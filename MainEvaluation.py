"""
MainEvaluation-complete.py -- multi-dataset defense benchmarking.

This is the multi-dataset extension of ``MainEvaluation.py`` (the original file
is left untouched, see the ``-complete`` convention).  It keeps every behavior
of the original while adding:

* **Combined training**: every defense model is trained once on the combined
  dataset pickle produced by ``TrainDataGeneration-complete.py`` (all datasets
  live in the same ``data`` list).
* **Leakage-safe per-tag evaluation**: for each configured evaluation dataset
  tag the orchestrator excludes (a) the indexes used to *train on that same
  tag* and (b) the indexes selected by the HPS pool for that same tag, so
  training/eval leakage is impossible.
* **Per-dataset results**: evaluation results are saved in separate files per
  dataset tag (``<output_dir>/<tag>/<model>.json``); the configured
  ``output_file`` holds a small summary mapping tags/models to those files.
* **Consolidated hyperparameter search** (``--hps``), which lives in this same
  file and in ``EvaluationDebateLoop-complete.py`` (no separate ``-HPS`` file,
  see R5).  The search saves the selected HPS indexes **per dataset tag**.

Config schema (new keys)::

    train_pkl_path: data/multi-train.pkl
    output_file: results/multi-eval/summary.json

    eval_datasets:                  # list of evaluation datasets (multi mode)
      - tag: MMLUPRO                # config tag (resolved to a loader TAG)
        loader_tag: MMLUPro         # optional explicit loader TAG
        num_questions: 1            # fixed-topology question count
        n_questions_on_random_topo: 1
        questions_random_seed: 28   # optional per-tag seed
        ma_dataset_path: ...        # optional per-tag dataset path
        hps_indexes: hps/...-index.pkl  # optional explicit per-tag HPS indexes

    hyperparameter_search:          # optional section => enables --hps mode
      total_samples: 4              # fixed HPS pool size, per dataset tag
      run_samples: 2                # per-run subset size, per dataset tag
      split_seed: 42
      index_pkl: hps/multi/index.pkl       # combined per-tag index record
      index_pkl_dir: hps/multi             # per-tag index pickles
      results_csv: hps/multi/results.csv

When ``eval_datasets`` is absent, the legacy single
``live_evaluation_config.questions_dataset_tag`` setup is wrapped into a
single-entry list, so existing configs keep working.  Legacy top-level HPS keys
(``hps_total_samples``, ``hps_run_samples``, ``hps_split_seed``, ``index_pkl``,
``results_csv``) are also accepted.
"""

import argparse
import copy
import csv
import gc
import hashlib
import importlib.util
import inspect
import itertools
import json
import os
import pickle
import sys
import tempfile
import traceback
from pathlib import Path
from time import time

import numpy as np
import yaml

from Utils import load_config_from_path, AttrDict, _to_attrdict
from LoggingUtils import (
    log_section,
    log_info,
    log_warn,
    log_error,
    log_done,
    log_config,
    fmt_seconds,
    print_stats_table,
    print_timing_report,
)


# ---------------------------------------------------------------------------
#  Consolidated evaluation-loop module (hyphenated filename -> load by path)
# ---------------------------------------------------------------------------

def _load_edl_module():
    here = Path(__file__).resolve().parent
    mod_path = here / "EvaluationDebateLoop-complete.py"
    spec = importlib.util.spec_from_file_location("EvaluationDebateLoop_complete", mod_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["EvaluationDebateLoop_complete"] = module
    spec.loader.exec_module(module)
    return module


_EDL = _load_edl_module()
LiveDebateOrchestration = _EDL.LiveDebateOrchestration
build_hps_pool_loader = _EDL.build_hps_pool_loader
draw_hps_run_subset = _EDL.draw_hps_run_subset
resolve_loader_tag_from_path = _EDL.resolve_loader_tag_from_path


# ---------------------------------------------------------------------------
#  Topology helpers (unchanged from MainEvaluation.py)
# ---------------------------------------------------------------------------

def adjacency_matrix_symmetric(n, topology):
    if n < 1:
        raise ValueError("n must be >= 1")

    A = [[0] * n for _ in range(n)]

    if topology == "chain":
        for i in range(n - 1):
            A[i][i + 1] = 1
            A[i + 1][i] = 1
    elif topology == "star":
        for i in range(1, n):
            A[0][i] = 1
            A[i][0] = 1

        for i in range(1, n):
            j = i + 1 if i < n - 1 else 1
            A[i][j] = 1
            A[j][i] = 1
    elif topology == "tree":
        for i in range(n):
            left = 2 * i + 1
            right = 2 * i + 2
            if left < n:
                A[i][left] = 1
                A[left][i] = 1
            if right < n:
                A[i][right] = 1
                A[right][i] = 1
    else:
        raise ValueError("topology must be 'chain', 'star', or 'tree'")

    return A


def generate_topologies(num_agents: int):
    return {
        "tree": adjacency_matrix_symmetric(num_agents, "tree"),
        "chain": adjacency_matrix_symmetric(num_agents, "chain"),
        "star": adjacency_matrix_symmetric(num_agents, "star"),
    }


def _extract_adj(record):
    if not isinstance(record, dict):
        return None
    return record.get("adj_matrix") or record.get("topology") or record.get("adjacency_matrix")


def load_topologies_from_pickle(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and "data" in data:
        data = data["data"]

    topologies = {}
    for item in data:
        if not isinstance(item, dict):
            continue

        topo_name = item.get("topology_name")
        adj = _extract_adj(item)
        if topo_name and adj is not None and topo_name not in topologies:
            topologies[topo_name] = adj

        results = item.get("results")
        if isinstance(results, list):
            for debate in results:
                if not isinstance(debate, dict):
                    continue
                debate_topo_name = debate.get("topology_name")
                debate_adj = _extract_adj(debate)
                if debate_topo_name and debate_adj is not None and debate_topo_name not in topologies:
                    topologies[debate_topo_name] = debate_adj

    return topologies


def resolve_topologies(config, config_file_path):
    live_cfg = config.live_evaluation_config

    topologies_file = getattr(live_cfg, "topologies_file", None)
    if topologies_file:
        topologies_path = Path(topologies_file)
        if topologies_path.exists():
            with open(topologies_path, "r", encoding="utf-8") as f:
                log_info(f"Loading topologies from file: {topologies_path}")
                return json.load(f)
        log_warn(f"topologies_file not found: {topologies_path}. Falling back to generated topologies.")

    pkl_path = getattr(live_cfg, "topologies_from_pkl", None)
    if pkl_path:
        pkl_topologies_path = Path(pkl_path)
        if pkl_topologies_path.exists():
            topologies = load_topologies_from_pickle(pkl_topologies_path)
            if topologies:
                log_info(f"Loaded {len(topologies)} topologies from pickle: {pkl_topologies_path}")
                return topologies
            log_warn(f"No topologies found in pickle: {pkl_topologies_path}. Falling back to generated topologies.")
        else:
            log_warn(f"topologies_from_pkl not found: {pkl_topologies_path}. Falling back to generated topologies.")

    n_agents = getattr(live_cfg, "num_agents", None)
    if n_agents is None:
        raise ValueError(
            "Cannot resolve topologies: provide live_evaluation_config.num_agents "
            "or a valid live_evaluation_config.topologies_file/topologies_from_pkl."
        )

    generated = generate_topologies(n_agents)
    if getattr(live_cfg, "new_random_each_question", False):
        generated["random"] = None
    log_info("Using generated topologies from live_evaluation_config.num_agents")
    return generated


def load_embedded_model_configs(config_file_path: str):
    with open(config_file_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f) or {}

    # Global default training itinerary: every model that does not define its
    # own 'pkl_train' falls back to the top-level train_pkl_path.
    default_train_pkl = raw_config.get("train_pkl_path")

    section_candidates = [
        "defense_model_train_configs",
        "model_train_configs",
        "models_train_configs",
    ]
    for section_name in section_candidates:
        section = raw_config.get(section_name)
        if section is None:
            continue
        if not isinstance(section, dict):
            raise ValueError(f"'{section_name}' must be a mapping of model_name -> config dict.")

        normalized = {}
        for model_name, cfg in section.items():
            if isinstance(cfg, dict):
                if default_train_pkl is not None and "pkl_train" not in cfg:
                    cfg["pkl_train"] = default_train_pkl
                cfg.setdefault("run_name", model_name)
                normalized[model_name] = [cfg]
            elif isinstance(cfg, list):
                for i, c in enumerate(cfg):
                    if default_train_pkl is not None and "pkl_train" not in c:
                        c["pkl_train"] = default_train_pkl
                    c.setdefault("run_name", f"{model_name}_{i}")
                normalized[model_name] = cfg
            else:
                raise ValueError(f"Config for '{model_name}' must be a dict or list of dicts.")
        return normalized

    raise ValueError("Missing embedded defense train config section in main config.")


def _write_temp_model_config(model_name: str, model_config: dict):
    fd, temp_path = tempfile.mkstemp(prefix=f"{model_name}-", suffix=".yaml")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(model_config, f, sort_keys=False)
    except Exception:
        Path(temp_path).unlink(missing_ok=True)
        raise
    return temp_path


def get_models_from_path(path, embedded_model_configs):
    models = {}
    folder = Path(path)
    for file in sorted(folder.glob("*.py")):
        module_name = file.stem
        spec = importlib.util.spec_from_file_location(module_name, file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not (hasattr(module, "Master") and inspect.isclass(getattr(module, "Master"))):
            continue

        cls = getattr(module, "Master")

        if module_name not in embedded_model_configs:
            log_info(f"No embedded config found for '{module_name}'. Skipping \u2014 model will not be evaluated or trained.")
            continue
        else:
            configs = embedded_model_configs[module_name]

        for model_cfg in configs:
            run_name = model_cfg.get("run_name", module_name)
            if not isinstance(model_cfg, dict):
                raise ValueError(f"Embedded config for model '{module_name}' must be a dict.")

            temp_config_path = _write_temp_model_config(run_name, model_cfg)
            models[run_name] = {
                "master": cls(temp_config_path),
                "config_path": f"embedded:defense_model_train_configs.{module_name}[{run_name}]",
                "temp_config_path": temp_config_path,
            }
            log_info(f"Loaded run '{run_name}' from model '{module_name}'.")

    return models


def _update_name_with_threshold(name: str, new_threshold: float) -> str:
    parts = name.split("_")
    updated = []
    for p in parts:
        if p.startswith("threshold") and not p == "threshold":
            val_str = p[len("threshold"):].replace("-", ".")
            try:
                float(val_str)
                formatted = f"threshold{str(new_threshold).replace('.', '-')}"
                updated.append(formatted)
                continue
            except ValueError:
                pass
        updated.append(p)
    return "_".join(updated)


def _cleanup_model(model_instance) -> None:
    del model_instance
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _safe_filename(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(name))
    return cleaned.strip("_") or "unnamed"


# ---------------------------------------------------------------------------
#  Multi-dataset configuration
# ---------------------------------------------------------------------------

def _get_eval_dataset_entries(config):
    """Return the evaluation dataset entries (multi-dataset aware).

    Each entry is a dict with ``tag`` (config tag), ``loader_tag`` (resolved
    canonical TAG), per-dataset question counts and optional overrides.
    """
    live_cfg = config.live_evaluation_config
    raw_entries = getattr(config, "eval_datasets", None)

    if not raw_entries:
        legacy_tag = getattr(
            live_cfg, "questions_dataset_tag", getattr(live_cfg, "dataset_tag", None)
        )
        if not legacy_tag:
            raise ValueError(
                "No evaluation datasets configured: provide a top-level 'eval_datasets' "
                "list or live_evaluation_config.questions_dataset_tag."
            )
        raw_entries = [
            {
                "tag": legacy_tag,
                "num_questions": getattr(live_cfg, "num_questions", 0),
                "n_questions_on_random_topo": getattr(live_cfg, "n_questions_on_random_topo", 0),
                "questions_random_seed": getattr(live_cfg, "questions_random_seed", None),
                "ma_dataset_path": getattr(live_cfg, "ma_dataset_path", None),
                "hps_indexes": getattr(config, "HPS_indexes", None),
            }
        ]

    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("'eval_datasets' must be a non-empty list of dataset entries.")

    entries = []
    seen = set()
    for idx, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, dict):
            raise ValueError(f"eval_datasets[{idx}] must be a mapping.")
        if not raw_entry.get("tag"):
            raise ValueError(f"eval_datasets[{idx}] is missing required key 'tag'.")

        loader_cls = resolve_loader_tag_from_path(
            live_cfg.questions_path,
            raw_entry["tag"],
            explicit_loader_tag=raw_entry.get("loader_tag"),
        )
        loader_tag = loader_cls.TAG

        if loader_tag in seen:
            raise ValueError(
                f"Duplicate evaluation tag '{raw_entry['tag']}' (loader TAG "
                f"'{loader_tag}') in eval_datasets."
            )
        seen.add(loader_tag)

        entry = {
            "tag": raw_entry["tag"],
            "loader_tag": loader_tag,
            "num_questions": raw_entry.get(
                "num_questions", getattr(live_cfg, "num_questions", 0)
            ),
            "n_questions_on_random_topo": raw_entry.get(
                "n_questions_on_random_topo",
                getattr(live_cfg, "n_questions_on_random_topo", 0),
            ),
            "questions_random_seed": raw_entry.get(
                "questions_random_seed", getattr(live_cfg, "questions_random_seed", None)
            ),
            "ma_dataset_path": raw_entry.get(
                "ma_dataset_path", getattr(live_cfg, "ma_dataset_path", None)
            ),
            "hps_indexes": raw_entry.get("hps_indexes", None),
        }
        entries.append(entry)

    return entries


def _load_train_indexes_per_tag(pkl_path):
    """Load ``idx_metadata`` from a training pickle as ``{tag: [indexes]}``.

    Supports the multi-dataset schema (``idx_metadata`` dict) and the legacy
    flat list schema (stored under the wildcard key ``"*"``).
    """
    if not pkl_path:
        return {}
    path = Path(pkl_path)
    if not path.exists():
        log_warn(f"Training pickle not found: {pkl_path}. No training indexes will be excluded.")
        return {}

    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        log_warn(f"Could not read training indexes from {pkl_path}: {e}")
        return {}

    idx_metadata = data.get("idx_metadata") if isinstance(data, dict) else None
    result = {}
    if isinstance(idx_metadata, dict):
        for tag, indexes in idx_metadata.items():
            if indexes is None:
                continue
            result[str(tag)] = sorted({int(i) for i in indexes})
    elif isinstance(idx_metadata, (list, tuple)):
        result["*"] = sorted({int(i) for i in idx_metadata})
    return result


def _indexes_for_entry(indexes_by_tag, entry):
    """Return the per-tag exclusion indexes for an evaluation entry."""
    if not indexes_by_tag:
        return set()

    candidates = {str(entry["tag"]), str(entry["loader_tag"])}
    normalized_candidates = {
        "".join(ch for ch in c.upper() if ch.isalnum()) for c in candidates
    }

    for key, indexes in indexes_by_tag.items():
        if key in candidates:
            return set(indexes)
    for key, indexes in indexes_by_tag.items():
        normalized_key = "".join(ch for ch in str(key).upper() if ch.isalnum())
        if normalized_key in normalized_candidates:
            return set(indexes)
    if "*" in indexes_by_tag:
        return set(indexes_by_tag["*"])
    return set()


def _parse_per_tag_index_pickle(path):
    """Read an HPS/Train index pickle and return ``(flat_indices, per_tag)``.

    Accepted structures:
      * ``{"indices": [...], ...}`` -> flat list
      * ``{"indices_per_tag": {tag: [...]}, ...}`` -> per-tag map
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Index pickle '{path}' has an unexpected structure.")

    per_tag = {}
    raw_per_tag = data.get("indices_per_tag")
    if isinstance(raw_per_tag, dict):
        for tag, indexes in raw_per_tag.items():
            per_tag[str(tag)] = [int(i) for i in (indexes or [])]

    flat = []
    if "indices" in data and isinstance(data["indices"], (list, tuple)):
        flat = [int(i) for i in data["indices"]]
    return flat, per_tag


def _load_hps_indexes_for_entry(config, entry, index_pkl_dir=None):
    """Resolve the HPS exclusion indexes for one evaluation dataset tag.

    Precedence:
      1. explicit ``eval_datasets[i].hps_indexes`` path,
      2. ``hps_index_pkl_dir`` / ``hyperparameter_search.index_pkl_dir``
         ``<tag>-index.pkl`` / ``<loader_tag>-index.pkl``,
      3. legacy top-level ``HPS_indexes`` file (per-tag or flat).
    """
    candidates = []

    explicit = entry.get("hps_indexes")
    if explicit:
        candidates.append(Path(explicit))

    if not index_pkl_dir:
        hps_section = getattr(config, "hyperparameter_search", None)
        if hps_section is not None:
            index_pkl_dir = getattr(hps_section, "index_pkl_dir", None)
    if not index_pkl_dir:
        index_pkl_dir = getattr(config, "hps_index_pkl_dir", None)

    if index_pkl_dir:
        base = Path(index_pkl_dir)
        for tag in (entry["tag"], entry["loader_tag"]):
            candidates.append(base / f"{_safe_filename(tag)}-index.pkl")

    legacy = getattr(config, "HPS_indexes", None)
    if legacy:
        candidates.append(Path(legacy))

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            flat, per_tag = _parse_per_tag_index_pickle(candidate)
        except Exception as e:
            log_warn(f"Could not read HPS indexes from {candidate}: {e}")
            continue

        if per_tag:
            indexes = _indexes_for_entry(per_tag, entry)
            if indexes:
                log_info(f"HPS index exclusion for '{entry['tag']}': {len(indexes)} index(es) from {candidate}")
                return indexes
            # The combined file exists but has no entry for this tag: for a
            # single-dataset legacy setup fall back to the flat list.
            if len(per_tag) == 1 and flat:
                return set(flat)
        elif flat:
            log_info(f"HPS index exclusion for '{entry['tag']}': {len(flat)} index(es) from {candidate}")
            return set(flat)

    return set()


def _build_live_config_for_tag(base_live_cfg, entry):
    """Clone the shared live config with per-tag overrides applied."""
    live = AttrDict({k: _to_attrdict(v) for k, v in base_live_cfg.items()})
    live.questions_dataset_tag = entry["loader_tag"]
    live.num_questions = int(entry.get("num_questions") or 0)
    live.n_questions_on_random_topo = int(entry.get("n_questions_on_random_topo") or 0)
    if entry.get("questions_random_seed") is not None:
        live.questions_random_seed = int(entry["questions_random_seed"])
    if entry.get("ma_dataset_path"):
        live.ma_dataset_path = entry["ma_dataset_path"]
    return live


def _per_tag_result_path(output_path: Path, tag: str, model_name: str) -> Path:
    return output_path.parent / _safe_filename(tag) / f"{_safe_filename(model_name)}.json"


# ---------------------------------------------------------------------------
#  Summary persistence
# ---------------------------------------------------------------------------

def _read_summary(output_path: Path) -> dict:
    if not output_path.exists():
        return {}
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        log_warn("Could not read existing results file. Starting fresh.")
        return {}


def _write_summary(output_path: Path, summary: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)


def _register_result(summary, tag, model_name, per_tag_path, train_indexes, hps_indexes):
    per_dataset = summary.setdefault("per_dataset_results", {})
    per_dataset.setdefault(tag, {})[model_name] = str(per_tag_path)

    excluded = summary.setdefault("excluded_indexes", {})
    excluded[tag] = {
        "train": sorted(int(i) for i in train_indexes),
        "hps": sorted(int(i) for i in hps_indexes),
    }


def _refresh_completed_runs(summary, eval_tags):
    per_dataset = summary.get("per_dataset_results", {})
    model_names = set()
    for tag in eval_tags:
        model_names.update(per_dataset.get(tag, {}).keys())
    completed = sorted(
        name for name in model_names
        if all(name in per_dataset.get(tag, {}) for tag in eval_tags)
    )
    summary["completed_runs"] = completed
    return completed


# ---------------------------------------------------------------------------
#  Standard evaluation
# ---------------------------------------------------------------------------

def _evaluate_model_on_tag(
    model_label,
    model_instance,
    entry,
    topologies,
    config,
    base_live,
    train_indexes_by_tag,
    output_path,
    index_pkl_dir=None,
):
    live_cfg = _build_live_config_for_tag(base_live, entry)
    train_indexes = _indexes_for_entry(train_indexes_by_tag, entry)
    hps_indexes = _load_hps_indexes_for_entry(config, entry, index_pkl_dir=index_pkl_dir)
    excluded = set(train_indexes) | set(hps_indexes)

    log_info(
        f"Evaluating '{model_label}' on dataset '{entry['tag']}' "
        f"(loader TAG={entry['loader_tag']}); excluded indexes: "
        f"{len(train_indexes)} training + {len(hps_indexes)} HPS = {len(excluded)} total"
    )

    orchestrator = LiveDebateOrchestration(
        live_cfg,
        train_indexes=sorted(train_indexes),
        excluded_indexes=sorted(hps_indexes),
        dataset_tag=entry["tag"],
        loader_tag=entry["loader_tag"],
    )
    traces = orchestrator.run_evaluation_single_defense_model_all_topos(
        model_instance, topologies
    )
    stats = orchestrator.parse_stats_single_model(traces)
    used_indexes = [int(i) for i in list(getattr(orchestrator.dataloader, "indexes", []))]

    payload = {
        "dataset_tag": entry["tag"],
        "loader_tag": entry["loader_tag"],
        "model": model_label,
        "train_excluded_indexes": sorted(int(i) for i in train_indexes),
        "hps_excluded_indexes": sorted(int(i) for i in hps_indexes),
        "used_indexes": used_indexes,
        "n_used_indexes": len(used_indexes),
        "results": stats,
    }
    per_tag_path = _per_tag_result_path(output_path, entry["tag"], model_label)
    per_tag_path.parent.mkdir(parents=True, exist_ok=True)
    with open(per_tag_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    log_info(f"Per-dataset results saved to {per_tag_path}")

    return stats, per_tag_path, train_indexes, hps_indexes


def _run_standard(config, parsed_args):
    log_section("Configuration Loading (standard multi-dataset evaluation)")
    log_info(f"Config file: {parsed_args.config_file}")

    embedded_model_configs = load_embedded_model_configs(parsed_args.config_file)
    models = get_models_from_path(config.models_directory, embedded_model_configs)

    overall_t0 = time()
    output_path = Path(config.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timing = {}

    if parsed_args.clean:
        if output_path.exists():
            output_path.unlink()
            log_info(f"Deleted existing results: {output_path.name}")
        report_path = output_path.with_name(f"report-{output_path.name}")
        if report_path.exists():
            report_path.unlink()
            log_info(f"Deleted existing report: {report_path.name}")

    summary = _read_summary(output_path)
    eval_entries = _get_eval_dataset_entries(config)
    eval_tags = [entry["tag"] for entry in eval_entries]
    log_info(
        "Evaluation datasets: "
        + ", ".join(f"{e['tag']}->{e['loader_tag']}" for e in eval_entries)
    )

    train_pkl_path = getattr(config, "train_pkl_path", None)
    train_indexes_by_tag = _load_train_indexes_per_tag(train_pkl_path)
    if train_indexes_by_tag:
        log_info(
            f"Loaded training indexes for {len(train_indexes_by_tag)} tag(s) "
            f"from {train_pkl_path}."
        )

    # Optional per-tag HPS index directory (written by the --hps mode).
    hps_section = getattr(config, "hyperparameter_search", None)
    index_pkl_dir = getattr(hps_section, "index_pkl_dir", None) if hps_section is not None else None
    if not index_pkl_dir:
        index_pkl_dir = getattr(config, "hps_index_pkl_dir", None)

    try:
        log_section("Topology Resolution")
        topologies = resolve_topologies(config, parsed_args.config_file)
        base_live = config.live_evaluation_config

        summary.setdefault("train_pkl_path", str(train_pkl_path) if train_pkl_path else None)
        summary["eval_datasets"] = [
            {"tag": e["tag"], "loader_tag": e["loader_tag"]} for e in eval_entries
        ]

        if getattr(base_live, "no_defense_baseline", False) and \
                "no_defense_baseline" not in summary.get("completed_runs", []):
            log_section("No-Defense Baseline")
            baseline_missing = [
                e for e in eval_entries
                if "no_defense_baseline" not in summary.get("per_dataset_results", {}).get(e["tag"], {})
            ]
            if baseline_missing:
                for entry in baseline_missing:
                    t0 = time()
                    live_cfg = _build_live_config_for_tag(base_live, entry)
                    train_indexes = _indexes_for_entry(train_indexes_by_tag, entry)
                    hps_indexes = _load_hps_indexes_for_entry(config, entry, index_pkl_dir=index_pkl_dir)
                    orchestrator = LiveDebateOrchestration(
                        live_cfg,
                        train_indexes=sorted(train_indexes),
                        excluded_indexes=sorted(hps_indexes),
                        dataset_tag=entry["tag"],
                        loader_tag=entry["loader_tag"],
                    )
                    questions = orchestrator.dataloader.get_formatted_questions()
                    baseline_traces = orchestrator.run_debate_no_defense(questions, topologies)
                    baseline_stats = orchestrator.parse_stats_single_model(baseline_traces)
                    used_indexes = [int(i) for i in list(getattr(orchestrator.dataloader, "indexes", []))]
                    payload = {
                        "dataset_tag": entry["tag"],
                        "loader_tag": entry["loader_tag"],
                        "model": "no_defense_baseline",
                        "train_excluded_indexes": sorted(int(i) for i in train_indexes),
                        "hps_excluded_indexes": sorted(int(i) for i in hps_indexes),
                        "used_indexes": used_indexes,
                        "n_used_indexes": len(used_indexes),
                        "results": baseline_stats,
                    }
                    per_tag_path = _per_tag_result_path(output_path, entry["tag"], "no_defense_baseline")
                    per_tag_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(per_tag_path, "w", encoding="utf-8") as f:
                        json.dump(payload, f, indent=2)
                    _register_result(
                        summary, entry["tag"], "no_defense_baseline",
                        per_tag_path, train_indexes, hps_indexes,
                    )
                    elapsed = time() - t0
                    timing[f"no_defense_baseline::{entry['tag']}"] = elapsed
                    print_stats_table(baseline_stats, model_name=f"no_defense_baseline [{entry['tag']}]")
                    log_info(f"Evaluation completed in {fmt_seconds(elapsed)}")
                    log_info(f"Results saved to {per_tag_path}")
                    del baseline_traces, baseline_stats
                    gc.collect()
                _refresh_completed_runs(summary, eval_tags)
                _write_summary(output_path, summary)
            else:
                log_info("No-Defense baseline already completed for every dataset tag.")

        completed_runs = _refresh_completed_runs(summary, eval_tags)
        if completed_runs:
            for name in list(models.keys()):
                if name in completed_runs:
                    log_info(f"Skipping '{name}' \u2014 already present in {output_path.name}.")
                    temp_cfg = models[name].get("temp_config_path")
                    if temp_cfg:
                        Path(temp_cfg).unlink(missing_ok=True)
                    del models[name]

        total_models = len(models)
        if total_models == 0:
            log_info("All planned models are already completed. Nothing to do.")
        else:
            log_info(f"Processing {total_models} model(s).")

        for idx, (model_name, model_info) in enumerate(models.items(), start=1):
            model_t0 = time()
            model_instance = None
            try:
                log_section(f"Model {idx}/{total_models}: {model_name}")
                log_config("config", model_info["config_path"])

                train_t0 = time()
                model_train_pkl = (
                    getattr(getattr(model_info["master"], "args", None), "pkl_train", None)
                    or config.train_pkl_path
                )
                metrics, model_instance = model_info["master"]._run(model_train_pkl)
                effective_name = model_name
                computed_threshold = metrics.get("computed_threshold") if isinstance(metrics, dict) else None
                if computed_threshold is not None:
                    effective_name = _update_name_with_threshold(model_name, computed_threshold)
                    log_info(f"Threshold computed: {computed_threshold:.6f} (config default overridden)")
                    if effective_name != model_name:
                        log_info(f"Effective run name: {effective_name}")
                log_info(f"Training completed in {fmt_seconds(time() - train_t0)}")

                for entry in eval_entries:
                    tag = entry["tag"]
                    if tag in summary.get("per_dataset_results", {}) and \
                            effective_name in summary["per_dataset_results"][tag]:
                        log_info(f"[{tag}] '{effective_name}' already evaluated; skipping.")
                        continue
                    eval_t0 = time()
                    stats, per_tag_path, train_indexes, hps_indexes = _evaluate_model_on_tag(
                        effective_name,
                        model_instance,
                        entry,
                        topologies,
                        config,
                        base_live,
                        train_indexes_by_tag,
                        output_path,
                        index_pkl_dir=index_pkl_dir,
                    )
                    elapsed = time() - eval_t0
                    timing[f"{effective_name}::{tag}"] = elapsed
                    print_stats_table(stats, model_name=f"{effective_name} [{tag}]")
                    log_info(f"Evaluation on '{tag}' completed in {fmt_seconds(elapsed)}")
                    _register_result(
                        summary, tag, effective_name, per_tag_path,
                        train_indexes, hps_indexes,
                    )
                    _refresh_completed_runs(summary, eval_tags)
                    _write_summary(output_path, summary)
                    del stats

                _cleanup_model(model_instance)
                model_instance = None
                log_done("Resources cleaned up")

                total_elapsed = time() - model_t0
                timing[effective_name] = total_elapsed
                log_info(f"Total elapsed: {fmt_seconds(total_elapsed)}")

            except KeyboardInterrupt:
                if model_instance is not None:
                    _cleanup_model(model_instance)
                raise
            except Exception as e:
                if model_instance is not None:
                    _cleanup_model(model_instance)
                elapsed = time() - model_t0
                log_error(f"Model '{model_name}' failed after {fmt_seconds(elapsed)}: {e}")
                for line in traceback.format_exc().strip().splitlines():
                    log_error(line)
                log_warn("Previously completed results are preserved. Moving to next model.")
                continue

    except KeyboardInterrupt:
        log_warn("KeyboardInterrupt received. All previously completed results have been saved.")
    except Exception as e:
        tb = traceback.format_exc()
        log_error(f"Unhandled exception: {e}")
        for line in tb.strip().splitlines():
            log_error(line)
        log_warn("Previously completed results are preserved in the output file.")
    finally:
        for model_info in models.values():
            temp_cfg = model_info.get("temp_config_path")
            if temp_cfg:
                Path(temp_cfg).unlink(missing_ok=True)

        _refresh_completed_runs(summary, eval_tags)
        _write_summary(output_path, summary)

        total_elapsed = time() - overall_t0
        timing["total_seconds"] = total_elapsed
        report_filename = f"report-{output_path.name}"
        if not report_filename.lower().endswith(".json"):
            report_filename += ".json"
        if timing:
            report_path = output_path.with_name(report_filename)
            with open(report_path, "w", encoding="utf-8") as report_file:
                json.dump(timing, report_file, indent=2)
            print()
            print_timing_report(timing, total_elapsed)
            log_info(f"Timing report saved to {report_path}")


# ---------------------------------------------------------------------------
#  Hyperparameter search (consolidated in this file, see R5)
# ---------------------------------------------------------------------------

HPS_INTERNAL_KEYS = {
    "hps_total_samples",
    "hps_run_samples",
    "hps_split_seed",
    "index_pkl",
    "index_pkl_dir",
    "results_csv",
}

STRUCTURAL_LIST_NAMES = {
    "density_range_for_random_topo",
    "topology",
    "topologies",
    "random_topology_data",
}

BASE_COLUMNS = [
    "config_signature",
    "model_name",
    "run_name",
    "dataset_tag",
    "topology",
    "total_questions",
    "correct_answers",
    "overall_accuracy",
    "overall_AUROC",
    "effective_config",
]


def _is_scalar(v):
    return isinstance(v, (int, float, str, bool)) or v is None


def _is_scalar_list(lst):
    return isinstance(lst, list) and len(lst) > 0 and all(_is_scalar(x) for x in lst)


def _find_list_params(obj, prefix=()):
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in HPS_INTERNAL_KEYS:
                continue
            path = prefix + (k,)
            if isinstance(v, list):
                if k in STRUCTURAL_LIST_NAMES:
                    _descend_structural(v, path, found)
                elif _is_scalar_list(v):
                    found.append((path, list(v)))
                else:
                    _descend_structural(v, path, found)
            elif isinstance(v, dict):
                found.extend(_find_list_params(v, path))
    return found


def _descend_structural(lst, path, found):
    for i, item in enumerate(lst):
        if isinstance(item, dict):
            found.extend(_find_list_params(item, path + (i,)))
        elif isinstance(item, list):
            _descend_structural(item, path + (i,), found)


def _get_path(obj, path):
    cur = obj
    for p in path:
        cur = cur[p]
    return cur


def _set_path(obj, path, value):
    cur = obj
    for p in path[:-1]:
        cur = cur[p]
    cur[path[-1]] = value


def _path_to_str(path):
    return ".".join(str(p) for p in path)


def _sanitize_name_value(v):
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, float):
        return f"{v}".replace(".", "-")
    if v is None:
        return "none"
    s = str(v)
    return s.replace(".", "-").replace(" ", "_").replace("/", "_")


def _make_hp_suffix(varied):
    leaves = {}
    for k in varied:
        leaves.setdefault(k.split(".")[-1], []).append(k)
    parts = []
    for k, v in sorted(varied.items()):
        leaf = k.split(".")[-1]
        name = leaf if len(leaves[leaf]) == 1 else k.replace(".", "").replace("_", "")
        parts.append(f"{name}{_sanitize_name_value(v)}")
    return "_".join(parts) if parts else ""


def _model_section_key(raw):
    for key in ("defense_model_train_configs", "model_train_configs", "models_train_configs"):
        if isinstance(raw.get(key), dict):
            return key
    return None


def expand_model_config(model_cfg):
    list_params = _find_list_params(model_cfg)
    if not list_params:
        return [(copy.deepcopy(model_cfg), {})]

    list_params.sort(key=lambda x: _path_to_str(x[0]))
    paths = [p for p, _ in list_params]
    value_lists = [vs for _, vs in list_params]

    expanded = []
    for combo in itertools.product(*value_lists):
        eff = copy.deepcopy(model_cfg)
        varied = {}
        for path, val in zip(paths, combo):
            _set_path(eff, path, val)
            varied[_path_to_str(path)] = val
        expanded.append((eff, varied))
    return expanded


def build_run_plans(raw):
    section_key = _model_section_key(raw)
    if section_key is None:
        raise ValueError(
            "Missing embedded defense train config section in main config "
            "(expected 'defense_model_train_configs', 'model_train_configs' or "
            "'models_train_configs')."
        )
    section = raw[section_key]
    global_cfg = _strip_hps_internal(raw)

    plans = []
    for model_name, model_cfg in section.items():
        if isinstance(model_cfg, dict):
            entries = [model_cfg]
        elif isinstance(model_cfg, list):
            entries = model_cfg
        else:
            log_warn(
                f"Skipping model '{model_name}': config must be a dict or list of dicts."
            )
            continue

        for entry in entries:
            if not isinstance(entry, dict):
                log_warn(f"Skipping model '{model_name}': config entry must be a dict.")
                continue
            base_name = entry.get("run_name", model_name)

            for combo_cfg, model_varied in expand_model_config(entry):
                varied = {
                    f"{section_key}.{model_name}.{k}": v
                    for k, v in model_varied.items()
                }
                hp_suffix = _make_hp_suffix(varied)
                run_name = f"{base_name}_{hp_suffix}" if hp_suffix else base_name

                combo_cfg["run_name"] = run_name

                eff = copy.deepcopy(global_cfg)
                eff[section_key] = {model_name: combo_cfg}

                identity_cfg = copy.deepcopy(combo_cfg)
                identity_cfg.pop("run_name", None)
                signature = _config_signature(model_name, identity_cfg)

                plans.append(
                    {
                        "model_name": model_name,
                        "run_name": run_name,
                        "eff": eff,
                        "varied": varied,
                        "signature": signature,
                    }
                )
    return plans


def _strip_hps_internal(cfg):
    out = copy.deepcopy(cfg)
    for k in HPS_INTERNAL_KEYS:
        out.pop(k, None)
    out.pop("hyperparameter_search", None)
    return out


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, default=str)


def _config_signature(model_name, effective_dict):
    payload = {"model": model_name, "config": effective_dict}
    h = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
    return h


def _write_temp_main_config(effective_dict):
    fd, temp_path = tempfile.mkstemp(prefix="hps-main-", suffix=".yaml")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(effective_dict, f, sort_keys=False)
    except Exception:
        Path(temp_path).unlink(missing_ok=True)
        raise
    return temp_path


def _flatten_stats_rows(stats, max_rounds, base_row):
    rows = []
    for topo_result in stats:
        row = dict(base_row)
        row["topology"] = topo_result.get("topology", "unknown")
        row["total_questions"] = topo_result.get("total_questions", 0)
        row["correct_answers"] = topo_result.get("correct_answers", 0)
        row["overall_accuracy"] = topo_result.get("overall_accuracy", 0)
        row["overall_AUROC"] = topo_result.get("overall_AUROC", 0)

        round_counts = topo_result.get("round_counts", {})
        rounds_rates = topo_result.get("rounds_rates", [])

        metric_keys = ["ASR", "UnFlagASR", "ADR", "AIR", "FPR", "F1"]
        for r in range(max_rounds):
            prefix = f"round_{r + 1}_"
            rr = rounds_rates[r] if r < len(rounds_rates) else {}
            row[prefix + "count"] = round_counts.get(r, "")
            for m in metric_keys:
                row[prefix + m] = rr.get(m, "")
                row[prefix + f"{m}_ci95"] = rr.get(f"{m}_ci95", "")
            row[prefix + "AUROC"] = rr.get("AUROC", "")
            row[prefix + "AUROC_ci95"] = rr.get("AUROC_ci95", "")
            row[prefix + "pooled_AUROC"] = rr.get("pooled_AUROC", "")
        rows.append(row)
    return rows


def _metric_columns(max_rounds):
    cols = []
    metric_keys = ["ASR", "UnFlagASR", "ADR", "AIR", "FPR", "F1"]
    for r in range(max_rounds):
        prefix = f"round_{r + 1}_"
        cols.append(prefix + "count")
        for m in metric_keys:
            cols.append(prefix + m)
            cols.append(prefix + f"{m}_ci95")
        cols.append(prefix + "AUROC")
        cols.append(prefix + "AUROC_ci95")
        cols.append(prefix + "pooled_AUROC")
    return cols


def _read_completed_pairs(csv_path):
    pairs = set()
    signatures = set()
    if not Path(csv_path).exists():
        return signatures, pairs
    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and "config_signature" in reader.fieldnames:
                for row in reader:
                    sig = row.get("config_signature")
                    if not sig:
                        continue
                    signatures.add(sig)
                    tag = row.get("dataset_tag")
                    if tag:
                        pairs.add((sig, tag))
    except Exception as e:
        log_warn(f"Could not read existing CSV ({csv_path}): {e}. Starting fresh.")
        return set(), set()
    return signatures, pairs


def _open_csv_writer(csv_path, fieldnames):
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    f = open(path, "a" if exists else "w", encoding="utf-8", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    if not exists:
        writer.writeheader()
        f.flush()
    return f, writer


def _parse_hps_settings(raw_config, config):
    section = raw_config.get("hyperparameter_search") or {}

    def pick(new_key, legacy_key, default=None):
        value = section.get(new_key) if isinstance(section, dict) else None
        if value is None:
            value = getattr(config, legacy_key, None)
        return default if value is None else value

    total = pick("total_samples", "hps_total_samples")
    run_samples = pick("run_samples", "hps_run_samples")
    split_seed = pick("split_seed", "hps_split_seed", 42)
    index_pkl = pick("index_pkl", "index_pkl")
    index_pkl_dir = pick("index_pkl_dir", "hps_index_pkl_dir")
    results_csv = pick("results_csv", "results_csv")

    if total is None:
        raise ValueError("Hyperparameter search requires 'total_samples' (hps_total_samples).")
    if index_pkl_dir is None:
        if index_pkl is None:
            raise ValueError(
                "Hyperparameter search requires 'index_pkl' and/or 'index_pkl_dir' "
                "to persist the selected per-tag indexes."
            )
        index_pkl_dir = str(Path(index_pkl).parent)
    if results_csv is None:
        raise ValueError("Hyperparameter search requires 'results_csv'.")

    return {
        "total_samples": int(total),
        "run_samples": int(run_samples) if run_samples is not None else None,
        "split_seed": int(split_seed),
        "index_pkl": str(index_pkl) if index_pkl else None,
        "index_pkl_dir": str(index_pkl_dir),
        "results_csv": str(results_csv),
    }


def _completed_run_names_from_csv(signatures, plans):
    completed = {
        plan["run_name"] for plan in plans if plan["signature"] in signatures
    }
    return completed


def _run_hps(raw_config, config, parsed_args):
    log_section("Hyperparameter Search (multi-dataset)")

    # The base config (as loaded) may contain list-valued model params; the raw
    # YAML is used for Cartesian expansion, exactly like the original HPSearch.
    settings = _parse_hps_settings(raw_config, config)
    total_samples = settings["total_samples"]
    run_samples = settings["run_samples"]
    split_seed = settings["split_seed"]
    index_pkl = settings["index_pkl"]
    index_pkl_dir = Path(settings["index_pkl_dir"])
    results_csv = Path(settings["results_csv"])

    log_config("hps_total_samples", total_samples)
    log_config("hps_run_samples", run_samples)
    log_config("hps_split_seed", split_seed)
    log_config("index_pkl", index_pkl)
    log_config("index_pkl_dir", str(index_pkl_dir))
    log_config("results_csv", str(results_csv))

    if parsed_args.clean:
        for p in (results_csv, index_pkl):
            if p and Path(p).exists():
                Path(p).unlink()
                log_info(f"--clean: removed {p}")
        if index_pkl_dir.exists():
            for p in index_pkl_dir.glob("*-index.pkl"):
                p.unlink()
                log_info(f"--clean: removed {p}")

    eval_entries = _get_eval_dataset_entries(config)
    eval_tags = [entry["tag"] for entry in eval_entries]
    log_info(
        "HPS evaluation datasets: "
        + ", ".join(f"{e['tag']}->{e['loader_tag']}" for e in eval_entries)
    )

    train_pkl_path = getattr(config, "train_pkl_path", None)
    train_indexes_by_tag = _load_train_indexes_per_tag(train_pkl_path)
    log_info(
        f"Loaded training indexes for {len(train_indexes_by_tag)} tag(s) "
        f"from {train_pkl_path}."
    )

    run_plans = build_run_plans(raw_config)
    total_plans = len(run_plans)
    varied_key_set = set()
    for plan in run_plans:
        varied_key_set.update(plan["varied"].keys())
    varied_cols = sorted(varied_key_set)

    base_live = raw_config.get("live_evaluation_config", {}) or {}
    max_rounds = int(base_live.get("max_rounds", 0))
    metric_cols = _metric_columns(max_rounds)
    fieldnames = list(BASE_COLUMNS) + varied_cols + metric_cols

    log_info(
        f"Hyperparameter expansion: {total_plans} run(s) "
        f"({len(varied_cols)} varied parameter column(s))."
    )
    for plan in run_plans:
        log_info(
            f"  - {plan['model_name']} | {plan['run_name']} | "
            f"signature={plan['signature'][:12]}"
        )

    completed_signatures, completed_pairs = _read_completed_pairs(results_csv)
    if completed_signatures:
        log_info(
            f"Resume: {len(completed_signatures)} completed configuration(s) "
            f"found in {results_csv}"
        )

    csv_file, csv_writer = _open_csv_writer(results_csv, fieldnames)

    # -- Build one fixed HPS pool per dataset tag (exclude that tag's training
    #    indexes) and persist the selected indexes per tag.
    index_pkl_dir.mkdir(parents=True, exist_ok=True)
    pool_questions = {}
    pool_loaders = {}
    pool_indices_per_tag = {}
    pool_indices_flat = set()
    for entry in eval_entries:
        tag = entry["tag"]
        live_cfg = _build_live_config_for_tag(config.live_evaluation_config, entry)
        train_indexes = _indexes_for_entry(train_indexes_by_tag, entry)
        per_tag_index_path = index_pkl_dir / f"{_safe_filename(tag)}-index.pkl"
        pool_loader, pool_indices = build_hps_pool_loader(
            live_cfg,
            sorted(train_indexes),
            total_samples,
            split_seed,
            per_tag_index_path,
            dataset_tag=tag,
            loader_tag=entry["loader_tag"],
        )
        pool_loaders[tag] = pool_loader
        pool_questions[tag] = list(pool_loader.get_formatted_questions())
        pool_indices_per_tag[tag] = [int(i) for i in pool_indices]
        pool_indices_flat.update(int(i) for i in pool_indices)
        log_info(
            f"HPS pool ready for '{tag}': {len(pool_questions[tag])} question(s), "
            f"index pickle: {per_tag_index_path}"
        )

    if index_pkl:
        combined = {
            "indices": sorted(pool_indices_flat),
            "indices_per_tag": pool_indices_per_tag,
            "params": {
                "hps_total_samples": total_samples,
                "hps_run_samples": run_samples,
                "hps_split_seed": split_seed,
                "index_pkl_dir": str(index_pkl_dir),
                "dataset_tags": eval_tags,
            },
        }
        Path(index_pkl).parent.mkdir(parents=True, exist_ok=True)
        with open(index_pkl, "wb") as f:
            pickle.dump(combined, f)
        log_info(f"Combined per-tag HPS indexes saved to {index_pkl}")

    # Shared text processor (loaded once, reused across configurations).
    cached_tp = None
    cached_tp_key = None

    overall_t0 = time()
    elapsed_accum = 0.0
    done_before = 0
    temp_paths = []

    try:
        for idx, plan in enumerate(run_plans, start=1):
            eff = plan["eff"]
            varied = plan["varied"]
            signature = plan["signature"]
            model_name = plan["model_name"]
            run_name = plan["run_name"]

            pending_tags = [
                entry for entry in eval_entries
                if (signature, entry["tag"]) not in completed_pairs
            ]
            if not pending_tags:
                log_section(f"HP Search [{idx}/{total_plans}]: {model_name} -- SKIPPED")
                log_info(f"Skipping run (already present in CSV): {run_name}")
                continue

            temp_main = _write_temp_main_config(eff)
            temp_paths.append(temp_main)

            log_section(f"HP Search [{idx}/{total_plans}]: {model_name}")
            if varied:
                for k, v in sorted(varied.items()):
                    log_config(k, v)
            else:
                log_info("No varied parameters for this configuration.")
            log_config("run_name", run_name)
            if done_before > 0:
                avg = elapsed_accum / done_before
                remaining = (total_plans - done_before) * avg
                log_info(
                    f"Progress: {done_before}/{total_plans} done | "
                    f"elapsed {fmt_seconds(elapsed_accum)} | ETA ~{fmt_seconds(remaining)}"
                )

            combo_t0 = time()
            model_instance = None
            combo_model_temps = []
            try:
                combo_config = load_config_from_path(temp_main)
                live_base = combo_config.live_evaluation_config

                tp_key = (
                    getattr(live_base, "text_processor_path", None),
                    getattr(live_base, "text_processor_class_name", None),
                )
                if cached_tp is None or tp_key != cached_tp_key:
                    textProcessor = _EDL.load_class_from_path(
                        live_base.text_processor_path,
                        live_base.text_processor_class_name,
                    )
                    cached_tp = textProcessor(device="cpu")
                    cached_tp_key = tp_key
                    log_info(f"Text processor loaded: {cached_tp_key}")

                embedded = load_embedded_model_configs(temp_main)
                models = get_models_from_path(combo_config.models_directory, embedded)
                if not models:
                    log_warn(
                        f"No trainable model resolved for '{model_name}' in "
                        f"{combo_config.models_directory}; skipping this run."
                    )
                    continue

                for _mn, _mi in models.items():
                    if _mi.get("temp_config_path"):
                        combo_model_temps.append(_mi["temp_config_path"])

                model_train_pkl = (
                    getattr(getattr(next(iter(models.values()))["master"], "args", None), "pkl_train", None)
                    or getattr(combo_config, "train_pkl_path", None)
                )

                log_section(f"Training [{idx}/{total_plans}]: {model_name}")
                train_t0 = time()
                for _loaded_run_name, model_info in models.items():
                    metrics, model_instance = model_info["master"]._run(model_train_pkl)
                effective_name = run_name
                computed_threshold = (
                    metrics.get("computed_threshold") if isinstance(metrics, dict) else None
                )
                if computed_threshold is not None:
                    effective_name = _update_name_with_threshold(run_name, computed_threshold)
                    log_info(
                        f"Threshold computed: {computed_threshold:.6f} "
                        "(config default overridden)"
                    )
                log_info(f"Training completed in {fmt_seconds(time() - train_t0)}")

                log_section(f"Evaluating [{idx}/{total_plans}]: {effective_name}")
                eval_t0 = time()
                any_rows = False
                for entry in pending_tags:
                    tag = entry["tag"]
                    subset = draw_hps_run_subset(
                        pool_questions[tag], run_samples, split_seed, f"{signature}::{tag}"
                    )
                    live_cfg = _build_live_config_for_tag(live_base, entry)
                    live_cfg.num_questions = len(subset)
                    live_cfg.n_questions_on_random_topo = len(subset)
                    live_cfg.new_random_each_question = True
                    topologies = {"random": None}

                    orchestrator = LiveDebateOrchestration(
                        live_cfg,
                        dataloader=pool_loaders[tag],
                        text_processor=cached_tp,
                        train_indexes=sorted(_indexes_for_entry(train_indexes_by_tag, entry)),
                        dataset_tag=tag,
                        loader_tag=entry["loader_tag"],
                    )
                    traces = orchestrator.run_debate_with_defense(
                        subset, model_instance, topologies
                    )
                    stats = orchestrator.parse_stats_single_model(traces)
                    used_indexes = [
                        int(i) for i in list(getattr(orchestrator.dataloader, "indexes", []))
                    ]
                    log_info(
                        f"[{tag}] run subset: {len(subset)} question(s); "
                        f"pool indexes used: {len(used_indexes)}"
                    )

                    base_row = {
                        "config_signature": signature,
                        "model_name": model_name,
                        "run_name": effective_name,
                        "dataset_tag": tag,
                        "effective_config": _canonical(eff),
                    }
                    for col, val in varied.items():
                        base_row[col] = val

                    rows = _flatten_stats_rows(stats, max_rounds, base_row)
                    if not rows:
                        rows = [dict(base_row, topology="(no valid debates)")]
                    for r in rows:
                        csv_writer.writerow(r)
                    any_rows = True
                    completed_pairs.add((signature, tag))
                    del traces, stats

                csv_file.flush()
                try:
                    os.fsync(csv_file.fileno())
                except OSError:
                    pass
                if any_rows:
                    log_info(
                        f"Results flushed to CSV: {results_csv} "
                        f"({len(completed_pairs)} tag-run pair(s) completed)"
                    )
                    completed_signatures.add(signature)

                _cleanup_model(model_instance)
                model_instance = None
                log_info(f"Evaluation completed in {fmt_seconds(time() - eval_t0)}")
                log_done("Resources cleaned up")

                combo_elapsed = time() - combo_t0
                elapsed_accum += combo_elapsed
                done_before += 1
                log_info(
                    f"Run elapsed: {fmt_seconds(combo_elapsed)} | "
                    f"total so far: {fmt_seconds(time() - overall_t0)}"
                )

            except KeyboardInterrupt:
                if model_instance is not None:
                    _cleanup_model(model_instance)
                log_warn("KeyboardInterrupt received. Completed results are saved.")
                raise
            except Exception as e:
                if model_instance is not None:
                    _cleanup_model(model_instance)
                log_error(f"Run [{idx}/{total_plans}] failed: {e}")
                for line in traceback.format_exc().strip().splitlines():
                    log_error(line)
                log_warn(
                    "This run was NOT marked as completed. "
                    "It will be retried on the next run."
                )
                continue
            finally:
                if temp_main in temp_paths:
                    temp_paths.remove(temp_main)
                Path(temp_main).unlink(missing_ok=True)
                for p in combo_model_temps:
                    Path(p).unlink(missing_ok=True)
                combo_model_temps = []

    except KeyboardInterrupt:
        log_warn("KeyboardInterrupt. All previously completed results are saved.")
    except Exception as e:
        log_error(f"Unhandled exception: {e}")
        for line in traceback.format_exc().strip().splitlines():
            log_error(line)
    finally:
        try:
            csv_file.flush()
            try:
                os.fsync(csv_file.fileno())
            except OSError:
                pass
            csv_file.close()
        except Exception:
            pass
        for p in list(temp_paths):
            Path(p).unlink(missing_ok=True)
        total_elapsed = time() - overall_t0
        log_section("HPS Search Finished")
        log_info(f"Total elapsed: {fmt_seconds(total_elapsed)}")
        log_info(f"Results CSV: {results_csv}")
        log_info(f"Per-tag index pickles: {index_pkl_dir}")


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------

def _hps_requested(config) -> bool:
    if getattr(config, "hyperparameter_search", None) is not None:
        return True
    return getattr(config, "hps_total_samples", None) is not None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to the configuration file.")
    parser.add_argument("--clean", action="store_true", help="Delete existing results and start fresh.")
    parser.add_argument(
        "--hps",
        action="store_true",
        help="Run the consolidated hyperparameter search (multi-dataset).",
    )
    parsed_args = parser.parse_args()
    config = load_config_from_path(parsed_args.config_file)

    with open(parsed_args.config_file, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f) or {}

    if parsed_args.hps or _hps_requested(config):
        _run_hps(raw_config, config, parsed_args)
    else:
        _run_standard(config, parsed_args)
