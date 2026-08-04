"""
HPSearch.py  --  Hyperparameter search orchestration.

Repeatedly trains and evaluates defense models over model-specific
hyperparameter configurations, then immediately persists the evaluation results
to a CSV file.

The configuration format is identical to the standard evaluation configuration.
Each model under the model-config section (e.g. ``defense_model_train_configs``)
may declare *any* of its own parameters as a list of values.  The Cartesian
product of list-valued parameters is expanded **independently for each model**,
so the parameters of one model never combine with those of another model.

Every generated run receives the complete global (top-level) configuration
together with the selected model and one of that model's hyperparameter
combinations, so parameters such as ``train_pkl_path`` are propagated to the
training/evaluation pipeline exactly as in the standard evaluation.

New configuration:

    hps_total_samples: 100     # size of the fixed HPS evaluation pool
    hps_run_samples:   40      # per-run random subset drawn from the pool
    hps_split_seed:     42     # reproducibility seed (pool + per-run subsampling)
    index_pkl:  hps/index.pkl   # persisted pool indices (reused on resume)
    results_csv: hps/results.csv # appended-to CSV of all completed runs

Usage::

    python HPSearch.py <config_file> [--clean]

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
from types import SimpleNamespace

import numpy as np
import yaml

from Utils import load_config_from_path
from LoggingUtils import (
    log_section,
    log_info,
    log_warn,
    log_error,
    log_done,
    log_config,
    fmt_seconds,
    print_stats_table,
)

from MainEvaluation import (
    load_embedded_model_configs,
    get_models_from_path,
    resolve_topologies,
    _update_name_with_threshold,
    _cleanup_model,
    _write_temp_model_config,
    generate_topologies,
)


# ---------------------------------------------------------------------------
#  Dynamic import of the hyphenated HPS module (EvaluationDebateLoop-HPS.py)
# ---------------------------------------------------------------------------

def _load_hps_module():
    here = Path(__file__).resolve().parent
    mod_path = here / "EvaluationDebateLoop-HPS.py"
    spec = importlib.util.spec_from_file_location("EvaluationDebateLoop_HPS", mod_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["EvaluationDebateLoop_HPS"] = module
    spec.loader.exec_module(module)
    return module


_HPS = _load_hps_module()
LiveDebateOrchestrationHPS = _HPS.LiveDebateOrchestrationHPS
build_hps_pool_loader = _HPS.build_hps_pool_loader


# ---------------------------------------------------------------------------
#  Top-level HPS configuration knobs (never expanded as search parameters)
# ---------------------------------------------------------------------------

HPS_INTERNAL_KEYS = {
    "hps_total_samples",
    "hps_run_samples",
    "hps_split_seed",
    "index_pkl",
    "results_csv",
}

STRUCTURAL_LIST_NAMES = {
    "density_range_for_random_topo",
    "topology",
    "topologies",
    "random_topology_data",
}


# ---------------------------------------------------------------------------
#  Configuration expansion helpers
# ---------------------------------------------------------------------------

def _is_scalar(v):
    return isinstance(v, (int, float, str, bool)) or v is None


def _is_scalar_list(lst):
    return isinstance(lst, list) and len(lst) > 0 and all(_is_scalar(x) for x in lst)


def _find_list_params(obj, prefix=()):
    """
    Recursively locate every list-valued scalar parameter in *obj*.

    A list is treated as a varying parameter when every element is a scalar and
    its name is not a known structural parameter.  Structural lists (lists of
    dicts, lists of lists, e.g. topology matrices or model-config sections) are
    traversed instead so that nested scalar lists params inside them are found.

    Returns a list of (path_tuple, values).
    """
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
            # scalars -> nothing
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
    # Use the trailing key of each dotted path; if leaf names collide, fall back
    # to the full compact path so run_names stay unique.
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
    """Return the top-level key holding the per-model training configs."""
    for key in ("defense_model_train_configs", "model_train_configs", "models_train_configs"):
        if isinstance(raw.get(key), dict):
            return key
    return None


def expand_model_config(model_cfg):
    """
    Expand list-valued scalar parameters *within a single model configuration*.

    Returns a list of ``(effective_cfg, varied_dict)`` where ``effective_cfg`` is
    a deep copy of ``model_cfg`` with one combination applied, and ``varied_dict``
    maps model-relative dotted path-strings to the chosen scalar value.

    The Cartesian product is computed only over parameters belonging to this one
    model, so hyperparameters of other models are never combined here.  When the
    model declares no list-valued parameters, a single ``(deepcopy, {})`` entry
    is returned so the model still produces exactly one run.
    """
    list_params = _find_list_params(model_cfg)
    if not list_params:
        return [(copy.deepcopy(model_cfg), {})]

    # Deterministic ordering for reproducible combination ordering.
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
    """
    Build one independent, self-contained run plan per (model, hyperparameter
    combination).

    The Cartesian product of list-valued parameters is generated separately for
    every model inside the model-config section, so parameters belonging to
    different models are never combined.  Each returned plan is:

    * ``eff`` -- the complete effective configuration: the original top-level
      (global) configuration plus the *single* selected model carrying one of
      that model's hyperparameter combinations.  No other model is present, so
      no model leaks into another model's run.
    * ``run_name`` -- ``<base>_<hp_suffix>`` (or ``<base>`` when the model has no
      varying parameters), uniquely naming the run.
    * ``signature`` -- stable run identity derived from the model name and that
      model's own effective hyperparameter configuration only.

    Returns a list of plan dicts::

        {"model_name", "run_name", "eff", "varied", "signature"}

    A model with ``n`` combinations produces exactly ``n`` plans; the total
    number of plans is the sum of all per-model combinations.
    """
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
                # Prefix varied paths with the model location so CSV columns are
                # unambiguous across models.
                varied = {
                    f"{section_key}.{model_name}.{k}": v
                    for k, v in model_varied.items()
                }
                hp_suffix = _make_hp_suffix(varied)
                run_name = f"{base_name}_{hp_suffix}" if hp_suffix else base_name

                combo_cfg["run_name"] = run_name

                # Complete effective configuration: global top-level config +
                # the single selected model with this combination applied.
                eff = copy.deepcopy(global_cfg)
                eff[section_key] = {model_name: combo_cfg}

                # Stable identity based on the model name + this model's own
                # hyperparameter configuration only (never other models).
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
    """Return a copy of *cfg* without the HPS-only top-level keys."""
    out = copy.deepcopy(cfg)
    for k in HPS_INTERNAL_KEYS:
        out.pop(k, None)
    return out


# ---------------------------------------------------------------------------
#  Train-index loading
# ---------------------------------------------------------------------------

def _collect_train_pkl_paths(raw):
    paths = []
    top = raw.get("train_pkl_path")
    if top:
        paths.append(top)
    section = raw.get("defense_model_train_configs") or raw.get(
        "model_train_configs"
    ) or raw.get("models_train_configs")
    if isinstance(section, dict):
        for cfg in section.values():
            pkl = None
            if isinstance(cfg, dict):
                pkl = cfg.get("pkl_train")
            elif isinstance(cfg, list):
                for c in cfg:
                    if isinstance(c, dict):
                        pkl = c.get("pkl_train")
                        if pkl:
                            break
            if pkl and pkl not in paths:
                paths.append(pkl)
    return paths


def _load_train_indexes(raw):
    train_set = set()
    for pkl_path in _collect_train_pkl_paths(raw):
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            idx = data.get("idx_metadata", []) if isinstance(data, dict) else []
            train_set.update(int(i) for i in idx)
        except Exception as e:
            log_warn(f"Could not load train indexes from {pkl_path}: {e}")
    return sorted(train_set)


# ---------------------------------------------------------------------------
#  Temp main-config writer
# ---------------------------------------------------------------------------

def _write_temp_main_config(effective_dict):
    fd, temp_path = tempfile.mkstemp(prefix="hps-main-", suffix=".yaml")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(effective_dict, f, sort_keys=False)
    except Exception:
        Path(temp_path).unlink(missing_ok=True)
        raise
    return temp_path


# ---------------------------------------------------------------------------
#  Configuration signature
# ---------------------------------------------------------------------------

def _canonical(obj):
    return json.dumps(obj, sort_keys=True, default=str)


def _config_signature(model_name, effective_dict):
    payload = {"model": model_name, "config": effective_dict}
    h = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
    return h


# ---------------------------------------------------------------------------
#  Per-run reproducible subsampling
# ---------------------------------------------------------------------------

def _subsampling_seed(hps_split_seed, effective_dict, model_name):
    payload = {"seed": hps_split_seed, "config": effective_dict, "model": model_name}
    h = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def _draw_run_subset(pool_questions, hps_run_samples, hps_split_seed,
                     effective_dict, model_name):
    pool_size = len(pool_questions)
    if hps_run_samples is None or hps_run_samples >= pool_size:
        return list(pool_questions)
    rng = np.random.default_rng(_subsampling_seed(hps_split_seed, effective_dict, model_name))
    chosen = rng.choice(pool_size, size=hps_run_samples, replace=False)
    chosen.sort()
    return [pool_questions[int(i)] for i in chosen]


# ---------------------------------------------------------------------------
#  Stats flattening -> CSV rows
# ---------------------------------------------------------------------------

def _flatten_stats_rows(stats, max_rounds, base_row):
    """
    Produce one CSV row per topology.

    ``stats`` is the list returned by ``parse_stats_single_model`` (one dict per
    topology).
    """
    rows = []
    for topo_result in stats:
        row = dict(base_row)
        topo = topo_result.get("topology", "unknown")
        row["topology"] = topo
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


# ---------------------------------------------------------------------------
#  CSV I/O
# ---------------------------------------------------------------------------

BASE_COLUMNS = [
    "config_signature",
    "model_name",
    "run_name",
    "topology",
    "total_questions",
    "correct_answers",
    "overall_accuracy",
    "overall_AUROC",
    "effective_config",
]


def _read_completed_signatures(csv_path):
    signatures = set()
    if not Path(csv_path).exists():
        return signatures
    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and "config_signature" in reader.fieldnames:
                for row in reader:
                    sig = row.get("config_signature")
                    if sig:
                        signatures.add(sig)
    except Exception as e:
        log_warn(f"Could not read existing CSV ({csv_path}): {e}. Starting fresh.")
        return set()
    return signatures


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


# ---------------------------------------------------------------------------
#  Logging helpers
# ---------------------------------------------------------------------------

def _log_combo_header(idx, total, base_name, varied, run_name, t0,
                      elapsed_prev, completed_prev):
    log_section(f"HP Search [{idx}/{total}]: {base_name}")
    if varied:
        for k, v in sorted(varied.items()):
            log_config(k, v)
    else:
        log_info("No varied parameters for this configuration.")
    log_config("run_name", run_name)
    if completed_prev > 0:
        done = idx - 1
        avg = elapsed_prev / done if done else 0
        remaining = (total - done) * avg
        log_info(
            f"Progress: {done}/{total} done | elapsed {fmt_seconds(elapsed_prev)} "
            f"| ETA ~{fmt_seconds(remaining)}"
        )


def _log_effective_config(eff):
    """Expose the complete effective configuration for a run for debugging."""
    log_info("Effective configuration for this run:")
    try:
        text = yaml.safe_dump(eff, sort_keys=False, default_flow_style=False)
        for line in text.strip().splitlines():
            log_info(f"    {line}")
    except Exception:
        log_info(f"    {_canonical(eff)}")


# ---------------------------------------------------------------------------
#  Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter search over the evaluation configuration."
    )
    parser.add_argument("config_file", type=str, help="Path to the HPS configuration file.")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing CSV / index pickle and start fresh.",
    )
    args = parser.parse_args()

    log_section("HPS Configuration Loading")
    log_info(f"Config file: {args.config_file}")

    with open(args.config_file, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # -- HPS knobs ---------------------------------------------------------
    hps_total_samples = raw.get("hps_total_samples")
    hps_run_samples = raw.get("hps_run_samples")
    hps_split_seed = raw.get("hps_split_seed", 42)
    index_pkl = raw.get("index_pkl")
    results_csv = raw.get("results_csv")

    if hps_total_samples is None:
        raise ValueError("Configuration must define 'hps_total_samples'.")
    if index_pkl is None:
        raise ValueError("Configuration must define 'index_pkl'.")
    if results_csv is None:
        raise ValueError("Configuration must define 'results_csv'.")
    hps_total_samples = int(hps_total_samples)
    hps_run_samples = int(hps_run_samples) if hps_run_samples is not None else None
    hps_split_seed = int(hps_split_seed)

    log_config("hps_total_samples", hps_total_samples)
    log_config("hps_run_samples", hps_run_samples)
    log_config("hps_split_seed", hps_split_seed)
    log_config("index_pkl", index_pkl)
    log_config("results_csv", results_csv)

    if args.clean:
        for p in (results_csv, index_pkl):
            if p and Path(p).exists():
                Path(p).unlink()
                log_info(f"--clean: removed {p}")

    # -- Train indexes (for pool exclusion) --------------------------------
    train_indexes = _load_train_indexes(raw)
    log_info(f"Loaded {len(train_indexes)} training indices (excluded from pool).")

    # -- Expand hyperparameter combinations (independently per model) ------
    run_plans = build_run_plans(raw)
    total_plans = len(run_plans)
    # Union of all varied-parameter column names (stable ordering).
    varied_key_set = set()
    for plan in run_plans:
        varied_key_set.update(plan["varied"].keys())
    varied_cols = sorted(varied_key_set)

    # max_rounds is constant across configs (read from the base live cfg).
    base_live = raw.get("live_evaluation_config", {})
    max_rounds = int(base_live.get("max_rounds", 0))
    metric_cols = _metric_columns(max_rounds)
    fieldnames = list(BASE_COLUMNS) + varied_cols + metric_cols

    log_info(
        f"Hyperparameter expansion: {total_plans} run(s) across models "
        f"({len(varied_cols)} varied parameter column(s))."
    )
    for plan in run_plans:
        log_info(
            f"  - {plan['model_name']} | {plan['run_name']} | "
            f"signature={plan['signature'][:12]}"
        )

    results_csv_path = Path(results_csv)
    completed_signatures = _read_completed_signatures(results_csv_path)
    if completed_signatures:
        log_info(
            f"Resume: {len(completed_signatures)} completed configuration(s) "
            f"found in {results_csv_path}"
        )
        log_info(f"Location of CSV: {results_csv_path}")
        log_info(f"Location of index pickle: {Path(index_pkl)}")

    csv_file, csv_writer = _open_csv_writer(results_csv_path, fieldnames)

    # -- Build the fixed HPS evaluation pool ONCE ---------------------------
    # The pool depends only on the question loader config + training indices,
    # which are assumed constant across HP combinations.
    base_live_cfg = load_config_from_path(args.config_file).live_evaluation_config
    pool_loader, pool_indices = build_hps_pool_loader(
        base_live_cfg,
        train_indexes,
        hps_total_samples,
        hps_split_seed,
        index_pkl,
    )
    pool_questions = list(pool_loader.get_formatted_questions())
    log_info(f"HPS pool ready: {len(pool_questions)} questions.")
    log_info(f"Location of index pickle: {Path(index_pkl)}")
    log_info(f"Location of CSV: {results_csv_path}")

    # Shared text-processor (loaded once, reused across configurations when
    # the text-processor path/class does not change).
    from EvaluationDebateLoop import load_class_from_path as _load_cls
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

            if signature in completed_signatures:
                log_section(f"HP Search [{idx}/{total_plans}]: {model_name} -- SKIPPED")
                log_info(f"Skipping run (already present in CSV): {run_name}")
                if varied:
                    for k, v in sorted(varied.items()):
                        log_config(k, v)
                continue

            # Write the effective config to a temp main-config file.
            temp_main = _write_temp_main_config(eff)
            temp_paths.append(temp_main)

            _log_combo_header(
                idx, total_plans, model_name, varied, run_name,
                overall_t0, elapsed_accum, done_before
            )
            _log_effective_config(eff)

            combo_t0 = time()
            model_instance = None
            combo_model_temps = []
            try:
                # Load the effective config as a namespace.
                config = load_config_from_path(temp_main)

                # Per-run reproducible subset of the pool.
                subset = _draw_run_subset(
                    pool_questions, hps_run_samples, hps_split_seed,
                    eff, model_name,
                )
                log_info(
                    f"Per-run subset: {len(subset)}/{len(pool_questions)} questions "
                    f"(seed derived from hps_split_seed={hps_split_seed})"
                )

                # Build / reuse the text processor.
                live_cfg = config.live_evaluation_config
                # Mutate live config question counts so run_debate_with_defense
                # uses the whole subset.
                live_cfg.num_questions = len(subset)
                live_cfg.n_questions_on_random_topo = len(subset)

                tp_key = (
                    getattr(live_cfg, "text_processor_path", None),
                    getattr(live_cfg, "text_processor_class_name", None),
                )
                if cached_tp is None or tp_key != cached_tp_key:
                    textProcessor = _load_cls(
                        live_cfg.text_processor_path,
                        live_cfg.text_processor_class_name,
                    )
                    cached_tp = textProcessor(device="cpu")
                    cached_tp_key = tp_key
                    log_info(f"Text processor loaded: {cached_tp_key}")

                # Build the HPS orchestrator.
                orchestrator = LiveDebateOrchestrationHPS(
                    live_cfg,
                    pool_loader,
                    text_processor=cached_tp,
                    train_indexes=train_indexes,
                )

                # Resolve topologies for this run's live config.
                log_section(f"Topology Resolution [{idx}/{total_plans}]")
                topologies = resolve_topologies(
                    SimpleNamespace(live_evaluation_config=live_cfg),
                    temp_main,
                )

                # Build / train the model(s) for this effective config.
                # Only the single model selected for this run is embedded, so no
                # other model is instantiated or evaluated as part of this run.
                embedded = load_embedded_model_configs(temp_main)
                models = get_models_from_path(config.models_directory, embedded)

                if not models:
                    log_warn(
                        f"No trainable model resolved for '{model_name}' in "
                        f"{config.models_directory}; skipping this run."
                    )
                    continue

                for _mn, _mi in models.items():
                    if _mi.get("temp_config_path"):
                        combo_model_temps.append(_mi["temp_config_path"])

                for _loaded_run_name, model_info in models.items():
                    log_section(f"Training [{idx}/{total_plans}]: {model_name}")
                    train_t0 = time()
                    # Pass the top-level train_pkl_path to the training code
                    # exactly like MainEvaluation, so the complete global
                    # configuration reaches the model's training.
                    metrics, model_instance = model_info["master"]._run(
                        getattr(config, "train_pkl_path", None)
                    )
                    computed_threshold = (
                        metrics.get("computed_threshold") if isinstance(metrics, dict) else None
                    )
                    effective_name = run_name
                    if computed_threshold is not None:
                        effective_name = _update_name_with_threshold(
                            run_name, computed_threshold
                        )
                        log_info(
                            f"Threshold computed: {computed_threshold:.6f} "
                            "(config default overridden)"
                        )
                    log_info(f"Training completed in {fmt_seconds(time() - train_t0)}")

                    log_section(f"Evaluating [{idx}/{total_plans}]: {effective_name}")
                    eval_t0 = time()
                    traces = orchestrator.run_debate_with_defense(
                        subset, model_instance, topologies
                    )
                    stats = orchestrator.parse_stats_single_model(traces)
                    log_info(
                        f"Evaluation completed in {fmt_seconds(time() - eval_t0)}"
                    )

                    print_stats_table(stats, model_name=effective_name)

                    base_row = {
                        "config_signature": signature,
                        "model_name": model_name,
                        "run_name": effective_name,
                        "effective_config": _canonical(eff),
                    }
                    for col, val in varied.items():
                        base_row[col] = val

                    rows = _flatten_stats_rows(stats, max_rounds, base_row)
                    for r in rows:
                        csv_writer.writerow(r)
                    csv_file.flush()
                    try:
                        os.fsync(csv_file.fileno())
                    except OSError:
                        pass
                    log_info(
                        f"Results flushed to CSV: {results_csv_path} "
                        f"({len(rows)} topology row(s))"
                    )
                    completed_signatures.add(signature)

                    del traces, stats
                    _cleanup_model(model_instance)
                    model_instance = None
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
                log_error(
                    f"Run [{idx}/{total_plans}] failed: {e}"
                )
                for line in traceback.format_exc().strip().splitlines():
                    log_error(line)
                log_warn(
                    "This run was NOT marked as completed. "
                    "It will be retried on the next run."
                )
                continue
            finally:
                # Remove the per-run temp main config.
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
        # Clean up any model temp config files left by get_models_from_path.
        # (Those are tracked inside get_models_from_path; ensure stragglers die.)
        for p in list(temp_paths):
            Path(p).unlink(missing_ok=True)
        total_elapsed = time() - overall_t0
        log_section("HPS Search Finished")
        log_info(f"Total elapsed: {fmt_seconds(total_elapsed)}")
        log_info(f"Results CSV: {results_csv_path}")
        log_info(f"Index pickle: {Path(index_pkl)}")


if __name__ == "__main__":
    main()