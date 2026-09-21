"""
MainEvaluation.py -- multi-dataset defense benchmarking.

Multi-dataset defense benchmarking for GAMMAF:

* **Combined training**: every defense model is trained once on the combined
  dataset pickle produced by ``TrainDataGeneration.py`` (all datasets
  live in the same ``data`` list).
* **Leakage-safe per-tag evaluation**: for each configured evaluation dataset
  tag the orchestrator excludes (a) the indexes used to *train on that same
  tag* and (b) the indexes selected by the HPS pool for that same tag, so
  training/eval leakage is impossible.
* **Per-dataset results**: evaluation results are saved in separate files per
  dataset tag (``<output_dir>/<tag>/<model>.json``); the configured
  ``output_file`` holds a small summary mapping tags/models to those files.
* **Consolidated hyperparameter search** (``--hps``), which lives in this same
  file and in ``EvaluationDebateLoop.py``.  The search saves the selected HPS
  indexes **per dataset tag**.

All configuration loading and validation is delegated to
``EvaluationConfigCheck.py``; this module only consumes the normalised config.

Config schema (see ``config-examples/evaluation-config.yaml``)::

    models_directory: defense-models
    output_file: results/summary.json
    train_pkl_path: data/train-data.pkl     # optional per-model fallback

    llm: {timeout, llm_max_retries, max_concurrent_inference}
    debate: {num_agents, num_malicious_agents, malicious_seed, max_rounds,
             consensus_threshold, check_consensus_only_unflagged,
             no_consensus_check, new_random_each_question, random_topo_seed,
             density_range_for_random_topo}

    datasets:
      - tag: MMLU
        loader_tag: null
        num_questions: 20
        num_questions_on_random_topo: 20
        questions_random_seed: 28
        ma_dataset_path: null
        hps_indexes: null

    evaluation:
      questions_path: DatasetManager.py
      questions_class_name: null
      python_seed: 28
      numpy_seed: 28
      answer_seed: 28
      top_k_defense: 2
      no_defense_baseline: true
      save_traces: false
      debug_mode: false
      static_adjacency_mode: false
      topologies_file: null
      topologies_from_pkl: null

    hyperparameter_search:          # optional section => enables --hps mode
      total_samples: 100
      run_samples: 40
      split_seed: 42
      index_pkl: hps/index.pkl
      index_pkl_dir: hps/indexes
      results_csv: hps/results.csv

    defense_model_train_configs:    # one entry per defense model file stem
      BlindGuard: {...}
"""

import argparse
import csv
import gc
import importlib.util
import inspect
import json
import os
import pickle
import traceback
from pathlib import Path
from time import time

import numpy as np

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
from Utils import AttrDict


# ---------------------------------------------------------------------------
#  Consolidated evaluation loop
# ---------------------------------------------------------------------------

from EvaluationDebateLoop import (
    LiveDebateOrchestration,
    build_hps_pool_loader,
    draw_hps_run_subset,
)
from EvaluationConfigCheck import (
    build_run_plans,
    load_evaluation_config,
    write_effective_config,
    write_model_config,
)


# ---------------------------------------------------------------------------
#  Topology helpers
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


def resolve_topologies(config):
    topologies_file = config.evaluation.topologies_file
    if topologies_file:
        with open(topologies_file, "r", encoding="utf-8") as f:
            log_info(f"Loading topologies from file: {topologies_file}")
            return json.load(f)

    pkl_path = config.evaluation.topologies_from_pkl
    if pkl_path:
        topologies = load_topologies_from_pickle(Path(pkl_path))
        if topologies:
            log_info(f"Loaded {len(topologies)} topologies from pickle: {pkl_path}")
            return topologies
        log_warn(f"No topologies found in pickle: {pkl_path}. Falling back to generated topologies.")

    generated = generate_topologies(config.debate.num_agents)
    if config.debate.new_random_each_question:
        generated["random"] = None
    log_info("Using generated topologies from debate.num_agents")
    return generated


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
            temp_config_path = write_model_config(run_name, model_cfg)
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
#  Exclusion indexes
# ---------------------------------------------------------------------------

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


def _load_hps_indexes_for_entry(entry):
    """Read the HPS exclusion indexes for one evaluation dataset entry."""
    path = entry.get("hps_index_path")
    if not path or not Path(path).exists():
        return set()

    try:
        flat, per_tag = _parse_per_tag_index_pickle(path)
    except Exception as e:
        log_warn(f"Could not read HPS indexes from {path}: {e}")
        return set()

    if per_tag:
        indexes = _indexes_for_entry(per_tag, entry)
        if indexes:
            log_info(f"HPS index exclusion for '{entry['tag']}': {len(indexes)} index(es) from {path}")
        return indexes
    if flat:
        log_info(f"HPS index exclusion for '{entry['tag']}': {len(flat)} index(es) from {path}")
        return set(flat)
    return set()


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
    train_indexes_by_tag,
    output_path,
):
    train_indexes = _indexes_for_entry(train_indexes_by_tag, entry)
    hps_indexes = _load_hps_indexes_for_entry(entry)
    excluded = set(train_indexes) | set(hps_indexes)

    log_info(
        f"Evaluating '{model_label}' on dataset '{entry.tag}' "
        f"(loader TAG={entry.loader_tag}); excluded indexes: "
        f"{len(train_indexes)} training + {len(hps_indexes)} HPS = {len(excluded)} total"
    )

    orchestrator = LiveDebateOrchestration(
        config,
        entry,
        train_indexes=sorted(train_indexes),
        excluded_indexes=sorted(hps_indexes),
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

    models = get_models_from_path(config.models_directory, config.defense_model_train_configs)

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
    eval_entries = config.datasets
    eval_tags = [entry.tag for entry in eval_entries]
    log_info(
        "Evaluation datasets: "
        + ", ".join(f"{e.tag}->{e.loader_tag}" for e in eval_entries)
    )

    train_pkl_path = config.train_pkl_path
    train_indexes_by_tag = _load_train_indexes_per_tag(train_pkl_path)
    if train_indexes_by_tag:
        log_info(
            f"Loaded training indexes for {len(train_indexes_by_tag)} tag(s) "
            f"from {train_pkl_path}."
        )

    try:
        log_section("Topology Resolution")
        topologies = resolve_topologies(config)

        summary.setdefault("train_pkl_path", str(train_pkl_path) if train_pkl_path else None)
        summary["datasets"] = [
            {"tag": e.tag, "loader_tag": e.loader_tag} for e in eval_entries
        ]

        if config.evaluation.no_defense_baseline and \
                "no_defense_baseline" not in summary.get("completed_runs", []):
            log_section("No-Defense Baseline")
            baseline_missing = [
                e for e in eval_entries
                if "no_defense_baseline" not in summary.get("per_dataset_results", {}).get(e.tag, {})
            ]
            if baseline_missing:
                for entry in baseline_missing:
                    t0 = time()
                    train_indexes = _indexes_for_entry(train_indexes_by_tag, entry)
                    hps_indexes = _load_hps_indexes_for_entry(entry)
                    orchestrator = LiveDebateOrchestration(
                        config,
                        entry,
                        train_indexes=sorted(train_indexes),
                        excluded_indexes=sorted(hps_indexes),
                    )
                    questions = orchestrator.dataloader.get_formatted_questions()
                    baseline_traces = orchestrator.run_debate_no_defense(questions, topologies)
                    baseline_stats = orchestrator.parse_stats_single_model(baseline_traces)
                    used_indexes = [int(i) for i in list(getattr(orchestrator.dataloader, "indexes", []))]
                    payload = {
                        "dataset_tag": entry.tag,
                        "loader_tag": entry.loader_tag,
                        "model": "no_defense_baseline",
                        "train_excluded_indexes": sorted(int(i) for i in train_indexes),
                        "hps_excluded_indexes": sorted(int(i) for i in hps_indexes),
                        "used_indexes": used_indexes,
                        "n_used_indexes": len(used_indexes),
                        "results": baseline_stats,
                    }
                    per_tag_path = _per_tag_result_path(output_path, entry.tag, "no_defense_baseline")
                    per_tag_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(per_tag_path, "w", encoding="utf-8") as f:
                        json.dump(payload, f, indent=2)
                    _register_result(
                        summary, entry.tag, "no_defense_baseline",
                        per_tag_path, train_indexes, hps_indexes,
                    )
                    elapsed = time() - t0
                    timing[f"no_defense_baseline::{entry.tag}"] = elapsed
                    print_stats_table(baseline_stats, model_name=f"no_defense_baseline [{entry.tag}]")
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
                    tag = entry.tag
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
                        train_indexes_by_tag,
                        output_path,
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


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, default=str)


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


def _run_hps(config, config_file, parsed_args):
    log_section("Hyperparameter Search (multi-dataset)")

    settings = config.hyperparameter_search
    if settings is None:
        raise ValueError(
            "Hyperparameter search requires a 'hyperparameter_search' section in the "
            "evaluation configuration"
        )
    total_samples = settings.total_samples
    run_samples = settings.run_samples
    split_seed = settings.split_seed
    index_pkl = settings.index_pkl
    index_pkl_dir = Path(settings.index_pkl_dir)
    results_csv = Path(settings.results_csv)

    log_config("total_samples", total_samples)
    log_config("run_samples", run_samples)
    log_config("split_seed", split_seed)
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

    eval_entries = config.datasets
    eval_tags = [entry.tag for entry in eval_entries]
    log_info(
        "HPS evaluation datasets: "
        + ", ".join(f"{e.tag}->{e.loader_tag}" for e in eval_entries)
    )

    train_pkl_path = config.train_pkl_path
    train_indexes_by_tag = _load_train_indexes_per_tag(train_pkl_path)
    log_info(
        f"Loaded training indexes for {len(train_indexes_by_tag)} tag(s) "
        f"from {train_pkl_path}."
    )

    run_plans = build_run_plans(config_file)
    total_plans = len(run_plans)
    varied_key_set = set()
    for plan in run_plans:
        varied_key_set.update(plan["varied"].keys())
    varied_cols = sorted(varied_key_set)

    max_rounds = config.debate.max_rounds
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
        tag = entry.tag
        train_indexes = _indexes_for_entry(train_indexes_by_tag, entry)
        per_tag_index_path = index_pkl_dir / f"{_safe_filename(tag)}-index.pkl"
        pool_loader, pool_indices = build_hps_pool_loader(
            entry.loader_class,
            sorted(train_indexes),
            total_samples,
            split_seed,
            per_tag_index_path,
            ma_dataset_path=entry.ma_dataset_path,
            dataset_tag=tag,
            loader_tag=entry.loader_tag,
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
                "total_samples": total_samples,
                "run_samples": run_samples,
                "split_seed": split_seed,
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
                if (signature, entry.tag) not in completed_pairs
            ]
            if not pending_tags:
                log_section(f"HP Search [{idx}/{total_plans}]: {model_name} -- SKIPPED")
                log_info(f"Skipping run (already present in CSV): {run_name}")
                continue

            temp_main = write_effective_config(eff)
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
                combo_config = load_evaluation_config(temp_main)
                combo_config.debate.new_random_each_question = True

                tp_key = (
                    combo_config.text_processor_path,
                    combo_config.text_processor_class_name,
                )
                if cached_tp is None or tp_key != cached_tp_key:
                    textProcessor = load_class_from_path(
                        combo_config.text_processor_path,
                        combo_config.text_processor_class_name,
                    )
                    processor_kwargs = dict(combo_config.text_processor_kwargs)
                    processor_kwargs.setdefault("device", combo_config.text_processor_device)
                    cached_tp = textProcessor(**processor_kwargs)
                    cached_tp_key = tp_key
                    log_info(f"Text processor loaded: {cached_tp_key}")

                models = get_models_from_path(
                    combo_config.models_directory,
                    combo_config.defense_model_train_configs,
                )
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
                    or combo_config.train_pkl_path
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
                    tag = entry.tag
                    subset = draw_hps_run_subset(
                        pool_questions[tag], run_samples, split_seed, f"{signature}::{tag}"
                    )
                    subset_entry = AttrDict(dict(entry))
                    subset_entry.num_questions = len(subset)
                    subset_entry.num_questions_on_random_topo = len(subset)
                    topologies = {"random": None}

                    orchestrator = LiveDebateOrchestration(
                        combo_config,
                        subset_entry,
                        dataloader=pool_loaders[tag],
                        text_processor=cached_tp,
                        train_indexes=sorted(_indexes_for_entry(train_indexes_by_tag, entry)),
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
    return config.hyperparameter_search is not None


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
    config = load_evaluation_config(parsed_args.config_file)

    if parsed_args.hps or _hps_requested(config):
        _run_hps(config, parsed_args.config_file, parsed_args)
    else:
        _run_standard(config, parsed_args)
