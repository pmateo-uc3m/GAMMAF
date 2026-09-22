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

    defense_model_train_configs:    # one entry per defense model file stem
      BlindGuard: {...}
"""

import argparse
import gc
import importlib.util
import inspect
import json
import pickle
import traceback
from pathlib import Path
from time import time

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

from EvaluationDebateLoop import LiveDebateOrchestration
from EvaluationConfigCheck import (
    load_evaluation_config,
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
#  Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to the configuration file.")
    parser.add_argument("--clean", action="store_true", help="Delete existing results and start fresh.")
    parsed_args = parser.parse_args()
    config = load_evaluation_config(parsed_args.config_file)
    _run_standard(config, parsed_args)
