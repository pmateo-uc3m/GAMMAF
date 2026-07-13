import argparse
import inspect
from pathlib import Path
import importlib.util
import os
import tempfile
from Utils import load_config_from_path
from EvaluationDebateLoop import LiveDebateOrchestration
import json
import pickle
from time import time
import yaml
import gc
from LoggingUtils import log_section, log_info, log_warn, log_error, log_done, log_config, fmt_seconds, print_stats_table, print_timing_report


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
                cfg.setdefault("run_name", model_name)
                normalized[model_name] = [cfg]
            elif isinstance(cfg, list):
                for i, c in enumerate(cfg):
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


def _append_model_result(output_path: Path, model_name: str, stats) -> None:
    results = {}
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
        except (json.JSONDecodeError, OSError):
            log_warn("Could not read existing results file. Starting fresh.")
            results = {}
    results[model_name] = stats
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


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


def _get_completed_run_names(output_path: Path) -> set:
    if not output_path.exists():
        return set()
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        if isinstance(results, dict):
            return set(results.keys())
        return set()
    except (json.JSONDecodeError, OSError):
        log_warn("Could not read existing results file. Starting fresh.")
        return set()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to the configuration file.")
    parser.add_argument("--clean", action="store_true", help="Delete existing results and start fresh.")
    parsed_args = parser.parse_args()
    config = load_config_from_path(parsed_args.config_file)

    log_section("Configuration Loading")
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
        completed_run_names = set()
    else:
        completed_run_names = _get_completed_run_names(output_path)

    try:
        log_section("Topology Resolution")
        topologies = resolve_topologies(config, parsed_args.config_file)
        liveEvaluator = LiveDebateOrchestration(config.live_evaluation_config)

        live_cfg = config.live_evaluation_config
        if getattr(live_cfg, "no_defense_baseline", False):
            if "no_defense_baseline" in completed_run_names:
                log_info(f"Skipping no_defense_baseline \u2014 already present in {output_path.name}.")
            else:
                log_section("No-Defense Baseline")
                t0 = time()
                questions = liveEvaluator.dataloader.get_formatted_questions()
                baseline_traces = liveEvaluator.run_debate_no_defense(questions, topologies)
                baseline_stats = liveEvaluator.parse_stats_single_model(baseline_traces)
                _append_model_result(output_path, "no_defense_baseline", baseline_stats)
                elapsed = time() - t0
                timing["no_defense_baseline"] = elapsed
                print_stats_table(baseline_stats, model_name="no_defense_baseline")
                log_info(f"Elapsed: {fmt_seconds(elapsed)}")
                log_info(f"Results saved to {output_path}")
                del baseline_traces, baseline_stats
                gc.collect()

        if completed_run_names:
            total_planned = len(models)
            completed_models = [name for name in models if name in completed_run_names]
            for name in completed_models:
                log_info(f"Skipping '{name}' \u2014 already present in {output_path.name}.")
                del models[name]
            remaining = len(models)
            if remaining < total_planned:
                log_section(f"Resuming: skipped {total_planned - remaining}/{total_planned} completed runs, {remaining} remaining")

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
                log_config(f"config", model_info["config_path"])

                train_t0 = time()
                metrics, model_instance = model_info["master"]._run()
                effective_name = model_name
                computed_threshold = metrics.get("computed_threshold") if isinstance(metrics, dict) else None
                if computed_threshold is not None:
                    effective_name = _update_name_with_threshold(model_name, computed_threshold)
                    log_info(f"Threshold computed: {computed_threshold:.6f} (config default overridden)")
                    if effective_name != model_name:
                        log_info(f"Effective run name: {effective_name}")
                log_info(f"Training completed in {fmt_seconds(time() - train_t0)}")

                eval_t0 = time()
                log_section(f"Evaluating Model {idx}/{total_models}: {effective_name}")
                traces = liveEvaluator.run_evaluation_single_defense_model_all_topos(
                    model_instance, topologies
                )
                stats = liveEvaluator.parse_stats_single_model(traces)
                log_info(f"Evaluation completed in {fmt_seconds(time() - eval_t0)}")

                _append_model_result(output_path, effective_name, stats)
                print_stats_table(stats, model_name=effective_name)
                log_info(f"Results saved to {output_path}")
                del traces, stats

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
                log_warn("Previously completed results are preserved. Moving to next model.")
                continue

    except KeyboardInterrupt:
        log_warn("KeyboardInterrupt received. All previously completed results have been saved.")
    except Exception as e:
        log_error(f"Unhandled exception: {e}")
        log_warn("Previously completed results are preserved in the output file.")
    finally:
        for model_info in models.values():
            temp_cfg = model_info.get("temp_config_path")
            if temp_cfg:
                Path(temp_cfg).unlink(missing_ok=True)

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
