"""
MainEvaluation-search.py  --  Hyperparameter search extension of MainEvaluation.py.

Automatically expands list-valued configuration parameters into Cartesian product
evaluations, generating one independent evaluation run per parameter combination.

Usage:  python MainEvaluation-search.py <config_file> [--clean]

When no configuration parameter is specified as a list, behaviour is identical
to that of MainEvaluation.py.
"""

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
import itertools
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

from MainEvaluation import (
    adjacency_matrix_symmetric,
    generate_topologies,
    _extract_adj,
    load_topologies_from_pickle,
    resolve_topologies,
    _write_temp_model_config,
    get_models_from_path,
    _update_name_with_threshold,
    _cleanup_model,
    _get_completed_run_names,
)


# ---------------------------------------------------------------------------
#  Hyperparameter expansion helpers
# ---------------------------------------------------------------------------

def _sanitize_name_value(v):
    """Convert a parameter value to a string safe for embedding in a run name."""
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, float):
        return f"{v}".replace(".", "-")
    if v is None:
        return "none"
    s = str(v)
    return s.replace(".", "-").replace(" ", "_")


def _make_hp_params_suffix(params):
    """Build a deterministic, human-readable suffix from a dict of varied parameters."""
    sorted_items = sorted(params.items(), key=lambda x: x[0])
    parts = []
    for k, v in sorted_items:
        pname = k.replace("_", "")
        pval = _sanitize_name_value(v)
        parts.append(f"{pname}{pval}")
    return "_".join(parts)


def _expand_single_config(cfg, base_name):
    """
    Expand a single config dict into a list of (config_dict, varied_params) tuples.

    If *any* top-level parameter value is a list, the Cartesian product of all
    list-valued parameters is generated.  Otherwise *cfg* is returned unchanged.
    """
    list_params = {}
    for k, v in cfg.items():
        if k == "run_name":
            continue
        if isinstance(v, list):
            if len(v) == 0:
                log_warn(f"Parameter '{k}' is an empty list \u2014 skipping")
                continue
            list_params[k] = v

    if not list_params:
        return [(dict(cfg), {})]

    keys = sorted(list_params.keys())
    values = [list_params[k] for k in keys]
    base_run_name = cfg.get("run_name", base_name)

    expanded = []
    for combo in itertools.product(*values):
        new_cfg = dict(cfg)
        varied = {}
        for k, v in zip(keys, combo):
            new_cfg[k] = v
            varied[k] = v

        suffix = _make_hp_params_suffix(varied)
        new_cfg["run_name"] = f"{base_run_name}_{suffix}"
        expanded.append((new_cfg, varied))

    return expanded


def load_expanded_model_configs(config_file_path):
    """
    Load model configs from YAML and expand any list-valued parameters.

    Returns
    -------
    expanded_configs : dict[str, list[dict]]
        Same structure as ``load_embedded_model_configs`` but with every
        list-valued parameter expanded into multiple individual configs.
    run_cfg_map : dict[str, dict]
        Maps expanded run_name -> expanded config dict.
    run_varied_map : dict[str, dict]
        Maps expanded run_name -> dict of only the parameters that varied.
    run_base_map : dict[str, str]
        Maps expanded run_name -> original YAML section name.
    """
    from MainEvaluation import load_embedded_model_configs

    base_configs = load_embedded_model_configs(config_file_path)

    expanded = {}
    run_cfg_map = {}
    run_varied_map = {}
    run_base_map = {}

    for model_name, configs in base_configs.items():
        all_expanded = []
        for cfg in configs:
            for exp_cfg, varied in _expand_single_config(cfg, model_name):
                rn = exp_cfg["run_name"]
                all_expanded.append(exp_cfg)
                run_cfg_map[rn] = exp_cfg
                run_varied_map[rn] = varied
                run_base_map[rn] = model_name
        expanded[model_name] = all_expanded

    total_original = sum(len(v) for v in base_configs.values())
    total_expanded = sum(len(v) for v in expanded.values())
    if total_expanded != total_original:
        log_info(
            f"Hyperparameter expansion: {total_original} base "
            f"\u2192 {total_expanded} evaluations "
            f"(+{total_expanded - total_original})"
        )

    return expanded, run_cfg_map, run_varied_map, run_base_map


# ---------------------------------------------------------------------------
#  JSON persistence
# ---------------------------------------------------------------------------

def _append_model_result_with_config(
    output_path: Path, model_name: str, stats, model_config: dict
) -> None:
    """Append model results and store the expanded configuration."""
    results = {}
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
        except (json.JSONDecodeError, OSError):
            log_warn("Could not read existing results file. Starting fresh.")
            results = {}

    results[model_name] = stats

    if "_configs" not in results:
        results["_configs"] = {}
    results["_configs"][model_name] = model_config

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


def _get_completed_run_names_filtered(output_path: Path) -> set:
    """Return completed run names, excluding internal meta-keys (starting with ``_``)."""
    names = _get_completed_run_names(output_path)
    return {n for n in names if not n.startswith("_")}


# ---------------------------------------------------------------------------
#  Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file", type=str, help="Path to the configuration file."
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing results and start fresh.",
    )
    parsed_args = parser.parse_args()
    config = load_config_from_path(parsed_args.config_file)

    log_section("Configuration Loading")
    log_info(f"Config file: {parsed_args.config_file}")

    embedded_model_configs, run_cfg_map, run_varied_map, run_base_map = (
        load_expanded_model_configs(parsed_args.config_file)
    )
    models = get_models_from_path(
        config.models_directory, embedded_model_configs
    )

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
        completed_run_names = _get_completed_run_names_filtered(output_path)

    try:
        log_section("Topology Resolution")
        topologies = resolve_topologies(config, parsed_args.config_file)
        liveEvaluator = LiveDebateOrchestration(
            config.live_evaluation_config
        )

        live_cfg = config.live_evaluation_config
        if getattr(live_cfg, "no_defense_baseline", False):
            if "no_defense_baseline" in completed_run_names:
                log_info(
                    f"Skipping no_defense_baseline \u2014 "
                    f"already present in {output_path.name}."
                )
            else:
                log_section("No-Defense Baseline")
                t0 = time()
                questions = liveEvaluator.dataloader.get_formatted_questions()
                baseline_traces = liveEvaluator.run_debate_no_defense(
                    questions, topologies
                )
                baseline_stats = liveEvaluator.parse_stats_single_model(
                    baseline_traces
                )
                _append_model_result_with_config(
                    output_path,
                    "no_defense_baseline",
                    baseline_stats,
                    {"type": "no_defense_baseline"},
                )
                elapsed = time() - t0
                timing["no_defense_baseline"] = elapsed
                print_stats_table(
                    baseline_stats, model_name="no_defense_baseline"
                )
                log_info(f"Elapsed: {fmt_seconds(elapsed)}")
                log_info(f"Results saved to {output_path}")
                del baseline_traces, baseline_stats
                gc.collect()

        if completed_run_names:
            total_planned = len(models)
            completed_models = [
                name for name in models if name in completed_run_names
            ]
            for name in completed_models:
                log_info(
                    f"Skipping '{name}' \u2014 "
                    f"already present in {output_path.name}."
                )
                del models[name]
            remaining = len(models)
            if remaining < total_planned:
                log_section(
                    f"Resuming: skipped "
                    f"{total_planned - remaining}/{total_planned} completed "
                    f"runs, {remaining} remaining"
                )

        total_models = len(models)
        if total_models == 0:
            log_info(
                "All planned models are already completed. Nothing to do."
            )
        else:
            log_info(f"Processing {total_models} model(s).")

        for idx, (model_name, model_info) in enumerate(
            models.items(), start=1
        ):
            model_t0 = time()
            model_instance = None
            try:
                varied = run_varied_map.get(model_name, {})
                base_name = run_base_map.get(model_name, model_name)
                has_hp = bool(varied)

                if has_hp:
                    params_desc = ", ".join(
                        f"{k}={v}" for k, v in sorted(varied.items())
                    )
                    log_section(
                        f"HP Search [{idx}/{total_models}]: "
                        f"{base_name}  |  {params_desc}"
                    )
                else:
                    log_section(
                        f"Model {idx}/{total_models}: {model_name}"
                    )

                log_config("config", model_info["config_path"])

                train_t0 = time()
                metrics, model_instance = model_info["master"]._run()
                effective_name = model_name
                computed_threshold = (
                    metrics.get("computed_threshold")
                    if isinstance(metrics, dict)
                    else None
                )
                if computed_threshold is not None:
                    effective_name = _update_name_with_threshold(
                        model_name, computed_threshold
                    )
                    log_info(
                        f"Threshold computed: {computed_threshold:.6f} "
                        "(config default overridden)"
                    )
                    if effective_name != model_name:
                        log_info(
                            f"Effective run name: {effective_name}"
                        )
                log_info(
                    f"Training completed in "
                    f"{fmt_seconds(time() - train_t0)}"
                )

                eval_t0 = time()
                log_section(
                    f"Evaluating Model {idx}/{total_models}: "
                    f"{effective_name}"
                )
                traces = liveEvaluator.run_evaluation_single_defense_model_all_topos(
                    model_instance, topologies
                )
                stats = liveEvaluator.parse_stats_single_model(traces)
                log_info(
                    f"Evaluation completed in "
                    f"{fmt_seconds(time() - eval_t0)}"
                )

                stored_config = run_cfg_map.get(model_name, {})
                _append_model_result_with_config(
                    output_path, effective_name, stats, stored_config
                )
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
                log_error(
                    f"Model '{model_name}' failed after "
                    f"{fmt_seconds(elapsed)}: {e}"
                )
                log_warn(
                    "Previously completed results are preserved. "
                    "Moving to next model."
                )
                continue

    except KeyboardInterrupt:
        log_warn(
            "KeyboardInterrupt received. "
            "All previously completed results have been saved."
        )
    except Exception as e:
        log_error(f"Unhandled exception: {e}")
        log_warn(
            "Previously completed results are preserved "
            "in the output file."
        )
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
            with open(
                report_path, "w", encoding="utf-8"
            ) as report_file:
                json.dump(timing, report_file, indent=2)
            print()
            print_timing_report(timing, total_elapsed)
            log_info(f"Timing report saved to {report_path}")
