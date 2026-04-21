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


def _print_section(title: str, width: int = 72):
    print()
    print("=" * width)
    print(f"[INFO] {title}")
    print("=" * width)


def _fmt_seconds(seconds: float) -> str:
    minutes = int(seconds // 60)
    rem_seconds = seconds % 60
    return f"{minutes}m {rem_seconds:.2f}s"


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
                print(f"[INFO] Loading topologies from file: {topologies_path}")
                return json.load(f)
        print(f"[WARNING] topologies_file not found: {topologies_path}. Falling back to generated topologies.")

    pkl_path = getattr(live_cfg, "topologies_from_pkl", None)
    if pkl_path:
        pkl_topologies_path = Path(pkl_path)
        if pkl_topologies_path.exists():
            topologies = load_topologies_from_pickle(pkl_topologies_path)
            if topologies:
                print(f"[INFO] Loaded {len(topologies)} topologies from pickle: {pkl_topologies_path}")
                return topologies
            print(f"[WARNING] No topologies found in pickle: {pkl_topologies_path}. Falling back to generated topologies.")
        else:
            print(f"[WARNING] topologies_from_pkl not found: {pkl_topologies_path}. Falling back to generated topologies.")

    n_agents = getattr(live_cfg, "num_agents", None)
    if n_agents is None:
        raise ValueError(
            "Cannot resolve topologies: provide live_evaluation_config.num_agents "
            "or a valid live_evaluation_config.topologies_file/topologies_from_pkl."
        )

    generated = generate_topologies(n_agents)
    if getattr(live_cfg, "new_random_each_question", False):
        generated["random"] = None
    print("[INFO] Using generated topologies from live_evaluation_config.num_agents")
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
            raise ValueError(
                f"'{section_name}' must be a mapping of model_name -> config dict."
            )
        return section

    raise ValueError(
        "Missing embedded defense train config section in main config. "
        "Add 'defense_model_train_configs' with one child section per defense model file name."
    )


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
        
        if hasattr(module, "Master"):
            cls = getattr(module, "Master")
            if inspect.isclass(cls):
                if module_name not in embedded_model_configs:
                    raise ValueError(
                        f"Missing embedded training config for defense model '{module_name}'. "
                        "Add a matching section under 'defense_model_train_configs' in main config."
                    )
                model_cfg = embedded_model_configs[module_name]
                if not isinstance(model_cfg, dict):
                    raise ValueError(
                        f"Embedded config for model '{module_name}' must be a mapping/dict."
                    )

                temp_config_path = _write_temp_model_config(module_name, model_cfg)
                models[module_name] = {
                    "master": cls(temp_config_path),
                    "config_path": f"embedded:defense_model_train_configs.{module_name}",
                    "temp_config_path": temp_config_path,
                }
                print(
                    f"[INFO] Loaded model {module_name} with embedded configuration "
                    f"from defense_model_train_configs.{module_name}."
                )
    return models

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to the configuration file.")
    parsed_args = parser.parse_args()
    config = load_config_from_path(parsed_args.config_file)
    embedded_model_configs = load_embedded_model_configs(parsed_args.config_file)
    models = get_models_from_path(config.models_directory, embedded_model_configs)
    overall_t0 = time()
    training_t0 = overall_t0
    try:
        total_models = len(models)
        for idx, (model_name, model_info) in enumerate(models.items(), start=1):
            _print_section(f"Training Model {idx}/{total_models}: {model_name}")
            print(f"  config........: {model_info['config_path']}")
            model_t0 = time()
            models[model_name]['evaluation metrics'], models[model_name]['model'] = model_info["master"]._run()
            print(f"  elapsed.......: {_fmt_seconds(time() - model_t0)}")

        training_elapsed = time() - training_t0
        print(f"[INFO] All models trained in {_fmt_seconds(training_elapsed)}")

        live_t0 = time()

        _print_section("Starting Live Defense Benchmarking")
        liveEvaluator = LiveDebateOrchestration(config.live_evaluation_config)
        models_list = [(model_name, model_info['model']) for model_name, model_info in models.items()]
        topologies = resolve_topologies(config, parsed_args.config_file)
        print(f"  models........: {len(models_list)}")
        print(f"  topologies....: {len(topologies)}")
        live_results = liveEvaluator._run(models_list, topologies)
        output_path = Path(config.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config.output_file, 'w') as f:
            json.dump(live_results, f, indent=4)
        print(f"[INFO] Live evaluation completed. Results saved to {config.output_file}")
        live_elapsed = time() - live_t0
        print(f"[INFO] Live evaluation elapsed: {_fmt_seconds(live_elapsed)}")

        total_elapsed = time() - overall_t0
        timing_report = {
            "training_seconds": training_elapsed,
            "live_evaluation_seconds": live_elapsed,
            "total_seconds": total_elapsed,
        }
        report_filename = f"report-{output_path.name}"
        if not report_filename.lower().endswith(".json"):
            report_filename += ".json"
        report_path = output_path.with_name(report_filename)
        with open(report_path, "w", encoding="utf-8") as report_file:
            json.dump(timing_report, report_file, indent=2)
        print(f"[INFO] Timing report saved to {report_path}")
    finally:
        for model_info in models.values():
            temp_cfg = model_info.get("temp_config_path")
            if temp_cfg:
                Path(temp_cfg).unlink(missing_ok=True)

    
    
    