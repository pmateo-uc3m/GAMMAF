from DebateConfigLoader import DebateConfig
from DebateDataGenerationLoop import DebateOrchestration
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib
import importlib.util
import os
import pickle
import json
from typing import Any, cast
from Utils import load_config
from tqdm import tqdm

from TextProcessingManager import RoundProcessor


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return {k: to_jsonable(v) for k, v in vars(obj).items()}
    return obj

def is_valid_debate(debate_data):
    """
    Checks if a debate result is complete and valid.
    """
    if debate_data is None:
        return False
    
    rounds = debate_data.get("debate_rounds", [])
    if not rounds:
        return False
        
    for round_data in rounds:
        if not round_data:
            return False
            
        for agent_resp in round_data:
            # Check answer validity
            ans = agent_resp.get("answer")
            if ans is None:
                return False
            if isinstance(ans, str) and not ans.strip():
                return False
                
            # Check reason validity (or presence of embeddings)
            reason = agent_resp.get("reason")
            has_embeddings = 'st_embedding' in agent_resp
            
            # If we have embeddings, we assume reason was valid before processing
            # If we don't have embeddings, reason must be valid
            if not has_embeddings:
                if reason is None:
                    return False
                if isinstance(reason, str) and not reason.strip():
                    return False
                    
    return True


def get_debate_invalid_reasons(debate_data):
    """Return a list of reason codes describing why a debate is invalid."""
    reasons = []

    if debate_data is None:
        return ["debate_is_none"]

    if not isinstance(debate_data, dict):
        return ["debate_not_dict"]

    rounds = debate_data.get("debate_rounds", [])
    if not rounds:
        reasons.append("missing_or_empty_rounds")
        return reasons

    for round_idx, round_data in enumerate(rounds):
        if not round_data:
            reasons.append(f"round_{round_idx}_empty")
            continue

        for agent_idx, agent_resp in enumerate(round_data):
            if not isinstance(agent_resp, dict):
                reasons.append(f"round_{round_idx}_agent_{agent_idx}_not_dict")
                continue

            ans = agent_resp.get("answer")
            if ans is None:
                reasons.append(f"round_{round_idx}_agent_{agent_idx}_missing_answer")
            elif isinstance(ans, str) and not ans.strip():
                reasons.append(f"round_{round_idx}_agent_{agent_idx}_empty_answer")

            reason = agent_resp.get("reason")
            has_embeddings = "st_embedding" in agent_resp
            if not has_embeddings:
                if reason is None:
                    reasons.append(f"round_{round_idx}_agent_{agent_idx}_missing_reason")
                elif isinstance(reason, str) and not reason.strip():
                    reasons.append(f"round_{round_idx}_agent_{agent_idx}_empty_reason")

    return reasons

def adjacency_matrix_symmetric(n, topology):
    if n < 1:
        raise ValueError("n must be >= 1")

    # Initialize n x n matrix with zeros
    A = [[0] * n for _ in range(n)]

    if topology == "chain":
        # i <-> i + 1 (Symmetric)
        for i in range(n - 1):
            A[i][i + 1] = 1
            A[i + 1][i] = 1

    elif topology == "star":
    # 0 <-> all other nodes (center of the star)
        for i in range(1, n):
            A[0][i] = 1
            A[i][0] = 1

        # Connect exterior nodes in a ring: 1-2-3-...-(n-1)-1
        for i in range(1, n):
            j = i + 1 if i < n - 1 else 1  # wrap last node to node 1
            A[i][j] = 1
            A[j][i] = 1


    elif topology == "tree":
        # Binary tree connections (Symmetric)
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

def generate_topologies(num_agents: int, random_config = None):
    topologies = {
        "tree" : adjacency_matrix_symmetric(num_agents, "tree"),
        "chain": adjacency_matrix_symmetric(num_agents, "chain"),
        "star" : adjacency_matrix_symmetric(num_agents, "star")
    }
    return topologies


def load_text_processor(args, config_path: str | None = None):
    """Load text processor class from config path/module + class name."""
    processor_class_name = getattr(args, "text_processor_class_name", "RoundProcessor")
    processor_path = getattr(args, "text_processor_path", None)

    # Backward compatible default.
    if processor_path is None:
        return RoundProcessor()

    # Accept Python module path (e.g., package.module) or script path (e.g., ./postprocessing.py).
    is_script_path = processor_path.endswith(".py") or os.path.sep in processor_path or "/" in processor_path
    if is_script_path:
        if os.path.isabs(processor_path):
            resolved_path = processor_path
        else:
            # Relative paths are always resolved from current working directory.
            resolved_path = os.path.abspath(processor_path)

        if not os.path.isfile(resolved_path):
            raise FileNotFoundError(
                f"Processor script not found: '{processor_path}'. Tried '{resolved_path}'"
            )

        spec = importlib.util.spec_from_file_location("dynamic_text_processor", resolved_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load processor module from path: {resolved_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(processor_path)

    processor_cls = getattr(module, processor_class_name, None)
    if processor_cls is None:
        raise AttributeError(
            f"Class '{processor_class_name}' not found in processor module '{processor_path}'."
        )

    processor_kwargs = getattr(args, "text_processor_kwargs", None)
    if not isinstance(processor_kwargs, dict):
        processor_kwargs = {}

    text_processor_device = getattr(args, "text_processor_device", None)
    if text_processor_device is not None and "device" not in processor_kwargs:
        processor_kwargs["device"] = text_processor_device

    try:
        return processor_cls(**processor_kwargs)
    except TypeError:
        if processor_kwargs:
            print(
                "[WARNING] text_processor_kwargs/text_processor_device not supported by processor class; "
                "falling back to default constructor."
            )
        return processor_cls()


def process_single_debate(
    debate: Any,
    processor: RoundProcessor,
):
    """Process one debate's rounds into embeddings; preserves invalid entries as-is."""
    # If debate is None, we skip processing (it will be filtered later if clean-data is on)
    if debate is None or not isinstance(debate, dict):
        return debate

    worked_rounds = []
    debate_rounds = debate.get("debate_rounds", [])
    if not isinstance(debate_rounds, list):
        return debate

    for round_data in debate_rounds:
        processed_round = processor.process_round(round_data)
        worked_rounds.append(processed_round)
    debate["debate_rounds"] = worked_rounds
    return debate



def main():
    from time import time
    t0 = time()
    arguments = argparse.ArgumentParser(description="XG-Guard Anomaly Detection with Graph Neural Networks")
    arguments.add_argument('config', type=str, default=None, help='Path to YAML config file with all parameters')    
    parsed_config = arguments.parse_args()
    args = load_config(parsed_config)

    # Support both nested schema (debate_config.*) and legacy flat keys.
    debate_cfg = getattr(args, "debate_config", args)
    n_agents = getattr(debate_cfg, "num_agents", getattr(args, "num_agents", None))
    if n_agents is None:
        raise ValueError("Missing number of agents in config: set debate_config.num_agents (or num_agents).")

    n_questions_random_topo = getattr(
        debate_cfg,
        "n_questions_random_topo",
        getattr(args, "n_questions_random_topo", 0),
    )
    n_questions_fixed = getattr(debate_cfg, "n_questions", getattr(args, "num_questions", 0))
    max_rounds = getattr(debate_cfg, "max_rounds", getattr(args, "max_rounds", 3))
    num_malicious = getattr(debate_cfg, "num_malicious", getattr(args, "num_malicious", 0))
    consensus_threshold = getattr(debate_cfg, "consensus_threshold", getattr(args, "consensus_threshold", 1.0))
    malicious_randomization_seed = getattr(
        debate_cfg,
        "malicious_randomization_seed",
        getattr(args, "random_malicious_seed", 42),
    )
    random_topo_seed = getattr(debate_cfg, "random_topo_seed", getattr(args, "random_topo", 24))

    density_cfg = getattr(debate_cfg, "density", None)
    density_min = getattr(density_cfg, "min", getattr(args, "density_min", 0.3))
    density_max = getattr(density_cfg, "max", getattr(args, "density_max", 0.7))

    base_question_seed = getattr(args, "questions_random_seed", getattr(args, "random_debate", 0))
    process_text = getattr(args, "process_text", False)
    clean_data = getattr(args, "clean_data", False)
    text_process_workers = int(getattr(args, "text_process_workers", 0) or 0)
    save_data_dir = getattr(args, "save_data_dir", "data")
    file_name = getattr(args, "file_name", "train-data.pkl")

    # random_config = {
    #     'num_topologies': args.num_random_topologies,
    #     'density': args.density,
    #     'rng': np.random.default_rng(args.random_topo)
    # }
    
    topologies = generate_topologies(n_agents, 
                                    #  random_config
                                     )
    if n_questions_random_topo > 0:
        topologies['random'] = [[0]*n_agents for _ in range(n_agents)]
    
    # if args.save_topologies:
    #     with open(args.save_topologies, 'w') as f:
    #         json.dump(topologies, f, indent=4)
    #     print(f"Topologies saved to {args.save_topologies}")
    # print(topologies)
    
    # Aqui ya tengo las topologias generadas en el diccionario
    # El siguiente paso es generar los datos para cada dataset y topologia (guardar los datasets por separado?)
    processor = None
    if process_text:
        processor = load_text_processor(args, getattr(parsed_config, "config", None))
    if process_text and processor is None:
        raise RuntimeError("process_text is enabled but no processor could be initialized.")
    all_results = []
    total_initial_debates = 0
    total_valid_debates = 0
    total_topologies = len(topologies)
    for i, (topo_name, adj_matrix) in enumerate(topologies.items(), start=1):
        # Use unique seed per topology so each gets different questions.
        topology_seed = base_question_seed + (i - 1)
        questions_for_topology = n_questions_random_topo if topo_name == "random" else n_questions_fixed

        print()
        print("=" * 72)
        print(f"[TOPOLOGY {i}/{total_topologies}] {topo_name.upper()}")
        print(f"  seed..............: {topology_seed}")
        print(f"  planned_questions.: {questions_for_topology}")
        print("=" * 72)
        
        config = DebateConfig(
            timeout=args.timeout,
            is_random_topology = True if topo_name=="random" else False,
            random_topology_data = {
                "seed": random_topo_seed,
                "density interval": (density_min, density_max)
            },
            max_rounds=max_rounds,
            number_of_agents=n_agents,
            number_malicious_agents=num_malicious,
            consensus_threshold=consensus_threshold,
            topology=adj_matrix,
            prompts_file=args.prompts if args.prompts is not None else "prompts_blindguard.json",
            malicious_randomization_seed=malicious_randomization_seed,
            parallel_questions=args.parallel_questions,
            parallel_agents=True,
            save_logs_json=False,
            save_logs_dir=f"debate_logs_{topo_name}",
            verbose=args.verbose,
            num_questions=questions_for_topology,
            questions_random_seed=args.questions_random_seed + (i - 1), # Could remove +(i-1) to repeat questions across topologies
            dataset_tag=args.dataset_tag, # TAG defined for the dataset class at dataset_loader
        )

        debate_orchestration = DebateOrchestration(config)
        results, _ = debate_orchestration.run_evaluation()
        if results is None:
            print(f"[WARNING] No results returned for topology {topo_name}; skipping.")
            all_results.append(
                {
                    "topology_name": topo_name,
                    "topology": adj_matrix,
                    "results": [],
                }
            )
            continue

        initial_count = len(results)
        total_initial_debates += initial_count
        # Aqui deberia eliminar las rondas en las rondas (o debate entero?) en los que haya None por timeout
        
        if clean_data:
            print('[INFO] Cleaning data: Removing debates with invalid/empty responses...')
            cleaned_results = []
            invalid_reason_counts = {}
            invalid_examples = []

            for debate_idx, debate in enumerate(results):
                invalid_reasons = get_debate_invalid_reasons(debate)
                if not invalid_reasons:
                    cleaned_results.append(debate)
                    continue

                for reason in invalid_reasons:
                    invalid_reason_counts[reason] = invalid_reason_counts.get(reason, 0) + 1

                if len(invalid_examples) < 5:
                    invalid_examples.append(
                        {
                            "debate_index": debate_idx,
                            "reasons": invalid_reasons[:6],
                        }
                    )

            results = cleaned_results
            removed_count = initial_count - len(results)
            print(f"[INFO] Cleaned data: Removed {removed_count} invalid/empty debates from topology {topo_name}")

            if removed_count > 0:
                print(f"[INFO] Clean report for topology {topo_name}:")
                print(f"  - Total debates: {initial_count}")
                print(f"  - Kept debates: {len(results)}")
                print(f"  - Removed debates: {removed_count}")
                print("  - Reason counts:")
                for reason, count in sorted(invalid_reason_counts.items(), key=lambda x: (-x[1], x[0])):
                    print(f"      * {reason}: {count}")

                if invalid_examples:
                    print("  - Example removed debates (first up to 5):")
                    for ex in invalid_examples:
                        reasons_str = ", ".join(ex["reasons"])
                        print(f"      * debate_index={ex['debate_index']}: {reasons_str}")

        if process_text:
            print("[INFO] Starting text processing on cleaned debates...")
            assert processor is not None
            total_to_process = len(results)

            # Auto workers: CPU count when unset; cap to avoid over-subscription.
            workers = text_process_workers if text_process_workers > 0 else min(8, os.cpu_count() or 1)
            processor_device = getattr(processor, "device", "cpu")

            # Keep GPU path sequential by default to avoid concurrent CUDA/model contention.
            if processor_device != "cpu":
                if workers > 1:
                    print(
                        f"[INFO] text_process_workers={workers} requested, but processor device is '{processor_device}'. "
                        "Falling back to sequential text processing for safety."
                    )
                workers = 1

            if workers <= 1:
                for debate in tqdm(
                    cast(list[Any], results),
                    total=total_to_process,
                    desc=f"Text Processing [{topo_name}]",
                    leave=False,
                ):
                    process_single_debate(debate, processor)
            else:
                print(f"[INFO] Processing debate text in parallel with {workers} workers...")
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    results = list(
                        tqdm(
                            executor.map(
                                lambda d: process_single_debate(d, processor),
                                cast(list[Any], results),
                            ),
                            total=total_to_process,
                            desc=f"Text Processing [{topo_name}]",
                            leave=False,
                        )
                    )

        total_valid_debates += len(results)
            
        all_results.append(
            {"topology_name": topo_name, 
             "topology": adj_matrix,
             "results": results}
            )
    
    # En este momento tengo los datos de rondas con el processing (embeddings)
    
    # Guardar los datos en pickle
    os.makedirs(save_data_dir, exist_ok=True)
    output_filepath = os.path.join(save_data_dir, file_name)
    with open(output_filepath, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"[INFO] Processed text data saved to {output_filepath}")
    elapsed_seconds = time() - t0
    elapsed_minutes = int(elapsed_seconds // 60)
    remaining_seconds = elapsed_seconds % 60
    print(f"[INFO] Total execution time: {elapsed_minutes} min {remaining_seconds:.2f} sec")

    avg_seconds_per_initial = (
        elapsed_seconds / total_initial_debates if total_initial_debates > 0 else 0.0
    )
    avg_seconds_per_valid = (
        elapsed_seconds / total_valid_debates if total_valid_debates > 0 else 0.0
    )

    print("[INFO] Timing report:")
    print(f"  - Initial debates/questions generated: {total_initial_debates}")
    print(f"  - Final valid debates kept: {total_valid_debates}")
    print(f"  - Average time per initial debate/question: {avg_seconds_per_initial:.4f} sec")
    print(f"  - Average time per final valid debate: {avg_seconds_per_valid:.4f} sec")

    timing_report = {
        "total_seconds": elapsed_seconds,
        "total_minutes": elapsed_minutes,
        "remaining_seconds": remaining_seconds,
        "initial_debates": total_initial_debates,
        "valid_debates": total_valid_debates,
        "avg_seconds_per_initial": avg_seconds_per_initial,
        "avg_seconds_per_valid": avg_seconds_per_valid,
    }
    report_filename = f"report-{file_name}"
    if not report_filename.lower().endswith(".json"):
        report_filename += ".json"
    report_filepath = os.path.join(save_data_dir, report_filename)
    with open(report_filepath, "w", encoding="utf-8") as report_file:
        json.dump(timing_report, report_file, indent=2)
    print(f"[INFO] Timing report saved to {report_filepath}")
    
    
    
    # Save config
    # config_filename = os.path.splitext(file_name)[0] + "-config.json"
    # config_filepath = os.path.join(save_data_dir, config_filename)
    # with open(config_filepath, 'w') as f:
    #     json.dump(to_jsonable(args), f, indent=4)
    # print(f"[INFO] Configuration saved to {config_filepath}")

if __name__ == "__main__":
    main()