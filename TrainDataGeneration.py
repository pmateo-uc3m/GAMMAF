"""
TrainDataGeneration.py -- multi-dataset training-data generation.

Generates debate data for **several datasets in a single run**, while recording,
per dataset tag, the exact dataset indexes that were used ("used indexes") so
that later stages (training, hyperparameter search, evaluation) can avoid
train/eval leakage.

Config schema (see ``config-examples/generation-config.yaml``)::

    llm: {timeout, llm_max_retries, max_concurrent_inference}
    debate: {num_agents, num_malicious_agents, malicious_seed, max_rounds,
             consensus_threshold, random_topo_seed,
             density_range_for_random_topo}

    datasets:                       # list of dataset entries (multi-dataset mode)
      - tag: MMLUPRO                # config tag (resolved to a loader TAG)
        loader_tag: MMLUPro         # optional explicit loader TAG override
        num_questions: 10           # fixed-topology question count for this dataset
        num_questions_on_random_topo: 10 # random-topology question count
        questions_random_seed: 1    # per-dataset seed
        ma_dataset_path: ...        # optional per-dataset dataset path

    output_dir: data
    output_file: train_data.pkl
    process_text: true
    clean_debates: true

All configuration loading and validation is delegated to
``GenerationConfigCheck.py``.

Output pickle schema::

    {
        "data": [ {topology_name, topology, dataset_tag, results: [...]}, ... ],
        "idx_metadata": { config_tag: [used dataset indexes], ... },
        "idx_metadata_flat": [flat union of every used index],
        "dataset_tags": [config tags in generation order],
    }

Each debate additionally carries ``dataset_tag`` and ``dataset_index`` so a
debate can always be traced back to the dataset instance it was generated from.
"""

from GenerationConfigCheck import build_debate_config, load_generation_config
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib
import importlib.util
import os
import pickle
import json
from typing import Any, cast
from tqdm import tqdm
from LoggingUtils import log_section, log_info, log_warn, log_error, log_done, fmt_seconds, print_timing_report

from TextProcessingManager import RoundProcessor
from DebateDataGenerationLoop import DebateOrchestration


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return {k: to_jsonable(v) for k, v in vars(obj).items()}
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            return obj
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

            # Check message validity (or presence of embeddings)
            message = agent_resp.get("message")
            has_embeddings = 'st_embedding' in agent_resp

            # If we have embeddings, we assume message was valid before processing
            # If we don't have embeddings, message must be valid
            if not has_embeddings:
                if message is None:
                    return False
                if isinstance(message, str) and not message.strip():
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

            message = agent_resp.get("message")
            has_embeddings = "st_embedding" in agent_resp
            if not has_embeddings:
                if message is None:
                    reasons.append(f"round_{round_idx}_agent_{agent_idx}_missing_message")
                elif isinstance(message, str) and not message.strip():
                    reasons.append(f"round_{round_idx}_agent_{agent_idx}_empty_message")

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


def load_text_processor(config):
    """Instantiate the configured text processor class."""
    processor_class_name = config.text_processor_class_name
    processor_path = config.text_processor_path

    # Accept Python module path (e.g., package.module) or script path (e.g., ./postprocessing.py).
    is_script_path = processor_path.endswith(".py") or os.path.sep in processor_path or "/" in processor_path
    if is_script_path:
        if os.path.isabs(processor_path):
            resolved_path = processor_path
        else:
            # Relative paths are always resolved from current working directory.
            resolved_path = os.path.abspath(processor_path)

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

    processor_kwargs = dict(config.text_processor_kwargs)
    if config.text_processor_device is not None:
        processor_kwargs.setdefault("device", config.text_processor_device)

    return processor_cls(**processor_kwargs)


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


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _generate_for_dataset(dataset_entry, config, processor):
    """Generate all topologies for ONE dataset entry.

    Returns ``(dataset_results, used_dataset_indexes, stats)`` where
    ``dataset_results`` is the list of ``{topology_name, topology, dataset_tag,
    results}`` records, ``used_dataset_indexes`` are the dataset indexes that
    actually survived cleaning, and ``stats`` holds the debate counters.
    """
    resolved_tag = dataset_entry.loader_tag
    tag_label = dataset_entry.tag
    n_questions_fixed = dataset_entry.num_questions
    n_questions_random_topo = dataset_entry.num_questions_on_random_topo
    base_question_seed = dataset_entry.questions_random_seed

    # TA (InjecAgent) semantics: an empty agent answer means no tool was called,
    # which is a safe outcome, so debate cleaning is never applied to TA.
    is_ta = str(resolved_tag).upper() == "TA"

    log_section(f"Dataset {tag_label} (loader TAG={resolved_tag})")
    log_info(
        f"Planned questions: fixed={n_questions_fixed}, random={n_questions_random_topo}, "
        f"seed={base_question_seed}"
    )

    topologies = generate_topologies(config.debate.num_agents)
    if n_questions_random_topo > 0:
        topologies['random'] = [[0] * config.debate.num_agents for _ in range(config.debate.num_agents)]

    dataset_results = []
    used_global_indexes = []
    dataset_initial_debates = 0
    dataset_valid_debates = 0
    total_topologies = len(topologies)

    for i, (topo_name, adj_matrix) in enumerate(topologies.items(), start=1):
        # Use unique seed per topology so each gets different questions.
        topology_seed = base_question_seed + (i - 1)
        questions_for_topology = (
            n_questions_random_topo if topo_name == "random" else n_questions_fixed
        )

        log_section(
            f"[{tag_label}] Topology {i}/{total_topologies}: {topo_name.upper()}"
        )
        log_info(f"Seed: {topology_seed}")
        log_info(f"Planned questions: {questions_for_topology}")

        runtime_config = build_debate_config(
            config,
            dataset_entry,
            topo_name,
            adj_matrix,
            questions_for_topology,
            topology_seed,
        )

        debate_orchestration = DebateOrchestration(runtime_config)
        results, _ = debate_orchestration.run_evaluation()
        if results is None:
            log_warn(f"No results returned for topology {topo_name}; skipping.")
            dataset_results.append(
                {
                    "topology_name": topo_name,
                    "topology": adj_matrix,
                    "dataset_tag": tag_label,
                    "results": [],
                }
            )
            continue

        # Trace each debate back to the dataset instance it came from.  The
        # dataloader exposes the selected dataset indexes in question order, so
        # result position ``pos`` maps to ``dataloader.indexes[pos]``.
        dataloader = getattr(debate_orchestration, "dataloader", None)
        dataset_indexes = getattr(dataloader, "indexes", None)

        def _index_for_position(pos):
            if dataset_indexes is None or pos >= len(dataset_indexes):
                return None
            return int(dataset_indexes[pos])

        for debate_pos, debate in enumerate(results):
            if isinstance(debate, dict):
                debate["dataset_tag"] = tag_label
                debate_index = _index_for_position(debate_pos)
                if debate_index is not None:
                    debate["dataset_index"] = debate_index

        initial_count = len(results)
        dataset_initial_debates += initial_count

        if config.clean_debates and not is_ta:
            log_info("Cleaning data: removing debates with invalid/empty responses...")
            cleaned_results = []
            kept_positions = []
            invalid_reason_counts = {}
            invalid_examples = []

            for debate_idx, debate in enumerate(results):
                invalid_reasons = get_debate_invalid_reasons(debate)
                if not invalid_reasons:
                    cleaned_results.append(debate)
                    kept_positions.append(debate_idx)
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
            log_info(f"Cleaned data: removed {removed_count} invalid/empty debates from topology {topo_name}")

            if removed_count > 0:
                log_info(f"Clean report for topology {topo_name}:")
                print(f"    Total debates      : {initial_count}")
                print(f"    Kept debates       : {len(results)}")
                print(f"    Removed debates    : {removed_count}")
                log_info("Reason counts:")
                for reason, count in sorted(invalid_reason_counts.items(), key=lambda x: (-x[1], x[0])):
                    print(f"      - {reason}: {count}")

                if invalid_examples:
                    log_info("Example removed debates (first up to 5):")
                    for ex in invalid_examples:
                        reasons_str = ", ".join(ex["reasons"])
                        print(f"      - debate_index={ex['debate_index']}: {reasons_str}")

        else:
            if config.clean_debates and is_ta:
                log_info(
                    "Skipping debate cleaning for TA: an empty answer means no tool was "
                    "called (safe), so TA debates are never dropped for empty responses."
                )
            kept_positions = [
                j for j, debate in enumerate(results) if debate is not None
            ]

        if config.process_text:
            log_info("Starting text processing on cleaned debates...")
            assert processor is not None
            total_to_process = len(results)

            workers = config.text_process_workers if config.text_process_workers > 0 else min(8, os.cpu_count() or 1)
            processor_device = getattr(processor, "device", "cpu")

            if processor_device != "cpu":
                if workers > 1:
                    log_info(
                        f"text_process_workers={workers} requested, but processor device is "
                        f"'{processor_device}'. Falling back to sequential processing."
                    )
                workers = 1

            if workers <= 1:
                for debate in tqdm(
                    cast(list[Any], results),
                    total=total_to_process,
                    desc=f"Text Processing [{tag_label}/{topo_name}]",
                    leave=False,
                ):
                    process_single_debate(debate, processor)
            else:
                log_info(f"Processing debate text in parallel with {workers} workers...")
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    results = list(
                        tqdm(
                            executor.map(
                                lambda d: process_single_debate(d, processor),
                                cast(list[Any], results),
                            ),
                            total=total_to_process,
                            desc=f"Text Processing [{tag_label}/{topo_name}]",
                            leave=False,
                        )
                    )

        dataset_valid_debates += len(results)

        # Keep the dataset indexes of the debates that actually made it into the
        # saved data, dropping any cleaned/invalid or failed (None) ones.
        if dataset_indexes is not None:
            used_global_indexes.extend(
                int(dataset_indexes[pos])
                for pos in kept_positions
                if pos < len(dataset_indexes)
            )

        dataset_results.append(
            {
                "topology_name": topo_name,
                "topology": adj_matrix,
                "dataset_tag": tag_label,
                "results": results,
            }
        )

    stats = {
        "initial_debates": dataset_initial_debates,
        "valid_debates": dataset_valid_debates,
    }
    return dataset_results, used_global_indexes, stats


def main():
    from time import time
    t0 = time()
    arguments = argparse.ArgumentParser(description="XG-Guard Anomaly Detection with Graph Neural Networks")
    arguments.add_argument('config', type=str, default=None, help='Path to YAML config file with all parameters')
    parsed_config = arguments.parse_args()
    config = load_generation_config(parsed_config.config)
    log_section("Training Data Generation (multi-dataset)")

    log_info(
        "Datasets to generate: "
        + ", ".join(f"{e.tag}->{e.loader_tag}" for e in config.datasets)
    )

    processor = None
    if config.process_text:
        processor = load_text_processor(config)

    all_results = []
    idx_metadata = {}
    total_initial_debates = 0
    total_valid_debates = 0

    for entry in config.datasets:
        dataset_results, used_global_indexes, stats = _generate_for_dataset(
            entry, config, processor
        )
        tag = entry.tag

        all_results.extend(dataset_results)
        idx_metadata[tag] = used_global_indexes
        total_initial_debates += stats["initial_debates"]
        total_valid_debates += stats["valid_debates"]

        log_info(
            f"Dataset '{tag}': {stats['valid_debates']} valid debates, "
            f"{len(used_global_indexes)} used indexes recorded."
        )

    os.makedirs(config.output_dir, exist_ok=True)
    output_filepath = os.path.join(config.output_dir, config.output_file)
    legacy_flat = sorted({int(i) for indexes in idx_metadata.values() for i in indexes})
    output = {
        "data": all_results,
        "idx_metadata": idx_metadata,
        "idx_metadata_flat": legacy_flat,
        "dataset_tags": [entry.tag for entry in config.datasets],
    }
    total_used = sum(len(v) for v in idx_metadata.values())
    if total_used:
        log_info(
            f"Stored per-tag used indexes in idx_metadata ({total_used} total)."
        )
        for tag, indexes in idx_metadata.items():
            print(f"    - {tag}: {len(indexes)} index(es)")
    with open(output_filepath, 'wb') as f:
        pickle.dump(output, f)
    log_info(f"Processed text data saved to {output_filepath}")

    elapsed_seconds = time() - t0
    log_info(f"Total execution time: {fmt_seconds(elapsed_seconds)}")

    avg_seconds_per_initial = (
        elapsed_seconds / total_initial_debates if total_initial_debates > 0 else 0.0
    )
    avg_seconds_per_valid = (
        elapsed_seconds / total_valid_debates if total_valid_debates > 0 else 0.0
    )

    print()
    log_info("Timing report:")
    print(f"    Initial debates/questions generated : {total_initial_debates}")
    print(f"    Final valid debates kept            : {total_valid_debates}")
    print(f"    Avg time per initial debate/question: {avg_seconds_per_initial:.4f} sec")
    print(f"    Avg time per final valid debate     : {avg_seconds_per_valid:.4f} sec")

    timing_report = {
        "total_seconds": elapsed_seconds,
        "total_minutes": int(elapsed_seconds // 60),
        "remaining_seconds": elapsed_seconds % 60,
        "initial_debates": total_initial_debates,
        "valid_debates": total_valid_debates,
        "avg_seconds_per_initial": avg_seconds_per_initial,
        "avg_seconds_per_valid": avg_seconds_per_valid,
        "datasets": [
            {
                "config_tag": entry.tag,
                "loader_tag": entry.loader_tag,
                "num_questions": entry.num_questions,
                "num_questions_on_random_topo": entry.num_questions_on_random_topo,
                "used_indexes": list(idx_metadata.get(entry.tag, [])),
            }
            for entry in config.datasets
        ],
        "used_indexes_per_tag": {
            tag: len(indexes) for tag, indexes in idx_metadata.items()
        },
    }
    report_filename = f"report-{config.output_file}"
    if not report_filename.lower().endswith(".json"):
        report_filename += ".json"
    report_filepath = os.path.join(config.output_dir, report_filename)
    with open(report_filepath, "w", encoding="utf-8") as report_file:
        json.dump(to_jsonable(timing_report), report_file, indent=2)
    log_info(f"Timing report saved to {report_filepath}")


if __name__ == "__main__":
    main()
