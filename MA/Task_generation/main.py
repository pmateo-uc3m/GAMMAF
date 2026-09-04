"""Entry point for the MS MARCO contaminated-passage task generator.

Runs the full generation pipeline:

1. Load configuration (``config.json``) and LLM settings (``llm_settings.json``).
2. Load ``microsoft/ms_marco`` (v2.1) and consume entries in a seeded order.
3. For each entry, select all passages with ``is_selected == 1`` (skipping
   entries that have no selected passage).
4. Ask the LLM to produce a coordinated set of contaminated passages, one per
   selected passage.
5. Write the final benchmark JSON with the required schema.

``num_entries`` is the *target* number of generated tasks. Entries that are
skipped (malformed or with no selected passage) or that fail LLM generation
are replaced by additional entries from the dataset, so the requested number
of tasks is produced whenever the dataset has enough valid entries.

Usage:
    python main.py [--config PATH] [--limit NUM]

``--limit`` (optional) overrides ``num_entries`` for a quick test run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from dataset_utils import load_msmarco, extract_entry, DatasetEntryError  # noqa: E402
from passage_selection import select_selected_passages  # noqa: E402
from contamination_llm import ContaminationLLM, LLMGenerationError  # noqa: E402
from prompts import (  # noqa: E402
    load_prompts,
    load_answer_prompts,
    build_user_prompt,
    build_answer_prompt,
)


def _is_no_answer(answers: List[str]) -> bool:
    """True when the entry is an MS MARCO unanswerable query.

    MS MARCO v2.1 marks unanswerable queries with the placeholder answer
    ``"No Answer Present."``.
    """
    for a in answers:
        if str(a).strip().lower() == "no answer present.":
            return True
    return False


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(base_dir: str, p: str) -> str:
    """Resolve a configured path.

    Relative paths are first resolved against the config file's directory
    (``base_dir``).  If that path does not exist, fall back to the package
    directory (``MA/Task_generation``) so bundled prompt files are found even
    when the config file lives elsewhere.
    """
    if os.path.isabs(p):
        return p
    candidate = os.path.join(base_dir, p)
    if os.path.exists(candidate):
        return candidate
    fallback = os.path.join(_THIS_DIR, p)
    if os.path.exists(fallback):
        return fallback
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MS MARCO contaminated-passage benchmark generator"
    )
    parser.add_argument(
        "--config", default=os.path.join(_THIS_DIR, "config.json"),
        help="Path to config.json",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Override num_entries for a quick test run",
    )
    parser.add_argument(
        "--out", default=None,
        help="Override output JSON path",
    )
    args = parser.parse_args()

    config = _read_json(args.config)
    base_dir = os.path.dirname(os.path.abspath(args.config))

    # --- dataset settings ---
    ds_cfg = config["dataset"]
    dataset_name = ds_cfg["name"]
    dataset_version = ds_cfg["config"]
    split = ds_cfg.get("split", "validation")

    num_entries = int(config.get("num_entries", 0))
    if args.limit is not None:
        num_entries = args.limit

    max_concurrent_calls = int(config.get("max_concurrent_calls", 200))

    entry_seed = int(config["entry_selection_seed"])

    # --- output settings ---
    out_cfg = config.get("output", {})
    out_dir = resolve_path(base_dir, out_cfg.get("dir", "output"))
    out_file = out_cfg.get("file_name", "msmarco_contaminated_benchmark.json")
    if args.out is not None:
        out_path = args.out if os.path.isabs(args.out) else resolve_path(base_dir, args.out)
    else:
        out_path = os.path.join(out_dir, out_file)

    # --- llm settings + prompts ---
    llm_settings_path = resolve_path(
        base_dir, config.get("llm_settings_file", "llm_settings.json")
    )
    llm_settings = _read_json(llm_settings_path)

    prompt_cfg = config.get("prompts", {})
    system_file = resolve_path(
        base_dir, prompt_cfg.get("system_prompt_file", "prompts_system.txt")
    )
    user_file = resolve_path(
        base_dir, prompt_cfg.get("user_prompt_file", "prompts_user.txt")
    )
    prompts = load_prompts(system_file, user_file)

    answer_system_file = resolve_path(
        base_dir,
        prompt_cfg.get("answer_system_prompt_file", "prompts_answer_system.txt"),
    )
    answer_user_file = resolve_path(
        base_dir,
        prompt_cfg.get("answer_user_prompt_file", "prompts_answer_user.txt"),
    )
    answer_prompts = load_answer_prompts(answer_system_file, answer_user_file)

    retry_cfg = config.get("retry", {})

    # --- load dataset ---
    print("Loading dataset %s (config=%s, split=%s) ..." % (
        dataset_name, dataset_version, split,
    ))
    try:
        dataset = load_msmarco(dataset_name, dataset_version, split)
    except Exception as exc:
        print(f"[ERROR] Failed to load dataset: {exc}")
        return 1
    total_rows = len(dataset)
    print(f"Loaded {total_rows} rows.")

    # --- deterministic entry ordering ---
    rng_entry = np.random.default_rng(entry_seed)
    if num_entries <= 0:
        raise ValueError("num_entries must be a positive integer")
    row_order = rng_entry.permutation(total_rows)

    llm = ContaminationLLM(llm_settings)

    stats = {
        "considered": 0,
        "generated": 0,
        "skipped_no_selected": 0,
        "skipped_malformed": 0,
        "llm_failed": 0,
        "llm_retries": 0,
    }

    output_entries: List[Dict[str, Any]] = []
    os.makedirs(out_dir, exist_ok=True)

    start_time = time.time()

    # --- job generation: consume entries in seeded order ---
    # Entries that are malformed or have no selected passage are skipped;
    # the generator keeps yielding valid jobs so that, when combined with the
    # failure-replenishing executor below, exactly ``num_entries`` tasks are
    # produced (or the dataset is exhausted first).
    def iter_jobs():
        for idx in row_order:
            stats["considered"] += 1
            row = dataset[int(idx)]

            try:
                entry = extract_entry(row)
            except DatasetEntryError as exc:
                print(f"[WARN] Skipping row {int(idx)} (malformed): {exc}")
                stats["skipped_malformed"] += 1
                continue

            try:
                safe_passages, _ = select_selected_passages(
                    entry["passages"], entry["is_selected"]
                )
            except ValueError:
                # No selected passage: skip per documented policy.
                stats["skipped_no_selected"] += 1
                continue

            yield {
                "entry": entry,
                "safe_passages": safe_passages,
                "needs_answer_gen": _is_no_answer(entry["answers"]),
                "correct_answer": ", ".join(entry["answers"]),
            }

    def process_job(job: Dict[str, Any]) -> Dict[str, Any]:
        """Run answer generation (if needed) then contamination for one entry."""
        entry = job["entry"]
        safe_passages = job["safe_passages"]
        n_job = len(safe_passages)

        if job["needs_answer_gen"]:
            answer_user = build_answer_prompt(
                answer_prompts["user"], entry["query"], safe_passages
            )
            answer_result = llm.generate_answer(
                answer_prompts["system"],
                answer_user,
                retry_settings=retry_cfg,
            )
            effective_answers = [answer_result["answer"]]
            correct_answer = answer_result["answer"]
        else:
            effective_answers = entry["answers"]
            correct_answer = job["correct_answer"]

        user_prompt = build_user_prompt(
            prompts["user"], entry["query"], correct_answer, safe_passages, n_job
        )
        result = llm.generate(
            prompts["system"], user_prompt, n_job, retry_settings=retry_cfg
        )
        return {
            "query_id": entry["query_id"],
            "query": entry["query"],
            "answers": effective_answers,
            "safe_passages": safe_passages,
            "adv_passages": result["adv_passages"],
        }

    # --- parallel LLM contamination with failure replenishment ---
    completed: Dict[int, Dict[str, Any]] = {}
    target = num_entries
    job_gen = iter_jobs()

    with ThreadPoolExecutor(max_workers=max_concurrent_calls) as executor:
        pending: Dict[Any, int] = {}
        next_index = 0

        def fill() -> None:
            """Submit jobs to keep the window full without overshooting.

            The number of in-flight jobs never exceeds the remaining number of
            tasks still needed, so in-flight calls are not wasted once the
            target is nearly reached.
            """
            nonlocal next_index
            if stats["generated"] >= target:
                return
            cap = max(1, min(max_concurrent_calls, target - stats["generated"]))
            while len(pending) < cap:
                try:
                    job = next(job_gen)
                except StopIteration:
                    return
                future = executor.submit(process_job, job)
                pending[future] = next_index
                next_index += 1

        fill()

        with tqdm(total=target, desc="Contaminating passages", unit="entry") as bar:
            while pending:
                done, _ = wait(list(pending.keys()), return_when=FIRST_COMPLETED)
                for future in done:
                    if stats["generated"] >= target:
                        break
                    i = pending.pop(future)
                    try:
                        entry_out = future.result()
                    except LLMGenerationError:
                        stats["llm_failed"] += 1
                    except Exception as exc:
                        print(f"[WARN] Unexpected LLM error (job {i}): {exc}")
                        stats["llm_failed"] += 1
                    else:
                        if len(entry_out["safe_passages"]) == 0 or len(entry_out["safe_passages"]) != len(entry_out["adv_passages"]):
                            stats["skipped_malformed"] += 1
                        else:
                            completed[i] = entry_out
                            stats["generated"] += 1
                            bar.update(1)
                if stats["generated"] >= target:
                    break
                fill()

        for future in pending:
            future.cancel()

    if stats["generated"] < target:
        print(
            f"[WARN] Dataset exhausted: generated {stats['generated']} tasks "
            f"instead of the requested {target}."
        )

    # --- preserve deterministic input order in the output ---
    for i in sorted(completed):
        output_entries.append(completed[i])

    elapsed = time.time() - start_time

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_entries, f, ensure_ascii=False, indent=2)

    # --- summary ---
    print()
    print("=" * 72)
    print("  MS MARCO Contaminated-Passage Generator — Summary")
    print("=" * 72)
    print(f"  {'entries considered':.<30s} {stats['considered']}")
    print(f"  {'entries generated':.<30s} {stats['generated']}")
    print(f"  {'skipped (no selected passages)':.<30s} {stats['skipped_no_selected']}")
    print(f"  {'skipped (malformed entries)':.<30s} {stats['skipped_malformed']}")
    print(f"  {'LLM failures':.<30s} {stats['llm_failed']}")
    print(f"  {'output location':.<30s} {out_path}")
    print(f"  {'elapsed':.<30s} {elapsed:.2f}s")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
