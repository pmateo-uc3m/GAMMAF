"""Entry point for the MS MARCO contaminated-passage task generator.

Runs the full generation pipeline:

1. Load configuration (``config.json``) and LLM settings (``llm_settings.json``).
2. Load ``microsoft/ms_marco`` (v2.1) and deterministically select entries.
3. For each entry, deterministically select ``n`` passages (prioritizing
   ``is_selected == 1``), skipping entries with fewer than ``n`` passages.
4. Ask the LLM to produce a coordinated set of contaminated passages.
5. Write the final benchmark JSON with the required schema.

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
from typing import Any, Dict, List

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from dataset_utils import load_msmarco, extract_entry, DatasetEntryError  # noqa: E402
from passage_selection import select_passages  # noqa: E402
from contamination_llm import ContaminationLLM, LLMGenerationError  # noqa: E402
from prompts import load_prompts, build_user_prompt  # noqa: E402


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

    n = int(config["n"])
    num_entries = int(config.get("num_entries", 0))
    if args.limit is not None:
        num_entries = args.limit

    entry_seed = int(config["entry_selection_seed"])
    passage_seed = int(config["passage_selection_seed"])

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

    # --- deterministic entry selection ---
    rng_entry = np.random.default_rng(entry_seed)
    if num_entries <= 0:
        raise ValueError("num_entries must be a positive integer")
    if num_entries > total_rows:
        print(f"[WARN] num_entries={num_entries} > available rows={total_rows}; using {total_rows}")
        num_entries = total_rows
    row_indices = rng_entry.choice(total_rows, size=num_entries, replace=False)

    # --- passage selection rng ---
    rng_passage = np.random.default_rng(passage_seed)

    llm = ContaminationLLM(llm_settings)

    stats = {
        "considered": 0,
        "generated": 0,
        "skipped_insufficient": 0,
        "skipped_malformed": 0,
        "llm_failed": 0,
        "llm_retries": 0,
    }

    output_entries: List[Dict[str, Any]] = []
    os.makedirs(out_dir, exist_ok=True)

    start_time = time.time()

    for idx in row_indices:
        stats["considered"] += 1
        row = dataset[int(idx)]

        # --- entry field extraction / validation ---
        try:
            entry = extract_entry(row)
        except DatasetEntryError as exc:
            print(f"[WARN] Skipping row {int(idx)} (malformed): {exc}")
            stats["skipped_malformed"] += 1
            continue

        passages = entry["passages"]
        is_selected = entry["is_selected"]

        # --- passage selection ---
        try:
            safe_passages, _ = select_passages(
                passages, is_selected, n, rng_passage
            )
        except ValueError as exc:
            # Fewer than n passages: skip per documented policy.
            stats["skipped_insufficient"] += 1
            continue

        correct_answer = ", ".join(entry["answers"])
        user_prompt = build_user_prompt(
            prompts["user"], entry["query"], correct_answer, safe_passages, n
        )

        # --- LLM contamination ---
        try:
            result = llm.generate(
                prompts["system"], user_prompt, n, retry_settings=retry_cfg
            )
        except LLMGenerationError as exc:
            print(f"[WARN] LLM failure for query_id={entry['query_id']}: {exc}")
            stats["llm_failed"] += 1
            continue

        # Internal metadata (target answer) is captured for debugging but not
        # written into the final benchmark schema.
        entry_out = {
            "query_id": entry["query_id"],
            "query": entry["query"],
            "answers": entry["answers"],
            "safe_passages": safe_passages,
            "adv_passages": result["adv_passages"],
        }
        _debug_meta = {
            "incorrect_answer": result["incorrect_answer"],
        }

        # Final safety check: exactly n vs n.
        if len(entry_out["safe_passages"]) != n or len(entry_out["adv_passages"]) != n:
            print(f"[WARN] Skipping query_id={entry['query_id']} (size validation failed)")
            stats["skipped_malformed"] += 1
            continue

        output_entries.append(entry_out)
        stats["generated"] += 1

        print(
            f"[OK] query_id={entry['query_id']}  "
            f"incorrect_answer={_debug_meta['incorrect_answer']!r}"
        )

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
    print(f"  {'skipped (insufficient passages)':.<30s} {stats['skipped_insufficient']}")
    print(f"  {'skipped (malformed entries)':.<30s} {stats['skipped_malformed']}")
    print(f"  {'LLM failures':.<30s} {stats['llm_failed']}")
    print(f"  {'output location':.<30s} {out_path}")
    print(f"  {'elapsed':.<30s} {elapsed:.2f}s")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
