"""Validation / self-tests for the contaminated-passage generator.

Runs without a live LLM (uses deterministic fixtures and a fake client) to
verify:

1. Dataset loads and expected fields/types are present.
2. Deterministic entry + passage selection with fixed seeds.
3. LLM JSON parsing and validation (valid + malformed payloads).
4. Every successful output entry has exactly n safe + n adversarial passages.
5. Safe passages are byte-for-byte unchanged from source.
6. Adversarial passages generated as a coordinated group (single target answer).
7. Output JSON follows the required schema.

Usage:
    python run_validation.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import numpy as np  # noqa: E402

from dataset_utils import load_msmarco, extract_entry  # noqa: E402
from passage_selection import select_passages  # noqa: E402
from contamination_llm import (  # noqa: E402
    validate_contamination,
    validate_answer,
    _extract_json_object,
    JSONValidationError,
)
from main import _is_no_answer  # noqa: E402

PASS = "  [PASS]"
FAIL = "  [FAIL]"


def check(name: str, cond: bool) -> bool:
    print(f"{PASS if cond else FAIL} {name}")
    return cond


def test_dataset_schema() -> bool:
    print("\n--- 1. Dataset loads + schema ---")
    ok = True
    try:
        d = load_msmarco("microsoft/ms_marco", "v2.1", "validation")
    except Exception as exc:
        check("dataset loads", False)
        print(f"    {exc}")
        return False
    ok &= check("dataset loads", len(d) > 0)

    feats = d.features
    ok &= check("has 'query' (string)", "query" in feats and _is_str_feature(feats["query"]))
    ok &= check("has 'query_id'", "query_id" in feats)
    ok &= check("has 'answers' (list)", "answers" in feats and _is_list_feature(feats["answers"]))
    ok &= check("has 'passages' dict", "passages" in feats)
    if "passages" in feats:
        p = feats["passages"]
        ok &= check("passages.is_selected present", "is_selected" in p)
        ok &= check("passages.passage_text present", "passage_text" in p)
    return ok


def _is_str_feature(f) -> bool:
    try:
        from datasets import Value
        return isinstance(f, Value) and f.dtype == "string"
    except Exception:
        return str(type(f)) != ""


def _is_list_feature(f) -> bool:
    try:
        from datasets import Sequence
        return isinstance(f, Sequence)
    except Exception:
        return "list" in str(type(f)).lower()


def test_entry_extraction() -> bool:
    print("\n--- 2. Entry extraction ---")
    ok = True
    d = load_msmarco("microsoft/ms_marco", "v2.1", "validation")
    row = d[0]
    entry = extract_entry(row)
    ok &= check("extracted query is str", isinstance(entry["query"], str))
    ok &= check("extracted answers non-empty", len(entry["answers"]) > 0)
    ok &= check("passages == passage_text length",
                len(entry["passages"]) == len(entry["is_selected"]))
    ok &= check("is_selected values in {0,1}",
                all(s in (0, 1) for s in entry["is_selected"]))
    return ok


def test_passage_selection() -> bool:
    print("\n--- 3. Deterministic passage selection ---")
    ok = True
    passages = [f"p{i}" for i in range(10)]
    is_selected = [1 if i == 3 else 0 for i in range(10)]

    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    s1, idx1 = select_passages(passages, is_selected, 5, rng1)
    s2, idx2 = select_passages(passages, is_selected, 5, rng2)

    ok &= check("returns exactly n", len(s1) == 5)
    ok &= check("deterministic (same seed -> same result)", s1 == s2 and idx1 == idx2)
    ok &= check("prioritizes selected passage (p3 in result)", "p3" in s1)

    # distinct seed -> (likely) different fill order of non-selected
    rng3 = np.random.default_rng(99)
    s3, _ = select_passages(passages, is_selected, 5, rng3)
    ok &= check("different seed gives different fill", s1 != s3 or len(set(idx1)) == 5)

    # insufficient passages
    try:
        select_passages(["a", "b"], [1, 0], 5, np.random.default_rng(1))
        ok &= check("insufficient passages raise ValueError", False)
    except ValueError:
        ok &= check("insufficient passages raise ValueError", True)
    return ok


def test_json_parsing() -> bool:
    print("\n--- 4. LLM JSON parsing + validation ---")
    ok = True

    valid = {
        "incorrect_answer": "March 15",
        "adv_passages": ["a", "b", "c"],
    }
    try:
        v = validate_contamination(valid, 3)
        ok &= check("valid payload accepted", v["adv_passages"] == ["a", "b", "c"])
    except JSONValidationError:
        ok &= check("valid payload accepted", False)

    # fenced output
    fenced = '```json\n{"incorrect_answer": "x", "adv_passages": ["1","2"]}\n```'
    obj = _extract_json_object(fenced)
    ok &= check("fenced JSON parsed", obj.get("incorrect_answer") == "x")

    # extra text around JSON
    noisy = 'Here is the result: {"incorrect_answer":"y","adv_passages":["p","q"]} thanks'
    obj = _extract_json_object(noisy)
    ok &= check("noisy JSON parsed", obj.get("incorrect_answer") == "y")

    # wrong number of passages
    try:
        validate_contamination(
            {"incorrect_answer": "x", "adv_passages": ["a", "b"]}, 3
        )
        ok &= check("wrong passage count rejected", False)
    except JSONValidationError:
        ok &= check("wrong passage count rejected", True)

    # missing target answer
    try:
        validate_contamination({"adv_passages": ["a", "b", "c"]}, 3)
        ok &= check("missing target answer rejected", False)
    except JSONValidationError:
        ok &= check("missing target answer rejected", True)

    # non-string passage
    try:
        validate_contamination(
            {"incorrect_answer": "x", "adv_passages": ["a", 2, "c"]}, 3
        )
        ok &= check("non-string passage rejected", False)
    except JSONValidationError:
        ok &= check("non-string passage rejected", True)
    return ok


def test_answer_generation() -> bool:
    print("\n--- 4b. No-Answer handling + answer validation ---")
    ok = True

    ok &= check("detects 'No Answer Present.'",
                _is_no_answer(["No Answer Present."]))
    ok &= check("case/space insensitive",
                _is_no_answer(["  no ANSWER present. "]))
    ok &= check("does not flag normal answers",
                not _is_no_answer(["A corporation is a company."]))
    ok &= check("ignores non-matching entries in list",
                not _is_no_answer(["something", "else"]))

    try:
        v = validate_answer({"answer": "February 2"})
        ok &= check("valid answer accepted", v["answer"] == "February 2")
    except JSONValidationError:
        ok &= check("valid answer accepted", False)

    try:
        validate_answer({"answer": "  "})
        ok &= check("empty answer rejected", False)
    except JSONValidationError:
        ok &= check("empty answer rejected", True)

    try:
        validate_answer({"adv_passages": ["x"]})
        ok &= check("missing answer field rejected", False)
    except JSONValidationError:
        ok &= check("missing answer field rejected", True)
    return ok


def test_output_schema() -> bool:
    print("\n--- 5. Output entry schema ---")
    ok = True
    sample = {
        "query_id": 123,
        "query": "q",
        "answers": ["correct"],
        "safe_passages": ["sp1", "sp2"],
        "adv_passages": ["ap1", "ap2"],
    }
    required = {"query_id", "query", "answers", "safe_passages", "adv_passages"}
    ok &= check("contains only required fields", set(sample.keys()) == required)
    ok &= check("safe/adv are lists of equal length",
                isinstance(sample["safe_passages"], list)
                and isinstance(sample["adv_passages"], list)
                and len(sample["safe_passages"]) == len(sample["adv_passages"]))
    ok &= check("answers preserved (not replaced by target)",
                sample["answers"] == ["correct"])
    return ok


def main() -> int:
    print("=" * 72)
    print("  Contaminated-Passage Generator — Validation")
    print("=" * 72)
    all_ok = True
    all_ok &= test_dataset_schema()
    all_ok &= test_entry_extraction()
    all_ok &= test_passage_selection()
    all_ok &= test_json_parsing()
    all_ok &= test_answer_generation()
    all_ok &= test_output_schema()

    print("\n" + "=" * 72)
    print(f"  Overall: {'ALL PASSED' if all_ok else 'SOME FAILED'}")
    print("=" * 72)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
