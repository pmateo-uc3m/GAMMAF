#!/usr/bin/env python3
"""Compact summariser for MainEvaluation result JSON files.

Usage:
    python summarize_results.py <results.json> [<results.json> ...]
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_run(name, stats):
    rows = []
    per_round = defaultdict(lambda: {"debates": 0, "flags": 0, "agent_rounds": 0,
                                     "f1": [], "fpr": []})
    total_questions = 0
    total_correct = 0
    topo_metrics = {}
    for topo in stats:
        topo_name = topo.get("topology", "?")
        n_q = topo.get("total_questions", 0)
        correct = topo.get("correct_answers", 0)
        total_questions += n_q
        total_correct += correct
        topo_metrics[topo_name] = {
            "questions": n_q,
            "correct": correct,
            "accuracy": (correct / n_q) if n_q else None,
        }
        for r_idx, rr in enumerate(topo.get("rounds_rates", [])):
            count = topo.get("round_counts", {}).get(str(r_idx), topo.get("round_counts", {}).get(r_idx, 0))
            f1 = rr.get("F1")
            fpr = rr.get("FPR")
            if f1 is not None:
                per_round[r_idx]["f1"].append(f1)
            if fpr is not None:
                per_round[r_idx]["fpr"].append(fpr)
            per_round[r_idx]["debates"] += count
    for r_idx, agg in sorted(per_round.items()):
        f1_values = agg["f1"] if agg["f1"] else [float("nan")]
        fpr_values = agg["fpr"] if agg["fpr"] else [float("nan")]
        rows.append(
            {
                "round": r_idx + 1,
                "topology_entries": len(agg["f1"]),
                "macro_F1": float(np.nanmean(f1_values)),
                "macro_FPR": float(np.nanmean(fpr_values)),
            }
        )
    return {
        "name": name,
        "questions": total_questions,
        "accuracy": (total_correct / total_questions) if total_questions else None,
        "topologies": topo_metrics,
        "per_round": rows,
    }


def main(paths):
    results = []
    for path in paths:
        data = load(path)
        for model_name, stats in data.items():
            if model_name.startswith("_"):
                continue
            results.append(summarize_run(Path(path).stem + ":" + model_name, stats))
    header = f"{'run':<48} {'qs':>5} {'acc':>7} {'r1_F1':>7} {'r2_F1':>7} {'r3_F1':>7} {'r1_FPR':>7} {'r2_FPR':>7} {'r3_FPR':>7}"
    print(header)
    print("-" * len(header))
    for result in results:
        f1 = {row["round"]: row["macro_F1"] for row in result["per_round"]}
        fpr = {row["round"]: row["macro_FPR"] for row in result["per_round"]}
        acc = result["accuracy"]
        print(
            f"{result['name']:<48} {result['questions']:>5} "
            f"{(acc if acc is not None else float('nan')):>7.3f} "
            f"{f1.get(1, float('nan')):>7.3f} {f1.get(2, float('nan')):>7.3f} {f1.get(3, float('nan')):>7.3f} "
            f"{fpr.get(1, float('nan')):>7.3f} {fpr.get(2, float('nan')):>7.3f} {fpr.get(3, float('nan')):>7.3f}"
        )
        for topo, metrics in sorted(result["topologies"].items()):
            print(
                f"    {topo:<16} q={metrics['questions']:<4} "
                f"acc={metrics['accuracy'] if metrics['accuracy'] is not None else float('nan'):.3f}"
            )


if __name__ == "__main__":
    main(sys.argv[1:])
