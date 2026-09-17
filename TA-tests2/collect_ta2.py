"""Aggregate TA2 experiment metrics into a summary table.

Reads ``TA-tests2/arm-X/exp-XX/metrics.json`` (written by
``auxiliary/compute_ta_asr.py``) and, when a matching
``TA-tests2/arm-X/exp-XX-control/metrics.json`` exists, reports the
control-corrected excess.

Usage:
    python TA-tests2/collect_ta2.py
"""

import glob
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize(arm, exp):
    exp_dir = os.path.join(BASE, f"arm-{arm}", f"exp-{exp:02d}")
    metrics = load(os.path.join(exp_dir, "metrics.json"))
    if not metrics:
        return None
    control = load(os.path.join(BASE, f"arm-{arm}", f"exp-{exp:02d}-control", "metrics.json"))
    real = metrics["real"]
    row = {
        "arm": arm,
        "experiment": exp,
        "asr_name": real["overall"]["rates"]["asr_name"],
        "asr_verified": real["overall"]["rates"]["asr_verified"],
        "ben_verified": real["overall"]["rates"]["ben_asr_verified"],
        "mal_verified": real["overall"]["rates"]["mal_asr_verified"],
        "final_name": real["final"]["asr_name"],
        "final_verified": real["final"]["asr_verified"],
        "hallucinated": real["overall"]["rates"]["hallucinated"],
        "tool_call_rate": real["overall"]["rates"]["tool_call_rate"],
        "excess_name": None,
        "excess_verified": None,
        "topologies": {
            name: {
                "asr_name": data["rates"]["asr_name"],
                "asr_verified": data["rates"]["asr_verified"],
            }
            for name, data in real["topologies"].items()
        },
    }
    if control:
        control_real = control["real"]
        row["excess_name"] = row["asr_name"] - control_real["overall"]["rates"]["asr_name"]
        row["excess_verified"] = row["asr_verified"] - control_real["overall"]["rates"]["asr_verified"]
        row["control_asr_name"] = control_real["overall"]["rates"]["asr_name"]
        row["control_asr_verified"] = control_real["overall"]["rates"]["asr_verified"]
    return row


def main():
    summary = {}
    for arm in ("A", "B"):
        rows = []
        for exp in range(1, 16):
            row = summarize(arm, exp)
            if row:
                rows.append(row)
        summary[f"arm-{arm}"] = rows

    out_path = os.path.join(BASE, "summary.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Wrote {out_path}")
    print()
    for arm in ("A", "B"):
        rows = summary[f"arm-{arm}"]
        if not rows:
            continue
        print(f"=== Arm {arm} ===")
        print(f"{'exp':<5}{'ASR-name':>9}{'ASR-ver':>9}{'ben-ver':>9}{'mal-ver':>9}"
              f"{'final-ver':>10}{'halluc%':>9}{'calls%':>8}{'excess-ver':>11}")
        for row in rows:
            excess = row["excess_verified"]
            excess_s = f"{excess:>11.2f}" if excess is not None else f"{'n/a':>11}"
            print(f"{row['experiment']:<5}{row['asr_name']:>9.2f}{row['asr_verified']:>9.2f}"
                  f"{row['ben_verified']:>9.2f}{row['mal_verified']:>9.2f}"
                  f"{row['final_verified']:>10.2f}{row['hallucinated']:>9.2f}"
                  f"{row['tool_call_rate']:>8.2f}{excess_s}")
        print()


if __name__ == "__main__":
    main()
