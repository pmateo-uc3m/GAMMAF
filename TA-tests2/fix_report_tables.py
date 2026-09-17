"""Rebuild the results tables in FINAL-REPORT.md from summary.json.

Ensures every number in the report matches the machine-readable metrics.

Usage:
    python TA-tests2/fix_report_tables.py
"""

import json
import os
import re

BASE = os.path.dirname(os.path.abspath(__file__))

MECHANISMS = {
    1: "Reference (best original TA prompt)",
    2: "Relay of pending steps",
    3: "Precedent / standard procedure",
    4: "Completion pressure",
    5: "Group norm",
    6: "Memory persistence",
    7: "Perform, don't describe",
    8: "Confirmation loop",
    9: "Ownership",
    10: "Urgency",
    11: "Minimal prompts",
    12: "Parameter fidelity",
    13: "Relay + authority (combo)",
    14: "Precedent + completion (combo)",
    15: "**Precedent + parameter fidelity (adaptive)**",
}


def cell(value, digits=2):
    return f"{value:.{digits}f}"


def build_table(rows, title):
    lines = [f"### {title}", ""]
    lines.append(
        "| Exp | Mechanism | ASR-name | **ASR-verified** | Excess-verified | Benign ver | "
        "Malicious ver | Final-round ver | Tree | Chain | Star | Random | Calls % |"
    )
    lines.append(
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for row in rows:
        exp = row["experiment"]
        topo = row["topologies"]
        excess = cell(row["excess_verified"]) if row["excess_verified"] is not None else "n/a"
        lines.append(
            f"| {exp:02d} | {MECHANISMS[exp]} | {cell(row['asr_name'])} | **{cell(row['asr_verified'])}** | "
            f"{excess} | {cell(row['ben_verified'])} | {cell(row['mal_verified'])} | {cell(row['final_verified'])} | "
            f"{cell(topo['tree']['asr_verified'])} | {cell(topo['chain']['asr_verified'])} | "
            f"{cell(topo['star']['asr_verified'])} | {cell(topo['random']['asr_verified'])} | "
            f"{cell(row['tool_call_rate'], 1)} |"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    with open(os.path.join(BASE, "summary.json"), "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    path = os.path.join(BASE, "FINAL-REPORT.md")
    report = open(path, "r", encoding="utf-8").read()

    block_a = build_table(summary["arm-A"], "Arm A — operational-instruction framing")
    block_b = build_table(summary["arm-B"], "Arm B — parameter-redirection framing")

    report, count_a = re.subn(
        r"### Arm A — operational-instruction framing\n\n(?:\|.*\n)+",
        block_a + "\n",
        report,
        count=1,
    )
    report, count_b = re.subn(
        r"### Arm B — parameter-redirection framing\n\n(?:\|.*\n)+",
        block_b + "\n",
        report,
        count=1,
    )
    if count_a != 1 or count_b != 1:
        raise RuntimeError(f"table replacement failed (A={count_a}, B={count_b})")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(report)
    print("FINAL-REPORT.md tables rebuilt from summary.json")


if __name__ == "__main__":
    main()
