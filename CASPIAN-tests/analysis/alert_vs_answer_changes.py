#!/usr/bin/env python3
"""Relate CASPIAN alerts to genuine answer revisions in benign debates.

Joins the generated MMLUPro debates (which contain per-round agent answers)
with replay results (which contain per-round flags) and reports whether
alerting debates differ in answer-revision behaviour from silent ones.
"""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.pkl, "rb") as handle:
        payload = pickle.load(handle)
    with open(args.replay, "r", encoding="utf-8") as handle:
        replay = json.load(handle)

    debates = {}
    for topo in payload["data"]:
        name = topo["topology_name"]
        for idx, debate in enumerate(topo["results"]):
            rounds = debate.get("debate_rounds", [])
            revisions = 0
            rounds_with_revision = 0
            for r in range(1, len(rounds)):
                changed = 0
                for a_idx in range(len(rounds[r])):
                    if rounds[r][a_idx]["answer"] != rounds[r - 1][a_idx]["answer"]:
                        changed += 1
                revisions += changed
                if changed:
                    rounds_with_revision += 1
            debates[(name, idx)] = {
                "n_rounds": len(rounds),
                "revisions": revisions,
                "rounds_with_revision": rounds_with_revision,
                "revision_rate": revisions / (len(rounds) - 1) if len(rounds) > 1 else 0.0,
            }

    rows = []
    for record in replay.get("records", []):
        key = (record["topology"], record["debate_index"])
        info = debates.get(key)
        if info is None:
            continue
        flagged = record["emitted_at"] is not None
        rows.append({"flagged": flagged, **info})

    def stats(select):
        chosen = [r for r in rows if select(r)]
        if not chosen:
            return {"n": 0}
        revisions = np.asarray([r["revisions"] for r in chosen], dtype=float)
        rates = np.asarray([r["revision_rate"] for r in chosen], dtype=float)
        rounds = np.asarray([r["n_rounds"] for r in chosen], dtype=float)
        return {
            "n": len(chosen),
            "mean_revisions": float(revisions.mean()),
            "mean_revision_rate": float(rates.mean()),
            "frac_with_any_revision": float(np.mean(revisions > 0)),
            "mean_rounds": float(rounds.mean()),
        }

    summary = {
        "flagged": stats(lambda r: r["flagged"]),
        "not_flagged": stats(lambda r: not r["flagged"]),
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "rows": rows}, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
