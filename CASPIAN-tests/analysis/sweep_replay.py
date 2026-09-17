#!/usr/bin/env python3
"""Small deterministic parameter sweep for CASPIAN over fixed generated data.

Replays a single implementation over one generated pickle for a grid of
implementation-level parameters and reports false-positive behaviour.  The
sweep is used for sensitivity analysis (not score chasing): all debates are
benign, so lower flag rates are only meaningful together with the signal
statistics and with the qualitative effect on the detector's operating regime.

Usage:
    python sweep_replay.py --pkl <pkl> --impl <CASPIAN.py> --out <json>
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_caspian import DEFAULTS, replay, summarise  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--impl", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--grid", required=True,
                        help='JSON dict, e.g. \'{"target_ema_decay": [0.5, 0.8]}\'')
    args = parser.parse_args()

    grid = json.loads(args.grid)
    keys = sorted(grid)
    rows = []
    for combo in itertools.product(*(grid[key] for key in keys)):
        params = dict(DEFAULTS)
        params.update(dict(zip(keys, combo)))
        records, used = replay(Path(args.pkl), Path(args.impl), params)
        summary = summarise(records, used)
        rows.append(
            {
                "params": {key: value for key, value in zip(keys, combo)},
                "cascades": summary["debates_with_cascade"],
                "rounds_with_flags": summary["rounds_with_flags"],
                "flags": summary["total_flags"],
                "fpr_agent_level": summary["fpr_agent_level"],
                "gap_mean": summary["signals"]["normalised_gap"]["mean"],
                "gap_median": summary["signals"]["normalised_gap"]["median"],
                "amplification_mean": summary["signals"]["amplification"]["mean"],
            }
        )
        print(rows[-1])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump({"impl": args.impl, "grid": grid, "runs": rows}, handle, indent=2)
    print("saved", out_path)


if __name__ == "__main__":
    main()
