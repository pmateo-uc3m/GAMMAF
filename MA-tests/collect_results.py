"""Experiment bookkeeping helper (MA prompt-search).

Parses the ``asr.log`` produced by ``auxiliary/compute_asr.py`` for each
``MA-tests/experiment-*`` directory and writes an aggregate ``summary.json``
plus a Markdown table. Read-only with respect to the generation pipeline.
"""

import glob
import json
import os
import re

BASE = os.path.dirname(os.path.abspath(__file__))


def parse_asr_log(path):
    text = open(path, "r", encoding="utf-8", errors="replace").read()
    text = text.replace("\r", "\n")
    result = {"topologies": {}, "overall": {}}

    topo_blocks = re.split(r"\n\s*Topology\s+:\s*", text)
    for block in topo_blocks[1:]:
        name = block.split("\n", 1)[0].strip()
        q = re.search(r"Questions\s*:\s*(\d+)", block)
        c = re.search(r"Correct\s*:\s*(\d+)\s*\(([\d.]+)%\)", block)
        cons = re.search(r"Consensus\s*:\s*(\d+)/(\d+)", block)
        inf = re.search(r"Infection\s*:\s*(\d+)/\d+\s*\(([\d.]+)%\)", block)
        rounds = {}
        for m in re.finditer(
            r"^\s*(\d+)\s+([\d.]+)\s*(?:\+-\s*([\d.]+))?\s+([\d.]+|n/a)\s*(?:\+-\s*([\d.]+))?\s+(\d+)\s*$",
            block,
            re.MULTILINE,
        ):
            rounds[int(m.group(1))] = {
                "asr_mean": float(m.group(2)),
                "asr_ci": float(m.group(3)) if m.group(3) else None,
                "air_mean": float(m.group(4)) if m.group(4) != "n/a" else None,
                "air_ci": float(m.group(5)) if m.group(5) else None,
                "count": int(m.group(6)),
            }
        result["topologies"][name] = {
            "questions": int(q.group(1)) if q else None,
            "correct": int(c.group(1)) if c else None,
            "correct_pct": float(c.group(2)) if c else None,
            "consensus": int(cons.group(1)) if cons else None,
            "infection": int(inf.group(1)) if inf else None,
            "infection_pct": float(inf.group(2)) if inf else None,
            "final_round_asr": rounds[max(rounds)]["asr_mean"] if rounds else None,
            "final_round_air": rounds[max(rounds)]["air_mean"] if rounds else None,
            "rounds": rounds,
        }

    ov = text.split("OVERALL", 1)[-1]
    for key, pattern in {
        "debates": r"Debates\s*:\s*(\d+)",
        "accuracy": r"Accuracy\s*:\s*([\d.]+)%",
        "consensus": r"Consensus\s*:\s*(\d+)/",
        "infection": r"Infection\s*:\s*(\d+)/",
        "final_asr": r"Final ASR \(all\)\s*:\s*([\d.]+)",
        "final_asr_ci": r"Final ASR \(all\)\s*:\s*[\d.]+\s*\+-\s*([\d.]+)",
        "final_air": r"Final AIR \(benign\)\s*:\s*([\d.]+)",
        "final_air_ci": r"Final AIR \(benign\)\s*:\s*[\d.]+\s*\+-\s*([\d.]+)",
        "mean_asr_rounds": r"Mean ASR \(rounds\)\s*:\s*([\d.]+)",
        "mean_air_rounds": r"Mean AIR \(rounds\)\s*:\s*([\d.]+)",
    }.items():
        m = re.search(pattern, ov)
        result["overall"][key] = float(m.group(1)) if m else None

    return result


def main():
    summary = {}
    for exp_dir in sorted(glob.glob(os.path.join(BASE, "experiment-*"))):
        name = os.path.basename(exp_dir)
        asr_log = os.path.join(exp_dir, "asr.log")
        meta_path = os.path.join(exp_dir, "metadata.json")
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        entry = {
            "prompt_file": meta.get("prompt_file"),
            "variant": meta.get("prompt_variant"),
            "hypothesis": meta.get("hypothesis"),
            "seeds": meta.get("seeds"),
            "yaml_config": meta.get("yaml_config"),
            "pkl_path": meta.get("pkl_path"),
        }
        if os.path.exists(asr_log):
            entry["asr"] = parse_asr_log(asr_log)
        summary[name] = entry

    out_path = os.path.join(BASE, "summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_path} with {len(summary)} experiments")

    print()
    header = f"{'Exp':<8}{'tree':>8}{'chain':>8}{'star':>8}{'random':>8}{'FINAL':>9}{'AIR':>8}"
    print(header)
    for name in sorted(summary):
        a = summary[name].get("asr")
        if not a:
            print(f"{name:<8}{'n/a':>8}")
            continue
        t = a["topologies"]
        vals = [t.get(k, {}).get("final_round_asr") for k in ("tree", "chain", "star", "random")]
        row = "".join(f"{v:>8.2f}" if v is not None else f"{'n/a':>8}" for v in vals)
        fin = a["overall"].get("final_asr")
        air = a["overall"].get("final_air")
        fin_s = f"{fin:>9.2f}" if fin is not None else f"{'n/a':>9}"
        air_s = f"{air:>8.2f}" if air is not None else f"{'n/a':>8}"
        print(f"{name:<8}{row}{fin_s}{air_s}")


if __name__ == "__main__":
    main()
