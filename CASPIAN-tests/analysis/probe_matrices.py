#!/usr/bin/env python3
"""Probe CASPIAN influence matrices on generated data.

Prints, for selected debates, the raw influence matrix, the degree-normalised
matrix, its singular values and the derived per-turn signals.
"""

import argparse
import importlib.util
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

np.set_printoptions(precision=6, suppress=True, linewidth=200)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--impl", required=True)
    parser.add_argument("--topology", default=None)
    parser.add_argument("--debates", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    module = load_module(args.impl, "caspian_probe")
    detector = module.CASPIANDetector(
        SimpleNamespace(epsilon=1e-8, target_ema_decay=0.8, influence_ema_decay=0.8,
                        max_persistence_window=64, spine_top_k=3, top_k=2)
    )
    with open(args.pkl, "rb") as handle:
        payload = pickle.load(handle)

    for topo in payload["data"]:
        name = topo["topology_name"]
        if args.topology and name != args.topology:
            continue
        print("=" * 100)
        print(f"TOPOLOGY {name}")
        for deb_idx, debate in enumerate(topo["results"][: args.debates]):
            adjacency = debate.get("topology") or topo.get("topology")
            trace_id = (name, deb_idx)
            detector.begin_trace(trace_id, adjacency)
            print("-" * 100)
            print(f"debate {deb_idx}  rounds={len(debate['debate_rounds'])}")
            for r_idx, round_agents in enumerate(debate["debate_rounds"][: args.rounds]):
                round_data = [{"agent_id": a["agent_id"], "st_embedding": a["st_embedding"]}
                              for a in round_agents]
                flags, scores = detector.predict(round_data, adjacency, trace_id=trace_id)
                state = detector._states[detector._key(trace_id)]
                raw = state["influence"]
                norm = detector._normalised_influence(raw)
                singular = np.linalg.svd(norm, compute_uv=False)
                print(f"\nround {r_idx}: flags={np.asarray(flags)}")
                print("raw:\n", raw)
                print("normalised:\n", norm)
                print("singular values:", singular)
                print("lambda1 - prev (floating point):", state["last_signals"]["lambda1"],
                      "ratio:", state["last_signals"]["coupling_ratio"],
                      "gap:", state["last_signals"]["normalised_gap"])
                print("watch:", state["last_signals"]["watch"],
                      "phase_shift:", state["last_signals"]["phase_shift"],
                      "instant/multi:", state["last_signals"]["cascade"])
            detector.end_trace(trace_id)


if __name__ == "__main__":
    main()
