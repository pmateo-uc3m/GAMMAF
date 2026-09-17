#!/usr/bin/env python3
"""Offline replay harness for CASPIAN over generated MMLUPro debate data.

This is an evaluation artifact: it does not modify any framework source file.
It reloads a generated training-data pickle (rounds already carry
``st_embedding``) and replays a chosen CASPIAN implementation round by round,
exactly as ``LiveDebateOrchestration.debate_question`` would:

  * ``begin_trace(trace_id, adjacency)`` once per debate,
  * ``predict(round_data, adjacency, trace_id)`` per round,
  * ``modify_adjacency(flags, adjacency)`` after each round,
  * ``end_trace(trace_id)`` at the end.

It records the returned flags/scores plus the detector's internal spectral
signals (``last_signals``) so validity and false-positive behaviour can be
analysed against the all-benign ground truth.

Usage:
    python replay_caspian.py --pkl <pkl> --impl <CASPIAN.py> --out <json>
                             [--param name=value ...]
"""

import argparse
import importlib.util
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULTS = {
    "epsilon": 1e-8,
    "target_ema_decay": 0.8,
    "influence_ema_decay": 0.8,
    "max_persistence_window": 64,
    "spine_top_k": 3,
    "top_k": 2,
}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def modify_adjacency(flags, adjacency_matrix):
    modified = [row[:] for row in adjacency_matrix]
    for i in range(len(flags)):
        if flags[i] == 1:
            for j in range(len(adjacency_matrix)):
                modified[i][j] = 0
                modified[j][i] = 0
    return modified


def as_round_data(round_agents):
    """Keep the keys CASPIAN reads; embeddings are already present."""
    result = []
    for agent in round_agents:
        entry = {"agent_id": agent.get("agent_id", len(result))}
        if "st_embedding" in agent:
            entry["st_embedding"] = agent["st_embedding"]
        else:
            entry["st_embedding"] = None
        result.append(entry)
    return result


def replay(pkl_path, impl_path, params, max_rounds=None, limit_per_topo=None):
    caspian = load_module(impl_path, "caspian_replay_impl")
    config = SimpleNamespace(**params)
    detector = caspian.CASPIANDetector(config)

    with open(pkl_path, "rb") as handle:
        payload = pickle.load(handle)
    data = payload.get("data", payload if isinstance(payload, list) else [])

    records = []
    for topo_entry in data:
        topo_name = topo_entry.get("topology_name", "unknown")
        results = topo_entry.get("results", [])
        if limit_per_topo is not None:
            results = results[:limit_per_topo]
        for deb_idx, debate in enumerate(results):
            if not isinstance(debate, dict):
                continue
            rounds = debate.get("debate_rounds", [])
            adjacency = debate.get("topology") or topo_entry.get("topology")
            if adjacency is None:
                continue
            if max_rounds is not None:
                rounds = rounds[:max_rounds]
            trace_id = (topo_name, deb_idx)
            detector.begin_trace(trace_id, adjacency)
            current_adj = [list(row) for row in adjacency]
            emitted_at = None
            per_round = []
            for r_idx, round_agents in enumerate(rounds):
                round_data = as_round_data(round_agents)
                if any(agent["st_embedding"] is None for agent in round_data):
                    per_round.append({"round": r_idx, "error": "missing_embedding"})
                    break
                flags, scores = detector.predict(round_data, current_adj, trace_id=trace_id)
                flags = [int(f) for f in np.asarray(flags).reshape(-1)]
                scores = [float(s) for s in np.asarray(scores, dtype=float).reshape(-1)]
                state = detector._states[detector._key(trace_id)]
                signals = dict(state.get("last_signals", {}))
                attribution = detector.last_attribution
                emitted = bool(state.get("cascade_emitted"))
                if emitted and emitted_at is None:
                    emitted_at = r_idx
                per_round.append(
                    {
                        "round": r_idx,
                        "flags": flags,
                        "scores": scores,
                        "n_flags": int(sum(flags)),
                        "cascade": emitted,
                        "attribution": attribution if (emitted and emitted_at == r_idx) else None,
                        "signals": {k: (v if not isinstance(v, np.ndarray) else v.tolist()) for k, v in signals.items()},
                    }
                )
                current_adj = modify_adjacency(flags, current_adj)
            detector.end_trace(trace_id)
            records.append(
                {
                    "topology": topo_name,
                    "debate_index": deb_idx,
                    "n_rounds": len(per_round),
                    "emitted_at": emitted_at,
                    "per_round": per_round,
                }
            )
    return records, params


def summarise(records, params):
    n_debates = len(records)
    n_rounds = sum(r["n_rounds"] for r in records)
    n_agent_rounds = 0
    total_flags = 0
    rounds_with_flags = 0
    debates_with_flags = 0
    debates_with_cascade = 0
    per_round_stats = defaultdict(list)
    signal_names = [
        "lambda1", "lambda2", "energy", "amplification", "coupling_ratio",
        "normalised_gap", "gap_contraction", "phase_magnitude", "watch",
        "phase_shift", "cross_channel", "weak_link",
    ]
    signal_values = {name: [] for name in signal_names}
    per_topo = defaultdict(lambda: {"debates": 0, "rounds": 0, "flags": 0,
                                    "rounds_with_flags": 0, "cascade": 0,
                                    "debates_with_flags": 0})
    for record in records:
        topo = record["topology"]
        per_topo[topo]["debates"] += 1
        debate_has_flags = False
        if record["emitted_at"] is not None:
            debates_with_cascade += 1
            per_topo[topo]["cascade"] += 1
        for rr in record["per_round"]:
            if "error" in rr:
                continue
            n_agents = len(rr["flags"])
            n_agent_rounds += n_agents
            total_flags += rr["n_flags"]
            per_topo[topo]["rounds"] += 1
            per_topo[topo]["flags"] += rr["n_flags"]
            if rr["n_flags"] > 0:
                rounds_with_flags += 1
                debate_has_flags = True
                per_topo[topo]["rounds_with_flags"] += 1
            per_round_stats[rr["round"]].append(rr["n_flags"])
            for name in signal_names:
                value = rr["signals"].get(name)
                if isinstance(value, bool):
                    value = int(value)
                if value is not None:
                    signal_values[name].append(float(value))
        if debate_has_flags:
            debates_with_flags += 1
            per_topo[topo]["debates_with_flags"] += 1

    def stat(values):
        if not values:
            return {"count": 0}
        arr = np.asarray(values, dtype=float)
        return {
            "count": int(arr.size),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p05": float(np.percentile(arr, 5)),
            "p95": float(np.percentile(arr, 95)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    f1_rounds = []
    for r_idx, flags in sorted(per_round_stats.items()):
        f1_rounds.append(
            {
                "round": r_idx,
                "debates": len(flags),
                "rounds_with_any_flag": int(sum(1 for f in flags if f > 0)),
                "f1_n_mal_0": 1.0 if all(f == 0 for f in flags) else 0.0,
                "fpr_agent_level": float(sum(flags) / (len(flags) * 8)) if flags else 0.0,
            }
        )

    summary = {
        "params": params,
        "n_debates": n_debates,
        "n_rounds": n_rounds,
        "n_agent_rounds": n_agent_rounds,
        "total_flags": total_flags,
        "rounds_with_flags": rounds_with_flags,
        "debates_with_flags": debates_with_flags,
        "debates_with_cascade": debates_with_cascade,
        "fpr_agent_level": (total_flags / n_agent_rounds) if n_agent_rounds else 0.0,
        "debate_level_flag_rate": (debates_with_flags / n_debates) if n_debates else 0.0,
        "round_level_flag_rate": (rounds_with_flags / n_rounds) if n_rounds else 0.0,
        "per_round_f1": f1_rounds,
        "per_topology": {k: dict(v) for k, v in sorted(per_topo.items())},
        "signals": {name: stat(values) for name, values in signal_values.items()},
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--impl", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--limit-per-topo", type=int, default=None)
    parser.add_argument("--param", action="append", default=[],
                        help="Override a CASPIAN config parameter: name=value")
    parser.add_argument("--dump-records", action="store_true")
    args = parser.parse_args()

    params = dict(DEFAULTS)
    for override in args.param:
        key, _, raw = override.partition("=")
        key = key.strip()
        raw = raw.strip()
        try:
            if raw.lower() in ("true", "false"):
                value = raw.lower() == "true"
            elif "." in raw or "e" in raw.lower():
                value = float(raw)
            else:
                value = int(raw)
        except ValueError:
            value = raw
        params[key] = value

    records, used_params = replay(
        Path(args.pkl), Path(args.impl), params,
        max_rounds=args.max_rounds, limit_per_topo=args.limit_per_topo,
    )
    summary = summarise(records, used_params)
    output = {"summary": summary}
    if args.dump_records:
        output["records"] = records
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
