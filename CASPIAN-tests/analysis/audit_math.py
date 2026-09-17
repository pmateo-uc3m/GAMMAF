#!/usr/bin/env python3
"""Numerical audit of CASPIAN internal computations against the paper formulas.

Loads an implementation file (baseline or modified) and checks the detector's
documented math:

  * degree-aware normalization  A~ = A / (sqrt(out_i * in_j) + eps)
  * lambda1/lambda2 = two largest singular values of A~
  * energy = lambda1 + lambda2
  * amplification = E_t / (E_{t-1} + eps)
  * coupling ratio R = lambda2 / (lambda1 + eps), gap = 1 - R
  * gap contraction = g_{t-1} - g_t
  * phase magnitude = |R_t - R_{t-1}| / (R_{t-1} + eps)
  * Watch = amp>1 & dg>0 & lambda1 increases
  * weak-link bottleneck / energy scale by brute-force path enumeration
  * attribution origin/amplifier/bridge by brute-force interval aggregation

Usage: python audit_math.py --impl <CASPIAN.py>
"""

import argparse
import importlib.util
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_detector(module, **overrides):
    params = dict(epsilon=1e-8, target_ema_decay=0.8, influence_ema_decay=0.8,
                  max_persistence_window=64, spine_top_k=3, top_k=2)
    params.update(overrides)
    return module.CASPIANDetector(SimpleNamespace(**params))


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    return ok


def brute_force_bottleneck(structural, weights, max_edges):
    n = structural.shape[0]
    nodes = list(range(n))
    best = 0.0
    for length in range(2, max_edges + 2):
        if length > n:
            break
        for path in itertools.permutations(nodes, length):
            if all(structural[path[i], path[i + 1]] for i in range(length - 1)):
                values = [weights[path[i], path[i + 1]] for i in range(len(path) - 1)]
                best = max(best, min(values))
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    module = load_module(args.impl, "caspian_audit")
    rng = np.random.default_rng(args.seed)
    all_ok = True

    # --- degree-aware normalization and spectral values ---------------------
    det = make_detector(module)
    raw = rng.uniform(0, 2, size=(8, 8))
    raw[rng.uniform(size=(8, 8)) < 0.4] = 0.0
    np.fill_diagonal(raw, 0.0)
    normalised = det._normalised_influence(raw)
    outgoing = raw.sum(axis=1, keepdims=True)
    incoming = raw.sum(axis=0, keepdims=True)
    reference = raw / (np.sqrt(outgoing * incoming) + det.epsilon)
    all_ok &= check("degree-aware normalisation", np.allclose(normalised, reference))

    lambda1, lambda2 = det._spectral_values(normalised)
    singular = np.linalg.svd(normalised, compute_uv=False)
    all_ok &= check("lambda1/lambda2 = top-2 singular values",
                    abs(lambda1 - singular[0]) < 1e-12 and abs(lambda2 - singular[1]) < 1e-12)

    energy = lambda1 + lambda2
    ratio = lambda2 / (lambda1 + det.epsilon)
    gap = 1.0 - ratio
    all_ok &= check("R_t = lambda2/(lambda1+eps)",
                    abs(ratio - singular[1] / (singular[0] + det.epsilon)) < 1e-12)
    all_ok &= check("g_t = 1 - R_t", abs(gap - (1 - ratio)) < 1e-12)

    # --- weak link brute force ---------------------------------------------
    structural = (raw > 0).astype(float)
    feasible, bottleneck, energy_scale = det._weak_link({"structural": structural}, normalised)
    reference_bottleneck = brute_force_bottleneck(
        structural > 0, normalised, det._graph_diameter(structural)
    )
    mask = structural > 0
    reference_energy_scale = (np.sum(normalised[mask] ** 2) /
                              (np.sum(normalised[mask]) + det.epsilon))
    all_ok &= check("weak-link bottleneck (brute force)",
                    abs(bottleneck - reference_bottleneck) < 1e-9,
                    f"impl={bottleneck:.6f} ref={reference_bottleneck:.6f}")
    all_ok &= check("weak-link energy scale",
                    abs(energy_scale - reference_energy_scale) < 1e-9,
                    f"impl={energy_scale:.6f} ref={reference_energy_scale:.6f}")
    all_ok &= check("weak-link feasibility direction",
                    feasible == (reference_bottleneck + det.epsilon >= reference_energy_scale
                                 and np.sum(normalised[mask]) > det.epsilon))

    # --- graph diameter -----------------------------------------------------
    diameter = det._graph_diameter(structural)
    dist = np.full((8, 8), np.inf)
    np.fill_diagonal(dist, 0)
    dist[structural > 0] = 1
    for k in range(8):
        for i in range(8):
            for j in range(8):
                dist[i, j] = min(dist[i, j], dist[i, k] + dist[k, j])
    finite = dist[np.isfinite(dist) & (dist > 0)]
    reference_diameter = int(np.max(finite)) if finite.size else 0
    all_ok &= check("graph diameter", diameter == reference_diameter,
                    f"impl={diameter} ref={reference_diameter}")

    # --- signal evolution against the paper formulas ------------------------
    n = 8
    det2 = make_detector(module)
    adjacency = (rng.uniform(size=(n, n)) < 0.5).astype(float)
    np.fill_diagonal(adjacency, 0)
    trace = "audit"
    det2.begin_trace(trace, adjacency)
    previous = None
    for step in range(4):
        embeddings = rng.normal(size=(n, 16))
        flags, scores = det2.predict(
            [{"agent_id": i, "st_embedding": embeddings[i]} for i in range(n)],
            adjacency, trace_id=trace,
        )
        state = det2._states[trace]
        signals = state["last_signals"]
        current_raw = state["influence"]
        current_norm = det2._normalised_influence(current_raw)
        singular = np.linalg.svd(current_norm, compute_uv=False)
        e = singular[0] + singular[1]
        r = singular[1] / (singular[0] + det2.epsilon)
        g = 1 - r
        if previous is not None:
            amp = e / (previous["energy"] + det2.epsilon)
            dg = previous["gap"] - g
            phi = abs(r - previous["ratio"]) / (previous["ratio"] + det2.epsilon)
            all_ok &= check(f"step {step}: amplification", abs(signals["amplification"] - amp) < 1e-9)
            all_ok &= check(f"step {step}: gap contraction", abs(signals["gap_contraction"] - dg) < 1e-9)
            all_ok &= check(f"step {step}: phase magnitude", abs(signals["phase_magnitude"] - phi) < 1e-9)
            watch_ref = bool(amp > 1.0 and dg > 0.0 and singular[0] > previous["lambda1"])
            all_ok &= check(f"step {step}: Watch", signals["watch"] == watch_ref)
            all_ok &= check(f"step {step}: phase shift", signals["phase_shift"] == bool(phi > dg))
        previous = {"energy": e, "ratio": r, "gap": g, "lambda1": singular[0]}
    det2.end_trace(trace)

    # --- attribution formulas on a synthetic interval -----------------------
    det3 = make_detector(module)
    state = det3._new_state(adjacency)
    records = []
    for _ in range(3):
        raw = rng.uniform(0, 1, size=(n, n)) * adjacency
        norm = det3._normalised_influence(raw)
        records.append({"raw": raw, "normalised": norm})
    attribution = det3._attribute(state, records)
    outgoing_total = records[0]["normalised"].sum(axis=1)
    origin_ref = int(np.argmax(outgoing_total))
    all_ok &= check("attribution origin", attribution["origin"] == origin_ref)
    amplifier_scores = np.zeros(n)
    bridge_scores = np.zeros(n)
    max_norm = np.zeros_like(records[0]["normalised"])
    for record in records:
        norm, raw = record["normalised"], record["raw"]
        amplifier_scores += norm.sum(axis=1) / (norm.sum(axis=0) + det3.epsilon)
        bridge_scores += raw.sum(axis=1) * raw.sum(axis=0)
        max_norm = np.maximum(max_norm, norm)
    all_ok &= check("attribution amplifier", attribution["amplifier"] == int(np.argmax(amplifier_scores)))
    all_ok &= check("attribution bridge", attribution["bridge"] == int(np.argmax(bridge_scores)))
    if attribution["spines"]:
        top = attribution["spines"][0]
        values = [max_norm[top["path"][i], top["path"][i + 1]] for i in range(len(top["path"]) - 1)]
        all_ok &= check("attribution top spine bottleneck",
                        abs(top["bottleneck"] - min(values)) < 1e-12)

    print()
    print("AUDIT RESULT:", "ALL PASS" if all_ok else "FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
