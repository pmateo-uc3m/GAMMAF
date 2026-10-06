import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

from display_results import render_table

METRICS = ["ASR", "UnFlagASR", "ADR", "AIR", "FPR", "F1", "AUROC_gt", "AUROC_beh"]
QUADRUPLE_DECIMALS = {"F1", "AUROC_gt", "AUROC_beh"}
BASELINE = "no_defense_baseline"


def _read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_artifact_path(stored, summary_path):
    if not stored:
        return None
    candidate = Path(stored)
    if candidate.is_file():
        return candidate
    relative = summary_path.parent / stored
    if relative.is_file():
        return relative
    return candidate


def _collect_artifacts(paths):
    artifacts = []
    notes = []
    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            raise SystemExit(f"Error: file not found: {path}")
        data = _read_json(path)
        if isinstance(data, dict) and isinstance(data.get("items"), list):
            artifacts.append((data, path))
            continue
        runs = data.get("runs") if isinstance(data, dict) else None
        if not isinstance(runs, dict):
            raise SystemExit(
                f"Error: unrecognized JSON in {path}: expected a MainEvaluation "
                "summary ('runs') or a per-item artifact ('items')"
            )
        for model, run in runs.items():
            datasets = run.get("datasets") if isinstance(run, dict) else None
            if not isinstance(datasets, dict):
                continue
            for tag, entry in datasets.items():
                if not isinstance(entry, dict):
                    continue
                resolved = _resolve_artifact_path(entry.get("per_item_file"), path)
                if resolved is None:
                    notes.append(
                        f"{model}/{tag}: no per-item artifact recorded "
                        "(run with evaluation.save_scores_artifact: true)"
                    )
                    continue
                if not resolved.is_file():
                    notes.append(f"{model}/{tag}: artifact missing at {resolved}")
                    continue
                artifacts.append((_read_json(resolved), resolved))
    return artifacts, notes


def _group_artifacts(artifacts):
    grouped = {}
    for data, path in artifacts:
        model = data.get("model")
        tag = data.get("dataset_tag")
        if model is None or tag is None:
            continue
        key = (tag, model)
        if key in grouped:
            print(f"Warning: duplicate artifact for {model}/{tag}; using {path}", file=sys.stderr)
        grouped[key] = data
    return grouped


def _as_selection(values, default):
    selected = []
    for value in values:
        selected.extend(part.strip() for part in str(value).split(",") if part.strip())
    return selected or list(default)


def _item_key(record):
    loader = record.get("loader_index")
    if isinstance(loader, int):
        return ("loader_index", loader)
    return ("question_index", record.get("question_index"))


def _topology_values(payload, metric, round_zero, drop_padded, topologies_filter):
    """{question_key: {topology: value}} for one round of one artifact."""
    out = {}
    for record in payload.get("items", []):
        if not isinstance(record, dict):
            continue
        if not record.get("included", True):
            continue
        if int(record.get("round", -1)) != round_zero:
            continue
        if drop_padded and record.get("padded"):
            continue
        topology = record.get("topology")
        if topologies_filter and topology not in topologies_filter:
            continue
        metrics = record.get("metrics") or {}
        if metric not in metrics:
            continue
        out.setdefault(_item_key(record), {})[topology] = float(metrics[metric])
    return out


def _rounds_with_items(payload, metric, topologies_filter, drop_padded):
    rounds = set()
    for record in payload.get("items", []):
        if not isinstance(record, dict) or not record.get("included", True):
            continue
        if topologies_filter and record.get("topology") not in topologies_filter:
            continue
        if drop_padded and record.get("padded"):
            continue
        if metric not in (record.get("metrics") or {}):
            continue
        rounds.add(int(record.get("round", -1)) + 1)
    return sorted(rounds)


def _question_pairs(values_a, values_b):
    """Paired (a, b) per question: each side averaged over the topologies
    present in both artifacts for that question."""
    pairs = []
    for key in sorted(set(values_a) & set(values_b), key=str):
        common = set(values_a[key]) & set(values_b[key])
        if not common:
            continue
        mean_a = float(np.mean([values_a[key][topology] for topology in common]))
        mean_b = float(np.mean([values_b[key][topology] for topology in common]))
        pairs.append((mean_a, mean_b))
    return pairs


def _paired_stats(pairs, alpha):
    n = len(pairs)
    if n < 2:
        return None
    a = np.asarray([pair[0] for pair in pairs], dtype=float)
    b = np.asarray([pair[1] for pair in pairs], dtype=float)
    d = a - b
    mean_d = float(d.mean())
    sd = float(d.std(ddof=1))
    if sd > 0:
        tcrit = float(stats.t.ppf(1 - alpha / 2.0, df=n - 1))
        ci_half = tcrit * sd / np.sqrt(n)
        _, t_p = stats.ttest_rel(a, b)
        d_z = mean_d / sd
    else:
        ci_half = 0.0
        t_p = 1.0 if mean_d == 0 else 0.0
        d_z = 0.0
    try:
        _, w_p = stats.wilcoxon(d)
        w_p = float(w_p)
    except ValueError:
        w_p = 1.0
    return {
        "n": n,
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": mean_d,
        "ci95_low": mean_d - ci_half,
        "ci95_high": mean_d + ci_half,
        "t_p": float(t_p),
        "wilcoxon_p": w_p,
        "d_z": float(d_z),
    }


def _tag_mean_stats(tag_pairs, alpha):
    """Equal-tags weighting: one mean difference per dataset, tested across
    datasets with a one-sample t-test / Wilcoxon."""
    tag_diffs = []
    tag_means_a = []
    tag_means_b = []
    total_questions = 0
    for pairs in tag_pairs:
        if len(pairs) < 2:
            continue
        a = np.asarray([pair[0] for pair in pairs], dtype=float)
        b = np.asarray([pair[1] for pair in pairs], dtype=float)
        tag_diffs.append(float((a - b).mean()))
        tag_means_a.append(float(a.mean()))
        tag_means_b.append(float(b.mean()))
        total_questions += len(pairs)
    n = len(tag_diffs)
    if n < 2:
        return None
    d = np.asarray(tag_diffs, dtype=float)
    mean_d = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    tcrit = float(stats.t.ppf(1 - alpha / 2.0, df=n - 1))
    ci_half = tcrit * se
    _, t_p = stats.ttest_1samp(d, 0.0)
    try:
        _, w_p = stats.wilcoxon(d)
        w_p = float(w_p)
    except ValueError:
        w_p = 1.0
    return {
        "n": n,
        "n_questions": total_questions,
        "mean_a": float(np.mean(tag_means_a)),
        "mean_b": float(np.mean(tag_means_b)),
        "mean_diff": mean_d,
        "ci95_low": mean_d - ci_half,
        "ci95_high": mean_d + ci_half,
        "t_p": float(t_p),
        "wilcoxon_p": w_p,
        "d_z": mean_d / sd if sd > 0 else 0.0,
    }


def _holm(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order, start=1):
        running = max(running, min(1.0, (m - rank + 1) * pvals[idx]))
        adjusted[idx] = running
    return adjusted


def _benjamini_hochberg(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m)
    running = 1.0
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        running = min(running, min(1.0, pvals[idx] * m / rank))
        adjusted[idx] = running
    return adjusted


def _fmt_p(value):
    if value < 1e-4:
        return "<1e-4"
    return f"{value:.4f}"


def _fmt_value(metric, value):
    if metric in QUADRUPLE_DECIMALS:
        return f"{value:.4f}"
    return f"{value:.2f}"


def _select_rounds(rounds_present, requested, final_only):
    if final_only:
        return [max(rounds_present)] if rounds_present else []
    if requested is None:
        return list(rounds_present)
    return [round_number for round_number in rounds_present if round_number in requested]


def _per_topology_rows(grouped, tags, metrics, args, topologies_filter, requested_rounds, final_only):
    rows = []
    for tag in tags:
        available = sorted(model for dataset, model in grouped if dataset == tag)
        selected = [model for model in _as_selection([args.models], available) if (tag, model) in grouped]
        if args.vs_baseline:
            if (tag, BASELINE) not in grouped:
                print(f"Warning: no '{BASELINE}' artifact for dataset '{tag}'; skipping", file=sys.stderr)
                continue
            selected = [model for model in selected if model != BASELINE]
            pairs = [(model, BASELINE) for model in selected]
        else:
            pairs = list(combinations(selected, 2))
        if not pairs:
            print(f"Warning: fewer than two models available for dataset '{tag}'; skipping", file=sys.stderr)
            continue

        for model_a, model_b in pairs:
            payload_a = grouped[(tag, model_a)]
            payload_b = grouped[(tag, model_b)]
            topologies = sorted(
                {record.get("topology") for record in payload_a.get("items", []) if isinstance(record, dict)}
                & {record.get("topology") for record in payload_b.get("items", []) if isinstance(record, dict)}
            )
            for topology in topologies:
                if topologies_filter and topology not in topologies_filter:
                    continue
                for metric in metrics:
                    rounds_present = _rounds_with_items(payload_a, metric, [topology], args.drop_padded)
                    for round_number in _select_rounds(rounds_present, requested_rounds, final_only):
                        values_a = _topology_values(payload_a, metric, round_number - 1, args.drop_padded, [topology])
                        values_b = _topology_values(payload_b, metric, round_number - 1, args.drop_padded, [topology])
                        paired = _question_pairs(values_a, values_b)
                        stats_row = _paired_stats(paired, args.alpha)
                        if stats_row is None:
                            continue
                        stats_row.update(
                            {
                                "dataset_tag": tag,
                                "topology": topology,
                                "model_a": model_a,
                                "model_b": model_b,
                                "metric": metric,
                                "round": round_number,
                            }
                        )
                        rows.append(stats_row)
    return rows


def _aggregate_rows(grouped, tags, metrics, args, topologies_filter, requested_rounds, final_only, level):
    """level='dataset': one row per dataset (questions pooled over topologies).
    level='model': one row per model pair (datasets pooled too, optionally
    equal-weighted)."""
    rows = []
    available_by_tag = {tag: {model for dataset, model in grouped if dataset == tag} for tag in tags}
    if level == "model":
        common = set.intersection(*available_by_tag.values()) if available_by_tag else set()
        candidates = [model for model in _as_selection([args.models], sorted(common)) if model in common]
        if args.vs_baseline:
            if BASELINE not in common:
                print(f"Warning: '{BASELINE}' is not available for every selected dataset; skipping", file=sys.stderr)
                return rows
            candidates = [model for model in candidates if model != BASELINE]
            pairs = [(model, BASELINE) for model in candidates]
        else:
            pairs = list(combinations(candidates, 2))
        if not pairs:
            print("Warning: fewer than two models available across all datasets; skipping", file=sys.stderr)
            return rows
    else:
        pairs = None

    pair_list = []
    if level == "dataset":
        for tag in tags:
            available = sorted(available_by_tag[tag])
            selected = [model for model in _as_selection([args.models], available) if model in available]
            if args.vs_baseline:
                if BASELINE not in available:
                    print(f"Warning: no '{BASELINE}' artifact for dataset '{tag}'; skipping", file=sys.stderr)
                    continue
                selected = [model for model in selected if model != BASELINE]
                pair_list.extend((tag, model, BASELINE) for model in selected)
            else:
                pair_list.extend((tag, model_a, model_b) for model_a, model_b in combinations(selected, 2))
    else:
        for model_a, model_b in pairs:
            for tag in tags:
                pair_list.append((tag, model_a, model_b))

    grouped_pairs = {}
    for tag, model_a, model_b in pair_list:
        if (tag, model_a) not in grouped or (tag, model_b) not in grouped:
            continue
        grouped_pairs.setdefault((model_a, model_b), []).append(tag)
    if level == "model":
        grouped_pairs = {key: sorted(set(value)) for key, value in grouped_pairs.items()}

    if level == "dataset":
        for tag, model_a, model_b in pair_list:
            payload_a = grouped.get((tag, model_a))
            payload_b = grouped.get((tag, model_b))
            if payload_a is None or payload_b is None:
                continue
            for metric in metrics:
                rounds_present = _rounds_with_items(payload_a, metric, topologies_filter, args.drop_padded)
                for round_number in _select_rounds(rounds_present, requested_rounds, final_only):
                    values_a = _topology_values(payload_a, metric, round_number - 1, args.drop_padded, topologies_filter)
                    values_b = _topology_values(payload_b, metric, round_number - 1, args.drop_padded, topologies_filter)
                    paired = _question_pairs(values_a, values_b)
                    stats_row = _paired_stats(paired, args.alpha)
                    if stats_row is None:
                        continue
                    stats_row.update(
                        {
                            "dataset_tag": tag,
                            "topology": "aggregated",
                            "model_a": model_a,
                            "model_b": model_b,
                            "metric": metric,
                            "round": round_number,
                        }
                    )
                    rows.append(stats_row)
        return rows

    for (model_a, model_b), tag_list in sorted(grouped_pairs.items()):
        for metric in metrics:
            rounds_present = set()
            for tag in tag_list:
                payload_a = grouped[(tag, model_a)]
                payload_b = grouped[(tag, model_b)]
                rounds_present.update(
                    set(_rounds_with_items(payload_a, metric, topologies_filter, args.drop_padded))
                    & set(_rounds_with_items(payload_b, metric, topologies_filter, args.drop_padded))
                )
            for round_number in _select_rounds(sorted(rounds_present), requested_rounds, final_only):
                tag_pairs = []
                tag_means_a = []
                tag_means_b = []
                pooled = []
                for tag in tag_list:
                    payload_a = grouped[(tag, model_a)]
                    payload_b = grouped[(tag, model_b)]
                    values_a = _topology_values(payload_a, metric, round_number - 1, args.drop_padded, topologies_filter)
                    values_b = _topology_values(payload_b, metric, round_number - 1, args.drop_padded, topologies_filter)
                    paired = _question_pairs(values_a, values_b)
                    if len(paired) < 2:
                        continue
                    tag_pairs.append(paired)
                    pooled.extend(paired)
                    tag_means_a.append(float(np.mean([pair[0] for pair in paired])))
                    tag_means_b.append(float(np.mean([pair[1] for pair in paired])))
                if args.weighting == "equal-tags":
                    stats_row = _tag_mean_stats(tag_pairs, args.alpha)
                else:
                    stats_row = _paired_stats(pooled, args.alpha)
                if stats_row is None:
                    continue
                stats_row.update(
                    {
                        "dataset_tag": "ALL",
                        "topology": "aggregated",
                        "model_a": model_a,
                        "model_b": model_b,
                        "metric": metric,
                        "round": round_number,
                        "datasets_used": tag_list,
                        "n_datasets": len(tag_pairs),
                    }
                )
                rows.append(stats_row)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Paired between-method comparison from MainEvaluation per-item score "
            "artifacts (requires evaluation.save_scores_artifact: true)."
        )
    )
    parser.add_argument(
        "results",
        nargs="+",
        help="MainEvaluation summary JSON and/or per-item artifact JSON file(s).",
    )
    parser.add_argument("--models", default="", help="Comma-separated model names (default: all found).")
    parser.add_argument(
        "--vs-baseline",
        action="store_true",
        help=f"Compare each selected model against '{BASELINE}'.",
    )
    parser.add_argument("--dataset", action="append", default=[], help="Only this dataset tag (repeatable).")
    parser.add_argument("--topology", action="append", default=[], help="Only this topology (repeatable).")
    parser.add_argument(
        "--aggregate",
        choices=["none", "dataset", "model"],
        default="none",
        help=(
            "none: one test per dataset/topology (default). "
            "dataset: average each question over its topologies, then test per dataset. "
            "model: same, pooled over datasets, for a single whole-model verdict."
        ),
    )
    parser.add_argument(
        "--weighting",
        choices=["pooled", "equal-tags"],
        default="pooled",
        help="For --aggregate model: pool all (dataset, question) units equally (default) or test per-dataset means.",
    )
    parser.add_argument(
        "--metrics",
        default="all",
        help="'all' or comma-separated metric names. Available: " + ", ".join(METRICS),
    )
    parser.add_argument(
        "--rounds",
        default="",
        help="Comma-separated 1-based rounds, or 'final' for the last executed round (default: all).",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level (default: 0.05).")
    parser.add_argument(
        "--correction",
        choices=["holm", "bh", "none"],
        default="holm",
        help="Multiple-comparison correction for the primary test (default: holm).",
    )
    parser.add_argument(
        "--primary-test",
        choices=["t", "wilcoxon"],
        default="t",
        help="Which p-value the correction applies to (default: t).",
    )
    parser.add_argument("--drop-padded", action="store_true", help="Drop early-stop padded rounds.")
    parser.add_argument("--json-out", default=None, help="Write the comparison rows as JSON.")
    args = parser.parse_args()

    if args.metrics.strip().lower() == "all":
        metrics = list(METRICS)
    else:
        metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]
    unknown = [metric for metric in metrics if metric not in METRICS]
    if unknown:
        raise SystemExit(f"Error: unknown metric(s): {', '.join(unknown)}")

    final_only = args.rounds.strip().lower() == "final"
    if final_only:
        requested_rounds = None
    elif args.rounds.strip():
        requested_rounds = {int(item) for item in args.rounds.split(",") if item.strip()}
        if any(round_number < 1 for round_number in requested_rounds):
            raise SystemExit("Error: --rounds must be >= 1 or 'final'")
    else:
        requested_rounds = None

    artifacts, notes = _collect_artifacts(args.results)
    grouped = _group_artifacts(artifacts)
    if not grouped:
        for note in notes:
            print(f"Note: {note}", file=sys.stderr)
        raise SystemExit("Error: no per-item artifacts found.")

    tags = _as_selection(args.dataset, sorted({tag for tag, _ in grouped}))
    topologies_filter = _as_selection(args.topology, [])

    if args.aggregate == "none":
        rows = _per_topology_rows(grouped, tags, metrics, args, topologies_filter, requested_rounds, final_only)
    else:
        rows = _aggregate_rows(
            grouped, tags, metrics, args, topologies_filter, requested_rounds, final_only, args.aggregate
        )

    if not rows:
        raise SystemExit("Error: no paired comparisons could be built (check filters and artifacts).")

    primary = [row["t_p"] if args.primary_test == "t" else row["wilcoxon_p"] for row in rows]
    if args.correction == "holm":
        adjusted = _holm(primary)
    elif args.correction == "bh":
        adjusted = _benjamini_hochberg(primary)
    else:
        adjusted = np.asarray(primary, dtype=float)
    for row, adjusted_p in zip(rows, adjusted):
        row["p_adjusted"] = float(adjusted_p)

    rows.sort(
        key=lambda row: (
            row["dataset_tag"],
            row["topology"],
            row["model_a"],
            row["model_b"],
            row["metric"],
            row["round"],
        )
    )

    print("=" * 78)
    print("  GAMMAF - paired between-method comparison")
    print("=" * 78)
    print(f"  Corrections : {args.correction} on the {args.primary_test} test")
    print(f"  Alpha       : {args.alpha}")
    print(f"  Aggregate   : {args.aggregate}" + (f" ({args.weighting})" if args.aggregate == "model" else ""))
    print(f"  Rounds      : {'final' if final_only else (sorted(requested_rounds) if requested_rounds else 'all')}")
    print(f"  Padded      : {'dropped' if args.drop_padded else 'included'}")
    print(f"  Tests       : {len(rows)}")
    print()

    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row["dataset_tag"],
                row["topology"],
                row["model_a"],
                row["model_b"],
                row["metric"],
                str(row["round"]),
                str(row["n"]),
                _fmt_value(row["metric"], row["mean_a"]),
                _fmt_value(row["metric"], row["mean_b"]),
                _fmt_value(row["metric"], row["mean_diff"]),
                f"[{_fmt_value(row['metric'], row['ci95_low'])}, {_fmt_value(row['metric'], row['ci95_high'])}]",
                _fmt_p(row["t_p"]),
                _fmt_p(row["wilcoxon_p"]),
                _fmt_p(row["p_adjusted"]),
                "*" if row["p_adjusted"] < args.alpha else "",
            ]
        )
    render_table(
        [
            "Dataset",
            "Topology",
            "Model A",
            "Model B",
            "Metric",
            "Round",
            "n",
            "Mean A",
            "Mean B",
            "Diff",
            "95% CI (diff)",
            "t p",
            "Wilcoxon p",
            "p adj",
            "sig",
        ],
        table_rows,
        aligns=["l"] * 6 + ["r"] * 9,
    )
    print()
    if args.aggregate == "none":
        print("  Paired over matched questions within each dataset/topology.")
    elif args.aggregate == "dataset":
        print("  Each question is averaged over its topologies, then paired across questions of the dataset.")
    else:
        if args.weighting == "equal-tags":
            print("  Each dataset contributes one mean difference; the test runs across dataset means (n = datasets).")
        else:
            print("  Each question is averaged over its topologies; (dataset, question) units are pooled across datasets.")
    print("  Diff = Mean A - Mean B; p adj applies to the selected primary test.")

    for note in notes:
        print(f"Note: {note}", file=sys.stderr)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "correction": args.correction,
                    "primary_test": args.primary_test,
                    "alpha": args.alpha,
                    "aggregate": args.aggregate,
                    "weighting": args.weighting,
                    "drop_padded": args.drop_padded,
                    "final_round_only": final_only,
                    "rows": rows,
                },
                handle,
                indent=2,
            )
        print(f"Comparison JSON saved to {args.json_out}")


if __name__ == "__main__":
    main()
