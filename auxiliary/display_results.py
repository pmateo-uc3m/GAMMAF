import argparse
import json
import sys
from pathlib import Path

ROUND_METRICS = ["ASR", "UnFlagASR", "ADR", "AIR", "FPR", "F1", "AUROC_gt", "AUROC_beh"]
LEGACY_ROUND_METRICS = ["AUROC"]
AVAILABLE_METRICS = (
    ROUND_METRICS
    + LEGACY_ROUND_METRICS
    + [f"{metric}_ci95" for metric in ROUND_METRICS + LEGACY_ROUND_METRICS]
    + ["pooled_AUROC_gt", "pooled_AUROC_beh", "pooled_AUROC"]
)
_FOUR_DECIMALS = {
    "F1",
    "AUROC_gt",
    "AUROC_beh",
    "pooled_AUROC_gt",
    "pooled_AUROC_beh",
    "AUROC",
    "pooled_AUROC",
}


def _round_metric(round_rates, metric):
    value = round_rates.get(metric)
    if value is None and metric == "AUROC_gt":
        value = round_rates.get("AUROC")
    if value is None and metric == "pooled_AUROC_gt":
        value = round_rates.get("pooled_AUROC")
    return value


def _fmt(value, kind=None):
    if value is None:
        return "-"
    try:
        if kind == "int":
            return str(int(value))
        if kind == "pct":
            return f"{float(value) * 100:.2f}%"
        if kind == "f4":
            return f"{float(value):.4f}"
        if kind == "f2":
            return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)
    return str(value)


def _metric_kind(metric):
    if metric in _FOUR_DECIMALS or metric.endswith("_ci95"):
        return "f4" if metric in _FOUR_DECIMALS else "f2"
    return "f2"


def render_table(headers, rows, aligns):
    if not rows:
        print("  (no rows match the selected filters)")
        return
    widths = [len(str(header)) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(str(cell)))

    def border(left, middle, right):
        return left + middle.join("─" * (width + 2) for width in widths) + right

    def format_row(cells):
        parts = []
        for cell, width, align in zip(cells, widths, aligns):
            text = str(cell)
            parts.append(text.ljust(width) if align == "l" else text.rjust(width))
        return "│ " + " │ ".join(parts) + " │"

    print(border("┌", "┬", "┐"))
    print(format_row(headers))
    print(border("├", "┼", "┤"))
    for row in rows:
        print(format_row(row))
    print(border("└", "┴", "┘"))


def _read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _result_path_candidates(model, tag, entry, source_path):
    if entry.get("result_file"):
        yield Path(entry["result_file"])
    if source_path is not None:
        yield source_path.parent / str(tag) / f"{model}.json"


def _payload_from_entry(model, tag, entry, source_path):
    if isinstance(entry.get("results"), list):
        return {
            "model": entry.get("model", model),
            "dataset_tag": tag,
            "loader_tag": entry.get("loader_tag"),
            "n_used_indexes": entry.get("n_used_indexes"),
            "results": entry["results"],
            "status": entry.get("status"),
            "duration_seconds": entry.get("duration_seconds"),
        }
    for candidate in _result_path_candidates(model, tag, entry, source_path):
        if candidate.is_file():
            payload = _read_json(candidate)
            if isinstance(payload, dict) and isinstance(payload.get("results"), list):
                return payload
    return None


def iter_payloads(data, source_path):
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object")
    if isinstance(data.get("results"), list):
        yield data
        return

    runs = data.get("runs")
    if isinstance(runs, dict):
        for model, run in runs.items():
            if not isinstance(run, dict):
                continue
            datasets = run.get("datasets")
            if isinstance(datasets, dict) and datasets:
                for tag, entry in datasets.items():
                    if not isinstance(entry, dict):
                        continue
                    payload = _payload_from_entry(model, tag, entry, source_path)
                    if payload is not None:
                        yield payload
            elif run.get("status") == "failed":
                yield {
                    "model": run.get("model", model),
                    "dataset_tag": None,
                    "results": [],
                    "status": "failed",
                    "error": run.get("error"),
                }
        return

    per_dataset = data.get("per_dataset_results")
    if isinstance(per_dataset, dict):
        for tag, models in per_dataset.items():
            if not isinstance(models, dict):
                continue
            for model, result_path in models.items():
                path = Path(result_path)
                if path.is_file():
                    payload = _read_json(path)
                    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
                        yield payload
        return

    raise ValueError(
        "Unrecognized JSON: expected a MainEvaluation summary ('runs' or "
        "'per_dataset_results') or a per-dataset result file ('results')."
    )


def _as_selection(values):
    selected = []
    for value in values:
        selected.extend(part.strip() for part in str(value).split(",") if part.strip())
    return {item.lower() for item in selected}


def _is_selected(value, selected):
    return not selected or str(value).lower() in selected


def _print_header(data, path):
    print("=" * 78)
    print("  GAMMAF — MainEvaluation results")
    print("=" * 78)
    print(f"  File      : {path}")
    for key, label in (
        ("script", "Script"),
        ("config_file", "Config"),
        ("created_at", "Created"),
        ("updated_at", "Updated"),
        ("model", "Model"),
        ("dataset_tag", "Dataset"),
    ):
        if data.get(key):
            print(f"  {label:<10}: {data[key]}")
    completed = data.get("completed_runs")
    if isinstance(completed, list):
        print(f"  {'Completed':<10}: {', '.join(completed) if completed else '-'}")
    print()


def _display_tables(payloads, metrics, model_filter, dataset_filter, topology_filter, round_filter):
    overall_rows = []
    round_rows = []

    for payload in payloads:
        model = payload.get("model", "-")
        tag = payload.get("dataset_tag") or "-"
        if not _is_selected(model, model_filter) or not _is_selected(tag, dataset_filter):
            continue

        for topology_result in payload.get("results", []):
            if not isinstance(topology_result, dict):
                continue
            topology = topology_result.get("topology", "unknown")
            if not _is_selected(topology, topology_filter):
                continue

            overall_auroc_gt = topology_result.get("overall_AUROC_gt")
            if overall_auroc_gt is None:
                overall_auroc_gt = topology_result.get("overall_AUROC")
            overall_rows.append(
                [
                    model,
                    tag,
                    topology,
                    _fmt(topology_result.get("total_questions"), "int"),
                    _fmt(topology_result.get("correct_answers"), "int"),
                    _fmt(topology_result.get("overall_accuracy"), "pct"),
                    _fmt(overall_auroc_gt, "f4"),
                    _fmt(topology_result.get("overall_AUROC_beh"), "f4"),
                ]
            )

            round_counts = topology_result.get("round_counts", {})
            for index, round_rates in enumerate(topology_result.get("rounds_rates", [])):
                number = index + 1
                if round_filter is not None and number != round_filter:
                    continue
                if not isinstance(round_rates, dict):
                    continue
                count = None
                if isinstance(round_counts, dict):
                    count = round_counts.get(str(index), round_counts.get(index))
                round_rows.append(
                    [model, tag, topology, number]
                    + [_fmt(_round_metric(round_rates, metric), _metric_kind(metric)) for metric in metrics]
                    + [_fmt(count, "int")]
                )

    print("  Overall")
    render_table(
        ["Model", "Dataset", "Topology", "Questions", "Correct", "Accuracy", "AUROC_gt", "AUROC_beh"],
        overall_rows,
        aligns=["l", "l", "l", "r", "r", "r", "r", "r"],
    )
    print()
    print("  Per-round metrics")
    render_table(
        ["Model", "Dataset", "Topology", "Round"] + metrics + ["Count"],
        round_rows,
        aligns=["l", "l", "l", "r"] + ["r"] * (len(metrics) + 1),
    )


def _print_failures(data):
    runs = data.get("runs") if isinstance(data, dict) else None
    if not isinstance(runs, dict):
        return
    failed = [
        (name, run.get("error", "failed"))
        for name, run in runs.items()
        if isinstance(run, dict) and run.get("status") == "failed"
    ]
    if failed:
        print()
        print("  Failed runs:")
        for name, error in failed:
            print(f"    {name}: {error}")


def _is_scalar_mapping(data):
    if not isinstance(data, dict) or not data:
        return False
    return all(isinstance(value, (int, float, str, bool)) for value in data.values())


def _format_duration(value):
    return f"{int(value // 60)}m {value % 60:.2f}s"


def _display_scalar_mapping(data, path):
    _print_header(data, path)
    looks_timing = any("seconds" in str(key) for key in data)
    rows = []
    for key, value in data.items():
        if looks_timing and isinstance(value, (int, float)) and not isinstance(value, bool):
            value = _format_duration(value)
        rows.append([key, value])
    render_table(["Key", "Value"], rows, aligns=["l", "r"])


def main():
    parser = argparse.ArgumentParser(
        description="Render a MainEvaluation result JSON as formatted tables."
    )
    parser.add_argument(
        "results_file",
        help="MainEvaluation summary.json or a per-dataset result JSON.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Only show this model (repeatable / comma-separated).",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Only show this dataset tag (repeatable / comma-separated).",
    )
    parser.add_argument(
        "--topology",
        action="append",
        default=[],
        help="Only show this topology (repeatable / comma-separated).",
    )
    parser.add_argument(
        "--round",
        type=int,
        help="Only show this round (1-based; default: all rounds).",
    )
    parser.add_argument(
        "--metrics",
        default=",".join(ROUND_METRICS),
        help="Comma-separated round metrics to show. Available: "
        + ", ".join(AVAILABLE_METRICS),
    )
    args = parser.parse_args()

    path = Path(args.results_file)
    if not path.is_file():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        data = _read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Error reading {path}: {exc}", file=sys.stderr)
        sys.exit(1)

    metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]
    unknown = [metric for metric in metrics if metric not in AVAILABLE_METRICS]
    if unknown:
        print(
            f"Error: unknown metric(s): {', '.join(unknown)}. "
            f"Available: {', '.join(AVAILABLE_METRICS)}",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.round is not None and args.round < 1:
        print("Error: --round must be >= 1", file=sys.stderr)
        sys.exit(1)

    if _is_scalar_mapping(data):
        _display_scalar_mapping(data, path)
        return

    try:
        payloads = list(iter_payloads(data, path))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    _print_header(data, path)
    _display_tables(
        payloads,
        metrics,
        _as_selection(args.model),
        _as_selection(args.dataset),
        _as_selection(args.topology),
        args.round,
    )
    _print_failures(data)


if __name__ == "__main__":
    main()
