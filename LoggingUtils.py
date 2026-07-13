import json
import sys
from typing import Any, List, Dict

SECTION_WIDTH = 72

def log_section(title: str, width: int = SECTION_WIDTH):
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def log_subsection(title: str):
    print()
    print(f"  ─── {title} ───")


def log_info(msg: str):
    print(f"  [INFO] {msg}")


def log_warn(msg: str):
    print(f"  [WARN] {msg}")


def log_error(msg: str):
    print(f"  [ERROR] {msg}")


def log_done(msg: str):
    print(f"  [DONE] {msg}")


def log_config(label: str, value: Any):
    print(f"  {label:.<30s} {value}")


def fmt_seconds(seconds: float) -> str:
    minutes = int(seconds // 60)
    rem_seconds = seconds % 60
    return f"{minutes}m {rem_seconds:.2f}s"


def print_stats_table(stats: List[Dict], model_name: str = ""):
    if not stats:
        log_info("No statistics available.")
        return

    print(f"  {'─' * 68}")
    header = f"  Evaluation Results"
    if model_name:
        header += f" — {model_name}"
    print(header)
    print(f"  {'─' * 68}")

    for topo_result in stats:
        topo = topo_result.get("topology", "unknown")
        total_q = topo_result.get("total_questions", 0)
        correct = topo_result.get("correct_answers", 0)
        acc = topo_result.get("overall_accuracy", 0)
        acc_ci = topo_result.get("overall_accuracy_ci95", 0)

        print(f"    Topology   : {topo}")
        print(f"    Questions  : {total_q}")
        print(f"    Correct    : {correct}")
        print(f"    Accuracy   : {acc * 100:.2f}%  \u00b1 {acc_ci * 100:.2f}%")

        rounds_rates = topo_result.get("rounds_rates", [])
        if rounds_rates:
            print()
            print(f"    {'Round':>5}  {'ASR':>7}  {'UnFlagASR':>10}  {'ADR':>7}  {'AIR':>7}  {'FPR':>7}  {'F1':>8}  {'AUROC':>8}")
            print(f"    {'─' * 67}")
            for i, rr in enumerate(rounds_rates):
                print(f"    {i + 1:>5}  {rr.get('ASR', 0):>7.2f}  {rr.get('UnFlagASR', 0):>10.2f}  {rr.get('ADR', 0):>7.2f}  {rr.get('AIR', 0):>7.2f}  {rr.get('FPR', 0):>7.2f}  {rr.get('F1', 0):>8.4f}  {rr.get('AUROC', 0):>8.4f}")
        print()


def print_timing_report(timing: Dict, total_seconds: float):
    print()
    log_subsection("Timing Report")
    for key, val in timing.items():
        if key == "total_seconds":
            continue
        if isinstance(val, (int, float)):
            print(f"    {key:.<35s} {fmt_seconds(val)}")
    print(f"    {'total':.<35s} {fmt_seconds(total_seconds)}")


def print_epoch_log(epoch: int, total_epochs: int, train_loss: float, val_loss: float, lr: float, is_best: bool = False):
    line = f"    Epoch {epoch:03d}/{total_epochs:03d}  |  Train Loss: {train_loss:.6f}  |  Val Loss: {val_loss:.6f}  |  LR: {lr:.6e}"
    if is_best:
        line += "  [BEST]"
    print(line)
