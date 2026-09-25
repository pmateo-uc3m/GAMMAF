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

        print(f"    Topology   : {topo}")
        print(f"    Questions  : {total_q}")
        print(f"    Correct    : {correct}")
        print(f"    Accuracy   : {acc * 100:.2f}%")

        rounds_rates = topo_result.get("rounds_rates", [])
        round_counts = topo_result.get("round_counts", {})
        if rounds_rates:
            print()
            print(f"    {'Round':>5}  {'ASR':>7}  {'UnFlagASR':>10}  {'ADR':>7}  {'AIR':>7}  {'FPR':>7}  {'F1':>8}  {'AUROC':>8}  {'Count':>6}")
            print(f"    {'─' * 75}")
            for i, rr in enumerate(rounds_rates):
                cnt = round_counts.get(i, 0)
                print(f"    {i + 1:>5}  {rr.get('ASR', 0):>7.2f}  {rr.get('UnFlagASR', 0):>10.2f}  {rr.get('ADR', 0):>7.2f}  {rr.get('AIR', 0):>7.2f}  {rr.get('FPR', 0):>7.2f}  {rr.get('F1', 0):>8.4f}  {rr.get('AUROC', 0):>8.4f}  {cnt:>6}")
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


def print_epoch_log(epoch: int, total_epochs: int, train_loss: float, val_loss: float, lr: float, is_best: bool = False, extra: str = None):
    line = f"    Epoch {epoch:03d}/{total_epochs:03d}  |  Train Loss: {train_loss:.6f}  |  Validation Loss: {val_loss:.6f}  |  LR: {lr:.6e}"
    if extra:
        line += f"  |  {extra}"
    if is_best:
        line += "  [BEST]"
    print(line)


class LRPlateauReducer:
    """Reduce the learning rate and/or stop training on validation plateaus.

    A validation loss counts as an improvement when it beats the best loss
    seen so far by at least the corresponding percentual threshold.  After
    ``patience`` consecutive epochs without such an improvement the learning
    rate is multiplied by ``factor``, floored at ``min_lr``.  Independently,
    after ``early_stop_patience`` consecutive epochs without an improvement of
    at least ``early_stop_improvement_pct`` percent, ``step`` reports that
    training should stop.  The early-stop counter is deliberately not reset by
    an LR reduction, so the LR has a chance to act before training stops.

    ``step(val_loss)`` returns ``(is_best, reduced, should_stop)``: ``is_best``
    marks a new best validation loss (used for best-checkpoint selection),
    ``reduced`` marks epochs on which the learning rate was lowered and
    ``should_stop`` marks epochs on which training should be stopped.
    """

    def __init__(self, optimizer, patience: int = 5, factor: float = 0.5, min_lr: float = 1e-6,
                 min_improvement_pct: float = 1.0, early_stop_patience: int = None,
                 early_stop_improvement_pct: float = None):
        self.optimizer = optimizer
        self.patience = max(1, int(patience))
        self.factor = float(factor)
        self.min_lr = float(min_lr)
        self.min_improvement_pct = float(min_improvement_pct)
        self.early_stop_patience = None if early_stop_patience is None else max(1, int(early_stop_patience))
        self.early_stop_improvement_pct = (
            self.min_improvement_pct if early_stop_improvement_pct is None else float(early_stop_improvement_pct)
        )
        self.best_loss = float("inf")
        self.epochs_without_improvement = 0
        self.epochs_without_improvement_early_stop = 0
        self.reductions = 0

    @property
    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def step(self, val_loss: float):
        is_best = val_loss < self.best_loss
        if is_best:
            previous = self.best_loss
            self.best_loss = float(val_loss)
            if previous == float("inf") or previous <= 0.0:
                improvement_pct = float("inf")
            else:
                improvement_pct = (previous - val_loss) / previous * 100.0
            if improvement_pct >= self.min_improvement_pct:
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            if improvement_pct >= self.early_stop_improvement_pct:
                self.epochs_without_improvement_early_stop = 0
            else:
                self.epochs_without_improvement_early_stop += 1
        else:
            self.epochs_without_improvement += 1
            self.epochs_without_improvement_early_stop += 1

        reduced = False
        if self.epochs_without_improvement >= self.patience:
            for group in self.optimizer.param_groups:
                group["lr"] = max(group["lr"] * self.factor, self.min_lr)
            self.epochs_without_improvement = 0
            self.reductions += 1
            reduced = True

        should_stop = (
            self.early_stop_patience is not None
            and self.epochs_without_improvement_early_stop >= self.early_stop_patience
        )
        return is_best, reduced, should_stop
