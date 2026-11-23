"""runs every experiment from the proposal then makes summary.csv and the charts

run: .venv/Scripts/python experiments.py
runs that already finished (have a metrics.json) get skipped so you can restart it whenever
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import load_splits
from train import RESULTS_DIR, run_name, train_run

# (method, augmentation on?, images per class, None = all of them)
RUNS = [
    # 1. training method
    ("scratch", True, None), ("frozen", True, None), ("finetune", True, None),
    # 2. augmentation off
    ("scratch", False, None), ("finetune", False, None),
    # 3. less data, 10 and 20 per class
    ("scratch", True, 10), ("frozen", True, 10), ("finetune", True, 10),
    ("scratch", True, 20), ("frozen", True, 20), ("finetune", True, 20),
]
CSV_COLUMNS = ["run", "method", "aug", "per_class", "n_train", "best_epoch",
               "test_top1", "test_top5", "test_macro_f1"]
METHOD_ORDER = ["scratch", "frozen", "finetune"]


def pending_runs(results_dir: Path = RESULTS_DIR) -> list[tuple]:
    return [r for r in RUNS if not (Path(results_dir) / run_name(*r) / "metrics.json").exists()]


def _bar_labels(ax, bars):
    for b in bars:
        ax.annotate(f"{b.get_height():.2f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=8)


def summarize(results_dir: Path = RESULTS_DIR) -> list[dict]:
    results_dir = Path(results_dir)
    runs = {}
    for f in sorted(results_dir.glob("*/metrics.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        runs[m["run"]] = m
    rows = [{c: runs[r][c] for c in CSV_COLUMNS} for r in sorted(runs)]
    with open(results_dir / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for row in rows:
            w.writerow({**row, "per_class": "all" if row["per_class"] is None else row["per_class"]})

    def get(method, aug, k):
        return runs.get(run_name(method, aug, k))

    methods = [m for m in METHOD_ORDER if get(m, True, None)]
    if methods:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(methods))
        for i, (key, label) in enumerate([("test_top1", "top-1"), ("test_top5", "top-5"),
                                          ("test_macro_f1", "macro-F1")]):
            _bar_labels(ax, ax.bar(x + (i - 1) * 0.27, [get(m, True, None)[key] for m in methods],
                                   0.27, label=label))
        ax.set(xticks=x, xticklabels=methods, ylim=(0, 1.08), ylabel="test score",
               title="Training method (augmentation on, all data)")
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(results_dir / "method_comparison.png", dpi=130)
        plt.close(fig)

    methods = [m for m in METHOD_ORDER if get(m, True, None) and get(m, False, None)]
    if methods:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        x = np.arange(len(methods))
        _bar_labels(ax, ax.bar(x - 0.2, [get(m, False, None)["test_top1"] for m in methods], 0.4, label="aug off"))
        _bar_labels(ax, ax.bar(x + 0.2, [get(m, True, None)["test_top1"] for m in methods], 0.4, label="aug on"))
        ax.set(xticks=x, xticklabels=methods, ylim=(0, 1.08), ylabel="test top-1",
               title="Effect of data augmentation (all data)")
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(results_dir / "augmentation.png", dpi=130)
        plt.close(fig)

    # chart 3: x is the avg images per class that actually got used
    # (the 20 setting caps out around 18 for the smallest classes, all is ~32)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    drew = False
    for m in METHOD_ORDER:
        pts = [r for r in (get(m, True, k) for k in (10, 20, None)) if r]
        if len(pts) >= 2:
            xs = [r["n_train"] / len(r["classes"]) for r in pts]
            ax.plot(xs, [r["test_top1"] for r in pts], marker="o", label=m)
            drew = True
    if drew:
        ax.set(xlabel="training images per class (mean)", ylabel="test top-1", ylim=(0, 1),
               title="Accuracy vs. amount of training data (augmentation on)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(results_dir / "data_amount.png", dpi=130)
    plt.close(fig)
    return rows


def main():
    splits = load_splits()
    todo = pending_runs()
    print(f"{len(RUNS) - len(todo)} of {len(RUNS)} runs already done; {len(todo)} to go.")
    for i, (method, aug, k) in enumerate(todo, 1):
        print(f"\n=== run {i}/{len(todo)}: {run_name(method, aug, k)} ===", flush=True)
        train_run(method, aug, k, splits, RESULTS_DIR / run_name(method, aug, k))
    rows = summarize()
    print(f"\n{'run':22s} {'n_train':>7s} {'top-1':>6s} {'top-5':>6s} {'F1':>6s}")
    for r in rows:
        print(f"{r['run']:22s} {r['n_train']:7d} {r['test_top1']:6.3f} {r['test_top5']:6.3f} "
              f"{r['test_macro_f1']:6.3f}")
    print(f"\nWrote {RESULTS_DIR / 'summary.csv'} and charts.")


if __name__ == "__main__":
    main()
