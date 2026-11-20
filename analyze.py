"""Error analysis: which species does the model mix up?

Run: .venv/Scripts/python analyze.py            (best "all data" run)
     .venv/Scripts/python analyze.py --run finetune_aug_all
Uses the test predictions saved by train.py, so no GPU or retraining is needed.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import load_image, load_splits
from train import RESULTS_DIR

OUT_DIR = RESULTS_DIR / "analysis"


def top_confusions(preds: list, classes: list, n: int = 10) -> list[tuple[str, str, int]]:
    """Most frequent (true species, predicted species, count) among wrong predictions."""
    counts = Counter((classes[t], classes[p]) for _, t, p in preds if t != p)
    return [(t, p, c) for (t, p), c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))][:n]


def per_class_accuracy(preds: list, classes: list) -> list[tuple[str, float, int]]:
    """(species, test accuracy, number of test images), worst first."""
    total, right = Counter(), Counter()
    for _, t, p in preds:
        total[t] += 1
        right[t] += t == p
    return sorted(((classes[c], right[c] / total[c], total[c]) for c in total), key=lambda r: (r[1], r[0]))


def best_run(results_dir: Path = RESULTS_DIR) -> str:
    scores = {f.parent.name: json.loads(f.read_text(encoding="utf-8"))["test_top1"]
              for f in Path(results_dir).glob("*_all/metrics.json")}
    if not scores:
        raise SystemExit("No finished '*_all' runs in results/. Run experiments.py first.")
    return max(scores, key=scores.get)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="run folder name inside results/ (default: best all-data run)")
    run = ap.parse_args().run or best_run()
    m = json.loads((RESULTS_DIR / run / "metrics.json").read_text(encoding="utf-8"))
    classes, preds = m["classes"], m["test_predictions"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Analyzing {run}: test top-1 {m['test_top1']:.3f}")

    # Confusion matrix (rows = true species, columns = predicted)
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    for _, t, p in preds:
        cm[t, p] += 1
    fig, ax = plt.subplots(figsize=(22, 22))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n), classes, rotation=90, fontsize=5)
    ax.set_yticks(range(n), classes, fontsize=5)
    ax.set(xlabel="predicted", ylabel="true", title=f"Confusion matrix — {run}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    pairs = top_confusions(preds, classes)
    with open(OUT_DIR / "top_confusions.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["true", "predicted", "count"])
        w.writerows(pairs)
    with open(OUT_DIR / "per_class_accuracy.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["species", "accuracy", "n_test"])
        w.writerows((s, f"{a:.3f}", k) for s, a, k in per_class_accuracy(preds, classes))

    # Picture of each confused pair: a misclassified test image next to a real example of
    # the species the model guessed.
    if pairs:
        train = load_splits()["train"]
        first_train = {}
        for path, label in train:
            first_train.setdefault(label, path)
        idx = {c: i for i, c in enumerate(classes)}
        fig, axes = plt.subplots(len(pairs), 2, figsize=(5, 2.5 * len(pairs)), squeeze=False)
        for row, (t, p, c) in zip(axes, pairs):
            wrong = next(path for path, tt, pp in preds if tt == idx[t] and pp == idx[p])
            row[0].imshow(load_image(wrong))
            row[0].set_title(f"true: {t}  (x{c} as {p})", fontsize=8)
            row[1].imshow(load_image(first_train[idx[p]]))
            row[1].set_title(f"a real {p}", fontsize=8)
            for a in row:
                a.axis("off")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "confused_pairs.png", dpi=130)
        plt.close(fig)

    print("\nMost confused pairs (true -> predicted, count):")
    for t, p, c in pairs:
        print(f"  {t:>14s} -> {p:<14s} {c}")
    print(f"\nWrote confusion_matrix.png, top_confusions.csv, per_class_accuracy.csv, "
          f"confused_pairs.png to {OUT_DIR}")


if __name__ == "__main__":
    main()
