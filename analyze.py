"""Error analysis: which species does the model mix up?

Run: .venv/Scripts/python analyze.py            (best "all data" run)
     .venv/Scripts/python analyze.py --run finetune_aug_all
     .venv/Scripts/python analyze.py --run refine/cosine30_s42   (final model)
Output: results/analysis/<run>/
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


def confused_classes(preds: list, n: int = 10) -> list[int]:
    """Sorted indices of every class in the n most frequent (true, predicted) mistakes."""
    counts = Counter((t, p) for _, t, p in preds if t != p)
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
    return sorted({c for pair, _ in top for c in pair})


def per_class_accuracy(preds: list, classes: list) -> list[tuple[str, float, int]]:
    """(species, test accuracy, number of test images), worst first."""
    total, right = Counter(), Counter()
    for _, t, p in preds:
        total[t] += 1
        right[t] += t == p
    return sorted(((classes[c], right[c] / total[c], total[c]) for c in total), key=lambda r: (r[1], r[0]))


def suspect_labels(m: dict, min_conf: float = 0.9) -> list[tuple[str, str, str, float]]:
    """Test images the model gets wrong while being very sure: (path, label, prediction, confidence),
    most confident first. These are the best candidates for mislabeled images."""
    classes = m["classes"]
    rows = [(path, classes[t], classes[p], conf)
            for (path, t, p), conf in zip(m["test_predictions"], m["test_confidence"])
            if t != p and conf >= min_conf]
    return sorted(rows, key=lambda r: -r[3])


def plot_pairs(rows: list[tuple[str, str, str]], out_path: Path) -> None:
    """One row per (image path, left title, predicted species): the image next to a training
    example of the predicted species."""
    classes_train = {}
    s = load_splits()
    for path, label in s["train"]:
        classes_train.setdefault(s["classes"][label], path)
    fig, axes = plt.subplots(len(rows), 2, figsize=(5, 2.5 * len(rows)), squeeze=False)
    for ax_row, (path, title, pred) in zip(axes, rows):
        ax_row[0].imshow(load_image(path))
        ax_row[0].set_title(title, fontsize=8)
        ax_row[1].imshow(load_image(classes_train[pred]))
        ax_row[1].set_title(f"a real {pred}", fontsize=8)
        for a in ax_row:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


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
    out = OUT_DIR / Path(run).name  # e.g. results/analysis/cosine30_s42
    out.mkdir(parents=True, exist_ok=True)
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
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Readable zoom: only the species involved in the 10 most frequent mistakes.
    idx = confused_classes(preds, 10)
    if idx:
        sub = cm[np.ix_(idx, idx)]
        fig, ax = plt.subplots(figsize=(0.55 * len(idx) + 2.5, 0.55 * len(idx) + 2))
        ax.imshow(sub, cmap="Blues")
        for i in range(len(idx)):
            for j in range(len(idx)):
                if sub[i, j]:
                    ax.text(j, i, sub[i, j], ha="center", va="center", fontsize=8,
                            color="white" if sub[i, j] > sub.max() / 2 else "black")
        names = [classes[i] for i in idx]
        ax.set_xticks(range(len(idx)), names, rotation=90, fontsize=8)
        ax.set_yticks(range(len(idx)), names, fontsize=8)
        ax.set(xlabel="predicted", ylabel="true", title="Most-confused species (test counts)")
        fig.tight_layout()
        fig.savefig(out / "confusion_zoom.png", dpi=150)
        plt.close(fig)

    pairs = top_confusions(preds, classes)
    with open(out / "top_confusions.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["true", "predicted", "count"])
        w.writerows(pairs)
    with open(out / "per_class_accuracy.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["species", "accuracy", "n_test"])
        w.writerows((s, f"{a:.3f}", k) for s, a, k in per_class_accuracy(preds, classes))

    # Picture of each confused pair: a misclassified test image next to a real example of
    # the species the model guessed.
    if pairs:
        idx = {c: i for i, c in enumerate(classes)}
        rows = [(next(path for path, tt, pp in preds if tt == idx[t] and pp == idx[p]),
                 f"true: {t}  (x{c} as {p})", p) for t, p, c in pairs]
        plot_pairs(rows, out / "confused_pairs.png")

    print("\nMost confused pairs (true -> predicted, count):")
    for t, p, c in pairs:
        print(f"  {t:>14s} -> {p:<14s} {c}")
    print(f"\nWrote confusion_matrix.png, confusion_zoom.png, top_confusions.csv, "
          f"per_class_accuracy.csv, confused_pairs.png to {out}")


if __name__ == "__main__":
    main()
