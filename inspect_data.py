"""quick look at the dataset: images per class, bad files, duplicates, and 20 random pics

run: .venv/Scripts/python inspect_data.py
"""
import random
import statistics
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import SEED, load_image, load_splits
from train import RESULTS_DIR


def main():
    s = load_splits()
    classes = s["classes"]
    items = s["train"] + s["val"] + s["test"]
    per_class = Counter(classes[label] for _, label in items)
    counts = sorted(per_class.values())

    print(f"Usable images: {len(items)}   classes: {len(classes)}")
    print(f"Images per class: min {counts[0]}, median {statistics.median(counts)}, max {counts[-1]}")
    print(f"Split: train {len(s['train'])}, val {len(s['val'])}, test {len(s['test'])}")
    print("Smallest classes:", ", ".join(f"{c} ({n})" for c, n in sorted(per_class.items(), key=lambda kv: kv[1])[:5]))
    print(f"\nSkipped (not readable as an image): {len(s['skipped'])}")
    for p in s["skipped"]:
        print("  ", p)
    print(f"\nExact duplicate groups: {len(s['duplicates'])} "
          "(same species: one copy kept; different species: all copies dropped)")
    for g in s["duplicates"]:
        print("  ", " == ".join(g))

    RESULTS_DIR.mkdir(exist_ok=True)
    sample = random.Random(SEED).sample(items, 20)
    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    for ax, (path, label) in zip(axes.flat, sample):
        ax.imshow(load_image(path))
        ax.set_title(classes[label], fontsize=9)
        ax.axis("off")
    fig.suptitle("20 random images with their labels")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "sample_grid.png", dpi=110)
    plt.close(fig)
    print(f"\nSaved {RESULTS_DIR / 'sample_grid.png'}")


if __name__ == "__main__":
    main()
