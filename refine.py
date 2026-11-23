"""tries to improve the fine-tuned model, ONLY picking by validation accuracy (avg of 3 seeds)

run: .venv/Scripts/python refine.py   (takes hours, finished runs get skipped on restart)

stage B  one change at a time vs the baseline fine-tuned resnet-18, 3 seeds each
stage C  "combined": best backbone + every option that beat the baseline on val
stage D  final = whatever has the best mean val top-1. only now does the test set get looked at:
         baseline vs final, mean +- std over the same 3 seeds
stage A  scratch / frozen retrained for up to 60 epochs (were they cut off too early at 20?)
audit    test images the final model gets wrong with >= 90% confidence (probably bad labels)
everything goes to results/refine/
"""
import csv
import json
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze import plot_pairs, suspect_labels
from data import load_splits
from train import RESULTS_DIR, train_run

OUT = RESULTS_DIR / "refine"
SEEDS = (42, 43, 44)
BASE = {"method": "finetune", "aug": True}
SINGLE = {  # changes compared to BASE (finetune resnet-18, aug on, 20 epochs, patience 5)
    "base": {},
    "noaug": {"aug": False},
    "cosine30": {"cosine": True, "epochs": 30, "patience": None},
    "smooth": {"label_smoothing": 0.1},
    "whitefill": {"rot_fill": 255},
    "effb0": {"arch": "efficientnet_b0"},
    "r50": {"arch": "resnet50"},
}
LONG = {  # stage A, only seed 42
    "long_scratch_aug": {"method": "scratch", "aug": True, "epochs": 60, "patience": 10},
    "long_scratch_noaug": {"method": "scratch", "aug": False, "epochs": 60, "patience": 10},
    "long_frozen_aug": {"method": "frozen", "aug": True, "epochs": 60, "patience": 10},
}
# cosine30 without the schedule, is the gain from the schedule or just from training longer?
ABLATION = {"long30": {"epochs": 30, "patience": None}}


def seed_stats(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def combined_changes(val_mean: dict) -> dict:
    arch = max(("base", "effb0", "r50"), key=lambda k: val_mean[k])
    changes = dict(SINGLE[arch])
    for opt in ("noaug", "cosine30", "smooth", "whitefill"):
        if val_mean[opt] > val_mean["base"]:
            changes.update(SINGLE[opt])
    if changes.get("aug") is False:
        changes.pop("rot_fill", None)  # no augmentation means no rotation
    return changes


def pick_final(val_mean: dict) -> str:
    return max(val_mean, key=val_mean.get)


LABELS = {"base": "baseline (ResNet-18)", "noaug": "augmentation off", "cosine30": "cosine LR, 30 ep",
          "smooth": "label smoothing 0.1", "whitefill": "white rotation fill", "effb0": "EfficientNet-B0",
          "r50": "ResNet-50", "combined": "combined", "long30": "30 ep, no cosine"}


def plot_configs(stats: dict, path) -> None:
    names = sorted(stats, key=lambda n: stats[n][0])
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(names) + 1.2))
    ax.barh([LABELS.get(n, n) for n in names], [stats[n][0] for n in names],
            xerr=[stats[n][1] for n in names], capsize=3,
            color=["tab:green" if n == names[-1] else "tab:blue" for n in names])
    ax.set_xlim(min(m - s for m, s in stats.values()) - 0.01,
                min(1.0, max(m + s for m, s in stats.values()) + 0.01))
    ax.set(xlabel="validation top-1 (mean of 3 seeds, bars = std)", title="Refining the fine-tuned model")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_long(acc: dict, path) -> None:
    labels = list(acc)
    fig, ax = plt.subplots(figsize=(6, 4))
    for off, i, name in ((-0.2, 0, "20 epochs"), (0.2, 1, "up to 60 epochs")):
        bars = ax.bar([x + off for x in range(len(labels))], [acc[l][i] for l in labels], width=0.4, label=name)
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xticks(range(len(labels)), labels)
    ax.set(ylabel="test top-1", ylim=(0, 1), title="Were scratch and frozen cut short?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run_config(name: str, changes: dict, seed: int, splits: dict) -> dict:
    out = OUT / f"{name}_s{seed}"
    if (out / "metrics.json").exists():
        return json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    kw = {**BASE, **changes}
    method, aug = kw.pop("method"), kw.pop("aug")
    print(f"\n=== {name} seed {seed} ===", flush=True)
    return train_run(method, aug, None, splits, out, seed=seed, name=f"{name}_s{seed}", **kw)


def main():
    splits = load_splits()
    configs = dict(SINGLE)
    runs = {name: [run_config(name, ch, s, splits) for s in SEEDS] for name, ch in configs.items()}
    val_mean = {n: seed_stats([r["val_top1"] for r in rs])[0] for n, rs in runs.items()}

    combo = combined_changes(val_mean)
    same = next((n for n, ch in SINGLE.items() if ch == combo), None)
    if same:
        print(f"\nCombined config equals '{same}', no extra run needed.")
    else:
        configs["combined"] = combo
        runs["combined"] = [run_config("combined", combo, s, splits) for s in SEEDS]
        val_mean["combined"] = seed_stats([r["val_top1"] for r in runs["combined"]])[0]
    final = pick_final(val_mean)

    # picking table, validation only
    with open(OUT / "selection.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "changes", "val_top1_mean", "val_top1_std", "best_epochs"])
        for n in sorted(val_mean, key=val_mean.get, reverse=True):
            m, s = seed_stats([r["val_top1"] for r in runs[n]])
            w.writerow([n, json.dumps(configs[n]), f"{m:.4f}", f"{s:.4f}",
                        " ".join(str(r["best_epoch"]) for r in runs[n])])

    # test comparison: baseline vs the final config val picked, same seeds
    report = {"final_config": final, "final_changes": configs[final], "seeds": list(SEEDS),
              "combined_changes": combo}
    for n in dict.fromkeys(["base", final]):
        for k in ("test_top1", "test_top5", "test_macro_f1", "val_top1"):
            report[f"{n}_{k}"] = seed_stats([r[k] for r in runs[n]])
    diffs = [f["test_top1"] - b["test_top1"] for f, b in zip(runs[final], runs["base"])]
    report["test_top1_gain_per_seed"] = diffs
    report["test_top1_gain"] = seed_stats(diffs)
    (OUT / "final.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

    print("\nValidation top-1 (mean of 3 seeds), used for every choice:")
    for n in sorted(val_mean, key=val_mean.get, reverse=True):
        m, s = seed_stats([r["val_top1"] for r in runs[n]])
        print(f"  {n:10s} {m:.4f} +- {s:.4f}   {configs[n]}")
    print(f"\nFinal config (picked on validation): {final} {configs[final]}")
    for n in dict.fromkeys(["base", final]):
        (m1, s1), (m5, _), (mf, _) = (report[f"{n}_{k}"] for k in ("test_top1", "test_top5", "test_macro_f1"))
        print(f"  TEST {n:10s} top-1 {m1:.4f} +- {s1:.4f}   top-5 {m5:.4f}   macro-F1 {mf:.4f}")
    print(f"  TEST top-1 gain per seed: {[round(d, 4) for d in diffs]}")

    # stage A: longer training for the weaker methods
    long_runs = {}
    for name, ch in LONG.items():
        r = long_runs[name] = run_config(name, ch, 42, splits)
        print(f"  {name}: val {r['val_top1']:.4f}  test {r['test_top1']:.4f}  best epoch "
              f"{r['best_epoch']} of {len(r['history'])}")

    # audit: confident mistakes from the final model (seed 42)
    m = runs[final][0]
    sus = suspect_labels(m, 0.9)
    with open(OUT / "suspect_labels.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "label", "model_says", "confidence"])
        w.writerows((p, t, q, f"{c:.3f}") for p, t, q, c in sus)
    if sus:
        plot_pairs([(p, f"label: {t}  (model: {q} {c:.0%})", q) for p, t, q, c in sus[:12]],
                   OUT / "suspect_labels.png")
    plot_configs({n: seed_stats([r["val_top1"] for r in runs[n]]) for n in val_mean}, OUT / "configs.png")
    short = {n: json.loads((RESULTS_DIR / n / "metrics.json").read_text(encoding="utf-8"))["test_top1"]
             for n in ("scratch_aug_all", "scratch_noaug_all", "frozen_aug_all")}
    plot_long({"scratch, aug": (short["scratch_aug_all"], long_runs["long_scratch_aug"]["test_top1"]),
               "scratch, no aug": (short["scratch_noaug_all"], long_runs["long_scratch_noaug"]["test_top1"]),
               "frozen, aug": (short["frozen_aug_all"], long_runs["long_frozen_aug"]["test_top1"])},
              OUT / "long_training.png")
    print(f"\n{len(sus)} test images wrong with >= 90% confidence -> {OUT / 'suspect_labels.csv'}")

    # ablation, just to understand it, not used to pick anything
    runs.update({n: [run_config(n, ch, s, splits) for s in SEEDS] for n, ch in ABLATION.items()})
    abl = {n: {k: seed_stats([r[k] for r in runs[n]]) for k in ("val_top1", "test_top1")}
           for n in ("base", *ABLATION, "cosine30")}
    (OUT / "ablation.json").write_text(json.dumps(abl, indent=1), encoding="utf-8")
    print("\nAblation (mean +- std over 3 seeds):")
    for n, d in abl.items():
        print(f"  {n:10s} val {d['val_top1'][0]:.4f} +- {d['val_top1'][1]:.4f}   "
              f"test {d['test_top1'][0]:.4f} +- {d['test_top1'][1]:.4f}")


if __name__ == "__main__":
    main()
