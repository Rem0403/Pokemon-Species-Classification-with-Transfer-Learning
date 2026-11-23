"""trains and tests one setup

example: .venv/Scripts/python train.py --method finetune --aug on --per-class all
"""
import argparse
import copy
import json
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save plots to files, no window popping up
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from data import SEED, ImageDataset, build_transform, load_splits, subsample
from models import METHODS, build_model, param_groups, set_train_mode

RESULTS_DIR = Path("results")


def run_name(method: str, aug: bool, per_class: int | None) -> str:
    return f"{method}_{'aug' if aug else 'noaug'}_{'all' if per_class is None else f'k{per_class}'}"


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).float().cpu())
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    k = min(k, logits.shape[1])
    hits = (logits.topk(k, dim=1).indices == labels[:, None]).any(dim=1)
    return hits.float().mean().item()


def save_curves(history: list[dict], path: Path, title: str) -> None:
    epochs = [h["epoch"] for h in history]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    a.plot(epochs, [h["train_loss"] for h in history], marker="o")
    a.set(title="Training loss", xlabel="epoch", ylabel="cross-entropy")
    b.plot(epochs, [h["val_top1"] for h in history], marker="o", color="tab:green")
    b.set(title="Validation top-1 accuracy", xlabel="epoch", ylabel="accuracy", ylim=(0, 1))
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def train_run(method: str, aug: bool, per_class: int | None, splits: dict, out_dir: Path,
              epochs: int = 20, patience: int | None = 5, batch_size: int = 32, num_workers: int = 4,
              pretrained: bool = True, seed: int = SEED, arch: str = "resnet18",
              label_smoothing: float = 0.0, cosine: bool = False, rot_fill: int = 0,
              name: str | None = None) -> dict:
    """trains one model, keeps the epoch with the best val top-1, then tests it.

    patience=None trains every epoch. cosine=True lowers all the learning rates along a cosine curve
    down to ~0 by the last epoch. the defaults are what the original 11 experiments used.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    num_classes = len(splits["classes"])
    model = build_model(method, num_classes, pretrained, arch)  # right after seeding so each seed gets the same init
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_items = subsample(splits["train"], per_class)
    loader_kw = dict(batch_size=batch_size, num_workers=num_workers,
                     pin_memory=device.type == "cuda", persistent_workers=num_workers > 0)
    train_loader = DataLoader(ImageDataset(train_items, build_transform(True, aug, rot_fill)), shuffle=True,
                              generator=torch.Generator().manual_seed(seed), **loader_kw)
    val_loader = DataLoader(ImageDataset(splits["val"], build_transform(False, aug)), **loader_kw)
    test_loader = DataLoader(ImageDataset(splits["test"], build_transform(False, aug)), **loader_kw)

    optimizer = torch.optim.AdamW(param_groups(model, method))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) if cosine else None
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    history, best_val, best_epoch, best_state, waited = [], -1.0, 0, None, 0
    name = name or run_name(method, aug, per_class)

    for epoch in range(1, epochs + 1):
        lr = optimizer.param_groups[-1]["lr"]  # the new head's lr this epoch
        set_train_mode(model, method)
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
        if scheduler:
            scheduler.step()
        val_top1 = topk_accuracy(*predict(model, val_loader, device), 1)
        history.append({"epoch": epoch, "train_loss": total_loss / len(train_items), "val_top1": val_top1,
                        "lr": lr})
        print(f"[{name}] epoch {epoch:2d}  loss {history[-1]['train_loss']:.3f}  val top-1 {val_top1:.3f}",
              flush=True)
        if val_top1 > best_val:
            best_val, best_epoch, waited = val_top1, epoch, 0
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        else:
            waited += 1
            if patience is not None and waited >= patience:
                print(f"[{name}] early stop: no improvement for {patience} epochs", flush=True)
                break

    model.load_state_dict(best_state)  # test the best val weights, not the last epoch
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_dir / "best.pt")

    logits, y_true = predict(model, test_loader, device)
    preds = logits.argmax(dim=1)
    metrics = {
        "run": name, "method": method, "aug": aug, "per_class": per_class,
        "config": {"seed": seed, "label_smoothing": label_smoothing, "cosine": cosine,
                   "rot_fill": rot_fill, "arch": arch, "epochs": epochs, "patience": patience},
        "n_train": len(train_items), "history": history, "best_epoch": best_epoch,
        "val_top1": best_val,
        "test_top1": topk_accuracy(logits, y_true, 1),
        "test_top5": topk_accuracy(logits, y_true, 5),
        "test_macro_f1": float(f1_score(y_true, preds, average="macro",
                                        labels=list(range(num_classes)), zero_division=0)),
        "classes": splits["classes"],
        "test_predictions": [[p, int(t), int(q)] for (p, _), t, q
                             in zip(splits["test"], y_true.tolist(), preds.tolist())],
        # how confident the model is in its own answer, per test image
        "test_confidence": logits.softmax(dim=1).max(dim=1).values.tolist(),
    }
    save_curves(history, out_dir / "curves.png", name)
    # metrics.json is what marks a run as done, so write it last and atomically (temp file + rename).
    # that way a run that gets interrupted never leaves a half written one behind
    tmp = out_dir / "metrics.json.tmp"
    tmp.write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, out_dir / "metrics.json")
    print(f"[{name}] TEST top-1 {metrics['test_top1']:.3f}  top-5 {metrics['test_top5']:.3f}  "
          f"macro-F1 {metrics['test_macro_f1']:.3f}  (best epoch {best_epoch})", flush=True)
    return metrics


def parse_per_class(value: str) -> int | None:
    return None if value == "all" else int(value)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=METHODS, required=True)
    ap.add_argument("--aug", choices=["on", "off"], default="on")
    ap.add_argument("--per-class", type=parse_per_class, default=None, help="'all' or a number, e.g. 10")
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()
    aug = args.aug == "on"
    train_run(args.method, aug, args.per_class, load_splits(),
              RESULTS_DIR / run_name(args.method, aug, args.per_class), epochs=args.epochs)
