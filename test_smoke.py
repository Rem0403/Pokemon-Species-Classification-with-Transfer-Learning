"""Quick self-check on a tiny fake dataset. Run: .venv/Scripts/python test_smoke.py"""
import functools
import json
import shutil
import tempfile
from pathlib import Path

import torch
from PIL import Image

from data import (ImageDataset, build_transform, find_class_root, load_image,
                  load_splits, make_splits, scan, subsample)
from models import ARCHS, METHODS, build_model, head, param_groups, set_train_mode
from train import run_name, train_run
from experiments import RUNS, pending_runs, summarize
from analyze import per_class_accuracy, suspect_labels, top_confusions
from refine import SINGLE, combined_changes, pick_final, seed_stats

TMP = Path(tempfile.mkdtemp())
CLASSES = ["Abra", "Alolan Fake", "Mew"]  # "Alolan Fake" has a space, like "Alolan Sandslash"


@functools.cache
def fake_root() -> Path:
    """TMP/fake/archive/PokemonData/<class>/... with tricky files mixed in."""
    root = TMP / "fake"
    class_root = root / "archive" / "PokemonData"
    for c, name in enumerate(CLASSES):
        d = class_root / name
        d.mkdir(parents=True)
        for i in range(20):  # every image a different color -> no accidental duplicates
            Image.new("RGB", (32, 32), (c * 80, i * 12, 100)).save(d / f"{i}.jpg")
    abra, mew = class_root / "Abra", class_root / "Mew"
    Image.new("RGB", (32, 32), (250, 1, 1)).save(abra / "pokémon_0.jpg")  # non-ASCII name
    shutil.copy(abra / "0.jpg", abra / "dup.jpg")  # exact duplicate, same class
    shutil.copy(abra / "1.jpg", mew / "also_abra.jpg")  # same image under two species
    rgba = Image.new("RGBA", (32, 32), (0, 0, 255, 255))
    rgba.putpixel((0, 0), (0, 0, 0, 0))  # one fully transparent pixel
    rgba.save(mew / "clear.png")
    (mew / "broken.svg").write_text("<svg></svg>")  # not loadable by Pillow
    return root


def class_root() -> Path:
    return fake_root() / "archive" / "PokemonData"


@functools.cache
def splits() -> dict:
    return make_splits(fake_root())


# ---------- data.py ----------

def test_find_class_root_wrapper():
    assert find_class_root(fake_root()).name == "PokemonData"


def test_scan():
    s = scan(class_root())
    assert s["classes"] == CLASSES
    assert any(p.endswith("broken.svg") for p in s["skipped"])
    assert len(s["duplicates"]) == 2 and all(len(g) == 2 for g in s["duplicates"])
    # 60 colors + pokémon_0.jpg + clear.png = 62; dup.jpg removed; svg skipped;
    # Abra/1.jpg and Mew/also_abra.jpg both dropped (label is ambiguous) -> 61
    assert len(s["items"]) == 61
    paths = [p for p, _ in s["items"]]
    assert not any(p.endswith(("dup.jpg", "Abra/1.jpg", "also_abra.jpg")) for p in paths)
    assert any(p.endswith("Abra/0.jpg") for p in paths)  # first copy of same-class dup kept


def test_load_image_modes():
    p_img = Image.new("P", (8, 8), 0)
    p_img.putpalette([0, 0, 0, 10, 20, 30] + [0] * 762)
    p_img.info["transparency"] = 0
    p_img.save(TMP / "p.png", transparency=0)
    Image.new("L", (8, 8), 128).save(TMP / "gray.png")
    clear = class_root() / "Mew" / "clear.png"
    for path in (clear, TMP / "p.png", TMP / "gray.png"):
        assert load_image(path).mode == "RGB"
    assert load_image(clear).getpixel((0, 0)) == (255, 255, 255)  # transparent -> white
    assert load_image(clear).getpixel((5, 5)) == (0, 0, 255)
    assert load_image(TMP / "p.png").getpixel((0, 0)) == (255, 255, 255)


def test_splits():
    sp = splits()
    tr, va, te = ({p for p, _ in sp[k]} for k in ("train", "val", "test"))
    assert not (tr & va or tr & te or va & te)
    assert len(tr) + len(va) + len(te) == 61
    for k in ("train", "val", "test"):
        assert {label for _, label in sp[k]} == {0, 1, 2}


def test_splits_stable():
    a = load_splits(fake_root(), TMP / "s.json")
    b = load_splits(fake_root(), TMP / "s.json")
    assert a == b
    assert any("pokémon" in p for p, _ in a["train"] + a["val"] + a["test"])


def test_subsample():
    train = splits()["train"]
    out = subsample(train, 5)
    assert all(sum(1 for _, label in out if label == c) == 5 for c in range(3))
    assert subsample(train, 10_000) == train
    assert subsample(train, None) == train


def test_dataset_tensor():
    x, y = ImageDataset(splits()["train"], build_transform(True, True))[0]
    assert tuple(x.shape) == (3, 224, 224) and isinstance(y, int)


# ---------- models.py ----------

def test_models_shapes():
    for m in METHODS:
        model = build_model(m, 7, pretrained=False).eval()
        assert model(torch.randn(2, 3, 224, 224)).shape == (2, 7)


def test_frozen_only_fc_trainable():
    model = build_model("frozen", 7, pretrained=False)
    assert {n for n, p in model.named_parameters() if p.requires_grad} == {"fc.weight", "fc.bias"}
    set_train_mode(model, "frozen")
    assert not model.training
    set_train_mode(model, "finetune")
    assert model.training


def test_param_groups():
    ft = build_model("finetune", 7, pretrained=False)
    groups = param_groups(ft, "finetune")
    assert [g["lr"] for g in groups] == [1e-4, 1e-3]
    n_params = len(list(ft.parameters()))
    assert sum(len(g["params"]) for g in groups) == n_params
    assert [g["lr"] for g in param_groups(build_model("scratch", 7), "scratch")] == [1e-3]
    fz = param_groups(build_model("frozen", 7, pretrained=False), "frozen")
    assert [g["lr"] for g in fz] == [1e-3] and len(fz[0]["params"]) == 2


# ---------- train.py ----------

def _quick_run(method: str, out: str, **kw) -> dict:
    return train_run(method, True, None, splits(), TMP / out, epochs=1, batch_size=8,
                     num_workers=0, pretrained=False, **kw)


def test_run_name():
    assert run_name("scratch", False, 10) == "scratch_noaug_k10"
    assert run_name("finetune", True, None) == "finetune_aug_all"


def test_train_run_all_methods():
    for m in METHODS:
        _quick_run(m, m)
        saved = json.loads((TMP / m / "metrics.json").read_text(encoding="utf-8"))
        for k in ("test_top1", "test_top5", "test_macro_f1", "val_top1"):
            assert 0.0 <= saved[k] <= 1.0, (k, saved[k])
        assert saved["test_top5"] == 1.0  # 3 classes -> top-k uses k=3, always hits
        assert len(saved["test_predictions"]) == len(splits()["test"])
        assert saved["n_train"] == len(splits()["train"]) and saved["best_epoch"] == 1
        assert saved["classes"] == CLASSES and len(saved["history"]) == 1
        assert (TMP / m / "curves.png").exists() and (TMP / m / "best.pt").exists()


def test_frozen_backbone_unchanged():
    torch.manual_seed(42)  # train_run seeds 42 right before building the model
    before = build_model("frozen", 3, pretrained=False).state_dict()
    _quick_run("frozen", "fz")
    after = torch.load(TMP / "fz" / "best.pt", map_location="cpu")
    backbone = [k for k in before if not k.startswith("fc.")]
    assert all(torch.equal(before[k], after[k]) for k in backbone)  # includes BatchNorm stats
    assert not torch.equal(before["fc.weight"], after["fc.weight"])


def test_train_run_subsample():
    met = train_run("scratch", False, 5, splits(), TMP / "k5", epochs=1, batch_size=8,
                    num_workers=0, pretrained=False)
    assert met["n_train"] == 15 and met["per_class"] == 5 and met["aug"] is False


# ---------- refinement options ----------

def test_archs_head_and_freeze():
    assert ARCHS == ("resnet18", "resnet50", "efficientnet_b0")
    for arch in ARCHS:
        model = build_model("frozen", 7, pretrained=False, arch=arch).eval()
        assert model(torch.randn(1, 3, 224, 224)).shape == (1, 7)
        assert head(model).out_features == 7
        trainable = {id(p) for p in model.parameters() if p.requires_grad}
        assert trainable == {id(p) for p in head(model).parameters()}, arch
        ft = build_model("finetune", 7, pretrained=False, arch=arch)
        g = param_groups(ft, "finetune")
        assert [x["lr"] for x in g] == [1e-4, 1e-3]
        assert {id(p) for p in g[1]["params"]} == {id(p) for p in head(ft).parameters()}
        assert sum(len(x["params"]) for x in g) == len(list(ft.parameters()))


def test_rotation_fill_white():
    torch.manual_seed(0)
    white = Image.new("RGB", (64, 64), (255, 255, 255))
    t = build_transform(True, True, fill=255)
    # every pixel of a white image stays white (no black rotation corners); jitter can't change pure white
    # by more than brightness*255, so check the normalized value of channel 0 stays near white
    for _ in range(5):
        x = t(white)
        assert x[0].min() > (0.7 - 0.485) / 0.229, x[0].min()
    xb = build_transform(True, True, fill=0)
    torch.manual_seed(1)
    mins = [xb(white)[0].min().item() for _ in range(10)]
    assert min(mins) < (0.1 - 0.485) / 0.229  # default: black corners appear on at least one rotation


def test_train_run_options():
    sp = splits()
    a = train_run("finetune", True, None, sp, TMP / "opt", epochs=3, patience=None, batch_size=8,
                  num_workers=0, pretrained=False, seed=7, label_smoothing=0.1, cosine=True,
                  rot_fill=255, arch="efficientnet_b0", name="custom")
    assert a["run"] == "custom" and len(a["history"]) == 3
    lrs = [h["lr"] for h in a["history"]]
    assert lrs[0] == 1e-3 and lrs[1] < lrs[0] and lrs[2] < lrs[1]  # head LR decays with cosine
    assert a["config"] == {"seed": 7, "label_smoothing": 0.1, "cosine": True, "rot_fill": 255,
                           "arch": "efficientnet_b0", "epochs": 3, "patience": None}
    assert len(a["test_confidence"]) == len(sp["test"])
    assert all(0.0 < c <= 1.0 for c in a["test_confidence"])
    saved = json.loads((TMP / "opt" / "metrics.json").read_text(encoding="utf-8"))
    assert saved["config"]["arch"] == "efficientnet_b0"
    b = _quick_run("scratch", "lr_const")
    assert [h["lr"] for h in b["history"]] == [1e-3]  # no schedule by default


def test_seed_sets_init():
    # A frozen backbone never changes, so its saved weights reveal which seed built the model.
    train_run("frozen", True, None, splits(), TMP / "s1", epochs=1, batch_size=8, num_workers=0,
              pretrained=False, seed=1)
    saved = torch.load(TMP / "s1" / "best.pt")["conv1.weight"]
    torch.manual_seed(1)
    assert torch.equal(saved, build_model("frozen", 3, pretrained=False).state_dict()["conv1.weight"])
    torch.manual_seed(42)
    assert not torch.equal(saved, build_model("frozen", 3, pretrained=False).state_dict()["conv1.weight"])


# ---------- experiments.py / analyze.py ----------

def test_runs_table():
    assert len(RUNS) == 11 and len(set(RUNS)) == 11
    assert {("scratch", True, None), ("frozen", True, None), ("finetune", True, None)} <= set(RUNS)
    assert {("scratch", False, None), ("finetune", False, None)} <= set(RUNS)
    assert {(m, True, k) for m in METHODS for k in (10, 20)} <= set(RUNS)


def test_pending_runs():
    d = TMP / "res"
    (d / "scratch_aug_all").mkdir(parents=True)
    (d / "scratch_aug_all" / "metrics.json").write_text("{}")
    pending = pending_runs(d)
    assert ("scratch", True, None) not in pending and len(pending) == 10


def test_summarize():
    d = TMP / "sum"
    for m in METHODS:  # copy the 1-epoch runs from test_train_run_all_methods
        shutil.copytree(TMP / m, d / run_name(m, True, None))
    rows = summarize(d)
    assert [r["run"] for r in rows] == ["finetune_aug_all", "frozen_aug_all", "scratch_aug_all"]
    header = (d / "summary.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header == "run,method,aug,per_class,n_train,best_epoch,test_top1,test_top5,test_macro_f1"
    assert (d / "method_comparison.png").exists()


def test_confusions():
    preds = [["a", 0, 1], ["b", 0, 1], ["c", 1, 0], ["d", 2, 2], ["e", 0, 0]]
    assert top_confusions(preds, ["X", "Y", "Z"]) == [("X", "Y", 2), ("Y", "X", 1)]
    acc = per_class_accuracy(preds, ["X", "Y", "Z"])
    assert acc[0] == ("Y", 0.0, 1) and acc[-1] == ("Z", 1.0, 1)
    assert ("X", 1 / 3, 3) in acc


# ---------- refine.py ----------

def test_combined_changes():
    val = {"base": 0.90, "noaug": 0.89, "cosine30": 0.91, "smooth": 0.92, "whitefill": 0.95,
           "effb0": 0.93, "r50": 0.88}
    ch = combined_changes(val)
    assert ch == {"arch": "efficientnet_b0", "cosine": True, "epochs": 30, "patience": None,
                  "label_smoothing": 0.1, "rot_fill": 255}
    # augmentation off wins -> rotation fill is meaningless and dropped; base arch kept
    val2 = {**val, "noaug": 0.99, "effb0": 0.5, "whitefill": 0.95}
    assert combined_changes(val2) == {"aug": False, "cosine": True, "epochs": 30, "patience": None,
                                      "label_smoothing": 0.1}
    assert combined_changes({k: (1.0 if k == "base" else 0.1) for k in SINGLE}) == {}


def test_pick_final_and_stats():
    assert pick_final({"base": 0.9, "smooth": 0.93, "combined": 0.92}) == "smooth"
    mean, std = seed_stats([0.90, 0.92, 0.94])
    assert abs(mean - 0.92) < 1e-12 and abs(std - 0.02) < 1e-12
    assert seed_stats([0.5]) == (0.5, 0.0)


def test_suspect_labels():
    m = {"classes": ["X", "Y"],
         "test_predictions": [["a", 0, 1], ["b", 0, 1], ["c", 1, 1], ["d", 1, 0]],
         "test_confidence": [0.95, 0.60, 0.99, 0.97]}
    assert suspect_labels(m, 0.9) == [("d", "Y", "X", 0.97), ("a", "X", "Y", 0.95)]


def test_confused_classes():
    from analyze import confused_classes
    preds = [["a", 0, 1], ["b", 0, 1], ["c", 3, 2], ["f", 3, 2], ["d", 4, 4], ["e", 1, 0]]
    assert confused_classes(preds, n=2) == [0, 1, 2, 3]  # every class in the top-2 pairs, sorted
    assert confused_classes(preds, n=1) == [0, 1]


def test_predict_image():
    from predict import predict_image
    model = build_model("scratch", len(CLASSES), pretrained=False)
    img = class_root() / "Abra" / "0.jpg"
    top = predict_image(img, model, CLASSES, torch.device("cpu"), k=5)
    assert len(top) == 3  # k is capped at the number of classes
    assert sorted(n for n, _ in top) == sorted(CLASSES)
    probs = [p for _, p in top]
    assert probs == sorted(probs, reverse=True) and abs(sum(probs) - 1) < 1e-5


def test_refine_charts():
    from refine import plot_configs, plot_long
    plot_configs({"base": (0.94, 0.003), "cosine30": (0.95, 0.002)}, TMP / "cfg.png")
    plot_long({"scratch, aug": (0.41, 0.57), "frozen, aug": (0.72, 0.75)}, TMP / "long.png")
    assert (TMP / "cfg.png").stat().st_size > 0 and (TMP / "long.png").stat().st_size > 0


if __name__ == "__main__":
    try:
        for name, fn in list(globals().items()):
            if name.startswith("test_") and callable(fn):
                fn()
                print("ok  ", name)
        print("ALL PASSED")
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
