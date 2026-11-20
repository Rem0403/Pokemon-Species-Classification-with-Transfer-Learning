# Pokémon Species Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build PyTorch scripts that train and compare scratch / frozen / fine-tuned ResNet-18 classifiers on the 150-class Kaggle Pokémon dataset, then run every experiment and an error analysis.

**Architecture:** Flat scripts at repo root, each with one job: `data.py` (scan, dedupe, split, load) → `models.py` (3 models) → `train.py` (one run → `results/<run>/metrics.json`) → `experiments.py` (11 runs + summary charts) → `analyze.py` (confusions from stored predictions). `test_smoke.py` is a plain-assert self-check on a fake dataset.

**Tech Stack:** Python 3.12, torch + torchvision (CUDA 12.6 wheels), scikit-learn, matplotlib, numpy, pillow.

**Spec:** `docs/superpowers/specs/2026-11-20-pokemon-classifier-design.md`

## Global Constraints

- **Never `git commit` or `git push`** (user instruction). There are no commit steps.
- Python: always `.venv/Scripts/python` (Windows venv in repo root).
- Seed 42 everywhere; split 70/15/15 stratified; images 224×224; ImageNet mean/std normalization.
- Batch 32, max 20 epochs, early-stopping patience 5 on val top-1, AdamW (default weight decay), cross-entropy.
- LRs: scratch 1e-3; frozen fc 1e-3; finetune backbone 1e-4, fc 1e-3.
- Augmentation: `RandomResizedCrop(224, scale=(0.7, 1.0))`, `RandomHorizontalFlip()`, `RandomRotation(15)`, `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)`; **no hue**.
- Transparent images are composited onto white before RGB conversion.
- Any script that creates a DataLoader runs under `if __name__ == "__main__":` (Windows spawn).
- `data/`, `.venv/`, `*.pt` stay git-ignored (`.gitignore` already exists).

## Review Focus

1. Class folders with spaces in the name ("Alolan Sandslash"): must scan, split and label normally. Covered in Task 1 (fake class `"Alolan Fake"`).
2. Palette / grayscale / RGBA images: must load as 3-channel RGB with transparent areas turned white. Task 1 `test_load_image_modes`.
3. Dataset unzipped inside an extra wrapper folder (`data/archive/PokemonData/...`): `find_class_root` must still find the species folder. Task 1 `test_find_class_root_wrapper`.
4. Re-running after an interruption: `load_splits` must return the identical split, not reshuffle, and `experiments.py` must skip runs that are already finished. Task 1 `test_splits_stable`, Task 4 `test_pending_runs`.
5. Non-ASCII filenames on Windows: must load and round-trip through `splits.json`. Task 1 (fake file `pokémon_0.jpg`).

---

### Task 1: Environment + data pipeline

**Files:**
- Create: `requirements.txt`, `data.py`, `test_smoke.py`

**Interfaces:**
- Produces (`data.py`):
  - `SEED = 42`, `SPLITS_PATH = Path("splits.json")`, `DATA_DIR = Path("data")`
  - `find_class_root(data_dir: Path) -> Path`: the shallowest directory (breadth-first) that has ≥ 2 subdirectories, each directly containing files.
  - `load_image(path: str | Path) -> PIL.Image.Image`: returns RGB. If the image has transparency (mode `RGBA`/`LA`, or `P` with a `transparency` entry in `info`), convert to RGBA and `alpha_composite` it onto white first.
  - `scan(class_root: Path) -> dict`: keys `classes` (sorted folder names), `items` (list of `[posix_path_str, label_int]`, sorted), `skipped` (paths `load_image` failed on), `duplicates` (list of path groups with identical MD5; the first path of each group is kept in `items`).
  - `make_splits(data_dir: Path = DATA_DIR, seed: int = SEED) -> dict`: keys `classes`, `train`, `val`, `test` (each a list of `[path, label]`), `skipped`, `duplicates`. Two `train_test_split` calls with `stratify`: first 70% vs 30%, then the 30% split 50/50 into val and test.
  - `load_splits(data_dir: Path = DATA_DIR, path: Path = SPLITS_PATH, seed: int = SEED) -> dict`: if `path` exists, read it; otherwise `make_splits` and write JSON (`ensure_ascii=False`, utf-8).
  - `subsample(items: list, k: int | None, seed: int = SEED) -> list`: `None` returns `items` unchanged; otherwise `random.Random(seed).sample` of `min(k, n)` per label, keeping the original order.
  - `build_transform(train: bool, aug: bool) -> transforms.Compose`: augment only when `train and aug`; otherwise `Resize((224, 224))`. Always `ToTensor` + ImageNet `Normalize`.
  - `ImageDataset(items: list, transform)`: a `torch.utils.data.Dataset` returning `(tensor, label)`.

- [ ] **Step 1: Create venv and install**

```
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
`requirements.txt` lists: `torch`, `torchvision`, `scikit-learn`, `matplotlib`, `numpy`, `pillow`.
Expected: `True NVIDIA GeForce GTX 1660 Ti`.

- [ ] **Step 2: Write tests in `test_smoke.py`**

The file has a helper `make_fake_dataset(root: Path) -> Path` that builds `root/archive/PokemonData/` with 3 classes (`"Abra"`, `"Alolan Fake"`, `"Mew"`), 20 solid-color 32×32 images each (each image a different color, so no accidental duplicates). It also adds: one RGBA PNG with a fully transparent pixel in `Mew`; a file `pokémon_0.jpg` in `Abra`; an exact byte copy of one `Abra` image named `dup.jpg`; and `broken.svg` holding text. A `__main__` block runs every `test_*` function in the file and prints `ALL PASSED`. Tests use `tempfile.TemporaryDirectory`.

```python
def test_find_class_root_wrapper():  # root/archive/PokemonData found from root
    assert find_class_root(root).name == "PokemonData"

def test_scan():
    s = scan(class_root)
    assert s["classes"] == ["Abra", "Alolan Fake", "Mew"]
    assert any(p.endswith("broken.svg") for p in s["skipped"])
    assert len(s["duplicates"]) == 1 and len(s["duplicates"][0]) == 2
    assert len(s["items"]) == 62          # 60 colors + pokémon_0.jpg + RGBA png; dup.jpg removed, broken.svg skipped

def test_load_image_modes():
    for img in [load_image(rgba_png), load_image(p_with_transparency), load_image(gray_L)]:
        assert img.mode == "RGB"
    assert load_image(rgba_png).getpixel((0, 0)) == (255, 255, 255)  # transparent pixel -> white

def test_splits():
    sp = make_splits(root)
    tr, va, te = ({p for p, _ in sp[k]} for k in ("train", "val", "test"))
    assert not (tr & va or tr & te or va & te)
    for k in ("train", "val", "test"):
        assert {l for _, l in sp[k]} == {0, 1, 2}

def test_splits_stable():   # second load reads file, identical result
    a = load_splits(root, tmp / "s.json"); b = load_splits(root, tmp / "s.json")
    assert a == b and any("pokémon" in p for p, _ in a["train"] + a["val"] + a["test"])

def test_subsample():
    out = subsample(sp["train"], 5)
    assert all(sum(1 for _, l in out if l == c) == 5 for c in range(3))
    assert subsample(sp["train"], 10_000) == sp["train"] and subsample(sp["train"], None) == sp["train"]

def test_dataset_tensor():
    x, y = ImageDataset(sp["train"], build_transform(True, True))[0]
    assert tuple(x.shape) == (3, 224, 224) and isinstance(y, int)
```
`p_with_transparency` (mode `P` with `info["transparency"]`) and `gray_L` (mode `L`) are created by the test in the temp dir, outside the fake dataset.

- [ ] **Step 3: Run and see it fail**

Run: `.venv/Scripts/python test_smoke.py`. Expected: `ModuleNotFoundError: No module named 'data'`.

- [ ] **Step 4: Implement `data.py` per the Interfaces above.**

- [ ] **Step 5: Run and see it pass**

Run: `.venv/Scripts/python test_smoke.py`. Expected: `ALL PASSED`.

---

### Task 2: Models

**Files:**
- Create: `models.py`
- Modify: `test_smoke.py` (add tests)

**Interfaces:**
- Produces (`models.py`):
  - `METHODS = ("scratch", "frozen", "finetune")`
  - `build_model(method: str, num_classes: int, pretrained: bool = True) -> nn.Module`. scratch: 4 × (Conv3×3 pad 1 → BN → ReLU → MaxPool2), channels 32/64/128/256, then `AdaptiveAvgPool2d(1)` → `Flatten` → `Dropout(0.3)` → `Linear(256, n)`. frozen/finetune: `resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)` with `fc = Linear(512, n)`; frozen sets `requires_grad=False` on every non-`fc` parameter.
  - `param_groups(model: nn.Module, method: str) -> list[dict]`: scratch `[{"params": all, "lr": 1e-3}]`; frozen `[{"params": fc, "lr": 1e-3}]`; finetune `[{"params": backbone, "lr": 1e-4}, {"params": fc, "lr": 1e-3}]`.
  - `set_train_mode(model: nn.Module, method: str) -> None`: `model.eval()` for frozen (keeps BN stats fixed), else `model.train()`.

- [ ] **Step 1: Add tests**

```python
def test_models_shapes():
    for m in METHODS:
        assert build_model(m, 7, pretrained=False)(torch.randn(2, 3, 224, 224)).shape == (2, 7)

def test_frozen_only_fc_trainable():
    model = build_model("frozen", 7, pretrained=False)
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert names == {"fc.weight", "fc.bias"}
    set_train_mode(model, "frozen"); assert not model.training

def test_finetune_groups():
    g = param_groups(build_model("finetune", 7, pretrained=False), "finetune")
    assert [x["lr"] for x in g] == [1e-4, 1e-3]
```

- [ ] **Step 2: Run, expect `ModuleNotFoundError: No module named 'models'`.**
- [ ] **Step 3: Implement `models.py`.**
- [ ] **Step 4: Run `.venv/Scripts/python test_smoke.py`, expect `ALL PASSED`.**

---

### Task 3: Training + evaluation of one run

**Files:**
- Create: `train.py`
- Modify: `test_smoke.py`

**Interfaces:**
- Consumes: all of `data.py` and `models.py`.
- Produces (`train.py`):
  - `run_name(method: str, aug: bool, per_class: int | None) -> str`, e.g. `"finetune_aug_all"`, `"scratch_noaug_k10"`.
  - `train_run(method: str, aug: bool, per_class: int | None, splits: dict, out_dir: Path, epochs: int = 20, patience: int = 5, batch_size: int = 32, num_workers: int = 4, pretrained: bool = True) -> dict`. Seeds Python/NumPy/torch with 42; device is cuda if available. Each epoch: `set_train_mode`, one pass of train, then evaluate val top-1. Keep a CPU copy of the best `state_dict` (by val top-1); stop after `patience` epochs with no improvement. Restore the best weights, save `out_dir/best.pt`, evaluate on test, write `out_dir/metrics.json` and `out_dir/curves.png`, return the metrics dict.
  - `metrics.json` keys: `run`, `method`, `aug`, `per_class`, `n_train`, `history` (list of `{"epoch", "train_loss", "val_top1"}`), `best_epoch`, `val_top1`, `test_top1`, `test_top5`, `test_macro_f1`, `classes`, `test_predictions` (list of `[path, true, pred]`).
  - top-5 uses `k = min(5, num_classes)`; macro-F1 uses `sklearn.metrics.f1_score(average="macro", labels=range(num_classes), zero_division=0)`.
  - CLI: `python train.py --method {scratch,frozen,finetune} --aug {on,off} --per-class {all,10,20} [--epochs 20]`. Writes to `results/<run_name>/`.

- [ ] **Step 1: Add tests**

```python
def test_train_run_all_methods():
    for m in METHODS:
        met = train_run(m, True, None, sp, tmp / m, epochs=1, batch_size=8, num_workers=0, pretrained=False)
        saved = json.loads((tmp / m / "metrics.json").read_text(encoding="utf-8"))
        for k in ("test_top1", "test_top5", "test_macro_f1"):
            assert 0.0 <= saved[k] <= 1.0
        assert len(saved["test_predictions"]) == len(sp["test"])
        assert saved["test_top5"] == 1.0          # 3 classes -> top-k with k=3 always hits

def test_frozen_backbone_unchanged():
    torch.manual_seed(42); ref = build_model("frozen", 3, pretrained=False)
    before = {k: v.clone() for k, v in ref.state_dict().items()}
    train_run("frozen", True, None, sp, tmp / "fz", epochs=1, batch_size=8, num_workers=0, pretrained=False)
    after = torch.load(tmp / "fz" / "best.pt")
    assert all(torch.equal(before[k], after[k]) for k in before if not k.startswith("fc."))  # incl. BN buffers

def test_run_name():
    assert run_name("scratch", False, 10) == "scratch_noaug_k10" and run_name("finetune", True, None) == "finetune_aug_all"
```
`test_frozen_backbone_unchanged` relies on `train_run` calling `build_model` immediately after seeding with 42 (no other RNG use in between), so its initial weights equal `ref`'s.

- [ ] **Step 2: Run, expect `ModuleNotFoundError: No module named 'train'`.**
- [ ] **Step 3: Implement `train.py`.**
- [ ] **Step 4: Run `.venv/Scripts/python test_smoke.py`, expect `ALL PASSED`.**
- [ ] **Step 5: Real-data check:** `.venv/Scripts/python train.py --method frozen --aug on --per-class all --epochs 1`. Expected: `splits.json` is created, CUDA is used, and `results/frozen_aug_all/metrics.json` exists with `test_top1` > 0.3. Then delete that results folder so the full run starts clean.

---

### Task 4: Experiments, error analysis, data inspection

**Files:**
- Create: `experiments.py`, `analyze.py`, `inspect_data.py`
- Modify: `test_smoke.py`

**Interfaces:**
- Consumes: `train_run`, `run_name`, `load_splits`, `load_image`.
- Produces:
  - `experiments.py`: `RUNS` is the list of 11 `(method, aug, per_class)` tuples from the spec table. `pending_runs(results_dir: Path) -> list[tuple]` returns the runs without a `metrics.json`. `summarize(results_dir: Path) -> list[dict]` writes `summary.csv` (columns: run, method, aug, per_class, n_train, best_epoch, test_top1, test_top5, test_macro_f1) plus `method_comparison.png` (grouped bars of top-1/top-5/F1 for the 3 methods, aug on, all data), `augmentation.png` (top-1 on vs. off for scratch and finetune), and `data_amount.png` (top-1 vs. per_class 10/20/all, one line per method; the x-axis label "all" shows the mean training images per class). Charts whose runs are missing are skipped. `main()` trains the pending runs, then summarizes.
  - `analyze.py`: `top_confusions(preds: list, classes: list, n: int = 10) -> list[tuple[str, str, int]]` returns (true, predicted, count) for true≠pred, sorted by count descending then by name. `per_class_accuracy(preds, classes) -> list[tuple[str, float, int]]` returns (class, accuracy, n_test), sorted by accuracy ascending. `main()` with `--run`: the default is the best `*_all` run by `test_top1`. It writes `confusion_matrix.png`, `top_confusions.csv`, `per_class_accuracy.csv`, and `confused_pairs.png` (one row per pair: a misclassified test image beside the first training image of the predicted class) into `results/analysis/`.
  - `inspect_data.py`: `main()` prints the total, class count, min/median/max per class, the 5 smallest classes, the skipped files and the duplicate groups; it saves `results/sample_grid.png` (20 random images, seed 42, titled with the species).

- [ ] **Step 1: Add tests**

```python
def test_runs_table():
    assert len(RUNS) == 11 and len(set(RUNS)) == 11

def test_pending_runs():
    d = tmp / "res"; (d / "scratch_aug_all").mkdir(parents=True)
    (d / "scratch_aug_all" / "metrics.json").write_text("{}")
    assert ("scratch", True, None) not in pending_runs(d) and len(pending_runs(d)) == 10

def test_confusions():
    preds = [["a", 0, 1], ["b", 0, 1], ["c", 1, 0], ["d", 2, 2]]
    assert top_confusions(preds, ["X", "Y", "Z"]) == [("X", "Y", 2), ("Y", "X", 1)]
    assert per_class_accuracy(preds, ["X", "Y", "Z"])[0][1] == 0.0
```

- [ ] **Step 2: Run, expect `ModuleNotFoundError`.**
- [ ] **Step 3: Implement the three scripts.**
- [ ] **Step 4: Run `.venv/Scripts/python test_smoke.py`, expect `ALL PASSED`.**
- [ ] **Step 5: Run `.venv/Scripts/python inspect_data.py`.** Expected: 150 classes, 18 skipped files (spec), and `results/sample_grid.png` written. Open the grid and confirm the images and labels match.

---

### Task 5: README + full run

**Files:**
- Modify: `README.md` (rewrite as UTF-8; the current file is UTF-16)

- [ ] **Step 1: Write `README.md`.** Cover: what the project is, a link to the dataset, where to unzip it (`data/PokemonData/...`), the venv and install commands from Task 1, then the run order: `test_smoke.py` → `inspect_data.py` → `experiments.py` → `analyze.py`. Also explain where each output file lands and the known limits from the spec.
- [ ] **Step 2: Full run** (in the background, about 1 h): `.venv/Scripts/python experiments.py`. Expected: 11 `metrics.json` files, `summary.csv`, and 3 charts.
- [ ] **Step 3:** `.venv/Scripts/python analyze.py`. Expected: the 4 files in `results/analysis/`.
- [ ] **Step 4: Sanity-check the numbers.** Expect finetune_aug_all > frozen_aug_all > scratch_aug_all on top-1, and accuracy to rise with per_class for each method. If either doesn't hold, investigate before reporting (superpowers:systematic-debugging).
- [ ] **Step 5:** Confirm `git status` shows no files under `data/`, `.venv/`, or `*.pt`, and that nothing is committed.
