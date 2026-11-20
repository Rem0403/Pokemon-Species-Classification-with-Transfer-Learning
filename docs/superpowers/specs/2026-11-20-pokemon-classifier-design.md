# Pokémon Species Classification — Design Spec

Date: 2026-11-20 · Course: CMPS 4410 · Scope: **source code only** (report and slides are written by the student).

## Goal

Train a classifier that names a Pokémon species from one cropped image, and answer the proposal's three questions:

1. How much does transfer learning help vs. training from scratch?
2. How fast does accuracy fall as training images per class shrink?
3. Which species get confused, and why?

## Dataset (verified 2026-11-20)

- Source: Kaggle `lantian773030/pokemonclassification`, unzipped manually to `data/PokemonData/<Species>/<image>`.
- 6,838 files, 150 class folders, 26–66 files per class.
- Formats: 6,659 `.jpg`, 40 `.jpeg`, 121 `.png`; plus 18 non-image or broken files (12 `.svg`, 1 `.zip`, 1 `.asp`, 4 with mangled extensions).
- Includes "Alolan Sandslash" (a Gen-7 regional form). **Kept as a class**; noted as a dataset quirk.
- `data/` is git-ignored.

## Environment

- Local Windows PC, GTX 1660 Ti (6 GB), Python 3.12.
- Project-local `.venv` with CUDA-enabled PyTorch + torchvision, scikit-learn, matplotlib, numpy, pillow (`requirements.txt`).
- Falls back to CPU automatically if CUDA is unavailable.

## Data pipeline (`data.py`)

1. **Scan**: find the folder under `data/` whose subfolders are the species. Try to open each file with Pillow; files that fail are skipped and listed.
2. **Deduplicate**: MD5 of file bytes; keep the first of each exact-duplicate group (so the same image can never sit in both train and test). Near-duplicates are not detected (known limit).
3. **Split**: stratified 70/15/15 (scikit-learn `train_test_split`, seed 42), saved to `splits.json`. Created once; every experiment reuses it, so all runs share the same test set.
4. **Subsample (data-amount experiment)**: training set only, `k` images per class (k = 10, 20), deterministic with the seed. Classes with fewer than `k` training images use all they have; the actual count is recorded.
5. **Loading**: images converted to RGB; transparent PNGs are composited onto a **white** background (plain `convert("RGB")` would make transparent areas black).
6. **Transforms** (all normalized with ImageNet mean/std, 224×224):
   - Eval / no-augmentation: `Resize((224, 224))`.
   - Train with augmentation: `RandomResizedCrop(224, scale=(0.7, 1.0))`, `RandomHorizontalFlip`, `RandomRotation(15)`, `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)`. **No hue jitter**: color is a key species cue (e.g. shifting hue can turn one species' palette into another's).

`inspect_data.py` prints per-class counts, lists skipped files and duplicate groups, and saves a grid of 20 random images to `results/sample_grid.png` (proposal prep step 1).

## Models (`models.py`)

| Name | What | Trainable | Learning rate |
|---|---|---|---|
| `scratch` | 4 conv blocks (Conv-BN-ReLU-MaxPool; 32→64→128→256 channels) → global average pool → dropout 0.3 → linear(150). Random init. | all | 1e-3 |
| `frozen` | torchvision ResNet-18, `IMAGENET1K_V1` weights, `fc` replaced by linear(150). | `fc` only | 1e-3 |
| `finetune` | Same as `frozen`. | all | backbone 1e-4, `fc` 1e-3 |

For `frozen`, the backbone is held in **eval mode** during training so BatchNorm running statistics stay at their ImageNet values (otherwise "frozen" weights would still drift through BN stats).

## Training (`train.py`)

- Loss: cross-entropy. Optimizer: AdamW (default weight decay). Batch size 32. Max 20 epochs.
- Early stopping on validation top-1 accuracy, patience 5; best-validation weights are restored before testing.
- Seed 42 for Python, NumPy, and torch.
- Run as a CLI for a single configuration: `python train.py --method finetune --aug on --per-class all`.
- Outputs to `results/<run_name>/`:
  - `metrics.json`: config, train image count, per-epoch train loss / val accuracy, best epoch, test top-1, top-5, macro-F1, and per-test-image `(path, true, predicted)`.
  - `curves.png`: training loss and validation accuracy per epoch.
  - `best.pt`: best weights (git-ignored).
- Windows-safe DataLoader (`num_workers=4`, code under `if __name__ == "__main__"`).

## Experiments (`experiments.py`)

11 runs; a run is skipped if its `metrics.json` already exists (so an interrupted session can resume).

| Experiment | Runs |
|---|---|
| Training method | `scratch`, `frozen`, `finetune` — aug on, all data |
| Augmentation | `scratch`, `finetune` — aug off, all data (compared to the "on" runs above) |
| Data amount | `scratch`, `frozen`, `finetune` × k = 10, 20 — aug on (plus the "all" runs above) |

Writes `results/summary.csv` (one row per run) and charts: `method_comparison.png`, `augmentation.png`, `data_amount.png`.

## Error analysis (`analyze.py`)

Uses the stored test predictions of the best run (highest test top-1 among "all data" runs; overridable with `--run`). No GPU needed.

- `confusion_matrix.png`: 150×150 heatmap.
- `top_confusions.csv`: the 10 most frequent (true → predicted) pairs with counts.
- `confused_pairs.png`: for each top pair, a misclassified test image beside a training example of the predicted species.
- `per_class_accuracy.csv`: per-class test accuracy, sorted from worst to best.

The "why" (shape, color, evolution line) is the student's interpretation, made from these images.

## Testing (`test_smoke.py`)

Plain-assert script (`python test_smoke.py`), no framework. It builds a tiny fake dataset in a temp folder (3 classes, including one transparent PNG, one duplicate, and one corrupt file) and checks that:

- the corrupt file is skipped and the duplicate removed;
- train/val/test do not overlap and every class appears in each split;
- subsampling gives `min(k, available)` per class;
- each of the 3 methods trains 1 epoch and writes `metrics.json` with top-1, top-5, and macro-F1;
- in `frozen`, only `fc` parameters change.

## Known limits (deliberate)

- One seed per run, so no error bars. Multiple seeds would triple runtime.
- No learning-rate schedule or hyperparameter search. The settings are the proposal's starting values.
- The "20 per class" setting is capped by the smallest classes (~18 training images).

## Deliverables

Files at repo root: `requirements.txt`, `data.py`, `inspect_data.py`, `models.py`, `train.py`, `experiments.py`, `analyze.py`, `test_smoke.py`, an updated `README.md` (setup + run steps), and `.gitignore` (already done). Nothing is committed or pushed.
