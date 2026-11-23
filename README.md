# Pokémon Species Classification with Transfer Learning

CMPS 4410 project. A deep learning model names a Pokémon species from a single image. The project compares three ways of training it:

| Method | What it is |
|---|---|
| `scratch` | A small 4-block CNN that starts from random weights |
| `frozen` | ResNet-18 pretrained on ImageNet; only the new last layer learns |
| `finetune` | ResNet-18 pretrained on ImageNet; every layer learns (the main model) |

The experiments answer three questions:
1. How much does transfer learning help?
2. How fast does accuracy drop with fewer images per class?
3. Which species get mixed up?

## Dataset

[7,000 Labeled Pokemon](https://www.kaggle.com/datasets/lantian773030/pokemonclassification) on Kaggle. Download the zip and unzip it into `data/`, so you get `data/PokemonData/<Species>/<image>`. The `data/` folder is git-ignored.

After cleaning, 6,808 images remain across 150 classes (25–66 per class). The cleaning:
- 12 `.svg` files can't be read as images and are skipped.
- 7 exact duplicates within a species: one copy is kept.
- 5 images appear under two species (always an evolution pair, e.g. Pidgeot/Pidgeotto). All copies are dropped, because there's no single correct label.
- One class, "Alolan Sandslash", is not a Generation 1 Pokémon. It is kept as a class.

The split is a stratified 70/15/15 (train 4,765 / val 1,021 / test 1,022, seed 42). It is saved in `splits.json` so every experiment uses the same test images.

## Setup (Windows, NVIDIA GPU)

```
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python -m pip install -r requirements.txt
```
Without an NVIDIA GPU, install torch with plain `pip install torch torchvision`. Everything still runs, but on the CPU, which is much slower.

## Run, in this order

| Command | What it does | Output |
|---|---|---|
| `.venv\Scripts\python test_smoke.py` | Quick self-check on a tiny fake dataset (about 1 min) | prints `ALL PASSED` |
| `.venv\Scripts\python inspect_data.py` | Counts images per class, lists bad files and duplicates, saves 20 random images | `results/sample_grid.png` |
| `.venv\Scripts\python experiments.py` | Trains all 11 runs (about 1–2 h on a GTX 1660 Ti). Finished runs are skipped, so it can be restarted. | `results/<run>/`, `results/summary.csv`, 3 charts |
| `.venv\Scripts\python analyze.py` | Error analysis on the best all-data run (`--run refine/cosine30_s42` for the final model) | `results/analysis/<run>/` |
| `.venv\Scripts\python refine.py` | Optional. Tries changes to the fine-tuned model (3 seeds each, picked on validation only), retrains scratch/frozen for up to 60 epochs, and lists likely mislabeled test images (several hours, restartable) | `results/refine/`, see `FINDINGS.md` |
| `.venv\Scripts\python predict.py <image> ...` | Names the Pokémon in your own images: top-5 species with confidence (final model by default) | printed |
| `.venv\Scripts\python demo.py` | Desktop window: click "Open image…", pick a picture, see it with the top-5 species (final model) | window |

To train a single configuration, for example: `.venv\Scripts\python train.py --method finetune --aug on --per-class 10`

### The 11 runs

| Experiment | Runs |
|---|---|
| Training method | scratch / frozen / finetune, augmentation on, all data |
| Augmentation | scratch / finetune with augmentation off (compared with the "on" runs) |
| Data amount | each method with 10 and 20 training images per class (compared with "all") |

## Files

| File | Purpose |
|---|---|
| `data.py` | Finds images, skips unreadable files, removes duplicates, makes the split, loads images (transparent areas become white), and holds the augmentations |
| `models.py` | The three models, their learning rates, and the frozen model's eval mode |
| `train.py` | Trains one run: AdamW, batch 32, up to 20 epochs, early stopping (patience 5) on validation accuracy. Then tests it. |
| `experiments.py` | Runs all 11 experiments and builds `summary.csv` and the charts |
| `analyze.py` | Confusion matrix (full and zoomed on the most-confused species), most-confused pairs (with pictures), and per-class accuracy |
| `predict.py` | Top-5 prediction for any image, for demos |
| `refine.py` | Refinement experiments: seeds, learning-rate schedule, label smoothing, other backbones, longer training, label audit |
| `inspect_data.py` | Dataset statistics and a sample image grid |
| `test_smoke.py` | Self-check |

### Outputs of each run (`results/<run>/`)
- `metrics.json`: test top-1, top-5, and macro-F1, the training history, and every test prediction
- `curves.png`: training loss and validation accuracy per epoch
- `best.pt`: the best model weights (git-ignored)

## Settings and choices

- Images are 224×224 and normalized with the ImageNet mean and standard deviation.
- Training augmentation: random resized crop (70–100% of the image), horizontal flip, ±15° rotation, and brightness/contrast/saturation jitter. There is **no hue jitter**, because color is a main clue for telling species apart.
- Learning rates: 1e-3 for new layers, 1e-4 for the pretrained layers being fine-tuned.
- The frozen model's BatchNorm layers stay in eval mode, so its ImageNet statistics really stay frozen.

## Known limits

- Each of the 11 main configurations runs once with one seed, so there are no error bars. `refine.py` repeats the fine-tuned runs over 3 seeds: the seed alone moves test top-1 by about ±0.5 points.
- The 11 main runs use the proposal's starting values (no learning-rate schedule). `refine.py` tests a small set of changes; there is no broad hyperparameter search.
- GPU training is not bit-for-bit reproducible (cuDNN), so a rerun can differ by a few tenths of a point.
- "20 per class" is capped by the smallest classes (as low as 17–18 training images). The data-amount chart's x-axis therefore shows the mean number of images per class actually used.
- Only exact duplicates are detected. Near-duplicates (resized or re-encoded copies of the same picture) may still sit in different splits.
