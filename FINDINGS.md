# Findings

Notes from all the experiments. They are meant as source material for the report. Raw numbers are in `results/summary.csv`, `results/analysis/`, and `results/refine/`.

Test set: 1,022 images, 150 species. One test image is worth about 0.1 points of accuracy.

## 1. Main experiments (`experiments.py`, one seed each, 20 epochs max)

| Run | Test top-1 | Test top-5 | Macro-F1 |
|---|---|---|---|
| finetune, aug, all data | **0.933** | 0.984 | 0.931 |
| finetune, no aug, all data | 0.932 | 0.983 | 0.930 |
| finetune, aug, 20 per class | 0.918 | 0.983 | 0.916 |
| finetune, aug, 10 per class | 0.886 | 0.971 | 0.884 |
| frozen, aug, all data | 0.719 | 0.912 | 0.711 |
| frozen, aug, 20 per class | 0.700 | 0.905 | 0.696 |
| frozen, aug, 10 per class | 0.601 | 0.823 | 0.595 |
| scratch, aug, all data | 0.407 | 0.711 | 0.370 |
| scratch, no aug, all data | 0.438 | 0.747 | 0.412 |
| scratch, aug, 20 per class | 0.313 | 0.632 | 0.287 |
| scratch, aug, 10 per class | 0.211 | 0.534 | 0.179 |

The scratch runs had their best epoch at 19–20 of 20, so they were still improving when training stopped (see section 3).

- **Transfer learning is by far the biggest factor.** Fine-tuning beats training from scratch by about 50 points, and beats the frozen backbone by about 21 points.
- **Fine-tuning holds up well with little data.** With only 10 images per class it still scores 0.886, more than twice what the scratch CNN reaches with all the data.
- **Augmentation made no clear difference with 20 epochs.** For scratch it was even slightly worse (0.407 vs 0.438). Section 3 explains this: the scratch model was still learning when it was cut off.

## 2. Refining the fine-tuned model (`refine.py`)

**Method.** Each change was tested on its own against the baseline (fine-tuned ResNet-18, augmentation on, 20 epochs, early stopping with patience 5). Each one ran with 3 seeds (42, 43, 44), and every choice was made on **mean validation top-1**. The test set was used only once, for the baseline vs. the final pick, so the test score is not tuned.

| Change | Val top-1 (mean ± std, 3 seeds) | Best epochs |
|---|---|---|
| **Cosine learning-rate decay, 30 epochs, no early stopping** | **0.953 ± 0.003** | 26, 23, 29 |
| Combined: EfficientNet-B0 + no aug + cosine 30 | 0.949 ± 0.003 | 16, 22, 25 |
| Augmentation off | 0.948 ± 0.003 | 8, 13, 16 |
| EfficientNet-B0 backbone | 0.947 ± 0.004 | 8, 13, 8 |
| Baseline | 0.943 ± 0.003 | 8, 11, 7 |
| White corners for rotation (instead of black) | 0.943 ± 0.002 | 12, 5, 8 |
| Label smoothing 0.1 | 0.943 ± 0.003 | 11, 10, 11 |
| ResNet-50 backbone | 0.936 ± 0.003 | 3, 5, 5 |

**Final pick: cosine learning-rate decay over 30 epochs.** On the test set, using the same 3 seeds:

| | Test top-1 | Test top-5 | Macro-F1 |
|---|---|---|---|
| Baseline | 0.933 ± 0.006 | 0.988 | 0.932 |
| Cosine 30 | **0.947 ± 0.001** | 0.991 | 0.946 |

Gain per seed: +1.9, +0.7, +1.6 points (mean **+1.4 ± 0.6**). The gain is positive for every seed. The final model is also more stable across seeds (std 0.1 vs 0.6 points).

What this shows:
- **The learning-rate schedule mattered more than the architecture.** Slowly lowering the learning rate lets the model settle into a better minimum.
- **The gain comes from the schedule, not from training longer.** Ablation (`results/refine/ablation.json`): the same 30 epochs with no early stopping but a *constant* learning rate gives val 0.943 ± 0.007 and test 0.935 ± 0.006, the same as the baseline. Its best epochs (8, 11, 11) match the baseline's, so the extra epochs never helped. Only the lowering learning rate let the model keep improving, to best epochs 23–29.
- **A bigger model did worse.** ResNet-50 peaked at epoch 3–5 and then overfit. About 4,800 training images is too few to make use of the extra capacity.
- **The good changes did not add up.** "Combined" (0.949) scored below cosine alone (0.953). This is a known effect: settings tuned one at a time interact.
- **Label smoothing and white rotation corners changed nothing** (within ±0.1 points).
- **Seed noise is real.** The seed alone moved baseline test top-1 from 0.928 to 0.939. A single-seed difference under about 1 point should not be trusted. GPU training is also not bit-for-bit repeatable: the same seed gave 0.933 in section 1 and 0.928 here.

## 3. Were scratch and frozen cut short? (60 epochs, patience 10, seed 42)

| Run | 20 epochs | 60 epochs | Best epoch |
|---|---|---|---|
| scratch, aug | 0.407 | **0.569** | 53 of 60 |
| scratch, no aug | 0.438 | **0.586** | 57 of 60 |
| frozen, aug | 0.719 | **0.748** | 24 (stopped at 34) |

- **Yes, especially scratch.** It gained about 15 points and was *still improving* at epoch 60. The 20-epoch numbers in section 1 understate it. Even so, it remains far behind fine-tuning (about 0.59 vs 0.93), so the main conclusion holds.
- **Frozen gained about 3 points and then truly converged** (early stop at epoch 34). Its limit comes from the fixed ImageNet features, not from the training time.
- Augmentation still did not help scratch at 60 epochs. Its benefit usually shows up in even longer training.
- These are single-seed runs, but the gaps are much larger than the seed noise.

## 4. Error analysis

**Confused pairs.** The original model is in `results/analysis/finetune_aug_all/`, and the final model in `results/analysis/cosine30_s42/` (`confusion_zoom.png` is the readable version for slides). The repeated mistakes are between two stages of **the same evolution line**: Cubone→Marowak, Primeape→Mankey, Dugtrio→Diglett, Venusaur→Ivysaur, and Dewgong→Seel. These species share colors and shapes and often differ mainly in size or small details. The final model fixed Primeape→Mankey. Its only pairs seen more than once are Cubone→Marowak (3), Dugtrio→Diglett (2), and Venusaur→Ivysaur (2). The rest are one-off mistakes, some between unrelated species (e.g. Dewgong→Mewtwo, Drowzee→Raichu). Because so many pairs tie at a count of 1, the bottom of the "top 10" list is ordered alphabetically and should not be read as a ranking.

**Likely mislabeled test images** (`results/refine/suspect_labels.csv` and `.png`). The final model gets 11 test images wrong while being at least 90% sure. All 11 are evolution-line pairs. Checked by eye (not an expert judgment):

| Labeled as | Model says | My read of the picture |
|---|---|---|
| Mankey | Primeape | looks like Primeape (wrist cuffs, large body): label likely wrong |
| Pidgeot | Pidgeotto | short crest, no long plume, looks like Pidgeotto: label likely wrong |
| Rapidash | Ponyta | no horn, small flame mane, looks like Ponyta: label likely wrong |
| Venusaur | Ivysaur | flower bud, not a full flower: label likely wrong |
| Dewgong | Seel | pale blue body with a nose horn, looks like Seel: label likely wrong |
| Cubone | Marowak | looks like Marowak; less certain |
| Poliwhirl | Poliwrath | muscular, angry; could be Poliwrath; unclear |
| Golbat | Zubat | plush toy; unclear |
| Krabby | Kingler | unclear |
| Weedle | Kakuna | pixel art; unclear |
| Pidgeotto | Pidgey | fan art; unclear |

- **About 5 of the 11 confident mistakes are probably errors in the dataset, not the model.** That means the true accuracy is about 0.5 points higher than measured. More importantly, the dataset has some label noise right where the model struggles most: evolution pairs.
- Drawings, plush toys, and fan art also make the task harder, because they don't look like the official art.
- The labels were **not** changed. The test set was left as downloaded so that the scores stay comparable.

## 5. Short version for the report

1. Fine-tuning a pretrained ResNet-18 is the clear winner: 0.933 test top-1, against 0.72 for the frozen backbone and 0.41 for scratch at 20 epochs (0.75 and 0.59 with 60 epochs).
2. Transfer learning is most valuable with little data: 0.886 with only 10 images per class.
3. The best refinement was a cosine learning-rate schedule with 30 epochs: test top-1 **0.947 ± 0.001** over 3 seeds, +1.4 points over the baseline. It was picked on validation only. An ablation shows the gain comes from the schedule: 30 epochs at a constant learning rate gives no gain.
4. Bigger backbones, label smoothing, and stacking several changes did not help.
5. The remaining errors are mostly evolution-line look-alikes, and several of the most confident "errors" are actually mislabeled images.

## Figures for the slides

| Slide topic | File |
|---|---|
| The dataset | `results/sample_grid.png` |
| Scratch vs frozen vs fine-tune | `results/method_comparison.png` |
| Augmentation | `results/augmentation.png` |
| Data amount | `results/data_amount.png` |
| Refinement results | `results/refine/configs.png` (the x-axis starts at 0.925, not 0, so the differences look bigger) |
| Longer training | `results/refine/long_training.png` |
| Confusions | `results/analysis/cosine30_s42/confusion_zoom.png` and `confused_pairs.png` |
| Mislabeled images | `results/refine/suspect_labels.png` |
| Training curves | `results/refine/cosine30_s42/curves.png` vs `results/refine/base_s42/curves.png` |

**Live demo:** `.venv\Scripts\python predict.py <image>` prints the final model's top-5 guesses. A test image, for example `data/PokemonData/Weezing/5762079c803a40228b49cd8ee22844e4.jpg`, gives 100% Weezing. The mislabeled "Venusaur" image `data/PokemonData/Venusaur/caca9ce6f70d4af3895096b1730ab44c.jpg` gives 99% Ivysaur, which makes a good talking point. Images found online work too, but expect lower accuracy on fan art and plush toys.

## Limits

- Each refinement config has 3 seeds, the long runs have 1 seed, and the main 11 runs have 1 seed.
- Only exact duplicates were removed. Near-duplicates may sit across splits and inflate accuracy a little.
- The mislabel check is a visual judgment of 11 images, not a full audit of the dataset.
