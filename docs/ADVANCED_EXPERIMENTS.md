# Advanced ordinal and multimodal experiments

This branch adds an experimental pipeline without changing the legacy model or
checkpoint formats. It is intentionally not executed as part of repository
setup because the intended experiments require a GPU.

## What is implemented

- Untouched, building-grouped test sets with optional spatial-group holdout.
- Explicit split manifests and metadata for each seed.
- Swin-T, ConvNeXt-Tiny, and DINOv2 ViT-S/14 encoders.
- Configurable image resolution using original image paths instead of upscaling
  the legacy 224-pixel arrays. The controlled matrix fixes all primary models at
  224 pixels.
- Multi-view attention pooling at the building level.
- Cross-entropy combined with squared EMD and a rank-consistent cumulative head.
- Equal-dimensional image/tabular projections with gated, FiLM, or concatenation
  fusion.
- Image-only and tabular-only auxiliary heads plus tabular-modality dropout.
- Tabular-only ANN (CE and ordinal), logistic-regression, and random-forest baselines.
- Real, masked, and shuffled-tabular inference conditions for fusion models.
- Accuracy, balanced accuracy, macro metrics, class-index MAE, expected-index
  MAE, quadratic weighted kappa, and calibration error.
- Active-learning ranking using entropy, prediction margin, and disagreement
  between image and tabular heads.

The existing augmentation and optimization policy is retained. This branch does
not implement the separate augmentation/regularization recommendations.

## 1. Create leakage-safe split manifests

Building-level repeated splits:

```bash
python -m examples.create_advanced_splits \
  --input data/all_data.csv \
  --output-dir data/advanced_splits \
  --seeds 13 37 71
```

For the stronger spatial generalization experiment, first add a neighborhood or
grid-block column to the input CSV, then pass it as the grouping unit:

```bash
python -m examples.create_advanced_splits \
  --input data/all_data_with_blocks.csv \
  --output-dir data/advanced_spatial_splits \
  --seeds 13 37 71 \
  --spatial-group-column spatial_block
```

The test set is selected from dataset 1 so every model can be evaluated on the
same buildings. Dataset-2-only buildings may enter training or validation but
are never appended to the common test set. No balancing operation changes the
test set.

## 2. Build path-based datasets

Build dataset 1 for one split:

```bash
python -m examples.build_advanced_dataset \
  --manifest data/advanced_splits/seed_13_building.csv \
  --images-dir data/images \
  --output-dir input/advanced/seed_13 \
  --dataset dataset1
```

Repeat with `--dataset dataset2` for the larger image-only experiment. If a CSV
contains multiple image filename columns for each building, supply them with
`--image-columns view_1 view_2 view_3`. The builder stores file paths, allowing
the trainer to read the original files at 224, 392, or another resolution.

## 3. Configure the experiment

The controlled matrix in `configs/advanced_matrix.json` contains 14
configurations for each seed:

| Family | Dataset 1 experiments | Dataset 2 experiment |
|---|---|---|
| Swin-T | CE image, ordinal image, ordinal concatenation, ordinal gating | Ordinal image |
| DINOv2 | CE image, ordinal image, ordinal concatenation, ordinal gating | Ordinal image |
| Tabular | ANN CE, ANN ordinal, logistic regression, random forest | Not applicable |

All image experiments use one 224-by-224 image per building. Dataset-1
image-only results are the controlled baselines for fusion; dataset-2 results
measure the separate effect of additional image data. DINOv2 uses 14-pixel
patches, and 224 produces an exact 16-by-16 patch grid. The first DINOv2 run
retrieves the pretrained model through PyTorch Hub; pre-populate the server
cache if compute nodes have no internet.

## 4. Train on the GPU server

```bash
python -m examples.train_advanced \
  --config configs/advanced_experiment.json \
  --data-dir input/advanced/seed_13 \
  --output-dir output/advanced/seed_13_dinov2_gated_ordinal
```

The directory contains the resolved configuration, training history,
training-only tabular normalization constants, and the best checkpoint. Model
selection uses validation macro-F1 with a configurable MAE penalty. The test set
must not be used for model or probability-blend selection.

To materialize the complete 14-model by three-seed matrix without starting
training, run:

```bash
python -m examples.run_advanced_matrix
```

This writes 42 resolved configuration files and a `commands.txt` file. After
checking the paths and GPU batch size, execute the matrix sequentially with:

```bash
python -m examples.run_advanced_matrix --execute --evaluate
```

The runner supports regular-expression filtering. For example,
`--include-pattern '^(swin_|ann_)'` selects only the five Swin-T and two ANN
configurations per seed. Logistic regression and random forest are deliberately
not matched by that filter.

Run only the Swin-T and ANN configurations on physical GPU 1 with:

```bash
mkdir -p logs

CUDA_VISIBLE_DEVICES=1 nohup python -u -m examples.run_advanced_matrix \
  --data-root input/advanced \
  --output-root output/swin_ann_matrix \
  --include-pattern '^(swin_|ann_)' \
  --execute \
  --evaluate \
  > logs/swin_ann_matrix_gpu1.log 2>&1 &

echo $! > logs/swin_ann_matrix_gpu1.pid
```

This selects 21 training jobs: five Swin-T and two ANN configurations for each
of three seeds. Test evaluation adds real-condition evaluation for image-only
models and real, masked, and shuffled conditions for fused Swin-T models.

## 5. Evaluate once on the held-out test set

```bash
python -m examples.evaluate_advanced \
  --checkpoint output/advanced/seed_13_dinov2_gated_ordinal/best_model.pth \
  --data input/advanced/seed_13/test_dataset1.pkl \
  --output-dir output/advanced/seed_13_dinov2_gated_ordinal/test
```

For any model trained with tabular inputs, repeat evaluation with the tabular
inputs masked and shuffled:

```bash
python -m examples.evaluate_advanced \
  --checkpoint output/advanced/seed_13_dinov2_gated_ordinal/best_model.pth \
  --data input/advanced/seed_13/test_dataset1.pkl \
  --output-dir output/advanced/seed_13_dinov2_gated_ordinal/test \
  --tabular-condition masked

python -m examples.evaluate_advanced \
  --checkpoint output/advanced/seed_13_dinov2_gated_ordinal/best_model.pth \
  --data input/advanced/seed_13/test_dataset1.pkl \
  --output-dir output/advanced/seed_13_dinov2_gated_ordinal/test \
  --tabular-condition shuffled \
  --shuffle-seed 2026
```

Passing `--evaluate` to the matrix runner performs the real evaluation for
image-only models and all three conditions for concatenated/gated models.
Tabular-only training writes its test metrics and predictions directly.

Run every ablation on the same test manifest within a seed. Aggregate results
across the three seeds and use paired building-level bootstrap intervals for
comparisons.

For any two models evaluated on the same buildings:

```bash
python -m examples.compare_predictions \
  --reference output/advanced/baseline/test/predictions.csv \
  --candidate output/advanced/proposed/test/predictions.csv \
  --output output/advanced/comparisons/proposed_vs_baseline.json
```

## 6. Rank buildings for targeted labeling

Export fused and unimodal probabilities for an unlabeled pool, then rank it:

```bash
python -m examples.predict_active_pool \
  --checkpoint output/advanced/seed_13_dinov2_gated_ordinal/best_model.pth \
  --pool data/unlabeled_pool.csv \
  --images-dir data/images \
  --output output/advanced/unlabeled_pool/predictions.csv

python -m examples.rank_active_learning \
  --predictions output/advanced/unlabeled_pool/predictions.csv \
  --output output/advanced/unlabeled_pool/labeling_priority.csv
```

Prioritize high-scoring buildings while monitoring class, neighborhood, source,
and façade-quality diversity. After annotation, regenerate all split manifests;
never insert newly selected buildings directly into an existing test set.

## Reproducibility notes

- Commit each manifest and its JSON metadata or archive them with a checksum.
- Record the exact branch commit, GPU type, CUDA/PyTorch versions, and model-cache
  revision.
- Keep image processing, tabular preprocessing, loss weights, and probability
  blend fixed before evaluating the test set.
- Report nominal and ordinal metrics together. Accuracy alone is dominated by
  the most frequent construction periods.
