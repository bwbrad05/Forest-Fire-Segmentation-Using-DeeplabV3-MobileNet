# Segmentasi Area Kebakaran Hutan dan Lahan pada Citra Satelit  
## Menggunakan DeepLabV3+ dengan Backbone MobileViT untuk Wilayah Tropis Indonesia

**Bradley Widjaja — 5025231036**  
Final Year Research (Tugas Akhir)

---

## Research Overview

This repository implements burned-area segmentation for Indonesian tropical
forests and peatlands using **DeepLabV3+ with a MobileViT backbone**.

The design prioritises:
- Computational efficiency suitable for limited resources
- Robustness on small satellite image datasets (227 images)
- Handling of Indonesian tropical conditions (cloud cover, peatland burn signatures)
- Landsat-8 Surface Reflectance imagery

---

## Architecture

```
Input Satellite Image (B, C, H, W)
        │
        ▼
┌─────────────────────────────────┐
│        MobileViT Encoder        │  ← Hybrid CNN + Transformer
│  stride-4  │ stride-8           │    (MobileViT-XXS or XS variant)
│  stride-16 │ stride-32          │
└──────────────────┬──────────────┘
                   │ multi-scale feature maps
        ┌──────────▼──────────┐
        │    ASPP Module      │  ← Atrous Spatial Pyramid Pooling
        │  rates: 6, 12, 18   │    captures multi-scale context
        └──────────┬──────────┘
        ┌──────────▼──────────┐
        │ DeepLabV3+ Decoder  │  ← Skip connection from stride-4 features
        │  (skip connection)  │    refines boundary details
        └──────────┬──────────┘
        ┌──────────▼──────────┐
        │  Segmentation Head  │  ← 1×1 conv + bilinear ×4 upsample
        │  Burned / Non-burned│
        └─────────────────────┘
              Output Mask (B, 2, H, W)
```

---

## Ablation Study

Three backbone variants are compared to validate MobileViT's advantage:

| Experiment | Backbone         | Config File                           |
|------------|-----------------|---------------------------------------|
| A          | ResNet-50       | `configs/model/deeplabv3plus_resnet50.yaml` |
| B          | MobileNetV3     | `configs/model/deeplabv3plus_mobilenetv3.yaml` |
| **C** ✓   | **MobileViT-XXS** | `configs/model/deeplabv3plus_mobilevit_xxs.yaml` |
| C (larger) | MobileViT-XS   | `configs/model/deeplabv3plus_mobilevit_xs.yaml` |

---

## Datasets

### Primary — Indonesian Burned Area Dataset
- **Sensor**: Landsat-8 Surface Reflectance
- **Size**: 227 images, 512×512 pixels
- **Bands**: B2 B3 B4 B5 (NIR) B6 (SWIR1) B7 (SWIR2) → 6 channels
- **Labels**: Binary mask — burned (1) / non-burned (0)
- **Split**: 5-fold cross-validation
- **Location**: `data/indonesia/`

---

## Dataset Directory Layout

```
data/
  indonesia/
    images/
      <id>.tif              ← multi-band GeoTIFF (6 bands)
    masks/
      <id>_mask.tif         ← binary burned-area mask
    cloud_masks/            ← optional; same naming as images
      <id>.tif
    splits.parquet           ← file IDs with fold assignments (0–4)
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Training

```bash
# Primary model — Experiment C (MobileViT-XXS, Landsat-8)
python main.py mode=train model=deeplabv3plus_mobilevit_xxs dataset=indonesia

# Ablation A — ResNet-50 baseline
python main.py mode=train model=deeplabv3plus_resnet50 dataset=indonesia

# Ablation B — MobileNetV3 baseline
python main.py mode=train model=deeplabv3plus_mobilenetv3 dataset=indonesia

# 5-fold cross-validation
python main.py mode=crossval model=deeplabv3plus_mobilevit_xxs dataset=indonesia
```

---

## Evaluation

```bash
# Full evaluation (all metrics + efficiency)
python evaluate.py ckpt_path=<checkpoint.ckpt> model=deeplabv3plus_mobilevit_xxs

# Efficiency metrics only (no dataset needed)
python evaluate.py mode=efficiency model=deeplabv3plus_mobilevit_xxs
```

### Metrics Reported
| Metric           | Description                              |
|-----------------|------------------------------------------|
| IoU              | Intersection over Union (burned class)  |
| Dice Score       | 2·TP / (2·TP + FP + FN)                |
| Precision        | TP / (TP + FP)                          |
| Recall           | TP / (TP + FN)                          |
| Accuracy         | (TP + TN) / total pixels                |
| F1 Score         | Harmonic mean of Precision and Recall   |
| Parameters (M)   | Trainable model parameters              |
| Model size (MB)  | Weight storage in float32               |
| Inference (ms)   | Per-image wall-clock time               |

---

## Visualization

Render qualitative result panels (input imagery, ground truth, prediction,
overlays, and burned-probability heatmap) to inspect results visually.

```bash
# All test-split samples from a trained checkpoint (Experiment C, MobileViT-XXS)
python visualize.py ckpt_path=checkpoints/last.ckpt num_samples=null

# A handful of samples (good for slides)
python visualize.py ckpt_path=checkpoints/last.ckpt split=test num_samples=8

# An ablation backbone, into a separate folder
python visualize.py ckpt_path=checkpoints/resnet50.ckpt model=deeplabv3plus_resnet50 out_dir=outputs/viz_resnet
```

> **Windows note:** if the virtualenv is not activated, call its Python directly:
> `.venv/Scripts/python.exe visualize.py ckpt_path=checkpoints/last.ckpt`

`ckpt_path` is **required** — running `python visualize.py` alone raises an error.

### Options (Hydra overrides)
| Key           | Default                  | Description                                   |
|---------------|--------------------------|-----------------------------------------------|
| `ckpt_path`   | `null` (**required**)    | Path to a trained `.ckpt` checkpoint          |
| `split`       | `test`                   | Which split to render: `train` / `val` / `test` |
| `num_samples` | `8`                      | Samples to render (`null` = all)              |
| `out_dir`     | `outputs/visualizations` | Where PNG panels are saved                     |
| `model`       | `deeplabv3plus_mobilevit_xxs` | Backbone config (must match the checkpoint) |
| `dpi`         | `130`                    | Figure resolution                              |

Each sample is saved as `<out_dir>/<split>_<sample_id>.png` with seven panels:
true-colour RGB (B4/B3/B2), burn-highlight false colour (B7/B5/B4), ground truth,
prediction, ground-truth overlay, prediction overlay, and burned-class probability.

> Inputs are z-score normalised, so display bands use a per-band 2–98 %
> percentile stretch (a plain `×255` would render noise).

---

## Loss Functions

Two imbalance-aware losses are supported (burned pixels < 5% of total):

**Asymmetric Unified Focal Loss** (default — `loss_fn: asymmetric_unified_focal`)  
Combines Asymmetric Focal Tversky Loss + Asymmetric Focal Loss.
Weights foreground (burned) pixels more heavily than background.

**BCE + Dice Loss** (`loss_fn: bce_dice`)  
Pixel-level cross-entropy + region-level Dice overlap.
`pos_weight` controls the burned-class weight in BCE.

---

## Preprocessing Pipeline

1. **Band normalisation** — Z-score using pre-computed statistics
   (Landsat-8 statistics are hard-coded in `indonesia_datamodule.py`)
2. **Cloud masking** — pixels flagged as cloud/shadow are zero-filled;
   patches with > 30% cloud coverage are skipped
3. **Spatial resizing** — images are resized to 512×512 if necessary
4. **Augmentation** (training only):
   - Random horizontal / vertical flip
   - Random 90° rotation (k ∈ {1, 2, 3})
   - Mild per-band brightness shift (±10%)

---

## Project Structure

```
.
├── main.py                         ← Training entry point
├── evaluate.py                     ← Evaluation entry point
├── visualize.py                    ← Qualitative visualization entry point
├── requirements.txt
├── configs/
│   ├── train.yaml                  ← Default training config
│   ├── evaluate.yaml               ← Evaluation config
│   ├── visualize.yaml              ← Visualization config
│   ├── model/
│   │   ├── deeplabv3plus_mobilevit_xxs.yaml   ← Experiment C (primary)
│   │   ├── deeplabv3plus_mobilevit_xs.yaml    ← Experiment C (larger)
│   │   ├── deeplabv3plus_resnet50.yaml        ← Experiment A (ablation)
│   │   ├── deeplabv3plus_mobilenetv3.yaml     ← Experiment B (ablation)
│   │   ├── deeplabv3plus.yaml                 ← SMP default baseline
│   │   └── deeplabv3plus_mobile.yaml
│   ├── dataset/
│   │   └── indonesia.yaml          ← Landsat-8 primary dataset
│   ├── trainer/
│   │   ├── default_gpu.yaml
│   │   └── default_cpu.yaml
│   └── logger/
│       └── comet.yaml
├── neural_net/
│   ├── __init__.py
│   ├── mobilevit_backbone.py       ← MobileViT encoder (timm wrapper)
│   ├── attention.py                ← CBAM + Coordinate Attention, encoder/decoder wrappers
│   ├── strip_pooling.py            ← Strip-pooling ASPP branch (5 → 6 branches)
│   ├── enhancements.py             ← Applies the optional modules to a built model
│   ├── deeplabv3plus_mobilevit.py  ← Primary Lightning module
│   └── deeplabv3plus.py            ← Ablation baseline Lightning module
├── lightning_modules/
│   ├── __init__.py
│   └── indonesia_datamodule.py     ← Landsat-8 data pipeline
├── loss/
│   ├── __init__.py
│   ├── unified_focal_loss.py       ← Asymmetric Unified Focal Loss
│   └── bce_dice_loss.py            ← BCE + Dice Loss
├── utils/
│   ├── __init__.py
│   ├── conversions.py              ← Visualisation helpers
│   ├── schedulers.py               ← poly / warm-up + cosine LR schedules
│   ├── tta.py                      ← 8-way test-time augmentation
│   └── efficiency.py               ← Parameter count / inference timing
└── dict_transforms/
    └── dict_transforms.py          ← Spatial augmentation transforms
```

---

## Configuration Override Examples

```bash
# Override batch size and test fold via CLI
python main.py dataset.batch_size=4 dataset.test_fold=2

# Use BCE+Dice loss instead of focal (= weighted CE + Dice)
python main.py model.loss_fn=bce_dice model.bce_weight=0.6

# Enhanced DeepLabV3+ modules (see CHANGES_SINCE_MEETING.md Part F)
python main.py model.strip_pooling=true model.attention=ca

# CPU debug run
python main.py trainer=default_cpu dataset.test_fold=0
```

