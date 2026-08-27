# Changes Log — Response to Advisor Meeting (`p.txt`)

This documents the code changes made **in response to the advisor notes in `p.txt`**,
plus the smaller data-pipeline updates that were made **before** that meeting for context.

---

## Part A — Fixes driven by the `p.txt` advisor notes

`p.txt` listed six points. Here is what each produced.

### 1. Learning rate
- **ResNet-50 ablation LR corrected: `0.0001` → `0.01`** (paper §IV-E uses `0.01` for the
  ResNet family, `0.0001` for the mobile family). The MobileViT / MobileNetV3 configs were
  already at the correct `0.0001` and were left unchanged.
- File: `configs/model/deeplabv3plus_resnet50.yaml`

### 2. Parameter count / overfit check
- **Corrected the parameter-count comments** in all four model configs — they were badly
  wrong (XXS said "~5.7M", real value **1.45M**; XS said "~10.9M", real **2.13M**).
- **Diagnosed overfit vs. underfit** from the existing `version_6` run: the model is
  **underfitting** (train IoU below val IoU, val loss never rises, curves still climbing at
  epoch 55). Conclusion: scale capacity / training, do **not** add regularization.
- Files: all four `configs/model/*.yaml`

### 3. Check the network
- **Fixed a crash that blocked the ablation models from running at all:** the code called
  `tm.Dice`, removed in the installed torchmetrics version, so ResNet-50 and MobileNetV3
  could not even instantiate. (Dice = F1 for binary; `test_f1` is already reported.)
- **Fixed a channel-detection bug** in efficiency mode (ablations were profiled at 6
  channels instead of 8).
- Files: `neural_net/deeplabv3plus.py`, `main.py`

### 4. GFLOPs
- **Added GFLOPs and GMACs** to the efficiency report (PyTorch's built-in FLOP counter — no
  new dependency).
- **Added `mode=efficiency_sweep`** — one command profiles all four models and writes a
  params / GFLOPs / GMACs / latency comparison table (`lightning_logs/efficiency_report.csv`).
- Files: `utils/efficiency.py`, `main.py`

### 5 & 6. "Balance vs. the big model" / "complex images need a complex network"
- **No code change — these require GPU experiments, not edits.** They are addressed by the
  underfitting finding (Part A.2): the path is more capacity (MobileViT-XS) + more epochs +
  a tuned LR. See "Next experiments" below.

---

## Part B — Additional correctness fixes made in the same pass

### Spatial-leakage fix (scene-aware split)
- Tiles from one Landsat acquisition (`L8_<pathrow>_<date>_*`) were scattered across
  train/val/test, inflating IoU. Fold assignment is now **scene-aware** — whole scenes stay
  in one fold (verified **0 leaked scenes** vs. 36 before).
- Also fixed the `id` vs `files` column-name bug, and made a malformed `splits.parquet` a
  hard error instead of silently reverting to the leaky split.
- Two protocols supported: `--mode scene` (honest, default) and `--mode tile`
  (paper-comparable).
- Files: new `lightning_modules/scene_splits.py`, `lightning_modules/indonesia_datamodule.py`,
  `scripts/sanity_and_splits.py`
- **Implication:** the old IoU 0.6436 was inflated; report the scene-aware number as the
  honest result.

### Test-time augmentation (TTA)
- Optional 8-way (D4) TTA at inference (`model.tta=true`) for a small, retraining-free
  IoU/F1 gain.
- Files: new `utils/tta.py`, both model classes, model configs.

### `lr_find` made usable
- `mode=lr_find` now **saves** the suggested LR + the full loss-vs-LR sweep (CSV) + a plot,
  instead of only printing a number.
- File: `main.py`

### Visualization band indices fixed for the 8-band stack (2026-08-14)

`visualize.py` still used the old 6-band indices, so panels rendered **B3/B2/B1** as
"true colour" and **B6/B4/B3** as "burn-highlight" — the burn composite had no SWIR2 in it
at all. Display bands are now chosen from the channel count (8-band → B4/B3/B2 and
B7/B5/B4; the 6-band mapping is kept for legacy checkpoints).
File: `visualize.py`

### Training-curve plotting script (2026-08-14)

TensorBoard is not installed, so Lightning falls back to the CSV logger and there was no way
to see loss/IoU curves. `scripts/plot_curves.py` renders them from `metrics.csv` (single run
or all crossval folds overlaid) and prints the best epoch plus the train-vs-val IoU gap.
Re-run on `version_6` it reproduces the Part A.2 finding directly: best val_loss 0.1401 @
epoch 51, val_iou 0.6174 vs. train_iou 0.5673 — gap −0.0501, **underfitting**.
File: new `scripts/plot_curves.py`

### GeoTIFF reading no longer depends on GDAL (2026-08-14)

The GPU machine could not read a single tile: pip's `rasterio` wheel bundles its own GDAL,
which corrupted the heap next to conda's copy and killed the process with no traceback
(exit `0xC0000374`), while the conda-forge build failed with `DLL load failed while
importing _base`.

Nothing downstream uses geospatial metadata — only the pixel array — so `_read_tif` now
tries rasterio → **tifffile** → xarray. tifffile needs no GDAL and returns bit-identical
arrays (verified against rasterio on this dataset: max abs diff **0.0**). Correctly
installed machines still use rasterio; broken ones fall through with a warning instead of
dying. A shared `_to_chw` helper normalises the layout, since tifffile returns `(H, W, C)`
where rasterio returns `(C, H, W)`.
File: `lightning_modules/indonesia_datamodule.py`

### Augmentation pickling fix (enables fast data loading on Windows)
- The data augmentation was a nested closure and could not be pickled to DataLoader worker
  processes, so `num_workers > 0` crashed on Windows. Rewrote it as a module-level class.
- File: `lightning_modules/indonesia_datamodule.py`
- See Sync status for what is still uncommitted.

---

## Part C — Data-pipeline work done *before* the meeting (not yet presented)

These predate the `p.txt` fixes. They were never shown to the advisor, so they are written
out in full here.

### C.1 `IndonesiaDataModule` rewritten from the original stub

The datamodule was a placeholder. It now implements the full path from GeoTIFF to batch:
per-tile read (rasterio, xarray fallback), optional cloud mask applied to the imagery,
tiles above `cloud_threshold` (30 %) skipped, resize to 512×512 (bilinear for imagery,
nearest for masks), per-band z-score normalisation, and augmentation on the train split
only. Fold logic: `test = fold k`, `val = fold (k+1) % 5`, `train =` the remaining three.

File: `lightning_modules/indonesia_datamodule.py`

### C.2 6 → 8 input bands — **the "B8" change**

Naming note for the presentation: the commit is titled "B8 processing" but it means
**8 bands**, not Landsat band 8. Band 8 (panchromatic, 15 m) is *excluded*; the 8 channels
are **B1 B2 B3 B4 B5 B6 B7 B9**, which is the standard Landsat-8 30 m stack and matches the
paper's Table I.

**The problem.** The band order in the GeoTIFFs was assumed to start at B2. It actually
starts at B1. Two independent checks over all 227 images pinned it down:

- the NIR reflectance peak sits at **index 4**, so the stack begins at B1, not B2;
- **index 7** is spectral, not a QA/cloud mask — ~8k distinct values, no zero pixels, 0.80
  neighbour correlation. Its ~0.0 correlation with every surface band is exactly what B9
  (cirrus) looks like, since water-vapour absorption keeps it from seeing the ground.

**Why it mattered.** The earlier 6-band cut took channels 0–5, i.e. B1–B6 — so it silently
**dropped B7 (SWIR2)**, the single band burned-area detection depends on most. The
Normalized Burn Ratio is NBR = (B5 − B7)/(B5 + B7); without B7 the model had no access to
the burn signal the literature is built on. This is the substantive point to present: it was
not a "use more channels" tweak, it was a missing input band.

**Consequences.** `in_channels: 8` in all four model configs, 8-element normalisation
statistics, and **every pre-24-July checkpoint is now unloadable** (6-channel stem vs.
8-channel stem — a hard shape mismatch, not a warning). `checkpoints/last-v6.ckpt` is the
only 8-band checkpoint on the laptop.

Final band order in the input tensor:

| idx | 0  | 1  | 2  | 3  | 4  | 5    | 6    | 7  |
|-----|----|----|----|----|----|------|------|----|
|band | B1 | B2 | B3 | B4 | B5 | B6   | B7   | B9 |
|     |Coastal|Blue|Green|Red|NIR|SWIR1|SWIR2|Cirrus|

### C.3 Band normalisation recomputed

The previous mean/std were generic Landsat-8 surface-reflectance defaults assuming a DN
range of 0–10000. This dataset's values reach ~18000, so inputs were arriving at
**mean ≈ +12 / std ≈ 8** instead of the intended mean ≈ 0 / std ≈ 1. Statistics are now
computed per band over every pixel of all 227 images.

### C.4 Cross-validation

`mode=crossval` runs all 5 folds sequentially, each with its own model, callbacks and CSV
logger, then writes a mean ± std table to
`lightning_logs/crossval_<model>/summary.csv`.

> The `summary.csv` currently in `lightning_logs/` is from a **1-epoch, 6-band smoke run**
> (fold checkpoints are `epoch=0`). Do not present those numbers — they need a real rerun.

---

## Part D — Results (GPU run, 2026-08-14)

First full training run on the 8-band pipeline, MobileViT-XXS, 100-epoch budget.
Source: `lightning_logs/version_0` (training) and `lightning_logs/version_1/metrics.csv` (test).

| Metric | 8-band run (new) | 6-band run (old, `version_6`) | Δ |
|---|---|---|---|
| **test_iou** | **0.7191** | 0.6436 | **+0.0755** |
| **test_f1** | **0.8366** | 0.7832 | **+0.0534** |
| test_precision | 0.8096 | 0.7796 | +0.0300 |
| test_recall | 0.8655 | 0.7868 | +0.0787 |
| test_accuracy | 0.9597 | 0.9557 | +0.0040 |
| inference (ms/image) | 65.9 (GPU) | 239.3 (laptop CPU) | not comparable |

### How to present this

The headline is not just "IoU went up". The old 0.6436 came from a **6-band, spatially
leaky** setup — inflated *and* missing SWIR2. The new 0.7191 is from an **8-band,
scene-aware** setup, which is the harder and more honest protocol. Restoring B7 more than
paid for removing the leakage: the number went up while the evaluation got stricter.

Recall (0.8655) exceeding precision (0.8096) means the model over-predicts burned area
slightly — it finds most of the burn scars and adds some false positives. For a fire-mapping
application that is the preferable direction to err.

### Caveats to state honestly

- **Single fold.** This is `test_fold=0` only. A single fold is not defensible for the
  thesis — fold-to-fold std can reach several IoU points. `mode=crossval` is still required
  for the reported number.
- **Still underfitting.** `save_top_k=3` kept epochs **73, 82 and 88**, so val_loss was
  still improving near the end of the 100-epoch budget. More epochs and/or a tuned LR should
  push this further. This matches the Part A.2 diagnosis.
- **The inference times are on different machines** (GPU vs. laptop CPU). Do not put them in
  the same comparison — use `mode=efficiency_sweep` on one machine for the latency table.
- ~~**Confirm the split protocol** before quoting the number as scene-aware.~~
  **Resolved 2026-08-28.** Checked directly on the GPU box: folds of 46/46/45/45/45 tiles,
  81 scenes, **0 scenes crossing folds** — identical to the laptop. The run used the
  scene-aware split, so **0.7191 is the honest figure** and can be quoted as such.

---

## Part E — Accuracy improvements from the literature (2026-08-21)

Two changes aimed at F1/IoU, both grounded in the burned-area literature. They are
independent, so they can be ablated separately for the thesis.

### E.1 ImageNet transfer learning was silently disabled

**This is the big one, and it was a bug rather than a design choice.** Three separate places
switched pretraining off whenever the input was not 3-channel — which, at 8 bands, is
always:

| File | Code | Effect |
|---|---|---|
| `neural_net/mobilevit_backbone.py` | `pretrained=(pretrained and in_channels == 3)` | never loads weights |
| `neural_net/mobilevit_backbone.py` | `_adapt_first_conv` → `kaiming_normal_` | re-randomises the stem |
| `neural_net/deeplabv3plus.py` | `encoder_weights if n_channels == 3 else None` | same, for the ablations |

So every run so far — including the IoU 0.7191 in Part D — trained a Vision Transformer
**from scratch on 227 tiles**. Transformers are markedly more data-hungry than CNNs, which
is a strong candidate explanation for the persistent underfitting in Part A.2.

The premise behind those guards is wrong: **both timm and SMP adapt a pretrained first
convolution to any band count**, tiling/rescaling the RGB filters rather than discarding
them. Verified here — with `in_chans=8`, deep-layer weights differ from random init
(std 0.190 vs 0.059) and the stem's per-channel means repeat with period 3
(`0.0059, -0.0101, -0.0021, 0.0059, …`), i.e. the RGB filters tiled across 8 bands.

Fix: pass `in_chans` to timm, drop the stem surgery, remove the 3-channel guards. All four
configs now default to `encoder_weights: imagenet`; set `model.encoder_weights=null` to
ablate against from-scratch.

### E.2 Spectral indices as extra input channels

The literature is consistent that NBR is the decisive variable for burned-area mapping, and
that appending indices to the band stack beats bands alone. `dataset.spectral_indices=true`
appends three channels computed from the **raw** bands before z-scoring (a normalised
difference of z-scores is meaningless):

- **NBR** = (B5 − B7)/(B5 + B7) — the standard burned-area index
- **NDVI** = (B5 − B4)/(B5 + B4) — vegetation greenness, hence its absence over scars
- **NBR2** = (B6 − B7)/(B6 + B7) — post-fire moisture/char, complementary to NBR

Measured separability on all 227 tiles of this dataset (mean index value inside vs. outside
the burn mask):

| index | burned | unburned | gap |
|---|---|---|---|
| **NBR** | 0.1641 | 0.5196 | **−0.3556** |
| NDVI | 0.2855 | 0.4184 | −0.1329 |
| NBR2 | 0.1705 | 0.3223 | −0.1517 |

NBR separates by more than twice either alternative — this dataset agrees with the papers.
Enabling the flag requires `model.in_channels=11`.

### E.3 CBAM attention module — the closest match in the literature

[Zhang et al., *Scientific Reports* 2024](https://www.nature.com/articles/s41598-024-66060-7)
build a lightweight DeepLabV3+ for **burned-area identification** on 512×512 tiles, Adam at
lr 1e-4, 100 epochs — the same task, the same tile size, the same optimiser settings as this
thesis. Their **Table 4, "Accuracy evaluation of ablation experiments"**:

| Row (their naming) | MIoU (%) | Δ |
|---|---|---|
| `Deeplab v3+` | 73.54 | — |
| `LDeeplab v3+` (lightweight backbone) | 72.97 | −0.57 |
| **`CB-Deeplab v3+`** (backbone + CBAM) | **80.12** | **+7.15** |
| `Ours` (+ deep transitive transfer learning) | 83.62 | +3.50 |

**Two caveats to state accurately, because an examiner will check them.**

1. *The paper never attributes the +7.15 to CBAM by name.* The supporting sentence reads:
   "When the improved MobileNet V2 network is used as the backbone network, MIoU, OA and
   Kappa are improved, increasing by 7.15%, 2.71% and 0.0941 respectively." Their "improved
   MobileNet V2" is the CBAM-embedded backbone, so the attribution is sound — but it comes
   from differencing table rows, not from prose. Cite it as *the CBAM-augmented backbone*.

2. *Their metric is MIoU; ours is foreground IoU.* `test_iou` here is
   `JaccardIndex(task="binary")` — the burned class only. MIoU averages burned and
   background, and background IoU in this task typically sits near 0.95. That puts their
   baseline foreground IoU around **0.51**, against **0.7191** here (Part D). Their headroom
   was far larger, and a +7.15 MIoU delta would correspond to roughly +14 points of
   foreground IoU. **Do not present 6–7 % as the expected gain for this thesis.** Gains
   compress as the baseline rises.


**Implementation** — `model.attention=cbam`. New file `neural_net/attention.py` provides
`ChannelAttention`, `SpatialAttention`, `CBAM` and a `CBAMEncoder` wrapper.

Placement deliberately differs from the paper. They embed CBAM inside the MobileNetV2
bottlenecks; the equivalent inside MobileViT would mean editing timm's blocks and inserting
randomly-initialised layers into the middle of an ImageNet-pretrained encoder — which would
fight E.1. Instead CBAM refines the two feature maps the DeepLabV3+ decoder actually
consumes: the stride-16 map entering the ASPP and the stride-4 low-level skip. The
pretrained encoder is untouched, and the refinement still occurs "before the features
proceed to the ASPP module and decoder" as the paper describes. The wrapper follows the SMP
encoder interface, so it applies identically to MobileViT, ResNet-50 and MobileNetV3 and
keeps the ablation fair.

**Cost.** MobileViT-XXS goes from **1.449 M → 1.450 M parameters** — a 0.07 % increase. For a
thesis whose selling point is edge-deployability, an accuracy module that is essentially free
is a strong result in its own right. (ResNet-50: 26.69 M → 27.23 M, since CBAM scales with
channel count.)

Verified here: all four backbones build with and without CBAM, output shapes unchanged, and
all six CBAM parameter tensors receive non-zero gradients.

### References

- [Assessment of vegetation indices for mapping burned areas using a deep learning method and
  a comprehensive forest fire dataset from Landsat collection](https://www.sciencedirect.com/science/article/pii/S0273117724012134)
  — closest match to this setup (Landsat, U-Net, trained with and without NDVI/NBR).
- [Improving wildfire burned area mapping from Sentinel-2 imagery using a context-aware
  transformer UNet (FTSUNet)](https://www.sciencedirect.com/science/article/pii/S2590123026027933)
  — NBR/BAI/BAIS2 as inputs; Dice 95.50 %, IoU 91.49 %.
- [Burned area detection using Sentinel-1 and Sentinel-2 features and analysis of ensemble
  models with explainable AI](https://link.springer.com/article/10.1007/s10651-025-00693-3)
  — NBR, NBRSWIR and the SWIR/NIR bands rank as the most decisive variables.
- [FLOGA: a machine-learning-ready dataset, benchmark and model for burnt area mapping](https://arxiv.org/pdf/2311.03339)
  — benchmark protocol worth comparing against.
- [Semantic Segmentation of Remote Sensing Images Using Transfer Learning and Deep CNN with
  Dense Connection](https://www.researchgate.net/publication/342368106_Semantic_Segmentation_of_Remote_Sensing_Images_Using_Transfer_Learning_and_Deep_Convolutional_Neural_Network_With_Dense_Connection)
  — transfer learning specifically for insufficient labelled data and class imbalance.
- [A lightweight DeepLab V3+ network integrating deep transitive transfer learning and
  attention mechanism for burned area identification](https://www.nature.com/articles/s41598-024-66060-7)
  — **the primary reference for E.3.** Same task, tile size and optimiser; CBAM contributes
  +7.15 MIoU in their ablation.
- [An improved semantic segmentation algorithm for high-resolution remote sensing images
  based on DeepLabv3+](https://www.nature.com/articles/s41598-024-60375-1)
  — attention-augmented DeepLabV3+; MIoU 64.26 → 69.61 on ISPRS.
- [Self-Supervised Encoders Are Better Transfer Learners in Remote Sensing Applications](https://www.mdpi.com/2072-4292/14/21/5500)
  — if ImageNet transfer helps, SSL-pretrained remote-sensing encoders are the next step.

---

## Part F — Enhanced DeepLabV3+ modules from the wheat-grain paper (2026-08-28)

Source: Lyu, Liu, Sun, Guan & Lyu, **"Enhanced DeepLabV3+ for wheat grain
segmentation: Addressing tiny targets, blurred edges, and class imbalance"**,
*Journal of Agriculture and Food Research* **26** (2026) 102680.
https://doi.org/10.1016/j.jafr.2026.102680 (open access, CC BY-NC-ND)

Brought in by the advisor. Two of its four contributions are genuinely new to
this project and are now implemented; the other two were already here in
different form.

### F.1 Where the "+6 %" actually comes from

The number is real and it is traceable, but it must be quoted precisely, because
an examiner can open the paper and check.

It is **not** the headline 86.93 % vs 77.58 %. That +9.35 is the *whole* enhanced
model — new backbone, strip pooling, coordinate attention, the joint loss, and
the regularization/scheduling package all together. The +6.28 comes from the
sentence in their conclusion (section 5.1):

> "Multi-scale feature enhancement, by embedding SP and integrating CA modules,
> improves the mIoU by 6.28 % compared with the benchmark model"

and it differences two rows of their ablation Table 4:

| Their row | Modules | mIoU % |
|---|---|---|
| Models_1 | none (plain DeepLabV3+) | 77.58 |
| **Models_7** | **SP + CA** | **83.86** |

83.86 − 77.58 = **6.28**. So the +6 % is **strip pooling plus coordinate
attention**, measured *without* their MobileNetV2 backbone swap and *without*
the loss change. That is exactly the pair implemented below.

**Two things to state honestly when presenting this.**

1. **Their metric is 4-class mIoU; ours is foreground IoU.** Their Table 2 lets
   us convert. Their baseline DeepLabV3+ scores 72.89 / 71.31 / 68.68 on the
   three wheat grades and 97.44 on background — mean 77.58 ✓. Dropping
   background gives a **foreground IoU of 70.96** for their baseline, against
   **71.91** for this project's Part D run. Their full model reaches a
   foreground IoU of 82.77.

   This matters: unlike the CBAM paper in Part E.3, whose baseline sat around
   0.51 foreground IoU and therefore had far more headroom than we do, this
   paper's baseline starts from *almost exactly where we are*. The headroom
   objection is much weaker here, which makes the advisor's suggestion a
   reasonable one on its own terms.

2. **The tasks still are not equivalent.** Their three foreground classes are
   wheat grains at 790, 805 and 810 g/L bulk density — visually near-identical,
   so most of their foreground error is *class confusion between lookalikes*.
   Our error is binary and mostly *boundary and extent*. Similar IoU numbers do
   not imply the same error structure, so do not promise +6 points. Present it
   as "the mechanism is plausible for our target geometry, and the ablation will
   tell us".

### F.2 What the paper proposes, and what was actually new here

| Their contribution | Status in this project |
|---|---|
| Lightweight backbone (MobileNetV2 replacing Xception) | **Already done, and further.** MobileViT-XXS is 1.45 M params against their MobileNetV2's 5.81 M. Ablations B/C already cover the lightweight-backbone question. |
| Dynamic downsampling (per-stage dilation/stride adjustment) | **Not applicable.** This is how MobileNetV2 is made to hit output stride 8 or 16 — SMP and timm already do it, and our MobileViT encoder bottoms out at stride 16 by construction. No gain available. |
| **Strip Pooling (SP) in the ASPP** | **NEW — implemented.** |
| **Coordinate Attention (CA) at the backbone output** | **NEW — implemented.** |
| CA-weighted fusion in the decoder | **NEW — implemented** (with a stated deviation, see F.4). |
| WCE + Dice joint loss with weight λ | **Mostly present.** `BCEDiceLoss` was already exactly this; λ and the class weight are now exposed and the class weight is now *measured* rather than guessed (F.5). |
| Warm-up + cosine annealing LR | **NEW — implemented** as an option; we were on PolynomialLR. |
| Dropout 0.5, weight normalization, ×10 augmentation | **Deliberately not adopted** — see F.7. |

### F.3 Strip Pooling — what it does and why it should transfer

`neural_net/strip_pooling.py`. Enable with `model.strip_pooling=true`.

The ASPP gathers context with **square** dilated kernels. Any square window wide
enough to span a long thin structure necessarily also swallows a large amount of
unrelated area. Strip pooling (Hou et al., CVPR 2020) instead averages a whole
row and a whole column independently, producing an **anisotropic** receptive
field that follows elongated structure at almost no cost:

```
x -> avgpool to (H,1) -> 1-D conv along H -.
x -> avgpool to (1,W) -> 1-D conv along W -+-> sum -> ReLU -> 1x1 -> sigmoid -> * x
```

It is added as a **sixth ASPP branch**, so the fusion convolution widens from 5×
to 6× branch channels — the paper's Fig. 5, reproduced exactly.

**The argument for our task.** The paper wanted this for elongated wheat grains.
Burn scars are not elongated *objects*, but their **boundaries and extents are**:
fire fronts run along ridge lines, stop at rivers and roads, and follow
plantation-block edges, all of which are long, thin, roughly axis-aligned
structures spanning a 512×512 Landsat tile. A row-wide and column-wide average
is a cheap prior for exactly that geometry. This is a plausibility argument, not
a result — the ablation in F.8 is what decides it.

### F.4 Coordinate Attention — and why it is paired with SP

`neural_net/attention.py`, class `CoordinateAttention`. Enable with
`model.attention=ca`.

CA (Hou et al., CVPR 2021) factorises attention into two **full-length 1-D
profiles**, one per image row and one per image column, so the weight at pixel
(i, j) is informed by the whole of row i and the whole of column j:

```
Z_h = mean over W  ->  (B,C,H,1)          direction-sensitive pooling   [their eq. 4]
Z_v = mean over H  ->  (B,C,1,W)                                        [their eq. 5]
A   = sigmoid(f_1x1(concat(Z_h, Z_v)))    joint channel-spatial weights [their eq. 6]
Y   = X * A_h * A_v                       dynamic feature weighting     [their eq. 7]
```

This is **not** interchangeable with the CBAM already added in Part E.3. CBAM's
spatial branch collapses the channel axis and convolves a 7×7 window, so its
"where" is *local*. CA's is *global along each axis* — the same anisotropic
geometry as strip pooling.

That is precisely why the paper proposes them as a pair, and their section 2.3
states the mechanism plainly: SP's aggressive long-range aggregation drags
background into the features, and CA is the filter that suppresses it again.
Their Table 4 supports this — SP alone on top of the lightweight backbone
(Models_4, 84.07) is actually *worse* than the backbone alone (Models_2, 84.89);
only once CA is added does the combination beat both (Models_8, 85.92).

**Practical consequence for our ablation: do not run `strip_pooling=true` on its
own and conclude SP is useless.** The paper's own numbers predict that result.
SP+CA is the unit being claimed.

**Three deviations, all deliberate, all stated so they can be defended:**

1. **Placement is at the encoder output, not inside the backbone.** Same choice
   and same reason as CBAM in Part E.3: embedding attention inside the blocks
   would inject randomly-initialised layers into the middle of an
   ImageNet-pretrained encoder and undo the transfer learning restored in
   Part E.1. CA refines the two maps the decoder actually consumes — the
   stride-16 map entering the ASPP and the stride-4 low-level skip. The paper's
   own Fig. 3 places CA at the backbone output too, so this is closer to their
   design than the CBAM case was to Zhang et al.'s.

2. **The CA hidden width is floored at 8 channels.** The paper's compression
   ratio r = 32 assumes a wide backbone. MobileViT-XXS carries 64 channels at
   stride 16 and 16 at stride 4, so a literal `C // 32` would give 2 and **0**
   channels. The reference CA implementation's `max(8, C // r)` is used instead.

3. **The decoder's "dynamic weight allocation" uses a second CA instance.** The
   paper's equations 19–20 reuse the *same* weight map `A` from the backbone-output
   CA to scale the fused features (`F_out = F_cat ⊙ A`). That is not literally
   reproducible in this decoder: `A` is generated at stride 16 with the
   encoder's channel count, while `F_cat` is at stride 4 with 256 channels, so
   it neither broadcasts nor upsamples meaningfully. A second CA on the fused
   map delivers the stated intent without inventing a reshape the paper never
   describes. Separate flag: `model.decoder_attention=ca`.

**Deviation in the strip-pooling branch, for cost reasons.** Two changes, both
measured:

- *The branch projects to 256 channels first, then strip-pools.* Gating at the
  raw encoder width and projecting afterwards sizes the strip convolutions by
  the backbone: **+30 M parameters on ResNet-50** (2048 channels at stride 16)
  against +0.05 M on MobileViT-XXS. That would both wreck the edge-deployability
  claim and make the ablation unfair, since "+SP" would mean a different amount
  of added capacity for each backbone. Projecting first pins the gate at 256
  channels for every backbone.
- *The two 1-D convolutions are depthwise*, with the 1×1 fusion convolution
  doing the channel mixing — the same depthwise-separable factorisation the
  paper applies to its own backbone (their section 2.2.1). This takes the gate
  from 458 K to 67 K parameters. `depthwise=False` restores Hou et al.'s
  original full convolutions.

### F.5 Multi-loss — already present, now correctly parameterised

The paper's L_total = λ·L_WCE + (1−λ)·L_Dice (their equations 10–12) **is** this
project's existing `BCEDiceLoss`: `bce_weight` is λ and `pos_weight` is their
per-class weight ω_c. Two things were missing.

- **λ was hard-coded to 0.5.** Now `model.bce_weight`. Their grid search over
  0.1–0.9 (their Table 5) peaks at **0.6**.
- **ω_c was a guess.** The configs carried `pos_weight = 10.0` with a comment
  saying "~10–20 is reasonable". Their equation (10) defines it as
  N_total / N_c accumulated over the **entire training set**, explicitly not
  per mini-batch. `scripts/class_weight.py` now computes it:

  ```
  test_fold=0  train folds=[2,3,4]  tiles=135
    burned pixels     : 1,846,275 (5.217 %)
    background pixels : 33,543,165
    --> bce_pos_weight = 18.168
  ```

  So the true ratio is **18.17, not 10.0** — the burned class was being
  under-weighted by nearly half. (It counts training folds only; computing it
  over val/test would leak label statistics.) The configs now carry the measured
  value. This is inert for the current runs, which use
  `loss_fn=asymmetric_unified_focal`, and only takes effect under
  `loss_fn=bce_dice`.

### F.6 Warm-up + cosine annealing

`utils/schedulers.py`, `model.scheduler=warmup_cosine`. Implements their
equation (21): quadratic warm-up over the first 10 % of epochs, cosine anneal to
`lr_min`, optional constant tail. Their section 4.5 credits it with removing the
loss/mIoU discontinuity their unscheduled run shows at epoch 50.

Relevance here is independent of that: the Part D run kept its three best
checkpoints at epochs **73, 82 and 88** of a 100-epoch budget, i.e. validation
loss was still improving at the end. Warm-up avoids burning the first epochs on
a randomly-initialised decoder bolted to a pretrained encoder, and the cosine
tail spends the end refining rather than oscillating. `poly` remains the default
so nothing already run changes.

### F.7 What was deliberately **not** adopted

- **Dropout 0.5 on the ASPP fusion** (their section 2.2.5). The Part A.2
  diagnosis for this project is **under**-fitting, not overfitting — adding
  regularization is the wrong direction. Exposed as `model.aspp_dropout` and
  defaulted to `null`, which leaves each path's existing behaviour untouched.
- **Weight normalization on the ASPP fusion conv** (their equation 13). Stacking
  WN on top of the BatchNorm that is already there has no clear justification,
  and it would change every ASPP state-dict key for a claimed benefit the paper
  never isolates in its ablation.
- **Their ×10 augmentation regime.** We already augment; theirs is tuned to a
  150-image lab dataset on a black tray.
- **Instance segmentation / Mask R-CNN** (their future work, section 5.2 item 5).
  Out of scope.

**Also found while wiring this up — an existing inconsistency in the ablation,
not introduced here.** SMP's `DeepLabV3Plus` defaults to `aspp_dropout=0.5`,
`aspp_separable=True` and `atrous_rates=(12,24,36)`, whereas the MobileViT path
in `_build_model` explicitly builds its decoder with `0.0`, `False` and
`(6,12,18)`. So Experiments A/B and Experiment C have **not** been running the
same decoder. Nothing was changed here, because doing so would silently
invalidate Part D; but it is worth equalising before the final ablation table is
reported, and `model.aspp_dropout=0.0` on the A/B configs is the one-line half
of that fix.

### F.8 Cost — measured, not estimated

MobileViT-XXS, 8×512×512 input, `utils.compute_gflops`:

| Config | Params | Δ | GFLOPs | Δ |
|---|---|---|---|---|
| baseline | 1.449 M | — | 6.92 | — |
| + CA | 1.451 M | +0.2 % | 6.92 | +0.0 % |
| + SP | 1.600 M | +10.4 % | 7.23 | +4.4 % |
| **+ SP + CA** | **1.602 M** | **+10.6 %** | **7.23** | **+4.4 %** |
| + SP + CA + decoder-CA | 1.609 M | +11.0 % | 7.23 | +4.4 % |

CA is essentially free (+2.3 K parameters, no measurable FLOPs). Strip pooling
is the whole cost, and 65 K of its 151 K is the mandatory widening of the ASPP
fusion convolution from 5× to 6× branches.

Across backbones, "+SP +CA" costs +10.6 % (MobileViT-XXS), +7.4 %
(MobileViT-XS), +14.6 % (MobileNetV3-Small) and +4.0 % (ResNet-50) in
parameters. At 1.60 M the primary model is still well under the 5.81 M
MobileNetV2 the paper itself calls lightweight.

### F.9 Verification performed

- All four backbones build and run a 512×512 forward pass under all six switch
  combinations; output shape stays (B, 2, 512, 512) in every case.
- Every new parameter tensor receives a non-zero gradient (9 strip-pooling, 14
  encoder-CA, 7 decoder-CA).
- The ASPP really has 6 branches and a 1536-channel fusion input once SP is on.
- CBAM state-dict keys are unchanged by the wrapper refactor
  (`model.encoder.blocks.2.channel.mlp.0.weight` etc.), so Part E.3 checkpoints
  still load.
- `warmup_cosine` factors behave: 0.01 at epoch 0, 1.0 at epoch 9–10 (end of
  warm-up), 0.59 at 50, 0.001 at 99.
- Hydra instantiates all four model configs with and without the new keys.
- End-to-end `fast_dev_run` passes on the MobileViT path with all modules on,
  and on the MobileNetV3 ablation path with `loss_fn=bce_dice bce_weight=0.6`.

**Not verified: whether any of this improves IoU.** That needs GPU runs.

### F.10 Files

| File | Change |
|---|---|
| `neural_net/strip_pooling.py` | **new** — `StripPooling`, `StripPoolingBranch`, `add_strip_pooling`, `set_aspp_dropout` |
| `neural_net/attention.py` | **+** `CoordinateAttention`, `build_attention`, `AttentionEncoder`, `AttentionDecoder`; `CBAMEncoder` becomes a thin subclass (keys preserved) |
| `neural_net/enhancements.py` | **new** — one `apply_enhancements()` used by both Lightning modules so every switch means the same thing on every backbone |
| `utils/schedulers.py` | **new** — `poly` / `warmup_cosine` |
| `scripts/class_weight.py` | **new** — measures the paper's ω_c from the training folds |
| `neural_net/deeplabv3plus_mobilevit.py` | new constructor args; scheduler routed through `utils.build_scheduler`; λ plumbed into the loss |
| `neural_net/deeplabv3plus.py` | same args, so the A/B ablations stay comparable |
| `configs/model/*.yaml` (4 files) | the new switches, documented inline |
| `utils/__init__.py`, `README.md` | exports and file-tree update |

Defaults are unchanged throughout: with no overrides the model is byte-identical
to the Part E configuration, so Part D remains reproducible.

### F.11 Experiments to run (GPU)

The priority from Part D is still **`mode=crossval` on the current best
configuration** — a single fold cannot be the reported number. Do that first;
these modules are a second axis, not a replacement for it.

Then, per the paper's own ablation logic (F.4: SP alone is expected to *not*
help):

| # | Command addition | Tests |
|---|---|---|
| 1 | *(none — current best)* | baseline for this table |
| 2 | `model.attention=ca` | CA alone |
| 3 | `model.strip_pooling=true` | SP alone — expected flat or slightly down |
| 4 | `model.strip_pooling=true model.attention=ca` | **the paper's +6.28 claim** |
| 5 | `+ model.decoder_attention=ca` | their full structural package |
| 6 | `model.attention=cbam` | CBAM vs CA, head to head (Part E.3 vs F.4) |
| 7 | `model.scheduler=warmup_cosine` | orthogonal — combine with the winner |

Run 4 before 2 and 3 if GPU time is short: it is the claim being tested, and
2 and 3 only explain *why* it worked.

### F.12 Command reference

```powershell
# The paper's SP + CA pair — the +6.28 mIoU configuration
python main.py mode=train model=deeplabv3plus_mobilevit_xxs model.strip_pooling=true model.attention=ca

# Full structural package (adds the decoder's dynamic weight allocation)
python main.py mode=train model=deeplabv3plus_mobilevit_xxs model.strip_pooling=true model.attention=ca model.decoder_attention=ca

# Their loss at their tuned lambda, with the measured class weight
python main.py mode=train model.loss_fn=bce_dice model.bce_weight=0.6 model.bce_pos_weight=18.168

# Their LR schedule (equation 21)
python main.py mode=train model.scheduler=warmup_cosine

# Cross-validation with the modules on — the number to report
python main.py mode=crossval model=deeplabv3plus_mobilevit_xxs model.strip_pooling=true model.attention=ca

# Measure the WCE class weight for a given fold (or every fold)
python scripts/class_weight.py --root data/indonesia --test_fold 0
python scripts/class_weight.py --root data/indonesia --all_folds

# Cost table including the new modules
python main.py mode=efficiency_sweep
```
### F.13 Running it on the GPU box (2026-08-28)

Four supporting pieces were added so the sweep is one command and the results
are readable afterwards.

**`scripts/setup_gpu.ps1`** — creates the conda environment in the order that
actually works. The two failure modes this project already hit are both
ordering/mixing problems: installing `requirements-gpu.txt` before torch silently
gets a CPU-only wheel from PyPI (GPU idle, no error), and letting conda supply
scientific packages next to pip's torch aborts the process with `OMP: Error #15`.
So: conda gives Python and nothing else, torch comes first from the CUDA index,
everything else from PyPI. It then verifies CUDA *and* builds an SP+CA model
before declaring success — a multi-hour run should not be the thing that
discovers a broken env.

```powershell
.\scripts\setup_gpu.ps1                  # creates 'firenet-gpu', cu124
.\scripts\setup_gpu.ps1 -Cuda cu121      # different driver
```

**`tensorboard` added to `requirements-gpu.txt`.** Without it Lightning silently
falls back to CSV only, which is why there were no curves to look at before.

**`run_name` (new config key, `main.py`).** Previously every run landed in an
auto-numbered `lightning_logs/version_N` and every checkpoint piled into one
`checkpoints/` folder, colliding into `last.ckpt`, `last-v1.ckpt`, ... — which is
exactly why there are 12 unattributable checkpoints in this repo. With
`run_name=<x>` a run gets `lightning_logs/<x>/` for logs *and* checkpoints, and
both TensorBoard and CSV loggers are attached (TensorBoard to watch live, CSV
because the plotting scripts read `metrics.csv`). Default is unchanged when the
key is unset. The same fix was applied to `mode=crossval`, which had the
identical per-fold collision.

**`scripts/run_ablation.ps1`** — the sweep. One variant per row of F.11, each
into its own `run_name`. `mode=train` already runs the test phase against the
best checkpoint, so every variant yields `test_iou`/`test_f1` from one command.
Finished runs are skipped, so the sweep survives being interrupted; a failed
variant does not abort the rest.

```powershell
.\scripts
un_ablation.ps1                                  # all 7
.\scripts
un_ablation.ps1 -Only abl_baseline,abl_sp_ca     # the claim + reference
.\scripts
un_ablation.ps1 -DryRun                          # print the plan, no GPU needed
```

**`scripts/compare_runs.py`** — reads every run's `metrics.csv`, ranks by
`test_iou`, and reports each variant's delta against the baseline. Writes
`ablation_summary.md` (paste-ready), `ablation_summary.csv` and
`ablation_curves.png` (val IoU and val loss, all runs overlaid) into
`lightning_logs/comparison/`. `--folds` aggregates a cross-validation run as
mean ± std instead.

Verified against the existing logs: it reproduces `version_6`'s recorded
`test_iou 0.6436` and `best val_loss 0.1401 @ epoch 51` exactly, matching Part D.

**Order of work on the GPU box:**

```powershell
conda activate firenet-gpu
python scripts/sanity_and_splits.py --root data/indonesia     # confirm scene-aware split
.\scripts
un_ablation.ps1 -Only abl_baseline,abl_sp_ca      # ~2 runs, the claim
python scripts/compare_runs.py --runs "lightning_logs/abl_*" --baseline abl_baseline
```

`tensorboard --logdir lightning_logs` from a second terminal shows curves while
training runs.

---

## Sync status (laptop ↔ GPU machine)

- **Parts A, B and C are committed** in `ecde31a "feat: add cross validation method and B8
  processing"`, except the four items below.
- **Pending (uncommitted) changes:**
  1. `lightning_modules/indonesia_datamodule.py` — augmentation pickling fix **and** the
     GDAL-free `_read_tif` fallback
  2. `visualize.py` — 8-band display indices
  3. `scripts/plot_curves.py` — new file
  4. `requirements-gpu.txt` — new file
  5. `CHANGES_SINCE_MEETING.md` — this document
  6. Part F (2026-08-28): `neural_net/strip_pooling.py`, `neural_net/enhancements.py`,
     `utils/schedulers.py`, `scripts/class_weight.py`, `scripts/compare_runs.py`,
     `scripts/run_ablation.ps1`, `scripts/setup_gpu.ps1` (all new);
     `neural_net/attention.py`, both model classes, `main.py`, the four model
     configs, `configs/train.yaml`, `requirements-gpu.txt`, `utils/__init__.py`
     and `README.md` (modified)
- To bring the GPU machine up to date: commit these on the laptop and `git pull` on the GPU
  box. Until item 1 is pulled, train with `dataset.num_workers=0` on Windows.
- Note: the GPU box already has item 1 pasted in by hand. `git pull` may report a conflict
  there — take the incoming version (`git checkout --theirs` or just discard the local edit
  with `git checkout -- lightning_modules/indonesia_datamodule.py` before pulling).

---

## Next experiments (require GPU — no more code needed)

1. ~~**`mode=lr_find`** on MobileViT-XXS~~ — not yet run; LR is still the config default 1e-4.
2. ~~**Full training** at 8 bands, ~100 epochs, single-fold check.~~ **Done** — see Part D
   (IoU 0.7191 / F1 0.8366).
3. **`mode=crossval`** → mean ± std for the thesis. **This is now the priority** — Part D is
   a single fold and cannot be the reported result on its own.
4. Optionally repeat with `--mode tile` split for the paper-comparable number, and with
   `model.tta=true`.
5. **Strip pooling + coordinate attention** (Part F) — the advisor's paper. Run
   `model.strip_pooling=true model.attention=ca` first; it is the configuration their
   +6.28 mIoU claim actually refers to. Full ablation grid in Part F.11.

---

## Appendix — command reference

All commands run from the repo root. Hydra configs live in `configs/`; anything in a config
can be overridden as `key=value` on the command line. On Windows, prefix with
`.venv\Scripts\python.exe` instead of `python` if the venv is not activated.

### Data preparation (once)

```powershell
# Verify image/mask pairing and report the fold split
python scripts/sanity_and_splits.py --root data/indonesia

# (Re)create the scene-aware split — honest protocol, no spatial leakage
python scripts/sanity_and_splits.py --root data/indonesia --create --overwrite

# Per-tile split instead — leaky, but matches the reference paper's protocol
python scripts/sanity_and_splits.py --root data/indonesia --create --mode tile --overwrite
```

Current `data/indonesia/splits.parquet` is scene-aware: 227 tiles / 81 scenes,
folds of 46/46/45/45/45 tiles, 0 scenes crossing folds.

### Training

```powershell
# Pipeline smoke test — CPU, one batch, no checkpoints written
python main.py mode=train trainer=default_cpu dataset.num_workers=0 +trainer.fast_dev_run=true

# Learning-rate finder (writes lr_find_<model>.csv + .png to lightning_logs/)
python main.py mode=lr_find model=deeplabv3plus_mobilevit_xxs

# Primary model — Experiment C, MobileViT-XXS
python main.py mode=train model=deeplabv3plus_mobilevit_xxs dataset=indonesia

# Ablations
python main.py mode=train model=deeplabv3plus_resnet50 dataset=indonesia      # Experiment A
python main.py mode=train model=deeplabv3plus_mobilenetv3 dataset=indonesia   # Experiment B
python main.py mode=train model=deeplabv3plus_mobilevit_xs dataset=indonesia  # C variant

# 5-fold cross-validation — the number to report in the thesis
python main.py mode=crossval model=deeplabv3plus_mobilevit_xxs dataset=indonesia
```

Common overrides (chain any of them onto the commands above):

| Override | Effect |
|---|---|
| `model.learning_rate=3e-4` | LR (defaults: 1e-4 mobile family, 1e-2 ResNet-50) |
| `trainer.max_epochs=150` | epoch cap (default 100) |
| `trainer.max_time=null` | remove the 6-hour wall-clock limit |
| `dataset.batch_size=8` | batch size |
| `dataset.num_workers=0` | required on Windows until the pickling fix is pulled |
| `dataset.test_fold=2` | which fold is held out (single-fold runs) |
| `model.tta=true` | 8-way test-time augmentation at eval |
| `trainer=default_cpu` | CPU debug trainer (5 epochs, 3 batches) |
| `~logger` | drop Comet, log to CSV only |

`logger=comet` is the default; with no Comet API key it warns and falls back automatically,
so no action is needed.

### Evaluation

```powershell
# Test from a checkpoint
python main.py mode=test model=deeplabv3plus_mobilevit_xxs ckpt_path=checkpoints/last-v6.ckpt

# Same, with test-time augmentation
python main.py mode=test model=deeplabv3plus_mobilevit_xxs ckpt_path=checkpoints/last-v6.ckpt model.tta=true

# Efficiency table for all four models -> lightning_logs/efficiency_report.csv
python main.py mode=efficiency_sweep

# Single-model efficiency report
python main.py mode=efficiency model=deeplabv3plus_mobilevit_xxs
```

### Visualization

**Qualitative panels** — 7 columns per sample: true colour (B4/B3/B2), burn-highlight
(B7/B5/B4), ground truth, prediction, both overlays, and the burned-probability heatmap.

```powershell
# Default: test split, 8 samples -> outputs/visualizations/
python visualize.py ckpt_path=checkpoints/last-v6.ckpt

# All test tiles of the current fold, custom output directory
python visualize.py ckpt_path=checkpoints/last-v6.ckpt num_samples=null out_dir=outputs/viz_xxs

# Validation split, a different fold, an ablation backbone
python visualize.py ckpt_path=<ckpt> model=deeplabv3plus_resnet50 split=val dataset.test_fold=2 out_dir=outputs/viz_resnet
```

The checkpoint must be 8-band; 6-band checkpoints raise a stem shape mismatch.

**Training curves** — loss and IoU vs. epoch, from the CSV logs.

```powershell
# One run -> lightning_logs/version_6/curves.png
python scripts/plot_curves.py --run lightning_logs/version_6

# All five folds overlaid
python scripts/plot_curves.py --run lightning_logs/crossval_deeplabv3plus_mobilevit_xxs --folds
```

The script also prints the best epoch and the train-vs-val IoU gap, which is the
overfit/underfit verdict.
