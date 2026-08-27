# IndonesiaDataModule — fully rewritten from the incomplete stub.
# Handles:
#   - Landsat-8 (227 images, 512×512)
#   - Band normalisation tuned for Indonesian tropical conditions
#   - Cloud-mask filtering
#   - Small-dataset augmentation for robustness

import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Band normalisation statistics
# --------------------------------------------------------------------------- #

# Per-band mean/std computed from ALL 227 images of THIS dataset (all 8 bands).
# Previous values were generic Landsat-8 SR defaults (DN 0-10000) that did NOT
# match this dataset's scale (actual values reach ~18000), leaving inputs at
# mean ~+12 / std ~8 instead of the intended mean~0 / std~1. Recomputed 2026-07-10.
# Band order, established empirically from all 227 images (2026-07-17):
#   0=B1 Coastal  1=B2 Blue  2=B3 Green  3=B4 Red
#   4=B5 NIR      5=B6 SWIR1 6=B7 SWIR2  7=B9 Cirrus
# This is the standard Landsat-8 30 m stack (B8 Pan is 15 m and omitted), and
# matches the 8 channels reported in the paper (Table I).
#
# Two facts pinned the order down:
#  - the NIR reflectance peak sits at index 4, so the stack starts at B1, not
#    B2. An earlier 6-band cut therefore silently dropped B7 (SWIR2) -- the band
#    the Normalized Burn Ratio depends on, NBR = (B5 - B7)/(B5 + B7).
#  - index 7 is spectral, not a QA mask: ~8k distinct values, no zero pixels and
#    0.80 neighbour correlation. Its ~0.0 correlation with every surface band is
#    expected of B9, which water vapour absorption keeps from seeing the ground.
#
# Stats are per-band mean/std over every pixel of all 227 images.
LANDSAT8_MEANS = torch.tensor([11592.3, 10282.3, 9015.6, 7896.5, 18340.9, 11298.5, 6352.3, 485.8])
LANDSAT8_STDS  = torch.tensor([ 6431.2,  6780.0,  6697.5, 7130.1,  7922.3,  6089.8, 4739.9, 855.6])

SENSOR_STATS = {
    "landsat8":  (LANDSAT8_MEANS,  LANDSAT8_STDS),
}


# --------------------------------------------------------------------------- #
# Spectral indices
# --------------------------------------------------------------------------- #

# Band positions in the 8-band stack (see the band-order note above).
_B4_RED, _B5_NIR, _B6_SWIR1, _B7_SWIR2 = 3, 4, 5, 6

SPECTRAL_INDEX_NAMES = ("NBR", "NDVI", "NBR2")
N_SPECTRAL_INDICES = len(SPECTRAL_INDEX_NAMES)


def _normalised_difference(a, b):
    """(a - b) / (a + b), guarded against the 0/0 that cloud-filled pixels give."""
    denom = a + b
    return torch.where(
        denom.abs() < 1e-6,
        torch.zeros_like(denom),
        (a - b) / torch.where(denom.abs() < 1e-6, torch.ones_like(denom), denom),
    )


def _spectral_indices(image):
    """Compute (3, H, W) of NBR, NDVI, NBR2 from a RAW 8-band tensor.

    Must be called before ``_normalise``: these are ratios of reflectance, and
    (a-b)/(a+b) over z-scored values is not an index of anything.

    NBR  = (NIR - SWIR2)/(NIR + SWIR2)  -- the standard burned-area index; the
           burn signal this dataset's 6-band cut used to be missing entirely.
    NDVI = (NIR - Red)/(NIR + Red)      -- vegetation greenness, hence its
           absence over burn scars.
    NBR2 = (SWIR1 - SWIR2)/(SWIR1 + SWIR2) -- sensitive to post-fire moisture
           and char, and complementary to NBR.

    All three are normalised differences, so they already land in [-1, 1] and
    need no further scaling to sit alongside the z-scored bands.
    """
    nbr  = _normalised_difference(image[_B5_NIR],   image[_B7_SWIR2])
    ndvi = _normalised_difference(image[_B5_NIR],   image[_B4_RED])
    nbr2 = _normalised_difference(image[_B6_SWIR1], image[_B7_SWIR2])
    return torch.stack([nbr, ndvi, nbr2], dim=0)

# Cloud-masking helper

def apply_cloud_mask(image, cloud_mask, fill_value=0.0):
    if cloud_mask is None:
        return image
    return torch.where(cloud_mask.bool(), torch.full_like(image, fill_value), image)


def cloud_coverage_fraction(cloud_mask):
    if cloud_mask is None:
        return 0.0
    return cloud_mask.float().mean().item()


# Dataset

class IndonesiaDataset(Dataset):
    """
    Dataset for Indonesian burned-area segmentation.

    Directory layout expected:
        root/
          images/<id>.tif         -- multi-band satellite image (GeoTIFF)
          masks/<id>_mask.tif     -- binary burned-area mask (0/1)
          cloud_masks/<id>.tif    -- optional cloud mask
          splits.parquet          -- polars DataFrame: files(str), fold(int 0-4)

    Parameters
    ----------
    root : str or Path
    folds : list of int
    sensor : str  'landsat8'
    augment : bool
    cloud_threshold : float  Skip patches with cloud fraction > threshold
    patch_size : int
    """

    def __init__(
        self,
        root,
        folds,
        sensor="landsat8",
        augment=False,
        cloud_threshold=0.3,
        patch_size=512,
        spectral_indices=False,
    ):
        super().__init__()
        self.root = Path(root)
        self.sensor = sensor
        self.augment = augment
        self.cloud_threshold = cloud_threshold
        self.patch_size = patch_size
        self.spectral_indices = spectral_indices
        self.means, self.stds = SENSOR_STATS[sensor]

        # Resolve fold membership. Prefer a reproducible splits.parquet; if it is
        # absent, fall back to an on-the-fly SCENE-aware split so tiles from the
        # same acquisition never leak across folds. A splits.parquet that exists
        # but is malformed is a hard error — never a silent fall-through, which is
        # how the old code masked a wrong column name and reverted to a leaky
        # per-tile mod-5 split without warning.
        splits_path = self.root / "splits.parquet"
        if splits_path.exists():
            import polars as pl
            df = pl.read_parquet(splits_path)
            if "files" not in df.columns or "fold" not in df.columns:
                raise KeyError(
                    f"{splits_path} has columns {df.columns}; expected 'files' and "
                    "'fold'. Regenerate it with:\n  python scripts/sanity_and_splits.py "
                    f"--root {self.root} --create --overwrite"
                )
            file_ids = df.filter(pl.col("fold").is_in(folds))["files"].to_list()
        else:
            from lightning_modules.scene_splits import assign_scene_folds
            logger.warning(
                "splits.parquet not found at %s — using an on-the-fly scene-aware "
                "fold split (deterministic, but NOT the paper's per-tile split). "
                "Run scripts/sanity_and_splits.py --create for a reproducible file.",
                splits_path,
            )
            all_files = sorted(p.stem for p in (self.root / "images").glob("*.tif"))
            fold_map = assign_scene_folds(all_files, n_folds=5)
            file_ids = [f for f in all_files if fold_map[f] in folds]

        self.samples = []
        for fid in file_ids:
            img_path   = self.root / "images"      / f"{fid}.tif"
            mask_path  = self.root / "masks"       / f"{fid}_mask.tif"
            cloud_path = self.root / "cloud_masks" / f"{fid}.tif"
            if not img_path.exists() or not mask_path.exists():
                continue
            entry = {"image": img_path, "mask": mask_path, "id": fid}
            if cloud_path.exists():
                entry["cloud"] = cloud_path
            self.samples.append(entry)

        logger.info(
            "IndonesiaDataset [folds=%s, sensor=%s]: %d samples",
            folds, sensor, len(self.samples),
        )

        self._augment_fn = _build_augmentation() if augment else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        entry = self.samples[index]

        image = _read_tif(entry["image"])
        mask  = _read_tif(entry["mask"])

        cloud = None
        if "cloud" in entry:
            cloud = _read_tif(entry["cloud"])
            if cloud_coverage_fraction(cloud) > self.cloud_threshold:
                return self[(index + 1) % len(self)]

        image = apply_cloud_mask(image, cloud)
        # Indices are derived from the raw bands, then appended after the bands
        # themselves are z-scored -- see _spectral_indices for why the order
        # matters.
        indices = _spectral_indices(image) if self.spectral_indices else None
        image = self._normalise(image)
        if indices is not None:
            image = torch.cat([image, indices], dim=0)
        image = _ensure_size(image, self.patch_size)
        mask  = _ensure_size(mask,  self.patch_size, is_mask=True)

        if self._augment_fn is not None:
            image, mask = self._augment_fn(image, mask)

        return {
            "post": image,
            "mask": mask,
            "id":   entry["id"],
        }

    def _normalise(self, image):
        # Handle variable band counts: use min(image_bands, stats_bands)
        n_bands = min(image.shape[0], self.means.shape[0])
        image = image[:n_bands]
        means = self.means[:n_bands].view(-1, 1, 1)
        stds  = self.stds[:n_bands].view(-1, 1, 1)
        return (image - means) / (stds + 1e-8)


# DataModule

class IndonesiaDataModule(pl.LightningDataModule):
    """
    LightningDataModule for Indonesian burned-area dataset (Landsat-8, 227 images).

    5-fold cross-validation:
      - test  = fold test_fold
      - val   = fold (test_fold + 1) % 5
      - train = remaining 3 folds

    Parameters
    ----------
    root : str
    batch_size : int
    num_workers : int
    test_fold : int  [0-4]
    sensor : str  'landsat8'
    patch_size : int
    cloud_threshold : float
    spectral_indices : bool
        Append NBR, NDVI and NBR2 as three extra input channels. When enabled the
        model's ``in_channels`` must be raised from 8 to 11 to match.
    """

    def __init__(
        self,
        root="data/indonesia",
        batch_size=8,
        num_workers=4,
        test_fold=0,
        sensor="landsat8",
        patch_size=512,
        cloud_threshold=0.3,
        spectral_indices=False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        val_fold    = (test_fold + 1) % 5
        train_folds = [f for f in range(5) if f not in (test_fold, val_fold)]
        self._fold_map = {
            "train": train_folds,
            "val":   [val_fold],
            "test":  [test_fold],
        }
        self.train_dataset = None
        self.val_dataset   = None
        self.test_dataset  = None

    def setup(self, stage=None):
        hp = self.hparams
        kw = dict(
            root=hp.root,
            sensor=hp.sensor,
            patch_size=hp.patch_size,
            cloud_threshold=hp.cloud_threshold,
            spectral_indices=hp.spectral_indices,
        )
        if stage in ("fit", None):
            self.train_dataset = IndonesiaDataset(folds=self._fold_map["train"], augment=True,  **kw)
            self.val_dataset   = IndonesiaDataset(folds=self._fold_map["val"],   augment=False, **kw)
        if stage in ("test", "predict", None):
            self.test_dataset  = IndonesiaDataset(folds=self._fold_map["test"],  augment=False, **kw)

    def _loader(self, dataset, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
            pin_memory=True,
            drop_last=False,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def train_dataloader(self):   return self._loader(self.train_dataset, shuffle=True)
    def val_dataloader(self):     return self._loader(self.val_dataset)
    def test_dataloader(self):    return self._loader(self.test_dataset)
    def predict_dataloader(self): return self._loader(self.test_dataset)


# I/O helpers

_RASTERIO_WARNED = False

def _to_chw(data):
    """Normalise a raw array to (C, H, W) float32.

    Readers disagree on layout: rasterio yields (C, H, W), tifffile yields
    (H, W, C) for contiguous-planar files, and masks come back 2-D. Band counts
    are small (8) next to the 512-px spatial dims, so the short axis is the
    channel axis.
    """
    data = data.astype(np.float32)
    if data.ndim == 2:
        data = data[np.newaxis]
    elif data.ndim == 3 and data.shape[-1] < data.shape[0]:
        data = np.transpose(data, (2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(data))


def _read_tif(path):
    """Read a GeoTIFF and return float32 tensor of shape (C, H, W).

    Three readers are tried in order because GDAL is the fragile part of this
    stack on Windows: a pip rasterio wheel bundles its own GDAL and can corrupt
    the heap next to conda's (silent 0xC0000374 crash), while a mismatched
    conda-forge build fails with "DLL load failed while importing _base".

    Nothing downstream uses geospatial metadata — only the pixel array — so
    tifffile is a complete substitute and needs no GDAL at all. It is the
    fallback rather than the default so that correctly-installed machines keep
    rasterio's broader format support.
    """
    try:
        import rasterio
        with rasterio.open(path) as src:
            return _to_chw(src.read())
    except (ImportError, OSError) as e:
        # Warn once per process, not once per tile: this runs for every read, so
        # at ~181 tiles per epoch it otherwise emits tens of thousands of
        # identical lines and buries the training metrics.
        global _RASTERIO_WARNED
        if not _RASTERIO_WARNED:
            _RASTERIO_WARNED = True
            logger.warning(
                "rasterio unavailable (%s: %s); falling back to tifffile for all "
                "reads. This is fine — the pipeline uses pixel values only. "
                "(Reported once per process.)",
                type(e).__name__, e,
            )

    try:
        import tifffile
        return _to_chw(tifffile.imread(str(path)))
    except ImportError:
        pass

    # Last resort: xarray, as in the original codebase.
    import xarray as xr
    return _to_chw(xr.open_dataarray(path).fillna(0).to_numpy())


def _ensure_size(tensor, size, is_mask=False):
    _, h, w = tensor.shape
    if h == size and w == size:
        return tensor
    mode  = "nearest" if is_mask else "bilinear"
    align = None if is_mask else False
    return F.interpolate(
        tensor.unsqueeze(0), size=(size, size), mode=mode, align_corners=align
    ).squeeze(0)


# Augmentation

class _Augmentation:
    """Random flips / 90° rotations / intensity shift.

    Defined at module level (not a nested closure) so that a Dataset holding an
    instance stays picklable — DataLoader workers on spawn/forkserver platforms
    (Windows, and Python 3.14+ on POSIX) must pickle the dataset to hand it to
    each worker, and a local closure would raise
    ``Can't pickle local object '_build_augmentation.<locals>.augment'``.
    """

    def __call__(self, image, mask):
        if random.random() < 0.5:
            image = torch.flip(image, dims=[-1])
            mask  = torch.flip(mask,  dims=[-1])
        if random.random() < 0.5:
            image = torch.flip(image, dims=[-2])
            mask  = torch.flip(mask,  dims=[-2])
        if random.random() < 0.5:
            k = random.randint(1, 3)
            image = torch.rot90(image, k, dims=[-2, -1])
            mask  = torch.rot90(mask,  k, dims=[-2, -1])
        if random.random() < 0.3:
            shift = torch.empty(image.shape[0], 1, 1).uniform_(-0.1, 0.1)
            image = image + shift
        return image, mask


def _build_augmentation():
    return _Augmentation()
