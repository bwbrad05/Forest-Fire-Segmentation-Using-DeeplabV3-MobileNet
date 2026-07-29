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

# Per-band mean/std computed from ALL 227 images of THIS dataset (first 6 bands).
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
    ):
        super().__init__()
        self.root = Path(root)
        self.sensor = sensor
        self.augment = augment
        self.cloud_threshold = cloud_threshold
        self.patch_size = patch_size
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
        image = self._normalise(image)
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

def _read_tif(path):
    """Read a GeoTIFF and return float32 tensor of shape (C, H, W)."""
    try:
        import rasterio
        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)
        return torch.from_numpy(data)
    except ImportError:
        # Fallback: use xarray (available in original codebase)
        import xarray as xr
        data = xr.open_dataarray(path).fillna(0).to_numpy().astype(np.float32)
        if data.ndim == 2:
            data = data[np.newaxis]
        return torch.from_numpy(data)


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

def _build_augmentation():
    def augment(image, mask):
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
    return augment
