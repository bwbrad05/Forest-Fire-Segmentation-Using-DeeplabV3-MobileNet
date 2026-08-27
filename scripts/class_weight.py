"""
class_weight.py

Compute the Weighted Cross-Entropy class weight from the training data.

Lyu et al., *J. Agric. Food Res.* 26 (2026) 102680, equation (10) defines the
WCE weight for class c as

    w_c = N_total / N_c

and stresses (their section 2.2.4) that it must be accumulated over the *entire
training set* rather than per mini-batch, so batch-to-batch fluctuation in the
burned-pixel fraction does not move the weight around.

This project's ``BCEDiceLoss`` takes a single ``pos_weight`` — the weight of the
burned class relative to background in ``BCEWithLogitsLoss`` — so the ratio that
matters here is

    pos_weight = N_background / N_burned

which is ``w_burned / w_background`` under the paper's formula, i.e. the same
quantity expressed the way PyTorch wants it. The configs currently carry a
hand-picked ``bce_pos_weight: 10.0``; run this to replace that guess with the
measured value.

Fold handling matches ``IndonesiaDataModule``: test = ``--test_fold``,
val = ``(test_fold + 1) % 5``, train = the remaining three. Only the training
folds are counted, because computing the weight over val/test would leak label
statistics from data the model is not supposed to have seen.

Usage:
  python scripts/class_weight.py --root data/indonesia
  python scripts/class_weight.py --root data/indonesia --test_fold 2
  python scripts/class_weight.py --root data/indonesia --all_folds
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lightning_modules.indonesia_datamodule import _read_tif  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/indonesia", help="Dataset root (images/, masks/)")
    p.add_argument("--test_fold", type=int, default=0, help="Held-out fold [0-4]")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--all_folds", action="store_true",
                   help="Report the weight for every choice of test_fold")
    return p.parse_args()


def train_folds(test_fold: int, n_folds: int):
    """Mirror IndonesiaDataModule: test = k, val = k+1, train = the rest."""
    val_fold = (test_fold + 1) % n_folds
    return [f for f in range(n_folds) if f not in (test_fold, val_fold)]


def load_fold_map(root: Path):
    """Return {file_id: fold}, from splits.parquet when present."""
    splits_path = root / "splits.parquet"
    if splits_path.exists():
        import polars as pl

        df = pl.read_parquet(splits_path)
        if "files" not in df.columns or "fold" not in df.columns:
            raise SystemExit(
                f"{splits_path} has columns {df.columns}; expected 'files' and 'fold'. "
                "Regenerate with: python scripts/sanity_and_splits.py --create"
            )
        return dict(zip(df["files"].to_list(), df["fold"].to_list()))

    from lightning_modules.scene_splits import assign_scene_folds

    ids = sorted(p.stem for p in (root / "images").glob("*"))
    print("WARNING: no splits.parquet — using an on-the-fly scene-aware split.")
    return assign_scene_folds(ids, n_folds=5)


def count_pixels(root: Path, file_ids):
    """Return (burned_pixels, total_pixels) accumulated over the given tiles."""
    burned = 0
    total = 0
    missing = 0
    for fid in file_ids:
        mask_path = root / "masks" / f"{fid}_mask.tif"
        if not mask_path.exists():
            missing += 1
            continue
        mask = _read_tif(mask_path)
        arr = np.asarray(mask, dtype=np.float32)
        burned += int((arr > 0).sum())
        total += int(arr.size)
    if missing:
        print(f"  ({missing} tiles skipped — no matching mask)")
    return burned, total


def report(root: Path, fold_map, test_fold: int, n_folds: int):
    folds = train_folds(test_fold, n_folds)
    ids = [f for f, k in fold_map.items() if k in folds]
    burned, total = count_pixels(root, ids)

    if burned == 0 or total == 0:
        raise SystemExit("No burned pixels counted — check the mask directory.")

    background = total - burned
    frac = burned / total
    pos_weight = background / burned

    print(f"test_fold={test_fold}  train folds={folds}  tiles={len(ids)}")
    print(f"  burned pixels     : {burned:,} ({100 * frac:.3f} %)")
    print(f"  background pixels : {background:,}")
    print(f"  N_total / N_burned     = {total / burned:.3f}   (paper w_c, burned class)")
    print(f"  N_total / N_background = {total / background:.3f}   (paper w_c, background)")
    print(f"  --> bce_pos_weight     = {pos_weight:.3f}")
    print(f"      model.bce_pos_weight={pos_weight:.3f}")
    return pos_weight


def main():
    args = parse_args()
    root = Path(args.root)
    if not (root / "masks").exists():
        raise SystemExit(f"ERROR: {root / 'masks'} not found")

    fold_map = load_fold_map(root)

    if args.all_folds:
        weights = []
        for k in range(args.n_folds):
            weights.append(report(root, fold_map, k, args.n_folds))
            print()
        print(f"mean over folds: {np.mean(weights):.3f} +/- {np.std(weights):.3f}")
        print("Cross-validation uses a different train set per fold; the spread above "
              "shows whether one shared value is defensible.")
    else:
        report(root, fold_map, args.test_fold, args.n_folds)


if __name__ == "__main__":
    main()
