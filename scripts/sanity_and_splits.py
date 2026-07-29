"""
sanity_and_splits.py

Verify dataset structure and optionally create a 5-fold ``splits.parquet``.

Two split protocols are supported (see the thesis's spatial-leakage discussion):

  scene  (default) — every tile of one Landsat acquisition (same path/row + date)
                     stays in ONE fold. No spatial leakage; honest generalisation.
  tile             — random per-tile assignment. Leaky, but matches the reference
                     paper's protocol, so it is kept for apples-to-apples numbers.

Usage examples:
  # Show counts / mismatches only
  python scripts/sanity_and_splits.py --root data/indonesia

  # Create a scene-aware splits.parquet
  python scripts/sanity_and_splits.py --root data/indonesia --create

  # Create a per-tile (paper-style) splits.parquet, overwriting any existing file
  python scripts/sanity_and_splits.py --root data/indonesia --create --mode tile --overwrite

The output parquet has columns ``files`` (tile stem), ``fold`` (0..n-1) and
``scene`` — matching what lightning_modules/indonesia_datamodule.py reads.
"""

import argparse
import random
import sys
from pathlib import Path

# Allow running as `python scripts/sanity_and_splits.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lightning_modules.scene_splits import (  # noqa: E402
    assign_scene_folds,
    count_cross_fold_scenes,
    fold_summary,
    scene_key,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/indonesia", help="Dataset root (images/, masks/)")
    p.add_argument("--create", action="store_true", help="Create splits.parquet if missing")
    p.add_argument("--mode", choices=["scene", "tile"], default="scene",
                   help="scene = leak-free per-acquisition (default); tile = random per-tile (paper-style)")
    p.add_argument("--n_folds", type=int, default=5, help="Number of folds to create")
    p.add_argument("--seed", type=int, default=47, help="Random seed (tile mode only)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing splits.parquet")
    return p.parse_args()


def list_files(path: Path):
    return sorted([p for p in path.glob("*") if p.is_file()])


def build_fold_map(ids, mode, n_folds, seed):
    """Return {file_id: fold} for the requested protocol."""
    if mode == "scene":
        return assign_scene_folds(ids, n_folds=n_folds)
    # tile: random per-tile assignment (reproducible via seed)
    shuffled = list(ids)
    random.seed(seed)
    random.shuffle(shuffled)
    return {fid: i % n_folds for i, fid in enumerate(shuffled)}


def report_split(fold_map, n_folds):
    """Print per-fold counts and the scene-leakage figure."""
    print("\nFold distribution:")
    for row in fold_summary(fold_map, n_folds=n_folds):
        print(f"  fold {row['fold']}: {row['tiles']:>4d} tiles across {row['scenes']:>3d} scenes")
    leaked = count_cross_fold_scenes(fold_map)
    n_scenes = len({scene_key(f) for f in fold_map})
    print(f"\nScenes total          : {n_scenes}")
    print(f"Scenes split >1 fold  : {leaked}   "
          f"({'OK - no spatial leakage' if leaked == 0 else 'LEAKAGE present'})")


def write_parquet(splits_path, ids, fold_map):
    """Write files/fold/scene, preferring polars, then pandas, then CSV."""
    rows_files = list(ids)
    rows_fold  = [fold_map[i] for i in ids]
    rows_scene = [scene_key(i) for i in ids]

    try:
        import polars as pl
        pl.DataFrame({"files": rows_files, "fold": rows_fold, "scene": rows_scene}) \
            .write_parquet(splits_path)
        print(f"\nWrote {splits_path} using polars ({len(rows_files)} tiles).")
        return
    except Exception:
        pass

    try:
        import pandas as pd
        df = pd.DataFrame({"files": rows_files, "fold": rows_fold, "scene": rows_scene})
        try:
            df.to_parquet(splits_path)
            print(f"\nWrote {splits_path} using pandas ({len(rows_files)} tiles).")
            return
        except Exception:
            csvp = splits_path.with_suffix(".csv")
            df.to_csv(csvp, index=False)
            print(f"\nWrote {csvp} (parquet unavailable). NOTE: the datamodule reads "
                  "parquet — install polars/pyarrow to produce splits.parquet.")
            return
    except Exception:
        csvp = splits_path.with_suffix(".csv")
        with open(csvp, "w", encoding="utf8") as f:
            f.write("files,fold,scene\n")
            for fid in ids:
                f.write(f"{fid},{fold_map[fid]},{scene_key(fid)}\n")
        print(f"\nWrote {csvp} (no polars/pandas). NOTE: the datamodule reads parquet.")


def main():
    args = parse_args()
    root = Path(args.root)
    imgs_dir = root / "images"
    masks_dir = root / "masks"

    if not imgs_dir.exists() or not masks_dir.exists():
        print("ERROR: expected folders:", imgs_dir, "and", masks_dir)
        sys.exit(2)

    imgs = list_files(imgs_dir)
    msks = list_files(masks_dir)

    img_ids = [p.stem for p in imgs]
    mask_ids = [p.stem.replace("_mask", "") for p in msks]

    imgs_set = set(img_ids)
    msks_set = set(mask_ids)

    print(f"Found {len(imgs)} images and {len(msks)} masks")
    print("Sample images:", [p.name for p in imgs[:5]])
    print("Sample masks:", [p.name for p in msks[:5]])

    only_imgs = sorted(imgs_set - msks_set)
    only_msks = sorted(msks_set - imgs_set)
    print(f"Pairs: {len(imgs_set & msks_set)}")
    if only_imgs:
        print("Images without matching masks (sample 10):", only_imgs[:10])
    if only_msks:
        print("Masks without matching images (sample 10):", only_msks[:10])

    splits_path = root / "splits.parquet"

    if not args.create:
        if splits_path.exists():
            print("\nsplits.parquet exists at", splits_path)
        else:
            print("\nsplits.parquet not found. Run with --create to generate one.")
        return

    if splits_path.exists() and not args.overwrite:
        print("\nsplits.parquet already exists at", splits_path)
        print("Use --overwrite to replace it")
        return

    ids = sorted(imgs_set & msks_set)
    if not ids:
        print("No matching image/mask pairs to create splits from. Aborting.")
        return

    print(f"\nCreating '{args.mode}' split over {len(ids)} tiles, {args.n_folds} folds.")
    fold_map = build_fold_map(ids, args.mode, args.n_folds, args.seed)
    report_split(fold_map, args.n_folds)
    write_parquet(splits_path, ids, fold_map)


if __name__ == "__main__":
    main()
