"""
sanity_and_splits.py

Quick helper to verify dataset structure and optionally create a 5-fold `splits.parquet`.

Usage examples:
  # Show counts and mismatches
  python scripts/sanity_and_splits.py --root data/indonesia

  # Create a 5-fold splits.parquet (uses polars if available)
  python scripts/sanity_and_splits.py --root data/indonesia --create

The script is safe to run repeatedly; it will not overwrite an existing
`splits.parquet` unless you pass `--overwrite`.
"""

import argparse
import random
from pathlib import Path
import sys


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/indonesia", help="Dataset root (images/, masks/)")
    p.add_argument("--create", action="store_true", help="Create splits.parquet if missing")
    p.add_argument("--n_folds", type=int, default=5, help="Number of folds to create")
    p.add_argument("--seed", type=int, default=47, help="Random seed")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing splits.parquet")
    return p.parse_args()


def list_files(path: Path):
    return sorted([p for p in path.glob("*") if p.is_file()])


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

    only_imgs = sorted(list(imgs_set - msks_set))
    only_msks = sorted(list(msks_set - imgs_set))
    paired = len(imgs_set & msks_set)

    print(f"Pairs: {paired}")
    if only_imgs:
        print("Images without matching masks (sample 10):", only_imgs[:10])
    if only_msks:
        print("Masks without matching images (sample 10):", only_msks[:10])

    splits_path = root / "splits.parquet"

    if args.create:
        if splits_path.exists() and not args.overwrite:
            print("splits.parquet already exists at", splits_path)
            print("Use --overwrite to replace it")
            return

        ids = sorted(list(imgs_set & msks_set))
        if not ids:
            print("No matching image/mask pairs to create splits from. Aborting.")
            return

        random.seed(args.seed)
        random.shuffle(ids)
        folds = [i % args.n_folds for i in range(len(ids))]

        # Try to use polars for fast parquet writing, fall back to pandas + pyarrow
        try:
            import polars as pl
            df = pl.DataFrame({"id": ids, "fold": folds})
            df.write_parquet(splits_path)
            print("Wrote splits.parquet using polars:", splits_path)
            return
        except Exception:
            pass

        try:
            import pandas as pd
            df = pd.DataFrame({"id": ids, "fold": folds})
            try:
                df.to_parquet(splits_path)
                print("Wrote splits.parquet using pandas:", splits_path)
                return
            except Exception:
                csvp = root / "splits.csv"
                df.to_csv(csvp, index=False)
                print("Wrote splits.csv (parquet write failed):", csvp)
                return
        except Exception:
            csvp = root / "splits.csv"
            with open(csvp, "w", encoding="utf8") as f:
                f.write("id,fold\n")
                for i, _id in enumerate(ids):
                    f.write(f"{_id},{i%args.n_folds}\n")
            print("Wrote simple splits.csv (no pandas/polars available):", csvp)
            return

    else:
        if splits_path.exists():
            print("splits.parquet exists at", splits_path)
        else:
            print("splits.parquet not found. Run with --create to generate one from paired IDs.")


if __name__ == "__main__":
    main()
