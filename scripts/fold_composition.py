"""
fold_composition.py

Explain why some cross-validation folds score lower than others by describing
what each fold's TEST tiles actually contain: tile count, number of scenes,
burned-pixel fraction, and how many tiles have little or no burn.

Reads data/indonesia/splits.parquet and the masks only (no model, no GPU).

Usage:
  python scripts/fold_composition.py
  python scripts/fold_composition.py --root data/indonesia --scenes   # per-scene table too
"""

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lightning_modules.scene_splits import scene_key  # noqa: E402

SMALL_TILE_FRAC = 0.01   # tiles with < 1% burned pixels count as "small/near-empty"


def burned_fraction(mask_path: Path) -> float:
    m = tifffile.imread(str(mask_path))
    return float((m > 0).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/indonesia")
    p.add_argument("--scenes", action="store_true", help="also print one row per scene")
    args = p.parse_args()

    root = Path(args.root)
    df = pl.read_parquet(root / "splits.parquet")

    fold_tiles = defaultdict(list)   # fold -> [(tile_id, burned_frac)]
    for fid, fold in zip(df["files"].to_list(), df["fold"].to_list()):
        mask_path = root / "masks" / f"{fid}_mask.tif"
        if not mask_path.exists():
            continue
        fold_tiles[int(fold)].append((fid, burned_fraction(mask_path)))

    head = (f"{'fold':>4} {'tiles':>6} {'scenes':>7} {'burned%':>8} {'median%':>8} "
            f"{'empty':>6} {'<1%':>5} {'max%':>7}")
    print("\nBurned-pixel composition per fold (these tiles are that fold's TEST set)\n")
    print(head)
    print("-" * len(head))
    for fold in sorted(fold_tiles):
        fr = np.array([f for _, f in fold_tiles[fold]])
        scenes = {scene_key(t) for t, _ in fold_tiles[fold]}
        print(f"{fold:>4} {len(fr):>6} {len(scenes):>7} {fr.mean()*100:>8.2f} "
              f"{np.median(fr)*100:>8.2f} {(fr == 0).sum():>6} "
              f"{(fr < SMALL_TILE_FRAC).sum():>5} {fr.max()*100:>7.2f}")

    all_fr = np.array([f for tiles in fold_tiles.values() for _, f in tiles])
    print("-" * len(head))
    print(f" all {len(all_fr):>6} {'':>7} {all_fr.mean()*100:>8.2f} {np.median(all_fr)*100:>8.2f} "
          f"{(all_fr == 0).sum():>6} {(all_fr < SMALL_TILE_FRAC).sum():>5} {all_fr.max()*100:>7.2f}")
    print("\nburned% = mean per-tile burned fraction; empty = tiles with zero burned pixels;"
          f"\n<1% = tiles under {SMALL_TILE_FRAC:.0%} burned (hard for IoU: a few pixels swing the score)\n")

    if args.scenes:
        print("Per-scene breakdown\n")
        print(f"{'fold':>4}  {'scene':<22} {'tiles':>5} {'burned%':>8}")
        rows = []
        for fold, tiles in fold_tiles.items():
            by_scene = defaultdict(list)
            for t, f in tiles:
                by_scene[scene_key(t)].append(f)
            for s, fs in by_scene.items():
                rows.append((fold, s, len(fs), statistics.fmean(fs) * 100))
        for fold, s, n, b in sorted(rows):
            print(f"{fold:>4}  {s:<22} {n:>5} {b:>8.2f}")
        print()


if __name__ == "__main__":
    main()
