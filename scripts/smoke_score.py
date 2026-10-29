"""
smoke_score.py

Detect smoke / haze in the Landsat-8 tiles and rank every tile by haze level (median
HOT over pixels that are neither cloud nor water). Smoke is told apart from cloud
with SWIR1 and from water with NIR vs green (see utils/haze.py).

Outputs (under outputs/smoke/):
  smoke_scores.csv   one row per tile: scene, fold, hot_median, smoke_frac, cloud_frac, water_frac, burned_frac
  smoke_sheet.png    contact sheet of the most-smoked tiles, fixed colour stretch, for visual review

Optionally (--write-list) the tiles with hot_median at or above --min-hot are written to
configs/exclusions/smoke_tiles.txt, which the datamodule can read to drop them
(dataset.exclude_file=...). That file is tracked in git, so review the sheet first.

Usage:
  python scripts/smoke_score.py --calibrate          # print the two constants for utils/haze.py
  python scripts/smoke_score.py                      # rank tiles, write CSV and sheet
  python scripts/smoke_score.py --min-hot 0.15 --drop <ids> --write-list
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import utils.haze as haze  # noqa: E402
from lightning_modules.scene_splits import scene_key  # noqa: E402

OUT_DIR = Path("outputs/smoke")
LIST_PATH = Path("configs/exclusions/smoke_tiles.txt")


def read_tile(root: Path, fid: str) -> np.ndarray:
    return np.transpose(tifffile.imread(str(root / "images" / f"{fid}.tif")), (2, 0, 1))


def tile_ids(root: Path):
    return sorted(p.stem for p in (root / "images").glob("*.tif"))


def calibrate(root: Path):
    """Print HOT_SMOKE_THR and REF_DARK from the cleanest quarter of tiles."""
    ids = tile_ids(root)
    clear_hot, clear_bands, medians = {}, {}, {}
    for fid in ids:
        refl = haze.to_reflectance(read_tile(root, fid)[:7])
        dn = read_tile(root, fid)
        clear = ~haze.nodata_mask(dn) & ~haze.cloud_mask(refl) & ~haze.water_mask(refl)
        if clear.sum() < 5000:
            continue
        hot = haze.hot_map(refl)
        medians[fid] = float(np.median(hot[clear]))
        clear_hot[fid] = hot[clear][::8]
        clear_bands[fid] = np.stack([refl[b][clear][::8] for b in range(4)])
    cutoff = np.quantile(list(medians.values()), 0.25)
    ref = [f for f, m in medians.items() if m <= cutoff]
    pooled_hot = np.concatenate([clear_hot[f] for f in ref])
    pooled_bands = np.concatenate([clear_bands[f] for f in ref], axis=1)
    thr = float(np.percentile(pooled_hot, 95))
    dark = [float(np.percentile(pooled_bands[b], 2)) for b in range(4)]
    print(f"reference tiles: {len(ref)} (lowest-quartile median HOT <= {cutoff:.4f})")
    print("paste into utils/haze.py:")
    print(f"HOT_SMOKE_THR = {thr:.4f}")
    print(f"REF_DARK = ({dark[0]:.4f}, {dark[1]:.4f}, {dark[2]:.4f}, {dark[3]:.4f})")


def build_table(root: Path, hot_thr):
    fold_of = {}
    sp = root / "splits.parquet"
    if sp.exists():
        import polars as pl
        df = pl.read_parquet(sp)
        fold_of = dict(zip(df["files"].to_list(), df["fold"].to_list()))
    rows = []
    for fid in tile_ids(root):
        dn = read_tile(root, fid)
        feat = haze.smoke_features(dn, hot_thr)
        mask_path = root / "masks" / f"{fid}_mask.tif"
        burned = float((tifffile.imread(str(mask_path)) > 0).mean()) if mask_path.exists() else float("nan")
        rows.append(dict(id=fid, scene=scene_key(fid), fold=fold_of.get(fid, -1),
                         burned_frac=burned, **feat))
    return pd.DataFrame(rows).sort_values("hot_median", ascending=False).reset_index(drop=True)


def contact_sheet(root: Path, table: pd.DataFrame, n: int, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sample = [read_tile(root, f)[[3, 2, 1]][:, ::8, ::8] for f in table.id.iloc[::10]]
    flat = np.concatenate([s.reshape(3, -1) for s in sample], axis=1)
    lo, hi = np.percentile(flat, 2, axis=1), np.percentile(flat, 98, axis=1)

    sel = table.head(n)
    cols = 6
    rows = int(np.ceil(len(sel) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.9))
    for ax in np.ravel(axs):
        ax.axis("off")
    for ax, (rank, r) in zip(np.ravel(axs), sel.iterrows()):
        x = read_tile(root, r.id)[[3, 2, 1]][:, ::4, ::4].astype(float)
        rgb = np.stack([np.clip((x[c] - lo[c]) / (hi[c] - lo[c]), 0, 1) for c in range(3)], -1)
        ax.imshow(rgb)
        ax.set_title(f"#{rank + 1} {r.id[3:]}\nHOT {r.hot_median:.3f}  cloud {r.cloud_frac:.2f}  water {r.water_frac:.2f}", fontsize=7)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=80)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/indonesia")
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--hot-thr", type=float, default=None, help="override utils.haze.HOT_SMOKE_THR")
    p.add_argument("--min-hot", type=float, default=0.15,
                   help="a tile counts as smoke-covered when its median HOT >= this")
    p.add_argument("--drop", nargs="*", default=[],
                   help="tile ids to remove from the flagged list after visual review (recorded in the list header)")
    p.add_argument("--sheet-n", type=int, default=36, help="tiles shown in the contact sheet")
    p.add_argument("--write-list", action="store_true", help=f"write {LIST_PATH}")
    args = p.parse_args()
    root = Path(args.root)

    if args.calibrate:
        calibrate(root)
        return

    thr = haze.HOT_SMOKE_THR if args.hot_thr is None else args.hot_thr
    if thr == 0.0:
        raise SystemExit("HOT_SMOKE_THR is not set. Run with --calibrate and paste the constants into utils/haze.py.")
    table = build_table(root, thr)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "smoke_scores.csv", index=False)
    contact_sheet(root, table, args.sheet_n, OUT_DIR / "smoke_sheet.png")

    flagged = table[(table.hot_median >= args.min_hot) & ~table.id.isin(args.drop)]
    print(f"\nsmoke-covered if median HOT >= {args.min_hot}")
    print(f"flagged {len(flagged)} of {len(table)} tiles "
          f"({flagged.scene.nunique()} scenes); burned px share in flagged tiles: "
          f"{flagged.burned_frac.sum() / table.burned_frac.sum():.1%}")
    print("\nflagged tiles per fold (test set of that fold):")
    print(flagged.groupby("fold").size().reindex(range(5), fill_value=0).to_string())
    print("\ntop 15:")
    print(table.head(15)[["id", "fold", "hot_median", "smoke_frac", "cloud_frac", "water_frac", "burned_frac"]].round(3).to_string(index=False))
    print(f"\nwrote {OUT_DIR / 'smoke_scores.csv'} and {OUT_DIR / 'smoke_sheet.png'}")

    if args.write_list:
        LIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LIST_PATH, "w", encoding="utf8") as fh:
            fh.write(f"# Smoke-covered tiles: median HOT >= {args.min_hot} (cloud and water pixels excluded).\n")
            fh.write("# Generated by scripts/smoke_score.py; review outputs/smoke/smoke_sheet.png before using.\n")
            if args.drop:
                fh.write("# Dropped after visual review (cloud or water, not smoke): " + " ".join(args.drop) + "\n")
            for fid in sorted(flagged.id):
                fh.write(fid + "\n")
        print(f"wrote {LIST_PATH} ({len(flagged)} tiles)")


if __name__ == "__main__":
    main()
