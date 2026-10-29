"""
dehaze_dataset.py

Prototype smoke/haze correction (utils/haze.py: HOT-binned dark-object subtraction on
B1-B4) applied to the whole dataset, plus a before/after preview.

The corrected copy is a drop-in dataset: images are rewritten, masks and
splits.parquet are copied unchanged, so training and testing read it through
``dataset.root=data/indonesia_dehazed`` with no code change.

This is a prototype, not a validated product. Judge it on the preview sheet and on
the model results, not on the fact that it runs.

Usage:
  python scripts/dehaze_dataset.py --preview-only        # just render outputs/smoke/dehaze_preview.png
  python scripts/dehaze_dataset.py                       # write data/indonesia_dehazed
  python scripts/dehaze_dataset.py --only-flagged        # correct only the tiles in the smoke list
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import utils.haze as haze  # noqa: E402

DEFAULT_LIST = "configs/exclusions/smoke_tiles.txt"


def read_list(path):
    with open(path, encoding="utf8") as fh:
        return [l.strip() for l in fh if l.strip() and not l.lstrip().startswith("#")]


def read_tile(root, fid):
    return np.transpose(tifffile.imread(str(Path(root) / "images" / f"{fid}.tif")), (2, 0, 1))


def preview(root, ids, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    allids = sorted(p.stem for p in (Path(root) / "images").glob("*.tif"))
    flat = np.concatenate([read_tile(root, f)[[3, 2, 1]][:, ::8, ::8].reshape(3, -1) for f in allids[::10]], axis=1)
    lo, hi = np.percentile(flat, 2, axis=1), np.percentile(flat, 98, axis=1)

    def rgb(x, bands, rng):
        a = x[list(bands)][:, ::2, ::2].astype(float)
        return np.stack([np.clip((a[i] - rng[0][i]) / (rng[1][i] - rng[0][i]), 0, 1) for i in range(3)], -1)

    flat7 = np.concatenate([read_tile(root, f)[[6, 4, 3]][:, ::8, ::8].reshape(3, -1) for f in allids[::10]], axis=1)
    lo7, hi7 = np.percentile(flat7, 2, axis=1), np.percentile(flat7, 98, axis=1)

    fig, axs = plt.subplots(len(ids), 4, figsize=(11, 2.8 * len(ids)))
    axs = np.atleast_2d(axs)
    for r, fid in enumerate(ids):
        before = read_tile(root, fid)
        after = haze.dehaze_hot_dos(before)
        panels = [("true colour before", rgb(before, (3, 2, 1), (lo, hi))),
                  ("true colour after", rgb(after, (3, 2, 1), (lo, hi))),
                  ("B7/B5/B4 before", rgb(before, (6, 4, 3), (lo7, hi7))),
                  ("B7/B5/B4 after", rgb(after, (6, 4, 3), (lo7, hi7)))]
        for c, (name, img) in enumerate(panels):
            axs[r, c].imshow(img)
            axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
            axs[r, c].set_title(f"{fid[3:]}  {name}" if c == 0 else name, fontsize=7)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=80)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/indonesia")
    p.add_argument("--out", default="data/indonesia_dehazed")
    p.add_argument("--list", default=DEFAULT_LIST)
    p.add_argument("--only-flagged", action="store_true")
    p.add_argument("--preview-only", action="store_true")
    p.add_argument("--preview-n", type=int, default=6)
    p.add_argument("--preview-out", default="outputs/smoke/dehaze_preview.png")
    args = p.parse_args()

    flagged = read_list(args.list)
    preview(args.root, flagged[: args.preview_n], args.preview_out)
    print(f"wrote {args.preview_out}")
    if args.preview_only:
        return

    src, dst = Path(args.root), Path(args.out)
    (dst / "images").mkdir(parents=True, exist_ok=True)
    shutil.copytree(src / "masks", dst / "masks", dirs_exist_ok=True)
    if (src / "splits.parquet").exists():
        shutil.copy2(src / "splits.parquet", dst / "splits.parquet")
    if (src / "cloud_masks").exists():
        shutil.copytree(src / "cloud_masks", dst / "cloud_masks", dirs_exist_ok=True)

    fl = set(flagged)
    ids = sorted(p.stem for p in (src / "images").glob("*.tif"))
    changed = []
    for i, fid in enumerate(ids, 1):
        arr = read_tile(src, fid)
        new = haze.dehaze_hot_dos(arr) if (not args.only_flagged or fid in fl) else arr
        shift = float(np.abs(new[:4].astype(float) - arr[:4].astype(float)).mean())
        changed.append((fid, shift))
        tifffile.imwrite(str(dst / "images" / f"{fid}.tif"), np.transpose(new, (1, 2, 0)))
        if i % 25 == 0:
            print(f"  {i}/{len(ids)}")
    big = sorted(changed, key=lambda t: -t[1])[:5]
    print(f"wrote {len(ids)} tiles to {dst}")
    print("largest mean change in B1-B4 (DN): " + ", ".join(f"{f[3:]}={s:.0f}" for f, s in big))
    print(f"median change over all tiles: {np.median([s for _, s in changed]):.0f} DN")


if __name__ == "__main__":
    main()
