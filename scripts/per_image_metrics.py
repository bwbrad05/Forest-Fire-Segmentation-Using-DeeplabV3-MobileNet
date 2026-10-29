"""
per_image_metrics.py

Score every test tile of a cross-validation run on its own, and show how the
smoke-covered tiles (configs/exclusions/smoke_tiles.txt) differ from the rest.

Each tile is a member of exactly one fold's test set, so running every fold's best
checkpoint on its own test fold covers all 227 tiles once. The prediction is the
plain argmax of the network: no test-time augmentation and no post-processing.

Writes, next to the run:
  per_image_metrics.csv   one row per tile (TP/FP/FN, IoU, F1, precision, recall, smoke flag, haze level)
  iou_vs_haze.png         tile IoU against haze level, smoke-covered tiles highlighted
and optionally one panel PNG per tile (--panels DIR) with its own scores in the title.

It prints (1) the usual per-fold mean +/- std, (2) the same with the smoke-covered
tiles removed from the evaluation, and (3) smoky tiles against clean tiles.

Each fold's checkpoint is the one Lightning itself recorded as best: the newest
last*.ckpt in the fold stores ModelCheckpoint.best_model_path. If that is missing it
falls back to the lowest val_loss among the three newest epoch=*.ckpt files, because
folders reused by an earlier, crashed run also hold that run's checkpoints.

Usage:
  python scripts/per_image_metrics.py --crossval lightning_logs/crossval_cv_baseline
  python scripts/per_image_metrics.py --crossval lightning_logs/crossval_cv_baseline --panels outputs/per_image_baseline
  python scripts/per_image_metrics.py --crossval ... --device cpu        # if the GPU is busy
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import utils.haze as haze  # noqa: E402
from lightning_modules import IndonesiaDataset  # noqa: E402
from lightning_modules.indonesia_datamodule import _read_id_list  # noqa: E402
from lightning_modules.scene_splits import scene_key  # noqa: E402
from neural_net import DeepLabV3PlusMobileViT  # noqa: E402

DEFAULT_LIST = "configs/exclusions/smoke_tiles.txt"
DEFAULT_SMOKE_CSV = "outputs/smoke/smoke_scores.csv"


def best_checkpoint(fold_dir: Path):
    """Return (path, how) for the fold's best checkpoint."""
    ckpt_dir = fold_dir / "checkpoints"
    lasts = sorted(ckpt_dir.glob("last*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if lasts:
        state = torch.load(lasts[0], map_location="cpu", weights_only=False)
        for key, val in state.get("callbacks", {}).items():
            if key.startswith("ModelCheckpoint") and val.get("best_model_path"):
                cand = ckpt_dir / Path(str(val["best_model_path"]).replace("\\", "/")).name
                if cand.exists():
                    return cand, f"recorded as best in {lasts[0].name}"
    epochs = sorted(ckpt_dir.glob("epoch=*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)[:3]
    if epochs:
        def val_loss(p):
            m = re.search(r"val_loss=([0-9]*\.[0-9]+)", p.name)
            return float(m.group(1)) if m else float("inf")
        return min(epochs, key=val_loss), "lowest val_loss among the 3 newest checkpoints"
    if lasts:
        return lasts[0], "WARNING: no best checkpoint found, using the newest last*.ckpt"
    raise SystemExit(f"No checkpoints in {ckpt_dir}")


def ratio(num, den):
    return num / den if den else float("nan")


def tile_scores(tp, fp, fn, tn):
    return dict(
        iou=ratio(tp, tp + fp + fn),
        f1=ratio(2 * tp, 2 * tp + fp + fn),
        precision=ratio(tp, tp + fp),
        recall=ratio(tp, tp + fn),
        accuracy=ratio(tp + tn, tp + fp + fn + tn),
    )


def pooled(df):
    tp, fp, fn, tn = (df[c].sum() for c in ("tp", "fp", "fn", "tn"))
    return tile_scores(tp, fp, fn, tn)


def fold_table(df, label):
    """Per-fold pooled scores and their mean +/- std (population std, as main.py)."""
    rows = [dict(fold=f, tiles=len(g), **pooled(g)) for f, g in df.groupby("fold")]
    t = pd.DataFrame(rows)
    print(f"\n{label}")
    print(t.round(4).to_string(index=False))
    stats = "   ".join(f"{m} {t[m].mean():.4f} +- {t[m].std(ddof=0):.4f}"
                       for m in ("iou", "f1", "precision", "recall"))
    print(f"mean +- std over {len(t)} folds:  {stats}")
    return t


def load_haze_table(root, smoke_csv):
    path = Path(smoke_csv)
    if path.exists():
        return pd.read_csv(path)
    from smoke_score import build_table
    print(f"{smoke_csv} not found; computing smoke scores from the raw tiles")
    table = build_table(Path(root), haze.HOT_SMOKE_THR)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--crossval", required=True, help="crossval run dir holding fold*/checkpoints")
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--root", default="data/indonesia")
    p.add_argument("--smoke-list", default=DEFAULT_LIST)
    p.add_argument("--smoke-csv", default=DEFAULT_SMOKE_CSV)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--panels", default=None, help="directory for one result panel per tile")
    p.add_argument("--worst", type=int, default=10, help="how many worst tiles to list")
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    crossval = Path(args.crossval)
    smoky_ids = _read_id_list(args.smoke_list)
    hz = load_haze_table(args.root, args.smoke_csv).set_index("id")

    panel_fn = None
    if args.panels:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from visualize import _build_panel as panel_fn  # noqa: E402
        Path(args.panels).mkdir(parents=True, exist_ok=True)

    rows = []
    for fold in args.folds:
        ckpt, how = best_checkpoint(crossval / f"fold{fold}")
        print(f"fold {fold}: {ckpt.name}  ({how})")
        model = DeepLabV3PlusMobileViT.load_from_checkpoint(
            ckpt, map_location=device, encoder_weights=None).to(device).eval()
        ds = IndonesiaDataset(root=args.root, folds=[fold], augment=False)
        with torch.no_grad():
            for i in range(len(ds)):
                s = ds[i]
                x = s["post"].float().unsqueeze(0).to(device)
                logits = model(x)
                pred = (logits.argmax(dim=1)[0] > 0).cpu()
                gt = (s["mask"].squeeze() > 0)
                tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum())
                fn = int((~pred & gt).sum()); tn = int((~pred & ~gt).sum())
                tid = s["id"]
                sc = tile_scores(tp, fp, fn, tn)
                hot = float(hz.loc[tid, "hot_median"]) if tid in hz.index else float("nan")
                smoky = tid in smoky_ids
                rows.append(dict(id=tid, scene=scene_key(tid), fold=fold, ckpt=ckpt.name,
                                 gt_px=int(gt.sum()), pred_px=int(pred.sum()),
                                 tp=tp, fp=fp, fn=fn, tn=tn, **sc, smoky=smoky, hot_median=hot))
                if panel_fn is not None:
                    prob = torch.softmax(logits, dim=1)[0, 1].cpu()
                    title = (f"{tid} | fold {fold} | IoU {sc['iou']:.3f}  F1 {sc['f1']:.3f}  "
                             f"P {sc['precision']:.3f}  R {sc['recall']:.3f} | "
                             f"{'SMOKE-COVERED' if smoky else 'not flagged'} (HOT {hot:.3f})")
                    fig = panel_fn(s["post"].float(), gt.long(), pred.long(), prob, title)
                    fig.savefig(Path(args.panels) / f"{tid}.png", dpi=110, bbox_inches="tight")
                    plt.close(fig)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    out_csv = crossval / "per_image_metrics.csv"
    df.to_csv(out_csv, index=False)

    fold_table(df, "ALL test tiles (matches the run's own test metrics up to fp16/fp32 rounding)")
    clean = df[~df.smoky]
    fold_table(clean, f"Smoke-covered tiles REMOVED from evaluation ({int(df.smoky.sum())} tiles dropped)")

    print("\nSmoke-covered vs other tiles (pixels pooled over all folds)")
    for name, g in (("smoke-covered", df[df.smoky]), ("not flagged", clean)):
        if len(g):
            s = pooled(g)
            print(f"  {name:<14} tiles={len(g):>3}  IoU {s['iou']:.4f}  F1 {s['f1']:.4f}  "
                  f"precision {s['precision']:.4f}  recall {s['recall']:.4f}")

    burned = df[df.gt_px > 0].sort_values("iou")
    print(f"\n{args.worst} lowest-IoU tiles that contain burn")
    print(burned.head(args.worst)[["id", "fold", "gt_px", "fn", "fp", "iou", "recall", "smoky", "hot_median"]]
          .round(3).to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ok = df[(df.gt_px > 0) & df.hot_median.notna()]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(ok[~ok.smoky].hot_median, ok[~ok.smoky].iou, s=18, c="#2563eb", label="not flagged")
    ax.scatter(ok[ok.smoky].hot_median, ok[ok.smoky].iou, s=26, c="#dc2626", label="smoke-covered")
    ax.set_xlabel("haze level (median HOT)")
    ax.set_ylabel("tile IoU (tiles with burn)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(crossval / "iou_vs_haze.png", dpi=140)
    print(f"\nwrote {out_csv} and {crossval / 'iou_vs_haze.png'}")
    if args.panels:
        print(f"wrote {len(df)} panels to {args.panels}")


if __name__ == "__main__":
    main()
