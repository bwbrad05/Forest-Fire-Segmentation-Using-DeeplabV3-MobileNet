"""
fold_tile_errors.py

Break a fold's test errors down by tile and by scene, to see WHERE the model
misses burned pixels (false negatives) or invents them (false positives).

For each test tile it reports burned pixels in the ground truth, TP / FP / FN,
and tile IoU; scenes are then aggregated and ranked by how much they contribute
to the fold's total false negatives and false positives. Pooled precision /
recall / IoU are printed first so they can be checked against the fold's
reported test metrics.

Usage (from the project root, in the training env):
  python scripts/fold_tile_errors.py --ckpt "lightning_logs/crossval_cv_baseline/fold3/checkpoints/epoch=93-val_loss=0.0820-val_iou=0.6926.ckpt" --fold 3
  python scripts/fold_tile_errors.py --ckpt <ckpt> --fold 3 --tta        # with 8-way TTA
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import utils  # noqa: E402
from lightning_modules import IndonesiaDataset  # noqa: E402
from lightning_modules.scene_splits import scene_key  # noqa: E402
from neural_net import DeepLabV3PlusMobileViT  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--fold", type=int, required=True, help="the fold used as TEST set")
    p.add_argument("--root", default="data/indonesia")
    p.add_argument("--tta", action="store_true")
    p.add_argument("--top", type=int, default=12, help="rows to print in each ranked table")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DeepLabV3PlusMobileViT.load_from_checkpoint(
        args.ckpt, map_location=device, encoder_weights=None
    ).to(device).eval()

    ds = IndonesiaDataset(root=args.root, folds=[args.fold], augment=False)

    rows = []
    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            x = s["post"].float().unsqueeze(0).to(device)
            gt = (s["mask"].squeeze() > 0)
            if args.tta:
                pred = utils.tta_predict(model.forward, x).argmax(dim=1)[0]
            else:
                pred = model(x).argmax(dim=1)[0]
            pred = (pred > 0).cpu()
            tp = int((pred & gt).sum())
            fp = int((pred & ~gt).sum())
            fn = int((~pred & gt).sum())
            rows.append(dict(id=s["id"], scene=scene_key(s["id"]),
                             gt=int(gt.sum()), tp=tp, fp=fp, fn=fn))

    def ratios(tp, fp, fn):
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        iou = tp / (tp + fp + fn) if tp + fp + fn else float("nan")
        return prec, rec, iou

    tot = {k: sum(r[k] for r in rows) for k in ("gt", "tp", "fp", "fn")}
    prec, rec, iou = ratios(tot["tp"], tot["fp"], tot["fn"])
    print(f"\nfold {args.fold}  tiles={len(rows)}  TTA={'on' if args.tta else 'off'}")
    print(f"pooled  precision={prec:.4f}  recall={rec:.4f}  IoU={iou:.4f}"
          "   (compare with the fold's test metrics; small AMP differences are normal)")
    print(f"pixels  GT burned={tot['gt']:,}  FN={tot['fn']:,}  FP={tot['fp']:,}\n")

    scenes = defaultdict(lambda: dict(tiles=0, gt=0, tp=0, fp=0, fn=0))
    for r in rows:
        a = scenes[r["scene"]]
        a["tiles"] += 1
        for k in ("gt", "tp", "fp", "fn"):
            a[k] += r[k]

    def share(v, total):
        return 100 * v / total if total else 0.0

    print("Scenes ranked by share of the fold's missed burned pixels (FN)")
    print(f"{'scene':<20}{'tiles':>6}{'GT px':>10}{'recall':>8}{'FN share':>10}{'FP share':>10}")
    for name, a in sorted(scenes.items(), key=lambda kv: -kv[1]["fn"])[: args.top]:
        _, r_, _ = ratios(a["tp"], a["fp"], a["fn"])
        print(f"{name:<20}{a['tiles']:>6}{a['gt']:>10,}{r_:>8.3f}"
              f"{share(a['fn'], tot['fn']):>9.1f}%{share(a['fp'], tot['fp']):>9.1f}%")

    print("\nTiles ranked by false positives (burn predicted where none is labelled)")
    print(f"{'tile':<26}{'GT px':>9}{'FP':>9}{'FN':>9}")
    for r in sorted(rows, key=lambda r: -r["fp"])[: args.top]:
        print(f"{r['id']:<26}{r['gt']:>9,}{r['fp']:>9,}{r['fn']:>9,}")

    print("\nTiles ranked by false negatives (burn missed)")
    print(f"{'tile':<26}{'GT px':>9}{'FN':>9}{'tile IoU':>10}")
    for r in sorted(rows, key=lambda r: -r["fn"])[: args.top]:
        _, _, t_iou = ratios(r["tp"], r["fp"], r["fn"])
        print(f"{r['id']:<26}{r['gt']:>9,}{r['fn']:>9,}{t_iou:>10.3f}")
    print()


if __name__ == "__main__":
    main()
