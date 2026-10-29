"""
compare_smoke_runs.py

Put several cross-validation runs side by side using their per_image_metrics.csv
(written by scripts/per_image_metrics.py). For each run it reports, as mean +/- std
over folds (pixels pooled inside each fold, population std, as main.py does):

  all tiles         every test tile
  clean tiles       the smoke-covered tiles removed from the test set
and the pooled IoU / recall on the smoke-covered tiles alone.

Because every run is scored on the same tiles, runs trained with different data
(with or without smoky tiles, or on the dehazed copy) can be compared directly.

Usage:
  python scripts/compare_smoke_runs.py lightning_logs/crossval_cv_baseline lightning_logs/crossval_cv_nosmoke_train
  python scripts/compare_smoke_runs.py --md <run> <run> ...        # Markdown table for the thesis
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))


def ratio(num, den):
    return num / den if den else float("nan")


def pooled(df):
    tp, fp, fn = (df[c].sum() for c in ("tp", "fp", "fn"))
    return dict(iou=ratio(tp, tp + fp + fn), f1=ratio(2 * tp, 2 * tp + fp + fn),
                precision=ratio(tp, tp + fp), recall=ratio(tp, tp + fn))


def per_fold(df):
    return pd.DataFrame([pooled(g) for _, g in df.groupby("fold")])


def cell(t, m):
    return f"{t[m].mean():.4f} +- {t[m].std(ddof=0):.4f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="crossval run directories")
    p.add_argument("--md", action="store_true", help="print a Markdown table")
    args = p.parse_args()

    rows = []
    for run in args.runs:
        df = pd.read_csv(Path(run) / "per_image_metrics.csv")
        all_t, clean_t = per_fold(df), per_fold(df[~df.smoky])
        smoky = pooled(df[df.smoky]) if df.smoky.any() else dict(iou=float("nan"), recall=float("nan"))
        rows.append(dict(
            run=Path(run).name, folds=df.fold.nunique(), tiles=len(df), clean=int((~df.smoky).sum()),
            all_iou=cell(all_t, "iou"), all_f1=cell(all_t, "f1"),
            clean_iou=cell(clean_t, "iou"), clean_f1=cell(clean_t, "f1"),
            smoky_iou=f"{smoky['iou']:.4f}", smoky_recall=f"{smoky['recall']:.4f}"))
    t = pd.DataFrame(rows)

    if args.md:
        print("| run | tiles | IoU (all tiles) | F1 (all tiles) | IoU (clean tiles) | F1 (clean tiles) | IoU (smoky tiles) | recall (smoky tiles) |")
        print("|---|---|---|---|---|---|---|---|")
        for r in rows:
            print(f"| {r['run']} | {r['tiles']} / {r['clean']} clean | {r['all_iou']} | {r['all_f1']} | "
                  f"{r['clean_iou']} | {r['clean_f1']} | {r['smoky_iou']} | {r['smoky_recall']} |")
    else:
        print(t.to_string(index=False))
    if (t.folds < 5).any():
        print("\nnote: at least one run covers fewer than 5 folds, so its mean/std is not comparable.")


if __name__ == "__main__":
    main()
