"""
plot_curves.py

Render training curves (loss and IoU vs. epoch) from a Lightning CSVLogger run.

TensorBoard is not installed in this environment, so PyTorch Lightning falls back
to the CSV logger and every run writes ``metrics.csv`` instead of event files.
This script turns that CSV into the loss/IoU figures the thesis needs, and prints
the best epoch so overfit-vs-underfit can be read off directly.

Usage examples:
  # Single run
  python scripts/plot_curves.py --run lightning_logs/version_6

  # A cross-validation fold
  python scripts/plot_curves.py --run lightning_logs/crossval_deeplabv3plus_mobilevit_xxs/fold0

  # All five folds on one pair of axes
  python scripts/plot_curves.py --run lightning_logs/crossval_deeplabv3plus_mobilevit_xxs --folds

Output: ``curves.png`` inside the run directory (override with --out).
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")               # headless-safe
import matplotlib.pyplot as plt     # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True,
                   help="Run directory containing metrics.csv, or a crossval parent with --folds")
    p.add_argument("--folds", action="store_true",
                   help="Treat --run as a crossval parent and overlay every fold*/metrics.csv")
    p.add_argument("--out", default=None, help="Output PNG (default: <run>/curves.png)")
    p.add_argument("--dpi", type=int, default=130)
    return p.parse_args()


def read_metrics(csv_path: Path):
    """Return {metric: [(epoch, value), ...]} from a Lightning metrics.csv.

    Lightning writes one row per logging event, so most cells are blank and a
    metric's epoch appears across several rows. Values are collapsed per epoch
    (last write wins) to give one point per epoch per metric.
    """
    per_epoch = defaultdict(dict)
    with open(csv_path, newline="", encoding="utf8") as fh:
        for row in csv.DictReader(fh):
            epoch = row.get("epoch", "")
            if epoch in ("", None):
                continue
            epoch = int(float(epoch))
            for key, value in row.items():
                if key in ("epoch", "step") or value in ("", None):
                    continue
                try:
                    per_epoch[key][epoch] = float(value)
                except ValueError:
                    continue
    return {k: sorted(v.items()) for k, v in per_epoch.items()}


def plot_series(ax, metrics, keys, label_prefix=""):
    plotted = False
    for key, style in keys:
        series = metrics.get(key)
        if not series:
            continue
        xs = [e for e, _ in series]
        ys = [v for _, v in series]
        ax.plot(xs, ys, style, label=f"{label_prefix}{key}")
        plotted = True
    return plotted


def summarise(name, metrics):
    """Print best val_loss / val_iou and the train-vs-val gap at that epoch."""
    val_loss = metrics.get("val_loss", [])
    val_iou = metrics.get("val_iou", [])
    train_iou = dict(metrics.get("train_iou", []))
    if not val_loss:
        print(f"{name}: no val_loss logged")
        return
    best_epoch, best_loss = min(val_loss, key=lambda t: t[1])
    iou_at_best = dict(val_iou).get(best_epoch)
    line = f"{name}: best val_loss {best_loss:.4f} @ epoch {best_epoch}"
    if iou_at_best is not None:
        line += f" | val_iou {iou_at_best:.4f}"
    if best_epoch in train_iou and iou_at_best is not None:
        gap = train_iou[best_epoch] - iou_at_best
        verdict = "overfitting" if gap > 0.05 else "underfitting" if gap < -0.02 else "balanced"
        line += f" | train_iou {train_iou[best_epoch]:.4f} (gap {gap:+.4f} -> {verdict})"
    print(line)


def main():
    args = parse_args()
    run = Path(args.run)

    if args.folds:
        runs = sorted(d for d in run.glob("fold*") if (d / "metrics.csv").exists())
        if not runs:
            raise SystemExit(f"No fold*/metrics.csv under {run}")
    else:
        if not (run / "metrics.csv").exists():
            raise SystemExit(f"No metrics.csv in {run}")
        runs = [run]

    fig, (ax_loss, ax_iou) = plt.subplots(ncols=2, figsize=(12, 4.5))

    for d in runs:
        metrics = read_metrics(d / "metrics.csv")
        prefix = f"{d.name} " if args.folds else ""
        plot_series(ax_loss, metrics,
                    [("train_loss_epoch", "-"), ("val_loss", "--")], prefix)
        plot_series(ax_iou, metrics,
                    [("train_iou", "-"), ("val_iou", "--")], prefix)
        summarise(d.name, metrics)

    ax_loss.set_title("Loss"); ax_iou.set_title("IoU")
    for ax in (ax_loss, ax_iou):
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(run.name)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = Path(args.out) if args.out else run / "curves.png"
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"\nSaved {out.resolve()}")


if __name__ == "__main__":
    main()
