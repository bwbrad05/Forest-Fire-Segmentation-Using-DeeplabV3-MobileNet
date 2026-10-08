"""
plot_fold_metrics.py

Render a per-fold comparison of the final test metrics from a cross-validation run.

``plot_curves.py`` shows how each fold trained (loss/IoU vs epoch); this script
answers the other question the thesis needs: how the folds compare on the held-out
test set, and how stable the model is across folds. It reads every
``fold*/metrics.csv`` under a crossval directory, takes each fold's final test_*
value, and draws two panels:

  left   grouped bars of IoU / F1 / precision / recall / accuracy, per fold
  right  mean +/- std across folds, per metric (the number you report)

It also prints the per-fold table and the mean +/- std line to the console.

Usage examples:
  # Auto-detect the crossval directory under lightning_logs/
  python scripts/plot_fold_metrics.py

  # Point at a specific run
  python scripts/plot_fold_metrics.py --crossval lightning_logs/crossval_deeplabv3plus_mobilevit_xxs

Output: ``fold_metrics.png`` inside the crossval directory (override with --out).
"""

import argparse
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")               # headless-safe (same as the other scripts)
import matplotlib.pyplot as plt     # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_curves import read_metrics  # noqa: E402

# Metric key -> (axis label, bar color). Order is the plotting/report order.
METRICS = [
    ("test_iou",       "IoU",       "#2563eb"),
    ("test_f1",        "F1",        "#d97706"),
    ("test_precision", "Precision", "#0d9488"),
    ("test_recall",    "Recall",    "#7c3aed"),
    ("test_accuracy",  "Accuracy",  "#6b7280"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--crossval", default=None,
                   help="Crossval directory holding fold*/metrics.csv. "
                        "Default: the first lightning_logs/crossval* directory found.")
    p.add_argument("--out", default=None,
                   help="Output PNG (default: <crossval>/fold_metrics.png)")
    p.add_argument("--dpi", type=int, default=140)
    p.add_argument("--title", default=None, help="Figure title (default: the run name)")
    return p.parse_args()


def find_crossval():
    root = Path("lightning_logs")
    if not root.exists():
        raise SystemExit("No lightning_logs/ directory. Run from the project root.")
    candidates = sorted(d for d in root.glob("crossval*") if d.is_dir())
    if not candidates:
        raise SystemExit(
            "No lightning_logs/crossval* directory found. "
            "Pass one explicitly with --crossval."
        )
    return candidates[0]


def fold_test_metrics(crossval: Path):
    """Return (fold_names, {metric: [value per fold]}) from fold*/metrics.csv.

    Each fold's final logged value of a test_* metric is used (test runs once at
    the end, so the last write is the result).
    """
    fold_dirs = sorted(d for d in crossval.glob("fold*")
                       if d.is_dir() and (d / "metrics.csv").exists())
    if not fold_dirs:
        raise SystemExit(f"No fold*/metrics.csv under {crossval}")

    names = [d.name for d in fold_dirs]
    values = {key: [] for key, _, _ in METRICS}
    for d in fold_dirs:
        metrics = read_metrics(d / "metrics.csv")
        for key, _, _ in METRICS:
            series = metrics.get(key, [])
            values[key].append(series[-1][1] if series else float("nan"))
    return names, values


def is_finite(x):
    return x == x  # NaN != NaN


def print_table(names, values):
    head = ["fold"] + [label for _, label, _ in METRICS]
    widths = [max(len(head[0]), max(len(n) for n in names))] + [max(len(h), 8) for h in head[1:]]
    def row(cells):
        return "  ".join(str(c).rjust(w) for c, w in zip(cells, widths))
    print(row(head))
    print(row(["-" * w for w in widths]))
    for i, name in enumerate(names):
        cells = [name] + [f"{values[key][i]:.4f}" if is_finite(values[key][i]) else "-"
                          for key, _, _ in METRICS]
        print(row(cells))
    # mean +/- std
    means, stds = {}, {}
    for key, _, _ in METRICS:
        vals = [v for v in values[key] if is_finite(v)]
        means[key] = statistics.fmean(vals) if vals else float("nan")
        stds[key] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    print(row(["-" * w for w in widths]))
    print(row(["mean"] + [f"{means[k]:.4f}" for k, _, _ in METRICS]))
    print(row(["std"] + [f"{stds[k]:.4f}" for k, _, _ in METRICS]))
    return means, stds


def plot(names, values, means, stds, title, out: Path, dpi: int):
    n_folds = len(names)
    n_metrics = len(METRICS)
    fig, (ax_fold, ax_mean) = plt.subplots(
        1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [3, 2]})

    # ---- left: grouped bars, x = fold, one bar per metric ---------------- #
    group_w = 0.8
    bar_w = group_w / n_metrics
    x = range(n_folds)
    for j, (key, label, color) in enumerate(METRICS):
        offsets = [xi - group_w / 2 + bar_w * (j + 0.5) for xi in x]
        heights = [values[key][i] if is_finite(values[key][i]) else 0 for i in range(n_folds)]
        bars = ax_fold.bar(offsets, heights, bar_w, label=label, color=color,
                           edgecolor="none")
        for rect, h in zip(bars, heights):
            if h > 0.02:
                ax_fold.annotate(f"{h:.2f}", (rect.get_x() + rect.get_width() / 2, h),
                                 ha="center", va="bottom", fontsize=7, color=color)
    ax_fold.set_xticks(list(x))
    ax_fold.set_xticklabels([n.replace("fold", "Fold ") for n in names])
    ax_fold.set_ylabel("score")
    ax_fold.set_ylim(0, 1)
    ax_fold.set_title("Test metrics by fold")
    ax_fold.grid(axis="y", alpha=0.3)
    ax_fold.legend(fontsize=8, ncol=n_metrics, loc="upper center",
                   bbox_to_anchor=(0.5, -0.08), frameon=False)

    # ---- right: mean +/- std per metric ---------------------------------- #
    labels = [label for _, label, _ in METRICS]
    colors = [color for _, _, color in METRICS]
    m = [means[key] for key, _, _ in METRICS]
    e = [stds[key] for key, _, _ in METRICS]
    xm = range(n_metrics)
    ax_mean.bar(xm, m, 0.62, yerr=e, color=colors, edgecolor="none",
                capsize=4, error_kw={"elinewidth": 1.2, "ecolor": "#444"})
    for xi, mi, ei in zip(xm, m, e):
        ax_mean.annotate(f"{mi:.3f}\n±{ei:.3f}", (xi, mi + ei),
                         ha="center", va="bottom", fontsize=7)
    ax_mean.set_xticks(list(xm))
    ax_mean.set_xticklabels(labels, rotation=20, ha="right")
    ax_mean.set_ylim(0, 1)
    ax_mean.set_title(f"Mean ± std across {n_folds} folds")
    ax_mean.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    crossval = Path(args.crossval) if args.crossval else find_crossval()
    if not crossval.is_dir():
        raise SystemExit(f"Not a directory: {crossval}")

    names, values = fold_test_metrics(crossval)
    print(f"crossval: {crossval}\n")
    means, stds = print_table(names, values)

    out = Path(args.out) if args.out else crossval / "fold_metrics.png"
    title = args.title or crossval.name
    plot(names, values, means, stds, title, out, args.dpi)
    print(f"\nSaved {out.resolve()}")


if __name__ == "__main__":
    main()
