"""
compare_runs.py

Turn a finished ablation sweep into one table and one figure.

``scripts/run_ablation.ps1`` gives every variant its own ``run_name``, so each
lands in ``lightning_logs/<run_name>/version_N/metrics.csv``. This script reads
all of them, ranks them, and reports each variant's delta against the baseline —
which is the form the ablation table in the thesis needs.

Usage examples:
  # Every run directory under lightning_logs/ that has a metrics.csv
  python scripts/compare_runs.py

  # Only the ablation runs, with an explicit baseline for the delta column
  python scripts/compare_runs.py --runs lightning_logs/abl_* --baseline abl_baseline

  # Cross-validation folds of one configuration (mean +/- std instead of deltas)
  python scripts/compare_runs.py --runs lightning_logs/crossval_* --folds

Outputs (into --out_dir, default lightning_logs/comparison/):
  ablation_summary.csv   one row per run, all metrics
  ablation_summary.md    the same table, ready to paste into the thesis
  ablation_curves.png    val_iou and val_loss vs epoch, one line per run
"""

import argparse
import csv
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")               # headless-safe
import matplotlib.pyplot as plt     # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_curves import read_metrics  # noqa: E402

# Reported in this order; the first is the ranking key.
TEST_METRICS = [
    "test_iou",
    "test_f1",
    "test_precision",
    "test_recall",
    "test_accuracy",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="*", default=None,
                   help="Run directories or globs. Default: every lightning_logs/* "
                        "subtree containing a metrics.csv.")
    p.add_argument("--baseline", default=None,
                   help="Run name used as the reference for the delta column. "
                        "Default: a run whose name ends in 'baseline', else the first.")
    p.add_argument("--folds", action="store_true",
                   help="Treat each run directory as a crossval parent and aggregate "
                        "its fold*/metrics.csv as mean +/- std.")
    p.add_argument("--out_dir", default="lightning_logs/comparison")
    p.add_argument("--dpi", type=int, default=140)
    return p.parse_args()


def find_metrics_csv(run_dir: Path):
    """Return the newest metrics.csv under a run directory, or None.

    Handles both ``<run>/metrics.csv`` and ``<run>/version_N/metrics.csv``; when
    a run was restarted, the highest version number wins.
    """
    direct = run_dir / "metrics.csv"
    if direct.exists():
        return direct
    candidates = sorted(run_dir.glob("*/metrics.csv"), key=lambda p: p.parent.name)
    return candidates[-1] if candidates else None


def discover_runs(patterns):
    """Expand globs / paths into run directories that actually hold metrics."""
    if patterns:
        dirs = []
        for pattern in patterns:
            path = Path(pattern)
            if path.is_dir():
                dirs.append(path)
            else:
                # Shells on Windows do not expand globs for us.
                dirs.extend(sorted(Path(path.parent).glob(path.name)))
    else:
        root = Path("lightning_logs")
        dirs = sorted(d for d in root.iterdir() if d.is_dir()) if root.exists() else []

    return [d for d in dirs if d.is_dir() and d.name != "comparison"]


def summarise_run(run_dir: Path):
    """Extract one row of results from a single run directory."""
    csv_path = find_metrics_csv(run_dir)
    if csv_path is None:
        return None
    metrics = read_metrics(csv_path)

    row = {"run": run_dir.name}

    val_loss = metrics.get("val_loss", [])
    val_iou = dict(metrics.get("val_iou", []))
    train_iou = dict(metrics.get("train_iou", []))

    if val_loss:
        best_epoch, best_loss = min(val_loss, key=lambda t: t[1])
        row["best_epoch"] = best_epoch
        row["best_val_loss"] = best_loss
        row["val_iou_at_best"] = val_iou.get(best_epoch)
        if best_epoch in train_iou and best_epoch in val_iou:
            row["train_val_iou_gap"] = train_iou[best_epoch] - val_iou[best_epoch]
        row["epochs_run"] = max(e for e, _ in val_loss) + 1

    for key in TEST_METRICS:
        series = metrics.get(key, [])
        if series:
            row[key] = series[-1][1]

    row["_curves"] = metrics
    return row


def summarise_folds(run_dir: Path):
    """Aggregate fold*/metrics.csv under one crossval directory."""
    fold_dirs = sorted(d for d in run_dir.glob("fold*") if d.is_dir())
    rows = [r for r in (summarise_run(d) for d in fold_dirs) if r]
    if not rows:
        return None

    agg = {"run": run_dir.name, "n_folds": len(rows)}
    for key in TEST_METRICS + ["best_val_loss", "val_iou_at_best"]:
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            continue
        agg[key] = statistics.fmean(vals)
        agg[key + "_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    # Show the first fold's curves so the figure still has something to draw.
    agg["_curves"] = rows[0]["_curves"]
    return agg


def pick_baseline(rows, requested):
    if requested:
        for r in rows:
            if r["run"] == requested:
                return r
        print(f"WARNING: baseline '{requested}' not found; using the first run.")
    for r in rows:
        if r["run"].endswith("baseline"):
            return r
    return rows[0] if rows else None


def fmt(value, digits=4):
    return "-" if value is None else f"{value:.{digits}f}"


def write_table(rows, baseline, out_dir: Path, folds: bool):
    columns = ["run"]
    if folds:
        columns.append("n_folds")
    else:
        columns += ["epochs_run", "best_epoch"]
    columns += TEST_METRICS + ["best_val_loss", "val_iou_at_best"]
    if not folds:
        columns.append("train_val_iou_gap")

    # --- CSV ------------------------------------------------------------- #
    csv_path = out_dir / "ablation_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf8") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns + (["delta_test_iou"] if baseline else []))
        for r in rows:
            line = [r.get(c, "") for c in columns]
            if baseline:
                if r.get("test_iou") is not None and baseline.get("test_iou") is not None:
                    line.append(round(r["test_iou"] - baseline["test_iou"], 6))
                else:
                    line.append("")
            writer.writerow(line)

    # --- Markdown -------------------------------------------------------- #
    key = "test_iou"
    header = ["run", "test_iou", "delta", "test_f1", "precision", "recall", "best_val_loss"]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        delta = "-"
        if baseline and r.get(key) is not None and baseline.get(key) is not None:
            d = r[key] - baseline[key]
            delta = "baseline" if r is baseline else f"{d:+.4f}"
        lines.append("| " + " | ".join([
            r["run"],
            fmt(r.get("test_iou")),
            delta,
            fmt(r.get("test_f1")),
            fmt(r.get("test_precision")),
            fmt(r.get("test_recall")),
            fmt(r.get("best_val_loss")),
        ]) + " |")

    md_path = out_dir / "ablation_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf8")

    print("\n".join(lines))
    print(f"\n  table : {csv_path}")
    print(f"  table : {md_path}")


def plot_curves(rows, out_dir: Path, dpi: int):
    fig, (ax_iou, ax_loss) = plt.subplots(1, 2, figsize=(13, 5))

    drew = False
    for r in rows:
        curves = r.get("_curves", {})
        for ax, key in ((ax_iou, "val_iou"), (ax_loss, "val_loss")):
            series = curves.get(key)
            if not series:
                continue
            ax.plot([e for e, _ in series], [v for _, v in series], label=r["run"], lw=1.4)
            drew = True

    if not drew:
        plt.close(fig)
        print("  (no val curves found — nothing to plot)")
        return

    ax_iou.set_title("Validation IoU")
    ax_iou.set_xlabel("epoch")
    ax_iou.set_ylabel("val_iou")
    ax_loss.set_title("Validation loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("val_loss")
    for ax in (ax_iou, ax_loss):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Ablation comparison", fontsize=13)
    fig.tight_layout()
    out_path = out_dir / "ablation_curves.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure: {out_path}")


def main():
    args = parse_args()
    run_dirs = discover_runs(args.runs)
    if not run_dirs:
        raise SystemExit("No run directories found. Train something first.")

    collect = summarise_folds if args.folds else summarise_run
    rows = []
    for d in run_dirs:
        row = collect(d)
        if row is None:
            continue
        rows.append(row)

    if not rows:
        raise SystemExit(
            "Found run directories but no metrics.csv in them.\n"
            "Runs logged to Comet or TensorBoard only will not appear here — "
            "re-run with '~logger' so the CSV logger is attached."
        )

    # Rank by test IoU when available; runs still training sort last.
    rows.sort(key=lambda r: (r.get("test_iou") is None, -(r.get("test_iou") or 0)))

    baseline = None if args.folds else pick_baseline(rows, args.baseline)
    if baseline:
        print(f"baseline: {baseline['run']}\n")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_table(rows, baseline, out_dir, args.folds)
    plot_curves(rows, out_dir, args.dpi)


if __name__ == "__main__":
    main()
