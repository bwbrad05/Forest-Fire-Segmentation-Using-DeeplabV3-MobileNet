"""
main.py — Training entry point for DeepLabV3+ MobileViT research.

Research: Segmentasi Area Kebakaran Hutan dan Lahan pada Citra Satelit
          Menggunakan DeepLabV3+ dengan Backbone MobileViT
          untuk Wilayah Tropis Indonesia

Usage
-----
# Train primary model (Experiment C: MobileViT backbone)
python main.py mode=train model=deeplabv3plus_mobilevit_xxs dataset=indonesia

# Train ablation A (ResNet50)
python main.py mode=train model=deeplabv3plus_resnet50 dataset=indonesia

# Train ablation B (MobileNetV3)
python main.py mode=train model=deeplabv3plus_mobilenetv3 dataset=indonesia

# Test from checkpoint
python main.py mode=test model=deeplabv3plus_mobilevit_xxs ckpt_path=<path>

# Run 5-fold cross-validation
python main.py mode=crossval model=deeplabv3plus_mobilevit_xxs dataset=indonesia

# Efficiency comparison table for all four thesis models (params / GFLOPs / latency)
python main.py mode=efficiency_sweep
"""

import csv
import logging
from pathlib import Path

import hydra
import pytorch_lightning as pl
import pytorch_lightning.loggers as loggers
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.tuner import Tuner

import utils

log = logging.getLogger(__name__)


def make_callbacks(dirpath=None):
    """Build a fresh set of callbacks.

    ModelCheckpoint and EarlyStopping hold per-run state (best_model_path,
    patience counters), so each cross-validation fold needs its own instances.

    ``dirpath=None`` keeps Lightning's default location. Pass a directory to
    stop several runs of an ablation sweep from piling their checkpoints into
    one folder, where they collide into last.ckpt, last-v1.ckpt, last-v2.ckpt ...
    and become impossible to attribute.
    """
    return [
        pl.callbacks.ModelCheckpoint(
            dirpath=dirpath,
            monitor="val_loss",
            save_top_k=3,
            mode="min",
            save_last=True,
            filename="{epoch}-{val_loss:.4f}-{val_iou:.4f}",
        ),
        pl.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            mode="min",
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
        pl.callbacks.RichModelSummary(max_depth=3),
        pl.callbacks.RichProgressBar(),
    ]


def default_loggers(run_name=None):
    """Local loggers to use when Comet is unavailable or disabled.

    Without ``run_name`` this returns ``True``, i.e. Lightning's own default —
    unchanged behaviour, runs land in auto-numbered ``lightning_logs/version_N``.

    With ``run_name`` every run gets its own ``lightning_logs/<run_name>/``
    subtree, which is what makes an ablation sweep readable afterwards. Both
    TensorBoard and CSV are attached: TensorBoard for watching a run live, CSV
    because ``scripts/plot_curves.py`` and ``scripts/compare_runs.py`` read
    ``metrics.csv``. TensorBoard is skipped if it is not installed.
    """
    if not run_name:
        return True

    active = [loggers.CSVLogger(save_dir="lightning_logs", name=run_name)]
    try:
        import tensorboard  # noqa: F401
    except ImportError:
        log.warning("tensorboard not installed — logging to CSV only.")
    else:
        active.insert(
            0, loggers.TensorBoardLogger(save_dir="lightning_logs", name=run_name)
        )
    return active


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    # Reproducibility
    pl.seed_everything(47, True)

    mode = cfg.get("mode", "train")

    # ------------------------------------------------------------------ #
    # efficiency_sweep — self-contained: profiles all four thesis models
    # in one command. Handled up front so it needs no Trainer/DataModule
    # (and therefore runs on a CPU-only box regardless of the default GPU
    # trainer). Emits a params / GFLOPs / GMACs / latency comparison table.
    # ------------------------------------------------------------------ #
    if mode == "efficiency_sweep":
        from omegaconf import OmegaConf
        from utils import efficiency_table

        model_dir = Path(__file__).parent / "configs" / "model"
        thesis_models = [
            "deeplabv3plus_mobilevit_xxs",   # Experiment C (primary)
            "deeplabv3plus_mobilevit_xs",    # Experiment C (variant)
            "deeplabv3plus_mobilenetv3",     # Experiment B (ablation)
            "deeplabv3plus_resnet50",        # Experiment A (ablation)
        ]
        entries = []
        for name in thesis_models:
            mcfg = OmegaConf.load(model_dir / f"{name}.yaml")
            model = instantiate(mcfg)
            n_ch = mcfg.get("in_channels", mcfg.get("n_channels", 8))
            entries.append((name, model, n_ch))

        efficiency_table(
            entries,
            csv_path=Path("lightning_logs") / "efficiency_report.csv",
        )
        return

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    pl_model = instantiate(cfg["model"])

    # Log parameter count
    n_params = pl_model.count_parameters() if hasattr(pl_model, "count_parameters") else \
               sum(p.numel() for p in pl_model.parameters() if p.requires_grad)
    log.info(
        "Model: %s | Parameters: %.2f M",
        pl_model.__class__.__name__,
        n_params / 1e6,
    )

    # ------------------------------------------------------------------ #
    # Logger
    # ------------------------------------------------------------------ #
    run_name = cfg.get("run_name", None)
    experiment_name = run_name or (
        f"{pl_model.__class__.__name__}_{cfg.get('model_name', '')}"
    )
    if cfg.get("logger"):
        try:
            logger = loggers.CometLogger(
                **cfg["logger"], experiment_name=experiment_name
            )
        except ValueError as e:
            # Fall back to TensorBoard if Comet API key is missing
            if "API key" in str(e):
                log.warning("Comet API key not found; logging locally instead")
                logger = default_loggers(run_name)
            else:
                raise
    else:
        logger = default_loggers(run_name)

    # Callbacks
    ckpt_dir = str(Path("lightning_logs") / run_name / "checkpoints") if run_name else None
    if run_name:
        log.info("run_name=%s -> logs and checkpoints under lightning_logs/%s/",
                 run_name, run_name)
    callbacks = make_callbacks(ckpt_dir)

    # Trainer
    trainer = pl.Trainer(
        **cfg["trainer"],
        logger=logger,
        callbacks=callbacks,
    )

    # DataModule
    datamodule = instantiate(cfg["dataset"])

    # Execution modes (``mode`` resolved near the top of main())
    if mode == "train":
        trainer.fit(pl_model, datamodule=datamodule)
        # fast_dev_run disables checkpointing, so there is no "best" checkpoint
        # to test against — skip the test phase (pipeline check only).
        if trainer.fast_dev_run:
            log.info("fast_dev_run enabled: pipeline OK, skipping test phase.")
            return
        if isinstance(logger, loggers.CometLogger):
            logger.experiment.log_model(
                "Best model", trainer.checkpoint_callback.best_model_path
            )
        trainer.test(pl_model, datamodule=datamodule, ckpt_path="best")

    elif mode == "test":
        trainer.test(pl_model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    elif mode == "predict":
        trainer.predict(pl_model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    elif mode == "crossval":
        # Run 5-fold cross-validation sequentially
        fold_results = []
        for fold in range(5):
            log.info("=== Cross-validation fold %d/5 ===", fold + 1)
            cfg["dataset"]["test_fold"] = fold
            fold_dm = instantiate(cfg["dataset"])
            fold_model = instantiate(cfg["model"])
            # Same protocol as `mode=train`: monitored checkpointing so that
            # ckpt_path="best" below resolves to the best-val_loss epoch rather
            # than the last one, plus a per-fold CSV log for the loss/IoU curves.
            fold_dir = (
                Path("lightning_logs")
                / f"crossval_{cfg.get('run_name') or cfg.get('model_name', 'model')}"
                / f"fold{fold}"
            )
            fold_trainer = pl.Trainer(
                **cfg["trainer"],
                logger=loggers.CSVLogger(
                    "lightning_logs",
                    name=fold_dir.parent.name,
                    version=f"fold{fold}",
                ),
                # Per-fold checkpoint directory: without it every fold writes
                # into the same folder and collides into last-v1, last-v2, ...
                callbacks=make_callbacks(str(fold_dir / "checkpoints")),
            )
            fold_trainer.fit(fold_model, datamodule=fold_dm)
            results = fold_trainer.test(fold_model, datamodule=fold_dm, ckpt_path="best")
            fold_results.append(results[0] if results else {})
            log.info("Fold %d results: %s", fold, results)

        # Print aggregated results, and persist them so a closed terminal
        # does not lose the outcome of a multi-hour run.
        if fold_results:
            all_keys = sorted(fold_results[0].keys())
            summary = {}
            print("\n===== 5-Fold Cross-Validation Summary =====")
            for key in all_keys:
                vals = [r.get(key, 0) for r in fold_results]
                mean = sum(vals) / len(vals)
                std  = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
                summary[key] = (mean, std)
                print(f"  {key:<35s}: {mean:.4f} ± {std:.4f}")
            print("=" * 47 + "\n")

            out_dir = Path("lightning_logs") / f"crossval_{cfg.get('model_name', 'model')}"
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "summary.csv", "w", newline="", encoding="utf8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["metric"] + [f"fold{i}" for i in range(len(fold_results))]
                                + ["mean", "std"])
                for key in all_keys:
                    mean, std = summary[key]
                    writer.writerow([key] + [r.get(key, "") for r in fold_results]
                                    + [f"{mean:.6f}", f"{std:.6f}"])
            log.info("Cross-validation summary written to %s", out_dir / "summary.csv")

    elif mode == "efficiency":
        from utils import model_efficiency_report
        # DeepLabV3PlusMobileViT stores the band count as ``in_channels`` while the
        # SMP-native ablation baseline (DeepLabV3Plus) stores it as ``n_channels``.
        # Check both so ResNet/MobileNetV3 aren't silently profiled at 6 channels.
        hp = pl_model.hparams
        n_ch = hp.get("in_channels", hp.get("n_channels", 8))
        model_efficiency_report(pl_model, input_shape=(1, n_ch, 512, 512))
        return

    elif mode == "lr_find":
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(pl_model, datamodule=datamodule)
        suggestion = lr_finder.suggestion()
        log.info("Suggested LR: %s", suggestion)

        # Persist the result so a headless/overnight run isn't lost: the
        # suggested LR, the full loss-vs-LR sweep (for the thesis plot), and a
        # rendered PNG of the curve with the suggestion marked.
        name = cfg.get("model_name", "model")
        out_dir = Path("lightning_logs")
        out_dir.mkdir(parents=True, exist_ok=True)

        results = getattr(lr_finder, "results", {}) or {}
        lrs = results.get("lr", [])
        losses = results.get("loss", [])
        with open(out_dir / f"lr_find_{name}.csv", "w", newline="", encoding="utf8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["suggested_lr", suggestion])
            writer.writerow(["lr", "loss"])
            for lr, loss in zip(lrs, losses):
                writer.writerow([lr, loss])

        try:
            import matplotlib
            matplotlib.use("Agg")            # headless-safe
            fig = lr_finder.plot(suggest=True)
            fig.savefig(out_dir / f"lr_find_{name}.png", dpi=120, bbox_inches="tight")
        except Exception as e:                # plotting is best-effort
            log.warning("Could not save lr_find plot: %s", e)

        print(f"\nSuggested LR for {name}: {suggestion}")
        print(f"  sweep: {out_dir / f'lr_find_{name}.csv'}")
        print(f"  plot : {out_dir / f'lr_find_{name}.png'}\n")
        return

    # Log model name tag
    if isinstance(logger, loggers.CometLogger):
        logger.experiment.log_parameter("model", cfg.get("model_name", ""))
        logger.experiment.end()


if __name__ == "__main__":
    main()
