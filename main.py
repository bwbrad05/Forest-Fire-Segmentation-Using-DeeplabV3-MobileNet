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
"""

import logging

import hydra
import pytorch_lightning as pl
import pytorch_lightning.loggers as loggers
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.tuner import Tuner

import utils

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    # Reproducibility
    pl.seed_everything(47, True)

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
    experiment_name = f"{pl_model.__class__.__name__}_{cfg.get('model_name', '')}"
    if cfg.get("logger"):
        try:
            logger = loggers.CometLogger(
                **cfg["logger"], experiment_name=experiment_name
            )
        except ValueError as e:
            # Fall back to TensorBoard if Comet API key is missing
            if "API key" in str(e):
                log.warning("Comet API key not found; using TensorBoard instead")
                logger = True
            else:
                raise
    else:
        logger = True   # default TensorBoard

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #
    callbacks = [
        pl.callbacks.ModelCheckpoint(
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

    # ------------------------------------------------------------------ #
    # Trainer
    # ------------------------------------------------------------------ #
    trainer = pl.Trainer(
        **cfg["trainer"],
        logger=logger,
        callbacks=callbacks,
    )

    # ------------------------------------------------------------------ #
    # DataModule
    # ------------------------------------------------------------------ #
    datamodule = instantiate(cfg["dataset"])

    # ------------------------------------------------------------------ #
    # Execution modes
    # ------------------------------------------------------------------ #
    mode = cfg.get("mode", "train")

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
            fold_trainer = pl.Trainer(**cfg["trainer"], logger=False, enable_progress_bar=True)
            fold_trainer.fit(fold_model, datamodule=fold_dm)
            results = fold_trainer.test(fold_model, datamodule=fold_dm, ckpt_path="best")
            fold_results.append(results[0] if results else {})
            log.info("Fold %d results: %s", fold, results)

        # Print aggregated results
        if fold_results:
            all_keys = fold_results[0].keys()
            print("\n===== 5-Fold Cross-Validation Summary =====")
            for key in sorted(all_keys):
                vals = [r.get(key, 0) for r in fold_results]
                mean = sum(vals) / len(vals)
                std  = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
                print(f"  {key:<35s}: {mean:.4f} ± {std:.4f}")
            print("=" * 47 + "\n")

    elif mode == "efficiency":
        from utils import model_efficiency_report
        n_ch = getattr(pl_model.hparams, "in_channels", 6)
        model_efficiency_report(pl_model, input_shape=(1, n_ch, 512, 512))
        return

    elif mode == "lr_find":
        tuner = Tuner(trainer)
        suggestion = tuner.lr_find(pl_model, datamodule=datamodule).suggestion()
        log.info("Suggested LR: %s", suggestion)
        return

    # Log model name tag
    if isinstance(logger, loggers.CometLogger):
        logger.experiment.log_parameter("model", cfg.get("model_name", ""))
        logger.experiment.end()


if __name__ == "__main__":
    main()
