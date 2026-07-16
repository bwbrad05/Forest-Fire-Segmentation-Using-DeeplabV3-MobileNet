"""
evaluate.py — Evaluation pipeline for DeepLabV3+ MobileViT research.

Usage examples
--------------
# Evaluate primary model (Experiment C: MobileViT backbone)
python evaluate.py ckpt_path=checkpoints/mobilevit_xxs.ckpt model=deeplabv3plus_mobilevit_xxs

# Evaluate ablation A (ResNet backbone)
python evaluate.py ckpt_path=checkpoints/resnet50.ckpt model=deeplabv3plus_resnet50

# Run efficiency report only (no dataset needed)
python evaluate.py mode=efficiency model=deeplabv3plus_mobilevit_xxs

Metrics reported
----------------
  IoU, Dice Score, Precision, Recall, Accuracy, F1
  Parameter count, model size (MB), inference time (ms), throughput
"""

import logging
import time
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

import utils
from lightning_modules import IndonesiaDataModule
from neural_net import DeepLabV3PlusMobileViT, DeepLabV3Plus

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="evaluate")
def main(cfg: DictConfig):
    pl.seed_everything(47, True)

    # ------------------------------------------------------------------ #
    # Build model
    # ------------------------------------------------------------------ #
    if cfg.ckpt_path:
        # Load from checkpoint — auto-detects model class
        ckpt_path = Path(cfg.ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        pl_model = instantiate(cfg["model"])
        log.info("Loading checkpoint: %s", ckpt_path)
    else:
        pl_model = instantiate(cfg["model"])

    # ------------------------------------------------------------------ #
    # Efficiency report (always computed)
    # ------------------------------------------------------------------ #
    n_channels = getattr(pl_model.hparams, "in_channels", 6)
    report = utils.model_efficiency_report(
        pl_model,
        input_shape=(1, n_channels, 512, 512),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    log.info("Efficiency: %s", report)

    if cfg.get("mode") == "efficiency":
        return

    # ------------------------------------------------------------------ #
    # Setup datamodule
    # ------------------------------------------------------------------ #
    datamodule = instantiate(cfg["dataset"])

    # ------------------------------------------------------------------ #
    # Setup trainer
    # ------------------------------------------------------------------ #
    trainer = pl.Trainer(
        **cfg["trainer"],
        logger=False,      # No cloud logging during evaluation by default
        enable_progress_bar=True,
    )

    # ------------------------------------------------------------------ #
    # Run test
    # ------------------------------------------------------------------ #
    ckpt = cfg.get("ckpt_path")
    results = trainer.test(
        pl_model,
        datamodule=datamodule,
        ckpt_path=ckpt if ckpt else None,
    )

    log.info("Test results: %s", results)

    # Print summary table
    if results:
        r = results[0]
        print("\n" + "=" * 55)
        print("  EVALUATION RESULTS")
        print("=" * 55)
        for key, value in sorted(r.items()):
            print(f"  {key:<35s}: {value:.4f}")
        print("-" * 55)
        print(f"  Parameters       : {report['parameters_M']:.3f} M")
        print(f"  Model size       : {report['model_size_MB']:.2f} MB")
        print(f"  Inference time   : {report['inference_time_ms']:.3f} ms/image")
        print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
