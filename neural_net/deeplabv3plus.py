# Research: Segmentasi Area Kebakaran Hutan dan Lahan pada Citra Satelit
# Menggunakan DeepLabV3+ dengan Backbone MobileViT untuk Wilayah Tropis Indonesia
#
# DeepLabV3Plus — Ablation baseline model.
# Used for Experiment A (ResNet backbone) and Experiment B (MobileNetV3 backbone).
# The primary experiment (C) uses DeepLabV3PlusMobileViT instead.

from typing import Optional

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torchmetrics as tm

import utils
from loss import AsymmetricUnifiedFocalLoss, BCEDiceLoss


class DeepLabV3Plus(pl.LightningModule):
    """
    Baseline DeepLabV3+ using SMP-native encoders (ResNet, MobileNetV3).

    Used for the ablation study:
        Experiment A → encoder_name='resnet50'
        Experiment B → encoder_name='timm-mobilenetv3_small_100'

    Parameters
    ----------
    encoder_name : str
        Any encoder supported by segmentation_models_pytorch.
    n_channels : int
        Number of input satellite bands.
    n_classes : int
        Number of output classes (2 for binary burned/non-burned).
    learning_rate : float
        AdamW learning rate.
    loss_fn : str
        'asymmetric_unified_focal' or 'bce_dice'.
    encoder_weights : str or None
        Pretrained weights (e.g. 'imagenet'). Set None for multi-band inputs.
    """

    def __init__(
        self,
        encoder_name: str = "resnet50",
        n_channels: int = 6,
        n_classes: int = 2,
        learning_rate: float = 1e-4,
        loss_fn: str = "asymmetric_unified_focal",
        encoder_weights: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights if n_channels == 3 else None,
            in_channels=n_channels,
            classes=n_classes,
        )

        if loss_fn == "bce_dice":
            self.loss = BCEDiceLoss(bce_weight=0.5, pos_weight=10.0)
            self._binary_loss = True
        else:
            self.loss = AsymmetricUnifiedFocalLoss(0.5, 0.6, 0.1)
            self._binary_loss = False

        metric_kwargs = dict(task="binary") if n_classes == 2 else dict(
            task="multiclass", num_classes=n_classes
        )
        self.test_metrics = tm.MetricCollection(
            {
                "test_iou": tm.JaccardIndex(**metric_kwargs),
                "test_dice": tm.Dice(**metric_kwargs) if n_classes == 2 else tm.Dice(
                    num_classes=n_classes, average="macro"
                ),
                "test_precision": tm.Precision(**metric_kwargs),
                "test_recall": tm.Recall(**metric_kwargs),
                "test_accuracy": tm.Accuracy(**metric_kwargs),
                "test_f1": tm.F1Score(**metric_kwargs),
            }
        )
        self.batch_to_log = [0, 5]

    def forward(self, x):
        return self.model(x)

    def _compute_loss(self, logits, masks):
        if self._binary_loss:
            return self.loss(logits, masks)
        probs = torch.softmax(logits, dim=1)
        return self.loss(probs, masks.long())

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer,
            total_iters=self.trainer.max_epochs if self.trainer else 60,
            power=0.9,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    def training_step(self, batch, batch_idx):
        masks, images = batch["mask"].float(), batch["post"].float()
        masks = masks.squeeze(1)
        logits = self(images)
        loss = self._compute_loss(logits, masks)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        masks, images = batch["mask"].float(), batch["post"].float()
        masks = masks.squeeze(1)
        logits = self(images)
        loss = self._compute_loss(logits, masks)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        masks, images = batch["mask"].float(), batch["post"].float()
        masks = masks.squeeze(1)
        logits = self(images)
        preds = logits.argmax(dim=1)
        self.test_metrics.update(preds, masks.long())

        if batch_idx in self.batch_to_log and self.logger is not None:
            for i, figure in enumerate(utils.draw_figure(masks, preds)):
                if hasattr(self.logger.experiment, "log_figure"):
                    self.logger.experiment.log_figure(
                        figure=figure,
                        figure_name=f"testS{self.global_step}N{i}",
                        step=self.global_step,
                    )

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()
