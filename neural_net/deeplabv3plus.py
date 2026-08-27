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
from neural_net.enhancements import apply_enhancements


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
        Pretrained weights (e.g. 'imagenet'), or None to train from scratch.
        Valid for multi-band inputs — SMP adapts the first convolution.
    attention : str or None
        'cbam' (Zhang et al., Sci. Rep. 2024) or 'ca' (Coordinate Attention,
        Lyu et al., J. Agric. Food Res. 2026) refines the ASPP input and the
        low-level skip. None leaves the encoder untouched.
    strip_pooling : bool
        Add the strip-pooling branch to the ASPP, 5 -> 6 branches (Lyu et al.,
        section 2.2.3).
    decoder_attention : str or None
        Attention block re-weighting the decoder's fused features.
    aspp_dropout : float or None
        Dropout on the ASPP fusion projection. None keeps SMP's own default of
        0.5 so previously-run ablations stay reproducible.
    bce_weight : float
        Lambda in lambda * WCE + (1 - lambda) * Dice; only used with
        loss_fn='bce_dice'.
    bce_pos_weight : float
        Per-class weight w_c for the WCE term.
    scheduler : str
        'poly' or 'warmup_cosine' (Lyu et al., equation 21).
    """

    def __init__(
        self,
        encoder_name: str = "resnet50",
        n_channels: int = 6,
        n_classes: int = 2,
        learning_rate: float = 1e-4,
        loss_fn: str = "asymmetric_unified_focal",
        encoder_weights: Optional[str] = None,
        tta: bool = False,
        attention: Optional[str] = None,
        strip_pooling: bool = False,
        decoder_attention: Optional[str] = None,
        aspp_dropout: Optional[float] = None,
        bce_weight: float = 0.5,
        bce_pos_weight: float = 10.0,
        scheduler: str = "poly",
        warmup_epochs: Optional[int] = None,
        min_lr_ratio: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # SMP rescales a pretrained first convolution to any ``in_channels``, so
        # the old ``if n_channels == 3`` guard silently discarded ImageNet
        # weights for every multi-band run — i.e. always, here.
        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=n_channels,
            classes=n_classes,
            **({} if aspp_dropout is None else {"decoder_aspp_dropout": aspp_dropout}),
        )

        # Same helper as the MobileViT model, so every switch means the same
        # thing across all three backbones and the ablation stays comparable.
        apply_enhancements(
            self.model, attention, strip_pooling, decoder_attention, aspp_dropout
        )

        if loss_fn == "bce_dice":
            self.loss = BCEDiceLoss(bce_weight=bce_weight, pos_weight=bce_pos_weight)
            self._binary_loss = True
        else:
            self.loss = AsymmetricUnifiedFocalLoss(0.5, 0.6, 0.1)
            self._binary_loss = False

        metric_kwargs = dict(task="binary") if n_classes == 2 else dict(
            task="multiclass", num_classes=n_classes
        )
        # For binary segmentation the Dice coefficient equals the F1 score, which
        # is already reported as ``test_f1`` — so no separate Dice metric is kept.
        # (``tm.Dice`` was also removed in torchmetrics >= 1.8, breaking instantiation.)
        # This matches the metric set of the primary DeepLabV3PlusMobileViT model.
        self.test_metrics = tm.MetricCollection(
            {
                "test_iou": tm.JaccardIndex(**metric_kwargs),
                "test_f1": tm.F1Score(**metric_kwargs),
                "test_precision": tm.Precision(**metric_kwargs),
                "test_recall": tm.Recall(**metric_kwargs),
                "test_accuracy": tm.Accuracy(**metric_kwargs),
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
        scheduler = utils.build_scheduler(
            optimizer,
            name=self.hparams.get("scheduler", "poly"),
            total_epochs=self.trainer.max_epochs if self.trainer else 60,
            warmup_epochs=self.hparams.get("warmup_epochs", None),
            min_lr_ratio=self.hparams.get("min_lr_ratio", 1e-3),
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
        if self.hparams.get("tta", False):
            # 8-way D4 test-time augmentation (see utils.tta_predict).
            preds = utils.tta_predict(self.forward, images).argmax(dim=1)
        else:
            preds = self(images).argmax(dim=1)
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
