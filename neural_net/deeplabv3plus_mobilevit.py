# Research: Segmentasi Area Kebakaran Hutan dan Lahan pada Citra Satelit
# Menggunakan DeepLabV3+ dengan Backbone MobileViT untuk Wilayah Tropis Indonesia
#
# DeepLabV3Plus with pluggable backbone (MobileViT / MobileNetV3 / ResNet).
# Supports binary burned-area segmentation with imbalanced-aware loss functions.

import time
from typing import Dict, Literal, Optional, Tuple

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torchmetrics as tm
from torch.nn import functional as F

import utils
from loss import AsymmetricUnifiedFocalLoss, BCEDiceLoss
from neural_net.enhancements import apply_enhancements
from neural_net.mobilevit_backbone import MobileViTEncoder


# --------------------------------------------------------------------------- #
# Backbone registry
# --------------------------------------------------------------------------- #

def _build_model(
    backbone: str,
    in_channels: int,
    n_classes: int,
    encoder_weights: Optional[str],
    attention: Optional[str] = None,
    strip_pooling: bool = False,
    decoder_attention: Optional[str] = None,
    aspp_dropout: Optional[float] = None,
) -> nn.Module:
    """
    Build a DeepLabV3+ model with the requested backbone.

    Parameters
    ----------
    backbone : str
        One of:
        - 'mobilevit_xxs' / 'mobilevit_xs'   → MobileViT (this research's primary)
        - 'mobilenet_v3_small' / 'mobilenet_v3_large' → MobileNetV3 (ablation B)
        - 'resnet50' / 'resnet101'            → ResNet (ablation A)
    in_channels : int
        Number of input bands.
    n_classes : int
        Number of segmentation classes (2 for binary burned/non-burned).
    encoder_weights : str or None
        Pretrained weights tag (e.g. 'imagenet'), or None to train from scratch.
        Valid at any ``in_channels`` — both timm and SMP adapt the pretrained
        first convolution to the requested band count.
    attention : str or None
        'cbam' or 'ca' wraps the encoder so the ASPP input and the low-level
        skip are refined by that attention block. None leaves the encoder
        untouched.
    strip_pooling : bool
        Add the strip-pooling branch to the ASPP (5 -> 6 branches).
    decoder_attention : str or None
        Attention block re-weighting the decoder's fused features.
    aspp_dropout : float or None
        Dropout on the ASPP fusion projection. None keeps each path's existing
        default (0.0 for MobileViT, SMP's own 0.5 for the ablation encoders) so
        previous runs stay reproducible; set a number to pin both paths to it.

    Returns
    -------
    nn.Module
        A DeepLabV3+ segmentation model.
    """
    mobilevit_variants = ("mobilevit_xxs", "mobilevit_xs", "mobilevit_s")

    if backbone in mobilevit_variants:
        # ------------------------------------------------------------------ #
        # MobileViT path: custom encoder + SMP DeepLabV3+ decoder
        # ------------------------------------------------------------------ #
        encoder = MobileViTEncoder(
            variant=backbone,
            in_channels=in_channels,
            output_stride=16,
            pretrained=(encoder_weights == "imagenet"),
        )
        model = smp.DeepLabV3Plus(
            encoder_name="resnet18",        # placeholder — will be replaced
            encoder_weights=None,
            in_channels=in_channels,
            classes=n_classes,
        )
        # Swap the SMP-installed encoder for our MobileViT encoder
        model.encoder = encoder
        # Rebuild decoder to match MobileViT channel sizes
        from segmentation_models_pytorch.decoders.deeplabv3.decoder import (
            DeepLabV3PlusDecoder,
        )
        from segmentation_models_pytorch.base import SegmentationHead

        model.decoder = DeepLabV3PlusDecoder(
            encoder_channels=encoder.out_channels,
            encoder_depth=len(encoder.out_channels) - 1,
            aspp_separable=False,
            aspp_dropout=0.0 if aspp_dropout is None else aspp_dropout,
            out_channels=256,
            atrous_rates=(6, 12, 18),
            output_stride=16,
        )
        model.segmentation_head = SegmentationHead(
            in_channels=256,
            out_channels=n_classes,
            kernel_size=1,
            activation=None,
            upsampling=4,
        )
        # Applied after the decoder is built: the encoder wrapper preserves
        # out_channels, so the wiring above is unaffected, and the ASPP surgery
        # needs the decoder to exist.
        return apply_enhancements(
            model, attention, strip_pooling, decoder_attention, aspp_dropout
        )

    else:
        # ------------------------------------------------------------------ #
        # SMP-native path: ResNet / MobileNetV3 via segmentation_models_pytorch
        # ------------------------------------------------------------------ #
        # SMP adapts pretrained encoder weights to any ``in_channels`` itself
        # (it rescales the first conv), so the old ``if in_channels == 3`` guard
        # silently forced the ablations to train from scratch as well. Keeping
        # pretraining on for all three backbones is also what makes the ablation
        # a fair comparison.
        model = smp.DeepLabV3Plus(
            encoder_name=backbone,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=n_classes,
            **({} if aspp_dropout is None else {"decoder_aspp_dropout": aspp_dropout}),
        )
        return apply_enhancements(
            model, attention, strip_pooling, decoder_attention, aspp_dropout
        )


# --------------------------------------------------------------------------- #
# Lightning Module
# --------------------------------------------------------------------------- #

class DeepLabV3PlusMobileViT(pl.LightningModule):
    """
    PyTorch Lightning module for DeepLabV3+ with MobileViT backbone.

    Architecture
    ------------
    Input (B, C, H, W)
        → MobileViT Encoder  [multi-scale feature maps]
        → ASPP Module        [atrous spatial pyramid pooling, rates 6/12/18]
        → DeepLabV3+ Decoder [skip connection from stride-4 features]
        → Segmentation Head  [bilinear ×4 upsample → n_classes channels]

    This is the primary model (Experiment C) in the ablation study.
    The same class handles Experiments A (ResNet) and B (MobileNetV3) via
    the ``backbone`` parameter.

    Parameters
    ----------
    backbone : str
        Encoder variant. Default is 'mobilevit_xxs'.
        Experiment A: 'resnet50', Experiment B: 'mobilenet_v3_small'.
    in_channels : int
        Number of satellite bands (6 for Landsat-8).
    n_classes : int
        Number of output classes. Use 2 for binary segmentation.
    learning_rate : float
        Initial learning rate for AdamW optimiser.
    loss_fn : str
        Loss function. One of 'asymmetric_unified_focal' (default) or 'bce_dice'.
    encoder_weights : str or None
        Pretrained weights tag, e.g. 'imagenet'. Valid at any band count.
    bce_pos_weight : float
        Positive class weight for BCE+Dice loss (only used when loss_fn='bce_dice').
        This is the per-class weight w_c of the wheat paper's Weighted
        Cross-Entropy, equation (10) — they set it to N_total / N_c over the whole
        training set. ``scripts/class_weight.py`` computes that value for this
        dataset.
    bce_weight : float
        The wheat paper's lambda in L_total = lambda * L_WCE + (1 - lambda) * L_Dice
        (equation 12). Their grid search over 0.1-0.9 peaks at 0.6 (their Table 5).
        Only used when loss_fn='bce_dice'.
    attention : str or None
        Attention block on the ASPP input and the low-level skip.
        'cbam' — Convolutional Block Attention Module. Zhang et al. (Sci. Rep.
        2024) report it as the largest single gain for lightweight DeepLabV3+
        burned-area mapping (+7.15 MIoU), ahead of transfer learning.
        'ca' — Coordinate Attention. Lyu et al. (J. Agric. Food Res. 2026) pair
        it with strip pooling for +6.28 mIoU over their DeepLabV3+ baseline.
    strip_pooling : bool
        Add the strip-pooling branch to the ASPP, 5 -> 6 branches (Lyu et al.,
        section 2.2.3). Designed for elongated targets; burn-scar boundaries
        follow ridges, rivers and plantation edges rather than forming blobs.
    decoder_attention : str or None
        Attention block re-weighting the decoder's fused features — the paper's
        "dynamic weight allocation" (their equations 19-20).
    aspp_dropout : float or None
        Dropout on the ASPP fusion projection. None keeps the existing default
        (0.0 here). The paper uses 0.5, which is contraindicated while this model
        is still underfitting.
    scheduler : str
        'poly' (DeepLab default, power 0.9) or 'warmup_cosine' (the paper's
        equation 21: quadratic warm-up then cosine annealing).
    warmup_epochs : int or None
        Warm-up length for 'warmup_cosine'. None -> 10 % of max_epochs, as the
        paper specifies.
    min_lr_ratio : float
        lr_min / lr_init for 'warmup_cosine'. The paper's 5e-6 / 5e-3 = 1e-3.
    tta : bool
        If True, apply 8-way (D4) test-time augmentation during ``test``/``predict``:
        each tile is predicted in all flip/rotation orientations and the class
        probabilities are averaged. Adds inference cost (~8×) for a small,
        retraining-free IoU/F1 gain. Ignored for the ``bce_dice`` binary path.
    """

    def __init__(
        self,
        backbone: str = "mobilevit_xxs",
        in_channels: int = 6,
        n_classes: int = 2,
        learning_rate: float = 1e-4,
        loss_fn: Literal["asymmetric_unified_focal", "bce_dice"] = "asymmetric_unified_focal",
        encoder_weights: Optional[str] = None,
        bce_pos_weight: float = 10.0,
        bce_weight: float = 0.5,
        tta: bool = False,
        attention: Optional[str] = None,
        strip_pooling: bool = False,
        decoder_attention: Optional[str] = None,
        aspp_dropout: Optional[float] = None,
        scheduler: Literal["poly", "warmup_cosine"] = "poly",
        warmup_epochs: Optional[int] = None,
        min_lr_ratio: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Model ------------------------------------------------------------ #
        self.model = _build_model(
            backbone,
            in_channels,
            n_classes,
            encoder_weights,
            attention,
            strip_pooling,
            decoder_attention,
            aspp_dropout,
        )

        # Loss ------------------------------------------------------------- #
        if loss_fn == "bce_dice":
            # bce_weight is the paper's lambda; pos_weight is its w_c, so this
            # branch *is* their WCE + Dice joint objective (equations 10-12).
            self.loss = BCEDiceLoss(bce_weight=bce_weight, pos_weight=bce_pos_weight)
            self._binary_loss = True
        else:
            self.loss = AsymmetricUnifiedFocalLoss(weight=0.5, delta=0.6, gamma=0.1)
            self._binary_loss = False

        # Metrics ---------------------------------------------------------- #
        metric_kwargs = dict(task="binary") if n_classes == 2 else dict(
            task="multiclass", num_classes=n_classes
        )

        self.train_metrics = tm.MetricCollection(
            {
                "train_iou": tm.JaccardIndex(**metric_kwargs),
            }
        )
        self.val_metrics = tm.MetricCollection(
            {
                "val_iou": tm.JaccardIndex(**metric_kwargs),
                "val_f1": tm.F1Score(**metric_kwargs),
            }
        )
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
        self._inference_times: list = []

    # ---------------------------------------------------------------------- #
    # Forward
    # ---------------------------------------------------------------------- #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run a full forward pass through DeepLabV3+ with MobileViT encoder.

        Parameters
        ----------
        x : torch.Tensor  shape (B, C, H, W)

        Returns
        -------
        torch.Tensor  shape (B, n_classes, H, W)  — raw logits
        """
        return self.model(x)

    # ---------------------------------------------------------------------- #
    # Training / Validation / Test steps
    # ---------------------------------------------------------------------- #

    def _unpack_batch(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract image and mask tensors from a data-loader batch dict."""
        images = batch["post"].float()
        masks = batch["mask"].float().squeeze(1)          # (B, H, W)
        return images, masks

    def _compute_loss(
        self, logits: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        if self._binary_loss:
            # BCEDiceLoss handles both (B,1,H,W) and (B,2,H,W) logits
            return self.loss(logits, masks)
        else:
            # AsymmetricUnifiedFocalLoss expects softmax probabilities
            probs = torch.softmax(logits, dim=1)
            return self.loss(probs, masks.long())

    def training_step(self, batch, batch_idx):
        images, masks = self._unpack_batch(batch)
        logits = self(images)
        loss = self._compute_loss(logits, masks)

        if torch.isnan(loss):
            return None

        preds = logits.argmax(dim=1) if not self._binary_loss else (logits.squeeze(1) > 0).long()
        self.train_metrics.update(preds, masks.long())

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute(), prog_bar=False)
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        images, masks = self._unpack_batch(batch)
        logits = self(images)
        loss = self._compute_loss(logits, masks)

        preds = logits.argmax(dim=1) if not self._binary_loss else (logits.squeeze(1) > 0).long()
        self.val_metrics.update(preds, masks.long())

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), prog_bar=True)
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        images, masks = self._unpack_batch(batch)

        # Measure inference time for efficiency evaluation
        start = time.perf_counter()
        if self.hparams.get("tta", False) and not self._binary_loss:
            # 8-way D4 test-time augmentation: average class probabilities over
            # all flip/rotation orientations, then take the hard label.
            preds = utils.tta_predict(self.forward, images).argmax(dim=1)
        else:
            logits = self(images)
            preds = logits.argmax(dim=1) if not self._binary_loss else (logits.squeeze(1) > 0).long()
        elapsed = time.perf_counter() - start
        self._inference_times.append(elapsed / images.shape[0])

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
        metrics = self.test_metrics.compute()
        self.log_dict(metrics)
        self.test_metrics.reset()

        # Log efficiency metrics
        if self._inference_times:
            avg_ms = (sum(self._inference_times) / len(self._inference_times)) * 1000
            self.log("test_inference_time_ms_per_image", avg_ms)
            self._inference_times.clear()

    # ---------------------------------------------------------------------- #
    # Optimiser & Scheduler
    # ---------------------------------------------------------------------- #

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=1e-4,
        )
        total_epochs = self.trainer.max_epochs if self.trainer else 60
        scheduler = utils.build_scheduler(
            optimizer,
            name=self.hparams.get("scheduler", "poly"),
            total_epochs=total_epochs,
            warmup_epochs=self.hparams.get("warmup_epochs", None),
            min_lr_ratio=self.hparams.get("min_lr_ratio", 1e-3),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    # ---------------------------------------------------------------------- #
    # Efficiency introspection
    # ---------------------------------------------------------------------- #

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_size_mb(self) -> float:
        """Return approximate model size in megabytes (float32)."""
        return self.count_parameters() * 4 / (1024 ** 2)
