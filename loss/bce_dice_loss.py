# BCE + Dice Loss for imbalanced burned-area segmentation
# Burned area pixels are typically <5% of total pixels in Indonesian datasets.
# This compound loss balances pixel-level cross-entropy with region-level overlap.

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    """
    Weighted combination of Binary Cross-Entropy and Dice Loss.

    For a binary segmentation problem (burned vs non-burned) with severe
    class imbalance, this loss penalises both pixel misclassification (BCE)
    and poor region overlap (Dice) simultaneously.

    Parameters
    ----------
    bce_weight : float
        Weight of the BCE term. ``dice_weight = 1 - bce_weight``.
    pos_weight : float or None
        Class weight for the burned class in BCE. When None, no re-weighting
        is applied. A value of ~10–20 is reasonable for Indonesian datasets
        where burned pixels are rare.
    smooth : float
        Laplace smoothing added to Dice numerator and denominator to prevent
        division by zero.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        pos_weight: float = 10.0,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = 1.0 - bce_weight
        self.smooth = smooth

        pw = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : torch.Tensor
            Raw (pre-sigmoid) model output, shape (B, 1, H, W) or (B, H, W).
        targets : torch.Tensor
            Binary ground-truth mask, shape (B, H, W) with values in {0, 1}.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        # Normalise shapes ------------------------------------------------- #
        if logits.dim() == 4 and logits.shape[1] == 1:
            logits = logits.squeeze(1)          # (B, H, W)
        if logits.dim() == 4 and logits.shape[1] == 2:
            # Multi-class head: use burned-class channel
            logits = logits[:, 1]               # (B, H, W)

        targets = targets.float()

        # BCE term --------------------------------------------------------- #
        bce_loss = self.bce(logits, targets)

        # Dice term -------------------------------------------------------- #
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(-1, -2))
        union = probs.sum(dim=(-1, -2)) + targets.sum(dim=(-1, -2))
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score.mean()

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
