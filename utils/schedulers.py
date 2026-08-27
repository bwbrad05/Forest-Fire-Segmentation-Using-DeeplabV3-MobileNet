# Research: Segmentasi Area Kebakaran Hutan dan Lahan pada Citra Satelit
# Menggunakan DeepLabV3+ dengan Backbone MobileViT untuk Wilayah Tropis Indonesia
#
# Learning-rate schedules.
#
# ``poly`` is the DeepLab default this project has always used. ``warmup_cosine``
# implements equation (21) of Lyu et al., *J. Agric. Food Res.* 26 (2026) 102680
# (section 2.2.5), which their section 4.5 credits with removing the loss/mIoU
# discontinuity their unscheduled run shows at epoch 50.
#
# Relevance here: the Part D run of this project kept its three best checkpoints
# at epochs 73, 82 and 88 of a 100-epoch budget, i.e. validation loss was still
# improving at the end. A warm-up avoids wasting the first epochs on an
# untrained decoder attached to a pretrained encoder, and cosine annealing
# spends the tail refining rather than oscillating.

import math
from typing import Optional

import torch


def warmup_cosine_lambda(
    total_epochs: int,
    warmup_epochs: Optional[int] = None,
    stable_epochs: int = 0,
    min_lr_ratio: float = 1e-3,
):
    """Return the multiplicative factor function for ``LambdaLR``.

    Reproduces the paper's three-stage schedule as a fraction of the initial LR:

        t <= T_warmup                    ->  (t / T_warmup) ** 2
        T_warmup < t <= T_total - T_stable ->  cosine from 1.0 down to min_lr_ratio
        t > T_total - T_stable           ->  min_lr_ratio

    Parameters
    ----------
    total_epochs : int
        Length of the run (the paper's ``T_total``).
    warmup_epochs : int or None
        The paper's ``T_warmup``; defaults to 10 % of ``total_epochs``, as it
        specifies, with a floor of 1.
    stable_epochs : int
        The paper's ``T_stable`` — a constant-LR tail, capped by them at 15
        epochs. Defaults to 0 so the cosine runs to the end unless asked for.
    min_lr_ratio : float
        ``lr_min / lr_init``. The paper uses 5e-6 / 5e-3 = 1e-3.
    """
    total_epochs = max(1, int(total_epochs))
    if warmup_epochs is None:
        warmup_epochs = max(1, int(round(0.1 * total_epochs)))
    warmup_epochs = max(0, min(int(warmup_epochs), total_epochs - 1))
    stable_epochs = max(0, min(int(stable_epochs), total_epochs - warmup_epochs - 1))

    cosine_epochs = max(1, total_epochs - warmup_epochs - stable_epochs)

    def factor(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            # Quadratic warm-up: (t / T_warmup) ** 2, offset by one so the first
            # epoch is not a dead step at lr = 0.
            return ((epoch + 1) / warmup_epochs) ** 2

        progress = (epoch - warmup_epochs) / cosine_epochs
        if progress >= 1.0:
            return min_lr_ratio

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return factor


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    name: str,
    total_epochs: int,
    warmup_epochs: Optional[int] = None,
    stable_epochs: int = 0,
    min_lr_ratio: float = 1e-3,
):
    """Build the requested epoch-interval LR scheduler.

    Parameters
    ----------
    name : str
        ``'poly'`` (DeepLab default, power 0.9) or ``'warmup_cosine'``.
    """
    if name == "poly":
        return torch.optim.lr_scheduler.PolynomialLR(
            optimizer,
            total_iters=total_epochs,
            power=0.9,  # Common in DeepLab training schedules
        )

    if name == "warmup_cosine":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=warmup_cosine_lambda(
                total_epochs, warmup_epochs, stable_epochs, min_lr_ratio
            ),
        )

    raise ValueError(f"Unknown scheduler '{name}'. Choose from 'poly', 'warmup_cosine'.")
