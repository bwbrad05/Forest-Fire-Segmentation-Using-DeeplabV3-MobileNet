# Research: Segmentasi Area Kebakaran Hutan dan Lahan pada Citra Satelit
# Menggunakan DeepLabV3+ dengan Backbone MobileViT untuk Wilayah Tropis Indonesia
#
# Strip Pooling (SP) — a sixth ASPP branch.
#
# Origin: Hou et al., "Strip Pooling: Rethinking Spatial Pooling for Scene
# Parsing", CVPR 2020. Brought into DeepLabV3+ by Lyu et al., *Journal of
# Agriculture and Food Research* 26 (2026) 102680, "Enhanced DeepLabV3+ for wheat
# grain segmentation" — sections 2.2.3 and Figs. 4/5. Their ablation (Table 4)
# credits SP together with Coordinate Attention with +6.28 mIoU over the
# DeepLabV3+ baseline (Models_1 77.58 -> Models_7 83.86); that pairing is what
# this module and ``CoordinateAttention`` in ``neural_net.attention`` reproduce.
#
# Why it should transfer to burned-area mapping
# ---------------------------------------------
# The ASPP samples context with square dilated kernels, so any square window
# large enough to span an elongated structure also drags in a large amount of
# unrelated area. Strip pooling instead aggregates a full row and a full column
# independently, giving an anisotropic receptive field that follows long, thin
# structures at negligible cost. The wheat paper wanted this for elongated
# grains; here the same geometry describes burn-scar boundaries, which follow
# ridge lines, rivers and plantation-block edges across a Landsat tile rather
# than forming compact blobs.
#
# What the branch computes (paper Fig. 4)
# ---------------------------------------
#     x -> avgpool to (H, 1) -> 1-D conv along H -.
#     x -> avgpool to (1, W) -> 1-D conv along W -+-> sum -> ReLU
#                                                  -> 1x1 conv -> sigmoid -> * x
#
# i.e. the module is *modulatory*: it returns its input re-weighted by a
# long-range horizontal/vertical context map, not a new feature space.
#
# Two deliberate deviations, both about parameter cost
# ----------------------------------------------------
# 1. **The branch projects first, then strip-pools.** Every ASPP branch has to
#    emit ``out_channels`` (256) before the concatenation. Gating at the raw
#    encoder width and projecting afterwards would size the strip convolutions by
#    the backbone: measured at +30 M parameters on ResNet-50 (2048 channels at
#    stride 16) against +0.05 M on MobileViT-XXS (64 channels). That would both
#    wreck this thesis's edge-deployability claim and make the ablation unfair,
#    since "+SP" would mean a different amount of added capacity per backbone.
#    Projecting first fixes the gate width at 256 for every backbone.
# 2. **The two 1-D convolutions are depthwise**, with the 1x1 fusion convolution
#    doing the cross-channel mixing. This is the same depthwise-separable
#    factorisation the paper applies to its own backbone (their section 2.2.1),
#    and it takes the gate from 458 K parameters to 67 K. Set
#    ``depthwise=False`` for Hou et al.'s original full convolutions.
#
# Net cost of ``strip_pooling=True`` on MobileViT-XXS: 1.449 M -> 1.514 M
# parameters (+4.5 %), of which 65 K is the mandatory widening of the ASPP fusion
# convolution from 5x to 6x branches (the paper's Fig. 5).

from typing import Optional

import torch
import torch.nn as nn


class StripPooling(nn.Module):
    """Long-range horizontal/vertical context gate (Hou et al., CVPR 2020).

    Output shape equals the input shape; only the values are re-weighted.

    Parameters
    ----------
    channels : int
        Channel count of the input feature map (preserved on output).
    kernel_size : int
        Width of the 1-D convolutions applied along the pooled strips.
    depthwise : bool
        Run the 1-D convolutions per channel (default) instead of mixing
        channels, leaving the mixing to the 1x1 fusion convolution. See the
        module docstring.
    """

    def __init__(self, channels: int, kernel_size: int = 3, depthwise: bool = True):
        super().__init__()
        pad = kernel_size // 2
        groups = channels if depthwise else 1

        # Pool over the width axis -> one value per row, then convolve along H.
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.conv_h = nn.Sequential(
            nn.Conv2d(
                channels, channels, (kernel_size, 1),
                padding=(pad, 0), groups=groups, bias=False,
            ),
            nn.BatchNorm2d(channels),
        )

        # Pool over the height axis -> one value per column, then convolve along W.
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv_w = nn.Sequential(
            nn.Conv2d(
                channels, channels, (1, kernel_size),
                padding=(0, pad), groups=groups, bias=False,
            ),
            nn.BatchNorm2d(channels),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]

        # ``expand`` rather than ``interpolate``: the pooled maps are already
        # (H, 1) and (1, W), so broadcasting *is* the paper's "expand" step and
        # allocates nothing.
        ctx_h = self.conv_h(self.pool_h(x)).expand(-1, -1, h, w)
        ctx_w = self.conv_w(self.pool_w(x)).expand(-1, -1, h, w)

        gate = torch.sigmoid(self.fuse(torch.relu(ctx_h + ctx_w)))
        return x * gate


class StripPoolingBranch(nn.Module):
    """ASPP branch: 1x1 projection to the branch width, then the strip gate.

    The five stock ASPP branches all emit ``out_channels``; this one matches them
    so it can be concatenated alongside. See the module docstring for why the
    projection comes first.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        depthwise: bool = True,
    ):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.strip = StripPooling(out_channels, kernel_size, depthwise)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.strip(self.project(x))


# --------------------------------------------------------------------------- #
# Surgery helpers — insert the branch into an already-built SMP ASPP
# --------------------------------------------------------------------------- #

def _first_conv(module: nn.Module) -> nn.Conv2d:
    """Return the first Conv2d found in ``module`` (depth-first)."""
    for sub in module.modules():
        if isinstance(sub, nn.Conv2d):
            return sub
    raise ValueError(f"No Conv2d found in {type(module).__name__}")


def _find_aspp(model: nn.Module) -> nn.Module:
    """Locate the ASPP block inside a built SMP DeepLabV3+ decoder."""
    from segmentation_models_pytorch.decoders.deeplabv3.decoder import ASPP

    for sub in model.modules():
        if isinstance(sub, ASPP):
            return sub
    raise ValueError("No ASPP module found — is this a DeepLabV3/DeepLabV3+ model?")


def add_strip_pooling(
    model: nn.Module, kernel_size: int = 3, depthwise: bool = True
) -> nn.Module:
    """Add a strip-pooling branch to the ASPP of a built DeepLabV3+ model.

    Mirrors Fig. 5 of the wheat paper: the branch count goes 5 -> 6 and the
    fusion 1x1 convolution widens from ``5 * out`` to ``6 * out`` input channels.
    The projection is necessarily re-initialised, which is harmless because the
    DeepLabV3+ decoder is trained from scratch in this project either way (only
    the encoder carries ImageNet weights).

    Works for both model paths here — the SMP-native encoders and the MobileViT
    encoder — because both end up with the same ``DeepLabV3PlusDecoder``.

    Returns the same ``model``, modified in place.
    """
    aspp = _find_aspp(model)

    in_channels = _first_conv(aspp.convs[0]).in_channels
    out_channels = _first_conv(aspp.project).out_channels

    dropout_p = next(
        (m.p for m in aspp.project.modules() if isinstance(m, nn.Dropout)), 0.0
    )

    aspp.convs.append(
        StripPoolingBranch(in_channels, out_channels, kernel_size, depthwise)
    )

    aspp.project = nn.Sequential(
        nn.Conv2d(len(aspp.convs) * out_channels, out_channels, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.Dropout(dropout_p),
    )
    return model


def set_aspp_dropout(model: nn.Module, p: float) -> nn.Module:
    """Set the dropout rate on the ASPP fusion projection.

    The wheat paper uses p = 0.5 (their section 2.2.5). Kept configurable and
    defaulted off here because this project's diagnosis is *under*-fitting, not
    over-fitting — see ``CHANGES_SINCE_MEETING.md`` Part A.2.
    """
    for sub in _find_aspp(model).project.modules():
        if isinstance(sub, nn.Dropout):
            sub.p = p
    return model
