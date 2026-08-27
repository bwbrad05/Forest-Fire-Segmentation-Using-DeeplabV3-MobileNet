# Research: Segmentasi Area Kebakaran Hutan dan Lahan pada Citra Satelit
# Menggunakan DeepLabV3+ dengan Backbone MobileViT untuk Wilayah Tropis Indonesia
#
# CBAM — Convolutional Block Attention Module (Woo et al., ECCV 2018).
#
# Motivation for this thesis: Zhang et al. (Sci. Rep. 2024,
# https://www.nature.com/articles/s41598-024-66060-7) build a lightweight
# DeepLabV3+ for *burned-area identification* on 512x512 tiles and report CBAM as
# by far the largest single contributor in their ablation — MIoU 72.97 -> 80.12
# (+7.15), ahead of transfer learning (+0.63) and deep transitive transfer
# learning (+2.87). That is the same task, backbone class and tile size as here.
#
# Placement differs from theirs by design. They embed CBAM inside the MobileNetV2
# bottlenecks; doing the equivalent inside MobileViT would mean editing timm's
# blocks and would put randomly-initialised layers in the middle of the
# ImageNet-pretrained encoder. Instead CBAM is applied to the two feature maps the
# DeepLabV3+ decoder actually consumes — the stride-16 map entering the ASPP and
# the stride-4 low-level skip. The pretrained encoder stays untouched, and the
# refinement still happens "before the features proceed to the ASPP module and
# decoder" as the paper describes.

from typing import Sequence

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Squeeze features to a per-channel weight from avg- and max-pooled context."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)      # 16-channel maps would floor to 0
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.mlp(x.mean(dim=(2, 3), keepdim=True))
        mx = self.mlp(x.amax(dim=(2, 3), keepdim=True))
        return torch.sigmoid(avg + mx)


class SpatialAttention(nn.Module):
    """Per-pixel weight from the channel-wise avg and max response."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    """Channel attention then spatial attention, each applied multiplicatively."""

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel(x)
        return x * self.spatial(x)


# --------------------------------------------------------------------------- #
# Coordinate Attention
# --------------------------------------------------------------------------- #

class CoordinateAttention(nn.Module):
    """Coordinate Attention (Hou et al., CVPR 2021).

    Adopted here from Lyu et al., *J. Agric. Food Res.* 26 (2026) 102680, section
    2.2.2 and Fig. 3, where CA sits at the output end of the backbone and is
    credited — jointly with strip pooling — with +6.28 mIoU over the DeepLabV3+
    baseline (their Table 4, Models_1 77.58 -> Models_7 83.86).

    CBAM's spatial branch collapses the channel axis and then convolves a 7x7
    window, so its notion of "where" is local. CA instead factorises attention
    into two 1-D, *full-length* profiles — one per image row and one per image
    column — so the weight at pixel (i, j) is informed by the whole of row i and
    the whole of column j. That is the same anisotropic, long-range geometry as
    strip pooling, which is why the two are proposed together: SP widens the
    receptive field along strips, CA then suppresses the background that the
    widened field drags in (paper section 2.3).

    Steps (paper equations 4-7):
        Z_h = mean over W  -> (B, C, H, 1)        direction-sensitive pooling
        Z_v = mean over H  -> (B, C, 1, W)
        A   = sigmoid(f_1x1(concat(Z_h, Z_v)))    joint channel-spatial weights
        Y   = X * A_h * A_v                       dynamic feature weighting

    Parameters
    ----------
    channels : int
        Input/output channel count.
    reduction : int
        Channel compression ratio (the paper's r = 32). The hidden width is
        floored at ``min_hidden`` because the maps this project attends to are
        narrow — MobileViT-XXS carries 64 channels at stride-16 and 16 at
        stride-4, so a bare ``C // 32`` would collapse to 2 and 0 channels.
    min_hidden : int
        Lower bound on the compressed width, following the reference
        implementation's ``max(8, inp // reduction)``.
    """

    def __init__(self, channels: int, reduction: int = 32, min_hidden: int = 8):
        super().__init__()
        hidden = max(min_hidden, channels // reduction)
        self.reduce = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.Hardswish(inplace=True),
        )
        self.attn_h = nn.Conv2d(hidden, channels, 1)
        self.attn_w = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]

        z_h = x.mean(dim=3, keepdim=True)                      # (B, C, H, 1)
        z_w = x.mean(dim=2, keepdim=True).transpose(2, 3)      # (B, C, W, 1)

        # Sharing one 1x1 conv across both directions is what lets the vertical
        # profile inform the horizontal one and vice versa.
        y = self.reduce(torch.cat([z_h, z_w], dim=2))          # (B, hidden, H+W, 1)
        y_h, y_w = torch.split(y, [h, w], dim=2)

        a_h = torch.sigmoid(self.attn_h(y_h))                  # (B, C, H, 1)
        a_w = torch.sigmoid(self.attn_w(y_w.transpose(2, 3)))  # (B, C, 1, W)
        return x * a_h * a_w


def build_attention(name: str, channels: int) -> nn.Module:
    """Factory mapping a config string to an attention block."""
    if name == "cbam":
        return CBAM(channels)
    if name == "ca":
        return CoordinateAttention(channels)
    raise ValueError(f"Unknown attention '{name}'. Choose from 'cbam', 'ca'.")


# --------------------------------------------------------------------------- #
# Encoder / decoder wrappers
# --------------------------------------------------------------------------- #

class AttentionEncoder(nn.Module):
    """Wrap an SMP-style encoder, refining the levels the decoder consumes.

    Works for any encoder following the SMP interface — the MobileViT wrapper and
    SMP's own ResNet / MobileNetV3 encoders alike — so the ablation stays a fair
    comparison across all three backbones.

    ``levels`` indexes the feature list the encoder returns. The DeepLabV3+
    decoder reads ``features[-1]`` for the ASPP and ``features[2]`` for the
    low-level skip, which is what the defaults target.

    Both source papers place their attention module at the backbone output,
    before the ASPP: Zhang et al. (Sci. Rep. 2024) for CBAM, Lyu et al. (JAFR
    2026, Fig. 3) for CA. Neither is embedded inside the backbone here, since
    that would insert randomly-initialised layers into the middle of an
    ImageNet-pretrained encoder and undo the transfer learning restored in
    Part E.1.
    """

    def __init__(
        self,
        encoder: nn.Module,
        block: str = "cbam",
        levels: Sequence[int] = (2, -1),
    ):
        super().__init__()
        self.encoder = encoder
        self.block_name = block
        channels = tuple(encoder.out_channels)
        self.levels = tuple(sorted({level % len(channels) for level in levels}))
        # ModuleDict keys must be strings; index by level so forward stays explicit.
        self.blocks = nn.ModuleDict(
            {str(level): build_attention(block, channels[level]) for level in self.levels}
        )

    # -- SMP encoder interface passthrough --------------------------------- #
    @property
    def out_channels(self):
        return self.encoder.out_channels

    @property
    def output_stride(self):
        return self.encoder.output_stride

    def __getattr__(self, name):
        # nn.Module.__getattr__ only runs for attributes not found normally, so
        # this forwards encoder-specific API (e.g. make_dilated) without
        # shadowing the submodules registered above.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules["encoder"], name)

    def forward(self, x: torch.Tensor):
        features = list(self.encoder(x))
        for level in self.levels:
            features[level] = self.blocks[str(level)](features[level])
        return features


class CBAMEncoder(AttentionEncoder):
    """``AttentionEncoder`` fixed to CBAM.

    Kept as a named class so the Part E.3 write-up, existing configs and any
    CBAM checkpoint keep working — the submodule names (``blocks.<level>.*``)
    are unchanged by the refactor, so those state dicts still load.
    """

    def __init__(
        self,
        encoder: nn.Module,
        levels: Sequence[int] = (2, -1),
        reduction: int = 16,
        kernel_size: int = 7,
    ):
        super().__init__(encoder, block="cbam", levels=levels)
        if (reduction, kernel_size) != (16, 7):
            channels = tuple(encoder.out_channels)
            self.blocks = nn.ModuleDict(
                {
                    str(level): CBAM(channels[level], reduction, kernel_size)
                    for level in self.levels
                }
            )


class AttentionDecoder(nn.Module):
    """Re-weight the decoder's fused features with an attention block.

    This is the wheat paper's "dynamic weight allocation" (section 2.2.5,
    equations 19 and 20): after the low-level skip is concatenated with the ASPP
    output, the fusion is scaled by coordinate-attention weights before the
    segmentation head.

    Deviation, stated plainly: the paper reuses the *same* weight map ``A`` that
    the backbone-output CA produced (``F_out = F_cat * A``). That is not
    literally reproducible in this decoder — ``A`` is generated at stride 16 with
    the encoder's channel count, while ``F_cat`` is at stride 4 with 256
    channels, so it neither broadcasts nor upsamples meaningfully. A second CA
    instance on the fused map delivers the stated intent — coordinate-attention
    weighting of the fused features — without inventing a reshape the paper
    never describes.
    """

    def __init__(self, decoder: nn.Module, channels: int, block: str = "ca"):
        super().__init__()
        self.decoder = decoder
        self.attention = build_attention(block, channels)

    def forward(self, features):
        return self.attention(self.decoder(features))
