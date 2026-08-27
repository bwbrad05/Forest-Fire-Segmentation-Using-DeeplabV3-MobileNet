# Research: Segmentasi Area Kebakaran Hutan dan Lahan pada Citra Satelit
# Menggunakan DeepLabV3+ dengan Backbone MobileViT untuk Wilayah Tropis Indonesia
#
# MobileViT Backbone
# Wraps Apple's MobileViT (via timm) and exposes multi-level feature maps
# compatible with the segmentation_models_pytorch encoder interface.

from typing import List, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


# Channel configurations for each MobileViT variant
# Keys correspond to timm model names
_MOBILEVIT_CHANNELS = {
    "mobilevit_xxs": {
        # (stride-4, stride-8, stride-16, stride-32)
        "out_channels": (16, 32, 64, 96),
        "final_channels": 320,
    },
    "mobilevit_xs": {
        "out_channels": (32, 48, 96, 160),
        "final_channels": 384,
    },
    "mobilevit_s": {
        "out_channels": (32, 64, 128, 192),
        "final_channels": 640,
    },
}


class MobileViTEncoder(nn.Module):
    """
    MobileViT encoder that produces multi-scale feature maps compatible with
    DeepLabV3PlusDecoder from segmentation_models_pytorch.

    The encoder exposes the following output levels:
      - features[0]: dummy tensor (SMP convention: index 0 = input-scale info)
      - features[1]: stride-2  feature map
      - features[2]: stride-4  feature map  (low-level, used for skip connection)
      - features[3]: stride-8  feature map
      - features[4]: stride-16 feature map  (deepest level, fed to ASPP)

    Verified at 512x512 input: shapes are 256, 128, 64, 32 px, i.e. strides
    2/4/8/16. timm's MobileViT with out_indices=(0,1,2,3) bottoms out at
    stride-16, which is what DeepLabV3+ wants for output_stride=16 -- there is
    no stride-32 level here.

    Parameters
    ----------
    variant : str
        MobileViT size variant. One of 'mobilevit_xxs', 'mobilevit_xs', 'mobilevit_s'.
    in_channels : int
        Number of input channels (e.g. 6 for Landsat-8 RGB+NIR+SWIR).
    output_stride : int
        Target output stride. Only 16 is fully supported by MobileViT out-of-the-box.
    pretrained : bool
        Load ImageNet pretrained weights. Works at any ``in_channels``: timm
        adapts the pretrained RGB stem filters to the requested band count.
    """

    def __init__(
        self,
        variant: str = "mobilevit_xxs",
        in_channels: int = 6,
        output_stride: int = 16,
        pretrained: bool = False,
    ):
        super().__init__()
        if variant not in _MOBILEVIT_CHANNELS:
            raise ValueError(
                f"Unknown MobileViT variant '{variant}'. "
                f"Choose from {list(_MOBILEVIT_CHANNELS.keys())}"
            )
        if output_stride not in (16, 32):
            raise ValueError("MobileViT encoder only supports output_stride 16 or 32.")

        self.variant = variant
        self._in_channels = in_channels
        self._output_stride = output_stride

        # ------------------------------------------------------------------ #
        # Build the timm MobileViT backbone
        # ------------------------------------------------------------------ #
        # timm's MobileViT with out_indices=(0,1,2,3) yields feature maps at
        # reductions (2, 4, 8, 16) — the deepest level is stride-16, which is
        # exactly what DeepLabV3+ needs for output_stride=16. We read the real
        # channel sizes from feature_info instead of hard-coding them, so the
        # decoder is always wired to match the actual encoder output.
        # ``in_chans`` is passed to timm rather than rebuilding the stem
        # afterwards. timm adapts the pretrained RGB stem to any band count by
        # tiling/rescaling the three-channel filters, so ImageNet weights stay
        # usable on 8-band Landsat. The previous form disabled transfer learning
        # twice over: ``pretrained and in_channels == 3`` is always False here,
        # and ``_adapt_first_conv`` then kaiming-reinitialised the stem. With
        # 227 training tiles, training the encoder from scratch is the single
        # biggest thing holding the scores back.
        self.backbone = timm.create_model(
            variant,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),  # reductions: 2, 4, 8, 16
            in_chans=in_channels,
        )

        # SMP convention: index 0 is a dummy input-scale placeholder, followed
        # by the encoder feature levels. The DeepLabV3PlusDecoder uses
        # features[-1] (stride-16) for the ASPP and features[2] (stride-4) for
        # the low-level skip connection.
        self._out_channels = (1,) + tuple(self.backbone.feature_info.channels())

        # No stem surgery needed: timm already built it at ``in_channels`` above,
        # preserving pretrained weights. ``set_in_channels`` remains available
        # for reconfiguring after construction.

    # ---------------------------------------------------------------------- #
    # SMP-compatible interface
    # ---------------------------------------------------------------------- #

    @property
    def out_channels(self) -> Tuple[int, ...]:
        """Channel sizes at each feature level (SMP convention)."""
        return self._out_channels

    @property
    def output_stride(self) -> int:
        """Output stride of the encoder (required by SMP DeepLabV3Plus)."""
        return self._output_stride

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W).

        Returns
        -------
        List[torch.Tensor]
            SMP-style list [dummy, stride2, stride4, stride8, stride16].
            The DeepLabV3PlusDecoder consumes features[-1] (stride-16) for the
            ASPP and features[2] (stride-4) for the low-level skip connection.
        """
        features = self.backbone(x)  # [stride2, stride4, stride8, stride16]

        # Build SMP-style list with a dummy input-scale tensor at index 0.
        dummy = x.new_zeros(x.shape[0], 1, x.shape[2], x.shape[3])
        return [dummy] + list(features)

    # ---------------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------------- #

    def _adapt_first_conv(self, in_channels: int):
        """Replace the stem convolution to accept `in_channels` inputs."""
        # Locate the first Conv2d in the backbone
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
                parent = (
                    self.backbone.get_submodule(parent_name)
                    if parent_name
                    else self.backbone
                )
                # Rebuild with new in_channels, same other params
                new_conv = nn.Conv2d(
                    in_channels,
                    module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    bias=module.bias is not None,
                )
                nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
                setattr(parent, child_name, new_conv)
                return  # Only patch the very first conv

    def set_in_channels(self, in_channels: int):
        """Public API to reconfigure the input channels after construction."""
        if in_channels != self._in_channels:
            self._adapt_first_conv(in_channels)
            self._in_channels = in_channels
