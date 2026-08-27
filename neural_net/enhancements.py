# Research: Segmentasi Area Kebakaran Hutan dan Lahan pada Citra Satelit
# Menggunakan DeepLabV3+ dengan Backbone MobileViT untuk Wilayah Tropis Indonesia
#
# One place that applies the optional architecture modules to a built DeepLabV3+.
#
# Both LightningModules in this package (``DeepLabV3PlusMobileViT`` for the
# MobileViT experiments, ``DeepLabV3Plus`` for the ResNet / MobileNetV3
# ablations) call this, so every switch behaves identically across backbones and
# the ablation table stays a fair comparison.
#
# The modules come from two papers on the same architecture family:
#   * CBAM on the encoder outputs — Zhang et al., Sci. Rep. 2024
#     (lightweight DeepLabV3+ for burned-area identification).
#   * Strip pooling in the ASPP, coordinate attention on the encoder output, and
#     coordinate-attention weighting of the fused decoder features — Lyu et al.,
#     J. Agric. Food Res. 26 (2026) 102680 (enhanced DeepLabV3+ for wheat grain
#     segmentation), sections 2.2.2, 2.2.3 and 2.2.5.
#
# Each is a separate flag precisely so it can be switched on alone, which is what
# an ablation table needs.

from typing import Optional

import torch.nn as nn

from neural_net.attention import AttentionDecoder, AttentionEncoder
from neural_net.strip_pooling import add_strip_pooling, set_aspp_dropout


def apply_enhancements(
    model: nn.Module,
    attention: Optional[str] = None,
    strip_pooling: bool = False,
    decoder_attention: Optional[str] = None,
    aspp_dropout: Optional[float] = None,
) -> nn.Module:
    """Attach the requested optional modules to a built SMP DeepLabV3+ model.

    Parameters
    ----------
    model : nn.Module
        A constructed ``smp.DeepLabV3Plus`` (or the MobileViT variant assembled
        from the same decoder), modified in place.
    attention : str or None
        ``'cbam'`` or ``'ca'`` — refines the ASPP input (stride 16) and the
        low-level skip (stride 4) at the encoder output.
    strip_pooling : bool
        Add the strip-pooling branch to the ASPP, taking it from five branches
        to six.
    decoder_attention : str or None
        ``'ca'`` (or ``'cbam'``) — re-weights the decoder's fused features, the
        paper's "dynamic weight allocation".
    aspp_dropout : float or None
        Dropout rate on the ASPP fusion projection. The paper uses 0.5; None
        (the default) leaves whatever the model was built with, because this
        project is underfitting rather than overfitting.

    Returns
    -------
    nn.Module
        The same ``model``.
    """
    if attention:
        model.encoder = AttentionEncoder(model.encoder, block=attention)

    if strip_pooling:
        add_strip_pooling(model)

    if aspp_dropout and aspp_dropout > 0.0:
        set_aspp_dropout(model, aspp_dropout)

    if decoder_attention:
        # The segmentation head reads the decoder output, so its input width is
        # the fused-feature width regardless of backbone or decoder settings.
        fused_channels = model.segmentation_head[0].in_channels
        model.decoder = AttentionDecoder(
            model.decoder, fused_channels, block=decoder_attention
        )

    return model
