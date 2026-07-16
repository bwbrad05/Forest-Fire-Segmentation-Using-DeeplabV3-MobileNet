# Research: Segmentasi Area Kebakaran Hutan dan Lahan pada Citra Satelit
# Menggunakan DeepLabV3+ dengan Backbone MobileViT untuk Wilayah Tropis Indonesia
#
# Efficiency reporting for model comparison (Objective 4: Computational Efficiency)

import time
from typing import Dict, Tuple

import torch
import torch.nn as nn


def model_efficiency_report(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 6, 512, 512),
    n_warmup: int = 5,
    n_runs: int = 20,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Measure and return efficiency metrics for the given model.

    Metrics
    -------
    parameters_M : float
        Number of trainable parameters in millions.
    model_size_MB : float
        Approximate model size in megabytes (float32 weights).
    inference_time_ms : float
        Average per-image inference time in milliseconds.
    throughput_img_per_sec : float
        Images processed per second.

    Parameters
    ----------
    model : nn.Module
        The model to profile (will be set to eval mode).
    input_shape : tuple
        (batch, channels, height, width).
    n_warmup : int
        Number of warm-up passes before timing.
    n_runs : int
        Number of timed passes.
    device : str
        'cpu' or 'cuda'.

    Returns
    -------
    dict
        Dictionary of efficiency metrics.
    """
    model = model.to(device).eval()
    batch_size = input_shape[0]
    dummy = torch.zeros(input_shape, device=device)

    # Warm-up ---------------------------------------------------------------- #
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)

    # Timed runs ------------------------------------------------------------- #
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    total_images  = n_runs * batch_size
    elapsed_sec   = t1 - t0
    per_image_ms  = elapsed_sec / total_images * 1000.0
    throughput    = total_images / elapsed_sec

    # Parameter count -------------------------------------------------------- #
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb  = n_params * 4 / (1024 ** 2)   # float32

    report = {
        "parameters_M":           round(n_params / 1e6, 3),
        "model_size_MB":          round(size_mb, 2),
        "inference_time_ms":      round(per_image_ms, 3),
        "throughput_img_per_sec": round(throughput, 2),
    }

    print("\n===== Model Efficiency Report =====")
    print(f"  Parameters       : {report['parameters_M']:.3f} M")
    print(f"  Model size       : {report['model_size_MB']:.2f} MB")
    print(f"  Inference time   : {report['inference_time_ms']:.3f} ms/image")
    print(f"  Throughput       : {report['throughput_img_per_sec']:.1f} img/s")
    print("===================================\n")

    return report
