# Research: Segmentasi Area Kebakaran Hutan dan Lahan pada Citra Satelit
# Menggunakan DeepLabV3+ dengan Backbone MobileViT untuk Wilayah Tropis Indonesia
#
# Efficiency reporting for model comparison (Objective 4: Computational Efficiency)

import csv
from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn


def _measure_latency(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int],
    n_warmup: int,
    n_runs: int,
    device: str,
) -> Tuple[float, float]:
    """Return (per_image_ms, throughput_img_per_sec) for the given model.

    Warm-up passes are excluded from the timing. On CUDA the device is
    synchronised around the timed region so the wall-clock reflects real GPU work.
    """
    model = model.to(device).eval()
    batch_size = input_shape[0]
    dummy = torch.zeros(input_shape, device=device)

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    total_images = n_runs * batch_size
    elapsed_sec  = t1 - t0
    per_image_ms = elapsed_sec / total_images * 1000.0
    throughput   = total_images / elapsed_sec
    return per_image_ms, throughput


def compute_gflops(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 8, 512, 512),
    device: str = "cpu",
) -> Optional[float]:
    """
    Compute per-image GFLOPs for a single forward pass.

    Uses PyTorch's built-in ``torch.utils.flop_counter.FlopCounterMode`` so no
    external dependency (fvcore/thop/ptflops) is required. That counter reports
    *true* FLOPs — it counts each multiply-accumulate as 2 operations — so
    ``GMACs = GFLOPs / 2`` if the reference you compare against reports MACs.

    Returns the per-image GFLOPs, or ``None`` if the FLOP count is unavailable
    (e.g. an op without a registered formula on this PyTorch build).
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None

    model = model.to(device).eval()
    dummy = torch.zeros(input_shape, device=device)
    counter = FlopCounterMode(display=False)
    try:
        with torch.no_grad(), counter:
            model(dummy)
    except Exception:
        return None

    batch_size = max(1, input_shape[0])
    total_flops = counter.get_total_flops()
    if not total_flops:
        return None
    return total_flops / batch_size / 1e9


def model_efficiency_report(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 8, 512, 512),
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
    gflops : float
        Per-image forward-pass GFLOPs (see :func:`compute_gflops`).
        Omitted if the FLOP count is unavailable on this build.
    gmacs : float
        Per-image multiply-accumulate count in billions (``gflops / 2``),
        matching the convention some papers label as "GFLOPs".
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
    # Latency ---------------------------------------------------------------- #
    per_image_ms, throughput = _measure_latency(
        model, input_shape, n_warmup, n_runs, device
    )

    # Parameter count -------------------------------------------------------- #
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb  = n_params * 4 / (1024 ** 2)   # float32

    # Compute cost ----------------------------------------------------------- #
    gflops = compute_gflops(model, input_shape=input_shape, device=device)

    report = {
        "parameters_M":           round(n_params / 1e6, 3),
        "model_size_MB":          round(size_mb, 2),
        "inference_time_ms":      round(per_image_ms, 3),
        "throughput_img_per_sec": round(throughput, 2),
    }
    if gflops is not None:
        report["gflops"] = round(gflops, 3)
        report["gmacs"]  = round(gflops / 2.0, 3)

    print("\n===== Model Efficiency Report =====")
    print(f"  Parameters       : {report['parameters_M']:.3f} M")
    print(f"  Model size       : {report['model_size_MB']:.2f} MB")
    if gflops is not None:
        print(f"  Compute          : {report['gflops']:.3f} GFLOPs  ({report['gmacs']:.3f} GMACs) / image @ {input_shape[2]}x{input_shape[3]}")
    else:
        print(f"  Compute          : (GFLOPs unavailable on this build)")
    print(f"  Inference time   : {report['inference_time_ms']:.3f} ms/image")
    print(f"  Throughput       : {report['throughput_img_per_sec']:.1f} img/s")
    print("===================================\n")

    return report


def efficiency_table(
    entries: Sequence[Tuple[str, nn.Module, int]],
    input_size: int = 512,
    device: str = "cpu",
    n_warmup: int = 3,
    n_runs: int = 10,
    csv_path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, object]]:
    """
    Profile several models and print one comparison table (params / GFLOPs / latency).

    Parameters
    ----------
    entries : sequence of (label, model, in_channels)
        Models to profile, in display order.
    input_size : int
        Spatial size H = W of the dummy input (512 for this dataset).
    device : str
        'cpu' or 'cuda'. Latency is only meaningful on the device you report;
        the paper measures compute cost as GFLOPs (device-independent) and
        recommends timing on GPU.
    n_warmup, n_runs : int
        Warm-up / timed passes per model for the latency column.
    csv_path : str or Path, optional
        If given, the table is also written here as CSV (ready for the thesis).

    Returns
    -------
    list of dict
        One row of metrics per model.
    """
    rows: List[Dict[str, object]] = []
    for label, model, in_ch in entries:
        shape = (1, in_ch, input_size, input_size)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        gflops = compute_gflops(model, input_shape=shape, device=device)
        latency_ms, throughput = _measure_latency(
            model, shape, n_warmup, n_runs, device
        )
        rows.append({
            "model":         label,
            "in_channels":   in_ch,
            "parameters_M":  round(n_params / 1e6, 3),
            "model_size_MB": round(n_params * 4 / (1024 ** 2), 2),
            "gflops":        round(gflops, 3) if gflops is not None else None,
            "gmacs":         round(gflops / 2.0, 3) if gflops is not None else None,
            "latency_ms":    round(latency_ms, 2),
            "throughput_ips": round(throughput, 2),
        })

    # Pretty-print ----------------------------------------------------------- #
    header = (
        f"{'Model':<32}{'Params(M)':>11}{'Size(MB)':>10}"
        f"{'GFLOPs':>9}{'GMACs':>9}{'Lat(ms)':>10}{'img/s':>9}"
    )
    print(f"\n===== Efficiency Comparison  (input {input_size}x{input_size}, device={device}) =====")
    print(header)
    print("-" * len(header))
    for r in rows:
        g = f"{r['gflops']:.2f}" if r["gflops"] is not None else "n/a"
        m = f"{r['gmacs']:.2f}" if r["gmacs"] is not None else "n/a"
        print(
            f"{r['model']:<32}{r['parameters_M']:>11.3f}{r['model_size_MB']:>10.2f}"
            f"{g:>9}{m:>9}{r['latency_ms']:>10.2f}{r['throughput_ips']:>9.2f}"
        )
    print("-" * len(header))
    print(f"Note: latency/throughput are {device.upper()} numbers; report GPU timings for "
          "efficiency claims. GFLOPs are device-independent.\n")

    # Persist ---------------------------------------------------------------- #
    if csv_path is not None:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys()) if rows else []
        with open(csv_path, "w", newline="", encoding="utf8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Efficiency table written to {csv_path}\n")

    return rows
