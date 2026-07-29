"""
visualize.py — Qualitative result visualization for DeepLabV3+ MobileViT research.

Renders per-sample panels so results can be inspected visually (input imagery,
ground-truth burned-area mask, model prediction, overlays, and probability heatmap).

Usage
-----
# Visualize the test split of a trained checkpoint (Experiment C, MobileViT-XXS)
python visualize.py ckpt_path=checkpoints/last.ckpt

# Different backbone / split / number of samples
python visualize.py ckpt_path=checkpoints/resnet50.ckpt model=deeplabv3plus_resnet50 \
    split=val num_samples=12 out_dir=outputs/viz_resnet

Each sample is saved as a PNG panel:
    <out_dir>/<split>_<sample_id>.png

Panel columns
-------------
  True-colour RGB        B4/B3/B2  — natural colour
  Burn-highlight RGB     B7/B5/B4  — SWIR2/NIR/Red false colour (burn scars glow)
  Ground truth           binary burned-area mask
  Prediction             model output (argmax)
  GT overlay             ground truth (red) over imagery
  Prediction overlay     prediction (red) over imagery
  Burned probability     softmax probability heatmap for the burned class
"""

import logging
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.utils as vutils
from hydra.utils import instantiate
from omegaconf import DictConfig

log = logging.getLogger(__name__)

# Landsat-8 band order in the input tensor (see indonesia_datamodule.py):
#   idx 0=B2 Blue, 1=B3 Green, 2=B4 Red, 3=B5 NIR, 4=B6 SWIR1, 5=B7 SWIR2
TRUE_COLOR = (2, 1, 0)          # Red, Green, Blue      → natural colour
BURN_COLOR = (5, 3, 2)          # SWIR2, NIR, Red       → burn-scar false colour


def _percentile_stretch(chw: torch.Tensor, p_low: float = 2.0, p_high: float = 98.0):
    """Per-band percentile stretch to [0,1] for display.

    Inputs are z-score normalised (mean~0, std~1), so a simple ``*255`` would
    clip to noise. A robust 2–98 % stretch per band gives a viewable composite
    regardless of the normalisation applied upstream.
    """
    out = torch.empty_like(chw)
    for c in range(chw.shape[0]):
        band = chw[c]
        lo = torch.quantile(band, p_low / 100.0)
        hi = torch.quantile(band, p_high / 100.0)
        out[c] = ((band - lo) / (hi - lo + 1e-8)).clamp(0.0, 1.0)
    return out


def _rgb_uint8(image: torch.Tensor, channels) -> torch.Tensor:
    """Build a (3,H,W) uint8 display image from selected bands of a (C,H,W) tensor."""
    n = image.shape[0]
    chans = [c for c in channels if c < n]
    while len(chans) < 3:                       # pad if fewer bands than expected
        chans.append(chans[-1])
    rgb = image[chans[:3]].float()
    rgb = _percentile_stretch(rgb)
    return (rgb * 255).round().byte()


@torch.no_grad()
def _predict(model, image: torch.Tensor):
    """Return (pred_mask HxW long, burned_prob HxW float) for a single (C,H,W) image."""
    logits = model(image.unsqueeze(0))          # (1, n_classes, H, W)
    if logits.shape[1] == 1:                    # binary-logit head
        prob = torch.sigmoid(logits[:, 0])
        pred = (prob > 0.5).long()
    else:                                       # 2-class softmax head (default)
        probs = torch.softmax(logits, dim=1)
        prob = probs[:, 1]
        pred = logits.argmax(dim=1)
    return pred.squeeze(0), prob.squeeze(0)


def _build_panel(image, gt_mask, pred_mask, burned_prob, title):
    """Assemble a single matplotlib figure for one sample."""
    true_rgb = _rgb_uint8(image, TRUE_COLOR)
    burn_rgb = _rgb_uint8(image, BURN_COLOR)

    gt = (gt_mask > 0).cpu()
    pr = (pred_mask > 0).cpu()

    gt_overlay = vutils.draw_segmentation_masks(true_rgb, gt.bool(), colors=["red"], alpha=0.5)
    pr_overlay = vutils.draw_segmentation_masks(true_rgb, pr.bool(), colors=["red"], alpha=0.5)

    panels = [
        ("True-colour (B4/B3/B2)", true_rgb.permute(1, 2, 0).numpy(), None),
        ("Burn-highlight (B7/B5/B4)", burn_rgb.permute(1, 2, 0).numpy(), None),
        ("Ground truth", gt.numpy(), "gray"),
        ("Prediction", pr.numpy(), "gray"),
        ("GT overlay", gt_overlay.permute(1, 2, 0).numpy(), None),
        ("Prediction overlay", pr_overlay.permute(1, 2, 0).numpy(), None),
        ("Burned probability", burned_prob.cpu().numpy(), "inferno"),
    ]

    fig, axs = plt.subplots(ncols=len(panels), figsize=(3.2 * len(panels), 3.6))
    fig.suptitle(title, fontsize=13)
    for ax, (name, data, cmap) in zip(axs, panels):
        im = ax.imshow(data, cmap=cmap, vmin=0 if cmap in ("inferno", "gray") else None,
                       vmax=1 if cmap in ("inferno", "gray") else None)
        ax.set_title(name, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


@hydra.main(version_base=None, config_path="configs", config_name="visualize")
def main(cfg: DictConfig):
    pl.seed_everything(47, True)

    if not cfg.get("ckpt_path"):
        raise ValueError("ckpt_path is required, e.g. ckpt_path=checkpoints/last.ckpt")
    ckpt_path = Path(cfg.ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------ #
    # Model: instantiate architecture, then load trained weights
    # ------------------------------------------------------------------ #
    model = instantiate(cfg["model"])
    state = torch.load(ckpt_path, map_location=device)
    state_dict = state.get("state_dict", state)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        log.warning("Missing keys when loading checkpoint: %d", len(missing))
    if unexpected:
        log.warning("Unexpected keys when loading checkpoint: %d", len(unexpected))
    model.eval().to(device)
    log.info("Loaded checkpoint: %s", ckpt_path)

    # ------------------------------------------------------------------ #
    # Data: pick the requested split
    # ------------------------------------------------------------------ #
    datamodule = instantiate(cfg["dataset"])
    stage = "fit" if cfg.split in ("train", "val") else "test"
    datamodule.setup(stage)
    dataset = {
        "train": datamodule.train_dataset,
        "val": datamodule.val_dataset,
        "test": datamodule.test_dataset,
    }[cfg.split]
    log.info("Split '%s': %d samples", cfg.split, len(dataset))

    n = len(dataset) if cfg.get("num_samples") in (None, "null") else min(cfg.num_samples, len(dataset))

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Render
    # ------------------------------------------------------------------ #
    for i in range(n):
        sample = dataset[i]
        image = sample["post"].float().to(device)
        gt_mask = sample["mask"].squeeze(0) if sample["mask"].dim() == 3 else sample["mask"]
        sample_id = sample.get("id", f"idx{i}")

        pred_mask, burned_prob = _predict(model, image)

        fig = _build_panel(
            image.cpu(), gt_mask.cpu(), pred_mask, burned_prob,
            title=f"{cfg.split} · {sample_id}",
        )
        out_path = out_dir / f"{cfg.split}_{sample_id}.png"
        fig.savefig(out_path, dpi=cfg.get("dpi", 130), bbox_inches="tight")
        plt.close(fig)
        log.info("[%d/%d] saved %s", i + 1, n, out_path)

    print(f"\nSaved {n} visualization panel(s) to: {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
