from itertools import groupby, product
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF
import torchvision.utils as vutils
import utils

Position = Tuple[int, int]


def extract_rgb(img: torch.Tensor, rgb_channels: Tuple[int, int, int]) -> torch.Tensor:
    return (img[rgb_channels, :] * 255).round().byte()


def crop_image(
    image: torch.Tensor, crop_size: int
) -> Tuple[torch.Tensor, List[Position]]:
    cropped_images = []
    crop_position = []
    for top, left in product(
        range(0, image.size()[-1], crop_size),
        range(0, image.size()[-2], crop_size),
    ):
        cropped_images.append(
            TF.crop(image, top, left, crop_size, crop_size).unsqueeze(0)
        )
        crop_position.append((top, left))
    return torch.concat(cropped_images), crop_position


def recompose_image(crops: torch.Tensor, positions: List[Position]) -> torch.Tensor:
    zipped = sorted(zip(positions, crops), key=lambda x: x[0])
    rows = [
        torch.concat([el[1] for el in g], dim=2)
        for k, g in groupby(zipped, key=lambda x: x[0][0])
    ]
    return torch.concat(rows, dim=1)


def draw_figure(
    masks: torch.Tensor,
    pred_mask: torch.Tensor,
    images: torch.Tensor = None,
    logits: torch.Tensor = None,
) -> List[plt.Figure]:
    rgb_indexes = (3, 2, 1)

    masks = (masks.byte().cpu() > 0).unsqueeze(1)
    pred_mask = (pred_mask.byte().cpu() > 0).unsqueeze(1)
    images = (
        (images[:, rgb_indexes] * 255).cpu().byte().round()
        if images is not None
        else None
    )
    logits = logits.cpu() if logits is not None else None

    figures = []

    for i, (gt, pr) in enumerate(zip(masks, pred_mask)):
        collection = {
            "ground truth mask": (gt, "gray"),
            "prediction mask": (pr, "gray"),
        }
        if images is not None:
            img = images[i]
            collection["original image"] = (img, None)
            collection["ground truth with image"] = (
                vutils.draw_segmentation_masks(img, gt.bool(), colors=["red"]),
                None,
            )
            collection["prediction with image"] = (
                vutils.draw_segmentation_masks(img, pr.bool(), colors=["red"]),
                None,
            )
        if logits is not None:
            collection["prediction heatmap"] = (
                logits[i][1].unsqueeze(0).cpu(),
                "viridis",
            )

        collection = {
            k: (v.permute(1, 2, 0).numpy(), cmap) for k, (v, cmap) in collection.items()
        }

        figure, axs = plt.subplots(ncols=len(collection), figsize=(20, 20))
        figure.tight_layout()
        for ax, (k, (v, cmap)) in zip(axs, collection.items()):
            ax.imshow(v, cmap=cmap)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_title(k, {"fontsize": 15})
        figures.append(figure)

    return figures
