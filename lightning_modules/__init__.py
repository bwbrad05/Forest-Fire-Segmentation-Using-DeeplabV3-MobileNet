# Indonesian burned-area dataset modules (Landsat-8).

from .indonesia_datamodule import (
    IndonesiaDataModule,   # primary dataset
    IndonesiaDataset,
)

__all__ = [
    "IndonesiaDataModule",
    "IndonesiaDataset",
]
