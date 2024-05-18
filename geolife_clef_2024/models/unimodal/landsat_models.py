"""Models that use only the landsat time series."""

import dataclasses
import functools
from collections.abc import Sequence

import albumentations as A
import polars as pl
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from geolife_clef_2024 import datasets
from geolife_clef_2024.models import common as common_models
from geolife_clef_2024.models import utils as model_utils


@dataclasses.dataclass()
class LandsatModifiedResNetConfig(model_utils.WandbTrackedModeConfig):
    """Model using ModifiedResNet18."""

    batch_size: int = 64
    transforms: Sequence[A.TransformType] = ()
    learning_rate: float = 0.0002


def train_loader_from_config(config: LandsatModifiedResNetConfig):
    """Create a train_loader from config."""
    return DataLoader(
        datasets.TimeSeriesDataset(
            split="train", transforms=config.transforms, which="landsat_time_series"
        ),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
    )


def test_load_from_config(config: LandsatModifiedResNetConfig):
    """Create a test loader from config."""
    return DataLoader(
        datasets.TimeSeriesDataset(
            split="test", transforms=config.transforms, which="landsat_time_series"
        ),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
    )


def optimizer_from_config(model: torch.nn.Module, config: LandsatModifiedResNetConfig):
    """Create an optimizer from config."""
    return torch.optim.AdamW(model.parameters(), lr=config.learning_rate)


def main(args: Sequence[str] | None = None):
    """Train/eval the model."""
    model_utils.WandbTrackedModel[LandsatModifiedResNetConfig](
        checkpoint_prefix="resnet18-with-landsat-cubes",
        config_class=LandsatModifiedResNetConfig,
        tags=("ModifiedResNet18", "landsat", "unimodal"),
        model=common_models.ModifiedResNet18(
            [6, 4, 21],
            num_classes=datasets.load_observation_data(split="train")
            .select(pl.col("speciesId").unique().count())
            .collect()
            .item(),
        ),
        train_loader=train_loader_from_config,
        test_loader=test_load_from_config,
        optimizer=optimizer_from_config,
        scheduler=functools.partial(CosineAnnealingLR, T_max=25),  # type:ignore
        loss_factory=torch.nn.BCEWithLogitsLoss,  # type:ignore
    ).parse_args(args=args)


if __name__ == "__main__":
    main()
