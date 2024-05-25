"""Models that only use the satellite images."""

import dataclasses
import functools
from collections.abc import Sequence
from typing import Literal

import albumentations as A
import polars as pl
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from geolife_clef_2024 import dataset_utils, datasets
from geolife_clef_2024.models import common as common_models
from geolife_clef_2024.models import utils as model_utils


@dataclasses.dataclass()
class SatellitePatchesSwinTransformerConfig(model_utils.WandbTrackedModeConfig):
    """Model using SwinTransformer."""

    batch_size: int = 64
    transforms: Sequence[A.TransformType] = (
        A.RandomBrightnessContrast(p=0.2),
        A.ColorJitter(p=0.2),
        A.OpticalDistortion(p=0.2),
    )
    learning_rate: float = 0.0002
    model: common_models.SwinTransformer = "swin_v2_s"
    weights: Literal["IMAGENET1K_V1"] = "IMAGENET1K_V1"
    num_epochs: int = 10
    use_mixing: bool = False


def model_from_config(config: SatellitePatchesSwinTransformerConfig):
    """Create the model from the config."""
    return common_models.swin_transformer_model(
        num_classes=datasets.load_observation_data(split="train")
        .select(pl.col("speciesId").unique().count())
        .collect()
        .item(),
        model=config.model,
        weights=config.weights,
    )


def train_loader_from_config(config: SatellitePatchesSwinTransformerConfig):
    """Create a train_loader from config."""
    collate_fn = None
    if config.use_mixing:
        collate_fn = dataset_utils.create_cutmix_or_mixup_collate_function(
            datasets.load_observation_data(split="train")
            .select(pl.col("speciesId").unique().count())
            .collect()
            .item()
        )
    return DataLoader(
        datasets.SatellitePatchesDataset(split="train", transforms=config.transforms),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )


def test_load_from_config(config: SatellitePatchesSwinTransformerConfig):
    """Create a test loader from config."""
    return DataLoader(
        datasets.SatellitePatchesDataset(split="test", transforms=config.transforms),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
    )


def optimizer_from_config(
    model: torch.nn.Module, config: SatellitePatchesSwinTransformerConfig
):
    """Create an optimizer from config."""
    return torch.optim.AdamW(model.parameters(), lr=config.learning_rate)


def main(args: Sequence[str] | None = None):
    """Train/eval the model."""
    model_utils.WandbTrackedModel[SatellitePatchesSwinTransformerConfig](
        checkpoint_prefix="swinv2-satellite-patches",
        config_class=SatellitePatchesSwinTransformerConfig,
        model=model_from_config,
        train_loader=train_loader_from_config,
        test_loader=test_load_from_config,
        optimizer=optimizer_from_config,
        scheduler=functools.partial(CosineAnnealingLR, T_max=25),  # type:ignore
        loss_factory=torch.nn.BCEWithLogitsLoss,  # type:ignore
    ).parse_args(args=args)


if __name__ == "__main__":
    main()
