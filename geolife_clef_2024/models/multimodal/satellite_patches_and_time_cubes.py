"""Module containing siamese network."""

import dataclasses
import functools
from collections.abc import Sequence

import albumentations as A
import polars as pl
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from geolife_clef_2024 import dataset_utils, datasets
from geolife_clef_2024.models import common as common_models
from geolife_clef_2024.models import utils as model_utils


@dataclasses.dataclass()
class MultiModalEnsembleConfig(model_utils.WandbTrackedModeConfig):
    """Model using ModifiedResNet18."""

    batch_size: int = 64

    landsat_transforms: Sequence[A.TransformType] = ()
    bio_climatic_transforms: Sequence[A.TransformType] = ()
    sentinel_transforms: Sequence[A.TransformType] = (A.Normalize(mean=0.5, std=0.5),)

    learning_rate: float = 0.00025
    num_epochs: int = 10


def train_loader_from_config(config: MultiModalEnsembleConfig):
    """Create a train_loader from config."""
    return DataLoader(
        dataset_utils.FlattenedDataset(
            dataset_utils.DatasetSlicer[:2](
                datasets.TimeSeriesDataset(
                    split="train",
                    transforms=config.landsat_transforms,
                    which="landsat_time_series",
                )
            ),
            dataset_utils.DatasetSlicer[1](
                datasets.TimeSeriesDataset(
                    split="train",
                    transforms=config.bio_climatic_transforms,
                    which="bioclimatic_monthly",
                )
            ),
            dataset_utils.DatasetSlicer[1:](
                datasets.SatellitePatchesDataset(
                    split="train", transforms=config.sentinel_transforms
                )
            ),
        ),
        shuffle=True,
        batch_size=config.batch_size,
    )


def test_load_from_config(config: MultiModalEnsembleConfig):
    """Create a test loader from config."""
    return DataLoader(
        dataset_utils.FlattenedDataset(
            dataset_utils.DatasetSlicer[:2](
                datasets.TimeSeriesDataset(
                    split="test",
                    transforms=config.landsat_transforms,
                    which="landsat_time_series",
                )
            ),
            dataset_utils.DatasetSlicer[1](
                datasets.TimeSeriesDataset(
                    split="test",
                    transforms=config.bio_climatic_transforms,
                    which="bioclimatic_monthly",
                )
            ),
            dataset_utils.DatasetSlicer[1:](
                datasets.SatellitePatchesDataset(
                    split="test", transforms=config.sentinel_transforms
                )
            ),
        ),
        shuffle=False,
        batch_size=config.batch_size,
    )


def optimizer_from_config(model: torch.nn.Module, config: MultiModalEnsembleConfig):
    """Create an optimizer from config."""
    return torch.optim.AdamW(model.parameters(), lr=config.learning_rate)


class MultiModalEnsemble(nn.Module):
    """Ensemble model combining satellite patches with time cubes."""

    def __init__(self, num_classes: int):
        """Create an ensemble of the sentinel data and the time series cubes."""
        super().__init__()

        self.landsat_model = nn.Sequential(
            model_utils.unhead(
                common_models.ModifiedResNet18([6, 4, 21], num_classes=num_classes)
            ),
            nn.LayerNorm(1000),
        )
        self.bio_climatic_model = nn.Sequential(
            model_utils.unhead(
                common_models.ModifiedResNet18([4, 19, 12], num_classes=num_classes)
            ),
            nn.LayerNorm(1000),
        )

        self.sentinel_model = model_utils.unhead(
            common_models.swin_transformer_model(
                num_classes=num_classes, model="swin_t", weights="IMAGENET1K_V1"
            )
        )

        self.head = nn.Sequential(
            nn.Linear(2768, 4096), nn.Dropout(p=0.1), nn.Linear(4096, num_classes)
        )

    def forward(
        self,
        landsat: torch.Tensor,
        bio_climatic: torch.Tensor,
        sentinel: torch.Tensor,
    ):
        """Run the individual models and combine the output."""
        landsat = self.landsat_model(landsat)
        bio_climatic = self.bio_climatic_model(bio_climatic)
        sentinel = self.sentinel_model(sentinel)
        xyz = torch.cat((landsat, bio_climatic, sentinel), dim=1)
        out = self.head(xyz)
        return out


def main(args: Sequence[str] | None = None):
    """Train/eval the model."""
    model_utils.WandbTrackedModel[MultiModalEnsembleConfig](
        checkpoint_prefix="ensemble-nn-sentinel-time_series-cube",
        config_class=MultiModalEnsembleConfig,
        model=MultiModalEnsemble(
            datasets.load_observation_data(split="train")
            .select(pl.col("speciesId").unique().count())
            .collect()
            .item()
        ),
        train_loader=train_loader_from_config,
        test_loader=test_load_from_config,
        optimizer=optimizer_from_config,
        scheduler=functools.partial(CosineAnnealingLR, T_max=25),  # type:ignore
        loss_factory=torch.nn.BCEWithLogitsLoss,  # type:ignore
    ).parse_args(args=args)


if __name__ == "__main__":
    main()
