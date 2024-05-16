"""Models that only use the satellite images."""

import dataclasses
import os
import re
from collections.abc import Sequence
from typing import Literal

import albumentations
import polars as pl
import torch
import torchvision.models as models
import tqdm
from magicbox import random as random_utils
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import wandb
from geolife_clef_2024 import constants, datasets


@dataclasses.dataclass()
class Swinv2:
    """Model using Swinv2."""

    batch_size: int = 64
    transforms: Sequence[albumentations.TransformType] = (
        albumentations.RandomBrightnessContrast(p=0.2),
        albumentations.ColorJitter(p=0.2),
        albumentations.OpticalDistortion(p=0.2),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate: float = 0.0002
    num_epochs: int = 10
    positive_weight_factor: float = 1.0
    weights: Literal["IMAGENET1K_V1"] = "IMAGENET1K_V1"
    run_id: str | None = "v1.1"
    checkpoint_prefix = "resnet18-with-landsat-cubes"
    _model: nn.Module = dataclasses.field(init=False, repr=False, compare=False)

    def __post_init__(
        self,
    ):
        """Initialize the model."""
        model = models.swin_v2_s(weights=self.weights)
        model.features[0][0] = nn.Conv2d(4, 96, kernel_size=(4, 4), stride=(4, 4))
        model.head = nn.Linear(
            in_features=768,
            out_features=datasets.load_observation_data(split="train")
            .select(pl.col("speciesId").unique().count())
            .collect()
            .item(),
            bias=True,
        )
        self._model = model

    def _checkpoint_path(self, epoch: int):
        return os.path.join(
            constants.CHECKPOINT_DIR,
            f"{self.checkpoint_prefix}-epoch_{epoch+1}.pth",
        )

    def _train(self):
        run = wandb.init(
            project=constants.WANDB_PROJECT,
            tags=["Swinv2", "satellite-image", "unimodal"],
            config={
                k: v
                for k, v in dataclasses.asdict(self).items()
                if not k.startswith("_")
            },
            id=self.run_id,
        )
        model, num_epochs, positive_weight_factor, device = (
            self._model,
            self.num_epochs,
            self.positive_weight_factor,
            self.device,
        )
        model = model.train().to(device, dtype=torch.float32)
        train_loader = DataLoader(
            datasets.SatellitePatchesDataset(split="train", transforms=self.transforms),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=25)

        for epoch in tqdm.tqdm(range(num_epochs)):
            for _, data, targets in tqdm.tqdm(
                train_loader,
                leave=False,
                position=1,
            ):
                data = data.to(device, dtype=torch.float32)
                targets = targets.to(device, dtype=torch.float32)

                optimizer.zero_grad()
                outputs = model(data)

                pos_weight = targets * positive_weight_factor
                criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                loss = criterion(outputs, targets)

                loss.backward()
                optimizer.step()

            scheduler.step()
            run.log(
                {
                    "loss": loss.item(),
                    "epoch": epoch + 1,
                    "lr": scheduler.get_last_lr(),
                },
                step=epoch,
            )

            torch.save(
                model.state_dict(),
                (model_path := self._checkpoint_path(epoch=epoch)),
            )
            run.log_model(path=model_path)
        run.finish()

    def fit(self):
        """Fit the training data/or load a checkpoint."""
        random_utils.set_seed(constants.SEED)
        wandb.login()
        try:
            api = wandb.Api()
            last_run, *_ = api.runs(constants.WANDB_PROJECT)
            if last_run.state == "finished":
                epoch_pattern = re.compile(
                    rf"run-{self.run_id}-{self.checkpoint_prefix}-.*?epoch_(\d+)"
                )
                collections_by_epoch = {
                    int(epoch_pattern.findall(artifact.name)[0]): artifact
                    for artifact in api.artifact_type(
                        type_name="model", project=constants.WANDB_PROJECT
                    ).collections()
                    if artifact.name.startswith(
                        f"run-{self.run_id}-{self.checkpoint_prefix}"
                    )
                }
                if len(collections_by_epoch):
                    last_epoch = max(collections_by_epoch.keys())
                    checkpoint_path = self._checkpoint_path(last_epoch - 1)

                    collections_by_epoch[last_epoch].artifacts()[0].download(
                        constants.CHECKPOINT_DIR
                    )
                    print(f"Loading checkpoint {checkpoint_path}")
                    self._model.train().to(
                        self.device, dtype=torch.float32
                    ).load_state_dict(torch.load(checkpoint_path))
                    return

        except Exception as e:
            print("Tried loading an old run. Failed!")
            print(e)
        return self._train()


if __name__ == "__main__":
    Swinv2().fit()