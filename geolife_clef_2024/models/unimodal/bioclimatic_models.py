"""Models that only use the satellite images."""

import dataclasses
import os
import re
from collections.abc import Sequence

import albumentations as A
import numpy as np
import polars as pl
import torch
from magicbox import random as random_utils
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

import wandb
from geolife_clef_2024 import constants, datasets, submissions
from geolife_clef_2024.models import common as common_models


@dataclasses.dataclass()
class BioClimaticModifiedResNet:
    """Model using ModifiedResNet18."""

    batch_size: int = 64
    transforms: Sequence[A.TransformType] = ()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    learning_rate: float = 0.0002
    num_epochs: int = 20
    positive_weight_factor: float = 1.0
    run_id: str = ""
    checkpoint_prefix = "resnet18-with-bioclimatic-cubes"
    _model: nn.Module = dataclasses.field(init=False, repr=False, compare=False)

    def __post_init__(
        self,
    ):
        """Initialize the model."""
        self._model = common_models.ModifiedResNet18(
            [4, 19, 12],
            num_classes=datasets.load_observation_data(split="train")
            .select(pl.col("speciesId").unique().count())
            .collect()
            .item(),
        )

    def _checkpoint_path(self, epoch: int):
        return os.path.join(
            constants.CHECKPOINT_DIR,
            f"{self.checkpoint_prefix}-epoch_{epoch+1}.pth",
        )

    def _train(self):
        run = wandb.init(
            project=constants.WANDB_PROJECT,
            tags=["ModifiedResNet18", "bioclimatic", "unimodal"],
            config={
                k: v
                for k, v in dataclasses.asdict(self).items()
                if not k.startswith("_")
            },
            id=self.run_id or None,
        )
        model, num_epochs, positive_weight_factor, device = (
            self._model,
            self.num_epochs,
            self.positive_weight_factor,
            torch.device(self.device),
        )
        model = model.train().to(device, dtype=torch.float32)
        train_loader = DataLoader(
            datasets.TimeSeriesDataset(
                split="train", transforms=self.transforms, which="bioclimatic_monthly"
            ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=25)

        for epoch in tqdm(range(num_epochs)):
            for _, data, targets in tqdm(
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
        if os.path.exists(
            checkpoint_path := self._checkpoint_path(self.num_epochs - 1)
        ):
            print(f"Loading checkpoint {checkpoint_path}")
            self._model.train().to(
                torch.device(self.device), dtype=torch.float32
            ).load_state_dict(torch.load(checkpoint_path))
            return
        wandb.login()
        if self.run_id:
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
                            torch.device(self.device), dtype=torch.float32
                        ).load_state_dict(torch.load(checkpoint_path))
                        return

            except Exception as e:
                print("Tried loading an old run. Failed!")
                print(e)
        return self._train()

    @torch.inference_mode()
    def transform(self):
        """Get the submission output."""
        device = torch.device(self.device)
        model = self._model.eval()
        decoder = datasets.create_species_decoder()
        test_loader = DataLoader(
            datasets.TimeSeriesDataset(
                split="test", transforms=self.transforms, which="bioclimatic_monthly"
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
        )
        all_survey_ids = pl.Series("surveyId", dtype=pl.Int64)
        all_predictions = pl.Series("predictions", dtype=pl.List(pl.Int64))

        for survey_id, data in tqdm(
            test_loader,
            leave=False,
            position=1,
        ):
            data = data.to(device, dtype=torch.float32)
            outputs = torch.sigmoid(model(data))
            all_survey_ids.extend(pl.Series(values=np.asarray(survey_id).flatten()))
            all_predictions.extend(
                decoder(np.argsort(-outputs.cpu().numpy(), axis=1)[:, :25])
            )

        return pl.DataFrame((all_survey_ids, all_predictions))


def main(args: Sequence[str] | None = None):
    """Allow running from the terminal."""
    import argparse_dataclass

    parser = argparse_dataclass.ArgumentParser(BioClimaticModifiedResNet)
    for action in parser._actions:
        if (option := action.option_strings[0]).startswith("---"):
            parser._handle_conflict_resolve(None, [(option, action)])
    model, _ = parser.parse_known_args(args)
    model.fit()
    submissions.save_predictions(
        os.path.join(
            constants.ROOT_DIR,
            "submissions",
            f"resnet18-bioclimatic-{model.run_id}.csv",
        ),
        model.transform(),
    )


if __name__ == "__main__":
    main()
