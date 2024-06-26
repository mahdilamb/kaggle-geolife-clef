"""Models that only use the satellite images."""

import dataclasses
import os
from collections.abc import Callable, Sequence
from typing import Generic, Protocol, TypeVar

import argparse_dataclass
import numpy as np
import polars as pl
import torch
from magicbox import random as random_utils
from torch import nn
from torch.optim import lr_scheduler, optimizer
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

import wandb
from geolife_clef_2024 import constants, datasets, submissions


@dataclasses.dataclass()
class WandbTrackedModeConfig:
    """Base dataclass for a config."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs: int = 20
    positive_weight_factor: float = 1.0
    run_id: str = ""


class WeightedLoss(Protocol):
    """Protocol for a loss function that takes positional weights."""

    def __call__(
        self,
        pos_weight: torch.Tensor | None = None,
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Call the loss function."""
        ...


T = TypeVar("T", bound=WandbTrackedModeConfig)
M = TypeVar("M", bound=nn.Module)


@dataclasses.dataclass(kw_only=True)
class WandbTrackedModel(Generic[T]):
    """Model using ModifiedResNet18."""

    checkpoint_prefix: str
    model: nn.Module | Callable[[T], nn.Module]
    train_loader: Callable[[T], DataLoader]
    test_loader: Callable[[T], DataLoader]
    optimizer: Callable[[nn.Module, T], optimizer.Optimizer]
    scheduler: type[lr_scheduler.LRScheduler]
    loss_factory: WeightedLoss = torch.nn.BCEWithLogitsLoss  # type:ignore
    tags: Sequence[str] = ()
    run_id: str | None = None
    config_class: type[T] | None = None
    config: T | None = dataclasses.field(init=False, repr=False, default=None)

    def __post_init__(self):
        """Check that the config is parsable."""
        if self.config_class is None and self.config is None:
            raise ValueError("Must provide either a config or config class.")
        if self.config is not None:
            self.config_class = type(self.config)

    @property
    def __model(self):
        model = self.model
        if not isinstance(model, nn.Module):
            return model(self.__safe_config)
        return model

    def __checkpoint_path(self, epoch: int):
        return os.path.join(
            constants.CHECKPOINT_DIR,
            f"{self.checkpoint_prefix}-epoch_{epoch+1}.pth",
        )

    @property
    def __safe_config(self):
        if self.config is None:
            raise ValueError("Please either supply a config or parse args.")
        return self.config

    def _train(self, run):
        config = self.__safe_config
        model, num_epochs, positive_weight_factor, device = (
            self.__model,
            config.num_epochs,
            config.positive_weight_factor,
            config.device,
        )
        device = torch.device(device)
        model = model.train().to(device, dtype=torch.float32)
        train_loader = self.train_loader(config)
        optimizer = self.optimizer(model, config)
        scheduler = self.scheduler(optimizer)
        criterion_factory = self.loss_factory
        for epoch in tqdm(range(num_epochs)):
            accuracies = []
            for _, *data, targets in tqdm(
                train_loader,
                leave=False,
                position=1,
            ):
                data = [d.to(device, dtype=torch.float32) for d in data]
                targets = targets.to(device, dtype=torch.float32)

                optimizer.zero_grad()
                outputs = model(*data)
                accuracies.append(
                    torch.mean(
                        ((torch.sigmoid(outputs) >= 0.5).long() == targets).float(),
                        dim=1,
                    )
                    .cpu()
                    .numpy()
                    .mean()
                )

                pos_weight = targets * positive_weight_factor
                criterion = criterion_factory(pos_weight=pos_weight)
                loss = criterion(outputs, targets)

                loss.backward()
                optimizer.step()

            scheduler.step()
            run.log(
                {
                    "loss": loss.item(),
                    "lr": scheduler.get_last_lr(),
                    "train/accuracy": np.mean(accuracies),
                },
                step=epoch,
            )

            torch.save(
                model.state_dict(),
                (model_path := self.__checkpoint_path(epoch=epoch)),
            )
            run.log_model(path=model_path)
        run.finish()

    def fit(self):
        """Fit the training data/or load a checkpoint."""
        random_utils.set_seed(constants.SEED)
        config = self.__safe_config
        device = torch.device(config.device)
        run = wandb.init(
            job_type="eval" if self.run_id else "train",
            project=constants.WANDB_PROJECT,
            tags=self.tags,
            config=dataclasses.asdict(config),
            id=self.run_id or None,
            group=self.checkpoint_prefix,
        )
        if self.run_id:
            checkpoint_path = run.use_model(
                name=f"{run.entity}/{constants.WANDB_PROJECT}/run-{self.run_id}-{self.checkpoint_prefix}-epoch_{config.num_epochs}.pth:latest",
            )
            run.finish()
            self.__model.train().to(device, dtype=torch.float32).load_state_dict(
                torch.load(checkpoint_path)
            )
            return

        return self._train(run)

    @torch.inference_mode()
    def transform(self):
        """Get the submission output."""
        config = self.__safe_config
        model = self.__model.eval()
        decoder = datasets.create_species_decoder()
        test_loader = self.test_loader(config)
        device = torch.device(config.device)
        all_survey_ids = pl.Series("surveyId", dtype=pl.Int64)
        all_predictions = pl.Series("predictions", dtype=pl.List(pl.Int64))

        for survey_id, *data in tqdm(
            test_loader,
            leave=False,
            position=1,
        ):
            data = [d.to(device, dtype=torch.float32) for d in data]
            outputs = torch.sigmoid(model(*data))
            all_survey_ids.extend(pl.Series(values=np.asarray(survey_id).flatten()))
            all_predictions.extend(
                decoder(np.argsort(-outputs.cpu().numpy(), axis=1)[:, :25])
            )

        return pl.DataFrame((all_survey_ids, all_predictions))

    def save_submission(self):
        """Save the submissions."""
        submissions.save_predictions(
            os.path.join(
                constants.ROOT_DIR,
                "submissions",
                f"{self.checkpoint_prefix}-{self.run_id}.csv",
            ),
            self.transform(),
        )

    def parse_args(self, args: Sequence[str] | None = None):
        """Parse the args to generate a config."""
        if self.config_class is None:
            raise ValueError("Must supply a config class to to parse args.")
        parser = argparse_dataclass.ArgumentParser(self.config_class)
        self.config, args = parser.parse_known_args(args)
        if any(arg == "--check" for arg in args):
            # TODO print available runs.
            raise Exception("Checking is currently not supported")
        self.config.run_id = self.config.run_id or None
        self.fit()
        self.save_submission()


def unhead(model: M, head_attr: str = "head") -> M:
    """Remove the head of a model."""
    setattr(model, head_attr, nn.Identity())
    return model
