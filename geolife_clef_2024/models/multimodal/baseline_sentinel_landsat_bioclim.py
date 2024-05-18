"""Modified version of the multi-modal baseline."""

import dataclasses
import functools
import os
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm.autonotebook import tqdm

from geolife_clef_2024 import constants, dataset_utils
from geolife_clef_2024.models import utils as model_utils

NUM_CLASSES = 11255


@dataclasses.dataclass()
class MultiModalEnsembleConfig(model_utils.WandbTrackedModeConfig):
    """Model using ModifiedResNet18."""

    batch_size: int = 64
    learning_rate: float = 0.00025
    num_epochs: int = 10
    use_mixing: bool = False


def construct_patch_path(data_path, survey_id):
    """Construct the patch file path based on plot_id as './CD/AB/XXXXABCD.jpeg'."""
    survey_id = str(survey_id)
    return os.path.join(
        data_path, survey_id[-2:], survey_id[-4:-2], f"{survey_id}.jpeg"
    )


class TrainDataset(Dataset):
    def __init__(
        self,
        bioclim_data_dir: str,
        landsat_data_dir: str,
        sentinel_data_dir: str,
        metadata: pd.DataFrame,
        transform: transforms.Compose,
    ):
        self.transform = transform
        self.sentinel_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.5, 0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5, 0.5)
                ),
            ]
        )

        self.bioclim_data_dir = bioclim_data_dir
        self.landsat_data_dir = landsat_data_dir
        self.sentinel_data_dir = sentinel_data_dir
        self.metadata = metadata.dropna(subset="speciesId").reset_index(drop=True)
        self.metadata["speciesId"] = self.metadata["speciesId"].astype(int)
        self.label_dict = (
            self.metadata.groupby("surveyId")["speciesId"].apply(list).to_dict()
        )

        self.metadata = self.metadata.drop_duplicates(subset="surveyId").reset_index(
            drop=True
        )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        survey_id = self.metadata.surveyId[idx]

        landsat_sample = torch.nan_to_num(
            torch.load(
                os.path.join(
                    self.landsat_data_dir,
                    f"GLC24-PA-train-landsat-time-series_{survey_id}_cube.pt",
                )
            )
        )
        bioclim_sample = torch.nan_to_num(
            torch.load(
                os.path.join(
                    self.bioclim_data_dir,
                    f"GLC24-PA-train-bioclimatic_monthly_{survey_id}_cube.pt",
                )
            )
        )

        rgb_sample = np.array(
            Image.open(construct_patch_path(self.sentinel_data_dir, survey_id))
        )
        nir_sample = np.array(
            Image.open(
                construct_patch_path(
                    self.sentinel_data_dir.replace("rgb", "nir").replace("RGB", "NIR"),
                    survey_id,
                )
            )
        )
        sentinel_sample = np.concatenate((rgb_sample, nir_sample[..., None]), axis=2)

        species_ids = self.label_dict.get(
            survey_id, []
        )  # Get list of species IDs for the survey ID
        label = torch.zeros(NUM_CLASSES)  # Initialize label tensor
        for species_id in species_ids:
            label[species_id] = (
                1  # Set the corresponding class index to 1 for each species
            )

        if isinstance(landsat_sample, torch.Tensor):
            landsat_sample = landsat_sample.permute(
                1, 2, 0
            )  # Change tensor shape from (C, H, W) to (H, W, C)
            landsat_sample = landsat_sample.numpy()  # Convert tensor to numpy array

        if isinstance(bioclim_sample, torch.Tensor):
            bioclim_sample = bioclim_sample.permute(
                1, 2, 0
            )  # Change tensor shape from (C, H, W) to (H, W, C)
            bioclim_sample = bioclim_sample.numpy()  # Convert tensor to numpy array

        landsat_sample = self.transform(landsat_sample)
        bioclim_sample = self.transform(bioclim_sample)
        sentinel_sample = self.sentinel_transform(sentinel_sample)

        return survey_id, landsat_sample, bioclim_sample, sentinel_sample, label


class TestDataset(TrainDataset):
    def __init__(
        self,
        bioclim_data_dir,
        landsat_data_dir,
        sentinel_data_dir,
        metadata,
        transform,
    ):
        self.transform = transform
        self.sentinel_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.5, 0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5, 0.5)
                ),
            ]
        )

        self.bioclim_data_dir = bioclim_data_dir
        self.landsat_data_dir = landsat_data_dir
        self.sentinel_data_dir = sentinel_data_dir
        self.metadata = metadata

    def __getitem__(self, idx):
        survey_id = self.metadata.surveyId[idx]
        landsat_sample = torch.nan_to_num(
            torch.load(
                os.path.join(
                    self.landsat_data_dir,
                    f"GLC24-PA-test-landsat_time_series_{survey_id}_cube.pt",
                )
            )
        )
        bioclim_sample = torch.nan_to_num(
            torch.load(
                os.path.join(
                    self.bioclim_data_dir,
                    f"GLC24-PA-test-bioclimatic_monthly_{survey_id}_cube.pt",
                )
            )
        )

        rgb_sample = np.array(
            Image.open(construct_patch_path(self.sentinel_data_dir, survey_id))
        )
        nir_sample = np.array(
            Image.open(
                construct_patch_path(
                    self.sentinel_data_dir.replace("rgb", "nir").replace("RGB", "NIR"),
                    survey_id,
                )
            )
        )
        sentinel_sample = np.concatenate((rgb_sample, nir_sample[..., None]), axis=2)

        if isinstance(landsat_sample, torch.Tensor):
            landsat_sample = landsat_sample.permute(
                1, 2, 0
            )  # Change tensor shape from (C, H, W) to (H, W, C)
            landsat_sample = landsat_sample.numpy()  # Convert tensor to numpy array

        if isinstance(bioclim_sample, torch.Tensor):
            bioclim_sample = bioclim_sample.permute(
                1, 2, 0
            )  # Change tensor shape from (C, H, W) to (H, W, C)
            bioclim_sample = bioclim_sample.numpy()  # Convert tensor to numpy array

        landsat_sample = self.transform(landsat_sample)
        bioclim_sample = self.transform(bioclim_sample)
        sentinel_sample = self.sentinel_transform(sentinel_sample)

        return survey_id, landsat_sample, bioclim_sample, sentinel_sample


def train_loader_from_config(config: MultiModalEnsembleConfig):
    """Create a train_loader from config."""
    collate_fn = None
    if config.use_mixing:
        collate_fn = dataset_utils.create_cutmix_or_mixup_collate_function(NUM_CLASSES)
    return DataLoader(
        TrainDataset(
            bioclim_data_dir=os.path.join(
                constants.DATA_DIR,
                "TimeSeries-Cubes",
                "TimeSeries-Cubes",
                "GLC24-PA-train-bioclimatic_monthly",
            ),
            landsat_data_dir=os.path.join(
                constants.DATA_DIR,
                "TimeSeries-Cubes",
                "TimeSeries-Cubes",
                "GLC24-PA-train-landsat_time_series",
            ),
            sentinel_data_dir=os.path.join(
                constants.DATA_DIR,
                "PA_Train_SatellitePatches_RGB",
                "pa_train_patches_rgb",
            ),
            metadata=pd.read_csv(
                os.path.join(constants.DATA_DIR, "GLC24_PA_metadata_train.csv")
            ),
            transform=transforms.Compose([transforms.ToTensor()]),
        ),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )


def test_load_from_config(config: MultiModalEnsembleConfig):
    """Create a test loader from config."""
    return DataLoader(
        TestDataset(
            bioclim_data_dir=os.path.join(
                constants.DATA_DIR,
                "TimeSeries-Cubes",
                "TimeSeries-Cubes",
                "GLC24-PA-test-bioclimatic_monthly",
            ),
            landsat_data_dir=os.path.join(
                constants.DATA_DIR,
                "TimeSeries-Cubes",
                "TimeSeries-Cubes",
                "GLC24-PA-test-landsat_time_series",
            ),
            sentinel_data_dir=os.path.join(
                constants.DATA_DIR,
                "PA_Test_SatellitePatches_RGB",
                "pa_test_patches_rgb",
            ),
            metadata=pd.read_csv(
                os.path.join(constants.DATA_DIR, "GLC24_PA_metadata_test.csv")
            ),
            transform=transforms.Compose([transforms.ToTensor()]),
        ),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
    )


class MultimodalEnsemble(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.landsat_norm = nn.LayerNorm([6, 4, 21])
        self.landsat_model = models.resnet18(weights=None)
        # Modify the first convolutional layer to accept 6 channels instead of 3
        self.landsat_model.conv1 = nn.Conv2d(
            6, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.landsat_model.maxpool = nn.Identity()

        self.bioclim_norm = nn.LayerNorm([4, 19, 12])
        self.bioclim_model = models.resnet18(weights=None)
        # Modify the first convolutional layer to accept 4 channels instead of 3
        self.bioclim_model.conv1 = nn.Conv2d(
            4, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bioclim_model.maxpool = nn.Identity()

        self.sentinel_model = models.swin_t(weights="IMAGENET1K_V1")
        # Modify the first layer to accept 4 channels instead of 3
        self.sentinel_model.features[0][0] = nn.Conv2d(
            4, 96, kernel_size=(4, 4), stride=(4, 4)
        )
        self.sentinel_model.head = nn.Identity()

        self.ln1 = nn.LayerNorm(1000)
        self.ln2 = nn.LayerNorm(1000)
        self.fc1 = nn.Linear(2768, 4096)
        self.fc2 = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, y, z):
        x = self.landsat_norm(x)
        x = self.landsat_model(x)
        x = self.ln1(x)

        y = self.bioclim_norm(y)
        y = self.bioclim_model(y)
        y = self.ln2(y)

        z = self.sentinel_model(z)

        xyz = torch.cat((x, y, z), dim=1)
        xyz = self.fc1(xyz)
        xyz = self.dropout(xyz)
        out = self.fc2(xyz)
        return out


def optimizer_from_config(model: torch.nn.Module, config: MultiModalEnsembleConfig):
    """Create an optimizer from config."""
    return torch.optim.AdamW(model.parameters(), lr=config.learning_rate)


def main(args: Sequence[str] | None = None):
    """Train/eval the model."""
    model = model_utils.WandbTrackedModel[MultiModalEnsembleConfig](
        checkpoint_prefix="baseline-sentinel-landsat-bioclim",
        config_class=MultiModalEnsembleConfig,
        model=lambda config: MultimodalEnsemble(NUM_CLASSES),
        train_loader=train_loader_from_config,
        test_loader=test_load_from_config,
        optimizer=optimizer_from_config,
        scheduler=functools.partial(CosineAnnealingLR, T_max=25),  # type:ignore
        loss_factory=torch.nn.BCEWithLogitsLoss,  # type:ignore
    )

    @torch.inference_mode()
    def transform(self: model_utils.WandbTrackedModel[MultiModalEnsembleConfig]):
        config = self.__safe_config
        device = torch.device(config.device)
        test_loader = self.test_loader(config)
        with torch.no_grad():
            surveys = []
            top_k_indices = None
            for survey_id, *data in tqdm(
                test_loader,
                leave=False,
                position=1,
            ):
                data = [d.to(device, dtype=torch.float32) for d in data]

                outputs = model(*data)
                predictions = torch.sigmoid(outputs).cpu().numpy()

                # Sellect top-25 values as predictions
                top_25 = np.argsort(-predictions, axis=1)[:, :25]
                if top_k_indices is None:
                    top_k_indices = top_25
                else:
                    top_k_indices = np.concatenate((top_k_indices, top_25), axis=0)

                surveys.extend(survey_id.cpu().numpy())
        data_concatenated = [" ".join(map(str, row)) for row in top_k_indices]

        return pd.DataFrame(
            {
                "surveyId": surveys,
                "predictions": data_concatenated,
            }
        )

    def save_submission(self):
        return self.transform().to_csv(
            os.path.join(
                constants.ROOT_DIR,
                "submissions",
                f"{self.checkpoint_prefix}-{self.__safe_config.run_id}.csv",
            ),
            index=False,
        )

    model.transform = transform
    model.save_submission = save_submission
    model.parse_args(args=args)


if __name__ == "__main__":
    main()
