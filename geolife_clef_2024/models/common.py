"""Shared models."""

from typing import Literal, TypeAlias, TypeVar

import torch
import torchvision.models as models
from torch import nn
from torchvision.models import swin_transformer

SwinTransformer: TypeAlias = Literal[
    "swin_t",
    "swin_s",
    "swin_b",
    "swin_v2_t",
    "swin_v2_s",
    "swin_v2_b",
]
T = TypeVar("T", bound=nn.Module)


class ModifiedResNet18(nn.Module):
    """Modified ResNet18 model."""

    def __init__(self, input_shape: list[int], num_classes: int):
        """Create a new modified ResNet18 model."""
        super().__init__()

        self.norm_input = nn.LayerNorm(input_shape)
        self.resnet18 = models.resnet18(weights=None)
        self.resnet18.conv1 = nn.Conv2d(
            input_shape[0], 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.resnet18.maxpool = nn.Identity()
        self.head: nn.Module = nn.Sequential(
            nn.LayerNorm(1000), nn.Linear(1000, 2056), nn.Linear(2056, num_classes)
        )

    def forward(self, x: torch.Tensor):
        """Apply the model to the input tensor."""
        x = self.norm_input(x)
        x = self.resnet18(x)
        x = self.head(x)
        return x


def swin_transformer_model(
    num_classes: int,
    model: SwinTransformer,
    weights: Literal["IMAGENET1K_V1"] = "IMAGENET1K_V1",
):
    """Get a swin transformer model, modified for a specific number of classes.."""
    swin_model: swin_transformer.SwinTransformer = getattr(models, model)(
        weights=weights
    )
    swin_model.features[0][0] = nn.Conv2d(4, 96, kernel_size=(4, 4), stride=(4, 4))
    swin_model.head = nn.Linear(
        in_features=768,
        out_features=num_classes,
        bias=True,
    )
    return swin_model
