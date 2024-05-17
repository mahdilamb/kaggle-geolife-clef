"""Shared models."""

import torch
import torchvision.models as models
from torch import nn


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
        self.ln = nn.LayerNorm(1000)
        self.fc1 = nn.Linear(1000, 2056)
        self.fc2 = nn.Linear(2056, num_classes)

    def forward(self, x: torch.Tensor):
        """Apply the model to the input tensor."""
        x = self.norm_input(x)
        x = self.resnet18(x)
        x = self.ln(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
