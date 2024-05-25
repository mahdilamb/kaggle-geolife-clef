"""Module containing utility functions for images."""

from typing import Protocol

import torch
from torch import nn


class ChannelWeightFunction(Protocol):
    def __call__(self, input: torch.Tensor, dim: int) -> torch.Tensor: ...


def adjust_channel_number(
    conv_2d: nn.Conv2d,
    channel_numbers: int | None = None,
    channel_weights: ChannelWeightFunction = torch.mean,
) -> nn.Conv2d:
    if channel_numbers is None:
        channel_numbers = conv_2d.in_channels
    conv_2d_weights = conv_2d.weight
    out_conv_2d = nn.Conv2d(
        in_channels=channel_numbers,
        **{
            attr: getattr(
                conv_2d,
                attr,
            )
            for attr in (
                "out_channels",
                "kernel_size",
                "stride",
                "padding",
                "dilation",
                "groups",
                "padding_mode",
            )
        },
        bias=conv_2d.bias is not None,
        device=conv_2d_weights.device,
        dtype=conv_2d_weights.dtype,
    )
    with torch.no_grad():
        out_conv_2d.weight.copy_(
            torch.concat(
                (
                    conv_2d_weights,
                    channel_weights(conv_2d_weights, dim=1)[:, None, ...],
                ),
                dim=1,
            )
        )
    out_conv_2d.requires_grad = conv_2d.requires_grad_()
    return out_conv_2d
