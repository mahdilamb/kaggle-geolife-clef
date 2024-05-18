"""Utility functions for working with datasets."""

import functools
import operator
from collections.abc import Sequence
from unittest import mock

import PIL.Image
import torch
from torch.utils._pytree import tree_flatten, tree_unflatten
from torch.utils.data import Dataset, default_collate
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2._utils import (
    has_any,
)


class DatasetSlicerMetaClass(type):
    """Metaclass for a dataset slicer."""

    def __getitem__(self, idx: slice | Sequence[int] | int):
        """Apply the slice to a dataset."""

        def slicer(dataset: Dataset):
            def new_get_item(self, *args, **kwargs):
                result = dataset.__getitem__(*args, **kwargs)
                return result[idx]

            return type(
                f"Sliced{dataset.__class__}",
                (Dataset,),
                {
                    "__getitem__": new_get_item,
                    "__len__": dataset.__len__,
                    "__add__": dataset.__add__,
                },
            )()

        return slicer


class DatasetSlicer(metaclass=DatasetSlicerMetaClass):
    """Class from which to use the slicer."""


class FlattenedDataset(Dataset):
    """Stacked dataset where the resulting sequences are squeezed."""

    def __init__(self, dataset: Dataset, *datasets: Dataset):
        """Create a stacked dataset where the resulting sequences are squeezed."""
        self._datasets = dataset, *datasets
        if not all(len(dataset) == len(_) for _ in datasets):
            raise ValueError("All datasets must be the same length.")

    def __getitem__(self, index) -> tuple[torch.Tensor, ...]:
        """Get the items at the given index."""
        return tuple(
            functools.reduce(
                operator.iconcat,
                (
                    (ds,) if isinstance(ds, torch.Tensor) else ds
                    for ds in (dataset[index] for dataset in self._datasets)
                ),
                [],
            )
        )

    def __len__(self):
        """Get the length of the dataset."""
        return len(self._datasets[0])


def _modified_base_mixup_cutmix__mixup_label(
    self, label: torch.Tensor, *, lam: float
) -> torch.Tensor:
    # Modified to not do OHE
    if not label.dtype.is_floating_point:
        label = label.float()
    return label.roll(1, 0).mul_(1.0 - lam).add_(label.mul(lam))


def _modified_base_mixup_cutmix_forward(self, *inputs):
    inputs = inputs if len(inputs) > 1 else inputs[0]
    flat_inputs, spec = tree_flatten(inputs)
    needs_transform_list = self._needs_transform_list(flat_inputs)

    if has_any(flat_inputs, PIL.Image.Image, tv_tensors.BoundingBoxes, tv_tensors.Mask):
        raise ValueError(
            f"{type(self).__name__}() does not support PIL images, "
            + "bounding boxes and masks."
        )

    labels = inputs[-1]  # Modified as labels are always the last
    if not isinstance(labels, torch.Tensor):
        raise ValueError(
            f"The labels must be a tensor, but got {type(labels)} instead."
        )

    params = {
        "labels": labels,
        "batch_size": labels.shape[0],
        **self._get_params(
            [
                inpt
                for (inpt, needs_transform) in zip(
                    flat_inputs, needs_transform_list, strict=False
                )
                if needs_transform
            ]
        ),
    }

    needs_transform_list[
        next(idx for idx, inpt in enumerate(flat_inputs) if inpt is labels)
    ] = True
    flat_outputs = [
        self._transform(inpt, params) if needs_transform else inpt
        for (inpt, needs_transform) in zip(
            flat_inputs, needs_transform_list, strict=False
        )
    ]

    return tree_unflatten(flat_outputs, spec)


def create_cutmix_or_mixup_collate_function(num_classes: int):
    """Create a CutMix/Mixup collate function."""
    cutmix = v2.CutMix(num_classes=num_classes)
    mixup = v2.MixUp(num_classes=num_classes)
    cutmix_or_mixup = v2.RandomChoice((cutmix, mixup))

    def collate_fn(batch):
        survey_id, *tensors = default_collate(batch)
        with (
            mock.patch(
                "torchvision.transforms.v2._augment._BaseMixUpCutMix.forward",
                _modified_base_mixup_cutmix_forward,
            ),
            mock.patch(
                "torchvision.transforms.v2._augment._BaseMixUpCutMix._mixup_label",
                _modified_base_mixup_cutmix__mixup_label,
            ),
        ):
            return survey_id, *cutmix_or_mixup(*tensors)

    return collate_fn
