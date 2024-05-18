"""Utility functions for working with datasets."""

import functools
import operator
from collections.abc import Sequence

import torch
from torch.utils.data import Dataset


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
