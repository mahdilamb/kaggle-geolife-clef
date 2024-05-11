"""Tests for dataset loaders."""

import functools
from collections.abc import Callable
from typing import (
    Generic,
    NamedTuple,
    TypeVar,
)

import numpy as np
import pandas as pd
import polars as pl
import pytest
from torch.utils.data import Dataset

from geolife_clef_2024 import datasets

T = TypeVar("T")


class DatasetTest(NamedTuple, Generic[T]):
    """Test set for dataset loading."""

    fn: Callable[[], T] | str
    tests: dict[Callable[[T], bool], str]
    errors: tuple[type[Exception], ...]


@pytest.mark.parametrize(
    ("fn", "tests", "errors"),
    [
        DatasetTest(
            fn=functools.partial(datasets.load_observation_data, split="train"),
            tests={
                lambda val: isinstance(
                    val, pl.LazyFrame
                ): "Expected the default dataset to be loaded as a polars dataframe."
            },
            errors=(),
        ),
        DatasetTest(
            fn=functools.partial(
                datasets.load_observation_data, split="test", as_pandas=True
            ),
            tests={
                lambda val: isinstance(
                    val, pd.DataFrame
                ): "Expected the pandas dataframe to load as a pandas dataframe."
            },
            errors=(),
        ),
        DatasetTest(
            fn=functools.partial(
                datasets.load_observation_data, split="test", group="P0"
            ),
            tests={},
            errors=(ValueError,),
        ),
        DatasetTest(
            fn=functools.partial(datasets.load_satellite_patches, split="test"),
            tests={
                lambda val: isinstance(
                    val, Dataset
                ): "Expected the satellite patches to be a dataset.",
                lambda val: isinstance(
                    val[0][1], np.ndarray
                ): "Expected the image to be an ndarray.",
                lambda val: val[0][1].shape[-1]
                == 4: "Expected to get an RGB+NIR image.",
            },
            errors=(),
        ),
    ],
)
def test_load_dataset(
    fn: Callable[[], T],
    tests: dict[Callable[[T], bool], str],
    errors: tuple[type[Exception], ...],
):
    """Test the various dataset loaders."""
    actual: T
    if errors:
        with pytest.raises(errors):
            actual = fn()
    else:
        actual = fn()
    for test_fn, message in tests.items():
        assert test_fn(actual), message


if __name__ == "__main__":
    pytest.main([__file__])
