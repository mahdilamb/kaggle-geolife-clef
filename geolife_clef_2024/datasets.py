"""Dataset loaders."""

import os
from typing import Literal, NoReturn, overload

import pandas as pd
import polars as pl

from geolife_clef_2024 import _type_aliases, constants, metadata_schemas


@overload
def load_metadata(
    dataset: _type_aliases.Dataset = "train",
    group: _type_aliases.DatasetGroup = "PA",
    as_pandas: Literal[False] = ...,
) -> pl.LazyFrame: ...
@overload
def load_metadata(
    dataset: _type_aliases.Dataset = "train",
    group: _type_aliases.DatasetGroup = "PA",
    as_pandas: Literal[True] = True,
) -> pd.DataFrame: ...


def load_metadata(
    dataset: _type_aliases.Dataset = "train",
    group: _type_aliases.DatasetGroup = "PA",
    as_pandas: bool = False,
) -> pl.LazyFrame | pd.DataFrame | NoReturn:
    """Load the metadata for each dataset."""
    if group == "P0" and dataset == "test":
        raise ValueError(f"No such dataset {group}/{dataset}")
    if as_pandas:
        return pd.read_csv(
            os.path.join(constants.DATA_DIR, f"GLC24_{group}_metadata_{dataset}.csv"),
        )
    schema = {
        k: pl.String
        if isinstance(v, pl.Enum)
        else (pl.Float64 if isinstance(v, pl.Int64) or v == pl.Int64 else v)
        for k, v in getattr(metadata_schemas, f"{group}_{dataset.upper()}").items()
    }
    return pl.scan_csv(
        os.path.join(constants.DATA_DIR, f"GLC24_{group}_metadata_{dataset}.csv"),
        schema=schema,
    ).cast(
        {
            k: v
            for k, v in getattr(metadata_schemas, f"{group}_{dataset.upper()}").items()
            if isinstance(v, pl.Enum | pl.Int64) or v == pl.Int64
        }
    )
