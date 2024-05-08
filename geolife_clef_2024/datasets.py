"""Dataset loaders."""

import os
from typing import Literal, NoReturn, overload

import pandas as pd
import polars as pl
from jupyter_utils import polars as pl_utils

from geolife_clef_2024 import constants, metadata_schemas, type_aliases


@overload
def load_metadata(
    *,
    split: type_aliases.Dataset = "train",
    group: type_aliases.DatasetGroup = "PA",
    as_pandas: Literal[False] = ...,
) -> pl.LazyFrame: ...
@overload
def load_metadata(
    *,
    split: type_aliases.Dataset = "train",
    group: type_aliases.DatasetGroup = "PA",
    as_pandas: Literal[True] = True,
) -> pd.DataFrame: ...


def load_metadata(
    *,
    split: type_aliases.Dataset = "train",
    group: type_aliases.DatasetGroup = "PA",
    as_pandas: bool = False,
) -> pl.LazyFrame | pd.DataFrame | NoReturn:
    """Load the metadata for each dataset."""
    if group == "P0" and split == "test":
        raise ValueError(f"No such dataset {group}/{split}")
    if as_pandas:
        return pd.read_csv(
            os.path.join(constants.DATA_DIR, f"GLC24_{group}_metadata_{split}.csv"),
            dtype=pl_utils.to_pandas_schema(
                getattr(metadata_schemas, f"{group}_{split.upper()}")
            ),
            engine="pyarrow",
        )
    schema = {
        k: pl.String
        if isinstance(v, pl.Enum)
        else (pl.Float64 if isinstance(v, pl.Int64) or v == pl.Int64 else v)
        for k, v in getattr(metadata_schemas, f"{group}_{split.upper()}").items()
    }
    return pl.scan_csv(
        os.path.join(constants.DATA_DIR, f"GLC24_{group}_metadata_{split}.csv"),
        schema=schema,
    ).cast(
        {
            k: v
            for k, v in getattr(metadata_schemas, f"{group}_{split.upper()}").items()
            if isinstance(v, pl.Enum | pl.Int64) or v == pl.Int64
        }
    )
