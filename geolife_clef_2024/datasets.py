"""Dataset loaders."""

import functools
import glob
import os
import re
from collections.abc import Sequence
from typing import Literal, NoReturn, overload

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.utils
import torch.utils.data
from magicbox import polars as pl_utils
from torch.utils.data import Dataset

from geolife_clef_2024 import (
    constants,
    metadata_schemas,
    satellite_patches,
    type_aliases,
)


@functools.cache
def all_satellite_patch_survey_ids(
    split: type_aliases.Dataset,
) -> Sequence[str]:
    """Get the survey_ids of all the satellite patches."""
    return pl.Series(
        values=glob.iglob(
            os.path.join(
                constants.DATA_DIR,
                f"PA_{split.title()}_SatellitePatches_RGB",
                f"pa_{split}_patches_rgb",
                "**",
                "**",
                "**.jpeg",
            )
        )
    ).str.extract(r"([^\/]*?)\.")  # type: ignore


@functools.cache
def survey_ids_with_satellite_patches(
    split: type_aliases.Dataset,
) -> Sequence[str]:
    """Get a sequence of survey ids that have an associated patch."""
    return (
        load_observation_data(split=split)
        .select(pl.col("surveyId").cast(pl.String))
        .filter(pl.col("surveyId").is_in(all_satellite_patch_survey_ids(split=split)))
        .sort("surveyId")
        .collect()
        .to_series()
    )  # type: ignore


class SatellitePatchesDataset(Dataset[tuple[str, np.ndarray]]):
    """Dataset containing the satellite patches."""

    def __init__(
        self,
        split: type_aliases.Dataset,
        include_nir: bool = True,
    ) -> None:
        """Create a new dataset for a specific dataset split."""
        self._include_nir = include_nir
        self._split = split
        self._survey_ids = survey_ids_with_satellite_patches(split=split)

    def __len__(self) -> int:
        """Get the number of patches."""
        return len(self._survey_ids)

    def __getitem__(self, idx: int) -> tuple[str, np.ndarray]:
        """Get a patch stack from a specific survey."""
        survey_id = self._survey_ids[idx]
        return survey_id, satellite_patches.load_patch(
            survey_id=survey_id, split=self._split, include_nir=self._include_nir
        )


class TimeSeriesDataset(Dataset[tuple[str, torch.Tensor]]):
    """Dataset for the satellite time series."""

    def __init__(
        self,
        split: type_aliases.Dataset,
        which: Literal["landsat_time_series", "bioclimatic_monthly"],
        **torch_kwargs,
    ):
        """Create a time series dataset."""
        self._split = split
        self._load_kwargs = torch_kwargs
        self._files = sorted(
            glob.iglob(
                os.path.join(
                    constants.DATA_DIR,
                    "TimeSeries-Cubes",
                    "TimeSeries-Cubes",
                    f"GLC24-PA-{split}-{which}",
                    "*.pt",
                )
            )
        )

    def __len__(self) -> int:
        """Get the number of elements in the time series dataset."""
        return len(self._files)

    def __getitem__(self, index) -> tuple[str, torch.Tensor]:
        """Get an item from the time series."""
        file = self._files[index]
        survey_id = next(re.finditer(r"(\d+)_cube", file)).group(1)
        arr = torch.load(file, **self._load_kwargs)
        return survey_id, arr


@overload
def load_observation_data(
    *,
    split: type_aliases.Dataset = "train",
    group: type_aliases.DatasetGroup = "PA",
    as_pandas: Literal[False] = ...,
) -> pl.LazyFrame: ...
@overload
def load_observation_data(
    *,
    split: type_aliases.Dataset = "train",
    group: type_aliases.DatasetGroup = "PA",
    as_pandas: Literal[True] = True,
) -> pd.DataFrame: ...


def load_observation_data(
    *,
    split: type_aliases.Dataset = "train",
    group: type_aliases.DatasetGroup = "PA",
    as_pandas: bool = False,
) -> pl.LazyFrame | pd.DataFrame | NoReturn:
    """Load the metadata for each dataset.

    The species related training data comprises:

    1. Presence-Absence (PA) surveys: including around 90 thousand surveys with roughly
        10,000 species of the European flora. The presence-absence data (PA) is provided
        to compensate for the problem of false-absences of PO data and calibrate models
        to avoid associated biases.
    2. Presence-Only (PO) occurrences: combines around five million observations from
        numerous datasets gathered from the Global Biodiversity Information Facility
        (GBIF, www.gbif.org). This data constitutes the larger piece of the training
        data and covers all countries of our study area, but it has been sampled
        opportunistically (without standardized sampling protocol), leading to various
        sampling biases. The local absence of a species among PO data doesn't mean it
        is truly absent. An observer might not have reported it because it was difficult
        to "see" it at this time of the year, to identify it as not a monitoring target,
        or just unattractive.
    There are two CSVs with species occurrence data on the Seafile available for
        training. The detailed description is provided again on SeaFile in separate
        ReadME files in relevant folders.

    - The PO metadata are available in
        PresenceOnlyOccurences/GLC24_PO_metadata_train.csv.
    - The PA metadata are available in
        PresenceAbsenceSurveys/GLC24_PA_metadata_train.csv.
    """
    if group == "P0" and split == "test":
        raise ValueError(f"No such dataset {group}/{split}")
    csv_file = os.path.join(constants.DATA_DIR, f"GLC24_{group}_metadata_{split}.csv")
    if as_pandas:
        return pd.read_csv(
            csv_file,
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
    return pl.scan_csv(csv_file, schema=schema).cast(
        {
            k: v
            for k, v in getattr(metadata_schemas, f"{group}_{split.upper()}").items()
            if isinstance(v, pl.Enum | pl.Int64) or v == pl.Int64
        }
    )


def load_satellite_patches(
    split: type_aliases.Dataset = "train", include_nir: bool = True
) -> SatellitePatchesDataset:
    """Get a dataset loader for the satellite patches.

    1280mx1280m RGB and NIR patches (four bands) centered at the observation geolocation
        and taken the same year. The patches are compressed in two zip files
        (patchs_rgb.zip, patchs_nir.zip) accessible in folder /SatelliteImages/.

    Format: 128x128 JPEG images, a color JPEG file for RGB data and a grayscale one for
        Near-Infrared.
    Resolution: 10 meters per pixel
    Source: Sentinel2 remote sensing data pre-processed by the Ecodatacube platform
    Access: First, one must download and decompress the provided zip files. Each JPEG
        file corresponds to a unique observation location (via "surveyId"). To load the
        RGB or NIR patch for a selected observation, take the "surveyId" from any
        occurrence CSV and load it following this rule --> '…/CD/AB/XXXXABCD.jpeg'. For
        example, the image location for the surveyId 3018575 is "./75/85/3018575.jpeg".
        For all "surveyId" with less than four digits, you can use a similar rule. For a
        "surveyId" 1 is "./1/1.jpeg".
    """
    return SatellitePatchesDataset(include_nir=include_nir, split=split)


@overload
def load_environmental_rasters(
    *,
    raster: type_aliases.Raster,
    scale: type_aliases.TimeGranularity = "monthly",
    split: type_aliases.Dataset = "train",
    group: type_aliases.DatasetGroup = "PA",
    as_pandas: Literal[False] = ...,
) -> pl.LazyFrame: ...
@overload
def load_environmental_rasters(
    *,
    raster: type_aliases.Raster,
    scale: type_aliases.TimeGranularity = "monthly",
    split: type_aliases.Dataset = "train",
    group: type_aliases.DatasetGroup = "PA",
    as_pandas: Literal[True] = True,
) -> pd.DataFrame: ...


def load_environmental_rasters(
    *,
    raster: type_aliases.Raster,
    scale: type_aliases.TimeGranularity = "monthly",
    split: type_aliases.Dataset = "train",
    group: type_aliases.DatasetGroup = "PA",
    as_pandas: bool = False,
) -> pd.DataFrame | pl.LazyFrame:
    """Load the environmental rasters.

    Args:
        raster (type_aliases.Raster): The raster type.
        scale (type_aliases.TimeGranularity, optional): The time scale (ignored
            if raster is not `climate`). Defaults to "monthly".
        split (type_aliases.Dataset, optional): Which split to use. Defaults to "train".
        group (type_aliases.DatasetGroup, optional): Which group to use.
            Defaults to "PA".
        as_pandas (bool, optional): Whether to output as pandas . Defaults to False.

    """
    if group == "P0" and split == "test":
        raise ValueError(f"No such dataset {group}/{split}")

    csv_dir = os.path.join(
        constants.DATA_DIR,
        "EnvironmentalRasters",
        "EnvironmentalRasters",
        (
            {
                "climate": "Climate",
                "elevation": "Elevation",
                "human_footprint": "Human Footprint",
                "land_cover": "LandCover",
                "soil_grids": "SoilGrids",
            }
        )[raster],
    )
    if raster == "climate":
        csv_dir = os.path.join(
            csv_dir,
            "Monthly" if scale == "monthly" else "Average 1981-2010",
        )
    csv_file = f"GLC24-{group}-{split}-"
    csv_file += {
        "climate": "bioclimatic",
        "elevation": "elevation",
        "human_footprint": "human_footprint",
        "land_cover": "landcover",
        "soil_grids": "soilgrids",
    }[raster]
    if raster == "climate" and scale == "monthly":
        csv_file += "_monthly"
    csv_file = os.path.join(csv_dir, f"{csv_file}.csv")
    if as_pandas:
        return pd.read_csv(
            os.path.join(csv_file),
            engine="pyarrow",
        )
    return pl.scan_csv(csv_file)


@overload
def load_satellite_time_series(
    *,
    split: type_aliases.Dataset = "train",
    group: type_aliases.DatasetGroup = "PA",
    format: Literal["pt"] = ...,
) -> torch.utils.data.Dataset: ...
@overload
def load_satellite_time_series(
    *,
    split: type_aliases.Dataset = "train",
    group: type_aliases.DatasetGroup = "PA",
    format: Literal["pd"] = "pd",
) -> pd.DataFrame: ...
@overload
def load_satellite_time_series(
    *,
    split: type_aliases.Dataset = "train",
    group: type_aliases.DatasetGroup = "PA",
    format: Literal["pl"] = "pl",
) -> pl.LazyFrame: ...


def load_satellite_time_series(
    *,
    split: type_aliases.Dataset = "train",
    group: type_aliases.DatasetGroup = "PA",
    format: type_aliases.SatelliteFormat = "pt",
) -> pl.LazyFrame | pd.DataFrame | torch.utils.data.Dataset | NoReturn:
    """Load the satellite time series.

    Each observation is associated with the time series of the satellite median point
        values over each season since the winter of 1999 for six satellite bands (R, G,
        B, NIR, SWIR1, and SWIR2). This data carries a high-resolution local signature
        of the past 20 years' succession of seasonal vegetation changes, potential
        extreme natural events (fires), or land use changes.

    Format1: Six CSV files, one per band. The corresponds to the "surveyId," and the
        columns are the 84 seasons from winter 2000 until autumn 2020.
    Format2: . TimeSeries-Cubes - The above-mentioned CSV aggregated into 3d tensors
        with axes as BAND, QUARTER, and YEAR.
    Resolution: The original satellite data has a resolution of 30m per pixel
    Source: Landsat remote sensing data pre-processed by the Ecodatacube platform
    Access: /SatelliteTimeSeries/

    """
    if format == "pt":
        return TimeSeriesDataset(split=split, which="landsat_time_series")
    csv_files = (
        (file, next(re.finditer(r"([a-z0-9]*)\.", file)).group(1))
        for file in glob.iglob(
            os.path.join(
                constants.DATA_DIR,
                f"{group}-{split}-landsat_time_series",
                f"GLC24-{group}-{split}-landsat_time_series-*.csv",
            )
        )
    )
    result = pl.DataFrame()
    for csv_file, channel in csv_files:
        df = pl.scan_csv(csv_file)
        result = result.vstack(
            df.melt(
                id_vars="surveyId",
                value_vars=df.columns[1:],
                value_name="median",
                variable_name="raw_season",
            )
            .with_columns(channel=pl.lit(channel))
            .collect()
        )
    result = result.lazy().with_columns(
        year=pl.col("raw_season").str.slice(0, 4).cast(pl.Int64),
        season=pl.col("raw_season")
        .str.slice(5)
        .replace({1: "winter", 2: "spring", 3: "summer", 4: "fall"})
        .cast(pl.Enum(("spring", "summer", "fall", "winter"))),
        channel=pl.col("channel").cast(
            pl.Enum(("red", "green", "blue", "nir", "swir1", "swir2"))
        ),
    )
    if format == "pl":
        return result
    else:
        return result.collect().to_pandas().replace(np.nan, pd.NA)
