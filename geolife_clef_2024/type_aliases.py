"""Type aliases used throughout the package."""

from typing import Literal, TypeAlias

Dataset: TypeAlias = Literal["train", "test"]
DatasetGroup: TypeAlias = Literal["PA", "P0"]
TargetFeature: TypeAlias = Literal["surveyId"]
P0Feature: TypeAlias = Literal[
    "publisher",
    "year",
    "month",
    "day",
    "lat",
    "lon",
    "geoUncertaintyInM",
    "taxonRank",
    "date",
    "dayOfYear",
    "speciesId",
]
PAFeature: TypeAlias = Literal[
    "lon",
    "lat",
    "year",
    "geoUncertaintyInM",
    "areaInM2",
    "region",
    "country",
    "speciesId",
]
Raster: TypeAlias = Literal[
    "climate", "elevation", "human_footprint", "land_cover", "soil_grids"
]
TimeGranularity: TypeAlias = Literal["monthly", "all"]
SatelliteFormat: TypeAlias = Literal["pl", "pd", "pt"]
