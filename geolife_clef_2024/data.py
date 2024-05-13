"""Module containing additional data inputs."""

import os

import polars as pl

from geolife_clef_2024 import constants, datasets
from geolife_clef_2024.locations import location_information


def with_location_data(df: pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
    """Add reverse geocoded data to a data frame."""
    if not os.path.exists(
        location_data_path := os.path.join(constants.DATA_DIR, ".location-data.jsonl")
    ):
        all_df = pl.concat(
            (
                datasets.load_observation_data(),
                datasets.load_observation_data(group="P0"),
                datasets.load_observation_data(split="test"),
            ),
        )

        location_information(all_df).write_ndjson(location_data_path)
    reverse_geocoded_data = pl.scan_ndjson(location_data_path)
    return (
        df.lazy()
        .join(
            reverse_geocoded_data,
            left_on=("lat", "lon"),
            right_on=("raw_lat", "raw_lon"),
            how="left",
        )
        .select(pl.exclude("lat", "lon"))
        .rename(
            {
                "lat_right": "lat",
                "lon_right": "lon",
                "cc": "countryCode",
                "admin1": "county",
                "admin2": "district",
                "name": "locationName",
            }
        )
    )
