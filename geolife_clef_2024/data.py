import os

import polars as pl

from geolife_clef_2024 import constants, datasets, type_aliases
from geolife_clef_2024.locations import location_information


def add_location_data(
    split: type_aliases.Dataset = "train",
    group: type_aliases.DatasetGroup = "PA",
) -> pl.LazyFrame:
    if not os.path.exists(
        location_data_path := os.path.join(constants.DATA_DIR, ".location-data.jsonl")
    ):
        df = pl.concat(
            (
                datasets.load_observation_data(),
                datasets.load_observation_data(group="P0"),
                datasets.load_observation_data(split="test"),
            ),
        )

        location_information(df).write_ndjson(location_data_path)
    reverse_geocoded_data = pl.scan_ndjson(location_data_path)
    return (
        datasets.load_observation_data(split=split, group=group)
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


