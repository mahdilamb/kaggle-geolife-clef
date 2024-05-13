import polars as pl
import reverse_geocoder
from magicbox import polars as pl_utils


def location_information(
    df: pl.LazyFrame | pl.DataFrame,
) -> pl.DataFrame:
    df = df.lazy()
    coords = df.select(raw_lat=pl.col("lat"), raw_lon=pl.col("lon")).unique().collect()

    return pl.DataFrame(
        reverse_geocoder.search(
            tuple(
                map(
                    tuple,
                    zip(
                        pl_utils.column_to_series(coords, "raw_lat"),
                        pl_utils.column_to_series(coords, "raw_lon"),
                        strict=True,
                    ),
                )
            ),
            mode=1,
        ),
        schema_overrides={"lat": pl.Float64, "lon": pl.Float64},
    ).hstack(coords)
