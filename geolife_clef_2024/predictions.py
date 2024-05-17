import polars as pl


def format_predictions(df: pl.LazyFrame | pl.DataFrame):
    df = df.lazy().rename(
        dict(zip(df.columns, ("surveyId", "predictions"), strict=False))
    )
    return df.select(
        "surveyId", pl.col("predictions").cast(pl.List(pl.String)).list.join(" ")
    ).collect()


def save_predictions(path: str, df: pl.LazyFrame | pl.DataFrame):
    format_predictions(df).rechunk().write_csv(path)
