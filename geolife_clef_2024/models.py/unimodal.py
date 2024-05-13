from collections.abc import Sequence
from typing import Literal, TypeAlias

import polars as pl

from geolife_clef_2024 import data

Column: TypeAlias = Literal["country", "district", "region"]


class NaiveTopForAll:
    """Model that does no training.

    Mostly akin to https://www.kaggle.com/code/picekl/naive-approach-baselines-0-20567.
    """

    def __init__(
        self,
        top_by: Column | Sequence[Column] | None = None,
        top_k: int = 25,
    ) -> None:
        self._results: str | None | pl.LazyFrame = None
        self._top_k = top_k
        self._top_for: Sequence[Column] | None = (
            (top_by,)
            if top_by is not None and not isinstance(top_by, Sequence)
            else top_by
        )

    def train(self):
        train_df = data.add_location_data()
        if self._top_for is None:
            self._results = (
                train_df.group_by("speciesId")
                .agg(pl.len())
                .top_k(self._top_k, by="len")
                .select("speciesId")
                .collect()
                .transpose()
                .select(pl.concat_str(pl.all(), separator=" "))
                .item()
            )
        else:
            self._results = (
                train_df.group_by("speciesId", *self._top_for)
                .agg(pl.len())
                .sort("len", descending=True)
                .group_by(*self._top_for, maintain_order=True)
                .agg(pl.col("speciesId"))
                .with_columns(
                    pl.col("speciesId")
                    .list.slice(0, self._top_k)
                    .cast(pl.List(pl.String))
                    .list.join(" ")
                )
            )

    def test(self):
        if self._results is None:
            self.train()
        if self._top_for is None:
            return data.add_location_data("test").select(
                "surveyId", predictions=pl.lit(self._results)
            )
        return (
            data.add_location_data("test")
            .select("surveyId", *self._top_for)
            .join(self._results, on=self._top_for, how="left")
            .select("surveyId", pl.col("speciesId").alias("predictions"))
        )
