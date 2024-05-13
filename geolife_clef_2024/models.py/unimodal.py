"""Module containing unimodal models."""

from collections.abc import Sequence
from types import NoneType
from typing import Literal, TypeAlias

import polars as pl

from geolife_clef_2024 import data, datasets

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
        """Create a Naive model that will return the same k top for each selection.

        The selection is chosen by `top_by`.

        Args:
            top_by (Column | Sequence[Column] | None, optional): The columns to get the
                top k for. Defaults to None.
            top_k (int, optional): The number of top K elements to select. Defaults to
                25.
        """
        self._results: str | None | pl.LazyFrame = None
        self._top_k = top_k
        self._top_by: Sequence[Column] | None = (
            top_by if isinstance(top_by, Sequence | NoneType) else (top_by,)
        )

    def fit(self):
        """Find the top K elements."""
        train_df = data.with_location_data(datasets.load_observation_data())
        if self._top_by is None:
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
                train_df.group_by("speciesId", *self._top_by)
                .agg(pl.len())
                .sort("len", descending=True)
                .group_by(*self._top_by, maintain_order=True)
                .agg(pl.col("speciesId"))
                .with_columns(
                    pl.col("speciesId")
                    .list.slice(0, self._top_k)
                    .cast(pl.List(pl.String))
                    .list.join(" ")
                )
            )

    def transform(self):
        """Append the top K predictions."""
        if self._results is None:
            self.fit()
        test_df = datasets.load_observation_data(split="test")
        if self._top_by is None:
            return data.with_location_data(test_df).select(
                "surveyId", predictions=pl.lit(self._results)
            )
        return (
            data.with_location_data(test_df)
            .select("surveyId", *self._top_by)
            .join(self._results, on=self._top_by, how="left")
            .select("surveyId", pl.col("speciesId").alias("predictions"))
        )
