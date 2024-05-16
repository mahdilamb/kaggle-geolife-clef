"""Module containing utilities for species ids."""


import polars as pl
import torch
from magicbox import polars as pl_utils

from geolife_clef_2024 import datasets


def encoded_species_idxs():
    """Get the mapping of species ids to survey ids.

    Note: the species ids will be in OHE.
    """
    df = datasets.load_observation_data(split="train")
    all_ids = df.select(pl.col("speciesId").unique().sort()).collect()
    all_ids = all_ids.with_columns(
        pl.int_range(0, (num_classes := all_ids.height), eager=True)
    )
    mapping = pl_utils.to_dict(
        df.group_by("surveyId").agg(
            speciesId=pl.col("speciesId").replace(pl_utils.to_dict(all_ids)).unique()
        )
    )

    def encode(survey_id: str | int, dtype=torch.float32):
        survey_id = int(survey_id)
        label = torch.zeros(num_classes, dtype=dtype)
        label[mapping[survey_id]] = 1
        return label

    return encode
