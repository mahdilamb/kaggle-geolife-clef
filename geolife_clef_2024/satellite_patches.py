"""Module containing torch datasets."""

import os

import numpy as np
from skimage import io

from geolife_clef_2024 import constants, type_aliases


def img_paths(split: type_aliases.Dataset, survey_id: str) -> tuple[str, str] | None:
    """Get the NIR and RGB image paths.

    Returns None if the associated path's can't be found.
    """
    ending = survey_id[-2:], survey_id[-4:-2], f"{survey_id}.jpeg"
    if os.path.exists(
        nir_path := os.path.join(
            constants.DATA_DIR,
            f"PA_{split.title()}_SatellitePatches_NIR",
            f"pa_{split}_patches_nir",
            *ending,
        )
    ):
        return nir_path, os.path.join(
            constants.DATA_DIR,
            f"PA_{split.title()}_SatellitePatches_RGB",
            f"pa_{split}_patches_rgb",
            *ending,
        )
    return None


def load_patch(
    survey_id: str,
    split: type_aliases.Dataset,
    include_nir: bool = True,
) -> np.ndarray:
    """Load a specific patch.

    Note that, when nir is included, it will be the first channel.
    """
    paths = img_paths(survey_id=survey_id, split=split)
    if paths:
        nir_path, rgb_path = paths
    else:
        raise FileExistsError(f"No satellite patches for surveyId: {survey_id}")
    img = io.imread(rgb_path)
    if include_nir:
        img = np.dstack((io.imread(nir_path), img))
    return img
