"""Module containing torch datasets."""

import os

import numpy as np
from PIL import Image

from geolife_clef_2024 import constants, type_aliases


def img_paths(split: type_aliases.Dataset, survey_id: str | int) -> tuple[str, str]:
    """Get the NIR and RGB image paths.

    Returns None if the associated path's can't be found.
    """
    survey_id = str(survey_id)
    ending = survey_id[-2:], survey_id[-4:-2], f"{survey_id}.jpeg"
    nir_path = os.path.join(
        constants.DATA_DIR,
        f"PA_{split.title()}_SatellitePatches_NIR",
        f"pa_{split}_patches_nir",
        *ending,
    )
    return nir_path, os.path.join(
        constants.DATA_DIR,
        f"PA_{split.title()}_SatellitePatches_RGB",
        f"pa_{split}_patches_rgb",
        *ending,
    )


def load_patch(
    survey_id: str,
    split: type_aliases.Dataset,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a specific patch.

    Note that, when nir is included, it will be the first channel.
    """
    nir_path, rgb_path = img_paths(survey_id=survey_id, split=split)
    return np.array(Image.open(nir_path)), np.array(Image.open(rgb_path))


def load_rgb_patch(
    survey_id: str,
    split: type_aliases.Dataset,
) -> np.ndarray:
    """Load a specific patch (RGB only)."""
    _, rgb_path = img_paths(survey_id=survey_id, split=split)
    return np.array(Image.open(rgb_path))


def load_nir_patch(
    survey_id: str,
    split: type_aliases.Dataset,
) -> np.ndarray:
    """Load a specific patch (NIR only)."""
    nir_path, _ = img_paths(survey_id=survey_id, split=split)
    return np.array(Image.open(nir_path))
