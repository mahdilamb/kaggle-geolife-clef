"""Constants used throughout the package."""

import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))  # The root of the package
ROOT_DIR = os.path.dirname(PACKAGE_DIR)  # The parent to the package
DATA_DIR = os.getenv(
    "GEOLIFE_CLEF_DATA", os.path.join(ROOT_DIR, "data")
)  # The location of the data
CHECKPOINT_DIR = os.getenv(
    "GEOLIFE_CLEF_CHECKPOINTS", os.path.join(ROOT_DIR, "checkpoints")
)

WANDB_PROJECT = "geolife-clef-2024"

SEED = 42
