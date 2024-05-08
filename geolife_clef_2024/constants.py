"""Constants used throughout the package."""

import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))  # The root of the package
ROOT_DIR = os.path.dirname(PACKAGE_DIR)  # The parent to the package
DATA_DIR = os.path.join(ROOT_DIR, "data")  # The location of the data

SEED = 42
