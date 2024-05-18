"""Test modifications to dataset for multimodal."""

import pytest

from geolife_clef_2024 import dataset_utils, datasets


def test_dataset_slicer():
    """Test dataset slicer."""
    dataset = dataset_utils.DatasetSlicer[0](datasets.load_satellite_patches())
    print(dataset.__getitem__)
    assert next(iter(dataset)) == 212, "Expected to only get the survey index."


if __name__ == "__main__":
    pytest.main([__file__])
