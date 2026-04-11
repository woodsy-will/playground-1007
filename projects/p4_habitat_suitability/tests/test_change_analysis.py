"""Tests for habitat change analysis between current and future suitability maps."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from rasterio.transform import from_bounds

from projects.p4_habitat_suitability.src.change_analysis import (
    GAIN,
    LOSS,
    STABLE_SUITABLE,
    STABLE_UNSUITABLE,
    compute_change,
    summarize_change,
)


@pytest.fixture()
def sample_profile() -> dict:
    """A minimal rasterio-like profile with a 10x10 m cell transform."""
    transform = from_bounds(0, 0, 100, 100, 10, 10)  # 10m cells
    return {"transform": transform, "width": 10, "height": 10, "crs": "EPSG:3310"}


class TestComputeChange:
    """Tests for compute_change()."""

    def test_all_classes_present(self) -> None:
        """All four change classes should appear when conditions exist."""
        current = np.array([[0.8, 0.1], [0.8, 0.1]])
        future = np.array([[0.8, 0.8], [0.1, 0.1]])

        result = compute_change(current, future)

        unique_classes = set(np.unique(result))
        assert unique_classes == {
            STABLE_UNSUITABLE,
            STABLE_SUITABLE,
            GAIN,
            LOSS,
        }

    def test_all_unsuitable(self) -> None:
        """All pixels should be stable_unsuitable when both surfaces are below threshold."""
        current = np.zeros((5, 5))
        future = np.zeros((5, 5))

        result = compute_change(current, future)

        assert (result == STABLE_UNSUITABLE).all()

    def test_all_suitable_refugia(self) -> None:
        """All pixels should be stable_suitable (refugia) when both surfaces are above threshold."""
        current = np.ones((5, 5))
        future = np.ones((5, 5))

        result = compute_change(current, future)

        assert (result == STABLE_SUITABLE).all()

    def test_custom_threshold(self) -> None:
        """Custom threshold should change how pixels are classified."""
        current = np.full((3, 3), 0.3)
        future = np.full((3, 3), 0.3)

        # Default threshold 0.5 -> unsuitable
        result_default = compute_change(current, future, threshold=0.5)
        assert (result_default == STABLE_UNSUITABLE).all()

        # Lower threshold 0.2 -> refugia
        result_low = compute_change(current, future, threshold=0.2)
        assert (result_low == STABLE_SUITABLE).all()

    def test_output_dtype(self) -> None:
        """Output raster should have uint8 dtype."""
        current = np.random.default_rng(0).random((4, 4))
        future = np.random.default_rng(1).random((4, 4))

        result = compute_change(current, future)

        assert result.dtype == np.uint8


class TestSummarizeChange:
    """Tests for summarize_change()."""

    def test_returns_dataframe(self, sample_profile: dict) -> None:
        """summarize_change should return a pandas DataFrame."""
        change = np.zeros((10, 10), dtype=np.uint8)
        result = summarize_change(change, sample_profile)

        assert isinstance(result, pd.DataFrame)

    def test_area_calculation(self, sample_profile: dict) -> None:
        """Area values should match pixel count * cell area."""
        # 10x10 grid with 10m cells -> cell area = 100 m2
        change = np.zeros((10, 10), dtype=np.uint8)  # all stable_unsuitable
        result = summarize_change(change, sample_profile)

        row = result[result["class_code"] == STABLE_UNSUITABLE].iloc[0]
        expected_area_m2 = 100 * 100.0  # 100 pixels * 100 m2/pixel
        assert row["area_m2"] == pytest.approx(expected_area_m2)
        assert row["area_ha"] == pytest.approx(expected_area_m2 / 10_000.0)

    def test_all_classes_in_output(self, sample_profile: dict) -> None:
        """All four change classes should appear in the summary, even with zero counts."""
        change = np.zeros((10, 10), dtype=np.uint8)
        result = summarize_change(change, sample_profile)

        assert set(result["class_code"].values) == {
            STABLE_UNSUITABLE,
            STABLE_SUITABLE,
            GAIN,
            LOSS,
        }

    def test_pixel_count_matches(self, sample_profile: dict) -> None:
        """Pixel counts in summary should match actual pixel counts in the raster."""
        change = np.array(
            [[0, 1, 2, 3, 0], [1, 2, 3, 0, 1], [2, 3, 0, 1, 2], [3, 0, 1, 2, 3]],
            dtype=np.uint8,
        )
        # Adjust profile for this raster size
        profile = {
            "transform": from_bounds(0, 0, 50, 40, 5, 4),
            "width": 5,
            "height": 4,
        }

        result = summarize_change(change, profile)

        for code in [STABLE_UNSUITABLE, STABLE_SUITABLE, GAIN, LOSS]:
            expected = int((change == code).sum())
            actual = result[result["class_code"] == code]["pixel_count"].iloc[0]
            assert actual == expected
