"""Integration tests for P4 change analysis pipeline.

These tests exercise the compute_change -> summarize_change workflow
using pure numpy (no sklearn needed), so they always run.
"""

from __future__ import annotations

import numpy as np
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


class TestChangeAnalysisPipeline:
    """Integration test: compute_change -> summarize_change."""

    def test_compute_then_summarize(self) -> None:
        """Full pipeline should produce consistent area totals."""
        rng = np.random.default_rng(99)
        rows, cols = 50, 50
        current = rng.random((rows, cols))
        future = rng.random((rows, cols))

        # 30m cells (common for Landsat-derived products)
        transform = from_bounds(0, 0, cols * 30, rows * 30, cols, rows)
        profile = {"transform": transform, "width": cols, "height": rows}
        cell_area_m2 = 30.0 * 30.0  # 900 m2

        # Step 1: compute change
        change = compute_change(current, future, threshold=0.5)
        assert change.shape == (rows, cols)
        assert change.dtype == np.uint8

        # Step 2: summarize
        summary = summarize_change(change, profile)

        # Total pixel count should equal total raster pixels
        total_pixels = summary["pixel_count"].sum()
        assert total_pixels == rows * cols

        # Total area should equal raster extent area
        total_area_m2 = summary["area_m2"].sum()
        expected_area = rows * cols * cell_area_m2
        assert total_area_m2 == pytest.approx(expected_area)

        # Hectare conversion should be consistent
        for _, row in summary.iterrows():
            assert row["area_ha"] == pytest.approx(row["area_m2"] / 10_000.0)

        # All four classes should appear in summary
        assert set(summary["class_code"].values) == {
            STABLE_UNSUITABLE,
            STABLE_SUITABLE,
            GAIN,
            LOSS,
        }

        # Each class pixel count should match what we can verify directly
        for _, row in summary.iterrows():
            code = row["class_code"]
            expected_count = int((change == code).sum())
            assert row["pixel_count"] == expected_count
