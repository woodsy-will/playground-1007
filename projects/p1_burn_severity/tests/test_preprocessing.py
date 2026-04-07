"""Tests for cloud masking and reprojection preprocessing."""

from __future__ import annotations

import numpy as np
import pytest

from projects.p1_burn_severity.src.preprocessing import apply_cloud_mask


class TestApplyCloudMask:
    """Tests for apply_cloud_mask."""

    def test_masks_cloud_pixels(self, config: dict, scl_array: np.ndarray) -> None:
        data = np.ones((20, 20), dtype=np.float32)
        masked = apply_cloud_mask(data, scl_array, config)
        # SCL=9 block at [2:5, 2:5] and SCL=3 at [7,7] should be NaN
        assert np.isnan(masked[3, 3])
        assert np.isnan(masked[7, 7])

    def test_preserves_non_cloud(self, config: dict, scl_array: np.ndarray) -> None:
        data = np.full((20, 20), 42.0, dtype=np.float32)
        masked = apply_cloud_mask(data, scl_array, config)
        # Non-cloud pixel should remain
        assert masked[0, 0] == 42.0
        assert masked[10, 10] == 42.0

    def test_3d_input(self, config: dict, scl_array: np.ndarray) -> None:
        data = np.ones((2, 20, 20), dtype=np.float32)
        masked = apply_cloud_mask(data, scl_array, config)
        assert masked.shape == (2, 20, 20)
        assert np.isnan(masked[0, 3, 3])
        assert np.isnan(masked[1, 3, 3])

    def test_output_dtype(self, config: dict, scl_array: np.ndarray) -> None:
        data = np.ones((20, 20), dtype=np.float32)
        masked = apply_cloud_mask(data, scl_array, config)
        assert masked.dtype == np.float64

    def test_cloud_count(self, config: dict, scl_array: np.ndarray) -> None:
        data = np.ones((20, 20), dtype=np.float32)
        masked = apply_cloud_mask(data, scl_array, config)
        # 3x3 block of class 9 = 9 pixels + 1 pixel of class 3 = 10
        nan_count = np.count_nonzero(np.isnan(masked))
        assert nan_count == 10
