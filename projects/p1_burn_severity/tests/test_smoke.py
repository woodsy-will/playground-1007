"""Smoke tests for P1 Burn Severity.

Quick sanity checks that core modules import, key functions are callable,
and basic operations produce expected types. Designed to run in < 1 second.
"""

from __future__ import annotations

import numpy as np


class TestP1Smoke:
    """Fast smoke tests for P1 burn severity analysis."""

    def test_imports(self):
        """All core P1 modules should import without error."""
        from projects.p1_burn_severity.src import (
            preprocessing,  # noqa: F401
            recovery,  # noqa: F401
            severity,  # noqa: F401
        )

    def test_compute_nbr_runs(self):
        """compute_nbr should accept arrays and return an array."""
        from projects.p1_burn_severity.src.severity import compute_nbr

        nir = np.array([[0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
        swir = np.array([[0.1, 0.2], [0.1, 0.2]], dtype=np.float32)
        result = compute_nbr(nir, swir)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)

    def test_classify_severity_runs(self):
        """classify_severity should produce integer class codes."""
        from projects.p1_burn_severity.src.severity import classify_severity

        dnbr = np.array([[0.05, 0.15, 0.3, 0.5, 0.8]])
        config = {
            "processing": {
                "severity_thresholds": {
                    "unburned": (-0.1, 0.1),
                    "low": (0.1, 0.27),
                    "moderate_low": (0.27, 0.44),
                    "moderate_high": (0.44, 0.66),
                    "high": (0.66, 1.3),
                }
            }
        }
        result = classify_severity(dnbr, config)
        assert result.dtype == np.uint8
        assert set(np.unique(result)).issubset({0, 1, 2, 3, 4, 255})

    def test_cloud_mask_runs(self):
        """apply_cloud_mask should zero out cloud pixels."""
        from projects.p1_burn_severity.src.preprocessing import apply_cloud_mask

        band = np.ones((4, 4), dtype=np.float32)
        scl = np.zeros((4, 4), dtype=np.uint8)
        scl[0, 0] = 9  # cloud
        config = {"preprocessing": {"cloud_classes": [9]}}
        result = apply_cloud_mask(band, scl, config)
        assert np.isnan(result[0, 0])
        assert result[1, 1] == 1.0

    def test_vegetation_index_runs(self):
        """compute_vegetation_index should return a finite array."""
        from projects.p1_burn_severity.src.recovery import compute_vegetation_index

        nir = np.array([[0.5, 0.6]])
        red = np.array([[0.1, 0.2]])
        ndvi = compute_vegetation_index(nir, red, index_type="NDVI")
        assert np.all(np.isfinite(ndvi))
