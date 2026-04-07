"""Tests for vegetation recovery tracking and curve fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projects.p1_burn_severity.src.recovery import (
    build_recovery_timeseries,
    compute_vegetation_index,
    fit_recovery_model,
)


class TestComputeVegetationIndex:
    """Tests for compute_vegetation_index."""

    def test_ndvi_known(self) -> None:
        nir = np.array([0.5, 0.3, 0.0])
        red = np.array([0.1, 0.3, 0.0])
        ndvi = compute_vegetation_index(nir, red, "NDVI")
        np.testing.assert_allclose(ndvi[0], (0.5 - 0.1) / (0.5 + 0.1), atol=1e-10)
        np.testing.assert_allclose(ndvi[1], 0.0, atol=1e-10)
        assert np.isnan(ndvi[2])

    def test_evi_known(self) -> None:
        nir = np.array([0.5])
        red = np.array([0.1])
        evi = compute_vegetation_index(nir, red, "EVI")
        expected = 2.5 * (0.5 - 0.1) / (0.5 + 2.4 * 0.1 + 1.0)
        np.testing.assert_allclose(evi[0], expected, atol=1e-10)

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            compute_vegetation_index(np.array([1.0]), np.array([1.0]), "BAD")


class TestBuildRecoveryTimeseries:
    """Tests for build_recovery_timeseries."""

    def test_output_columns(self, config: dict) -> None:
        rng = np.random.default_rng(0)
        severity = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        rasters = [rng.uniform(0.2, 0.8, (2, 2)).astype(np.float32) for _ in range(3)]
        ts = build_recovery_timeseries(rasters, severity, config)
        assert set(ts.columns) >= {
            "year",
            "severity_class",
            "mean_index",
            "std_index",
            "pixel_count",
        }

    def test_year_range(self, config: dict) -> None:
        severity = np.zeros((5, 5), dtype=np.uint8)
        rasters = [np.full((5, 5), 0.5, dtype=np.float32) for _ in range(5)]
        ts = build_recovery_timeseries(rasters, severity, config)
        assert ts["year"].min() == 1
        assert ts["year"].max() == 5


class TestFitRecoveryModel:
    """Tests for fit_recovery_model with synthetic exponential data."""

    @pytest.fixture()
    def synthetic_ts(self) -> pd.DataFrame:
        """Create a synthetic time series following an exponential curve."""
        rng = np.random.default_rng(42)
        records = []
        for sev in [1, 2, 3]:
            a, b, c = 0.6, 0.4, 0.2
            for yr in range(1, 8):
                y = a * (1 - np.exp(-b * yr)) + c + rng.normal(0, 0.01)
                records.append(
                    {
                        "year": yr,
                        "severity_class": sev,
                        "mean_index": y,
                        "std_index": 0.01,
                        "pixel_count": 100,
                    }
                )
        return pd.DataFrame(records)

    def test_fit_returns_params(self, synthetic_ts: pd.DataFrame, config: dict) -> None:
        result = fit_recovery_model(synthetic_ts, config)
        assert len(result) == 3  # 3 severity classes
        assert "a" in result.columns
        assert "b" in result.columns
        assert "c" in result.columns

    def test_r_squared_high(self, synthetic_ts: pd.DataFrame, config: dict) -> None:
        result = fit_recovery_model(synthetic_ts, config)
        for _, row in result.iterrows():
            assert row["r_squared"] > 0.95

    def test_recovery_time_positive(
        self, synthetic_ts: pd.DataFrame, config: dict
    ) -> None:
        result = fit_recovery_model(synthetic_ts, config)
        for _, row in result.iterrows():
            assert row["years_to_90pct_recovery"] > 0
