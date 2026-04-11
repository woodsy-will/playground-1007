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

    def test_fit_recovery_too_few_points(self, config: dict) -> None:
        """Severity class with <3 data points should be skipped."""
        records = []
        # Class 1 has only 2 points — too few to fit
        for yr in range(1, 3):
            records.append(
                {
                    "year": yr,
                    "severity_class": 1,
                    "mean_index": 0.3 + 0.05 * yr,
                    "std_index": 0.01,
                    "pixel_count": 50,
                }
            )
        # Class 2 has 5 points — enough to fit
        for yr in range(1, 6):
            records.append(
                {
                    "year": yr,
                    "severity_class": 2,
                    "mean_index": 0.6 * (1 - np.exp(-0.4 * yr)) + 0.2,
                    "std_index": 0.01,
                    "pixel_count": 100,
                }
            )
        df = pd.DataFrame(records)
        result = fit_recovery_model(df, config)
        # Class 1 should be absent from results, only class 2 fitted
        assert 1 not in result["severity_class"].values
        assert 2 in result["severity_class"].values

    def test_fit_recovery_no_convergence(self, config: dict) -> None:
        """When curve_fit raises RuntimeError, the class should be skipped."""
        from unittest.mock import patch

        records = []
        for yr in range(1, 8):
            records.append(
                {
                    "year": yr,
                    "severity_class": 0,
                    "mean_index": 0.5,
                    "std_index": 0.01,
                    "pixel_count": 100,
                }
            )
        df = pd.DataFrame(records)
        with patch(
            "projects.p1_burn_severity.src.recovery.curve_fit",
            side_effect=RuntimeError("Optimal parameters not found"),
        ):
            result = fit_recovery_model(df, config)
        # Convergence failure means the class is skipped entirely
        assert len(result) == 0


class TestTimeseriesAllNanClass:
    """Cover the `continue` when all values are NaN for a severity class."""

    def test_timeseries_all_nan_class(self, config: dict) -> None:
        """A severity class whose index values are ALL NaN should be skipped."""
        severity = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        # Raster where class-1 pixels are NaN
        raster = np.array([[0.5, np.nan], [np.nan, 0.6]], dtype=np.float32)
        ts = build_recovery_timeseries([raster], severity, config)
        # Class 1 should not appear (all NaN)
        assert 1 not in ts["severity_class"].values
        # Class 0 should appear
        assert 0 in ts["severity_class"].values
