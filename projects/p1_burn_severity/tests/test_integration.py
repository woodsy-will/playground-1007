"""Integration tests chaining P1 burn severity modules.

Tests the full analysis pipeline: preprocessing -> severity -> recovery,
using synthetic data from conftest fixtures.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from projects.p1_burn_severity.src.preprocessing import apply_cloud_mask
from projects.p1_burn_severity.src.recovery import (
    build_recovery_timeseries,
    compute_vegetation_index,
    fit_recovery_model,
)
from projects.p1_burn_severity.src.severity import (
    classify_severity,
    compute_dnbr,
    compute_nbr,
)

# ---------------------------------------------------------------------------
# TestPreprocessingToSeverity
# ---------------------------------------------------------------------------


class TestPreprocessingToSeverity:
    """Chain: apply_cloud_mask -> compute_nbr -> compute_dnbr -> classify_severity."""

    def test_cloud_mask_to_severity_chain(
        self,
        synthetic_nir_swir: dict[str, np.ndarray],
        scl_array: np.ndarray,
        config: dict[str, Any],
    ):
        """Masked pixels should propagate as NaN through NBR/dNBR and become
        nodata (255) in the severity classification."""
        bands = synthetic_nir_swir

        # Step 1: Apply cloud mask to all four bands
        pre_nir = apply_cloud_mask(bands["pre_nir"], scl_array, config)
        pre_swir = apply_cloud_mask(bands["pre_swir"], scl_array, config)
        post_nir = apply_cloud_mask(bands["post_nir"], scl_array, config)
        post_swir = apply_cloud_mask(bands["post_swir"], scl_array, config)

        # Cloud pixels (SCL 9 at [2:5, 2:5] and SCL 3 at [7, 7]) should be NaN
        assert np.isnan(pre_nir[3, 3]), "Cloud pixel should be NaN after masking"
        assert np.isnan(pre_nir[7, 7]), "Cloud shadow pixel should be NaN"

        # Step 2: Compute pre- and post-fire NBR
        pre_nbr = compute_nbr(pre_nir, pre_swir)
        post_nbr = compute_nbr(post_nir, post_swir)

        # NaN should propagate through NBR
        assert np.isnan(pre_nbr[3, 3])
        assert np.isnan(post_nbr[3, 3])

        # Step 3: Compute dNBR
        dnbr = compute_dnbr(pre_nbr, post_nbr)

        # Burned top half should have positive dNBR (high severity)
        burn_zone = dnbr[:10, :]
        valid_burn = burn_zone[~np.isnan(burn_zone)]
        assert np.mean(valid_burn) > 0.1, "Burned zone should have positive dNBR"

        # Unburned bottom half should have dNBR near zero
        unburn_zone = dnbr[10:, :]
        valid_unburn = unburn_zone[~np.isnan(unburn_zone)]
        assert abs(np.mean(valid_unburn)) < 0.1, "Unburned zone dNBR should be near zero"

        # Step 4: Classify severity
        severity = classify_severity(dnbr, config)

        # Cloud-masked pixels should be nodata (255)
        assert severity[3, 3] == 255, "Cloud pixel should be nodata in classification"
        assert severity[7, 7] == 255, "Cloud shadow should be nodata in classification"

        # Burned zone should have elevated severity classes (>= 1)
        burn_sev = severity[:10, :]
        valid_burn_sev = burn_sev[(burn_sev != 255)]
        assert np.mean(valid_burn_sev) > 1, "Burned zone should have severity > low"

        # Unburned zone should be mostly class 0 (unburned)
        unburn_sev = severity[10:, :]
        valid_unburn_sev = unburn_sev[(unburn_sev != 255)]
        assert np.mean(valid_unburn_sev) < 1, "Unburned zone should be mostly class 0"


# ---------------------------------------------------------------------------
# TestSeverityToRecovery
# ---------------------------------------------------------------------------


class TestSeverityToRecovery:
    """Chain: synthetic NDVI timeseries -> build_recovery_timeseries -> fit_recovery_model."""

    def test_severity_to_recovery_chain(self, config: dict[str, Any]):
        """A synthetic exponential recovery signal should be fit with
        reasonable R-squared and finite time-to-90% recovery."""
        rng = np.random.default_rng(42)
        h, w = 20, 20

        # Create a severity raster: top half = class 4 (high), bottom = class 0
        severity = np.zeros((h, w), dtype=np.uint8)
        severity[:10, :] = 4  # high severity

        # Simulate 5 years of post-fire NDVI recovery for the burned area.
        # Unburned area stays at high NDVI; burned area recovers exponentially.
        annual_ndvi: list[np.ndarray] = []
        for year in range(1, 6):
            ndvi = np.full((h, w), 0.75, dtype=np.float64)  # unburned baseline
            # Exponential recovery: NDVI = 0.65 * (1 - exp(-0.4 * t)) + 0.1
            recovery_val = 0.65 * (1.0 - np.exp(-0.4 * year)) + 0.1
            ndvi[:10, :] = recovery_val + rng.normal(0, 0.02, (10, w))
            annual_ndvi.append(ndvi)

        # Step 1: Build time series
        ts_df = build_recovery_timeseries(annual_ndvi, severity, config)

        assert isinstance(ts_df, pd.DataFrame)
        assert len(ts_df) > 0
        assert "year" in ts_df.columns
        assert "severity_class" in ts_df.columns
        assert "mean_index" in ts_df.columns

        # Both severity classes should be present
        classes_present = set(ts_df["severity_class"].unique())
        assert 0 in classes_present, "Unburned class should be in time series"
        assert 4 in classes_present, "High severity class should be in time series"

        # Step 2: Fit recovery model
        model_df = fit_recovery_model(ts_df, config)

        assert isinstance(model_df, pd.DataFrame)
        assert len(model_df) > 0

        # High severity class should have a fitted model
        high_sev = model_df[model_df["severity_class"] == 4]
        assert len(high_sev) == 1, "Should have one model for high severity"

        row = high_sev.iloc[0]
        assert np.isfinite(row["r_squared"]), "R-squared should be finite"
        assert row["r_squared"] > 0.5, "Fit quality should be reasonable"
        assert np.isfinite(
            row["years_to_90pct_recovery"]
        ), "Time to 90% recovery should be finite"
        assert row["years_to_90pct_recovery"] > 0, "Recovery time should be positive"


# ---------------------------------------------------------------------------
# TestFullAnalysisChain
# ---------------------------------------------------------------------------


class TestFullAnalysisChain:
    """End-to-end chain: raw data -> cloud mask -> NBR -> dNBR -> severity
    -> vegetation index timeseries -> recovery model."""

    def test_end_to_end(
        self,
        synthetic_nir_swir: dict[str, np.ndarray],
        scl_array: np.ndarray,
        config: dict[str, Any],
    ):
        """Full pipeline from raw bands to fitted recovery model."""
        bands = synthetic_nir_swir
        rng = np.random.default_rng(123)
        h, w = bands["pre_nir"].shape

        # --- Stage 1: Preprocessing (cloud masking) ---
        pre_nir = apply_cloud_mask(bands["pre_nir"], scl_array, config)
        pre_swir = apply_cloud_mask(bands["pre_swir"], scl_array, config)
        post_nir = apply_cloud_mask(bands["post_nir"], scl_array, config)
        post_swir = apply_cloud_mask(bands["post_swir"], scl_array, config)

        # --- Stage 2: Severity analysis ---
        pre_nbr = compute_nbr(pre_nir, pre_swir)
        post_nbr = compute_nbr(post_nir, post_swir)
        dnbr = compute_dnbr(pre_nbr, post_nbr)
        severity = classify_severity(dnbr, config)

        # Sanity: we should have multiple severity classes
        unique_classes = set(np.unique(severity))
        assert len(unique_classes) >= 2, "Should have at least 2 severity classes"

        # --- Stage 3: Vegetation index computation ---
        # Use compute_vegetation_index to get NDVI from NIR and a synthetic Red band
        red_band = rng.uniform(0.05, 0.15, (h, w)).astype(np.float32)
        ndvi_post = compute_vegetation_index(post_nir, red_band, index_type="NDVI")
        assert ndvi_post.shape == (h, w)

        # --- Stage 4: Recovery time series ---
        # Generate 5 years of recovery NDVI rasters based on severity
        annual_ndvi: list[np.ndarray] = []
        for year in range(1, 6):
            ndvi_year = np.full((h, w), 0.7, dtype=np.float64)
            # Pixels that were high severity recover more slowly
            high_mask = severity == 4
            if np.any(high_mask):
                recovery = 0.6 * (1.0 - np.exp(-0.3 * year)) + 0.1
                ndvi_year[high_mask] = recovery + rng.normal(0, 0.01, np.sum(high_mask))
            # Moderate severity pixels recover faster
            mod_mask = (severity == 2) | (severity == 3)
            if np.any(mod_mask):
                recovery = 0.6 * (1.0 - np.exp(-0.6 * year)) + 0.15
                ndvi_year[mod_mask] = recovery + rng.normal(0, 0.01, np.sum(mod_mask))
            # Nodata pixels stay NaN
            ndvi_year[severity == 255] = np.nan
            annual_ndvi.append(ndvi_year)

        ts_df = build_recovery_timeseries(annual_ndvi, severity, config)
        assert len(ts_df) > 0, "Time series should have data"

        # --- Stage 5: Recovery model fitting ---
        model_df = fit_recovery_model(ts_df, config)

        # Should produce fitted models for classes with >= 3 data points
        assert isinstance(model_df, pd.DataFrame)
        if len(model_df) > 0:
            # R-squared may be NaN for constant series (e.g. unburned class
            # where NDVI does not change).  Verify that at least one class
            # has a finite R-squared and that all finite values are positive.
            finite_r2 = model_df["r_squared"].dropna()
            assert len(finite_r2) > 0, "At least one class should have a fitted R-squared"
            assert (finite_r2 > 0).all(), "Finite R-squared values should be positive"
            # All recovery times should be positive where finite
            finite_t90 = model_df["years_to_90pct_recovery"][
                model_df["years_to_90pct_recovery"].apply(np.isfinite)
            ]
            assert (finite_t90 > 0).all(), "Recovery times should be positive"
