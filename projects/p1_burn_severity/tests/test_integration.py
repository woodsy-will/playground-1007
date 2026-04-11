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


# ---------------------------------------------------------------------------
# TestRBRChain
# ---------------------------------------------------------------------------


class TestRBRChain:
    """Chain: compute_nbr -> compute_rbr (Relativized Burn Ratio)."""

    def test_nbr_to_rbr_chain(
        self,
        synthetic_nir_swir: dict[str, np.ndarray],
        config: dict[str, Any],
    ):
        """Compute pre-NBR, compute RBR, verify finite output and correct shape."""
        from projects.p1_burn_severity.src.severity import compute_rbr

        bands = synthetic_nir_swir

        # Step 1: Compute pre- and post-fire NBR
        pre_nbr = compute_nbr(bands["pre_nir"], bands["pre_swir"])
        post_nbr = compute_nbr(bands["post_nir"], bands["post_swir"])

        # Step 2: Compute dNBR
        dnbr = compute_dnbr(pre_nbr, post_nbr)

        # Step 3: Compute RBR = dNBR / (pre_NBR + 1.001)
        rbr = compute_rbr(dnbr, pre_nbr)

        # Verify shape matches input
        assert rbr.shape == pre_nbr.shape

        # Where pre_nbr and post_nbr are finite, RBR should also be finite
        # (the +1.001 prevents division by zero)
        finite_mask = np.isfinite(pre_nbr) & np.isfinite(post_nbr)
        assert np.all(np.isfinite(rbr[finite_mask])), (
            "RBR should be finite wherever both NBR inputs are finite"
        )

        # Burned region (top half) should have positive RBR on average
        h = pre_nbr.shape[0]
        burn_rbr = rbr[:h // 2, :]
        valid_burn = burn_rbr[np.isfinite(burn_rbr)]
        assert np.mean(valid_burn) > 0, "Burned zone should have positive RBR"


# ---------------------------------------------------------------------------
# TestMultiClassRecovery
# ---------------------------------------------------------------------------


class TestMultiClassRecovery:
    """Chain: classify with multiple severity classes -> build timeseries
    -> fit models for each."""

    def test_multi_class_recovery_fitting(self, config: dict[str, Any]):
        """Create severity raster with classes 1-4, build recovery timeseries
        for each, fit models, assert separate model fits per class."""
        rng = np.random.default_rng(55)
        h, w = 40, 40

        # Create a severity raster with four burn classes in quadrants
        severity = np.zeros((h, w), dtype=np.uint8)
        severity[:20, :20] = 1   # low severity (upper-left)
        severity[:20, 20:] = 2   # moderate-low (upper-right)
        severity[20:, :20] = 3   # moderate-high (lower-left)
        severity[20:, 20:] = 4   # high severity (lower-right)

        # Simulate 5 years of recovery NDVI with different rates per class
        # Higher severity => slower recovery
        recovery_rates = {1: 0.8, 2: 0.6, 3: 0.4, 4: 0.25}
        annual_ndvi: list[np.ndarray] = []
        for year in range(1, 6):
            ndvi = np.full((h, w), 0.75, dtype=np.float64)
            for sev_class, rate in recovery_rates.items():
                mask = severity == sev_class
                recovery_val = 0.65 * (1.0 - np.exp(-rate * year)) + 0.1
                ndvi[mask] = recovery_val + rng.normal(0, 0.01, np.sum(mask))
            annual_ndvi.append(ndvi)

        # Step 1: Build time series
        ts_df = build_recovery_timeseries(annual_ndvi, severity, config)

        # All four severity classes should be present
        classes_present = set(ts_df["severity_class"].unique())
        for cls in [1, 2, 3, 4]:
            assert cls in classes_present, f"Class {cls} missing from timeseries"

        # Step 2: Fit recovery models
        model_df = fit_recovery_model(ts_df, config)

        # Should get separate models for each severity class
        fitted_classes = set(model_df["severity_class"].values)
        assert len(fitted_classes) >= 4, (
            f"Expected models for 4 classes, got {fitted_classes}"
        )

        # All R-squared values should be finite and positive (clean signal)
        for _, row in model_df.iterrows():
            assert np.isfinite(row["r_squared"]), (
                f"R-squared should be finite for class {row['severity_class']}"
            )
            assert row["r_squared"] > 0.5, (
                f"R-squared too low for class {row['severity_class']}"
            )

        # Higher severity classes should have longer recovery times
        model_sorted = model_df.sort_values("severity_class")
        t90_values = model_sorted["years_to_90pct_recovery"].values
        for i in range(len(t90_values) - 1):
            assert t90_values[i] < t90_values[i + 1], (
                "Higher severity should have longer recovery times"
            )


# ---------------------------------------------------------------------------
# TestPreprocessingEdgeCases
# ---------------------------------------------------------------------------


class TestPreprocessingEdgeCases:
    """Chain: cloud mask with 100% clouds -> NBR -> verify all NaN."""

    def test_full_cloud_cover_propagation(self, config: dict[str, Any]):
        """All pixels masked -> NBR should be all NaN -> severity all nodata."""
        rng = np.random.default_rng(77)
        h, w = 20, 20

        # Create synthetic bands
        nir = rng.uniform(0.3, 0.5, (h, w)).astype(np.float32)
        swir = rng.uniform(0.05, 0.15, (h, w)).astype(np.float32)

        # SCL with 100% cloud cover (class 9 = cloud high probability)
        scl_all_cloud = np.full((h, w), 9, dtype=np.uint8)

        # Step 1: Apply cloud mask — everything should become NaN
        masked_nir = apply_cloud_mask(nir, scl_all_cloud, config)
        masked_swir = apply_cloud_mask(swir, scl_all_cloud, config)

        assert np.all(np.isnan(masked_nir)), "All pixels should be NaN after masking"
        assert np.all(np.isnan(masked_swir)), "All pixels should be NaN after masking"

        # Step 2: Compute NBR — should be all NaN
        nbr = compute_nbr(masked_nir, masked_swir)
        assert np.all(np.isnan(nbr)), "NBR should be all NaN with full cloud mask"

        # Step 3: Compute dNBR — all NaN
        dnbr = compute_dnbr(nbr, nbr)
        assert np.all(np.isnan(dnbr)), "dNBR should be all NaN"

        # Step 4: Classify severity — should be all nodata (255)
        severity = classify_severity(dnbr, config)
        assert np.all(severity == 255), (
            "All severity pixels should be nodata (255) with full cloud cover"
        )
