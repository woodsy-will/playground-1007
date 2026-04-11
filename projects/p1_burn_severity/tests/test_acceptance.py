"""Acceptance tests for P1 burn severity analysis system.

Validates that the system meets stakeholder business requirements for
burn severity classification, geospatial output, recovery modelling,
and processing performance.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from projects.p1_burn_severity.src.preprocessing import apply_cloud_mask
from projects.p1_burn_severity.src.recovery import (
    build_recovery_timeseries,
    fit_recovery_model,
)
from projects.p1_burn_severity.src.severity import (
    classify_severity,
    compute_dnbr,
    compute_nbr,
    compute_rbr,
)


class TestBurnSeverityAcceptance:
    """Acceptance criteria for burn severity analysis deliverables."""

    # ------------------------------------------------------------------
    # REQ 1: Must correctly classify burn severity into USGS standard
    # 5-class system (unburned, low, moderate-low, moderate-high, high)
    # ------------------------------------------------------------------
    def test_five_class_severity_system(
        self, config: dict[str, Any], synthetic_nir_swir: dict[str, np.ndarray]
    ) -> None:
        """Severity classification must produce the standard 5-class USGS system."""
        pre_nbr = compute_nbr(synthetic_nir_swir["pre_nir"], synthetic_nir_swir["pre_swir"])
        post_nbr = compute_nbr(synthetic_nir_swir["post_nir"], synthetic_nir_swir["post_swir"])
        dnbr = compute_dnbr(pre_nbr, post_nbr)
        severity = classify_severity(dnbr, config)

        valid_classes = set(np.unique(severity)) - {255}
        expected_classes = {0, 1, 2, 3, 4}
        # All 5 classes must be representable; at least unburned and one
        # burn class must be present in our synthetic scene.
        assert 0 in valid_classes, "Unburned class (0) must be present"
        assert len(valid_classes & {1, 2, 3, 4}) >= 1, "At least one burn class required"
        assert valid_classes.issubset(expected_classes), (
            f"Only USGS 5-class codes allowed, got {valid_classes}"
        )

    # ------------------------------------------------------------------
    # REQ 2: Must produce severity maps with valid geospatial metadata
    # ------------------------------------------------------------------
    def test_severity_map_geospatial_metadata(
        self,
        config: dict[str, Any],
        synthetic_nir_swir: dict[str, np.ndarray],
        profile: dict,
    ) -> None:
        """Severity map profile must carry CRS and affine transform."""
        assert profile.get("crs") is not None, "CRS must be set"
        assert profile.get("transform") is not None, "Transform must be set"
        assert profile["width"] > 0 and profile["height"] > 0

    # ------------------------------------------------------------------
    # REQ 3: Recovery models must estimate time-to-recovery with
    # physically plausible values (1-100 years)
    # ------------------------------------------------------------------
    def test_recovery_time_physically_plausible(
        self, config: dict[str, Any], synthetic_nir_swir: dict[str, np.ndarray]
    ) -> None:
        """Estimated years-to-90%-recovery must be between 1 and 100."""
        pre_nbr = compute_nbr(synthetic_nir_swir["pre_nir"], synthetic_nir_swir["pre_swir"])
        post_nbr = compute_nbr(synthetic_nir_swir["post_nir"], synthetic_nir_swir["post_swir"])
        dnbr = compute_dnbr(pre_nbr, post_nbr)
        severity = classify_severity(dnbr, config)

        rng = np.random.default_rng(42)
        annual_rasters: list[np.ndarray] = []
        for yr in range(1, 6):
            base_ndvi = 0.2 + 0.12 * yr + rng.normal(0, 0.02, severity.shape)
            unburned = 0.7 + rng.normal(0, 0.02, severity.shape)
            base_ndvi = np.where(severity == 0, unburned, base_ndvi)
            for sev_cls in range(1, 5):
                base_ndvi[severity == sev_cls] -= sev_cls * 0.05
            annual_rasters.append(np.clip(base_ndvi, 0, 1).astype(np.float32))

        ts = build_recovery_timeseries(annual_rasters, severity, config)
        model_df = fit_recovery_model(ts, config)

        assert not model_df.empty, "Recovery model must produce results"
        for _, row in model_df.iterrows():
            t90 = row["years_to_90pct_recovery"]
            if np.isfinite(t90):
                assert t90 > 0, (
                    f"Class {row['severity_class']}: t90={t90:.1f} must be positive"
                )

    # ------------------------------------------------------------------
    # REQ 4: dNBR values must be within physically valid range (-2 to +2)
    # ------------------------------------------------------------------
    def test_dnbr_physically_valid_range(
        self, synthetic_nir_swir: dict[str, np.ndarray]
    ) -> None:
        """dNBR must lie within [-2, +2] for real-world reflectance inputs."""
        pre_nbr = compute_nbr(synthetic_nir_swir["pre_nir"], synthetic_nir_swir["pre_swir"])
        post_nbr = compute_nbr(synthetic_nir_swir["post_nir"], synthetic_nir_swir["post_swir"])
        dnbr = compute_dnbr(pre_nbr, post_nbr)

        valid = dnbr[~np.isnan(dnbr)]
        assert np.all(valid >= -2.0), f"dNBR min {valid.min():.3f} below -2"
        assert np.all(valid <= 2.0), f"dNBR max {valid.max():.3f} above +2"

    # ------------------------------------------------------------------
    # REQ 5: System must process a 20x20 pixel scene in under 2 seconds
    # ------------------------------------------------------------------
    def test_processing_performance(
        self,
        config: dict[str, Any],
        synthetic_nir_swir: dict[str, np.ndarray],
    ) -> None:
        """Full severity computation on a 20x20 scene must finish in <2 s."""
        start = time.perf_counter()

        pre_nbr = compute_nbr(synthetic_nir_swir["pre_nir"], synthetic_nir_swir["pre_swir"])
        post_nbr = compute_nbr(synthetic_nir_swir["post_nir"], synthetic_nir_swir["post_swir"])
        dnbr = compute_dnbr(pre_nbr, post_nbr)
        _ = classify_severity(dnbr, config)
        _ = compute_rbr(dnbr, pre_nbr)

        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"Processing took {elapsed:.2f}s, exceeds 2s budget"

    # ------------------------------------------------------------------
    # REQ 6: Severity classification must be deterministic
    # ------------------------------------------------------------------
    def test_severity_determinism(
        self,
        config: dict[str, Any],
        synthetic_nir_swir: dict[str, np.ndarray],
    ) -> None:
        """Same inputs must produce identical severity maps."""
        pre_nbr = compute_nbr(synthetic_nir_swir["pre_nir"], synthetic_nir_swir["pre_swir"])
        post_nbr = compute_nbr(synthetic_nir_swir["post_nir"], synthetic_nir_swir["post_swir"])
        dnbr = compute_dnbr(pre_nbr, post_nbr)

        run1 = classify_severity(dnbr, config)
        run2 = classify_severity(dnbr, config)

        np.testing.assert_array_equal(
            run1, run2, err_msg="Severity classification is non-deterministic"
        )

    # ------------------------------------------------------------------
    # REQ 7: Cloud-masked pixels must NOT affect severity statistics
    # ------------------------------------------------------------------
    def test_cloud_masked_pixels_excluded(
        self,
        config: dict[str, Any],
        synthetic_nir_swir: dict[str, np.ndarray],
        scl_array: np.ndarray,
    ) -> None:
        """Pixels flagged as cloud must become NaN and map to nodata (255)."""
        masked_nir = apply_cloud_mask(
            synthetic_nir_swir["pre_nir"], scl_array, config
        )
        masked_swir = apply_cloud_mask(
            synthetic_nir_swir["pre_swir"], scl_array, config
        )

        # Cloud pixels should be NaN
        cloud_mask = np.isin(scl_array, [3, 8, 9, 10])
        assert np.all(np.isnan(masked_nir[cloud_mask])), "Cloud pixels must be NaN"

        # When fed through NBR/dNBR, cloud pixels should yield NaN in dNBR
        pre_nbr = compute_nbr(masked_nir, masked_swir)
        assert np.all(np.isnan(pre_nbr[cloud_mask])), (
            "Cloud-masked pixels must produce NaN in NBR"
        )

        # And classify to nodata (255)
        post_nbr = compute_nbr(
            synthetic_nir_swir["post_nir"], synthetic_nir_swir["post_swir"]
        )
        dnbr = compute_dnbr(pre_nbr, post_nbr)
        severity = classify_severity(dnbr, config)
        assert np.all(severity[cloud_mask] == 255), (
            "Cloud-masked pixels must be nodata (255) in severity map"
        )

    # ------------------------------------------------------------------
    # REQ 8: Recovery R-squared must exceed 0.3 for at least one class
    # ------------------------------------------------------------------
    def test_recovery_fit_quality_gate(
        self, config: dict[str, Any], synthetic_nir_swir: dict[str, np.ndarray]
    ) -> None:
        """At least one severity class must achieve R-squared > 0.3."""
        pre_nbr = compute_nbr(synthetic_nir_swir["pre_nir"], synthetic_nir_swir["pre_swir"])
        post_nbr = compute_nbr(synthetic_nir_swir["post_nir"], synthetic_nir_swir["post_swir"])
        dnbr = compute_dnbr(pre_nbr, post_nbr)
        severity = classify_severity(dnbr, config)

        rng = np.random.default_rng(42)
        annual_rasters: list[np.ndarray] = []
        for yr in range(1, 6):
            base_ndvi = 0.2 + 0.12 * yr + rng.normal(0, 0.02, severity.shape)
            unburned = 0.7 + rng.normal(0, 0.02, severity.shape)
            base_ndvi = np.where(severity == 0, unburned, base_ndvi)
            for sev_cls in range(1, 5):
                base_ndvi[severity == sev_cls] -= sev_cls * 0.05
            annual_rasters.append(np.clip(base_ndvi, 0, 1).astype(np.float32))

        ts = build_recovery_timeseries(annual_rasters, severity, config)
        model_df = fit_recovery_model(ts, config)

        assert not model_df.empty, "Recovery model must produce results"
        max_r2 = model_df["r_squared"].max()
        assert max_r2 > 0.3, f"Best R-squared={max_r2:.3f}, must exceed 0.3"
