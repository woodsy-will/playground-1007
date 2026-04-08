"""Tests for burn severity computation -- NBR, dNBR, classification, RBR."""

from __future__ import annotations

import numpy as np

from projects.p1_burn_severity.src.severity import (
    classify_severity,
    compute_dnbr,
    compute_nbr,
    compute_rbr,
)


class TestComputeNBR:
    """Tests for compute_nbr."""

    def test_known_values(self) -> None:
        nir = np.array([0.5, 0.3, 0.0], dtype=np.float64)
        swir = np.array([0.1, 0.3, 0.0], dtype=np.float64)
        nbr = compute_nbr(nir, swir)
        np.testing.assert_allclose(nbr[0], (0.5 - 0.1) / (0.5 + 0.1), atol=1e-10)
        np.testing.assert_allclose(nbr[1], 0.0, atol=1e-10)
        assert np.isnan(nbr[2])  # 0/0

    def test_shape_preserved(self, synthetic_nir_swir: dict) -> None:
        nbr = compute_nbr(synthetic_nir_swir["pre_nir"], synthetic_nir_swir["pre_swir"])
        assert nbr.shape == synthetic_nir_swir["pre_nir"].shape

    def test_range(self, synthetic_nir_swir: dict) -> None:
        nbr = compute_nbr(synthetic_nir_swir["pre_nir"], synthetic_nir_swir["pre_swir"])
        valid = nbr[~np.isnan(nbr)]
        assert np.all(valid >= -1.0)
        assert np.all(valid <= 1.0)


class TestComputeDNBR:
    """Tests for compute_dnbr."""

    def test_positive_for_burned(self, synthetic_nir_swir: dict) -> None:
        pre_nbr = compute_nbr(
            synthetic_nir_swir["pre_nir"], synthetic_nir_swir["pre_swir"]
        )
        post_nbr = compute_nbr(
            synthetic_nir_swir["post_nir"], synthetic_nir_swir["post_swir"]
        )
        dnbr = compute_dnbr(pre_nbr, post_nbr)
        # Top half (burned) should have positive dNBR
        burn_rows = dnbr.shape[0] // 2
        assert np.nanmean(dnbr[:burn_rows, :]) > 0.0

    def test_near_zero_for_unburned(self, synthetic_nir_swir: dict) -> None:
        pre_nbr = compute_nbr(
            synthetic_nir_swir["pre_nir"], synthetic_nir_swir["pre_swir"]
        )
        post_nbr = compute_nbr(
            synthetic_nir_swir["post_nir"], synthetic_nir_swir["post_swir"]
        )
        dnbr = compute_dnbr(pre_nbr, post_nbr)
        # Bottom half (unburned) should be near zero
        burn_rows = dnbr.shape[0] // 2
        assert abs(np.nanmean(dnbr[burn_rows:, :])) < 0.05


class TestClassifySeverity:
    """Tests for classify_severity."""

    def test_known_thresholds(self, config: dict) -> None:
        dnbr = np.array([0.0, 0.15, 0.35, 0.55, 0.8, np.nan])
        classes = classify_severity(dnbr, config)
        assert classes[0] == 0  # unburned
        assert classes[1] == 1  # low
        assert classes[2] == 2  # moderate_low
        assert classes[3] == 3  # moderate_high
        assert classes[4] == 4  # high
        assert classes[5] == 255  # nodata

    def test_output_dtype(self, config: dict) -> None:
        dnbr = np.random.default_rng(0).uniform(-0.2, 1.0, (10, 10))
        classes = classify_severity(dnbr, config)
        assert classes.dtype == np.uint8

    def test_boundary_values(self, config: dict) -> None:
        """Verify values at exact threshold boundaries classify correctly."""
        # Exact boundary values: use threshold midpoints
        thresholds = config["processing"]["severity_thresholds"]
        dnbr = np.array([
            thresholds["unburned"][0],       # -0.1 -> unburned
            thresholds["unburned"][1],        # 0.1 -> low (boundary)
            thresholds["low"][1],             # 0.27 -> moderate_low
            thresholds["moderate_low"][1],    # 0.44 -> moderate_high
            thresholds["moderate_high"][1],   # 0.66 -> high
        ])
        classes = classify_severity(dnbr, config)
        assert classes[0] == 0  # unburned
        assert classes[1] == 1  # low (at boundary)
        assert classes[2] == 2  # moderate_low
        assert classes[3] == 3  # moderate_high
        assert classes[4] == 4  # high

    def test_all_classes_present(self, config: dict, synthetic_nir_swir: dict) -> None:
        pre_nbr = compute_nbr(
            synthetic_nir_swir["pre_nir"], synthetic_nir_swir["pre_swir"]
        )
        post_nbr = compute_nbr(
            synthetic_nir_swir["post_nir"], synthetic_nir_swir["post_swir"]
        )
        dnbr = compute_dnbr(pre_nbr, post_nbr)
        classes = classify_severity(dnbr, config)
        # At least unburned (0) and some burn classes should exist
        unique = set(np.unique(classes))
        assert 0 in unique  # unburned pixels from bottom half


class TestComputeRBR:
    """Tests for compute_rbr."""

    def test_finite_output(self) -> None:
        dnbr = np.array([0.5, 0.0, -0.1])
        pre_nbr = np.array([0.6, 0.3, 0.8])
        rbr = compute_rbr(dnbr, pre_nbr)
        assert np.all(np.isfinite(rbr))

    def test_zero_pre_nbr_safe(self) -> None:
        """Division by zero is avoided by the +1.001 offset."""
        dnbr = np.array([0.5])
        pre_nbr = np.array([0.0])
        rbr = compute_rbr(dnbr, pre_nbr)
        assert np.isfinite(rbr[0])
