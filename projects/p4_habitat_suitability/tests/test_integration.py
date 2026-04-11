"""Integration tests for P4 habitat suitability pipeline.

These tests exercise multi-module chains across the P4 project:
- change analysis pipeline (compute_change -> summarize_change)
- occurrence loading and thinning -> predictor extraction
- background sampling and PA matrix construction
- topographic derivative integration

No sklearn is required for any of these tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from rasterio.transform import from_bounds

from projects.p4_habitat_suitability.src.background import (
    create_pa_matrix,
    generate_background_points,
)
from projects.p4_habitat_suitability.src.change_analysis import (
    GAIN,
    LOSS,
    STABLE_SUITABLE,
    STABLE_UNSUITABLE,
    compute_change,
    summarize_change,
)
from projects.p4_habitat_suitability.src.occurrences import (
    load_occurrences,
    thin_occurrences,
)
from projects.p4_habitat_suitability.src.predictors import (
    build_predictor_stack,
    extract_values_at_points,
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


# ---------------------------------------------------------------------------
# TestOccurrencesToPredictors
# ---------------------------------------------------------------------------


class TestOccurrencesToPredictors:
    """Chain: load_occurrences -> thin_occurrences -> build_predictor_stack
    -> extract_values_at_points."""

    def test_load_thin_extract_chain(
        self,
        occurrences_path: Path,
        predictor_dir: Path,
        default_config: dict,
    ) -> None:
        """Load occurrences, thin them, build predictor stack, extract
        values at points.  Assert extracted DataFrame has correct number
        of bands."""
        # Step 1: Load occurrences
        occ = load_occurrences(default_config)
        assert len(occ) > 0

        # Step 2: Thin occurrences
        thinned = thin_occurrences(occ, distance_km=0.01)
        assert len(thinned) > 0
        assert len(thinned) <= len(occ)

        # Step 3: Build predictor stack
        stack, profile, band_names = build_predictor_stack(default_config)
        assert stack.ndim == 3
        assert len(band_names) == stack.shape[0]

        # Step 4: Extract values at points
        extracted = extract_values_at_points(stack, profile, thinned, band_names)
        assert len(extracted) > 0
        # Every band name should appear as a column (plus x, y)
        for name in band_names:
            assert name in extracted.columns
        assert extracted.shape[1] == len(band_names) + 2  # bands + x + y


# ---------------------------------------------------------------------------
# TestBackgroundSamplingChain
# ---------------------------------------------------------------------------


class TestBackgroundSamplingChain:
    """Chain: load_occurrences -> thin -> build_stack ->
    generate_background_points -> verify points."""

    def test_occurrence_to_background_chain(
        self,
        occurrences_path: Path,
        predictor_dir: Path,
        default_config: dict,
    ) -> None:
        """Full chain producing background points.  Assert correct count,
        presence=0, and points within raster extent."""
        # Step 1: Load and thin occurrences
        occ = load_occurrences(default_config)
        thinned = thin_occurrences(occ, distance_km=0.01)

        # Step 2: Build predictor stack
        stack, profile, band_names = build_predictor_stack(default_config)

        # Step 3: Generate background points
        n_bg = 100
        bg = generate_background_points(
            thinned, stack, profile, default_config, n_points=n_bg,
        )

        # Verify count
        assert len(bg) == n_bg

        # All background points should have presence=0
        assert (bg["presence"] == 0).all()

        # All points should fall within the raster extent
        transform = profile["transform"]
        height = profile["height"]
        width = profile["width"]
        xmin = transform.c
        ymax = transform.f
        xmax = xmin + width * transform.a
        ymin = ymax + height * transform.e  # transform.e is negative

        for geom in bg.geometry:
            assert geom.x >= xmin and geom.x <= xmax
            assert geom.y >= ymin and geom.y <= ymax


# ---------------------------------------------------------------------------
# TestProjectionChain
# ---------------------------------------------------------------------------


class TestProjectionChain:
    """Chain: compute_change -> summarize_change with threshold sensitivity."""

    def test_threshold_sensitivity(self) -> None:
        """Run change analysis at threshold=0.3 and 0.7, verify different
        class distributions."""
        rng = np.random.default_rng(77)
        rows, cols = 40, 40
        current = rng.random((rows, cols))
        future = rng.random((rows, cols))

        change_03 = compute_change(current, future, threshold=0.3)
        change_07 = compute_change(current, future, threshold=0.7)

        # Different thresholds should produce different class distributions
        count_suitable_03 = int((change_03 == STABLE_SUITABLE).sum())
        count_suitable_07 = int((change_07 == STABLE_SUITABLE).sum())

        # At lower threshold more pixels are classified as suitable, so
        # stable suitable count should differ
        assert count_suitable_03 != count_suitable_07

        # At threshold 0.3 more pixels should be "suitable" overall
        # (both current and future above 0.3 is more likely than above 0.7)
        assert count_suitable_03 > count_suitable_07

    def test_change_to_summary_areas_consistent(self) -> None:
        """Verify area calculations match pixel counts exactly."""
        rng = np.random.default_rng(88)
        rows, cols = 30, 30
        current = rng.random((rows, cols))
        future = rng.random((rows, cols))
        cell_size = 30.0

        transform = from_bounds(0, 0, cols * cell_size, rows * cell_size, cols, rows)
        profile = {"transform": transform, "width": cols, "height": rows}
        cell_area_m2 = cell_size * cell_size

        change = compute_change(current, future, threshold=0.5)
        summary = summarize_change(change, profile)

        # For every class, area_m2 must be exactly pixel_count * cell_area_m2
        for _, row in summary.iterrows():
            expected_area = row["pixel_count"] * cell_area_m2
            assert row["area_m2"] == pytest.approx(expected_area)


# ---------------------------------------------------------------------------
# TestPAMatrixChain
# ---------------------------------------------------------------------------


class TestPAMatrixChain:
    """Chain: load -> thin -> build_stack -> generate_background ->
    create_pa_matrix."""

    def test_full_pa_matrix_chain(
        self,
        occurrences_path: Path,
        predictor_dir: Path,
        default_config: dict,
    ) -> None:
        """Assert X has shape (n, bands), y has matching length, y contains
        0s and 1s."""
        # Step 1: Load and thin occurrences
        occ = load_occurrences(default_config)
        thinned = thin_occurrences(occ, distance_km=0.01)

        # Step 2: Build predictor stack
        stack, profile, band_names = build_predictor_stack(default_config)

        # Step 3: Generate background points
        n_bg = 50
        bg = generate_background_points(
            thinned, stack, profile, default_config, n_points=n_bg,
        )

        # Step 4: Create PA matrix
        x_mat, y = create_pa_matrix(thinned, bg, stack, profile, band_names)

        # x_mat should have shape (n_samples, n_bands)
        assert x_mat.ndim == 2
        assert x_mat.shape[1] == len(band_names)

        # y should match x_mat in length
        assert len(y) == x_mat.shape[0]

        # y should contain both 0s and 1s
        assert 0 in y
        assert 1 in y

        # Total samples should not exceed presence + background
        assert len(y) <= len(thinned) + n_bg


# ---------------------------------------------------------------------------
# TestTopoDerivativesChain
# ---------------------------------------------------------------------------


class TestTopoDerivativesChain:
    """Chain: build_predictor_stack (reads topo files) ->
    extract_values_at_points -> verify derivatives are finite."""

    def test_predictors_with_topo_derivatives(
        self,
        occurrences_path: Path,
        predictor_dir: Path,
        default_config: dict,
    ) -> None:
        """Build stack, verify topo bands included and values are finite."""
        # Step 1: Build predictor stack (the synthetic predictors include
        # elevation, slope, tpi which are topographic derivatives)
        stack, profile, band_names = build_predictor_stack(default_config)

        # Verify topographic bands are present in the stack
        topo_bands = {"elevation", "slope", "tpi"}
        present_topo = topo_bands.intersection(set(band_names))
        assert len(present_topo) >= 3, (
            f"Expected at least 3 topo bands, found {present_topo}"
        )

        # Step 2: Load occurrences and extract values
        occ = load_occurrences(default_config)
        thinned = thin_occurrences(occ, distance_km=0.01)

        extracted = extract_values_at_points(stack, profile, thinned, band_names)
        assert len(extracted) > 0

        # Verify all topo-derived values are finite (no NaN/inf)
        for topo_name in present_topo:
            assert topo_name in extracted.columns
            vals = extracted[topo_name].values
            assert np.all(np.isfinite(vals)), (
                f"Non-finite values found in {topo_name}"
            )
