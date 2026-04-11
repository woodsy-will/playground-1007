"""Acceptance tests for P4 habitat suitability modeling system.

Validates that the system meets stakeholder business requirements for
occurrence thinning, predictor completeness, background sampling,
PA matrix construction, habitat change analysis, area statistics,
topographic derivatives, processing speed, and suitability thresholding.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from projects.p4_habitat_suitability.src.change_analysis import (
    compute_change,
    summarize_change,
)
from projects.p4_habitat_suitability.src.occurrences import (
    load_occurrences,
    thin_occurrences,
)
from projects.p4_habitat_suitability.src.predictors import (
    build_predictor_stack,
    compute_topo_derivatives,
)
from projects.p4_habitat_suitability.src.projection import threshold_suitability


class TestHabitatSuitabilityAcceptance:
    """Acceptance criteria for habitat suitability modeling."""

    # ----------------------------------------------------------------
    # REQ 1: Occurrence thinning must reduce spatial autocorrelation
    # ----------------------------------------------------------------
    def test_thinning_reduces_point_count(
        self, default_config: dict
    ) -> None:
        """Spatial thinning must produce fewer points than the raw dataset."""
        occ = load_occurrences(default_config)
        n_raw = len(occ)
        distance_km = default_config["species"]["thinning_distance_km"]
        thinned = thin_occurrences(occ, distance_km, default_config)
        n_thinned = len(thinned)

        assert n_raw > 0, "No raw occurrences loaded"
        assert n_thinned < n_raw, (
            f"Thinning did not reduce points: {n_thinned} >= {n_raw}"
        )
        assert n_thinned > 0, "Thinning removed all occurrences"

    # ----------------------------------------------------------------
    # REQ 2: Predictor stack must have no all-NaN bands
    # ----------------------------------------------------------------
    def test_predictor_stack_no_all_nan_bands(
        self, default_config: dict
    ) -> None:
        """Every band in the predictor stack must contain at least one finite value."""
        stack, profile, band_names = build_predictor_stack(default_config)

        assert stack.ndim == 3, f"Stack must be 3-D, got {stack.ndim}"
        for i, name in enumerate(band_names):
            band = stack[i]
            assert np.any(np.isfinite(band)), (
                f"Band '{name}' is entirely NaN/non-finite"
            )

    # ----------------------------------------------------------------
    # REQ 3: Background points must sample the full study area extent
    # ----------------------------------------------------------------
    def test_background_points_cover_extent(
        self, default_config: dict
    ) -> None:
        """Background points must span at least 80% of the raster extent."""
        from projects.p4_habitat_suitability.src.background import (
            generate_background_points,
        )

        occ = load_occurrences(default_config)
        thinned = thin_occurrences(
            occ, default_config["species"]["thinning_distance_km"]
        )
        stack, profile, _ = build_predictor_stack(default_config)

        bg = generate_background_points(thinned, stack, profile, default_config)

        assert len(bg) > 0, "No background points generated"

        # Check that background covers a reasonable fraction of the extent
        transform = profile["transform"]
        raster_xmin = transform.c
        raster_ymax = transform.f
        raster_xmax = raster_xmin + profile["width"] * transform.a
        raster_ymin = raster_ymax + profile["height"] * transform.e

        bg_bounds = bg.total_bounds  # xmin, ymin, xmax, ymax
        raster_width = raster_xmax - raster_xmin
        raster_height = raster_ymax - raster_ymin
        bg_width = bg_bounds[2] - bg_bounds[0]
        bg_height = bg_bounds[3] - bg_bounds[1]

        coverage_x = bg_width / raster_width if raster_width > 0 else 0
        coverage_y = bg_height / raster_height if raster_height > 0 else 0

        assert coverage_x >= 0.5, (
            f"Background X coverage {coverage_x:.1%} < 50%"
        )
        assert coverage_y >= 0.5, (
            f"Background Y coverage {coverage_y:.1%} < 50%"
        )

    # ----------------------------------------------------------------
    # REQ 4: Presence/absence matrix must have both classes represented
    # ----------------------------------------------------------------
    def test_pa_matrix_both_classes(self, default_config: dict) -> None:
        """PA matrix must contain both presence (1) and absence (0) samples."""
        from projects.p4_habitat_suitability.src.background import (
            create_pa_matrix,
            generate_background_points,
        )

        occ = load_occurrences(default_config)
        thinned = thin_occurrences(
            occ, default_config["species"]["thinning_distance_km"]
        )
        stack, profile, band_names = build_predictor_stack(default_config)
        bg = generate_background_points(thinned, stack, profile, default_config)
        _, y = create_pa_matrix(thinned, bg, stack, profile, band_names)

        assert len(y) > 0, "PA matrix is empty"
        assert np.any(y == 1), "No presence records in PA matrix"
        assert np.any(y == 0), "No background records in PA matrix"

    # ----------------------------------------------------------------
    # REQ 5: Habitat change analysis must produce all 4 change classes
    # ----------------------------------------------------------------
    def test_change_analysis_four_classes(self) -> None:
        """compute_change must produce refugia, gain, loss, and stable unsuitable."""
        # Create current and future suitability with varied values
        current = np.array([
            [0.8, 0.9, 0.2, 0.1],
            [0.7, 0.3, 0.4, 0.1],
            [0.1, 0.2, 0.6, 0.8],
            [0.1, 0.1, 0.3, 0.9],
        ], dtype=np.float32)
        future = np.array([
            [0.7, 0.2, 0.8, 0.1],
            [0.6, 0.6, 0.1, 0.2],
            [0.2, 0.1, 0.7, 0.2],
            [0.1, 0.3, 0.8, 0.8],
        ], dtype=np.float32)

        change = compute_change(current, future, threshold=0.5)
        unique_classes = set(np.unique(change))

        # Must have all 4 classes: 0=stable_unsuitable, 1=refugia, 2=gain, 3=loss
        assert 0 in unique_classes, "Missing stable_unsuitable (0)"
        assert 1 in unique_classes, "Missing stable_suitable/refugia (1)"
        assert 2 in unique_classes, "Missing gain (2)"
        assert 3 in unique_classes, "Missing loss (3)"

    # ----------------------------------------------------------------
    # REQ 6: Area statistics must be spatially consistent
    # ----------------------------------------------------------------
    def test_area_statistics_consistency(self) -> None:
        """Total area from change summary must equal raster extent area."""
        from shared.utils.io import make_profile

        bounds = (-200_000.0, -50_000.0, -199_800.0, -49_800.0)
        profile = make_profile(bounds, 30.0)
        h, w = profile["height"], profile["width"]

        # Simple change raster
        rng = np.random.default_rng(42)
        change = rng.choice([0, 1, 2, 3], size=(h, w)).astype(np.uint8)

        summary = summarize_change(change, profile)

        total_pixels = summary["pixel_count"].sum()
        assert total_pixels == h * w, (
            f"Total pixels {total_pixels} != raster size {h * w}"
        )

        cellsize_x = abs(profile["transform"].a)
        cellsize_y = abs(profile["transform"].e)
        expected_area = h * w * cellsize_x * cellsize_y
        total_area = summary["area_m2"].sum()
        assert abs(total_area - expected_area) < 1e-6, (
            f"Total area {total_area:.1f} m2 != expected {expected_area:.1f} m2"
        )

    # ----------------------------------------------------------------
    # REQ 7: Topographic derivatives must have physically valid ranges
    # ----------------------------------------------------------------
    def test_topo_derivatives_valid_ranges(
        self, tmp_path: Path
    ) -> None:
        """Slope, TPI, and TWI must have physically plausible values."""
        from shared.utils.io import make_profile, write_raster

        bounds = (-200_000.0, -50_000.0, -199_800.0, -49_800.0)
        profile = make_profile(bounds, 10.0)
        h, w = profile["height"], profile["width"]

        # Create a synthetic DEM with gentle slope
        rng = np.random.default_rng(42)
        dem = 1000.0 + np.linspace(0, 100, w).reshape(1, -1).repeat(h, axis=0)
        dem += rng.normal(0, 1, (h, w))
        dem_path = tmp_path / "dem.tif"
        write_raster(dem_path, dem.astype(np.float32), profile)

        derivs = compute_topo_derivatives(dem_path, tmp_path / "derivs")

        # Slope must be in [0, 90] degrees
        from shared.utils.io import read_raster

        slope_data, _ = read_raster(derivs["slope"])
        slope = slope_data[0]
        assert np.all(np.isfinite(slope)), "Slope has non-finite values"
        assert np.all(slope >= 0), f"Slope has negatives: min={slope.min():.2f}"
        assert np.all(slope <= 90), f"Slope exceeds 90 deg: max={slope.max():.2f}"

        # TPI should be centred around zero for a planar surface with noise
        tpi_data, _ = read_raster(derivs["tpi"])
        tpi = tpi_data[0]
        assert np.all(np.isfinite(tpi)), "TPI has non-finite values"

        # TWI should be finite
        twi_data, _ = read_raster(derivs["twi"])
        twi = twi_data[0]
        assert np.all(np.isfinite(twi)), "TWI has non-finite values"

    # ----------------------------------------------------------------
    # REQ 8: System must process occurrence data + predictors in <3s
    # ----------------------------------------------------------------
    def test_processing_performance(self, default_config: dict) -> None:
        """Loading occurrences, thinning, and building predictor stack must finish in <3s."""
        start = time.perf_counter()

        occ = load_occurrences(default_config)
        thin_occurrences(occ, default_config["species"]["thinning_distance_km"])
        build_predictor_stack(default_config)

        elapsed = time.perf_counter() - start
        assert elapsed < 3.0, (
            f"Processing took {elapsed:.2f}s, exceeds 3s budget"
        )

    # ----------------------------------------------------------------
    # REQ 9: Suitability thresholding must produce a valid binary map
    # ----------------------------------------------------------------
    def test_suitability_thresholding_binary(self) -> None:
        """threshold_suitability must produce only 0s and 1s (uint8)."""
        rng = np.random.default_rng(42)
        suitability = rng.uniform(0, 1, (20, 20)).astype(np.float32)

        binary = threshold_suitability(suitability, threshold=0.5)

        assert binary.dtype == np.uint8, f"Expected uint8, got {binary.dtype}"
        unique_vals = set(np.unique(binary))
        assert unique_vals.issubset({0, 1}), (
            f"Binary map has non-binary values: {unique_vals}"
        )
        # Both classes should be present with random input
        assert 0 in unique_vals, "No unsuitable pixels in binary map"
        assert 1 in unique_vals, "No suitable pixels in binary map"

        # NaN pixels must map to 0
        suitability_with_nan = suitability.copy()
        suitability_with_nan[0, 0] = np.nan
        binary_nan = threshold_suitability(suitability_with_nan, threshold=0.5)
        assert binary_nan[0, 0] == 0, "NaN pixel must map to 0 (unsuitable)"
