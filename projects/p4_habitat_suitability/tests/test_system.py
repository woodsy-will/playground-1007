"""System tests for the P4 habitat suitability pipeline.

Verifies the complete system end-to-end: occurrence data and environmental
predictors are loaded, processed, and combined into suitability surfaces
with change analysis.  Tests focus on the data pipeline portion and do not
require sklearn.
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
    compute_topo_derivatives,
)
from shared.utils.io import read_raster

# -----------------------------------------------------------------------
# System test class
# -----------------------------------------------------------------------


class TestHabitatDataSystem:
    """System-level tests for the P4 habitat suitability data pipeline."""

    def test_system_occurrence_loading_and_thinning(
        self,
        occurrences_path: Path,
        default_config: dict,
    ) -> None:
        """Load occurrences, thin, verify fewer points after thinning,
        all points within valid CRS."""
        occ = load_occurrences(default_config)
        assert len(occ) > 0
        assert occ.crs is not None

        thinned = thin_occurrences(occ, distance_km=1.0, config=default_config)
        assert len(thinned) > 0
        assert len(thinned) <= len(occ), (
            f"Thinned ({len(thinned)}) should be <= original ({len(occ)})"
        )

        # All thinned points should still have a valid CRS
        assert thinned.crs is not None
        # All geometries should be valid points
        assert thinned.geometry.is_valid.all()

    def test_system_predictor_stack_complete(
        self,
        predictor_dir: Path,
        default_config: dict,
    ) -> None:
        """Build predictor stack, verify all expected bands present,
        no all-NaN bands."""
        stack, profile, band_names = build_predictor_stack(default_config)

        assert stack.ndim == 3
        assert len(band_names) == stack.shape[0]
        assert len(band_names) > 0

        # Verify no all-NaN bands
        for i, name in enumerate(band_names):
            band = stack[i]
            assert not np.all(np.isnan(band)), (
                f"Band '{name}' is entirely NaN"
            )

    def test_system_background_sampling_unbiased(
        self,
        occurrences_path: Path,
        predictor_dir: Path,
        default_config: dict,
    ) -> None:
        """Generate background points, verify they cover the raster extent,
        presence column is all 0."""
        occ = load_occurrences(default_config)
        thinned = thin_occurrences(occ, distance_km=0.01)
        stack, profile, band_names = build_predictor_stack(default_config)

        n_bg = 200
        bg = generate_background_points(
            thinned, stack, profile, default_config, n_points=n_bg,
        )

        assert len(bg) == n_bg

        # All background points should have presence=0
        assert (bg["presence"] == 0).all(), "Background points should all have presence=0"

        # Points should cover the raster extent
        transform = profile["transform"]
        height = profile["height"]
        width = profile["width"]
        xmin = transform.c
        ymax = transform.f
        xmax = xmin + width * transform.a
        ymin = ymax + height * transform.e  # transform.e is negative

        for geom in bg.geometry:
            assert xmin <= geom.x <= xmax, (
                f"Point x={geom.x} outside raster extent [{xmin}, {xmax}]"
            )
            assert ymin <= geom.y <= ymax, (
                f"Point y={geom.y} outside raster extent [{ymin}, {ymax}]"
            )

    def test_system_pa_matrix_balanced(
        self,
        occurrences_path: Path,
        predictor_dir: Path,
        default_config: dict,
    ) -> None:
        """Create PA matrix, verify both classes present, features finite."""
        occ = load_occurrences(default_config)
        thinned = thin_occurrences(occ, distance_km=0.01)
        stack, profile, band_names = build_predictor_stack(default_config)

        n_bg = 50
        bg = generate_background_points(
            thinned, stack, profile, default_config, n_points=n_bg,
        )

        x_mat, y = create_pa_matrix(thinned, bg, stack, profile, band_names)

        # Both classes should be present
        assert 1 in y, "No presence samples in PA matrix"
        assert 0 in y, "No background samples in PA matrix"

        # Features should be finite (no NaN or inf)
        assert np.all(np.isfinite(x_mat)), "PA matrix contains non-finite values"

        # Shape consistency
        assert x_mat.ndim == 2
        assert x_mat.shape[0] == len(y)
        assert x_mat.shape[1] == len(band_names)

    def test_system_change_analysis_end_to_end(self) -> None:
        """Create current/future suitability surfaces, compute change,
        summarize. Verify all 4 classes present, areas sum to total extent."""
        rng = np.random.default_rng(42)
        rows, cols = 50, 50
        cell_size = 30.0

        # Synthetic suitability surfaces
        current = rng.random((rows, cols))
        future = rng.random((rows, cols))

        transform = from_bounds(0, 0, cols * cell_size, rows * cell_size, cols, rows)
        profile = {"transform": transform, "width": cols, "height": rows}

        # Compute change
        change = compute_change(current, future, threshold=0.5)
        assert change.shape == (rows, cols)
        assert change.dtype == np.uint8

        # Summarize
        summary = summarize_change(change, profile)

        # All 4 classes should be present
        class_codes = set(summary["class_code"].values)
        expected_codes = {STABLE_UNSUITABLE, STABLE_SUITABLE, GAIN, LOSS}
        assert class_codes == expected_codes, (
            f"Expected all 4 change classes, got {class_codes}"
        )

        # Total area should equal raster extent
        total_pixels = summary["pixel_count"].sum()
        assert total_pixels == rows * cols

        cell_area_m2 = cell_size * cell_size
        total_area = summary["area_m2"].sum()
        expected_area = rows * cols * cell_area_m2
        assert total_area == pytest.approx(expected_area)

    def test_system_topo_derivatives_chain(
        self,
        predictor_dir: Path,
        default_config: dict,
        tmp_path: Path,
    ) -> None:
        """Compute topo derivatives from DEM, verify slope/TPI/TWI all
        finite and within expected ranges."""
        # Use the synthetic elevation raster as a DEM
        dem_path = predictor_dir / "elevation.tif"
        assert dem_path.exists(), f"Elevation raster not found: {dem_path}"

        output_dir = tmp_path / "topo_output"
        results = compute_topo_derivatives(dem_path, output_dir)

        # Should produce slope, tpi, twi
        assert "slope" in results
        assert "tpi" in results
        assert "twi" in results

        # Read and verify slope: should be >= 0 degrees and finite
        slope_data, _ = read_raster(results["slope"])
        slope = slope_data[0]
        assert np.all(np.isfinite(slope)), "Slope contains non-finite values"
        assert np.all(slope >= 0), "Slope has negative values"
        assert np.all(slope <= 90), "Slope exceeds 90 degrees"

        # Read and verify TPI: should be finite
        tpi_data, _ = read_raster(results["tpi"])
        tpi = tpi_data[0]
        assert np.all(np.isfinite(tpi)), "TPI contains non-finite values"

        # Read and verify TWI: should be finite
        twi_data, _ = read_raster(results["twi"])
        twi = twi_data[0]
        assert np.all(np.isfinite(twi)), "TWI contains non-finite values"
