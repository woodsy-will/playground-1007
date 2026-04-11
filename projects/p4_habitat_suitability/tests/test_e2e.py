"""End-to-end tests for the habitat suitability pipeline.

User story: A wildlife biologist loads species occurrence data and
environmental predictors, generates background points, builds a habitat
model, and runs change analysis.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

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
from shared.utils.io import make_profile, read_raster, write_raster


class TestHabitatSuitabilityE2E:
    """E2E tests simulating a wildlife biologist's habitat modelling workflow."""

    # ------------------------------------------------------------------ #
    # 1. Load and prepare occurrence data
    # ------------------------------------------------------------------ #
    def test_user_loads_and_prepares_occurrence_data(
        self, default_config: dict
    ) -> None:
        occ = load_occurrences(default_config)
        assert isinstance(occ, gpd.GeoDataFrame)
        assert len(occ) > 0

        distance_km = default_config["species"]["thinning_distance_km"]
        thinned = thin_occurrences(occ, distance_km, default_config)

        assert isinstance(thinned, gpd.GeoDataFrame)
        assert len(thinned) <= len(occ), "Thinning should not add points"
        assert len(thinned) > 0, "Thinning removed all points"

        # CRS preserved
        crs_str = str(thinned.crs)
        assert "3310" in crs_str

    # ------------------------------------------------------------------ #
    # 2. Build environmental predictor stack
    # ------------------------------------------------------------------ #
    def test_user_builds_environmental_predictors(
        self, default_config: dict
    ) -> None:
        stack, profile, band_names = build_predictor_stack(default_config)

        # 3D array
        assert stack.ndim == 3
        n_bands, height, width = stack.shape
        assert n_bands > 0
        assert height > 0
        assert width > 0

        # All bands named
        assert len(band_names) == n_bands

        # No all-NaN bands
        for i, name in enumerate(band_names):
            assert not np.all(np.isnan(stack[i])), f"Band {name!r} is all NaN"

    # ------------------------------------------------------------------ #
    # 3. Full data preparation chain: load -> thin -> stack -> bg -> PA
    # ------------------------------------------------------------------ #
    def test_user_generates_training_data(
        self, default_config: dict
    ) -> None:
        occ = load_occurrences(default_config)
        distance_km = default_config["species"]["thinning_distance_km"]
        thinned = thin_occurrences(occ, distance_km, default_config)

        stack, profile, band_names = build_predictor_stack(default_config)

        bg = generate_background_points(
            thinned, stack, profile, default_config, n_points=200
        )
        assert len(bg) == 200

        x_mat, y = create_pa_matrix(thinned, bg, stack, profile, band_names)

        # x_mat shape: (n_samples, n_predictors)
        assert x_mat.ndim == 2
        assert x_mat.shape[1] == len(band_names)

        # y is binary
        unique_labels = set(np.unique(y))
        assert unique_labels.issubset({0, 1})
        assert 1 in unique_labels, "No presence samples in y"
        assert 0 in unique_labels, "No background samples in y"

        # Features are finite
        assert np.all(np.isfinite(x_mat)), "Non-finite values in predictor matrix"

    # ------------------------------------------------------------------ #
    # 4. Change analysis: synthetic current/future suitability
    # ------------------------------------------------------------------ #
    def test_user_runs_change_analysis(self, synthetic_dir: Path) -> None:
        rng = np.random.default_rng(42)
        shape = (50, 50)

        current = rng.uniform(0, 1, shape).astype(np.float32)
        future = rng.uniform(0, 1, shape).astype(np.float32)

        change = compute_change(current, future)
        assert change.shape == shape

        # All four class codes should be present in a random 50x50 grid
        unique_codes = set(np.unique(change))
        assert unique_codes.issubset(
            {STABLE_UNSUITABLE, STABLE_SUITABLE, GAIN, LOSS}
        )

        profile = make_profile(
            (-200_000, -50_000, -199_500, -49_500), 10.0
        )
        summary = summarize_change(change, profile)

        assert isinstance(summary, pd.DataFrame)
        assert "class_code" in summary.columns
        assert "area_ha" in summary.columns
        assert len(summary) == 4

    # ------------------------------------------------------------------ #
    # 5. Export change map to GeoTIFF and round-trip
    # ------------------------------------------------------------------ #
    def test_user_exports_change_map(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(99)
        shape = (40, 40)

        current = rng.uniform(0, 1, shape).astype(np.float32)
        future = rng.uniform(0, 1, shape).astype(np.float32)
        change = compute_change(current, future)

        profile = make_profile(
            (-200_000, -50_000, -199_600, -49_600), 10.0
        )

        out_path = tmp_path / "habitat_change.tif"
        write_raster(out_path, change, profile, dtype="uint8", nodata=255)
        assert out_path.exists()

        data, _ = read_raster(out_path)
        roundtrip = data[0]

        unique_codes = set(np.unique(roundtrip))
        expected = {STABLE_UNSUITABLE, STABLE_SUITABLE, GAIN, LOSS}
        assert unique_codes.issubset(expected)
        assert len(unique_codes) == 4, (
            f"Expected all 4 change classes; got {unique_codes}"
        )

    # ------------------------------------------------------------------ #
    # 6. Topo derivatives from a synthetic DEM
    # ------------------------------------------------------------------ #
    def test_user_computes_topo_derivatives(self, tmp_path: Path) -> None:
        # Create a synthetic DEM with a gradient
        rng = np.random.default_rng(42)
        shape = (50, 50)
        yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
        dem = (1000.0 + 5.0 * xx + 3.0 * yy + rng.normal(0, 0.5, shape)).astype(
            np.float32
        )

        profile = make_profile(
            (-200_000, -50_000, -199_500, -49_500), 10.0
        )
        dem_path = tmp_path / "dem.tif"
        write_raster(dem_path, dem, profile)

        out_dir = tmp_path / "topo_out"
        results = compute_topo_derivatives(dem_path, out_dir)

        # All three derivatives produced
        for name in ("slope", "tpi", "twi"):
            assert name in results, f"Missing derivative: {name}"
            assert results[name].exists(), f"{name} file not written"

        # Slope >= 0 degrees
        slope_data, _ = read_raster(results["slope"])
        slope = slope_data[0]
        assert np.all(slope >= 0), "Negative slope values"
        assert np.all(np.isfinite(slope)), "Non-finite slope values"

        # TPI is finite
        tpi_data, _ = read_raster(results["tpi"])
        assert np.all(np.isfinite(tpi_data[0])), "Non-finite TPI values"

        # TWI is finite
        twi_data, _ = read_raster(results["twi"])
        assert np.all(np.isfinite(twi_data[0])), "Non-finite TWI values"
