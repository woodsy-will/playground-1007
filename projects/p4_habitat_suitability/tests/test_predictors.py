"""Tests for predictor stack construction and value extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pyproj import CRS

from shared.utils.io import make_profile, write_raster


class TestBuildPredictorStack:
    """Verify predictor stack alignment, shape, and CRS."""

    def test_stack_shape(self, default_config: dict):
        from projects.p4_habitat_suitability.src.predictors import build_predictor_stack

        stack, profile, band_names = build_predictor_stack(default_config)
        n_bands, h, w = stack.shape
        assert n_bands == len(band_names)
        assert h == profile["height"]
        assert w == profile["width"]
        assert n_bands > 0

    def test_stack_crs(self, default_config: dict):
        from projects.p4_habitat_suitability.src.predictors import build_predictor_stack

        _, profile, _ = build_predictor_stack(default_config)
        assert CRS(profile["crs"]) == CRS.from_epsg(3310)

    def test_stack_finite_values(self, default_config: dict):
        from projects.p4_habitat_suitability.src.predictors import build_predictor_stack

        stack, _, _ = build_predictor_stack(default_config)
        assert np.all(np.isfinite(stack)), "Stack contains non-finite values"

    def test_band_names_match_files(self, default_config: dict):
        from projects.p4_habitat_suitability.src.predictors import build_predictor_stack

        _, _, band_names = build_predictor_stack(default_config)
        # Synthetic data generates these predictors
        expected = {"bio1_mean_temp", "bio12_annual_precip", "canopy_cover",
                    "elevation", "slope", "tpi"}
        assert set(band_names) == expected

    def test_empty_predictor_dir_raises(self, tmp_path: Path):
        """build_predictor_stack raises FileNotFoundError if no .tif files."""
        from projects.p4_habitat_suitability.src.predictors import build_predictor_stack

        empty_dir = tmp_path / "empty_predictors"
        empty_dir.mkdir()
        config = {
            "data": {"predictor_dir": str(empty_dir)},
            "modeling": {"crs": "EPSG:3310"},
        }
        with pytest.raises(FileNotFoundError, match="No .tif files"):
            build_predictor_stack(config)

    def test_mismatched_raster_alignment(self, tmp_path: Path):
        """Rasters with different dimensions are reprojected to the reference grid."""
        from projects.p4_habitat_suitability.src.predictors import build_predictor_stack

        pred_dir = tmp_path / "predictors"
        pred_dir.mkdir()

        # Reference raster: 10x10 at 10m resolution
        ref_profile = make_profile(bounds=(0, 0, 100, 100), resolution=10.0)
        ref_data = np.ones((ref_profile["height"], ref_profile["width"]), dtype=np.float32)
        write_raster(pred_dir / "aaa_ref.tif", ref_data, ref_profile)

        # Mismatched raster: 5x5 at 20m resolution (same extent, different grid)
        mis_profile = make_profile(bounds=(0, 0, 100, 100), resolution=20.0)
        mis_data = np.ones(
            (mis_profile["height"], mis_profile["width"]), dtype=np.float32
        ) * 2.0
        write_raster(pred_dir / "bbb_mis.tif", mis_data, mis_profile)

        config = {
            "data": {"predictor_dir": str(pred_dir)},
            "modeling": {"crs": "EPSG:3310"},
        }
        stack, profile, band_names = build_predictor_stack(config)

        # Both bands should have the reference dimensions
        assert stack.shape == (2, ref_profile["height"], ref_profile["width"])
        assert band_names == ["aaa_ref", "bbb_mis"]
        assert np.all(np.isfinite(stack))


class TestComputeTopoDerivatives:
    """Verify topographic derivative computation from a DEM."""

    @staticmethod
    def _create_synthetic_dem(tmp_path: Path) -> Path:
        """Write a small synthetic DEM raster and return its path."""
        profile = make_profile(
            bounds=(0.0, 0.0, 500.0, 500.0),
            resolution=50.0,
        )
        rng = np.random.default_rng(99)
        # Create a tilted surface with some noise so derivatives are non-trivial
        h, w = profile["height"], profile["width"]
        row_idx, col_idx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        dem = (100.0 + 5.0 * row_idx + 3.0 * col_idx + rng.normal(0, 0.5, (h, w))).astype(
            np.float32
        )
        dem_path = tmp_path / "dem.tif"
        write_raster(dem_path, dem, profile)
        return dem_path

    def test_returns_dict_with_expected_keys(self, tmp_path: Path) -> None:
        from projects.p4_habitat_suitability.src.predictors import compute_topo_derivatives

        dem_path = self._create_synthetic_dem(tmp_path)
        out_dir = tmp_path / "topo_out"
        result = compute_topo_derivatives(dem_path, out_dir)

        assert isinstance(result, dict)
        assert set(result.keys()) == {"slope", "tpi", "twi"}

    def test_slope_values_non_negative(self, tmp_path: Path) -> None:
        from projects.p4_habitat_suitability.src.predictors import compute_topo_derivatives

        dem_path = self._create_synthetic_dem(tmp_path)
        out_dir = tmp_path / "topo_out"
        result = compute_topo_derivatives(dem_path, out_dir)

        from shared.utils.io import read_raster

        slope_data, _ = read_raster(result["slope"])
        assert np.all(slope_data >= 0), "Slope contains negative values"

    def test_output_files_exist(self, tmp_path: Path) -> None:
        from projects.p4_habitat_suitability.src.predictors import compute_topo_derivatives

        dem_path = self._create_synthetic_dem(tmp_path)
        out_dir = tmp_path / "topo_out"
        result = compute_topo_derivatives(dem_path, out_dir)

        for name, path in result.items():
            assert Path(path).exists(), f"Output file missing for {name}"

    def test_tpi_values_finite(self, tmp_path: Path) -> None:
        from projects.p4_habitat_suitability.src.predictors import compute_topo_derivatives

        dem_path = self._create_synthetic_dem(tmp_path)
        out_dir = tmp_path / "topo_out"
        result = compute_topo_derivatives(dem_path, out_dir)

        from shared.utils.io import read_raster

        tpi_data, _ = read_raster(result["tpi"])
        assert np.all(np.isfinite(tpi_data)), "TPI contains non-finite values"


class TestExtractValuesAtPoints:
    """Verify raster value extraction at point locations."""

    def test_extraction_returns_all_bands(self, default_config: dict):
        from projects.p4_habitat_suitability.src.occurrences import load_occurrences
        from projects.p4_habitat_suitability.src.predictors import (
            build_predictor_stack,
            extract_values_at_points,
        )

        gdf = load_occurrences(default_config)
        stack, profile, band_names = build_predictor_stack(default_config)
        df = extract_values_at_points(stack, profile, gdf, band_names)

        assert len(df) > 0
        for name in band_names:
            assert name in df.columns

    def test_extraction_no_nans(self, default_config: dict):
        from projects.p4_habitat_suitability.src.occurrences import load_occurrences
        from projects.p4_habitat_suitability.src.predictors import (
            build_predictor_stack,
            extract_values_at_points,
        )

        gdf = load_occurrences(default_config)
        stack, profile, band_names = build_predictor_stack(default_config)
        df = extract_values_at_points(stack, profile, gdf, band_names)

        assert not df.isna().any().any(), "Extracted values contain NaN"
