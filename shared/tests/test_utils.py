"""Unit tests for all shared utility modules."""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from pyproj import CRS

from shared.utils.config import load_config
from shared.utils.crs import DEFAULT_CRS, validate_crs
from shared.utils.io import (
    list_files,
    make_profile,
    read_raster,
    read_vector,
    write_raster,
    write_vector,
)
from shared.utils.logging import get_logger
from shared.utils.raster import apply_nodata_mask, clip_raster_to_bounds

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------

class TestLoadConfig:
    """Tests for shared.utils.config.load_config."""

    def test_loads_valid_yaml(self, tmp_yaml: Path):
        cfg = load_config(tmp_yaml)
        assert isinstance(cfg, dict)
        assert cfg["processing"]["crs"] == "EPSG:3310"

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_returns_nested_dict(self, tmp_yaml: Path):
        cfg = load_config(tmp_yaml)
        assert "data" in cfg
        assert cfg["data"]["output_dir"] == "/tmp/out"


# -----------------------------------------------------------------------
# CRS
# -----------------------------------------------------------------------

class TestValidateCRS:
    """Tests for shared.utils.crs.validate_crs."""

    def test_matching_crs(self):
        assert validate_crs("EPSG:3310") is True

    def test_non_matching_crs(self):
        assert validate_crs("EPSG:4326") is False

    def test_from_int(self):
        assert validate_crs(3310) is True

    def test_from_crs_object(self):
        assert validate_crs(CRS.from_epsg(3310)) is True

    def test_default_crs_is_3310(self):
        assert DEFAULT_CRS == CRS.from_epsg(3310)


# -----------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------

class TestRasterIO:
    """Tests for read_raster, write_raster, make_profile."""

    def test_round_trip(self, tmp_path: Path):
        profile = make_profile(bounds=(0, 0, 50, 50), resolution=10.0)
        data = np.ones((1, 5, 5), dtype=np.float32) * 42.0
        path = write_raster(tmp_path / "rt.tif", data, profile)
        read_data, read_profile = read_raster(path)
        np.testing.assert_array_equal(read_data, data)
        assert read_profile["width"] == 5
        assert read_profile["height"] == 5

    def test_write_2d_array(self, tmp_path: Path):
        profile = make_profile(bounds=(0, 0, 30, 30), resolution=10.0)
        data_2d = np.zeros((3, 3), dtype=np.float32)
        path = write_raster(tmp_path / "2d.tif", data_2d, profile)
        read_data, _ = read_raster(path)
        assert read_data.shape == (1, 3, 3)

    def test_make_profile_dimensions(self):
        profile = make_profile(bounds=(0, 0, 100, 200), resolution=10.0)
        assert profile["width"] == 10
        assert profile["height"] == 20
        assert profile["crs"] == DEFAULT_CRS

    def test_make_profile_custom_crs(self):
        profile = make_profile(
            bounds=(0, 0, 100, 100), resolution=10.0, crs="EPSG:4326"
        )
        assert CRS(profile["crs"]) == CRS.from_epsg(4326)

    def test_write_creates_parent_dirs(self, tmp_path: Path):
        profile = make_profile(bounds=(0, 0, 10, 10), resolution=10.0)
        data = np.zeros((1, 1, 1), dtype=np.float32)
        path = write_raster(tmp_path / "sub" / "dir" / "out.tif", data, profile)
        assert path.exists()


class TestVectorIO:
    """Tests for read_vector and write_vector."""

    def test_round_trip_gpkg(self, tmp_path: Path, sample_gdf: gpd.GeoDataFrame):
        out = tmp_path / "test.gpkg"
        write_vector(sample_gdf, out)
        result = read_vector(out)
        assert len(result) == 2
        assert "name" in result.columns

    def test_round_trip_geojson(self, tmp_path: Path, sample_gdf: gpd.GeoDataFrame):
        out = tmp_path / "test.geojson"
        write_vector(sample_gdf, out)
        result = read_vector(out)
        assert len(result) == 2

    def test_write_creates_parent_dirs(self, tmp_path: Path, sample_gdf: gpd.GeoDataFrame):
        out = tmp_path / "nested" / "dir" / "data.gpkg"
        write_vector(sample_gdf, out)
        assert out.exists()


class TestListFiles:
    """Tests for list_files."""

    def test_finds_matching_files(self, tmp_path: Path):
        (tmp_path / "a.txt").touch()
        (tmp_path / "b.txt").touch()
        (tmp_path / "c.csv").touch()
        result = list_files(tmp_path, "*.txt")
        assert len(result) == 2

    def test_no_matches_returns_empty(self, tmp_path: Path):
        result = list_files(tmp_path, "*.xyz")
        assert result == []

    def test_returns_sorted(self, tmp_path: Path):
        (tmp_path / "b.laz").touch()
        (tmp_path / "a.laz").touch()
        result = list_files(tmp_path, "*.laz")
        assert result[0].name == "a.laz"


# -----------------------------------------------------------------------
# Raster processing
# -----------------------------------------------------------------------

class TestApplyNodataMask:
    """Tests for raster.apply_nodata_mask."""

    def test_replaces_nodata_with_nan(self):
        data = np.array([1.0, -9999.0, 3.0])
        result = apply_nodata_mask(data, nodata=-9999.0)
        assert np.isnan(result[1])
        assert result[0] == 1.0
        assert result[2] == 3.0

    def test_preserves_valid_values(self):
        data = np.array([10.0, 20.0, 30.0])
        result = apply_nodata_mask(data, nodata=-9999.0)
        np.testing.assert_array_equal(result, data)

    def test_custom_fill(self):
        data = np.array([1.0, -9999.0])
        result = apply_nodata_mask(data, nodata=-9999.0, fill=0.0)
        assert result[1] == 0.0


class TestClipRasterToBounds:
    """Tests for raster.clip_raster_to_bounds."""

    def test_clips_correctly(self, tmp_raster: Path):
        data, profile = read_raster(tmp_raster)
        # Clip to the upper-left quarter (bounds in CRS coordinates)
        clipped, new_profile = clip_raster_to_bounds(
            data, profile, bounds=(0.0, 50.0, 50.0, 100.0)
        )
        assert clipped.shape[1] <= profile["height"]
        assert clipped.shape[2] <= profile["width"]
        assert new_profile["width"] == clipped.shape[2]
        assert new_profile["height"] == clipped.shape[1]

    def test_full_bounds_returns_same(self, tmp_raster: Path):
        data, profile = read_raster(tmp_raster)
        clipped, _ = clip_raster_to_bounds(
            data, profile, bounds=(0.0, 0.0, 100.0, 100.0)
        )
        assert clipped.shape == data.shape

    def test_clip_raster_2d_array(self):
        """Clipping a 2-D raster (no band dimension) produces correct shape."""
        profile = make_profile(bounds=(0.0, 0.0, 100.0, 100.0), resolution=10.0)
        data_2d = np.arange(100, dtype=np.float32).reshape(10, 10)

        clipped, new_profile = clip_raster_to_bounds(
            data_2d, profile, bounds=(0.0, 50.0, 50.0, 100.0)
        )
        assert clipped.ndim == 2
        assert clipped.shape == (new_profile["height"], new_profile["width"])


class TestReprojectRaster:
    """Tests for raster.reproject_raster."""

    def test_changes_crs(self, tmp_raster: Path, tmp_path: Path):
        from shared.utils.raster import reproject_raster

        dst = tmp_path / "reprojected.tif"
        reproject_raster(tmp_raster, dst, dst_crs="EPSG:4326")
        _, profile = read_raster(dst)
        assert CRS(profile["crs"]) == CRS.from_epsg(4326)

    def test_output_exists(self, tmp_raster: Path, tmp_path: Path):
        from shared.utils.raster import reproject_raster

        dst = tmp_path / "out.tif"
        result = reproject_raster(tmp_raster, dst)
        assert result.exists()


class TestResampleRaster:
    """Tests for raster.resample_raster."""

    def test_changes_resolution(self, tmp_raster: Path, tmp_path: Path):
        from shared.utils.raster import resample_raster

        dst = tmp_path / "resampled.tif"
        resample_raster(tmp_raster, dst, target_resolution=20.0)
        _, profile = read_raster(dst)
        # Coarser resolution → fewer pixels
        assert profile["width"] <= 10
        assert profile["height"] <= 10


# -----------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------

class TestGetLogger:
    """Tests for shared.utils.logging.get_logger."""

    def test_returns_logger(self):
        lgr = get_logger("test_logger_1")
        assert isinstance(lgr, logging.Logger)

    def test_correct_level(self):
        lgr = get_logger("test_logger_2", level="DEBUG")
        assert lgr.level == logging.DEBUG

    def test_no_duplicate_handlers(self):
        name = "test_logger_no_dup"
        lgr1 = get_logger(name)
        lgr2 = get_logger(name)
        assert lgr1 is lgr2
        assert len(lgr2.handlers) == 1


# -----------------------------------------------------------------------
# Synthetic data generation
# -----------------------------------------------------------------------

class TestGenerateAll:
    """Tests for shared.data.generate_synthetic.generate_all."""

    def test_generate_all_creates_expected_files(self, tmp_path: Path):
        from shared.data.generate_synthetic import generate_all

        paths = generate_all(tmp_path)
        assert isinstance(paths, dict)
        assert len(paths) > 0

        # Verify every returned path actually exists on disk
        for name, p in paths.items():
            assert Path(p).exists(), f"Expected file '{name}' not found at {p}"

        # Verify key datasets are present
        expected_keys = {
            "lidar_csv", "cruise_plots", "chm", "dtm",  # P3
            "pre_nir", "pre_swir", "post_nir", "post_swir", "fire_perimeter",  # P1
            "occurrences",  # P4
            "geopackage",  # P2
        }
        assert expected_keys.issubset(set(paths.keys())), (
            f"Missing keys: {expected_keys - set(paths.keys())}"
        )

    def test_cruise_plots_without_reference(self, tmp_path: Path):
        """generate_synthetic_cruise_plots uses fallback when no ref CSV exists."""
        import pandas as pd

        from shared.data.generate_synthetic import generate_synthetic_cruise_plots

        # Call on a fresh directory with no synthetic_tree_reference.csv
        path = generate_synthetic_cruise_plots(tmp_path, n_trees=3)
        assert path.exists()
        df = pd.read_csv(path)
        assert len(df) == 3
        assert "dbh_inches" in df.columns
