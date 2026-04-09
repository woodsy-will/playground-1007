"""Tests for cloud masking and reprojection preprocessing."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from shapely.geometry import box

from projects.p1_burn_severity.src.preprocessing import apply_cloud_mask


class TestApplyCloudMask:
    """Tests for apply_cloud_mask."""

    def test_masks_cloud_pixels(self, config: dict, scl_array: np.ndarray) -> None:
        data = np.ones((20, 20), dtype=np.float32)
        masked = apply_cloud_mask(data, scl_array, config)
        # SCL=9 block at [2:5, 2:5] and SCL=3 at [7,7] should be NaN
        assert np.isnan(masked[3, 3])
        assert np.isnan(masked[7, 7])

    def test_preserves_non_cloud(self, config: dict, scl_array: np.ndarray) -> None:
        data = np.full((20, 20), 42.0, dtype=np.float32)
        masked = apply_cloud_mask(data, scl_array, config)
        # Non-cloud pixel should remain
        assert masked[0, 0] == 42.0
        assert masked[10, 10] == 42.0

    def test_3d_input(self, config: dict, scl_array: np.ndarray) -> None:
        data = np.ones((2, 20, 20), dtype=np.float32)
        masked = apply_cloud_mask(data, scl_array, config)
        assert masked.shape == (2, 20, 20)
        assert np.isnan(masked[0, 3, 3])
        assert np.isnan(masked[1, 3, 3])

    def test_output_dtype(self, config: dict, scl_array: np.ndarray) -> None:
        data = np.ones((20, 20), dtype=np.float32)
        masked = apply_cloud_mask(data, scl_array, config)
        assert masked.dtype == np.float64

    def test_cloud_count(self, config: dict, scl_array: np.ndarray) -> None:
        data = np.ones((20, 20), dtype=np.float32)
        masked = apply_cloud_mask(data, scl_array, config)
        # 3x3 block of class 9 = 9 pixels + 1 pixel of class 3 = 10
        nan_count = np.count_nonzero(np.isnan(masked))
        assert nan_count == 10


def _write_test_raster(path: Path, data: np.ndarray, crs: str, bounds: tuple) -> None:
    """Write a small GeoTIFF for testing."""
    h, w = data.shape[-2], data.shape[-1]
    transform = from_bounds(*bounds, w, h)
    count = data.shape[0] if data.ndim == 3 else 1
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    profile = {
        "driver": "GTiff",
        "dtype": data.dtype.name,
        "width": w,
        "height": h,
        "count": count,
        "crs": CRS.from_user_input(crs),
        "transform": transform,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)


def _write_test_perimeter(path: Path, bounds: tuple, crs: str) -> None:
    """Write a GeoPackage with a single bounding-box polygon."""
    gdf = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[box(*bounds)],
        crs=crs,
    )
    gdf.to_file(path, driver="GPKG")


class TestReprojectAndClip:
    """Tests for reproject_and_clip with CRS handling."""

    def test_reprojection_changes_crs(self, tmp_path: Path, config: dict) -> None:
        """Raster in EPSG:4326 should be reprojected to EPSG:3310."""
        from projects.p1_burn_severity.src.preprocessing import reproject_and_clip

        # Create a small raster in WGS84 (lon/lat) over the Sierra Nevada
        data = np.random.default_rng(42).uniform(0.1, 0.5, (20, 20)).astype(np.float32)
        raster_path = tmp_path / "test_4326.tif"
        _write_test_raster(
            raster_path, data, "EPSG:4326",
            bounds=(-120.5, 37.5, -120.0, 38.0),
        )

        # Perimeter in the same WGS84 area
        perim_path = tmp_path / "perim.gpkg"
        _write_test_perimeter(
            perim_path,
            bounds=(-120.4, 37.6, -120.1, 37.9),
            crs="EPSG:4326",
        )

        clipped, profile = reproject_and_clip(raster_path, perim_path, config)

        # Output CRS should be EPSG:3310
        out_crs = CRS.from_user_input(profile["crs"])
        assert out_crs == CRS.from_epsg(3310)
        assert clipped.size > 0

    def test_same_crs_skips_reprojection(self, tmp_path: Path, config: dict) -> None:
        """Raster already in EPSG:3310 should not be reprojected."""
        from projects.p1_burn_severity.src.preprocessing import reproject_and_clip

        # Create a raster already in EPSG:3310
        data = np.ones((10, 10), dtype=np.float32)
        raster_path = tmp_path / "test_3310.tif"
        _write_test_raster(
            raster_path, data, "EPSG:3310",
            bounds=(-200_000, -50_000, -199_000, -49_000),
        )

        perim_path = tmp_path / "perim.gpkg"
        _write_test_perimeter(
            perim_path,
            bounds=(-199_800, -49_800, -199_200, -49_200),
            crs="EPSG:3310",
        )

        clipped, profile = reproject_and_clip(raster_path, perim_path, config)

        out_crs = CRS.from_user_input(profile["crs"])
        assert out_crs == CRS.from_epsg(3310)
        # Clipped should be smaller than original
        assert clipped.shape[-2] <= 10
        assert clipped.shape[-1] <= 10

    def test_configurable_resampling(self, tmp_path: Path, config: dict) -> None:
        """Resampling method should be read from config."""
        from projects.p1_burn_severity.src.preprocessing import reproject_and_clip

        data = np.arange(100, dtype=np.float32).reshape(10, 10)
        raster_path = tmp_path / "test_4326.tif"
        _write_test_raster(
            raster_path, data, "EPSG:4326",
            bounds=(-120.5, 37.5, -120.0, 38.0),
        )

        perim_path = tmp_path / "perim.gpkg"
        _write_test_perimeter(
            perim_path,
            bounds=(-120.4, 37.6, -120.1, 37.9),
            crs="EPSG:4326",
        )

        # Use nearest resampling
        cfg = dict(config)
        cfg["processing"] = {**cfg["processing"], "resampling": "nearest"}

        clipped, profile = reproject_and_clip(raster_path, perim_path, cfg)
        assert clipped.size > 0
        assert CRS.from_user_input(profile["crs"]) == CRS.from_epsg(3310)

    def test_output_values_preserved(self, tmp_path: Path, config: dict) -> None:
        """Reprojection should not introduce extreme value distortion."""
        from projects.p1_burn_severity.src.preprocessing import reproject_and_clip

        rng = np.random.default_rng(99)
        data = rng.uniform(0.2, 0.8, (20, 20)).astype(np.float32)
        raster_path = tmp_path / "test_4326.tif"
        _write_test_raster(
            raster_path, data, "EPSG:4326",
            bounds=(-120.5, 37.5, -120.0, 38.0),
        )

        perim_path = tmp_path / "perim.gpkg"
        _write_test_perimeter(
            perim_path,
            bounds=(-120.45, 37.55, -120.05, 37.95),
            crs="EPSG:4326",
        )

        clipped, _ = reproject_and_clip(raster_path, perim_path, config)
        valid = clipped[np.isfinite(clipped)]
        # Bilinear resampling should keep values in roughly the same range
        assert valid.min() >= 0.0
        assert valid.max() <= 1.0
