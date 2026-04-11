"""Tests for occurrence loading and spatial thinning."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from pyproj import CRS
from shapely.geometry import Point


class TestLoadOccurrences:
    """Verify occurrence loading and CRS reprojection."""

    def test_load_returns_geodataframe(self, default_config: dict):
        from projects.p4_habitat_suitability.src.occurrences import load_occurrences

        gdf = load_occurrences(default_config)
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) > 0

    def test_load_crs_is_3310(self, default_config: dict):
        from projects.p4_habitat_suitability.src.occurrences import load_occurrences

        gdf = load_occurrences(default_config)
        assert CRS(gdf.crs) == CRS.from_epsg(3310)

    def test_load_occurrences_from_csv(self, tmp_path: Path):
        """Loading a CSV file with x, y, species columns returns a GeoDataFrame."""
        from projects.p4_habitat_suitability.src.occurrences import load_occurrences

        csv_path = tmp_path / "occ.csv"
        df = pd.DataFrame({
            "x": [-200_000.0, -199_990.0, -199_980.0],
            "y": [-50_000.0, -49_990.0, -49_980.0],
            "species": ["Pekania pennanti"] * 3,
        })
        df.to_csv(csv_path, index=False)

        config = {
            "data": {"occurrences_path": str(csv_path)},
            "modeling": {"crs": "EPSG:3310"},
        }
        gdf = load_occurrences(config)
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 3
        assert CRS(gdf.crs) == CRS.from_epsg(3310)

    def test_load_occurrences_from_gpkg(self, tmp_path: Path):
        """Loading a GeoPackage file returns a GeoDataFrame with correct CRS."""
        from projects.p4_habitat_suitability.src.occurrences import load_occurrences

        gpkg_path = tmp_path / "occ.gpkg"
        gdf_in = gpd.GeoDataFrame(
            {"species": ["test"] * 2},
            geometry=[Point(-200_000, -50_000), Point(-199_990, -49_990)],
            crs="EPSG:3310",
        )
        gdf_in.to_file(gpkg_path, driver="GPKG")

        config = {
            "data": {"occurrences_path": str(gpkg_path)},
            "modeling": {"crs": "EPSG:3310"},
        }
        gdf = load_occurrences(config)
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 2
        assert CRS(gdf.crs) == CRS.from_epsg(3310)

    def test_load_occurrences_from_gpkg_no_crs(self, tmp_path: Path):
        """Loading a GeoPackage without CRS sets the target CRS."""
        from projects.p4_habitat_suitability.src.occurrences import load_occurrences

        gpkg_path = tmp_path / "occ_nocrs.gpkg"
        gdf_in = gpd.GeoDataFrame(
            {"species": ["test"]},
            geometry=[Point(-200_000, -50_000)],
        )
        # Write without CRS
        gdf_in.to_file(gpkg_path, driver="GPKG")

        config = {
            "data": {"occurrences_path": str(gpkg_path)},
            "modeling": {"crs": "EPSG:3310"},
        }
        gdf = load_occurrences(config)
        assert CRS(gdf.crs) == CRS.from_epsg(3310)

    def test_load_occurrences_reprojects_different_crs(self, tmp_path: Path):
        """Loading a file in EPSG:4326 reprojects to EPSG:3310."""
        from projects.p4_habitat_suitability.src.occurrences import load_occurrences

        gpkg_path = tmp_path / "occ_4326.gpkg"
        gdf_in = gpd.GeoDataFrame(
            {"species": ["test"]},
            geometry=[Point(-120.0, 37.0)],
            crs="EPSG:4326",
        )
        gdf_in.to_file(gpkg_path, driver="GPKG")

        config = {
            "data": {"occurrences_path": str(gpkg_path)},
            "modeling": {"crs": "EPSG:3310"},
        }
        gdf = load_occurrences(config)
        assert CRS(gdf.crs) == CRS.from_epsg(3310)


class TestThinOccurrences:
    """Verify spatial thinning reduces point count and preserves CRS."""

    def test_thinning_reduces_count(self, default_config: dict):
        from projects.p4_habitat_suitability.src.occurrences import (
            load_occurrences,
            thin_occurrences,
        )

        gdf = load_occurrences(default_config)
        # Use a very small distance so at least some points survive
        thinned = thin_occurrences(gdf, distance_km=0.01)
        assert len(thinned) <= len(gdf)
        assert len(thinned) > 0

    def test_thinning_preserves_crs(self, default_config: dict):
        from projects.p4_habitat_suitability.src.occurrences import (
            load_occurrences,
            thin_occurrences,
        )

        gdf = load_occurrences(default_config)
        thinned = thin_occurrences(gdf, distance_km=0.01)
        assert CRS(thinned.crs) == CRS(gdf.crs)

    def test_large_distance_thins_aggressively(self, default_config: dict):
        from projects.p4_habitat_suitability.src.occurrences import (
            load_occurrences,
            thin_occurrences,
        )

        gdf = load_occurrences(default_config)
        thinned = thin_occurrences(gdf, distance_km=100.0)
        # With 100 km thinning on a ~200 m AOI, only 1 point should remain
        assert len(thinned) == 1

    def test_thin_occurrences_zero_distance(self, default_config: dict):
        """Thinning with distance=0 should return all points unchanged."""
        from projects.p4_habitat_suitability.src.occurrences import (
            load_occurrences,
            thin_occurrences,
        )

        gdf = load_occurrences(default_config)
        thinned = thin_occurrences(gdf, distance_km=0)
        assert len(thinned) == len(gdf)

    def test_thin_occurrences_empty_geodataframe(self):
        """Thinning an empty GeoDataFrame returns an empty copy."""
        from projects.p4_habitat_suitability.src.occurrences import thin_occurrences

        empty = gpd.GeoDataFrame(
            {"species": pd.Series([], dtype=str)},
            geometry=[],
            crs="EPSG:3310",
        )
        result = thin_occurrences(empty, distance_km=1.0)
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 0


class TestSplitSources:
    """Verify source splitting produces expected groups."""

    def test_split_returns_known_sources(self, default_config: dict):
        from projects.p4_habitat_suitability.src.occurrences import (
            load_occurrences,
            split_sources,
        )

        gdf = load_occurrences(default_config)
        splits = split_sources(gdf)
        assert isinstance(splits, dict)
        assert len(splits) > 0
        for name, sub in splits.items():
            assert isinstance(sub, gpd.GeoDataFrame)
            assert len(sub) > 0

    def test_split_sources_no_source_column(self):
        """Without a 'source' column, returns {'unknown': gdf.copy()}."""
        from projects.p4_habitat_suitability.src.occurrences import split_sources

        gdf = gpd.GeoDataFrame(
            {"species": ["test_a", "test_b"]},
            geometry=[Point(0, 0), Point(1, 1)],
            crs="EPSG:3310",
        )
        splits = split_sources(gdf)
        assert list(splits.keys()) == ["unknown"]
        assert len(splits["unknown"]) == 2
