"""Tests for occurrence loading and spatial thinning."""

from __future__ import annotations

import geopandas as gpd
from pyproj import CRS


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
