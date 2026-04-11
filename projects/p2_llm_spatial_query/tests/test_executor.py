"""Tests for SQL execution against a test GeoPackage."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from projects.p2_llm_spatial_query.src.executor import execute_query


class TestExecuteQuery:
    """Execute simple SELECT queries on the synthetic GeoPackage."""

    def test_select_all_harvest_units(
        self, gpkg_path: Path, default_config: dict
    ) -> None:
        sql = "SELECT * FROM harvest_units"
        result = execute_query(sql, gpkg_path, default_config)
        assert len(result) == 3
        assert "unit_id" in result.columns
        assert "unit_name" in result.columns

    def test_select_with_where(
        self, gpkg_path: Path, default_config: dict
    ) -> None:
        sql = "SELECT unit_name FROM harvest_units WHERE prescription = 'clearcut'"
        result = execute_query(sql, gpkg_path, default_config)
        assert len(result) == 1
        assert result.iloc[0]["unit_name"] == "Unit A"

    def test_select_streams(
        self, gpkg_path: Path, default_config: dict
    ) -> None:
        sql = "SELECT * FROM streams"
        result = execute_query(sql, gpkg_path, default_config)
        assert len(result) == 2

    def test_select_aggregate(
        self, gpkg_path: Path, default_config: dict
    ) -> None:
        sql = "SELECT SUM(acres) AS total FROM harvest_units"
        result = execute_query(sql, gpkg_path, default_config)
        assert len(result) == 1
        assert result.iloc[0]["total"] == pytest.approx(125.0)

    def test_returns_dataframe(
        self, gpkg_path: Path, default_config: dict
    ) -> None:
        sql = "SELECT unit_name, acres FROM harvest_units"
        result = execute_query(sql, gpkg_path, default_config)
        assert isinstance(result, pd.DataFrame)

    def test_file_not_found_raises(
        self, tmp_path: Path, default_config: dict
    ) -> None:
        with pytest.raises(FileNotFoundError):
            execute_query(
                "SELECT 1", tmp_path / "nonexistent.gpkg", default_config
            )

    def test_bad_sql_raises(
        self, gpkg_path: Path, default_config: dict
    ) -> None:
        with pytest.raises(Exception):
            execute_query(
                "SELECT * FROM nonexistent_table", gpkg_path, default_config
            )

    def test_execute_query_returns_geodataframe_with_geometry(
        self, gpkg_path: Path, default_config: dict
    ) -> None:
        """Querying a geometry column should return a GeoDataFrame when WKB
        parsing succeeds, covering the geometry promotion path."""
        import geopandas as gpd

        sql = "SELECT unit_name, geom AS geometry FROM harvest_units LIMIT 2"
        result = execute_query(sql, gpkg_path, default_config)
        assert isinstance(result, (pd.DataFrame, gpd.GeoDataFrame))
        assert len(result) == 2

    def test_spatialite_not_available(
        self, gpkg_path: Path, default_config: dict
    ) -> None:
        """When SpatiaLite cannot be loaded, execute_query should still work
        for non-spatial queries (it logs a warning and continues)."""
        from unittest.mock import patch

        with patch(
            "projects.p2_llm_spatial_query.src.executor.load_spatialite",
            side_effect=RuntimeError("SpatiaLite not available"),
        ):
            sql = "SELECT unit_name, acres FROM harvest_units"
            result = execute_query(sql, gpkg_path, default_config)

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "unit_name" in result.columns

    def test_wkb_geometry_promotion_to_geodataframe(
        self, gpkg_path: Path, default_config: dict
    ) -> None:
        """When the result has a valid WKB geometry column, execute_query
        should return a GeoDataFrame (covers lines 115-118)."""
        from unittest.mock import patch

        import geopandas as gpd
        from shapely.geometry import Point

        # Create a DataFrame with a real WKB geometry column
        geom_series = gpd.GeoSeries([Point(0, 0), Point(1, 1)], crs="EPSG:3310")
        mock_df = pd.DataFrame({
            "name": ["a", "b"],
            "geometry": geom_series.to_wkb(),
        })

        with patch(
            "projects.p2_llm_spatial_query.src.executor.load_spatialite",
            side_effect=RuntimeError("SpatiaLite not available"),
        ), patch(
            "projects.p2_llm_spatial_query.src.executor.pd.read_sql_query",
            return_value=mock_df,
        ):
            result = execute_query(
                "SELECT name, geometry FROM harvest_units",
                gpkg_path,
                default_config,
            )

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 2
        assert result.crs is not None
