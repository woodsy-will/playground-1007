"""Tests for the result and error formatter module.

Covers human-readable formatting of DataFrames, GeoDataFrames, empty
results, row truncation, and user-friendly error messages.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from projects.p2_llm_spatial_query.src.formatter import format_error, format_results

# -----------------------------------------------------------------------
# format_results
# -----------------------------------------------------------------------

class TestFormatResults:
    """Tests for format_results()."""

    def test_empty_dataframe(self) -> None:
        """An empty DataFrame should report 0 rows and 'No results found.'."""
        df = pd.DataFrame(columns=["unit_id", "unit_name", "acres"])
        result = format_results(df, "Show all units")
        assert "Rows returned: 0" in result
        assert "No results found." in result

    def test_with_data_includes_row_count(self) -> None:
        """Non-empty DataFrame output must contain the correct row count."""
        df = pd.DataFrame({
            "unit_id": [1, 2, 3],
            "unit_name": ["Alpha", "Bravo", "Charlie"],
            "acres": [10.0, 20.0, 30.0],
        })
        result = format_results(df, "List harvest units")
        assert "Rows returned: 3" in result
        assert "Query: List harvest units" in result

    def test_geodataframe_includes_geometry_types(self) -> None:
        """GeoDataFrame output must report geometry types."""
        gdf = gpd.GeoDataFrame(
            {"name": ["a", "b"]},
            geometry=[Point(0, 0), Point(1, 1)],
            crs="EPSG:3310",
        )
        result = format_results(gdf, "Find points")
        assert "Geometry types:" in result
        assert "Point" in result

    def test_geodataframe_includes_extent(self) -> None:
        """GeoDataFrame output must report spatial extent."""
        gdf = gpd.GeoDataFrame(
            {"name": ["a", "b"]},
            geometry=[Point(100, 200), Point(300, 400)],
            crs="EPSG:3310",
        )
        result = format_results(gdf, "Extent check")
        assert "Spatial extent:" in result
        assert "xmin=100.0" in result
        assert "ymin=200.0" in result
        assert "xmax=300.0" in result
        assert "ymax=400.0" in result

    def test_truncation_over_10_rows(self) -> None:
        """Results exceeding MAX_DISPLAY_ROWS (10) must show a truncation message."""
        df = pd.DataFrame({
            "id": list(range(15)),
            "val": [f"v{i}" for i in range(15)],
        })
        result = format_results(df, "Many rows")
        assert "Rows returned: 15" in result
        assert "5 more rows" in result

    def test_plain_dataframe(self) -> None:
        """A plain (non-geo) DataFrame should not mention geometry or extent."""
        df = pd.DataFrame({
            "unit_id": [1],
            "acres": [42.5],
        })
        result = format_results(df, "Plain data")
        assert "Geometry types:" not in result
        assert "Spatial extent:" not in result
        assert "42.5" in result


# -----------------------------------------------------------------------
# format_error
# -----------------------------------------------------------------------

class TestFormatError:
    """Tests for format_error()."""

    def test_includes_query(self) -> None:
        """The error output must contain the original user query."""
        result = format_error("table not found", "Show all trees")
        assert "Query: Show all trees" in result

    def test_includes_error_msg(self) -> None:
        """The error output must contain the technical error message."""
        result = format_error("no such table: bogus", "Select from bogus")
        assert "no such table: bogus" in result

    def test_includes_suggestions(self) -> None:
        """The error output must contain actionable suggestions."""
        result = format_error("syntax error", "Bad query")
        assert "Suggestions:" in result
        assert "Rephrase" in result
        assert "SELECT" in result
