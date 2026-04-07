"""Format query results and errors for human-readable output.

Produces concise textual summaries of DataFrames and GeoDataFrames that
are suitable for display in a notebook, CLI, or chat interface.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd

from shared.utils.logging import get_logger

logger = get_logger("p2_formatter")

MAX_DISPLAY_ROWS = 10


def format_results(
    results: gpd.GeoDataFrame | pd.DataFrame,
    user_query: str,
) -> str:
    """Create a human-readable summary of query results.

    Parameters
    ----------
    results : GeoDataFrame or DataFrame
        Query output.
    user_query : str
        The original natural language question (for context).

    Returns
    -------
    str
        Formatted summary string.
    """
    lines: list[str] = []
    lines.append(f"Query: {user_query}")
    lines.append(f"Rows returned: {len(results)}")
    lines.append(f"Columns: {', '.join(results.columns.tolist())}")
    lines.append("")

    if len(results) == 0:
        lines.append("No results found.")
        return "\n".join(lines)

    # Show first rows
    n_show = min(len(results), MAX_DISPLAY_ROWS)
    display_df = results.head(n_show)

    # If GeoDataFrame, summarise geometry rather than printing WKB
    if isinstance(results, gpd.GeoDataFrame) and "geometry" in results.columns:
        display_copy = display_df.drop(columns=["geometry"])
        geom_types = results.geometry.geom_type.value_counts().to_dict()
        lines.append(f"Geometry types: {geom_types}")
        bounds = results.total_bounds
        lines.append(
            f"Spatial extent: "
            f"xmin={bounds[0]:.1f}, ymin={bounds[1]:.1f}, "
            f"xmax={bounds[2]:.1f}, ymax={bounds[3]:.1f}"
        )
        lines.append("")
        lines.append(display_copy.to_string(index=False))
    else:
        lines.append(display_df.to_string(index=False))

    if len(results) > MAX_DISPLAY_ROWS:
        lines.append(f"\n... ({len(results) - MAX_DISPLAY_ROWS} more rows)")

    return "\n".join(lines)


def format_error(error_msg: str, user_query: str) -> str:
    """Create a user-friendly error message with suggestions.

    Parameters
    ----------
    error_msg : str
        The technical error message.
    user_query : str
        The original natural language question.

    Returns
    -------
    str
        Formatted error string with guidance for the user.
    """
    lines: list[str] = [
        f"Query: {user_query}",
        "",
        f"Error: {error_msg}",
        "",
        "Suggestions:",
        "  - Rephrase your question using simpler terms.",
        "  - Ensure you are asking about available layers: harvest_units, "
        "streams, roads, sensitive_habitats, ownership_parcels, lidar_tile_index.",
        "  - Spatial distances should be specified in feet or meters.",
        "  - Only read-only (SELECT) queries are supported.",
    ]
    return "\n".join(lines)
