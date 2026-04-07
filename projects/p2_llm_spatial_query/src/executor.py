"""Execute validated SQL against a SpatiaLite-enabled GeoPackage.

Loads the ``mod_spatialite`` extension into a sqlite3 connection and runs
the user query.  If the result contains a geometry column the output is
returned as a GeoDataFrame; otherwise a plain DataFrame is returned.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from shared.utils.logging import get_logger

logger = get_logger("p2_executor")


def load_spatialite(conn: sqlite3.Connection) -> None:
    """Load the SpatiaLite extension into *conn*.

    Tries the common shared-library names in order:
    ``mod_spatialite``, ``libspatialite``.

    Parameters
    ----------
    conn : sqlite3.Connection
        An open SQLite connection.

    Raises
    ------
    RuntimeError
        If SpatiaLite cannot be loaded.
    """
    conn.enable_load_extension(True)

    for lib in ("mod_spatialite", "libspatialite"):
        try:
            conn.load_extension(lib)
            logger.debug("Loaded SpatiaLite via %s", lib)
            return
        except sqlite3.OperationalError:
            continue

    raise RuntimeError(
        "Could not load SpatiaLite extension. "
        "Ensure mod_spatialite is installed and on the library path."
    )


def execute_query(
    sql: str,
    gpkg_path: str | Path,
    config: dict[str, Any],
) -> gpd.GeoDataFrame | pd.DataFrame:
    """Execute a validated SQL query against a GeoPackage.

    Opens the GeoPackage via sqlite3 with the SpatiaLite extension,
    executes the query, and returns results as a DataFrame or
    GeoDataFrame.

    Parameters
    ----------
    sql : str
        A validated SELECT statement.
    gpkg_path : str or Path
        Path to the GeoPackage file.
    config : dict
        Project configuration dict.

    Returns
    -------
    GeoDataFrame or DataFrame
        Query results.  A GeoDataFrame is returned when the result
        includes a ``geometry`` column; otherwise a plain DataFrame.

    Raises
    ------
    FileNotFoundError
        If *gpkg_path* does not exist.
    sqlite3.OperationalError
        If the SQL fails to execute.
    """
    gpkg_path = Path(gpkg_path)
    if not gpkg_path.exists():
        raise FileNotFoundError(f"GeoPackage not found: {gpkg_path}")

    conn = sqlite3.connect(str(gpkg_path))
    try:
        # Attempt to load SpatiaLite for spatial function support.
        # If it is not available we fall back to plain SQLite, which
        # still works for non-spatial queries.
        try:
            load_spatialite(conn)
        except RuntimeError:
            logger.warning(
                "SpatiaLite not available — spatial functions will not work"
            )

        logger.info("Executing SQL: %s", sql)
        df = pd.read_sql_query(sql, conn)

    finally:
        conn.close()

    # Attempt to promote to GeoDataFrame if a geometry column is present
    if "geometry" in df.columns:
        try:
            df["geometry"] = gpd.GeoSeries.from_wkb(df["geometry"])
            gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:3310")
            logger.info("Returning GeoDataFrame with %d rows", len(gdf))
            return gdf
        except Exception:
            logger.warning("Could not parse geometry column — returning DataFrame")

    logger.info("Returning DataFrame with %d rows", len(df))
    return df
