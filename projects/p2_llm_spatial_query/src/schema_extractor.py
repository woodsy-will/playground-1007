"""Extract and load GeoPackage schema metadata.

Provides introspection of GeoPackage tables via sqlite3 and loading of
pre-authored schema metadata from YAML configuration files.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import yaml

from shared.utils.logging import get_logger

logger = get_logger("p2_schema")


def extract_schema(gpkg_path: str | Path) -> dict[str, Any]:
    """Introspect a GeoPackage and return its schema structure.

    Opens the GeoPackage as a SQLite database and queries the
    ``gpkg_contents`` and ``gpkg_geometry_columns`` metadata tables to
    discover layers, columns, geometry types, and SRIDs.

    Parameters
    ----------
    gpkg_path : str or Path
        Path to the GeoPackage file.

    Returns
    -------
    dict
        Nested dict of ``{layer_name: {description, columns: {col: {type, ...}}}}``.

    Raises
    ------
    FileNotFoundError
        If *gpkg_path* does not exist.
    """
    gpkg_path = Path(gpkg_path)
    if not gpkg_path.exists():
        raise FileNotFoundError(f"GeoPackage not found: {gpkg_path}")

    schema: dict[str, Any] = {"layers": {}}

    conn = sqlite3.connect(str(gpkg_path))
    try:
        cursor = conn.cursor()

        # Discover layers from gpkg_contents
        cursor.execute(
            "SELECT table_name, identifier, description "
            "FROM gpkg_contents WHERE data_type IN ('features', 'tiles')"
        )
        layers = cursor.fetchall()

        # Build geometry info lookup
        geom_info: dict[str, dict[str, Any]] = {}
        try:
            cursor.execute(
                "SELECT table_name, column_name, geometry_type_name, srs_id "
                "FROM gpkg_geometry_columns"
            )
            for row in cursor.fetchall():
                geom_info[row[0]] = {
                    "column": row[1],
                    "type": row[2].lower() if row[2] else "geometry",
                    "srid": row[3],
                }
        except sqlite3.OperationalError:
            logger.warning("No gpkg_geometry_columns table found")

        for table_name, identifier, description in layers:
            layer_meta: dict[str, Any] = {
                "description": description or identifier or table_name,
                "columns": {},
            }

            # Query column info via PRAGMA
            cursor.execute(f"PRAGMA table_info('{table_name}')")
            for col_row in cursor.fetchall():
                col_name = col_row[1]
                col_type = (col_row[2] or "text").lower()

                # Check if this is the geometry column
                gi = geom_info.get(table_name, {})
                if col_name == gi.get("column"):
                    layer_meta["columns"][col_name] = {
                        "type": gi["type"],
                        "srid": gi["srid"],
                    }
                else:
                    layer_meta["columns"][col_name] = {"type": col_type}

            schema["layers"][table_name] = layer_meta

    finally:
        conn.close()

    logger.info("Extracted schema for %d layers", len(schema["layers"]))
    return schema


def load_schema_metadata(config: dict[str, Any]) -> dict[str, Any]:
    """Load pre-authored schema metadata from a YAML file.

    Parameters
    ----------
    config : dict
        Project configuration dict.  Expected key:
        ``config["rag"]["schema_metadata"]`` pointing to the YAML path
        (relative to the config file location or absolute).

    Returns
    -------
    dict
        Schema metadata dict with ``layers`` key.

    Raises
    ------
    FileNotFoundError
        If the schema metadata file does not exist.
    """
    meta_path = Path(config["rag"]["schema_metadata"])
    if not meta_path.exists():
        raise FileNotFoundError(f"Schema metadata not found: {meta_path}")

    with open(meta_path) as f:
        metadata = yaml.safe_load(f)

    logger.info("Loaded schema metadata for %d layers", len(metadata.get("layers", {})))
    return metadata
