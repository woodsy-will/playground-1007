"""Species occurrence data loading, thinning, and source splitting.

Handles GeoPackage and CSV formats, reprojects to EPSG:3310, and applies
spatial thinning to reduce sampling bias.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from pyproj import CRS

from shared.utils.logging import get_logger

logger = get_logger("p4_occurrences")


def load_occurrences(config: dict[str, Any]) -> gpd.GeoDataFrame:
    """Load species occurrence records from GeoPackage or CSV.

    Parameters
    ----------
    config : dict
        Project configuration. Expected keys:
        - ``data.occurrences_path``: path to occurrence file (.gpkg or .csv).
        - ``modeling.crs``: target CRS string (default EPSG:3310).

    Returns
    -------
    GeoDataFrame
        Occurrence records projected to the target CRS.
    """
    data_cfg = config.get("data", {})
    occ_path = Path(data_cfg["occurrences_path"])
    target_crs = CRS(config.get("modeling", {}).get("crs", "EPSG:3310"))

    if occ_path.suffix.lower() == ".csv":
        import pandas as pd

        df = pd.read_csv(occ_path)
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["x"], df["y"]),
            crs=target_crs,
        )
    else:
        gdf = gpd.read_file(occ_path)

    if gdf.crs is None:
        gdf = gdf.set_crs(target_crs)
    elif CRS(gdf.crs) != target_crs:
        gdf = gdf.to_crs(target_crs)

    logger.info("Loaded %d occurrences from %s", len(gdf), occ_path.name)
    return gdf


def thin_occurrences(
    gdf: gpd.GeoDataFrame,
    distance_km: float,
    config: dict[str, Any] | None = None,
) -> gpd.GeoDataFrame:
    """Spatially thin occurrence records using a greedy distance filter.

    For each point (in random order), removes all other points within
    ``distance_km`` kilometres.  Assumes the GeoDataFrame is already in
    a projected CRS with metre units.

    Parameters
    ----------
    gdf : GeoDataFrame
        Occurrence records in a projected CRS.
    distance_km : float
        Minimum distance between retained points, in kilometres.
    config : dict, optional
        Project configuration (unused, reserved for future options).

    Returns
    -------
    GeoDataFrame
        Thinned subset of the input records.
    """
    if len(gdf) == 0:
        return gdf.copy()

    distance_m = distance_km * 1000.0
    gdf = gdf.copy().reset_index(drop=True)
    indices = list(gdf.index)
    rng = np.random.default_rng(42)
    rng.shuffle(indices)

    sindex = gdf.sindex
    keep: list[int] = []
    removed: set[int] = set()

    for idx in indices:
        if idx in removed:
            continue
        keep.append(idx)
        point = gdf.geometry.iloc[idx]
        buffer = point.buffer(distance_m)
        candidates = list(sindex.intersection(buffer.bounds))
        for c in candidates:
            if c != idx and c not in removed:
                if gdf.geometry.iloc[c].distance(point) < distance_m:
                    removed.add(c)

    result = gdf.loc[keep].copy().reset_index(drop=True)
    logger.info(
        "Thinned %d -> %d occurrences (distance=%.1f km)",
        len(gdf), len(result), distance_km,
    )
    return result


def split_sources(gdf: gpd.GeoDataFrame) -> dict[str, gpd.GeoDataFrame]:
    """Split occurrence records by data source.

    Parameters
    ----------
    gdf : GeoDataFrame
        Occurrence records with a ``source`` column.

    Returns
    -------
    dict[str, GeoDataFrame]
        Mapping of source name to subset GeoDataFrame.
    """
    if "source" not in gdf.columns:
        return {"unknown": gdf.copy()}
    return {name: group.copy() for name, group in gdf.groupby("source")}
