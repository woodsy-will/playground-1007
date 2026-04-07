"""Crown segmentation via marker-controlled watershed on the inverted CHM.

Treetop locations serve as markers; the inverted CHM is the relief surface.
Resulting label regions are vectorised into crown polygons and filtered by
minimum area.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio.features
from scipy.ndimage import label as ndlabel
from shapely.geometry import shape

from shared.utils.io import read_raster
from shared.utils.logging import get_logger

logger = get_logger("p3_segment")


def segment_crowns(
    chm_path: str | Path,
    treetops_gdf: gpd.GeoDataFrame,
    config: dict[str, Any],
) -> gpd.GeoDataFrame:
    """Delineate individual tree crowns using marker-controlled watershed.

    Parameters
    ----------
    chm_path : str or Path
        Path to the CHM GeoTIFF.
    treetops_gdf : GeoDataFrame
        Treetop points (must contain ``tree_id`` and ``geometry``).
    config : dict
        Project configuration dictionary.

    Returns
    -------
    GeoDataFrame
        Crown polygons with columns ``tree_id``, ``crown_area_m2``,
        ``crown_diameter_m``, and ``geometry`` (Polygon).
    """
    skimage_seg = __import__(
        "skimage.segmentation", fromlist=["watershed"]
    )
    watershed = skimage_seg.watershed

    chm_path = Path(chm_path)
    proc = config.get("processing", {})
    min_height = proc.get("min_tree_height", 5.0)
    min_area = proc.get("min_crown_area", 4.0)
    crs = proc.get("crs", "EPSG:3310")

    chm_data, profile = read_raster(chm_path)
    chm = chm_data[0].astype(np.float64)
    transform = profile["transform"]

    # Build marker array from treetop points
    markers = np.zeros(chm.shape, dtype=np.int32)
    for _, row in treetops_gdf.iterrows():
        pt = row.geometry
        col = int((pt.x - transform.c) / transform.a)
        r = int((pt.y - transform.f) / transform.e)
        if 0 <= r < chm.shape[0] and 0 <= col < chm.shape[1]:
            markers[r, col] = int(row.tree_id)

    # Invert CHM for watershed (basins become peaks)
    inverted = chm.max() - chm

    # Mask: only segment where CHM >= min_tree_height
    seg_mask = chm >= min_height

    labels = watershed(
        inverted,
        markers=markers,
        mask=seg_mask,
    )

    # Vectorise labelled regions to polygons
    shapes_gen = rasterio.features.shapes(
        labels.astype(np.int32),
        transform=transform,
    )

    records: list[dict[str, Any]] = []
    for geom, value in shapes_gen:
        value = int(value)
        if value == 0:
            continue
        poly = shape(geom)
        area = poly.area
        if area < min_area:
            continue
        diameter = 2.0 * np.sqrt(area / np.pi)
        records.append(
            {
                "tree_id": value,
                "crown_area_m2": round(area, 2),
                "crown_diameter_m": round(diameter, 2),
                "geometry": poly,
            }
        )

    logger.info(
        "Segmented %d crowns (min_area=%.1f m²)",
        len(records),
        min_area,
    )

    gdf = gpd.GeoDataFrame(records, crs=crs)
    return gdf
