"""Treetop detection via variable-window local maxima filtering.

Applies a height-adaptive maximum filter to a Canopy Height Model,
returning point locations of detected treetops.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from scipy.ndimage import maximum_filter
from shapely.geometry import Point

from shared.utils.io import read_raster
from shared.utils.logging import get_logger

logger = get_logger("p3_treetops")


def _window_size_from_height(height: float) -> int:
    """Compute local-maximum window size (in pixels) from tree height.

    Uses a simple linear model calibrated for Sierra Nevada mixed-conifer:
        window_m = 1.5 + 0.15 * height_m

    The result is rounded up to the next odd integer so it can serve as
    a symmetric filter kernel size.

    Parameters
    ----------
    height : float
        Tree height in meters.

    Returns
    -------
    int
        Odd-valued window size in pixels (assumes 1 m resolution).
    """
    win = 1.5 + 0.15 * height
    win_int = int(np.ceil(win))
    if win_int % 2 == 0:
        win_int += 1
    return max(win_int, 3)


def detect_treetops(
    chm_path: str | Path,
    config: dict[str, Any],
) -> gpd.GeoDataFrame:
    """Detect individual treetops from a CHM using variable-window local maxima.

    Parameters
    ----------
    chm_path : str or Path
        Path to Canopy Height Model GeoTIFF.
    config : dict
        Project configuration dictionary.

    Returns
    -------
    GeoDataFrame
        Point features with columns ``tree_id``, ``height``, and
        ``geometry`` (Point in the raster CRS).
    """
    chm_path = Path(chm_path)
    proc = config.get("processing", {})
    min_height = proc.get("min_tree_height", 5.0)
    crs = proc.get("crs", "EPSG:3310")

    chm_data, profile = read_raster(chm_path)
    chm = chm_data[0].astype(np.float64)
    transform = profile["transform"]

    # Mask pixels below minimum tree height
    mask = chm >= min_height

    # Variable-window local maxima: iterate over a set of candidate window
    # sizes and mark pixels that are local maxima at the appropriate scale.
    height_bins = np.arange(min_height, float(np.nanmax(chm)) + 5.0, 2.0)
    is_max = np.zeros_like(chm, dtype=bool)

    for h_low, h_high in zip(height_bins[:-1], height_bins[1:]):
        win = _window_size_from_height((h_low + h_high) / 2.0)
        local_max = maximum_filter(chm, size=win)
        band_mask = mask & (chm >= h_low) & (chm < h_high) & (chm == local_max)
        is_max |= band_mask

    # Catch the tallest bin edge
    win = _window_size_from_height(float(height_bins[-1]))
    local_max = maximum_filter(chm, size=win)
    band_mask = mask & (chm >= height_bins[-1]) & (chm == local_max)
    is_max |= band_mask

    rows, cols = np.where(is_max)
    xs = transform.c + (cols + 0.5) * transform.a
    ys = transform.f + (rows + 0.5) * transform.e
    heights = chm[rows, cols]

    logger.info(
        "Detected %d treetops (min_height=%.1f m)",
        len(rows),
        min_height,
    )

    gdf = gpd.GeoDataFrame(
        {
            "tree_id": np.arange(1, len(rows) + 1),
            "height": heights,
        },
        geometry=[Point(x, y) for x, y in zip(xs, ys)],
        crs=crs,
    )
    return gdf
