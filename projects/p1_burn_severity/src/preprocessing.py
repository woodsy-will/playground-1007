"""Cloud masking and reprojection for Sentinel-2 burn severity imagery.

Applies Scene Classification Layer (SCL) masks and reprojects rasters to
EPSG:3310 clipped to a fire perimeter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from shared.utils.crs import DEFAULT_CRS
from shared.utils.io import read_raster, read_vector
from shared.utils.logging import get_logger
from shared.utils.raster import clip_raster_to_bounds

logger = get_logger("p1.preprocessing")

# SCL classes to mask: cloud shadows (3), cloud medium (8), cloud high (9),
# thin cirrus (10)
_CLOUD_SCL_CLASSES: list[int] = [3, 8, 9, 10]


def apply_cloud_mask(
    data: np.ndarray,
    scl_data: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    """Mask cloud and cloud-shadow pixels using the Sentinel-2 SCL band.

    Pixels classified as cloud shadow (3), cloud medium probability (8),
    cloud high probability (9), or thin cirrus (10) are set to ``np.nan``.

    Parameters
    ----------
    data : np.ndarray
        Raster data to mask (2-D or 3-D).
    scl_data : np.ndarray
        SCL classification raster (2-D, same spatial extent as *data*).
    config : dict
        Project configuration (reserved for future customisation of
        mask classes).

    Returns
    -------
    np.ndarray
        Masked copy of *data* with cloud pixels set to NaN.
    """
    masked = data.astype(np.float64, copy=True)
    cloud_mask = np.isin(scl_data, _CLOUD_SCL_CLASSES)

    if masked.ndim == 3:
        # Broadcast 2-D mask across bands
        masked[:, cloud_mask] = np.nan
    else:
        masked[cloud_mask] = np.nan

    n_masked = int(np.count_nonzero(cloud_mask))
    n_total = int(cloud_mask.size)
    logger.info(
        "Cloud mask applied: %d / %d pixels masked (%.1f%%)",
        n_masked,
        n_total,
        100.0 * n_masked / max(n_total, 1),
    )
    return masked


def reproject_and_clip(
    raster_path: str | Path,
    fire_perimeter_path: str | Path,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict]:
    """Reproject a raster to EPSG:3310 and clip to a fire perimeter.

    Parameters
    ----------
    raster_path : str or Path
        Path to input raster (GeoTIFF).
    fire_perimeter_path : str or Path
        Path to fire perimeter vector (GeoPackage / Shapefile).
    config : dict
        Project configuration (reads ``processing.crs``).

    Returns
    -------
    tuple[np.ndarray, dict]
        Clipped raster array and updated rasterio profile.
    """
    target_crs = config.get("processing", {}).get("crs", str(DEFAULT_CRS))

    # Read the raster
    data, profile = read_raster(raster_path)

    # Read fire perimeter and reproject to target CRS
    perimeter = read_vector(fire_perimeter_path)
    perimeter = perimeter.to_crs(target_crs)
    bounds = tuple(perimeter.total_bounds)  # (xmin, ymin, xmax, ymax)

    # Clip to perimeter bounds
    clipped, clip_profile = clip_raster_to_bounds(data, profile, bounds)

    logger.info(
        "Reprojected to %s and clipped to perimeter bounds %s",
        target_crs,
        bounds,
    )
    return clipped, clip_profile
