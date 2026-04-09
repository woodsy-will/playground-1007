"""Per-tree biometric extraction from crown polygons and the CHM.

Zonal statistics (max / mean height) are computed for each crown polygon,
then allometric equations convert crown geometry to DBH and stem volume.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.errors
import rasterio.mask

from shared.utils.allometry import dbh_from_crown_diameter, stem_volume_cuft
from shared.utils.logging import get_logger

logger = get_logger("p3_metrics")

# Metres-to-feet conversion factor
M_TO_FT = 3.28084


def extract_tree_metrics(
    crowns_gdf: gpd.GeoDataFrame,
    chm_path: str | Path,
    config: dict[str, Any],
) -> gpd.GeoDataFrame:
    """Enrich crown polygons with zonal height statistics and allometric estimates.

    Parameters
    ----------
    crowns_gdf : GeoDataFrame
        Crown polygons (must include ``tree_id``, ``crown_area_m2``,
        ``crown_diameter_m``, ``geometry``).
    chm_path : str or Path
        Path to the CHM GeoTIFF.
    config : dict
        Project configuration dictionary (unused beyond pass-through).

    Returns
    -------
    GeoDataFrame
        Input dataframe augmented with ``max_height_m``, ``mean_height_m``,
        ``quality_flag`` (0 = OK, 1 = extraction failed),
        ``dbh_inches``, and ``stem_volume_cuft`` columns.
    """
    chm_path = Path(chm_path)

    max_heights: list[float] = []
    mean_heights: list[float] = []
    quality_flags: list[int] = []
    error_count = 0

    with rasterio.open(chm_path) as src:
        for _, row in crowns_gdf.iterrows():
            geom = [row.geometry.__geo_interface__]
            try:
                out_image, _ = rasterio.mask.mask(src, geom, crop=True, filled=True)
                arr = out_image[0]
                valid = arr[arr > 0]
                if valid.size > 0:
                    max_heights.append(float(np.max(valid)))
                    mean_heights.append(float(np.mean(valid)))
                else:
                    max_heights.append(0.0)
                    mean_heights.append(0.0)
                quality_flags.append(0)
            except (ValueError, IndexError, rasterio.errors.RasterioError) as exc:
                logger.debug("Could not extract metrics for crown: %s", exc)
                max_heights.append(0.0)
                mean_heights.append(0.0)
                quality_flags.append(1)
                error_count += 1

    if error_count > 0:
        logger.warning(
            "Metric extraction failed for %d of %d crowns", error_count, len(crowns_gdf)
        )

    result = crowns_gdf.copy()
    result["max_height_m"] = max_heights
    result["mean_height_m"] = mean_heights
    result["quality_flag"] = quality_flags

    # Allometric estimates
    cd_m = result["crown_diameter_m"].values.astype(np.float64)
    dbh = dbh_from_crown_diameter(cd_m)
    result["dbh_inches"] = np.round(dbh, 2)

    height_ft = np.array(max_heights) * M_TO_FT
    result["stem_volume_cuft"] = np.round(stem_volume_cuft(dbh, height_ft), 2)

    logger.info("Extracted metrics for %d crowns", len(result))
    return result
