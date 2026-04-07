"""Canopy Height Model (CHM) generation.

Computes CHM = DSM - DTM, applies Gaussian smoothing, and clamps negative
values to zero.  If the input is a LAZ file a DSM is first derived from
first-return points via PDAL.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from shared.utils.io import read_raster, write_raster
from shared.utils.logging import get_logger

logger = get_logger("p3_chm")


def _generate_dsm_from_laz(laz_path: Path, config: dict[str, Any]) -> Path:
    """Create a DSM raster from first-return points in a LAZ file."""
    import pdal  # noqa: PLC0415

    proc = config.get("processing", {})
    output_dir = Path(config.get("data", {}).get("output_dir", "data/processed"))
    output_dir.mkdir(parents=True, exist_ok=True)

    resolution = proc.get("dtm_resolution", 1.0)
    crs = proc.get("crs", "EPSG:3310")
    dsm_path = output_dir / "dsm.tif"

    pipeline_json = json.dumps(
        {
            "pipeline": [
                {"type": "readers.las", "filename": str(laz_path)},
                {
                    "type": "filters.range",
                    "limits": "ReturnNumber[1:1]",
                },
                {
                    "type": "writers.gdal",
                    "filename": str(dsm_path),
                    "resolution": resolution,
                    "output_type": "max",
                    "gdaldriver": "GTiff",
                    "override_srs": crs,
                },
            ]
        }
    )

    logger.info("Generating DSM from first returns: %s", laz_path.name)
    pipeline = pdal.Pipeline(pipeline_json)
    pipeline.execute()
    return dsm_path


def generate_chm(
    dsm_path_or_laz: str | Path,
    dtm_path: str | Path,
    config: dict[str, Any],
) -> Path:
    """Generate a smoothed Canopy Height Model.

    Parameters
    ----------
    dsm_path_or_laz : str or Path
        Path to a DSM GeoTIFF **or** a LAZ file (in which case a DSM is
        generated on-the-fly from first returns).
    dtm_path : str or Path
        Path to the DTM GeoTIFF.
    config : dict
        Project configuration dictionary.

    Returns
    -------
    Path
        Path to the output CHM GeoTIFF.
    """
    dsm_path_or_laz = Path(dsm_path_or_laz)
    dtm_path = Path(dtm_path)
    proc = config.get("processing", {})
    output_dir = Path(config.get("data", {}).get("output_dir", "data/processed"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # If input is a point cloud, derive DSM first
    if dsm_path_or_laz.suffix.lower() in {".laz", ".las"}:
        dsm_path = _generate_dsm_from_laz(dsm_path_or_laz, config)
    else:
        dsm_path = dsm_path_or_laz

    dsm_data, dsm_profile = read_raster(dsm_path)
    dtm_data, _ = read_raster(dtm_path)

    # Both rasters are (1, rows, cols); squeeze to 2-D for arithmetic
    dsm_2d = dsm_data[0].astype(np.float32)
    dtm_2d = dtm_data[0].astype(np.float32)

    chm = dsm_2d - dtm_2d

    # Clamp negatives
    chm[chm < 0] = 0.0

    # Gaussian smoothing
    sigma = proc.get("chm_smoothing_sigma", 0.67)
    resolution = proc.get("dtm_resolution", 1.0)
    sigma_pixels = sigma / resolution
    logger.info("Smoothing CHM with sigma=%.2f m (%.2f px)", sigma, sigma_pixels)
    chm = gaussian_filter(chm, sigma=sigma_pixels).astype(np.float32)

    out_path = output_dir / "chm.tif"
    write_raster(out_path, chm, dsm_profile)
    logger.info("CHM written to %s", out_path)
    return out_path
