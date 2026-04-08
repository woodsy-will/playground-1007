"""Digital Terrain Model generation from classified LiDAR point clouds.

Extracts ground-classified points (class 2) and interpolates them onto a
regular grid using the PDAL ``writers.gdal`` stage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from shared.utils.logging import get_logger

logger = get_logger("p3_dtm")


def generate_dtm(classified_laz: str | Path, config: dict[str, Any]) -> Path:
    """Generate a DTM raster from a ground-classified LAZ file.

    Parameters
    ----------
    classified_laz : str or Path
        Path to classified LAZ file (must contain class-2 ground points).
    config : dict
        Project configuration dictionary.

    Returns
    -------
    Path
        Path to the output DTM GeoTIFF.
    """
    import pdal  # noqa: PLC0415

    classified_laz = Path(classified_laz)
    proc = config.get("processing", {})
    output_dir = Path(config.get("data", {}).get("output_dir", "data/processed"))
    output_dir.mkdir(parents=True, exist_ok=True)

    resolution = proc.get("dtm_resolution", 1.0)
    crs = proc.get("crs", "EPSG:3310")
    out_path = output_dir / "dtm.tif"

    pipeline_json = json.dumps(
        {
            "pipeline": [
                {"type": "readers.las", "filename": str(classified_laz)},
                {
                    "type": "filters.range",
                    "limits": "Classification[2:2]",
                },
                {
                    "type": "writers.gdal",
                    "filename": str(out_path),
                    "resolution": resolution,
                    "output_type": "idw",
                    "gdaldriver": "GTiff",
                    "override_srs": crs,
                },
            ]
        }
    )

    logger.info("Generating DTM at %.1f m resolution from %s", resolution, classified_laz.name)

    pipeline = pdal.Pipeline(pipeline_json)
    pipeline.execute()

    logger.info("DTM written to %s", out_path)
    return out_path
