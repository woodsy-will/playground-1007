"""Ground point classification using PDAL SMRF filter.

Classifies ground returns in a LAZ point cloud using the Simple
Morphological Filter (SMRF) algorithm, writing the classified result
to a new LAZ file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from shared.utils.logging import get_logger

logger = get_logger("p3_ground")


def classify_ground(laz_path: str | Path, config: dict[str, Any]) -> Path:
    """Classify ground points in a LAZ file using SMRF.

    Parameters
    ----------
    laz_path : str or Path
        Path to input LAZ/LAS point cloud.
    config : dict
        Project configuration dictionary.  SMRF parameters are read from
        ``config["processing"]`` with keys ``smrf_cell``, ``smrf_slope``,
        and ``smrf_window``.

    Returns
    -------
    Path
        Path to the classified output LAZ file.
    """
    import pdal  # noqa: PLC0415 – optional heavy dependency

    laz_path = Path(laz_path)
    proc = config.get("processing", {})
    output_dir = Path(config.get("data", {}).get("output_dir", "data/processed"))
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"{laz_path.stem}_classified.laz"

    cell = proc.get("smrf_cell", 1.0)
    slope = proc.get("smrf_slope", 0.15)
    window = proc.get("smrf_window", 18.0)

    pipeline_json = json.dumps(
        {
            "pipeline": [
                {"type": "readers.las", "filename": str(laz_path)},
                {
                    "type": "filters.smrf",
                    "cell": cell,
                    "slope": slope,
                    "window": window,
                },
                {"type": "writers.las", "filename": str(out_path)},
            ]
        }
    )

    logger.info("Classifying ground points: %s", laz_path.name)
    logger.info("SMRF params — cell=%.2f, slope=%.3f, window=%.1f", cell, slope, window)

    pipeline = pdal.Pipeline(pipeline_json)
    pipeline.execute()

    logger.info("Classified LAZ written to %s", out_path)
    return out_path
