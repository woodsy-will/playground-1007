"""Habitat change analysis between current and future suitability maps.

Classifies pixels into refugia, gains, losses, and stable unsuitable
areas, and computes area statistics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from shared.utils.logging import get_logger

logger = get_logger("p4_change")

# Change class codes
STABLE_UNSUITABLE = 0
STABLE_SUITABLE = 1  # refugia
GAIN = 2
LOSS = 3

CHANGE_LABELS = {
    STABLE_UNSUITABLE: "stable_unsuitable",
    STABLE_SUITABLE: "stable_suitable (refugia)",
    GAIN: "gain",
    LOSS: "loss",
}


def compute_change(
    current: np.ndarray,
    future: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Classify habitat change between two suitability surfaces.

    Parameters
    ----------
    current : np.ndarray
        Current suitability (continuous 0--1 or binary).
    future : np.ndarray
        Future suitability (continuous 0--1 or binary).
    threshold : float
        Cut-off for binarising continuous inputs.

    Returns
    -------
    np.ndarray
        Integer raster with values:
        - 0 = stable unsuitable
        - 1 = stable suitable (refugia)
        - 2 = gain (newly suitable)
        - 3 = loss (no longer suitable)
    """
    cur_bin = (current >= threshold).astype(np.uint8)
    fut_bin = (future >= threshold).astype(np.uint8)

    change = np.full_like(cur_bin, STABLE_UNSUITABLE, dtype=np.uint8)
    change[(cur_bin == 1) & (fut_bin == 1)] = STABLE_SUITABLE
    change[(cur_bin == 0) & (fut_bin == 1)] = GAIN
    change[(cur_bin == 1) & (fut_bin == 0)] = LOSS

    logger.info(
        "Change classes — refugia: %d, gain: %d, loss: %d, stable_unsuit: %d",
        int((change == STABLE_SUITABLE).sum()),
        int((change == GAIN).sum()),
        int((change == LOSS).sum()),
        int((change == STABLE_UNSUITABLE).sum()),
    )
    return change


def summarize_change(
    change_raster: np.ndarray,
    profile: dict,
) -> pd.DataFrame:
    """Compute area statistics for each habitat-change class.

    Parameters
    ----------
    change_raster : np.ndarray
        Integer change raster from :func:`compute_change`.
    profile : dict
        Rasterio profile with a transform for pixel size.

    Returns
    -------
    DataFrame
        Columns: ``class_code``, ``class_name``, ``pixel_count``,
        ``area_m2``, ``area_ha``.
    """
    cellsize_x = abs(profile["transform"].a)
    cellsize_y = abs(profile["transform"].e)
    cell_area_m2 = cellsize_x * cellsize_y

    rows: list[dict] = []
    for code, label in CHANGE_LABELS.items():
        count = int((change_raster == code).sum())
        area_m2 = count * cell_area_m2
        rows.append({
            "class_code": code,
            "class_name": label,
            "pixel_count": count,
            "area_m2": area_m2,
            "area_ha": area_m2 / 10_000.0,
        })

    df = pd.DataFrame(rows)
    logger.info("Change summary:\n%s", df.to_string(index=False))
    return df
