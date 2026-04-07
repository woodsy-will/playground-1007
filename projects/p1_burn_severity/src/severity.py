"""Core burn severity analysis — NBR, dNBR, classification, and RBR.

Implements the standard Normalized Burn Ratio approach following
Key & Benson (2006) and severity thresholds from Miller & Thode (2007).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from shared.utils.logging import get_logger

logger = get_logger("p1.severity")


def compute_nbr(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """Compute Normalized Burn Ratio.

    NBR = (NIR - SWIR) / (NIR + SWIR)

    Parameters
    ----------
    nir : np.ndarray
        Near-infrared band (e.g. Sentinel-2 B8A).
    swir : np.ndarray
        Short-wave infrared band (e.g. Sentinel-2 B12).

    Returns
    -------
    np.ndarray
        NBR values in [-1, 1].  Pixels where NIR + SWIR == 0 are set
        to ``np.nan``.
    """
    nir = nir.astype(np.float64)
    swir = swir.astype(np.float64)
    denom = nir + swir
    nbr = np.where(denom != 0, (nir - swir) / denom, np.nan)
    return nbr


def compute_dnbr(pre_nbr: np.ndarray, post_nbr: np.ndarray) -> np.ndarray:
    """Compute differenced Normalized Burn Ratio.

    dNBR = pre_NBR - post_NBR

    Higher values indicate greater burn severity.

    Parameters
    ----------
    pre_nbr : np.ndarray
        Pre-fire NBR raster.
    post_nbr : np.ndarray
        Post-fire NBR raster.

    Returns
    -------
    np.ndarray
        dNBR raster.
    """
    return pre_nbr.astype(np.float64) - post_nbr.astype(np.float64)


def classify_severity(
    dnbr: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    """Classify dNBR into burn severity categories.

    Classes (from config thresholds):
        0 — Unburned / Very Low
        1 — Low Severity
        2 — Moderate-Low Severity
        3 — Moderate-High Severity
        4 — High Severity

    Parameters
    ----------
    dnbr : np.ndarray
        Differenced NBR raster.
    config : dict
        Project configuration with ``processing.severity_thresholds``.

    Returns
    -------
    np.ndarray
        Integer classification raster (dtype ``uint8``).  NaN pixels in
        the input are mapped to 255 (nodata).
    """
    thresholds = config["processing"]["severity_thresholds"]

    classes = np.full(dnbr.shape, 255, dtype=np.uint8)

    # Map class names to integer codes
    class_map: dict[str, int] = {
        "unburned": 0,
        "low": 1,
        "moderate_low": 2,
        "moderate_high": 3,
        "high": 4,
    }

    for name, code in class_map.items():
        low, high = thresholds[name]
        mask = (dnbr >= low) & (dnbr < high)
        classes[mask] = code

    # Pixels below unburned range
    ub_low = thresholds["unburned"][0]
    classes[dnbr < ub_low] = 0

    # Pixels above high range
    hi_high = thresholds["high"][1]
    classes[dnbr >= hi_high] = 4

    # NaN pixels stay as 255 (nodata)
    classes[np.isnan(dnbr)] = 255

    # Log summary
    for name, code in class_map.items():
        count = int(np.count_nonzero(classes == code))
        logger.info("  %s (class %d): %d pixels", name, code, count)

    return classes


def compute_rbr(dnbr: np.ndarray, pre_nbr: np.ndarray) -> np.ndarray:
    """Compute Relativized Burn Ratio.

    RBR = dNBR / (pre_NBR + 1.001)

    The small offset (1.001) prevents division by zero and follows the
    convention of Parks et al. (2014).

    Parameters
    ----------
    dnbr : np.ndarray
        Differenced NBR raster.
    pre_nbr : np.ndarray
        Pre-fire NBR raster.

    Returns
    -------
    np.ndarray
        RBR raster.
    """
    return dnbr / (pre_nbr.astype(np.float64) + 1.001)
