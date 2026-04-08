"""Vegetation recovery tracking using spectral indices and curve fitting.

Builds per-severity-class time series of NDVI / EVI and fits an
exponential recovery model to estimate time-to-recovery.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from shared.utils.logging import get_logger

logger = get_logger("p1.recovery")


def compute_vegetation_index(
    nir: np.ndarray,
    red_or_swir: np.ndarray,
    index_type: str = "NDVI",
) -> np.ndarray:
    """Compute a vegetation spectral index.

    Supported indices:
        - **NDVI** = (NIR - Red) / (NIR + Red)
        - **EVI**  = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
          Simplified two-band EVI (no blue band) uses
          EVI2 = 2.5 * (NIR - Red) / (NIR + 2.4*Red + 1).

    Parameters
    ----------
    nir : np.ndarray
        Near-infrared band values.
    red_or_swir : np.ndarray
        Red band (for NDVI/EVI) or SWIR band values.
    index_type : str
        ``"NDVI"`` or ``"EVI"``.

    Returns
    -------
    np.ndarray
        Vegetation index values.  Division-by-zero pixels are NaN.
    """
    nir = nir.astype(np.float64)
    red = red_or_swir.astype(np.float64)

    if index_type.upper() == "NDVI":
        denom = nir + red
        vi = np.where(denom != 0, (nir - red) / denom, np.nan)
    elif index_type.upper() == "EVI":
        # Two-band Enhanced Vegetation Index (EVI2)
        denom = nir + 2.4 * red + 1.0
        vi = np.where(denom != 0, 2.5 * (nir - red) / denom, np.nan)
    else:
        raise ValueError(f"Unsupported index type: {index_type!r}. Use 'NDVI' or 'EVI'.")

    return vi


def build_recovery_timeseries(
    annual_index_rasters: list[np.ndarray],
    severity_raster: np.ndarray,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Compute mean vegetation index per severity class for each year.

    Parameters
    ----------
    annual_index_rasters : list[np.ndarray]
        One vegetation-index raster per post-fire year (ordered
        chronologically).
    severity_raster : np.ndarray
        Integer severity classification raster (0-4; 255 = nodata).
    config : dict
        Project configuration (reads ``processing.years_post_fire``).

    Returns
    -------
    pd.DataFrame
        Columns: ``year``, ``severity_class``, ``mean_index``,
        ``std_index``, ``pixel_count``.
    """
    years_post = config.get("processing", {}).get("years_post_fire", 5)
    n_years = min(len(annual_index_rasters), years_post)

    records: list[dict[str, Any]] = []
    for yr_idx in range(n_years):
        raster = annual_index_rasters[yr_idx]
        for sev_class in range(5):
            mask = severity_raster == sev_class
            if not np.any(mask):
                continue
            vals = raster[mask]
            valid = vals[~np.isnan(vals)]
            if len(valid) == 0:
                continue
            records.append(
                {
                    "year": yr_idx + 1,
                    "severity_class": sev_class,
                    "mean_index": float(np.mean(valid)),
                    "std_index": float(np.std(valid)),
                    "pixel_count": int(len(valid)),
                }
            )

    df = pd.DataFrame(records)
    logger.info("Built recovery time series with %d rows", len(df))
    return df


def _exponential_model(t: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Exponential recovery: y = a * (1 - exp(-b * t)) + c."""
    return a * (1.0 - np.exp(-b * t)) + c


def fit_recovery_model(
    ts_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Fit exponential recovery curves per severity class.

    Model: ``y = a * (1 - exp(-b * t)) + c``

    Parameters
    ----------
    ts_df : pd.DataFrame
        Time series dataframe from :func:`build_recovery_timeseries`.
    config : dict
        Project configuration.

    Returns
    -------
    pd.DataFrame
        One row per severity class with fitted parameters ``a``, ``b``,
        ``c``, estimated ``years_to_90pct_recovery``, and ``r_squared``.
    """
    results: list[dict[str, Any]] = []

    for sev_class, grp in ts_df.groupby("severity_class"):
        t = grp["year"].values.astype(np.float64)
        y = grp["mean_index"].values.astype(np.float64)

        if len(t) < 3:
            logger.warning(
                "Severity class %d: only %d points, skipping fit", sev_class, len(t)
            )
            continue

        try:
            popt, _ = curve_fit(
                _exponential_model,
                t,
                y,
                p0=[np.max(y) - np.min(y), 0.5, np.min(y)],
                maxfev=5000,
            )
            a, b, c = popt

            # Predicted values for R-squared
            y_pred = _exponential_model(t, *popt)
            ss_res = float(np.sum((y - y_pred) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

            # Estimate time to 90% of asymptotic recovery (a + c)
            if b > 0 and a != 0:
                t_90 = -np.log(1.0 - 0.9) / b  # solve 0.9 = 1 - exp(-b*t)
            else:
                t_90 = np.nan

            results.append(
                {
                    "severity_class": int(sev_class),
                    "a": float(a),
                    "b": float(b),
                    "c": float(c),
                    "years_to_90pct_recovery": float(t_90),
                    "r_squared": float(r_sq),
                }
            )
            logger.info(
                "Class %d: a=%.4f, b=%.4f, c=%.4f, t90=%.1f yr, R2=%.3f",
                sev_class,
                a,
                b,
                c,
                t_90,
                r_sq,
            )
        except RuntimeError:
            logger.warning(
                "Curve fit did not converge for severity class %d", sev_class
            )

    return pd.DataFrame(results)
