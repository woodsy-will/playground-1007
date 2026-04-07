"""Background point generation and presence/absence matrix construction.

Implements target-group background sampling for species distribution
modelling.
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from shared.utils.logging import get_logger

logger = get_logger("p4_background")


def generate_background_points(
    occurrences_gdf: gpd.GeoDataFrame,
    predictor_stack: np.ndarray,
    profile: dict,
    config: dict[str, Any],
    n_points: int = 10_000,
) -> gpd.GeoDataFrame:
    """Generate background (pseudo-absence) points via target-group sampling.

    Random points are placed within the raster extent.  Points are weighted
    by proximity to existing occurrences as a proxy for sampling effort,
    following the target-group background approach (Phillips et al. 2009).

    Parameters
    ----------
    occurrences_gdf : GeoDataFrame
        Presence records used to derive a sampling-effort surface.
    predictor_stack : np.ndarray
        Predictor raster stack ``(bands, rows, cols)``.
    profile : dict
        Rasterio profile (transform, CRS, width, height).
    config : dict
        Project configuration.
    n_points : int
        Number of background points to generate.

    Returns
    -------
    GeoDataFrame
        Background points with ``presence=0`` column.
    """
    rng = np.random.default_rng(42)
    transform = profile["transform"]
    height = profile["height"]
    width = profile["width"]
    crs = profile["crs"]

    # Build a sampling-effort weight surface from occurrence density
    effort = np.ones((height, width), dtype=np.float64)
    for geom in occurrences_gdf.geometry:
        col = int((geom.x - transform.c) / transform.a)
        row = int((geom.y - transform.f) / transform.e)
        if 0 <= row < height and 0 <= col < width:
            r_lo = max(0, row - 2)
            r_hi = min(height, row + 3)
            c_lo = max(0, col - 2)
            c_hi = min(width, col + 3)
            effort[r_lo:r_hi, c_lo:c_hi] += 1.0

    # Mask out pixels that are nodata in any band
    valid = np.all(np.isfinite(predictor_stack), axis=0)
    effort[~valid] = 0.0

    flat_weights = effort.ravel()
    total = flat_weights.sum()
    if total == 0:
        raise ValueError("No valid pixels for background sampling")
    probs = flat_weights / total

    chosen = rng.choice(height * width, size=n_points, replace=True, p=probs)
    rows, cols = np.unravel_index(chosen, (height, width))

    xs = transform.c + (cols + 0.5) * transform.a
    ys = transform.f + (rows + 0.5) * transform.e

    bg_gdf = gpd.GeoDataFrame(
        {"presence": np.zeros(n_points, dtype=int)},
        geometry=[Point(x, y) for x, y in zip(xs, ys)],
        crs=crs,
    )
    logger.info("Generated %d background points", len(bg_gdf))
    return bg_gdf


def create_pa_matrix(
    presence_gdf: gpd.GeoDataFrame,
    background_gdf: gpd.GeoDataFrame,
    stack: np.ndarray,
    profile: dict,
    band_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Build predictor matrix (X) and response vector (y).

    Extracts predictor values for both presence and background points,
    combines them, and returns clean numpy arrays suitable for model
    training.

    Parameters
    ----------
    presence_gdf : GeoDataFrame
        Presence records.
    background_gdf : GeoDataFrame
        Background points.
    stack : np.ndarray
        Predictor stack ``(bands, rows, cols)``.
    profile : dict
        Rasterio profile.
    band_names : list[str]
        Band names for the predictor stack.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(X, y)`` where X has shape ``(n_samples, n_predictors)`` and y
        is a 1-D binary array (1 = presence, 0 = background).
    """
    from projects.p4_habitat_suitability.src.predictors import extract_values_at_points

    pres_df = extract_values_at_points(stack, profile, presence_gdf, band_names)
    pres_df["presence"] = 1

    bg_df = extract_values_at_points(stack, profile, background_gdf, band_names)
    bg_df["presence"] = 0

    combined = pd.concat([pres_df, bg_df], ignore_index=True).dropna()

    X = combined[band_names].values.astype(np.float32)  # noqa: N806
    y = combined["presence"].values.astype(np.int32)

    logger.info(
        "PA matrix: %d samples (%d presence, %d background), %d predictors",
        len(y), int(y.sum()), int((y == 0).sum()), X.shape[1],
    )
    return X, y
