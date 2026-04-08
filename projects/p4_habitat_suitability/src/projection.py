"""Project trained SDMs onto current and future climate scenarios.

Generates continuous suitability surfaces (0--1 probability) and binary
habitat maps from a fitted model and a predictor raster stack.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from shared.utils.logging import get_logger

logger = get_logger("p4_projection")


def project_suitability(
    model: Any,
    future_stack: np.ndarray,
    profile: dict,
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict]:
    """Apply a fitted model to a predictor stack and produce a suitability map.

    Parameters
    ----------
    model : fitted model
        Must expose ``predict_proba(X) -> array``.  If the model has an
        attached ``scaler_`` attribute (as with the MaxEnt approximation),
        features are scaled before prediction.
    future_stack : np.ndarray
        Predictor stack of shape ``(bands, rows, cols)``.
    profile : dict
        Rasterio profile for the predictor grid.
    config : dict, optional
        Project configuration (reserved for threshold overrides).

    Returns
    -------
    tuple[np.ndarray, dict]
        - suitability: 2-D array ``(rows, cols)`` with values in [0, 1].
        - profile: updated rasterio profile (single band, float32).
    """
    n_bands, height, width = future_stack.shape
    flat = future_stack.reshape(n_bands, -1).T  # (pixels, bands)

    # Identify valid (finite) pixels
    valid_mask = np.all(np.isfinite(flat), axis=1)
    probs = np.full(flat.shape[0], np.nan, dtype=np.float32)

    if valid_mask.any():
        X_valid = flat[valid_mask].astype(np.float32)  # noqa: N806
        if hasattr(model, "scaler_"):
            X_valid = model.scaler_.transform(X_valid)  # noqa: N806
        probs[valid_mask] = model.predict_proba(X_valid)[:, 1].astype(np.float32)

    suitability = probs.reshape(height, width)

    out_profile = profile.copy()
    out_profile.update(count=1, dtype="float32")

    logger.info(
        "Projected suitability: min=%.3f, max=%.3f, valid_pix=%d",
        float(np.nanmin(suitability)),
        float(np.nanmax(suitability)),
        int(valid_mask.sum()),
    )
    return suitability, out_profile


def threshold_suitability(
    suitability: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Convert continuous suitability to a binary habitat map.

    Parameters
    ----------
    suitability : np.ndarray
        2-D suitability surface with values in [0, 1].
    threshold : float
        Probability cut-off.  Pixels >= threshold are classified as
        suitable (1), others as unsuitable (0).

    Returns
    -------
    np.ndarray
        Binary array (dtype uint8): 1 = suitable, 0 = unsuitable.
        NaN pixels in the input are mapped to 0.
    """
    binary = np.where(np.isnan(suitability), 0, (suitability >= threshold).astype(np.uint8))
    logger.info(
        "Thresholded at %.2f: %d suitable, %d unsuitable pixels",
        threshold,
        int((binary == 1).sum()),
        int((binary == 0).sum()),
    )
    return binary.astype(np.uint8)
