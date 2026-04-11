"""Validation of predicted tree inventory against field cruise data.

Performs nearest-neighbour matching between remotely-sensed tree locations
and field-measured stems, then computes detection / error metrics overall
and by configurable strata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from shared.utils.logging import get_logger

logger = get_logger("p3_validation")


def validate_against_cruise(
    predicted_gdf: Any,
    cruise_csv: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Match predicted trees to cruise stems and compute accuracy metrics.

    Parameters
    ----------
    predicted_gdf : GeoDataFrame
        Predicted tree inventory with ``geometry`` (Point centroids) and at
        minimum ``dbh_inches`` and ``max_height_m`` columns.
    cruise_csv : str or Path
        Path to cruise plot CSV with columns ``stem_x``, ``stem_y``,
        ``dbh_inches``, ``height_ft``, ``species``, ``diameter_class``.
    config : dict
        Project configuration dictionary.

    Returns
    -------
    dict
        Dictionary containing:
        - ``n_predicted``: int
        - ``n_reference``: int
        - ``n_matched``: int
        - ``detection_rate``: float
        - ``omission_rate``: float
        - ``commission_rate``: float
        - ``rmse_height_m``: float (or ``None``)
        - ``rmse_dbh_inches``: float (or ``None``)
        - ``by_stratum``: dict mapping stratum values to sub-dicts
    """
    cruise_csv = Path(cruise_csv)
    val_cfg = config.get("validation", {})
    match_dist = val_cfg.get("match_distance", 3.0)
    stratify_by = val_cfg.get("stratify_by", [])

    cruise = pd.read_csv(cruise_csv)

    # Reference coordinates
    ref_xy = cruise[["stem_x", "stem_y"]].values

    # Predicted coordinates (centroids)
    pred_xy = np.array(
        [[g.x, g.y] for g in predicted_gdf.geometry]
    )

    n_ref = len(ref_xy)
    n_pred = len(pred_xy)

    if n_pred == 0 or n_ref == 0:
        return {
            "n_predicted": n_pred,
            "n_reference": n_ref,
            "n_matched": 0,
            "detection_rate": 0.0,
            "omission_rate": 1.0,
            "commission_rate": 1.0 if n_pred > 0 else 0.0,
            "rmse_height_m": None,
            "rmse_dbh_inches": None,
            "by_stratum": {},
        }

    # Build KD-tree on reference stems
    tree = cKDTree(ref_xy)
    dists, idxs = tree.query(pred_xy, k=1)

    matched_mask = dists <= match_dist
    n_matched = int(matched_mask.sum())

    # Unique reference matches (one-to-one)
    matched_ref_ids = set(idxs[matched_mask])

    detection_rate = len(matched_ref_ids) / n_ref if n_ref > 0 else 0.0
    omission_rate = 1.0 - detection_rate
    commission_rate = 1.0 - (n_matched / n_pred) if n_pred > 0 else 0.0

    # Height RMSE (metres)
    rmse_height: float | None = None
    rmse_dbh: float | None = None

    if n_matched > 0:
        matched_pred = predicted_gdf.iloc[matched_mask].copy()
        matched_cruise = cruise.iloc[idxs[matched_mask]].copy()

        if "max_height_m" in matched_pred.columns and "height_ft" in matched_cruise.columns:
            pred_h = matched_pred["max_height_m"].values
            ref_h = matched_cruise["height_ft"].values / 3.28084  # ft -> m
            rmse_height = float(np.sqrt(np.mean((pred_h - ref_h) ** 2)))

        if "dbh_inches" in matched_pred.columns and "dbh_inches" in matched_cruise.columns:
            pred_d = matched_pred["dbh_inches"].values
            ref_d = matched_cruise["dbh_inches"].values
            rmse_dbh = float(np.sqrt(np.mean((pred_d - ref_d) ** 2)))

    # Stratified metrics
    by_stratum: dict[str, Any] = {}
    for col in stratify_by:
        if col not in cruise.columns:
            continue
        strata_metrics: dict[str, Any] = {}
        for val in cruise[col].dropna().unique():
            sub_cruise = cruise[cruise[col] == val]
            sub_ref_xy = sub_cruise[["stem_x", "stem_y"]].values
            if len(sub_ref_xy) == 0:  # pragma: no cover
                continue
            sub_tree = cKDTree(sub_ref_xy)
            sub_dists, _ = sub_tree.query(pred_xy, k=1)
            sub_matched = int((sub_dists <= match_dist).sum())
            sub_det = min(sub_matched, len(sub_ref_xy)) / len(sub_ref_xy)
            strata_metrics[str(val)] = {
                "n_reference": len(sub_ref_xy),
                "detection_rate": round(sub_det, 3),
            }
        by_stratum[col] = strata_metrics

    metrics = {
        "n_predicted": n_pred,
        "n_reference": n_ref,
        "n_matched": n_matched,
        "detection_rate": round(detection_rate, 3),
        "omission_rate": round(omission_rate, 3),
        "commission_rate": round(commission_rate, 3),
        "rmse_height_m": round(rmse_height, 3) if rmse_height is not None else None,
        "rmse_dbh_inches": round(rmse_dbh, 3) if rmse_dbh is not None else None,
        "by_stratum": by_stratum,
    }

    logger.info(
        "Validation: %d predicted, %d reference, %d matched (det=%.1f%%)",
        n_pred,
        n_ref,
        n_matched,
        detection_rate * 100,
    )
    return metrics
