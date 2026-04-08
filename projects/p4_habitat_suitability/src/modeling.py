"""Species distribution model training, cross-validation, and diagnostics.

Supports MaxEnt (via L1-penalised logistic regression) and Random Forest.
Spatial block cross-validation prevents spatial autocorrelation leakage.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from shared.utils.logging import get_logger

logger = get_logger("p4_modeling")


def train_maxent(
    X_train: np.ndarray,  # noqa: N803
    y_train: np.ndarray,
    config: dict[str, Any] | None = None,
) -> LogisticRegression:
    """Train a MaxEnt-approximation model using L1-penalised logistic regression.

    Logistic regression with an L1 (lasso) penalty on feature-transformed
    inputs approximates maximum entropy modelling (Phillips et al. 2006;
    Renner & Warton 2013).

    Parameters
    ----------
    X_train : np.ndarray
        Predictor matrix ``(n_samples, n_features)``.
    y_train : np.ndarray
        Binary response vector (1 = presence, 0 = background).
    config : dict, optional
        Project configuration for hyper-parameter overrides.

    Returns
    -------
    LogisticRegression
        Fitted model.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)  # noqa: N806

    model = LogisticRegression(
        solver="saga",
        C=1.0,
        l1_ratio=1.0,
        max_iter=2000,
        random_state=42,
    )
    model.fit(X_scaled, y_train)

    # Attach scaler for use at prediction time
    model.scaler_ = scaler  # type: ignore[attr-defined]
    logger.info("MaxEnt (L1 LR) trained on %d samples", len(y_train))
    return model


def train_random_forest(
    X_train: np.ndarray,  # noqa: N803
    y_train: np.ndarray,
    config: dict[str, Any] | None = None,
) -> RandomForestClassifier:
    """Train a Random Forest classifier with default hyper-parameters.

    Parameters
    ----------
    X_train : np.ndarray
        Predictor matrix ``(n_samples, n_features)``.
    y_train : np.ndarray
        Binary response vector.
    config : dict, optional
        Project configuration (reserved for hyper-parameter tuning).

    Returns
    -------
    RandomForestClassifier
        Fitted model.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    logger.info("Random Forest trained on %d samples", len(y_train))
    return model


def _compute_tss(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    """Compute True Skill Statistic (sensitivity + specificity - 1)."""
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return sensitivity + specificity - 1.0


def spatial_block_cv(
    X: np.ndarray,  # noqa: N803
    y: np.ndarray,
    coords: np.ndarray,
    model_fn: Callable[[np.ndarray, np.ndarray], Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Spatial block cross-validation with AUC and TSS metrics.

    Divides the study area into spatial blocks along the x-axis, assigns
    blocks to folds, and evaluates model performance on held-out blocks
    to account for spatial autocorrelation (Valavi et al. 2019).

    Parameters
    ----------
    X : np.ndarray
        Predictor matrix ``(n_samples, n_features)``.
    y : np.ndarray
        Binary response vector.
    coords : np.ndarray
        Spatial coordinates ``(n_samples, 2)`` — columns ``[x, y]``.
    model_fn : callable
        Function accepting ``(X_train, y_train)`` and returning a fitted
        model with a ``predict_proba`` method.
    config : dict, optional
        Project configuration.  Uses ``modeling.cv_folds`` (default 5).

    Returns
    -------
    dict
        Keys: ``auc_mean``, ``auc_std``, ``auc_ci95``, ``tss_mean``,
        ``tss_std``, ``tss_ci95``, ``fold_aucs``, ``fold_tss``.
    """
    n_folds = 5
    if config:
        n_folds = config.get("modeling", {}).get("cv_folds", 5)

    x_coords = coords[:, 0]
    # Create spatial blocks by quantile along x
    block_edges = np.quantile(x_coords, np.linspace(0, 1, n_folds + 1))
    fold_ids = np.digitize(x_coords, block_edges[1:-1])

    aucs: list[float] = []
    tss_scores: list[float] = []

    for fold in range(n_folds):
        test_mask = fold_ids == fold
        train_mask = ~test_mask

        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        if len(np.unique(y[test_mask])) < 2:
            continue

        model = model_fn(X[train_mask], y[train_mask])

        if hasattr(model, "scaler_"):
            X_test_scaled = model.scaler_.transform(X[test_mask])  # noqa: N806
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_prob = model.predict_proba(X[test_mask])[:, 1]

        auc = roc_auc_score(y[test_mask], y_prob)
        tss = _compute_tss(y[test_mask], y_prob)
        aucs.append(auc)
        tss_scores.append(tss)

    if not aucs:
        return {
            "auc_mean": np.nan, "auc_std": np.nan, "auc_ci95": (np.nan, np.nan),
            "tss_mean": np.nan, "tss_std": np.nan, "tss_ci95": (np.nan, np.nan),
            "fold_aucs": [], "fold_tss": [],
        }

    auc_arr = np.array(aucs)
    tss_arr = np.array(tss_scores)

    def _ci95(arr: np.ndarray) -> tuple[float, float]:
        m = float(arr.mean())
        se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
        return (m - 1.96 * se, m + 1.96 * se)

    result = {
        "auc_mean": float(auc_arr.mean()),
        "auc_std": float(auc_arr.std()),
        "auc_ci95": _ci95(auc_arr),
        "tss_mean": float(tss_arr.mean()),
        "tss_std": float(tss_arr.std()),
        "tss_ci95": _ci95(tss_arr),
        "fold_aucs": aucs,
        "fold_tss": tss_scores,
    }
    logger.info(
        "Spatial block CV (%d folds): AUC=%.3f +/- %.3f, TSS=%.3f +/- %.3f",
        len(aucs), result["auc_mean"], result["auc_std"],
        result["tss_mean"], result["tss_std"],
    )
    return result


def compute_variable_importance(
    model: Any,
    X: np.ndarray,  # noqa: N803
    y: np.ndarray,
    band_names: list[str],
) -> pd.DataFrame:
    """Compute permutation-based variable importance.

    Each predictor column is randomly shuffled and the drop in AUC is
    recorded as the importance score.

    Parameters
    ----------
    model : fitted model
        Must have a ``predict_proba`` method.
    X : np.ndarray
        Predictor matrix.
    y : np.ndarray
        Binary response vector.
    band_names : list[str]
        Names for each predictor column.

    Returns
    -------
    DataFrame
        Columns: ``variable``, ``importance`` (AUC drop), sorted
        descending.
    """
    rng = np.random.default_rng(42)

    if hasattr(model, "scaler_"):
        X_eval = model.scaler_.transform(X)  # noqa: N806
    else:
        X_eval = X  # noqa: N806

    base_prob = model.predict_proba(X_eval)[:, 1]
    if len(np.unique(y)) < 2:
        return pd.DataFrame({"variable": band_names, "importance": 0.0})
    base_auc = roc_auc_score(y, base_prob)

    importances: list[float] = []
    for i in range(X.shape[1]):
        X_perm = X.copy()  # noqa: N806
        rng.shuffle(X_perm[:, i])
        if hasattr(model, "scaler_"):
            X_perm_eval = model.scaler_.transform(X_perm)  # noqa: N806
        else:
            X_perm_eval = X_perm  # noqa: N806
        perm_prob = model.predict_proba(X_perm_eval)[:, 1]
        perm_auc = roc_auc_score(y, perm_prob)
        importances.append(base_auc - perm_auc)

    df = pd.DataFrame({"variable": band_names, "importance": importances})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)
