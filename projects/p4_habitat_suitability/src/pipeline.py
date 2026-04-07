"""End-to-end habitat suitability modelling pipeline.

Orchestrates occurrence loading, predictor stack construction, model
training with spatial block CV, projection under climate scenarios, and
habitat change analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from shared.utils.config import load_config
from shared.utils.io import write_raster
from shared.utils.logging import get_logger

logger = get_logger("p4_pipeline")


def run_pipeline(config_path: str | Path) -> dict[str, Any]:
    """Execute the full habitat suitability workflow.

    Parameters
    ----------
    config_path : str or Path
        Path to the project YAML configuration file.

    Returns
    -------
    dict
        Dictionary with keys:
        - ``occurrences_raw``: path to raw occurrence file
        - ``occurrences_thinned``: count of thinned records
        - ``predictor_stack_shape``: tuple of stack dimensions
        - ``band_names``: list of predictor names
        - ``cv_metrics``: dict of cross-validation results
        - ``suitability_current``: path to current suitability raster
        - ``change_summary``: DataFrame of change statistics
        - ``models``: dict of fitted model objects
    """
    from projects.p4_habitat_suitability.src.background import (
        create_pa_matrix,
        generate_background_points,
    )
    from projects.p4_habitat_suitability.src.change_analysis import (
        compute_change,
        summarize_change,
    )
    from projects.p4_habitat_suitability.src.modeling import (
        compute_variable_importance,
        spatial_block_cv,
        train_maxent,
        train_random_forest,
    )
    from projects.p4_habitat_suitability.src.occurrences import (
        load_occurrences,
        thin_occurrences,
    )
    from projects.p4_habitat_suitability.src.predictors import (
        build_predictor_stack,
    )
    from projects.p4_habitat_suitability.src.projection import (
        project_suitability,
        threshold_suitability,
    )

    config = load_config(config_path)
    output_dir = Path(config.get("data", {}).get("output_dir", "data/processed"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load and thin occurrences ---
    logger.info("Step 1: Loading occurrences")
    occ_gdf = load_occurrences(config)
    distance_km = config.get("species", {}).get("thinning_distance_km", 1.0)
    occ_thinned = thin_occurrences(occ_gdf, distance_km, config)

    # --- 2. Build predictor stack ---
    logger.info("Step 2: Building predictor stack")
    stack, profile, band_names = build_predictor_stack(config)

    # --- 3. Generate background points ---
    logger.info("Step 3: Generating background points")
    bg_gdf = generate_background_points(occ_thinned, stack, profile, config)

    # --- 4. Create presence/absence matrix ---
    logger.info("Step 4: Creating PA matrix")
    X, y = create_pa_matrix(occ_thinned, bg_gdf, stack, profile, band_names)  # noqa: N806

    # Coordinates for spatial CV
    import numpy as np

    pres_coords = np.column_stack([
        occ_thinned.geometry.x.values[:int(y.sum())],
        occ_thinned.geometry.y.values[:int(y.sum())],
    ])
    bg_coords = np.column_stack([
        bg_gdf.geometry.x.values[: int((y == 0).sum())],
        bg_gdf.geometry.y.values[: int((y == 0).sum())],
    ])
    coords = np.vstack([pres_coords, bg_coords])[:len(X)]

    # --- 5. Train models ---
    logger.info("Step 5: Training models")
    models: dict[str, Any] = {}
    algorithms = config.get("modeling", {}).get("algorithms", ["maxent", "random_forest"])

    cv_metrics: dict[str, Any] = {}
    for algo in algorithms:
        if algo == "maxent":
            model_fn = lambda Xt, yt: train_maxent(Xt, yt, config)  # noqa: N803
            models["maxent"] = train_maxent(X, y, config)
        else:
            model_fn = lambda Xt, yt: train_random_forest(Xt, yt, config)  # noqa: N803
            models["random_forest"] = train_random_forest(X, y, config)

        cv_result = spatial_block_cv(X, y, coords, model_fn, config)
        cv_metrics[algo] = cv_result

    # --- 6. Variable importance (use first available model) ---
    primary_model = models.get("random_forest", models.get("maxent"))
    if primary_model is not None:
        importance_df = compute_variable_importance(primary_model, X, y, band_names)
        importance_df.to_csv(output_dir / "variable_importance.csv", index=False)

    # --- 7. Project current suitability ---
    logger.info("Step 6: Projecting current suitability")
    if primary_model is not None:
        suitability, suit_profile = project_suitability(
            primary_model, stack, profile, config,
        )
        suit_path = output_dir / "suitability_current.tif"
        write_raster(suit_path, suitability, suit_profile)

        # --- 8. Future projections and change analysis ---
        # In a real workflow, future_stack would come from downscaled
        # CMIP6 predictors.  Here we demonstrate the change pipeline
        # using the current stack as a stand-in.
        future_suit, _ = project_suitability(primary_model, stack, profile, config)

        change = compute_change(suitability, future_suit)
        change_path = output_dir / "habitat_change.tif"
        write_raster(change_path, change, suit_profile, dtype="uint8")

        change_df = summarize_change(change, suit_profile)
        change_df.to_csv(output_dir / "change_summary.csv", index=False)
    else:
        suit_path = None
        change_df = None

    results: dict[str, Any] = {
        "occurrences_raw": len(occ_gdf),
        "occurrences_thinned": len(occ_thinned),
        "predictor_stack_shape": stack.shape,
        "band_names": band_names,
        "cv_metrics": cv_metrics,
        "suitability_current": str(suit_path) if suit_path else None,
        "change_summary": change_df,
        "models": models,
    }
    logger.info("Pipeline complete. Outputs in %s", output_dir)
    return results
