"""End-to-end ITC delineation pipeline.

Orchestrates ground classification, DTM/CHM generation, treetop detection,
crown segmentation, metric extraction, and validation against cruise data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from shared.utils.config import load_config
from shared.utils.logging import get_logger

logger = get_logger("p3_pipeline")


def run_pipeline(config_path: str | Path) -> dict[str, Any]:
    """Execute the full ITC delineation workflow.

    Parameters
    ----------
    config_path : str or Path
        Path to the project YAML configuration file.

    Returns
    -------
    dict
        Dictionary with keys:
        - ``classified_laz``: Path to classified point cloud
        - ``dtm``: Path to DTM raster
        - ``chm``: Path to CHM raster
        - ``treetops``: Path to treetop vector file
        - ``crowns``: Path to crown polygon vector file
        - ``metrics``: Path to enriched crown metrics vector file
        - ``validation``: dict of validation statistics
    """
    from projects.p3_itc_delineation.src.chm import generate_chm
    from projects.p3_itc_delineation.src.dtm import generate_dtm
    from projects.p3_itc_delineation.src.ground_classify import classify_ground
    from projects.p3_itc_delineation.src.metrics import extract_tree_metrics
    from projects.p3_itc_delineation.src.segmentation import segment_crowns
    from projects.p3_itc_delineation.src.treetops import detect_treetops
    from projects.p3_itc_delineation.src.validation import validate_against_cruise

    from shared.utils.io import list_files, write_vector

    config = load_config(config_path)
    data_cfg = config.get("data", {})
    output_dir = Path(data_cfg.get("output_dir", "data/processed"))
    output_dir.mkdir(parents=True, exist_ok=True)

    lidar_dir = Path(data_cfg.get("lidar_dir", "data/raw/lidar"))
    laz_files = list_files(lidar_dir, "*.laz")
    if not laz_files:
        raise FileNotFoundError(f"No LAZ files found in {lidar_dir}")

    laz_path = laz_files[0]
    logger.info("Processing tile: %s", laz_path.name)

    # 1. Ground classification
    classified_laz = classify_ground(laz_path, config)

    # 2. DTM generation
    dtm_path = generate_dtm(classified_laz, config)

    # 3. CHM generation (from classified LAZ + DTM)
    chm_path = generate_chm(classified_laz, dtm_path, config)

    # 4. Treetop detection
    treetops_gdf = detect_treetops(chm_path, config)
    treetops_path = output_dir / "treetops.gpkg"
    write_vector(treetops_gdf, treetops_path)

    # 5. Crown segmentation
    crowns_gdf = segment_crowns(chm_path, treetops_gdf, config)
    crowns_path = output_dir / "crowns.gpkg"
    write_vector(crowns_gdf, crowns_path)

    # 6. Metric extraction
    metrics_gdf = extract_tree_metrics(crowns_gdf, chm_path, config)
    metrics_path = output_dir / "tree_metrics.gpkg"
    write_vector(metrics_gdf, metrics_path)

    # 7. Validation (if cruise data exists)
    validation_results: dict[str, Any] = {}
    cruise_path = Path(data_cfg.get("cruise_plots", "data/raw/cruise_plots.csv"))
    if cruise_path.exists():
        # Use crown centroids for matching
        pred_centroids = metrics_gdf.copy()
        pred_centroids["geometry"] = pred_centroids.geometry.centroid
        validation_results = validate_against_cruise(pred_centroids, cruise_path, config)
        logger.info("Validation results: %s", validation_results)
    else:
        logger.warning("No cruise data at %s — skipping validation", cruise_path)

    results = {
        "classified_laz": classified_laz,
        "dtm": dtm_path,
        "chm": chm_path,
        "treetops": treetops_path,
        "crowns": crowns_path,
        "metrics": metrics_path,
        "validation": validation_results,
    }
    logger.info("Pipeline complete. Outputs in %s", output_dir)
    return results
