"""End-to-end burn severity and recovery analysis pipeline.

Orchestrates: acquisition -> preprocessing -> severity -> recovery.
When real imagery is unavailable (no STAC credentials or offline), the
pipeline falls back to synthetic data for demonstration purposes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from shared.utils.config import load_config
from shared.utils.io import read_raster, write_raster, make_profile
from shared.utils.logging import get_logger

from projects.p1_burn_severity.src.severity import (
    classify_severity,
    compute_dnbr,
    compute_nbr,
    compute_rbr,
)
from projects.p1_burn_severity.src.recovery import (
    build_recovery_timeseries,
    compute_vegetation_index,
    fit_recovery_model,
)

logger = get_logger("p1.pipeline")


def run_pipeline(config_path: str | Path) -> dict[str, Any]:
    """Execute the full burn severity and recovery pipeline.

    Steps
    -----
    1. Load configuration.
    2. Load pre-fire and post-fire NIR / SWIR rasters (from config paths
       or synthetic data).
    3. Compute NBR, dNBR, severity classification, and RBR.
    4. Generate synthetic annual recovery rasters and build a recovery
       time series.
    5. Fit exponential recovery model.

    Parameters
    ----------
    config_path : str or Path
        Path to YAML configuration file.

    Returns
    -------
    dict[str, Any]
        ``output_paths`` — dict of written file paths.
        ``summary`` — dict with pixel counts per severity class and
        recovery model parameters.
    """
    config = load_config(config_path)
    data_cfg = config.get("data", {})
    proc_cfg = config.get("processing", {})
    output_dir = Path(data_cfg.get("output_dir", "data/processed"))
    output_dir.mkdir(parents=True, exist_ok=True)

    imagery_dir = Path(data_cfg.get("imagery_dir", "data/raw/imagery"))
    output_paths: dict[str, Path] = {}

    # ------------------------------------------------------------------
    # 1. Load imagery (fall back to synthetic if not present)
    # ------------------------------------------------------------------
    pre_nir_path = imagery_dir / "pre_nir.tif"
    pre_swir_path = imagery_dir / "pre_swir.tif"
    post_nir_path = imagery_dir / "post_nir.tif"
    post_swir_path = imagery_dir / "post_swir.tif"

    if not pre_nir_path.exists():
        logger.info("Imagery not found — generating synthetic data")
        from shared.data.generate_synthetic import generate_synthetic_burn_rasters

        synth = generate_synthetic_burn_rasters(imagery_dir)
        pre_nir_path = synth["pre_nir"]
        pre_swir_path = synth["pre_swir"]
        post_nir_path = synth["post_nir"]
        post_swir_path = synth["post_swir"]

    pre_nir, profile = read_raster(pre_nir_path)
    pre_swir, _ = read_raster(pre_swir_path)
    post_nir, _ = read_raster(post_nir_path)
    post_swir, _ = read_raster(post_swir_path)

    # Squeeze to 2-D if needed
    if pre_nir.ndim == 3:
        pre_nir = pre_nir[0]
    if pre_swir.ndim == 3:
        pre_swir = pre_swir[0]
    if post_nir.ndim == 3:
        post_nir = post_nir[0]
    if post_swir.ndim == 3:
        post_swir = post_swir[0]

    # ------------------------------------------------------------------
    # 2. Compute burn severity
    # ------------------------------------------------------------------
    pre_nbr = compute_nbr(pre_nir, pre_swir)
    post_nbr = compute_nbr(post_nir, post_swir)
    dnbr = compute_dnbr(pre_nbr, post_nbr)
    severity = classify_severity(dnbr, config)
    rbr = compute_rbr(dnbr, pre_nbr)

    # Write outputs
    sev_path = output_dir / "severity.tif"
    dnbr_path = output_dir / "dnbr.tif"
    rbr_path = output_dir / "rbr.tif"

    write_raster(sev_path, severity, profile, dtype="uint8", nodata=255)
    write_raster(dnbr_path, dnbr.astype(np.float32), profile)
    write_raster(rbr_path, rbr.astype(np.float32), profile)

    output_paths["severity"] = sev_path
    output_paths["dnbr"] = dnbr_path
    output_paths["rbr"] = rbr_path

    logger.info("Severity classification written to %s", sev_path)

    # ------------------------------------------------------------------
    # 3. Build recovery time series (synthetic annual NDVI)
    # ------------------------------------------------------------------
    years_post = proc_cfg.get("years_post_fire", 5)
    rng = np.random.default_rng(42)
    annual_rasters: list[np.ndarray] = []

    for yr in range(1, years_post + 1):
        # Simulate gradual recovery: NDVI increases each year
        base_ndvi = 0.2 + 0.12 * yr + rng.normal(0, 0.02, pre_nir.shape)
        # Unburned areas remain stable
        base_ndvi = np.where(severity == 0, 0.7 + rng.normal(0, 0.02, pre_nir.shape), base_ndvi)
        # Scale recovery by severity
        for sev_cls in range(1, 5):
            mask = severity == sev_cls
            penalty = sev_cls * 0.05
            base_ndvi[mask] -= penalty
        base_ndvi = np.clip(base_ndvi, 0, 1).astype(np.float32)
        annual_rasters.append(base_ndvi)

    recovery_ts = build_recovery_timeseries(annual_rasters, severity, config)
    recovery_model = fit_recovery_model(recovery_ts, config)

    ts_path = output_dir / "recovery_timeseries.csv"
    model_path = output_dir / "recovery_model.csv"
    recovery_ts.to_csv(ts_path, index=False)
    recovery_model.to_csv(model_path, index=False)
    output_paths["recovery_timeseries"] = ts_path
    output_paths["recovery_model"] = model_path

    logger.info("Recovery time series written to %s", ts_path)

    # ------------------------------------------------------------------
    # 4. Summary statistics
    # ------------------------------------------------------------------
    valid = severity[severity != 255]
    class_counts = {
        int(cls): int(np.count_nonzero(valid == cls)) for cls in range(5)
    }

    summary: dict[str, Any] = {
        "total_pixels": int(len(valid)),
        "class_counts": class_counts,
        "mean_dnbr": float(np.nanmean(dnbr)),
        "recovery_model": (
            recovery_model.to_dict(orient="records")
            if not recovery_model.empty
            else []
        ),
    }

    logger.info("Pipeline complete. Summary: %s", summary)
    return {"output_paths": output_paths, "summary": summary}
