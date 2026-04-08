"""Shared pytest fixtures for P1 burn severity tests.

Generates deterministic synthetic rasters and configuration for fast,
reproducible unit and integration testing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from shared.utils.config import load_config
from shared.utils.io import make_profile, write_raster

# ---------------------------------------------------------------------------
# Synthetic raster dimensions
# ---------------------------------------------------------------------------
_RNG = np.random.default_rng(99)
_H, _W = 20, 20
_BOUNDS = (-200_000.0, -50_000.0, -199_800.0, -49_800.0)
_RESOLUTION = 10.0


@pytest.fixture()
def config() -> dict[str, Any]:
    """Load the default P1 configuration."""
    cfg_path = (
        Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    )
    return load_config(cfg_path)


@pytest.fixture()
def profile() -> dict:
    """Rasterio-compatible profile for synthetic rasters."""
    return make_profile(_BOUNDS, _RESOLUTION)


@pytest.fixture()
def synthetic_nir_swir() -> dict[str, np.ndarray]:
    """Pre/post NIR and SWIR arrays with a known burn in the top half.

    Returns dict with keys ``pre_nir``, ``pre_swir``, ``post_nir``,
    ``post_swir``.
    """
    pre_nir = _RNG.uniform(0.3, 0.5, (_H, _W)).astype(np.float32)
    pre_swir = _RNG.uniform(0.05, 0.15, (_H, _W)).astype(np.float32)

    post_nir = pre_nir.copy()
    post_swir = pre_swir.copy()

    burn_rows = _H // 2
    post_nir[:burn_rows, :] = _RNG.uniform(0.05, 0.15, (burn_rows, _W))
    post_swir[:burn_rows, :] = _RNG.uniform(0.2, 0.4, (burn_rows, _W))

    return {
        "pre_nir": pre_nir,
        "pre_swir": pre_swir,
        "post_nir": post_nir,
        "post_swir": post_swir,
    }


@pytest.fixture()
def synthetic_raster_paths(
    tmp_path: Path, synthetic_nir_swir: dict[str, np.ndarray], profile: dict
) -> dict[str, Path]:
    """Write synthetic rasters to disk and return paths."""
    paths: dict[str, Path] = {}
    for name, data in synthetic_nir_swir.items():
        p = tmp_path / f"{name}.tif"
        write_raster(p, data, profile)
        paths[name] = p
    return paths


@pytest.fixture()
def scl_array() -> np.ndarray:
    """Synthetic SCL band with a few cloud pixels."""
    scl = np.full((_H, _W), 4, dtype=np.uint8)  # 4 = vegetation
    # Add cloud pixels in a 3x3 block
    scl[2:5, 2:5] = 9  # cloud high probability
    scl[7, 7] = 3  # cloud shadow
    return scl
