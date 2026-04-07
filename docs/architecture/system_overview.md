# Sierra Spatial Portfolio — System Architecture

## Overview

This monorepo implements four complementary geospatial analytics projects
demonstrating advanced remote sensing, LiDAR processing, species distribution
modeling, and AI-powered spatial query capabilities for Sierra Nevada forest
management.

## Monorepo Structure

```
resume-spatial-portfoli0/
├── shared/                         # Cross-project shared code
│   ├── utils/
│   │   ├── allometry.py            # Pillsbury & Kirkley allometric equations
│   │   ├── config.py               # YAML config loader
│   │   ├── crs.py                  # CRS validation (EPSG:3310)
│   │   ├── io.py                   # Raster/vector I/O wrappers
│   │   ├── raster.py               # Reproject, resample, clip, mask
│   │   └── logging.py              # Standardized logger setup
│   └── data/
│       ├── generate_synthetic.py   # Synthetic test data for all projects
│       ├── download_3dep.py        # USGS 3DEP LiDAR acquisition
│       ├── download_sentinel2.py   # Sentinel-2 L2A via STAC
│       ├── download_occurrences.py # GBIF species occurrence download
│       └── download_worldclim.py   # WorldClim v2 bioclimatic rasters
│
├── projects/
│   ├── p1_burn_severity/           # Post-wildfire burn severity & recovery
│   │   ├── src/                    # acquisition, preprocessing, severity,
│   │   │                           # recovery, dashboard, pipeline
│   │   ├── tests/
│   │   ├── notebooks/
│   │   ├── docs/
│   │   └── configs/default.yaml
│   │
│   ├── p2_llm_spatial_query/       # Natural language → spatial SQL
│   │   ├── src/                    # schema_extractor, prompt_builder,
│   │   │                           # sql_generator, sql_validator,
│   │   │                           # executor, formatter, pipeline
│   │   ├── tests/
│   │   ├── notebooks/
│   │   ├── docs/
│   │   └── configs/
│   │       ├── default.yaml
│   │       ├── schema_metadata.yaml
│   │       └── few_shot_queries.yaml
│   │
│   ├── p3_itc_delineation/         # Individual tree crown delineation
│   │   ├── src/                    # ground_classify, dtm, chm, treetops,
│   │   │                           # segmentation, metrics, validation,
│   │   │                           # pipeline
│   │   ├── tests/
│   │   ├── notebooks/
│   │   ├── docs/
│   │   └── configs/default.yaml
│   │
│   └── p4_habitat_suitability/     # Species distribution modeling
│       ├── src/                    # occurrences, predictors, background,
│       │                           # modeling, projection, change_analysis,
│       │                           # pipeline
│       ├── tests/
│       ├── notebooks/
│       ├── docs/
│       └── configs/default.yaml
│
├── agents/AGENT_TEAM.md            # Agent team architecture
├── docs/architecture/              # This directory
├── .github/workflows/ci.yml        # CI/CD pipeline
├── environment.yml                 # Conda environment
└── pyproject.toml                  # Project metadata, pytest, ruff config
```

## Data Flow Between Projects

```
                    USGS 3DEP LiDAR
                         │
                         ▼
              ┌──────────────────────┐
              │  P3: ITC Delineation │
              │  LAZ → DTM → CHM →  │
              │  Treetops → Crowns → │
              │  Metrics → Validate  │
              └──────────┬───────────┘
                         │ CHM (canopy cover predictor)
                         ▼
    Sentinel-2       ┌──────────────────────────┐
        │            │  P4: Habitat Suitability  │     GBIF/CNDDB
        │            │  Occurrences → Predictors │ ◄── Occurrences
        │            │  → MaxEnt/RF → Project →  │     WorldClim
        │            │  Change Analysis          │
        │            └──────────┬────────────────┘
        ▼                       │
  ┌──────────────────┐          │
  │  P1: Burn        │          │
  │  Severity        │          │
  │  NBR → dNBR →    │          │
  │  Classify →      │          │
  │  Recovery →      │          │
  │  Dashboard       │          │
  └────────┬─────────┘          │
           │                    │
           ▼                    ▼
     ┌─────────────────────────────────┐
     │  GeoPackage (shared data store) │
     │  harvest_units, streams, roads, │
     │  severity maps, tree inventory, │
     │  suitability surfaces           │
     └──────────────┬──────────────────┘
                    │
                    ▼
         ┌────────────────────────┐
         │  P2: LLM Spatial Query │
         │  NL → RAG → SQL →     │
         │  Validate → Execute →  │
         │  Format Results        │
         └────────────────────────┘
```

## Key Design Principles

### 1. Config-Driven Processing
All parameters (CRS, thresholds, model hyperparameters, file paths) are stored
in YAML configuration files. No hardcoded values in source code.

### 2. Consistent CRS
All projects use **EPSG:3310 (California Albers NAD 83)** as the default CRS,
ensuring spatial consistency across all outputs. CRS validation is enforced
via `shared/utils/crs.py`.

### 3. Shared Utilities
Common operations (I/O, CRS handling, raster operations, allometry) are
centralized in `shared/utils/` to avoid code duplication.

### 4. Testable with Synthetic Data
All projects include synthetic data generators (`shared/data/generate_synthetic.py`)
enabling full test coverage without requiring real datasets. Real data download
scripts are provided for integration testing.

### 5. Safety-First for P2
The LLM spatial query interface enforces a strict SQL whitelist (SELECT only),
blocks all destructive operations, and validates every generated query before
execution against the GeoPackage.

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Language | Python ≥ 3.11 |
| CRS | EPSG:3310 (California Albers NAD 83) |
| Raster I/O | rasterio, rioxarray |
| Vector I/O | geopandas, fiona |
| Point Clouds | PDAL, laspy |
| Terrain | WhiteboxTools, scipy |
| Machine Learning | scikit-learn, maxnet/elapid |
| Visualization | matplotlib, plotly, Dash |
| Satellite Data | pystac-client (STAC) |
| Database | SpatiaLite, GeoPackage |
| LLM | Llama 3 8B (GPTQ-4bit), OpenAI-compatible API |
| Testing | pytest, pytest-cov |
| Linting | ruff |
| CI/CD | GitHub Actions |
| Environment | Conda (environment.yml) |
