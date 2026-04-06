# Agent Team Architecture

## Overview

This document defines the five-agent team responsible for developing the Sierra Spatial Portfolio. Each agent has a defined scope, input/output contract, and handoff protocol.

## Agent Definitions

### 1. Orchestrator

**Scope:** Task decomposition, scheduling, dependency management, quality gates.

**Responsibilities:**
- Receive high-level objectives and decompose into atomic, assignable tasks
- Track inter-project dependencies (e.g., P3 CHM products feed P4 canopy cover predictor)
- Enforce quality gates before phase transitions
- Resolve blocks raised by downstream agents
- Maintain the project milestone tracker

**Does NOT:** Write code, perform analysis, or generate documentation content.

### 2. Data Engineer

**Scope:** Data acquisition, preprocessing, schema validation, pipeline infrastructure.

**Responsibilities:**
- Download and index source data (3DEP LiDAR, Sentinel-2, GBIF, WorldClim, fire perimeters)
- Validate data integrity (checksums, CRS consistency, completeness)
- Build ETL pipelines with config-driven parameters
- Maintain `shared/data/` download scripts and data manifests
- Schema validation — verify column names, dtypes, and spatial extents before handoff

**Output Contract:**
```yaml
manifest:
  dataset: "<name>"
  source_url: "<url>"
  crs: "EPSG:3310"
  extent: [xmin, ymin, xmax, ymax]
  resolution: <float>  # meters, for rasters
  row_count: <int>     # for vectors/tables
  columns: [...]       # exact names and dtypes
  checksum: "<sha256>"
  known_issues: [...]
```

### 3. Spatial Analyst

**Scope:** Core geospatial algorithm implementation — raster math, vector operations, LiDAR processing, terrain derivatives.

**Responsibilities:**
- Implement analysis pipelines: CHM generation, severity classification, suitability projection, spatial queries
- All code must be config-driven (no hardcoded thresholds, paths, or CRS)
- Vectorized operations preferred over explicit loops
- Every function includes docstring with parameter types, return types, and units

**Input Requirement:** Must receive a Data Engineer manifest before starting. If manifest is missing or incomplete, raise a block to Orchestrator.

**Output Contract:**
```yaml
manifest:
  output_type: "raster" | "vector" | "table"
  path: "<relative path>"
  crs: "EPSG:3310"
  description: "<what this output represents>"
  method: "<algorithm name and key parameters>"
  assumptions: [...]
  limitations: [...]
```

### 4. ML Engineer

**Scope:** Model training, evaluation, hyperparameter optimization, uncertainty quantification.

**Responsibilities:**
- Train and validate models (Random Forest, MaxEnt, allometric regressions, recovery curve fitting)
- Implement proper spatial cross-validation (blocked k-fold)
- Report evaluation metrics: AUC, TSS, RMSE, R², with confidence intervals
- Feature importance and partial dependence analysis
- Experiment tracking with reproducible configs

**Output Contract:**
```yaml
manifest:
  model_type: "<algorithm>"
  target_variable: "<name>"
  evaluation:
    metric: <value>
    ci_95: [lower, upper]
    cv_method: "spatial_block_5fold"
  feature_importance: {<feature>: <importance>, ...}
  artifacts: ["model.joblib", "metrics.json", "pdp_plots/"]
```

### 5. Technical Writer

**Scope:** Documentation, white papers, visualization, portfolio presentation.

**Responsibilities:**
- Maintain all README files and project documentation
- Produce technical white papers with methods, results, limitations, and management implications
- Create publication-quality figures (matplotlib/seaborn static, plotly interactive)
- Build dashboards and interactive visualizations
- Ensure all figures have labeled axes with units, colorblind-safe palettes, and statistical annotations

**Input Requirement:** Only begins deliverable production after Spatial Analyst and ML Engineer outputs pass quality gates.

## Handoff Protocol

```
User Request
    │
    ▼
Orchestrator ──► Decomposes into tasks
    │               │
    ├──► Data Engineer (Phase 1)
    │         │
    │         ▼ manifest
    │
    ├──► Spatial Analyst (Phase 2–3)
    │         │
    │         ▼ manifest
    │
    ├──► ML Engineer (Phase 4–5)
    │         │
    │         ▼ manifest
    │
    └──► Technical Writer (Phase 6–7)
              │
              ▼ deliverables
```

### Block Protocol

When an agent cannot proceed:

1. Agent identifies the missing input or invalid upstream output
2. Agent raises a **block** with:
   - `blocked_agent`: who is blocked
   - `blocking_dependency`: what is missing
   - `suggested_resolution`: how to unblock
3. Orchestrator routes the block to the responsible upstream agent
4. Upstream agent resolves and re-delivers with updated manifest
5. Blocked agent resumes

### Quality Gate Checklist

Before any phase transition, Orchestrator verifies:

- [ ] All upstream manifests present and complete
- [ ] Tests pass for new code (`pytest -v`)
- [ ] No hardcoded paths or magic numbers
- [ ] CRS consistent across all outputs (EPSG:3310 unless overridden)
- [ ] Outputs reproducible from `config.yaml` + source data + code
- [ ] Docstrings present on all public functions
- [ ] Git commit with conventional message format
