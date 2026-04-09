# Claude Code Instructions

## Git Workflow

- **Always `git fetch origin` and check branch state before starting any task.** Verify whether prior PRs have been merged and whether the working branch still exists or has diverged.
- Use **one branch per feature/issue**. Create from latest `main`. Never reuse a branch after its PR is merged.
- After a PR is merged by the user, start a new branch from the updated `main` for the next task.
- Never push new commits to a branch whose PR has already been merged.
- If `git push` fails with 503, use the MCP `push_files` tool as a fallback.

## Repo Structure

- **Monorepo** with 4 geospatial analytics projects for Sierra Nevada forest management.
- `projects/p1_burn_severity/` — Post-wildfire burn severity & recovery (Sentinel-2, dNBR, exponential recovery models)
- `projects/p2_llm_spatial_query/` — LLM-powered natural-language to spatial SQL (GeoPackage, SpatiaLite, safety validator)
- `projects/p3_itc_delineation/` — Individual tree crown delineation (LiDAR CHM, watershed segmentation, allometric biometrics)
- `projects/p4_habitat_suitability/` — Habitat suitability modeling (MaxEnt + Random Forest ensemble, spatial block CV, climate projections)
- `shared/` — Cross-project utilities (CRS, I/O, allometry, logging, raster ops, synthetic data generators)
- Each project: `src/`, `tests/`, `configs/`, `notebooks/`, `docs/`

## Testing & CI

- **pytest** with `pyproject.toml` config. Run: `python -m pytest projects/ -v --tb=short`
- **pytest-cov** configured: 80% minimum threshold. Run with `--cov=projects --cov=shared --cov-config=pyproject.toml`
- **ruff** for linting: `ruff check . --config pyproject.toml`
- E402 ignored for notebooks (imports after `sys.path.insert` are expected)
- CI workflow: `.github/workflows/ci.yml` — lint job + test-with-coverage job
- Always verify ruff clean + all tests pass before pushing.

## Open Issues

- **#2** — P2: Replace regex SQL validator with a real SQL parser (enhancement, security)
- **#4** — P3: Improve exception handling in metrics with specific error types and quality flags (enhancement)

## Completed Issues (for reference)

- **#3** — P1: Raster reprojection in `reproject_and_clip()` — implemented with configurable resampling
- **#5** — CI: pytest-cov integration with 80% threshold and coverage artifact upload
- **#6** — P4: AUC-weighted ensemble model support with uncertainty maps

## Key Design Decisions

- P2 SQL validator uses regex + `_ALWAYS_BLOCKED` frozenset (PRAGMA, ATTACH, DETACH, VACUUM, LOAD_EXTENSION). Defense-in-depth: executor also disables `enable_load_extension` after SpatiaLite init.
- P4 ensemble weights models by CV AUC; NaN AUC defaults to 0.5. Single model degenerates correctly.
- Default CRS is EPSG:3310 (California Albers NAD 83) across all projects.
- Synthetic data generators in `shared/data/generate_synthetic.py` produce deterministic test fixtures.
