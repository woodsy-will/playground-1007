# P4 — Dynamic Habitat Suitability Modeling under Climate Scenarios

## Objective

Model current and projected habitat suitability for Pacific fisher (*Pekania pennanti*) using topographic, climatic, and vegetative predictors. Project onto CMIP6 scenarios to identify refugia and range shifts.

## Data Requirements

- **Occurrences:** GBIF, CNDDB — spatially thinned to 1 km grid
- **Predictors:** DEM derivatives (slope, TPI, TWI), bioclimatic variables (WorldClim v2 / PRISM), canopy cover (NLCD or LiDAR CHM)
- **Climate Projections:** CMIP6 downscaled via Basin Characterization Model or Cal-Adapt (SSP2-4.5, SSP5-8.5)
- **CRS:** EPSG:3310

## Pipeline

```
Occurrence data → spatial thinning → presence/background points
                                            │
Predictor stack alignment (common extent, resolution, CRS)
                                            │
MaxEnt + Random Forest → blocked spatial CV (5-fold)
                                            │
AUC, TSS, variable importance, partial dependence
                                            │
Project onto SSP2-4.5 / SSP5-8.5 at 2050, 2090
                                            │
Suitability change: gain / loss / stable refugia
                                            │
Cartographic output + white paper
```

## Key References

- Zielinski, W.J., et al. (2017). Resting habitat selection by fishers in the Sierra Nevada. *Journal of Wildlife Management*, 81(7), 1267–1277.
- Valavi, R., et al. (2019). blockCV: An R package for generating spatially blocked folds for cross-validation. *Methods in Ecology and Evolution*, 10(2), 225–232.
- Phillips, S.J., et al. (2006). Maximum entropy modeling of species geographic distributions. *Ecological Modelling*, 190(3–4), 231–259.

## Status

🔲 Not started
