# P3 — Individual Tree Crown Delineation & Biometric Extraction

## Objective

Process raw LiDAR point clouds to delineate individual tree crowns, extract per-tree structural biometrics (height, crown area, crown diameter), estimate DBH and volume via allometric equations, and validate against field cruise data.

## Data Requirements

- **LiDAR:** USGS 3DEP QL1 or better (≥ 8 pts/m²), LAZ format
- **Field Data:** Timber cruise plots with stem-mapped trees (PlotID, Species, DBH, Height, UTM coordinates)
- **CRS:** EPSG:3310 (California Albers NAD 83)

## Pipeline

```
LAZ tiles → PDAL ground classification (SMRF) → DTM (1.0 m)
                                                     │
                                              DSM - DTM = CHM
                                                     │
                                        Gaussian smoothing (σ=0.67 m)
                                                     │
                                   Variable-window local maxima → treetops
                                                     │
                                     Marker-controlled watershed → crowns
                                                     │
                                       Per-tree metrics extraction
                                                     │
                                    Allometric DBH/volume estimation
                                                     │
                                    Validation vs. cruise plots
```

## Key References

- Popescu, S.C., & Wynne, R.H. (2004). Seeing the Trees in the Forest. *Photogrammetric Engineering & Remote Sensing*, 70(5), 589–604.
- Dalponte, M., & Coomes, D.A. (2016). Tree-centric mapping of forest carbon density. *Methods in Ecology and Evolution*, 7(10), 1236–1245.
- Pillsbury, N.H., & Kirkley, M.L. (1984). Equations for total, wood, and saw-log volume for thirteen California hardwoods. USDA Forest Service PNW-GTR-414.

## Status

🔲 Not started
