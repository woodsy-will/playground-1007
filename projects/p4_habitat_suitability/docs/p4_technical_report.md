# P4 Technical Report: Dynamic Habitat Suitability Modelling under Climate Scenarios

## 1. Objective

Model the current and projected habitat suitability of the Pacific fisher
(*Pekania pennanti*) across the Sierra Nevada under multiple CMIP6 climate
scenarios (SSP2-4.5 and SSP5-8.5).  The workflow identifies climate refugia,
areas of potential range expansion, and zones of projected habitat loss to
inform conservation prioritisation.

## 2. Focal Species Ecology

The Pacific fisher is a medium-sized carnivore of the weasel family
(Mustelidae) endemic to the forests of western North America.  In the
Sierra Nevada, fishers depend on structurally complex, late-successional
conifer forests with high canopy cover, large-diameter snags for denning,
and moderate-to-low elevations (900--2,400 m).  Key habitat associations
include:

- Dense canopy cover (> 60 %) for thermal regulation and predator
  avoidance.
- Large trees and snags (> 90 cm DBH) for denning and resting.
- Moderate winter snowpack --- deep snow limits foraging access.
- Low-to-moderate slopes and mesic topographic positions.

The species' sensitivity to canopy structure and snow conditions makes it
a useful indicator for climate-driven habitat shifts in Sierra Nevada
forests (Zielinski et al. 2017).

## 3. Data Sources

### 3.1 Species Occurrence Records

| Source | Description                                    |
|--------|------------------------------------------------|
| GBIF   | Global Biodiversity Information Facility        |
| CNDDB  | California Natural Diversity Database           |

Records are filtered for spatial quality, thinned at 1 km to reduce
sampling bias, and projected to California Albers NAD83 (EPSG:3310).

### 3.2 Environmental Predictors

**Topographic variables** derived from a 30 m DEM (USGS 3DEP):

| Variable  | Description                                     |
|-----------|-------------------------------------------------|
| Elevation | Metres above sea level                          |
| Slope     | Terrain gradient (degrees) via `np.gradient`    |
| TPI       | Topographic Position Index (focal mean diff.)   |
| TWI       | Topographic Wetness Index: ln(a / tan(slope))   |

**Bioclimatic variables** from WorldClim v2 (Fick & Hijmans 2017):

| Variable           | Description                          |
|--------------------|--------------------------------------|
| BIO1               | Annual mean temperature              |
| BIO12              | Annual precipitation                 |

**Land cover** from the National Land Cover Database (NLCD):

| Variable     | Description                              |
|--------------|------------------------------------------|
| Canopy cover | Percent tree canopy (30 m resolution)    |

### 3.3 Future Climate Projections

Downscaled CMIP6 data from the Basin Characterization Model (Flint &
Flint 2014) at 270 m resolution, re-gridded to match the predictor stack:

| Scenario | Time step | Description                         |
|----------|-----------|-------------------------------------|
| SSP2-4.5 | 2050      | Moderate emissions, mid-century     |
| SSP2-4.5 | 2090      | Moderate emissions, late-century    |
| SSP5-8.5 | 2050      | High emissions, mid-century         |
| SSP5-8.5 | 2090      | High emissions, late-century        |

## 4. Methods

### 4.1 Spatial Thinning

Occurrence records are thinned to a minimum inter-point distance of 1 km
using a greedy spatial filter to reduce spatial autocorrelation and
geographic sampling bias (Aiello-Lammens et al. 2015).

### 4.2 Background Sampling

Target-group background points (n = 10,000) are generated within the
study extent, weighted by a kernel-density proxy of sampling effort
derived from all occurrence records (Phillips et al. 2009).

### 4.3 Species Distribution Models

Two complementary algorithms are trained:

**MaxEnt approximation** --- L1-penalised (lasso) logistic regression
with standardised features, following Renner & Warton (2013).  This
approximation avoids the Java dependency of the original MaxEnt software
while producing equivalent predictions under a Poisson point process
framework.

**Random Forest** --- An ensemble of 100 classification trees with
default hyper-parameters (min_samples_leaf = 5) from scikit-learn.

### 4.4 Spatial Block Cross-Validation

Model performance is evaluated using spatial block cross-validation
(Valavi et al. 2019) with 5 folds.  The study extent is divided into
spatial blocks along the easting axis, and blocks are assigned to folds
to prevent spatial autocorrelation leakage between training and test
sets.

**Evaluation metrics:**

| Metric | Description                                              |
|--------|----------------------------------------------------------|
| AUC    | Area under the ROC curve (discrimination ability)        |
| TSS    | True Skill Statistic (sensitivity + specificity - 1)     |

Metrics are reported as means with 95 % confidence intervals across
folds.

### 4.5 Variable Importance

Permutation-based variable importance (Breiman 2001) quantifies the
decrease in AUC when each predictor is randomly shuffled, providing a
model-agnostic measure of each variable's contribution to predictive
performance.

### 4.6 Suitability Projection

The best-performing model is applied to each scenario-time step
combination, producing continuous suitability surfaces (probability
0--1).  A binary habitat threshold is selected using the maximum TSS
criterion.

### 4.7 Change Analysis

Binary current and future maps are compared pixel-by-pixel to classify
each cell into one of four categories:

| Code | Class               | Description                          |
|------|---------------------|--------------------------------------|
| 0    | Stable unsuitable   | Not suitable in either period        |
| 1    | Refugia             | Suitable in both periods             |
| 2    | Gain                | Newly suitable under future climate  |
| 3    | Loss                | Suitable now but lost in future      |

Area statistics (hectares) are summarised per class and scenario.

## 5. Expected Outputs

1. **Thinned occurrence dataset** --- GeoPackage of filtered records
2. **Predictor stack** --- Aligned multi-band GeoTIFF
3. **Model diagnostics** --- CV metrics (AUC, TSS), variable importance
4. **Current suitability map** --- GeoTIFF (0--1 probability)
5. **Future suitability maps** --- one per scenario/time step
6. **Change rasters** --- gain/loss/refugia classification
7. **Change statistics** --- CSV summary table with area per class
8. **Reproducible pipeline** --- config-driven via `run_pipeline()`

## 6. References

- Aiello-Lammens, M. E., Boria, R. A., Radosavljevic, A., Vilela, B.,
  & Anderson, R. P. (2015). spThin: an R package for spatial thinning of
  species occurrence records for use in ecological niche models.
  *Ecography*, 38(5), 541--545.

- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5--32.

- Fick, S. E., & Hijmans, R. J. (2017). WorldClim 2: new 1-km spatial
  resolution climate surfaces for global land areas. *International Journal
  of Climatology*, 37(12), 4302--4315.

- Flint, L. E., & Flint, A. L. (2014). California Basin Characterization
  Model: a dataset of historical and future hydrologic response to climate
  change. *U.S. Geological Survey Data Release*.

- Phillips, S. J., Anderson, R. P., & Schapire, R. E. (2006). Maximum
  entropy modeling of species geographic distributions. *Ecological
  Modelling*, 190(3--4), 231--259.

- Phillips, S. J., Dudik, M., Elith, J., Graham, C. H., Lehmann, A.,
  Leathwick, J., & Ferrier, S. (2009). Sample selection bias and
  presence-only distribution models: implications for background and
  pseudo-absence data. *Ecological Applications*, 19(1), 181--197.

- Renner, I. W., & Warton, D. I. (2013). Equivalence of MAXENT and
  Poisson point process models for joint variable-selection/variable-
  regularization in species distribution modeling. *Biometrics*, 69(1),
  274--281.

- Valavi, R., Elith, J., Lahoz-Monfort, J. J., & Guillera-Arroita, G.
  (2019). blockCV: an R package for generating spatially or
  environmentally separated folds for k-fold cross-validation of species
  distribution models. *Methods in Ecology and Evolution*, 10(2), 225--232.

- Zielinski, W. J., Thompson, C. M., Purcell, K. L., & Garner, J. D.
  (2017). An assessment of fisher (*Pekania pennanti*) tolerance to
  forest management intensity on the landscape. *Forest Ecology and
  Management*, 310, 821--826.
