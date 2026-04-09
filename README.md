# Sierra Spatial Portfolio

![CI](https://github.com/woodsy-will/resume-spatial-portfoli0/actions/workflows/ci.yml/badge.svg)
![Coverage](https://img.shields.io/badge/coverage-%E2%89%A580%25-brightgreen)

**Advanced geospatial analytics for Sierra Nevada forest management.**

A monorepo containing four portfolio projects that demonstrate LiDAR processing, remote sensing, machine learning, and AI-powered spatial analysis — all grounded in operational forestry contexts.

---

## Repository Structure

```
sierra-spatial-portfolio/
├── agents/                          # Agent team definitions and orchestration
│   ├── AGENT_TEAM.md                # Agent roles, responsibilities, handoff protocol
│   └── orchestrator.py              # Task router and dependency manager
├── projects/
│   ├── p1_burn_severity/            # Post-Wildfire Burn Severity & Recovery Tracker
│   ├── p2_llm_spatial_query/        # LLM-Powered Spatial Query Interface
│   ├── p3_itc_delineation/          # Individual Tree Crown Delineation & Biometrics
│   └── p4_habitat_suitability/      # Habitat Suitability Modeling under Climate Scenarios
├── shared/                          # Cross-project utilities
│   ├── utils/                       # CRS handling, allometrics, I/O helpers
│   ├── data/                        # Shared data indices and download scripts
│   └── configs/                     # Environment configs, default parameters
├── docs/
│   └── architecture/                # System diagrams, data flow, agent architecture
├── .github/workflows/               # CI: linting, tests, dependency audits
├── environment.yml                  # Conda environment specification
├── pyproject.toml                   # Project metadata and dependencies
└── README.md                        # This file
```