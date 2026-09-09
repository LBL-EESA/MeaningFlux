<p align="center">
  <img src="docs/MeaningFlux_logo.png" width="300" alt="MeaningFlux logo">
</p>

# MeaningFlux

**MeaningFlux** is an open-source Python graphical workflow for exploratory analysis, machine learning, and information-theoretic interpretation of eddy-covariance (EC) data.

MeaningFlux links conventional EC data exploration and quality assessment with machine-learning prediction and information-theory diagnostics. The workflow is designed to help users ask not only **how well a model predicts a flux**, but also **whether model predictions preserve the observed information relationships between fluxes and environmental drivers**.

---

## Overview

MeaningFlux is intended as a **post-processing and interpretation layer** for eddy-covariance datasets. It brings together visualization, QA/QC, gap filling, footprint context, machine learning, and information theory within a common graphical interface.

<p align="center">
  <img src="docs/1_main.png" width="650" alt="MeaningFlux main interface">
</p>

A typical workflow is:

**Load EC data → Explore and QA/QC → Gap-fill when needed → Run ML → Create the ML-to-IT bridge → Analyze observed and modeled information structure**

MeaningFlux does not replace standard EC processing, site-specific quality control, or process-based scientific interpretation.

---

## Core Capabilities

### EC exploration and visualization

- Time-series visualization
- Daily-average visualization
- Density and scatter plots
- Correlation analysis
- Wind-rose visualization
- Directional flux-contribution diagnostics
- Data-availability assessment
- Flux-budget calculations

### QA/QC

- Guided standard QA/QC
- AmeriFlux BASE QA/QC
- Missing-data and completeness assessment
- Support for identifying usable analysis periods

### Footprint and source-area context

- Flux Footprint Prediction (FFP) calculations
- Footprint climatology
- Fetch-rose diagnostics
- Directional source-area visualization
- BADM/site metadata support

### Gap filling and flux completion

- ONEFlux-based gap filling
- CO2 flux partitioning utilities
- N2O gap filling
- CH4 gap filling
- Multiple machine-learning options for CH4 gap filling.

Original and gap-filled variables are kept distinguishable so users can preserve data provenance in downstream analyses.

Optional CH4 gap-filling dependency:

```bash
pip install xgboost
---

## Machine Learning Toolbox

The Machine Learning Toolbox evaluates out-of-sample predictability of EC fluxes and creates aligned outputs for subsequent information-theory analysis.

### Supported models

- Linear Regression
- Random Forest
- Multilayer Perceptron (MLP)
- Long Short-Term Memory network (LSTM)
- Hysteresis-Gate LSTM (H-LSTM)

The current H-LSTM uses the same Keras sequence architecture and training framework as the standard LSTM, with an additional gate-derived input channel for controlled comparison.

### Target variables

MeaningFlux includes target-aware support for common EC variables such as:

- FC / NEE
- FCH4
- FN2O
- LE
- H
- GPP
- RECO

Other numeric targets can also be selected.

### Predictor support

MeaningFlux provides literature-informed predictor presets that update with the selected target. Candidate predictors can include variables representing:

- radiation
- air temperature
- vapor pressure deficit or relative humidity
- soil temperature
- soil moisture
- precipitation
- atmospheric pressure
- wind and turbulence
- vegetation state
- other site-specific environmental drivers

Predictor presets are starting points and should be reviewed according to the site, target, temporal scale, and scientific question.

### Temporal aggregation and validation

Analyses can be performed at:

- Native resolution
- Daily resolution
- Weekly resolution

The ML workflow includes:

- chronological holdout validation
- expanding-window blocked cross-validation
- configurable training/test fractions
- configurable fold structure
- minimum within-period coverage
- target-aware aggregation rules
- user-defined aggregation overrides
- leakage-aware scaling for neural-network models
- gap-safe sequence construction for LSTM/H-LSTM

### Model diagnostics

MeaningFlux reports and visualizes:

- R²
- RMSE
- normalized RMSE
- MAE
- fold-level performance
- observed-versus-predicted behavior
- repeated model comparisons
- held-out Random Forest permutation importance

### Exploratory predictor screening

An optional predictor-screening workflow compares evidence from:

- literature-informed predictor presets
- Random Forest permutation importance
- Pearson correlation
- standardized linear coefficients
- partial-dependence diagnostics

This module is intended for exploratory screening rather than definitive attribution of ecosystem controls.

---

## ML-to-IT Bridge

A central feature of MeaningFlux is the **ML-to-IT bridge**.

After fitting ML models, MeaningFlux can export a timestamp-aligned table containing:

- environmental drivers
- observed target flux
- model predictions
- residuals
- timestamps and temporal metadata
- evaluation labels

This common structure allows the Information Theory Toolbox to analyze the same environmental relationships in both the observations and the model predictions.

The bridge supports the transition from:

> **How well does the model predict?**

to:

> **Does the model preserve the information structure observed in the measurements?**

---

## Information Theory Toolbox

The Information Theory Toolbox provides complementary diagnostics for statistical dependence, temporal organization, interactions among drivers, and model information fidelity.

### Entropy

- Entropy, H(X)

Entropy quantifies the uncertainty or information content of a selected variable.

### Mutual information

- Mutual Information, I(X;Y)
- Normalized mutual information
- MI Driver Ranking
- Correlation vs MI Ranking

Mutual information captures statistical dependence without restricting the relationship to a linear form.

### Conditional mutual information

- Conditional MI, I(X;Y|Z)

Conditional MI evaluates the remaining dependence between two variables after conditioning on a third variable.

### Lagged mutual information

- Lagged MI, I(X_t;Y_{t+lag})

Lagged MI evaluates time-shifted dependence while preserving the temporal grid and missing time steps before shifting.

### Partial information decomposition

- Two-source PID
- Pairwise PID matrices

PID decomposes information supplied by two drivers into:

- redundant information
- unique information from each driver
- synergistic information

This provides a complementary perspective to conventional feature-importance methods by explicitly distinguishing overlapping and joint information among environmental drivers.

### Transfer entropy

- Transfer Entropy, TE(X→Y)
- Transfer Entropy vs Lag
- TE networks
- bidirectional source-target evaluation
- directed network visualizations

Transfer entropy evaluates directional lagged information transfer while conditioning on the target's own past.

**Transfer entropy should be interpreted as a diagnostic of directional information structure and timing, not as proof of physical causality.**

### Temporal support and significance

MeaningFlux includes support for:

- temporal surrogate testing
- circular-shift, block-shuffle, and random surrogates
- configurable lag ranges
- minimum aligned-sample thresholds
- maximum-statistic support across searched lags

### Analysis windows

Information-theory analyses can be restricted to:

- all available data
- daytime
- nighttime
- growing season
- non-growing season
- selected months
- custom date ranges

These options are particularly important when the interpretation of a flux depends on ecosystem state, seasonality, or day/night processes.

---

## Model Information Fidelity

MeaningFlux directly compares the information structure of **observed** and **modeled** fluxes.

Available diagnostics include:

### Model-vs-Observed MI

For the same environmental driver, MeaningFlux compares its normalized mutual information with:

- the observed target, and
- the model-predicted target.

The resulting information mismatch indicates whether the modeled target overrepresents, underrepresents, or closely reproduces the observed driver-target dependence.

### Model-vs-Observed PID

MeaningFlux compares observed and modeled pairwise PID structure, including differences in:

- redundancy
- unique information
- synergy

### Functional Performance Summary

MeaningFlux combines individual-driver MI fidelity and pairwise PID fidelity into a compact functional-performance summary.

These diagnostics complement conventional predictive metrics such as R² and RMSE. A model can predict a flux well while still misrepresenting the environmental information relationships present in the observations.

MeaningFlux therefore treats **predictive skill** and **information fidelity** as complementary dimensions of model evaluation.

---

## Scientific Interpretation

Information-theory metrics identify statistical information relationships in the data; they do not by themselves establish physical mechanism or causality.

Results should be interpreted in the context of:

- ecosystem processes
- the physical meaning of the target flux
- temporal aggregation
- day/night structure
- seasonality
- autocorrelation
- variable selection
- data coverage
- site characteristics
- preprocessing choices

---

## Version

Current release: **v1.0**

---

## Installation

Clone the repository:

```bash
git clone https://github.com/LBL-EESA/MeaningFlux.git
cd MeaningFlux
```

Create a Python 3.10 environment:

```bash
conda create -n meaningflux python=3.10
conda activate meaningflux
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The standard installation includes the dependencies required for the current ML workflow, including TensorFlow/Keras for LSTM and H-LSTM and XGBoost for the corresponding CH4 gap-filling option.

---

## Running MeaningFlux

From the repository root directory:

```bash
python scripts/MeaningFlux_main.py
```

This opens the MeaningFlux graphical interface.

---

## Data Requirements

MeaningFlux is designed for post-processed eddy-covariance datasets stored as tabular files such as CSV.

A typical input dataset contains:

- a timestamp column, preferably `TIMESTAMP_START`
- one or more EC flux variables
- environmental or ecosystem predictor variables
- a regular or inferable temporal resolution

Common target variables include FC/NEE, FCH4, FN2O, LE, H, GPP, and RECO.

Typical environmental predictors include radiation, temperature, atmospheric demand, soil moisture, precipitation, wind/turbulence, and vegetation variables.

AmeriFlux-style missing-value codes such as `-9999` are treated as missing data where supported by the workflow.

MeaningFlux does not require every dataset to contain the same variables. Users should select the target, candidate drivers, temporal window, and preprocessing choices appropriate to their scientific question.

---

## Repository Structure

```text
MeaningFlux/
│
├── scripts/
│   └── MeaningFlux_main.py
├── src/
│   ├── open_machine_learning_toolbox.py
│   ├── open_information_theory_toolbox.py
│   ├── oneflux_py3/
│   └── additional visualization, QA/QC, gap-filling, and footprint modules
├── docs/
│   └── images and documentation
├── requirements.txt
├── CITATION.cff
└── README.md
```

---

## AmeriFlux 2026 Hands-On Workshop

MeaningFlux is being used in the AmeriFlux 2026 breakout session:

**Hands-On Machine Learning and Information Theory for Eddy-Covariance Data**

The workshop uses MeaningFlux to move from prediction to interpretation by asking:

> **Can an ML model predict an eddy-covariance flux well while failing to preserve the environmental information relationships observed in the measurements?**

The hands-on workflow will demonstrate how to:

1. load an EC dataset,
2. define a target flux and environmental drivers,
3. train and compare ML models,
4. evaluate observed driver-flux dependence with mutual information,
5. examine redundant, unique, and synergistic information with PID,
6. create the ML-to-IT bridge,
7. compare observed and modeled information structure, and
8. interpret predictive skill and information fidelity together.

Workshop materials will be added to this repository.

---

## Citation

If MeaningFlux contributes to your research, analysis, figures, or publication, please cite the software.

**Hernandez Rodriguez, L. C. (2026). _MeaningFlux v1.0_ [Computer software]. Lawrence Berkeley National Laboratory.**

Official DOE CODE / OSTI software record:

https://www.osti.gov/doecode/biblio/178023

Source code:

https://github.com/LBL-EESA/MeaningFlux

The repository also includes a `CITATION.cff` file for citation export through GitHub.

A manuscript describing the MeaningFlux analytical framework and technical validation has been submitted to the *Journal of Geophysical Research: Biogeosciences*.

---

## Author

**Leila Hernandez Rodriguez**  
Postdoctoral Research Fellow  
Energy Geosciences Division  
Lawrence Berkeley National Laboratory

GitHub: https://github.com/leilaher  
LinkedIn: https://www.linkedin.com/in/leilaher/

---

## Funding

Development of MeaningFlux and portions of the associated analyses were supported by the SMARTFARM program of the U.S. Department of Energy's Advanced Research Projects Agency–Energy (ARPA-E).

Additional project-specific funding and acknowledgments are provided in the associated scientific publications.

---

## Feedback and Contributions

MeaningFlux is intended for use across diverse eddy-covariance datasets and ecosystem types.

Users are encouraged to:

- test MeaningFlux with their own EC datasets,
- report bugs or unexpected behavior,
- suggest workflow or diagnostic improvements, and
- contribute reproducible examples and use cases.

Issues and suggestions can be submitted through the GitHub repository.

---

## Disclaimer

MeaningFlux is provided for research and educational purposes.

Users are responsible for:

- verifying input-data quality,
- selecting scientifically appropriate variables and analysis windows,
- evaluating assumptions associated with statistical and machine-learning methods,
- validating model outputs, and
- interpreting results in the appropriate ecological and physical context.

Information-theory metrics quantify statistical information relationships and should not be interpreted as direct evidence of physical causality without additional supporting analysis.

---

## Copyright

*** Copyright Notice ***

MeaningFlux Copyright (c) 2026, The Regents of the University of California,
through Lawrence Berkeley National Laboratory (subject to receipt of
any required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at:

IPO@lbl.gov

NOTICE. This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights. As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative
works, and perform publicly and display publicly, and to permit others to do so.

See `LICENSE.txt` for the software license and terms of use.
