# MeaningFlux

Open-source Python GUI implementing the MeaningFlux analytical framework for eddy covariance data: standardized visualization, gap-filling, footprint/fetch metrics, machine-learning predictability, and information-theoretic diagnostics.

---

## Overview

MeaningFlux is a Python-based graphical user interface (GUI) designed to support reproducible and transparent analysis of eddy covariance (EC) datasets.

The platform integrates visualization, diagnostics, and advanced analytical tools within a unified framework to facilitate exploration of ecosystem flux dynamics and their environmental drivers.

Core capabilities include:

- Time-series visualization of eddy covariance variables
- Data quality inspection and filtering
- Gap-filling tools for flux data
- Flux footprint and fetch metrics
- Annual flux budget estimation for flux variables
- Machine-learning-based predictability analysis
- Information-theoretic diagnostics (e.g., mutual information, transfer entropy)

The framework is intended to support both exploratory analysis and reproducible scientific workflows for ecosystem flux research.

---

## Version

Current release: **v1.0**

---

## Installation

Clone the repository:

```bash
git clone https://github.com/LBL-EESA/MeaningFlux.git
cd MeaningFlux

Create a Python environment (recommended):
conda create -n meaningflux python=3.10
conda activate meaningflux

Install required dependencies:
pip install -r requirements.txt

---

## Running MeaningFlux
From the repository root directory, launch the GUI with:

python MeaningFlux_main.py

This will open the MeaningFlux graphical interface.

---

## Data Requirements

MeaningFlux expects eddy covariance datasets formatted as tabular data files (e.g., CSV).

Typical requirements include:

A timestamp column named TIMESTAMP_START

Regular time intervals (e.g., half-hourly or hourly)

Standardized variable names

Missing values such as -9999 are automatically treated as missing data.

---

## Repository Structure

MeaningFlux/
│
├── MeaningFlux_main.py        # Main GUI launcher
├── src/                       # Core analysis modules
├── docs/                      # Documentation assets (figures, logo, screenshots)
├── requirements.txt           # Python dependencies
└── README.md

---

## Citation

If you use MeaningFlux in academic work, please cite the software repository:

Hernandez Rodriguez, L., & Wu, Y. (2026).
MeaningFlux (Version 1.0).
Lawrence Berkeley National Laboratory.
GitHub repository: https://github.com/LBL-EESA/MeaningFlux

A manuscript describing the MeaningFlux analytical framework is currently in preparation.

---

## Disclaimer

This software is provided for research and educational purposes. Users are responsible for verifying results and ensuring appropriate interpretation of outputs when applying the software to scientific analyses.

License and Copyright

*** Copyright Notice ***

MeaningFlux Copyright (c) 2026, The Regents of the University of California,
through Lawrence Berkeley National Laboratory (subject to receipt of
any required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov
.

NOTICE. This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights. As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative
works, and perform publicly and display publicly, and to permit others to do so.
