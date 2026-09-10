# MeaningFlux AmeriFlux 2026 Workshop

This folder contains the materials for the hands-on MeaningFlux workshop at the 2026 AmeriFlux Annual Meeting.

## Workshop files

* `MeaningFlux_AmeriFlux2026_Tutorial.pdf`: Step-by-step participant instructions
* `MeaningFlux_AmeriFlux2026_Workshop_Slides.pdf`: Workshop presentation
* `AMF_US-Var_BASE_HH_27-5.csv`: AmeriFlux BASE sample dataset for US-Var

## Workshop analysis

We will use:

* Target variable: `LE`
* Environmental drivers: `NETRAD`, `VPD_PI`, `TA`, and `SWC_PI_1_2_A`
* Models: Linear Regression and Random Forest
* Analyses: model evaluation, mutual information, partial information decomposition, and observed-versus-modeled information fidelity

## Installation

From the main MeaningFlux repository folder, run:

```bash
conda create -n meaningflux-workshop python=3.10
conda activate meaningflux-workshop
pip install -r requirements-workshop.txt
python scripts/MeaningFlux_main.py
```

Please install MeaningFlux and confirm that the application opens before the workshop.

## Main repository

[MeaningFlux on GitHub](https://github.com/LBL-EESA/MeaningFlux)
