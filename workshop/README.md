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

## Installation instructions

### 1. Install Anaconda or Miniconda

If you do not already have Anaconda or Miniconda, install one of them:

- [Download Anaconda](https://www.anaconda.com/download)
- [Download Miniconda](https://docs.conda.io/projects/miniconda/en/latest/)

### 2. Download MeaningFlux

1. Open the [MeaningFlux repository](https://github.com/LBL-EESA/MeaningFlux).
2. Click the green **Code** button.
3. Select **Download ZIP**.
4. Open your Downloads folder.
5. Unzip `MeaningFlux-main.zip`.

### 3. Open a terminal

- **Windows:** Open **Anaconda Prompt** from the Start menu.
- **macOS:** Open the **Terminal** application.

### 4. Move into the MeaningFlux folder

Type `cd`, add one space, and then drag the unzipped `MeaningFlux-main` folder from your file browser into the terminal window.

The command will look similar to:

**macOS**

```bash
cd /Users/your-name/Downloads/MeaningFlux-main

Please install MeaningFlux and confirm that the application opens before the workshop.

## Main repository

[MeaningFlux on GitHub](https://github.com/LBL-EESA/MeaningFlux)
