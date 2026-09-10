# MeaningFlux AmeriFlux 2026 Workshop

This folder contains the materials for the hands-on MeaningFlux workshop at the 2026 AmeriFlux Annual Meeting.

## Workshop files

* `MeaningFlux_AmeriFlux2026_Tutorial.pdf`: Step-by-step participant guide
* `MeaningFlux_AmeriFlux2026_Workshop_Slides.pdf`: Workshop presentation
* `AMF_US-Var_BASE_HH_27-5.csv`: AmeriFlux BASE sample dataset for US-Var

## Workshop analysis

We will use:

* Target: `LE`
* Drivers: `NETRAD`, `VPD_PI`, `TA`, and `SWC_PI_1_2_A`
* Models: Linear Regression and Random Forest
* Analyses: model evaluation, mutual information, partial information decomposition, and observed-versus-modeled information fidelity

## Installation instructions

Please complete these steps before the workshop.

### 1. Install Conda

Install either:

* [Anaconda](https://www.anaconda.com/download)
* [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/)

### 2. Download MeaningFlux

1. Open the [MeaningFlux repository](https://github.com/LBL-EESA/MeaningFlux).
2. Click the green **Code** button.
3. Select **Download ZIP**.
4. Unzip `MeaningFlux-main.zip`.

### 3. Open a terminal

* **Windows:** Open **Anaconda Prompt**.
* **macOS:** Open **Terminal**.

### 4. Enter the MeaningFlux folder

Type `cd` followed by a space. Then drag the unzipped `MeaningFlux-main` folder into the terminal and press **Enter**.

For example:

```bash
cd /Users/your-name/Downloads/MeaningFlux-main
```

### 5. Install and start MeaningFlux

Copy and run each command separately:

```bash
conda create -n meaningflux-workshop python=3.10 -y
```

```bash
conda activate meaningflux-workshop
```

```bash
pip install -r requirements-workshop.txt
```

```bash
python scripts/MeaningFlux_main.py
```

The MeaningFlux window should open. Keep the terminal open while using the program.

Please confirm that MeaningFlux opens successfully before the workshop.

## Troubleshooting

If MeaningFlux does not open:

1. Confirm that `(meaningflux-workshop)` appears at the beginning of the terminal line.
2. Confirm that you are inside the `MeaningFlux-main` folder.
3. Copy the complete error message and bring it to the workshop.

## Main repository

[MeaningFlux on GitHub](https://github.com/LBL-EESA/MeaningFlux)
