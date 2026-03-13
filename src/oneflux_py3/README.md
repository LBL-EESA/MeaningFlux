# oneflux_py3 (minimal, pure-Python helpers)

This mini-toolkit gives you:
- **Nighttime NEE partitioning** (Reichstein 2005/Lloyd–Taylor) in pure Python 3 (`partition_nt.partition_nee_nighttime`).
- **Simplified MDS gap-filling** for NEE and met variables in pure Python (`mds_gapfill.mds_fill_nee`, `mds_gapfill.mds_fill_met`).
- A **scaffold** for daytime partitioning (`partition_dt.partition_nee_daytime`) so you can plug in a solver later.

No SciPy required (uses linearization + NumPy/Pandas only).

## Install (local)
```bash
# no pip needed — just keep the folder in your project or add to PYTHONPATH
```

## Quick start
```python
import pandas as pd
from oneflux_py3.partition_nt import partition_nee_nighttime
from oneflux_py3.mds_gapfill import mds_fill_nee, mds_fill_met

# Suppose df has DateTimeIndex and columns: NEE, TA, VPD, RG
df = pd.read_csv('your_site.csv', parse_dates=True, index_col=0)

# Fill gaps (order matters: fill drivers first if needed)
df['TA_f']  = mds_fill_met(df['TA'])
df['VPD_f'] = mds_fill_met(df['VPD'])
df['RG_f']  = mds_fill_met(df['RG'])
df['NEE_f'] = mds_fill_nee(df['NEE'], df['RG_f'], df['TA_f'], df['VPD_f'])

# Nighttime partitioning → Reco & GPP
part = partition_nee_nighttime(df['NEE_f'], df['TA_f'], df['RG_f'])
df['Reco_nt'] = part['Reco']
df['GPP_nt']  = part['GPP']
df['Rref']    = part['Rref']
df['E0']      = part['E0']
```

### Notes
- **Sign convention**: NEE < 0 means uptake. At night, NEE ≈ Reco (>0). The code enforces GPP ≥ 0.
- **Parameters**: You can tweak radiation threshold (`rg_thresh`), window length (`window_days`), and minimum points (`min_pts`) in `partition_nee_nighttime`.
- **MDS strategy**: Tries ±7/14/28-day windows; relaxes similarity tolerances on (Rg, Ta, VPD); then falls back to mean diurnal course.

### Daytime partition (scaffold)
`partition_nee_daytime(...)` currently raises `NotImplementedError`. If you want Lasslop (2010) fitted, plug in `scipy.optimize.curve_fit` or `lmfit` to estimate the rectangular-hyperbola parameters in moving windows.

## License
MIT (for this helper code). This is **not** the official ONEFlux codebase.

## What’s new in this version
- **Robust nighttime partitioning**: iterative MAD outlier rejection, parameter bounds, and per-timestep **fit stats** (`n_used`, `r2`, and `qf_partition_nt`).
- **Smarter time-step inference**: works with **10/15/30/60-min** data; bins by within-day step length.
- **Improved MDS**: dynamic tolerances based on local variability, method **flags** embedded (see `filled.attrs['mds_method_codes']`), and a **monthly climatology** fallback.
- **Convenience pipeline**: `pipeline.process_site(df)` wires everything together.

### Minimal end-to-end
```python
from oneflux_py3.pipeline import process_site
df = pd.read_csv('your_site.csv', parse_dates=True, index_col=0)
df_out = process_site(df, nee_col='NEE', ta_col='TA', vpd_col='VPD', rg_col='RG')
```

### Outputs & QA
- `Reco_nt`, `GPP_nt`, `Rref`, `E0` from nighttime partitioning
- `qf_partition_nt` flag: 0=good, 2=weak fit (r2<0.3 or few points)
- Filled series `*_f` for `NEE`, `TA`, `VPD`, `RG`
```python
nee_methods_legend = df_out['NEE_f'].attrs.get('mds_method_codes', {})
```

> NOTE: This aims for **clarity and portability**, not bitwise identity with the official ONEFlux outputs. For validation, compare against a site-year using MAE/MEDIAN absolute diffs.
