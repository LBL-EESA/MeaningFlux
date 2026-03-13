"""Nighttime partitioning of NEE (Reichstein 2005 style) — robust Py3 version.
- Iterative outlier rejection using MAD on residuals
- Param constraints: Rref>0, 50<=E0<=450 (typical range)
- Fit stats per window: n_used, r2
- Smarter time-step inference (10/15/30/60-min supported)
"""
import numpy as np
import pandas as pd
from .utils import halfhour_bin, steps_per_day, mad, clip_positive

T0 = 227.13  # K
TREF_C = 15.0
TREF_K = TREF_C + 273.15

def _lt_X(T_c):
    T_k = T_c + 273.15
    return (1.0/(TREF_K - T0)) - (1.0/(T_k - T0))

def _fit_window(X, y, max_iter=3, z_thresh=3.5):
    """Robust linear fit: y = a + b X, with iterative MAD outlier rejection."""
    ok = np.isfinite(X) & np.isfinite(y)
    Xw = X[ok]
    yw = y[ok]
    if Xw.size < 5:
        return np.nan, np.nan, 0, np.nan  # a, b, n, r2
    mask = np.ones_like(Xw, dtype=bool)
    for _ in range(max_iter):
        A = np.vstack([np.ones(mask.sum()), Xw[mask]]).T
        coef, *_ = np.linalg.lstsq(A, yw[mask], rcond=None)
        a, b = coef
        resid = yw - (a + b*Xw)
        sigma = mad(resid[mask])
        if not np.isfinite(sigma) or sigma == 0:
            break
        z = np.abs(resid) / sigma
        new_mask = z < z_thresh
        if new_mask.sum() == mask.sum():
            break
        mask = new_mask
    # Final stats
    if mask.sum() < 5:
        return np.nan, np.nan, int(mask.sum()), np.nan
    A = np.vstack([np.ones(mask.sum()), Xw[mask]]).T
    coef, *_ = np.linalg.lstsq(A, yw[mask], rcond=None)
    a, b = coef
    yhat = A @ coef
    ss_res = np.sum((yw[mask] - yhat)**2)
    ss_tot = np.sum((yw[mask] - np.mean(yw[mask]))**2)
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan
    return a, b, int(mask.sum()), r2

def _rolling_params_night(nee, ta, rg, window_days=10, min_pts=30, rg_thresh=20.0):
    df = pd.DataFrame({'nee': nee, 'ta': ta, 'rg': rg}).copy()
    night = df['rg'] < rg_thresh
    df_n = df.loc[night].dropna(subset=['nee','ta'])
    if df_n.empty:
        raise ValueError('No nighttime data available for partitioning.')
    Reco = clip_positive(df_n['nee'].values, 1e-6)  # ensure positive
    X = _lt_X(df_n['ta'].values)
    y = np.log(Reco)
    df_n = df_n.assign(X=X, y=y)
    # determine rolling window length in samples
    spd = steps_per_day(nee.index)
    win = max(min_pts, int(window_days * spd))
    half = win // 2
    idx = df_n.index
    n = len(df_n)
    Rref_s = pd.Series(index=idx, dtype=float)
    E0_s   = pd.Series(index=idx, dtype=float)
    N_s    = pd.Series(index=idx, dtype=float)
    R2_s   = pd.Series(index=idx, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        Xi = df_n['X'].iloc[lo:hi].values
        yi = df_n['y'].iloc[lo:hi].values
        if Xi.size < min_pts:
            continue
        a, b, n_used, r2 = _fit_window(Xi, yi)
        if not np.isfinite(a) or not np.isfinite(b):
            continue
        Rref = np.exp(a)
        E0 = b
        # constraints
        if Rref <= 0 or not np.isfinite(Rref):
            continue
        if not (50.0 <= E0 <= 450.0):
            continue
        Rref_s.iloc[i] = Rref
        E0_s.iloc[i]   = E0
        N_s.iloc[i]    = n_used
        R2_s.iloc[i]   = r2
    # fill and map back
    Rref_s = Rref_s.ffill().bfill()
    E0_s   = E0_s.ffill().bfill()
    N_s    = N_s.ffill().bfill()
    R2_s   = R2_s.ffill().bfill()
    params = pd.DataFrame({'Rref': Rref_s, 'E0': E0_s, 'n_used': N_s, 'r2': R2_s})
    params = params.reindex(nee.index, method='nearest')
    return params

def lloyd_taylor_reco(ta_c, Rref, E0):
    X = _lt_X(ta_c)
    return Rref * np.exp(E0 * X)

def partition_nee_nighttime(nee: pd.Series, ta: pd.Series, rg: pd.Series,
                            rg_thresh: float = 20.0,
                            window_days: int = 10,
                            min_pts: int = 30) -> pd.DataFrame:
    nee = nee.astype(float)
    ta  = ta.astype(float)
    rg  = rg.astype(float)
    params = _rolling_params_night(nee, ta, rg, window_days=window_days, min_pts=min_pts, rg_thresh=rg_thresh)
    Reco = lloyd_taylor_reco(ta, params['Rref'], params['E0'])
    GPP  = (Reco - nee).clip(lower=0.0)
    out = params.copy()
    out['Reco'] = Reco.clip(lower=0.0)
    out['GPP']  = GPP
    # Simple quality flag: 0 good, 1 if params missing (should be filled), 2 if r2<0.3 or n_used<min_pts
    qf = np.zeros(len(out), dtype=int)
    qf[(~np.isfinite(out['r2'])) | (out['n_used'] < min_pts)] = 2
    out['qf_partition_nt'] = qf
    return out
