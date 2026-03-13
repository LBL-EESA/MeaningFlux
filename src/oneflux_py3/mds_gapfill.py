"""Improved MDS gap-filling for NEE and meteorology.
Adds:
- Dynamic tolerances (relative to local variability)
- Method flags per filled point
- Monthly climatology fallback after MDC
"""
import numpy as np
import pandas as pd
from .utils import halfhour_bin, doy, steps_per_day

METHOD_CODES = {
    0: 'original',
    1: 'similar_RgTaVPD',
    2: 'relaxed_RgTaVPD',
    3: 'MDC',
    4: 'monthly_climatology',
    5: 'window_mean'
}

def _monthly_climatology(series: pd.Series) -> pd.Series:
    return series.groupby([series.index.month, series.index.hour, series.index.minute]).transform('mean')

def _dynamic_tols(df_window, base_r_t=50.0, base_t_t=2.5, base_v_t=5.0):
    # Scale with variability in the window (IQR)
    rg_iqr = (df_window['rg'].quantile(0.75) - df_window['rg'].quantile(0.25))
    ta_iqr = (df_window['ta'].quantile(0.75) - df_window['ta'].quantile(0.25))
    vpd_iqr = (df_window['vpd'].quantile(0.75) - df_window['vpd'].quantile(0.25))
    rg_tol = max(base_r_t, 0.5 * rg_iqr) if np.isfinite(rg_iqr) else base_r_t
    ta_tol = max(base_t_t, 0.5 * ta_iqr) if np.isfinite(ta_iqr) else base_t_t
    vpd_tol = max(base_v_t, 0.5 * vpd_iqr) if np.isfinite(vpd_iqr) else base_v_t
    return rg_tol, ta_tol, vpd_tol

def _select_similar(sel, hh, rg, ta, vpd, rg_tol, ta_tol, vpd_tol, min_count):
    m = (sel['hh'] == hh)
    if np.isfinite(rg):
        m &= (np.abs(sel['rg'] - rg) <= rg_tol)
    if np.isfinite(ta):
        m &= (np.abs(sel['ta'] - ta) <= ta_tol)
    if np.isfinite(vpd):
        m &= (np.abs(sel['vpd'] - vpd) <= vpd_tol)
    cand = sel[m]
    if len(cand) >= min_count:
        return np.median(cand['nee'])
    return np.nan

def mds_fill_nee(nee: pd.Series, rg: pd.Series, ta: pd.Series, vpd: pd.Series,
                 windows=(7, 14, 28), min_count=4) -> pd.Series:
    df = pd.DataFrame({'nee': nee, 'rg': rg, 'ta': ta, 'vpd': vpd}).copy()
    df['hh'] = halfhour_bin(df.index)
    filled = df['nee'].copy()
    method = pd.Series(index=df.index, dtype=int)
    # precompute monthly clim and MDC maps
    clim = _monthly_climatology(df['nee'])
    # MDC: mean per hh within window will be computed on-the-fly
    gaps = df['nee'].isna().values
    for i, isgap in enumerate(gaps):
        if not isgap:
            method.iloc[i] = 0
            continue
        ts = df.index[i]
        hh = df['hh'].iloc[i]
        rg_i = df['rg'].iloc[i]
        ta_i = df['ta'].iloc[i]
        vpd_i = df['vpd'].iloc[i]
        val = np.nan
        # step 1-2: similar conditions in expanding windows with dynamic tolerances
        found = False
        for k, wnd in enumerate(windows):
            lo = ts - pd.Timedelta(days=wnd)
            hi = ts + pd.Timedelta(days=wnd)
            sel = df.loc[lo:hi].dropna()
            if sel.empty:
                continue
            rg_tol, ta_tol, vpd_tol = _dynamic_tols(sel)
            val = _select_similar(sel, hh, rg_i, ta_i, vpd_i, rg_tol, ta_tol, vpd_tol, min_count)
            if np.isfinite(val):
                method.iloc[i] = 1 if k == 0 else 2
                found = True
                break
        if not found:
            # step 3: MDC in ±7/14/28
            for wnd in windows:
                lo = ts - pd.Timedelta(days=wnd)
                hi = ts + pd.Timedelta(days=wnd)
                seg = df.loc[lo:hi]['nee'].dropna()
                if seg.empty:
                    continue
                by_hh = seg.groupby(halfhour_bin(seg.index)).mean()
                if hh in by_hh.index and np.isfinite(by_hh.loc[hh]):
                    val = by_hh.loc[hh]
                    method.iloc[i] = 3
                    found = True
                    break
        if not found:
            # step 4: monthly climatology (same month, hh)
            key = (ts.month, ts.hour, ts.minute)
            # We created a Series with the same index; take mean over same (month, hh) grouping
            val = clim.iloc[i]
            if np.isfinite(val):
                method.iloc[i] = 4
                found = True
        if not found:
            # step 5: last resort — local window mean of all finite NEE
            lo = ts - pd.Timedelta(days=30)
            hi = ts + pd.Timedelta(days=30)
            seg = df.loc[lo:hi]['nee'].dropna()
            val = seg.mean() if not seg.empty else np.nan
            method.iloc[i] = 5
        filled.iloc[i] = val
    filled.attrs['mds_method_codes'] = METHOD_CODES
    return filled

def mds_fill_met(series: pd.Series, windows=(7, 14, 28)) -> pd.Series:
    s = series.copy()
    idx = s.index
    # First attempt MDC
    hh = halfhour_bin(idx)
    out = s.copy()
    gaps = out.isna().values
    for i, isgap in enumerate(gaps):
        if not isgap:
            continue
        ts = idx[i]
        bin_i = hh[i]
        val = np.nan
        for wnd in windows:
            lo = ts - pd.Timedelta(days=wnd)
            hi = ts + pd.Timedelta(days=wnd)
            seg = s.loc[lo:hi].dropna()
            if seg.empty:
                continue
            by_bin = seg.groupby(halfhour_bin(seg.index)).mean()
            if bin_i in by_bin.index and np.isfinite(by_bin.loc[bin_i]):
                val = by_bin.loc[bin_i]
                break
        if not np.isfinite(val):
            # monthly climatology fallback
            grp = s.groupby([s.index.month, s.index.hour, s.index.minute]).transform('mean')
            val = grp.iloc[i]
        if not np.isfinite(val):
            # broader mean
            lo = ts - pd.Timedelta(days=30)
            hi = ts + pd.Timedelta(days=30)
            seg = s.loc[lo:hi].dropna()
            val = seg.mean() if not seg.empty else np.nan
        out.iloc[i] = val
    return out
