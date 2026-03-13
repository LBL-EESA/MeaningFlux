# oneflux_py3/vpd_tools.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Tuple

__all__ = ["es_hpa", "estimate_vpd_hpa", "ensure_vpd_series"]

def es_hpa(Tc: pd.Series | np.ndarray | float) -> pd.Series:
    T = pd.to_numeric(pd.Series(Tc), errors="coerce")
    return 6.112 * np.exp(17.62 * T / (243.12 + T))

def estimate_vpd_hpa(ta_c: pd.Series, rh_pct: Optional[pd.Series] = None,
                     tdew_c: Optional[pd.Series] = None, method: Optional[str] = None) -> pd.Series:
    T = pd.to_numeric(ta_c, errors="coerce")
    m = (method or "").upper().strip()
    if (m == "TA+RH") or (not m and rh_pct is not None):
        es = es_hpa(T); rh = pd.to_numeric(rh_pct, errors="coerce") / 100.0 if rh_pct is not None else np.nan
        vpd = es * (1.0 - rh)
    elif (m == "TA+TDEW") or (not m and tdew_c is not None):
        es  = es_hpa(T); esd = es_hpa(pd.to_numeric(tdew_c, errors="coerce")); vpd = es - esd
    else:
        raise ValueError("estimate_vpd_hpa: need RH% or Tdew °C (or set method).")
    return pd.Series(np.clip(vpd.values, 0.0, None), index=T.index, name="VPD")

def ensure_vpd_series(df: pd.DataFrame, ta_col: str, vpd_col: Optional[str] = None,
                      rh_col: Optional[str] = None, tdew_col: Optional[str] = None,
                      method: str = "TA+RH", out_col: str = "VPD",
                      persist_estimate_as: str = "VPD_est_hPa", inplace: bool = True) -> Tuple[pd.DataFrame, str]:
    _df = df if inplace else df.copy()
    if vpd_col and vpd_col in _df.columns:
        _df[out_col] = pd.to_numeric(_df[vpd_col], errors="coerce")
        return _df, out_col
    m = (method or "TA+RH").upper()
    have_rh = bool(rh_col) and (rh_col in _df.columns)
    have_td = bool(tdew_col) and (tdew_col in _df.columns)
    if m == "TA+RH" and have_rh:
        vpd = estimate_vpd_hpa(_df[ta_col], rh_pct=_df[rh_col], method="TA+RH")
    elif m == "TA+TDEW" and have_td:
        vpd = estimate_vpd_hpa(_df[ta_col], tdew_c=_df[tdew_col], method="TA+Tdew")
    elif have_rh:
        vpd = estimate_vpd_hpa(_df[ta_col], rh_pct=_df[rh_col], method="TA+RH")
    elif have_td:
        vpd = estimate_vpd_hpa(_df[ta_col], tdew_c=_df[tdew_col], method="TA+Tdew")
    else:
        raise ValueError("ensure_vpd_series: no VPD column and cannot estimate (need RH% or Tdew °C).")
    _df[out_col] = vpd
    if persist_estimate_as and persist_estimate_as != out_col:
        _df[persist_estimate_as] = vpd.copy()
    return _df, out_col
