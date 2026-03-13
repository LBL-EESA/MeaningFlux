"""Daytime partitioning scaffold (Lasslop et al. 2010 style).
This module provides an interface-compatible function but raises NotImplementedError
by default. You can plug in a custom solver (e.g., scipy.optimize) to fit the
rectangular-hyperbola light-response with VPD effects.
"""
import pandas as pd

def partition_nee_daytime(nee: pd.Series, rg: pd.Series, vpd: pd.Series, ta: pd.Series):
    raise NotImplementedError("Daytime partitioning is scaffolded only. Use nighttime method or add a solver.")
