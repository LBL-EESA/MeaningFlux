import numpy as np
import pandas as pd

def infer_minutes_per_step(index: pd.DatetimeIndex) -> int:
    """Infer minutes per step from the most common diff (round to int)."""
    if len(index) < 2:
        return 30
    diffs = pd.Series(index[1:] - index[:-1]).dt.total_seconds() / 60.0
    mode = diffs.round().mode()
    if len(mode) == 0:
        return int(round(diffs.median()))
    return int(mode.iloc[0])

def steps_per_day(index: pd.DatetimeIndex) -> int:
    mps = infer_minutes_per_step(index)
    return int(round(1440.0 / mps))

def halfhour_bin(index: pd.DatetimeIndex) -> np.ndarray:
    """Return within-day bin index based on inferred step length (0..n-1)."""
    step = infer_minutes_per_step(index)
    mins = index.hour * 60 + index.minute
    return (mins // step).astype(int)

def doy(index: pd.DatetimeIndex) -> np.ndarray:
    return index.dayofyear.values

def mad(x):
    med = np.nanmedian(x)
    return np.nanmedian(np.abs(x - med)) * 1.4826  # consistent with std for normal

def clip_positive(x, eps=1e-6):
    return np.where(x > eps, x, eps)
