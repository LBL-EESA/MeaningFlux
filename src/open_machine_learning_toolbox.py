#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MeaningFlux — Machine Learning Toolbox 
Author: Leila C. Hernandez (LBNL)
"""

import time, math, queue, threading, re, platform
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
from tkinter import (
    Frame, Label, Button, Listbox, OptionMenu, StringVar, IntVar,
    Radiobutton, MULTIPLE, messagebox
)

# --- matplotlib (Tk backend) ---
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- sklearn ---
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

# --- keras (optional) ---
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Input
except Exception:
    tf = None
    Sequential = None
    Dense = None
    LSTM = None
    Input = None

# --- torch (optional) ---
_HAS_TORCH = False
try:
    import torch
    from torch import nn
    _HAS_TORCH = True
except Exception:
    torch = None
    nn = None

# The GUI already runs model fitting in a background thread. Restricting
# PyTorch's internal CPU pools avoids nested-thread oversubscription and makes
# H-LSTM training responsive and reproducible on laptops and workstations.
if _HAS_TORCH:
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

# --- external Sequence class (optional, NOT used for H-LSTM now) ---
_HAS_EXT_SEQUENCE = False
try:
    from lstm_classes import Sequence
    _HAS_EXT_SEQUENCE = True
except Exception:
    Sequence = None

_stop_event = threading.Event()
_GLOBAL_SEED = 42
_MIN_RESAMPLE_COVERAGE = 0.75
_rng = np.random.RandomState(_GLOBAL_SEED)

def _set_reproducible_seed(seed=_GLOBAL_SEED):
    """Set NumPy, TensorFlow, and PyTorch seeds when those libraries are available."""
    seed = int(seed)
    np.random.seed(seed)
    if tf is not None:
        try:
            tf.keras.utils.set_random_seed(seed)
        except Exception:
            try:
                tf.random.set_seed(seed)
            except Exception:
                pass
    if _HAS_TORCH:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
        except Exception:
            pass

_set_reproducible_seed(_GLOBAL_SEED)

# =============================================================================
#                               SIMPLE THEME
# =============================================================================
def set_simple_theme(root, accent="#0F8C8C"):
    """Lightweight modern theme."""
    from tkinter import ttk
    import matplotlib as mpl
    s = ttk.Style(root)
    try:
        s.theme_use("clam")
    except Exception:
        pass
    s.configure(".", background="#FFFFFF", foreground="#1D2B34")
    s.configure("TFrame", background="#FFFFFF")
    s.configure("TLabelframe", background="#F6FBFD", bordercolor="#D2EEF2")
    s.configure("TLabelframe.Label", background="#F6FBFD")
    s.configure("TNotebook", background="#FFFFFF")
    s.configure("TNotebook.Tab", background="#F6FBFD", padding=(6,4))
    s.map("TNotebook.Tab", background=[("selected", "#E9F6F8")])
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#CCCCCC",
        "axes.labelcolor": "#1D2B34",
        "text.color": "#1D2B34",
        "xtick.color": "#536873",
        "ytick.color": "#536873",
        "grid.color": "#E5ECEF",
        "grid.linestyle": "-",
        "grid.alpha": 0.35,
    })


# =============================================================================
#                         CONCEPTUAL QUESTION BANNERS
# =============================================================================
def _concept_question(parent, question, details=None, width=380):
    """Add a small conceptual-question banner to a tab/control panel."""
    box = ttk.LabelFrame(parent, text="Question")
    box.grid(sticky="ew", pady=(0, 8))
    box.columnconfigure(0, weight=1)
    tk.Message(
        box,
        width=width,
        text=question,
        fg="#0B4F9C",
        font=("TkDefaultFont", 9, "bold"),
    ).grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 2))
    if details:
        tk.Message(
            box,
            width=width,
            text=details,
            fg="#444444",
        ).grid(row=1, column=0, sticky="ew", padx=6, pady=(0, 6))
    return box


def _scrollable_controls(splitter, width=410, padding=(10, 10)):
    """Create a compact, scrollable left control sidebar for long GUI workflows.

    This keeps the main MeaningFlux ML window usable on smaller screens while
    preserving all workflow steps and explanatory notes.
    """
    outer = ttk.Frame(splitter)
    outer.columnconfigure(0, weight=1)
    outer.rowconfigure(0, weight=1)

    canvas = tk.Canvas(
        outer,
        width=width,
        borderwidth=0,
        highlightthickness=0,
        background="#FFFFFF",
    )
    vscroll = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vscroll.set)

    canvas.grid(row=0, column=0, sticky="nsew")
    vscroll.grid(row=0, column=1, sticky="ns")

    controls = ttk.Frame(canvas, padding=padding)
    controls.columnconfigure(0, weight=1)
    window_id = canvas.create_window((0, 0), window=controls, anchor="nw")

    def _refresh_scrollregion(_event=None):
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _fit_inner_width(event):
        canvas.itemconfigure(window_id, width=event.width)

    controls.bind("<Configure>", _refresh_scrollregion)
    canvas.bind("<Configure>", _fit_inner_width)

    def _on_mousewheel(event):
        # Windows/Linux trackpads often use +/-120; macOS can use smaller deltas.
        delta = event.delta
        if delta == 0:
            return
        step = -1 if delta > 0 else 1
        canvas.yview_scroll(step, "units")

    def _bind_wheel(_event=None):
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

    def _unbind_wheel(_event=None):
        canvas.unbind_all("<MouseWheel>")

    outer.bind("<Enter>", _bind_wheel)
    outer.bind("<Leave>", _unbind_wheel)

    splitter.add(outer, weight=0)
    return controls

# =============================================================================
#                               UTILITIES
# =============================================================================


def _software_versions():
    """Return a compact reproducibility record for the active Python environment."""
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import sklearn
        versions["scikit_learn"] = sklearn.__version__
    except Exception:
        versions["scikit_learn"] = None
    versions["numpy"] = np.__version__
    versions["pandas"] = pd.__version__
    try:
        versions["tensorflow"] = tf.__version__ if tf is not None else None
    except Exception:
        versions["tensorflow"] = None
    try:
        versions["torch"] = torch.__version__ if _HAS_TORCH else None
    except Exception:
        versions["torch"] = None
    return versions


def _sorted_unique_columns(df_or_columns):
    """Return unique column names sorted alphabetically for easier GUI search."""
    cols_in = list(df_or_columns.columns) if hasattr(df_or_columns, "columns") else list(df_or_columns)
    seen = set()
    cols = []
    for c in cols_in:
        if c not in seen:
            cols.append(c)
            seen.add(c)
    return sorted(cols, key=lambda c: str(c).lower())

def _is_main_thread():
    return threading.current_thread() is threading.main_thread()

def ensure_on_main(widget, fn, *args, **kwargs):
    if _is_main_thread():
        return fn(*args, **kwargs)
    widget.after(0, lambda: fn(*args, **kwargs))

def on_main(widget, fn, *args, **kwargs):
    widget.after(0, lambda: fn(*args, **kwargs))


def on_main_sync(widget, fn, *args, **kwargs):
    """Run a Tk/matplotlib callback on the main thread and wait for completion."""
    if _is_main_thread():
        return fn(*args, **kwargs)
    done = threading.Event()
    result = {"value": None, "error": None}
    def _wrapped():
        try:
            result["value"] = fn(*args, **kwargs)
        except Exception as exc:
            result["error"] = exc
        finally:
            done.set()
    widget.after(0, _wrapped)
    done.wait()
    if result["error"] is not None:
        raise result["error"]
    return result["value"]

def _to_datetime_1d(obj):
    """
    Safely convert to datetime, even if `obj` is a DataFrame
    (e.g., duplicate timestamp columns). Always returns a 1-D Series-like.
    """
    if isinstance(obj, pd.DataFrame):
        obj = obj.iloc[:, 0]
    return pd.to_datetime(obj, errors="coerce")

def _is_accumulated_variable(column):
    """Return True for interval totals that should be summed during resampling.

    The rule is intentionally conservative. It recognizes common EC precipitation,
    irrigation, and fertilizer-amount names while avoiding atmospheric pressure (PA).
    """
    name = str(column).upper()
    parts = _split_name_parts(name) if "_split_name_parts" in globals() else [p for p in re.split(r"[^A-Z0-9]+", name) if p]
    compact = "".join(parts)
    # Explicit rate/intensity columns must be averaged, not summed. MeaningFlux
    # assumes the standard EC precipitation variable P is an interval amount.
    if any(token in parts for token in ("RATE", "INTENSITY", "FLUX")):
        return False
    exact = {"P", "PRECIP", "PRECIPITATION", "PREC", "PPT", "RAIN", "RAINFALL"}
    if name in exact or (parts and parts[0] in exact):
        return True
    tokens = ("PRECIP", "RAINFALL", "RAIN_MM", "IRRIG", "FERTILIZER_AMOUNT", "FERT_AMOUNT", "N_FERT")
    return any(tok in name or tok.replace("_", "") in compact for tok in tokens)


def _resampling_method_for_column(column, sum_columns=None, mean_columns=None):
    """Return the aggregation method, honoring explicit user overrides."""
    name = str(column)
    sum_set = {str(c) for c in (sum_columns or [])}
    mean_set = {str(c) for c in (mean_columns or [])}
    if name in mean_set:
        return "mean"
    if name in sum_set:
        return "sum"
    return "sum" if _is_accumulated_variable(column) else "mean"


def _parse_column_overrides(text, available_columns):
    """Parse comma-separated aggregation overrides and validate column names."""
    raw = [item.strip() for item in str(text or "").split(",") if item.strip()]
    available = {str(c) for c in available_columns}
    unknown = [c for c in raw if c not in available]
    if unknown:
        raise ValueError("Unknown aggregation override column(s): " + ", ".join(unknown))
    return list(dict.fromkeys(raw))


def _resample_interval_seconds(interval):
    """Return nominal bin duration in seconds for supported resampling labels."""
    text = str(interval).strip().upper()
    match = re.fullmatch(r"(\d+)?\s*([DWH])", text)
    if match:
        mult = int(match.group(1) or 1)
        unit = match.group(2)
        return float(mult * {"H": 3600, "D": 86400, "W": 604800}[unit])
    try:
        return float(pd.to_timedelta(interval).total_seconds())
    except Exception:
        return np.nan


def _required_resample_count(series, interval, min_coverage=_MIN_RESAMPLE_COVERAGE):
    """Infer the minimum nonmissing samples needed for one aggregate value.

    Expected sampling frequency is estimated separately for each variable from
    its median positive time step. This accommodates mixed-frequency columns
    while preventing a daily/weekly statistic from being based on a few isolated
    native observations.
    """
    valid_times = pd.DatetimeIndex(series.index[series.notna()])
    if len(valid_times) < 2:
        return 1
    deltas = np.diff(valid_times.asi8) / 1e9
    deltas = deltas[np.isfinite(deltas) & (deltas > 0)]
    bin_seconds = _resample_interval_seconds(interval)
    if deltas.size == 0 or not np.isfinite(bin_seconds) or bin_seconds <= 0:
        return 1
    cadence = float(np.nanmedian(deltas))
    expected = max(1, int(round(bin_seconds / cadence)))
    return max(1, int(math.ceil(float(min_coverage) * expected)))


def _resample_df(df, interval, ts_col, min_coverage=_MIN_RESAMPLE_COVERAGE,
                 sum_columns=None, mean_columns=None, drop_complete_cases=True,
                 coverage_relative_to_available_rows=False):
    """Resample with explicit aggregation and a completeness threshold.

    State and rate variables are averaged. Detected interval accumulations are
    summed unless the user forces them to mean. Set ``drop_complete_cases=False``
    when building coverage reports so variable-specific missingness is preserved.

    ``coverage_relative_to_available_rows`` is used for optional daytime/nighttime
    sensitivity analyses. After filtering to daylight or darkness, the expected
    number of records varies seasonally. In that mode, each variable must be
    present for the requested fraction of the retained subdaily timestamps in
    that day/week, rather than for the same fraction of a full 24-hour interval.
    """
    df = df.copy()
    df[ts_col] = _to_datetime_1d(df[ts_col])
    df = df.dropna(subset=[ts_col]).set_index(ts_col).sort_index()
    numeric_cols = list(df.columns)
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    grouped = df.resample(interval)
    available_rows = grouped.size()
    pieces = {}
    for c in numeric_cols:
        method = _resampling_method_for_column(
            c, sum_columns=sum_columns, mean_columns=mean_columns
        )
        aggregated = grouped[c].sum(min_count=1) if method == "sum" else grouped[c].mean()
        observed = grouped[c].count()
        if coverage_relative_to_available_rows:
            required = np.ceil(float(min_coverage) * available_rows).clip(lower=1).astype(int)
            pieces[c] = aggregated.where(observed >= required)
        else:
            required = _required_resample_count(df[c], interval, min_coverage=min_coverage)
            pieces[c] = aggregated.where(observed >= required)
    out = pd.DataFrame(pieces)
    return out.dropna(how="any") if drop_complete_cases else out


def _reporting_frame(df, columns, ts_col, rs_mode, min_coverage,
                     sum_columns=None, mean_columns=None,
                     coverage_relative_to_available_rows=False):
    """Return the period-filtered analysis table without complete-case deletion."""
    cols = list(dict.fromkeys(list(columns)))
    base = df[cols + [ts_col]].copy()
    base[ts_col] = _to_datetime_1d(base[ts_col])
    base = base.dropna(subset=[ts_col])
    if rs_mode == 1:
        return _resample_df(
            base, "D", ts_col, min_coverage=min_coverage,
            sum_columns=sum_columns, mean_columns=mean_columns,
            drop_complete_cases=False,
            coverage_relative_to_available_rows=coverage_relative_to_available_rows,
        )
    if rs_mode == 2:
        return _resample_df(
            base, "W", ts_col, min_coverage=min_coverage,
            sum_columns=sum_columns, mean_columns=mean_columns,
            drop_complete_cases=False,
            coverage_relative_to_available_rows=coverage_relative_to_available_rows,
        )
    base = base.set_index(ts_col).sort_index()
    for c in cols:
        base[c] = pd.to_numeric(base[c], errors="coerce")
    return base.replace([np.inf, -np.inf], np.nan)


def _infer_expected_timestep(index):
    """Infer the dominant positive timestep from a datetime index."""
    idx = pd.DatetimeIndex(pd.to_datetime(index, errors="coerce")).dropna().sort_values().unique()
    if len(idx) < 2:
        return pd.NaT
    deltas = pd.Series(pd.to_timedelta(np.diff(idx.asi8), unit="ns"))
    deltas = deltas[deltas > pd.Timedelta(0)]
    if deltas.empty:
        return pd.NaT
    mode = deltas.mode()
    return pd.Timedelta(mode.iloc[0] if not mode.empty else deltas.median())


def _timestamp_gap_diagnostics(index, expected_step=None):
    """Summarize gaps relative to the expected timestep."""
    idx = pd.DatetimeIndex(pd.to_datetime(index, errors="coerce")).dropna().sort_values().unique()
    step = pd.Timedelta(expected_step) if expected_step is not None and not pd.isna(expected_step) else _infer_expected_timestep(idx)
    out = {
        "expected_timestep": str(step) if not pd.isna(step) else "not available",
        "n_gaps": 0,
        "max_interval": "not available",
        "max_interval_days": np.nan,
    }
    if len(idx) < 2 or pd.isna(step):
        return out
    diffs = pd.to_timedelta(np.diff(idx.asi8), unit="ns")
    tolerance = max(pd.Timedelta(seconds=1), step * 1e-6)
    gap_mask = diffs > (step + tolerance)
    max_interval = pd.Timedelta(diffs.max()) if len(diffs) else pd.NaT
    out.update({
        "n_gaps": int(np.sum(gap_mask)),
        "max_interval": str(max_interval) if not pd.isna(max_interval) else "not available",
        "max_interval_days": (float(max_interval / pd.Timedelta(days=1)) if not pd.isna(max_interval) else np.nan),
    })
    return out


def _variable_coverage_summary(frame, columns, target, gate_variable=None,
                               sum_columns=None, mean_columns=None):
    """Create one concise coverage row per modeled variable."""
    rows = []
    n_total = int(len(frame))
    expected_step = _infer_expected_timestep(frame.index)
    for c in columns:
        series = pd.to_numeric(frame[c], errors="coerce") if c in frame.columns else pd.Series(dtype=float)
        valid = series.dropna()
        gaps = _timestamp_gap_diagnostics(valid.index, expected_step=expected_step)
        role = "target" if c == target else "predictor"
        if gate_variable and c == gate_variable:
            role = "predictor; H-LSTM gate"
        rows.append({
            "variable": c,
            "role": role,
            "aggregation": _resampling_method_for_column(
                c, sum_columns=sum_columns, mean_columns=mean_columns
            ),
            "n_period_rows": n_total,
            "n_valid": int(valid.shape[0]),
            "n_missing": int(max(0, n_total - valid.shape[0])),
            "valid_percent": (100.0 * valid.shape[0] / n_total if n_total else np.nan),
            "first_valid": valid.index.min() if not valid.empty else pd.NaT,
            "last_valid": valid.index.max() if not valid.empty else pd.NaT,
            "n_gaps": gaps["n_gaps"],
            "max_interval_days": gaps["max_interval_days"],
        })
    return pd.DataFrame(rows)

def _resampling_summary(columns, min_coverage=_MIN_RESAMPLE_COVERAGE,
                        sum_columns=None, mean_columns=None):
    summed = [str(c) for c in columns
              if _resampling_method_for_column(c, sum_columns=sum_columns, mean_columns=mean_columns) == "sum"]
    coverage = f"minimum {100 * float(min_coverage):.0f}% inferred within-period coverage"
    if summed:
        return "Mean for state/rate variables; sum for: " + ", ".join(summed) + f"; {coverage}."
    return f"Mean for selected state/rate variables; no accumulated variables detected; {coverage}."


def _metric_bundle(y_true, y_pred):
    """Return original-unit test metrics plus robust normalized RMSE.

    nRMSE is RMSE divided by the observed 5th-95th percentile range, expressed
    as percent. This remains meaningful for fluxes that cross zero and is less
    sensitive to isolated extremes than max-min normalization.
    """
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    valid = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[valid], yp[valid]
    if yt.size == 0:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "nrmse_p5_p95": np.nan, "n": 0,
                "observed_p05": np.nan, "observed_p95": np.nan}
    rmse = math.sqrt(mean_squared_error(yt, yp))
    mae = mean_absolute_error(yt, yp)
    r2 = r2_score(yt, yp) if yt.size >= 2 else np.nan
    p05, p95 = np.nanpercentile(yt, [5, 95])
    robust_range = float(p95 - p05)
    nrmse = 100.0 * rmse / robust_range if np.isfinite(robust_range) and robust_range > 0 else np.nan
    return {
        "rmse": float(rmse), "mae": float(mae), "r2": float(r2) if np.isfinite(r2) else np.nan,
        "nrmse_p5_p95": float(nrmse) if np.isfinite(nrmse) else np.nan,
        "n": int(yt.size), "observed_p05": float(p05), "observed_p95": float(p95),
    }


def _metrics(y_true, y_pred):
    m = _metric_bundle(y_true, y_pred)
    return m["rmse"], m["mae"], m["r2"]


def _nrmse_with_reference_range(rmse, observed_p05, observed_p95):
    """Normalize RMSE with one predefined observed P5-P95 range.

    Figure 3B fold intervals must use the same site-level denominator; otherwise
    fold-to-fold changes in observed variability are confounded with changes in
    prediction error.
    """
    robust_range = float(observed_p95) - float(observed_p05)
    if not np.isfinite(rmse) or not np.isfinite(robust_range) or robust_range <= 0:
        return np.nan
    return float(100.0 * float(rmse) / robust_range)


def _fold_metric_record(fold, y_true, y_pred, train_idx=None, test_idx=None,
                        timestamps=None, all_timestamps=None,
                        sequence_diagnostics=None):
    """Return fold metrics plus exact train/test periods and gap-window counts."""
    m = _metric_bundle(y_true, y_pred)
    out = {"fold": int(fold), **m}
    out["n_train"] = int(len(train_idx)) if train_idx is not None else np.nan
    out["n_test"] = int(len(test_idx)) if test_idx is not None else int(m["n"])

    train_ts = pd.DatetimeIndex([])
    if all_timestamps is not None and train_idx is not None:
        try:
            all_ts = pd.DatetimeIndex(pd.to_datetime(all_timestamps, errors="coerce"))
            train_ts = all_ts[np.asarray(train_idx, dtype=int)].dropna()
        except Exception:
            train_ts = pd.DatetimeIndex([])
    test_ts = pd.DatetimeIndex([])
    if timestamps is not None:
        try:
            test_ts = pd.DatetimeIndex(pd.to_datetime(timestamps, errors="coerce")).dropna()
        except Exception:
            test_ts = pd.DatetimeIndex([])

    out["train_start"] = train_ts.min() if len(train_ts) else pd.NaT
    out["train_end"] = train_ts.max() if len(train_ts) else pd.NaT
    out["test_start"] = test_ts.min() if len(test_ts) else pd.NaT
    out["test_end"] = test_ts.max() if len(test_ts) else pd.NaT

    rejected = pd.DatetimeIndex([])
    if sequence_diagnostics:
        try:
            rejected = pd.DatetimeIndex(sequence_diagnostics.get("rejected_timestamps", [])).dropna()
        except Exception:
            rejected = pd.DatetimeIndex([])
    out["n_rejected_gap_windows_train"] = (
        int(((rejected >= train_ts.min()) & (rejected <= train_ts.max())).sum())
        if len(rejected) and len(train_ts) else 0
    )
    out["n_rejected_gap_windows_test"] = (
        int(((rejected >= test_ts.min()) & (rejected <= test_ts.max())).sum())
        if len(rejected) and len(test_ts) else 0
    )
    return out

def _plain_model_name(model_name):
    """Short user-facing label for model names."""
    txt = str(model_name)
    if "Random Forest" in txt:
        return "Random Forest"
    if "Linear" in txt:
        return "Linear Regression"
    if "Hysteresis" in txt:
        return "H-LSTM"
    if "LSTM" in txt:
        return "LSTM"
    if "MLP" in txt or "Neural" in txt:
        return "MLP"
    return txt

def _comparison_run_label(run, index=None):
    """Return a compact unique label, including settings for repeated models."""
    base = _plain_model_name(run.get("model", "model"))
    params = run.get("model_parameters") or {}
    details = []
    if base in ("LSTM", "H-LSTM"):
        if run.get("sequence_length") is not None:
            details.append(f"seq={run.get('sequence_length')}")
        if run.get("hidden_size") is not None:
            details.append(f"h={run.get('hidden_size')}")
        if base == "H-LSTM" and run.get("gate_variable"):
            details.append(f"gate={run.get('gate_variable')}")
    elif base == "Random Forest":
        if params.get("n_estimators") is not None:
            details.append(f"trees={params.get('n_estimators')}")
        if params.get("min_samples_leaf") is not None:
            details.append(f"leaf={params.get('min_samples_leaf')}")
    elif base == "MLP":
        if params.get("hidden_units") is not None:
            details.append(f"h={params.get('hidden_units')}")
        elif run.get("hidden_size") is not None:
            details.append(f"h={run.get('hidden_size')}")
    diurnal = str(run.get("diurnal_mode") or "All observations")
    if not diurnal.lower().startswith("all"):
        details.append(diurnal)
    prefix = f"Run {int(index)+1}: " if index is not None else ""
    return prefix + base + (" [" + ", ".join(details) + "]" if details else "")


def _interpret_ml_result(model_name, target, features, rmse, mae, r2, resample_label, nrmse=np.nan):
    """Create plain-language interpretation for the latest ML result."""
    name = _plain_model_name(model_name)
    features_txt = ", ".join(map(str, features[:8])) + (f", ... (+{len(features)-8})" if len(features) > 8 else "")
    if np.isfinite(r2):
        if r2 >= 0.80:
            skill = "high predictive skill"
        elif r2 >= 0.50:
            skill = "moderate predictive skill"
        elif r2 >= 0.20:
            skill = "weak-to-moderate predictive skill"
        elif r2 >= 0:
            skill = "weak predictive skill"
        else:
            skill = "poor predictive skill; the mean of the observations may outperform this model"
    else:
        skill = "predictive skill could not be assessed"

    return (
        f"Latest model: {name}\n"
        f"Target flux/variable: {target}\n"
        f"Predictors used: {features_txt or 'none'}\n"
        f"Temporal aggregation: {resample_label}\n\n"
        f"Predictive interpretation:\n"
        f"• R² = {r2:.4g}. This indicates {skill}.\n"
        f"• RMSE = {rmse:.4g}; MAE = {mae:.4g}. These are in the original target units.\n"
        f"• nRMSE (5th–95th percentile range) = {nrmse:.3g}% and can be compared across different flux targets.\n"
        f"• The residual plot shows whether errors are centered near zero or biased. The scatter plot shows whether predictions follow the 1:1 line.\n\n"
        f"How to connect this to Information Theory:\n"
        f"• A good R² only means the model predicts the target well. It does not prove that the model uses the drivers in the same way as the observed flux.\n"
        f"• Use Compare → Export for IT to create an aligned table containing drivers, observed {target}, model predictions, and residuals.\n"
        f"• In the Information Theory toolbox, run Model-vs-Observed MI and Model-vs-Observed PID Matrix to test whether the model preserves observed driver–flux dependence, redundancy, uniqueness, and synergy.\n"
    )

def _ml_workflow_guide_text():
    return (
        "MeaningFlux Machine Learning Toolbox — guided workflow\n\n"
        "Step 1 — Choose the model and its parameters.\n"
        "Select Linear Regression, Random Forest, MLP, LSTM, or H-LSTM first. The GUI then shows only parameters that the selected method actually uses. Keep the same settings across sites for a controlled manuscript comparison. The random seed is user-controlled and recorded in exports.\n\n"
        "Step 2 — Choose the target flux or variable.\n"
        "Examples: FC, FCH4, FN2O, LE, H, GPP, or RECO. The target defines the prediction problem and activates a literature-informed predictor preset.\n\n"
        "Step 3 — Define the analysis period and temporal aggregation.\n"
        "Choose the timestamp, record limits, and Native, Daily, or Weekly resolution. Review the minimum within-period coverage and aggregation overrides. State/rate variables are averaged; interval precipitation and management amounts are summed.\n\n"
        "Step 4 — Review the predictor pool.\n"
        "Remove the target family, gap-filled target variants, QC/uncertainty fields, residuals, and prediction columns. For model comparison, use the same ordered predictor list for every model within a site. The H-LSTM gate must also be in that common list.\n\n"
        "Step 5 — Define temporal validation.\n"
        "Blocked expanding-window validation is recommended. Choose the number of future folds and the initial training fraction. Chronological holdout is a simpler alternative; random splits are exploratory because they mix time periods and can inflate skill for autocorrelated EC data.\n\n"
        "Model parameters exposed in Step 1\n"
        "• Linear Regression: fit intercept.\n"
        "• Random Forest: trees, minimum leaf size, maximum depth, maximum features, and held-out permutation repeats.\n"
        "• MLP: hidden units, maximum iterations, L2 alpha, and learning rate. Random internal early stopping is intentionally disabled because it would not preserve time order.\n"
        "• LSTM: sequence length, hidden units, epochs per fold, batch size, learning rate, and dropout.\n"
        "• LSTM and H-LSTM: sequence length, hidden units, epochs, batch size, learning rate, dropout, and gradient clipping are shared. H-LSTM uses the same Keras model and differs only by the added Δgate input channel.\n\n"
        "Step 6 — Read the results together.\n"
        "Use Metrics, Series, Scatter, Residuals, Feature Importance, Training, and Interpretation. Metrics are calculated only from held-out/out-of-fold predictions in original target units.\n\n"
        "Step 7 — Compare and export.\n"
        "Use Export Fig 3B results to align all five models on identical held-out timestamps. Use Save ML–IT bridge or Open in IT to evaluate whether model predictions preserve observed driver–flux information structure.\n"
    )


# =============================================================================
#             LITERATURE-INFORMED PREDICTOR PRESETS FOR EC FLUXES
# =============================================================================
def _clean_name_for_matching(name):
    """Uppercase alphanumeric name used for loose EC-variable matching."""
    return "".join(ch for ch in str(name).upper() if ch.isalnum())

def _split_name_parts(name):
    """Split a column name into uppercase parts, preserving EC-style base names."""
    import re
    return [p for p in re.split(r"[^A-Za-z0-9]+", str(name).upper()) if p]

def _looks_like_bad_predictor(col, target=None, timestamp_col=None):
    """Avoid timestamps, QC flags, uncertainty flags, and the target itself."""
    c = str(col)
    cu = c.upper()
    cc = _clean_name_for_matching(c)
    if timestamp_col is not None and c == str(timestamp_col):
        return True
    if target is not None and c == str(target):
        return True
    # Strongly avoid QC / flags / counts / uncertainty columns as predictors.
    bad_suffixes = ("_QC", "_FLAG", "_FQC", "_N", "_SD", "_SE", "_UNC", "_JOINTUNC", "_RANDUNC")
    if any(cu.endswith(s) for s in bad_suffixes):
        return True
    if "QC" in _split_name_parts(c) or "FLAG" in _split_name_parts(c):
        return True
    # Avoid model prediction/residual columns as predictors.
    if any(k in cu for k in ["_PRED", "PRED_", "_MODEL", "MODEL_", "_RESID", "RESID_"]):
        return True
    # Avoid direct target-family leakage: e.g., FC_F, FC_PI, NEE, FCH4_F when target is FC/FCH4.
    if target is not None:
        tfam = _target_family(target)
        parts = _split_name_parts(c)
        if tfam == "CO2 flux / NEE" and any(p in parts for p in ["FC", "NEE"]):
            return True
        if tfam == "Methane flux" and any(p in parts for p in ["FCH4", "CH4"]):
            return True
        if tfam == "Nitrous oxide flux" and any(p in parts for p in ["FN2O", "N2O"]):
            return True
        if tfam == "Latent heat / evapotranspiration" and any(p in parts for p in ["LE", "ET"]):
            return True
        if tfam == "Sensible heat" and parts and parts[0] in ["H", "SH"]:
            return True
        if tfam == "Gross primary productivity" and any(p in parts for p in ["GPP"]):
            return True
        if tfam == "Ecosystem respiration" and any(p in parts for p in ["RECO", "ER", "RESP"]):
            return True
    return False

def _target_family(target):
    """Map a target column to a broad EC flux family."""
    s = _clean_name_for_matching(target)
    parts = _split_name_parts(target)
    # Order matters: FCH4 and FN2O should be checked before FC.
    if "FCH4" in s or ("CH4" in parts) or s.startswith("CH4"):
        return "Methane flux"
    if "FN2O" in s or ("N2O" in parts) or s.startswith("N2O"):
        return "Nitrous oxide flux"
    if s in ["FC", "NEE"] or s.startswith("FC") or s.startswith("NEE"):
        return "CO2 flux / NEE"
    if s.startswith("GPP"):
        return "Gross primary productivity"
    if s.startswith("RECO") or s.startswith("ER") or "RESP" in s:
        return "Ecosystem respiration"
    if s == "LE" or s.startswith("LE") or s.startswith("ET"):
        return "Latent heat / evapotranspiration"
    if s == "H" or s.startswith("H") or s.startswith("SH") or s.startswith("SENSIBLE"):
        return "Sensible heat"
    return "Generic flux / target"

def _guess_default_target(cols, timestamp_col=None):
    """Choose a sensible default flux target so the first user action is target review/selection.

    Preference is given to common EC flux targets rather than the first file column
    (which is often TIMESTAMP). The user can still change this immediately, and
    predictor presets update automatically when the target changes.
    """
    if not cols:
        return ""

    def bad_default(c):
        cu = str(c).upper()
        if timestamp_col is not None and str(c) == str(timestamp_col):
            return True
        if any(k in cu for k in ["TIMESTAMP", "DATE", "TIME"]):
            return True
        if any(cu.endswith(s) for s in ["_QC", "_FLAG", "_N", "_SD", "_SE", "_UNC"]):
            return True
        return False

    # Prefer targets used in the manuscript/demo workflow. Order matters.
    preferred_exact = [
        "FC", "NEE", "FCH4", "CH4", "FN2O", "N2O", "LE", "H", "GPP", "RECO", "ER"
    ]
    for target in preferred_exact:
        for c in cols:
            if bad_default(c):
                continue
            parts = _split_name_parts(c)
            cc = _clean_name_for_matching(c)
            if parts and parts[0] == target:
                return c
            if cc == target:
                return c

    # Then accept broader family matches, but avoid obvious predictors where possible.
    for c in cols:
        if bad_default(c):
            continue
        fam = _target_family(c)
        if fam != "Generic flux / target":
            return c

    # Fallback to the first non-time column.
    for c in cols:
        if not bad_default(c):
            return c
    return cols[0]

_PREDICTOR_ALIASES = {
    "radiation_shortwave": ["SW_IN", "SWIN", "SW_IN_F", "SWIN_F", "PPFD", "PAR", "APAR", "PPFD_IN", "PAR_IN"],
    "radiation_net": ["NETRAD", "NETRAD_F", "RNET", "RN", "NET_RADIATION", "SWCNETRAD"],
    "air_temperature": ["TA", "TA_F", "TA_PI", "T_AIR", "TAIR", "AIR_TEMP", "AIRTEMP", "TEMP_AIR", "AT"],
    "soil_temperature": ["TS", "TS_F", "TS_PI", "T_SOIL", "TSOIL", "SOIL_TEMP", "SOILTEMP", "TG"],
    "vpd": ["VPD", "VPD_F", "VPD_PI", "VAPOR_PRESSURE_DEFICIT"],
    "humidity": ["RH", "RH_F", "RH_PI", "RELATIVE_HUMIDITY", "HUMIDITY"],
    "soil_water": ["SWC", "SWC_F", "SWC_PI", "SWC_AVR", "VWC", "SOIL_WATER", "SOIL_MOISTURE", "SM", "WTD", "WATER_TABLE", "WT"],
    "turbulence": ["USTAR", "USTAR_PI", "U_STAR", "UST", "WS", "WS_F", "WIND_SPEED", "WIND", "WD"],
    "precipitation": ["P", "P_F", "P_1", "PRECIP", "PREC", "PPT", "RAIN", "RAINFALL"],
    "pressure": ["PA", "PA_F", "PRESSURE", "ATM_PRESSURE", "BP", "BARO"],
    "vegetation": ["NDVI", "EVI", "LAI", "FPAR", "FAPAR", "GREENNESS", "CANOPY"],
    "soil_nitrogen": ["NO3", "NH4", "NITRATE", "AMMONIUM", "SOIL_N", "N_FERT", "FERT", "FERTILIZER"],
    "ground_heat": ["G", "G_F", "SHF", "SOIL_HEAT", "GROUND_HEAT"],
}

def _column_matches_alias(col, alias):
    """Loose but safe matching between available columns and EC variable aliases."""
    cu = str(col).upper()
    au = str(alias).upper()
    cc = _clean_name_for_matching(cu)
    aa = _clean_name_for_matching(au)
    parts = _split_name_parts(cu)
    if not aa:
        return False
    # Single-letter aliases are dangerous; require exact first part.
    if len(aa) <= 2:
        return parts and parts[0] == aa
    if cc == aa or cc.startswith(aa):
        return True
    # For common base variables with suffixes like _1_1_1, _F, _PI.
    if parts:
        joined_first_two = "".join(parts[:2])
        if parts[0] == au or joined_first_two == aa:
            return True
    return False

def _find_predictor_columns(cols, alias_group, target=None, timestamp_col=None, max_count=1):
    """Find available columns matching one physical predictor group."""
    aliases = _PREDICTOR_ALIASES.get(alias_group, [])
    found = []
    for col in cols:
        if _looks_like_bad_predictor(col, target=target, timestamp_col=timestamp_col):
            continue
        for a in aliases:
            if _column_matches_alias(col, a):
                if col not in found:
                    found.append(col)
                break
    return found[:max_count]

def _add_group(selected, cols, group, target, timestamp_col, max_count=1):
    for c in _find_predictor_columns(cols, group, target=target, timestamp_col=timestamp_col, max_count=max_count):
        if c not in selected:
            selected.append(c)

def _recommended_predictors_for_target(cols, target, timestamp_col=None):
    """
    Return literature-informed predictor suggestions and a user-facing rationale.

    These are intentionally conservative presets. They select common EC drivers
    likely to be available in AmeriFlux-style files, but users should always review
    the final list for site-specific variables and leakage.
    """
    family = _target_family(target)
    selected = []

    if family == "CO2 flux / NEE":
        # Photosynthesis/respiration controls: light, temperature, atmospheric demand, soil state, turbulence.
        for group, maxn in [
            ("radiation_shortwave", 1), ("radiation_net", 1), ("air_temperature", 1),
            ("vpd", 1), ("humidity", 1), ("soil_temperature", 1), ("soil_water", 1),
            ("turbulence", 2), ("precipitation", 1), ("vegetation", 1)
        ]:
            _add_group(selected, cols, group, target, timestamp_col, maxn)
        rationale = (
            "CO₂ flux / NEE is commonly driven by light availability, canopy/air temperature, atmospheric demand, "
            "soil temperature and moisture, turbulence/mixing, and seasonal vegetation state. The preset avoids GPP, RECO, NEE/FC variants, QC flags, and model outputs to reduce target leakage."
        )
    elif family == "Methane flux":
        for group, maxn in [
            ("soil_water", 2), ("soil_temperature", 1), ("air_temperature", 1),
            ("radiation_shortwave", 1), ("vpd", 1), ("humidity", 1),
            ("turbulence", 2), ("pressure", 1), ("precipitation", 1), ("vegetation", 1)
        ]:
            _add_group(selected, cols, group, target, timestamp_col, maxn)
        rationale = (
            "Methane flux is often linked to wetness/water table or soil water, soil temperature, vegetation-mediated transport, atmospheric/turbulent mixing, pressure, and recent moisture inputs. The preset prioritizes wetness and thermal controls."
        )
    elif family == "Nitrous oxide flux":
        for group, maxn in [
            ("soil_water", 1), ("soil_temperature", 1), ("air_temperature", 1),
            ("precipitation", 1), ("soil_nitrogen", 2), ("vpd", 1), ("humidity", 1), ("turbulence", 1)
        ]:
            _add_group(selected, cols, group, target, timestamp_col, maxn)
        rationale = (
            "N₂O flux is commonly event-driven and linked to soil moisture, soil temperature, precipitation, nitrogen availability/fertilization, and mixing/turbulence. If fertilizer or soil-N columns exist, they are prioritized."
        )
    elif family == "Latent heat / evapotranspiration":
        for group, maxn in [
            ("radiation_net", 1), ("radiation_shortwave", 1), ("air_temperature", 1),
            ("vpd", 1), ("humidity", 1), ("turbulence", 2), ("soil_water", 1),
            ("precipitation", 1), ("vegetation", 1), ("ground_heat", 1)
        ]:
            _add_group(selected, cols, group, target, timestamp_col, maxn)
        rationale = (
            "Latent heat / ET is controlled by available energy, atmospheric demand, aerodynamic transport, water availability, and vegetation state. The preset prioritizes radiation, VPD/RH, wind/u*, and soil water."
        )
    elif family == "Sensible heat":
        for group, maxn in [
            ("radiation_net", 1), ("radiation_shortwave", 1), ("air_temperature", 1),
            ("soil_temperature", 1), ("vpd", 1), ("humidity", 1), ("turbulence", 2),
            ("soil_water", 1), ("ground_heat", 1)
        ]:
            _add_group(selected, cols, group, target, timestamp_col, maxn)
        rationale = (
            "Sensible heat is controlled by available energy, thermal gradients, turbulence/aerodynamic transport, and surface wetness. The preset prioritizes radiation, temperature, wind/u*, and soil state."
        )
    elif family == "Gross primary productivity":
        for group, maxn in [
            ("radiation_shortwave", 1), ("radiation_net", 1), ("air_temperature", 1),
            ("vpd", 1), ("humidity", 1), ("soil_water", 1), ("soil_temperature", 1),
            ("vegetation", 2), ("turbulence", 1)
        ]:
            _add_group(selected, cols, group, target, timestamp_col, maxn)
        rationale = (
            "GPP is primarily controlled by light, canopy/air temperature, atmospheric demand, water availability, and vegetation state. The preset prioritizes radiation, VPD, temperature, soil water, and LAI/NDVI when available."
        )
    elif family == "Ecosystem respiration":
        for group, maxn in [
            ("soil_temperature", 1), ("air_temperature", 1), ("soil_water", 1),
            ("precipitation", 1), ("vegetation", 1), ("turbulence", 1), ("radiation_shortwave", 1)
        ]:
            _add_group(selected, cols, group, target, timestamp_col, maxn)
        rationale = (
            "Ecosystem respiration is commonly linked to soil/canopy temperature, soil moisture, recent precipitation, vegetation state, and nighttime/turbulence filtering. The preset prioritizes temperature and soil water controls."
        )
    else:
        # Generic fallback: physically common EC drivers, excluding obvious target/model/QC columns.
        for group, maxn in [
            ("radiation_shortwave", 1), ("radiation_net", 1), ("air_temperature", 1),
            ("vpd", 1), ("humidity", 1), ("soil_temperature", 1), ("soil_water", 1),
            ("turbulence", 2), ("precipitation", 1), ("pressure", 1), ("vegetation", 1)
        ]:
            _add_group(selected, cols, group, target, timestamp_col, maxn)
        rationale = (
            "Generic EC preset: selects common meteorological, soil, turbulence, radiation, and vegetation variables while avoiding the target, QC flags, and model-output columns. Review the list before running."
        )

    # Remove duplicates and keep the list manageable for first-time users.
    selected = list(dict.fromkeys(selected))[:10]
    if selected:
        selected_text = ", ".join(selected)
    else:
        selected_text = "No matching predictors were found automatically. Select drivers manually."
    guide = (
        f"Literature-informed preset for: {family}\n"
        f"Suggested predictors: {selected_text}\n\n"
        f"Why these predictors? {rationale}\n\n"
        "Review before running: remove variables that are derived from the target, gap-filled target variants, QC flags, or variables unavailable for your scientific question."
    )
    return selected, family, guide

def _apply_predictor_preset_to_listbox(lb, cols, target, timestamp_col=None, status_widget=None, select=True):
    """Apply or preview the target-specific preset in a Tk listbox."""
    selected, family, guide = _recommended_predictors_for_target(cols, target, timestamp_col=timestamp_col)
    if select:
        lb.selection_clear(0, "end")
        for i, c in enumerate(cols):
            if c in selected:
                lb.selection_set(i)
                try:
                    lb.see(i)
                except Exception:
                    pass
    if status_widget is not None:
        try:
            status_widget.configure(text=guide)
        except Exception:
            pass
    return selected, family, guide

def _resample_view(df_local, rs_mode, tscol, min_coverage=_MIN_RESAMPLE_COVERAGE,
                   sum_columns=None, mean_columns=None,
                   coverage_relative_to_available_rows=False,
                   drop_complete_cases=True):
    """Return a native/daily/weekly analysis frame.

    Sequence construction must preserve rows where the target is missing because
    the target is needed only at the prediction endpoint, not at every predictor
    step. Set ``drop_complete_cases=False`` for LSTM/H-LSTM source frames and
    create a separate complete endpoint frame for nonsequence models.
    """
    if rs_mode == 1:
        return _resample_df(
            df_local, "D", tscol, min_coverage=min_coverage,
            sum_columns=sum_columns, mean_columns=mean_columns,
            coverage_relative_to_available_rows=coverage_relative_to_available_rows,
            drop_complete_cases=drop_complete_cases,
        ), "Daily"
    if rs_mode == 2:
        return _resample_df(
            df_local, "W", tscol, min_coverage=min_coverage,
            sum_columns=sum_columns, mean_columns=mean_columns,
            coverage_relative_to_available_rows=coverage_relative_to_available_rows,
            drop_complete_cases=drop_complete_cases,
        ), "Weekly"
    # Native; ensure datetime column exists and is valid.
    dfw = df_local.copy()
    dfw[tscol] = _to_datetime_1d(dfw[tscol])
    dfw = dfw.dropna(subset=[tscol]).set_index(tscol, drop=False).sort_index()
    numeric_cols = [c for c in dfw.columns if c != tscol]
    for c in numeric_cols:
        dfw[c] = pd.to_numeric(dfw[c], errors="coerce")
    dfw = dfw.replace([np.inf, -np.inf], np.nan)
    if drop_complete_cases:
        dfw = dfw.dropna(subset=numeric_cols, how="any")
    return dfw, "Native"


def _available_period(df, tscol):
    """Return full start/end datetime strings for the selected timestamp column."""
    try:
        if tscol not in df.columns:
            return "", ""
        ts = _to_datetime_1d(df[tscol]).dropna()
        if ts.empty:
            return "", ""
        return ts.min().strftime("%Y-%m-%d %H:%M"), ts.max().strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "", ""


def _guess_timestamp_column(columns):
    """Prefer the native EC timestamp over date-only helper columns.

    Alphabetical sorting must not cause DATESTAMP_START or another date-only
    column to become the default time coordinate.
    """
    cols = list(columns)
    preferred = [
        "TIMESTAMP_START", "TIMESTAMP_END", "timestamp_start", "timestamp_end",
        "timestamp", "datetime", "time",
    ]
    for name in preferred:
        for c in cols:
            if str(c) == name or str(c).lower() == name.lower():
                return c
    # Then prefer columns containing timestamp/datetime, but avoid DATESTAMP.
    for c in cols:
        cl = str(c).lower()
        if "datestamp" not in cl and ("timestamp" in cl or "datetime" in cl):
            return c
    # Last resort: any time-like column, still avoiding DATESTAMP when possible.
    for c in cols:
        cl = str(c).lower()
        if "datestamp" not in cl and "time" in cl:
            return c
    return cols[0] if cols else ""

def _filter_analysis_period(df_local, tscol, start_text=None, end_text=None):
    """Filter a dataframe to the selected analysis period before resampling/modeling."""
    if tscol not in df_local.columns:
        raise ValueError(f"Timestamp column not found: {tscol}")
    out = df_local.copy()
    out[tscol] = _to_datetime_1d(out[tscol])
    out = out.dropna(subset=[tscol])
    start_text_clean = str(start_text or "").strip()
    end_text_clean = str(end_text or "").strip()
    start = pd.to_datetime(start_text_clean, errors="coerce") if start_text_clean else pd.NaT
    end = pd.to_datetime(end_text_clean, errors="coerce") if end_text_clean else pd.NaT
    if start_text_clean and pd.isna(start):
        raise ValueError("Invalid start date/time. Use YYYY-MM-DD HH:MM, for example 2013-01-01 00:00.")
    if end_text_clean and pd.isna(end):
        raise ValueError("Invalid end date/time. Use YYYY-MM-DD HH:MM, for example 2022-12-31 23:30.")
    if pd.notna(start) and pd.notna(end) and start > end:
        raise ValueError("Start date/time must be earlier than End date/time.")
    if pd.notna(start):
        out = out[out[tscol] >= start]
    if pd.notna(end):
        # Exact date/time is inclusive. A date-only value includes that whole day.
        date_only = len(end_text_clean) <= 10
        if date_only:
            out = out[out[tscol] < end.normalize() + pd.Timedelta(days=1)]
        else:
            out = out[out[tscol] <= end]
    if out.empty:
        raise ValueError("No data remain after applying the selected analysis period.")
    return out


def _guess_daylight_column(columns):
    """Choose a sensible radiation variable for optional day/night filtering."""
    cols = list(columns)
    priorities = (
        ("SW_IN", "SWIN", "SHORTWAVE_IN"),
        ("PPFD_IN", "PAR_IN"),
        ("PPFD", "PAR"),
        ("NETRAD", "RNET", "NET_RADIATION", "RN"),
        ("PPFD_DIF",),
    )
    for aliases in priorities:
        for c in cols:
            u = str(c).upper()
            parts = _split_name_parts(c) if "_split_name_parts" in globals() else [u]
            if any(u == a or u.startswith(a + "_") or (parts and parts[0] == a) for a in aliases):
                if "OUT" not in u:
                    return c
    return cols[0] if cols else ""


def _apply_diurnal_subset(df_local, tscol, mode="All observations",
                           radiation_col=None, threshold=10.0):
    """Filter subdaily observations before aggregation for FC sensitivity tests.

    Daytime is defined as radiation strictly above ``threshold``; nighttime is
    radiation less than or equal to it. The classifier can be a shortwave, PPFD,
    PAR, or net-radiation column. The main Figure 3B workflow uses All observations.
    """
    mode_text = str(mode or "All observations")
    if mode_text.lower().startswith("all"):
        return df_local.copy(), {
            "diurnal_mode": "All observations",
            "daylight_variable": None,
            "daylight_threshold": np.nan,
            "rows_before_diurnal_filter": int(len(df_local)),
            "rows_after_diurnal_filter": int(len(df_local)),
        }
    if radiation_col not in df_local.columns:
        raise ValueError(
            "Select a valid radiation column for the daytime/nighttime sensitivity analysis."
        )
    threshold = float(threshold)
    out = df_local.copy()
    out[tscol] = _to_datetime_1d(out[tscol])
    radiation = pd.to_numeric(out[radiation_col], errors="coerce")
    if mode_text.lower().startswith("day"):
        keep = radiation > threshold
        normalized_mode = "Daytime"
    elif mode_text.lower().startswith("night"):
        keep = radiation <= threshold
        normalized_mode = "Nighttime"
    else:
        raise ValueError(f"Unknown diurnal subset: {mode_text}")
    out = out.loc[keep.fillna(False)].copy()
    if out.empty:
        raise ValueError(
            f"No {normalized_mode.lower()} rows remain using {radiation_col} and threshold {threshold:g}."
        )
    return out, {
        "diurnal_mode": normalized_mode,
        "daylight_variable": str(radiation_col),
        "daylight_threshold": threshold,
        "rows_before_diurnal_filter": int(len(df_local)),
        "rows_after_diurnal_filter": int(len(out)),
    }


def _seasonal_climatology_predictions(y, timestamps, train_idx, test_idx,
                                       window_days=7):
    """Predict test observations from training-only day-of-year climatology."""
    y = np.asarray(y, dtype=float).reshape(-1)
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps, errors="coerce"))
    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)
    train_y = y[train_idx]
    train_doy = ts[train_idx].dayofyear.to_numpy(dtype=int)
    fallback = float(np.nanmean(train_y))
    window = max(0, int(window_days))
    out = []
    for idx in test_idx:
        doy = int(ts[idx].dayofyear)
        distance = np.abs(train_doy - doy)
        circular = np.minimum(distance, 366 - distance)
        values = train_y[circular <= window]
        values = values[np.isfinite(values)]
        out.append(float(np.nanmean(values)) if values.size else fallback)
    return np.asarray(out, dtype=float)


def _baseline_skill_score(y_true, y_pred, baseline_pred):
    """Return 1 - model MSE / baseline MSE on identical observations."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    bp = np.asarray(baseline_pred, dtype=float)
    valid = np.isfinite(yt) & np.isfinite(yp) & np.isfinite(bp)
    if not np.any(valid):
        return np.nan
    model_mse = float(np.mean((yt[valid] - yp[valid]) ** 2))
    baseline_mse = float(np.mean((yt[valid] - bp[valid]) ** 2))
    if not np.isfinite(baseline_mse) or baseline_mse <= 0:
        return np.nan
    return float(1.0 - model_mse / baseline_mse)


def _predictor_overlap_summary(df_period, predictors, target, tscol):
    """Summarize target/predictor temporal overlap before resampling.

    Returns a tuple: (summary_dataframe, complete_case_mask, limiting_text).
    The table helps users identify variables that shorten the model-ready period
    because ML uses rows where the target and all selected predictors are present.
    """
    predictors = list(dict.fromkeys([p for p in predictors if p in df_period.columns]))
    cols = list(dict.fromkeys([target] + predictors))
    if target not in df_period.columns:
        raise ValueError(f"Target not found: {target}")
    if tscol not in df_period.columns:
        raise ValueError(f"Timestamp column not found: {tscol}")

    local = df_period[[tscol] + cols].copy()
    local[tscol] = _to_datetime_1d(local[tscol])
    local = local.dropna(subset=[tscol])
    n_period = int(len(local))
    if n_period == 0:
        raise ValueError("No rows remain in the selected analysis period.")

    numeric = local[cols].apply(pd.to_numeric, errors="coerce")
    complete_mask = numeric.notna().all(axis=1)
    n_complete = int(complete_mask.sum())

    rows = []
    for col in cols:
        valid = numeric[col].notna()
        n_valid = int(valid.sum())
        pct_valid = 100.0 * n_valid / max(1, n_period)
        first_valid = _fmt_time_range(local.loc[valid, tscol]) if n_valid else "n/a"

        # How many complete rows would be recovered if this predictor were removed?
        # For target, do not calculate removal gain; the target is required.
        if col == target:
            n_complete_without = n_complete
            gain = 0
            role = "target"
        else:
            other_cols = [c for c in cols if c != col]
            n_complete_without = int(numeric[other_cols].notna().all(axis=1).sum())
            gain = max(0, n_complete_without - n_complete)
            role = "predictor"

        if pct_valid < 50:
            note = "low coverage"
        elif gain > max(50, 0.10 * max(1, n_complete)):
            note = "limits overlap"
        else:
            note = "ok"

        rows.append({
            "variable": col,
            "role": role,
            "nonmissing_rows": n_valid,
            "nonmissing_pct": pct_valid,
            "valid_period": first_valid,
            "complete_rows_if_removed": n_complete_without if col != target else "required",
            "row_gain_if_removed": gain if col != target else "required",
            "note": note,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        # Target first, then strongest limiting predictors.
        out["_sort_gain"] = pd.to_numeric(out["row_gain_if_removed"], errors="coerce").fillna(-1)
        out["_sort_role"] = out["role"].map({"target": 0, "predictor": 1}).fillna(2)
        out = out.sort_values(["_sort_role", "_sort_gain", "nonmissing_pct"], ascending=[True, False, True]).drop(columns=["_sort_gain", "_sort_role"])

    limiting = out[(out["role"] == "predictor") & (out["note"].isin(["low coverage", "limits overlap"]))].head(3)
    if limiting.empty:
        limiting_text = "No obvious single predictor is strongly limiting overlap."
    else:
        limiting_text = "Potential overlap-limiting predictors: " + ", ".join(limiting["variable"].astype(str).tolist())
    return out, complete_mask, limiting_text

def _analysis_period_text(df, tscol, start_var=None, end_var=None):
    """Concise user-facing analysis period text.

    The available dataset range is used only to pre-fill the entries. Showing
    both available and selected periods in the GUI was redundant and confusing.
    """
    start = start_var.get().strip() if start_var is not None else ""
    end = end_var.get().strip() if end_var is not None else ""
    return f"Analysis period: {start or 'start'} to {end or 'end'}"

def _add_analysis_period_controls(parent, df, ts_var, row_start=2, width=380):
    """Add reusable start/end analysis-period controls to a timestamp panel."""
    start0, end0 = _available_period(df, ts_var.get())
    start_var = StringVar(parent, value=start0)
    end_var = StringVar(parent, value=end0)
    use_full = tk.BooleanVar(parent, value=True)

    ttk.Label(parent, text="Analysis period").grid(row=row_start, column=0, sticky="w", padx=6, pady=(6,3))
    ttk.Label(parent, text="Start date/time").grid(row=row_start, column=1, sticky="w", padx=6, pady=(6,3))
    start_entry = ttk.Entry(parent, textvariable=start_var, width=19)
    start_entry.grid(row=row_start, column=2, sticky="ew", padx=6, pady=(6,3))
    ttk.Label(parent, text="End date/time").grid(row=row_start+1, column=1, sticky="w", padx=6, pady=3)
    end_entry = ttk.Entry(parent, textvariable=end_var, width=19)
    end_entry.grid(row=row_start+1, column=2, sticky="ew", padx=6, pady=3)
    parent.columnconfigure(2, weight=1)

    def _set_entries_state():
        state = "disabled" if use_full.get() else "normal"
        start_entry.configure(state=state)
        end_entry.configure(state=state)

    def _refresh_period(*_):
        a, b = _available_period(df, ts_var.get())
        if use_full.get():
            start_var.set(a)
            end_var.set(b)
        _set_entries_state()

    ttk.Checkbutton(parent, text="Use full available record", variable=use_full, command=_refresh_period).grid(
        row=row_start+1, column=0, sticky="w", padx=6, pady=3
    )

    try:
        ts_var.trace_add("write", _refresh_period)
    except Exception:
        pass
    _refresh_period()
    return start_var, end_var, use_full, None

def _chronological_split(n, test_ratio=0.2):
    """Deterministic chronological split (train first, test last)."""
    n_test = int(round(test_ratio * n))
    n_test = max(1, min(n - 1, n_test)) if n > 1 else 0
    idx = np.arange(n)
    return idx[:-n_test], idx[-n_test:]

def _random_fixed_split(n, test_ratio=0.2):
    idx = np.arange(n)
    rng = np.random.RandomState(42)
    rng.shuffle(idx)
    n_test = int(round(test_ratio * n))
    n_test = max(1, min(n - 1, n_test)) if n > 1 else 0
    return idx[n_test:], idx[:n_test]  # train, test

def _random_new_split(n, test_ratio=0.2):
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_test = int(round(test_ratio * n))
    n_test = max(1, min(n - 1, n_test)) if n > 1 else 0
    return idx[n_test:], idx[:n_test]  # train, test

def _parse_fraction(value, default, min_value=0.0, max_value=1.0, label="fraction"):
    """Parse 20, 20%, or 0.20 and enforce a scientifically valid range."""
    try:
        txt = str(value).strip().replace("%", "")
        v = float(txt)
        if v > 1.0:
            v = v / 100.0
        if not np.isfinite(v) or not (float(min_value) <= v <= float(max_value)):
            raise ValueError
        return float(v)
    except Exception:
        raise ValueError(
            f"Invalid {label}. Enter a value between {100*min_value:.0f}% and {100*max_value:.0f}%."
        )


def _parse_test_fraction(value, default=0.20):
    """Return test fraction from user entry. Accepts 20, 20%, or 0.20."""
    try:
        return _parse_fraction(value, default, min_value=0.05, max_value=0.80,
                               label="holdout test fraction")
    except ValueError:
        return float(default)

def _make_split_indices(n, strategy="Chronological holdout", test_ratio=0.20):
    """Return train/test row indices for the selected split strategy."""
    if n < 3:
        raise ValueError("Not enough valid rows after filtering/resampling to create train/test split.")
    strategy = str(strategy or "Chronological holdout")
    if "Random fixed" in strategy:
        tr, ts = _random_fixed_split(n, test_ratio)
    elif "Random new" in strategy:
        tr, ts = _random_new_split(n, test_ratio)
    else:
        tr, ts = _chronological_split(n, test_ratio)
    if len(tr) < 2 or len(ts) < 1:
        raise ValueError(f"Invalid split: train={len(tr)}, test={len(ts)}. Use more data or a smaller test fraction.")
    return tr, ts


def _expanding_window_splits(n, n_folds=5, initial_train_fraction=0.50):
    """Create non-overlapping, expanding-window temporal validation folds.

    The first half of the record initializes training. The remaining observations
    are divided into consecutive test blocks. Training expands after each block.
    No future observations enter any fold's training set.
    """
    n = int(n)
    n_folds = max(2, int(n_folds))
    if n < max(20, n_folds + 3):
        raise ValueError("Not enough model-ready observations for blocked time-series cross-validation.")
    initial_train = max(2, int(math.floor(initial_train_fraction * n)))
    remaining = n - initial_train
    if remaining < n_folds:
        raise ValueError("Not enough observations after the initial training window for the requested number of folds.")
    test_blocks = [np.asarray(b, dtype=int) for b in np.array_split(np.arange(initial_train, n), n_folds) if len(b)]
    splits = []
    for fold, test_idx in enumerate(test_blocks, start=1):
        train_idx = np.arange(0, int(test_idx[0]), dtype=int)
        if len(train_idx) < 2 or len(test_idx) < 1:
            continue
        splits.append((fold, train_idx, test_idx))
    if len(splits) < 2:
        raise ValueError("Could not construct at least two valid expanding-window folds.")
    return splits


def _make_validation_splits(n, strategy="Chronological holdout", test_ratio=0.20,
                            n_folds=5, initial_train_fraction=0.50):
    strategy = str(strategy or "Chronological holdout")
    if "Blocked time-series CV" in strategy or "Expanding-window" in strategy:
        return _expanding_window_splits(
            n, n_folds=n_folds, initial_train_fraction=float(initial_train_fraction)
        )
    tr, ts = _make_split_indices(n, strategy, test_ratio)
    return [(1, tr, ts)]


def _make_timestamp_aligned_validation_splits(model_timestamps, reference_timestamps,
                                               strategy="Chronological holdout",
                                               test_ratio=0.20, n_folds=5,
                                               initial_train_fraction=0.50):
    """Map validation windows from one ordered reference timeline to model rows.

    For blocked CV, the reference timeline defines exactly ``n_folds`` future
    blocks. Model rows are then mapped to the same calendar cutoffs. This is
    especially important for LSTM/H-LSTM, where gap-safe sequence construction
    removes many candidate endpoints. Requested folds are never silently dropped:
    either every fold is returned or a clear error is raised.
    """
    model_raw = pd.Series(model_timestamps)
    ref_raw = pd.Series(reference_timestamps)
    model_parsed = _to_datetime_1d(model_raw)
    ref_parsed = _to_datetime_1d(ref_raw)
    if model_parsed.isna().any():
        raise ValueError("Model timestamps contain invalid values.")
    if ref_parsed.isna().any():
        raise ValueError("Reference timestamps contain invalid values.")

    model_ts = pd.DatetimeIndex(model_parsed)
    ref_ts = pd.DatetimeIndex(ref_parsed)
    if not model_ts.is_monotonic_increasing or model_ts.has_duplicates:
        raise ValueError("Model timestamps must be unique and sorted before validation.")
    if not ref_ts.is_monotonic_increasing or ref_ts.has_duplicates:
        raise ValueError("Reference timestamps must be unique and sorted before validation.")

    strategy = str(strategy or "Chronological holdout")
    blocked = "Blocked time-series CV" in strategy or "Expanding-window" in strategy
    if "Random" in strategy:
        return _make_validation_splits(
            len(model_ts), strategy, test_ratio, n_folds,
            initial_train_fraction=initial_train_fraction,
        )

    ref_splits = _make_validation_splits(
        len(ref_ts), strategy, test_ratio, n_folds,
        initial_train_fraction=initial_train_fraction,
    )
    if blocked and len(ref_splits) != int(n_folds):
        raise ValueError(
            f"Requested {int(n_folds)} blocked folds, but the reference timeline "
            f"produced {len(ref_splits)}. Reduce folds or use a longer period."
        )

    fold_starts = [ref_ts[int(ref_test[0])] for _, _, ref_test in ref_splits]
    final_reference_end = ref_ts[int(ref_splits[-1][2][-1])]
    mapped = []
    for pos, (fold, _, ref_test) in enumerate(ref_splits):
        start = fold_starts[pos]
        next_start = fold_starts[pos + 1] if pos + 1 < len(fold_starts) else None
        train_idx = np.flatnonzero(model_ts < start)
        if next_start is None:
            test_mask = (model_ts >= start) & (model_ts <= final_reference_end)
        else:
            test_mask = (model_ts >= start) & (model_ts < next_start)
        test_idx = np.flatnonzero(test_mask)
        if len(train_idx) < 2:
            raise ValueError(
                f"Fold {int(fold)} has only {len(train_idx)} training rows after "
                "timestamp alignment. Use an earlier analysis start or a smaller initial training fraction."
            )
        minimum_test_rows = 2 if blocked else 1
        if len(test_idx) < minimum_test_rows:
            raise ValueError(
                f"Fold {int(fold)} has only {len(test_idx)} valid model row(s) after "
                "sequence/gap alignment; at least 2 are required for blocked-CV metrics. "
                "MeaningFlux will not silently drop or fabricate a requested fold. Reduce the "
                "number of folds, shorten the sequence length, remove an overlap-limiting "
                "predictor, or use a period with more contiguous data."
            )
        mapped.append((int(fold), train_idx.astype(int), test_idx.astype(int)))

    if blocked:
        expected_ids = list(range(1, int(n_folds) + 1))
        actual_ids = [fold for fold, _, _ in mapped]
        if len(mapped) != int(n_folds) or actual_ids != expected_ids:
            raise ValueError(
                f"Requested blocked folds {expected_ids}, but obtained {actual_ids}. "
                "No fold reduction is allowed."
            )
    elif not mapped:
        raise ValueError("No model rows fall inside the requested temporal validation window.")
    return mapped

def _fmt_time_range(values):
    try:
        s = pd.Series(_to_datetime_1d(values)).dropna()
        if s.empty:
            return "n/a"
        return f"{s.min().date()} to {s.max().date()}"
    except Exception:
        return "n/a"


def _plot_gap_aware_line(ax, timestamps, values, *args, expected_step=None, **kwargs):
    """Plot a time series without drawing lines across missing time intervals.

    The first contiguous segment receives the requested legend label; subsequent
    segments use the same style but no duplicate legend label.
    """
    ts = pd.DatetimeIndex(pd.to_datetime(pd.Series(timestamps), errors="coerce"))
    vals = np.asarray(values, dtype=float).reshape(-1)
    valid = (~ts.isna()) & np.isfinite(vals)
    ts = ts[valid]
    vals = vals[valid]
    if len(ts) == 0:
        return []
    order = np.argsort(ts.asi8)
    ts = ts[order]
    vals = vals[order]
    step = (
        pd.Timedelta(expected_step)
        if expected_step is not None and not pd.isna(expected_step)
        else _infer_expected_timestep(ts)
    )
    if pd.isna(step) or step <= pd.Timedelta(0) or len(ts) < 2:
        return ax.plot(ts, vals, *args, **kwargs)
    tolerance = max(pd.Timedelta(seconds=1), step * 0.10)
    diffs = pd.to_timedelta(np.diff(ts.asi8), unit="ns")
    split_positions = np.flatnonzero(diffs > (step + tolerance)) + 1
    segments = np.split(np.arange(len(ts)), split_positions)
    label = kwargs.pop("label", None)
    artists = []
    shared_color = kwargs.get("color")
    for seg_i, seg in enumerate(segments):
        if len(seg) == 0:
            continue
        seg_kwargs = dict(kwargs)
        if shared_color is not None:
            seg_kwargs["color"] = shared_color
        seg_kwargs["label"] = label if seg_i == 0 else "_nolegend_"
        lines = ax.plot(ts[seg], vals[seg], *args, **seg_kwargs)
        artists.extend(lines)
        if shared_color is None and lines:
            # Matplotlib advances the color cycle for every ax.plot call. Lock
            # all later gap-separated segments to the first segment's color.
            shared_color = lines[0].get_color()
    return artists

def _split_summary_from_frame(dfw, tscol, train_idx, test_idx, strategy, test_ratio):
    """Return a compact dictionary describing the train/test pool."""
    try:
        if tscol in dfw.columns:
            tvals = _to_datetime_1d(dfw[tscol])
        else:
            tvals = _to_datetime_1d(dfw.index)
        tr_time = np.asarray(tvals)[train_idx]
        ts_time = np.asarray(tvals)[test_idx]
    except Exception:
        tr_time, ts_time = [], []
    return {
        "split_strategy": str(strategy),
        "test_fraction": float(test_ratio),
        "n_total": int(len(dfw)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "train_period": _fmt_time_range(tr_time),
        "test_period": _fmt_time_range(ts_time),
    }

def _split_summary_text(info):
    if not info:
        return "Split not defined yet. Click Preview split or run a model."
    n_folds = int(info.get("n_folds", 1) or 1)
    if n_folds > 1:
        initial_pct = 100 * float(info.get("initial_train_fraction", 0.50) or 0.50)
        split_header = (
            f"Validation: {info.get('split_strategy', 'n/a')} | folds={n_folds} | "
            f"initial training={initial_pct:.0f}%"
        )
    else:
        pct = 100 * float(info.get("test_fraction", np.nan))
        split_header = f"Split: {info.get('split_strategy', 'n/a')} | test={pct:.0f}%"
    lines = [split_header]
    if info.get("selected_analysis_period"):
        lines.append(f"Analysis period: {info.get('selected_analysis_period')}")
    if info.get("complete_case_period"):
        lines.append(f"Model-ready period: {info.get('complete_case_period')}")
    if info.get("n_period_rows") not in (None, ""):
        lines.append(
            f"Rows in analysis period: {info.get('n_period_rows', 'n/a')} | "
            f"usable rows after target/predictor filtering: {info.get('n_total', 'n/a')}"
        )
    else:
        lines.append(f"Rows after filtering/resampling: {info.get('n_total', 'n/a')}")
    lines.append(f"Train rows: {info.get('n_train', 'n/a')} | test rows: {info.get('n_test', 'n/a')}")
    lines.append(f"Train period: {info.get('train_period', 'n/a')}")
    lines.append(f"Prediction plot shows testing period: {info.get('test_period', 'n/a')}")
    if info.get("coverage_warning"):
        lines.append(f"Note: {info.get('coverage_warning')}")
    return "\n".join(lines)

# =============================================================================
#                         TRAINING MONITOR (for LSTM)
# =============================================================================
class TrainingMonitor:
    """Live loss curve + progress bar + ETA + log (thread-safe)."""
    def __init__(self, parent):
        self.parent = parent
        self.q = queue.Queue()

        top = Frame(parent); top.pack(fill="x")
        self.title = Label(top, text="Training", font=("TkDefaultFont", 10, "bold"))
        self.title.pack(side="left")
        self.status = Label(top, text="idle", fg="#555")
        self.status.pack(side="right")

        self.pb = ttk.Progressbar(parent, mode="determinate", maximum=100)
        self.pb.pack(fill="x", pady=2)
        self.eta = Label(parent, text="ETA: --:--:--")
        self.eta.pack(anchor="e")

        self.fig, self.ax = plt.subplots(figsize=(6.4, 2.2))
        self.ax.set_title("Loss"); self.ax.set_xlabel("Epoch"); self.ax.set_ylabel("Loss")
        self.line, = self.ax.plot([], [], marker=".", lw=1)
        self.losses = []
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.draw(); self.canvas.get_tk_widget().pack(fill="both", expand=True, pady=(4, 0))

        self.log = tk.Text(parent, height=6, wrap="word")
        self.log.pack(fill="both"); self.log.configure(state="disabled")

        self.total = 0; self.t0 = None
        self._poll_after_id = self.parent.after(100, self._poll)

    def reset(self, total_epochs, title, meta):
        self.total = max(1, int(total_epochs)); self.t0 = time.time(); self.losses = []
        self.ax.cla(); self.ax.set_title("Loss"); self.ax.set_xlabel("Epoch"); self.ax.set_ylabel("Loss")
        self.line, = self.ax.plot([], [], marker=".", lw=1)
        self.pb["value"] = 0; self._eta("--:--:--")
        self.title.config(text=title); self.status.config(text="running…", fg="#0a7")
        self._log(f"▶ {title}\n{meta}\n")

    def done(self, note="finished"):
        self.status.config(text=note, fg="#07a"); self._log(f"✓ {note}\n")

    def stopped(self):
        self.status.config(text="stopped", fg="#c60"); self._log("■ Training stopped by user.\n")

    def info_note(self, text):
        self.status.config(text="info", fg="#555"); self._log(text + "\n")

    def error(self, msg):
        self.status.config(text="error", fg="#b00"); self._log(f"✗ ERROR: {msg}\n")

    def push(self, epoch, loss):
        self.q.put(("p", epoch, float(loss)))

    def info(self, text):
        self.q.put(("i", text))

    def _poll(self):
        try:
            while True:
                t = self.q.get_nowait()
                if t[0] == "p":
                    self._on_progress(t[1], t[2])
                else:
                    self._log(t[1])
        except queue.Empty:
            pass
        try:
            self._poll_after_id = self.parent.after(100, self._poll)
        except Exception:
            self._poll_after_id = None

    def close(self):
        """Cancel the scheduled queue poll before the parent window is destroyed."""
        if getattr(self, "_poll_after_id", None) is not None:
            try:
                self.parent.after_cancel(self._poll_after_id)
            except Exception:
                pass
            self._poll_after_id = None

    def _on_progress(self, e, loss):
        while len(self.losses) <= e: self.losses.append(None)
        self.losses[e] = loss
        xs = [i+1 for i,v in enumerate(self.losses) if v is not None]
        ys = [v for v in self.losses if v is not None]
        self.line.set_data(xs, ys); self.ax.relim(); self.ax.autoscale_view()
        self.canvas.draw_idle()

        pct = 100.0 * (e + 1) / self.total; self.pb["value"] = pct
        elapsed = time.time() - (self.t0 or time.time())
        per_ep = elapsed / max(1, (e + 1)); remain = per_ep * max(0, self.total - (e + 1))
        self._eta(_fmt_sec(remain))
        self.status.config(text=f"epoch {e+1}/{self.total} | loss={loss:.4g}")

    def _log(self, txt):
        self.log.configure(state="normal"); self.log.insert("end", txt); self.log.see("end"); self.log.configure(state="disabled")

    def _eta(self, s): self.eta.config(text=f"ETA: {s}")

def _fmt_sec(sec):
    m, s = divmod(int(sec), 60); h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# =============================================================================
#                           MODEL RUNNERS
# =============================================================================
def run_linear(X, y, train_idx=None, test_idx=None, seed=_GLOBAL_SEED,
               fit_intercept=True):
    _set_reproducible_seed(seed)
    if train_idx is None or test_idx is None:
        X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.2, random_state=seed)
    else:
        X_tr, X_ts = X[train_idx], X[test_idx]
        y_tr, y_ts = y[train_idx], y[test_idx]
    m = LinearRegression(fit_intercept=bool(fit_intercept)).fit(X_tr, y_tr)
    return np.asarray(y_ts, dtype=float), np.asarray(m.predict(X_ts), dtype=float), m, (train_idx, test_idx)


def run_rf(X, y, train_idx=None, test_idx=None, seed=_GLOBAL_SEED,
           n_estimators=300, min_samples_leaf=1, max_depth=None,
           max_features=1.0):
    _set_reproducible_seed(seed)
    if train_idx is None or test_idx is None:
        X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.2, random_state=seed)
    else:
        X_tr, X_ts = X[train_idx], X[test_idx]
        y_tr, y_ts = y[train_idx], y[test_idx]
    m = RandomForestRegressor(
        n_estimators=int(n_estimators), random_state=seed, n_jobs=1,
        min_samples_leaf=int(min_samples_leaf),
        max_depth=(None if max_depth in (None, "", "None") else int(max_depth)),
        max_features=max_features,
    ).fit(X_tr, y_tr)
    return np.asarray(y_ts, dtype=float), np.asarray(m.predict(X_ts), dtype=float), m, (train_idx, test_idx)


def run_mlp(X, y, train_idx=None, test_idx=None, hidden=100,
            max_iter=1500, alpha=0.0001, learning_rate_init=0.001,
            activation="relu", batch_size="auto", seed=_GLOBAL_SEED):
    """Fit MLP with predictors and target standardized using training data only."""
    _set_reproducible_seed(seed)
    if train_idx is None or test_idx is None:
        train_idx, test_idx = _chronological_split(len(X), 0.20)
    X_tr, X_ts = np.asarray(X)[train_idx], np.asarray(X)[test_idx]
    y_tr, y_ts = np.asarray(y)[train_idx], np.asarray(y)[test_idx]
    x_scaler = StandardScaler().fit(X_tr)
    y_scaler = StandardScaler().fit(y_tr.reshape(-1, 1))
    X_tr_s = x_scaler.transform(X_tr)
    X_ts_s = x_scaler.transform(X_ts)
    y_tr_s = y_scaler.transform(y_tr.reshape(-1, 1)).ravel()
    m = MLPRegressor(
        hidden_layer_sizes=(int(hidden),), max_iter=int(max_iter),
        alpha=float(alpha), learning_rate_init=float(learning_rate_init),
        activation=str(activation), batch_size=batch_size,
        random_state=seed, solver="adam", early_stopping=False,
    ).fit(X_tr_s, y_tr_s)
    pred_s = m.predict(X_ts_s).reshape(-1, 1)
    pred = y_scaler.inverse_transform(pred_s).ravel()
    return y_ts.astype(float), pred.astype(float), m, (train_idx, test_idx)


def _contiguous_sequence_arrays(work, feature_cols, target, seq_len, expected_step=None):
    """Build fixed-cadence predictor sequences with endpoint-only targets.

    Predictor values must be finite at every step of the sequence. The target is
    required only at the final prediction timestamp. Missing target values at
    earlier sequence steps therefore do not invalidate an otherwise usable
    predictor window.
    """
    work = work.copy().sort_index()
    work = work[~work.index.duplicated(keep="last")]
    if not isinstance(work.index, pd.DatetimeIndex):
        work.index = pd.DatetimeIndex(pd.to_datetime(work.index, errors="coerce"))
        work = work[~work.index.isna()].sort_index()

    feature_cols = list(dict.fromkeys(list(feature_cols)))
    required_cols = feature_cols + [target]
    missing_cols = [c for c in required_cols if c not in work.columns]
    if missing_cols:
        raise ValueError("Sequence source is missing column(s): " + ", ".join(map(str, missing_cols)))

    for c in required_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work[required_cols] = work[required_cols].replace([np.inf, -np.inf], np.nan)

    T = int(seq_len)
    if T < 2:
        raise ValueError("Sequence length must be at least 2.")
    if len(work) < T:
        raise ValueError("Not enough source timestamps for the selected sequence length.")

    step = (
        pd.Timedelta(expected_step)
        if expected_step is not None and not pd.isna(expected_step)
        else _infer_expected_timestep(work.index)
    )
    if pd.isna(step) or step <= pd.Timedelta(0):
        raise ValueError("Could not infer a valid timestep for sequence construction.")
    tolerance = max(pd.Timedelta(seconds=1), step * 1e-6)

    xvals = work[feature_cols].to_numpy(dtype=float)
    yvals = work[target].to_numpy(dtype=float)
    idx = pd.DatetimeIndex(work.index)

    Xs, ys, ts_out = [], [], []
    rejected_gap_ts = []
    rejected_target_ts = []
    rejected_predictor_ts = []
    candidate_windows = max(0, len(work) - T + 1)

    for t in range(T - 1, len(work)):
        window_idx = idx[t - T + 1:t + 1]
        diffs = pd.to_timedelta(np.diff(window_idx.asi8), unit="ns")
        contiguous = bool(
            len(diffs) == T - 1
            and np.all(np.abs(diffs - step) <= tolerance)
        )
        if not contiguous:
            rejected_gap_ts.append(idx[t])
            continue

        # The target is required only at the endpoint.
        if not np.isfinite(yvals[t]):
            rejected_target_ts.append(idx[t])
            continue

        x_window = xvals[t - T + 1:t + 1]
        if not np.isfinite(x_window).all():
            rejected_predictor_ts.append(idx[t])
            continue

        Xs.append(x_window)
        ys.append(yvals[t])
        ts_out.append(idx[t])

    if not Xs:
        raise ValueError(
            "No valid sequences remain. Predictor windows must be complete and "
            "consecutive, while the target must be observed at the endpoint. "
            "Shorten the sequence, remove an overlap-limiting predictor, or "
            "choose a better-covered period."
        )

    gap_info = _timestamp_gap_diagnostics(idx, expected_step=step)
    complete_endpoint_rows = int(
        np.sum(np.isfinite(yvals) & np.isfinite(xvals).all(axis=1))
    )
    diagnostics = {
        "sequence_policy": (
            "fixed-cadence predictor windows; predictors complete at every "
            "sequence step; target required only at prediction endpoint; "
            "windows crossing gaps rejected"
        ),
        "target_requirement": "endpoint only",
        "predictor_requirement": "complete at every sequence step",
        "expected_timestep": str(step),
        "sequence_length": T,
        "n_source_rows_before_sequences": int(len(work)),
        # Backward-compatible field, now explicitly counting complete endpoint rows.
        "n_complete_rows_before_sequences": complete_endpoint_rows,
        "n_candidate_sequence_windows": int(candidate_windows),
        "n_candidate_endpoint_targets": int(
            np.sum(np.isfinite(yvals[T - 1:]))
        ),
        "n_valid_sequences": int(len(Xs)),
        "n_rejected_gap_windows": int(len(rejected_gap_ts)),
        "n_rejected_missing_endpoint_target": int(len(rejected_target_ts)),
        "n_rejected_missing_predictor_windows": int(len(rejected_predictor_ts)),
        "n_source_gaps": int(gap_info["n_gaps"]),
        "max_source_interval": gap_info["max_interval"],
        "max_source_interval_days": gap_info["max_interval_days"],
        "first_valid_sequence": pd.Timestamp(ts_out[0]),
        "last_valid_sequence": pd.Timestamp(ts_out[-1]),
        "rejected_timestamps": pd.DatetimeIndex(rejected_gap_ts),
        "rejected_missing_target_timestamps": pd.DatetimeIndex(rejected_target_ts),
        "rejected_missing_predictor_timestamps": pd.DatetimeIndex(rejected_predictor_ts),
    }
    return (
        np.asarray(Xs, dtype=float),
        np.asarray(ys, dtype=float),
        pd.DatetimeIndex(ts_out),
        diagnostics,
    )


def _make_lstm_sequence_data(df_resampled, features, target, seq_len,
                             return_diagnostics=False, expected_step=None):
    """Build unscaled LSTM sequences without requiring prior target values."""
    feature_cols = list(dict.fromkeys(list(features)))
    cols = feature_cols + [target]
    work = df_resampled[cols].copy()
    for c in cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan)
    X, y, ts, diagnostics = _contiguous_sequence_arrays(
        work, feature_cols, target, int(seq_len), expected_step=expected_step
    )
    return (X, y, ts, diagnostics) if return_diagnostics else (X, y, ts)

def run_keras_lstm(X_seq, y, epochs=100, batch_size=32, hidden=96,
                   learning_rate=0.001, dropout=0.0, gradient_clip=1.0,
                   train_idx=None, test_idx=None, progress_cb=None, info_cb=None,
                   seed=_GLOBAL_SEED):
    """Fit a true multistep Keras LSTM with leakage-free train-only scaling."""
    if Sequential is None:
        raise RuntimeError("TensorFlow/Keras not found. Install to enable LSTM.")
    _set_reproducible_seed(seed)
    if train_idx is None or test_idx is None:
        train_idx, test_idx = _chronological_split(len(X_seq), 0.20)
    X_seq = np.asarray(X_seq, dtype=float)
    y = np.asarray(y, dtype=float)
    X_tr, X_ts = X_seq[train_idx], X_seq[test_idx]
    y_tr, y_ts = y[train_idx], y[test_idx]

    n_features = X_seq.shape[2]
    x_scaler = StandardScaler().fit(X_tr.reshape(-1, n_features))
    y_scaler = StandardScaler().fit(y_tr.reshape(-1, 1))
    X_tr_s = x_scaler.transform(X_tr.reshape(-1, n_features)).reshape(X_tr.shape)
    X_ts_s = x_scaler.transform(X_ts.reshape(-1, n_features)).reshape(X_ts.shape)
    y_tr_s = y_scaler.transform(y_tr.reshape(-1, 1)).ravel()

    model = Sequential([
        Input(shape=(X_tr_s.shape[1], n_features)),
        LSTM(int(hidden), dropout=float(dropout)),
        Dense(1),
    ])
    optimizer_kwargs = {"learning_rate": float(learning_rate)}
    if gradient_clip is not None and float(gradient_clip) > 0:
        optimizer_kwargs["clipnorm"] = float(gradient_clip)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(**optimizer_kwargs),
        loss="mse",
    )

    class _CB(tf.keras.callbacks.Callback):
        def on_epoch_end(self, e, logs=None):
            if progress_cb and logs and "loss" in logs:
                progress_cb(e, float(logs["loss"]))
            if _stop_event.is_set():
                self.model.stop_training = True

    t0 = time.time()
    model.fit(
        X_tr_s, y_tr_s, epochs=int(epochs), batch_size=int(batch_size),
        verbose=0, callbacks=[_CB()], shuffle=False,
    )
    if info_cb:
        info_cb(f"{'Stopped' if _stop_event.is_set() else 'Training time'}: {time.time()-t0:.1f}s\n")
    pred_s = model.predict(X_ts_s, verbose=0).reshape(-1, 1)
    pred = y_scaler.inverse_transform(pred_s).ravel()
    return y_ts.astype(float), pred.astype(float), model, (train_idx, test_idx)


# ----------------- Hysteresis-aware helpers -----------------
def _prepare_h_lstm_frame(df, predictors, target, gate_col, tscol, rs_mode,
                          min_coverage=_MIN_RESAMPLE_COVERAGE,
                          sum_columns=None, mean_columns=None,
                          coverage_relative_to_available_rows=False):
    """Prepare an unscaled H-LSTM frame while preserving missing targets."""
    cols_need = list(dict.fromkeys(list(predictors) + [gate_col, target]))
    base = df[cols_need + [tscol]].copy()
    base[tscol] = _to_datetime_1d(base[tscol])
    base = base.dropna(subset=[tscol])
    if rs_mode == 1:
        base = _resample_df(
            base, "D", tscol, min_coverage=min_coverage,
            sum_columns=sum_columns, mean_columns=mean_columns,
            coverage_relative_to_available_rows=coverage_relative_to_available_rows,
            drop_complete_cases=False,
        )
    elif rs_mode == 2:
        base = _resample_df(
            base, "W", tscol, min_coverage=min_coverage,
            sum_columns=sum_columns, mean_columns=mean_columns,
            coverage_relative_to_available_rows=coverage_relative_to_available_rows,
            drop_complete_cases=False,
        )
    else:
        base = base.set_index(tscol).sort_index()
        for c in cols_need:
            base[c] = pd.to_numeric(base[c], errors="coerce")
        base = base.replace([np.inf, -np.inf], np.nan)
    return base


def _make_seq_data(df_resampled, features, target, gate_col, seq_len,
                   return_diagnostics=False, expected_step=None):
    """Build H-LSTM sequences with endpoint-only targets and a gap-safe Δgate."""
    feature_cols = list(dict.fromkeys(list(features) + [gate_col]))
    base = df_resampled[feature_cols + [target]].copy()
    for c in feature_cols + [target]:
        base[c] = pd.to_numeric(base[c], errors="coerce")
    base = base.replace([np.inf, -np.inf], np.nan).sort_index()
    base = base[~base.index.duplicated(keep="last")]
    if not isinstance(base.index, pd.DatetimeIndex):
        base.index = pd.DatetimeIndex(pd.to_datetime(base.index, errors="coerce"))
        base = base[~base.index.isna()].sort_index()
    if len(base) < int(seq_len) + 1:
        raise ValueError("Not enough source timestamps for the selected H-LSTM sequence length.")

    step = (
        pd.Timedelta(expected_step)
        if expected_step is not None and not pd.isna(expected_step)
        else _infer_expected_timestep(base.index)
    )
    if pd.isna(step) or step <= pd.Timedelta(0):
        raise ValueError("Could not infer a valid timestep for H-LSTM sequence construction.")
    tolerance = max(pd.Timedelta(seconds=1), step * 1e-6)
    timestamp_delta = base.index.to_series().diff()
    dgate = base[gate_col].diff()
    consecutive_from_previous = (timestamp_delta - step).abs() <= tolerance
    dgate = dgate.where(consecutive_from_previous.to_numpy())
    base["__DGATE__"] = dgate

    # Do not delete rows with missing target. The shared sequence builder checks
    # the target only at the final prediction timestamp.
    work = base[feature_cols + ["__DGATE__", target]]
    X, y, ts, diagnostics = _contiguous_sequence_arrays(
        work,
        feature_cols + ["__DGATE__"],
        target,
        int(seq_len),
        expected_step=step,
    )
    diagnostics["gate_variable"] = gate_col
    diagnostics["includes_delta_gate"] = True
    diagnostics["delta_gate_requirement"] = (
        "Δgate must be available at every sequence step; the first step therefore "
        "uses the immediately preceding gate observation"
    )
    diagnostics["n_gate_delta_break_rows"] = int(base["__DGATE__"].isna().sum())
    return (X, y, ts, diagnostics) if return_diagnostics else (X, y, ts)

def run_hysteresis_lstm_sequences(X_seq, y, hidden=96, epochs=100,
                                  batch_size=32, learning_rate=0.001,
                                  dropout=0.0, gradient_clip=1.0,
                                  train_idx=None, test_idx=None,
                                  progress_cb=None, info_cb=None,
                                  seed=_GLOBAL_SEED):
    """Fit the hysteresis-aware sequence model with the same Keras LSTM used
    by the standard LSTM.

    ``X_seq`` already contains the common predictors plus the internally
    calculated delta-gate channel. Architecture, optimizer, scaling, batch
    size, epochs, dropout, gradient clipping, and seed are otherwise identical
    to ``run_keras_lstm``. This isolates the added trajectory information as
    the intended model difference.
    """
    return run_keras_lstm(
        X_seq, y,
        epochs=epochs,
        batch_size=batch_size,
        hidden=hidden,
        learning_rate=learning_rate,
        dropout=dropout,
        gradient_clip=gradient_clip,
        train_idx=train_idx,
        test_idx=test_idx,
        progress_cb=progress_cb,
        info_cb=info_cb,
        seed=seed,
    )

def run_hysteresis_lstm(df_resampled, features, target, gate_col, seq_len=20,
                        hidden=96, epochs=100, batch_size=32, learning_rate=0.001,
                        dropout=0.0, gradient_clip=1.0,
                        train_idx=None, test_idx=None, progress_cb=None,
                        info_cb=None, seed=_GLOBAL_SEED, expected_step=None):
    """Compatibility wrapper that builds sequences once and fits H-LSTM."""
    X, y, ts_all = _make_seq_data(
        df_resampled, features, target, gate_col, int(seq_len),
        expected_step=expected_step,
    )
    yt, yp, model, split = run_hysteresis_lstm_sequences(
        X, y, hidden=hidden, epochs=epochs, batch_size=batch_size,
        learning_rate=learning_rate, dropout=dropout,
        gradient_clip=gradient_clip, train_idx=train_idx,
        test_idx=test_idx, progress_cb=progress_cb,
        info_cb=info_cb, seed=seed,
    )
    return yt, yp, ts_all[np.asarray(split[1], dtype=int)], split

# =============================================================================
#                 OPTIONAL PREDICTOR SCREENING ANALYTICS
# =============================================================================
def driver_perm_importance(df, features, target):
    X = df[features].values; y = df[target].values
    rf = RandomForestRegressor(n_estimators=300, random_state=42).fit(X, y)
    r = permutation_importance(rf, X, y, n_repeats=10, random_state=42)
    return rf, r.importances_mean, r.importances_std

def _make_hysteresis_view(df, target, gate, tscol, rs_mode, smooth=1):
    # Resample
    if rs_mode == 1:
        dfr = _resample_df(df[[target, gate, tscol]], "D", tscol)
    elif rs_mode == 2:
        dfr = _resample_df(df[[target, gate, tscol]], "W", tscol)
    else:
        dfr = df[[target, gate, tscol]].copy()
        dfr[tscol] = _to_datetime_1d(dfr[tscol])
        dfr = dfr.dropna(subset=[tscol]).set_index(tscol)
        dfr = dfr.dropna(how="any")
    # Smooth
    if smooth and int(smooth) > 1:
        s = int(smooth)
        dfr[target] = dfr[target].rolling(s, center=True, min_periods=max(1, s//2)).mean()
        dfr[gate]   = dfr[gate].rolling(s, center=True, min_periods=max(1, s//2)).mean()
        dfr = dfr.dropna(how="any")
    # Δgate & sign
    dgate = np.diff(dfr[gate].values)
    dfr = dfr.iloc[1:, :].copy()
    dfr["__DGATE__"] = dgate
    dfr["__SIGN__"]  = np.sign(dgate)
    dfr = dfr.reset_index().rename(columns={tscol: "__TS__"})
    return dfr[["__TS__", target, gate, "__DGATE__", "__SIGN__"]]

# =============================================================================
#                      PREDICT TAB: BUILD + PLOTTING
# =============================================================================
def _build_predict_tab(parent_notebook, df, inputname_site):
    tab_predict = ttk.Frame(parent_notebook)
    parent_notebook.add(tab_predict, text="1. Predictive Modeling")

    splitter = ttk.Panedwindow(tab_predict, orient="horizontal")
    splitter.pack(fill="both", expand=True)

    # LEFT controls: scrollable so the window does not need to be very tall
    controls = _scrollable_controls(splitter, width=420, padding=(10, 10))

    # RIGHT results
    results = ttk.Notebook(splitter)
    splitter.add(results, weight=1)

    Label(controls, text="Predictive Modeling", font=("TkDefaultFont", 11, "bold")).grid(sticky="w", pady=(0, 6))
    _concept_question(
        controls,
        "Can selected drivers predict the target flux out of sample?",
        "Model → target → analysis period → predictors → temporal validation. "
        "Open the completed model directly in Information Theory to evaluate information fidelity.",
    )

    # Top action bar
    bar = ttk.Frame(controls); bar.grid(sticky="ew", pady=(0, 8))
    bar.columnconfigure(5, weight=1)
    btn_run = ttk.Button(bar, text="▶ Run model"); btn_stop = ttk.Button(bar, text="■ Stop", state="disabled")
    btn_open_it = ttk.Button(bar, text="Open in IT", state="disabled")
    btn_export = ttk.Button(bar, text="⤓ Save results")
    btn_fig3b_defaults = ttk.Button(bar, text="Apply Fig 3B defaults")
    btn_run.grid(row=0, column=0, padx=(0,6)); btn_stop.grid(row=0, column=1, padx=(0,6))
    btn_open_it.grid(row=0, column=2, padx=(0,6)); btn_export.grid(row=0, column=3, padx=(0,6))
    btn_fig3b_defaults.grid(row=0, column=4, padx=(0,6))

    # Run progress shown for every model. Epoch-based models also report a more
    # detailed ETA in the Training tab.
    run_progress_box = ttk.Frame(controls)
    run_progress_box.grid(sticky="ew", pady=(0, 8))
    run_progress_box.columnconfigure(0, weight=1)
    run_progress = ttk.Progressbar(run_progress_box, mode="indeterminate")
    run_progress.grid(row=0, column=0, sticky="ew")
    run_status = StringVar(run_progress_box, value="Ready")
    run_time = StringVar(run_progress_box, value="")
    ttk.Label(run_progress_box, textvariable=run_status).grid(row=1, column=0, sticky="w", pady=(2,0))
    ttk.Label(run_progress_box, textvariable=run_time, foreground="#555555").grid(row=1, column=1, sticky="e", padx=(8,0), pady=(2,0))

    # Shared workflow variables. Resampling is displayed in Step 2; model-specific
    # options are displayed in Step 5.
    model = StringVar(controls, value="Random Forest")
    rs = IntVar(controls, value=0)

    # Data selectors (unique, alphabetized for easier search)
    cols = _sorted_unique_columns(df)

    # -------------------------------------------------------------------------
    # Step 1: model choice and model-specific settings
    # -------------------------------------------------------------------------
    default_ts = _guess_timestamp_column(cols)
    default_target = _guess_default_target(cols, timestamp_col=default_ts)
    initial_predictors, _, _ = _recommended_predictors_for_target(
        cols, default_target, timestamp_col=default_ts
    )
    preferred_gate_candidates = []
    for gate_group in ("vpd", "humidity", "soil_water", "air_temperature", "soil_temperature"):
        preferred_gate_candidates.extend(
            _find_predictor_columns(
                cols, gate_group, target=default_target,
                timestamp_col=default_ts, max_count=3,
            )
        )
    gate_default = next(
        (c for c in preferred_gate_candidates if c in initial_predictors),
        initial_predictors[0] if initial_predictors else (cols[0] if cols else ""),
    )

    box_cfg = ttk.LabelFrame(controls, text="Step 1 — Model and parameters")
    box_cfg.grid(sticky="ew", pady=6)
    box_cfg.columnconfigure(1, weight=1)
    box_cfg.columnconfigure(3, weight=1)

    ttk.Label(box_cfg, text="Model").grid(row=0, column=0, sticky="w", padx=6, pady=(6,3))
    ttk.OptionMenu(
        box_cfg, model, model.get(),
        "Linear Regression", "Random Forest", "Neural Network (MLP)",
        "LSTM (Keras)", "Hysteresis-Gate LSTM (H-LSTM)"
    ).grid(row=0, column=1, columnspan=3, sticky="ew", padx=6, pady=(6,3))

    model_settings_note = tk.Message(box_cfg, width=380, fg="#555")
    model_settings_note.grid(row=1, column=0, columnspan=4, sticky="ew", padx=6, pady=(2,4))

    reproducibility_box = ttk.LabelFrame(box_cfg, text="Reproducibility")
    reproducibility_box.grid(row=2, column=0, columnspan=4, sticky="ew", padx=6, pady=(3,6))
    ttk.Label(reproducibility_box, text="Random seed").grid(row=0, column=0, sticky="w", padx=6, pady=6)
    random_seed = StringVar(reproducibility_box, value=str(_GLOBAL_SEED))
    ttk.Entry(reproducibility_box, textvariable=random_seed, width=8).grid(row=0, column=1, sticky="w", padx=6, pady=6)
    ttk.Label(
        reproducibility_box,
        text="Keep identical across models/sites for the manuscript comparison.",
        foreground="#555",
    ).grid(row=0, column=2, sticky="w", padx=6, pady=6)

    # Linear Regression parameters
    linear_box = ttk.LabelFrame(box_cfg, text="Linear Regression settings")
    linear_fit_intercept = tk.BooleanVar(linear_box, value=True)
    ttk.Checkbutton(
        linear_box, text="Fit intercept", variable=linear_fit_intercept
    ).grid(row=0, column=0, sticky="w", padx=6, pady=6)

    # Random Forest parameters
    rf_box = ttk.LabelFrame(box_cfg, text="Random Forest settings")
    rf_n_estimators = StringVar(rf_box, value="300")
    rf_min_samples_leaf = StringVar(rf_box, value="1")
    rf_max_depth = StringVar(rf_box, value="")
    rf_max_features = StringVar(rf_box, value="All")
    rf_perm_repeats = StringVar(rf_box, value="10")
    ttk.Label(rf_box, text="Trees").grid(row=0, column=0, sticky="w", padx=6, pady=3)
    ttk.Entry(rf_box, textvariable=rf_n_estimators, width=8).grid(row=0, column=1, sticky="w", padx=6, pady=3)
    ttk.Label(rf_box, text="Min samples/leaf").grid(row=0, column=2, sticky="w", padx=6, pady=3)
    ttk.Entry(rf_box, textvariable=rf_min_samples_leaf, width=8).grid(row=0, column=3, sticky="w", padx=6, pady=3)
    ttk.Label(rf_box, text="Max depth (blank = none)").grid(row=1, column=0, sticky="w", padx=6, pady=3)
    ttk.Entry(rf_box, textvariable=rf_max_depth, width=8).grid(row=1, column=1, sticky="w", padx=6, pady=3)
    ttk.Label(rf_box, text="Max features").grid(row=1, column=2, sticky="w", padx=6, pady=3)
    ttk.OptionMenu(rf_box, rf_max_features, rf_max_features.get(), "All", "sqrt", "log2").grid(row=1, column=3, sticky="ew", padx=6, pady=3)
    ttk.Label(rf_box, text="Permutation repeats").grid(row=2, column=0, sticky="w", padx=6, pady=(3,6))
    ttk.Entry(rf_box, textvariable=rf_perm_repeats, width=8).grid(row=2, column=1, sticky="w", padx=6, pady=(3,6))

    # MLP parameters. Early stopping is intentionally not exposed because
    # sklearn's implementation uses a random internal validation subset.
    mlp_box = ttk.LabelFrame(box_cfg, text="MLP settings")
    mlp_hidden = StringVar(mlp_box, value="100")
    mlp_max_iter = StringVar(mlp_box, value="1500")
    mlp_alpha = StringVar(mlp_box, value="0.0001")
    mlp_learning_rate = StringVar(mlp_box, value="0.001")
    mlp_activation = StringVar(mlp_box, value="relu")
    mlp_batch_size = StringVar(mlp_box, value="auto")
    ttk.Label(mlp_box, text="Hidden units").grid(row=0, column=0, sticky="w", padx=6, pady=3)
    ttk.Entry(mlp_box, textvariable=mlp_hidden, width=8).grid(row=0, column=1, sticky="w", padx=6, pady=3)
    ttk.Label(mlp_box, text="Max iterations").grid(row=0, column=2, sticky="w", padx=6, pady=3)
    ttk.Entry(mlp_box, textvariable=mlp_max_iter, width=8).grid(row=0, column=3, sticky="w", padx=6, pady=3)
    ttk.Label(mlp_box, text="L2 alpha").grid(row=1, column=0, sticky="w", padx=6, pady=3)
    ttk.Entry(mlp_box, textvariable=mlp_alpha, width=10).grid(row=1, column=1, sticky="w", padx=6, pady=3)
    ttk.Label(mlp_box, text="Learning rate").grid(row=1, column=2, sticky="w", padx=6, pady=3)
    ttk.Entry(mlp_box, textvariable=mlp_learning_rate, width=10).grid(row=1, column=3, sticky="w", padx=6, pady=3)
    ttk.Label(mlp_box, text="Activation").grid(row=2, column=0, sticky="w", padx=6, pady=(3,6))
    ttk.OptionMenu(mlp_box, mlp_activation, mlp_activation.get(), "relu", "tanh", "logistic", "identity").grid(row=2, column=1, sticky="ew", padx=6, pady=(3,6))
    ttk.Label(mlp_box, text="Batch size (auto or integer)").grid(row=2, column=2, sticky="w", padx=6, pady=(3,6))
    ttk.Entry(mlp_box, textvariable=mlp_batch_size, width=10).grid(row=2, column=3, sticky="w", padx=6, pady=(3,6))

    # Shared sequence architecture parameters
    seq_box = ttk.LabelFrame(box_cfg, text="Sequence architecture")
    seq_len = StringVar(seq_box, value="20")
    hidden = StringVar(seq_box, value="96")
    epochs = StringVar(seq_box, value="100")
    ttk.Label(seq_box, text="Sequence length").grid(row=0, column=0, sticky="w", padx=6, pady=6)
    ttk.Entry(seq_box, textvariable=seq_len, width=8).grid(row=0, column=1, sticky="w", padx=6, pady=6)
    ttk.Label(seq_box, text="Hidden units").grid(row=0, column=2, sticky="w", padx=6, pady=6)
    ttk.Entry(seq_box, textvariable=hidden, width=8).grid(row=0, column=3, sticky="w", padx=6, pady=6)
    ttk.Label(seq_box, text="Epochs/fold").grid(row=0, column=4, sticky="w", padx=6, pady=6)
    ttk.Entry(seq_box, textvariable=epochs, width=8).grid(row=0, column=5, sticky="w", padx=6, pady=6)

    # Shared Keras sequence-model training parameters. These settings are used
    # identically by LSTM and H-LSTM so the only intended difference is the
    # additional delta-gate input channel supplied to H-LSTM.
    lstm_box = ttk.LabelFrame(box_cfg, text="Shared LSTM / H-LSTM training settings")
    lstm_batch_size = StringVar(lstm_box, value="32")
    lstm_learning_rate = StringVar(lstm_box, value="0.001")
    lstm_dropout = StringVar(lstm_box, value="0.0")
    lstm_gradient_clip = StringVar(lstm_box, value="1.0")
    ttk.Label(lstm_box, text="Batch size").grid(row=0, column=0, sticky="w", padx=6, pady=6)
    ttk.Entry(lstm_box, textvariable=lstm_batch_size, width=8).grid(row=0, column=1, sticky="w", padx=6, pady=6)
    ttk.Label(lstm_box, text="Learning rate").grid(row=0, column=2, sticky="w", padx=6, pady=6)
    ttk.Entry(lstm_box, textvariable=lstm_learning_rate, width=10).grid(row=0, column=3, sticky="w", padx=6, pady=6)
    ttk.Label(lstm_box, text="Dropout (0–0.9)").grid(row=0, column=4, sticky="w", padx=6, pady=6)
    ttk.Entry(lstm_box, textvariable=lstm_dropout, width=8).grid(row=0, column=5, sticky="w", padx=6, pady=6)
    ttk.Label(lstm_box, text="Gradient clip (0 = off)").grid(row=1, column=0, sticky="w", padx=6, pady=(3,6))
    ttk.Entry(lstm_box, textvariable=lstm_gradient_clip, width=8).grid(row=1, column=1, sticky="w", padx=6, pady=(3,6))

    # Gate choice is made in Step 4. H-LSTM deliberately has no separate
    # optimizer controls: it uses the same Keras training configuration above.
    hlstm_box = ttk.LabelFrame(box_cfg, text="H-LSTM controlled difference")
    gate_var = StringVar(hlstm_box, value=gate_default)
    hlstm_learning_rate = lstm_learning_rate
    hlstm_dropout = lstm_dropout
    hlstm_gradient_clip = lstm_gradient_clip
    tk.Message(
        hlstm_box, width=360, fg="#444",
        text=(
            "H-LSTM uses the same Keras LSTM, optimizer, batch size, epochs, "
            "dropout, gradient clipping, scaling, and seed as LSTM. Its only "
            "additional input is Δgate (current gate minus previous gate)."
        ),
    ).grid(row=0, column=0, sticky="ew", padx=6, pady=6)

    model_frames = [linear_box, rf_box, mlp_box, seq_box, lstm_box, hlstm_box]

    def _update_model_settings_note(*_):
        for frame in model_frames:
            frame.grid_remove()
        m = model.get()
        if m == "Linear Regression":
            msg = "Linear baseline. Fit-intercept is the only exposed model parameter."
            linear_box.grid(row=3, column=0, columnspan=4, sticky="ew", padx=6, pady=(3,6))
        elif m == "Random Forest":
            msg = "Nonlinear tree ensemble. Importance is evaluated by permutation on held-out folds."
            rf_box.grid(row=3, column=0, columnspan=4, sticky="ew", padx=6, pady=(3,6))
        elif m == "Neural Network (MLP)":
            msg = "Feed-forward neural benchmark with fold-specific predictor and target scaling."
            mlp_box.grid(row=3, column=0, columnspan=4, sticky="ew", padx=6, pady=(3,6))
        elif m == "LSTM (Keras)":
            msg = "Multistep Keras LSTM using the shared sequence-model training settings."
            if Sequential is None:
                msg += " TensorFlow/Keras is not available in this Python environment."
            seq_box.grid(row=3, column=0, columnspan=4, sticky="ew", padx=6, pady=(3,3))
            lstm_box.grid(row=4, column=0, columnspan=4, sticky="ew", padx=6, pady=(3,6))
        else:
            msg = (
                "Controlled H-LSTM: the same Keras LSTM and training settings as LSTM, "
                "with only Δgate added as an input channel. The gate must also be a common predictor."
            )
            if Sequential is None:
                msg += " TensorFlow/Keras is not available in this Python environment."
            seq_box.grid(row=3, column=0, columnspan=4, sticky="ew", padx=6, pady=(3,3))
            lstm_box.grid(row=4, column=0, columnspan=4, sticky="ew", padx=6, pady=(3,3))
            hlstm_box.grid(row=5, column=0, columnspan=4, sticky="ew", padx=6, pady=(3,6))
        model_settings_note.configure(text=msg)

    try:
        model.trace_add("write", _update_model_settings_note)
    except Exception:
        pass
    _update_model_settings_note()

    # -------------------------------------------------------------------------
    # Step 2: target
    # -------------------------------------------------------------------------
    box_target = ttk.LabelFrame(controls, text="Step 2 — Target flux")
    box_target.grid(sticky="ew", pady=6)
    box_target.columnconfigure(0, weight=1)

    ttk.Label(box_target, text="Target flux / variable (y)").grid(row=0, column=0, sticky="w", padx=6, pady=(6,0))
    y_var = StringVar(box_target, value=default_target)
    ttk.OptionMenu(box_target, y_var, y_var.get(), *cols).grid(row=1, column=0, columnspan=2, sticky="ew", padx=6, pady=(0,4))
    target_units = StringVar(box_target, value="")
    target_sign_convention = StringVar(box_target, value="")
    ttk.Label(box_target, text="Units (optional)").grid(row=2, column=0, sticky="w", padx=6, pady=3)
    ttk.Entry(box_target, textvariable=target_units).grid(row=2, column=1, sticky="ew", padx=6, pady=3)
    ttk.Label(box_target, text="Sign convention (optional)").grid(row=3, column=0, sticky="w", padx=6, pady=(3,6))
    ttk.Entry(box_target, textvariable=target_sign_convention).grid(row=3, column=1, sticky="ew", padx=6, pady=(3,6))

    # -------------------------------------------------------------------------
    # Step 3: timestamp and temporal aggregation
    # -------------------------------------------------------------------------
    box_time = ttk.LabelFrame(controls, text="Step 3 — Analysis period and aggregation")
    box_time.grid(sticky="ew", pady=6)
    box_time.columnconfigure(1, weight=1)

    ttk.Label(box_time, text="Timestamp column").grid(row=0, column=0, sticky="w", padx=6, pady=(6,3))
    ts_var = StringVar(box_time, value=default_ts)
    ttk.OptionMenu(box_time, ts_var, ts_var.get(), *cols).grid(row=0, column=1, columnspan=3, sticky="ew", padx=6, pady=(6,3))

    analysis_start, analysis_end, use_full_period, analysis_period_msg = _add_analysis_period_controls(
        box_time, df, ts_var, row_start=1, width=380
    )

    ttk.Label(box_time, text="Time resolution").grid(row=4, column=0, sticky="w", padx=6, pady=3)
    ttk.Radiobutton(box_time, text="Native", variable=rs, value=0).grid(row=4, column=1, sticky="w", padx=6, pady=3)
    ttk.Radiobutton(box_time, text="Daily", variable=rs, value=1).grid(row=4, column=2, sticky="w", padx=6, pady=3)
    ttk.Radiobutton(box_time, text="Weekly", variable=rs, value=2).grid(row=4, column=3, sticky="w", padx=6, pady=3)
    tk.Message(
        box_time,
        width=380,
        fg="#444",
        text=(
            "Native = original timestep; Daily/Weekly = aggregated before modeling. "
            "Detected interval accumulations are summed; states/rates are averaged."
        )
    ).grid(row=5, column=0, columnspan=4, sticky="ew", padx=6, pady=(3,3))

    ttk.Label(box_time, text="Minimum within-period coverage").grid(row=6, column=0, sticky="w", padx=6, pady=3)
    min_coverage_pct = StringVar(box_time, value="75%")
    ttk.Entry(box_time, textvariable=min_coverage_pct, width=8).grid(row=6, column=1, sticky="w", padx=6, pady=3)
    ttk.Label(box_time, text="Additional columns to SUM").grid(row=7, column=0, sticky="w", padx=6, pady=3)
    sum_override_text = StringVar(box_time, value="")
    ttk.Entry(box_time, textvariable=sum_override_text).grid(row=7, column=1, columnspan=3, sticky="ew", padx=6, pady=3)
    ttk.Label(box_time, text="Columns to force MEAN").grid(row=8, column=0, sticky="w", padx=6, pady=3)
    mean_override_text = StringVar(box_time, value="")
    ttk.Entry(box_time, textvariable=mean_override_text).grid(row=8, column=1, columnspan=3, sticky="ew", padx=6, pady=3)
    tk.Message(
        box_time, width=380, fg="#555",
        text="Aggregation overrides are optional comma-separated exact column names. Use them when automatic precipitation/management detection is wrong."
    ).grid(row=9, column=0, columnspan=4, sticky="ew", padx=6, pady=(3,6))

    # Optional FC sensitivity analysis. The locked main Figure 3B setting is All.
    ttk.Separator(box_time, orient="horizontal").grid(
        row=10, column=0, columnspan=4, sticky="ew", padx=6, pady=(4,5)
    )
    ttk.Label(box_time, text="FC day/night sensitivity").grid(
        row=11, column=0, sticky="w", padx=6, pady=3
    )
    diurnal_mode = StringVar(box_time, value="All observations (main Figure 3B)")
    diurnal_options = [
        "All observations (main Figure 3B)",
        "Daytime sensitivity",
        "Nighttime sensitivity",
    ]
    ttk.OptionMenu(
        box_time, diurnal_mode, diurnal_mode.get(), *diurnal_options
    ).grid(row=11, column=1, columnspan=3, sticky="ew", padx=6, pady=3)

    daylight_variable = StringVar(box_time, value=_guess_daylight_column(cols))
    daylight_threshold = StringVar(box_time, value="10")
    ttk.Label(box_time, text="Radiation classifier").grid(
        row=12, column=0, sticky="w", padx=6, pady=3
    )
    daylight_menu = ttk.OptionMenu(
        box_time, daylight_variable, daylight_variable.get(), *cols
    )
    daylight_menu.grid(row=12, column=1, columnspan=2, sticky="ew", padx=6, pady=3)
    ttk.Label(box_time, text="Threshold").grid(
        row=12, column=3, sticky="w", padx=6, pady=3
    )
    daylight_threshold_entry = ttk.Entry(
        box_time, textvariable=daylight_threshold, width=8
    )
    daylight_threshold_entry.grid(row=13, column=3, sticky="w", padx=6, pady=(0,3))
    tk.Message(
        box_time, width=380, fg="#555",
        text=(
            "Optional FC/NEE sensitivity only. Filtering occurs before daily aggregation: "
            "daytime = radiation > threshold; nighttime = radiation ≤ threshold. "
            "The Figure 3B defaults reset this option to All observations."
        ),
    ).grid(row=13, column=0, columnspan=3, sticky="ew", padx=6, pady=(0,6))

    def _update_diurnal_controls(*_):
        active = not str(diurnal_mode.get()).lower().startswith("all")
        state = "normal" if active else "disabled"
        daylight_menu.configure(state=state)
        daylight_threshold_entry.configure(state=state)

    try:
        diurnal_mode.trace_add("write", _update_diurnal_controls)
    except Exception:
        pass
    _update_diurnal_controls()

    # -------------------------------------------------------------------------
    # Step 4: predictors and target-based presets
    # -------------------------------------------------------------------------
    box_predictors = ttk.LabelFrame(controls, text="Step 4 — Predictor selection")
    box_predictors.grid(sticky="ew", pady=6)
    box_predictors.columnconfigure(0, weight=1)

    ttk.Label(box_predictors, text="Literature-informed predictors (review/edit)").grid(sticky="w", padx=6, pady=(6,0))
    pred_list_frame = ttk.Frame(box_predictors)
    pred_list_frame.grid(sticky="ew", padx=6, pady=(0,6))
    pred_list_frame.columnconfigure(0, weight=1)
    lb_X = Listbox(pred_list_frame, selectmode=MULTIPLE, exportselection=False, height=7)
    pred_scroll = ttk.Scrollbar(pred_list_frame, orient="vertical", command=lb_X.yview)
    lb_X.configure(yscrollcommand=pred_scroll.set)
    for i, c in enumerate(cols):
        lb_X.insert(i, c)
    lb_X.grid(row=0, column=0, sticky="ew")
    pred_scroll.grid(row=0, column=1, sticky="ns")

    preset_msg = tk.Message(
        box_predictors,
        width=380,
        text="Preset predictors appear after target selection.",
        fg="#444"
    )
    preset_msg.grid(row=2, column=0, columnspan=3, sticky="ew", padx=6, pady=(0,4))
    auto_preset_var = tk.BooleanVar(box_predictors, value=True)
    # This flag is activated only by Apply Fig 3B defaults. Normal literature-
    # informed presets remain unchanged throughout the rest of MeaningFlux.
    fig3b_predictor_mode = tk.BooleanVar(box_predictors, value=False)
    ttk.Checkbutton(
        box_predictors,
        text="Auto-select when target changes",
        variable=auto_preset_var
    ).grid(row=3, column=0, sticky="w", padx=6, pady=(0,6))

    def _excluded_from_fig3b_auto_selection(column):
        """Exclude only TS, SWC, and P families from the Fig 3B auto-preset.

        Exact EC base-name matching avoids accidentally excluding PA, PPFD,
        precipitation-independent variables, or other columns containing the
        same letters. Users may still select these variables manually.
        """
        parts = _split_name_parts(column)
        return bool(parts and parts[0] in {"TS", "SWC", "P"})

    def _fig3b_keep_one_radiation(selected):
        """Keep one primary radiation predictor in the Figure 3B auto-preset.

        Preference: incoming shortwave/total PPFD or PAR, then net radiation,
        then diffuse PPFD. The rule changes only automatic Figure 3B selection;
        users may add another radiation variable manually afterward.
        """
        selected = list(selected)
        radiation = []
        for c in selected:
            if (_find_predictor_columns([c], "radiation_shortwave", target=y_var.get(),
                                        timestamp_col=ts_var.get(), max_count=1)
                    or _find_predictor_columns([c], "radiation_net", target=y_var.get(),
                                               timestamp_col=ts_var.get(), max_count=1)):
                radiation.append(c)
        if len(radiation) <= 1:
            return selected

        def _priority(c):
            u = str(c).upper()
            if any(k in u for k in ("SW_IN", "SWIN", "PPFD_IN", "PAR_IN")):
                return 0
            if ("PPFD" in u or "PAR" in u) and "DIF" not in u and "OUT" not in u:
                return 1
            if any(k in u for k in ("NETRAD", "RNET", "NET_RADIATION")) or u == "RN":
                return 2
            if "PPFD_DIF" in u or ("PPFD" in u and "DIF" in u):
                return 3
            return 4

        keep = sorted(radiation, key=lambda c: (_priority(c), selected.index(c)))[0]
        return [c for c in selected if c not in radiation or c == keep]

    def _apply_predict_preset(show_note=False):
        selected, family, guide = _recommended_predictors_for_target(
            cols, y_var.get(), timestamp_col=ts_var.get()
        )
        if fig3b_predictor_mode.get():
            selected = [c for c in selected if not _excluded_from_fig3b_auto_selection(c)]
            selected = _fig3b_keep_one_radiation(selected)
            guide += (
                "\n\nFigure 3B auto-selection: TS, SWC, and P columns are not "
                "selected automatically. They remain available for manual selection."
            )
        lb_X.selection_clear(0, "end")
        for i, c in enumerate(cols):
            if c in selected:
                lb_X.selection_set(i)
        preset_msg.configure(text=guide)
        if gate_var.get() not in selected:
            gate_candidates = []
            for gate_group in ("vpd", "humidity", "soil_water", "air_temperature", "soil_temperature"):
                gate_candidates.extend(
                    _find_predictor_columns(
                        cols, gate_group, target=y_var.get(),
                        timestamp_col=ts_var.get(), max_count=3,
                    )
                )
            suggested_gate = next((c for c in gate_candidates if c in selected), None)
            if suggested_gate is not None:
                gate_var.set(suggested_gate)
        if show_note:
            messagebox.showinfo(
                "Literature-informed predictor preset",
                f"Applied preset for {family}.\n\nSelected {len(selected)} predictor(s):\n" +
                (", ".join(selected) if selected else "None found automatically.") +
                "\n\nPlease review the selection before running the model."
            )

    ttk.Button(
        box_predictors,
        text="Apply preset now",
        command=lambda: _apply_predict_preset(show_note=True)
    ).grid(row=3, column=1, sticky="e", padx=6, pady=(0,6))
    ttk.Button(
        box_predictors,
        text="Clear predictors",
        command=lambda: lb_X.selection_clear(0, "end")
    ).grid(row=3, column=2, sticky="e", padx=(0,6), pady=(0,6))
    ttk.Label(
        box_predictors,
        text="H-LSTM gate / common fold reference",
    ).grid(row=4, column=0, sticky="w", padx=6, pady=(3,3))
    ttk.OptionMenu(
        box_predictors, gate_var, gate_var.get(), *cols
    ).grid(row=4, column=1, columnspan=2, sticky="ew", padx=6, pady=(3,3))
    tk.Message(
        box_predictors, width=380, fg="#555",
        text=(
            "For blocked model comparison, select the gate before running any model and keep it unchanged. "
            "The gate must be in the common predictor list. MeaningFlux then defines all fold cutoffs from "
            "the same gap-safe H-LSTM-eligible endpoint timeline."
        ),
    ).grid(row=5, column=0, columnspan=3, sticky="ew", padx=6, pady=(0,5))
    ttk.Button(
        box_predictors,
        text="Check overlap",
        command=lambda: _preview_predict_split(show_overlap_tab=True)
    ).grid(row=6, column=0, columnspan=3, sticky="ew", padx=6, pady=(0,6))

    def _on_predict_target_changed(*_):
        if auto_preset_var.get():
            _apply_predict_preset(show_note=False)
        else:
            _apply_predictor_preset_to_listbox(
                lb_X, cols, y_var.get(), timestamp_col=ts_var.get(), status_widget=preset_msg, select=False
            )

    try:
        y_var.trace_add("write", _on_predict_target_changed)
        ts_var.trace_add("write", lambda *_: _on_predict_target_changed())
    except Exception:
        pass
    _apply_predict_preset(show_note=False)

    # Step 5: temporal validation controls
    box_split = ttk.LabelFrame(controls, text="Step 5 — Temporal validation")
    box_split.grid(sticky="ew", pady=6)
    box_split.columnconfigure(1, weight=1)

    ttk.Label(box_split, text="Split strategy").grid(row=0, column=0, sticky="w", padx=6, pady=(6,3))
    split_strategy = StringVar(box_split, value="Blocked time-series CV (expanding window)")
    split_options = [
        "Blocked time-series CV (expanding window)",
        "Chronological holdout (train earlier, test later)",
        "Random fixed holdout (same rows each run)",
        "Random new holdout (exploratory; changes each run)",
    ]
    ttk.OptionMenu(box_split, split_strategy, split_strategy.get(), *split_options).grid(row=0, column=1, columnspan=2, sticky="ew", padx=6, pady=(6,3))

    ttk.Label(box_split, text="Holdout test fraction").grid(row=1, column=0, sticky="w", padx=6, pady=3)
    test_fraction = StringVar(box_split, value="20%")
    test_fraction_entry = ttk.Entry(box_split, textvariable=test_fraction, width=8)
    test_fraction_entry.grid(row=1, column=1, sticky="w", padx=6, pady=3)

    ttk.Label(box_split, text="Blocked CV folds").grid(row=2, column=0, sticky="w", padx=6, pady=3)
    cv_folds = StringVar(box_split, value="5")
    cv_folds_entry = ttk.Entry(box_split, textvariable=cv_folds, width=8)
    cv_folds_entry.grid(row=2, column=1, sticky="w", padx=6, pady=3)

    ttk.Label(box_split, text="Initial training fraction").grid(row=3, column=0, sticky="w", padx=6, pady=3)
    initial_train_fraction = StringVar(box_split, value="50%")
    initial_train_entry = ttk.Entry(box_split, textvariable=initial_train_fraction, width=8)
    initial_train_entry.grid(row=3, column=1, sticky="w", padx=6, pady=3)
    ttk.Button(box_split, text="Preview validation", command=lambda: _preview_predict_split()).grid(row=1, column=2, rowspan=3, sticky="e", padx=6, pady=3)

    use_common_sequence_folds = tk.BooleanVar(box_split, value=False)
    ttk.Checkbutton(
        box_split,
        text="Use common gap-safe sequence endpoints (required for Fig 3B)",
        variable=use_common_sequence_folds,
    ).grid(row=4, column=0, columnspan=3, sticky="w", padx=6, pady=(3,3))

    split_msg = tk.Message(
        box_split, width=380, fg="#444",
        text=(
            "Blocked expanding-window validation is recommended for the manuscript. "
            "It trains only on earlier observations and tests consecutive future blocks."
        )
    )
    split_msg.grid(row=5, column=0, columnspan=3, sticky="ew", padx=6, pady=(3,6))

    def _update_validation_control_states(*_):
        blocked = "Blocked time-series CV" in split_strategy.get()
        test_fraction_entry.configure(state="disabled" if blocked else "normal")
        cv_folds_entry.configure(state="normal" if blocked else "disabled")
        initial_train_entry.configure(state="normal" if blocked else "disabled")

    try:
        split_strategy.trace_add("write", _update_validation_control_states)
    except Exception:
        pass
    _update_validation_control_states()

    def _apply_fig3b_defaults():
        """Apply locked manuscript defaults and the Fig 3B-only predictor rule."""
        # Activate and apply the restricted auto-preset only for Figure 3B.
        # Users may deliberately add excluded variables back afterward.
        previous_gate = gate_var.get()
        fig3b_predictor_mode.set(True)
        _apply_predict_preset(show_note=False)
        selected_now = [cols[i] for i in lb_X.curselection()]
        if previous_gate in selected_now:
            gate_var.set(previous_gate)
        random_seed.set("42")
        rs.set(1)  # Daily
        diurnal_mode.set("All observations (main Figure 3B)")
        min_coverage_pct.set("75%")
        sum_override_text.set("")
        mean_override_text.set("")
        split_strategy.set("Blocked time-series CV (expanding window)")
        use_common_sequence_folds.set(True)
        cv_folds.set("5")
        initial_train_fraction.set("50%")
        test_fraction.set("20%")
        linear_fit_intercept.set(True)
        rf_n_estimators.set("300")
        rf_min_samples_leaf.set("1")
        rf_max_depth.set("")
        rf_max_features.set("All")
        rf_perm_repeats.set("10")
        mlp_hidden.set("100")
        mlp_max_iter.set("1500")
        mlp_alpha.set("0.0001")
        mlp_learning_rate.set("0.001")
        mlp_activation.set("relu")
        mlp_batch_size.set("auto")
        seq_len.set("7")
        hidden.set("96")
        epochs.set("100")
        lstm_batch_size.set("32")
        lstm_learning_rate.set("0.001")
        lstm_dropout.set("0.0")
        hlstm_learning_rate.set("0.001")
        hlstm_dropout.set("0.0")
        hlstm_gradient_clip.set("1.0")
        _update_validation_control_states()
        messagebox.showinfo(
            "Figure 3B defaults",
            "Applied the locked Figure 3B model, aggregation, validation, and reproducibility defaults. "
            "TS, SWC, and P were removed from automatic Figure 3B predictor selection only; "
            "one primary radiation variable was retained; all variables remain available manually. "
            "Sequence length is 7 days, and five folds use common gap-safe sequence endpoints across models. "
            "The target and analysis period were not changed."
        )

    btn_fig3b_defaults.configure(command=_apply_fig3b_defaults)

    # Visible scientific workflow: model → target → data → predictors → validation.
    box_cfg.grid_configure(row=4)
    box_target.grid_configure(row=5)
    box_time.grid_configure(row=6)
    box_predictors.grid_configure(row=7)
    box_split.grid_configure(row=8)

    def _preview_predict_split(show_overlap_tab=False):
        try:
            all_cols_preview = _sorted_unique_columns(df)
            sel_preview = [all_cols_preview[i] for i in lb_X.curselection()]
            tgt_preview = y_var.get(); ts_preview = ts_var.get()
            if not sel_preview:
                split_msg.configure(text="No predictors selected yet. Select/review predictors before previewing the split.")
                return
            diurnal_mode_preview = str(diurnal_mode.get() or "All observations")
            diurnal_active_preview = not diurnal_mode_preview.lower().startswith("all")
            if diurnal_active_preview and rs.get() != 1:
                raise ValueError(
                    "Daytime/nighttime sensitivity is defined as a daily product. Select Daily resolution."
                )
            daylight_col_preview = str(daylight_variable.get() or "").strip()
            daylight_threshold_preview = float(daylight_threshold.get() or 10.0)
            needed_preview = list(dict.fromkeys(
                sel_preview + [tgt_preview, ts_preview] +
                ([daylight_col_preview] if diurnal_active_preview else [])
            ))
            df_preview_period = _filter_analysis_period(
                df[needed_preview], ts_preview,
                analysis_start.get(), analysis_end.get(),
            )
            selected_period_text = _fmt_time_range(df_preview_period[ts_preview])
            n_period_rows = int(len(df_preview_period))
            df_preview, diurnal_meta_preview = _apply_diurnal_subset(
                df_preview_period, ts_preview, diurnal_mode_preview,
                daylight_col_preview, daylight_threshold_preview,
            )
            n_subset_rows = int(len(df_preview))

            # Predictor/target overlap diagnostics before resampling. This explains
            # why the model-ready period may be much shorter than the selected period.
            overlap_df, complete_mask_raw, limiting_text = _predictor_overlap_summary(
                df_preview, sel_preview, tgt_preview, ts_preview
            )

            # Populate the Data overlap tab if it has been created.
            try:
                for item in overlap_tree.get_children():
                    overlap_tree.delete(item)
                for _, row in overlap_df.iterrows():
                    gain = row.get("row_gain_if_removed", "")
                    gain_txt = str(gain) if isinstance(gain, str) else f"{int(gain):,}"
                    overlap_tree.insert("", "end", values=(
                        row.get("variable", ""),
                        row.get("role", ""),
                        f"{float(row.get('nonmissing_pct', 0)):.1f}",
                        f"{int(row.get('nonmissing_rows', 0)):,}",
                        row.get("valid_period", ""),
                        gain_txt,
                        row.get("note", ""),
                    ))
                complete_rows = int(complete_mask_raw.sum())
                coverage_pct = 100.0 * complete_rows / max(1, n_subset_rows)
                subset_note = (
                    f"{diurnal_meta_preview['diurnal_mode']} rows: {n_subset_rows:,}   |   "
                    if diurnal_active_preview else ""
                )
                overlap_msg.configure(
                    text=(
                        f"Analysis rows: {n_period_rows:,}   |   {subset_note}"
                        f"Complete target + predictor rows: {complete_rows:,} "
                        f"({coverage_pct:.1f}%)   |   {limiting_text}"
                    )
                )

                if show_overlap_tab:
                    results.select(results_tabs["Data overlap"])
            except Exception:
                pass

            coverage_preview = _parse_fraction(
                min_coverage_pct.get(), _MIN_RESAMPLE_COVERAGE,
                min_value=0.10, max_value=1.00,
                label="minimum within-period coverage",
            )
            sum_overrides_preview = _parse_column_overrides(sum_override_text.get(), df.columns)
            mean_overrides_preview = _parse_column_overrides(mean_override_text.get(), df.columns)
            overlap_overrides = sorted(set(sum_overrides_preview) & set(mean_overrides_preview))
            if overlap_overrides:
                raise ValueError(
                    "A column cannot be forced to both SUM and MEAN: " + ", ".join(overlap_overrides)
                )
            dfw_preview_all, rs_label_preview = _resample_view(
                df_preview, rs.get(), ts_preview,
                min_coverage=coverage_preview,
                sum_columns=sum_overrides_preview,
                mean_columns=mean_overrides_preview,
                coverage_relative_to_available_rows=diurnal_active_preview,
                drop_complete_cases=False,
            )
            dfw_preview_complete = (
                dfw_preview_all[sel_preview + [tgt_preview]]
                .apply(pd.to_numeric, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna(how="any")
            )
            preview_timestamps = pd.Index(dfw_preview_complete.index)
            n_model_rows = len(dfw_preview_complete)
            sequence_note = ""
            gate_preview = str(gate_var.get() or "").strip()
            blocked_preview = "Blocked time-series CV" in split_strategy.get()
            use_common_preview = blocked_preview and bool(use_common_sequence_folds.get())
            expected_preview_step = (
                pd.Timedelta(days=1) if rs.get() == 1 else
                pd.Timedelta(days=7) if rs.get() == 2 else None
            )

            common_endpoint_ts = None
            if use_common_preview:
                if not gate_preview or gate_preview not in sel_preview:
                    raise ValueError(
                        "For Figure 3B, select the H-LSTM gate and include that same variable "
                        "in the common predictor list before previewing validation."
                    )
                needed_common = list(dict.fromkeys(
                    sel_preview + [gate_preview, tgt_preview, ts_preview]
                ))
                df_h_common = _prepare_h_lstm_frame(
                    df_preview[needed_common], sel_preview, tgt_preview,
                    gate_preview, ts_preview, rs.get(),
                    min_coverage=coverage_preview,
                    sum_columns=sum_overrides_preview,
                    mean_columns=mean_overrides_preview,
                    coverage_relative_to_available_rows=diurnal_active_preview,
                )
                _, _, common_endpoint_ts, common_diag = _make_seq_data(
                    df_h_common, sel_preview, tgt_preview, gate_preview,
                    int(seq_len.get()), return_diagnostics=True,
                    expected_step=expected_preview_step,
                )
                common_endpoint_ts = pd.Index(common_endpoint_ts)
                preview_timestamps = common_endpoint_ts
                n_model_rows = len(common_endpoint_ts)
                sequence_note = (
                    f" Figure 3B uses {n_model_rows:,} common gap-safe endpoints from "
                    f"{int(seq_len.get())}-step predictor sequences; the target is required "
                    "only at each endpoint, and all five models use these exact dates."
                )
            elif model.get() == "LSTM (Keras)":
                _, yp, tsp = _make_lstm_sequence_data(
                    dfw_preview_all, sel_preview, tgt_preview, int(seq_len.get()),
                    expected_step=expected_preview_step,
                )
                n_model_rows = len(yp)
                preview_timestamps = pd.Index(tsp)
                sequence_note = (
                    f" LSTM uses {int(seq_len.get())}-step contiguous predictor sequences; "
                    "the target is required only at the endpoint and gap-crossing windows are rejected."
                )
            elif model.get() == "Hysteresis-Gate LSTM (H-LSTM)":
                if not gate_preview or gate_preview not in sel_preview:
                    raise ValueError("The H-LSTM gate must also be selected as a predictor.")
                needed = list(dict.fromkeys(
                    sel_preview + [gate_preview, tgt_preview, ts_preview]
                ))
                df_h_preview = _prepare_h_lstm_frame(
                    df_preview[needed], sel_preview, tgt_preview, gate_preview,
                    ts_preview, rs.get(), min_coverage=coverage_preview,
                    sum_columns=sum_overrides_preview,
                    mean_columns=mean_overrides_preview,
                    coverage_relative_to_available_rows=diurnal_active_preview,
                )
                _, yp, tsp = _make_seq_data(
                    df_h_preview, sel_preview, tgt_preview, gate_preview,
                    int(seq_len.get()), expected_step=expected_preview_step,
                )
                n_model_rows = len(yp)
                preview_timestamps = pd.Index(tsp)
                sequence_note = (
                    f" H-LSTM uses {int(seq_len.get())}-step contiguous predictor sequences and "
                    f"Δ{gate_preview}; the target is required only at the endpoint."
                )

            validation_reference_timestamps = preview_timestamps
            validation_reference_text = (
                "common gap-safe sequence endpoints"
                if use_common_preview else "the current model-ready timeline"
            )

            frac_preview = _parse_test_fraction(test_fraction.get(), 0.20)
            folds_preview = max(2, int(cv_folds.get() or 5))
            initial_fraction_preview = _parse_fraction(
                initial_train_fraction.get(), 0.50,
                min_value=0.20, max_value=0.90,
                label="initial training fraction",
            )
            validation_splits = _make_timestamp_aligned_validation_splits(
                preview_timestamps, validation_reference_timestamps,
                split_strategy.get(), frac_preview, folds_preview,
                initial_train_fraction=initial_fraction_preview,
            )
            first_fold, first_tr, first_ts = validation_splits[0]
            last_fold, last_tr, last_ts = validation_splits[-1]
            all_test = np.concatenate([te for _, _, te in validation_splits])
            info_preview = {
                "split_strategy": split_strategy.get(),
                "test_fraction": frac_preview,
                "n_total": int(n_model_rows),
                "n_train": f"{len(first_tr):,}–{len(last_tr):,}" if len(validation_splits) > 1 else int(len(first_tr)),
                "n_test": int(len(all_test)),
                "train_period": _fmt_time_range(preview_timestamps[first_tr]),
                "test_period": _fmt_time_range(preview_timestamps[all_test]),
                "selected_analysis_period": selected_period_text,
                "complete_case_period": _fmt_time_range(preview_timestamps),
                "n_period_rows": n_period_rows,
                "n_folds": len(validation_splits),
                "initial_train_fraction": initial_fraction_preview,
            }
            preview_text = _split_summary_text(info_preview)
            if len(validation_splits) > 1:
                preview_text += (
                    f"\nValidation folds: {len(validation_splits)} expanding, non-overlapping future blocks "
                    f"defined from {validation_reference_text}."
                )
            preview_text += sequence_note
            if diurnal_active_preview:
                preview_text += (
                    f"\nDiurnal sensitivity: {diurnal_meta_preview['diurnal_mode']} using "
                    f"{daylight_col_preview} threshold {daylight_threshold_preview:g}; "
                    f"{n_subset_rows:,} subdaily rows retained before daily aggregation."
                )
            preview_text += "\nResampling: " + _resampling_summary(
                sel_preview + [tgt_preview], min_coverage=coverage_preview,
                sum_columns=sum_overrides_preview,
                mean_columns=mean_overrides_preview,
            )
            if n_model_rows < max(10, 0.25 * n_period_rows):
                preview_text += "\nNote: Complete model-ready cases are much fewer than selected rows. " + limiting_text
            split_msg.configure(text=preview_text)
        except Exception as e:
            split_msg.configure(text=f"Could not preview split: {e}")

    # Guidance lives in the Guide tab, model note, resampling note, split panel, and result tabs.

    # Results notebook tabs
    results_tabs = {}
    result_tab_labels = [
        ("Series", "Series"),
        ("Scatter", "Scatter"),
        ("Metrics", "Metrics"),
        ("Feature Importance", "Features"),
        ("Residuals", "Residuals"),
        ("Compare", "Compare"),
        ("Data overlap", "Overlap"),
        ("Training", "Train"),
        ("Interpretation", "Explain"),
        ("Guide", "Guide"),
    ]
    for key, label in result_tab_labels:
        fr = ttk.Frame(results)
        results.add(fr, text=label)
        results_tabs[key] = fr

    # Guide tab: explains the ML-to-IT workflow before the user runs anything.
    guide_box = ttk.Frame(results_tabs["Guide"], padding=10)
    guide_box.pack(fill="both", expand=True)
    guide_text = tk.Text(guide_box, wrap="word", height=20)
    guide_text.pack(fill="both", expand=True)
    guide_text.insert("1.0", _ml_workflow_guide_text())
    guide_text.configure(state="disabled")

    # Interpretation tab: updated after every model run.
    interp_box = ttk.Frame(results_tabs["Interpretation"], padding=10)
    interp_box.pack(fill="both", expand=True)
    interp_text = tk.Text(interp_box, wrap="word", height=20)
    interp_text.pack(fill="both", expand=True)
    interp_text.insert("1.0", "Run a model to see plain-language interpretation of the metrics and next steps for the Information Theory toolbox.\n")
    interp_text.configure(state="disabled")

    # Matplotlib canvases
    series_header = ttk.Frame(results_tabs["Series"], padding=(8, 6, 8, 0))
    series_header.pack(fill="x")
    lbl_series_note = ttk.Label(
        series_header,
        text=(
            "Showing model testing data."
        ),
        foreground="#444444",
        font=("TkDefaultFont", 9, "bold"),
    )
    lbl_series_note.pack(anchor="w")

    fig_series, ax_series = plt.subplots(figsize=(8.6, 3.2))
    can_series = FigureCanvasTkAgg(fig_series, master=results_tabs["Series"]); can_series.draw(); can_series.get_tk_widget().pack(fill="both", expand=True)

    fig_scat, ax_scat = plt.subplots(figsize=(8.6, 3.2))
    can_scat = FigureCanvasTkAgg(fig_scat, master=results_tabs["Scatter"]); can_scat.draw(); can_scat.get_tk_widget().pack(fill="both", expand=True)

    fig_resid, ax_resid = plt.subplots(figsize=(8.6, 3.2))
    can_resid = FigureCanvasTkAgg(fig_resid, master=results_tabs["Residuals"]); can_resid.draw(); can_resid.get_tk_widget().pack(fill="both", expand=True)

    fig_imp, ax_imp = plt.subplots(figsize=(8.6, 3.2))
    can_imp = FigureCanvasTkAgg(fig_imp, master=results_tabs["Feature Importance"]); can_imp.draw(); can_imp.get_tk_widget().pack(fill="both", expand=True)

    # Data overlap tab: shows which predictors shorten the usable ML period.
    overlap_frame = ttk.Frame(results_tabs["Data overlap"], padding=10)
    overlap_frame.pack(fill="both", expand=True)
    overlap_msg = tk.Message(
        overlap_frame,
        width=840,
        fg="#444",
        text=(
            "Click Check overlap or Preview split to see how much the selected target and predictors overlap in the analysis period. "
            "The ML model can only use rows where the target and all selected predictors are present."
        ),
    )
    overlap_msg.pack(fill="x", anchor="w", pady=(0, 6))

    overlap_cols = ("variable", "role", "nonmissing_pct", "nonmissing_rows", "valid_period", "gain", "note")
    overlap_tree = ttk.Treeview(overlap_frame, columns=overlap_cols, show="headings", height=12)
    overlap_headings = {
        "variable": "Variable",
        "role": "Role",
        "nonmissing_pct": "% present",
        "nonmissing_rows": "Rows present",
        "valid_period": "Valid period",
        "gain": "Rows gained if removed",
        "note": "Note",
    }
    overlap_widths = {"variable": 150, "role": 80, "nonmissing_pct": 80, "nonmissing_rows": 100, "valid_period": 190, "gain": 130, "note": 150}
    for c in overlap_cols:
        overlap_tree.heading(c, text=overlap_headings[c])
        overlap_tree.column(c, width=overlap_widths[c], anchor="w")
    overlap_scroll = ttk.Scrollbar(overlap_frame, orient="vertical", command=overlap_tree.yview)
    overlap_tree.configure(yscrollcommand=overlap_scroll.set)
    overlap_tree.pack(side="left", fill="both", expand=True)
    overlap_scroll.pack(side="right", fill="y")

    # Metrics labels
    box_metrics = ttk.Frame(results_tabs["Metrics"], padding=10); box_metrics.pack(fill="both", expand=True)
    lbl_rmse = ttk.Label(box_metrics, text="RMSE: —", font=("TkDefaultFont", 10, "bold"))
    lbl_nrmse = ttk.Label(box_metrics, text="nRMSE (P5–P95): —", font=("TkDefaultFont", 10, "bold"))
    lbl_mae  = ttk.Label(box_metrics, text="MAE: —",  font=("TkDefaultFont", 10, "bold"))
    lbl_r2   = ttk.Label(box_metrics, text="R²: —",   font=("TkDefaultFont", 10, "bold"))
    lbl_rmse.pack(anchor="w"); lbl_nrmse.pack(anchor="w"); lbl_mae.pack(anchor="w"); lbl_r2.pack(anchor="w")
    tk.Message(
        box_metrics,
        width=720,
        text=(
            "How to read these metrics:\n"
            "• R² close to 1 means predictions explain much of the observed variability. R² below 0 means the model is worse than predicting the mean.\n"
            "• RMSE penalizes large errors more strongly; MAE is the average absolute error. nRMSE divides RMSE by the observed 5th–95th percentile range, allowing comparison among flux targets.\n"
            "• Good predictive performance is necessary but not sufficient. Open in IT to test whether the model preserves driver–flux relationships."
        ),
        fg="#444"
    ).pack(anchor="w", pady=(10,0))

    baseline_msg = tk.Message(
        box_metrics,
        width=720,
        text=(
            "Baseline benchmark: run a model to compare it with the training-fold mean "
            "and a training-only ±7-day seasonal climatology on the exact same test dates."
        ),
        fg="#0B4F9C",
    )
    baseline_msg.pack(anchor="w", pady=(12, 0))

    # Training monitor
    monitor = TrainingMonitor(results_tabs["Training"])

    # ----------------------- Compare tab UI -----------------------
    tab_cmp = results_tabs["Compare"]
    row1 = ttk.Frame(tab_cmp); row1.pack(fill="x", pady=(6,4), padx=8)
    Label(row1, text="Metric:").pack(side="left")
    cmp_metric = StringVar(tab_cmp, value="RMSE")
    ttk.OptionMenu(row1, cmp_metric, "RMSE", "RMSE", "nRMSE", "MAE", "R²").pack(side="left", padx=(6,18))
    btn_refresh = ttk.Button(row1, text="Refresh")
    btn_overlay = ttk.Button(row1, text="Overlay selected predictions")
    btn_ecdf = ttk.Button(row1, text="Plot metric ECDF")
    btn_export_fig2c = ttk.Button(row1, text="Export Fig 2C")
    btn_export_publication = ttk.Button(row1, text="Export publication outputs")
    btn_export_fig3b = ttk.Button(row1, text="Export Fig 3B results")
    btn_export_it = ttk.Button(row1, text="Save ML–IT bridge")
    btn_clear   = ttk.Button(row1, text="Clear runs")
    btn_refresh.pack(side="left")
    btn_overlay.pack(side="left", padx=6)
    btn_ecdf.pack(side="left", padx=6)
    btn_export_fig2c.pack(side="left", padx=6)
    btn_export_publication.pack(side="left", padx=6)
    btn_export_fig3b.pack(side="left", padx=6)
    btn_export_it.pack(side="left", padx=6)
    btn_clear.pack(side="left", padx=6)

    tk.Message(
        tab_cmp,
        width=900,
        text=(
            "Compare tab guide:\n"
            "• Each row is one model run. Repeated runs of the same model are allowed and can be selected together. Lower RMSE/nRMSE/MAE and higher R² indicate better predictive skill.\n"
            "• Overlay selected predictions to see when models fail or diverge.\n"
            "• Open in IT sends the aligned drivers, observations, predictions, residuals, and train/test labels directly to the Information Theory toolbox.\n"
            "• Plot metric ECDF shows the distribution of model skill across all runs in this session. This is useful when comparing many models, sites, or time windows rather than relying on one average value."
        ),
        fg="#444"
    ).pack(fill="x", padx=8, pady=(0,6))

    cols_cmp = ("when", "model", "resample", "split", "folds", "n_train", "n_test", "target", "features", "rmse", "nrmse", "mae", "r2")
    tree = ttk.Treeview(tab_cmp, columns=cols_cmp, show="headings", height=6, selectmode="extended")
    for c in cols_cmp:
        tree.heading(c, text=c.upper())
        tree.column(c, width=110 if c not in ("features",) else 360, anchor="w")
    tree.pack(fill="x", padx=8, pady=(0,6))

    fig_cmp, ax_cmp = plt.subplots(figsize=(8.6, 3.0))
    can_cmp = FigureCanvasTkAgg(fig_cmp, master=tab_cmp); can_cmp.draw(); can_cmp.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0,6))

    fig_ov, ax_ov = plt.subplots(figsize=(8.6, 3.2))
    can_ov = FigureCanvasTkAgg(fig_ov, master=tab_cmp); can_ov.draw(); can_ov.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0,6))

    fig_ecdf, ax_ecdf = plt.subplots(figsize=(8.6, 3.0))
    can_ecdf = FigureCanvasTkAgg(fig_ecdf, master=tab_cmp); can_ecdf.draw(); can_ecdf.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0,8))

    # ----------------------- handles -----------------------
    handles = dict(
        model=model, lb_X=lb_X, y_var=y_var, target_units=target_units,
        target_sign_convention=target_sign_convention, ts_var=ts_var,
        rs=rs, gate_var=gate_var, seq_len=seq_len, hidden=hidden, epochs=epochs,
        random_seed=random_seed,
        linear_fit_intercept=linear_fit_intercept,
        rf_n_estimators=rf_n_estimators,
        rf_min_samples_leaf=rf_min_samples_leaf,
        rf_max_depth=rf_max_depth,
        rf_max_features=rf_max_features,
        rf_perm_repeats=rf_perm_repeats,
        mlp_hidden=mlp_hidden,
        mlp_max_iter=mlp_max_iter,
        mlp_alpha=mlp_alpha,
        mlp_learning_rate=mlp_learning_rate,
        mlp_activation=mlp_activation,
        mlp_batch_size=mlp_batch_size,
        lstm_batch_size=lstm_batch_size,
        lstm_learning_rate=lstm_learning_rate,
        lstm_dropout=lstm_dropout,
        hlstm_learning_rate=hlstm_learning_rate,
        hlstm_dropout=hlstm_dropout,
        hlstm_gradient_clip=hlstm_gradient_clip,
        btn_run=btn_run, btn_stop=btn_stop, btn_open_it=btn_open_it, btn_export=btn_export,
        run_progress=run_progress, run_status=run_status, run_time=run_time,
        current_it_bridge=None, current_it_metadata=None,
        run_started_at=None, run_timer_active=False,
        ax_series=ax_series, can_series=can_series, lbl_series_note=lbl_series_note,
        ax_scat=ax_scat, can_scat=can_scat,
        ax_resid=ax_resid, can_resid=can_resid,
        ax_imp=ax_imp, can_imp=can_imp,
        lbl_rmse=lbl_rmse, lbl_nrmse=lbl_nrmse, lbl_mae=lbl_mae, lbl_r2=lbl_r2,
        baseline_msg=baseline_msg, interp_text=interp_text,
        monitor=monitor, results_notebook=results, tab_feat=results_tabs["Feature Importance"], tab_series=results_tabs["Series"],
        # compare
        runs=[], cmp_metric=cmp_metric, tree=tree,
        ax_cmp=ax_cmp, can_cmp=can_cmp, ax_ov=ax_ov, can_ov=can_ov,
        ax_ecdf=ax_ecdf, can_ecdf=can_ecdf,
        # config
        split_strategy=split_strategy, test_fraction=test_fraction, cv_folds=cv_folds,
        initial_train_fraction=initial_train_fraction,
        use_common_sequence_folds=use_common_sequence_folds,
        split_msg=split_msg,
        analysis_start=analysis_start, analysis_end=analysis_end, use_full_period=use_full_period,
        min_coverage_pct=min_coverage_pct,
        sum_override_text=sum_override_text,
        mean_override_text=mean_override_text,
        diurnal_mode=diurnal_mode,
        daylight_variable=daylight_variable,
        daylight_threshold=daylight_threshold,
        predictor_preset_message=preset_msg, auto_preset=auto_preset_var,
        fig3b_predictor_mode=fig3b_predictor_mode
    )

    # ----------------------- IT bridge export logic -----------------------
    def _safe_model_tag(name):
        tag = str(name).replace("Hysteresis-Gate LSTM (H-LSTM)", "HLSTM")
        tag = tag.replace("Linear Regression", "MLR")
        tag = tag.replace("Random Forest", "RF")
        tag = tag.replace("Neural Network (MLP)", "MLP")
        tag = tag.replace("LSTM (Keras)", "LSTM")
        tag = "".join(ch if ch.isalnum() else "_" for ch in tag).strip("_")
        return tag or "model"

    def _build_it_bridge_table(selected_only=False):
        runs = handles.get("runs", [])
        if selected_only:
            sel_ids = tree.selection()
            if sel_ids:
                runs = [runs[int(i)] for i in sel_ids]
        if not runs:
            return None

        base = None
        meta_rows = []
        for r in runs:
            bdf = r.get("it_bridge_df")
            if bdf is None or getattr(bdf, "empty", True):
                continue
            bdf = bdf.copy()
            pred_col = r.get("prediction_column") or f'{r["target"]}_pred_{_safe_model_tag(r["model"])}'
            resid_col = r.get("residual_column") or f'{r["target"]}_resid_{_safe_model_tag(r["model"])}'

            key_cols = [c for c in ["timestamp", "index"] if c in bdf.columns]
            if not key_cols:
                bdf.insert(0, "index", range(len(bdf)))
                key_cols = ["index"]

            # First run contributes timestamp/index, drivers, observed target, split if present.
            if base is None:
                keep = key_cols + list(r.get("features", [])) + [f'{r["target"]}_obs']
                if "split" in bdf.columns:
                    keep.append("split")
                keep = [c for c in dict.fromkeys(keep) if c in bdf.columns]
                base = bdf[keep].copy()

            pred_keep = key_cols + [c for c in [pred_col, resid_col] if c in bdf.columns]
            if len(pred_keep) > len(key_cols):
                base = base.merge(bdf[pred_keep], on=key_cols, how="outer")

            meta_rows.append({
                "model": r.get("model"),
                "target": r.get("target"),
                "observed_column": f'{r.get("target")}_obs',
                "prediction_column": pred_col,
                "residual_column": resid_col,
                "features": ", ".join(r.get("features", [])),
                "resample": r.get("resample"),
                "diurnal_mode": r.get("diurnal_mode"),
                "daylight_variable": r.get("daylight_variable"),
                "daylight_threshold": r.get("daylight_threshold"),
                "resampling_summary": r.get("resampling_summary"),
                "min_resample_coverage": r.get("min_resample_coverage"),
                "sum_override_columns": ", ".join(r.get("sum_override_columns", [])),
                "mean_override_columns": ", ".join(r.get("mean_override_columns", [])),
                "split_strategy": r.get("split_strategy"),
                "test_fraction": r.get("test_fraction"),
                "n_folds": r.get("n_folds"),
                "requested_n_folds": r.get("requested_n_folds"),
                "validation_reference": r.get("validation_reference"),
                "comparison_gate_variable": r.get("comparison_gate_variable"),
                "use_common_sequence_folds": r.get("use_common_sequence_folds"),
                "n_train_min": r.get("n_train_min"),
                "n_train_max": r.get("n_train_max"),
                "n_test": r.get("n_test"),
                "analysis_start": r.get("analysis_start"),
                "analysis_end": r.get("analysis_end"),
                "diurnal_mode": r.get("diurnal_mode"),
                "daylight_variable": r.get("daylight_variable"),
                "daylight_threshold": r.get("daylight_threshold"),
                "train_period": (r.get("split_info") or {}).get("train_period"),
                "test_period": (r.get("split_info") or {}).get("test_period"),
                "rmse": r.get("rmse"),
                "nrmse_p5_p95": r.get("nrmse_p5_p95"),
                "mae": r.get("mae"),
                "r2": r.get("r2"),
                "skill_vs_training_mean": r.get("skill_vs_training_mean"),
                "skill_vs_seasonal_climatology": r.get("skill_vs_seasonal_climatology"),
                "folds_beating_training_mean": r.get("folds_beating_training_mean"),
                "folds_beating_seasonal_climatology": r.get("folds_beating_seasonal_climatology"),
                "sequence_length": r.get("sequence_length"),
                "hidden_size": r.get("hidden_size"),
                "epochs": r.get("epochs"),
                "gate_variable": r.get("gate_variable"),
                "random_seed": r.get("random_seed"),
                "scaling": r.get("scaling"),
                "missing_data_handling": r.get("missing_data_handling"),
                "outlier_handling": r.get("outlier_handling"),
                "diurnal_cycle_handling": r.get("diurnal_cycle_handling"),
                "model_parameters": str(r.get("model_parameters")),
                "skill_vs_training_mean": r.get("skill_vs_training_mean"),
                "skill_vs_seasonal_climatology": r.get("skill_vs_seasonal_climatology"),
                "folds_beating_training_mean": r.get("folds_beating_training_mean"),
                "folds_beating_seasonal_climatology": r.get("folds_beating_seasonal_climatology"),
                "software_versions": str(_software_versions()),
            })

        if base is None or base.empty:
            return None

        # Put a compact metadata block at the end of the file as normal columns is messy,
        # so we return it separately and save as two CSV files in _export_it_bridge.
        return base, pd.DataFrame(meta_rows)

    def _export_it_bridge():
        from tkinter.filedialog import asksaveasfilename
        package = _build_it_bridge_table(selected_only=True)
        if package is None:
            package = _build_it_bridge_table(selected_only=False)
        if package is None:
            messagebox.showwarning(
                "ML–IT bridge",
                "No bridge table is available. Run at least one model after defining the training/testing pool so drivers, observed target, and predictions can be aligned."
            )
            return
        bridge_df, meta_df = package
        target = str(handles["y_var"].get()) or "target"
        init = f"MeaningFlux_ML_to_IT_{target}.csv"
        path = asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                                 title="Save ML-to-IT bridge table", initialfile=init)
        if not path:
            return
        try:
            bridge_df.to_csv(path, index=False)
            meta_path = path.replace(".csv", "_metadata.csv") if path.lower().endswith(".csv") else path + "_metadata.csv"
            meta_df.to_csv(meta_path, index=False)
            messagebox.showinfo("ML–IT bridge", f"Saved bridge table:\n{path}\n\nSaved metadata:\n{meta_path}")
        except Exception as e:
            messagebox.showerror("ML–IT bridge error", str(e))

    def _open_in_information_theory():
        package = _build_it_bridge_table(selected_only=True)
        if package is None:
            package = _build_it_bridge_table(selected_only=False)
        if package is None:
            messagebox.showwarning(
                "Open in Information Theory",
                "Run at least one model first. MeaningFlux needs aligned observations, predictions, residuals, predictors, and train/test labels."
            )
            return
        bridge_df, meta_df = package
        try:
            try:
                from open_information_theory_toolbox import open_information_theory_toolbox
            except Exception:
                import importlib.util, os
                module_path = os.path.join(os.path.dirname(__file__), "open_information_theory_toolbox.py")
                spec = importlib.util.spec_from_file_location("meaningflux_it_toolbox", module_path)
                if spec is None or spec.loader is None:
                    raise ImportError("Could not locate open_information_theory_toolbox.py")
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                open_information_theory_toolbox = mod.open_information_theory_toolbox
            open_information_theory_toolbox(
                bridge_df,
                inputname_site,
                bridge_metadata=meta_df,
                source_label="Current ML results",
                current_dataset=df,
            )
        except Exception as exc:
            messagebox.showerror("Open in Information Theory", str(exc))

    btn_open_it.configure(command=_open_in_information_theory)

    def _target_axis_label(target):
        t = str(target).upper()
        if t in ("FC", "NEE") or t.startswith("FC_") or t.startswith("NEE"):
            return rf"{target} ($\mu$mol CO$_2$ m$^{{-2}}$ s$^{{-1}}$)"
        if "CH4" in t:
            return rf"{target} ($\mu$mol CH$_4$ m$^{{-2}}$ s$^{{-1}}$)"
        if "N2O" in t:
            return rf"{target} ($\mu$mol N$_2$O m$^{{-2}}$ s$^{{-1}}$)"
        return str(target)

    def _selected_or_last_run():
        runs = handles.get("runs", [])
        if not runs:
            return None
        sel_ids = tree.selection()
        if sel_ids:
            try:
                return runs[int(sel_ids[0])]
            except Exception:
                pass
        return runs[-1]

    def _export_manuscript_fig2c():
        """Export a publication-ready Panel C directly from the selected MeaningFlux ML run."""
        from tkinter.filedialog import asksaveasfilename
        r = _selected_or_last_run()
        if r is None:
            messagebox.showwarning("Export Fig 2C", "Run a model first. The selected or latest run will be used for the manuscript panel.")
            return
        pdf = r.get("pred_df")
        if pdf is None or getattr(pdf, "empty", True):
            messagebox.showwarning("Export Fig 2C", "No prediction table is available for this run.")
            return

        try:
            plot_df = pdf.copy()
            if "timestamp" in plot_df.columns:
                plot_df["timestamp"] = _to_datetime_1d(plot_df["timestamp"])
                plot_df = plot_df.dropna(subset=["timestamp"]).sort_values("timestamp")
            else:
                plot_df["timestamp"] = pd.RangeIndex(len(plot_df))

            plot_df = plot_df[["timestamp", "y_true", "y_pred"]].copy()
            plot_df["y_true"] = pd.to_numeric(plot_df["y_true"], errors="coerce")
            plot_df["y_pred"] = pd.to_numeric(plot_df["y_pred"], errors="coerce")
            plot_df = plot_df.dropna(how="any")
            if plot_df.empty:
                messagebox.showwarning("Export Fig 2C", "Prediction table has no valid observed/predicted pairs.")
                return

            # Metrics are calculated on native held-out test samples.
            metric_values = _metric_bundle(plot_df["y_true"].values, plot_df["y_pred"].values)
            rmse, mae, r2 = metric_values["rmse"], metric_values["mae"], metric_values["r2"]
            nrmse = metric_values["nrmse_p5_p95"]

            # Daily mean is only for readability when timestamps are available.
            use_daily = not isinstance(plot_df["timestamp"].iloc[0], (int, np.integer))
            if use_daily:
                to_plot = (plot_df.set_index("timestamp")[["y_true", "y_pred"]]
                           .resample("D").mean().dropna(how="any").reset_index())
                x = to_plot["timestamp"]
                x_label = "Date"
                title_note = "daily means shown; lines break at temporal gaps"
            else:
                to_plot = plot_df.copy()
                x = np.arange(len(to_plot))
                x_label = "Test sample"
                title_note = "native test samples shown"

            site = str(inputname_site).replace(".csv", "") if inputname_site else "site"
            target = str(r.get("target") or handles["y_var"].get() or "target")
            model_label = _plain_model_name(r.get("model", "model"))
            tag = _safe_model_tag(r.get("model", "model"))
            init = f"Figure2C_{site}_{target}_{tag}.png".replace(" ", "_")
            path = asksaveasfilename(defaultextension=".png",
                                     filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
                                     title="Save manuscript Panel C: observed vs predicted",
                                     initialfile=init)
            if not path:
                return

            fig, ax = plt.subplots(figsize=(6.6, 3.3))
            if use_daily:
                _plot_gap_aware_line(
                    ax, x, to_plot["y_true"].values, linewidth=1.8,
                    marker="o", markersize=2, label=f"Observed {target}",
                )
                _plot_gap_aware_line(
                    ax, x, to_plot["y_pred"].values, linewidth=1.8,
                    linestyle="--", marker="o", markersize=2,
                    label=f"Predicted {target} ({model_label})",
                )
            else:
                ax.plot(x, to_plot["y_true"].values, linewidth=1.8, label=f"Observed {target}")
                ax.plot(x, to_plot["y_pred"].values, linewidth=1.8, linestyle="--", label=f"Predicted {target} ({model_label})")
            ax.set_title(f"C. Held-out {target} prediction")
            ax.set_xlabel(x_label)
            ax.set_ylabel(_target_axis_label(target))
            ax.grid(True, alpha=0.25)
            ax.legend(frameon=False, loc="best")

            text = (f"{model_label}\n"
                    rf"$R^2$ = {r2:.2f}" + "\n"
                    f"RMSE = {rmse:.2f}\n"
                    f"nRMSE = {nrmse:.1f}%\n"
                    f"MAE = {mae:.2f}\n"
                    f"{title_note}")
            ax.text(0.02, 0.98, text, transform=ax.transAxes, va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7"))

            fig.tight_layout()
            fig.savefig(path, dpi=600, bbox_inches="tight")
            plt.close(fig)

            # Save plotted data and metadata next to the figure for reproducibility.
            base = path.rsplit(".", 1)[0]
            csv_path = base + "_data.csv"
            meta_path = base + "_metadata.csv"
            out_df = to_plot.copy()
            out_df = out_df.rename(columns={"y_true": f"{target}_observed", "y_pred": f"{target}_predicted_{tag}"})
            out_df.to_csv(csv_path, index=False)
            pd.DataFrame([{
                "site": site,
                "target": target,
                "model": r.get("model"),
                "split_strategy": r.get("split_strategy"),
                "test_fraction": r.get("test_fraction"),
                "n_train": r.get("n_train"),
                "n_test": r.get("n_test"),
                "analysis_start": r.get("analysis_start"),
                "analysis_end": r.get("analysis_end"),
                "train_period": (r.get("split_info") or {}).get("train_period"),
                "test_period": (r.get("split_info") or {}).get("test_period"),
                "metrics_calculated_on": "held-out/out-of-fold predictions in original target units",
                "plot_values": title_note,
                "rmse": rmse,
                "nrmse_p5_p95": nrmse,
                "mae": mae,
                "r2": r2,
                "n_folds": r.get("n_folds"),
                "requested_n_folds": r.get("requested_n_folds"),
                "validation_reference": r.get("validation_reference"),
                "comparison_gate_variable": r.get("comparison_gate_variable"),
                "use_common_sequence_folds": r.get("use_common_sequence_folds"),
                "sequence_length": r.get("sequence_length"),
                "hidden_size": r.get("hidden_size"),
                "epochs": r.get("epochs"),
                "gate_variable": r.get("gate_variable"),
                "random_seed": r.get("random_seed"),
                "scaling": r.get("scaling"),
            }]).to_csv(meta_path, index=False)

            messagebox.showinfo("Export Fig 2C", f"Saved manuscript panel:\n{path}\n\nSaved plotted data:\n{csv_path}\n\nSaved metadata:\n{meta_path}")
        except Exception as e:
            messagebox.showerror("Export Fig 2C error", str(e))


    def _export_publication_outputs():
        """Export a standardized, manuscript-ready package for the selected/latest ML run."""
        import os
        import json
        from pathlib import Path
        from tkinter.filedialog import askdirectory

        r = _selected_or_last_run()
        if r is None:
            messagebox.showwarning(
                "Export publication outputs",
                "Run a model first. The selected or latest run will be exported."
            )
            return

        pred_df = r.get("pred_df")
        if pred_df is None or getattr(pred_df, "empty", True):
            messagebox.showwarning(
                "Export publication outputs",
                "No prediction table is available for this run."
            )
            return

        root_dir = askdirectory(title="Choose folder for ML publication outputs")
        if not root_dir:
            return

        try:
            site = str(inputname_site).replace(".csv", "") if inputname_site else "site"
            target = str(r.get("target") or handles["y_var"].get() or "target")
            model_name = str(r.get("model") or "model")
            model_label = _plain_model_name(model_name)
            model_tag = _safe_model_tag(model_name)
            stamp = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")

            def _safe_token(value):
                token = "".join(ch if str(ch).isalnum() or ch in "-_" else "_" for ch in str(value))
                return token.strip("_") or "item"

            package_dir = Path(root_dir) / "results" / "machine_learning" / _safe_token(site) / f"{stamp}_{_safe_token(target)}_{_safe_token(model_tag)}"
            figures_dir = package_dir / "figures"
            data_dir = package_dir / "data"
            metadata_dir = package_dir / "metadata"
            figures_dir.mkdir(parents=True, exist_ok=True)
            data_dir.mkdir(parents=True, exist_ok=True)
            metadata_dir.mkdir(parents=True, exist_ok=True)

            work = pred_df.copy()
            if "timestamp" in work.columns:
                work["timestamp"] = _to_datetime_1d(work["timestamp"])
                work = work.dropna(subset=["timestamp"]).sort_values("timestamp")
            else:
                work["timestamp"] = pd.RangeIndex(len(work))
            work["y_true"] = pd.to_numeric(work["y_true"], errors="coerce")
            work["y_pred"] = pd.to_numeric(work["y_pred"], errors="coerce")
            work = work.dropna(subset=["y_true", "y_pred"])
            if work.empty:
                raise ValueError("Prediction table has no valid observed/predicted pairs.")

            metric_values = _metric_bundle(work["y_true"].to_numpy(), work["y_pred"].to_numpy())
            rmse, mae, r2 = metric_values["rmse"], metric_values["mae"], metric_values["r2"]
            nrmse = metric_values["nrmse_p5_p95"]
            resid = work["y_pred"].to_numpy() - work["y_true"].to_numpy()
            manifest_rows = []

            def _save_figure(fig_obj, stem):
                for ext, dpi in (("png", 600), ("pdf", 300), ("svg", 300)):
                    path = figures_dir / f"{stem}.{ext}"
                    fig_obj.savefig(path, dpi=dpi, bbox_inches="tight")
                    manifest_rows.append({"type": "figure", "name": stem, "format": ext, "path": str(path.relative_to(package_dir))})
                plt.close(fig_obj)

            # 1. Observed vs predicted time series.
            has_real_time = not isinstance(work["timestamp"].iloc[0], (int, np.integer))
            if has_real_time:
                ts_plot = (work.set_index("timestamp")[["y_true", "y_pred"]]
                           .resample("D").mean().dropna(how="any").reset_index())
                x = ts_plot["timestamp"]
                x_label = "Date"
                time_note = "Daily means shown with temporal gaps left unconnected; metrics use all held-out/out-of-fold samples."
            else:
                ts_plot = work[["timestamp", "y_true", "y_pred"]].copy()
                x = np.arange(len(ts_plot))
                x_label = "Test sample"
                time_note = "Held-out/out-of-fold samples shown."

            fig_ts, ax_ts = plt.subplots(figsize=(6.8, 3.5))
            if has_real_time:
                _plot_gap_aware_line(
                    ax_ts, x, ts_plot["y_true"], linewidth=1.6,
                    marker="o", markersize=2, label=f"Observed {target}",
                )
                _plot_gap_aware_line(
                    ax_ts, x, ts_plot["y_pred"], linewidth=1.6,
                    linestyle="--", marker="o", markersize=2,
                    label=f"Predicted {target}",
                )
            else:
                ax_ts.plot(x, ts_plot["y_true"], linewidth=1.6, label=f"Observed {target}")
                ax_ts.plot(x, ts_plot["y_pred"], linewidth=1.6, linestyle="--", label=f"Predicted {target}")
            ax_ts.set_title(f"Observed and predicted {target}")
            ax_ts.set_xlabel(x_label)
            ax_ts.set_ylabel(_target_axis_label(target))
            ax_ts.legend(frameon=False)
            ax_ts.grid(True, alpha=0.25)
            ax_ts.text(0.02, 0.98, f"{model_label}\n$R^2$ = {r2:.2f}\nRMSE = {rmse:.2f}\nnRMSE = {nrmse:.1f}%\nMAE = {mae:.2f}",
                       transform=ax_ts.transAxes, va="top", ha="left",
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7"))
            fig_ts.tight_layout()
            _save_figure(fig_ts, "observed_vs_predicted_timeseries")
            ts_plot.rename(columns={"y_true": f"{target}_observed", "y_pred": f"{target}_predicted_{model_tag}"}).to_csv(data_dir / "observed_vs_predicted_timeseries.csv", index=False)

            # 2. Observed-vs-predicted scatter.
            fig_sc, ax_sc = plt.subplots(figsize=(4.6, 4.2))
            ax_sc.scatter(work["y_true"], work["y_pred"], s=12, alpha=0.45)
            lo = float(np.nanmin([work["y_true"].min(), work["y_pred"].min()]))
            hi = float(np.nanmax([work["y_true"].max(), work["y_pred"].max()]))
            ax_sc.plot([lo, hi], [lo, hi], "--", linewidth=1)
            ax_sc.set_xlabel(f"Observed {target}")
            ax_sc.set_ylabel(f"Predicted {target}")
            ax_sc.set_title("Observed versus predicted")
            ax_sc.grid(True, alpha=0.25)
            ax_sc.text(0.04, 0.96, f"$R^2$ = {r2:.2f}\nRMSE = {rmse:.2f}\nnRMSE = {nrmse:.1f}%\nMAE = {mae:.2f}",
                       transform=ax_sc.transAxes, va="top",
                       bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.7"))
            fig_sc.tight_layout()
            _save_figure(fig_sc, "observed_vs_predicted_scatter")

            # 3. Residual distribution.
            fig_res, ax_res = plt.subplots(figsize=(6.0, 3.5))
            ax_res.hist(resid, bins=35, alpha=0.85)
            ax_res.axvline(0.0, linestyle="--", linewidth=1)
            ax_res.set_xlabel("Prediction error (predicted - observed)")
            ax_res.set_ylabel("Count")
            ax_res.set_title("Residual distribution")
            ax_res.grid(True, axis="y", alpha=0.25)
            fig_res.tight_layout()
            _save_figure(fig_res, "residual_distribution")

            # 4. Feature importance when available.
            importance_df = r.get("importance_df")
            if importance_df is not None and not getattr(importance_df, "empty", True):
                imp_df = importance_df.copy()
                value_col = "importance" if "importance" in imp_df.columns else imp_df.select_dtypes(include=[np.number]).columns[0]
                feature_col = "feature" if "feature" in imp_df.columns else imp_df.columns[0]
                imp_df[value_col] = pd.to_numeric(imp_df[value_col], errors="coerce")
                imp_df = imp_df.dropna(subset=[value_col]).sort_values(value_col, ascending=True)
                fig_imp, ax_imp = plt.subplots(figsize=(6.2, max(3.2, 0.42 * len(imp_df) + 1.4)))
                ypos = np.arange(len(imp_df))
                xerr = imp_df["importance_std"].to_numpy() if "importance_std" in imp_df.columns else None
                ax_imp.barh(ypos, imp_df[value_col], xerr=xerr, capsize=2 if xerr is not None else 0)
                ax_imp.set_yticks(ypos)
                ax_imp.set_yticklabels(imp_df[feature_col].astype(str))
                ax_imp.set_xlabel("Held-out permutation importance")
                ax_imp.set_title(f"{model_label} held-out feature importance")
                ax_imp.grid(True, axis="x", alpha=0.25)
                fig_imp.tight_layout()
                _save_figure(fig_imp, "feature_importance")
                imp_df.to_csv(data_dir / "feature_importance.csv", index=False)

            # Data products.
            work.rename(columns={"y_true": f"{target}_observed", "y_pred": f"{target}_predicted_{model_tag}"}).to_csv(data_dir / "predictions_test.csv", index=False)
            pd.DataFrame({"residual": resid}).to_csv(data_dir / "residuals_test.csv", index=False)
            pd.DataFrame([{
                "rmse": rmse, "nrmse_p5_p95": nrmse, "mae": mae, "r2": r2,
                "n_test": len(work), "observed_p05": metric_values["observed_p05"],
                "observed_p95": metric_values["observed_p95"],
                "skill_vs_training_mean": r.get("skill_vs_training_mean"),
                "skill_vs_seasonal_climatology": r.get("skill_vs_seasonal_climatology"),
                "folds_beating_training_mean": r.get("folds_beating_training_mean"),
                "folds_beating_seasonal_climatology": r.get("folds_beating_seasonal_climatology"),
            }]).to_csv(data_dir / "metrics.csv", index=False)
            baseline_metrics_df = r.get("baseline_metrics_df")
            if baseline_metrics_df is not None and not getattr(baseline_metrics_df, "empty", True):
                baseline_metrics_df.to_csv(data_dir / "baseline_metrics.csv", index=False)
            fold_metrics_df = r.get("fold_metrics_df")
            if fold_metrics_df is not None and not getattr(fold_metrics_df, "empty", True):
                fold_metrics_df.to_csv(data_dir / "fold_metrics.csv", index=False)

            bridge_df = r.get("it_bridge_df")
            if bridge_df is not None and not getattr(bridge_df, "empty", True):
                bridge_df.to_csv(data_dir / "ML_IT_bridge.csv", index=False)
                manifest_rows.append({"type": "data", "name": "ML_IT_bridge", "format": "csv", "path": "data/ML_IT_bridge.csv"})

            # Session run summary.
            runs_summary = pd.DataFrame([{
                "when": str(rr.get("when", "")),
                "model": rr.get("model"),
                "resample": rr.get("resample"),
                "target": rr.get("target"),
                "diurnal_mode": rr.get("diurnal_mode"),
                "daylight_variable": rr.get("daylight_variable"),
                "daylight_threshold": rr.get("daylight_threshold"),
                "features": ", ".join(rr.get("features", [])),
                "split_strategy": rr.get("split_strategy"),
                "test_fraction": rr.get("test_fraction"),
                "n_folds": rr.get("n_folds"),
                "requested_n_folds": rr.get("requested_n_folds"),
                "validation_reference": rr.get("validation_reference"),
                "comparison_gate_variable": rr.get("comparison_gate_variable"),
                "use_common_sequence_folds": rr.get("use_common_sequence_folds"),
                "n_train_min": rr.get("n_train_min"),
                "n_train_max": rr.get("n_train_max"),
                "n_test": rr.get("n_test"),
                "rmse": rr.get("rmse"),
                "nrmse_p5_p95": rr.get("nrmse_p5_p95"),
                "mae": rr.get("mae"),
                "r2": rr.get("r2"),
                "skill_vs_training_mean": rr.get("skill_vs_training_mean"),
                "skill_vs_seasonal_climatology": rr.get("skill_vs_seasonal_climatology"),
                "folds_beating_training_mean": rr.get("folds_beating_training_mean"),
                "folds_beating_seasonal_climatology": rr.get("folds_beating_seasonal_climatology"),
                "sequence_length": rr.get("sequence_length"),
                "hidden_size": rr.get("hidden_size"),
                "epochs": rr.get("epochs"),
                "gate_variable": rr.get("gate_variable"),
                "random_seed": rr.get("random_seed"),
                "scaling": rr.get("scaling"),
            } for rr in handles.get("runs", [])])
            runs_summary.to_csv(data_dir / "session_run_summary.csv", index=False)

            metadata = {
                "site": site,
                "target": target,
                "model": model_name,
                "model_label": model_label,
                "features": list(r.get("features", [])),
                "resample": r.get("resample"),
                "resampling_summary": r.get("resampling_summary"),
                "min_resample_coverage": r.get("min_resample_coverage"),
                "sum_override_columns": r.get("sum_override_columns"),
                "mean_override_columns": r.get("mean_override_columns"),
                "split_strategy": r.get("split_strategy"),
                "test_fraction": r.get("test_fraction"),
                "n_folds": r.get("n_folds"),
                "requested_n_folds": r.get("requested_n_folds"),
                "validation_reference": r.get("validation_reference"),
                "comparison_gate_variable": r.get("comparison_gate_variable"),
                "use_common_sequence_folds": r.get("use_common_sequence_folds"),
                "n_train_min": r.get("n_train_min"),
                "n_train_max": r.get("n_train_max"),
                "n_test": r.get("n_test"),
                "analysis_start": r.get("analysis_start"),
                "analysis_end": r.get("analysis_end"),
                "train_period": (r.get("split_info") or {}).get("train_period"),
                "test_period": (r.get("split_info") or {}).get("test_period"),
                "metrics_calculated_on": "held-out/out-of-fold predictions in original target units",
                "timeseries_display": time_note,
                "rmse": rmse,
                "nrmse_p5_p95": nrmse,
                "mae": mae,
                "r2": r2,
                "skill_vs_training_mean": r.get("skill_vs_training_mean"),
                "skill_vs_seasonal_climatology": r.get("skill_vs_seasonal_climatology"),
                "folds_beating_training_mean": r.get("folds_beating_training_mean"),
                "folds_beating_seasonal_climatology": r.get("folds_beating_seasonal_climatology"),
                "observed_p05": metric_values["observed_p05"],
                "observed_p95": metric_values["observed_p95"],
                "sequence_length": r.get("sequence_length"),
                "hidden_size": r.get("hidden_size"),
                "epochs": r.get("epochs"),
                "batch_size": r.get("batch_size"),
                "gate_variable": r.get("gate_variable"),
                "random_seed": r.get("random_seed"),
                "scaling": r.get("scaling"),
                "importance_scoring": r.get("importance_scoring"),
                "model_parameters": r.get("model_parameters"),
                "initial_train_fraction": r.get("initial_train_fraction"),
                "software_versions": _software_versions(),
                "created_utc": pd.Timestamp.utcnow().isoformat(),
            }
            with open(metadata_dir / "run_metadata.json", "w", encoding="utf-8") as fh:
                json.dump(metadata, fh, indent=2, default=str)

            # Manifest and README.
            for p in sorted(data_dir.glob("*.csv")):
                manifest_rows.append({"type": "data", "name": p.stem, "format": "csv", "path": str(p.relative_to(package_dir))})
            manifest_rows.append({"type": "metadata", "name": "run_metadata", "format": "json", "path": "metadata/run_metadata.json"})
            pd.DataFrame(manifest_rows).drop_duplicates().to_csv(package_dir / "publication_manifest.csv", index=False)

            readme = f"""MeaningFlux machine-learning publication outputs

Site: {site}
Target: {target}
Model: {model_label}
Predictors: {', '.join(r.get('features', []))}

Contents
- figures/: manuscript-ready PNG (600 dpi), PDF, and SVG outputs
- data/: predictions, residuals, model/baseline metrics, fold diagnostics, feature importance when available, session summary, and ML-IT bridge
- metadata/: run settings and reproducibility information
- publication_manifest.csv: index of exported files

Notes
- Predictive metrics are calculated only from held-out/out-of-fold predictions in original target units.
- nRMSE is RMSE divided by the observed 5th–95th percentile range.
- Baselines use the same held-out dates: the training-fold mean and a training-only ±7-day day-of-year climatology.
- Daytime/nighttime products are optional FC sensitivities and are not used by the locked main Figure 3B preset.
- Neural-model scalers are fitted inside each training fold and predictions are inverse-transformed before evaluation.
- The time-series figure may show daily means for readability; lines are broken at temporal gaps and this does not change the reported metrics.
- Panel letters are intentionally omitted so figures can be assembled in PowerPoint, Illustrator, or LaTeX.
"""
            (package_dir / "README.txt").write_text(readme, encoding="utf-8")

            messagebox.showinfo(
                "Export publication outputs",
                f"Saved publication package to:\n{package_dir}"
            )
        except Exception as exc:
            messagebox.showerror("Export publication outputs", str(exc))

    # ----------------------- Figure 3B common-timestamp export -----------------------
    def _runs_selected_or_all():
        runs = handles.get("runs", [])
        sel_ids = tree.selection()
        if sel_ids:
            return [runs[int(i)] for i in sel_ids]
        return list(runs)

    def _build_common_timestamp_comparison():
        """Build the final five-model Figure 3B comparison on common test dates."""
        runs = _runs_selected_or_all()
        expected_models = {
            "Linear Regression",
            "Random Forest",
            "Neural Network (MLP)",
            "LSTM (Keras)",
            "Hysteresis-Gate LSTM (H-LSTM)",
        }
        model_names = [str(r.get("model")) for r in runs]
        if len(model_names) != len(set(model_names)):
            raise ValueError(
                "Figure 3B requires exactly one run per model. Remove duplicate runs "
                "or select one row for each model."
            )
        if set(model_names) != expected_models:
            missing = sorted(expected_models - set(model_names))
            extra = sorted(set(model_names) - expected_models)
            details = []
            if missing:
                details.append("missing: " + ", ".join(missing))
            if extra:
                details.append("unexpected: " + ", ".join(extra))
            raise ValueError(
                "Figure 3B requires Linear Regression, Random Forest, MLP, LSTM, "
                "and H-LSTM (" + "; ".join(details) + ")."
            )

        targets = {str(r.get("target")) for r in runs}
        resolutions = {str(r.get("resample")) for r in runs}
        strategies = {str(r.get("split_strategy")) for r in runs}
        feature_sets = {tuple(r.get("features", [])) for r in runs}
        analysis_periods = {
            (str(r.get("analysis_start")), str(r.get("analysis_end"))) for r in runs
        }
        selected_periods = {
            (str(r.get("selected_analysis_start")), str(r.get("selected_analysis_end")))
            for r in runs
        }
        fold_counts = {int(r.get("n_folds") or 1) for r in runs}
        requested_fold_counts = {int(r.get("requested_n_folds") or r.get("n_folds") or 1) for r in runs}
        validation_references = {str(r.get("validation_reference")) for r in runs}
        comparison_gates = {str(r.get("comparison_gate_variable")) for r in runs}
        common_sequence_flags = {bool(r.get("use_common_sequence_folds")) for r in runs}
        initial_fractions = {r.get("initial_train_fraction") for r in runs}
        coverage_settings = {r.get("min_resample_coverage") for r in runs}
        sum_overrides = {tuple(r.get("sum_override_columns", [])) for r in runs}
        mean_overrides = {tuple(r.get("mean_override_columns", [])) for r in runs}
        seeds = {int(r.get("random_seed")) for r in runs}
        units = {str(r.get("target_units", "not provided")) for r in runs}
        sign_conventions = {
            str(r.get("target_sign_convention", "not provided")) for r in runs
        }
        diurnal_settings = {
            (
                str(r.get("diurnal_mode") or "All observations"),
                str(r.get("daylight_variable")),
                str(r.get("daylight_threshold")),
            )
            for r in runs
        }

        checks = [
            (len(targets) == 1, "the same target"),
            (len(resolutions) == 1, "the same temporal aggregation"),
            (len(strategies) == 1, "the same validation strategy"),
            (len(feature_sets) == 1, "the same ordered predictor list"),
            (len(analysis_periods) == 1 and len(selected_periods) == 1,
             "the same analysis period"),
            (len(fold_counts) == 1 and len(requested_fold_counts) == 1 and
             len(initial_fractions) == 1 and len(validation_references) == 1 and
             len(comparison_gates) == 1 and common_sequence_flags == {True},
             "identical requested/used folds, common endpoint validation reference, gate, and initial training fraction"),
            (len(coverage_settings) == 1 and len(sum_overrides) == 1 and len(mean_overrides) == 1,
             "identical aggregation settings"),
            (len(seeds) == 1, "the same random seed"),
            (len(units) == 1 and len(sign_conventions) == 1,
             "the same target units and sign convention"),
            (len(diurnal_settings) == 1,
             "the same daytime/nighttime setting"),
        ]
        failed = [label for ok, label in checks if not ok]
        if failed:
            raise ValueError("Figure 3B requires " + ", ".join(failed) + " for every model.")
        used_folds = next(iter(fold_counts))
        requested_fold_count = next(iter(requested_fold_counts))
        if requested_fold_count != 5 or used_folds != 5:
            raise ValueError(
                f"Figure 3B requires exactly 5 requested and 5 completed folds; "
                f"found requested={requested_fold_count}, completed={used_folds}."
            )
        diurnal_mode_for_export = next(iter(diurnal_settings))[0]
        if not diurnal_mode_for_export.lower().startswith("all"):
            raise ValueError(
                "The locked main Figure 3B uses All observations. Daytime/nighttime runs "
                "belong in the supplementary FC sensitivity analysis."
            )
        if any("Random" in strategy for strategy in strategies):
            raise ValueError(
                "Use blocked time-series CV or chronological holdout; random splits "
                "are exploratory only."
            )

        lstm_run = next(r for r in runs if r.get("model") == "LSTM (Keras)")
        hlstm_run = next(r for r in runs if r.get("model") == "Hysteresis-Gate LSTM (H-LSTM)")
        lstm_params = dict(lstm_run.get("model_parameters") or {})
        hlstm_params = dict(hlstm_run.get("model_parameters") or {})
        controlled_keys = (
            "sequence_length", "hidden_size", "epochs", "batch_size",
            "learning_rate", "dropout", "gradient_clip", "framework", "shuffle",
        )
        mismatched_sequence_settings = [
            key for key in controlled_keys
            if lstm_params.get(key) != hlstm_params.get(key)
        ]
        if mismatched_sequence_settings:
            raise ValueError(
                "For Figure 3B, LSTM and H-LSTM must use identical training settings. "
                "Mismatched: " + ", ".join(mismatched_sequence_settings) + "."
            )
        gate_for_comparison = hlstm_run.get("gate_variable")
        if not gate_for_comparison or gate_for_comparison not in hlstm_run.get("features", []):
            raise ValueError(
                "The H-LSTM gate must be included in the common predictor list."
            )

        common = None
        model_specs = []
        for idx, r in enumerate(runs):
            pdf = r.get("pred_df")
            if pdf is None or getattr(pdf, "empty", True) or "timestamp" not in pdf.columns:
                raise ValueError(f"{r.get('model')} has no timestamped held-out predictions.")
            d = pdf.copy()
            d["timestamp"] = _to_datetime_1d(d["timestamp"])
            d["y_true"] = pd.to_numeric(d["y_true"], errors="coerce")
            d["y_pred"] = pd.to_numeric(d["y_pred"], errors="coerce")
            d = d.dropna(subset=["timestamp", "y_true", "y_pred"])
            agg = {"y_true": "mean", "y_pred": "mean"}
            if "fold" in d.columns:
                agg["fold"] = "first"
            if idx == 0:
                if "pred_training_mean" in d.columns:
                    agg["pred_training_mean"] = "mean"
                if "pred_seasonal_climatology" in d.columns:
                    agg["pred_seasonal_climatology"] = "mean"
            d = d.groupby("timestamp", as_index=False).agg(agg)

            tag = _safe_model_tag(r.get("model", f"model_{idx+1}"))
            obs_col = f"observed_{tag}"
            pred_col = f"predicted_{tag}"
            fold_col = f"fold_{tag}"
            rename = {"y_true": obs_col, "y_pred": pred_col}
            if "fold" in d.columns:
                rename["fold"] = fold_col
            d = d.rename(columns=rename)
            keep = ["timestamp", obs_col, pred_col]
            if idx == 0:
                keep.extend([
                    c for c in ("pred_training_mean", "pred_seasonal_climatology")
                    if c in d.columns
                ])
            if fold_col in d.columns:
                keep.append(fold_col)
            common = d[keep] if common is None else common.merge(d[keep], on="timestamp", how="inner")
            model_specs.append((r, tag, obs_col, pred_col, fold_col))

        if common is None or common.empty:
            raise ValueError("The five models have no common held-out timestamps.")
        common = common.sort_values("timestamp").reset_index(drop=True)
        first_run = model_specs[0][0]
        first_obs = model_specs[0][2]
        common = common.rename(columns={first_obs: "observed"})
        for _, _, obs_col, _, _ in model_specs[1:]:
            diff = np.nanmax(
                np.abs(common[obs_col].to_numpy(dtype=float) - common["observed"].to_numpy(dtype=float))
            )
            if np.isfinite(diff) and diff > 1e-8:
                raise ValueError("Observed target values differ among models on common timestamps.")
            common = common.drop(columns=[obs_col])

        fold_cols = [spec[4] for spec in model_specs if spec[4] in common.columns]
        if len(fold_cols) != len(model_specs):
            raise ValueError("Every Figure 3B model must contain fold assignments.")
        fold_reference = pd.to_numeric(common[fold_cols[0]], errors="coerce")
        for fold_col in fold_cols[1:]:
            fold_other = pd.to_numeric(common[fold_col], errors="coerce")
            mismatch = ~(fold_reference.eq(fold_other) | (fold_reference.isna() & fold_other.isna()))
            if bool(mismatch.any()):
                raise ValueError(
                    "Validation fold assignments differ among models. Clear the runs and rerun "
                    "all five models without changing the validation configuration."
                )
        common.insert(1, "fold", fold_reference.astype("Int64"))
        common = common.drop(columns=fold_cols)
        common_fold_ids = sorted(common["fold"].dropna().astype(int).unique().tolist())
        expected_fold_ids = [1, 2, 3, 4, 5]
        if common_fold_ids != expected_fold_ids:
            raise ValueError(
                f"Common-timestamp comparison must retain folds {expected_fold_ids}; "
                f"found {common_fold_ids}. Rerun all models with the final validation code."
            )
        common_fold_counts = common.groupby("fold").size()
        too_small = common_fold_counts[common_fold_counts < 2]
        if not too_small.empty:
            raise ValueError(
                "At least two common observations are required in every fold to calculate R². "
                "Insufficient folds: " + ", ".join(
                    f"{int(k)} (n={int(v)})" for k, v in too_small.items()
                )
            )

        common_test_start = pd.to_datetime(common["timestamp"], errors="coerce").min()
        common_test_end = pd.to_datetime(common["timestamp"], errors="coerce").max()
        common_gap_info = _timestamp_gap_diagnostics(common["timestamp"])
        pooled_obs_p05, pooled_obs_p95 = np.nanpercentile(
            pd.to_numeric(common["observed"], errors="coerce"), [5, 95]
        )
        pooled_obs_range = float(pooled_obs_p95 - pooled_obs_p05)
        if "pred_training_mean" not in common.columns or "pred_seasonal_climatology" not in common.columns:
            raise ValueError(
                "The selected runs do not contain the final baseline predictions. Clear old runs and rerun all five models."
            )
        common_training_mean_metrics = _metric_bundle(
            common["observed"], common["pred_training_mean"]
        )
        common_seasonal_metrics = _metric_bundle(
            common["observed"], common["pred_seasonal_climatology"]
        )

        selected_analysis_start = first_run.get("selected_analysis_start")
        selected_analysis_end = first_run.get("selected_analysis_end")
        analysis_period_mode = first_run.get("analysis_period_mode", "not recorded")
        target_units = first_run.get("target_units", "not provided")
        target_sign = first_run.get("target_sign_convention", "not provided")

        metric_rows = []
        common_fold_rows = []
        metadata_rows = []
        for r, tag, _, pred_col, fold_col in model_specs:
            counts = dict(r.get("data_counts") or {})
            m = _metric_bundle(common["observed"], common[pred_col])
            metric_rows.append({
                "site": str(inputname_site),
                "target": r.get("target"),
                "target_units": target_units,
                "target_sign_convention": target_sign,
                "diurnal_mode": r.get("diurnal_mode"),
                "daylight_variable": r.get("daylight_variable"),
                "daylight_threshold": r.get("daylight_threshold"),
                "model": r.get("model"),
                "model_tag": tag,
                "requested_n_folds": r.get("requested_n_folds"),
                "completed_n_folds": r.get("n_folds"),
                "validation_reference": r.get("validation_reference"),
                "comparison_gate_variable": r.get("comparison_gate_variable"),
                "use_common_sequence_folds": r.get("use_common_sequence_folds"),
                "analysis_period_mode": analysis_period_mode,
                "selected_analysis_start": selected_analysis_start,
                "selected_analysis_end": selected_analysis_end,
                "model_ready_start": r.get("model_ready_start"),
                "model_ready_end": r.get("model_ready_end"),
                "common_test_start": common_test_start,
                "common_test_end": common_test_end,
                "n_common": int(len(common)),
                "common_test_gaps": int(common_gap_info["n_gaps"]),
                "common_test_max_interval_days": common_gap_info["max_interval_days"],
                "input_rows": counts.get("input_rows"),
                "selected_period_rows": counts.get("selected_period_rows"),
                "resampled_period_rows": counts.get("resampled_period_rows"),
                "complete_case_rows": counts.get("complete_case_rows"),
                "model_ready_rows": counts.get("model_ready_rows"),
                "rmse": m["rmse"],
                "nrmse_p5_p95": _nrmse_with_reference_range(
                    m["rmse"], pooled_obs_p05, pooled_obs_p95
                ),
                "mae": m["mae"],
                "r2": m["r2"],
                "training_mean_rmse": common_training_mean_metrics["rmse"],
                "training_mean_nrmse_p5_p95": _nrmse_with_reference_range(
                    common_training_mean_metrics["rmse"], pooled_obs_p05, pooled_obs_p95
                ),
                "training_mean_r2": common_training_mean_metrics["r2"],
                "seasonal_climatology_rmse": common_seasonal_metrics["rmse"],
                "seasonal_climatology_nrmse_p5_p95": _nrmse_with_reference_range(
                    common_seasonal_metrics["rmse"], pooled_obs_p05, pooled_obs_p95
                ),
                "seasonal_climatology_r2": common_seasonal_metrics["r2"],
                "skill_vs_training_mean": _baseline_skill_score(
                    common["observed"], common[pred_col], common["pred_training_mean"]
                ),
                "skill_vs_seasonal_climatology": _baseline_skill_score(
                    common["observed"], common[pred_col], common["pred_seasonal_climatology"]
                ),
                "observed_p05": float(pooled_obs_p05),
                "observed_p95": float(pooled_obs_p95),
                "nrmse_reference_range": pooled_obs_range,
                "nrmse_reference": "pooled common observed P5-P95 range",
                "sequence_policy": r.get("sequence_policy"),
                "expected_timestep": r.get("expected_timestep"),
                "n_candidate_sequence_windows": r.get("n_candidate_sequence_windows"),
                "n_valid_sequences": r.get("n_valid_sequences"),
                "n_rejected_gap_windows": r.get("n_rejected_gap_windows", 0),
                "n_rejected_missing_endpoint_target": r.get(
                    "n_rejected_missing_endpoint_target", 0
                ),
                "n_rejected_missing_predictor_windows": r.get(
                    "n_rejected_missing_predictor_windows", 0
                ),
                "target_requirement": r.get("target_requirement"),
                "predictor_requirement": r.get("predictor_requirement"),
            })

            original_folds = r.get("fold_metrics_df")
            for fold, g in common.groupby("fold"):
                fm = _metric_bundle(g["observed"], g[pred_col])
                mean_fm = _metric_bundle(g["observed"], g["pred_training_mean"])
                seasonal_fm = _metric_bundle(g["observed"], g["pred_seasonal_climatology"])
                original = None
                if original_folds is not None and not getattr(original_folds, "empty", True):
                    match = original_folds[pd.to_numeric(original_folds["fold"], errors="coerce") == int(fold)]
                    if not match.empty:
                        original = match.iloc[0]
                def _orig(name, default=np.nan):
                    return original.get(name, default) if original is not None else default
                common_fold_rows.append({
                    "site": str(inputname_site),
                    "target": r.get("target"),
                    "model": r.get("model"),
                    "fold": int(fold),
                    "train_start": _orig("train_start", pd.NaT),
                    "train_end": _orig("train_end", pd.NaT),
                    "n_train_model": _orig("n_train"),
                    "model_test_start": _orig("test_start", pd.NaT),
                    "model_test_end": _orig("test_end", pd.NaT),
                    "n_test_model": _orig("n_test"),
                    "common_fold_start": g["timestamp"].min(),
                    "common_fold_end": g["timestamp"].max(),
                    "n_common_fold": int(len(g)),
                    "n_rejected_gap_windows_train": _orig("n_rejected_gap_windows_train", 0),
                    "n_rejected_gap_windows_test": _orig("n_rejected_gap_windows_test", 0),
                    "rmse": fm["rmse"],
                    "nrmse_p5_p95": _nrmse_with_reference_range(
                        fm["rmse"], pooled_obs_p05, pooled_obs_p95
                    ),
                    "mae": fm["mae"],
                    "r2": fm["r2"],
                    "training_mean_rmse": mean_fm["rmse"],
                    "training_mean_nrmse_p5_p95": _nrmse_with_reference_range(
                        mean_fm["rmse"], pooled_obs_p05, pooled_obs_p95
                    ),
                    "seasonal_climatology_rmse": seasonal_fm["rmse"],
                    "seasonal_climatology_nrmse_p5_p95": _nrmse_with_reference_range(
                        seasonal_fm["rmse"], pooled_obs_p05, pooled_obs_p95
                    ),
                    "skill_vs_training_mean": _baseline_skill_score(
                        g["observed"], g[pred_col], g["pred_training_mean"]
                    ),
                    "skill_vs_seasonal_climatology": _baseline_skill_score(
                        g["observed"], g[pred_col], g["pred_seasonal_climatology"]
                    ),
                    "beats_training_mean": bool(fm["rmse"] < mean_fm["rmse"]),
                    "beats_seasonal_climatology": bool(fm["rmse"] < seasonal_fm["rmse"]),
                    "nrmse_reference_range": pooled_obs_range,
                })

            metadata_rows.append({
                "site": str(inputname_site),
                "target": r.get("target"),
                "target_units": target_units,
                "target_sign_convention": target_sign,
                "diurnal_mode": r.get("diurnal_mode"),
                "daylight_variable": r.get("daylight_variable"),
                "daylight_threshold": r.get("daylight_threshold"),
                "model": r.get("model"),
                "features": ", ".join(r.get("features", [])),
                "gate_variable": r.get("gate_variable"),
                "resample": r.get("resample"),
                "resampling_summary": r.get("resampling_summary"),
                "min_resample_coverage": r.get("min_resample_coverage"),
                "sum_override_columns": ", ".join(r.get("sum_override_columns", [])),
                "mean_override_columns": ", ".join(r.get("mean_override_columns", [])),
                "split_strategy": r.get("split_strategy"),
                "requested_n_folds": r.get("requested_n_folds"),
                "completed_n_folds": r.get("n_folds"),
                "validation_reference": r.get("validation_reference"),
                "comparison_gate_variable": r.get("comparison_gate_variable"),
                "use_common_sequence_folds": r.get("use_common_sequence_folds"),
                "initial_train_fraction": r.get("initial_train_fraction"),
                "random_seed": r.get("random_seed"),
                "analysis_period_mode": r.get("analysis_period_mode", analysis_period_mode),
                "requested_analysis_start": r.get("analysis_start"),
                "requested_analysis_end": r.get("analysis_end"),
                "selected_analysis_start": r.get("selected_analysis_start"),
                "selected_analysis_end": r.get("selected_analysis_end"),
                "model_ready_start": r.get("model_ready_start"),
                "model_ready_end": r.get("model_ready_end"),
                "common_test_start": common_test_start,
                "common_test_end": common_test_end,
                "n_common_test_observations": int(len(common)),
                "common_test_gaps": int(common_gap_info["n_gaps"]),
                "common_test_max_interval_days": common_gap_info["max_interval_days"],
                **counts,
                "sequence_length": r.get("sequence_length"),
                "sequence_policy": r.get("sequence_policy"),
                "expected_timestep": r.get("expected_timestep"),
                "n_candidate_sequence_windows": r.get("n_candidate_sequence_windows"),
                "n_valid_sequences": r.get("n_valid_sequences"),
                "n_rejected_gap_windows": r.get("n_rejected_gap_windows", 0),
                "n_rejected_missing_endpoint_target": r.get(
                    "n_rejected_missing_endpoint_target", 0
                ),
                "n_rejected_missing_predictor_windows": r.get(
                    "n_rejected_missing_predictor_windows", 0
                ),
                "target_requirement": r.get("target_requirement"),
                "predictor_requirement": r.get("predictor_requirement"),
                "n_source_gaps": r.get("n_source_gaps", 0),
                "max_source_interval_days": r.get("max_source_interval_days"),
                "hidden_size": r.get("hidden_size"),
                "epochs": r.get("epochs"),
                "batch_size": r.get("batch_size"),
                "scaling": r.get("scaling"),
                "missing_data_handling": r.get("missing_data_handling"),
                "outlier_handling": r.get("outlier_handling"),
                "model_parameters": str(r.get("model_parameters")),
                "software_versions": str(_software_versions()),
            })

        coverage_df = first_run.get("variable_coverage_df")
        if coverage_df is None or getattr(coverage_df, "empty", True):
            coverage_df = pd.DataFrame()
        else:
            coverage_df = coverage_df.copy()
            hlstm_run = next(r for r in runs if r.get("model") == "Hysteresis-Gate LSTM (H-LSTM)")
            gate = hlstm_run.get("gate_variable")
            if gate and "variable" in coverage_df.columns:
                coverage_df.loc[coverage_df["variable"] == gate, "role"] = "predictor; H-LSTM gate"
            coverage_df.insert(0, "site", str(inputname_site))
            coverage_df.insert(1, "target", first_run.get("target"))
            coverage_df.insert(2, "resample", first_run.get("resample"))
            coverage_df.insert(3, "selected_analysis_start", selected_analysis_start)
            coverage_df.insert(4, "selected_analysis_end", selected_analysis_end)

        return (
            common,
            pd.DataFrame(metric_rows),
            pd.DataFrame(common_fold_rows),
            pd.DataFrame(metadata_rows),
            coverage_df,
        )

    def _export_fig3b_comparison():
        from tkinter.filedialog import asksaveasfilename
        try:
            common, metrics_df, folds_df, metadata_df, coverage_df = _build_common_timestamp_comparison()
        except Exception as exc:
            messagebox.showerror("Figure 3B results", str(exc))
            return

        target = str(metrics_df["target"].iloc[0]) if not metrics_df.empty else "target"
        site = str(inputname_site).replace(".csv", "") if inputname_site else "site"
        path = asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            title="Save Figure 3B results (choose base filename)",
            initialfile=f"Figure3B_{site}_{target}_common_predictions.csv".replace(" ", "_"),
        )
        if not path:
            return
        base = path[:-4] if path.lower().endswith(".csv") else path
        paths = {
            "predictions": base + ".csv",
            "metrics": base + "_metrics.csv",
            "fold metrics": base + "_fold_metrics.csv",
            "metadata": base + "_metadata.csv",
            "coverage": base + "_coverage.csv",
        }
        common.to_csv(paths["predictions"], index=False)
        metrics_df.to_csv(paths["metrics"], index=False)
        folds_df.to_csv(paths["fold metrics"], index=False)
        metadata_df.to_csv(paths["metadata"], index=False)
        coverage_df.to_csv(paths["coverage"], index=False)

        import os
        save_folder = os.path.dirname(os.path.abspath(paths["predictions"]))
        file_list = "\n".join(f"• {os.path.basename(v)}" for v in paths.values())
        messagebox.showinfo(
            "Figure 3B results",
            f"Saved to:\n{save_folder}\n\nFiles:\n{file_list}\n\n"
            "Periods, data retention, fold train/test windows, gap-safe sequence counts, "
            "metrics, and variable coverage are included."
        )

    handles["build_fig3b_results"] = _build_common_timestamp_comparison
    handles["export_fig3b_results"] = _export_fig3b_comparison
    btn_export_fig3b.configure(command=_export_fig3b_comparison)

    # ----------------------- Compare tab logic -----------------------
    def _update_compare_tab():
        for r in tree.get_children(): tree.delete(r)
        runs = handles["runs"]
        for i, r in enumerate(runs):
            tree.insert("", "end", iid=str(i), values=(
                str(r["when"]).split(".")[0],
                r["model"], r["resample"],
                r.get("split_strategy", ""), r.get("n_folds", 1),
                r.get("n_train", ""), r.get("n_test", ""),
                r["target"],
                ", ".join(r["features"]),
                f'{r["rmse"]:.4g}', f'{r.get("nrmse_p5_p95", np.nan):.4g}', f'{r["mae"]:.4g}', f'{r["r2"]:.4g}'
            ))
        ax_cmp.cla()
        met = handles["cmp_metric"].get()
        if runs:
            labels = [_comparison_run_label(rr, i) for i, rr in enumerate(runs)]
            vals = [rr["rmse"] if met=="RMSE" else rr.get("nrmse_p5_p95", np.nan) if met=="nRMSE" else rr["mae"] if met=="MAE" else rr["r2"] for rr in runs]
            xpos = np.arange(len(vals))
            ax_cmp.bar(xpos, vals)
            ax_cmp.set_xticks(xpos); ax_cmp.set_xticklabels(labels, rotation=30, ha="right")
            ax_cmp.set_ylabel(met); ax_cmp.set_title(f"Model comparison by {met}")
        handles["can_cmp"].draw_idle()
        _plot_compare_ecdf()

    def _plot_compare_ecdf():
        """Plot an empirical cumulative distribution of the selected metric across runs.

        For one site this summarizes all model/configuration runs. Across many imported
        runs or repeated windows/sites, this becomes a SAGE-style distributional
        performance diagnostic rather than a single summary value.
        """
        ax = handles["ax_ecdf"]
        ax.cla()
        runs = handles.get("runs", [])
        met = handles["cmp_metric"].get()
        if not runs:
            ax.text(0.5, 0.5, "No runs yet. Run models first.", ha="center", va="center", transform=ax.transAxes)
            handles["can_ecdf"].draw_idle()
            return
        vals = []
        labels = []
        for rr in runs:
            v = rr["rmse"] if met == "RMSE" else rr.get("nrmse_p5_p95", np.nan) if met == "nRMSE" else rr["mae"] if met == "MAE" else rr["r2"]
            try:
                if np.isfinite(float(v)):
                    vals.append(float(v))
                    labels.append(str(rr.get("model", "model")))
            except Exception:
                pass
        if not vals:
            ax.text(0.5, 0.5, f"No finite {met} values.", ha="center", va="center", transform=ax.transAxes)
            handles["can_ecdf"].draw_idle()
            return
        vals = np.sort(np.asarray(vals, dtype=float))
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.step(vals, y, where="post")
        ax.scatter(vals, y, s=18)
        ax.set_xlabel(met)
        ax.set_ylabel("ECDF")
        direction = "lower is better" if met in ("RMSE", "nRMSE", "MAE") else "higher is better"
        ax.set_title(f"Distribution of {met} across model runs ({direction})")
        ax.grid(True, alpha=0.3)
        handles["can_ecdf"].draw_idle()

    def _overlay_selected_preds():
        sel_items = tree.selection()
        if not sel_items:
            messagebox.showinfo("Compare", "Select one or more rows in the table.")
            return
        ax_ov.cla()
        drew_true = False
        for selected_pos, iid in enumerate(sel_items):
            run_index = int(iid)
            r = handles["runs"][run_index]
            comparison_label = _comparison_run_label(r, run_index)
            pdf = r.get("pred_df")
            if pdf is None or pdf.empty: continue
            # x-axis
            if "timestamp" in pdf.columns:
                x = _to_datetime_1d(pdf["timestamp"])
                xlab = "Time"
            elif "index" in pdf.columns:
                x = pdf["index"].values
                xlab = "Index"
            else:
                x = np.arange(len(pdf)); xlab = "Index"
            # draw true once
            if not drew_true and "y_true" in pdf.columns:
                if "timestamp" in pdf.columns:
                    _plot_gap_aware_line(
                        ax_ov, x, pdf["y_true"].values,
                        label="True", alpha=0.8, marker="o", markersize=2,
                    )
                else:
                    ax_ov.plot(x, pdf["y_true"].values, label="True", alpha=0.8)
                drew_true = True
            # draw prediction with temporal gaps shown as breaks
            if "timestamp" in pdf.columns:
                _plot_gap_aware_line(
                    ax_ov, x, pdf["y_pred"].values, "--",
                    label=f"Pred: {comparison_label}", marker="o", markersize=2,
                )
            else:
                ax_ov.plot(x, pdf["y_pred"].values, "--", label=f"Pred: {comparison_label}")
        ax_ov.set_title("Overlay predictions")
        ax_ov.legend(loc="best")
        ax_ov.set_xlabel(xlab); ax_ov.set_ylabel(handles["y_var"].get())
        handles["can_ov"].draw_idle()

    btn_refresh.configure(command=_update_compare_tab)
    btn_overlay.configure(command=_overlay_selected_preds)
    btn_ecdf.configure(command=_plot_compare_ecdf)
    btn_export_fig2c.configure(command=_export_manuscript_fig2c)
    btn_export_publication.configure(command=_export_publication_outputs)
    # Figure 3B export command is attached after the common-timestamp helper is defined.
    btn_export_it.configure(command=_export_it_bridge)
    btn_clear.configure(command=lambda: (handles["runs"].clear(), _update_compare_tab()))

    handles["compare_refresh"] = _update_compare_tab
    handles.update(dict(last_pred_df=None, last_importance_df=None))
    return tab_predict, handles

def _make_it_bridge_df(dfw, features, target, y_true, y_pred, test_idx=None,
                       timestamp=None, model_name="model"):
    """Build a timestamp-aligned ML-to-IT table for one held-out model run.

    For sequence models, alignment is performed by prediction timestamp rather
    than by positional row index. This prevents shifted drivers or targets from
    entering downstream information-theory analyses.
    """
    def _safe_model_tag(name):
        tag = str(name).replace("Hysteresis-Gate LSTM (H-LSTM)", "HLSTM")
        tag = tag.replace("Linear Regression", "MLR")
        tag = tag.replace("Random Forest", "RF")
        tag = tag.replace("Neural Network (MLP)", "MLP")
        tag = tag.replace("LSTM (Keras)", "LSTM")
        return "".join(ch if ch.isalnum() else "_" for ch in tag).strip("_") or "model"

    tag = _safe_model_tag(model_name)
    pred_col = f"{target}_pred_{tag}"
    resid_col = f"{target}_resid_{tag}"
    obs_col = f"{target}_obs"
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.size != y_pred.size:
        raise ValueError("Observed and predicted arrays must have the same length.")

    try:
        if timestamp is not None and len(timestamp) == len(y_true):
            ts_values = pd.Series(_to_datetime_1d(pd.Series(timestamp))).reset_index(drop=True)
            out = pd.DataFrame({"timestamp": ts_values})

            if dfw is not None and len(features):
                base = dfw.copy()
                if "timestamp" in base.columns:
                    base_ts = _to_datetime_1d(base["timestamp"])
                else:
                    base_ts = _to_datetime_1d(pd.Series(base.index, index=base.index))
                base = base.copy()
                base["__MF_TS__"] = np.asarray(base_ts)
                base = base.dropna(subset=["__MF_TS__"])
                # Duplicate timestamps can occur after preprocessing; keep the
                # first deterministic occurrence to preserve one-to-one merging.
                base = base.drop_duplicates(subset=["__MF_TS__"], keep="first")
                driver_cols = [c for c in features if c in base.columns]
                if driver_cols:
                    drivers = base[["__MF_TS__"] + driver_cols].rename(columns={"__MF_TS__": "timestamp"})
                    out = out.merge(drivers, on="timestamp", how="left", validate="many_to_one")
        elif dfw is not None and test_idx is not None:
            base = dfw.iloc[np.asarray(test_idx, dtype=int)].copy()
            driver_cols = [c for c in features if c in base.columns]
            out = base[driver_cols].reset_index(drop=True)
            out.insert(0, "timestamp", _to_datetime_1d(pd.Series(base.index)).to_numpy())
        else:
            out = pd.DataFrame({"index": np.arange(len(y_true), dtype=int)})

        if len(out) != len(y_true):
            raise ValueError("Timestamp/driver alignment changed the number of prediction rows.")
        out[obs_col] = y_true
        out[pred_col] = y_pred
        out[resid_col] = y_pred - y_true
        out["split"] = "test"
        return out, pred_col, resid_col
    except Exception:
        # Preserve predictions even when driver alignment is unavailable; this
        # fallback is explicit and never silently shifts rows.
        out = pd.DataFrame({
            "index": np.arange(len(y_true), dtype=int),
            obs_col: y_true,
            pred_col: y_pred,
        })
        if timestamp is not None and len(timestamp) == len(y_true):
            out.insert(0, "timestamp", _to_datetime_1d(pd.Series(timestamp)).to_numpy())
            out = out.drop(columns=["index"])
        out[resid_col] = y_pred - y_true
        out["split"] = "test"
        return out, pred_col, resid_col


def _stash_exportables_predict(P, y_true, y_pred, timestamp, feature_names, importances,
                               fold_ids=None, importance_std=None, importance_scoring=None):
    """Store reproducible prediction and importance tables for export."""
    try:
        if timestamp is not None and len(timestamp) == len(y_true):
            pred_df = pd.DataFrame({"timestamp": _to_datetime_1d(timestamp), "y_true": y_true, "y_pred": y_pred})
        else:
            pred_df = pd.DataFrame({"index": np.arange(len(y_true)), "y_true": y_true, "y_pred": y_pred})
        if fold_ids is not None and len(fold_ids) == len(pred_df):
            pred_df["fold"] = np.asarray(fold_ids, dtype=int)
        P["last_pred_df"] = pred_df
    except Exception:
        P["last_pred_df"] = None
    try:
        if feature_names is not None and importances is not None:
            fn = list(feature_names)
            imp = np.asarray(importances, dtype=float).ravel()
            L = min(len(fn), imp.shape[0])
            out = pd.DataFrame({"feature": fn[:L], "importance": imp[:L]})
            if importance_std is not None:
                std = np.asarray(importance_std, dtype=float).ravel()
                out["importance_std"] = std[:L]
            if importance_scoring:
                out["scoring"] = str(importance_scoring)
            P["last_importance_df"] = out
        else:
            P["last_importance_df"] = None
    except Exception:
        P["last_importance_df"] = None


def _update_predict_plots(P, y_true, y_pred, title, y_var, timestamp=None,
                          feature_names=None, importances=None, split_info=None,
                          fold_ids=None, importance_std=None, importance_scoring=None):
    """Redraw held-out/out-of-fold diagnostics and original-unit metrics."""
    _stash_exportables_predict(
        P, y_true, y_pred, timestamp, feature_names, importances,
        fold_ids=fold_ids, importance_std=importance_std,
        importance_scoring=importance_scoring,
    )

    ax = P["ax_series"]
    ax.cla()
    note_parts = ["Showing held-out predictions"]
    if split_info:
        note_parts.append(f"Validation: {split_info.get('split_strategy', 'n/a')}")
        note_parts.append(f"Testing: {split_info.get('test_period', 'n/a')}")
        note_parts.append(f"OOF/test rows: {split_info.get('n_test', 'n/a')}")
        if split_info.get("n_folds"):
            note_parts.append(f"Folds: {split_info.get('n_folds')}")
    try:
        P["lbl_series_note"].configure(text=" | ".join(note_parts))
    except Exception:
        pass

    if timestamp is not None and len(timestamp) == len(y_true):
        order = np.argsort(np.asarray(_to_datetime_1d(timestamp)))
        t_axis = np.asarray(_to_datetime_1d(timestamp))[order]
        yt_plot = np.asarray(y_true)[order]
        yp_plot = np.asarray(y_pred)[order]
        _plot_gap_aware_line(
            ax, t_axis, yt_plot, label="Observed held-out", marker="o", markersize=2,
        )
        _plot_gap_aware_line(
            ax, t_axis, yp_plot, "--", label="Predicted held-out", marker="o", markersize=2,
        )
        ax.set_xlabel("Time")
    else:
        ax.plot(y_true, label="Observed held-out")
        ax.plot(y_pred, "--", label="Predicted held-out")
        ax.set_xlabel("Held-out sample")
    ax.set_title(f"Held-out prediction — {title}")
    ax.set_ylabel(y_var)
    ax.legend(loc="upper right", frameon=True, framealpha=0.88, borderpad=0.4)
    ax.grid(True, alpha=0.25)
    P["can_series"].draw_idle()

    ax = P["ax_scat"]
    ax.cla()
    ax.scatter(y_true, y_pred, s=10, alpha=0.6)
    mn = float(np.nanmin([np.nanmin(y_true), np.nanmin(y_pred)]))
    mx = float(np.nanmax([np.nanmax(y_true), np.nanmax(y_pred)]))
    ax.plot([mn, mx], [mn, mx], "k--", lw=1)
    ax.set_title("Observed vs predicted")
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.grid(True, alpha=0.25)
    P["can_scat"].draw_idle()

    ax = P["ax_resid"]
    ax.cla()
    resid = np.asarray(y_pred) - np.asarray(y_true)
    ax.hist(resid, bins=30, alpha=0.85)
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.set_title("Held-out residuals")
    ax.set_xlabel("Predicted - observed")
    ax.set_ylabel("Count")
    P["can_resid"].draw_idle()

    metrics = _metric_bundle(y_true, y_pred)
    P["lbl_rmse"].config(text=f"RMSE: {metrics['rmse']:.4g}")
    P["lbl_nrmse"].config(text=f"nRMSE (P5–P95): {metrics['nrmse_p5_p95']:.3g}%")
    P["lbl_mae"].config(text=f"MAE: {metrics['mae']:.4g}")
    P["lbl_r2"].config(text=f"R²: {metrics['r2']:.4g}")

    try:
        interp = _interpret_ml_result(
            title, y_var, list(feature_names or []), metrics["rmse"], metrics["mae"],
            metrics["r2"], title.split("|")[-1].strip() if "|" in title else "Native",
            nrmse=metrics["nrmse_p5_p95"],
        )
        if P.get("interp_text") is not None:
            P["interp_text"].configure(state="normal")
            P["interp_text"].delete("1.0", "end")
            P["interp_text"].insert("1.0", interp)
            P["interp_text"].configure(state="disabled")
    except Exception:
        pass

    ax = P["ax_imp"]
    ax.cla()
    if feature_names is not None and importances is not None:
        fn = list(feature_names)
        imp = np.asarray(importances, dtype=float).ravel()
        L = min(len(fn), len(imp))
        fn, imp = fn[:L], imp[:L]
        order = np.argsort(np.nan_to_num(imp, nan=-np.inf))
        y_pos = np.arange(L)
        xerr = None
        if importance_std is not None:
            std = np.asarray(importance_std, dtype=float).ravel()[:L]
            xerr = std[order]
        ax.barh(y_pos, imp[order], xerr=xerr, capsize=2 if xerr is not None else 0)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(np.asarray(fn, dtype=object)[order])
        ax.set_title("Held-out permutation importance")
        ax.set_xlabel("Decrease in held-out score when permuted")
        ax.grid(True, axis="x", alpha=0.25)
    else:
        ax.text(0.5, 0.5, "Held-out permutation importance is available for Random Forest", ha="center", va="center", transform=ax.transAxes)
    P["can_imp"].draw_idle()

    try:
        P["results_notebook"].select(P["tab_series"])
    except Exception:
        pass

# =============================================================================
#        OPTIONAL PREDICTOR SCREENING TAB: BUILD + PLOTTING
# =============================================================================
def _build_drivers_tab(parent_notebook, df, inputname_site):
    tab = ttk.Frame(parent_notebook)
    parent_notebook.add(tab, text="3. Optional Predictor Screening")

    splitter = ttk.Panedwindow(tab, orient="horizontal")
    splitter.pack(fill="both", expand=True)

    # LEFT controls: compact scrollable sidebar
    controls = _scrollable_controls(splitter, width=405, padding=(10, 10))

    # --- RIGHT: results notebook (unchanged) ---
    results = ttk.Notebook(splitter)
    splitter.add(results, weight=1)


    Label(controls, text="Optional Predictor Screening", font=("TkDefaultFont", 11, "bold")).grid(sticky="w", pady=(0,6))
    _concept_question(
        controls,
        "Do literature-informed predictors agree with statistical and predictive screening?",
        "Compare literature-based and statistical evidence for candidate predictors.",
        width=330,
    )

    # Unique, alphabetized column names for easier search
    cols = _sorted_unique_columns(df)

    selector_box = ttk.LabelFrame(controls, text="Step 1: select target and candidate drivers")
    selector_box.grid(sticky="ew", pady=6)
    selector_box.columnconfigure(0, weight=1)

    default_ts = _guess_timestamp_column(cols)
    default_target = _guess_default_target(cols, timestamp_col=default_ts)

    ttk.Label(selector_box, text="Target flux / variable (y)").grid(sticky="w", padx=6, pady=(6,0))
    yD = StringVar(selector_box, value=default_target)
    ttk.OptionMenu(selector_box, yD, yD.get(), *cols).grid(sticky="ew", padx=6, pady=(0,6))

    ttk.Label(selector_box, text="Timestamp column").grid(sticky="w", padx=6)
    tsD = StringVar(selector_box, value=default_ts)
    ttk.OptionMenu(selector_box, tsD, tsD.get(), *cols).grid(sticky="ew", padx=6, pady=(0,6))

    box_period_D = ttk.LabelFrame(controls, text="Analysis period for screening")
    box_period_D.grid(sticky="ew", pady=6)
    box_period_D.columnconfigure(1, weight=1)
    d_start, d_end, d_full, d_period_msg = _add_analysis_period_controls(
        box_period_D, df, tsD, row_start=0, width=330
    )

    ttk.Label(selector_box, text="Candidate predictors auto-selected from target preset (review/edit)").grid(sticky="w", padx=6)
    cand_list_frame = ttk.Frame(selector_box)
    cand_list_frame.grid(sticky="ew", padx=6, pady=(0,6))
    cand_list_frame.columnconfigure(0, weight=1)
    lb_D = Listbox(cand_list_frame, selectmode=MULTIPLE, exportselection=False, height=7)
    cand_scroll = ttk.Scrollbar(cand_list_frame, orient="vertical", command=lb_D.yview)
    lb_D.configure(yscrollcommand=cand_scroll.set)
    for i, c in enumerate(cols):
        lb_D.insert(i, c)
    lb_D.grid(row=0, column=0, sticky="ew")
    cand_scroll.grid(row=0, column=1, sticky="ns")

    # Target-aware predictor preset
    box_driver_preset = ttk.LabelFrame(controls, text="Target-aware predictor preset")
    box_driver_preset.grid(sticky="ew", pady=6)
    box_driver_preset.columnconfigure(0, weight=1)
    driver_preset_msg = tk.Message(
        box_driver_preset,
        width=320,
        text="Select a target flux to see recommended candidate drivers.",
        fg="#444"
    )
    driver_preset_msg.grid(row=0, column=0, columnspan=2, sticky="ew", padx=6, pady=(6,4))
    auto_driver_preset_var = tk.BooleanVar(box_driver_preset, value=True)
    ttk.Checkbutton(
        box_driver_preset,
        text="Auto-select when target changes",
        variable=auto_driver_preset_var
    ).grid(row=1, column=0, sticky="w", padx=6, pady=(0,6))

    def _apply_driver_preset(show_note=False):
        selected, family, guide = _apply_predictor_preset_to_listbox(
            lb_D, cols, yD.get(), timestamp_col=tsD.get(), status_widget=driver_preset_msg, select=True
        )
        if show_note:
            messagebox.showinfo(
                "Target-aware predictor preset",
                f"Applied preset for {family}.\n\nSelected {len(selected)} candidate predictor(s):\n" +
                (", ".join(selected) if selected else "None found automatically.") +
                "\n\nUse Analyze candidate predictors to compare literature, correlation, RF importance, and linear coefficients."
            )

    ttk.Button(
        box_driver_preset,
        text="Apply preset now",
        command=lambda: _apply_driver_preset(show_note=True)
    ).grid(row=1, column=1, sticky="e", padx=6, pady=(0,6))

    def _on_driver_target_changed(*_):
        if auto_driver_preset_var.get():
            _apply_driver_preset(show_note=False)
        else:
            _apply_predictor_preset_to_listbox(
                lb_D, cols, yD.get(), timestamp_col=tsD.get(), status_widget=driver_preset_msg, select=False
            )

    try:
        yD.trace_add("write", _on_driver_target_changed)
        tsD.trace_add("write", lambda *_: _on_driver_target_changed())
    except Exception:
        pass
    _apply_driver_preset(show_note=False)

    box_cfg = ttk.LabelFrame(controls, text="Step 2: temporal aggregation for driver screening"); box_cfg.grid(sticky="ew", pady=6)
    ttk.Label(box_cfg, text="Time resolution").grid(row=0, column=0, sticky="w", padx=6, pady=(6,4))
    rsD = IntVar(box_cfg, value=0)
    ttk.Radiobutton(box_cfg, text="Native", variable=rsD, value=0).grid(row=0, column=1, sticky="w")
    ttk.Radiobutton(box_cfg, text="Daily", variable=rsD, value=1).grid(row=0, column=2, sticky="w")
    ttk.Radiobutton(box_cfg, text="Weekly", variable=rsD, value=2).grid(row=0, column=3, sticky="w")
    box_cfg.columnconfigure(4, weight=1)

    ttk.Button(controls, text="Step 3: Analyze candidate predictors",
               command=lambda: threading.Thread(target=_run_drivers_worker,
                        args=(df, lb_D, yD, tsD, rsD, results), daemon=True).start()
               ).grid(sticky="ew", pady=8)

    # Link to the Hysteresis / Gate Explorer tab
    ttk.Separator(controls, orient="horizontal").grid(sticky="ew", pady=10)
    link_box = ttk.LabelFrame(controls, text="Step 4: follow-up diagnostic")
    link_box.grid(sticky="ew", pady=6)
    tk.Message(
        link_box,
        width=320,
        fg="#444",
        text=(
            "After identifying candidate predictors, open the Hysteresis / Gate Explorer tab to test whether one driver shows path-dependent behavior. "
            "Use this especially for possible H-LSTM gate variables such as VPD, SWC, radiation, temperature, or u*."
        ),
    ).grid(sticky="ew", padx=6, pady=6)

    # Results notebook tabs
    tab_guide = ttk.Frame(results); results.add(tab_guide, text="Guide")
    tab_imp = ttk.Frame(results); results.add(tab_imp, text="Importance")
    tab_corr = ttk.Frame(results); results.add(tab_corr, text="Correlation")
    tab_pdp = ttk.Frame(results); results.add(tab_pdp, text="PDP")
    tab_cmp = ttk.Frame(results); results.add(tab_cmp, text="Compare evidence")

    # Guide
    guide_frame = ttk.Frame(tab_guide, padding=10)
    guide_frame.pack(fill="both", expand=True)
    guide_text = tk.Text(guide_frame, wrap="word", height=18)
    guide_text.pack(fill="both", expand=True)
    guide_text.insert(
        "1.0",
        (
            "MeaningFlux Optional Predictor Screening — guided workflow\n\n"
            "Purpose\n"
            "Optional Predictor Screening is a screening step. It compares literature-informed candidate predictors with statistical and predictive screening results for a selected target flux. It should not be interpreted as mechanistic proof by itself.\n\n"
            "Step 1 — Select the target flux or variable.\n"
            "Examples include FC, FCH4, FN2O, LE, H, GPP, or RECO. The selected target defines the predictor-screening problem.\n\n"
            "Step 2 — Select the timestamp and temporal aggregation.\n"
            "Use Native for native half-hourly/hourly screening, daily aggregation for slower ecosystem-scale patterns, or weekly aggregation for seasonal behavior. Compare rankings only across runs with the same aggregation.\n\n"
            "Step 3 — Review candidate predictors.\n"
            "MeaningFlux applies a target-based preset when possible, but the user should remove target-derived variables, gap-filled target variants, QC flags, uncertainty fields, residuals, and model-prediction columns.\n\n"
            "Step 4 — Analyze predictor evidence.\n"
            "The toolbox compares Random Forest permutation importance, absolute Pearson correlation, and absolute linear-regression coefficients. Predictors supported across multiple evidence types are stronger candidates for downstream modeling and information-theoretic analysis.\n\n"
            "Step 5 — Inspect PDP and compare evidence.\n"
            "Partial dependence plots show the fitted marginal response for top drivers in the Random Forest model. Use them as exploratory diagnostics, not as causal evidence.\n\n"
            "Connection to Hysteresis / Gate Explorer\n"
            "After identifying plausible predictors, open the Hysteresis / Gate Explorer tab to test whether one selected driver produces different target-flux responses during rising versus falling conditions. This is especially useful for choosing an H-LSTM gate variable such as VPD, soil water content, radiation, temperature, or u*.\n\n"
            "Connection to Predictive Modeling and IT\n"
            "Use Optional Predictor Screening as a sensitivity check, not as the primary predictor-selection rule. Then run Predictive Modeling to quantify out-of-sample prediction skill and export the ML-to-IT bridge table. Finally, use the Information Theory toolbox to test whether the model preserves observed driver–flux information structure.\n"
        )
    )
    guide_text.configure(state="disabled")

    # Importance
    fig_imp, ax_imp = plt.subplots(figsize=(8.6, 3.2))
    can_imp = FigureCanvasTkAgg(fig_imp, master=tab_imp); can_imp.draw(); can_imp.get_tk_widget().pack(fill="both", expand=True)

    # Correlation
    fig_corr, ax_corr = plt.subplots(figsize=(8.6, 3.2))
    can_corr = FigureCanvasTkAgg(fig_corr, master=tab_corr); can_corr.draw(); can_corr.get_tk_widget().pack(fill="both", expand=True)

    # PDP
    fig_pdp, axs_pdp = plt.subplots(1, 3, figsize=(9.2, 3.2))
    can_pdp = FigureCanvasTkAgg(fig_pdp, master=tab_pdp); can_pdp.draw(); can_pdp.get_tk_widget().pack(fill="both", expand=True)

    # Compare evidence tab
    row_cmp = ttk.Frame(tab_cmp); row_cmp.pack(fill="x", padx=8, pady=(6,4))
    Label(row_cmp, text="Evidence metric:").pack(side="left")
    cmp_metric = StringVar(tab_cmp, value="RF Permutation Importance")
    ttk.OptionMenu(
        row_cmp,
        cmp_metric,
        "RF Permutation Importance",
        "RF Permutation Importance",
        "|Pearson r|",
        "|Standardized linear coef|"
    ).pack(side="left", padx=(6,18))
    tk.Message(
        tab_cmp,
        width=600,
        text=(
            "This view compares predictor evidence from target-aware presets and three screening methods:\n"
            "• RF Permutation Importance: held-out score decrease across expanding future blocks when each driver is shuffled.\n"
            "• |Pearson r|: absolute linear correlation with the target.\n"
            "• |Standardized linear coefficient|: comparable multivariate linear effect size after scaling predictors and target.\n"
            "Use it to see whether literature-informed predictors are also supported by linear association or nonlinear predictive importance. Strong candidates can then be inspected in the Hysteresis / Gate Explorer before running H-LSTM."
        ),
        fg="#444"
    ).pack(fill="x", padx=8, pady=(0,4))

    cols_ev = ("feature", "literature", "pearson", "rf", "linreg", "interpretation")
    evidence_tree = ttk.Treeview(tab_cmp, columns=cols_ev, show="headings", height=7)
    headings = {
        "feature": "PREDICTOR",
        "literature": "PRESET",
        "pearson": "|r|",
        "rf": "RF IMP.",
        "linreg": "|STD. LIN. COEF|",
        "interpretation": "INTERPRETATION",
    }
    widths = {"feature": 150, "literature": 70, "pearson": 80, "rf": 90, "linreg": 90, "interpretation": 460}
    for c in cols_ev:
        evidence_tree.heading(c, text=headings[c])
        evidence_tree.column(c, width=widths[c], anchor="w")
    evidence_tree.pack(fill="x", padx=8, pady=(0,6))

    fig_cmp, ax_cmp = plt.subplots(figsize=(8.6, 3.2))
    can_cmp = FigureCanvasTkAgg(fig_cmp, master=tab_cmp); can_cmp.draw(); can_cmp.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0,8))

    handles = dict(
        lb_D=lb_D, yD=yD, tsD=tsD, rsD=rsD,
        ax_imp=ax_imp, can_imp=can_imp,
        ax_corr=ax_corr, can_corr=can_corr,
        axs_pdp=axs_pdp, can_pdp=can_pdp,
        results=results,
        last_importance_df=None,
        # compare drivers
        cmp_metric=cmp_metric, ax_cmp=ax_cmp, can_cmp=can_cmp, evidence_tree=evidence_tree,
        imp_rf=None, imp_corr=None, imp_lin=None, evidence_df=None, all_cols=cols,
        predictor_preset_message=driver_preset_msg, auto_preset=auto_driver_preset_var,
        analysis_start=d_start, analysis_end=d_end, use_full_period=d_full
    )
    return tab, handles

def _rank_series(values, features):
    """Return a feature->rank dictionary where rank 1 is largest value."""
    try:
        arr = np.asarray(values, dtype=float)
        order = np.argsort(np.nan_to_num(arr, nan=-np.inf))[::-1]
        return {str(features[i]): int(k + 1) for k, i in enumerate(order)}
    except Exception:
        return {}

def _update_predictor_evidence_table(D, features, target):
    """Populate the evidence table that compares literature presets with screening metrics."""
    tree = D.get("evidence_tree")
    if tree is None:
        return
    for item in tree.get_children():
        tree.delete(item)

    preset = []
    try:
        preset, _, _ = _recommended_predictors_for_target(
            D.get("all_cols", []),
            target,
            timestamp_col=D.get("tsD").get() if D.get("tsD") is not None else None,
        )
    except Exception:
        preset = []

    rf_df = D.get("imp_rf")
    corr_df = D.get("imp_corr")
    lin_df = D.get("imp_lin")
    rf_vals = dict(zip(rf_df["feature"], rf_df["value"])) if rf_df is not None and not getattr(rf_df, "empty", True) else {}
    corr_vals = dict(zip(corr_df["feature"], corr_df["value"])) if corr_df is not None and not getattr(corr_df, "empty", True) else {}
    lin_vals = dict(zip(lin_df["feature"], lin_df["value"])) if lin_df is not None and not getattr(lin_df, "empty", True) else {}

    rf_rank = _rank_series([rf_vals.get(f, np.nan) for f in features], features)
    corr_rank = _rank_series([corr_vals.get(f, np.nan) for f in features], features)
    lin_rank = _rank_series([lin_vals.get(f, np.nan) for f in features], features)
    top_n = max(1, min(5, len(features)))

    rows = []
    for f in features:
        lit = "yes" if f in preset else "no"
        supports = 0
        if lit == "yes":
            supports += 1
        if corr_rank.get(f, 999) <= top_n:
            supports += 1
        if rf_rank.get(f, 999) <= top_n:
            supports += 1
        if lin_rank.get(f, 999) <= top_n:
            supports += 1

        if lit == "yes" and rf_rank.get(f, 999) <= top_n and corr_rank.get(f, 999) <= top_n:
            interp = "Consistent preset + linear + nonlinear support"
        elif lit == "yes" and rf_rank.get(f, 999) <= top_n:
            interp = "Preset-supported; nonlinear predictive support"
        elif lit == "yes" and corr_rank.get(f, 999) <= top_n:
            interp = "Preset-supported; linear association support"
        elif rf_rank.get(f, 999) <= top_n and corr_rank.get(f, 999) > top_n:
            interp = "Possible nonlinear or interaction-dependent predictor"
        elif lit == "no" and supports >= 2:
            interp = "Unexpected statistical/predictive candidate; inspect before using"
        elif supports <= 1:
            interp = "Weak or method-specific support"
        else:
            interp = "Mixed evidence; inspect with IT diagnostics"

        rows.append({
            "feature": f,
            "literature": lit,
            "pearson": corr_vals.get(f, np.nan),
            "rf": rf_vals.get(f, np.nan),
            "linreg": lin_vals.get(f, np.nan),
            "interpretation": interp,
            "support_count": supports,
        })

    rows = sorted(rows, key=lambda r: (r["support_count"], np.nan_to_num(r["rf"], nan=-np.inf)), reverse=True)
    for i, r in enumerate(rows):
        tree.insert("", "end", iid=str(i), values=(
            r["feature"],
            r["literature"],
            "" if not np.isfinite(r["pearson"]) else f'{r["pearson"]:.3g}',
            "" if not np.isfinite(r["rf"]) else f'{r["rf"]:.3g}',
            "" if not np.isfinite(r["linreg"]) else f'{r["linreg"]:.3g}',
            r["interpretation"],
        ))
    try:
        D["evidence_df"] = pd.DataFrame(rows)
    except Exception:
        D["evidence_df"] = None

def _refresh_driver_compare(D):
    """Update the 'Compare evidence' bar chart based on selected metric."""
    ax = D["ax_cmp"]; ax.cla()
    metric = D["cmp_metric"].get()

    if metric == "RF Permutation Importance":
        dfm = D.get("imp_rf")
        ylabel = "Permutation importance (Δ score)"
        title = "Predictor evidence: RF Permutation Importance"
    elif metric == "|Pearson r|":
        dfm = D.get("imp_corr")
        ylabel = "|Pearson r|"
        title = "Predictor evidence: |Pearson correlation|"
    else:  # |Standardized linear coef|
        dfm = D.get("imp_lin")
        ylabel = "|standardized coefficient|"
        title = "Predictor evidence: |standardized linear coefficient|"

    if dfm is None or getattr(dfm, "empty", True):
        ax.text(0.5, 0.5, "No rankings computed yet.\nClick 'Analyze candidate predictors' first.",
                ha="center", va="center", transform=ax.transAxes)
        D["can_cmp"].draw_idle()
        return

    feats = dfm["feature"].tolist()
    vals = dfm["value"].astype(float).values
    xpos = np.arange(len(feats))
    ax.bar(xpos, vals)
    ax.set_xticks(xpos)
    ax.set_xticklabels(feats, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    D["can_cmp"].draw_idle()

def _update_driver_plots(D, dfw, features, target, corr_kind="pearson"):
    """Update optional predictor screening with held-out nonlinear evidence.

    RF permutation importance is evaluated on expanding future blocks. Pearson
    correlation remains a descriptive association, and linear coefficients are
    standardized so their magnitudes are comparable among predictors.
    """
    X = dfw[features].to_numpy(dtype=float)
    y_target = dfw[target].to_numpy(dtype=float)

    pim = pis = None
    rf_full = None
    scoring_used = None
    try:
        splits = _expanding_window_splits(len(dfw), n_folds=5, initial_train_fraction=0.50)
        fold_means = []
        for fold, tr, te in splits:
            rf_fold = RandomForestRegressor(
                n_estimators=300, random_state=_GLOBAL_SEED + fold, n_jobs=1
            ).fit(X[tr], y_target[tr])
            try:
                perm = permutation_importance(
                    rf_fold, X[te], y_target[te], n_repeats=10,
                    random_state=_GLOBAL_SEED + fold, scoring="r2", n_jobs=1,
                )
                if not np.isfinite(perm.importances_mean).any():
                    raise ValueError("Non-finite R2 permutation importance")
                scoring_used = "R2 decrease"
            except Exception:
                perm = permutation_importance(
                    rf_fold, X[te], y_target[te], n_repeats=10,
                    random_state=_GLOBAL_SEED + fold, scoring="neg_mean_squared_error", n_jobs=1,
                )
                scoring_used = "negative-MSE decrease"
            fold_means.append(np.asarray(perm.importances_mean, dtype=float))
        fold_means = np.vstack(fold_means)
        pim = np.nanmean(fold_means, axis=0)
        pis = np.nanstd(fold_means, axis=0, ddof=1) if len(fold_means) > 1 else np.zeros_like(pim)
        # Full-data RF is used only to draw exploratory PDPs, not to score importance.
        rf_full = RandomForestRegressor(
            n_estimators=300, random_state=_GLOBAL_SEED, n_jobs=1
        ).fit(X, y_target)
    except Exception:
        pim, pis, rf_full = None, None, None

    ax = D["ax_imp"]
    ax.cla()
    if pim is not None and pim.size and len(features):
        L = min(len(features), pim.shape[0], pis.shape[0])
        fn = list(features)[:L]
        vals = np.asarray(pim[:L], dtype=float)
        errs = np.asarray(pis[:L], dtype=float)
        order = np.argsort(np.nan_to_num(vals, nan=-np.inf))
        ypos = np.arange(L)
        ax.barh(ypos, vals[order], xerr=errs[order], capsize=3)
        ax.set_title("Held-out RF permutation importance")
        ax.set_xlabel(f"Importance ({scoring_used or 'held-out score decrease'})")
        ax.set_yticks(ypos)
        ax.set_yticklabels(np.asarray(fn, dtype=object)[order])
        ax.grid(True, axis="x", alpha=0.25)
        D["last_importance_df"] = pd.DataFrame({
            "feature": fn,
            "perm_importance": vals,
            "between_fold_std": errs,
            "scoring": scoring_used,
            "validation": "5-fold expanding-window future blocks",
        })
        D["imp_rf"] = pd.DataFrame({"feature": fn, "value": vals})
    else:
        ax.text(0.5, 0.5, "Held-out importance unavailable", ha="center", va="center", transform=ax.transAxes)
        D["imp_rf"] = None
        D["last_importance_df"] = None
    D["can_imp"].draw_idle()

    # Descriptive correlation heatmap + |r| ranking.
    try:
        ax = D["ax_corr"]
        ax.cla()
        cols = [*features, target]
        corr = dfw[cols].corr(method=corr_kind)
        im = ax.imshow(corr.values, aspect="auto", vmin=-1, vmax=1)
        ax.set_title(f"Descriptive correlation heatmap ({corr_kind})")
        ax.set_xticks(range(corr.shape[1]))
        ax.set_yticks(range(corr.shape[0]))
        ax.set_xticklabels(corr.columns, rotation=30, ha="right")
        ax.set_yticklabels(corr.index)
        # Avoid stacking multiple colorbars on repeated runs.
        old_cb = D.get("corr_colorbar")
        if old_cb is not None:
            try:
                old_cb.remove()
            except Exception:
                pass
        D["corr_colorbar"] = D["can_corr"].figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        D["can_corr"].draw_idle()
        D["imp_corr"] = pd.DataFrame({
            "feature": features,
            "value": corr[target].loc[features].abs().values,
        })
    except Exception:
        D["imp_corr"] = None

    # Standardized multivariate linear coefficients.
    try:
        x_scaler = StandardScaler().fit(X)
        y_scaler = StandardScaler().fit(y_target.reshape(-1, 1))
        Xs = x_scaler.transform(X)
        ys = y_scaler.transform(y_target.reshape(-1, 1)).ravel()
        reg = LinearRegression().fit(Xs, ys)
        D["imp_lin"] = pd.DataFrame({"feature": features, "value": np.abs(reg.coef_)})
    except Exception:
        D["imp_lin"] = None

    # PDPs are exploratory fits to all available complete cases.
    for a in D["axs_pdp"]:
        a.cla()
    try:
        from sklearn.inspection import PartialDependenceDisplay
        if rf_full is not None and pim is not None and len(features) > 0:
            order = np.argsort(np.nan_to_num(pim, nan=-np.inf))[::-1][:3]
            for axp, i in zip(D["axs_pdp"], order):
                PartialDependenceDisplay.from_estimator(
                    rf_full, dfw[features], [int(i)], ax=axp, grid_resolution=20
                )
                axp.set_title(f"Exploratory PDP: {features[int(i)]}")
        else:
            for axp in D["axs_pdp"]:
                axp.text(0.5, 0.5, "PDP unavailable", ha="center", va="center", transform=axp.transAxes)
    except Exception:
        for axp in D["axs_pdp"]:
            axp.text(0.5, 0.5, "PDP unavailable", ha="center", va="center", transform=axp.transAxes)
    D["can_pdp"].draw_idle()

    _update_predictor_evidence_table(D, features, target)
    _refresh_driver_compare(D)

def _plot_hysteresis(ax, dfx, target, gate, arrows_every=20):
    ax.cla()
    rising = dfx[dfx["__SIGN__"] > 0]; falling = dfx[dfx["__SIGN__"] < 0]; flat = dfx[dfx["__SIGN__"] == 0]
    if not rising.empty:  ax.scatter(rising[gate], rising[target], s=10, alpha=0.7, label="rising Δgate")
    if not falling.empty: ax.scatter(falling[gate], falling[target], s=10, alpha=0.7, label="falling Δgate")
    if not flat.empty:    ax.scatter(flat[gate], flat[target], s=10, alpha=0.5, label="flat Δgate")

    if arrows_every and int(arrows_every) > 0:
        xs = dfx[gate].values; ys = dfx[target].values
        dx = np.diff(xs); dy = np.diff(ys)
        step = max(1, int(arrows_every))
        idx = np.arange(0, len(dx), step)
        ax.quiver(xs[idx], ys[idx], dx[idx], dy[idx], angles="xy", scale_units="xy", scale=1, width=0.0025)

    ax.set_title(f"Hysteresis: {target} vs {gate}"); ax.set_xlabel(gate); ax.set_ylabel(target)
    ax.legend(loc="best"); ax.grid(True, alpha=0.25)

# =============================================================================
#                          WORKERS (THREAD TARGETS)
# =============================================================================
def _export_df_to_csv(df, title="Save CSV"):
    from tkinter.filedialog import asksaveasfilename
    if df is None or (hasattr(df, "empty") and df.empty):
        messagebox.showwarning("Export", "Nothing to export yet."); return
    path = asksaveasfilename(defaultextension=".csv",
                             filetypes=[("CSV", "*.csv")],
                             title=title,
                             initialfile="meaningflux_export.csv")
    if not path: return
    try:
        df.to_csv(path, index=False)
        messagebox.showinfo("Export", f"Saved:\n{path}")
    except Exception as e:
        messagebox.showerror("Export error", str(e))

def _format_elapsed(seconds):
    """Format elapsed wall time for the GUI progress display."""
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def _start_run_progress(P, status="Preparing data…"):
    """Start the model-run progress indicator on the Tk main thread."""
    P["run_started_at"] = time.time()
    P["run_timer_active"] = True
    P["run_status"].set(status)
    P["run_time"].set("Elapsed 00:00")
    try:
        P["run_progress"].start(12)
    except Exception:
        pass

    widget = P["run_progress"]

    def _tick():
        if not P.get("run_timer_active"):
            return
        elapsed = time.time() - float(P.get("run_started_at") or time.time())
        P["run_time"].set(f"Elapsed {_format_elapsed(elapsed)}")
        widget.after(500, _tick)

    widget.after(500, _tick)


def _set_run_stage(P, text):
    """Update the run-status text without touching Tk from a worker thread."""
    on_main(P["run_progress"], P["run_status"].set, text)


def _finish_run_progress(P, text="Finished"):
    """Stop the progress indicator and report total elapsed time."""
    def _finish():
        P["run_timer_active"] = False
        try:
            P["run_progress"].stop()
            P["run_progress"]["value"] = 100
        except Exception:
            pass
        elapsed = time.time() - float(P.get("run_started_at") or time.time())
        P["run_status"].set(text)
        P["run_time"].set(f"Total {_format_elapsed(elapsed)}")

    on_main(P["run_progress"], _finish)


def _pick_validation_splits(P, n_samples):
    """Return validation folds and the user-facing validation settings."""
    frac = _parse_test_fraction(P["test_fraction"].get(), 0.20)
    strategy = P["split_strategy"].get()
    try:
        n_folds = max(2, int(P.get("cv_folds").get()))
    except Exception:
        n_folds = 5
    initial_fraction = _parse_fraction(
        P["initial_train_fraction"].get(), 0.50,
        min_value=0.20, max_value=0.90,
        label="initial training fraction",
    )
    splits = _make_validation_splits(
        n_samples, strategy, frac, n_folds,
        initial_train_fraction=initial_fraction,
    )
    return splits, strategy, frac, n_folds, initial_fraction


def _run_predict_worker(P, df):
    try:
        _set_run_stage(P, "Checking model, variables, and analysis period…")
        all_cols = _sorted_unique_columns(df)
        sel = [all_cols[i] for i in P["lb_X"].curselection()]
        tgt = P["y_var"].get()
        tscol = P["ts_var"].get()
        method = P["model"].get()
        target_units_value = str(P.get("target_units").get()).strip() if P.get("target_units") is not None else ""
        target_sign_value = str(P.get("target_sign_convention").get()).strip() if P.get("target_sign_convention") is not None else ""
        target_units_value = target_units_value or "not provided"
        target_sign_value = target_sign_value or "not provided"

        if not sel:
            return on_main(P["can_series"].get_tk_widget(), messagebox.showerror,
                           "Error", "Select at least one predictor.")
        if tgt not in df.columns:
            return on_main(P["can_series"].get_tk_widget(), messagebox.showerror,
                           "Error", "Target not found.")
        if tscol not in df.columns:
            return on_main(P["can_series"].get_tk_widget(), messagebox.showerror,
                           "Error", "Timestamp column not found.")

        sel = [c for c in dict.fromkeys(sel) if c not in (tgt, tscol)]
        if not sel:
            raise ValueError("After removing the target and timestamp, no valid predictors remain.")

        # ------------------------------------------------------------------
        # Parse and validate all user-controlled scientific parameters.
        # ------------------------------------------------------------------
        base_seed = int(P["random_seed"].get())
        if base_seed < 0:
            raise ValueError("Random seed must be zero or a positive integer.")

        rs_mode = P["rs"].get()
        rs_label = {0: "Native", 1: "Daily", 2: "Weekly"}.get(rs_mode, "Native")
        diurnal_mode_value = str(P.get("diurnal_mode").get() or "All observations")
        diurnal_active = not diurnal_mode_value.lower().startswith("all")
        daylight_variable_value = str(P.get("daylight_variable").get() or "").strip()
        daylight_threshold_value = float(P.get("daylight_threshold").get() or 10.0)
        if diurnal_active and rs_mode != 1:
            raise ValueError(
                "Daytime/nighttime sensitivity is a daily product. Select Daily resolution."
            )
        if diurnal_active and _target_family(tgt) != "CO2 flux / NEE":
            raise ValueError(
                "Daytime/nighttime sensitivity is intended for FC/NEE targets. "
                "Use All observations for FCH4, FN2O, and other targets."
            )
        period_start = P["analysis_start"].get() if P.get("analysis_start") is not None else None
        period_end = P["analysis_end"].get() if P.get("analysis_end") is not None else None
        use_full_period_value = bool(P["use_full_period"].get()) if P.get("use_full_period") is not None else False
        analysis_period_mode = "full available record" if use_full_period_value else "user-selected period"

        min_coverage = _parse_fraction(
            P["min_coverage_pct"].get(), _MIN_RESAMPLE_COVERAGE,
            min_value=0.10, max_value=1.00,
            label="minimum within-period coverage",
        )
        sum_overrides = _parse_column_overrides(P["sum_override_text"].get(), df.columns)
        mean_overrides = _parse_column_overrides(P["mean_override_text"].get(), df.columns)
        contradictory = sorted(set(sum_overrides) & set(mean_overrides))
        if contradictory:
            raise ValueError(
                "A column cannot be forced to both SUM and MEAN: " + ", ".join(contradictory)
            )

        seq_len_value = max(2, int(P["seq_len"].get() or 20))
        hidden_value = max(2, int(P["hidden"].get() or 96))
        epochs_value = max(1, int(P["epochs"].get() or 100))
        comparison_gate_value = str(P["gate_var"].get() or "").strip()
        gate_value = (
            comparison_gate_value
            if method == "Hysteresis-Gate LSTM (H-LSTM)" else None
        )
        expected_sequence_step = (
            pd.Timedelta(days=1) if rs_mode == 1 else
            pd.Timedelta(days=7) if rs_mode == 2 else None
        )

        linear_fit_intercept_value = bool(P["linear_fit_intercept"].get())

        rf_n_estimators_value = max(10, int(P["rf_n_estimators"].get() or 300))
        rf_min_samples_leaf_value = max(1, int(P["rf_min_samples_leaf"].get() or 1))
        rf_max_depth_text = str(P["rf_max_depth"].get() or "").strip()
        rf_max_depth_value = None if not rf_max_depth_text else max(1, int(rf_max_depth_text))
        rf_max_features_text = str(P["rf_max_features"].get() or "All")
        rf_max_features_value = 1.0 if rf_max_features_text == "All" else rf_max_features_text
        rf_perm_repeats_value = max(1, int(P["rf_perm_repeats"].get() or 10))

        mlp_hidden_value = max(2, int(P["mlp_hidden"].get() or 100))
        mlp_max_iter_value = max(50, int(P["mlp_max_iter"].get() or 1500))
        mlp_alpha_value = float(P["mlp_alpha"].get() or 0.0001)
        mlp_learning_rate_value = float(P["mlp_learning_rate"].get() or 0.001)
        mlp_activation_value = str(P["mlp_activation"].get() or "relu")
        mlp_batch_text = str(P["mlp_batch_size"].get() or "auto").strip().lower()
        mlp_batch_size_value = "auto" if mlp_batch_text == "auto" else max(1, int(mlp_batch_text))
        if mlp_alpha_value < 0 or mlp_learning_rate_value <= 0:
            raise ValueError("MLP alpha must be nonnegative and learning rate must be positive.")

        lstm_batch_size_value = max(1, int(P["lstm_batch_size"].get() or 32))
        lstm_learning_rate_value = float(P["lstm_learning_rate"].get() or 0.001)
        lstm_dropout_value = float(P["lstm_dropout"].get() or 0.0)
        if lstm_learning_rate_value <= 0 or not (0.0 <= lstm_dropout_value < 1.0):
            raise ValueError("LSTM learning rate must be positive and dropout must be between 0 and less than 1.")

        hlstm_learning_rate_value = float(P["hlstm_learning_rate"].get() or 0.001)
        hlstm_dropout_value = float(P["hlstm_dropout"].get() or 0.0)
        hlstm_gradient_clip_value = float(P["hlstm_gradient_clip"].get() or 0.0)
        if hlstm_learning_rate_value <= 0 or not (0.0 <= hlstm_dropout_value < 1.0):
            raise ValueError("H-LSTM learning rate must be positive and dropout must be between 0 and less than 1.")
        if hlstm_gradient_clip_value < 0:
            raise ValueError("H-LSTM gradient clipping must be zero or positive.")

        frac_for_split = _parse_test_fraction(P["test_fraction"].get(), 0.20)
        split_strategy_used = P["split_strategy"].get()
        requested_folds = max(2, int(P["cv_folds"].get() or 5))
        initial_train_fraction_value = _parse_fraction(
            P["initial_train_fraction"].get(), 0.50,
            min_value=0.20, max_value=0.90,
            label="initial training fraction",
        )
        blocked_validation = "Blocked time-series CV" in split_strategy_used
        use_common_sequence_folds_value = (
            blocked_validation and bool(P["use_common_sequence_folds"].get())
        )
        # Figure 3B folds use common gap-safe sequence endpoints.
        # The selected gate must be part of the common predictor list.

        # ------------------------------------------------------------------
        # Build one model-ready table before validation. Scaling remains fold-local.
        # ------------------------------------------------------------------
        _set_run_stage(P, "Building model-ready dataset…")
        fold_gate_value = comparison_gate_value if use_common_sequence_folds_value else gate_value
        if use_common_sequence_folds_value:
            if not fold_gate_value or fold_gate_value not in df.columns:
                raise ValueError(
                    "Select a valid H-LSTM gate before using the common Figure 3B endpoint timeline."
                )
            if fold_gate_value not in sel:
                raise ValueError(
                    "For Figure 3B, the H-LSTM gate must also be selected in the common predictor list."
                )

        needed = list(dict.fromkeys(
            sel + [tgt, tscol] + ([fold_gate_value] if fold_gate_value else []) +
            ([daylight_variable_value] if diurnal_active else [])
        ))
        df_period_full = _filter_analysis_period(df[needed], tscol, period_start, period_end)
        selected_analysis_start_actual = pd.to_datetime(df_period_full[tscol], errors="coerce").min()
        selected_analysis_end_actual = pd.to_datetime(df_period_full[tscol], errors="coerce").max()
        df_period, diurnal_metadata = _apply_diurnal_subset(
            df_period_full, tscol, diurnal_mode_value,
            daylight_variable_value, daylight_threshold_value,
        )

        reporting_frame = _reporting_frame(
            df_period, sel + [tgt], tscol, rs_mode, min_coverage,
            sum_columns=sum_overrides, mean_columns=mean_overrides,
            coverage_relative_to_available_rows=diurnal_active,
        )
        variable_coverage_df = _variable_coverage_summary(
            reporting_frame, sel + [tgt], tgt,
            gate_variable=fold_gate_value,
            sum_columns=sum_overrides, mean_columns=mean_overrides,
        )
        sequence_diagnostics = {}
        common_reference_diagnostics = {}

        # Preserve variable-specific missingness for sequence construction.
        # Predictor windows must be complete, but the target is required only at
        # the prediction endpoint. Nonsequence models use the separate complete
        # endpoint frame below.
        dfw_all, rs_label = _resample_view(
            df_period, rs_mode, tscol,
            min_coverage=min_coverage,
            sum_columns=sum_overrides,
            mean_columns=mean_overrides,
            coverage_relative_to_available_rows=diurnal_active,
            drop_complete_cases=False,
        )
        dfw_complete = (
            dfw_all[sel + [tgt]]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna(how="any")
        )

        common_endpoint_ts = None
        common_h_X = common_h_y = None
        dfw_h_common = None
        if use_common_sequence_folds_value:
            dfw_h_common = _prepare_h_lstm_frame(
                df_period, sel, tgt, fold_gate_value, tscol, rs_mode,
                min_coverage=min_coverage,
                sum_columns=sum_overrides,
                mean_columns=mean_overrides,
                coverage_relative_to_available_rows=diurnal_active,
            )
            common_h_X, common_h_y, common_endpoint_ts, common_reference_diagnostics = _make_seq_data(
                dfw_h_common, sel, tgt, fold_gate_value, seq_len_value,
                return_diagnostics=True, expected_step=expected_sequence_step,
            )
            common_endpoint_ts = pd.Index(common_endpoint_ts)
            minimum_endpoints = max(2 * requested_folds + 2, requested_folds + 3)
            if len(common_endpoint_ts) < minimum_endpoints:
                raise ValueError(
                    f"Only {len(common_endpoint_ts)} common gap-safe sequence endpoints remain. "
                    f"At least {minimum_endpoints} are needed for {requested_folds} expanding folds. "
                    "Shorten the sequence, remove an overlap-limiting predictor, or choose a better-covered period."
                )

        if method == "Hysteresis-Gate LSTM (H-LSTM)":
            if not gate_value or gate_value not in df.columns:
                raise ValueError("Select a valid H-LSTM gate variable.")
            if gate_value not in sel:
                raise ValueError(
                    "For a fair model comparison, the H-LSTM gate variable must also be selected "
                    "in the common predictor list."
                )
            if use_common_sequence_folds_value:
                dfw = dfw_h_common
                X_model = np.asarray(common_h_X, dtype=float)
                y_model = np.asarray(common_h_y, dtype=float)
                model_timestamps = common_endpoint_ts
                sequence_diagnostics = dict(common_reference_diagnostics)
            else:
                dfw = _prepare_h_lstm_frame(
                    df_period, sel, tgt, gate_value, tscol, rs_mode,
                    min_coverage=min_coverage,
                    sum_columns=sum_overrides,
                    mean_columns=mean_overrides,
                    coverage_relative_to_available_rows=diurnal_active,
                )
                X_model, y_model, model_timestamps, sequence_diagnostics = _make_seq_data(
                    dfw, sel, tgt, gate_value, seq_len_value,
                    return_diagnostics=True, expected_step=expected_sequence_step,
                )
            title = f"H-LSTM | {rs_label}"
        elif method == "LSTM (Keras)":
            dfw = dfw_all
            X_lstm, y_lstm, ts_lstm, sequence_diagnostics = _make_lstm_sequence_data(
                dfw, sel, tgt, seq_len_value, return_diagnostics=True,
                expected_step=expected_sequence_step,
            )
            if use_common_sequence_folds_value:
                indexer = pd.Index(ts_lstm).get_indexer(common_endpoint_ts)
                if np.any(indexer < 0):
                    raise ValueError(
                        "The standard LSTM could not align to every common H-LSTM endpoint. "
                        "Clear runs and verify that all models use the same predictors, gate, sequence length, and period."
                    )
                X_model = np.asarray(X_lstm, dtype=float)[indexer]
                y_model = np.asarray(y_lstm, dtype=float)[indexer]
                model_timestamps = common_endpoint_ts
                sequence_diagnostics = dict(sequence_diagnostics)
                sequence_diagnostics["aligned_common_endpoints"] = int(len(common_endpoint_ts))
            else:
                X_model, y_model, model_timestamps = X_lstm, y_lstm, pd.Index(ts_lstm)
            title = f"LSTM | {rs_label}"
        else:
            dfw = dfw_complete
            if use_common_sequence_folds_value:
                endpoint_frame = dfw_all.reindex(common_endpoint_ts)
                if endpoint_frame[sel + [tgt]].isna().any().any():
                    raise ValueError(
                        "A nonsequence model could not align to all common sequence endpoints. "
                        "Verify identical preprocessing and predictor selections."
                    )
                X_model = endpoint_frame[sel].to_numpy(dtype=float)
                y_model = endpoint_frame[tgt].to_numpy(dtype=float)
                model_timestamps = common_endpoint_ts
            else:
                X_model = dfw[sel].to_numpy(dtype=float)
                y_model = dfw[tgt].to_numpy(dtype=float)
                model_timestamps = pd.Index(dfw.index)
            title = f"{_plain_model_name(method)} | {rs_label}"

        if not sequence_diagnostics:
            time_info = _timestamp_gap_diagnostics(model_timestamps)
            sequence_diagnostics = {
                "sequence_policy": "not applicable",
                "expected_timestep": time_info["expected_timestep"],
                "sequence_length": None,
                "n_complete_rows_before_sequences": int(len(dfw_complete)),
                "n_candidate_sequence_windows": None,
                "n_valid_sequences": None,
                "n_rejected_gap_windows": 0,
                "n_source_gaps": int(time_info["n_gaps"]),
                "max_source_interval": time_info["max_interval"],
                "max_source_interval_days": time_info["max_interval_days"],
                "rejected_timestamps": pd.DatetimeIndex([]),
            }

        # Validation is defined on the ordered model-ready endpoint timeline.
        # In Figure 3B mode this is the exact same common gap-safe endpoint index
        # for all five models, so five requested folds cannot disappear during
        # later sequence alignment.
        validation_reference_timestamps = pd.Index(model_timestamps)
        validation_reference_text = (
            f"common gap-safe sequence endpoints (gate={fold_gate_value}, "
            f"sequence={seq_len_value}, n={len(model_timestamps)})"
            if use_common_sequence_folds_value else
            "the current model-ready endpoint timeline"
        )

        validation_splits = _make_timestamp_aligned_validation_splits(
            model_timestamps, validation_reference_timestamps,
            split_strategy_used, frac_for_split, requested_folds,
            initial_train_fraction=initial_train_fraction_value,
        )
        test_frac_used = (
            np.nan if "Blocked time-series CV" in split_strategy_used else frac_for_split
        )
        n_folds_used = len(validation_splits)
        all_test_idx = np.concatenate([te for _, _, te in validation_splits])
        n_train_values = [len(tr) for _, tr, _ in validation_splits]
        resampling_text = _resampling_summary(
            sel + [tgt], min_coverage=min_coverage,
            sum_columns=sum_overrides, mean_columns=mean_overrides,
        )
        split_info = {
            "split_strategy": split_strategy_used,
            "test_fraction": test_frac_used,
            "n_total": int(len(y_model)),
            "n_train": (
                f"{min(n_train_values):,}–{max(n_train_values):,}"
                if n_folds_used > 1 else int(n_train_values[0])
            ),
            "n_test": int(len(all_test_idx)),
            "train_period": _fmt_time_range(model_timestamps[validation_splits[0][1]]),
            "test_period": _fmt_time_range(model_timestamps[all_test_idx]),
            "selected_analysis_period": _fmt_time_range(df_period_full[tscol]),
            "diurnal_mode": diurnal_metadata["diurnal_mode"],
            "complete_case_period": _fmt_time_range(model_timestamps),
            "n_period_rows": int(len(df_period)),
            "n_resampled_rows": int(len(reporting_frame)),
            "n_complete_case_rows": int(len(dfw_complete)),
            "n_model_ready_rows": int(len(y_model)),
            "n_rejected_gap_windows": int(sequence_diagnostics.get("n_rejected_gap_windows", 0) or 0),
            "n_folds": int(n_folds_used),
            "requested_n_folds": int(requested_folds),
            "initial_train_fraction": initial_train_fraction_value,
            "validation_reference": validation_reference_text,
            "comparison_gate_variable": comparison_gate_value,
            "resampling_summary": resampling_text,
        }
        if len(y_model) < max(10, 0.25 * len(df_period)):
            split_info["coverage_warning"] = (
                "Complete model-ready cases are much fewer than selected rows. "
                "Check missing predictors, sequence length, or aggregation."
            )
        split_text = _split_summary_text(split_info)
        if n_folds_used > 1:
            split_text += (
                f"\nFolds: {n_folds_used}/{requested_folds} expanding-window future blocks; "
                f"cutoffs defined from {validation_reference_text}; "
                "pooled metrics use all out-of-fold predictions."
            )
        if diurnal_active:
            split_text += (
                f"\nDiurnal sensitivity: {diurnal_metadata['diurnal_mode']} using "
                f"{daylight_variable_value} threshold {daylight_threshold_value:g}; "
                f"{diurnal_metadata['rows_after_diurnal_filter']:,} subdaily rows retained."
            )
        split_text += "\nResampling: " + resampling_text
        if method in ("LSTM (Keras)", "Hysteresis-Gate LSTM (H-LSTM)"):
            split_text += (
                f"\nSequences: {sequence_diagnostics.get('n_valid_sequences', 0):,} valid; "
                f"target required only at endpoints; "
                f"{sequence_diagnostics.get('n_rejected_gap_windows', 0):,} rejected for temporal gaps; "
                f"{sequence_diagnostics.get('n_rejected_missing_predictor_windows', 0):,} rejected for incomplete predictor windows."
            )
        ensure_on_main(P["can_series"].get_tk_widget(), P["split_msg"].configure,
                       text=split_text)

        _stop_event.clear()
        if method in ("LSTM (Keras)", "Hysteresis-Gate LSTM (H-LSTM)"):
            on_main_sync(
                P["can_series"].get_tk_widget(), P["monitor"].reset,
                epochs_value * n_folds_used, method,
                f"Target={tgt} | X={sel} | seq={seq_len_value} | "
                f"hidden={hidden_value} | folds={n_folds_used}"
            )
        else:
            on_main_sync(
                P["can_series"].get_tk_widget(), P["monitor"].reset,
                1, method, f"Target={tgt} | X={sel} | folds={n_folds_used}"
            )
            on_main_sync(
                P["can_series"].get_tk_widget(), P["monitor"].info_note,
                "This model has no epoch curve. The worker reports fold progress.\n"
            )

        pooled_true, pooled_pred, pooled_time, pooled_fold = [], [], [], []
        pooled_training_mean, pooled_seasonal_climatology = [], []
        fold_records = []
        rf_importance_means, rf_importance_stds = [], []
        importance_scoring = None

        for fold_pos, (fold_id, tr, ts) in enumerate(validation_splits):
            if _stop_event.is_set():
                break
            seed = base_seed + int(fold_id)
            _set_run_stage(P, f"Training {method}: fold {fold_pos+1}/{n_folds_used}…")
            cb = None
            if method in ("LSTM (Keras)", "Hysteresis-Gate LSTM (H-LSTM)"):
                offset = fold_pos * epochs_value
                cb = lambda e, loss, off=offset: P["monitor"].push(off + e, loss)

            fitted_model = None
            if method == "Linear Regression":
                yt, yp, fitted_model, _ = run_linear(
                    X_model, y_model, tr, ts, seed=seed,
                    fit_intercept=linear_fit_intercept_value,
                )
            elif method == "Random Forest":
                yt, yp, fitted_model, _ = run_rf(
                    X_model, y_model, tr, ts, seed=seed,
                    n_estimators=rf_n_estimators_value,
                    min_samples_leaf=rf_min_samples_leaf_value,
                    max_depth=rf_max_depth_value,
                    max_features=rf_max_features_value,
                )
                try:
                    pim = permutation_importance(
                        fitted_model, X_model[ts], y_model[ts],
                        n_repeats=rf_perm_repeats_value,
                        random_state=seed, scoring="r2", n_jobs=1,
                    )
                    importance_scoring = "held-out permutation importance (R2 decrease)"
                except Exception:
                    pim = permutation_importance(
                        fitted_model, X_model[ts], y_model[ts],
                        n_repeats=rf_perm_repeats_value,
                        random_state=seed, scoring="neg_mean_squared_error", n_jobs=1,
                    )
                    importance_scoring = (
                        "held-out permutation importance (negative-MSE decrease)"
                    )
                rf_importance_means.append(np.asarray(pim.importances_mean, dtype=float))
                rf_importance_stds.append(np.asarray(pim.importances_std, dtype=float))
            elif method == "Neural Network (MLP)":
                yt, yp, fitted_model, _ = run_mlp(
                    X_model, y_model, tr, ts,
                    hidden=mlp_hidden_value,
                    max_iter=mlp_max_iter_value,
                    alpha=mlp_alpha_value,
                    learning_rate_init=mlp_learning_rate_value,
                    activation=mlp_activation_value,
                    batch_size=mlp_batch_size_value,
                    seed=seed,
                )
            elif method == "LSTM (Keras)":
                if Sequential is None:
                    raise RuntimeError("TensorFlow/Keras not found. Install to enable LSTM.")
                yt, yp, fitted_model, _ = run_keras_lstm(
                    X_model, y_model,
                    epochs=epochs_value,
                    batch_size=lstm_batch_size_value,
                    hidden=hidden_value,
                    learning_rate=lstm_learning_rate_value,
                    dropout=lstm_dropout_value,
                    gradient_clip=hlstm_gradient_clip_value,
                    train_idx=tr, test_idx=ts,
                    progress_cb=cb, info_cb=P["monitor"].info,
                    seed=seed,
                )
            elif method == "Hysteresis-Gate LSTM (H-LSTM)":
                yt, yp, fitted_model, _ = run_hysteresis_lstm_sequences(
                    X_model, y_model,
                    hidden=hidden_value,
                    epochs=epochs_value,
                    batch_size=lstm_batch_size_value,
                    learning_rate=hlstm_learning_rate_value,
                    dropout=hlstm_dropout_value,
                    gradient_clip=hlstm_gradient_clip_value,
                    train_idx=tr, test_idx=ts,
                    progress_cb=cb, info_cb=P["monitor"].info,
                    seed=seed,
                )
            else:
                raise ValueError(f"Unsupported model: {method}")

            fold_ts = pd.Index(model_timestamps)[ts]
            training_mean_pred = np.full(
                len(ts), float(np.nanmean(np.asarray(y_model, dtype=float)[tr])), dtype=float
            )
            seasonal_pred = _seasonal_climatology_predictions(
                y_model, model_timestamps, tr, ts, window_days=7
            )
            pooled_true.extend(np.asarray(yt, dtype=float).tolist())
            pooled_pred.extend(np.asarray(yp, dtype=float).tolist())
            pooled_training_mean.extend(training_mean_pred.tolist())
            pooled_seasonal_climatology.extend(seasonal_pred.tolist())
            pooled_time.extend(list(fold_ts))
            pooled_fold.extend([int(fold_id)] * len(yt))
            fold_record = _fold_metric_record(
                fold_id, yt, yp, tr, ts, fold_ts,
                all_timestamps=model_timestamps,
                sequence_diagnostics=sequence_diagnostics,
            )
            mean_metrics_fold = _metric_bundle(yt, training_mean_pred)
            seasonal_metrics_fold = _metric_bundle(yt, seasonal_pred)
            fold_record.update({
                "training_mean_rmse": mean_metrics_fold["rmse"],
                "training_mean_nrmse_p5_p95": mean_metrics_fold["nrmse_p5_p95"],
                "training_mean_mae": mean_metrics_fold["mae"],
                "training_mean_r2": mean_metrics_fold["r2"],
                "seasonal_climatology_rmse": seasonal_metrics_fold["rmse"],
                "seasonal_climatology_nrmse_p5_p95": seasonal_metrics_fold["nrmse_p5_p95"],
                "seasonal_climatology_mae": seasonal_metrics_fold["mae"],
                "seasonal_climatology_r2": seasonal_metrics_fold["r2"],
                "skill_vs_training_mean": _baseline_skill_score(yt, yp, training_mean_pred),
                "skill_vs_seasonal_climatology": _baseline_skill_score(yt, yp, seasonal_pred),
                "beats_training_mean": bool(_metric_bundle(yt, yp)["rmse"] < mean_metrics_fold["rmse"]),
                "beats_seasonal_climatology": bool(_metric_bundle(yt, yp)["rmse"] < seasonal_metrics_fold["rmse"]),
            })
            fold_records.append(fold_record)

        if not pooled_true:
            raise RuntimeError("No validation predictions were produced.")

        y_true = np.asarray(pooled_true, dtype=float)
        y_pred = np.asarray(pooled_pred, dtype=float)
        training_mean_pred = np.asarray(pooled_training_mean, dtype=float)
        seasonal_climatology_pred = np.asarray(pooled_seasonal_climatology, dtype=float)
        timestamp = pd.Index(pooled_time)
        fold_ids = np.asarray(pooled_fold, dtype=int)
        fold_metrics_df = pd.DataFrame(fold_records)
        actual_fold_ids = sorted(pd.to_numeric(fold_metrics_df.get("fold"), errors="coerce").dropna().astype(int).unique().tolist())
        if blocked_validation:
            expected_fold_ids = list(range(1, requested_folds + 1))
            if actual_fold_ids != expected_fold_ids:
                raise RuntimeError(
                    f"Blocked CV requested folds {expected_fold_ids}, but predictions were produced for "
                    f"{actual_fold_ids}. The run was not saved."
                )

        imp = imp_std = None
        importance_df = None
        if rf_importance_means:
            fold_imp = np.vstack(rf_importance_means)
            imp = np.nanmean(fold_imp, axis=0)
            imp_std = (
                np.nanstd(fold_imp, axis=0, ddof=1)
                if fold_imp.shape[0] > 1 else rf_importance_stds[0]
            )
            importance_df = pd.DataFrame({
                "feature": sel,
                "importance": imp,
                "importance_std": imp_std,
                "scoring": importance_scoring,
                "evaluation_data": "held-out test folds",
                "permutation_repeats": rf_perm_repeats_value,
            })

        _set_run_stage(P, "Calculating pooled out-of-fold metrics…")
        on_main_sync(
            P["can_series"].get_tk_widget(), _update_predict_plots,
            P, y_true, y_pred, title, tgt, timestamp,
            sel, imp, split_info, fold_ids, imp_std, importance_scoring,
        )
        if importance_df is not None:
            P["last_importance_df"] = importance_df.copy()

        it_bridge_df, pred_col, resid_col = _make_it_bridge_df(
            dfw, sel, tgt, y_true, y_pred, test_idx=None,
            timestamp=timestamp, model_name=method,
        )
        if it_bridge_df is not None and len(it_bridge_df) == len(fold_ids):
            it_bridge_df["fold"] = fold_ids

        metrics = _metric_bundle(y_true, y_pred)
        training_mean_metrics = _metric_bundle(y_true, training_mean_pred)
        seasonal_climatology_metrics = _metric_bundle(y_true, seasonal_climatology_pred)
        skill_vs_training_mean = _baseline_skill_score(
            y_true, y_pred, training_mean_pred
        )
        skill_vs_seasonal_climatology = _baseline_skill_score(
            y_true, y_pred, seasonal_climatology_pred
        )
        folds_beating_training_mean = int(
            pd.Series(fold_metrics_df.get("beats_training_mean", [])).fillna(False).astype(bool).sum()
        )
        folds_beating_seasonal = int(
            pd.Series(fold_metrics_df.get("beats_seasonal_climatology", [])).fillna(False).astype(bool).sum()
        )
        baseline_metrics_df = pd.DataFrame([
            {"benchmark": "Model", **metrics},
            {"benchmark": "Training-fold mean", **training_mean_metrics},
            {"benchmark": "Seasonal climatology ±7 DOY", **seasonal_climatology_metrics},
        ])
        baseline_summary = (
            f"Baseline benchmark on identical held-out dates:\n"
            f"• Training-fold mean: R²={training_mean_metrics['r2']:.3g}, "
            f"nRMSE={training_mean_metrics['nrmse_p5_p95']:.3g}%\n"
            f"• Seasonal climatology (training-only ±7 day-of-year): "
            f"R²={seasonal_climatology_metrics['r2']:.3g}, "
            f"nRMSE={seasonal_climatology_metrics['nrmse_p5_p95']:.3g}%\n"
            f"• Model skill vs mean={skill_vs_training_mean:.3g}; "
            f"vs climatology={skill_vs_seasonal_climatology:.3g}\n"
            f"• Model beats mean in {folds_beating_training_mean}/{n_folds_used} folds; "
            f"climatology in {folds_beating_seasonal}/{n_folds_used} folds."
        )
        ensure_on_main(
            P["can_series"].get_tk_widget(), P["baseline_msg"].configure,
            text=baseline_summary,
        )
        pred_df = P.get("last_pred_df")
        if pred_df is not None:
            pred_df = pred_df.copy()
            pred_df["fold"] = fold_ids
            pred_df["pred_training_mean"] = training_mean_pred
            pred_df["pred_seasonal_climatology"] = seasonal_climatology_pred
            pred_df["model"] = method
            pred_df["target"] = tgt
            P["last_pred_df"] = pred_df

        if method == "Linear Regression":
            model_parameters = {"fit_intercept": linear_fit_intercept_value}
            hidden_size_record = None
            epochs_record = None
            batch_size_record = None
        elif method == "Random Forest":
            model_parameters = {
                "n_estimators": rf_n_estimators_value,
                "min_samples_leaf": rf_min_samples_leaf_value,
                "max_depth": rf_max_depth_value,
                "max_features": rf_max_features_text,
                "permutation_repeats": rf_perm_repeats_value,
                "n_jobs": 1,
            }
            hidden_size_record = None
            epochs_record = None
            batch_size_record = None
        elif method == "Neural Network (MLP)":
            model_parameters = {
                "hidden_layer_sizes": [mlp_hidden_value],
                "max_iter": mlp_max_iter_value,
                "alpha": mlp_alpha_value,
                "learning_rate_init": mlp_learning_rate_value,
                "activation": mlp_activation_value,
                "batch_size": mlp_batch_size_value,
                "solver": "adam",
                "early_stopping": False,
            }
            hidden_size_record = mlp_hidden_value
            epochs_record = None
            batch_size_record = None
        elif method == "LSTM (Keras)":
            model_parameters = {
                "sequence_length": seq_len_value,
                "hidden_size": hidden_value,
                "epochs": epochs_value,
                "batch_size": lstm_batch_size_value,
                "learning_rate": lstm_learning_rate_value,
                "dropout": lstm_dropout_value,
                "gradient_clip": hlstm_gradient_clip_value,
                "framework": "TensorFlow/Keras",
                "shuffle": False,
            }
            hidden_size_record = hidden_value
            epochs_record = epochs_value
            batch_size_record = lstm_batch_size_value
        else:
            model_parameters = {
                "sequence_length": seq_len_value,
                "hidden_size": hidden_value,
                "epochs": epochs_value,
                "batch_size": lstm_batch_size_value,
                "learning_rate": hlstm_learning_rate_value,
                "dropout": hlstm_dropout_value,
                "gradient_clip": hlstm_gradient_clip_value,
                "gate_variable": gate_value,
                "includes_delta_gate": True,
                "framework": "TensorFlow/Keras",
                "controlled_difference": "same LSTM training as standard LSTM; adds delta-gate input channel only",
                "shuffle": False,
            }
            hidden_size_record = hidden_value
            epochs_record = epochs_value
            batch_size_record = lstm_batch_size_value

        sequence_diagnostics_export = {
            k: v for k, v in sequence_diagnostics.items()
            if not str(k).endswith("_timestamps") and k != "rejected_timestamps"
        }
        common_reference_diagnostics_export = {
            k: v for k, v in common_reference_diagnostics.items()
            if not str(k).endswith("_timestamps") and k != "rejected_timestamps"
        }
        data_counts = {
            "input_rows": int(len(df)),
            "selected_period_rows": int(len(df_period_full)),
            "diurnal_subset_rows": int(len(df_period)),
            "resampled_period_rows": int(len(reporting_frame)),
            "complete_case_rows": int(len(dfw_complete)),
            "model_ready_rows": int(len(y_model)),
            "held_out_predictions": int(len(y_true)),
        }

        run_rec = dict(
            when=pd.Timestamp.utcnow(), model=method, resample=rs_label,
            target=tgt, target_units=target_units_value,
            target_sign_convention=target_sign_value, features=sel,
            rmse=metrics["rmse"], mae=metrics["mae"], r2=metrics["r2"],
            nrmse_p5_p95=metrics["nrmse_p5_p95"],
            observed_p05=metrics["observed_p05"],
            observed_p95=metrics["observed_p95"],
            pred_df=P.get("last_pred_df"),
            importance_df=(importance_df.copy() if importance_df is not None else None),
            fold_metrics_df=fold_metrics_df,
            baseline_metrics_df=baseline_metrics_df,
            training_mean_metrics=training_mean_metrics,
            seasonal_climatology_metrics=seasonal_climatology_metrics,
            skill_vs_training_mean=skill_vs_training_mean,
            skill_vs_seasonal_climatology=skill_vs_seasonal_climatology,
            folds_beating_training_mean=folds_beating_training_mean,
            folds_beating_seasonal_climatology=folds_beating_seasonal,
            it_bridge_df=it_bridge_df,
            prediction_column=pred_col,
            residual_column=resid_col,
            split_strategy=split_strategy_used,
            test_fraction=test_frac_used,
            n_folds=n_folds_used,
            requested_n_folds=requested_folds,
            validation_reference=validation_reference_text,
            comparison_gate_variable=comparison_gate_value,
            use_common_sequence_folds=use_common_sequence_folds_value,
            common_reference_diagnostics=common_reference_diagnostics_export,
            n_train=(min(n_train_values) if n_folds_used == 1 else max(n_train_values)),
            n_train_min=min(n_train_values),
            n_train_max=max(n_train_values),
            n_test=len(y_true),
            split_info=split_info,
            analysis_period_mode=analysis_period_mode,
            analysis_start=period_start,
            analysis_end=period_end,
            diurnal_mode=diurnal_metadata["diurnal_mode"],
            daylight_variable=diurnal_metadata["daylight_variable"],
            daylight_threshold=diurnal_metadata["daylight_threshold"],
            selected_analysis_start=selected_analysis_start_actual,
            selected_analysis_end=selected_analysis_end_actual,
            model_ready_start=pd.to_datetime(pd.Index(model_timestamps), errors="coerce").min(),
            model_ready_end=pd.to_datetime(pd.Index(model_timestamps), errors="coerce").max(),
            sequence_length=(
                seq_len_value
                if method in ("LSTM (Keras)", "Hysteresis-Gate LSTM (H-LSTM)")
                else None
            ),
            hidden_size=hidden_size_record,
            epochs=epochs_record,
            batch_size=batch_size_record,
            gate_variable=gate_value,
            random_seed=base_seed,
            scaling=(
                "StandardScaler fitted separately within each training fold; target inverse-transformed"
                if method in ("Neural Network (MLP)", "LSTM (Keras)",
                              "Hysteresis-Gate LSTM (H-LSTM)")
                else "none"
            ),
            min_resample_coverage=min_coverage,
            sum_override_columns=sum_overrides,
            mean_override_columns=mean_overrides,
            resampling_summary=resampling_text,
            importance_scoring=importance_scoring,
            missing_data_handling=("no imputation; sequence predictors complete at every step; "
                                  "target required only at prediction endpoint; nonsequence models "
                                  "use complete predictor+target endpoint rows"),
            outlier_handling="no automatic statistical outlier removal; upstream QA/QC and user masks only",
            diurnal_cycle_handling=(
                (
                    f"{diurnal_metadata['diurnal_mode']} daily mean using "
                    f"{diurnal_metadata['daylight_variable']} threshold "
                    f"{diurnal_metadata['daylight_threshold']:g}; optional FC sensitivity"
                )
                if diurnal_active else
                "daily aggregation; diurnal cycle not modeled explicitly"
                if rs_label == "Daily" else
                "weekly aggregation; diurnal cycle not modeled explicitly"
                if rs_label == "Weekly" else
                "native resolution; diurnal cycle preserved unless represented by selected predictors"
            ),
            data_counts=data_counts,
            variable_coverage_df=variable_coverage_df.copy(),
            sequence_diagnostics=sequence_diagnostics_export,
            sequence_policy=sequence_diagnostics_export.get("sequence_policy"),
            expected_timestep=sequence_diagnostics_export.get("expected_timestep"),
            n_candidate_sequence_windows=sequence_diagnostics_export.get("n_candidate_sequence_windows"),
            n_valid_sequences=sequence_diagnostics_export.get("n_valid_sequences"),
            n_rejected_gap_windows=sequence_diagnostics_export.get("n_rejected_gap_windows", 0),
            n_rejected_missing_endpoint_target=sequence_diagnostics_export.get(
                "n_rejected_missing_endpoint_target", 0
            ),
            n_rejected_missing_predictor_windows=sequence_diagnostics_export.get(
                "n_rejected_missing_predictor_windows", 0
            ),
            target_requirement=sequence_diagnostics_export.get("target_requirement"),
            predictor_requirement=sequence_diagnostics_export.get("predictor_requirement"),
            n_source_gaps=sequence_diagnostics_export.get("n_source_gaps", 0),
            max_source_interval=sequence_diagnostics_export.get("max_source_interval"),
            max_source_interval_days=sequence_diagnostics_export.get("max_source_interval_days"),
            model_parameters=model_parameters,
            initial_train_fraction=(
                initial_train_fraction_value
                if "Blocked time-series CV" in split_strategy_used else None
            ),
        )
        P["runs"].append(run_rec)
        P["current_it_bridge"] = (
            it_bridge_df.copy() if it_bridge_df is not None else None
        )
        P["current_it_metadata"] = pd.DataFrame([{
            "model": method,
            "target": tgt,
            "target_units": target_units_value,
            "target_sign_convention": target_sign_value,
            "observed_column": f"{tgt}_obs",
            "prediction_column": pred_col,
            "residual_column": resid_col,
            "features": ", ".join(sel),
            "resample": rs_label,
            "resampling_summary": resampling_text,
            "min_resample_coverage": min_coverage,
            "sum_override_columns": ", ".join(sum_overrides),
            "mean_override_columns": ", ".join(mean_overrides),
            "split_strategy": split_strategy_used,
            "n_folds": n_folds_used,
            "requested_n_folds": requested_folds,
            "validation_reference": validation_reference_text,
            "comparison_gate_variable": comparison_gate_value,
            "use_common_sequence_folds": use_common_sequence_folds_value,
            "initial_train_fraction": run_rec["initial_train_fraction"],
            "test_fraction": test_frac_used,
            "n_train_min": min(n_train_values),
            "n_train_max": max(n_train_values),
            "n_test": len(y_true),
            "data_counts": str(data_counts),
            "sequence_diagnostics": str(sequence_diagnostics_export),
            "target_requirement": sequence_diagnostics_export.get("target_requirement"),
            "predictor_requirement": sequence_diagnostics_export.get("predictor_requirement"),
            "n_rejected_missing_endpoint_target": sequence_diagnostics_export.get(
                "n_rejected_missing_endpoint_target", 0
            ),
            "n_rejected_missing_predictor_windows": sequence_diagnostics_export.get(
                "n_rejected_missing_predictor_windows", 0
            ),
            "analysis_start": period_start,
            "analysis_end": period_end,
            "diurnal_mode": diurnal_metadata["diurnal_mode"],
            "daylight_variable": diurnal_metadata["daylight_variable"],
            "daylight_threshold": diurnal_metadata["daylight_threshold"],
            "test_period": split_info.get("test_period"),
            "rmse": metrics["rmse"],
            "nrmse_p5_p95": metrics["nrmse_p5_p95"],
            "mae": metrics["mae"],
            "r2": metrics["r2"],
            "skill_vs_training_mean": skill_vs_training_mean,
            "skill_vs_seasonal_climatology": skill_vs_seasonal_climatology,
            "folds_beating_training_mean": folds_beating_training_mean,
            "folds_beating_seasonal_climatology": folds_beating_seasonal,
            "sequence_length": run_rec["sequence_length"],
            "hidden_size": run_rec["hidden_size"],
            "epochs": run_rec["epochs"],
            "batch_size": run_rec["batch_size"],
            "gate_variable": gate_value,
            "random_seed": base_seed,
            "scaling": run_rec["scaling"],
            "missing_data_handling": run_rec["missing_data_handling"],
            "outlier_handling": run_rec["outlier_handling"],
            "diurnal_cycle_handling": run_rec["diurnal_cycle_handling"],
            "model_parameters": str(model_parameters),
        }])
        on_main(P["btn_open_it"], P["btn_open_it"].configure, state="normal")
        on_main(P["can_series"].get_tk_widget(), P["compare_refresh"])
        _finish_run_progress(P, "Finished — leakage-free held-out results ready")

    except Exception as e:
        on_main(P["can_series"].get_tk_widget(), P["monitor"].error, str(e))
        _finish_run_progress(P, "Stopped with error")
        on_main(P["can_series"].get_tk_widget(), messagebox.showerror,
                "Error", str(e))
    finally:
        on_main(P["btn_run"], P["btn_run"].configure, state="normal")
        on_main(P["btn_stop"], P["btn_stop"].configure, state="disabled")


# =============================================================================
#                    HYSTERESIS EXPLORER TAB: BUILD + PLOTTING
# =============================================================================
def _build_hysteresis_tab(parent_notebook, df, inputname_site):
    tab = ttk.Frame(parent_notebook)
    parent_notebook.add(tab, text="2. Hysteresis / Gate Explorer")

    splitter = ttk.Panedwindow(tab, orient="horizontal")
    splitter.pack(fill="both", expand=True)

    controls = _scrollable_controls(splitter, width=405, padding=(10, 10))

    results = ttk.Notebook(splitter)
    splitter.add(results, weight=1)

    Label(controls, text="Hysteresis / Gate Explorer", font=("TkDefaultFont", 11, "bold")).grid(sticky="w", pady=(0,6))
    _concept_question(
        controls,
        "Does the target flux respond differently when a driver is increasing versus decreasing?",
        "Inspect whether rising and falling driver states produce different flux responses.",
        width=360,
    )

    # Unique, alphabetized column names for easier search
    cols = _sorted_unique_columns(df)

    default_ts = _guess_timestamp_column(cols)
    default_target = _guess_default_target(cols, timestamp_col=default_ts)

    box_select = ttk.LabelFrame(controls, text="Step 1: select target and candidate gate driver")
    box_select.grid(sticky="ew", pady=6)
    box_select.columnconfigure(0, weight=1)

    ttk.Label(box_select, text="Target flux / variable (y)").grid(sticky="w", padx=6, pady=(6,0))
    yH = StringVar(box_select, value=default_target)
    ttk.OptionMenu(box_select, yH, yH.get(), *cols).grid(sticky="ew", padx=6, pady=(0,6))

    ttk.Label(box_select, text="Timestamp column").grid(sticky="w", padx=6)
    tsH = StringVar(box_select, value=default_ts)
    ttk.OptionMenu(box_select, tsH, tsH.get(), *cols).grid(sticky="ew", padx=6, pady=(0,6))

    box_period_H = ttk.LabelFrame(controls, text="Analysis period for hysteresis")
    box_period_H.grid(sticky="ew", pady=6)
    box_period_H.columnconfigure(1, weight=1)
    h_start, h_end, h_full, h_period_msg = _add_analysis_period_controls(
        box_period_H, df, tsH, row_start=0, width=360
    )

    ttk.Label(box_select, text="Candidate gate driver").grid(sticky="w", padx=6)
    gate_sel = StringVar(box_select, value=cols[0])
    ttk.OptionMenu(box_select, gate_sel, gate_sel.get(), *cols).grid(sticky="ew", padx=6, pady=(0,6))

    box_time = ttk.LabelFrame(controls, text="Step 2: define temporal view and smoothing")
    box_time.grid(sticky="ew", pady=6)
    box_time.columnconfigure(4, weight=1)
    ttk.Label(box_time, text="Time resolution").grid(row=0, column=0, sticky="w", padx=6, pady=(6,4))
    rsH = IntVar(box_time, value=0)
    ttk.Radiobutton(box_time, text="Native", variable=rsH, value=0).grid(row=0, column=1, sticky="w")
    ttk.Radiobutton(box_time, text="Daily", variable=rsH, value=1).grid(row=0, column=2, sticky="w")
    ttk.Radiobutton(box_time, text="Weekly", variable=rsH, value=2).grid(row=0, column=3, sticky="w")

    row_h = ttk.Frame(box_time)
    row_h.grid(row=2, column=0, columnspan=5, sticky="ew", padx=6, pady=(3,6))
    ttk.Label(row_h, text="Smoothing").grid(row=0, column=0, sticky="w")
    smooth_var = StringVar(row_h, value="3")
    ttk.Entry(row_h, textvariable=smooth_var, width=6).grid(row=0, column=1, sticky="w", padx=(6,12))
    ttk.Label(row_h, text="Arrows every").grid(row=0, column=2, sticky="w")
    arrows_var = StringVar(row_h, value="20")
    ttk.Entry(row_h, textvariable=arrows_var, width=6).grid(row=0, column=3, sticky="w", padx=(6,0))

    box_run = ttk.LabelFrame(controls, text="Step 3: inspect path dependence")
    box_run.grid(sticky="ew", pady=6)
    tk.Message(
        box_run,
        width=360,
        fg="#444",
        text=(
            "Points are colored by sign of Δgate and arrows show temporal evolution. "
            "Loops or separated branches suggest path dependence."
        ),
    ).grid(row=0, column=0, sticky="ew", padx=6, pady=(6,3))
    ttk.Button(
        box_run,
        text="Run Hysteresis / Gate Explorer",
        command=lambda: threading.Thread(
            target=_run_hysteresis_worker,
            args=(df, yH, gate_sel, tsH, rsH, smooth_var, arrows_var, results),
            daemon=True,
        ).start(),
    ).grid(row=1, column=0, sticky="ew", padx=6, pady=(3,6))

    tab_guide = ttk.Frame(results); results.add(tab_guide, text="Guide")
    tk.Message(
        tab_guide,
        width=760,
        fg="#444",
        text=(
            "Hysteresis / Gate Explorer is a follow-up diagnostic for candidate drivers identified through predictor screening or prior knowledge. "
            "It is not a driver-ranking method. Instead, it tests whether one selected driver produces different flux responses during rising versus falling conditions. "
            "If the hysteresis plot shows loops or branch separation, that driver may be a meaningful H-LSTM gate variable in the Predictive Modeling tab."
        ),
    ).pack(fill="x", padx=10, pady=10)

    tab_hyst = ttk.Frame(results); results.add(tab_hyst, text="Gate-response plot")
    fig_hyst, ax_hyst = plt.subplots(figsize=(8.8, 4.4))
    can_hyst = FigureCanvasTkAgg(fig_hyst, master=tab_hyst)
    can_hyst.draw()
    can_hyst.get_tk_widget().pack(fill="both", expand=True)

    handles = dict(
        yD=yH, tsD=tsH, rsD=rsH,
        gate_sel=gate_sel, smooth_var=smooth_var, arrows_var=arrows_var,
        ax_hyst=ax_hyst, can_hyst=can_hyst,
        results=results,
        analysis_start=h_start, analysis_end=h_end, use_full_period=h_full,
    )
    results._handles = handles
    return tab, handles

def _run_drivers_worker(df, lb_D, yD, tsD, rsD, results_notebook):
    try:
        all_cols = _sorted_unique_columns(df)
        sel = [all_cols[i] for i in lb_D.curselection()]
        tgt = yD.get(); tscol = tsD.get()
        if not sel: return on_main(results_notebook, messagebox.showerror, "Error", "Select at least one candidate driver.")
        if tgt not in df.columns: return on_main(results_notebook, messagebox.showerror, "Error", "Target not found.")
        if tscol not in df.columns: return on_main(results_notebook, messagebox.showerror, "Error", "Timestamp column not found.")

        D = getattr(results_notebook, "_handles", None)
        if D is None: return

        df_period = _filter_analysis_period(
            df[[*sel, tgt, tscol]], tscol,
            D["analysis_start"].get() if D.get("analysis_start") is not None else None,
            D["analysis_end"].get() if D.get("analysis_end") is not None else None,
        )
        dfw, _ = _resample_view(df_period, rsD.get(), tscol)
        if tscol in dfw.columns: dfw = dfw.drop(columns=[tscol])

        ensure_on_main(D["can_imp"].get_tk_widget(), _update_driver_plots, D, dfw, sel, tgt, "pearson")
    except Exception as e:
        on_main(results_notebook, messagebox.showerror, "Error", str(e))

def _run_hysteresis_worker(df, yD, gate_sel, tsD, rsD, smooth_var, arrows_var, results_notebook):
    try:
        tgt = yD.get(); gate = gate_sel.get(); tscol = tsD.get()
        D = getattr(results_notebook, "_handles", None)
        if D is None: return
        if tgt not in df.columns:  return on_main(results_notebook, messagebox.showerror, "Error", "Target not found.")
        if gate not in df.columns: return on_main(results_notebook, messagebox.showerror, "Error", "Gate not found.")
        if tscol not in df.columns: return on_main(results_notebook, messagebox.showerror, "Error", "Timestamp column not found.")
        df_period = _filter_analysis_period(
            df[[tgt, gate, tscol]], tscol,
            D["analysis_start"].get() if D.get("analysis_start") is not None else None,
            D["analysis_end"].get() if D.get("analysis_end") is not None else None,
        )
        dfx = _make_hysteresis_view(df_period, tgt, gate, tscol, rsD.get(), smooth=int(smooth_var.get()))
        def _draw():
            _plot_hysteresis(D["ax_hyst"], dfx, tgt, gate, arrows_every=int(arrows_var.get()))
            D["can_hyst"].draw_idle()
        ensure_on_main(D["can_hyst"].get_tk_widget(), _draw)
    except Exception as e:
        on_main(results_notebook, messagebox.showerror, "Error", str(e))

# =============================================================================
#                              MAIN WINDOW
# =============================================================================
_ml_window = None

def open_meaningflux_ml(df: pd.DataFrame, inputname_site: str):
    """Open compact MeaningFlux ML window with Predictive Modeling, Hysteresis/Gate Explorer, and Optional Predictor Screening tabs."""
    global _ml_window
    if _ml_window is not None and tk.Toplevel.winfo_exists(_ml_window):
        messagebox.showinfo("Info", "MeaningFlux ML window already open.")
        return

    _ml_window = tk.Toplevel()
    _ml_window.title(f"MeaningFlux — Machine Learning · {inputname_site}")
    _ml_window.geometry("1240x740")
    _ml_window.minsize(1100, 680)
    set_simple_theme(_ml_window)

    nb = ttk.Notebook(_ml_window); nb.pack(fill="both", expand=True)

    # Recommended workflow: predictive modeling first, then optional hysteresis and screening diagnostics.
    tab_predict, P = _build_predict_tab(nb, df, inputname_site)
    tab_hyst, H = _build_hysteresis_tab(nb, df, inputname_site)
    tab_drivers, D = _build_drivers_tab(nb, df, inputname_site)
    D["results"]._handles = D  # store handles for async updates
    H["results"]._handles = H  # store handles for async updates

    def _run():
        _stop_event.clear()
        P["btn_run"].configure(state="disabled")
        P["btn_stop"].configure(state="normal")
        _start_run_progress(P)
        threading.Thread(target=_run_predict_worker, args=(P, df), daemon=True).start()

    def _stop():
        _stop_event.set()
        P["monitor"].stopped()
        _finish_run_progress(P, "Stop requested")
        P["btn_stop"].configure(state="disabled")
        P["btn_run"].configure(state="normal")

    def _export():
        current_tab = P["results_notebook"].tab(P["results_notebook"].select(), "text")
        if current_tab in ("Series", "Scatter", "Residuals", "Metrics", "Training", "Feature Importance", "Compare"):
            if current_tab == "Feature Importance":
                _export_df_to_csv(P.get("last_importance_df"), "Save feature importance")
            elif current_tab == "Compare":
                if not P["runs"]:
                    messagebox.showwarning("Export", "No runs to export yet.")
                else:
                    df_runs = pd.DataFrame([{
                        "when": str(r["when"]).split(".")[0],
                        "model": r["model"],
                        "resample": r["resample"],
                        "target": r["target"],
                        "features": ", ".join(r["features"]),
                        "split_strategy": r.get("split_strategy", ""),
                        "test_fraction": r.get("test_fraction", np.nan),
                        "n_folds": r.get("n_folds", 1),
                        "n_train": r.get("n_train", np.nan),
                        "n_train_min": r.get("n_train_min", np.nan),
                        "n_train_max": r.get("n_train_max", np.nan),
                        "n_test": r.get("n_test", np.nan),
                        "analysis_start": r.get("analysis_start", ""),
                        "analysis_end": r.get("analysis_end", ""),
                        "train_period": (r.get("split_info") or {}).get("train_period", ""),
                        "test_period": (r.get("split_info") or {}).get("test_period", ""),
                        "rmse": r["rmse"], "nrmse_p5_p95": r.get("nrmse_p5_p95", np.nan),
                        "mae": r["mae"], "r2": r["r2"],
                        "sequence_length": r.get("sequence_length"), "hidden_size": r.get("hidden_size"),
                        "epochs": r.get("epochs"), "gate_variable": r.get("gate_variable"),
                        "random_seed": r.get("random_seed"), "scaling": r.get("scaling"),
                    } for r in P["runs"]])
                    _export_df_to_csv(df_runs, "Save runs summary")
            else:
                _export_df_to_csv(P.get("last_pred_df"), "Save predictions")
        else:
            _export_df_to_csv(D.get("last_importance_df"), "Save driver importance")

    P["btn_run"].configure(command=_run)
    P["btn_stop"].configure(command=_stop)
    P["btn_export"].configure(command=_export)

    def on_close():
        global _ml_window
        try:
            _stop_event.set()
            try:
                P["monitor"].close()
            except Exception:
                pass
            _ml_window.destroy()
        finally:
            _ml_window = None
    _ml_window.protocol("WM_DELETE_WINDOW", on_close)

def open_machine_learning_toolbox(df: pd.DataFrame, inputname_site: str):
    """Backwards-compatible wrapper."""
    return open_meaningflux_ml(df, inputname_site)


