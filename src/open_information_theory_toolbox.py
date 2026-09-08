#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MeaningFlux — Information Theory Toolbox (timestamp-preserving temporal update)
Author: Leila C. Hernandez (LBNL) 

Key points:
- Uses the same DataFrame philosophy as the ML toolbox:
  * We trust the incoming df (already cleaned upstream).
  * We do NOT pre-filter columns as "numeric only".
  * When a variable is selected, we convert that column to numeric on-the-fly
    with pd.to_numeric(errors='coerce') and drop NaNs.

Features:
- Entropy H(X)
- Mutual Information I(X;Y) (histogram or KDE-based TIP)
- MI Driver Ranking: ranks multiple drivers by I(driver; target), with optional permutation testing
- Normalized MI Ranking: reports I(driver; target) / H(target) for comparison across variables
- Correlation vs MI Ranking: compares linear association (Pearson r²) with MI
- Single-Site IT Summary: compact multi-panel diagnostic dashboard for one site/target
- Conditional Mutual Information I(X;Y|Z)
- Lagged Mutual Information I(X_t;Y_{t+lag}), including normalized MI and peak-lag summaries
- PID (two sources -> one target):
    * histogram-based min-information PID
    * KDE-based TIP decomposition (Goodwell & Kumar style)
- Pairwise PID Matrix: maps redundancy, total unique information, and synergy
  for all selected driver pairs with respect to one target
- Model-vs-Observed MI: compares normalized MI for observed and modeled targets
- Model-vs-Observed PID Matrix: maps model-minus-observed synergy, redundancy, and unique-information differences
- Functional Performance Summary: summarizes individual and pairwise information-fidelity diagnostics
- Transfer Entropy TE(X->Y) (simple discrete, first-order)
- Transfer Entropy vs Lag: evaluates raw and normalized TE(X->Y) over a positive lag window
- TE Network (matrix + heatmap, optional permutation test)
- Notebook-style output tabs: Method Guide, Plot, Results Table, Compare Runs, and Export
- Analysis period and window filters: explicit start/end period plus all-data, daytime, nighttime, growing-season, selected-month, or custom filters
- Model-output detection: helps users choose predicted/model columns for observed-vs-modeled diagnostics
- Surrogate significance options: random shuffle, block shuffle, and circular shift
- Maximum-statistic surrogate tests that account for searching across multiple lags
- Timestamp-preserving temporal analysis: missing calendar steps remain explicit before lagging
- Figure 6 Temporal Summary: Native/Daily/Weekly preparation, variable-specific aggregation, full lagged-MI and TE profiles, supported networks, and target-memory-conditioned temporal PID
- Row-wise numeric alignment to avoid timestamp mismatch when variables have different missing values
- Reproducibility metadata export for manuscript/SI reporting
- ML-to-IT bridge import: reads aligned prediction tables exported from the ML toolbox

IMPORTANT: The public entry point is flexible:
  - Pattern A (MeaningFlux main app):
        open_information_theory_toolbox(df, inputname_site)
  - Pattern B (standalone demo / other GUIs):
        open_information_theory_toolbox(parent_widget, df)
"""

from __future__ import annotations

from typing import Iterable, List, Dict, Tuple, Optional, Union
import math
import json
import re
import os
from datetime import datetime
from pathlib import Path
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import pandas as pd

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


# =============================================================================
# Generic helpers
# =============================================================================

ArrayLike = Union[np.ndarray, Iterable[float]]

_it_window: Optional[tk.Toplevel] = None  # single window like ML toolbox


def _looks_like_meaningflux_project(path: Path) -> bool:
    """Return True when *path* looks like the MeaningFlux project root."""
    try:
        path = path.expanduser().resolve()
    except Exception:
        return False
    has_code = (path / "code").is_dir()
    has_outputs = (path / "results").is_dir() or (path / "figures").is_dir()
    has_manuscript_code = (path / "code" / "manuscript_figures_code").is_dir()
    name_hint = "meaningflux" in path.name.lower()
    return bool(has_manuscript_code or (has_code and has_outputs) or (name_hint and has_code))


def _infer_meaningflux_project_root() -> Path:
    """Infer the nearest project root from this file, the working directory, or an environment variable."""
    env_value = os.environ.get("MEANINGFLUX_PROJECT_DIR", "").strip()
    if env_value:
        env_path = Path(env_value).expanduser()
        if env_path.exists():
            return env_path.resolve()

    starts = [Path(__file__).resolve().parent, Path.cwd().resolve()]
    seen: set[Path] = set()
    for start in starts:
        for candidate in (start, *start.parents):
            if candidate in seen:
                continue
            seen.add(candidate)
            if _looks_like_meaningflux_project(candidate):
                return candidate

    # A practical fallback: start the chooser near the code rather than at the filesystem root.
    return Path(__file__).resolve().parent


def _normalize_selected_project_root(path_value: str | os.PathLike[str]) -> Path:
    """Accept a project root or a nested results folder and return the project root."""
    path = Path(path_value).expanduser().resolve()
    lowered = [part.lower() for part in path.parts]
    if len(lowered) >= 2 and lowered[-2:] == ["results", "information_theory"]:
        return path.parents[1]
    if lowered and lowered[-1] == "results":
        return path.parent
    return path


# =============================================================================
# User-facing method guide text
# =============================================================================

METHOD_GUIDES: Dict[str, Dict[str, str]] = {
    "Entropy H(X)": {
        "what": "Quantifies the uncertainty or variability of one selected variable. Entropy is reported in bits.",
        "needs": "Select X. Choose the number of bins and discretizer if using the histogram estimator.",
        "interpret": "Higher H means the variable spans more states or is less predictable. Very low H often means the variable is nearly constant or has many repeated values. Entropy alone does not identify a driver; it describes the information content of one variable.",
    },
    "Mutual Information I(X;Y)": {
        "what": "Measures how much knowing X reduces uncertainty in Y. It captures nonlinear dependence, not only linear correlation.",
        "needs": "Select X as the driver/source and Y as the target flux or response variable.",
        "interpret": "Larger MI means stronger statistical dependence between X and Y. MI is symmetric, so I(X;Y)=I(Y;X). It does not prove causality or direction. A scatter plot is shown only as a visual aid; the MI value is the diagnostic.",
    },
    "MI Driver Ranking": {
        "what": "Ranks a pool of drivers by their mutual information with one target variable.",
        "needs": "Select Y as the target. Select drivers in the variable pool list. If no drivers are selected, the toolbox tries all available columns except the target.",
        "interpret": "Drivers with larger MI or normalized MI share more information with the target. Normalized MI = I(driver;target)/H(target), so it is easier to compare across targets or sites. This is a dependence ranking, not a causal ranking.",
    },
    "Correlation vs MI Ranking": {
        "what": "Compares linear association with nonlinear dependence for the same driver pool.",
        "needs": "Select Y as the target. Select drivers in the variable pool list.",
        "interpret": "Pearson r² highlights linear relationships. Normalized MI can remain high when a relationship is nonlinear, threshold-like, or asymmetric. Drivers with high MI but low r² are candidates for nonlinear controls that simple correlation may miss.",
    },
    "Model-vs-Observed MI": {
        "what": "Compares whether a modeled or predicted target reproduces observed driver-target dependencies at the individual-driver level.",
        "needs": "Select Y as the observed target, such as FC. Select Z as the modeled/predicted target, such as FC_pred_RF. Select drivers in the variable pool list.",
        "interpret": "The toolbox reports Delta I_n = I_n(model) - I_n(observed). Values near 0 mean the model reproduces the observed driver-target dependence. Positive values mean the model overestimates that dependence. Negative values mean the model underestimates it.",
    },
    "Model-vs-Observed PID Matrix": {
        "what": "Compares observed and modeled pairwise information structure: synergy, redundancy, and unique information.",
        "needs": "Select Y as the observed target. Select Z as the modeled/predicted target. Select at least two drivers in the variable pool list.",
        "interpret": "The heatmaps show model-minus-observed fractions for synergy, redundancy, and unique information. Positive values mean the model overestimates that information type. Negative values mean the model underestimates it. Values near 0 mean better functional agreement.",
    },
    "Functional Performance Summary": {
        "what": "Summarizes model functional performance using individual-driver MI and pairwise PID diagnostics.",
        "needs": "Select Y as the observed target, Z as the modeled/predicted target, and a driver pool.",
        "interpret": "Higher scores mean the modeled target better reproduces the observed information relationships. This complements predictive metrics such as R² or RMSE: a model can predict well but still reproduce the driver relationships poorly.",
    },
    "Single-Site IT Summary": {
        "what": "Creates a compact multi-panel information-theory dashboard for one site and target.",
        "needs": "Select Y as the target. Select a driver pool. X is used as the source for lagged/TE diagnostics. Z can be used as a second driver for PID, or the toolbox chooses a driver from the pool.",
        "interpret": "Use this as an overview figure. It combines driver ranking, correlation-vs-MI, lagged dependence, PID, TE-vs-lag, and TE network diagnostics. Interpret each panel as complementary, not as a single causal proof.",
    },
    "Figure 6 Temporal Summary": {
        "what": "Creates the single-site temporal outputs needed for manuscript Figure 6: full lagged-MI profiles, directional transfer-entropy profiles, supported source-target links, and target-memory-conditioned temporal PID for every selected driver pair.",
        "needs": "Select Y as the observed target and the harmonized driver pool. Open Figure 6 setup, choose Native, Daily, or Weekly resolution, review the variable-specific aggregation rules, then set the lag window, temporal-surrogate options, and bootstrap settings. At least two drivers are needed for temporal PID.",
        "interpret": "Lagged MI describes when dependence is strongest. Transfer entropy asks whether past source information reduces uncertainty in future target values beyond the target's own past. Temporal PID decomposes the joint information from two lagged drivers about the current target, within states of the previous target, into redundancy, unique information, and synergy. These diagnostics indicate timing and directional information structure, not definitive causality.",
    },
    "Conditional MI I(X;Y|Z)": {
        "what": "Measures the dependence between X and Y after accounting for Z.",
        "needs": "Select X, Y, and Z. Z should be a variable that may explain shared dependence, such as radiation, temperature, or seasonality proxy.",
        "interpret": "High conditional MI means X and Y still share information after conditioning on Z. Low conditional MI suggests their dependence may be largely explained by Z. Results depend strongly on the conditioning variable and binning.",
    },
    "Lagged MI I(X_t;Y_{t+lag})": {
        "what": "Measures time-shifted dependence between a source X and target Y across negative and positive lags.",
        "needs": "Select X as source and Y as target. Select a valid timestamp column and temporal grid, then set Max |lag| in time steps.",
        "interpret": "Positive lags compare X_t with Y at a later time, so X leads Y. Negative lags indicate Y leads X. The toolbox preserves missing calendar steps before shifting and reports both MI in bits and normalized MI. Lagged MI is not causal and can reflect seasonality or autocorrelation.",
    },
    "PID (X1,X2→Y)": {
        "what": "Decomposes the information from two drivers into redundancy, unique information, and synergy with respect to one target.",
        "needs": "Select X as driver 1, Y as driver 2, and Z as the target.",
        "interpret": "Redundancy means both drivers provide overlapping information. Unique information means one driver provides information not present in the other. Synergy means the two drivers provide information jointly that is not available from either one alone.",
    },
    "Pairwise PID Matrix": {
        "what": "Runs PID for every pair of selected drivers with respect to one target and maps redundancy, total unique information, and synergy.",
        "needs": "Select Y as the target. Select at least two drivers in the variable pool list.",
        "interpret": "Each cell is one driver pair. High redundancy indicates overlapping environmental information. High unique information indicates more independent contributions. High synergy indicates joint or multivariate control of the target.",
    },
    "Transfer Entropy TE(X→Y)": {
        "what": "Estimates directional, lagged information transfer from source X to target Y, conditioned on the target's own past.",
        "needs": "Select X as source, Y as target, and TE lag. Optional permutations provide an approximate significance test.",
        "interpret": "Higher TE means past X helps reduce uncertainty in future Y beyond what past Y already explains. TE is more directional than MI, but still should be interpreted carefully with autocorrelated EC data.",
    },
    "Transfer Entropy vs Lag": {
        "what": "Computes TE(X→Y) across a range of positive lags.",
        "needs": "Select X as source, Y as target, a valid timestamp column, and Max |lag| as the maximum positive lag. Choose temporal surrogates for peak support.",
        "interpret": "The toolbox reports raw TE, normalized TE, peak lag, and a maximum-statistic surrogate p-value that accounts for searching across lags. Supported peaks indicate directional timing structure, not direct proof of physical causation.",
    },
    "TE Network": {
        "what": "Computes pairwise transfer entropy among selected variables and displays a source-to-target matrix.",
        "needs": "Select two or more variables in the variable pool list. Set TE lag and optional permutations.",
        "interpret": "Rows are sources and columns are targets. Larger values suggest stronger directed lagged dependence. Use the network as an exploratory diagnostic; screen for autocorrelation, shared diurnal cycles, and physical plausibility.",
    },
}


def format_method_guide(method_name: str) -> str:
    guide = METHOD_GUIDES.get(method_name, None)
    if guide is None:
        return "No method guide is available for this option."
    return (
        f"Selected method: {method_name}\n\n"
        f"What it does:\n{guide['what']}\n\n"
        f"What you need to select:\n{guide['needs']}\n\n"
        f"How to interpret:\n{guide['interpret']}\n"
    )


def _it_workflow_guide_text() -> str:
    return (
        "MeaningFlux Information Theory Toolbox — guided workflow\n\n"
        "Use this toolbox after visual inspection and, when available, after running the Machine Learning toolbox.\n\n"
        "A. If you are analyzing observations only:\n"
        "1) Select an observed target flux in Y, such as FC, FCH4, FN2O, LE, or H.\n"
        "2) Select physically meaningful drivers in the variable pool, such as SW_IN, TA, VPD, TS, USTAR, WS, SWC, or precipitation.\n"
        "3) Start with MI Driver Ranking to identify which drivers share information with the target.\n"
        "4) Use Correlation vs MI Ranking to see whether relationships are mostly linear or nonlinear.\n"
        "5) Use Pairwise PID Matrix to separate redundancy, unique information, and synergy between driver pairs.\n"
        "6) Use Figure 6 Temporal Summary, Lagged MI, or TE vs Lag only with a valid timestamp column. The toolbox creates a complete regular grid and keeps missing time steps as NaN before lagging.\n\n"
        "B. If you are evaluating a machine-learning model:\n"
        "1) In the ML toolbox, run the model and click Export for IT.\n"
        "2) Here, click Load ML-to-IT CSV. The toolbox will try to detect observed target, prediction columns, and driver columns.\n"
        "3) Use Y = observed target, for example FC_obs.\n"
        "4) Use Z = predicted target, for example FC_pred_RF.\n"
        "5) Run Model-vs-Observed MI and Model-vs-Observed PID Matrix.\n"
        "6) Interpret model-minus-observed differences near zero as better functional agreement between model and observation.\n\n"
        "Core interpretation:\n"
        "• ML predictive performance asks: can the model predict the flux?\n"
        "• IT functional performance asks: does the model preserve the observed driver–flux information structure?\n"
        "• A model can have good R² but still overuse one driver, miss synergy, or misrepresent redundancy.\n\n"
        "Practical warning:\n"
        "Information-theory metrics show dependence and timing structure. They do not prove physical causality by themselves. Always interpret results with EC physics, seasonality, quality control, and footprint context.\n"
    )


def _to_1d_array(x: ArrayLike) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim > 1:
        x = x.reshape(-1)
    return x


def _validate_same_length(*arrays: ArrayLike) -> List[np.ndarray]:
    arrs = [_to_1d_array(a) for a in arrays]
    lengths = {len(a) for a in arrs}
    if len(lengths) > 1:
        raise ValueError("All input arrays must have the same length.")
    return arrs


def _remove_nan(*arrays: np.ndarray) -> List[np.ndarray]:
    if not arrays:
        return []
    mask = np.ones_like(arrays[0], dtype=bool)
    for a in arrays:
        mask &= ~np.isnan(a)
    return [a[mask] for a in arrays]


def _align_dropna(*arrs: np.ndarray) -> List[np.ndarray]:
    """
    Trim arrays to common length and remove positions with any NaN.

    This helper is kept for arrays that are already known to be row-aligned.
    For variables pulled from a DataFrame, prefer _pair_arrays_from_df() or
    _multi_arrays_from_df(), which drop missing values row-wise before arrays
    are separated.
    """
    if not arrs:
        return []
    L = min(len(a) for a in arrs)
    arrs = [np.asarray(a, dtype=float).reshape(-1)[:L] for a in arrs]
    mask = np.ones(L, dtype=bool)
    for a in arrs:
        mask &= ~np.isnan(a)
    return [a[mask] for a in arrs]


# =============================================================================
# DataFrame-safe alignment, filtering, and reproducibility helpers
# =============================================================================

def _numeric_frame_rowwise(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Return a numeric DataFrame with row-wise NaN removal for selected columns."""
    cols = list(dict.fromkeys([c for c in columns if c in df.columns]))
    if not cols:
        raise ValueError("No selected columns were found in the DataFrame.")
    out = df.loc[:, cols].apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    return out


def _pair_arrays_from_df(df: pd.DataFrame, x_name: str, y_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Get two row-aligned numeric arrays from a DataFrame."""
    dat = _numeric_frame_rowwise(df, [x_name, y_name])
    if dat.empty:
        raise ValueError(f"No overlapping numeric samples for {x_name} and {y_name}.")
    return dat[x_name].to_numpy(dtype=float), dat[y_name].to_numpy(dtype=float)


def _multi_arrays_from_df(df: pd.DataFrame, columns: Iterable[str]) -> List[np.ndarray]:
    """Get multiple row-aligned numeric arrays from a DataFrame."""
    cols = list(dict.fromkeys(columns))
    dat = _numeric_frame_rowwise(df, cols)
    if dat.empty:
        raise ValueError(f"No overlapping numeric samples for: {', '.join(cols)}.")
    return [dat[c].to_numpy(dtype=float) for c in cols]


def _parse_timestamp_series(values: pd.Series) -> pd.Series:
    """Parse ordinary and AmeriFlux-style timestamps without creating 1970 dates.

    Numeric AmeriFlux timestamps such as 201301010030 must be interpreted as
    YYYYMMDDHHMM rather than nanoseconds after 1970. The parser also accepts
    YYYYMMDD, YYYYMMDDHH, YYYYMMDDHHMMSS, and ordinary datetime strings.
    """
    s = pd.Series(values, copy=False)
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, errors="coerce")

    text_values = s.astype("string").str.strip()
    text_values = text_values.str.replace(r"\.0+$", "", regex=True)
    parsed = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    formats = {
        8: "%Y%m%d",
        10: "%Y%m%d%H",
        12: "%Y%m%d%H%M",
        14: "%Y%m%d%H%M%S",
    }
    digit_mask = text_values.str.fullmatch(r"\d+", na=False)
    for length, fmt in formats.items():
        mask = digit_mask & text_values.str.len().eq(length)
        if bool(mask.any()):
            parsed.loc[mask] = pd.to_datetime(
                text_values.loc[mask], format=fmt, errors="coerce"
            )

    remaining = parsed.isna() & text_values.notna() & text_values.ne("")
    if bool(remaining.any()):
        parsed.loc[remaining] = pd.to_datetime(
            text_values.loc[remaining], errors="coerce"
        )
    return parsed


def _infer_regular_timedelta(timestamps: pd.Series) -> pd.Timedelta:
    """Infer the dominant positive sampling interval from parsed timestamps."""
    t = _parse_timestamp_series(pd.Series(timestamps)).dropna().sort_values().drop_duplicates()
    if len(t) < 3:
        raise ValueError("At least three valid timestamps are required to infer a temporal grid.")
    diffs = t.diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        raise ValueError("Could not infer a positive timestamp interval.")
    counts = diffs.value_counts()
    step = counts.index[0] if not counts.empty else diffs.median()
    if not isinstance(step, pd.Timedelta):
        step = pd.to_timedelta(step)
    return step


def _infer_variable_aggregation(column_name: str) -> str:
    """Suggest a physically sensible aggregation method from a variable name."""
    token = re.sub(r"[^A-Z0-9]+", "_", str(column_name).upper()).strip("_")
    base = token.split("_")[0] if token else token

    # Circular variables must not use an arithmetic mean.
    if token == "WD" or token.startswith("WD_") or "WIND_DIRECTION" in token:
        return "circular_mean"

    # Accumulated inputs are normally summed. Users can override this when the
    # source column is a rate rather than an interval accumulation.
    accumulation_tokens = (
        "PRECIP", "P_F", "P_RAIN", "RAIN", "RAINFALL", "IRRIG", "SNOWFALL",
    )
    if token == "P" or any(k in token for k in accumulation_tokens):
        return "sum"

    if "_MAX" in token or token.endswith("MAX"):
        return "max"
    if "_MIN" in token or token.endswith("MIN"):
        return "min"

    # Flux rates, state variables, radiation rates, turbulence, and most EC
    # drivers are represented by their mean over the requested interval.
    return "mean"


def _aggregate_numeric_series(values: pd.Series, method: str) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if x.empty:
        return float("nan")
    method = str(method or "mean").lower()
    if method == "sum":
        return float(x.sum())
    if method == "median":
        return float(x.median())
    if method == "min":
        return float(x.min())
    if method == "max":
        return float(x.max())
    if method == "first":
        return float(x.iloc[0])
    if method == "last":
        return float(x.iloc[-1])
    if method == "circular_mean":
        radians = np.deg2rad(x.to_numpy(dtype=float) % 360.0)
        angle = np.arctan2(np.nanmean(np.sin(radians)), np.nanmean(np.cos(radians)))
        return float(np.rad2deg(angle) % 360.0)
    return float(x.mean())


def _regularize_temporal_frame(
    df: pd.DataFrame,
    timestamp_col: str,
    columns: Iterable[str],
    frequency: str = "auto",
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Create a complete regular grid while preserving missing temporal steps."""
    if timestamp_col not in df.columns:
        raise ValueError(
            f"Timestamp column '{timestamp_col}' was not found. Temporal methods require timestamps."
        )
    cols = list(dict.fromkeys([c for c in columns if c in df.columns]))
    if not cols:
        raise ValueError("No requested temporal-analysis variables were found.")

    work = pd.DataFrame({"__timestamp__": _parse_timestamp_series(df[timestamp_col])})
    for c in cols:
        work[c] = pd.to_numeric(df[c], errors="coerce")
    work = work.dropna(subset=["__timestamp__"]).sort_values("__timestamp__")
    if work.empty:
        raise ValueError(
            "No valid timestamps are available. Check the selected timestamp column; "
            "AmeriFlux numeric timestamps should look like YYYYMMDDHHMM."
        )

    work = work.groupby("__timestamp__", as_index=True)[cols].mean().sort_index()
    if len(work.index) < 3:
        raise ValueError("At least three unique timestamps are required for temporal analysis.")

    freq_text = str(frequency or "auto").strip()
    if not freq_text or freq_text.lower() == "auto":
        step = _infer_regular_timedelta(pd.Series(work.index))
        freq_used = str(step)
        full_index = pd.date_range(work.index.min(), work.index.max(), freq=step)
    else:
        try:
            offset = pd.tseries.frequencies.to_offset(freq_text)
        except Exception as exc:
            raise ValueError(
                "Temporal grid must be 'auto' or a pandas frequency such as D, H, or 30min."
            ) from exc
        freq_used = str(offset.freqstr)
        full_index = pd.date_range(work.index.min(), work.index.max(), freq=offset)

    regular = work.reindex(full_index)
    regular.index.name = timestamp_col
    original_unique = int(len(work))
    inserted = int(len(regular) - original_unique)
    meta = {
        "timestamp_column": timestamp_col,
        "temporal_frequency": freq_used,
        "temporal_resolution": "Native",
        "lag_unit": "native time steps",
        "temporal_rows_original_unique": original_unique,
        "temporal_rows_regular_grid": int(len(regular)),
        "temporal_rows_inserted_as_missing": max(0, inserted),
        "temporal_start": str(regular.index.min()),
        "temporal_end": str(regular.index.max()),
        "temporal_gap_policy": "complete regular grid; missing time steps retained as NaN before lagging",
    }
    return regular, meta


def _prepare_figure6_temporal_frame(
    df: pd.DataFrame,
    timestamp_col: str,
    columns: Iterable[str],
    resolution: str = "Daily",
    native_frequency: str = "auto",
    aggregation_rules: Optional[Dict[str, str]] = None,
    minimum_coverage_percent: float = 75.0,
    weekly_minimum_valid_days: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Prepare Native, Daily, or Weekly data internally for Figure 6.

    Daily and weekly values use variable-specific editable aggregation rules.
    Coverage is evaluated separately for every variable. Failed intervals remain
    NaN; the observed target is never gap-filled.
    """
    resolution = str(resolution or "Daily").title()
    if resolution not in {"Native", "Daily", "Weekly"}:
        raise ValueError("Temporal resolution must be Native, Daily, or Weekly.")
    coverage = float(minimum_coverage_percent)
    if not (0.0 < coverage <= 100.0):
        raise ValueError("Minimum valid coverage must be greater than 0 and at most 100 percent.")

    cols = list(dict.fromkeys([c for c in columns if c in df.columns]))
    if not cols:
        raise ValueError("No requested Figure 6 variables were found.")
    rules = {c: (aggregation_rules or {}).get(c, _infer_variable_aggregation(c)) for c in cols}

    if resolution == "Native":
        regular, meta = _regularize_temporal_frame(
            df, timestamp_col=timestamp_col, columns=cols, frequency=native_frequency,
        )
        meta.update({
            "temporal_resolution": "Native",
            "minimum_valid_coverage_percent": coverage,
            "weekly_minimum_valid_days": int(weekly_minimum_valid_days),
            "aggregation_rules": json.dumps(rules, sort_keys=True),
            "lag_unit": "native time steps",
        })
        return regular, meta

    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' was not found.")
    work = pd.DataFrame({"__timestamp__": _parse_timestamp_series(df[timestamp_col])})
    for c in cols:
        work[c] = pd.to_numeric(df[c], errors="coerce")
    work = work.dropna(subset=["__timestamp__"]).sort_values("__timestamp__")
    if work.empty:
        raise ValueError("No valid timestamps are available for temporal aggregation.")

    # Average exact duplicate timestamps before interval aggregation.
    work = work.groupby("__timestamp__", as_index=True)[cols].mean().sort_index()
    native_step = _infer_regular_timedelta(pd.Series(work.index))
    expected_per_day = max(1, int(round(pd.Timedelta(days=1) / native_step)))
    required_per_day = max(1, int(math.ceil(expected_per_day * coverage / 100.0)))

    day_key = work.index.normalize()
    day_index = pd.date_range(day_key.min(), day_key.max(), freq="D")
    daily = pd.DataFrame(index=day_index, columns=cols, dtype=float)
    daily_counts = pd.DataFrame(index=day_index, columns=cols, dtype=float)
    for c in cols:
        grouped = work[c].groupby(day_key)
        counts = grouped.count().reindex(day_index, fill_value=0)
        values = grouped.apply(lambda s, method=rules[c]: _aggregate_numeric_series(s, method)).reindex(day_index)
        values = values.where(counts >= required_per_day)
        daily[c] = values
        daily_counts[c] = counts
    daily.index.name = timestamp_col

    if resolution == "Daily":
        prepared = daily
        lag_unit = "days"
        frequency = "D"
        required_week = int(weekly_minimum_valid_days)
    else:
        required_week = max(1, min(7, int(weekly_minimum_valid_days)))
        week_starts = daily.index - pd.to_timedelta(daily.index.weekday, unit="D")
        first_week = week_starts.min()
        last_week = week_starts.max()
        week_index = pd.date_range(first_week, last_week, freq="7D")
        weekly = pd.DataFrame(index=week_index, columns=cols, dtype=float)
        for c in cols:
            grouped = daily[c].groupby(week_starts)
            counts = grouped.count().reindex(week_index, fill_value=0)
            values = grouped.apply(lambda s, method=rules[c]: _aggregate_numeric_series(s, method)).reindex(week_index)
            weekly[c] = values.where(counts >= required_week)
        weekly.index.name = timestamp_col
        prepared = weekly
        lag_unit = "weeks"
        frequency = "7D"

    meta = {
        "timestamp_column": timestamp_col,
        "input_temporal_frequency": str(native_step),
        "temporal_frequency": frequency,
        "temporal_resolution": resolution,
        "lag_unit": lag_unit,
        "minimum_valid_coverage_percent": coverage,
        "expected_native_records_per_day": int(expected_per_day),
        "required_native_records_per_day": int(required_per_day),
        "weekly_minimum_valid_days": int(required_week),
        "aggregation_rules": json.dumps(rules, sort_keys=True),
        "temporal_rows_original_unique": int(len(work)),
        "temporal_rows_regular_grid": int(len(prepared)),
        "temporal_rows_inserted_as_missing": int(prepared.isna().all(axis=1).sum()),
        "temporal_start": str(prepared.index.min()),
        "temporal_end": str(prepared.index.max()),
        "temporal_gap_policy": "complete prepared grid; intervals failing coverage retained as NaN; no target gap filling",
    }
    return prepared, meta

def _lagged_pair_arrays(x: ArrayLike, y: ArrayLike, lag: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create an aligned lag pair while preserving positions until after shifting."""
    x = _to_1d_array(x)
    y = _to_1d_array(y)
    if x.size != y.size:
        raise ValueError("x and y must have the same length.")
    lag = int(lag)
    if lag > 0:
        x_l, y_l = x[:-lag], y[lag:]
    elif lag < 0:
        k = -lag
        x_l, y_l = x[k:], y[:-k]
    else:
        x_l, y_l = x.copy(), y.copy()
    return tuple(_remove_nan(x_l, y_l))


def _guess_timestamp_columns(columns: Iterable[str]) -> List[str]:
    cols = list(columns)
    preferred = []
    patterns = ("timestamp", "time", "datetime", "date", "datestamp")
    for c in cols:
        if any(p in str(c).lower() for p in patterns):
            preferred.append(c)
    return list(dict.fromkeys(preferred + cols))


def _guess_radiation_columns(columns: Iterable[str]) -> List[str]:
    cols = list(columns)
    preferred = []
    patterns = ("sw_in", "ppfd", "par", "rg", "netrad", "rn", "swin")
    for c in cols:
        cl = str(c).lower()
        if any(p in cl for p in patterns):
            preferred.append(c)
    return list(dict.fromkeys(preferred + cols))


def detect_prediction_columns(columns: Iterable[str], observed_target: str = "") -> List[str]:
    cols = list(columns)
    pred_patterns = [
        r"(^|_)(pred|prediction|predicted|model|modeled|sim|simulated|hat)(_|$)",
        r"(_pred_|_pred$|^pred_)",
        r"(_rf$|_randomforest$|_lstm$|_mlr$|_linear$|_gb$|_svr$|_xgb$|_mlp$)",
    ]
    target_l = str(observed_target).lower()
    candidates = []
    for c in cols:
        cl = str(c).lower()
        if observed_target and cl == target_l:
            continue
        if observed_target and target_l and target_l not in cl:
            if not any(k in cl for k in ["pred", "model", "sim", "hat"]):
                continue
        if any(re.search(pat, cl) for pat in pred_patterns):
            candidates.append(c)
    return list(dict.fromkeys(candidates))


def _surrogate_series(y: np.ndarray, rng: np.random.Generator, surrogate_type: str = "random", block_size: int = 48) -> np.ndarray:
    y = np.asarray(y, dtype=float).reshape(-1)
    n = len(y)
    if n == 0:
        return y.copy()
    surrogate_type = (surrogate_type or "random").lower()
    if surrogate_type == "circular_shift":
        if n < 3:
            return rng.permutation(y)
        # Avoid trivial one-step shifts when possible. The block-size control
        # acts as the minimum separation from the original alignment.
        min_shift = int(max(1, min(block_size, max(1, n // 4))))
        if n > 2 * min_shift + 1:
            shift = int(rng.integers(min_shift, n - min_shift))
        else:
            shift = int(rng.integers(1, n))
        return np.roll(y, shift)
    if surrogate_type == "block_shuffle":
        b = int(max(2, min(block_size, n)))
        blocks = [y[i:i+b] for i in range(0, n, b)]
        order = rng.permutation(len(blocks))
        return np.concatenate([blocks[i] for i in order])[:n]
    return rng.permutation(y)


def surrogate_p_value(stat_fn, x: np.ndarray, y: np.ndarray, observed_value: Optional[float] = None,
                      n_perm: int = 200, surrogate_type: str = "random", block_size: int = 48,
                      seed: Optional[int] = None) -> float:
    if not n_perm or n_perm <= 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    x, y = _align_dropna(x, y)
    if len(x) < 3:
        return float("nan")
    stat_obs = float(observed_value if observed_value is not None else stat_fn(x, y))
    count = 0
    for _ in range(int(n_perm)):
        y_surr = _surrogate_series(y, rng, surrogate_type=surrogate_type, block_size=block_size)
        try:
            if float(stat_fn(x, y_surr)) >= stat_obs:
                count += 1
        except Exception:
            continue
    return float((count + 1) / (int(n_perm) + 1))


# =============================================================================
# Discretization
# =============================================================================

def discretize_equal_width(
    x: ArrayLike,
    n_bins: int = 10,
    return_edges: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Equal-width discretization. NaNs -> -1.
    """
    x = _to_1d_array(x)
    mask = ~np.isnan(x)
    x_valid = x[mask]

    if x_valid.size == 0:
        raise ValueError("Cannot discretize: all values are NaN.")

    xmin, xmax = np.min(x_valid), np.max(x_valid)
    if np.isclose(xmin, xmax):
        edges = np.array([xmin - 0.5, xmax + 0.5])
        labels = np.zeros_like(x, dtype=int)
    else:
        edges = np.linspace(xmin, xmax, n_bins + 1)
        labels = np.full_like(x, fill_value=-1, dtype=int)
        lab_valid = np.digitize(x_valid, edges[1:-1], right=False)
        lab_valid = np.clip(lab_valid, 0, n_bins - 1)
        labels[mask] = lab_valid

    if return_edges:
        return labels, edges
    return labels


def discretize_equal_frequency(
    x: ArrayLike,
    n_bins: int = 10,
    return_edges: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Equal-frequency (quantile) discretization. NaNs -> -1.
    """
    x = _to_1d_array(x)
    mask = ~np.isnan(x)
    x_valid = x[mask]

    if x_valid.size == 0:
        raise ValueError("Cannot discretize: all values are NaN.")

    qs = np.linspace(0, 100, n_bins + 1)
    edges = np.nanpercentile(x_valid, qs)
    edges = np.unique(edges)

    if edges.size < 2:
        edges = np.array([np.min(x_valid) - 0.5, np.max(x_valid) + 0.5])

    n_bins_eff = edges.size - 1
    labels = np.full_like(x, fill_value=-1, dtype=int)
    lab_valid = np.digitize(x_valid, edges[1:-1], right=False)
    lab_valid = np.clip(lab_valid, 0, n_bins_eff - 1)
    labels[mask] = lab_valid

    if return_edges:
        return labels, edges
    return labels


# =============================================================================
# Discrete entropy & mutual information
# =============================================================================

def _pmf_from_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    if labels.size == 0:
        raise ValueError("Cannot compute PMF: empty array.")
    max_label = labels.max()
    if max_label < 0:
        raise ValueError("Labels must be >= 0.")
    counts = np.bincount(labels, minlength=max_label + 1).astype(float)
    total = counts.sum()
    if total <= 0:
        raise ValueError("Counts sum to zero.")
    return counts / total


def entropy_discrete(labels: ArrayLike, base: float = 2.0) -> float:
    labels = _to_1d_array(labels)
    (labels,) = _remove_nan(labels)
    pmf = _pmf_from_labels(labels)
    pmf = pmf[pmf > 0]
    log_p = np.log(pmf) / np.log(base)
    return float(-np.sum(pmf * log_p))


def joint_entropy_discrete(
    x_labels: ArrayLike,
    y_labels: ArrayLike,
    base: float = 2.0,
) -> float:
    x, y = _validate_same_length(x_labels, y_labels)
    x, y = _remove_nan(x, y)
    x = x.astype(int)
    y = y.astype(int)

    if x.size == 0:
        raise ValueError("No valid samples after removing NaNs.")

    ny = int(y.max()) + 1
    joint_index = x * ny + y
    pmf = _pmf_from_labels(joint_index)
    pmf = pmf[pmf > 0]
    log_p = np.log(pmf) / np.log(base)
    return float(-np.sum(pmf * log_p))


def mutual_information_discrete(
    x_labels: ArrayLike,
    y_labels: ArrayLike,
    base: float = 2.0,
) -> float:
    x, y = _validate_same_length(x_labels, y_labels)
    x, y = _remove_nan(x, y)
    Hx = entropy_discrete(x, base=base)
    Hy = entropy_discrete(y, base=base)
    Hxy = joint_entropy_discrete(x, y, base=base)
    return float(Hx + Hy - Hxy)


def conditional_entropy_discrete(
    y_labels: ArrayLike,
    z_labels: ArrayLike,
    base: float = 2.0,
) -> float:
    y, z = _validate_same_length(y_labels, z_labels)
    y, z = _remove_nan(y, z)
    y = y.astype(int)
    z = z.astype(int)

    if y.size == 0:
        raise ValueError("No valid samples after removing NaNs.")

    z_unique, z_codes = np.unique(z, return_inverse=True)
    K = z_unique.size
    N = y.size
    H = 0.0
    base_log = np.log(base)

    for k in range(K):
        mask = (z_codes == k)
        yk = y[mask]
        if yk.size == 0:
            continue
        pmf_k = _pmf_from_labels(yk)
        pmf_k = pmf_k[pmf_k > 0]
        log_p_k = np.log(pmf_k) / base_log
        H_k = -np.sum(pmf_k * log_p_k)
        H += (yk.size / N) * H_k

    return float(H)


def conditional_mutual_information_discrete(
    x_labels: ArrayLike,
    y_labels: ArrayLike,
    z_labels: ArrayLike,
    base: float = 2.0,
) -> float:
    """
    I(X;Y|Z) = H(X|Z) + H(Y|Z) - H(X,Y|Z)
    """
    x, y, z = _validate_same_length(x_labels, y_labels, z_labels)
    x, y, z = _remove_nan(x, y, z)
    x = x.astype(int)
    y = y.astype(int)
    z = z.astype(int)

    if x.size == 0:
        raise ValueError("No valid samples after removing NaNs.")

    def H_given_Z(u: np.ndarray, zarr: np.ndarray) -> float:
        return conditional_entropy_discrete(u, zarr, base=base)

    ny = int(y.max()) + 1 if y.size > 0 else 1
    xy = x.astype(int) * ny + y.astype(int)

    Hx_z = H_given_Z(x, z)
    Hy_z = H_given_Z(y, z)
    Hxy_z = H_given_Z(xy, z)
    return float(Hx_z + Hy_z - Hxy_z)


# =============================================================================
# Continuous wrappers
# =============================================================================

def mutual_information(
    x: ArrayLike,
    y: ArrayLike,
    base: float = 2.0,
    method: str = "hist",
    n_bins: int = 10,
    disc: str = "equal_freq",
) -> float:
    if method != "hist":
        raise NotImplementedError("Only 'hist' method is implemented here.")

    x, y = _validate_same_length(x, y)
    x, y = _remove_nan(x, y)

    if disc == "equal_width":
        x_lab = discretize_equal_width(x, n_bins=n_bins)
        y_lab = discretize_equal_width(y, n_bins=n_bins)
    elif disc == "equal_freq":
        x_lab = discretize_equal_frequency(x, n_bins=n_bins)
        y_lab = discretize_equal_frequency(y, n_bins=n_bins)
    else:
        raise ValueError("disc must be 'equal_width' or 'equal_freq'.")

    return mutual_information_discrete(x_lab, y_lab, base=base)


def lagged_mutual_information(
    x: ArrayLike,
    y: ArrayLike,
    lags: Iterable[int],
    base: float = 2.0,
    method: str = "hist",
    n_bins: int = 10,
    disc: str = "equal_freq",
) -> Dict[int, float]:
    x = _to_1d_array(x)
    y = _to_1d_array(y)
    if x.size != y.size:
        raise ValueError("x and y must have the same length.")

    mi_by_lag: Dict[int, float] = {}
    for lag in lags:
        lag = int(lag)
        if lag > 0:
            x_l = x[:-lag]
            y_l = y[lag:]
        elif lag < 0:
            lag_abs = -lag
            x_l = x[lag_abs:]
            y_l = y[:-lag_abs]
        else:
            x_l = x.copy()
            y_l = y.copy()

        if x_l.size < 2:
            mi_by_lag[lag] = float("nan")
            continue

        mi_by_lag[lag] = mutual_information(
            x_l, y_l,
            base=base,
            method=method,
            n_bins=n_bins,
            disc=disc,
        )

    return mi_by_lag


def lagged_mutual_information_table(
    x: ArrayLike,
    y: ArrayLike,
    lags: Iterable[int],
    base: float = 2.0,
    method: str = "hist",
    n_bins: int = 10,
    disc: str = "equal_freq",
    min_samples: int = 20,
) -> pd.DataFrame:
    """Return lagged MI in bits and normalized by aligned target entropy.

    Missing values are removed only after each lag-specific pair is formed.
    This is essential when arrays originate from a complete regular time grid.
    """
    rows = []
    for lag in [int(v) for v in lags]:
        x_l, y_l = _lagged_pair_arrays(x, y, lag)
        n = int(len(y_l))
        if n < int(min_samples):
            rows.append({
                "lag": lag, "mi_bits": np.nan, "target_entropy_bits": np.nan,
                "mi_norm": np.nan, "n_samples": n,
            })
            continue
        try:
            mi = mutual_information(
                x_l, y_l, base=base, method=method, n_bins=n_bins, disc=disc,
            )
            hy = entropy_continuous_discrete(
                y_l, base=base, n_bins=n_bins, disc=disc,
            )
            mi_norm = float(mi / hy) if np.isfinite(hy) and hy > 0 else np.nan
        except Exception:
            mi = hy = mi_norm = np.nan
        rows.append({
            "lag": lag,
            "mi_bits": float(mi) if np.isfinite(mi) else np.nan,
            "target_entropy_bits": float(hy) if np.isfinite(hy) else np.nan,
            "mi_norm": float(mi_norm) if np.isfinite(mi_norm) else np.nan,
            "n_samples": n,
        })
    return pd.DataFrame(rows)


# =============================================================================
# MI driver ranking
# =============================================================================

def mutual_information_driver_ranking(
    df: pd.DataFrame,
    target: str,
    drivers: Iterable[str],
    base: float = 2.0,
    estimator: str = "hist",
    n_bins: int = 10,
    disc: str = "equal_freq",
    min_samples: int = 50,
    n_perm: int = 0,
    surrogate_type: str = "random",
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Rank multiple drivers by their mutual information with a target variable.

    Parameters
    ----------
    df : pandas.DataFrame
        Site-level data table.
    target : str
        Target variable, usually an ecosystem flux such as FC, FCH4, or FN2O.
    drivers : iterable of str
        Candidate source variables to compare against the target.
    base : float
        Logarithm base for information measures. base=2 gives bits.
    estimator : {"hist", "kde-tip"}
        Estimator used for MI. "hist" uses discretization and empirical PMFs.
        "kde-tip" uses the KDE/TIP backend.
    n_bins : int
        Number of bins for histogram estimation, or grid size for KDE/TIP.
    disc : {"equal_freq", "equal_width"}
        Discretization method for histogram estimation.
    min_samples : int
        Minimum number of overlapping valid samples required per driver.
    n_perm : int
        Number of permutations for optional p-values. If 0, p-values are not computed.
    seed : int or None
        Optional random seed used for permutation testing.

    Returns
    -------
    pandas.DataFrame
        Ranked table with columns including rank, driver, target, mi_bits,
        n_samples, estimator, n_bins, disc, and p_value when requested.
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' was not found in the DataFrame.")

    y_all = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)
    hy_bits = entropy_continuous_discrete(y_all, base=base, n_bins=n_bins, disc=disc)
    rows = []
    rng = np.random.default_rng(seed)

    # Preserve user selection order while removing repeated driver names.
    drivers = list(dict.fromkeys(drivers))

    for driver in drivers:
        if not driver or driver == target:
            continue
        if driver not in df.columns:
            continue

        x, y = _pair_arrays_from_df(df, driver, target)

        if len(x) < min_samples:
            continue

        try:
            if estimator == "kde-tip":
                def _stat(a, b):
                    return kde_mi_2d(
                        a,
                        b,
                        N=n_bins,
                        bin_scheme="global",
                        method="KDE",
                    )

                mi = _stat(x, y)
                used_disc = "kde-tip"
            else:
                def _stat(a, b):
                    return mutual_information(
                        a,
                        b,
                        base=base,
                        method="hist",
                        n_bins=n_bins,
                        disc=disc,
                    )

                mi = _stat(x, y)
                used_disc = disc

            p_value = np.nan
            if n_perm and n_perm > 0:
                p_value = surrogate_p_value(_stat, x, y, observed_value=float(mi),
                                            n_perm=int(n_perm), surrogate_type=surrogate_type,
                                            block_size=48, seed=seed)

        except Exception:
            # Skip variables that cannot be evaluated under the selected settings.
            continue

        rows.append({
            "driver": driver,
            "target": target,
            "mi_bits": float(mi),
            "mi_norm": float(mi / hy_bits) if hy_bits and not np.isnan(hy_bits) and hy_bits > 0 else np.nan,
            "target_entropy_bits": float(hy_bits),
            "n_samples": int(len(x)),
            "estimator": estimator,
            "n_bins": int(n_bins),
            "disc": used_disc,
            "p_value": float(p_value) if not np.isnan(p_value) else np.nan,
            "surrogate_type": surrogate_type if n_perm and n_perm > 0 else "",
        })

    if not rows:
        raise ValueError(
            "No valid MI rankings could be computed. Check the selected target, "
            "driver variables, and minimum sample requirement."
        )

    out = pd.DataFrame(rows)
    out = out.sort_values("mi_bits", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


# =============================================================================
# Single-site IT diagnostic helpers
# =============================================================================

def entropy_continuous_discrete(
    x: ArrayLike,
    base: float = 2.0,
    n_bins: int = 10,
    disc: str = "equal_freq",
) -> float:
    """
    Estimate entropy H(X) for a continuous variable after discretization.

    This helper is used to report normalized mutual information:
    I(X;Y) / H(Y). It uses the same histogram discretization settings as
    the MI estimator so the diagnostic remains internally consistent.
    """
    x = _to_1d_array(x)
    (x,) = _remove_nan(x)
    if x.size == 0:
        return float("nan")
    if disc == "equal_width":
        labels = discretize_equal_width(x, n_bins=n_bins)
    elif disc == "equal_freq":
        labels = discretize_equal_frequency(x, n_bins=n_bins)
    else:
        raise ValueError("disc must be 'equal_width' or 'equal_freq'.")
    return entropy_discrete(labels, base=base)


def _safe_pearson_r(x: ArrayLike, y: ArrayLike) -> float:
    """
    Compute Pearson correlation after NaN removal.
    Returns NaN if the relationship is not defined.
    """
    x, y = _validate_same_length(x, y)
    x, y = _remove_nan(x, y)
    if x.size < 2:
        return float("nan")
    if np.isclose(np.nanstd(x), 0.0) or np.isclose(np.nanstd(y), 0.0):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def correlation_mi_driver_ranking(
    df: pd.DataFrame,
    target: str,
    drivers: Iterable[str],
    base: float = 2.0,
    estimator: str = "hist",
    n_bins: int = 10,
    disc: str = "equal_freq",
    min_samples: int = 50,
) -> pd.DataFrame:
    """
    Compare linear association and mutual information for a driver pool.

    This single-site diagnostic computes Pearson r, Pearson r², and
    I(driver; target) for each selected driver. It is useful for identifying
    drivers that show nonlinear dependence with the target even when linear
    association is weak or moderate.
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' was not found in the DataFrame.")

    y_all = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)
    hy_bits = entropy_continuous_discrete(y_all, base=base, n_bins=n_bins, disc=disc)
    drivers = list(dict.fromkeys(drivers))
    rows = []

    for driver in drivers:
        if not driver or driver == target or driver not in df.columns:
            continue

        x, y = _pair_arrays_from_df(df, driver, target)
        if len(x) < min_samples:
            continue

        try:
            r = _safe_pearson_r(x, y)
            r2 = float(r ** 2) if not np.isnan(r) else np.nan

            if estimator == "kde-tip":
                mi = kde_mi_2d(x, y, N=n_bins, bin_scheme="global", method="KDE")
                used_disc = "kde-tip"
            else:
                mi = mutual_information(
                    x,
                    y,
                    base=base,
                    method="hist",
                    n_bins=n_bins,
                    disc=disc,
                )
                used_disc = disc

        except Exception:
            continue

        rows.append({
            "driver": driver,
            "target": target,
            "pearson_r": float(r),
            "pearson_r2": float(r2),
            "mi_bits": float(mi),
            "mi_norm": float(mi / hy_bits) if hy_bits and not np.isnan(hy_bits) and hy_bits > 0 else np.nan,
            "target_entropy_bits": float(hy_bits),
            "n_samples": int(len(x)),
            "estimator": estimator,
            "n_bins": int(n_bins),
            "disc": used_disc,
        })

    if not rows:
        raise ValueError(
            "No valid correlation-vs-MI ranking could be computed. Check the target, "
            "driver variables, and minimum sample requirement."
        )

    out = pd.DataFrame(rows)
    rank_col = "mi_norm" if "mi_norm" in out.columns else "mi_bits"
    out["rank_mi"] = out[rank_col].rank(ascending=False, method="min").astype(int)
    out["rank_r2"] = out["pearson_r2"].rank(ascending=False, method="min").astype(int)
    out["rank_difference"] = out["rank_r2"] - out["rank_mi"]
    out = out.sort_values("mi_bits", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def transfer_entropy_by_lag(
    x: ArrayLike,
    y: ArrayLike,
    max_lag: int = 24,
    base: float = 2.0,
    n_bins: int = 10,
    disc: str = "equal_freq",
    min_samples: int = 20,
) -> pd.DataFrame:
    """Compute raw and normalized transfer entropy for lags 1..max_lag.

    Normalized TE is TE divided by H(Y_future | Y_past) for the same aligned
    lag-specific sample. Missing values are removed only after lag construction.
    """
    x = _to_1d_array(x)
    y = _to_1d_array(y)
    if x.size != y.size:
        raise ValueError("x and y must have the same length.")
    if max_lag < 1:
        raise ValueError("max_lag must be >= 1.")

    rows = []
    for lag in range(1, int(max_lag) + 1):
        try:
            comp = transfer_entropy_components(
                x, y, lag=lag, base=base, n_bins=n_bins, disc=disc,
                min_samples=min_samples,
            )
        except Exception:
            comp = {
                "te_bits": np.nan,
                "target_cond_entropy_bits": np.nan,
                "te_norm": np.nan,
                "n_samples": 0,
            }
        rows.append({"lag": int(lag), **comp})
    return pd.DataFrame(rows)


def _selected_or_default_drivers(
    selected: Iterable[str],
    all_cols: Iterable[str],
    target: str,
    max_drivers: Optional[int] = None,
) -> List[str]:
    """
    Clean a selected driver list and fall back to all columns except the target.
    """
    selected = [c for c in list(dict.fromkeys(selected)) if c and c != target]
    if not selected:
        selected = [c for c in all_cols if c != target]
    if max_drivers is not None:
        selected = selected[:max_drivers]
    return selected


# =============================================================================
# PID: min-information decomposition
# =============================================================================

def pid_min_information(
    x1: ArrayLike,
    x2: ArrayLike,
    y: ArrayLike,
    base: float = 2.0,
    method: str = "hist",
    n_bins: int = 10,
    disc: str = "equal_freq",
) -> Dict[str, float]:
    x1, x2, y = _validate_same_length(x1, x2, y)
    x1, x2, y = _remove_nan(x1, x2, y)

    I_x1_y = mutual_information(
        x1, y,
        base=base,
        method=method,
        n_bins=n_bins,
        disc=disc,
    )
    I_x2_y = mutual_information(
        x2, y,
        base=base,
        method=method,
        n_bins=n_bins,
        disc=disc,
    )

    if disc == "equal_width":
        x1_lab = discretize_equal_width(x1, n_bins=n_bins)
        x2_lab = discretize_equal_width(x2, n_bins=n_bins)
        y_lab = discretize_equal_width(y, n_bins=n_bins)
    else:
        x1_lab = discretize_equal_frequency(x1, n_bins=n_bins)
        x2_lab = discretize_equal_frequency(x2, n_bins=n_bins)
        y_lab = discretize_equal_frequency(y, n_bins=n_bins)

    nx2 = int(x2_lab.max()) + 1 if x2_lab.size > 0 else 1
    x12_lab = x1_lab.astype(int) * nx2 + x2_lab.astype(int)

    I_x12_y = mutual_information_discrete(x12_lab, y_lab, base=base)

    R = min(I_x1_y, I_x2_y)
    U1 = I_x1_y - R
    U2 = I_x2_y - R
    S = I_x12_y - R - U1 - U2

    return {
        "redundant": float(R),
        "unique_x1": float(U1),
        "unique_x2": float(U2),
        "synergy": float(S),
        "I_x1_y": float(I_x1_y),
        "I_x2_y": float(I_x2_y),
        "I_x1x2_y": float(I_x12_y),
        "base": float(base),
    }


def conditional_temporal_pid_min_information(
    x1_past: ArrayLike,
    x2_past: ArrayLike,
    y_present: ArrayLike,
    y_past: ArrayLike,
    base: float = 2.0,
    n_bins: int = 5,
    disc: str = "equal_freq",
    min_samples: int = 50,
    min_condition_samples: int = 12,
) -> Dict[str, float]:
    """Target-memory-conditioned temporal PID using stratified I_min.

    The four series are aligned on the regular temporal grid before missing rows
    are removed. Variables are discretized globally. PID is then evaluated
    separately within each state of Y(t-1) and averaged by that state's empirical
    probability. This yields a transparent conditional decomposition without
    constructing a sparse high-dimensional joint histogram.
    """
    x1 = _to_1d_array(x1_past)
    x2 = _to_1d_array(x2_past)
    yf = _to_1d_array(y_present)
    yp = _to_1d_array(y_past)
    if not (x1.size == x2.size == yf.size == yp.size):
        raise ValueError("Temporal PID inputs must have the same length.")

    mask = np.isfinite(x1) & np.isfinite(x2) & np.isfinite(yf) & np.isfinite(yp)
    x1 = x1[mask]
    x2 = x2[mask]
    yf = yf[mask]
    yp = yp[mask]
    n = int(yf.size)
    if n < int(min_samples):
        return {
            "redundant_bits": np.nan,
            "unique_driver_1_bits": np.nan,
            "unique_driver_2_bits": np.nan,
            "synergy_bits": np.nan,
            "itot_bits": np.nan,
            "redundant_frac": np.nan,
            "unique_driver_1_frac": np.nan,
            "unique_driver_2_frac": np.nan,
            "unique_total_frac": np.nan,
            "synergy_frac": np.nan,
            "n_samples": n,
            "condition_states_used": 0,
        }

    discretizer = discretize_equal_width if disc == "equal_width" else discretize_equal_frequency
    x1_lab = discretizer(x1, n_bins=n_bins)
    x2_lab = discretizer(x2, n_bins=n_bins)
    yf_lab = discretizer(yf, n_bins=n_bins)
    yp_lab = discretizer(yp, n_bins=n_bins)

    totals = {"R": 0.0, "U1": 0.0, "U2": 0.0, "S": 0.0, "Itot": 0.0}
    states_used = 0
    weight_used = 0.0
    for state in np.unique(yp_lab):
        state_mask = yp_lab == state
        ns = int(np.sum(state_mask))
        if ns < int(min_condition_samples):
            continue
        a = x1_lab[state_mask]
        b = x2_lab[state_mask]
        yy = yf_lab[state_mask]
        i1 = float(mutual_information_discrete(a, yy, base=base))
        i2 = float(mutual_information_discrete(b, yy, base=base))
        nb = int(b.max()) + 1 if b.size else 1
        joint = a.astype(int) * nb + b.astype(int)
        ij = float(mutual_information_discrete(joint, yy, base=base))
        r = min(i1, i2)
        u1 = max(0.0, i1 - r)
        u2 = max(0.0, i2 - r)
        s = max(0.0, ij - r - u1 - u2)
        w = ns / n
        totals["R"] += w * r
        totals["U1"] += w * u1
        totals["U2"] += w * u2
        totals["S"] += w * s
        totals["Itot"] += w * ij
        weight_used += w
        states_used += 1

    if states_used == 0 or weight_used <= 0:
        return {
            "redundant_bits": np.nan,
            "unique_driver_1_bits": np.nan,
            "unique_driver_2_bits": np.nan,
            "synergy_bits": np.nan,
            "itot_bits": np.nan,
            "redundant_frac": np.nan,
            "unique_driver_1_frac": np.nan,
            "unique_driver_2_frac": np.nan,
            "unique_total_frac": np.nan,
            "synergy_frac": np.nan,
            "n_samples": n,
            "condition_states_used": 0,
        }

    # Renormalize if sparse conditioning states were excluded.
    for key in totals:
        totals[key] /= weight_used
    itot = totals["Itot"]
    if not np.isfinite(itot) or itot <= 0:
        fracs = {"R": np.nan, "U1": np.nan, "U2": np.nan, "S": np.nan}
    else:
        fracs = {k: totals[k] / itot for k in ("R", "U1", "U2", "S")}

    return {
        "redundant_bits": float(totals["R"]),
        "unique_driver_1_bits": float(totals["U1"]),
        "unique_driver_2_bits": float(totals["U2"]),
        "synergy_bits": float(totals["S"]),
        "itot_bits": float(itot),
        "redundant_frac": float(fracs["R"]),
        "unique_driver_1_frac": float(fracs["U1"]),
        "unique_driver_2_frac": float(fracs["U2"]),
        "unique_total_frac": float(fracs["U1"] + fracs["U2"]),
        "synergy_frac": float(fracs["S"]),
        "n_samples": n,
        "condition_states_used": int(states_used),
    }


def moving_block_bootstrap_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Circular moving-block bootstrap indices preserving local temporal order."""
    n = int(n)
    block_size = max(1, min(int(block_size), n))
    if n <= 0:
        return np.array([], dtype=int)
    n_blocks = int(math.ceil(n / block_size))
    starts = rng.integers(0, n, size=n_blocks)
    idx = np.concatenate([(start + np.arange(block_size)) % n for start in starts])
    return idx[:n].astype(int)


# =============================================================================
# Transfer entropy (simple discrete, first-order)
# =============================================================================

def transfer_entropy_components(
    x: ArrayLike,
    y: ArrayLike,
    lag: int = 1,
    base: float = 2.0,
    n_bins: int = 10,
    disc: str = "equal_freq",
    min_samples: int = 2,
) -> Dict[str, float]:
    """Return raw TE, target conditional entropy, normalized TE, and sample size."""
    x = _to_1d_array(x)
    y = _to_1d_array(y)
    if x.size != y.size:
        raise ValueError("x and y must have the same length.")
    if lag < 1:
        raise ValueError("lag must be >= 1.")

    # At lag ℓ, evaluate X_{t-ℓ} -> Y_t while conditioning on Y_{t-1}.
    # This keeps the target-history term first-order and comparable across lags.
    x_past = x[:-lag]
    y_past = y[lag - 1:-1]
    y_future = y[lag:]
    x_past, y_past, y_future = _remove_nan(x_past, y_past, y_future)
    n = int(y_future.size)
    if n < int(min_samples):
        return {
            "te_bits": np.nan,
            "target_cond_entropy_bits": np.nan,
            "te_norm": np.nan,
            "n_samples": n,
        }

    if disc == "equal_width":
        x_p = discretize_equal_width(x_past, n_bins=n_bins)
        y_p = discretize_equal_width(y_past, n_bins=n_bins)
        y_f = discretize_equal_width(y_future, n_bins=n_bins)
    elif disc == "equal_freq":
        x_p = discretize_equal_frequency(x_past, n_bins=n_bins)
        y_p = discretize_equal_frequency(y_past, n_bins=n_bins)
        y_f = discretize_equal_frequency(y_future, n_bins=n_bins)
    else:
        raise ValueError("disc must be 'equal_width' or 'equal_freq'.")

    h_target_remaining = conditional_entropy_discrete(y_f, y_p, base=base)
    nx = int(x_p.max()) + 1 if x_p.size > 0 else 1
    yp_xp = y_p.astype(int) * nx + x_p.astype(int)
    h_with_source = conditional_entropy_discrete(y_f, yp_xp, base=base)
    te = float(max(0.0, h_target_remaining - h_with_source))
    te_norm = (
        float(te / h_target_remaining)
        if np.isfinite(h_target_remaining) and h_target_remaining > 0
        else np.nan
    )
    return {
        "te_bits": te,
        "target_cond_entropy_bits": float(h_target_remaining),
        "te_norm": te_norm,
        "n_samples": n,
    }


def transfer_entropy(
    x: ArrayLike,
    y: ArrayLike,
    lag: int = 1,
    base: float = 2.0,
    n_bins: int = 10,
    disc: str = "equal_freq",
) -> float:
    """Backward-compatible raw transfer entropy in bits."""
    return float(transfer_entropy_components(
        x, y, lag=lag, base=base, n_bins=n_bins, disc=disc,
    )["te_bits"])


def source_surrogate_p_value(
    stat_fn,
    x: np.ndarray,
    y: np.ndarray,
    observed_value: Optional[float] = None,
    n_perm: int = 200,
    surrogate_type: str = "circular_shift",
    block_size: int = 7,
    seed: Optional[int] = None,
) -> float:
    """Test a directional statistic by disrupting the source while preserving target history."""
    if not n_perm or int(n_perm) <= 0:
        return float("nan")
    x = _to_1d_array(x)
    y = _to_1d_array(y)
    if x.size != y.size or x.size < 3:
        return float("nan")
    rng = np.random.default_rng(seed)
    stat_obs = float(observed_value if observed_value is not None else stat_fn(x, y))
    if not np.isfinite(stat_obs):
        return float("nan")
    count = 0
    valid = 0
    for _ in range(int(n_perm)):
        x_surr = _surrogate_series(
            x, rng, surrogate_type=surrogate_type, block_size=block_size,
        )
        try:
            value = float(stat_fn(x_surr, y))
        except Exception:
            continue
        if not np.isfinite(value):
            continue
        valid += 1
        if value >= stat_obs:
            count += 1
    return float((count + 1) / (valid + 1)) if valid else float("nan")


def temporal_max_statistic_test(
    x: ArrayLike,
    y: ArrayLike,
    lags: Iterable[int],
    statistic: str,
    base: float = 2.0,
    n_bins: int = 10,
    disc: str = "equal_freq",
    min_samples: int = 20,
    n_surrogates: int = 200,
    surrogate_type: str = "circular_shift",
    block_size: int = 7,
    seed: Optional[int] = None,
) -> Dict[str, object]:
    """Maximum-statistic temporal surrogate test across a complete lag window.

    The null distribution stores the maximum statistic from every surrogate,
    which accounts for selecting the largest value after searching many lags.
    The source series is surrogated so the target's own temporal structure is
    preserved for directional TE testing.
    """
    x = _to_1d_array(x)
    y = _to_1d_array(y)
    lags = [int(v) for v in lags]
    if x.size != y.size:
        raise ValueError("x and y must have the same length.")
    if not lags:
        raise ValueError("At least one lag is required.")

    def evaluate(source: np.ndarray) -> pd.DataFrame:
        if statistic == "lagged_mi_norm":
            return lagged_mutual_information_table(
                source, y, lags=lags, base=base, method="hist",
                n_bins=n_bins, disc=disc, min_samples=min_samples,
            )[["lag", "mi_norm"]].rename(columns={"mi_norm": "value"})
        if statistic == "te_norm":
            max_lag = max(lags)
            table = transfer_entropy_by_lag(
                source, y, max_lag=max_lag, base=base, n_bins=n_bins,
                disc=disc, min_samples=min_samples,
            )
            table = table[table["lag"].isin(lags)]
            return table[["lag", "te_norm"]].rename(columns={"te_norm": "value"})
        raise ValueError("statistic must be 'lagged_mi_norm' or 'te_norm'.")

    observed_table = evaluate(x)
    valid_obs = observed_table.replace([np.inf, -np.inf], np.nan).dropna(subset=["value"])
    if valid_obs.empty:
        return {
            "observed_peak": np.nan, "peak_lag": np.nan, "p_value": np.nan,
            "null_max_95": np.nan, "n_surrogates_requested": int(n_surrogates),
            "n_surrogates_valid": 0, "surrogate_type": surrogate_type,
        }
    peak_row = valid_obs.loc[valid_obs["value"].idxmax()]
    observed_peak = float(peak_row["value"])
    peak_lag = int(peak_row["lag"])

    if not n_surrogates or int(n_surrogates) <= 0:
        return {
            "observed_peak": observed_peak, "peak_lag": peak_lag,
            "p_value": np.nan, "null_max_95": np.nan,
            "n_surrogates_requested": 0, "n_surrogates_valid": 0,
            "surrogate_type": surrogate_type,
        }

    rng = np.random.default_rng(seed)
    null_maxima = []
    for _ in range(int(n_surrogates)):
        x_surr = _surrogate_series(
            x, rng, surrogate_type=surrogate_type, block_size=block_size,
        )
        try:
            table = evaluate(x_surr)
            vals = pd.to_numeric(table["value"], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(vals).any():
                null_maxima.append(float(np.nanmax(vals)))
        except Exception:
            continue

    if not null_maxima:
        p_value = null95 = np.nan
    else:
        null_arr = np.asarray(null_maxima, dtype=float)
        p_value = float((1 + np.sum(null_arr >= observed_peak)) / (len(null_arr) + 1))
        null95 = float(np.nanquantile(null_arr, 0.95))
    return {
        "observed_peak": observed_peak,
        "peak_lag": peak_lag,
        "p_value": p_value,
        "null_max_95": null95,
        "n_surrogates_requested": int(n_surrogates),
        "n_surrogates_valid": int(len(null_maxima)),
        "surrogate_type": surrogate_type,
    }


# =============================================================================
# KDE / TIP backend (Goodwell & Kumar)
# =============================================================================

def calc_info_measures(pdf: np.ndarray) -> Dict[str, float]:
    """
    Compute information measures from a given PDF (1D, 2D, 3D)
    following Goodwell & Kumar (2017) TIP framework.
    """
    dim = len(pdf.shape)
    v = pdf.shape

    if np.sum(np.array(v) == 1) > 0:
        dim -= 1

    N = pdf.shape[0]
    info: Dict[str, float] = {}

    # 1D: entropy
    if dim == 1:
        Hvect = pdf * np.log2(1.0 / pdf)
        Hvect[np.isnan(Hvect)] = 0
        Hx = float(np.sum(Hvect))
        info["Hx"] = Hx

    # 2D: H(X), H(Y), H(X|Y), H(Y|X), I(X;Y)
    if dim == 2:
        H_xgy = 0.0
        H_ygx = 0.0

        m_i = np.sum(pdf, axis=1)
        Hivect = m_i * np.log2(1.0 / m_i)
        Hivect[np.isnan(Hivect)] = 0
        Hx = float(np.sum(Hivect))

        m_j = np.sum(pdf, axis=0)
        Hjvect = m_j * np.log2(1.0 / m_j)
        Hjvect[np.isnan(Hjvect)] = 0
        Hy = float(np.sum(Hjvect))

        for i in range(N):
            for j in range(N):
                m_ij = pdf[i, j]
                mj = m_j[j]
                mi = m_i[i]

                if m_ij > 0 and mi > 0:
                    H_ygx += m_ij * np.log2(mi / m_ij)
                if m_ij > 0 and mj > 0:
                    H_xgy += m_ij * np.log2(mj / m_ij)

        info["Hx1"] = Hx
        info["Hx2"] = Hy
        info["H_xgy"] = H_xgy
        info["H_ygx"] = H_ygx
        info["I"] = min(Hx - H_xgy, Hy - H_ygx)

    # 3D: TIP decomposition
    if dim == 3:
        I_x1y = 0.0
        I_x2y = 0.0
        I_x1x2 = 0.0
        T = 0.0

        m_jk = np.sum(pdf, axis=0)
        m_ij = np.sum(pdf, axis=2)
        m_ik = np.sum(pdf, axis=1)

        m_i = np.sum(m_ij, axis=1)
        m_j = np.sum(m_ij, axis=0)
        m_k = np.sum(m_jk, axis=0)

        Hivect = m_i * np.log2(1.0 / m_i)
        Hivect[np.isnan(Hivect)] = 0
        Hx1 = float(np.sum(Hivect))

        Hjvect = m_j * np.log2(1.0 / m_j)
        Hjvect[np.isnan(Hjvect)] = 0
        Hx2 = float(np.sum(Hjvect))

        Hkvect = m_k * np.log2(1.0 / m_k)
        Hkvect[np.isnan(Hkvect)] = 0
        Hy = float(np.sum(Hkvect))

        eps = np.finfo(float).eps

        for i in range(N):
            for j in range(N):
                for k in range(N):
                    m_ijk = pdf[i, j, k]

                    if (
                        m_ijk > eps
                        and m_ij[i, j] > eps
                        and m_jk[j, k] > eps
                        and m_j[j] > eps
                    ):
                        T_add = m_ijk * np.log2(
                            (m_ijk * m_j[j]) / (m_ij[i, j] * m_jk[j, k])
                        )
                        if T_add > 0:
                            T += T_add

                    if j == 1:
                        if m_ik[i, k] > eps and m_i[i] > eps and m_k[k] > eps:
                            I_tau_add = m_ik[i, k] * np.log2(
                                m_ik[i, k] / (m_i[i] * m_k[k])
                            )
                            if I_tau_add > 0:
                                I_x1y += I_tau_add

                    if i == 1:
                        if m_jk[j, k] > eps and m_j[j] > eps and m_k[k] > eps:
                            I_tau2_add = m_jk[j, k] * np.log2(
                                m_jk[j, k] / (m_j[j] * m_k[k])
                            )
                            if I_tau2_add > 0:
                                I_x2y += I_tau2_add

                    if k == 1:
                        if m_ij[i, j] > eps and m_i[i] > eps and m_j[j] > eps:
                            I_tau3_add = m_ij[i, j] * np.log2(
                                m_ij[i, j] / (m_i[i] * m_j[j])
                            )
                            if I_tau3_add > 0:
                                I_x1x2 += I_tau3_add

        dI = T - I_x1y
        I_tot = dI + I_x1y + I_x2y

        I_sourcenorm = I_x1x2 / min(Hx1, Hx2) if min(Hx1, Hx2) > 0 else 0.0
        if np.isnan(I_sourcenorm):
            I_sourcenorm = 0.0

        Rmax = min(I_x1y, I_x2y)
        Rmin = max(0.0, I_x1y + I_x2y - I_tot)
        dR = Rmax - Rmin
        R = Rmin + dR * I_sourcenorm

        U1 = I_x1y - R
        U2 = I_x2y - R
        S = I_tot - (U1 + U2 + R)

        info["Hx1"] = Hx1
        info["Hx2"] = Hx2
        info["Hy"] = Hy
        info["I_x1y"] = I_x1y
        info["I_x2y"] = I_x2y
        info["T"] = T
        info["dI"] = dI
        info["Itot"] = I_tot
        info["R"] = R
        info["S"] = S
        info["U1"] = U1
        info["U2"] = U2

    return info


def compute_pdfGUI(
    Data: np.ndarray,
    N: int,
    bin_scheme: str,
    Range: np.ndarray,
    method: str,
    zeffect,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 1D/2D/3D PDFs using KDE or histogram binning.
    Range must be shape (2, dim): Range[0,i]=min, Range[1,i]=max.
    """
    Data = np.asarray(Data, dtype=float)
    nTup, dim = Data.shape

    if dim == 1:
        pdf = np.zeros((N,))
    elif dim == 2:
        pdf = np.zeros((N, N))
    elif dim == 3:
        pdf = np.zeros((N, N, N))
    else:
        raise ValueError("compute_pdfGUI only supports dim = 1, 2, or 3.")

    Coords = np.zeros((dim, N))
    Edges = np.zeros((dim, N + 1))
    xo = np.zeros(dim)

    for i in range(dim):
        if bin_scheme == "local":
            lo = np.nanmin(Data[:, i])
            hi = np.nanmax(Data[:, i])
        elif bin_scheme == "global":
            lo = Range[0, i]
            hi = Range[1, i]
        else:
            raise ValueError("bin_scheme must be 'local' or 'global'.")

        if np.isclose(lo, hi):
            lo -= 1e-6
            hi += 1e-6

        Edges[i, :] = np.linspace(lo - 1e-8, hi, N + 1)
        xo[i] = lo
        Coords[i, :] = (Edges[i, :-1] + Edges[i, 1:]) / 2.0
        Coords[i, 0] = 0.0

    delta = Coords[:, -1] - Coords[:, -2]

    if method == "KDE":
        var = np.var(Data, axis=0)
        ptp = np.ptp(Data, axis=0)
        ptp[ptp == 0] = 1e-6

        h1D = 1.06 * (nTup ** (-1.0 / 5.0)) * var / ptp
        h2D = 1.77 * (nTup ** (-1.0 / 6.0)) * var / ptp
        h3D = 2.78 * (nTup ** (-1.0 / 7.0)) * var / ptp

        if dim == 1:
            h = h1D
        elif dim == 2:
            h = h2D
        else:
            h = h3D

        pdfcenter = np.zeros_like(pdf)

        for n in range(nTup):
            dat = Data[n, :]

            minind = np.maximum(
                np.floor((dat - xo - h) / delta).astype(int) - 1, 0
            )
            maxind = np.minimum(
                np.ceil((dat - xo + h) / delta).astype(int) + 1, N - 1
            )

            ICoords = Coords[0, minind[0]: maxind[0] + 1]
            if dim > 1:
                JCoords = Coords[1, minind[1]: maxind[1] + 1]
                if dim == 3:
                    KCoords = Coords[2, minind[2]: maxind[2] + 1]

            if dim == 1:
                pdfcenter[minind[0]: maxind[0] + 1] += np.exp(
                    -0.5 * ((ICoords - dat[0]) / h[0]) ** 2
                ) / (h[0] * np.sqrt(2 * np.pi))

            elif dim == 2:
                grid = np.array(np.meshgrid(ICoords, JCoords)).T.reshape(-1, 2)
                vals = np.exp(-0.5 * np.sum(((grid - dat) / h) ** 2, axis=1))
                vals = vals.reshape(
                    maxind[0] - minind[0] + 1,
                    maxind[1] - minind[1] + 1,
                )
                pdfcenter[
                    minind[0]: maxind[0] + 1,
                    minind[1]: maxind[1] + 1,
                ] += vals / (np.prod(h) * (2 * np.pi) ** (dim / 2.0))

            elif dim == 3:
                grid = np.array(np.meshgrid(ICoords, JCoords, KCoords)).T.reshape(
                    -1, 3
                )
                vals = np.exp(-0.5 * np.sum(((grid - dat) / h) ** 2, axis=1))
                vals = vals.reshape(
                    maxind[0] - minind[0] + 1,
                    maxind[1] - minind[1] + 1,
                    maxind[2] - minind[2] + 1,
                )
                pdfcenter[
                    minind[0]: maxind[0] + 1,
                    minind[1]: maxind[1] + 1,
                    minind[2]: maxind[2] + 1,
                ] += vals / (np.prod(h) * (2 * np.pi) ** (dim / 2.0))

        pdf = pdfcenter / np.sum(pdfcenter)

    else:
        BinData = np.zeros_like(Data, dtype=int)
        for i in range(dim):
            edges = Edges[i]
            b = np.digitize(Data[:, i], bins=edges) - 1
            b[b == N] = N - 1
            BinData[:, i] = b

        rng = [(0, N)] * dim
        pdf, _ = np.histogramdd(BinData, bins=N, range=rng, density=True)

    return pdf, Coords


def kde_mi_2d(
    x: np.ndarray,
    y: np.ndarray,
    N: int = 50,
    bin_scheme: str = "global",
    method: str = "KDE",
) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    L = min(len(x), len(y))
    x, y = x[:L], y[:L]

    Range = np.zeros((2, 2))
    Range[0, 0], Range[1, 0] = np.nanmin(x), np.nanmax(x)
    Range[0, 1], Range[1, 1] = np.nanmin(y), np.nanmax(y)

    Data = np.column_stack([x, y])
    zeffect = [0, 0]

    pdf, _ = compute_pdfGUI(Data, N, bin_scheme, Range, method, zeffect)
    info = calc_info_measures(pdf)
    return float(info.get("I", np.nan))


def kde_tip_pid_3d(
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    N: int = 30,
    bin_scheme: str = "global",
    method: str = "KDE",
) -> Dict[str, float]:
    x1 = np.asarray(x1, dtype=float).reshape(-1)
    x2 = np.asarray(x2, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    L = min(len(x1), len(x2), len(y))
    x1, x2, y = x1[:L], x2[:L], y[:L]

    Range = np.zeros((2, 3))
    Range[0, 0], Range[1, 0] = np.nanmin(x1), np.nanmax(x1)
    Range[0, 1], Range[1, 1] = np.nanmin(x2), np.nanmax(x2)
    Range[0, 2], Range[1, 2] = np.nanmin(y), np.nanmax(y)

    Data = np.column_stack([x1, x2, y])
    zeffect = [0, 0, 0]

    pdf, _ = compute_pdfGUI(Data, N, bin_scheme, Range, method, zeffect)
    info = calc_info_measures(pdf)
    return info


# =============================================================================
# Model-vs-observed functional performance diagnostics
# =============================================================================

def mutual_information_functional_performance(
    df: pd.DataFrame,
    observed_target: str,
    modeled_target: str,
    drivers: Iterable[str],
    base: float = 2.0,
    estimator: str = "hist",
    n_bins: int = 10,
    disc: str = "equal_freq",
    min_samples: int = 50,
) -> pd.DataFrame:
    """
    Compare observed and modeled driver--target dependencies using normalized MI.

    For each driver X, this computes:
      In_obs = I(X; Y_obs) / H(Y_obs)
      In_mod = I(X; Y_mod) / H(Y_mod)
      Delta I_n = In_mod - In_obs

    Delta I_n close to zero indicates that the modeled target reproduces the
    observed driver--flux dependence. Positive values indicate overestimation
    by the model; negative values indicate underestimation.
    """
    if observed_target not in df.columns:
        raise ValueError(f"Observed target '{observed_target}' was not found in the DataFrame.")
    if modeled_target not in df.columns:
        raise ValueError(f"Modeled target '{modeled_target}' was not found in the DataFrame.")

    drivers = [c for c in list(dict.fromkeys(drivers)) if c and c not in {observed_target, modeled_target}]
    drivers = [c for c in drivers if c in df.columns]
    if not drivers:
        raise ValueError("Select at least one driver variable.")

    yobs_all = pd.to_numeric(df[observed_target], errors="coerce").to_numpy(dtype=float)
    ymod_all = pd.to_numeric(df[modeled_target], errors="coerce").to_numpy(dtype=float)
    h_obs = entropy_continuous_discrete(yobs_all, base=base, n_bins=n_bins, disc=disc)
    h_mod = entropy_continuous_discrete(ymod_all, base=base, n_bins=n_bins, disc=disc)

    rows = []
    for driver in drivers:
        x_obs, y_obs = _pair_arrays_from_df(df, driver, observed_target)
        x_mod, y_mod = _pair_arrays_from_df(df, driver, modeled_target)
        if len(x_obs) < min_samples or len(x_mod) < min_samples:
            continue

        try:
            if estimator == "kde-tip":
                mi_obs = kde_mi_2d(x_obs, y_obs, N=n_bins, bin_scheme="global", method="KDE")
                mi_mod = kde_mi_2d(x_mod, y_mod, N=n_bins, bin_scheme="global", method="KDE")
                used_disc = "kde-tip"
            else:
                mi_obs = mutual_information(x_obs, y_obs, base=base, method="hist", n_bins=n_bins, disc=disc)
                mi_mod = mutual_information(x_mod, y_mod, base=base, method="hist", n_bins=n_bins, disc=disc)
                used_disc = disc
        except Exception:
            continue

        in_obs = float(mi_obs / h_obs) if h_obs and not np.isnan(h_obs) and h_obs > 0 else np.nan
        in_mod = float(mi_mod / h_mod) if h_mod and not np.isnan(h_mod) and h_mod > 0 else np.nan
        delta_in = in_mod - in_obs if not (np.isnan(in_obs) or np.isnan(in_mod)) else np.nan

        rows.append({
            "driver": driver,
            "observed_target": observed_target,
            "modeled_target": modeled_target,
            "mi_obs_bits": float(mi_obs),
            "mi_mod_bits": float(mi_mod),
            "in_obs": float(in_obs),
            "in_mod": float(in_mod),
            "delta_in": float(delta_in),
            "abs_delta_in": float(abs(delta_in)) if not np.isnan(delta_in) else np.nan,
            "n_obs": int(len(x_obs)),
            "n_mod": int(len(x_mod)),
            "estimator": estimator,
            "n_bins": int(n_bins),
            "disc": used_disc,
        })

    if not rows:
        raise ValueError("No valid model-vs-observed MI diagnostics could be computed.")

    out = pd.DataFrame(rows).sort_values("abs_delta_in", ascending=True).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def pairwise_pid_difference_matrix(
    df: pd.DataFrame,
    observed_target: str,
    modeled_target: str,
    drivers: Iterable[str],
    base: float = 2.0,
    estimator: str = "hist",
    n_bins: int = 10,
    disc: str = "equal_freq",
    min_samples: int = 50,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], List[str]]:
    """
    Compare modeled and observed pairwise PID fractions for one site.

    The returned matrices contain model-minus-observed differences:
      Delta S = S_model - S_obs
      Delta R = R_model - R_obs
      Delta U = U_model - U_obs

    Positive values mean the modeled target overestimates that information type.
    Negative values mean the modeled target underestimates that information type.
    """
    drivers = [c for c in list(dict.fromkeys(drivers)) if c and c not in {observed_target, modeled_target}]
    drivers = [c for c in drivers if c in df.columns]
    if len(drivers) < 2:
        raise ValueError("Select at least two driver variables for model-vs-observed PID.")

    obs_table, obs_mats, pid_vars = pairwise_pid_matrix(
        df, observed_target, drivers, base=base, estimator=estimator,
        n_bins=n_bins, disc=disc, min_samples=min_samples,
    )
    mod_table, mod_mats, _ = pairwise_pid_matrix(
        df, modeled_target, pid_vars, base=base, estimator=estimator,
        n_bins=n_bins, disc=disc, min_samples=min_samples,
    )

    key_cols = ["driver_1", "driver_2"]
    merged = obs_table.merge(
        mod_table,
        on=key_cols,
        suffixes=("_obs", "_mod"),
        how="inner",
    )
    if merged.empty:
        raise ValueError("No overlapping PID pairs could be compared.")

    merged["delta_s"] = merged["synergy_frac_mod"] - merged["synergy_frac_obs"]
    merged["delta_r"] = merged["redundant_frac_mod"] - merged["redundant_frac_obs"]
    merged["delta_u"] = merged["unique_total_frac_mod"] - merged["unique_total_frac_obs"]
    merged["delta_ipart"] = merged[["delta_s", "delta_r", "delta_u"]].abs().sum(axis=1)
    merged["pairwise_score_0_1"] = (2.0 - merged["delta_ipart"]) / 2.0
    merged["pairwise_score_0_1"] = merged["pairwise_score_0_1"].clip(lower=0.0, upper=1.0)
    merged["observed_target"] = observed_target
    merged["modeled_target"] = modeled_target

    p = len(pid_vars)
    mats = {
        "delta_s": np.full((p, p), np.nan, dtype=float),
        "delta_r": np.full((p, p), np.nan, dtype=float),
        "delta_u": np.full((p, p), np.nan, dtype=float),
        "delta_ipart": np.full((p, p), np.nan, dtype=float),
    }
    idx = {v: k for k, v in enumerate(pid_vars)}
    for row in merged.itertuples(index=False):
        i = idx[getattr(row, "driver_1")]
        j = idx[getattr(row, "driver_2")]
        mats["delta_s"][i, j] = getattr(row, "delta_s")
        mats["delta_r"][i, j] = getattr(row, "delta_r")
        mats["delta_u"][i, j] = getattr(row, "delta_u")
        mats["delta_ipart"][i, j] = getattr(row, "delta_ipart")

    return merged.reset_index(drop=True), mats, pid_vars


def functional_performance_summary(
    df: pd.DataFrame,
    observed_target: str,
    modeled_target: str,
    drivers: Iterable[str],
    base: float = 2.0,
    estimator: str = "hist",
    n_bins: int = 10,
    disc: str = "equal_freq",
    min_samples: int = 50,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Summarize single-site functional performance for one modeled target.

    Returns
    -------
    summary : DataFrame
        One-row table with individual-source and pairwise functional scores.
    mi_table : DataFrame
        Model-vs-observed normalized MI diagnostics.
    pid_table : DataFrame
        Model-vs-observed pairwise PID diagnostics.
    """
    mi_table = mutual_information_functional_performance(
        df, observed_target, modeled_target, drivers, base=base, estimator=estimator,
        n_bins=n_bins, disc=disc, min_samples=min_samples,
    )
    pid_table, _, _ = pairwise_pid_difference_matrix(
        df, observed_target, modeled_target, drivers, base=base, estimator=estimator,
        n_bins=n_bins, disc=disc, min_samples=min_samples,
    )

    mi_fidelity_score = float(np.nanmean(1.0 - mi_table["abs_delta_in"].to_numpy(dtype=float)))
    mi_fidelity_score = max(min(mi_fidelity_score, 1.0), -np.inf)
    pid_fidelity_score_0_1 = float(np.nanmean(pid_table["pairwise_score_0_1"].to_numpy(dtype=float)))
    pid_fidelity_raw = float(np.nanmean(2.0 - pid_table["delta_ipart"].to_numpy(dtype=float)))

    summary = pd.DataFrame([{
        "observed_target": observed_target,
        "modeled_target": modeled_target,
        "n_drivers_mi": int(mi_table["driver"].nunique()),
        "n_pairs_pid": int(len(pid_table)),
        "mi_fidelity_score": mi_fidelity_score,
        "pid_fidelity_raw": pid_fidelity_raw,
        "pid_fidelity_score_0_1": pid_fidelity_score_0_1,
        "mean_abs_delta_in": float(np.nanmean(mi_table["abs_delta_in"])),
        "mean_delta_ipart": float(np.nanmean(pid_table["delta_ipart"])),
        "estimator": estimator,
        "n_bins": int(n_bins),
        "disc": disc,
    }])
    return summary, mi_table, pid_table


# =============================================================================
# Pairwise PID matrix
# =============================================================================

def pairwise_pid_matrix(
    df: pd.DataFrame,
    target: str,
    drivers: Iterable[str],
    base: float = 2.0,
    estimator: str = "hist",
    n_bins: int = 10,
    disc: str = "equal_freq",
    min_samples: int = 50,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], List[str]]:
    """
    Compute pairwise PID components for all unordered driver pairs.

    For each pair (X1, X2) and target Y, the function estimates:
    - redundant information R
    - unique information from X1 and X2
    - total unique information U = U1 + U2
    - synergy S
    - total information Itot

    The returned matrices contain normalized fractions of Itot so that the
    heatmaps are comparable across driver pairs. The returned table keeps both
    raw bit values and normalized fractions.
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' was not found in the DataFrame.")

    drivers = [c for c in list(dict.fromkeys(drivers)) if c and c != target]
    drivers = [c for c in drivers if c in df.columns]

    if len(drivers) < 2:
        raise ValueError("Select at least two driver variables for the PID matrix.")

    y_all = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)
    p = len(drivers)

    # Lower-triangle matrices. NaNs keep the upper triangle blank in the plot.
    mats = {
        "redundant_frac": np.full((p, p), np.nan, dtype=float),
        "unique_total_frac": np.full((p, p), np.nan, dtype=float),
        "synergy_frac": np.full((p, p), np.nan, dtype=float),
        "itot_bits": np.full((p, p), np.nan, dtype=float),
    }

    rows = []

    for i in range(p):
        for j in range(i + 1, p):
            d1 = drivers[i]
            d2 = drivers[j]

            x1_all = pd.to_numeric(df[d1], errors="coerce").to_numpy(dtype=float)
            x2_all = pd.to_numeric(df[d2], errors="coerce").to_numpy(dtype=float)
            x1, x2, y = _align_dropna(x1_all, x2_all, y_all)

            if len(y) < min_samples:
                continue

            try:
                if estimator == "kde-tip":
                    info_tip = kde_tip_pid_3d(
                        x1,
                        x2,
                        y,
                        N=n_bins,
                        bin_scheme="global",
                        method="KDE",
                    )
                    R = float(info_tip.get("R", np.nan))
                    S = float(info_tip.get("S", np.nan))
                    U1 = float(info_tip.get("U1", np.nan))
                    U2 = float(info_tip.get("U2", np.nan))
                    Itot = float(info_tip.get("Itot", R + U1 + U2 + S))
                    used = "kde-tip"
                else:
                    res = pid_min_information(
                        x1,
                        x2,
                        y,
                        base=base,
                        method="hist",
                        n_bins=n_bins,
                        disc=disc,
                    )
                    R = float(res["redundant"])
                    S = float(res["synergy"])
                    U1 = float(res["unique_x1"])
                    U2 = float(res["unique_x2"])
                    Itot = float(res["I_x1x2_y"])
                    used = disc
            except Exception:
                continue

            U_total = U1 + U2
            if not np.isfinite(Itot) or np.isclose(Itot, 0.0):
                R_frac = U_frac = S_frac = np.nan
            else:
                R_frac = R / Itot
                U_frac = U_total / Itot
                S_frac = S / Itot

            # Plot in lower triangle at row j, column i.
            mats["redundant_frac"][j, i] = R_frac
            mats["unique_total_frac"][j, i] = U_frac
            mats["synergy_frac"][j, i] = S_frac
            mats["itot_bits"][j, i] = Itot

            rows.append({
                "target": target,
                "driver_1": d1,
                "driver_2": d2,
                "redundant_bits": R,
                "unique_driver_1_bits": U1,
                "unique_driver_2_bits": U2,
                "unique_total_bits": U_total,
                "synergy_bits": S,
                "itot_bits": Itot,
                "redundant_frac": R_frac,
                "unique_total_frac": U_frac,
                "synergy_frac": S_frac,
                "n_samples": int(len(y)),
                "estimator": estimator,
                "n_bins": int(n_bins),
                "disc": used,
            })

    if not rows:
        raise ValueError(
            "No valid pairwise PID values could be computed. Check the selected "
            "target, driver variables, and sample size."
        )

    return pd.DataFrame(rows), mats, drivers


# =============================================================================
# Permutation test
# =============================================================================

def permutation_test(
    stat_fn,
    x: np.ndarray,
    y: np.ndarray,
    n_perm: int = 200,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    x, y = _align_dropna(x, y)

    stat_obs = float(stat_fn(x, y))
    count = 0
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        s = float(stat_fn(x, y_perm))
        if s >= stat_obs:
            count += 1
    p = (count + 1) / (n_perm + 1)
    return stat_obs, p


# =============================================================================
# Tkinter GUI (flexible entry point)
# =============================================================================

def open_information_theory_toolbox(
    arg1,
    arg2=None,
    title: str = "Information Theory Toolbox",
    bridge_metadata: Optional[pd.DataFrame] = None,
    source_label: str = "",
    current_dataset: Optional[pd.DataFrame] = None,
) -> Optional[tk.Toplevel]:
    """
    Flexible public entry point.

    Pattern A (MeaningFlux main app, like ML toolbox):
        open_information_theory_toolbox(df, inputname_site)

    Pattern B (standalone / other GUI):
        open_information_theory_toolbox(parent_widget, df)

    Parameters
    ----------
    arg1 : pandas.DataFrame or tk.Misc
        Either the DataFrame (Pattern A) or the parent widget (Pattern B).
    arg2 : str or pandas.DataFrame or None
        Either the site name (Pattern A) or the DataFrame (Pattern B).
    title : str
        Base window title (site name is appended if available).
    bridge_metadata : pandas.DataFrame or None
        Optional one-row metadata table supplied by the ML toolbox.
    source_label : str
        Optional input-source label, such as ``Current ML results``.
    current_dataset : pandas.DataFrame or None
        Optional full MeaningFlux dataset supplied when opening from the ML toolbox.
        This lets users switch between the complete current dataset and the aligned
        ML--IT bridge without reopening the toolbox.
    """
    global _it_window

    # Detect calling pattern
    parent: Optional[tk.Misc] = None
    df = None
    site_name = ""

    if isinstance(arg1, pd.DataFrame):
        # Pattern A: (df, site_name)
        df = arg1
        site_name = str(arg2) if arg2 is not None else ""
        parent = tk._default_root  # may be None; Toplevel can still be created
    else:
        # Pattern B: (parent, df)
        parent = arg1
        df = arg2
        site_name = ""

    # Coerce df into DataFrame
    if df is None:
        messagebox.showwarning("Warning", "Load the data first.")
        return None

    # If df is a string, try CSV path; otherwise attempt DataFrame()
    if isinstance(df, str):
        if os.path.isfile(df):
            try:
                df = pd.read_csv(df)
            except Exception as e:
                messagebox.showerror(
                    "Error",
                    f"Could not read data from file:\n{df}\n\n{e}",
                )
                return None
        else:
            messagebox.showerror(
                "Error",
                "The Information Theory Toolbox received a string instead of a "
                "data table, and it is not a valid file path.\n\n"
                "Please open/load a dataset in MeaningFlux first."
            )
            return None

    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Data must be convertible to a pandas DataFrame.\n"
                f"Got type: {type(df)}\n\n{e}",
            )
            return None

    if df.empty:
        messagebox.showwarning("Warning", "DataFrame is empty.")
        return None

    # Keep separate input sources. The IT toolbox is standalone: it can analyze
    # the current MeaningFlux dataset, current ML results, saved bridges, or any
    # external table. When opened from ML, ``df`` is the aligned bridge table and
    # ``current_dataset`` may optionally provide the full current dataset.
    incoming_df = df.copy()
    opened_from_ml = bool(source_label == "Current ML results" or bridge_metadata is not None)
    source_tables: Dict[str, pd.DataFrame] = {}
    if isinstance(current_dataset, pd.DataFrame) and not current_dataset.empty:
        source_tables["Current dataset"] = current_dataset.copy()
    elif not opened_from_ml:
        source_tables["Current dataset"] = incoming_df.copy()
    if opened_from_ml:
        source_tables["Current ML results"] = incoming_df.copy()
    initial_source = "Current ML results" if opened_from_ml else "Current dataset"
    if initial_source not in source_tables:
        initial_source = next(iter(source_tables))
    df = source_tables[initial_source].copy()

    # Deduplicated column names for UI (same pattern as ML toolbox)
    all_cols = sorted(list(dict.fromkeys(df.columns)), key=lambda c: str(c).lower())
    if not all_cols:
        messagebox.showerror("Error", "No columns available in the DataFrame.")
        return None

    # Single-window behavior (like ML toolbox)
    if _it_window is not None and tk.Toplevel.winfo_exists(_it_window):
        try:
            _it_window.lift()
        except Exception:
            pass
        return _it_window

    # ------------------------------------------------------------------
    # GUI layout
    # ------------------------------------------------------------------
    if parent is not None and isinstance(parent, tk.Misc):
        win = tk.Toplevel(parent)
    else:
        win = tk.Toplevel()

    _it_window = win

    full_title = "MeaningFlux — Information Theory"
    if site_name:
        full_title += f" · {site_name}"
    win.title(full_title)
    win.geometry("1440x900")
    win.minsize(1040, 680)

    # A fixed toolbar remains visible regardless of the left-panel scroll
    # position. It provides direct access to Figure 6 and controls for
    # widening/narrowing the configuration column on smaller displays.
    top_toolbar = ttk.Frame(win, padding=(10, 7, 10, 5))
    top_toolbar.grid(row=0, column=0, columnspan=2, sticky="ew")
    top_toolbar.columnconfigure(5, weight=1)

    top_figure6_btn = ttk.Button(
        top_toolbar, text="Figure 6 setup…", width=20
    )
    top_figure6_btn.grid(row=0, column=0, sticky="w", padx=(0, 6))
    top_methods_btn = ttk.Button(
        top_toolbar, text="All IT methods…", width=18
    )
    top_methods_btn.grid(row=0, column=1, sticky="w", padx=(0, 12))

    ttk.Separator(top_toolbar, orient="vertical").grid(
        row=0, column=2, sticky="ns", padx=(0, 10)
    )
    ttk.Label(top_toolbar, text="Controls width").grid(
        row=0, column=3, sticky="w", padx=(0, 5)
    )

    left_panel_width = tk.IntVar(value=420)

    # Match the ML toolbox: configuration panel on the left, results on the right.
    left_shell = ttk.Frame(win, padding=(10, 10, 4, 10), width=left_panel_width.get())
    left_shell.grid(row=1, column=0, sticky="nsew")
    left_shell.grid_propagate(False)
    right = ttk.Frame(win, padding=(4, 10, 10, 10))
    right.grid(row=1, column=1, sticky="nsew")

    win.columnconfigure(0, weight=0, minsize=left_panel_width.get())
    win.columnconfigure(1, weight=1)
    win.rowconfigure(1, weight=1)

    def _set_left_panel_width(value: int) -> None:
        width = int(max(330, min(650, value)))
        left_panel_width.set(width)
        left_shell.configure(width=width)
        win.columnconfigure(0, minsize=width)
        try:
            win.update_idletasks()
        except Exception:
            pass

    ttk.Button(
        top_toolbar, text="−", width=3,
        command=lambda: _set_left_panel_width(left_panel_width.get() - 50),
    ).grid(row=0, column=4, sticky="w")
    ttk.Button(
        top_toolbar, text="+", width=3,
        command=lambda: _set_left_panel_width(left_panel_width.get() + 50),
    ).grid(row=0, column=5, sticky="w", padx=(3, 0))
    ttk.Button(
        top_toolbar, text="Reset",
        command=lambda: _set_left_panel_width(420),
    ).grid(row=0, column=6, sticky="e", padx=(10, 0))
    ttk.Label(
        top_toolbar,
        text="The center controls scroll vertically; run/export buttons stay fixed.",
        foreground="gray35",
    ).grid(row=0, column=7, sticky="e", padx=(12, 0))
    left_shell.rowconfigure(3, weight=1)
    left_shell.columnconfigure(0, weight=1)

    # Fixed quick actions remain visible even when the configuration panel scrolls.
    quick_actions = ttk.LabelFrame(left_shell, text="Figure 6 status")
    quick_actions.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 8))
    quick_actions.columnconfigure(0, weight=1)

    # Fixed run/export area. The scientific controls can scroll without hiding
    # the buttons needed to compute and export an analysis.
    action_card = ttk.LabelFrame(left_shell, text="Run and export")
    action_card.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 0))
    action_card.columnconfigure(0, weight=1)
    action_card.columnconfigure(1, weight=1)

    ttk.Label(left_shell, text="Information Theory", font=("", 14, "bold")).grid(
        row=0, column=0, sticky="w", pady=(0, 4)
    )
    input_card = ttk.LabelFrame(left_shell, text="Input source")
    input_card.grid(row=1, column=0, sticky="ew", pady=(0, 8))
    input_card.columnconfigure(0, weight=1)

    input_source_var = tk.StringVar(value=initial_source)
    input_source_cb = ttk.Combobox(
        input_card, textvariable=input_source_var, state="readonly",
        values=list(source_tables.keys()),
    )
    input_source_cb.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=(6, 3))

    bridge_status_var = tk.StringVar(value="")
    ttk.Label(input_card, textvariable=bridge_status_var, wraplength=350).grid(
        row=1, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 4)
    )
    load_bridge_btn = ttk.Button(input_card, text="Load saved ML–IT bridge")
    load_bridge_btn.grid(row=2, column=0, sticky="ew", padx=(8, 3), pady=(0, 6))
    load_external_btn = ttk.Button(input_card, text="Load external dataset")
    load_external_btn.grid(row=2, column=1, sticky="ew", padx=(3, 8), pady=(0, 6))
    input_card.columnconfigure(1, weight=1)

    # Scrollable configuration panel for smaller displays.
    left_canvas = tk.Canvas(
        left_shell, highlightthickness=1, highlightbackground="#c7c7c7",
        borderwidth=0, takefocus=True,
    )
    left_scroll = ttk.Scrollbar(left_shell, orient="vertical", command=left_canvas.yview)
    left_canvas.configure(yscrollcommand=left_scroll.set)
    left_canvas.grid(row=3, column=0, sticky="nsew")
    left_scroll.grid(row=3, column=1, sticky="ns")
    left = ttk.Frame(left_canvas, padding=(2, 0, 8, 8))
    left_window = left_canvas.create_window((0, 0), window=left, anchor="nw")

    def _sync_left_scrollregion(_event=None):
        left_canvas.configure(scrollregion=left_canvas.bbox("all"))

    def _sync_left_width(event):
        left_canvas.itemconfigure(left_window, width=event.width)

    left.bind("<Configure>", _sync_left_scrollregion)
    left_canvas.bind("<Configure>", _sync_left_width)
    left.columnconfigure(0, weight=1)

    def _left_mousewheel(event):
        delta_value = getattr(event, "delta", 0)
        if delta_value:
            direction = -1 if delta_value > 0 else 1
        else:
            direction = -1 if getattr(event, "num", None) == 4 else 1
        left_canvas.yview_scroll(direction, "units")
        return "break"

    def _bind_left_scroll(_event=None):
        left_canvas.bind_all("<MouseWheel>", _left_mousewheel)
        left_canvas.bind_all("<Button-4>", _left_mousewheel)
        left_canvas.bind_all("<Button-5>", _left_mousewheel)

    def _unbind_left_scroll(_event=None):
        left_canvas.unbind_all("<MouseWheel>")
        left_canvas.unbind_all("<Button-4>")
        left_canvas.unbind_all("<Button-5>")

    left_canvas.bind("<Enter>", _bind_left_scroll)
    left_canvas.bind("<Leave>", _unbind_left_scroll)
    left_canvas.bind("<Prior>", lambda _e: left_canvas.yview_scroll(-1, "pages"))
    left_canvas.bind("<Next>", lambda _e: left_canvas.yview_scroll(1, "pages"))
    left_canvas.bind("<Home>", lambda _e: left_canvas.yview_moveto(0.0))
    left_canvas.bind("<End>", lambda _e: left_canvas.yview_moveto(1.0))

    # --- Controls (left) ---
    scroll_hint = ttk.Frame(left)
    scroll_hint.grid(row=0, column=0, sticky="ew", pady=(0, 6))
    scroll_hint.columnconfigure(0, weight=1)
    ttk.Label(
        scroll_hint, text="Scrollable scientific controls",
        font=("", 9, "bold"), foreground="gray30",
    ).grid(row=0, column=0, sticky="w")
    ttk.Button(
        scroll_hint, text="Top", width=5,
        command=lambda: left_canvas.yview_moveto(0.0),
    ).grid(row=0, column=1, sticky="e")
    ttk.Button(
        scroll_hint, text="Bottom", width=7,
        command=lambda: left_canvas.yview_moveto(1.0),
    ).grid(row=0, column=2, sticky="e", padx=(4, 0))

    ttk.Label(left, text="Information analysis", font=("", 10, "bold")).grid(
        row=1, column=0, sticky="w"
    )
    measure_var = tk.StringVar(value="MI Driver Ranking")
    measure_cb = ttk.Combobox(
        left,
        textvariable=measure_var,
        state="readonly",
        values=[
            "Entropy H(X)",
            "Mutual Information I(X;Y)",
            "MI Driver Ranking",
            "Correlation vs MI Ranking",
            "Model-vs-Observed MI",
            "Model-vs-Observed PID Matrix",
            "Functional Performance Summary",
            "Single-Site IT Summary",
            "Figure 6 Temporal Summary",
            "Conditional MI I(X;Y|Z)",
            "Lagged MI I(X_t;Y_{t+lag})",
            "PID (X1,X2→Y)",
            "Pairwise PID Matrix",
            "Transfer Entropy TE(X→Y)",
            "Transfer Entropy vs Lag",
            "TE Network",
        ],
    )
    measure_cb.grid(row=2, column=0, sticky="ew", pady=(0, 8))

    x_label = ttk.Label(left, text="Source / Driver X")
    x_label.grid(row=3, column=0, sticky="w")
    preferred_x = next((c for c in all_cols if str(c).upper().startswith(("SW_IN", "TA", "VPD", "TS", "USTAR"))), all_cols[0])
    x_var = tk.StringVar(value=preferred_x)
    x_cb = ttk.Combobox(left, textvariable=x_var, values=all_cols, state="readonly")
    x_cb.grid(row=4, column=0, sticky="ew")

    y_label = ttk.Label(left, text="Target Y")
    y_label.grid(row=5, column=0, sticky="w")
    preferred_y = next((c for c in all_cols if str(c).upper() in {"FC", "NEE", "FCH4", "FN2O", "LE", "H"}), all_cols[1] if len(all_cols) > 1 else all_cols[0])
    y_var = tk.StringVar(value=preferred_y)
    y_cb = ttk.Combobox(left, textvariable=y_var, values=all_cols, state="readonly")
    y_cb.grid(row=6, column=0, sticky="ew")

    z_label = ttk.Label(left, text="Modeled target / Variable Z")
    z_label.grid(row=7, column=0, sticky="w")
    z_var = tk.StringVar(value="")
    z_cb = ttk.Combobox(left, textvariable=z_var, values=[""] + all_cols, state="readonly")
    z_cb.grid(row=8, column=0, sticky="ew", pady=(0, 4))

    model_hint_var = tk.StringVar(value="")
    def _autofill_model_column():
        candidates = detect_prediction_columns(all_cols, y_var.get())
        if candidates:
            z_var.set(candidates[0])
            model_hint_var.set("Detected model columns: " + ", ".join(map(str, candidates[:4])))
        else:
            model_hint_var.set("No obvious model/prediction column detected for the selected target.")

    detect_model_btn = ttk.Button(left, text="Detect modeled target", command=_autofill_model_column)
    detect_model_btn.grid(row=9, column=0, sticky="ew", pady=(0, 2))
    model_hint_label = ttk.Label(left, textvariable=model_hint_var, foreground="gray40", wraplength=300)
    model_hint_label.grid(row=10, column=0, sticky="w", pady=(0, 6))

    params = ttk.LabelFrame(left, text="Analysis settings")
    params.grid(row=11, column=0, sticky="ew", pady=(4, 8))

    ttk.Label(params, text="Estimator").grid(row=0, column=0, sticky="w")
    est_var = tk.StringVar(value="hist")
    est_cb = ttk.Combobox(
        params,
        textvariable=est_var,
        state="readonly",
        values=["hist", "kde-tip"],
    )
    est_cb.grid(row=0, column=1, sticky="ew", padx=(6, 0))

    ttk.Label(params, text="Bins / KDE N").grid(row=1, column=0, sticky="w")
    bins_var = tk.IntVar(value=5)
    ttk.Spinbox(
        params, from_=2, to=128, textvariable=bins_var, width=6
    ).grid(row=1, column=1, sticky="w", padx=(6, 0))

    ttk.Label(params, text="Discretizer").grid(row=2, column=0, sticky="w")
    disc_var = tk.StringVar(value="freq")
    ttk.Combobox(
        params,
        textvariable=disc_var,
        state="readonly",
        values=["freq", "width"],
    ).grid(row=2, column=1, sticky="ew", padx=(6, 0))

    ttk.Label(params, text="Maximum |lag|").grid(row=3, column=0, sticky="w")
    maxlag_var = tk.IntVar(value=14)
    ttk.Spinbox(
        params, from_=0, to=200, textvariable=maxlag_var, width=6
    ).grid(row=3, column=1, sticky="w", padx=(6, 0))

    ttk.Label(params, text="TE lag").grid(row=4, column=0, sticky="w")
    delay_var = tk.IntVar(value=1)
    ttk.Spinbox(
        params, from_=1, to=96, textvariable=delay_var, width=6
    ).grid(row=4, column=1, sticky="w", padx=(6, 0))

    ttk.Label(params, text="Permutations").grid(
        row=5, column=0, sticky="w"
    )
    perm_var = tk.IntVar(value=500)
    ttk.Spinbox(
        params, from_=0, to=5000, increment=50, textvariable=perm_var, width=6
    ).grid(row=5, column=1, sticky="w", padx=(6, 0))

    ttk.Label(params, text="Surrogate type").grid(row=6, column=0, sticky="w")
    surrogate_var = tk.StringVar(value="circular_shift")
    ttk.Combobox(params, textvariable=surrogate_var, state="readonly",
                 values=["random", "block_shuffle", "circular_shift"]).grid(row=6, column=1, sticky="ew", padx=(6, 0))

    ttk.Label(params, text="Timestamp column").grid(row=7, column=0, sticky="w")
    ts_candidates = _guess_timestamp_columns(all_cols)
    ts_filter_var = tk.StringVar(value=ts_candidates[0] if ts_candidates else all_cols[0])
    ts_filter_cb = ttk.Combobox(params, textvariable=ts_filter_var, state="readonly", values=ts_candidates)
    ts_filter_cb.grid(row=7, column=1, sticky="ew", padx=(6, 0))

    ttk.Label(params, text="Analysis window").grid(row=8, column=0, sticky="w")
    window_var = tk.StringVar(value="All data")
    ttk.Combobox(params, textvariable=window_var, state="readonly",
                 values=["All data", "Daytime", "Nighttime", "Growing season", "Non-growing season", "Selected months", "Custom date range"]).grid(row=8, column=1, sticky="ew", padx=(6, 0))

    ttk.Label(params, text="Radiation column").grid(row=9, column=0, sticky="w")
    rad_candidates = _guess_radiation_columns(all_cols)
    rad_var = tk.StringVar(value=rad_candidates[0] if rad_candidates else all_cols[0])
    rad_cb = ttk.Combobox(params, textvariable=rad_var, state="readonly", values=rad_candidates)
    rad_cb.grid(row=9, column=1, sticky="ew", padx=(6, 0))

    ttk.Label(params, text="Day threshold").grid(row=10, column=0, sticky="w")
    day_threshold_var = tk.StringVar(value="20")
    ttk.Entry(params, textvariable=day_threshold_var, width=8).grid(row=10, column=1, sticky="w", padx=(6, 0))

    ttk.Label(params, text="Months").grid(row=11, column=0, sticky="w")
    months_var = tk.StringVar(value="5,6,7,8,9")
    ttk.Entry(params, textvariable=months_var, width=16).grid(row=11, column=1, sticky="ew", padx=(6, 0))

    ttk.Label(params, text="Analysis start date/time").grid(row=12, column=0, sticky="w")
    start_date_var = tk.StringVar(value="")
    start_entry = ttk.Entry(params, textvariable=start_date_var, width=18)
    start_entry.grid(row=12, column=1, sticky="ew", padx=(6, 0))

    ttk.Label(params, text="Analysis end date/time").grid(row=13, column=0, sticky="w")
    end_date_var = tk.StringVar(value="")
    end_entry = ttk.Entry(params, textvariable=end_date_var, width=18)
    end_entry.grid(row=13, column=1, sticky="ew", padx=(6, 0))

    use_full_period_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(
        params,
        text="Use full available record",
        variable=use_full_period_var,
    ).grid(row=14, column=0, columnspan=2, sticky="w", pady=(2, 0))

    ttk.Label(params, text="Temporal grid").grid(row=15, column=0, sticky="w")
    temporal_freq_var = tk.StringVar(value="auto")
    ttk.Combobox(
        params,
        textvariable=temporal_freq_var,
        state="readonly",
        values=["auto", "D", "H", "30min"],
    ).grid(row=15, column=1, sticky="ew", padx=(6, 0))

    ttk.Label(params, text="Surrogate block size").grid(row=16, column=0, sticky="w")
    block_size_var = tk.IntVar(value=7)
    ttk.Spinbox(
        params, from_=2, to=1000, textvariable=block_size_var, width=6
    ).grid(row=16, column=1, sticky="w", padx=(6, 0))

    # Dedicated Figure 6 settings edited through the fixed configuration button.
    fig6_min_samples_var = tk.IntVar(value=50)
    fig6_alpha_var = tk.DoubleVar(value=0.05)
    fig6_bidirectional_var = tk.BooleanVar(value=True)
    fig6_supported_only_var = tk.BooleanVar(value=True)
    fig6_resolution_var = tk.StringVar(value="Daily")
    fig6_coverage_var = tk.DoubleVar(value=75.0)
    fig6_weekly_days_var = tk.IntVar(value=5)
    fig6_aggregation_rules: Dict[str, str] = {}
    fig6_status_var = tk.StringVar(
        value="Figure 6: Daily preparation, variable-specific aggregation, ±14 lags."
    )

    figure6_config_note = ttk.Label(
        quick_actions, textvariable=fig6_status_var, foreground="gray35",
        wraplength=350, justify="left",
    )
    figure6_config_note.grid(row=0, column=0, sticky="w", padx=6, pady=6)

    def _detect_available_period_from_active_table() -> Tuple[str, str]:
        def _fmt_ts(v):
            return pd.Timestamp(v).strftime("%Y-%m-%d %H:%M")
        tscol = ts_filter_var.get()
        if tscol not in df.columns:
            return "", ""
        t = pd.to_datetime(df[tscol], errors="coerce").dropna()
        if t.empty:
            return "", ""
        return _fmt_ts(t.min()), _fmt_ts(t.max())

    def _refresh_period_fields(*_):
        start_avail, end_avail = _detect_available_period_from_active_table()
        if start_avail and end_avail:
            if use_full_period_var.get() or not start_date_var.get().strip():
                start_date_var.set(start_avail)
            if use_full_period_var.get() or not end_date_var.get().strip():
                end_date_var.set(end_avail)

    def _toggle_full_period(*_):
        if use_full_period_var.get():
            _refresh_period_fields()
            start_entry.configure(state="disabled")
            end_entry.configure(state="disabled")
        else:
            start_entry.configure(state="normal")
            end_entry.configure(state="normal")

    try:
        ts_filter_var.trace_add("write", _refresh_period_fields)
        use_full_period_var.trace_add("write", _toggle_full_period)
    except Exception:
        pass
    _refresh_period_fields()
    _toggle_full_period()

    pool_label = ttk.Label(left, text="Predictor / variable pool", font=("", 10, "bold"))
    pool_label.grid(row=12, column=0, sticky="w")
    search_var = tk.StringVar()
    search_entry = ttk.Entry(left, textvariable=search_var)
    search_entry.grid(row=13, column=0, sticky="ew", pady=(0, 4))

    pool_hint_label = ttk.Label(
        left,
        text="Select drivers for rankings, PID, and networks.",
        foreground="gray40",
    )
    pool_hint_label.grid(row=14, column=0, sticky="w", pady=(0, 4))

    driver_list_frame = ttk.Frame(left)
    driver_list_frame.grid(row=15, column=0, sticky="nsew")
    driver_list_frame.rowconfigure(0, weight=1)
    driver_list_frame.columnconfigure(0, weight=1)
    left.rowconfigure(15, weight=1)

    net_list = tk.Listbox(
        driver_list_frame, selectmode=tk.EXTENDED, height=10,
        exportselection=False,
    )
    driver_y = ttk.Scrollbar(driver_list_frame, orient="vertical", command=net_list.yview)
    driver_x = ttk.Scrollbar(driver_list_frame, orient="horizontal", command=net_list.xview)
    # Configure directly after scrollbar creation because lambdas resolve at runtime.
    net_list.configure(yscrollcommand=driver_y.set, xscrollcommand=driver_x.set)
    net_list.grid(row=0, column=0, sticky="nsew")
    driver_y.grid(row=0, column=1, sticky="ns")
    driver_x.grid(row=1, column=0, sticky="ew")

    def _driver_mousewheel(event):
        delta = -1 if getattr(event, "delta", 0) > 0 else 1
        net_list.yview_scroll(delta, "units")
        return "break"

    net_list.bind("<MouseWheel>", _driver_mousewheel)

    for c in all_cols:
        net_list.insert(tk.END, c)
    common_driver_prefixes = ("SW_IN", "TA", "TS", "VPD", "WS", "USTAR", "SWC")
    for i, c in enumerate(all_cols):
        if str(c).upper().startswith(common_driver_prefixes):
            net_list.selection_set(i)

    def _filter_net(*_):
        q = search_var.get().strip().lower()
        net_list.delete(0, tk.END)
        for c in all_cols:
            if q in c.lower():
                net_list.insert(tk.END, c)

    search_var.trace_add("write", _filter_net)

    list_buttons = ttk.Frame(left)
    list_buttons.grid(row=16, column=0, sticky="ew", pady=(4, 0))
    list_buttons.columnconfigure(0, weight=1)
    list_buttons.columnconfigure(1, weight=1)

    def _select_all_visible():
        net_list.select_set(0, tk.END)

    def _clear_selection():
        net_list.selection_clear(0, tk.END)

    def _set_driver_selection(driver_names: Iterable[str]):
        """Select exact driver names in the variable-pool list."""
        wanted = {str(v) for v in driver_names if v}
        search_var.set("")
        net_list.delete(0, tk.END)
        for c in all_cols:
            net_list.insert(tk.END, c)
        net_list.selection_clear(0, tk.END)
        for i, c in enumerate(all_cols):
            if str(c) in wanted:
                net_list.selection_set(i)
                net_list.see(i)

    def _suggest_figure6_drivers(target_name: str = "") -> List[str]:
        """Suggest one harmonized driver from each manuscript driver class."""
        target_name = str(target_name or y_var.get())
        candidates = [
            c for c in all_cols
            if c != target_name and not _looks_like_output_column(c)
        ]

        def first_match(patterns: Iterable[str]) -> Optional[str]:
            # Prefer exact/prefix matches before looser substring matches.
            for c in candidates:
                token = str(c).upper()
                if any(token == p or token.startswith(p + "_") for p in patterns):
                    return c
            for c in candidates:
                token = str(c).upper()
                if any(p in token for p in patterns):
                    return c
            return None

        selected = [
            first_match(("SW_IN", "PPFD", "PAR", "NETRAD", "RNET", "RN", "RG")),
            first_match(("TA", "T_AIR", "AIRTEMP", "AIR_TEMP")),
            first_match(("VPD", "RH", "RELHUM", "RELATIVE_HUMIDITY")),
            first_match(("USTAR", "U_STAR", "FRICTION_VELOCITY")),
        ]
        return list(dict.fromkeys([c for c in selected if c]))

    def _open_analysis_catalog():
        dlg = tk.Toplevel(win)
        dlg.title("MeaningFlux Information Theory — analysis catalog")
        dlg.geometry("760x650")
        dlg.minsize(620, 520)
        dlg.transient(win)

        outer = ttk.Frame(dlg, padding=12)
        outer.pack(fill="both", expand=True)
        ttk.Label(
            outer,
            text="Information-theory analyses available in this toolbox",
            font=("", 13, "bold"),
        ).pack(anchor="w", pady=(0, 4))
        ttk.Label(
            outer,
            text=(
                "The temporal update did not replace the original toolbox. "
                "It adds timestamp-preserving lagged diagnostics and a Figure 6 workflow."
            ),
            foreground="gray35",
            wraplength=700,
            justify="left",
        ).pack(anchor="w", pady=(0, 10))

        nb = ttk.Notebook(outer)
        nb.pack(fill="both", expand=True)
        groups = {
            "Core dependence": [
                "Entropy H(X)", "Mutual Information I(X;Y)",
                "MI Driver Ranking", "Correlation vs MI Ranking",
                "Conditional MI I(X;Y|Z)",
            ],
            "Multivariate structure": [
                "PID (X1,X2→Y)", "Pairwise PID Matrix",
            ],
            "ML information fidelity": [
                "Model-vs-Observed MI", "Model-vs-Observed PID Matrix",
                "Functional Performance Summary",
            ],
            "Temporal directionality": [
                "Lagged MI I(X_t;Y_{t+lag})", "Transfer Entropy TE(X→Y)",
                "Transfer Entropy vs Lag", "TE Network",
                "Figure 6 Temporal Summary",
            ],
            "Integrated summaries": [
                "Single-Site IT Summary",
            ],
        }
        for group_name, methods in groups.items():
            tab = ttk.Frame(nb, padding=10)
            nb.add(tab, text=group_name)
            tab.rowconfigure(0, weight=1)
            tab.columnconfigure(0, weight=1)
            box = tk.Text(tab, wrap="word", background="#fbfbfb", relief="solid", borderwidth=1)
            box.grid(row=0, column=0, sticky="nsew")
            lines = []
            for method in methods:
                guide = METHOD_GUIDES.get(method, {})
                lines.append(method)
                lines.append("  " + guide.get("what", ""))
                lines.append("")
            box.insert("1.0", "\n".join(lines))
            box.configure(state=tk.DISABLED)

        ttk.Button(outer, text="Close", command=dlg.destroy).pack(anchor="e", pady=(10, 0))

    def _open_figure6_configuration():
        """Open a dedicated, manuscript-oriented Figure 6 configuration dialog."""
        measure_var.set("Figure 6 Temporal Summary")
        update_states()
        _show_method_guide()

        dlg = tk.Toplevel(win)
        dlg.title("Configure manuscript Figure 6")
        screen_w = max(800, int(dlg.winfo_screenwidth()))
        screen_h = max(650, int(dlg.winfo_screenheight()))
        dialog_w = min(940, max(760, screen_w - 120))
        dialog_h = min(820, max(650, screen_h - 140))
        dlg.geometry(f"{dialog_w}x{dialog_h}")
        dlg.minsize(700, 600)
        dlg.transient(win)
        dlg.grab_set()

        # Local copies prevent accidental changes until Apply is clicked.
        target_local = tk.StringVar(value=y_var.get())
        timestamp_local = tk.StringVar(value=ts_filter_var.get())
        grid_local = tk.StringVar(value=temporal_freq_var.get())
        resolution_local = tk.StringVar(value=fig6_resolution_var.get())
        coverage_local = tk.DoubleVar(value=float(fig6_coverage_var.get()))
        weekly_days_local = tk.IntVar(value=int(fig6_weekly_days_var.get()))
        bins_local = tk.IntVar(value=int(bins_var.get()))
        disc_local = tk.StringVar(value=disc_var.get())
        maxlag_local = tk.IntVar(value=int(maxlag_var.get()))
        perm_local = tk.IntVar(value=int(perm_var.get()))
        surrogate_local = tk.StringVar(value=surrogate_var.get())
        block_local = tk.IntVar(value=int(block_size_var.get()))
        min_samples_local = tk.IntVar(value=int(fig6_min_samples_var.get()))
        alpha_local = tk.DoubleVar(value=float(fig6_alpha_var.get()))
        bidirectional_local = tk.BooleanVar(value=bool(fig6_bidirectional_var.get()))
        supported_only_local = tk.BooleanVar(value=bool(fig6_supported_only_var.get()))
        full_period_local = tk.BooleanVar(value=bool(use_full_period_var.get()))
        start_local = tk.StringVar(value=start_date_var.get())
        end_local = tk.StringVar(value=end_date_var.get())
        window_local = tk.StringVar(value=window_var.get())
        validation_local = tk.StringVar(value="Configuration not yet validated.")
        aggregation_local_vars: Dict[str, tk.StringVar] = {}

        outer = ttk.Frame(dlg, padding=12)
        outer.pack(fill="both", expand=True)
        ttk.Label(
            outer,
            text="Figure 6 — temporal and directional information",
            font=("", 14, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "Configure one site at a time using the continuous observed dataset. "
                "The dialog applies all settings to the main toolbox and preserves the full IT feature set."
            ),
            foreground="gray35",
            wraplength=760,
            justify="left",
        ).pack(anchor="w", pady=(2, 10))

        nb = ttk.Notebook(outer)
        nb.pack(fill="both", expand=True)
        tab_data = ttk.Frame(nb, padding=10)
        tab_agg = ttk.Frame(nb, padding=10)
        tab_time = ttk.Frame(nb, padding=10)
        tab_sig = ttk.Frame(nb, padding=10)
        tab_output = ttk.Frame(nb, padding=10)
        nb.add(tab_data, text="1. Data and drivers")
        nb.add(tab_agg, text="2. Aggregation")
        nb.add(tab_time, text="3. Time design")
        nb.add(tab_sig, text="4. Estimation and support")
        nb.add(tab_output, text="5. Network and export")

        # Data tab
        tab_data.columnconfigure(1, weight=1)
        tab_data.rowconfigure(4, weight=1)
        ttk.Label(tab_data, text="Input source").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Label(tab_data, text=input_source_var.get(), foreground="gray30").grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Label(tab_data, text="Observed target").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Combobox(tab_data, textvariable=target_local, values=all_cols, state="readonly").grid(row=1, column=1, sticky="ew", padx=(8, 0))
        ttk.Label(tab_data, text="Timestamp column").grid(row=2, column=0, sticky="w", pady=3)
        ttk.Combobox(tab_data, textvariable=timestamp_local, values=_guess_timestamp_columns(all_cols), state="readonly").grid(row=2, column=1, sticky="ew", padx=(8, 0))
        ttk.Label(tab_data, text="Harmonized drivers").grid(row=3, column=0, columnspan=2, sticky="w", pady=(10, 3))

        driver_frame = ttk.Frame(tab_data)
        driver_frame.grid(row=4, column=0, columnspan=2, sticky="nsew")
        driver_frame.rowconfigure(0, weight=1)
        driver_frame.columnconfigure(0, weight=1)
        driver_box = tk.Listbox(driver_frame, selectmode=tk.EXTENDED, exportselection=False, height=12)
        driver_scroll = ttk.Scrollbar(driver_frame, orient="vertical", command=driver_box.yview)
        driver_box.configure(yscrollcommand=driver_scroll.set)
        driver_box.grid(row=0, column=0, sticky="nsew")
        driver_scroll.grid(row=0, column=1, sticky="ns")
        for c in all_cols:
            driver_box.insert(tk.END, c)
        current_selected = {net_list.get(i) for i in net_list.curselection()}
        for i, c in enumerate(all_cols):
            if c in current_selected:
                driver_box.selection_set(i)

        driver_buttons = ttk.Frame(tab_data)
        driver_buttons.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        driver_buttons.columnconfigure((0, 1, 2), weight=1)

        def select_suggested():
            suggestions = set(_suggest_figure6_drivers(target_local.get()))
            driver_box.selection_clear(0, tk.END)
            for i, c in enumerate(all_cols):
                if c in suggestions:
                    driver_box.selection_set(i)
                    driver_box.see(i)
            try:
                dlg.after_idle(refresh_aggregation_rows)
            except Exception:
                pass

        ttk.Button(driver_buttons, text="Auto-select manuscript drivers", command=select_suggested).grid(row=0, column=0, sticky="ew", padx=(0, 3))
        ttk.Button(driver_buttons, text="Select all", command=lambda: driver_box.selection_set(0, tk.END)).grid(row=0, column=1, sticky="ew", padx=3)
        ttk.Button(driver_buttons, text="Clear drivers", command=lambda: driver_box.selection_clear(0, tk.END)).grid(row=0, column=2, sticky="ew", padx=(3, 0))

        ttk.Label(
            tab_data,
            text="Recommended classes: radiation, air temperature, RH or VPD, and USTAR.",
            foreground="gray35",
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(6, 0))

        # Aggregation tab: selected variables receive editable physical rules.
        tab_agg.columnconfigure(0, weight=1)
        ttk.Label(
            tab_agg,
            text="Temporal resolution",
            font=("", 10, "bold"),
        ).grid(row=0, column=0, sticky="w")
        resolution_row = ttk.Frame(tab_agg)
        resolution_row.grid(row=1, column=0, sticky="ew", pady=(4, 8))
        for value in ("Native", "Daily", "Weekly"):
            ttk.Radiobutton(
                resolution_row, text=value, value=value, variable=resolution_local
            ).pack(side="left", padx=(0, 14))
        ttk.Label(
            tab_agg,
            text=(
                "Daily and Weekly are prepared internally from the current dataset. "
                "No extra CSV is required. Figure 6 manuscript default: Daily."
            ),
            foreground="gray35", wraplength=680, justify="left",
        ).grid(row=2, column=0, sticky="w", pady=(0, 10))

        coverage_frame = ttk.Frame(tab_agg)
        coverage_frame.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(coverage_frame, text="Minimum valid coverage per day (%)").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(coverage_frame, from_=1, to=100, increment=5, textvariable=coverage_local, width=8).grid(row=0, column=1, sticky="w", padx=(8, 18))
        ttk.Label(coverage_frame, text="Minimum valid days per week").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(coverage_frame, from_=1, to=7, textvariable=weekly_days_local, width=6).grid(row=0, column=3, sticky="w", padx=(8, 0))

        ttk.Label(tab_agg, text="Variable-specific aggregation", font=("", 10, "bold")).grid(row=4, column=0, sticky="w", pady=(4, 4))
        aggregation_canvas = tk.Canvas(tab_agg, highlightthickness=0, height=260)
        aggregation_scroll = ttk.Scrollbar(tab_agg, orient="vertical", command=aggregation_canvas.yview)
        aggregation_canvas.configure(yscrollcommand=aggregation_scroll.set)
        aggregation_canvas.grid(row=5, column=0, sticky="nsew")
        aggregation_scroll.grid(row=5, column=1, sticky="ns")
        tab_agg.rowconfigure(5, weight=1)
        aggregation_inner = ttk.Frame(aggregation_canvas)
        aggregation_window = aggregation_canvas.create_window((0, 0), window=aggregation_inner, anchor="nw")
        aggregation_inner.bind("<Configure>", lambda _e: aggregation_canvas.configure(scrollregion=aggregation_canvas.bbox("all")))
        aggregation_canvas.bind("<Configure>", lambda e: aggregation_canvas.itemconfigure(aggregation_window, width=e.width))

        aggregation_methods = ["mean", "sum", "median", "min", "max", "circular_mean", "first", "last"]

        def refresh_aggregation_rows(*_):
            selected_now = [driver_box.get(i) for i in driver_box.curselection()]
            variables_now = list(dict.fromkeys([target_local.get()] + selected_now))
            for child in aggregation_inner.winfo_children():
                child.destroy()
            ttk.Label(aggregation_inner, text="Variable", font=("", 9, "bold")).grid(row=0, column=0, sticky="w", padx=4, pady=3)
            ttk.Label(aggregation_inner, text="Aggregation", font=("", 9, "bold")).grid(row=0, column=1, sticky="w", padx=4, pady=3)
            ttk.Label(aggregation_inner, text="Suggested role", font=("", 9, "bold")).grid(row=0, column=2, sticky="w", padx=4, pady=3)
            for row_i, variable in enumerate(variables_now, start=1):
                default_method = fig6_aggregation_rules.get(variable, _infer_variable_aggregation(variable))
                var_obj = aggregation_local_vars.get(variable)
                if var_obj is None:
                    var_obj = tk.StringVar(value=default_method)
                    aggregation_local_vars[variable] = var_obj
                ttk.Label(aggregation_inner, text=variable).grid(row=row_i, column=0, sticky="w", padx=4, pady=2)
                ttk.Combobox(
                    aggregation_inner, textvariable=var_obj, values=aggregation_methods,
                    state="readonly", width=18,
                ).grid(row=row_i, column=1, sticky="w", padx=4, pady=2)
                role = "accumulation" if default_method == "sum" else "direction" if default_method == "circular_mean" else "rate/state"
                ttk.Label(aggregation_inner, text=role, foreground="gray35").grid(row=row_i, column=2, sticky="w", padx=4, pady=2)
            aggregation_inner.columnconfigure(0, weight=1)

        ttk.Button(tab_agg, text="Refresh selected variables", command=refresh_aggregation_rows).grid(row=6, column=0, sticky="ew", pady=(6, 3))
        ttk.Label(
            tab_agg,
            text=(
                "Typical defaults: FC, radiation, TA, RH/VPD, and USTAR = mean; "
                "precipitation/irrigation = sum; wind direction = circular mean. "
                "Change a rule when the column units require a different operation."
            ),
            foreground="gray35", wraplength=680, justify="left",
        ).grid(row=7, column=0, sticky="w", pady=(4, 0))

        driver_box.bind("<<ListboxSelect>>", refresh_aggregation_rows)
        try:
            target_local.trace_add("write", refresh_aggregation_rows)
        except Exception:
            pass
        refresh_aggregation_rows()

        # Time tab
        tab_time.columnconfigure(1, weight=1)
        ttk.Label(tab_time, text="Native grid (used only for Native resolution)").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Combobox(tab_time, textvariable=grid_local, state="readonly", values=["auto", "D", "H", "30min"]).grid(row=0, column=1, sticky="ew", padx=(8, 0))
        ttk.Label(tab_time, text="Maximum absolute lag").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Spinbox(tab_time, from_=1, to=200, textvariable=maxlag_local).grid(row=1, column=1, sticky="w", padx=(8, 0))
        ttk.Label(tab_time, text="Minimum aligned samples per lag").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Spinbox(tab_time, from_=20, to=100000, increment=10, textvariable=min_samples_local).grid(row=2, column=1, sticky="w", padx=(8, 0))
        ttk.Label(tab_time, text="Analysis window").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Combobox(
            tab_time, textvariable=window_local, state="readonly",
            values=["All data", "Daytime", "Nighttime", "Growing season", "Non-growing season", "Selected months", "Custom date range"],
        ).grid(row=3, column=1, sticky="ew", padx=(8, 0))
        ttk.Checkbutton(tab_time, text="Use full available record", variable=full_period_local).grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 4))
        ttk.Label(tab_time, text="Start date/time").grid(row=5, column=0, sticky="w", pady=4)
        ttk.Entry(tab_time, textvariable=start_local).grid(row=5, column=1, sticky="ew", padx=(8, 0))
        ttk.Label(tab_time, text="End date/time").grid(row=6, column=0, sticky="w", pady=4)
        ttk.Entry(tab_time, textvariable=end_local).grid(row=6, column=1, sticky="ew", padx=(8, 0))
        ttk.Label(
            tab_time,
            text=(
                "For the manuscript daily-FC analysis, use grid D and maximum lag 14. "
                "Missing dates remain explicit NaN rows before lag construction."
            ),
            foreground="gray35", wraplength=650, justify="left",
        ).grid(row=7, column=0, columnspan=2, sticky="w", pady=(10, 0))

        # Estimation/significance tab
        tab_sig.columnconfigure(1, weight=1)
        ttk.Label(tab_sig, text="Estimator").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Label(tab_sig, text="Histogram (required for current TE implementation)", foreground="gray30").grid(row=0, column=1, sticky="w", padx=(8, 0))
        ttk.Label(tab_sig, text="Bins").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Spinbox(tab_sig, from_=2, to=64, textvariable=bins_local).grid(row=1, column=1, sticky="w", padx=(8, 0))
        ttk.Label(tab_sig, text="Discretizer").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Combobox(tab_sig, textvariable=disc_local, state="readonly", values=["freq", "width"]).grid(row=2, column=1, sticky="ew", padx=(8, 0))
        ttk.Label(tab_sig, text="Temporal surrogates").grid(row=3, column=0, sticky="w", pady=4)
        ttk.Spinbox(tab_sig, from_=100, to=5000, increment=100, textvariable=perm_local).grid(row=3, column=1, sticky="w", padx=(8, 0))
        ttk.Label(tab_sig, text="Surrogate type").grid(row=4, column=0, sticky="w", pady=4)
        ttk.Combobox(tab_sig, textvariable=surrogate_local, state="readonly", values=["circular_shift", "block_shuffle", "random"]).grid(row=4, column=1, sticky="ew", padx=(8, 0))
        ttk.Label(tab_sig, text="Minimum circular shift / block size").grid(row=5, column=0, sticky="w", pady=4)
        ttk.Spinbox(tab_sig, from_=2, to=1000, textvariable=block_local).grid(row=5, column=1, sticky="w", padx=(8, 0))
        ttk.Label(tab_sig, text="Support threshold α").grid(row=6, column=0, sticky="w", pady=4)
        ttk.Spinbox(tab_sig, from_=0.001, to=0.20, increment=0.005, textvariable=alpha_local).grid(row=6, column=1, sticky="w", padx=(8, 0))
        ttk.Label(
            tab_sig,
            text=(
                "The maximum-statistic test compares the observed peak with the maximum "
                "from every temporal surrogate, accounting for the search across lags."
            ),
            foreground="gray35", wraplength=650, justify="left",
        ).grid(row=7, column=0, columnspan=2, sticky="w", pady=(10, 0))

        # Output tab
        ttk.Checkbutton(
            tab_output,
            text="Compute bidirectional driver ↔ target TE links for the site network",
            variable=bidirectional_local,
        ).pack(anchor="w", pady=4)
        ttk.Checkbutton(
            tab_output,
            text="Show only supported links in the network viewer",
            variable=supported_only_local,
        ).pack(anchor="w", pady=4)
        ttk.Separator(tab_output).pack(fill="x", pady=10)
        ttk.Label(
            tab_output,
            text=(
                "The publication export contains the single-site three-panel preview; a tidy peak-summary CSV "
                "with lagged_mi_peak, te_peak, and te_network_edge rows; complete lag-by-lag MI and TE tables; "
                "the prepared temporal series; full metadata; and network outputs."
            ),
            wraplength=680, justify="left",
        ).pack(anchor="w")
        ttk.Label(
            tab_output,
            text=(
                "Figures 3–5 continue to use common out-of-fold held-out observations. "
                "Figure 6 uses the timestamp-preserving continuous observed series."
            ),
            foreground="gray35", wraplength=680, justify="left",
        ).pack(anchor="w", pady=(10, 0))

        status_box = ttk.Label(
            outer, textvariable=validation_local, foreground="gray30",
            wraplength=760, justify="left", relief="groove", padding=6,
        )
        status_box.pack(fill="x", pady=(10, 6))

        def selected_dialog_drivers() -> List[str]:
            return [driver_box.get(i) for i in driver_box.curselection()]

        def use_manuscript_defaults():
            target_guess = next(
                (c for c in all_cols if str(c).upper() in {"FC", "FC_OBS", "NEE"}),
                next(
                    (c for c in all_cols if str(c).upper().startswith(("FC_", "NEE_"))),
                    target_local.get(),
                ),
            )
            target_local.set(target_guess)
            timestamp_candidates = _guess_timestamp_columns(all_cols)
            if timestamp_candidates:
                timestamp_local.set(timestamp_candidates[0])
            resolution_local.set("Daily")
            grid_local.set("auto")
            coverage_local.set(75.0)
            weekly_days_local.set(5)
            bins_local.set(5)
            disc_local.set("freq")
            maxlag_local.set(14)
            perm_local.set(500)
            surrogate_local.set("circular_shift")
            block_local.set(7)
            min_samples_local.set(50)
            alpha_local.set(0.05)
            bidirectional_local.set(True)
            supported_only_local.set(True)
            full_period_local.set(True)
            window_local.set("All data")
            select_suggested()
            refresh_aggregation_rows()
            for variable, var_obj in aggregation_local_vars.items():
                var_obj.set(_infer_variable_aggregation(variable))
            validation_local.set("Manuscript defaults loaded: Daily, 75% coverage, variable-specific aggregation. Click Validate.")

        def validate_configuration(show_dialog: bool = True) -> bool:
            errors = []
            warnings_local = []
            target = target_local.get()
            tscol = timestamp_local.get()
            drivers = selected_dialog_drivers()
            if input_source_var.get() in {"Current ML results", "Saved ML–IT bridge"}:
                errors.append("Use Current dataset or an external continuous observed dataset, not an ML–IT bridge.")
            if target not in df.columns:
                errors.append("Select a valid observed target column.")
            if tscol not in df.columns:
                errors.append("Select a valid timestamp column.")
            if not drivers:
                errors.append("Select at least one environmental driver.")
            if target in drivers:
                errors.append("The target cannot also be selected as a driver.")
            try:
                if not errors:
                    current_rules = {
                        variable: aggregation_local_vars.get(
                            variable, tk.StringVar(value=_infer_variable_aggregation(variable))
                        ).get()
                        for variable in list(dict.fromkeys([target] + drivers))
                    }
                    regular, meta = _prepare_figure6_temporal_frame(
                        df, tscol, drivers + [target],
                        resolution=resolution_local.get(),
                        native_frequency=grid_local.get(),
                        aggregation_rules=current_rules,
                        minimum_coverage_percent=float(coverage_local.get()),
                        weekly_minimum_valid_days=int(weekly_days_local.get()),
                    )
                    minimum = int(min_samples_local.get())
                    insufficient = []
                    for driver in drivers:
                        valid = regular[[driver, target]].notna().all(axis=1)
                        if int(valid.sum()) < minimum:
                            insufficient.append(f"{driver} ({int(valid.sum())})")
                    if insufficient:
                        errors.append(
                            "Fewer than the minimum valid samples for: " + ", ".join(insufficient)
                        )
                    suggested = _suggest_figure6_drivers(target)
                    if len(drivers) < 4:
                        warnings_local.append("The main manuscript design uses four harmonized driver classes.")
                    if not suggested:
                        warnings_local.append("Could not automatically identify the harmonized driver classes from column names.")
                    validation_local.set(
                        f"Ready: {len(regular):,} {meta['temporal_resolution'].lower()} steps; "
                        f"{meta['temporal_rows_inserted_as_missing']:,} fully missing steps retained; "
                        f"{len(drivers)} drivers; lag window ±{int(maxlag_local.get())} {meta['lag_unit']}; "
                        f"{int(perm_local.get())} {surrogate_local.get()} surrogates."
                    )
            except Exception as exc:
                errors.append(str(exc))

            if errors:
                validation_local.set("Not ready: " + " | ".join(errors))
                if show_dialog:
                    messagebox.showerror("Figure 6 configuration", "\n• " + "\n• ".join(errors), parent=dlg)
                return False
            if warnings_local and show_dialog:
                messagebox.showwarning("Figure 6 configuration", "\n• " + "\n• ".join(warnings_local), parent=dlg)
            elif show_dialog:
                messagebox.showinfo("Figure 6 configuration", validation_local.get(), parent=dlg)
            return True

        def apply_configuration(run_after: bool = False):
            nonlocal fig6_aggregation_rules
            if not validate_configuration(show_dialog=True):
                return
            drivers = selected_dialog_drivers()
            y_var.set(target_local.get())
            ts_filter_var.set(timestamp_local.get())
            temporal_freq_var.set(grid_local.get())
            fig6_resolution_var.set(resolution_local.get())
            fig6_coverage_var.set(float(coverage_local.get()))
            fig6_weekly_days_var.set(int(weekly_days_local.get()))
            fig6_aggregation_rules = {
                variable: var_obj.get()
                for variable, var_obj in aggregation_local_vars.items()
                if variable in list(dict.fromkeys([target_local.get()] + drivers))
            }
            est_var.set("hist")
            bins_var.set(int(bins_local.get()))
            disc_var.set(disc_local.get())
            maxlag_var.set(int(maxlag_local.get()))
            perm_var.set(int(perm_local.get()))
            surrogate_var.set(surrogate_local.get())
            block_size_var.set(int(block_local.get()))
            fig6_min_samples_var.set(int(min_samples_local.get()))
            fig6_alpha_var.set(float(alpha_local.get()))
            fig6_bidirectional_var.set(bool(bidirectional_local.get()))
            fig6_supported_only_var.set(bool(supported_only_local.get()))
            use_full_period_var.set(bool(full_period_local.get()))
            start_date_var.set(start_local.get())
            end_date_var.set(end_local.get())
            window_var.set(window_local.get())
            _set_driver_selection(drivers)
            fig6_status_var.set(
                f"Configured: {resolution_local.get()}; {float(coverage_local.get()):g}% daily coverage; "
                f"±{int(maxlag_local.get())} lags; {int(bins_local.get())} bins; "
                f"{int(perm_local.get())} {surrogate_local.get()} surrogates; "
                f"α={float(alpha_local.get()):.3g}; {len(drivers)} drivers"
            )
            dlg.grab_release()
            dlg.destroy()
            if run_after:
                win.after(80, run_analysis)

        ttk.Label(
            outer,
            text="These buttons remain fixed. Use the tabs above; scroll inside driver and aggregation lists when needed.",
            foreground="gray35", justify="left",
        ).pack(fill="x", pady=(2, 2))

        button_row = ttk.Frame(outer)
        button_row.pack(fill="x", pady=(4, 0), side="bottom")
        for col in range(5):
            button_row.columnconfigure(col, weight=1)
        ttk.Button(button_row, text="Manuscript defaults", command=use_manuscript_defaults).grid(row=0, column=0, sticky="ew", padx=(0, 3))
        ttk.Button(button_row, text="Validate", command=validate_configuration).grid(row=0, column=1, sticky="ew", padx=3)
        ttk.Button(button_row, text="Apply", command=lambda: apply_configuration(False)).grid(row=0, column=2, sticky="ew", padx=3)
        ttk.Button(button_row, text="Apply and run", command=lambda: apply_configuration(True)).grid(row=0, column=3, sticky="ew", padx=3)
        ttk.Button(button_row, text="Cancel", command=dlg.destroy).grid(row=0, column=4, sticky="ew", padx=(3, 0))

        if not current_selected:
            select_suggested()

    top_figure6_btn.configure(command=_open_figure6_configuration)
    top_methods_btn.configure(command=_open_analysis_catalog)

    def _looks_like_output_column(c):
        cl = str(c).lower()
        return (
            cl in {"timestamp", "time", "datetime", "date", "index", "split"}
            or cl.endswith("_obs")
            or "_pred_" in cl
            or "_resid_" in cl
            or cl.endswith("_pred")
            or cl.endswith("_resid")
        )

    def _refresh_column_widgets_after_import(preferred_target=""):
        # Update all comboboxes and the variable pool after loading an ML-to-IT bridge table.
        x_cb.configure(values=all_cols)
        y_cb.configure(values=all_cols)
        z_cb.configure(values=[""] + all_cols)

        obs_cols = [c for c in all_cols if str(c).lower().endswith("_obs")]
        if preferred_target and preferred_target in all_cols:
            y_var.set(preferred_target)
        elif obs_cols:
            y_var.set(obs_cols[0])
        elif all_cols:
            y_var.set(all_cols[0])

        pred_cols = detect_prediction_columns(all_cols, y_var.get())
        if pred_cols:
            z_var.set(pred_cols[0])
            model_hint_var.set("Detected model columns: " + ", ".join(map(str, pred_cols[:5])))
        else:
            z_var.set("")
            model_hint_var.set("No model prediction columns detected yet.")

        # X defaults to first likely driver.
        driver_candidates = [c for c in all_cols if not _looks_like_output_column(c)]
        if driver_candidates:
            x_var.set(driver_candidates[0])
        elif all_cols:
            x_var.set(all_cols[0])

        # Refresh timestamp/radiation selectors.
        ts_new = _guess_timestamp_columns(all_cols)
        ts_filter_cb.configure(values=ts_new)
        if ts_new:
            start_date_var.set("")
            end_date_var.set("")
            ts_filter_var.set(ts_new[0])
            _refresh_period_fields()
        rad_new = _guess_radiation_columns(all_cols)
        rad_cb.configure(values=rad_new)
        if rad_new:
            rad_var.set(rad_new[0])

        # Refresh variable pool and select likely drivers.
        search_var.set("")
        net_list.delete(0, tk.END)
        for c in all_cols:
            net_list.insert(tk.END, c)
        net_list.selection_clear(0, tk.END)
        driver_set = set(driver_candidates)
        for i, c in enumerate(all_cols):
            if c in driver_set:
                net_list.selection_set(i)

    def _status_for_source(name: str, table: pd.DataFrame) -> str:
        parts = [name, f"{len(table):,} rows", f"{len(table.columns)} variables"]
        if name == "Current ML results" and bridge_metadata is not None and isinstance(bridge_metadata, pd.DataFrame) and not bridge_metadata.empty:
            meta_row = bridge_metadata.iloc[0]
            model_name = str(meta_row.get("model", meta_row.get("model_name", ""))).strip()
            target_name = str(meta_row.get("target", meta_row.get("target_variable", ""))).strip()
            if target_name and target_name.lower() != "nan":
                parts.insert(1, target_name)
            if model_name and model_name.lower() != "nan":
                parts.insert(1, model_name)
        return " · ".join(parts)

    def _activate_source(source_name: str, preferred_target: str = ""):
        nonlocal df, all_cols
        if source_name not in source_tables:
            return
        df = source_tables[source_name].copy()
        all_cols = sorted(list(dict.fromkeys(df.columns)), key=lambda c: str(c).lower())
        _refresh_column_widgets_after_import(preferred_target=preferred_target)
        bridge_status_var.set(_status_for_source(source_name, df))
        input_source_var.set(source_name)
        # Model-fidelity methods remain available only when prediction columns exist.
        if not detect_prediction_columns(all_cols, y_var.get()) and measure_var.get().startswith("Model-vs-Observed"):
            measure_var.set("MI Driver Ranking")
            update_states()

    def _on_input_source_change(*_):
        _activate_source(input_source_var.get())

    def _load_ml_bridge_csv():
        path = filedialog.askopenfilename(
            title="Load saved ML–IT bridge",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            new_df = pd.read_csv(path)
            if new_df.empty:
                raise ValueError("The selected file is empty.")
            source_tables["Saved ML–IT bridge"] = new_df
            input_source_cb.configure(values=list(source_tables.keys()))
            preferred = next((c for c in new_df.columns if str(c).lower().endswith("_obs")), "")
            _activate_source("Saved ML–IT bridge", preferred_target=preferred)
        except Exception as e:
            messagebox.showerror("Could not load ML–IT bridge", str(e))

    def _load_external_dataset():
        path = filedialog.askopenfilename(
            title="Load external dataset",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            new_df = pd.read_csv(path)
            if new_df.empty:
                raise ValueError("The selected file is empty.")
            source_tables["External dataset"] = new_df
            input_source_cb.configure(values=list(source_tables.keys()))
            _activate_source("External dataset")
        except Exception as e:
            messagebox.showerror("Could not load external dataset", str(e))

    input_source_cb.bind("<<ComboboxSelected>>", _on_input_source_change)
    load_bridge_btn.configure(command=_load_ml_bridge_csv)
    load_external_btn.configure(command=_load_external_dataset)
    bridge_status_var.set(_status_for_source(initial_source, df))

    # Add the two list buttons plus a bridge import button.
    ttk.Button(
        list_buttons,
        text="Select visible",
        command=_select_all_visible,
    ).grid(row=0, column=0, sticky="ew", padx=(0, 3))

    ttk.Button(
        list_buttons,
        text="Clear selection",
        command=_clear_selection,
    ).grid(row=0, column=1, sticky="ew", padx=(3, 0))

    run_btn = ttk.Button(action_card, text="Compute selected analysis")
    run_btn.grid(row=0, column=0, columnspan=2, sticky="ew", padx=6, pady=(6, 3))

    progress_var = tk.DoubleVar(value=0.0)
    progress_status_var = tk.StringVar(value="Ready")
    progress = ttk.Progressbar(action_card, variable=progress_var, maximum=100, mode="determinate")
    progress.grid(row=1, column=0, columnspan=2, sticky="ew", padx=6, pady=(1, 0))
    ttk.Label(action_card, textvariable=progress_status_var, foreground="gray40").grid(
        row=2, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 4)
    )

    saveplot_btn = ttk.Button(action_card, text="Save figure", state=tk.NORMAL)
    saveplot_btn.grid(row=3, column=0, sticky="ew", padx=(6, 3), pady=2)
    save_results_btn = ttk.Button(action_card, text="Save results", state=tk.DISABLED)
    save_results_btn.grid(row=3, column=1, sticky="ew", padx=(3, 6), pady=2)
    save_package_btn = ttk.Button(action_card, text="Results + metadata", state=tk.DISABLED)
    save_package_btn.grid(row=4, column=0, sticky="ew", padx=(6, 3), pady=2)
    export_publication_btn = ttk.Button(action_card, text="Publication package", state=tk.DISABLED)
    export_publication_btn.grid(row=4, column=1, sticky="ew", padx=(3, 6), pady=2)
    save_prepared_btn = ttk.Button(action_card, text="Save prepared series", state=tk.DISABLED)
    save_prepared_btn.grid(row=5, column=0, sticky="ew", padx=(6, 3), pady=(2, 6))
    save_btn = ttk.Button(action_card, text="Save TE matrix", state=tk.DISABLED)
    save_btn.grid(row=5, column=1, sticky="ew", padx=(3, 6), pady=(2, 6))

    # --- Results area (right): notebook-style like the ML toolbox ---
    right.rowconfigure(0, weight=1)
    right.columnconfigure(0, weight=1)

    style = ttk.Style(win)
    try:
        style.configure("IT.TNotebook.Tab", padding=(8, 4))
    except Exception:
        pass
    results_nb = ttk.Notebook(right, style="IT.TNotebook")
    results_nb.grid(row=0, column=0, sticky="nsew")

    tab_workflow = ttk.Frame(results_nb, padding=10)
    tab_guide = ttk.Frame(results_nb, padding=10)
    tab_plot = ttk.Frame(results_nb, padding=10)
    tab_table = ttk.Frame(results_nb, padding=10)
    tab_network = ttk.Frame(results_nb, padding=10)
    tab_compare = ttk.Frame(results_nb, padding=10)

    results_nb.add(tab_workflow, text="Summary")
    results_nb.add(tab_guide, text="Interpret")
    results_nb.add(tab_plot, text="Plot")
    results_nb.add(tab_table, text="Data")
    results_nb.add(tab_network, text="Network")
    results_nb.add(tab_compare, text="Compare")

    # Workflow tab: plain-language end-to-end instructions.
    tab_workflow.rowconfigure(0, weight=1)
    tab_workflow.columnconfigure(0, weight=1)
    workflow_txt = tk.Text(
        tab_workflow,
        wrap="word",
        background="#fbfbfb",
        foreground="#222222",
        relief="solid",
        borderwidth=1,
    )
    workflow_txt.grid(row=0, column=0, sticky="nsew")
    workflow_txt.insert("1.0", _it_workflow_guide_text())
    workflow_txt.configure(state=tk.DISABLED)

    # Guide + text results
    tab_guide.columnconfigure(0, weight=1)
    tab_guide.rowconfigure(3, weight=1)

    ttk.Label(tab_guide, text="Method guide", font=("", 10, "bold")).grid(
        row=0, column=0, sticky="w"
    )
    guide_txt = tk.Text(
        tab_guide,
        height=10,
        wrap="word",
        background="#f7f7f7",
        foreground="#222222",
        relief="solid",
        borderwidth=1,
    )
    guide_txt.grid(row=1, column=0, sticky="ew", pady=(0, 8))

    ttk.Label(tab_guide, text="Numerical results / interpretation notes", font=("", 10, "bold")).grid(
        row=2, column=0, sticky="w"
    )
    txt = tk.Text(tab_guide, height=12, wrap="word")
    txt.grid(row=3, column=0, sticky="nsew")

    # Plot tab
    tab_plot.rowconfigure(0, weight=1)
    tab_plot.columnconfigure(0, weight=1)
    fig = Figure(figsize=(8.8, 5.2), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_title("Plot")
    canvas = FigureCanvasTkAgg(fig, master=tab_plot)
    canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    # Results table tab
    tab_table.rowconfigure(1, weight=1)
    tab_table.columnconfigure(0, weight=1)
    table_top = ttk.Frame(tab_table)
    table_top.grid(row=0, column=0, sticky="ew", pady=(0, 6))
    ttk.Label(
        table_top,
        text="The table mirrors the latest exportable results. Use it to inspect values before saving.",
        foreground="gray35",
    ).pack(side="left")

    table_frame = ttk.Frame(tab_table)
    table_frame.grid(row=1, column=0, sticky="nsew")
    table_frame.rowconfigure(0, weight=1)
    table_frame.columnconfigure(0, weight=1)

    results_tree = ttk.Treeview(table_frame, show="headings")
    results_tree.grid(row=0, column=0, sticky="nsew")
    results_y = ttk.Scrollbar(table_frame, orient="vertical", command=results_tree.yview)
    results_x = ttk.Scrollbar(table_frame, orient="horizontal", command=results_tree.xview)
    results_tree.configure(yscrollcommand=results_y.set, xscrollcommand=results_x.set)
    results_y.grid(row=0, column=1, sticky="ns")
    results_x.grid(row=1, column=0, sticky="ew")

    # Network Viewer tab: MeaningFlux-specific result viewer.
    tab_network.rowconfigure(2, weight=1)
    tab_network.columnconfigure(0, weight=1)

    network_toolbar = ttk.Frame(tab_network)
    network_toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
    ttk.Label(network_toolbar, text="Link threshold").pack(side="left")
    network_threshold_var = tk.StringVar(value="0")
    ttk.Entry(network_toolbar, textvariable=network_threshold_var, width=8).pack(side="left", padx=(6, 10))
    network_label_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(network_toolbar, text="Show labels", variable=network_label_var).pack(side="left", padx=(0, 10))
    ttk.Label(network_toolbar, text="View").pack(side="left")
    network_view_var = tk.StringVar(value="Circos")
    network_view_cb = ttk.Combobox(
        network_toolbar,
        textvariable=network_view_var,
        state="readonly",
        values=["Circos", "Directed network"],
        width=16,
    )
    network_view_cb.pack(side="left", padx=(6, 10))
    refresh_network_btn = ttk.Button(network_toolbar, text="Refresh viewer")
    refresh_network_btn.pack(side="left", padx=(0, 6))
    save_network_figure_btn = ttk.Button(network_toolbar, text="Save network figure", state=tk.DISABLED)
    save_network_figure_btn.pack(side="left", padx=(0, 6))
    save_network_edges_btn = ttk.Button(network_toolbar, text="Save edge table", state=tk.DISABLED)
    save_network_edges_btn.pack(side="left")

    ttk.Label(
        tab_network,
        text=(
            "MeaningFlux network viewer. Nodes are variables; "
            "directed edges summarize source-to-target information links from TE Network outputs. "
            "Use this as an exploratory result viewer, not as causal proof."
        ),
        foreground="gray35",
        wraplength=900,
    ).grid(row=1, column=0, sticky="w", pady=(0, 6))

    network_pane = ttk.Panedwindow(tab_network, orient="vertical")
    network_pane.grid(row=2, column=0, sticky="nsew")
    network_plot_frame = ttk.Frame(network_pane)
    network_table_frame = ttk.Frame(network_pane)
    network_pane.add(network_plot_frame, weight=2)
    network_pane.add(network_table_frame, weight=1)

    network_plot_frame.rowconfigure(0, weight=1)
    network_plot_frame.columnconfigure(0, weight=1)
    network_fig = Figure(figsize=(8.8, 4.8), dpi=100)
    network_ax = network_fig.add_subplot(111)
    network_canvas = FigureCanvasTkAgg(network_fig, master=network_plot_frame)
    network_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    network_table_frame.rowconfigure(0, weight=1)
    network_table_frame.columnconfigure(0, weight=1)
    network_cols = ("source", "target", "metric", "value", "lag", "p_value", "significant")
    network_tree = ttk.Treeview(network_table_frame, columns=network_cols, show="headings", height=8)
    for c in network_cols:
        network_tree.heading(c, text=c.upper())
        network_tree.column(c, width=120, anchor="w")
    network_tree.grid(row=0, column=0, sticky="nsew")
    network_y = ttk.Scrollbar(network_table_frame, orient="vertical", command=network_tree.yview)
    network_x = ttk.Scrollbar(network_table_frame, orient="horizontal", command=network_tree.xview)
    network_tree.configure(yscrollcommand=network_y.set, xscrollcommand=network_x.set)
    network_y.grid(row=0, column=1, sticky="ns")
    network_x.grid(row=1, column=0, sticky="ew")

    # Compare-runs tab
    tab_compare.rowconfigure(2, weight=1)
    tab_compare.columnconfigure(0, weight=1)

    compare_toolbar = ttk.Frame(tab_compare)
    compare_toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
    ttk.Label(compare_toolbar, text="Compare metric:").pack(side="left")
    compare_metric_var = tk.StringVar(value="key_value")
    compare_metric_cb = ttk.Combobox(
        compare_toolbar,
        textvariable=compare_metric_var,
        state="readonly",
        values=["key_value", "best_driver_value", "mean_value", "rows"],
        width=18,
    )
    compare_metric_cb.pack(side="left", padx=(6, 8))
    clear_compare_btn = ttk.Button(compare_toolbar, text="Clear run history")
    clear_compare_btn.pack(side="left", padx=(4, 0))

    ttk.Label(
        tab_compare,
        text=(
            "Each row is one analysis run. This helps compare targets, variables, "
            "estimators, bin choices, and model-vs-observed diagnostics within one site."
        ),
        foreground="gray35",
    ).grid(row=1, column=0, sticky="w", pady=(0, 6))

    compare_pane = ttk.Panedwindow(tab_compare, orient="vertical")
    compare_pane.grid(row=2, column=0, sticky="nsew")

    compare_table_frame = ttk.Frame(compare_pane)
    compare_plot_frame = ttk.Frame(compare_pane)
    compare_pane.add(compare_table_frame, weight=1)
    compare_pane.add(compare_plot_frame, weight=2)

    compare_table_frame.rowconfigure(0, weight=1)
    compare_table_frame.columnconfigure(0, weight=1)
    compare_cols = (
        "run", "method", "x", "y", "z", "drivers", "estimator", "bins",
        "disc", "key_metric", "key_value", "rows",
    )
    compare_tree = ttk.Treeview(compare_table_frame, columns=compare_cols, show="headings", height=8)
    for c in compare_cols:
        compare_tree.heading(c, text=c.upper())
        compare_tree.column(c, width=110 if c not in ("drivers", "method") else 210, anchor="w")
    compare_tree.grid(row=0, column=0, sticky="nsew")
    compare_y = ttk.Scrollbar(compare_table_frame, orient="vertical", command=compare_tree.yview)
    compare_x = ttk.Scrollbar(compare_table_frame, orient="horizontal", command=compare_tree.xview)
    compare_tree.configure(yscrollcommand=compare_y.set, xscrollcommand=compare_x.set)
    compare_y.grid(row=0, column=1, sticky="ns")
    compare_x.grid(row=1, column=0, sticky="ew")

    compare_plot_frame.rowconfigure(0, weight=1)
    compare_plot_frame.columnconfigure(0, weight=1)
    compare_fig = Figure(figsize=(8.8, 3.2), dpi=100)
    compare_ax = compare_fig.add_subplot(111)
    compare_canvas = FigureCanvasTkAgg(compare_fig, master=compare_plot_frame)
    compare_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    last_matrix: Optional[np.ndarray] = None
    last_vars: Optional[List[str]] = None
    last_results_df: Optional[pd.DataFrame] = None
    last_network_edges: Optional[pd.DataFrame] = None
    last_metadata: Dict[str, object] = {}
    last_prepared_temporal_df: Optional[pd.DataFrame] = None
    last_figure6_lagged_mi_full_df: Optional[pd.DataFrame] = None
    last_figure6_transfer_entropy_full_df: Optional[pd.DataFrame] = None
    last_figure6_temporal_pid_df: Optional[pd.DataFrame] = None
    last_figure6_temporal_pid_bootstrap_df: Optional[pd.DataFrame] = None
    df_runtime: Dict[str, pd.DataFrame] = {"active": df}
    analysis_runs: List[Dict[str, object]] = []
    cbar = None  # track current colorbar so we can remove it on the next run


    # --- inner helpers (using df directly, ML-style) ---

    def _show(msg: str):
        txt.delete("1.0", tk.END)
        txt.insert(tk.END, msg)

    def _show_method_guide(*_):
        guide_txt.configure(state=tk.NORMAL)
        guide_txt.delete("1.0", tk.END)
        guide_txt.insert(tk.END, format_method_guide(measure_var.get()))
        guide_txt.configure(state=tk.DISABLED)

    def _format_cell_value(value) -> str:
        """Compact formatting for Treeview cells."""
        try:
            if pd.isna(value):
                return ""
        except Exception:
            pass
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        text_value = str(value)
        return text_value if len(text_value) <= 80 else text_value[:77] + "..."

    def _network_edge_table_from_matrix(mat: np.ndarray, vars_: List[str], metric: str = "TE") -> pd.DataFrame:
        """Convert a source x target matrix into a tidy edge table.

        Rows are source-to-target links. The diagonal is omitted.
        """
        rows = []
        arr = np.asarray(mat, dtype=float)
        for i, src in enumerate(vars_):
            for j, tar in enumerate(vars_):
                if i == j:
                    continue
                try:
                    value = float(arr[i, j])
                except Exception:
                    value = np.nan
                if not np.isfinite(value):
                    continue
                rows.append({
                    "source": src,
                    "target": tar,
                    "metric": metric,
                    "value": value,
                    "lag": int(delay_var.get()) if metric.upper().startswith("TE") else np.nan,
                    "p_value": np.nan,
                    "significant": "",
                })
        return pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)

    def _refresh_network_table(edges: Optional[pd.DataFrame]):
        network_tree.delete(*network_tree.get_children())
        if edges is None or getattr(edges, "empty", True):
            return
        display = edges.copy().head(500)
        for _, row in display.iterrows():
            network_tree.insert("", "end", values=tuple(_format_cell_value(row.get(c, "")) for c in network_cols))

    def _draw_network_viewer(edges: Optional[pd.DataFrame], vars_: Optional[List[str]] = None):
        """Draw either a Circos-style information map or a directed node network."""
        network_ax.cla()
        if edges is None or getattr(edges, "empty", True):
            network_ax.text(
                0.5, 0.5,
                "No network-style result yet.\nRun TE Network, then open this tab.",
                ha="center", va="center", transform=network_ax.transAxes,
            )
            network_ax.set_axis_off()
            network_canvas.draw_idle()
            _refresh_network_table(edges)
            save_network_edges_btn.configure(state=tk.DISABLED)
            save_network_figure_btn.configure(state=tk.DISABLED)
            return

        try:
            threshold = float(network_threshold_var.get() or 0.0)
        except Exception:
            threshold = 0.0
        ed = edges.copy()
        ed["value"] = pd.to_numeric(ed["value"], errors="coerce")
        ed = ed.replace([np.inf, -np.inf], np.nan).dropna(subset=["value"])
        ed = ed[ed["value"].abs() >= threshold]
        if ed.empty:
            network_ax.text(
                0.5, 0.5,
                "No links pass the selected threshold.",
                ha="center", va="center", transform=network_ax.transAxes,
            )
            network_ax.set_axis_off()
            network_canvas.draw_idle()
            _refresh_network_table(ed)
            enabled = tk.NORMAL if edges is not None and not edges.empty else tk.DISABLED
            save_network_edges_btn.configure(state=enabled)
            save_network_figure_btn.configure(state=tk.DISABLED)
            return

        if vars_ is None or not vars_:
            vars_ = list(dict.fromkeys(ed["source"].astype(str).tolist() + ed["target"].astype(str).tolist()))
        else:
            vars_ = [str(v) for v in vars_]
        # Keep only variables represented in the current edge table.
        represented = set(ed["source"].astype(str)) | set(ed["target"].astype(str))
        vars_ = [v for v in vars_ if v in represented]
        n = len(vars_)
        if n < 2:
            network_ax.text(0.5, 0.5, "At least two connected variables are required.",
                            ha="center", va="center", transform=network_ax.transAxes)
            network_ax.set_axis_off()
            network_canvas.draw_idle()
            return

        theta = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, n, endpoint=False)
        pos = {v: (np.cos(theta[k]), np.sin(theta[k])) for k, v in enumerate(vars_)}

        strength = {v: 0.0 for v in vars_}
        for row in ed.itertuples(index=False):
            src = str(getattr(row, "source")); tar = str(getattr(row, "target"))
            val = abs(float(getattr(row, "value")))
            if src in strength: strength[src] += val
            if tar in strength: strength[tar] += val
        max_strength = max(strength.values()) if strength else 1.0
        if max_strength <= 0 or not np.isfinite(max_strength):
            max_strength = 1.0
        max_val = float(ed["value"].abs().max()) if not ed.empty else 1.0
        if max_val <= 0 or not np.isfinite(max_val):
            max_val = 1.0

        metric = str(ed["metric"].iloc[0]) if "metric" in ed.columns and not ed.empty else "information"
        view = network_view_var.get()

        if view == "Circos":
            from matplotlib.path import Path
            from matplotlib.patches import PathPatch, Wedge
            from matplotlib import cm

            cmap = cm.get_cmap("tab20", max(n, 2))
            node_colors = {v: cmap(k) for k, v in enumerate(vars_)}

            # Draw outer variable sectors. Sector span is fixed; arc width represents total link strength.
            sector_half = min(0.115, np.pi / max(3 * n, 1))
            for k, v in enumerate(vars_):
                ang_deg = np.degrees(theta[k])
                wedge = Wedge(
                    (0, 0), 1.08,
                    ang_deg - np.degrees(sector_half),
                    ang_deg + np.degrees(sector_half),
                    width=0.10,
                    facecolor=node_colors[v], edgecolor="white", linewidth=1.0,
                    alpha=0.95, zorder=3,
                )
                network_ax.add_patch(wedge)

            # Curved Circos links. Width maps to information strength; source color identifies origin.
            ed_draw = ed.sort_values("value", ascending=True)
            for row in ed_draw.itertuples(index=False):
                src = str(getattr(row, "source")); tar = str(getattr(row, "target"))
                if src not in pos or tar not in pos:
                    continue
                val = abs(float(getattr(row, "value")))
                x1, y1 = pos[src]; x2, y2 = pos[tar]
                p1 = (0.98 * x1, 0.98 * y1)
                p2 = (0.98 * x2, 0.98 * y2)
                verts = [p1, (0.20 * x1, 0.20 * y1), (0.20 * x2, 0.20 * y2), p2]
                path = Path(verts, [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
                width = 0.7 + 8.0 * (val / max_val)
                patch = PathPatch(
                    path, facecolor="none", edgecolor=node_colors[src],
                    linewidth=width, alpha=0.42, capstyle="round", zorder=1,
                )
                network_ax.add_patch(patch)

            if network_label_var.get():
                for k, v in enumerate(vars_):
                    a = theta[k]
                    x0, y0 = 1.22 * np.cos(a), 1.22 * np.sin(a)
                    rot = np.degrees(a)
                    if x0 < 0:
                        rot += 180
                        ha = "right"
                    else:
                        ha = "left"
                    network_ax.text(x0, y0, v, rotation=rot, rotation_mode="anchor",
                                    ha=ha, va="center", fontsize=8)

            network_ax.set_title(f"Circos information map — {metric} links (n={len(ed)})", pad=18)
            network_ax.set_xlim(-1.55, 1.55)
            network_ax.set_ylim(-1.55, 1.55)
        else:
            from matplotlib.patches import FancyArrowPatch
            for row in ed.itertuples(index=False):
                src = str(getattr(row, "source")); tar = str(getattr(row, "target"))
                if src not in pos or tar not in pos:
                    continue
                val = abs(float(getattr(row, "value")))
                lw = 0.5 + 4.0 * (val / max_val)
                alpha = 0.25 + 0.65 * (val / max_val)
                x1, y1 = pos[src]; x2, y2 = pos[tar]
                arrow = FancyArrowPatch(
                    (x1, y1), (x2, y2), arrowstyle="-|>",
                    mutation_scale=10 + 8 * (val / max_val), linewidth=lw,
                    alpha=alpha, connectionstyle="arc3,rad=0.12",
                    shrinkA=16, shrinkB=16,
                )
                network_ax.add_patch(arrow)
            xs = [pos[v][0] for v in vars_]
            ys = [pos[v][1] for v in vars_]
            sizes = [250 + 1450 * (strength.get(v, 0.0) / max_strength) for v in vars_]
            network_ax.scatter(xs, ys, s=sizes, zorder=3, edgecolors="black", linewidths=0.8)
            if network_label_var.get():
                for v in vars_:
                    x0, y0 = pos[v]
                    network_ax.text(1.12 * x0, 1.12 * y0, v, ha="center", va="center", fontsize=9)
            network_ax.set_title(f"Directed information network — {metric} links (n={len(ed)})")
            network_ax.set_xlim(-1.35, 1.35)
            network_ax.set_ylim(-1.35, 1.35)

        network_ax.set_aspect("equal")
        network_ax.set_axis_off()
        network_fig.tight_layout()
        network_canvas.draw_idle()
        _refresh_network_table(ed)
        save_network_edges_btn.configure(state=tk.NORMAL)
        save_network_figure_btn.configure(state=tk.NORMAL)

    def _refresh_network_viewer():
        _draw_network_viewer(last_network_edges, last_vars)

    def _save_network_figure():
        fp = filedialog.asksaveasfilename(
            title="Save network/Circos figure",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("PDF document", "*.pdf"), ("All files", "*.*")],
        )
        if not fp:
            return
        try:
            network_fig.savefig(fp, dpi=300, bbox_inches="tight")
            messagebox.showinfo("Saved", f"Saved network figure to:\n{fp}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

    def _save_network_edges():
        if last_network_edges is None or getattr(last_network_edges, "empty", True):
            messagebox.showinfo("Nothing to save", "Run TE Network first to create a network edge table.")
            return
        fp = filedialog.asksaveasfilename(
            title="Save network edge table",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if not fp:
            return
        try:
            last_network_edges.to_csv(fp, index=False)
            messagebox.showinfo("Saved", f"Saved edge table to:\n{fp}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

    def _result_df_for_display(df_result: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Prepare the latest results table for display in the Table tab."""
        if df_result is None or getattr(df_result, "empty", True):
            return None
        out = df_result.copy()

        # Matrix-style results such as TE Network have meaningful index labels.
        # Keep them visible in the table instead of losing them.
        if not isinstance(out.index, pd.RangeIndex):
            out = out.reset_index().rename(columns={"index": "row"})

        # Avoid rendering very large tables in the GUI while preserving full CSV export.
        max_rows = 500
        if len(out) > max_rows:
            out = out.iloc[:max_rows].copy()
        return out

    def _refresh_results_table():
        """Update the Results Table tab from last_results_df."""
        display_df = _result_df_for_display(last_results_df)
        results_tree.delete(*results_tree.get_children())

        if display_df is None:
            results_tree.configure(columns=[])
            return

        cols_display = [str(c) for c in display_df.columns]
        results_tree.configure(columns=cols_display, show="headings")
        for c in cols_display:
            results_tree.heading(c, text=c)
            width = max(90, min(220, 10 * len(c) + 30))
            results_tree.column(c, width=width, anchor="w")

        for _, row in display_df.iterrows():
            values = [_format_cell_value(row[c]) for c in display_df.columns]
            results_tree.insert("", "end", values=values)

    def _extract_key_summary(method_name: str, df_result: Optional[pd.DataFrame]) -> Tuple[str, float, str, float, float]:
        """
        Return key_metric, key_value, best_driver, best_driver_value, mean_value
        for run-history comparisons.
        """
        if df_result is None or getattr(df_result, "empty", True):
            return "", float("nan"), "", float("nan"), float("nan")

        table = df_result.copy()
        numeric_cols = table.select_dtypes(include=[np.number]).columns.tolist()

        priority_by_method = {
            "Entropy H(X)": ["entropy_bits"],
            "Mutual Information I(X;Y)": ["mi_bits", "mi_norm"],
            "MI Driver Ranking": ["mi_norm", "mi_bits"],
            "Correlation vs MI Ranking": ["mi_norm", "pearson_r2", "mi_bits"],
            "Model-vs-Observed MI": ["abs_delta_in", "delta_in"],
            "Model-vs-Observed PID Matrix": ["delta_ipart", "pairwise_score_0_1"],
            "Functional Performance Summary": ["mi_fidelity_score", "pid_fidelity_score_0_1"],
            "Single-Site IT Summary": ["mi_norm", "te_bits", "synergy_frac"],
            "Figure 6 Temporal Summary": ["peak_te_norm", "peak_mi_norm", "value"],
            "Conditional MI I(X;Y|Z)": ["cmi_bits"],
            "Lagged MI I(X_t;Y_{t+lag})": ["mi_bits"],
            "PID (X1,X2→Y)": ["redundant", "unique_x1", "unique_x2", "synergy"],
            "Pairwise PID Matrix": ["synergy_frac", "redundant_frac", "unique_total_frac"],
            "Transfer Entropy TE(X→Y)": ["te_bits"],
            "Transfer Entropy vs Lag": ["te_bits"],
            "TE Network": numeric_cols,
        }

        candidates = priority_by_method.get(method_name, []) + numeric_cols
        key_col = next((c for c in candidates if c in table.columns), None)
        if key_col is None:
            return "rows", float(len(table)), "", float("nan"), float(len(table))

        vals = pd.to_numeric(table[key_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if vals.dropna().empty:
            return key_col, float("nan"), "", float("nan"), float("nan")

        # For error/difference columns, lower absolute value is better; for information quantities, max is more useful.
        use_min_abs = key_col.startswith("abs_") or "delta" in key_col or key_col in {"mean_abs_delta_in", "delta_ipart"}
        if use_min_abs:
            idx = vals.abs().idxmin()
        else:
            idx = vals.idxmax()

        key_value = float(vals.loc[idx])
        mean_value = float(vals.mean())

        best_label = ""
        for label_col in ["driver", "source", "driver_1", "component", "lag", "row"]:
            if label_col in table.columns:
                best_label = str(table.loc[idx, label_col])
                if label_col == "driver_1" and "driver_2" in table.columns:
                    best_label = f"{table.loc[idx, 'driver_1']} + {table.loc[idx, 'driver_2']}"
                break

        return key_col, key_value, best_label, key_value, mean_value

    def _selected_driver_names(max_names: int = 8) -> str:
        selected = [net_list.get(i) for i in net_list.curselection()]
        if not selected:
            return "auto/all"
        if len(selected) <= max_names:
            return ", ".join(selected)
        return ", ".join(selected[:max_names]) + f", ... (+{len(selected)-max_names})"

    def _add_run_record(method_name: str):
        """Append the latest analysis to Compare Runs."""
        key_metric, key_value, best_driver, best_driver_value, mean_value = _extract_key_summary(method_name, last_results_df)
        rec = {
            "run": len(analysis_runs) + 1,
            "method": method_name,
            "x": x_var.get(),
            "y": y_var.get(),
            "z": z_var.get(),
            "drivers": _selected_driver_names(),
            "estimator": est_var.get(),
            "bins": int(bins_var.get()),
            "disc": disc_var.get(),
            "key_metric": key_metric,
            "key_value": key_value,
            "best_driver": best_driver,
            "best_driver_value": best_driver_value,
            "mean_value": mean_value,
            "rows": int(len(last_results_df)) if last_results_df is not None else 0,
        }
        analysis_runs.append(rec)
        _refresh_compare_runs()

    def _refresh_compare_runs(*_):
        compare_tree.delete(*compare_tree.get_children())
        for rec in analysis_runs:
            compare_tree.insert(
                "", "end",
                values=(
                    rec["run"], rec["method"], rec["x"], rec["y"], rec["z"],
                    rec["drivers"], rec["estimator"], rec["bins"], rec["disc"],
                    rec["key_metric"], _format_cell_value(rec["key_value"]), rec["rows"],
                ),
            )

        compare_ax.cla()
        if not analysis_runs:
            compare_ax.text(
                0.5, 0.5,
                "No analysis runs yet.\nRun a method to populate this comparison tab.",
                ha="center", va="center", transform=compare_ax.transAxes,
            )
            compare_canvas.draw_idle()
            return

        metric = compare_metric_var.get()
        labels = [f"{rec['run']}: {rec['method'].split()[0]}" for rec in analysis_runs]
        vals = []
        for rec in analysis_runs:
            value = rec.get(metric, np.nan)
            try:
                value = float(value)
            except Exception:
                value = np.nan
            vals.append(value)

        xpos = np.arange(len(vals))
        compare_ax.bar(xpos, vals)
        compare_ax.set_xticks(xpos)
        compare_ax.set_xticklabels(labels, rotation=30, ha="right")
        compare_ax.set_ylabel(metric)
        compare_ax.set_title(f"Compare analysis runs by {metric}")
        compare_ax.grid(True, axis="y", alpha=0.25)
        compare_fig.tight_layout()
        compare_canvas.draw_idle()

    def _clear_run_history():
        analysis_runs.clear()
        _refresh_compare_runs()

    def _interpret_latest_result(method_name: str) -> str:
        """Plain-language interpretation of the latest result table."""
        table = last_results_df
        if table is None or getattr(table, "empty", True):
            return (
                "Interpretation:\n"
                "This analysis produced a plot or scalar result but no exportable table. Use the method guide above to interpret the numerical value and plot.\n"
            )

        try:
            dfres = table.copy()
            lines = ["", "=" * 72, "Plain-language interpretation", "=" * 72]

            if method_name in {"MI Driver Ranking", "Correlation vs MI Ranking"}:
                val_col = "mi_norm" if "mi_norm" in dfres.columns else "mi_bits"
                if "driver" in dfres.columns and val_col in dfres.columns:
                    top = dfres.sort_values(val_col, ascending=False).iloc[0]
                    lines.append(f"Top driver: {top['driver']} ({val_col} = {float(top[val_col]):.4g}).")
                    lines.append("Drivers near the top share more information with the selected target. This indicates stronger dependence, not proof of causality.")
                if method_name == "Correlation vs MI Ranking" and {"driver", "pearson_r2", "mi_norm"}.issubset(dfres.columns):
                    tmp = dfres.copy()
                    tmp["nonlinear_gap"] = pd.to_numeric(tmp["mi_norm"], errors="coerce") - pd.to_numeric(tmp["pearson_r2"], errors="coerce")
                    cand = tmp.sort_values("nonlinear_gap", ascending=False).iloc[0]
                    lines.append(f"Largest MI-vs-correlation gap: {cand['driver']}. This driver may contain nonlinear or threshold-like information not captured by linear correlation.")
                if "p_value" in dfres.columns:
                    sig = int((pd.to_numeric(dfres["p_value"], errors="coerce") < 0.05).sum())
                    lines.append(f"Significance screen: {sig} drivers have p < 0.05 using the selected surrogate option. Interpret with caution for autocorrelated EC data.")

            elif method_name == "Model-vs-Observed MI":
                if "abs_delta_in" in dfres.columns:
                    mean_abs = float(pd.to_numeric(dfres["abs_delta_in"], errors="coerce").mean())
                    best = dfres.sort_values("abs_delta_in", ascending=True).iloc[0] if "driver" in dfres.columns else None
                    worst = dfres.sort_values("abs_delta_in", ascending=False).iloc[0] if "driver" in dfres.columns else None
                    lines.append(f"Mean absolute Delta I_n = {mean_abs:.4g}. Values closer to 0 mean the model better preserves observed driver-target dependence.")
                    if best is not None:
                        lines.append(f"Best reproduced driver dependence: {best['driver']} (|Delta I_n| = {float(best['abs_delta_in']):.4g}).")
                    if worst is not None:
                        sign = float(worst.get("delta_in", np.nan))
                        direction = "overestimates" if sign > 0 else "underestimates" if sign < 0 else "matches"
                        lines.append(f"Largest mismatch: {worst['driver']} (Delta I_n = {sign:.4g}); the model {direction} this dependence relative to observations.")

            elif method_name == "Pairwise PID Matrix":
                for comp, label in [("synergy_frac", "synergy"), ("redundant_frac", "redundancy"), ("unique_total_frac", "unique information")]:
                    if comp in dfres.columns:
                        row = dfres.sort_values(comp, ascending=False).iloc[0]
                        pair = f"{row.get('driver_1','?')} + {row.get('driver_2','?')}"
                        lines.append(f"Highest {label}: {pair} ({float(row[comp]):.4g}).")
                lines.append("High redundancy means overlapping driver information. High uniqueness means more independent contributions. High synergy means the driver pair matters jointly.")

            elif method_name == "Model-vs-Observed PID Matrix":
                if "delta_ipart" in dfres.columns:
                    best = dfres.sort_values("delta_ipart", ascending=True).iloc[0]
                    worst = dfres.sort_values("delta_ipart", ascending=False).iloc[0]
                    lines.append("PID difference values are model minus observed. Near zero means better functional agreement.")
                    lines.append(f"Best reproduced pair: {best.get('driver_1','?')} + {best.get('driver_2','?')} (Delta I_part = {float(best['delta_ipart']):.4g}).")
                    lines.append(f"Largest pairwise mismatch: {worst.get('driver_1','?')} + {worst.get('driver_2','?')} (Delta I_part = {float(worst['delta_ipart']):.4g}).")
                    for comp in ["delta_s", "delta_r", "delta_u"]:
                        if comp in dfres.columns:
                            row = dfres.iloc[pd.to_numeric(dfres[comp], errors="coerce").abs().idxmax()]
                            lines.append(f"Largest |{comp}| occurs for {row.get('driver_1','?')} + {row.get('driver_2','?')} ({float(row[comp]):.4g}).")

            elif method_name == "Functional Performance Summary":
                row = dfres.iloc[0]
                if "mi_fidelity_score" in dfres.columns:
                    lines.append(f"Individual-driver MI fidelity score = {float(row['mi_fidelity_score']):.4g}; higher means closer model-observation agreement.")
                if "pid_fidelity_score_0_1" in dfres.columns:
                    lines.append(f"Pairwise PID fidelity score = {float(row['pid_fidelity_score_0_1']):.4g} on a 0–1 scale; higher means better preservation of redundancy/uniqueness/synergy.")
                lines.append("Compare these values with R²/RMSE from the ML toolbox. Good prediction plus good functional scores is the strongest result.")

            elif method_name == "Figure 6 Temporal Summary":
                if "result_type" in dfres.columns:
                    mi_rows = dfres[dfres["result_type"] == "lagged_mi_peak"]
                    te_rows = dfres[dfres["result_type"] == "te_peak"]
                    if not mi_rows.empty:
                        best = mi_rows.sort_values("peak_mi_norm", ascending=False).iloc[0]
                        lines.append(f"Strongest lagged dependence: {best.get('driver', '?')} at lag {int(best.get('peak_lag', 0))} (normalized MI = {float(best.get('peak_mi_norm', np.nan)):.4g}).")
                    if not te_rows.empty:
                        best = te_rows.sort_values("peak_te_norm", ascending=False).iloc[0]
                        lines.append(f"Strongest directional transfer to the target: {best.get('driver', '?')} at lag {int(best.get('peak_lag', 0))} (normalized TE = {float(best.get('peak_te_norm', np.nan)):.4g}, p = {float(best.get('p_peak_maxstat', np.nan)):.4g}).")
                        supported = int((pd.to_numeric(te_rows.get("p_peak_maxstat"), errors="coerce") < 0.05).sum())
                        lines.append(f"Supported driver-to-target TE peaks: {supported} of {len(te_rows)} using the maximum-statistic temporal surrogate test.")
                lines.append("Use continuous timestamp-preserving observations for this result. Directional information transfer is not definitive causality.")

            elif method_name in {"Lagged MI I(X_t;Y_{t+lag})", "Transfer Entropy vs Lag"}:
                val_col = "mi_bits" if "mi_bits" in dfres.columns else "te_bits" if "te_bits" in dfres.columns else None
                if val_col and "lag" in dfres.columns:
                    row = dfres.sort_values(val_col, ascending=False).iloc[0]
                    lag = int(row["lag"])
                    lines.append(f"Peak {val_col} occurs at lag {lag} ({float(row[val_col]):.4g}). Positive lags mean the source leads the target in the selected time-step units.")
                    lines.append("Peaks can reflect process memory, diurnal cycles, autocorrelation, or timing mismatch. Use physical interpretation and surrogate tests.")

            elif method_name == "TE Network":
                vals = dfres.select_dtypes(include=[np.number]).to_numpy(dtype=float)
                if vals.size:
                    lines.append(f"TE network maximum displayed value = {np.nanmax(vals):.4g}. Rows are sources and columns are targets; current network exports use normalized TE when available.")
                    lines.append("Use this as an exploratory directed-dependence map. Strong links should be checked with lag choice, surrogate tests, and physical plausibility.")

            else:
                key_metric, key_value, best_label, _, mean_value = _extract_key_summary(method_name, dfres)
                if key_metric:
                    lines.append(f"Key summary: {key_metric} = {key_value:.4g}; mean = {mean_value:.4g}.")
                    if best_label:
                        lines.append(f"Most notable variable/pair/lag: {best_label}.")
                lines.append("Use the method guide above to interpret this diagnostic in context.")

            lines.append(
                "\nNext step: save Results + Metadata for reproducibility, or run another method and compare it in the Compare Runs tab."
            )
            return "\n".join(lines) + "\n"
        except Exception as exc:
            return f"\n{'='*72}\nPlain-language interpretation\n{'='*72}\nCould not automatically summarize this result ({exc}). Use the guide and table to interpret it manually.\n"

    def _append_interpretation_to_results(method_name: str):
        try:
            txt.insert(tk.END, _interpret_latest_result(method_name))
            txt.see(tk.END)
        except Exception:
            pass

    def _finalize_success(method_name: str):
        """Common updates after any successful method run."""
        _refresh_results_table()
        _add_run_record(method_name)
        _append_interpretation_to_results(method_name)
        if last_results_df is not None and not last_results_df.empty:
            save_results_btn.configure(state=tk.NORMAL)
            save_package_btn.configure(state=tk.NORMAL)
            export_publication_btn.configure(state=tk.NORMAL)
        results_nb.select(tab_plot)

    clear_compare_btn.configure(command=_clear_run_history)
    compare_metric_cb.bind("<<ComboboxSelected>>", _refresh_compare_runs)
    refresh_network_btn.configure(command=_refresh_network_viewer)
    network_view_cb.bind("<<ComboboxSelected>>", lambda _e: _refresh_network_viewer())
    save_network_figure_btn.configure(command=_save_network_figure)
    save_network_edges_btn.configure(command=_save_network_edges)
    _refresh_compare_runs()
    _refresh_network_viewer()

    def _discretize(x: np.ndarray) -> np.ndarray:
        n_bins = int(bins_var.get())
        if disc_var.get() == "freq":
            return discretize_equal_frequency(x, n_bins=n_bins)
        return discretize_equal_width(x, n_bins=n_bins)

    def _col_as_numeric(col_name: str) -> np.ndarray:
        active = df_runtime["active"]
        if col_name not in active.columns:
            raise ValueError(f"Column '{col_name}' was not found in the filtered data table.")
        s = pd.to_numeric(active[col_name], errors="coerce")
        arr = s.to_numpy(dtype=float)
        (arr,) = _align_dropna(arr)
        if arr.size == 0:
            raise ValueError(f"Column '{col_name}' has no numeric values after conversion.\nSelect a different variable or check your data/filter.")
        return arr

    def _get_xy() -> Tuple[np.ndarray, np.ndarray, str, str]:
        xn, yn = x_var.get(), y_var.get()
        x, y = _pair_arrays_from_df(df_runtime["active"], xn, yn)
        return x, y, xn, yn

    def _get_xyz() -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, str, str]:
        xn, yn, zn = x_var.get(), y_var.get(), z_var.get()
        if not zn:
            raise ValueError("Select Z for this measure (target/model variable).")
        x, y, z = _multi_arrays_from_df(df_runtime["active"], [xn, yn, zn])
        return x, y, z, xn, yn, zn

    def _get_temporal_frame(columns: Iterable[str]) -> Tuple[pd.DataFrame, Dict[str, object]]:
        """Return selected variables on a complete regular timestamp grid."""
        return _regularize_temporal_frame(
            df_runtime["active"],
            timestamp_col=ts_filter_var.get(),
            columns=columns,
            frequency=temporal_freq_var.get(),
        )

    def _get_figure6_temporal_frame(columns: Iterable[str]) -> Tuple[pd.DataFrame, Dict[str, object]]:
        """Prepare Figure 6 at Native, Daily, or Weekly resolution internally."""
        rules = {
            c: fig6_aggregation_rules.get(c, _infer_variable_aggregation(c))
            for c in columns
        }
        return _prepare_figure6_temporal_frame(
            df_runtime["active"],
            timestamp_col=ts_filter_var.get(),
            columns=columns,
            resolution=fig6_resolution_var.get(),
            native_frequency=temporal_freq_var.get(),
            aggregation_rules=rules,
            minimum_coverage_percent=float(fig6_coverage_var.get()),
            weekly_minimum_valid_days=int(fig6_weekly_days_var.get()),
        )

    def _parse_months() -> List[int]:
        vals = []
        for part in months_var.get().replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            try:
                m = int(part)
                if 1 <= m <= 12:
                    vals.append(m)
            except ValueError:
                pass
        return sorted(set(vals)) or [5, 6, 7, 8, 9]

    def _get_active_df() -> pd.DataFrame:
        active = df.copy()
        mode = window_var.get()
        tscol = ts_filter_var.get()

        # Global analysis period: applied before all IT methods and before additional
        # daytime/month/window filters. This keeps IT outputs consistent with the ML
        # toolbox and with manuscript periods such as 2013--2022.
        needs_time = (
            (not use_full_period_var.get())
            or mode in {"Growing season", "Non-growing season", "Selected months", "Custom date range"}
        )
        if needs_time:
            if tscol not in active.columns:
                raise ValueError("A timestamp column is required for the selected analysis period/window.")
            t = pd.to_datetime(active[tscol], errors="coerce")
            active = active.loc[t.notna()].copy()
            t = pd.to_datetime(active[tscol], errors="coerce")

            # Explicit period filter. If Use full available period is checked, the
            # start/end boxes are documentation only unless Custom date range is selected.
            if (not use_full_period_var.get()) or mode == "Custom date range":
                start = start_date_var.get().strip()
                end = end_date_var.get().strip()
                mask = pd.Series(True, index=active.index)
                if start:
                    mask &= t >= pd.to_datetime(start)
                if end:
                    end_ts = pd.to_datetime(end)
                    if end_ts.time() == pd.Timestamp(end_ts.date()).time() and len(str(end).strip()) <= 10:
                        mask &= t < end_ts + pd.Timedelta(days=1)
                    else:
                        mask &= t <= end_ts
                active = active.loc[mask.to_numpy()].copy()
                t = pd.to_datetime(active[tscol], errors="coerce")

            if mode in {"Growing season", "Non-growing season", "Selected months"}:
                months = _parse_months()
                mask = t.dt.month.isin(months)
                if mode == "Non-growing season":
                    mask = ~mask
                active = active.loc[mask.to_numpy()].copy()

        if mode in {"Daytime", "Nighttime"}:
            rad = rad_var.get()
            if rad not in active.columns:
                raise ValueError("Select a valid radiation/light column for daytime/nighttime filtering.")
            try:
                thresh = float(day_threshold_var.get())
            except ValueError:
                thresh = 20.0
            r = pd.to_numeric(active[rad], errors="coerce")
            mask = r > thresh if mode == "Daytime" else r <= thresh
            active = active.loc[mask.fillna(False)].copy()

        if active.empty:
            raise ValueError(f"The selected analysis period/window produced no rows: {mode}.")
        return active

    def _current_metadata(method_name: str) -> Dict[str, object]:
        selected = [net_list.get(i) for i in net_list.curselection()]
        return {
            "site_name": site_name,
            "input_source": input_source_var.get(),
            "method": method_name,
            "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "n_rows_original": int(len(df)),
            "n_rows_analysis": int(len(df_runtime["active"])),
            "analysis_period_mode": "full_available" if use_full_period_var.get() else "custom_period",
            "analysis_window": window_var.get(),
            "timestamp_column": ts_filter_var.get(),
            "radiation_column": rad_var.get(),
            "day_threshold": day_threshold_var.get(),
            "months": months_var.get(),
            "analysis_start_date": start_date_var.get(),
            "analysis_end_date": end_date_var.get(),
            "x_source": x_var.get(),
            "y_target": y_var.get(),
            "z_model_or_pid_target": z_var.get(),
            "selected_drivers": selected,
            "estimator": est_var.get(),
            "n_bins_or_kde_n": int(bins_var.get()),
            "discretizer": "equal_freq" if disc_var.get() == "freq" else "equal_width",
            "max_abs_lag": int(maxlag_var.get()),
            "te_lag": int(delay_var.get()),
            "n_permutations": int(perm_var.get()),
            "surrogate_type": surrogate_var.get(),
            "surrogate_block_size": int(block_size_var.get()),
            "temporal_grid_setting": temporal_freq_var.get(),
            "figure6_min_samples_per_lag": int(fig6_min_samples_var.get()),
            "figure6_alpha": float(fig6_alpha_var.get()),
            "figure6_bidirectional_network": bool(fig6_bidirectional_var.get()),
            "figure6_supported_links_only": bool(fig6_supported_only_var.get()),
            "row_alignment": "row-wise numeric dropna for contemporaneous methods; timestamp-preserving regular grid for temporal methods",
        }

    # --- Run analysis ---

    def run_analysis():
        nonlocal last_matrix, last_vars, last_results_df, last_network_edges, last_metadata
        nonlocal last_prepared_temporal_df, last_figure6_lagged_mi_full_df
        nonlocal last_figure6_transfer_entropy_full_df, last_figure6_temporal_pid_df
        nonlocal last_figure6_temporal_pid_bootstrap_df, cbar, ax

        m = measure_var.get()
        n_bins = int(bins_var.get())
        disc = "equal_freq" if disc_var.get() == "freq" else "equal_width"
        nperm = int(perm_var.get())
        estimator = est_var.get()
        surrogate_type = surrogate_var.get()

        try:
            import time as _time
            _run_started = _time.perf_counter()
            progress_var.set(8)
            progress_status_var.set("Preparing data…")
            run_btn.configure(state=tk.DISABLED)
            win.update_idletasks()
            df_runtime["active"] = _get_active_df()
            progress_var.set(20)
            progress_status_var.set(f"Computing {m}…")
            win.update_idletasks()
            last_metadata = _current_metadata(m)
            # Reset export state so a failed or different run cannot export stale Figure 6 tables.
            last_results_df = None
            last_figure6_lagged_mi_full_df = None
            last_figure6_transfer_entropy_full_df = None
            last_figure6_temporal_pid_df = None
            last_figure6_temporal_pid_bootstrap_df = None
            if m != "Figure 6 Temporal Summary":
                last_prepared_temporal_df = None
                save_prepared_btn.configure(state=tk.DISABLED)
            if m != "TE Network":
                last_network_edges = None
            save_results_btn.configure(state=tk.DISABLED)
            save_package_btn.configure(state=tk.DISABLED)

            # --- hard reset of figure + axes + colorbar each run ---
            fig.clf()                     # remove ALL axes (main + old colorbars)
            fig.set_size_inches(7.5, 4.6)
            ax = fig.add_subplot(111)     # new fresh axes taking full space
            cbar = None                   # forget any previous colorbar handle
            last_results_df = None
            save_results_btn.configure(state=tk.DISABLED)
            save_package_btn.configure(state=tk.DISABLED)
            save_btn.configure(state=tk.DISABLED)
            # -------------------------------------------------------

            if m == "Entropy H(X)":
                xn = x_var.get()
                x = _col_as_numeric(xn)
                x_lab = _discretize(x)
                H = entropy_discrete(x_lab, base=2.0)
                _show(f"H({xn}) ≈ {H:.6f} bits\n")

                ax.hist(x, bins=max(10, n_bins))
                ax.set_title(f"Histogram: {xn}")
                ax.set_xlabel(xn)
                ax.set_ylabel("Count")
                canvas.draw_idle()

            elif m == "Mutual Information I(X;Y)":
                x, y, xn, yn = _get_xy()
                if estimator == "kde-tip":
                    mi = kde_mi_2d(x, y, N=n_bins, bin_scheme="global", method="KDE")
                    used = f"kde-tip (N={n_bins})"
                else:
                    mi = mutual_information(
                        x, y,
                        base=2.0,
                        method="hist",
                        n_bins=n_bins,
                        disc=disc,
                    )
                    used = f"hist (bins={n_bins}, {disc})"

                _show(f"I({xn}; {yn}) ≈ {mi:.6f} bits  [{used}]\n")

                ax.cla()
                ax.scatter(x, y, s=6, alpha=0.6)
                ax.set_title(f"Scatter: {xn} vs {yn}")
                ax.set_xlabel(xn)
                ax.set_ylabel(yn)
                canvas.draw_idle()

            elif m == "MI Driver Ranking":
                # Target is selected in the Y combobox.
                # Drivers are selected from the Network variables list.
                # If none are selected, all columns except the target are tried.
                yn = y_var.get()

                selected = [net_list.get(i) for i in net_list.curselection()]
                drivers = selected if selected else [c for c in all_cols if c != yn]

                ranking = mutual_information_driver_ranking(
                    df_runtime["active"],
                    target=yn,
                    drivers=drivers,
                    base=2.0,
                    estimator=estimator,
                    n_bins=n_bins,
                    disc=disc,
                    min_samples=50,
                    n_perm=nperm,
                    surrogate_type=surrogate_type,
                    seed=42,
                )
                last_results_df = ranking.copy()
                save_results_btn.configure(state=tk.NORMAL)

                used = (
                    f"kde-tip (N={n_bins})"
                    if estimator == "kde-tip"
                    else f"hist (bins={n_bins}, {disc})"
                )

                display_cols = ["rank", "driver", "mi_norm", "mi_bits", "n_samples"]
                if nperm > 0 and "p_value" in ranking.columns:
                    display_cols.append("p_value")

                lines = [
                    f"MI driver ranking for target {yn} [{used}]",
                    f"Drivers evaluated: {len(ranking)}",
                    "",
                    ranking[display_cols].to_string(
                        index=False,
                        float_format=lambda v: f"{v:.6f}",
                    ),
                ]
                _show("\n".join(lines) + "\n")

                ax.cla()

                value_col = "mi_norm" if "mi_norm" in ranking.columns else "mi_bits"
                plot_df = ranking.sort_values(value_col, ascending=True).copy()
                values = plot_df[value_col].to_numpy(dtype=float)
                y_pos = np.arange(len(plot_df))

                # Clean display labels without changing the underlying variable names.
                def _display_variable_name(name):
                    label = str(name)
                    label = re.sub(r"_(obs|pred(?:_[A-Za-z0-9]+)?|resid(?:_[A-Za-z0-9]+)?)$", "", label, flags=re.IGNORECASE)
                    label = re.sub(r"(?:_\d+){2,}$", "", label)
                    return label

                display_names = [_display_variable_name(v) for v in plot_df["driver"]]
                target_display = _display_variable_name(yn)

                # Horizontal lollipop ranking: lighter visual weight than bars while
                # preserving exact values and rank order.
                vmax = float(np.nanmax(values)) if len(values) else 0.0
                norm_values = values / vmax if vmax > 0 else np.zeros_like(values)
                point_colors = __import__("matplotlib").colormaps["viridis"](0.25 + 0.65 * norm_values)

                ax.hlines(y=y_pos, xmin=0.0, xmax=values, color="#c9ced6", linewidth=3.0, zorder=1)
                ax.scatter(values, y_pos, s=115, c=point_colors, edgecolors="white", linewidths=1.1, zorder=3)

                ranked_from_top = list(range(len(plot_df), 0, -1))
                rank_labels = [f"{rank}.  {name}" for rank, name in zip(ranked_from_top, display_names)]
                ax.set_yticks(y_pos)
                ax.set_yticklabels(rank_labels, fontsize=10)
                ax.set_xlabel(
                    "Normalized mutual information [-]"
                    if value_col == "mi_norm"
                    else "Mutual information [bits]",
                    fontsize=10,
                )
                ax.set_ylabel("")
                ax.set_title("Mutual information ranking", fontsize=14, fontweight="semibold", pad=18)
                ax.text(
                    0.0, 1.015, f"Target: {target_display}",
                    transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=9, color="#4b5563",
                )

                pad = 0.025 * vmax if vmax > 0 else 0.001
                ax.set_xlim(0, vmax + 7 * pad if vmax > 0 else 1.0)
                ax.grid(True, axis="x", alpha=0.22, linewidth=0.8)
                ax.grid(False, axis="y")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_visible(False)
                ax.tick_params(axis="y", length=0, pad=8)

                for i, (_, row) in enumerate(plot_df.iterrows()):
                    label = f"{row[value_col]:.3f}"
                    if nperm > 0 and not np.isnan(row.get("p_value", np.nan)):
                        label += f"  (p={row['p_value']:.3f})"
                    ax.text(
                        row[value_col] + pad, i, label,
                        va="center", ha="left", fontsize=9, color="#263238",
                    )

                fig.tight_layout(pad=1.8)
                canvas.draw_idle()

            elif m == "Correlation vs MI Ranking":
                # Target is selected in Y; drivers are selected from the variable pool.
                yn = y_var.get()
                selected = [net_list.get(i) for i in net_list.curselection()]
                drivers = _selected_or_default_drivers(selected, all_cols, yn)

                ranking = correlation_mi_driver_ranking(
                    df_runtime["active"],
                    target=yn,
                    drivers=drivers,
                    base=2.0,
                    estimator=estimator,
                    n_bins=n_bins,
                    disc=disc,
                    min_samples=50,
                )
                last_results_df = ranking.copy()
                save_results_btn.configure(state=tk.NORMAL)

                used = (
                    f"kde-tip (N={n_bins})"
                    if estimator == "kde-tip"
                    else f"hist (bins={n_bins}, {disc})"
                )

                display_cols = [
                    "rank", "driver", "pearson_r", "pearson_r2", "mi_norm", "mi_bits",
                    "rank_r2", "rank_mi", "n_samples",
                ]
                lines = [
                    f"Correlation vs MI ranking for target {yn} [{used}]",
                    "Pearson r² captures linear association; MI captures nonlinear dependence.",
                    f"Drivers evaluated: {len(ranking)}",
                    "",
                    ranking[display_cols].to_string(
                        index=False,
                        float_format=lambda v: f"{v:.6f}",
                    ),
                ]
                _show("\n".join(lines) + "\n")

                ax.cla()
                value_col = "mi_norm" if "mi_norm" in ranking.columns else "mi_bits"
                plot_df = ranking.sort_values(value_col, ascending=True).copy()
                y_pos = np.arange(len(plot_df))
                r2_vals = plot_df["pearson_r2"].to_numpy(dtype=float)
                mi_vals = plot_df[value_col].to_numpy(dtype=float)
                # Dumbbell plot emphasizes where nonlinear dependence exceeds linear association.
                for yi, a, b in zip(y_pos, r2_vals, mi_vals):
                    ax.plot([a, b], [yi, yi], color="0.78", linewidth=2.0, zorder=1)
                ax.scatter(r2_vals, y_pos, s=52, label="Pearson r²", zorder=3)
                ax.scatter(mi_vals, y_pos, s=52, marker="D",
                           label="Normalized MI" if value_col == "mi_norm" else "MI [bits]", zorder=3)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(plot_df["driver"].astype(str))
                ax.set_xlabel("Dependence strength")
                ax.set_ylabel("")
                ax.set_title("Linear versus nonlinear dependence", fontsize=14, fontweight="semibold", pad=14)
                ax.text(0.0, 1.01, f"Target: {yn}", transform=ax.transAxes,
                        ha="left", va="bottom", fontsize=9, color="0.35")
                ax.legend(frameon=False, loc="lower right")
                ax.grid(True, axis="x", alpha=0.25)
                for spine in ("top", "right", "left"):
                    ax.spines[spine].set_visible(False)
                fig.tight_layout(pad=1.6)
                canvas.draw_idle()

            elif m == "Model-vs-Observed MI":
                # Y is the observed target; Z is the modeled target/prediction column.
                yn = y_var.get()
                zn = z_var.get()
                if not zn:
                    raise ValueError("Select a modeled target/prediction column in Z.")
                selected = [net_list.get(i) for i in net_list.curselection()]
                drivers = _selected_or_default_drivers(selected, all_cols, yn)
                drivers = [d for d in drivers if d != zn]

                perf = mutual_information_functional_performance(
                    df_runtime["active"],
                    observed_target=yn,
                    modeled_target=zn,
                    drivers=drivers,
                    base=2.0,
                    estimator=estimator,
                    n_bins=n_bins,
                    disc=disc,
                    min_samples=50,
                )
                last_results_df = perf.copy()
                save_results_btn.configure(state=tk.NORMAL)

                display_cols = ["rank", "driver", "in_obs", "in_mod", "delta_in", "abs_delta_in", "n_obs"]
                lines = [
                    f"Model-vs-observed MI functional performance: {zn} vs {yn}",
                    "Delta I_n near 0 means the modeled target reproduces the observed driver--flux dependence.",
                    "Positive Delta I_n = model overestimates dependence; negative = underestimates.",
                    "",
                    perf[display_cols].to_string(index=False, float_format=lambda v: f"{v:.6f}"),
                ]
                _show("\n".join(lines) + "\n")

                ax.cla()
                plot_df = perf.sort_values("in_obs", ascending=True).copy()
                y_pos = np.arange(len(plot_df))
                obs_vals = plot_df["in_obs"].to_numpy(dtype=float)
                mod_vals = plot_df["in_mod"].to_numpy(dtype=float)
                for yi, a, b in zip(y_pos, obs_vals, mod_vals):
                    ax.plot([a, b], [yi, yi], color="0.78", linewidth=2.2, zorder=1)
                ax.scatter(obs_vals, y_pos, s=58, label="Observed", zorder=3)
                ax.scatter(mod_vals, y_pos, s=58, marker="D", label="Modeled", zorder=3)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(plot_df["driver"].astype(str))
                ax.set_xlabel("Normalized mutual information")
                ax.set_ylabel("")
                ax.set_title("Model information fidelity", fontsize=14, fontweight="semibold", pad=14)
                ax.text(0.0, 1.01, f"Observed: {yn}   |   Modeled: {zn}", transform=ax.transAxes,
                        ha="left", va="bottom", fontsize=9, color="0.35")
                ax.legend(frameon=False, loc="lower right")
                ax.grid(True, axis="x", alpha=0.25)
                for spine in ("top", "right", "left"):
                    ax.spines[spine].set_visible(False)
                fig.tight_layout(pad=1.6)
                canvas.draw_idle()

            elif m == "Model-vs-Observed PID Matrix":
                # Y is the observed target; Z is the modeled target/prediction column.
                yn = y_var.get()
                zn = z_var.get()
                if not zn:
                    raise ValueError("Select a modeled target/prediction column in Z.")
                selected = [net_list.get(i) for i in net_list.curselection()]
                drivers = _selected_or_default_drivers(selected, all_cols, yn, max_drivers=10)
                drivers = [d for d in drivers if d != zn]

                diff_table, diff_mats, diff_vars = pairwise_pid_difference_matrix(
                    df_runtime["active"],
                    observed_target=yn,
                    modeled_target=zn,
                    drivers=drivers,
                    base=2.0,
                    estimator=estimator,
                    n_bins=n_bins,
                    disc=disc,
                    min_samples=50,
                )
                last_results_df = diff_table.copy()
                save_results_btn.configure(state=tk.NORMAL)

                display_cols = ["driver_1", "driver_2", "delta_s", "delta_r", "delta_u", "delta_ipart", "pairwise_score_0_1"]
                lines = [
                    f"Model-vs-observed pairwise PID: {zn} vs {yn}",
                    "Heatmaps show model minus observed PID fractions.",
                    "Positive values mean the model overestimates that information type.",
                    "Negative values mean the model underestimates that information type.",
                    "",
                    diff_table[display_cols].to_string(index=False, float_format=lambda v: f"{v:.6f}"),
                ]
                _show("\n".join(lines) + "\n")

                fig.clf()
                axes = fig.subplots(1, 3)
                components = [
                    ("delta_s", "Delta S  synergy"),
                    ("delta_r", "Delta R  redundancy"),
                    ("delta_u", "Delta U  uniqueness"),
                ]
                last_im = None
                for ax_i, (key, title_i) in zip(axes, components):
                    last_im = ax_i.imshow(
                        diff_mats[key],
                        aspect="equal",
                        origin="upper",
                        vmin=-1.0,
                        vmax=1.0,
                        cmap="RdYlGn",
                    )
                    ax_i.axhline(-0.5, linewidth=0.5)
                    ax_i.set_xticks(range(len(diff_vars)))
                    ax_i.set_yticks(range(len(diff_vars)))
                    ax_i.set_xticklabels(diff_vars, rotation=45, ha="right", fontsize=8)
                    ax_i.set_yticklabels(diff_vars, fontsize=8)
                    ax_i.set_title(title_i)
                    ax_i.set_xlim(-0.5, len(diff_vars) - 0.5)
                    ax_i.set_ylim(len(diff_vars) - 0.5, -0.5)
                if last_im is not None:
                    fig.colorbar(
                        last_im,
                        ax=axes.ravel().tolist(),
                        fraction=0.046,
                        pad=0.04,
                        label="Model - observed fraction",
                    )
                fig.suptitle(f"Model-vs-observed PID matrix: {zn} vs {yn}", y=0.98)
                fig.tight_layout()
                canvas.draw_idle()

            elif m == "Functional Performance Summary":
                # Y is observed target; Z is modeled target/prediction column.
                yn = y_var.get()
                zn = z_var.get()
                if not zn:
                    raise ValueError("Select a modeled target/prediction column in Z.")
                selected = [net_list.get(i) for i in net_list.curselection()]
                drivers = _selected_or_default_drivers(selected, all_cols, yn, max_drivers=10)
                drivers = [d for d in drivers if d != zn]

                summary, mi_table, pid_table = functional_performance_summary(
                    df_runtime["active"],
                    observed_target=yn,
                    modeled_target=zn,
                    drivers=drivers,
                    base=2.0,
                    estimator=estimator,
                    n_bins=n_bins,
                    disc=disc,
                    min_samples=50,
                )
                combined = pd.concat(
                    [summary.assign(table="summary"), mi_table.assign(table="mi"), pid_table.assign(table="pid")],
                    ignore_index=True,
                    sort=False,
                )
                last_results_df = combined
                save_results_btn.configure(state=tk.NORMAL)

                row = summary.iloc[0]
                lines = [
                    f"Functional performance summary: {zn} vs {yn}",
                    "Higher scores indicate closer agreement with observed driver--flux relationships.",
                    "",
                    summary.to_string(index=False, float_format=lambda v: f"{v:.6f}"),
                ]
                _show("\n".join(lines) + "\n")

                ax.cla()
                labels = ["Individual MI", "Pairwise PID"]
                vals = [row["mi_fidelity_score"], row["pid_fidelity_score_0_1"]]
                xpos = np.arange(len(labels))
                ax.bar(xpos, vals)
                ax.set_xticks(xpos)
                ax.set_xticklabels(labels)
                ax.axhline(1.0, linestyle="--", linewidth=1)
                ax.set_ylim(0, 1.05)
                ax.set_ylabel("Functional score (0–1; higher is better)")
                ax.set_title(f"Functional performance: {zn} vs {yn}")
                for i, v in enumerate(vals):
                    ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom")
                canvas.draw_idle()

            elif m == "Single-Site IT Summary":
                # Compact dashboard: MI ranking, r²-vs-MI, lagged MI, PID, TE-vs-lag, TE network.
                yn = y_var.get()
                xn = x_var.get()
                zn = z_var.get()
                selected = [net_list.get(i) for i in net_list.curselection()]
                drivers = _selected_or_default_drivers(selected, all_cols, yn, max_drivers=8)

                ranking = correlation_mi_driver_ranking(
                    df_runtime["active"],
                    target=yn,
                    drivers=drivers,
                    base=2.0,
                    estimator=estimator,
                    n_bins=n_bins,
                    disc=disc,
                    min_samples=50,
                )
                last_results_df = ranking.copy()
                save_results_btn.configure(state=tk.NORMAL)

                # Choose source for lagged diagnostics.
                source_for_lag = xn if xn and xn != yn and xn in df.columns else ranking.iloc[0]["driver"]
                x_lag = pd.to_numeric(df[source_for_lag], errors="coerce").to_numpy(dtype=float)
                y_lag = pd.to_numeric(df[yn], errors="coerce").to_numpy(dtype=float)
                x_lag, y_lag = _align_dropna(x_lag, y_lag)
                max_lag = int(maxlag_var.get())
                max_lag_te = max(1, min(max_lag, int(delay_var.get()) if int(delay_var.get()) > 1 else max_lag))

                lagged_mi_df = pd.DataFrame({
                    "lag": list(range(-max_lag, max_lag + 1)),
                    "mi_bits": [
                        lagged_mutual_information(
                            x_lag, y_lag,
                            lags=[L],
                            base=2.0,
                            method="hist",
                            n_bins=n_bins,
                            disc=disc,
                        )[L]
                        for L in range(-max_lag, max_lag + 1)
                    ],
                })
                te_lag_df = transfer_entropy_by_lag(
                    x_lag,
                    y_lag,
                    max_lag=max_lag_te,
                    base=2.0,
                    n_bins=n_bins,
                    disc=disc,
                )

                # Choose PID drivers: X and Z if available; otherwise top two MI drivers.
                pid_x1 = xn if xn and xn != yn and xn in df.columns else None
                pid_x2 = zn if zn and zn != yn and zn in df.columns else None
                if pid_x1 is None or pid_x2 is None or pid_x1 == pid_x2:
                    top_drivers = [d for d in ranking["driver"].tolist() if d != yn]
                    if len(top_drivers) >= 2:
                        pid_x1, pid_x2 = top_drivers[0], top_drivers[1]

                pid_vals = None
                if pid_x1 and pid_x2 and pid_x1 != pid_x2:
                    x1_arr = pd.to_numeric(df[pid_x1], errors="coerce").to_numpy(dtype=float)
                    x2_arr = pd.to_numeric(df[pid_x2], errors="coerce").to_numpy(dtype=float)
                    y_arr = pd.to_numeric(df[yn], errors="coerce").to_numpy(dtype=float)
                    x1_arr, x2_arr, y_arr = _align_dropna(x1_arr, x2_arr, y_arr)
                    if estimator == "kde-tip":
                        info_tip = kde_tip_pid_3d(x1_arr, x2_arr, y_arr, N=n_bins, bin_scheme="global", method="KDE")
                        pid_vals = {
                            "Redundant": info_tip["R"],
                            f"Unique {pid_x1}": info_tip["U1"],
                            f"Unique {pid_x2}": info_tip["U2"],
                            "Synergy": info_tip["S"],
                        }
                    else:
                        res = pid_min_information(x1_arr, x2_arr, y_arr, base=2.0, method="hist", n_bins=n_bins, disc=disc)
                        pid_vals = {
                            "Redundant": res["redundant"],
                            f"Unique {pid_x1}": res["unique_x1"],
                            f"Unique {pid_x2}": res["unique_x2"],
                            "Synergy": res["synergy"],
                        }

                # Compute small TE network among selected drivers plus target.
                net_vars = [v for v in list(dict.fromkeys(drivers + [yn])) if v in df.columns][:8]
                te_mat = np.full((len(net_vars), len(net_vars)), np.nan)
                if len(net_vars) >= 2:
                    data = df_runtime["active"][net_vars].apply(pd.to_numeric, errors="coerce")
                    lag_for_net = int(delay_var.get())
                    lag_for_net = max(1, lag_for_net)
                    for j in range(len(net_vars)):
                        for i in range(len(net_vars)):
                            if i == j:
                                continue
                            xi = data.iloc[:, i].to_numpy(dtype=float)
                            yj = data.iloc[:, j].to_numpy(dtype=float)
                            xi, yj = _align_dropna(xi, yj)
                            if len(xi) >= lag_for_net + 2:
                                te_mat[i, j] = transfer_entropy(
                                    xi, yj,
                                    lag=lag_for_net,
                                    base=2.0,
                                    n_bins=n_bins,
                                    disc=disc,
                                )

                _show(
                    f"Single-site IT summary for target {yn}\n"
                    f"Drivers: {', '.join(drivers)}\n"
                    f"Lag source: {source_for_lag}\n"
                    f"PID drivers: {pid_x1}, {pid_x2}\n"
                    f"Estimator: {estimator}, bins/KDE N={n_bins}, disc={disc}\n"
                )

                fig.clf()
                fig.set_size_inches(11.0, 7.0)
                axes = fig.subplots(2, 3)

                # A: MI ranking
                ax0 = axes[0, 0]
                rank_plot = ranking.sort_values("mi_bits", ascending=True)
                ypos = np.arange(len(rank_plot))
                ax0.barh(ypos, rank_plot["mi_bits"])
                ax0.set_yticks(ypos)
                ax0.set_yticklabels(rank_plot["driver"], fontsize=8)
                ax0.set_xlabel("MI [bits]")
                ax0.set_title("MI driver ranking")

                # B: r² vs MI scatter
                ax1 = axes[0, 1]
                ax1.scatter(ranking["pearson_r2"], ranking["mi_bits"], s=25)
                for _, row in ranking.iterrows():
                    ax1.text(row["pearson_r2"], row["mi_bits"], f" {row['driver']}", fontsize=7)
                ax1.set_xlabel("Pearson r²")
                ax1.set_ylabel("MI [bits]")
                ax1.set_title("Linear vs nonlinear dependence")

                # C: lagged MI
                ax2 = axes[0, 2]
                ax2.plot(lagged_mi_df["lag"], lagged_mi_df["mi_bits"], marker="o", markersize=3)
                ax2.axvline(0, linestyle="--", linewidth=1)
                ax2.set_xlabel("Lag")
                ax2.set_ylabel("MI [bits]")
                ax2.set_title(f"Lagged MI: {source_for_lag} / {yn}")

                # D: PID
                ax3 = axes[1, 0]
                if pid_vals:
                    pid_labels = list(pid_vals.keys())
                    pid_values = list(pid_vals.values())
                    ax3.bar(range(len(pid_values)), pid_values)
                    ax3.set_xticks(range(len(pid_labels)))
                    ax3.set_xticklabels(pid_labels, rotation=25, ha="right", fontsize=8)
                    ax3.set_ylabel("Bits")
                    ax3.set_title("PID")
                else:
                    ax3.text(0.5, 0.5, "PID not available", ha="center", va="center")
                    ax3.set_axis_off()

                # E: TE vs lag
                ax4 = axes[1, 1]
                ax4.plot(te_lag_df["lag"], te_lag_df["te_bits"], marker="o", markersize=3)
                ax4.set_xlabel("Lag")
                ax4.set_ylabel("TE [bits]")
                ax4.set_title(f"TE vs lag: {source_for_lag} → {yn}")

                # F: TE network
                ax5 = axes[1, 2]
                if len(net_vars) >= 2:
                    im = ax5.imshow(te_mat, aspect="auto", origin="upper")
                    ax5.set_xticks(range(len(net_vars)))
                    ax5.set_yticks(range(len(net_vars)))
                    ax5.set_xticklabels(net_vars, rotation=45, ha="right", fontsize=7)
                    ax5.set_yticklabels(net_vars, fontsize=7)
                    ax5.set_xlabel("Target")
                    ax5.set_ylabel("Source")
                    ax5.set_title("TE network")
                    fig.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)
                else:
                    ax5.text(0.5, 0.5, "TE network not available", ha="center", va="center")
                    ax5.set_axis_off()

                fig.tight_layout()
                canvas.draw_idle()

            elif m == "Figure 6 Temporal Summary":
                yn = y_var.get()
                selected = [net_list.get(i) for i in net_list.curselection()]
                drivers = _selected_or_default_drivers(selected, all_cols, yn, max_drivers=8)
                drivers = [d for d in drivers if d in df_runtime["active"].columns and d != yn]
                if not drivers:
                    raise ValueError(
                        "Select at least one environmental driver for Figure 6, or use Configure Figure 6."
                    )
                if input_source_var.get() in {"Current ML results", "Saved ML–IT bridge"}:
                    raise ValueError(
                        "Figure 6 requires the continuous observed dataset. Switch Input source to "
                        "Current dataset or load an external continuous dataset."
                    )
                if int(maxlag_var.get()) < 1:
                    raise ValueError("Maximum |lag| must be at least 1.")
                if nperm <= 0:
                    raise ValueError(
                        "Figure 6 Temporal Summary requires temporal surrogates. "
                        "Set Permutations to at least 100; 500 is recommended for the manuscript."
                    )

                temporal, temporal_meta = _get_figure6_temporal_frame(drivers + [yn])
                last_prepared_temporal_df = temporal.reset_index()
                save_prepared_btn.configure(state=tk.NORMAL)
                last_metadata.update(temporal_meta)
                max_lag = int(maxlag_var.get())
                block_size = int(block_size_var.get())
                min_samples = int(fig6_min_samples_var.get())
                alpha = float(fig6_alpha_var.get())
                bidirectional = bool(fig6_bidirectional_var.get())
                supported_only = bool(fig6_supported_only_var.get())
                if not (0.0 < alpha < 1.0):
                    raise ValueError("Figure 6 support threshold alpha must be between 0 and 1.")
                y = temporal[yn].to_numpy(dtype=float)
                mi_rows = []
                te_rows = []
                edge_rows = []
                mi_full_frames = []
                te_full_frames = []
                pid_rows = []
                pid_bootstrap_rows = []

                for idx_driver, driver in enumerate(drivers):
                    progress_status_var.set(f"Figure 6: {driver} → {yn}…")
                    progress_var.set(20 + 65 * (idx_driver / max(1, len(drivers))))
                    win.update_idletasks()
                    x = temporal[driver].to_numpy(dtype=float)

                    mi_table = lagged_mutual_information_table(
                        x, y, lags=range(-max_lag, max_lag + 1),
                        base=2.0, method="hist", n_bins=n_bins,
                        disc=disc, min_samples=min_samples,
                    )
                    mi_test = temporal_max_statistic_test(
                        x, y, lags=range(-max_lag, max_lag + 1),
                        statistic="lagged_mi_norm", base=2.0, n_bins=n_bins,
                        disc=disc, min_samples=min_samples, n_surrogates=nperm,
                        surrogate_type=surrogate_type, block_size=block_size,
                        seed=4200 + idx_driver,
                    )
                    mi_peak = mi_table.loc[mi_table["mi_norm"].idxmax()] if mi_table["mi_norm"].notna().any() else None
                    mi_p = float(mi_test["p_value"]) if np.isfinite(mi_test["p_value"]) else np.nan
                    mi_rows.append({
                        "result_type": "lagged_mi_peak", "driver": driver,
                        "source": driver, "target": yn,
                        "peak_lag": mi_test["peak_lag"],
                        "peak_mi_bits": float(mi_peak["mi_bits"]) if mi_peak is not None else np.nan,
                        "peak_mi_norm": mi_test["observed_peak"],
                        "n_samples_peak": int(mi_peak["n_samples"]) if mi_peak is not None else 0,
                        "p_peak_maxstat": mi_p,
                        "null_max_95": mi_test["null_max_95"],
                        "supported_alpha": bool(np.isfinite(mi_p) and mi_p < alpha),
                        "supported_0_05": bool(np.isfinite(mi_p) and mi_p < 0.05),
                    })

                    mi_curve = mi_table.copy()
                    mi_curve.insert(0, "result_type", "lagged_mi_curve")
                    mi_curve.insert(1, "driver", driver)
                    mi_curve.insert(2, "source", driver)
                    mi_curve.insert(3, "target", yn)
                    mi_curve["is_peak"] = mi_curve["lag"].eq(mi_test["peak_lag"])
                    mi_curve["peak_lag"] = mi_test["peak_lag"]
                    mi_curve["p_peak_maxstat"] = mi_p
                    mi_curve["null_max_95"] = mi_test["null_max_95"]
                    mi_curve["supported_alpha"] = bool(np.isfinite(mi_p) and mi_p < alpha)
                    mi_curve["supported_0_05"] = bool(np.isfinite(mi_p) and mi_p < 0.05)
                    mi_full_frames.append(mi_curve)

                    te_table = transfer_entropy_by_lag(
                        x, y, max_lag=max_lag, base=2.0,
                        n_bins=n_bins, disc=disc, min_samples=min_samples,
                    )
                    te_test = temporal_max_statistic_test(
                        x, y, lags=range(1, max_lag + 1),
                        statistic="te_norm", base=2.0, n_bins=n_bins,
                        disc=disc, min_samples=min_samples, n_surrogates=nperm,
                        surrogate_type=surrogate_type, block_size=block_size,
                        seed=5200 + idx_driver,
                    )
                    te_peak = te_table.loc[te_table["te_norm"].idxmax()] if te_table["te_norm"].notna().any() else None
                    te_p = float(te_test["p_value"]) if np.isfinite(te_test["p_value"]) else np.nan
                    te_rows.append({
                        "result_type": "te_peak", "driver": driver,
                        "source": driver, "target": yn,
                        "peak_lag": te_test["peak_lag"],
                        "peak_te_bits": float(te_peak["te_bits"]) if te_peak is not None else np.nan,
                        "target_cond_entropy_bits": float(te_peak["target_cond_entropy_bits"]) if te_peak is not None else np.nan,
                        "peak_te_norm": te_test["observed_peak"],
                        "n_samples_peak": int(te_peak["n_samples"]) if te_peak is not None else 0,
                        "p_peak_maxstat": te_p,
                        "null_max_95": te_test["null_max_95"],
                        "supported_alpha": bool(np.isfinite(te_p) and te_p < alpha),
                        "supported_0_05": bool(np.isfinite(te_p) and te_p < 0.05),
                    })

                    # Reuse the driver→target TE test in the network so panels b and c
                    # cannot disagree because of separate surrogate random seeds.
                    directions = [(driver, yn, x, y, te_table, te_test, "driver_to_target")]
                    if bidirectional:
                        reverse_table = transfer_entropy_by_lag(
                            y, x, max_lag=max_lag, base=2.0,
                            n_bins=n_bins, disc=disc, min_samples=min_samples,
                        )
                        reverse_test = temporal_max_statistic_test(
                            y, x, lags=range(1, max_lag + 1),
                            statistic="te_norm", base=2.0, n_bins=n_bins,
                            disc=disc, min_samples=min_samples, n_surrogates=nperm,
                            surrogate_type=surrogate_type, block_size=block_size,
                            seed=7200 + idx_driver,
                        )
                        directions.append((yn, driver, y, x, reverse_table, reverse_test, "target_to_driver"))

                    for src_name, tar_name, src_arr, tar_arr, direction_table, edge_test, curve_role in directions:
                        edge_p = float(edge_test["p_value"]) if np.isfinite(edge_test["p_value"]) else np.nan
                        edge_supported = bool(np.isfinite(edge_p) and edge_p < alpha)
                        edge_rows.append({
                            "result_type": "te_network_edge", "driver": driver,
                            "source": src_name, "target": tar_name,
                            "peak_lag": edge_test["peak_lag"],
                            "value": edge_test["observed_peak"],
                            "peak_te_norm": edge_test["observed_peak"],
                            "p_peak_maxstat": edge_p,
                            "null_max_95": edge_test["null_max_95"],
                            "supported_alpha": edge_supported,
                            "supported_0_05": bool(np.isfinite(edge_p) and edge_p < 0.05),
                        })

                        te_curve = direction_table.copy()
                        te_curve.insert(0, "result_type", "transfer_entropy_curve")
                        te_curve.insert(1, "driver", driver)
                        te_curve.insert(2, "source", src_name)
                        te_curve.insert(3, "target", tar_name)
                        te_curve.insert(4, "curve_role", curve_role)
                        te_curve["is_peak"] = te_curve["lag"].eq(edge_test["peak_lag"])
                        te_curve["peak_lag"] = edge_test["peak_lag"]
                        te_curve["p_peak_maxstat"] = edge_p
                        te_curve["null_max_95"] = edge_test["null_max_95"]
                        te_curve["supported_alpha"] = edge_supported
                        te_curve["supported_0_05"] = bool(np.isfinite(edge_p) and edge_p < 0.05)
                        te_full_frames.append(te_curve)

                # Temporal PID: pair the source-specific TE peak lags and condition
                # the decomposition on the previous target state. This adds a
                # genuinely multivariate temporal diagnostic rather than another
                # rendering of pairwise TE.
                te_peak_lag_by_driver = {
                    str(row["driver"]): int(row["peak_lag"])
                    for row in te_rows
                    if np.isfinite(row.get("peak_lag", np.nan))
                }
                pid_bootstrap_reps = int(max(100, min(500, nperm)))
                pid_rng = np.random.default_rng(8600)
                for i in range(len(drivers)):
                    for j in range(i + 1, len(drivers)):
                        d1, d2 = drivers[i], drivers[j]
                        lag1 = max(1, int(te_peak_lag_by_driver.get(d1, 1)))
                        lag2 = max(1, int(te_peak_lag_by_driver.get(d2, 1)))
                        x1_past = temporal[d1].shift(lag1).to_numpy(dtype=float)
                        x2_past = temporal[d2].shift(lag2).to_numpy(dtype=float)
                        y_present = temporal[yn].to_numpy(dtype=float)
                        y_past = temporal[yn].shift(1).to_numpy(dtype=float)
                        pid_obs = conditional_temporal_pid_min_information(
                            x1_past, x2_past, y_present, y_past,
                            base=2.0, n_bins=n_bins, disc=disc,
                            min_samples=min_samples,
                            min_condition_samples=max(8, n_bins * 2),
                        )
                        if not np.isfinite(pid_obs.get("itot_bits", np.nan)):
                            continue

                        boot_records = []
                        n_grid = len(y_present)
                        for b in range(pid_bootstrap_reps):
                            idx = moving_block_bootstrap_indices(n_grid, block_size, pid_rng)
                            boot = conditional_temporal_pid_min_information(
                                x1_past[idx], x2_past[idx], y_present[idx], y_past[idx],
                                base=2.0, n_bins=n_bins, disc=disc,
                                min_samples=min_samples,
                                min_condition_samples=max(8, n_bins * 2),
                            )
                            if not np.isfinite(boot.get("itot_bits", np.nan)):
                                continue
                            rec = {
                                "result_type": "temporal_pid_bootstrap",
                                "driver_1": d1,
                                "driver_2": d2,
                                "target": yn,
                                "lag_driver_1": lag1,
                                "lag_driver_2": lag2,
                                "bootstrap_replicate": b + 1,
                            }
                            rec.update(boot)
                            boot_records.append(rec)
                        pid_bootstrap_rows.extend(boot_records)

                        row = {
                            "result_type": "temporal_pid_pair",
                            "driver": f"{d1}|{d2}",
                            "driver_1": d1,
                            "driver_2": d2,
                            "source": d1,
                            "second_source": d2,
                            "target": yn,
                            "lag_driver_1": lag1,
                            "lag_driver_2": lag2,
                            "target_memory_lag": 1,
                            "pid_estimator": "stratified_conditional_Imin",
                            "pid_condition": f"{yn}(t-1)",
                            "n_bootstrap_requested": pid_bootstrap_reps,
                            "n_bootstrap_valid": len(boot_records),
                        }
                        row.update(pid_obs)
                        for metric in (
                            "redundant_bits", "unique_driver_1_bits", "unique_driver_2_bits",
                            "synergy_bits", "itot_bits", "redundant_frac",
                            "unique_driver_1_frac", "unique_driver_2_frac",
                            "unique_total_frac", "synergy_frac",
                        ):
                            vals = np.asarray([r.get(metric, np.nan) for r in boot_records], dtype=float)
                            vals = vals[np.isfinite(vals)]
                            row[f"{metric}_ci_low"] = float(np.quantile(vals, 0.025)) if vals.size else np.nan
                            row[f"{metric}_ci_high"] = float(np.quantile(vals, 0.975)) if vals.size else np.nan
                        fractions = {
                            "Redundancy": row.get("redundant_frac", np.nan),
                            f"Unique {_display_var_name(d1)}": row.get("unique_driver_1_frac", np.nan),
                            f"Unique {_display_var_name(d2)}": row.get("unique_driver_2_frac", np.nan),
                            "Synergy": row.get("synergy_frac", np.nan),
                        }
                        finite_fracs = {k: v for k, v in fractions.items() if np.isfinite(v)}
                        row["dominant_component"] = max(finite_fracs, key=finite_fracs.get) if finite_fracs else ""
                        pid_rows.append(row)

                common_meta = {
                    "site": site_name, "n_bins": n_bins, "disc": disc,
                    "max_lag": max_lag, "min_samples_per_lag": min_samples,
                    "surrogate_type": surrogate_type,
                    "n_surrogates": nperm, "surrogate_block_size": block_size,
                    "support_alpha": alpha,
                    "bidirectional_network": bidirectional,
                    "temporal_frequency": temporal_meta["temporal_frequency"],
                    "temporal_resolution": temporal_meta.get("temporal_resolution", fig6_resolution_var.get()),
                    "lag_unit": temporal_meta.get("lag_unit", "time steps"),
                    "minimum_valid_coverage_percent": temporal_meta.get("minimum_valid_coverage_percent", np.nan),
                    "aggregation_rules": temporal_meta.get("aggregation_rules", ""),
                    "temporal_rows_regular_grid": temporal_meta["temporal_rows_regular_grid"],
                    "temporal_rows_inserted_as_missing": temporal_meta["temporal_rows_inserted_as_missing"],
                }
                tables = []
                for records in (mi_rows, te_rows, edge_rows, pid_rows):
                    frame = pd.DataFrame(records)
                    for k, v in common_meta.items():
                        frame[k] = v
                    tables.append(frame)
                last_results_df = pd.concat(tables, ignore_index=True, sort=False)

                last_figure6_lagged_mi_full_df = (
                    pd.concat(mi_full_frames, ignore_index=True, sort=False)
                    if mi_full_frames else pd.DataFrame()
                )
                last_figure6_transfer_entropy_full_df = (
                    pd.concat(te_full_frames, ignore_index=True, sort=False)
                    if te_full_frames else pd.DataFrame()
                )
                last_figure6_temporal_pid_df = pd.DataFrame(pid_rows)
                last_figure6_temporal_pid_bootstrap_df = pd.DataFrame(pid_bootstrap_rows)
                for full_frame in (
                    last_figure6_lagged_mi_full_df,
                    last_figure6_transfer_entropy_full_df,
                    last_figure6_temporal_pid_df,
                    last_figure6_temporal_pid_bootstrap_df,
                ):
                    if full_frame is not None and not full_frame.empty:
                        for k, v in common_meta.items():
                            full_frame[k] = v

                save_results_btn.configure(state=tk.NORMAL)

                edge_df = pd.DataFrame(edge_rows)
                last_network_edges = pd.DataFrame({
                    "source": edge_df["source"],
                    "target": edge_df["target"],
                    "metric": "peak_TE_norm",
                    "value": edge_df["peak_te_norm"],
                    "lag": edge_df["peak_lag"],
                    "p_value": edge_df["p_peak_maxstat"],
                    "significant": np.where(edge_df["supported_alpha"], "yes", "no"),
                })
                last_vars = list(dict.fromkeys(drivers + [yn]))
                viewer_edges = (
                    last_network_edges[last_network_edges["significant"] == "yes"]
                    if supported_only else last_network_edges
                )
                _draw_network_viewer(viewer_edges, last_vars)

                mi_plot = pd.DataFrame(mi_rows).sort_values("peak_mi_norm", ascending=False).reset_index(drop=True)
                te_plot = pd.DataFrame(te_rows).sort_values("peak_te_norm", ascending=False).reset_index(drop=True)
                pid_plot = pd.DataFrame(pid_rows).copy()
                sig_edges = edge_df[edge_df["supported_alpha"]].copy()

                fig.clf()
                fig.set_size_inches(16.6, 10.6)
                gs = fig.add_gridspec(
                    3, 3,
                    width_ratios=[1.18, 1.12, 1.12],
                    height_ratios=[1.02, 1.02, 1.42],
                    hspace=1.02,
                    wspace=0.72,
                )
                ax_a = fig.add_subplot(gs[0, 0])
                ax_b = fig.add_subplot(gs[1, 0])
                ax_c = fig.add_subplot(gs[0:2, 1:3])
                ax_d = fig.add_subplot(gs[2, :])

                # Panel A: full normalized lagged-MI structure.
                mi_curve_df = last_figure6_lagged_mi_full_df.copy()
                mi_lags = list(range(-max_lag, max_lag + 1))
                mi_heat = mi_curve_df.pivot_table(index="driver", columns="lag", values="mi_norm", aggfunc="first")
                mi_heat = mi_heat.reindex(index=drivers, columns=mi_lags)
                im_a = ax_a.imshow(mi_heat.to_numpy(dtype=float), aspect="auto", origin="upper", cmap="Blues")
                ax_a.set_yticks(np.arange(len(drivers)))
                ax_a.set_yticklabels([_display_var_name(v) for v in drivers], fontsize=9)
                tick_lags = [v for v in mi_lags if v % 7 == 0]
                ax_a.set_xticks([mi_lags.index(v) for v in tick_lags])
                ax_a.set_xticklabels([f"{v:+d}" for v in tick_lags], fontsize=8)
                ax_a.set_xlabel(f"Lag ({temporal_meta.get('lag_unit', 'time steps')}); positive = driver leads")
                ax_a.set_title("a  Lagged MI structure", loc="left", fontweight="bold", fontsize=13, pad=10)
                for r, d in enumerate(drivers):
                    peak = next((row for row in mi_rows if row["driver"] == d), None)
                    if peak and np.isfinite(peak.get("peak_lag", np.nan)):
                        c = mi_lags.index(int(peak["peak_lag"]))
                        ax_a.scatter(c, r, marker="o", s=58, facecolors="none",
                                     edgecolors="black" if peak["supported_alpha"] else "0.55",
                                     linewidths=1.3)
                cb_a = fig.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.03)
                cb_a.set_label("Normalized MI", fontsize=9)
                cb_a.ax.tick_params(labelsize=8)

                # Panel B: full driver-to-target TE structure.
                te_curve_df = last_figure6_transfer_entropy_full_df.copy()
                te_curve_df = te_curve_df[te_curve_df["curve_role"] == "driver_to_target"]
                te_lags = list(range(1, max_lag + 1))
                te_heat = te_curve_df.pivot_table(index="driver", columns="lag", values="te_norm", aggfunc="first")
                te_heat = te_heat.reindex(index=drivers, columns=te_lags)
                im_b = ax_b.imshow(te_heat.to_numpy(dtype=float), aspect="auto", origin="upper", cmap="Greens")
                ax_b.set_yticks(np.arange(len(drivers)))
                ax_b.set_yticklabels([_display_var_name(v) for v in drivers], fontsize=9)
                te_ticks = [v for v in te_lags if v in {1, 3, 7, 10, 14} or v == max_lag]
                te_ticks = sorted(set(te_ticks))
                ax_b.set_xticks([te_lags.index(v) for v in te_ticks])
                ax_b.set_xticklabels([str(v) for v in te_ticks], fontsize=8)
                ax_b.set_xlabel(f"Source lag ({temporal_meta.get('lag_unit', 'time steps')})")
                ax_b.set_title("b  Transfer-entropy structure", loc="left", fontweight="bold", fontsize=13, pad=10)
                for r, d in enumerate(drivers):
                    peak = next((row for row in te_rows if row["driver"] == d), None)
                    if peak and np.isfinite(peak.get("peak_lag", np.nan)):
                        c = te_lags.index(int(peak["peak_lag"]))
                        ax_b.scatter(c, r, marker="o", s=58, facecolors="none",
                                     edgecolors="black" if peak["supported_alpha"] else "0.55",
                                     linewidths=1.3)
                cb_b = fig.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.03)
                cb_b.set_label("Normalized TE", fontsize=9)
                cb_b.ax.tick_params(labelsize=8)

                # Panel C: large supported directional network.
                from matplotlib.patches import FancyArrowPatch
                display_drivers = [_display_var_name(v) for v in drivers]
                nodes = drivers + [yn]
                if len(drivers) == 4:
                    role_lookup = {display_drivers[i]: drivers[i] for i in range(len(drivers))}
                    pos = {}
                    def _assign_if_present(label_variants, coord):
                        for lbl in label_variants:
                            if lbl in role_lookup:
                                pos[role_lookup[lbl]] = coord
                                return
                    _assign_if_present(["TA"], (-1.55, 0.0))
                    _assign_if_present(["NETRAD", "SW_IN", "PPFD", "SW"], (1.55, 0.0))
                    _assign_if_present(["RH", "VPD"], (0.0, 1.32))
                    _assign_if_present(["USTAR"], (0.0, -1.32))
                    remaining = [d for d in drivers if d not in pos]
                    fallback = [(-1.55, 0.0), (1.55, 0.0), (0.0, 1.32), (0.0, -1.32)]
                    for d, coord in zip(remaining, [c for c in fallback if c not in pos.values()]):
                        pos[d] = coord
                else:
                    theta = np.linspace(0, 2 * np.pi, len(drivers), endpoint=False)
                    pos = {d: (1.55 * np.cos(theta[i]), 1.25 * np.sin(theta[i])) for i, d in enumerate(drivers)}
                pos[yn] = (0.0, 0.0)

                if not sig_edges.empty:
                    pairs = set((str(r.source), str(r.target)) for r in sig_edges.itertuples(index=False))
                    max_edge = float(sig_edges["peak_te_norm"].max())
                    if not np.isfinite(max_edge) or max_edge <= 0:
                        max_edge = 1.0
                    for row in sig_edges.itertuples(index=False):
                        src = str(row.source); tar = str(row.target)
                        x1, y1 = pos[src]; x2, y2 = pos[tar]
                        has_reverse = (tar, src) in pairs
                        rad = 0.22 if has_reverse and src < tar else (-0.22 if has_reverse else 0.0)
                        width = 1.3 + 5.5 * float(row.peak_te_norm) / max_edge
                        arrow = FancyArrowPatch(
                            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18,
                            linewidth=width, alpha=0.74, color="#2f3e46",
                            connectionstyle=f"arc3,rad={rad}", shrinkA=26, shrinkB=28,
                        )
                        ax_c.add_patch(arrow)
                        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                        if has_reverse:
                            dx, dy = (-(y2-y1), x2-x1)
                            scale = (dx**2 + dy**2) ** 0.5 or 1.0
                            sign = 1.0 if rad > 0 else -1.0
                            mx += 0.14 * sign * dx / scale
                            my += 0.14 * sign * dy / scale
                        ax_c.text(mx, my, f"lag {int(row.peak_lag)}", fontsize=9,
                                  ha="center", va="center",
                                  bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.72", alpha=0.96))

                palette = ["#3aaa35", "#2a84c9", "#f28e2b", "#e53935", "#17a398", "#b56576"]
                node_colors = {yn: "#9c77cf"}
                for i, d in enumerate(drivers):
                    node_colors[d] = palette[i % len(palette)]
                for node in nodes:
                    px, py = pos[node]
                    size = 2200 if node == yn else 1550
                    ax_c.scatter([px], [py], s=size, facecolors=node_colors[node],
                                 edgecolors="black", linewidths=1.2, zorder=3)
                    ax_c.text(px, py, _display_var_name(node), ha="center", va="center",
                              fontsize=10.6, zorder=4)
                if sig_edges.empty:
                    ax_c.text(0, -1.72, f"No links supported at p < {alpha:g}", ha="center", fontsize=10)
                ax_c.set_xlim(-2.1, 2.1); ax_c.set_ylim(-1.86, 1.86)
                ax_c.set_aspect("equal"); ax_c.set_axis_off()
                ax_c.set_title("c  Supported directional network", loc="center", fontweight="bold", fontsize=13, pad=12)

                # Panel D: target-memory-conditioned temporal PID.
                if not pid_plot.empty:
                    pid_plot = pid_plot.sort_values("synergy_frac", ascending=True).reset_index(drop=True)
                    yy = np.arange(len(pid_plot))
                    components = [
                        ("redundant_frac", "Redundancy", "#7b6fd0"),
                        ("unique_driver_1_frac", "Unique driver 1", "#3b82c4"),
                        ("unique_driver_2_frac", "Unique driver 2", "#f2a541"),
                        ("synergy_frac", "Synergy", "#d95f76"),
                    ]
                    left = np.zeros(len(pid_plot), dtype=float)
                    for col, label, color in components:
                        vals = pid_plot[col].fillna(0.0).to_numpy(dtype=float)
                        ax_d.barh(yy, vals, left=left, height=0.70, label=label,
                                  color=color, edgecolor="white", linewidth=0.7)
                        left += vals
                    pair_labels = [
                        f"{_display_var_name(r.driver_1)} + {_display_var_name(r.driver_2)}  "
                        f"(lags {int(r.lag_driver_1)}/{int(r.lag_driver_2)})"
                        for r in pid_plot.itertuples(index=False)
                    ]
                    ax_d.set_yticks(yy); ax_d.set_yticklabels(pair_labels, fontsize=9)
                    ax_d.set_xlim(0, 1.34)
                    ax_d.set_xlabel("Fraction of conditioned joint information")
                    ax_d.set_title(
                        f"d  Temporal PID given {_display_var_name(yn)}(t−1)",
                        loc="left", fontweight="bold", fontsize=13, pad=10,
                    )
                    ax_d.grid(True, axis="x", alpha=0.20)
                    ax_d.spines["top"].set_visible(False); ax_d.spines["right"].set_visible(False)
                    ax_d.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.08),
                                frameon=False, fontsize=8.5, columnspacing=1.2, handlelength=1.6)
                    for yi, row in enumerate(pid_plot.itertuples(index=False)):
                        lo = getattr(row, "synergy_frac_ci_low", np.nan)
                        hi = getattr(row, "synergy_frac_ci_high", np.nan)
                        ci_txt = f"S={row.synergy_frac:.2f} [{lo:.2f}, {hi:.2f}]" if np.isfinite(lo) and np.isfinite(hi) else f"S={row.synergy_frac:.2f}"
                        ax_d.text(1.02, yi, ci_txt, va="center", ha="left", fontsize=8.5)
                else:
                    ax_d.text(0.5, 0.5, "Temporal PID requires at least two valid drivers.",
                              ha="center", va="center", transform=ax_d.transAxes)
                    ax_d.set_axis_off()
                    ax_d.set_title("d  Temporal PID", loc="left", fontweight="bold", fontsize=14)
                fig.suptitle(
                    "Figure 6 temporal information structure",
                    fontsize=15, fontweight="semibold", y=0.982,
                )
                fig.text(
                    0.5, 0.953,
                    f"{site_name or 'site'} | target: {_display_var_name(yn)}",
                    ha="center", va="center", fontsize=11.2, color="0.20"
                )
                fig.subplots_adjust(top=0.89, bottom=0.07, left=0.18, right=0.985)
                canvas.draw_idle()

                supported_count = sum(bool(r["supported_alpha"]) for r in te_rows)
                _show(
                    f"Figure 6 temporal summary completed for {site_name or 'site'}; target={yn}.\n"
                    f"Drivers: {', '.join(drivers)}\n"
                    f"Prepared resolution: {temporal_meta.get('temporal_resolution', 'Native')} | "
                    f"frequency: {temporal_meta['temporal_frequency']} | "
                    f"fully missing steps retained: {temporal_meta['temporal_rows_inserted_as_missing']}\n"
                    f"Lag window: MI {-max_lag}..{max_lag}; TE 1..{max_lag} "
                    f"({temporal_meta.get('lag_unit', 'time steps')})\n"
                    f"Minimum aligned samples per lag: {min_samples}\n"
                    f"Surrogates: {nperm} {surrogate_type}; maximum-statistic test across lags.\n"
                    f"Support threshold: alpha={alpha:g}. Bidirectional network: {bidirectional}.\n"
                    f"Supported driver→target TE peaks: {supported_count} of {len(te_rows)}.\n"
                    f"Full curves retained for export: {len(last_figure6_lagged_mi_full_df)} MI rows and "
                    f"{len(last_figure6_transfer_entropy_full_df)} directional TE rows.\n"
                    f"Temporal PID retained for {len(last_figure6_temporal_pid_df)} driver pairs "
                    f"with {len(last_figure6_temporal_pid_bootstrap_df)} valid bootstrap replicates.\n"
                )

            elif m == "Conditional MI I(X;Y|Z)":
                x, y, z, xn, yn, zn = _get_xyz()
                xi = _discretize(x)
                yi = _discretize(y)
                zi = _discretize(z)
                cmi = conditional_mutual_information_discrete(xi, yi, zi, base=2.0)
                last_results_df = pd.DataFrame([{
                    "method": "Conditional MI I(X;Y|Z)",
                    "x": xn,
                    "y": yn,
                    "z": zn,
                    "cmi_bits": float(cmi),
                    "n_bins": int(n_bins),
                    "disc": disc,
                    "n_samples": int(len(x)),
                }])
                save_results_btn.configure(state=tk.NORMAL)
                _show(
                    f"I({xn}; {yn} | {zn}) ≈ {cmi:.6f} bits "
                    f"[discrete, bins={n_bins}, {disc}]\n"
                )

                ax.cla()
                ax.scatter(x, y, s=6, alpha=0.6)
                ax.set_title(f"Pairwise view: {xn} vs {yn}")
                ax.set_xlabel(xn)
                ax.set_ylabel(yn)
                canvas.draw_idle()

            elif m == "Lagged MI I(X_t;Y_{t+lag})":
                xn, yn = x_var.get(), y_var.get()
                max_lag = int(maxlag_var.get())
                if max_lag <= 0:
                    raise ValueError("Max |lag| must be > 0.")
                temporal, temporal_meta = _get_temporal_frame([xn, yn])
                last_metadata.update(temporal_meta)
                x = temporal[xn].to_numpy(dtype=float)
                y = temporal[yn].to_numpy(dtype=float)
                lags = list(range(-max_lag, max_lag + 1))

                lag_df = lagged_mutual_information_table(
                    x, y, lags=lags, base=2.0, method="hist",
                    n_bins=n_bins, disc=disc, min_samples=20,
                )
                lag_df.insert(0, "source", xn)
                lag_df.insert(1, "target", yn)
                test = temporal_max_statistic_test(
                    x, y, lags=lags, statistic="lagged_mi_norm",
                    base=2.0, n_bins=n_bins, disc=disc, min_samples=20,
                    n_surrogates=nperm, surrogate_type=surrogate_type,
                    block_size=int(block_size_var.get()), seed=42,
                )
                lag_df["is_peak"] = lag_df["lag"] == test["peak_lag"]
                lag_df["p_peak_maxstat"] = test["p_value"]
                lag_df["null_max_95"] = test["null_max_95"]
                lag_df["peak_supported_0_05"] = bool(
                    np.isfinite(test["p_value"]) and test["p_value"] < 0.05
                )
                lag_df["surrogate_type"] = surrogate_type if nperm > 0 else ""
                lag_df["n_surrogates"] = int(nperm)
                lag_df["temporal_frequency"] = temporal_meta["temporal_frequency"]
                last_results_df = lag_df.copy()
                save_results_btn.configure(state=tk.NORMAL)

                peak_text = (
                    f"Peak normalized MI at lag {test['peak_lag']}: {test['observed_peak']:.6f}; "
                    f"max-statistic p={test['p_value']:.4f}"
                    if np.isfinite(test["observed_peak"])
                    else "No valid lagged-MI peak."
                )
                _show(
                    f"Timestamp-preserving lagged MI: {xn} → {yn}\n"
                    f"Temporal grid: {temporal_meta['temporal_frequency']} | "
                    f"inserted missing steps: {temporal_meta['temporal_rows_inserted_as_missing']}\n"
                    f"{peak_text}\n\n"
                    + lag_df.to_string(index=False, float_format=lambda v: f"{v:.6f}")
                    + "\n"
                )

                ax.cla()
                Ls = lag_df["lag"].to_numpy(dtype=int)
                vals = lag_df["mi_norm"].to_numpy(dtype=float)
                ax.plot(Ls, vals, linewidth=2.0)
                ax.fill_between(Ls, 0, vals, alpha=0.12)
                ax.axvline(0, linestyle="--", linewidth=1, color="0.45")
                if np.isfinite(vals).any():
                    k = int(np.nanargmax(vals))
                    ax.scatter([Ls[k]], [vals[k]], s=72, zorder=4)
                    label = f"Peak: lag {Ls[k]}\n{vals[k]:.3f}"
                    if np.isfinite(test["p_value"]):
                        label += f"\np={test['p_value']:.3f}"
                    ax.annotate(label, xy=(Ls[k], vals[k]), xytext=(8, 12),
                                textcoords="offset points", fontsize=9,
                                arrowprops=dict(arrowstyle="-", color="0.4"))
                if np.isfinite(test["null_max_95"]):
                    ax.axhline(test["null_max_95"], linestyle=":", linewidth=1.2,
                               label="95% surrogate maximum")
                    ax.legend(frameon=False)
                ax.set_xlabel("Lag (time steps; positive = source leads target)")
                ax.set_ylabel("Normalized mutual information")
                ax.set_title("Timestamp-preserving lagged mutual information",
                             fontsize=14, fontweight="semibold", pad=14)
                ax.text(0.0, 1.01, f"{xn} → {yn}", transform=ax.transAxes,
                        ha="left", va="bottom", fontsize=9, color="0.35")
                ax.grid(True, alpha=0.25)
                for spine in ("top", "right"):
                    ax.spines[spine].set_visible(False)
                fig.tight_layout(pad=1.6)
                canvas.draw_idle()

            elif m == "PID (X1,X2→Y)":
                x, y, z, xn, yn, zn = _get_xyz()
                if estimator == "kde-tip":
                    info_tip = kde_tip_pid_3d(
                        x, y, z,
                        N=n_bins,
                        bin_scheme="global",
                        method="KDE",
                    )
                    R = info_tip["R"]
                    S = info_tip["S"]
                    U1 = info_tip["U1"]
                    U2 = info_tip["U2"]
                    label_source = "TIP / KDE"
                else:
                    res = pid_min_information(
                        x, y, z,
                        base=2.0,
                        method="hist",
                        n_bins=n_bins,
                        disc=disc,
                    )
                    R = res["redundant"]
                    S = res["synergy"]
                    U1 = res["unique_x1"]
                    U2 = res["unique_x2"]
                    label_source = "Min-information PID (hist)"

                last_results_df = pd.DataFrame([{
                    "method": "PID (X1,X2→Y)",
                    "driver_1": xn,
                    "driver_2": yn,
                    "target": zn,
                    "redundant": float(R),
                    "unique_x1": float(U1),
                    "unique_x2": float(U2),
                    "synergy": float(S),
                    "estimator": label_source,
                    "n_bins": int(n_bins),
                    "disc": disc,
                    "n_samples": int(len(x)),
                }])
                save_results_btn.configure(state=tk.NORMAL)

                lines = [
                    f"PID for drivers ({xn}, {yn}) → target {zn}",
                    f"Estimator: {label_source}",
                    f"Redundant = {R:.6f}",
                    f"Unique_{xn} = {U1:.6f}",
                    f"Unique_{yn} = {U2:.6f}",
                    f"Synergy = {S:.6f}",
                ]
                _show("\n".join(lines) + "\n")

                ax.cla()
                labels = ["Redundant", f"Unique {xn}", f"Unique {yn}", "Synergy"]
                vals = [R, U1, U2, S]
                idx = np.arange(len(labels))
                ax.bar(idx, vals)
                ax.set_xticks(idx)
                ax.set_xticklabels(labels, rotation=20, ha="right")
                ax.set_ylabel("Bits")
                ax.set_title(f"PID: {xn}, {yn} → {zn}")
                canvas.draw_idle()

            elif m == "Pairwise PID Matrix":
                # Target is selected in Y; drivers are selected from the variable pool.
                yn = y_var.get()
                selected = [net_list.get(i) for i in net_list.curselection()]
                drivers = _selected_or_default_drivers(selected, all_cols, yn, max_drivers=10)

                pid_table, pid_mats, pid_vars = pairwise_pid_matrix(
                    df_runtime["active"],
                    target=yn,
                    drivers=drivers,
                    base=2.0,
                    estimator=estimator,
                    n_bins=n_bins,
                    disc=disc,
                    min_samples=50,
                )
                last_results_df = pid_table.copy()
                save_results_btn.configure(state=tk.NORMAL)

                used = (
                    f"kde-tip (N={n_bins})"
                    if estimator == "kde-tip"
                    else f"hist (bins={n_bins}, {disc})"
                )

                display_cols = [
                    "driver_1", "driver_2", "redundant_frac",
                    "unique_total_frac", "synergy_frac", "itot_bits", "n_samples",
                ]
                lines = [
                    f"Pairwise PID matrix for target {yn} [{used}]",
                    "Heatmaps show fractions of total pairwise information.",
                    "The results table also stores raw bit values.",
                    f"Drivers evaluated: {len(pid_vars)}; pairs computed: {len(pid_table)}",
                    "",
                    pid_table[display_cols].to_string(
                        index=False,
                        float_format=lambda v: f"{v:.6f}",
                    ),
                ]
                _show("\n".join(lines) + "\n")

                fig.clf()
                axes = fig.subplots(1, 3)

                components = [
                    ("redundant_frac", "Redundancy R"),
                    ("unique_total_frac", "Unique U"),
                    ("synergy_frac", "Synergy S"),
                ]

                last_im = None
                for ax_i, (key, title_i) in zip(axes, components):
                    mat = pid_mats[key]
                    plot_mat = np.clip(mat, 0.0, 1.0)
                    last_im = ax_i.imshow(
                        plot_mat,
                        aspect="equal",
                        origin="upper",
                        vmin=0.0,
                        vmax=1.0,
                        cmap="YlGn",
                    )
                    ax_i.set_xticks(range(len(pid_vars)))
                    ax_i.set_yticks(range(len(pid_vars)))
                    ax_i.set_xticklabels(pid_vars, rotation=45, ha="right", fontsize=8)
                    ax_i.set_yticklabels(pid_vars, fontsize=8)
                    ax_i.set_title(title_i)

                    # Keep the upper triangle visually inactive.
                    ax_i.set_xlim(-0.5, len(pid_vars) - 0.5)
                    ax_i.set_ylim(len(pid_vars) - 0.5, -0.5)

                if last_im is not None:
                    fig.colorbar(
                        last_im,
                        ax=axes.ravel().tolist(),
                        fraction=0.046,
                        pad=0.04,
                        label="Fraction of total information",
                    )

                fig.suptitle(f"Pairwise PID matrix → {yn}", y=0.98)
                fig.tight_layout()
                canvas.draw_idle()

            elif m == "Transfer Entropy TE(X→Y)":
                xn, yn = x_var.get(), y_var.get()
                lag = int(delay_var.get())
                if lag < 1:
                    raise ValueError("TE lag must be ≥ 1.")
                temporal, temporal_meta = _get_temporal_frame([xn, yn])
                last_metadata.update(temporal_meta)
                x = temporal[xn].to_numpy(dtype=float)
                y = temporal[yn].to_numpy(dtype=float)
                comp = transfer_entropy_components(
                    x, y, lag=lag, base=2.0, n_bins=n_bins, disc=disc,
                    min_samples=20,
                )
                p = np.nan
                if nperm > 0:
                    p = source_surrogate_p_value(
                        lambda a, b: transfer_entropy_components(
                            a, b, lag=lag, base=2.0, n_bins=n_bins,
                            disc=disc, min_samples=20,
                        )["te_norm"],
                        x, y, observed_value=comp["te_norm"], n_perm=nperm,
                        surrogate_type=surrogate_type,
                        block_size=int(block_size_var.get()), seed=42,
                    )
                last_results_df = pd.DataFrame([{
                    "method": "Transfer Entropy TE(X→Y)",
                    "source": xn, "target": yn, "lag": lag,
                    **comp,
                    "p_value": p,
                    "significant_0_05": bool(np.isfinite(p) and p < 0.05),
                    "n_bins": n_bins, "disc": disc,
                    "surrogate_type": surrogate_type if nperm > 0 else "",
                    "n_surrogates": nperm,
                    "temporal_frequency": temporal_meta["temporal_frequency"],
                }])
                save_results_btn.configure(state=tk.NORMAL)
                _show(
                    f"Timestamp-preserving TE({xn}→{yn}), lag={lag}\n"
                    f"TE = {comp['te_bits']:.6f} bits\n"
                    f"Normalized TE = {comp['te_norm']:.6f}\n"
                    f"Target remaining entropy = {comp['target_cond_entropy_bits']:.6f} bits\n"
                    f"Source-surrogate p = {p:.4f} ({surrogate_type}, n={nperm})\n"
                    f"Temporal grid = {temporal_meta['temporal_frequency']}\n"
                )

                ax.cla()
                ax.plot(y, label=yn, lw=1.0)
                ax.plot(x, label=xn, lw=1.0, alpha=0.7)
                ax.set_title(f"Regular-grid series: target {yn} and source {xn}")
                ax.set_xlabel("Regular time-step index")
                ax.legend(frameon=False)
                canvas.draw_idle()

            elif m == "Transfer Entropy vs Lag":
                xn, yn = x_var.get(), y_var.get()
                max_lag = int(maxlag_var.get())
                if max_lag < 1:
                    raise ValueError("Max |lag| must be >= 1 for TE vs lag.")
                temporal, temporal_meta = _get_temporal_frame([xn, yn])
                last_metadata.update(temporal_meta)
                x = temporal[xn].to_numpy(dtype=float)
                y = temporal[yn].to_numpy(dtype=float)

                te_df = transfer_entropy_by_lag(
                    x, y, max_lag=max_lag, base=2.0,
                    n_bins=n_bins, disc=disc, min_samples=20,
                )
                test = temporal_max_statistic_test(
                    x, y, lags=range(1, max_lag + 1), statistic="te_norm",
                    base=2.0, n_bins=n_bins, disc=disc, min_samples=20,
                    n_surrogates=nperm, surrogate_type=surrogate_type,
                    block_size=int(block_size_var.get()), seed=42,
                )
                te_df.insert(0, "source", xn)
                te_df.insert(1, "target", yn)
                te_df["is_peak"] = te_df["lag"] == test["peak_lag"]
                te_df["p_peak_maxstat"] = test["p_value"]
                te_df["null_max_95"] = test["null_max_95"]
                te_df["peak_supported_0_05"] = bool(
                    np.isfinite(test["p_value"]) and test["p_value"] < 0.05
                )
                te_df["surrogate_type"] = surrogate_type if nperm > 0 else ""
                te_df["n_surrogates"] = int(nperm)
                te_df["temporal_frequency"] = temporal_meta["temporal_frequency"]
                last_results_df = te_df.copy()
                save_results_btn.configure(state=tk.NORMAL)

                _show(
                    f"Timestamp-preserving transfer entropy by lag: {xn} → {yn}\n"
                    f"Temporal grid: {temporal_meta['temporal_frequency']} | "
                    f"inserted missing steps: {temporal_meta['temporal_rows_inserted_as_missing']}\n"
                    f"Peak normalized TE = {test['observed_peak']:.6f} at lag {test['peak_lag']}; "
                    f"maximum-statistic p={test['p_value']:.4f}\n\n"
                    + te_df.to_string(index=False, float_format=lambda v: f"{v:.6f}")
                    + "\n"
                )

                ax.cla()
                lag_vals = te_df["lag"].to_numpy(dtype=float)
                te_vals = te_df["te_norm"].to_numpy(dtype=float)
                ax.plot(lag_vals, te_vals, linewidth=2.0)
                ax.fill_between(lag_vals, 0, te_vals, alpha=0.12)
                if np.isfinite(te_vals).any():
                    k = int(np.nanargmax(te_vals))
                    ax.scatter([lag_vals[k]], [te_vals[k]], s=72, zorder=4)
                    label = f"Peak: lag {int(lag_vals[k])}\n{te_vals[k]:.3f}"
                    if np.isfinite(test["p_value"]):
                        label += f"\np={test['p_value']:.3f}"
                    ax.annotate(label, xy=(lag_vals[k], te_vals[k]),
                                xytext=(8, 12), textcoords="offset points",
                                fontsize=9, arrowprops=dict(arrowstyle="-", color="0.4"))
                if np.isfinite(test["null_max_95"]):
                    ax.axhline(test["null_max_95"], linestyle=":", linewidth=1.2,
                               label="95% surrogate maximum")
                    ax.legend(frameon=False)
                ax.set_xlabel("Lag (time steps; source precedes target)")
                ax.set_ylabel("Normalized transfer entropy")
                ax.set_title("Timestamp-preserving transfer entropy across lags",
                             fontsize=14, fontweight="semibold", pad=14)
                ax.text(0.0, 1.01, f"{xn} → {yn}", transform=ax.transAxes,
                        ha="left", va="bottom", fontsize=9, color="0.35")
                ax.grid(True, alpha=0.25)
                for spine in ("top", "right"):
                    ax.spines[spine].set_visible(False)
                fig.tight_layout(pad=1.6)
                canvas.draw_idle()

            elif m == "TE Network":
                sel = [net_list.get(i) for i in net_list.curselection()]
                vars_ = sel if sel else all_cols
                if len(vars_) < 2:
                    messagebox.showwarning("Need variables", "Select at least two variables.")
                    return
                temporal, temporal_meta = _get_temporal_frame(vars_)
                last_metadata.update(temporal_meta)
                data = temporal[list(vars_)]
                p = len(vars_)
                mat = np.full((p, p), np.nan, dtype=float)
                raw_mat = np.full((p, p), np.nan, dtype=float)
                pmat = np.ones((p, p), dtype=float)
                lag = int(delay_var.get())
                if lag < 1:
                    raise ValueError("TE lag must be ≥ 1.")

                edge_rows = []
                for i in range(p):
                    for j in range(p):
                        if i == j:
                            continue
                        xi = data.iloc[:, i].to_numpy(dtype=float)
                        yj = data.iloc[:, j].to_numpy(dtype=float)
                        comp = transfer_entropy_components(
                            xi, yj, lag=lag, base=2.0,
                            n_bins=n_bins, disc=disc, min_samples=20,
                        )
                        mat[i, j] = comp["te_norm"]
                        raw_mat[i, j] = comp["te_bits"]
                        pval = np.nan
                        if nperm > 0 and np.isfinite(comp["te_norm"]):
                            pval = source_surrogate_p_value(
                                lambda a, b: transfer_entropy_components(
                                    a, b, lag=lag, base=2.0, n_bins=n_bins,
                                    disc=disc, min_samples=20,
                                )["te_norm"],
                                xi, yj, observed_value=comp["te_norm"],
                                n_perm=nperm, surrogate_type=surrogate_type,
                                block_size=int(block_size_var.get()), seed=42 + i * p + j,
                            )
                            pmat[i, j] = pval
                        edge_rows.append({
                            "source": vars_[i], "target": vars_[j], "lag": lag,
                            **comp, "p_value": pval,
                            "significant": "yes" if np.isfinite(pval) and pval < 0.05 else "no",
                            "surrogate_type": surrogate_type if nperm > 0 else "",
                            "n_surrogates": nperm,
                            "temporal_frequency": temporal_meta["temporal_frequency"],
                        })

                last_matrix = mat.copy()
                last_vars = list(vars_)
                last_results_df = pd.DataFrame(edge_rows).sort_values(
                    "te_norm", ascending=False, na_position="last"
                ).reset_index(drop=True)
                last_network_edges = pd.DataFrame({
                    "source": last_results_df["source"],
                    "target": last_results_df["target"],
                    "metric": "TE_norm",
                    "value": last_results_df["te_norm"],
                    "lag": last_results_df["lag"],
                    "p_value": last_results_df["p_value"],
                    "significant": last_results_df["significant"],
                })
                save_results_btn.configure(state=tk.NORMAL)

                _show(
                    f"Timestamp-preserving normalized TE network computed. shape={mat.shape}, "
                    f"lag={lag}, bins={n_bins}, {disc}, surrogates={nperm} ({surrogate_type}).\n"
                    f"Temporal grid: {temporal_meta['temporal_frequency']} | "
                    f"inserted missing steps: {temporal_meta['temporal_rows_inserted_as_missing']}\n"
                    f"Significant links (p<0.05): "
                    f"{int((last_results_df['significant'] == 'yes').sum())} of {len(last_results_df)}.\n"
                )

                ax.cla()
                im = ax.imshow(mat, aspect="auto", origin="upper", vmin=0)
                ax.set_xticks(range(p))
                ax.set_yticks(range(p))
                ax.set_xticklabels(vars_, rotation=30, ha="right")
                ax.set_yticklabels(vars_)
                ax.set_xlabel("Target")
                ax.set_ylabel("Source")
                ax.set_title("Normalized transfer-entropy network")
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("TE / H(target future | target past)")
                canvas.draw_idle()

                save_btn.configure(state=tk.NORMAL)
                _refresh_network_viewer()

            else:
                messagebox.showerror("Unknown measure", m)
                return

            progress_var.set(100)
            progress_status_var.set(f"Completed in {_time.perf_counter() - _run_started:.1f} s")
            run_btn.configure(state=tk.NORMAL)
            win.update_idletasks()
            _finalize_success(m)

        except Exception as e:
            progress_var.set(0)
            progress_status_var.set("Analysis failed")
            run_btn.configure(state=tk.NORMAL)
            messagebox.showerror("Analysis error", str(e))

    run_btn.configure(command=run_analysis)

    def save_matrix_csv():
        nonlocal last_matrix, last_vars
        if last_matrix is None or last_vars is None:
            messagebox.showinfo(
                "Nothing to save", "Run a TE Network first to create a matrix."
            )
            return
        fp = filedialog.asksaveasfilename(
            title="Save TE matrix CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if not fp:
            return
        try:
            pd.DataFrame(last_matrix, index=last_vars, columns=last_vars).to_csv(fp)
            messagebox.showinfo("Saved", f"Saved matrix to:\n{fp}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

    save_btn.configure(command=save_matrix_csv)

    def save_results_csv():
        nonlocal last_results_df
        if last_results_df is None or last_results_df.empty:
            messagebox.showinfo(
                "Nothing to save",
                "Run an analysis that produces a results table first."
            )
            return
        fp = filedialog.asksaveasfilename(
            title="Save results CSV",
            initialfile=f"{_default_export_base()}_data.csv",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if not fp:
            return
        try:
            last_results_df.to_csv(fp, index=False)
            messagebox.showinfo("Saved", f"Saved results table to:\n{fp}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

    save_results_btn.configure(command=save_results_csv)

    def save_prepared_temporal_series():
        nonlocal last_prepared_temporal_df
        if last_prepared_temporal_df is None or last_prepared_temporal_df.empty:
            messagebox.showinfo(
                "Nothing to save",
                "Run Figure 6 Temporal Summary first to create the prepared Native, Daily, or Weekly series."
            )
            return
        fp = filedialog.asksaveasfilename(
            title="Save prepared Figure 6 temporal series",
            initialfile=f"{_default_export_base()}_prepared_temporal_series.csv",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if not fp:
            return
        try:
            last_prepared_temporal_df.to_csv(fp, index=False)
            messagebox.showinfo("Saved", f"Saved prepared temporal series to:\n{fp}")
        except Exception as exc:
            messagebox.showerror("Save error", str(exc))

    save_prepared_btn.configure(command=save_prepared_temporal_series)

    def save_results_package():
        nonlocal last_results_df, last_metadata
        if last_results_df is None or last_results_df.empty:
            messagebox.showinfo("Nothing to save", "Run an analysis first to create results.")
            return
        fp = filedialog.asksaveasfilename(
            title="Save results CSV",
            initialfile=f"{_default_export_base()}_data.csv",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if not fp:
            return
        try:
            last_results_df.to_csv(fp, index=False)
            meta_fp = fp[:-4] + "_metadata.json" if fp.lower().endswith(".csv") else fp + "_metadata.json"
            with open(meta_fp, "w", encoding="utf-8") as f:
                json.dump(last_metadata, f, indent=2, default=str)
            messagebox.showinfo("Saved", f"Saved results to:\n{fp}\n\nSaved metadata to:\n{meta_fp}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

    save_package_btn.configure(command=save_results_package)
    
    def save_plot():
        fp = filedialog.asksaveasfilename(
            title="Save Plot",
            initialfile=f"{_default_export_base()}.png",
            defaultextension=".png",
            filetypes=[
                ("PNG Image", "*.png"),
                ("PDF Document", "*.pdf"),
                ("All Files", "*.*"),
            ],
        )
        if not fp:
            return
        try:
            fig.savefig(fp, dpi=300, bbox_inches="tight")
            messagebox.showinfo("Saved", f"Plot saved to:\n{fp}")
        except Exception as e:
            messagebox.showerror("Save error", f"Could not save plot:\n{e}")

    saveplot_btn.configure(command=save_plot)

    def _safe_file_tag(value: str) -> str:
        value = str(value or "analysis").strip().lower()
        value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
        return value or "analysis"

    def _method_export_tag(value: str) -> str:
        txt = str(value or "analysis").strip().lower()
        special = {
            "figure 6 temporal summary": "figure_6_temporal_summary",
        }
        return special.get(txt, _safe_file_tag(value))

    def _display_var_name(value: str) -> str:
        txt = str(value or "")
        txt = re.sub(r"_(?:\d+_){2}\d+$", "", txt)
        txt = re.sub(r"_(?:F|PI|QC|FLAG)$", "", txt)
        return txt or str(value)

    def _default_export_base() -> str:
        site_tag = _safe_file_tag(site_name or "site")
        method_tag = _method_export_tag(measure_var.get())
        return f"{site_tag}_{method_tag}"

    def _apply_publication_style(fig_obj, width=7.2, height=4.4, preserve_panel_typography=False):
        """Apply manuscript-oriented sizing without changing plotted values."""
        try:
            fig_obj.set_size_inches(float(width), float(height), forward=True)
            if not preserve_panel_typography:
                for axis in fig_obj.axes:
                    axis.title.set_fontsize(11)
                    axis.xaxis.label.set_size(10)
                    axis.yaxis.label.set_size(10)
                    axis.tick_params(axis="both", labelsize=9)
                    legend = axis.get_legend()
                    if legend is not None:
                        for txt_obj in legend.get_texts():
                            txt_obj.set_fontsize(9)
            if not preserve_panel_typography:
                fig_obj.tight_layout(pad=1.4)
        except Exception:
            pass

    def export_publication_outputs():
        """Export the current IT result as a reproducible publication package.

        The export includes standardized vector/raster figures, the full result
        table, metadata, run history, and available network products. It never
        recomputes analyses or changes the values shown in the toolbox.
        """
        nonlocal last_results_df, last_metadata, last_network_edges, last_prepared_temporal_df
        nonlocal last_figure6_lagged_mi_full_df, last_figure6_transfer_entropy_full_df
        nonlocal last_figure6_temporal_pid_df, last_figure6_temporal_pid_bootstrap_df
        if last_results_df is None or getattr(last_results_df, "empty", True):
            messagebox.showinfo("Nothing to export", "Run an analysis first.")
            return

        inferred_root = _infer_meaningflux_project_root()
        selected_dir = filedialog.askdirectory(
            parent=win,
            title=(
                "Choose the MeaningFlux project folder. "
                "Results will be saved under results/information_theory."
            ),
            initialdir=str(inferred_root),
            mustexist=True,
        )
        if not selected_dir:
            return
        root_dir = str(_normalize_selected_project_root(selected_dir))

        method_name = measure_var.get()
        site_tag = _safe_file_tag(site_name or "site")
        method_tag = _method_export_tag(method_name)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        package_dir = os.path.join(root_dir, "results", "information_theory", site_tag, f"{stamp}_{method_tag}")
        os.makedirs(package_dir, exist_ok=True)

        base_name = f"{site_tag}_{method_tag}"
        manifest = []
        is_figure6 = method_tag == "figure_6_temporal_summary"
        figure_width = 15.2 if is_figure6 else 7.2
        figure_height = 9.2 if is_figure6 else 4.4

        try:
            # Keep the deliberately wide Figure 6 layout and its larger network panel.
            _apply_publication_style(
                fig,
                width=figure_width,
                height=figure_height,
                preserve_panel_typography=is_figure6,
            )
            for ext, dpi in (("png", 600), ("pdf", 600), ("svg", 600)):
                out_path = os.path.join(package_dir, f"{base_name}.{ext}")
                fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
                manifest.append({"file": os.path.basename(out_path), "type": "main_figure", "format": ext})

            data_path = os.path.join(package_dir, f"{base_name}_data.csv")
            last_results_df.to_csv(data_path, index=False)
            manifest.append({"file": os.path.basename(data_path), "type": "peak_summary_data", "format": "csv"})

            if is_figure6 and last_figure6_lagged_mi_full_df is not None and not last_figure6_lagged_mi_full_df.empty:
                mi_full_path = os.path.join(package_dir, f"{site_tag}_figure_6_lagged_mi_full.csv")
                last_figure6_lagged_mi_full_df.to_csv(mi_full_path, index=False)
                manifest.append({
                    "file": os.path.basename(mi_full_path),
                    "type": "lagged_mi_full_curve_data",
                    "format": "csv",
                })

            if is_figure6 and last_figure6_transfer_entropy_full_df is not None and not last_figure6_transfer_entropy_full_df.empty:
                te_full_path = os.path.join(package_dir, f"{site_tag}_figure_6_transfer_entropy_full.csv")
                last_figure6_transfer_entropy_full_df.to_csv(te_full_path, index=False)
                manifest.append({
                    "file": os.path.basename(te_full_path),
                    "type": "transfer_entropy_full_curve_data",
                    "format": "csv",
                })

            if is_figure6 and last_figure6_temporal_pid_df is not None and not last_figure6_temporal_pid_df.empty:
                pid_path = os.path.join(package_dir, f"{site_tag}_figure_6_temporal_pid.csv")
                last_figure6_temporal_pid_df.to_csv(pid_path, index=False)
                manifest.append({
                    "file": os.path.basename(pid_path),
                    "type": "target_memory_conditioned_temporal_pid",
                    "format": "csv",
                })

            if is_figure6 and last_figure6_temporal_pid_bootstrap_df is not None and not last_figure6_temporal_pid_bootstrap_df.empty:
                pid_boot_path = os.path.join(package_dir, f"{site_tag}_figure_6_temporal_pid_bootstrap.csv")
                last_figure6_temporal_pid_bootstrap_df.to_csv(pid_boot_path, index=False)
                manifest.append({
                    "file": os.path.basename(pid_boot_path),
                    "type": "temporal_pid_moving_block_bootstrap",
                    "format": "csv",
                })

            if last_prepared_temporal_df is not None and not last_prepared_temporal_df.empty:
                prepared_path = os.path.join(package_dir, f"{base_name}_prepared_temporal_series.csv")
                last_prepared_temporal_df.to_csv(prepared_path, index=False)
                manifest.append({
                    "file": os.path.basename(prepared_path),
                    "type": "prepared_temporal_series",
                    "format": "csv",
                })

            metadata = dict(last_metadata or {})
            metadata.update({
                "publication_export_created_local": datetime.now().isoformat(timespec="seconds"),
                "publication_export_method": method_name,
                "publication_export_site": site_name,
                "figure_width_inches": figure_width,
                "figure_height_inches": figure_height,
                "png_dpi": 600,
                "formats": ["png", "pdf", "svg"],
                "figure6_full_lagged_mi_exported": bool(
                    is_figure6 and last_figure6_lagged_mi_full_df is not None
                    and not last_figure6_lagged_mi_full_df.empty
                ),
                "figure6_full_directional_te_exported": bool(
                    is_figure6 and last_figure6_transfer_entropy_full_df is not None
                    and not last_figure6_transfer_entropy_full_df.empty
                ),
                "figure6_temporal_pid_exported": bool(
                    is_figure6 and last_figure6_temporal_pid_df is not None
                    and not last_figure6_temporal_pid_df.empty
                ),
                "figure6_temporal_pid_bootstrap_exported": bool(
                    is_figure6 and last_figure6_temporal_pid_bootstrap_df is not None
                    and not last_figure6_temporal_pid_bootstrap_df.empty
                ),
                "figure6_temporal_pid_estimator": "stratified conditional I_min within target-memory states",
                "note": "Figure 6 uses full lag heatmaps, a large supported TE network, and temporal PID conditioned on target memory. Other methods use the standard publication size.",
            })
            meta_path = os.path.join(package_dir, f"{base_name}_metadata.json")
            with open(meta_path, "w", encoding="utf-8") as fmeta:
                json.dump(metadata, fmeta, indent=2, default=str)
            manifest.append({"file": os.path.basename(meta_path), "type": "metadata", "format": "json"})

            if analysis_runs:
                runs_path = os.path.join(package_dir, f"{site_tag}_analysis_run_history.csv")
                pd.DataFrame(analysis_runs).to_csv(runs_path, index=False)
                manifest.append({"file": os.path.basename(runs_path), "type": "run_history", "format": "csv"})

            # Export available Circos/directed-network products as separate figures.
            if last_network_edges is not None and not getattr(last_network_edges, "empty", True):
                _apply_publication_style(network_fig, width=7.6, height=6.2, preserve_panel_typography=True)
                network_base = f"{site_tag}_figure_6_information_network" if is_figure6 else f"{site_tag}_information_network"
                for ext, dpi in (("png", 600), ("pdf", 600), ("svg", 600)):
                    net_path = os.path.join(package_dir, f"{network_base}.{ext}")
                    network_fig.savefig(net_path, dpi=dpi, bbox_inches="tight")
                    manifest.append({"file": os.path.basename(net_path), "type": "network_figure", "format": ext})
                edges_name = (
                    f"{site_tag}_figure_6_information_network_edges.csv"
                    if is_figure6 else f"{site_tag}_information_network_edges.csv"
                )
                edges_path = os.path.join(package_dir, edges_name)
                last_network_edges.to_csv(edges_path, index=False)
                manifest.append({"file": os.path.basename(edges_path), "type": "network_edges", "format": "csv"})

            manifest_path = os.path.join(package_dir, "publication_manifest.csv")
            pd.DataFrame(manifest).to_csv(manifest_path, index=False)

            readme_path = os.path.join(package_dir, "README.txt")
            with open(readme_path, "w", encoding="utf-8") as freadme:
                freadme.write(
                    "MeaningFlux Information Theory publication outputs\n"
                    "==================================================\n\n"
                    f"Site: {site_name or 'not specified'}\n"
                    f"Analysis: {method_name}\n"
                    f"Created: {datetime.now().isoformat(timespec='seconds')}\n\n"
                    "Main figure formats: PNG (600 dpi), PDF, SVG.\n"
                    "The *_data.csv file contains peak MI/TE summaries, network edges, and temporal PID pair summaries used by the four-panel preview.\n"
                    "For Figure 6, *_lagged_mi_full.csv contains every evaluated MI lag, and \n"
                    "*_transfer_entropy_full.csv contains every evaluated directional TE lag, including reverse links when requested.\n"
                    "*_temporal_pid.csv contains target-memory-conditioned PID summaries for all driver pairs.\n"
                    "*_temporal_pid_bootstrap.csv contains the moving-block bootstrap replicates used for 95% intervals.\n"
                    "The prepared temporal-series CSV preserves the regular Native/Daily/Weekly grid and missing periods.\n"
                    "The JSON file records the analysis settings and reproducibility metadata.\n"
                    "Network products are included when a TE network has been computed.\n"
                    "Figure 6 keeps its full four-panel layout so the lag heatmaps, directional network, and PID composition remain legible.\n"
                )

            messagebox.showinfo(
                "Publication outputs exported",
                f"Created publication package:\n{package_dir}\n\n"
                "Includes PNG/PDF/SVG, peak summaries, full lag curves, temporal PID and bootstrap tables, prepared series, metadata, manifest, and network products."
            )
        except Exception as e:
            messagebox.showerror("Publication export error", str(e))

    export_publication_btn.configure(command=export_publication_outputs)


    def update_states(*_):
        """Show only controls needed by the selected analysis."""
        m = measure_var.get()
        save_btn.configure(state=tk.DISABLED)
        run_btn.configure(text="Compute selected analysis")

        # Variable selectors change by method; quick-action buttons stay visible.
        selector_widgets = [x_label, x_cb, y_label, y_cb, z_label, z_cb,
                            detect_model_btn, model_hint_label]
        pool_widgets = [pool_label, search_entry, pool_hint_label,
                        driver_list_frame, list_buttons]
        for w in selector_widgets + pool_widgets:
            try:
                w.grid_remove()
            except Exception:
                pass

        def show_x(label="Source / Driver X"):
            x_label.configure(text=label)
            x_label.grid()
            x_cb.grid()
            x_cb.configure(state="readonly")

        def show_y(label="Target Y"):
            y_label.configure(text=label)
            y_label.grid()
            y_cb.grid()
            y_cb.configure(state="readonly")

        def show_z(label="Variable Z"):
            z_label.configure(text=label)
            z_label.grid()
            z_cb.grid()
            z_cb.configure(state="readonly")

        def show_pool(hint="Select predictors / variables for this analysis."):
            pool_hint_label.configure(text=hint)
            for w in pool_widgets:
                w.grid()

        if m == "Entropy H(X)":
            show_x("Variable X")

        elif m == "Mutual Information I(X;Y)":
            show_x("Driver / Variable X")
            show_y("Target / Variable Y")

        elif m in {"MI Driver Ranking", "Correlation vs MI Ranking", "Pairwise PID Matrix"}:
            show_y("Target Y")
            show_pool("Select the predictors to evaluate against the target.")

        elif m == "Figure 6 Temporal Summary":
            show_y("Observed target Y")
            show_pool("Select harmonized environmental drivers. Use the always-visible Figure 6 setup button above.")
            run_btn.configure(text="Run configured Figure 6 analysis")
        elif m in {"Model-vs-Observed MI", "Model-vs-Observed PID Matrix", "Functional Performance Summary"}:
            show_y("Observed target Y")
            show_z("Modeled / predicted target Z")
            detect_model_btn.grid()
            model_hint_label.grid()
            show_pool("Select the environmental predictors used for the model comparison.")

        elif m == "Single-Site IT Summary":
            show_x("Lag / TE source X")
            show_y("Target Y")
            show_z("Optional second PID driver Z")
            show_pool("Select the driver pool used in the site summary.")

        elif m == "Conditional MI I(X;Y|Z)":
            show_x("Driver / Variable X")
            show_y("Target / Variable Y")
            show_z("Conditioning variable Z")

        elif m in {"Lagged MI I(X_t;Y_{t+lag})", "Transfer Entropy TE(X→Y)", "Transfer Entropy vs Lag"}:
            show_x("Source X")
            show_y("Target Y")

        elif m == "PID (X1,X2→Y)":
            show_x("Driver 1 (X1)")
            show_y("Driver 2 (X2)")
            show_z("PID target Y")

        elif m == "TE Network":
            show_pool("Select two or more variables for the directed TE network.")

    def _on_measure_change(*_):
        update_states()
        _show_method_guide()
        if measure_var.get() == "Figure 6 Temporal Summary":
            run_btn.configure(text="Run configured Figure 6 analysis")
        else:
            run_btn.configure(text="Compute selected analysis")

    measure_cb.bind("<<ComboboxSelected>>", _on_measure_change)
    update_states()
    _show_method_guide()

    def _on_close():
        global _it_window
        try:
            try:
                _unbind_left_scroll()
            except Exception:
                pass
            win.destroy()
        finally:
            _it_window = None

    win.protocol("WM_DELETE_WINDOW", _on_close)
    return win


# Standalone demo (optional)
if __name__ == "__main__":
    root = tk.Tk()
    root.title("MeaningFlux Demo – IT Toolbox")
    n = 1000
    df_demo = pd.DataFrame({
        "FC": np.random.randn(n),
        "SW_IN": np.random.rand(n) * 1000.0,
        "TA": 10 + 15 * np.random.rand(n),
        "VPD": np.random.rand(n) * 3.0,
        "TS": 8 + 10 * np.random.rand(n),
        "USTAR": np.random.rand(n),
        "WS": np.random.rand(n) * 5.0,
        "LE": np.random.randn(n) + 0.5,
        "SWC": np.random.rand(n),
        "TIMESTAMP_START": pd.date_range("2020-01-01", periods=n, freq="30min"),
        "STRING_COL": ["foo"] * n,  # non-numeric just to test robustness
    })
    ttk.Button(
        root,
        text="Open Information Theory Toolbox\n(pattern A: df, site_name)",
        command=lambda: open_information_theory_toolbox(df_demo, "DemoSite"),
    ).pack(padx=20, pady=(20, 10))
    ttk.Button(
        root,
        text="Open Information Theory Toolbox\n(pattern B: parent, df)",
        command=lambda: open_information_theory_toolbox(root, df_demo),
    ).pack(padx=20, pady=(0, 20))
    root.mainloop()
