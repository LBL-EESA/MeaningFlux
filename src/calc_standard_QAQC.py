# -*- coding: utf-8 -*-
"""
Standard QA/QC module for MeaningFlux (multi-variable + undo last)

Author: Leila C. Hernandez Rodriguez
Lawrence Berkeley National Laboratory, Berkeley, CA, USA (lchernandezrodriguez@lbl.gov)
ORCID: 0000-0001-8830-345X

- QA/QC multiple variables (one-by-one) and keep track of which variables were processed
- QA/QC summary panel listing variables that underwent QA/QC + counts flagged
- Export options:
    (A) Export FULL dataset (all columns) with QA/QC applied where available
    (B) Export ONLY the variables that underwent QA/QC (+ time cols + flags)
- Prevents cascading data loss on repeated Apply:
    * masks computed against ORIGINAL baseline per variable
    * cleaned series rebuilt from baseline for that variable each Apply
- Undo last apply (reverses only the last variable QA/QC action)
- Optional analysis subset filters: full record, custom date/time range, daytime/nighttime, months, growing/non-growing season
- Optional derived variables for downstream ML/IT: first difference, log10 transform, and rolling anomaly
- Exports a QA/QC summary metadata CSV next to the cleaned dataset
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


# ------------------- defaults (optional) -------------------
reasonable_limits: dict[str, tuple[float, float]] = {
    "FC": (-50, 50),
    "NEE": (-50, 50),
    "GPP": (0, 50),
    "RECO": (-50, 50),
    "FCH4": (-5, 5),
    "FN2O": (-2, 2),
    "TA": (-50, 60),
    "TS": (-50, 60),
    "SWC": (0, 100),
    "WS": (0, 60),
    "WD": (0, 360),
    "USTAR": (0, 2),
    "VPD": (0, 100),
    "SW_IN": (0, 1500),
    "NETRAD": (-200, 1000),
}

TIME_COLS = {
    "TIMESTAMP_START", "TIMESTAMP_END",
    "DATESTAMP_START", "DATESTAMP_END",
}


def _detect_time_columns(df: pd.DataFrame) -> list[str]:
    """Detect likely timestamp/date columns beyond the fixed AmeriFlux names."""
    detected: list[str] = []
    patterns = ("timestamp", "date", "time", "datetime", "datestamp")
    for c in df.columns:
        cl = str(c).lower()
        if c in TIME_COLS or any(p in cl for p in patterns):
            detected.append(c)
        elif pd.api.types.is_datetime64_any_dtype(df[c]):
            detected.append(c)
    return list(dict.fromkeys(detected))


def _guess_light_columns(df: pd.DataFrame) -> list[str]:
    """Return likely radiation/light columns for daytime/nighttime filtering."""
    patterns = ("sw_in", "swin", "ppfd", "par", "rg", "netrad", "rn")
    preferred = [c for c in df.columns if any(p in str(c).lower() for p in patterns)]
    try:
        numeric = _numeric_columns(df)
    except Exception:
        numeric = []
    return list(dict.fromkeys(preferred + numeric + list(df.columns)))


def _parse_datetime_or_none(value: str):
    value = (value or "").strip()
    if not value:
        return None
    return pd.to_datetime(value, errors="coerce")


def _parse_month_list(value: str) -> list[int]:
    out: list[int] = []
    for part in str(value or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            month = int(part)
            if 1 <= month <= 12:
                out.append(month)
        except Exception:
            pass
    return sorted(set(out)) or [5, 6, 7, 8, 9]


def _subset_mask(
    df: pd.DataFrame,
    mode: str,
    ts_col: str,
    light_col: str,
    day_threshold: str,
    months_text: str,
    start_text: str,
    end_text: str,
) -> pd.Series:
    """Build a row mask for applying QA/QC or plotting a selected analysis subset."""
    mask = pd.Series(True, index=df.index)
    mode = str(mode or "Full record")

    needs_time = mode in {"Custom date/time", "Selected months", "Growing season", "Non-growing season"}
    if needs_time:
        if ts_col not in df.columns:
            raise ValueError("A timestamp column is required for the selected subset.")
        t = pd.to_datetime(df[ts_col], errors="coerce")
        mask &= t.notna()
        if mode == "Custom date/time":
            start = _parse_datetime_or_none(start_text)
            end = _parse_datetime_or_none(end_text)
            if start is not None and pd.notna(start):
                mask &= t >= start
            if end is not None and pd.notna(end):
                mask &= t <= end
        else:
            months = _parse_month_list(months_text)
            m = t.dt.month.isin(months)
            mask &= (~m if mode == "Non-growing season" else m)

    if mode in {"Daytime", "Nighttime"}:
        if light_col not in df.columns:
            raise ValueError("Select a valid light/radiation column for daytime/nighttime filtering.")
        try:
            threshold = float(day_threshold)
        except Exception:
            threshold = 20.0
        light = pd.to_numeric(df[light_col], errors="coerce")
        valid_light = light.notna()
        day_mask = valid_light & (light > threshold)
        night_mask = valid_light & (light <= threshold)
        mask &= day_mask if mode == "Daytime" else night_mask

    return mask.fillna(False)


def _safe_filename_piece(value: str) -> str:
    value = str(value or "").strip()
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value) or "var"


def _qaqc_base_name(name: str) -> str:
    """Return an AmeriFlux-style base variable name for default QA/QC limits.

    Examples: FC_1_1_1 -> FC, TA_PI_1_1_1 -> TA, SW_IN_F -> SW_IN.
    """
    txt = str(name).upper()
    parts = txt.split("_")
    if not parts:
        return txt
    if len(parts) >= 2 and parts[0] in {"SW", "LW"} and parts[1] in {"IN", "OUT"}:
        return f"{parts[0]}_{parts[1]}"
    if len(parts) >= 2 and parts[0] == "PPFD" and parts[1] in {"IN", "OUT"}:
        return f"{parts[0]}_{parts[1]}"
    return parts[0]


# ------------------- helpers -------------------
def _center(win: tk.Tk | tk.Toplevel, w: int, h: int) -> None:
    win.update_idletasks()
    x = (win.winfo_screenwidth() // 2) - (w // 2)
    y = (win.winfo_screenheight() // 2) - (h // 2)
    win.geometry(f"{w}x{h}+{x}+{y}")


def _make_scrollable_frame(parent, width: int = 390):
    """Create a vertically scrollable frame for long QA/QC controls."""
    container = ttk.Frame(parent)
    container.grid_rowconfigure(0, weight=1)
    container.grid_columnconfigure(0, weight=1)

    canvas = tk.Canvas(container, width=width, borderwidth=0, highlightthickness=0)
    scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.grid(row=0, column=0, sticky="nsew")
    scrollbar.grid(row=0, column=1, sticky="ns")

    inner = ttk.Frame(canvas, padding=10)
    window_id = canvas.create_window((0, 0), window=inner, anchor="nw")

    def _on_configure(_event=None):
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _fit_width(event):
        canvas.itemconfigure(window_id, width=event.width)

    def _on_mousewheel(event):
        if event.delta:
            canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

    def _bind_wheel(_event=None):
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

    def _unbind_wheel(_event=None):
        canvas.unbind_all("<MouseWheel>")

    inner.bind("<Configure>", _on_configure)
    canvas.bind("<Configure>", _fit_width)
    container.bind("<Enter>", _bind_wheel)
    container.bind("<Leave>", _unbind_wheel)
    return container, inner


def _safe_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for c in df.columns:
        if c in TIME_COLS:
            continue
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
        else:
            test = pd.to_numeric(df[c], errors="coerce")
            if test.notna().sum() > 0:
                cols.append(c)
    return cols


def _mad_bounds(x: pd.Series, k: float):
    x0 = x.dropna()
    if x0.empty:
        return None, None
    med = float(x0.median())
    mad = float(np.median(np.abs(x0.values - med)))
    if not np.isfinite(med) or not np.isfinite(mad) or mad <= 0:
        return None, None
    robust_sigma = 1.4826 * mad
    return med - k * robust_sigma, med + k * robust_sigma


def _build_mask_from_baseline(
    x_baseline: pd.Series,
    use_physical: bool,
    phys_lo: float | None,
    phys_hi: float | None,
    use_sigma: bool,
    sigma: float | None,
    use_mad: bool,
    mad_k: float | None,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    x = _safe_numeric_series(x_baseline.copy())

    mask_bad = pd.Series(False, index=x.index)
    mask_phys = pd.Series(False, index=x.index)
    mask_sig = pd.Series(False, index=x.index)
    mask_mad = pd.Series(False, index=x.index)

    if use_physical and (phys_lo is not None) and (phys_hi is not None):
        mask_phys = (x < phys_lo) | (x > phys_hi)
        mask_bad |= mask_phys

    if use_sigma and (sigma is not None):
        mu = x.mean(skipna=True)
        sd = x.std(skipna=True)
        if np.isfinite(mu) and np.isfinite(sd) and sd > 0:
            lo = mu - sigma * sd
            hi = mu + sigma * sd
            mask_sig = (x < lo) | (x > hi)
            mask_bad |= mask_sig

    if use_mad and (mad_k is not None):
        lo, hi = _mad_bounds(x, k=mad_k)
        if lo is not None and hi is not None:
            mask_mad = (x < lo) | (x > hi)
            mask_bad |= mask_mad

    return mask_bad, mask_phys, mask_sig, mask_mad


def calc_standard_QAQC(df_in: pd.DataFrame, inputname_site: str, on_update_df=None) -> None:
    if df_in is None or df_in.empty:
        messagebox.showwarning("Warning", "Load EC data first.")
        return

    # Copies
    df_original = df_in.copy()
    df_working = df_in.copy()

    # Normalize detected time cols to datetime; never processed as numeric
    time_cols_detected = _detect_time_columns(df_working)
    for tc in time_cols_detected:
        if tc in df_working.columns:
            df_working[tc] = pd.to_datetime(df_working[tc], errors="coerce")
            df_original[tc] = pd.to_datetime(df_original[tc], errors="coerce")

    # Convert -9999 to NaN for non-time cols only
    for c in df_working.columns:
        if c in TIME_COLS or c in time_cols_detected or pd.api.types.is_datetime64_any_dtype(df_working[c]):
            continue
        df_working[c] = df_working[c].replace(-9999, np.nan)
        df_original[c] = df_original[c].replace(-9999, np.nan)

    vars_numeric = _numeric_columns(df_working)
    if not vars_numeric:
        messagebox.showerror("Error", "No numeric variables found to QA/QC.")
        return

    # Baseline per variable (original), and per-variable applied mask
    baseline = {v: _safe_numeric_series(df_original[v]) for v in vars_numeric}
    masks = {v: pd.Series(False, index=df_working.index) for v in vars_numeric}

    # Track which variables were QA/QC'd + stats
    qaqc_vars: set[str] = set()
    qaqc_stats: dict[str, dict] = {}

    # NEW: stack of applied actions (for undo)
    # each entry: {"var": v, "prev_mask": Series, "had_flag_col": bool, "prev_flag_series": Series|None, "was_in_set": bool}
    action_stack: list[dict] = []

    # ---------------- GUI ----------------
    win = tk.Toplevel() if tk._default_root else tk.Tk()
    win.title("Analysis Preparation and QA/QC")
    _center(win, 1260, 720)

    outer = ttk.Panedwindow(win, orient="horizontal")
    outer.pack(fill="both", expand=True, padx=10, pady=10)

    left_container, left = _make_scrollable_frame(outer, width=460)
    outer.add(left_container, weight=1)

    results_nb = ttk.Notebook(outer)
    outer.add(results_nb, weight=2)

    mid = ttk.Frame(results_nb, padding=10)
    right = ttk.Frame(results_nb, padding=10)
    guide_tab = ttk.Frame(results_nb, padding=10)
    results_nb.add(mid, text="Plot / Preview")
    results_nb.add(right, text="QA/QC Summary")
    results_nb.add(guide_tab, text="Guide")

    guide_text = tk.Text(guide_tab, wrap="word", height=20)
    guide_text.pack(fill="both", expand=True)
    guide_text.insert("1.0", (
        "MeaningFlux Standard QA/QC — guided workflow\n\n"
        "1. Choose rows: use the full record, a custom date/time range, day/night, or selected months. "
        "Rows outside the selected subset are preserved.\n\n"
        "2. Select one variable at a time. Physical limits remove impossible values; sigma and MAD identify statistical outliers within the selected rows.\n\n"
        "3. Preview selected rows before applying filters. This helps confirm that the period, day/night option, or months are correct.\n\n"
        "4. Apply QA/QC. The cleaned variable is rebuilt from the original baseline each time, so repeated Apply does not create cascading data loss.\n\n"
        "5. Use the Summary tab to review flagged counts by variable. Export the full dataset when the cleaned table should continue to ML or IT.\n\n"
        "Optional derived variables are for downstream diagnostics: first difference, log10, or rolling anomaly. Create them only when they support the analysis question."
    ))
    guide_text.configure(state="disabled")

    ttk.Label(left, text="Analysis preparation", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 4))
    ttk.Label(
        left,
        text="1. Select variable  •  2. Define subset  •  3. Apply QA/QC  •  4. Review  •  5. Export",
        wraplength=410,
        justify="left",
        foreground="#444",
    ).pack(anchor="w", pady=(0, 10))

    current_analysis_var = tk.StringVar(value="Current analysis: ready")
    ttk.Label(left, textvariable=current_analysis_var, foreground="#333", wraplength=410).pack(anchor="w", pady=(0, 10))

    ttk.Label(left, text="Variable", font=("Arial", 10, "bold")).pack(anchor="w", pady=(2, 2))
    ttk.Label(left, text="Choose one flux or driver. Apply can be repeated for multiple variables.", wraplength=410, foreground="#444").pack(anchor="w", pady=(0, 4))
    ttk.Label(left, text="Variable to clean").pack(anchor="w")
    var_cb = ttk.Combobox(left, values=vars_numeric, state="readonly", width=28)
    var_cb.set(vars_numeric[0])
    var_cb.pack(anchor="w", pady=(0, 10))

    # Analysis subset controls
    ts_candidates = time_cols_detected if time_cols_detected else list(df_working.columns)
    light_candidates = _guess_light_columns(df_working)
    subset_mode_var = tk.StringVar(value="Full record")
    ts_col_var = tk.StringVar(value=ts_candidates[0] if ts_candidates else "")
    light_col_var = tk.StringVar(value=light_candidates[0] if light_candidates else "")
    day_threshold_var = tk.StringVar(value="20")
    months_var = tk.StringVar(value="5,6,7,8,9")

    def _available_time_range_text(ts_col: str) -> tuple[str, str, str]:
        """Return prefilled start/end text and a readable available-range label."""
        if not ts_col or ts_col not in df_working.columns:
            return "", "", "Available period: no timestamp column detected."
        t = pd.to_datetime(df_working[ts_col], errors="coerce").dropna()
        if t.empty:
            return "", "", f"Available period for {ts_col}: no valid datetimes."
        start = t.min().strftime("%Y-%m-%d %H:%M")
        end = t.max().strftime("%Y-%m-%d %H:%M")
        return start, end, f"Available period in {ts_col}: {start} to {end}. You can keep these values or edit them."

    _default_start, _default_end, _range_label = _available_time_range_text(ts_col_var.get())
    start_dt_var = tk.StringVar(value=_default_start)
    end_dt_var = tk.StringVar(value=_default_end)
    available_range_var = tk.StringVar(value=_range_label)
    plot_subset_only_var = tk.BooleanVar(value=True)

    subset_box = ttk.LabelFrame(left, text="Analysis subset")
    subset_box.pack(anchor="w", fill="x", pady=(0, 10))
    tk.Message(
        subset_box,
        width=410,
        fg="#444",
        text=(
            "Rows outside the selected subset are preserved. Start and End are prefilled from the dataset and remain editable. "
            "Use subsets for seasons, day/night, droughts, or event windows."
        ),
    ).grid(row=0, column=0, columnspan=3, sticky="ew", padx=6, pady=(6, 4))

    ttk.Label(subset_box, text="Subset mode").grid(row=1, column=0, sticky="w", padx=6, pady=2)
    ttk.Combobox(
        subset_box,
        textvariable=subset_mode_var,
        state="readonly",
        values=["Full record", "Custom date/time", "Daytime", "Nighttime", "Selected months", "Growing season", "Non-growing season"],
        width=22,
    ).grid(row=1, column=1, columnspan=2, sticky="ew", padx=6, pady=2)

    ttk.Label(subset_box, text="Timestamp column").grid(row=2, column=0, sticky="w", padx=6, pady=2)
    ts_combo = ttk.Combobox(subset_box, textvariable=ts_col_var, state="readonly", values=ts_candidates, width=22)
    ts_combo.grid(row=2, column=1, columnspan=2, sticky="ew", padx=6, pady=2)

    ttk.Label(subset_box, text="Start date/time").grid(row=3, column=0, sticky="w", padx=6, pady=2)
    ttk.Entry(subset_box, textvariable=start_dt_var, width=22).grid(row=3, column=1, columnspan=2, sticky="ew", padx=6, pady=2)
    ttk.Label(subset_box, text="End date/time").grid(row=4, column=0, sticky="w", padx=6, pady=2)
    ttk.Entry(subset_box, textvariable=end_dt_var, width=22).grid(row=4, column=1, columnspan=2, sticky="ew", padx=6, pady=2)

    ttk.Label(subset_box, textvariable=available_range_var, foreground="gray40", wraplength=350).grid(
        row=5, column=0, columnspan=3, sticky="ew", padx=6, pady=(0, 4)
    )

    ttk.Button(subset_box, text="Use full available period", command=lambda: _set_full_period()).grid(
        row=6, column=0, columnspan=2, sticky="ew", padx=6, pady=(0, 4)
    )
    ttk.Button(subset_box, text="Set 2013--2022", command=lambda: _set_decade_period()).grid(
        row=6, column=2, sticky="ew", padx=(0, 6), pady=(0, 4)
    )

    ttk.Label(subset_box, text="Light/radiation column").grid(row=7, column=0, sticky="w", padx=6, pady=2)
    ttk.Combobox(subset_box, textvariable=light_col_var, state="readonly", values=light_candidates, width=22).grid(row=7, column=1, columnspan=2, sticky="ew", padx=6, pady=2)
    ttk.Label(subset_box, text="Day threshold").grid(row=8, column=0, sticky="w", padx=6, pady=2)
    ttk.Entry(subset_box, textvariable=day_threshold_var, width=10).grid(row=8, column=1, sticky="w", padx=6, pady=2)
    ttk.Label(subset_box, text="Months").grid(row=9, column=0, sticky="w", padx=6, pady=2)
    ttk.Entry(subset_box, textvariable=months_var, width=22).grid(row=9, column=1, columnspan=2, sticky="ew", padx=6, pady=2)
    ttk.Label(subset_box, text="Example: 5,6,7,8,9", foreground="gray40").grid(row=10, column=1, columnspan=2, sticky="w", padx=6, pady=(0,2))
    ttk.Checkbutton(subset_box, text="Plot only selected subset", variable=plot_subset_only_var).grid(row=11, column=0, columnspan=3, sticky="w", padx=6, pady=(2, 2))
    ttk.Button(subset_box, text="Preview selected rows", command=lambda: _preview_subset()).grid(row=12, column=0, columnspan=3, sticky="ew", padx=6, pady=(2, 6))
    subset_box.columnconfigure(1, weight=1)
    subset_box.columnconfigure(2, weight=1)

    def _set_full_period():
        start, end, label = _available_time_range_text(ts_col_var.get())
        start_dt_var.set(start)
        end_dt_var.set(end)
        available_range_var.set(label)

    def _set_decade_period():
        start_dt_var.set("2013-01-01 00:00")
        end_dt_var.set("2022-12-31 23:59")
        subset_mode_var.set("Custom date/time")

    def _on_timestamp_change(*_):
        _set_full_period()

    try:
        ts_col_var.trace_add("write", _on_timestamp_change)
    except Exception:
        pass

    # Options vars
    use_phys_var = tk.BooleanVar(value=True)
    phys_min_var = tk.StringVar(value="")
    phys_max_var = tk.StringVar(value="")

    use_sigma_var = tk.BooleanVar(value=True)
    sigma_val = tk.StringVar(value="3")

    use_mad_var = tk.BooleanVar(value=False)
    mad_val = tk.StringVar(value="3.5")

    export_as_9999 = tk.BooleanVar(value=True)

    ttk.Label(left, text="QA/QC filters", font=("Arial", 10, "bold")).pack(anchor="w", pady=(4, 2))
    ttk.Label(left, text="Physical limits remove impossible values. Sigma and MAD identify statistical outliers within the selected rows.", wraplength=350, foreground="#444").pack(anchor="w", pady=(0, 4))
    ttk.Checkbutton(left, text="Use physical limits", variable=use_phys_var).pack(anchor="w")
    phys_row = ttk.Frame(left)
    phys_row.pack(anchor="w", pady=(2, 10))
    ttk.Label(phys_row, text="Min:").grid(row=0, column=0, sticky="w")
    ttk.Entry(phys_row, textvariable=phys_min_var, width=10).grid(row=0, column=1, padx=(6, 12))
    ttk.Label(phys_row, text="Max:").grid(row=0, column=2, sticky="w")
    ttk.Entry(phys_row, textvariable=phys_max_var, width=10).grid(row=0, column=3, padx=(6, 0))

    ttk.Checkbutton(left, text="Use σ outlier filter (mean ± σ·std)", variable=use_sigma_var).pack(anchor="w")
    sig_row = ttk.Frame(left)
    sig_row.pack(anchor="w", pady=(2, 10))
    ttk.Label(sig_row, text="σ:").pack(side="left")
    ttk.Entry(sig_row, textvariable=sigma_val, width=8).pack(side="left", padx=6)

    ttk.Checkbutton(left, text="Use robust MAD filter (median ± k·MAD)", variable=use_mad_var).pack(anchor="w")
    mad_row = ttk.Frame(left)
    mad_row.pack(anchor="w", pady=(2, 10))
    ttk.Label(mad_row, text="k:").pack(side="left")
    ttk.Entry(mad_row, textvariable=mad_val, width=8).pack(side="left", padx=6)

    ttk.Checkbutton(left, text="Export missing as -9999", variable=export_as_9999).pack(anchor="w", pady=(0, 10))

    # Optional derived variables for downstream ML/IT
    derived_box = ttk.LabelFrame(left, text="Optional derived variables for ML/IT")
    derived_box.pack(anchor="w", fill="x", pady=(0, 10))
    tk.Message(
        derived_box,
        width=350,
        fg="#444",
        text=(
            "Create an additional column from the selected variable. "
            "First difference emphasizes changes, log10 helps skewed positive variables, "
            "and rolling anomaly removes a local moving-window mean."
        ),
    ).grid(row=0, column=0, columnspan=2, sticky="ew", padx=6, pady=(6, 2))
    transform_var = tk.StringVar(value="First difference")
    ttk.Label(derived_box, text="Transform").grid(row=1, column=0, sticky="w", padx=6, pady=(6,2))
    ttk.Combobox(
        derived_box, textvariable=transform_var, state="readonly",
        values=["First difference", "Log10", "Rolling anomaly"], width=20
    ).grid(row=1, column=1, sticky="ew", padx=6, pady=(6,2))
    ttk.Label(derived_box, text="Window for anomaly").grid(row=2, column=0, sticky="w", padx=6, pady=2)
    anomaly_window_var = tk.StringVar(value="48")
    ttk.Entry(derived_box, textvariable=anomaly_window_var, width=8).grid(row=2, column=1, sticky="w", padx=6, pady=2)
    derived_box.columnconfigure(1, weight=1)

    # -------- Plot area (mid) --------
    status_txt = tk.StringVar(value="Ready. Define the analysis subset, preview rows, apply QA/QC, then export.")
    ttk.Label(mid, textvariable=status_txt, wraplength=650, justify="left").pack(anchor="w", pady=(0, 6))
    plot_summary_var = tk.StringVar(value="Subset and QA/QC summary will appear here after preview or apply.")
    ttk.Label(mid, textvariable=plot_summary_var, wraplength=650, justify="left", foreground="#444").pack(anchor="w", pady=(0, 6))

    fig = plt.Figure(figsize=(6.7, 3.5), dpi=100)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=mid)
    canvas.get_tk_widget().pack(fill="both", expand=True)

    # -------- Summary (right) --------
    ttk.Label(right, text="QA/QC summary by variable", font=("Arial", 11, "bold")).pack(anchor="w")

    summary_cols = ("Variable", "Flagged", "PHYS", "SIG", "MAD")
    summary = ttk.Treeview(right, columns=summary_cols, show="headings", height=18)
    for c in summary_cols:
        summary.heading(c, text=c)
        summary.column(c, width=74 if c != "Variable" else 100, anchor="w")
    summary.pack(fill="both", expand=True, pady=(6, 8))

    def _refresh_summary():
        for row in summary.get_children():
            summary.delete(row)
        for v in sorted(qaqc_vars):
            st = qaqc_stats.get(v, {})
            summary.insert(
                "",
                "end",
                values=(
                    v,
                    st.get("n_flagged_total", 0),
                    st.get("n_flagged_phys", 0),
                    st.get("n_flagged_sigma", 0),
                    st.get("n_flagged_mad", 0),
                ),
            )

    # ---------------- actions ----------------
    def _parse_float(s: str) -> float | None:
        s = (s or "").strip()
        if s == "":
            return None
        try:
            return float(s)
        except Exception:
            return None

    def _preload_limits_for_var(v: str) -> None:
        key = v if v in reasonable_limits else _qaqc_base_name(v)
        if key in reasonable_limits:
            lo, hi = reasonable_limits[key]
            phys_min_var.set(str(lo))
            phys_max_var.set(str(hi))
        else:
            # don't overwrite user's custom limits if already filled
            if phys_min_var.get().strip() == "" and phys_max_var.get().strip() == "":
                phys_min_var.set("")
                phys_max_var.set("")

    def _current_subset_mask() -> pd.Series:
        return _subset_mask(
            df_working,
            subset_mode_var.get(),
            ts_col_var.get(),
            light_col_var.get(),
            day_threshold_var.get(),
            months_var.get(),
            start_dt_var.get(),
            end_dt_var.get(),
        )

    def _preview_subset() -> None:
        try:
            mask = _current_subset_mask()
            n_rows = int(mask.sum())
            n_total = int(len(df_working))
            status_txt.set(
                f"Subset preview: {subset_mode_var.get()}\n"
                f"Rows selected: {n_rows:,} of {n_total:,}\n"
                f"Timestamp: {ts_col_var.get()} | Start: {start_dt_var.get()} | End: {end_dt_var.get()}\n"
                f"Day/night uses {light_col_var.get()} > {day_threshold_var.get()} for daytime. Months field: {months_var.get()}"
            )
            _render_plot(var_cb.get().strip())
        except Exception as e:
            messagebox.showerror("Subset preview error", str(e))

    def _update_current_analysis_label() -> None:
        try:
            mask = _current_subset_mask()
            n_rows = int(mask.sum())
        except Exception:
            n_rows = 0
        current_analysis_var.set(
            f"Current analysis: {inputname_site} | {var_cb.get().strip()} | "
            f"{start_dt_var.get()} to {end_dt_var.get()} | selected rows: {n_rows:,}"
        )

    def _get_x_axis(df_: pd.DataFrame):
        ts_col = ts_col_var.get()
        if ts_col in df_.columns and pd.api.types.is_datetime64_any_dtype(df_[ts_col]):
            return df_[ts_col]
        if "TIMESTAMP_START" in df_.columns and pd.api.types.is_datetime64_any_dtype(df_["TIMESTAMP_START"]):
            return df_["TIMESTAMP_START"]
        return pd.RangeIndex(start=0, stop=len(df_), step=1)

    def _render_plot(v: str) -> None:
        _update_current_analysis_label()
        ax.clear()
        try:
            plot_mask = _current_subset_mask() if plot_subset_only_var.get() else pd.Series(True, index=df_working.index)
        except Exception as e:
            plot_mask = pd.Series(True, index=df_working.index)
            status_txt.set(f"Could not apply plot subset: {e}")
        x = _get_x_axis(df_working).loc[plot_mask] if hasattr(_get_x_axis(df_working), "loc") else np.asarray(_get_x_axis(df_working))[plot_mask.to_numpy()]
        y0 = _safe_numeric_series(df_original[v]).loc[plot_mask]
        y1 = _safe_numeric_series(df_working[v]).loc[plot_mask]
        ax.plot(x, y0, label="Original", linewidth=1.1)
        ax.plot(x, y1, label="After QA/QC", linewidth=1.1)
        flagged = masks.get(v, pd.Series(False, index=df_working.index)).loc[plot_mask]
        if flagged.any():
            ax.scatter(np.asarray(x)[flagged.to_numpy()], y0.loc[flagged], s=16, label="Flagged", zorder=3)
        ax.set_title(f"{v} — original and QA/QC-filtered observations")
        ax.set_xlabel("Time" if ts_col_var.get() in df_working.columns else "Index")
        ax.set_ylabel(v)
        ax.legend(loc="best")
        fig.tight_layout()
        canvas.draw()

    def _apply() -> None:
        nonlocal df_working
        v = var_cb.get().strip()

        phys_lo = _parse_float(phys_min_var.get())
        phys_hi = _parse_float(phys_max_var.get())
        if use_phys_var.get():
            if phys_lo is None or phys_hi is None:
                messagebox.showerror("Error", "Physical limits are enabled but Min/Max are not valid numbers.")
                return
            if phys_lo >= phys_hi:
                messagebox.showerror("Error", "Physical limits require Min < Max.")
                return

        sigma = None
        if use_sigma_var.get():
            sigma = _parse_float(sigma_val.get())
            if sigma is None or sigma <= 0:
                messagebox.showerror("Error", "σ must be a positive number (e.g., 2, 3, 4).")
                return

        mad_k = None
        if use_mad_var.get():
            mad_k = _parse_float(mad_val.get())
            if mad_k is None or mad_k <= 0:
                messagebox.showerror("Error", "MAD k must be a positive number (e.g., 3.5).")
                return

        # Save state for undo (only for this variable)
        flag_col = f"QAQC_FLAG_{v}"
        had_flag_col = flag_col in df_working.columns
        prev_flag_series = df_working[flag_col].copy() if had_flag_col else None
        prev_mask = masks[v].copy()
        was_in_set = v in qaqc_vars

        # Compute new mask from BASELINE, restricted to the selected subset.
        x_base = baseline[v]
        try:
            subset_mask = _current_subset_mask()
        except Exception as e:
            messagebox.showerror("Subset error", str(e))
            return

        mask_bad_raw, mask_phys_raw, mask_sig_raw, mask_mad_raw = _build_mask_from_baseline(
            x_baseline=x_base,
            use_physical=use_phys_var.get(),
            phys_lo=phys_lo,
            phys_hi=phys_hi,
            use_sigma=use_sigma_var.get(),
            sigma=sigma,
            use_mad=use_mad_var.get(),
            mad_k=mad_k,
        )

        # Preserve previous QA/QC decisions outside the selected subset.
        mask_bad = prev_mask.copy()
        mask_bad.loc[subset_mask] = mask_bad_raw.loc[subset_mask]
        mask_phys = pd.Series(False, index=df_working.index); mask_phys.loc[subset_mask] = mask_phys_raw.loc[subset_mask]
        mask_sig = pd.Series(False, index=df_working.index); mask_sig.loc[subset_mask] = mask_sig_raw.loc[subset_mask]
        mask_mad = pd.Series(False, index=df_working.index); mask_mad.loc[subset_mask] = mask_mad_raw.loc[subset_mask]

        # Push undo snapshot
        action_stack.append(
            {
                "var": v,
                "prev_mask": prev_mask,
                "had_flag_col": had_flag_col,
                "prev_flag_series": prev_flag_series,
                "was_in_set": was_in_set,
            }
        )

        # Apply
        masks[v] = mask_bad.copy()

        if flag_col not in df_working.columns:
            df_working[flag_col] = ""
        else:
            df_working[flag_col] = prev_flag_series.copy() if prev_flag_series is not None else ""
        df_working.loc[subset_mask, flag_col] = ""

        # rebuild this variable from baseline every time
        df_working[v] = x_base.mask(mask_bad, np.nan)

        df_working.loc[mask_phys, flag_col] = (df_working.loc[mask_phys, flag_col] + "|PHYS").str.strip("|")
        df_working.loc[mask_sig, flag_col] = (df_working.loc[mask_sig, flag_col] + "|SIG").str.strip("|")
        df_working.loc[mask_mad, flag_col] = (df_working.loc[mask_mad, flag_col] + "|MAD").str.strip("|")

        flag_text = df_working[flag_col].astype(str)
        stats = {
            "n_total": int(x_base.loc[subset_mask].notna().sum()),
            "n_subset_rows": int(subset_mask.sum()),
            "subset_mode": subset_mode_var.get(),
            "use_physical_limits": bool(use_phys_var.get()),
            "physical_min": phys_lo if use_phys_var.get() else "",
            "physical_max": phys_hi if use_phys_var.get() else "",
            "use_sigma_filter": bool(use_sigma_var.get()),
            "sigma_threshold": sigma if use_sigma_var.get() else "",
            "use_mad_filter": bool(use_mad_var.get()),
            "mad_k": mad_k if use_mad_var.get() else "",
            "n_flagged_total": int(mask_bad.loc[subset_mask].sum()),
            "n_flagged_phys": int(flag_text.loc[subset_mask].str.contains("PHYS", regex=False).sum()),
            "n_flagged_sigma": int(flag_text.loc[subset_mask].str.contains("SIG", regex=False).sum()),
            "n_flagged_mad": int(flag_text.loc[subset_mask].str.contains("MAD", regex=False).sum()),
        }

        qaqc_vars.add(v)
        qaqc_stats[v] = stats
        _refresh_summary()

        status_txt.set(
            f"Applied QA/QC to {v}\n"
            f"- Subset: {stats.get('subset_mode', 'Full record')} ({stats.get('n_subset_rows', len(df_working))} rows)\n"
            f"- Non-missing in subset: {stats['n_total']}\n"
            f"- Flagged total: {stats['n_flagged_total']} "
            f"(PHYS={stats['n_flagged_phys']}, SIG={stats['n_flagged_sigma']}, MAD={stats['n_flagged_mad']})\n"
            f"- Undo is available for the last Apply."
        )
        _render_plot(v)

    def _undo_last() -> None:
        nonlocal df_working
        if not action_stack:
            messagebox.showinfo("Undo", "Nothing to undo yet.")
            return

        last = action_stack.pop()
        v = last["var"]
        flag_col = f"QAQC_FLAG_{v}"

        # Restore mask and variable values based on restored mask
        masks[v] = last["prev_mask"].copy()
        df_working[v] = baseline[v].mask(masks[v], np.nan)

        # Restore flag column state
        if last["had_flag_col"]:
            df_working[flag_col] = last["prev_flag_series"].copy()
        else:
            # flag col didn't exist before, remove it if we created it
            if flag_col in df_working.columns:
                df_working.drop(columns=[flag_col], inplace=True)

        # Restore qaqc_vars membership + stats
        if last["was_in_set"]:
            qaqc_vars.add(v)
            # stats may be stale; recompute quick totals from restored mask
            st_prev = qaqc_stats.get(v, {})
            st_prev["n_flagged_total"] = int(masks[v].sum())
            qaqc_stats[v] = st_prev
        else:
            if v in qaqc_vars:
                qaqc_vars.remove(v)
            if v in qaqc_stats:
                qaqc_stats.pop(v, None)

        _refresh_summary()
        status_txt.set(f"Undid last Apply for {v}.")
        _render_plot(v)

    def _reset_all() -> None:
        nonlocal df_working
        df_working = df_original.copy()

        for c in df_working.columns:
            if c in TIME_COLS or c in time_cols_detected or pd.api.types.is_datetime64_any_dtype(df_working[c]):
                continue
            df_working[c] = df_working[c].replace(-9999, np.nan)

        for v in vars_numeric:
            masks[v] = pd.Series(False, index=df_working.index)

        qaqc_vars.clear()
        qaqc_stats.clear()
        action_stack.clear()
        _refresh_summary()

        status_txt.set("Reset: restored original dataset and cleared QA/QC summary.")
        _render_plot(var_cb.get().strip())

    def _export(full_dataset: bool) -> None:
        if df_working is None or df_working.empty:
            messagebox.showwarning("Warning", "Nothing to export.")
            return

        if not full_dataset and len(qaqc_vars) == 0:
            messagebox.showwarning("Warning", "No variables have undergone QA/QC yet.")
            return

        if full_dataset:
            default_name = f"{inputname_site}_StandardQAQC_FULL.csv"
        else:
            default_name = f"{inputname_site}_StandardQAQC_ONLY.csv"

        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV files", "*.csv")],
        )
        if not path:
            return

        if full_dataset:
            df_out = df_working.copy()
        else:
            cols = [c for c in df_working.columns if c in TIME_COLS or c in time_cols_detected]
            cols += sorted(qaqc_vars)
            cols += [f"QAQC_FLAG_{v}" for v in sorted(qaqc_vars) if f"QAQC_FLAG_{v}" in df_working.columns]
            seen = set()
            cols = [c for c in cols if not (c in seen or seen.add(c))]
            df_out = df_working[cols].copy()

        if export_as_9999.get():
            for c in df_out.columns:
                if c in TIME_COLS or c in time_cols_detected or pd.api.types.is_datetime64_any_dtype(df_out[c]):
                    continue
                test = pd.to_numeric(df_out[c], errors="coerce")
                if test.notna().sum() > 0:
                    df_out[c] = test.replace(np.nan, -9999)

        try:
            df_out.to_csv(path, index=False)
            meta_rows = []
            for v in sorted(qaqc_vars):
                st = qaqc_stats.get(v, {})
                meta_rows.append({
                    "site": inputname_site,
                    "variable": v,
                    "subset_mode": st.get("subset_mode", subset_mode_var.get()),
                    "timestamp_column": ts_col_var.get(),
                    "start_datetime": start_dt_var.get(),
                    "end_datetime": end_dt_var.get(),
                    "months": months_var.get(),
                    "light_column": light_col_var.get(),
                    "day_threshold": day_threshold_var.get(),
                    "n_subset_rows": st.get("n_subset_rows", ""),
                    "n_nonmissing_subset": st.get("n_total", ""),
                    "use_physical_limits": st.get("use_physical_limits", ""),
                    "physical_min": st.get("physical_min", ""),
                    "physical_max": st.get("physical_max", ""),
                    "use_sigma_filter": st.get("use_sigma_filter", ""),
                    "sigma_threshold": st.get("sigma_threshold", ""),
                    "use_mad_filter": st.get("use_mad_filter", ""),
                    "mad_k": st.get("mad_k", ""),
                    "n_flagged_total": st.get("n_flagged_total", ""),
                    "n_flagged_phys": st.get("n_flagged_phys", ""),
                    "n_flagged_sigma": st.get("n_flagged_sigma", ""),
                    "n_flagged_mad": st.get("n_flagged_mad", ""),
                    "export_missing_as": "-9999" if export_as_9999.get() else "NaN",
                    "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                })
            if meta_rows:
                meta_path = path[:-4] + "_QAQC_metadata.csv" if path.lower().endswith(".csv") else path + "_QAQC_metadata.csv"
                pd.DataFrame(meta_rows).to_csv(meta_path, index=False)
            else:
                meta_path = ""
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export:\n{e}")
            return

        extra = f"\n\nSaved metadata:\n{meta_path}" if meta_path else ""
        messagebox.showinfo("Exported", f"Saved:\n{path}{extra}")

    def _create_derived_variable() -> None:
        """Create a derived variable from the currently selected variable for downstream ML/IT."""
        nonlocal df_working, df_original
        v = var_cb.get().strip()
        if v not in df_working.columns:
            messagebox.showerror("Error", "Selected variable was not found.")
            return
        try:
            subset_mask = _current_subset_mask()
        except Exception as e:
            messagebox.showerror("Subset error", str(e))
            return

        x = _safe_numeric_series(df_working[v]).copy()
        transform = transform_var.get()
        suffix = ""
        out = pd.Series(np.nan, index=df_working.index, dtype=float)

        if transform == "First difference":
            suffix = "DIFF"
            out.loc[subset_mask] = x.loc[subset_mask].diff()
        elif transform == "Log10":
            suffix = "LOG10"
            vals = x.loc[subset_mask]
            out.loc[subset_mask] = pd.Series(np.where(vals > 0, np.log10(vals), np.nan), index=vals.index)
        else:
            suffix = "ANOM"
            try:
                window = int(float(anomaly_window_var.get()))
            except Exception:
                window = 48
            window = max(3, window)
            vals = x.loc[subset_mask]
            rolling_mean = vals.rolling(window=window, center=True, min_periods=max(2, window // 4)).mean()
            out.loc[subset_mask] = vals - rolling_mean

        new_col_base = f"{_safe_filename_piece(v)}_{suffix}"
        new_col = new_col_base
        counter = 2
        while new_col in df_working.columns:
            new_col = f"{new_col_base}_{counter}"
            counter += 1

        df_working[new_col] = out
        df_original[new_col] = out
        if new_col not in vars_numeric:
            vars_numeric.append(new_col)
            var_cb.configure(values=vars_numeric)
        baseline[new_col] = _safe_numeric_series(df_original[new_col])
        masks[new_col] = pd.Series(False, index=df_working.index)
        status_txt.set(
            f"Created derived variable: {new_col}\n"
            f"Transform: {transform}\n"
            f"Subset rows used: {int(subset_mask.sum())}. Values outside the subset are NaN."
        )
        var_cb.set(new_col)
        _render_plot(new_col)

    def _update_main() -> None:
        if on_update_df is None:
            messagebox.showinfo("Not connected", "No callback was provided to update the main dataset.")
            return
        try:
            on_update_df(df_working.copy())
            messagebox.showinfo("Updated", "Main dataset updated with Standard QA/QC output.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update main dataset:\n{e}")

    def _on_var_change(event=None) -> None:
        v = var_cb.get().strip()
        _preload_limits_for_var(v)
        _render_plot(v)

    var_cb.bind("<<ComboboxSelected>>", _on_var_change)
    for _var in [subset_mode_var, ts_col_var, light_col_var, day_threshold_var, months_var, start_dt_var, end_dt_var, plot_subset_only_var]:
        try:
            _var.trace_add("write", lambda *_: _render_plot(var_cb.get().strip()))
        except Exception:
            pass

    # Buttons (left, one column)
    ttk.Label(left, text="Apply or export", font=("Arial", 10, "bold")).pack(anchor="w", pady=(4, 2))
    ttk.Button(left, text="Apply QA/QC to selected variable", command=_apply).pack(anchor="w", pady=(6, 4), fill="x")
    ttk.Button(left, text="Create derived variable", command=_create_derived_variable).pack(anchor="w", pady=4, fill="x")
    ttk.Button(left, text="Undo last apply", command=_undo_last).pack(anchor="w", pady=4, fill="x")
    ttk.Button(left, text="Reset ALL", command=_reset_all).pack(anchor="w", pady=4, fill="x")
    ttk.Button(left, text="Export FULL dataset", command=lambda: _export(True)).pack(anchor="w", pady=4, fill="x")
    ttk.Button(left, text="Export ONLY QA/QC variables", command=lambda: _export(False)).pack(anchor="w", pady=4, fill="x")
    ttk.Button(left, text="Update main dataset", command=_update_main).pack(anchor="w", pady=(10, 4), fill="x")

    # Init
    _preload_limits_for_var(vars_numeric[0])
    _render_plot(vars_numeric[0])
    _refresh_summary()
    win.mainloop()
