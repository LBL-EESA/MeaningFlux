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
"""

from __future__ import annotations

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


# ------------------- helpers -------------------
def _center(win: tk.Tk | tk.Toplevel, w: int, h: int) -> None:
    win.update_idletasks()
    x = (win.winfo_screenwidth() // 2) - (w // 2)
    y = (win.winfo_screenheight() // 2) - (h // 2)
    win.geometry(f"{w}x{h}+{x}+{y}")


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

    # Normalize time cols to datetime; never processed as numeric
    for tc in TIME_COLS:
        if tc in df_working.columns:
            df_working[tc] = pd.to_datetime(df_working[tc], errors="coerce")
            df_original[tc] = pd.to_datetime(df_original[tc], errors="coerce")

    # Convert -9999 to NaN for non-time cols only
    for c in df_working.columns:
        if c in TIME_COLS or pd.api.types.is_datetime64_any_dtype(df_working[c]):
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
    win.title("Standard QA/QC")
    _center(win, 1100, 600)

    outer = ttk.Frame(win, padding=10)
    outer.pack(fill="both", expand=True)

    left = ttk.Frame(outer, padding=10)
    left.pack(side="left", fill="y")

    mid = ttk.Frame(outer, padding=10)
    mid.pack(side="left", fill="both", expand=True)

    right = ttk.Frame(outer, padding=10)
    right.pack(side="right", fill="y")

    ttk.Label(left, text="Standard QA/QC", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 8))

    desc = (
        "How to use:\n"
        "1) Select a variable.\n"
        "2) Choose filters and thresholds.\n"
        "3) Click 'Apply to variable' for as many variables as you want.\n"
        "4) The summary panel shows all variables processed.\n"
        "5) Export when ready.\n\n"
        "Inputs:\n"
        "• Physical limits: min/max expected range.\n"
        "• σ: mean ± σ·std (typical 2–4).\n"
        "• MAD k: robust outlier filter (typical 3–5).\n"
        "Note: applying again rebuilds from baseline (no cascading)."
    )
    ttk.Label(left, text=desc, wraplength=320, justify="left", foreground="#444").pack(anchor="w", pady=(0, 10))

    ttk.Label(left, text="Variable:").pack(anchor="w")
    var_cb = ttk.Combobox(left, values=vars_numeric, state="readonly", width=28)
    var_cb.set(vars_numeric[0])
    var_cb.pack(anchor="w", pady=(0, 10))

    # Options vars
    use_phys_var = tk.BooleanVar(value=True)
    phys_min_var = tk.StringVar(value="")
    phys_max_var = tk.StringVar(value="")

    use_sigma_var = tk.BooleanVar(value=True)
    sigma_val = tk.StringVar(value="3")

    use_mad_var = tk.BooleanVar(value=False)
    mad_val = tk.StringVar(value="3.5")

    export_as_9999 = tk.BooleanVar(value=True)

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

    # -------- Plot area (mid) --------
    status_txt = tk.StringVar(value="Ready.")
    ttk.Label(mid, textvariable=status_txt, wraplength=560, justify="left").pack(anchor="w", pady=(0, 8))

    fig = plt.Figure(figsize=(7.2, 3.6), dpi=100)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=mid)
    canvas.get_tk_widget().pack(fill="both", expand=True)

    # -------- Summary (right) --------
    ttk.Label(right, text="Variables QA/QC'd", font=("Arial", 11, "bold")).pack(anchor="w")

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
        if v in reasonable_limits:
            lo, hi = reasonable_limits[v]
            phys_min_var.set(str(lo))
            phys_max_var.set(str(hi))
        else:
            # don't overwrite user's custom limits if already filled
            if phys_min_var.get().strip() == "" and phys_max_var.get().strip() == "":
                phys_min_var.set("")
                phys_max_var.set("")

    def _get_x_axis(df_: pd.DataFrame):
        if "TIMESTAMP_START" in df_.columns and pd.api.types.is_datetime64_any_dtype(df_["TIMESTAMP_START"]):
            return df_["TIMESTAMP_START"]
        return pd.RangeIndex(start=0, stop=len(df_), step=1)

    def _render_plot(v: str) -> None:
        ax.clear()
        x = _get_x_axis(df_working)
        y0 = _safe_numeric_series(df_original[v])
        y1 = _safe_numeric_series(df_working[v])
        ax.plot(x, y0, label="Original")
        ax.plot(x, y1, label="After Standard QA/QC")
        ax.set_title(f"{v}: before vs after")
        ax.set_xlabel("Time" if "TIMESTAMP_START" in df_working.columns else "Index")
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

        # Compute new mask from BASELINE
        x_base = baseline[v]
        mask_bad, mask_phys, mask_sig, mask_mad = _build_mask_from_baseline(
            x_baseline=x_base,
            use_physical=use_phys_var.get(),
            phys_lo=phys_lo,
            phys_hi=phys_hi,
            use_sigma=use_sigma_var.get(),
            sigma=sigma,
            use_mad=use_mad_var.get(),
            mad_k=mad_k,
        )

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
        df_working[flag_col] = ""

        # rebuild this variable from baseline every time
        df_working[v] = x_base.mask(mask_bad, np.nan)

        df_working.loc[mask_phys, flag_col] = (df_working.loc[mask_phys, flag_col] + "|PHYS").str.strip("|")
        df_working.loc[mask_sig, flag_col] = (df_working.loc[mask_sig, flag_col] + "|SIG").str.strip("|")
        df_working.loc[mask_mad, flag_col] = (df_working.loc[mask_mad, flag_col] + "|MAD").str.strip("|")

        stats = {
            "n_total": int(x_base.notna().sum()),
            "n_flagged_total": int(mask_bad.sum()),
            "n_flagged_phys": int(mask_phys.sum()),
            "n_flagged_sigma": int(mask_sig.sum()),
            "n_flagged_mad": int(mask_mad.sum()),
        }

        qaqc_vars.add(v)
        qaqc_stats[v] = stats
        _refresh_summary()

        status_txt.set(
            f"Applied QA/QC to {v}\n"
            f"- Non-missing (baseline): {stats['n_total']}\n"
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
            if c in TIME_COLS or pd.api.types.is_datetime64_any_dtype(df_working[c]):
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
            cols = [c for c in df_working.columns if c in TIME_COLS]
            cols += sorted(qaqc_vars)
            cols += [f"QAQC_FLAG_{v}" for v in sorted(qaqc_vars) if f"QAQC_FLAG_{v}" in df_working.columns]
            seen = set()
            cols = [c for c in cols if not (c in seen or seen.add(c))]
            df_out = df_working[cols].copy()

        if export_as_9999.get():
            for c in df_out.columns:
                if c in TIME_COLS or pd.api.types.is_datetime64_any_dtype(df_out[c]):
                    continue
                test = pd.to_numeric(df_out[c], errors="coerce")
                if test.notna().sum() > 0:
                    df_out[c] = test.replace(np.nan, -9999)

        try:
            df_out.to_csv(path, index=False)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export:\n{e}")
            return

        messagebox.showinfo("Exported", f"Saved:\n{path}")

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

    # Buttons (left, one column)
    ttk.Button(left, text="Apply to variable", command=_apply).pack(anchor="w", pady=(6, 4), fill="x")
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
