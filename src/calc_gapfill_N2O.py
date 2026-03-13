# calc_gapfill_N2O.py
"""
Gap-filling FN2O (N₂O flux) for AmeriFlux-style eddy covariance datasets
(MeaningFlux module; Python port inspired by Goodrich & Wall workflow).

Goodrich, J. P., Campbell, D. I., Schipper, L. A., Clearwater, M. J., Rutledge, S.,
Wall, A. M., & Hunt, J. E. (2021). Improved gap-filling approach and uncertainty
estimation for eddy covariance N₂O fluxes. Agricultural and Forest Meteorology, 297, 108238.
https://doi.org/10.1016/j.agrformet.2020.108238

What this module does
---------------------
  • Detects the N₂O flux column by name (case-insensitive), preferring exact 'FN2O' or 'N2O',
    otherwise any column that STARTS WITH 'FN2O' or 'N2O' (e.g., 'FN2O_1_1_1', 'n2o_flux').
  • Optional short-gap interpolation for tiny holes.
  • kNN model using available driver variables (with simple median imputation).
  • Seasonal climatology fallback (month/week/DOY median; week handling is index-aligned).
  • Keeps original column; writes <target> with qualifier order like: FN2O_1_1_1 → FN2O_F_1_1_1
    (i.e., inserts “_F” before trailing numeric qualifiers; if none, appends “_F”).
  • Preserves TIMESTAMP_START format (YYYYMMDDHHMM) on save. Saved CSV converts numeric NaN ↔ -9999.
"""

import os
import re
import tkinter as tk
from tkinter import messagebox
from tkinter.ttk import Combobox, Progressbar
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ---------------- citation (for display in header) ----------------
CITATION = (
    "Goodrich, J. P., Campbell, D. I., Schipper, L. A., Clearwater, M. J., "
    "Rutledge, S., Wall, A. M., & Hunt, J. E. (2021). Improved gap-filling approach and "
    "uncertainty estimation for eddy covariance N₂O fluxes. Agricultural and Forest Meteorology, 297, 108238. "
    "https://doi.org/10.1016/j.agrformet.2020.108238"
)

# ---------------- helpers ----------------
def _center_window(win: tk.Toplevel, width: int, height: int):
    """Center and size the window."""
    win.update_idletasks()
    try:
        x = (win.winfo_screenwidth() // 2) - (width // 2)
        y = (win.winfo_screenheight() // 2) - (height // 2)
        win.geometry(f"{width}x{height}+{x}+{y}")
    except Exception:
        pass

def _candidate_n2o_cols(df: pd.DataFrame) -> list[str]:
    """All columns that look like FN2O/N2O (equal to or starting with; case-insensitive)."""
    out = []
    for c in df.columns:
        cl = c.lower()
        if cl == "fn2o" or cl == "n2o" or cl.startswith("fn2o") or cl.startswith("n2o"):
            out.append(c)
    return out

def _season_key(dt: pd.DatetimeIndex, freq: str) -> pd.Series:
    """
    Return a Series indexed by dt for alignment-safety.
    Supports 'month' | 'week' | 'doy'.
    """
    f = (freq or "month").lower()
    if f == "month":
        return pd.Series(dt.month, index=dt)
    if f == "week":
        wk = dt.isocalendar().week
        return pd.Series(np.asarray(wk, dtype="int64"), index=dt)
    return pd.Series(dt.dayofyear, index=dt)

def _nan_count(s: pd.Series) -> int:
    try:
        return int(pd.isna(s).sum())
    except Exception:
        return -1

def _median_minutes(idx: pd.DatetimeIndex):
    try:
        diffs = np.diff(idx.view("int64"))  # ns
        if len(diffs) == 0:
            return None
        med_ns = float(np.median(diffs))
        return med_ns / 1e9 / 60.0
    except Exception:
        return None

def _safe_start(pb):
    try:
        if pb: pb.start(8)
    except Exception:
        pass

def _safe_stop(pb):
    try:
        if pb: pb.stop()
    except Exception:
        pass

def _results_dir(inputCSV: str | None, sub="n2o_gapfill") -> str:
    base = os.path.dirname(inputCSV) if inputCSV else os.getcwd()
    outdir = os.path.join(base, sub)
    os.makedirs(outdir, exist_ok=True)
    return outdir

def _save_ameriflux_csv(df: pd.DataFrame, path: str) -> None:
    """
    Write CSV with TIMESTAMP_START formatted as YYYYMMDDHHMM and
    numeric NaN -> -9999. Ensures TIMESTAMP_START is the first column.
    """
    out = df.copy()

    if "TIMESTAMP_START" not in out.columns:
        try:
            idx_dt = pd.to_datetime(out.index, errors="coerce")
            if not idx_dt.isna().all():
                out["TIMESTAMP_START"] = pd.Series(idx_dt).dt.tz_localize(None).dt.strftime("%Y%m%d%H%M")
            else:
                out["TIMESTAMP_START"] = ""
        except Exception:
            out["TIMESTAMP_START"] = ""
    else:
        s = out["TIMESTAMP_START"]
        if np.issubdtype(s.dtype, np.datetime64):
            out["TIMESTAMP_START"] = pd.to_datetime(s, errors="coerce").dt.tz_localize(None).dt.strftime("%Y%m%d%H%M")
        else:
            s_str = s.astype(str).replace({"NaT": "", "nat": "", "NaN": "", "nan": ""})
            parsed = pd.to_datetime(s_str, errors="coerce")
            if parsed.notna().any():
                fmt = parsed.dt.tz_localize(None).dt.strftime("%Y%m%d%H%M")
                bad = parsed.isna()
                if bad.any():
                    scrub = s_str.str.replace(r"\D", "", regex=True)
                    scrub = scrub.where(scrub.str.len() >= 10, "")
                    scrub = np.where(scrub.str.len() == 10, scrub + "00", scrub)
                    scrub = np.where(pd.Series(scrub).str.len() == 12, scrub, "")
                    fmt = fmt.mask(bad, scrub)
                out["TIMESTAMP_START"] = fmt
            else:
                scrub = s_str.str.replace(r"\D", "", regex=True)
                scrub = np.where(pd.Series(scrub).str.len() == 10, pd.Series(scrub) + "00", scrub)
                scrub = np.where(pd.Series(scrub).str.len() == 12, scrub, "")
                out["TIMESTAMP_START"] = scrub

    out["TIMESTAMP_START"] = out["TIMESTAMP_START"].astype(str)
    cols = ["TIMESTAMP_START"] + [c for c in out.columns if c != "TIMESTAMP_START"]
    out = out[cols]
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].fillna(-9999)
    out.to_csv(path, index=False)

def _is_timestamp_like(colname: str) -> bool:
    """Hide only columns that START WITH 'TIMESTAMP' (case-insensitive)."""
    return str(colname).strip().lower().startswith("timestamp")

def _is_numeric_like(series: pd.Series) -> bool:
    """Heuristic: coerce to numeric and require some valid numbers."""
    v = pd.to_numeric(series.replace(-9999, np.nan), errors="coerce")
    return int(v.notna().sum()) >= max(5, int(0.02 * len(v)))

def _make_filled_name(orig: str) -> str:
    """
    Insert '_F' before trailing numeric qualifiers if present.
      'FN2O_1_1_1' -> 'FN2O_F_1_1_1'
      'FN2O'       -> 'FN2O_F'
      'N2O_flux'   -> 'N2O_flux_F'
    """
    if "_F" in orig:
        return orig
    m = re.match(r"^(.*?)(_\d+(?:[_\d]+)*)$", orig)
    if m:
        base, quals = m.group(1), m.group(2)
        return f"{base}_F{quals}"
    return f"{orig}_F"

# ---- NEW: plausible driver detection ----------------------------------------
def _is_plausible_driver_name(name: str) -> bool:
    """
    Heuristics for likely N₂O drivers (case-insensitive).
    Soil moisture/VWC/θ/WFPS, water table depth, temperature (TA/TS),
    VPD, RH, precipitation, radiation/energy terms, GPP.
    """
    n = name.lower()

    # hide timestamp-like regardless
    if _is_timestamp_like(name):
        return False

    # soil moisture & water content
    if any(tok in n for tok in ["swc", "soil_moist", "soilmoist", "vwc", "theta", "wfps", "soilw"]):
        return True

    # water table depth
    if any(tok in n for tok in ["wtd", "water_table", "wt_depth", "watertable"]):
        return True

    # air & soil temperature (careful to avoid accidental 'ta' in other words)
    if re.match(r"^(ta|tair|air[_]?temp|airtemp)\b", n):  # TA / Tair
        return True
    if re.match(r"^(ts|soil[_]?temp|soiltemp)\b", n):     # TS
        return True

    # vapor pressure deficit / humidity
    if "vpd" in n or "relative_humidity" in n or re.match(r"^rh(_|$)", n):
        return True

    # precipitation / rainfall
    if any(tok in n for tok in ["precip", "rain", "ppt", "prcp"]):
        return True

    # radiation / energy terms
    if ("sw_in" in n) or re.match(r"^rg(\b|_)", n) or "rnet" in n or "par" in n or "ppfd" in n:
        return True
    if "soil_heat" in n or re.match(r"^g(_|$)", n):  # ground heat flux 'G'
        return True

    # carbon flux proxies (sometimes informative)
    if re.match(r"^gpp(\b|_)", n) or re.match(r"^nee(\b|_)", n):
        return True

    return False

# ---------------- public entry point (window) ----------------
def calc_gapfill_N2O(
    parent: tk.Tk,
    df_in: pd.DataFrame,
    inputname_site: str | None = None,
    inputCSV: str | None = None,
    shared_progressbar=None,
    on_update_df=lambda df: None,
):
    """
    Open the MeaningFlux N₂O Gap-Fill window (interactive).
    On success, calls on_update_df(updated_df).
    """
    if df_in is None or df_in.empty:
        messagebox.showwarning("MeaningFlux", "Load data first.")
        return

    # Detect candidates and default target
    n2o_candidates = _candidate_n2o_cols(df_in)
    if not n2o_candidates:
        messagebox.showerror("MeaningFlux", "No FN2O/N2O-like columns found.")
        return
    default_target = n2o_candidates[0]

    # --- build window
    win = tk.Toplevel(parent)
    win.title("MeaningFlux: N₂O Gap-Fill")
    _center_window(win, width=900, height=1000)

    # Header
    tk.Label(win, text="MeaningFlux • N₂O Gap-Fill",
             font=("TkDefaultFont", 13, "bold")).pack(anchor="w", padx=12, pady=(8,4))
    tk.Label(win, text=CITATION, wraplength=880, fg="#444", justify="left").pack(anchor="w", padx=12, pady=(0,6))

    # How-to panel
    how = tk.LabelFrame(win, text="How this works")
    how.pack(fill="x", padx=12, pady=(0,6))
    tk.Label(
        how,
        justify="left", wraplength=880, fg="#333",
        text=(
            "• Target: Choose the N₂O flux column (auto-detected; verify here).\n"
            "• Interpolation steps: Small holes only (e.g., 1–6); larger gaps left for kNN/seasonal.\n"
            "• k (kNN): Typical 3–10; requires ≥ max(20, k+3) measured points.\n"
            "• Seasonal freq: Fallback median by month/week/DOY when kNN cannot fill remaining gaps.\n"
            "• Drivers: We preselect plausible N₂O drivers (soil moisture/θ/VWC/WFPS, water table, TA, TS, VPD, RH, precipitation, radiation/energy, GPP). "
            "Columns starting with 'TIMESTAMP' are hidden."
        )
    ).pack(anchor="w", padx=8, pady=6)

    # Target selector
    target_frame = tk.LabelFrame(win, text="Target (N₂O flux)")
    target_frame.pack(fill="x", padx=12, pady=6)
    tk.Label(target_frame, text="Target column").grid(row=0, column=0, sticky="w", padx=8, pady=6)
    var_target = tk.StringVar(value=default_target)
    cmb_target = Combobox(target_frame, textvariable=var_target, values=n2o_candidates, state="readonly", width=46)
    cmb_target.grid(row=0, column=1, sticky="w", padx=8, pady=6)

    # Options
    opts = tk.LabelFrame(win, text="Options")
    opts.pack(fill="x", padx=12, pady=6)

    tk.Label(opts, text="Interpolation steps").grid(row=0, column=0, sticky="e", padx=4, pady=4)
    interp_steps = tk.IntVar(value=3)
    tk.Entry(opts, textvariable=interp_steps, width=6).grid(row=0, column=1, sticky="w", padx=6)

    tk.Label(opts, text="k (kNN)").grid(row=0, column=2, sticky="e", padx=4, pady=4)
    k_var = tk.IntVar(value=5)
    tk.Entry(opts, textvariable=k_var, width=6).grid(row=0, column=3, sticky="w", padx=6)

    tk.Label(opts, text="Seasonal freq").grid(row=0, column=4, sticky="e", padx=4, pady=4)
    season_cb = Combobox(opts, values=["month", "week", "doy"], width=10)
    season_cb.set("month")
    season_cb.grid(row=0, column=5, sticky="w", padx=6)

    save_csv_var = tk.IntVar(value=1)
    keep_nan_memory_var = tk.IntVar(value=1)
    tk.Checkbutton(opts, text="Save CSV copy + metadata TXT (includes preview snapshot)", variable=save_csv_var)\
        .grid(row=1, column=0, columnspan=3, sticky="w", padx=6, pady=(2,4))
    tk.Checkbutton(opts, text="Keep NaN in memory (CSV writes -9999 for numerics)", variable=keep_nan_memory_var)\
        .grid(row=1, column=3, columnspan=3, sticky="w", padx=6, pady=(2,4))

    # Drivers (shorter list so bottom buttons remain visible)
    drv_frame = tk.LabelFrame(win, text="Select drivers for kNN (TIMESTAMP* columns hidden)")
    drv_frame.pack(fill="both", expand=False, padx=12, pady=6)
    listbox = tk.Listbox(drv_frame, selectmode="extended", height=12)
    listbox.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

    btns_drv = tk.Frame(drv_frame)
    btns_drv.grid(row=0, column=1, sticky="ns", padx=(0,6), pady=6)
    def _select_all_drivers():
        listbox.select_set(0, "end")
    def _clear_drivers():
        listbox.selection_clear(0, "end")
    tk.Button(btns_drv, text="Select all", command=_select_all_drivers, width=10).pack(pady=(0,6))
    tk.Button(btns_drv, text="Clear", command=_clear_drivers, width=10).pack()

    drv_frame.columnconfigure(0, weight=1)
    drv_frame.rowconfigure(0, weight=1)

    def _populate_drivers_and_preselect():
        listbox.delete(0, "end")
        current_target = var_target.get().strip()
        candidates = []
        for c in df_in.columns:
            if c == current_target:
                continue
            if _is_timestamp_like(c):
                continue
            candidates.append(c)
            listbox.insert("end", c)

        # Auto-preselect plausible drivers
        for i, name in enumerate(candidates):
            if _is_plausible_driver_name(name):
                listbox.select_set(i)

    _populate_drivers_and_preselect()
    cmb_target.bind("<<ComboboxSelected>>", lambda e: _populate_drivers_and_preselect())

    # Preview
    prev = tk.LabelFrame(win, text="Preview / Summary")
    prev.pack(fill="both", expand=True, padx=12, pady=6)
    txt = tk.Text(prev, height=18, width=120)
    txt.pack(fill="both", padx=6, pady=6)
    pb = Progressbar(prev, orient="horizontal", mode="indeterminate", length=320)
    pb.pack(padx=6, pady=(0, 6))

    last_preview_lines: list[str] = []

    def _clear_preview():
        txt.configure(state="normal")
        txt.delete("1.0", "end")
        txt.configure(state="disabled")

    def _writeln(m):
        last_preview_lines.append(m)
        txt.configure(state="normal")
        txt.insert("end", m + ("\n" if not m.endswith("\n") else ""))
        txt.see("end")
        txt.configure(state="disabled")

    def _selected_drivers_filtered() -> list[str]:
        raw_sel = [listbox.get(i) for i in listbox.curselection()]
        tgt = var_target.get().strip()
        sel = [c for c in raw_sel if c != tgt and not _is_timestamp_like(c)]
        return sel

    def _preview():
        last_preview_lines.clear()
        _clear_preview()
        try:
            col = var_target.get().strip()
            _writeln(f"Target (N₂O) column: {col}")

            if "TIMESTAMP_START" in df_in.columns:
                idx = pd.to_datetime(df_in["TIMESTAMP_START"], errors="coerce")
            else:
                idx = pd.to_datetime(df_in.index, errors="coerce")

            if idx.isna().any():
                _writeln("! Some timestamps could not be parsed (expect YYYYMMDDHHMM).")

            step = _median_minutes(pd.DatetimeIndex(idx))
            _writeln(f"Median timestep: {step:.1f} min" if step else "Median timestep: n/a")

            sel = _selected_drivers_filtered()
            if not sel:
                _writeln("Drivers: (none selected)")
            else:
                _writeln(f"Drivers (auto-preselected where plausible): {', '.join(sel)}")
                for c in sel[:24]:
                    if c in df_in.columns:
                        _writeln(f"  - {c}: NaN = {_nan_count(df_in[c])}")

            _writeln(f"Interpolation steps: {interp_steps.get()}")
            _writeln(f"kNN k: {k_var.get()}")
            _writeln(f"Seasonal frequency: {season_cb.get()}")
            _writeln(f"Save CSV + metadata TXT: {bool(save_csv_var.get())}")
            _writeln(f"Keep NaN in memory (CSV writes -9999): {bool(keep_nan_memory_var.get())}")

        except Exception as e:
            _writeln(f"! Preview failed: {e}")

    def _run():
        _safe_start(pb)
        _safe_start(shared_progressbar)
        try:
            try:
                win.configure(cursor="watch")
                parent.configure(cursor="watch")
            except Exception:
                pass

            col = var_target.get().strip()

            # y = N2O series; -9999 -> NaN for processing
            y = pd.to_numeric(df_in[col], errors="coerce").replace(-9999, np.nan).astype(float)
            filled = y.copy()

            # --- 1) short-gap interpolation
            steps = max(0, int(interp_steps.get()))
            if steps > 0:
                interp = y.interpolate(method="linear", limit=steps, limit_direction="both")
                mask_interp = y.isna() & interp.notna()
                filled[mask_interp] = interp[mask_interp]
                interp_ct = int(mask_interp.sum())
            else:
                interp_ct = 0

            # --- 2) kNN using selected drivers (median-imputed), timestamps ignored
            sel = _selected_drivers_filtered()
            knn_ct = 0
            sel_used = []
            if sel:
                X = pd.DataFrame(index=df_in.index)
                for c in sel:
                    series = pd.to_numeric(df_in[c], errors="coerce").replace(-9999, np.nan)
                    if not _is_numeric_like(series):
                        continue
                    med = np.nanmedian(series.values) if np.isfinite(np.nanmedian(series.values)) else 0.0
                    X[c] = series.fillna(med).values
                    sel_used.append(c)

                if not X.empty:
                    known = filled.notna()
                    unknown = ~known
                    k = max(1, int(k_var.get()))
                    if known.sum() >= max(20, k + 3):
                        pipe = Pipeline([
                            ("scaler", StandardScaler()),
                            ("knn", KNeighborsRegressor(n_neighbors=k, weights="distance")),
                        ])
                        pipe.fit(X.loc[known, sel_used], filled.loc[known])
                        pred = pipe.predict(X.loc[unknown, sel_used])
                        filled.loc[unknown] = pred
                        knn_ct = int(unknown.sum())

            # --- 3) seasonal fallback (median by month/week/doy) — alignment-safe
            if "TIMESTAMP_START" in df_in.columns:
                idx = pd.to_datetime(df_in["TIMESTAMP_START"], errors="coerce")
            else:
                idx = pd.to_datetime(df_in.index, errors="coerce")

            key = _season_key(pd.DatetimeIndex(idx), season_cb.get())
            key = key.reindex(filled.index)  # align

            tmp = pd.DataFrame({"f": filled, "k": key}, index=filled.index)
            med = tmp.groupby("k")["f"].transform("median")
            mask = filled.isna() & med.notna()
            filled.loc[mask] = med.loc[mask]
            seasonal_ct = int(mask.sum())

            # --- attach to output with proper AmeriFlux qualifier order for _F
            df_out = df_in.copy()
            new_col = _make_filled_name(col)
            df_out[new_col] = filled.values  # keep NaNs in memory

            # save if asked (CSV + metadata TXT with preview snapshot)
            saved_csv = None
            saved_meta = None
            if bool(save_csv_var.get()):
                outdir = _results_dir(inputCSV, sub="n2o_gapfill")
                ts = datetime.now().strftime("%Y%m%d%H%M%S")
                base = (inputname_site or "site").replace(" ", "_") or "site"
                csv_path  = os.path.join(outdir, f"{base}_n2o_gapfill_{ts}.csv")
                meta_path = os.path.join(outdir, f"{base}_n2o_gapfill_{ts}_meta.txt")

                to_save = df_out.copy()
                if "TIMESTAMP_START" not in to_save.columns:
                    try:
                        idx_dt = pd.to_datetime(to_save.index, errors="coerce")
                        if not idx_dt.isna().all():
                            to_save["TIMESTAMP_START"] = pd.Series(idx_dt).dt.strftime("%Y%m%d%H%M")
                    except Exception:
                        pass
                _save_ameriflux_csv(to_save, csv_path)

                with open(meta_path, "w", encoding="utf-8") as f:
                    f.write("MeaningFlux — N2O Gap-Fill\n")
                    f.write(f"Site: {inputname_site or ''}\n")
                    f.write(f"Input CSV: {inputCSV or '(not provided)'}\n")
                    f.write(f"Output CSV: {csv_path}\n")
                    f.write(f"Time: {ts}\n")
                    f.write(f"Detected/selected target: {col} → output {new_col}\n")
                    f.write(f"Drivers used (TIMESTAMP* ignored): {', '.join(sel_used) if sel_used else '(none)'}\n")
                    f.write(f"Interpolation steps: {steps}\n")
                    f.write(f"kNN k: {int(k_var.get())} (filled ~{knn_ct} points)\n")
                    f.write(f"Seasonal fallback: {season_cb.get()} (filled ~{seasonal_ct} points)\n")
                    f.write(f"Short-gap interpolation filled ~{interp_ct} points\n")
                    f.write("NaN policy: In-memory DF keeps NaN; saved CSV writes -9999 for numeric columns.\n")
                    f.write("\n--- Preview snapshot ---\n")
                    if last_preview_lines:
                        for line in last_preview_lines:
                            f.write(line.rstrip() + "\n")
                    else:
                        f.write("(No preview was run before execution.)\n")

                saved_csv, saved_meta = csv_path, meta_path

            # hand back to main app
            on_update_df(df_out)

            # completion message
            bits = [f"N₂O gap-fill completed.", f"New column: {new_col}"]
            if inputname_site: bits.append(f"Site: {inputname_site}")
            if inputCSV: bits.append(f"Input CSV: {inputCSV}")
            if saved_csv:
                bits.append(f"Saved CSV:\n{saved_csv}")
                bits.append(f"Metadata:\n{saved_meta}")
            messagebox.showinfo("MeaningFlux", "\n".join(bits))

        except Exception as e:
            messagebox.showerror("MeaningFlux", f"Failed:\n{e}")
        finally:
            _safe_stop(pb)
            _safe_stop(shared_progressbar)
            try:
                win.configure(cursor="")
                parent.configure(cursor="")
            except Exception:
                pass

    # Preview/Run Buttons
    prev_btns = tk.Frame(prev)
    prev_btns.pack(fill="x", padx=6, pady=(0, 6))
    tk.Button(prev_btns, text="Preview", command=_preview, width=12).pack(side="left", padx=6, pady=6)
    tk.Button(prev_btns, text="Run", command=_run, width=12).pack(side="left", padx=6, pady=6)
    tk.Button(prev_btns, text="Close", command=win.destroy, width=10).pack(side="right", padx=6, pady=6)
