"""
ONEFlux (Py) dialog for MeaningFlux — AmeriFlux-aligned gap-filling & nighttime partition
(MeaningFlux module; Python wrapper around ONEFlux methods and conventions).

Citation
--------
The FLUXNET2015 dataset and the ONEFlux processing pipeline for eddy covariance data.
Scientific Data (Nature Research), 2020. https://www.nature.com/articles/s41597-020-0534-3

What this window does
---------------------
• VPD estimation:
  If a VPD column is not provided, VPD is estimated via TA+RH (preferred) or TA+Tdew using
  oneflux_py3.vpd_tools.ensure_vpd_series. Output column “VPD” (and “VPD_est_hPa” if estimated).
• MDS gap-filling (AmeriFlux/ONEFlux style):
  Fills missing values in drivers (TA, VPD, RG/SW_IN) and, optionally, NEE/FC using Marginal
  Distribution Sampling. Gap-filled outputs use the “_F” suffix (AmeriFlux v3.1.3 convention).
• Nighttime partition:
  Computes ecosystem respiration (Reco_nt) by fitting a temperature response to nighttime data,
  then GPP_nt = Reco_nt − NEE. Also outputs Rref, E0, and qf_partition_nt.
• Preview & QA:
  Shows time step, duplicate-timestamp count, NaN counts, and the VPD estimation plan before running.
• Progress & UX:
  Local indeterminate progress bar (and shared header progress bar if provided) while processing.
• Saving & metadata:
  Writes a CSV and a TXT metadata file into a “oneflux” results folder next to the input CSV.

Outputs & file conventions
--------------------------
• CSV formatting:
  - TIMESTAMP_START is written as YYYYMMDDHHMM (string).
  - NaNs are written as -9999 (numeric columns), per AmeriFlux practice.
  - All original (non gap-filled) columns are preserved.
• New/derived columns:
  - *_F  : gap-filled variables (drivers and/or NEE/FC).
  - Reco_nt, GPP_nt, Rref, E0, qf_partition_nt from nighttime partition.
• UI label:
  The window header includes a “MeaningFlux • ONEFlux” label for consistency with the suite.

Notes
-----
• Variable naming follows AmeriFlux variable conventions where applicable.
• “_F” denotes gap-filled values (AmeriFlux v3.1.3 data variable qualifiers).
• The dialog returns the updated DataFrame to the main app via a callback and optionally saves files.
"""

from __future__ import annotations
import os
import tkinter as tk
from tkinter import messagebox
from tkinter.ttk import Combobox, Progressbar
import pandas as pd
import numpy as np
import webbrowser
from datetime import datetime

# --- ONEFlux (Py) imports ---
try:
    from oneflux_py3.partition_nt import partition_nee_nighttime
    from oneflux_py3.mds_gapfill import mds_fill_nee, mds_fill_met
    from oneflux_py3.vpd_tools import ensure_vpd_series
    HAVE_ONEFLUX_PY = True
    _ONEFLUX_IMPORT_ERR = None
except Exception as _e:
    HAVE_ONEFLUX_PY = False
    _ONEFLUX_IMPORT_ERR = _e

ONEFLUX_URL = "https://ameriflux.lbl.gov/data/flux-data-products/oneflux-processing/"
AMF_VARS_URL = "https://ameriflux.lbl.gov/data/aboutdata/data-variables/"

# ----------------- small helpers -----------------
def _safe_config(widget, **kwargs):
    try:
        if widget and widget.winfo_exists():
            widget.configure(**kwargs)
    except tk.TclError:
        pass

def _center_window(win: tk.Tk | tk.Toplevel, width: int, height: int):
    win.update_idletasks()
    x = (win.winfo_screenwidth() // 2) - (width // 2)
    y = (win.winfo_screenheight() // 2) - (height // 2)
    win.geometry(f"{width}x{height}+{x}+{y}")

def _guess_col(_df: pd.DataFrame, candidates: list[str]):
    for c in candidates:
        if c in _df.columns:
            return c
    return None

def _make_unique_dtindex(dt_index: pd.Index) -> pd.DatetimeIndex:
    """
    Add microsecond bumps to duplicated timestamps to make the index unique,
    preserving order and row count.
    """
    s = pd.Series(dt_index)
    bumps = s.groupby(s).cumcount()
    return pd.to_datetime(s.values) + pd.to_timedelta(bumps.values, unit="us")

def _time_step_minutes(dt_index: pd.DatetimeIndex) -> float | None:
    """Return median time step in minutes, or None if cannot compute."""
    try:
        diffs = np.diff(dt_index.view("int64"))  # ns diffs
        if len(diffs) == 0:
            return None
        med_ns = np.median(diffs)
        return float(med_ns) / 1e9 / 60.0
    except Exception:
        return None

def _nan_count(series: pd.Series) -> int:
    return int(pd.isna(series).sum())

def _results_dir(inputCSV: str | None) -> str:
    base = os.path.dirname(inputCSV) if inputCSV else os.getcwd()
    outdir = os.path.join(base, "oneflux")
    os.makedirs(outdir, exist_ok=True)
    return outdir

def _write_metadata_txt(path: str, meta: dict[str, str | int | float | bool]):
    lines = []
    for k, v in meta.items():
        lines.append(f"{k}: {v}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

# -------- Modal busy / waiting overlay --------
class _BusyOverlay:
    def __init__(self, parent: tk.Tk | tk.Toplevel, text: str = "Working..."):
        self.parent = parent
        self.top = tk.Toplevel(parent)
        self.top.title("Please wait")
        self.top.transient(parent)
        self.top.grab_set()                  # modal
        self.top.resizable(False, False)
        self._msg = tk.StringVar(value=text)

        # Center near parent
        self.top.update_idletasks()
        w, h = 360, 120
        try:
            px = parent.winfo_rootx() + (parent.winfo_width() - w) // 2
            py = parent.winfo_rooty() + (parent.winfo_height() - h) // 2
        except Exception:
            px = (self.top.winfo_screenwidth() - w) // 2
            py = (self.top.winfo_screenheight() - h) // 2
        self.top.geometry(f"{w}x{h}+{px}+{py}")

        frm = tk.Frame(self.top, padx=14, pady=12)
        frm.pack(fill="both", expand=True)
        tk.Label(frm, textvariable=self._msg, anchor="w").pack(fill="x", pady=(0, 8))

        self.pb = Progressbar(frm, orient="horizontal", mode="indeterminate", length=300)
        self.pb.pack(fill="x")
        self.pb.start(10)

        # Disable close while running
        self.top.protocol("WM_DELETE_WINDOW", lambda: None)

    def set(self, text: str):
        self._msg.set(text)
        self.top.update_idletasks()

    def close(self):
        try:
            self.pb.stop()
        except Exception:
            pass
        try:
            self.top.grab_release()
        except Exception:
            pass
        self.top.destroy()

# ----------------- public entry point -----------------
def open_oneflux_window(
    parent: tk.Tk,
    df_in: pd.DataFrame,
    inputname_site: str | None,
    inputCSV: str | None,
    shared_progressbar,         # ttk.Progressbar instance from main header (optional)
    on_update_df,               # callback: def on_update_df(new_df): ...
):
    """
    Open the ONEFlux (Py) mini-window. On success calls on_update_df(updated_df).
    Produces outputs: *_F (gap-filled), Reco_nt, GPP_nt, Rref, E0, qf_partition_nt.
    Also writes a TXT metadata file alongside the CSV in a 'oneflux' subfolder.
    """
    if not HAVE_ONEFLUX_PY:
        messagebox.showwarning(
            "ONEFlux (Py) not available",
            f"Could not import oneflux_py3 components.\n\n{_ONEFLUX_IMPORT_ERR}"
        )
        return
    if df_in is None or df_in.empty:
        messagebox.showwarning("Warning", "Load the data first (Step 1).")
        return

    win = tk.Toplevel(parent)
    win.title("ONEFlux (Py): Gap-fill + Nighttime Partition")
    _center_window(win, 1040, 900)

    # -------- Header / Help --------
    hdr = tk.Frame(win)
    hdr.grid(row=0, column=0, columnspan=10, sticky="we", padx=12, pady=(12, 6))
    tk.Label(hdr, text="MeaningFlux • ONEFlux", font=("TkDefaultFont", 13, "bold")).grid(row=0, column=0, sticky="w")
    tk.Button(hdr, text="Open ONEFlux docs", command=lambda: webbrowser.open(ONEFLUX_URL)).grid(row=0, column=1, padx=12)
    tk.Button(hdr, text="AmeriFlux variables", command=lambda: webbrowser.open(AMF_VARS_URL)).grid(row=0, column=2, padx=6)

    # Brief descriptions
    desc = tk.LabelFrame(win, text="What this does")
    desc.grid(row=1, column=0, columnspan=10, sticky="we", padx=12, pady=6)
    tk.Label(
        desc, justify="left", wraplength=980,
        text=(
            "• VPD estimation (hPa): If no VPD column is provided, VPD is estimated using TA+RH (preferred) "
            "or TA+Tdew via a standard saturation vapor pressure relation.\n"
            "• MDS gap-filling: Fills missing values in drivers (TA, VPD, RG/SW_IN) and optionally NEE/FC "
            "using the Marginal Distribution Sampling approach. Gap-filled outputs use the suffix “_F”.\n"
            "• Nighttime partition: Estimates ecosystem respiration (Reco_nt) by fitting a temperature response "
            "to nighttime (low shortwave) NEE; computes GPP_nt = Reco_nt − NEE. Also outputs Rref, E0, "
            "and qf_partition_nt.\n"
            "• Saving: The CSV is written with TIMESTAMP_START formatted as YYYYMMDDHHMM and NaNs replaced with -9999. "
            "All non gap-filled columns are kept."
        ),
    ).grid(row=0, column=0, sticky="w", padx=8, pady=6)

    # -------- Inputs --------
    inputs = tk.LabelFrame(win, text="Inputs (select columns; units: TA [°C], VPD [hPa], RG/SW_IN [W m⁻²])")
    inputs.grid(row=2, column=0, columnspan=10, sticky="we", padx=12, pady=6)

    cols = list(df_in.columns)
    nee_default = _guess_col(df_in, ["NEE", "FC", "NEE_F", "FC_F"])
    ta_default  = _guess_col(df_in, ["TA", "TA_F", "TA_1_1_1", "TA_ERA"])
    vpd_default = _guess_col(df_in, ["VPD", "VPD_F", "VPD_ERA"])
    rg_default  = _guess_col(df_in, ["RG", "SW_IN", "SW_IN_F", "SW_IN_ERA", "SW_IN_POT"])
    rh_default   = _guess_col(df_in, ["RH","RH_F","RH_1_1_1","RH_ERA","REL_HUM","RH2M"])
    tdew_default = _guess_col(df_in, ["TDEW","TD","TDP","DEW","TDEW_ERA","TD_F"])

    tk.Label(inputs, text="NEE / FC").grid(row=0, column=0, sticky="e", padx=6, pady=4)
    tk.Label(inputs, text="TA (°C)").grid(row=1, column=0, sticky="e", padx=6, pady=4)
    tk.Label(inputs, text="VPD (hPa)").grid(row=2, column=0, sticky="e", padx=6, pady=4)
    tk.Label(inputs, text="RG / SW_IN (W m⁻²)").grid(row=3, column=0, sticky="e", padx=6, pady=4)

    nee_cb = Combobox(inputs, values=cols, width=38); nee_cb.set(nee_default or (cols[0] if cols else ""))
    ta_cb  = Combobox(inputs, values=cols, width=38); ta_cb.set(ta_default or (cols[0] if cols else ""))
    vpd_cb = Combobox(inputs, values=["<auto/estimate>"] + cols, width=38); vpd_cb.set(vpd_default or "<auto/estimate>")
    rg_cb  = Combobox(inputs, values=cols, width=38); rg_cb.set(rg_default or (cols[0] if cols else ""))

    nee_cb.grid(row=0, column=1, padx=6, pady=2, sticky="w")
    ta_cb.grid(row=1, column=1, padx=6, pady=2, sticky="w")
    vpd_cb.grid(row=2, column=1, padx=6, pady=2, sticky="w")
    rg_cb.grid(row=3, column=1, padx=6, pady=2, sticky="w")

    # Optional for VPD estimation
    tk.Label(inputs, text="(Optional) RH (%)").grid(row=4, column=0, sticky="e", padx=6, pady=(8,2))
    rh_cb = Combobox(inputs, values=cols, width=38)
    if rh_default: rh_cb.set(rh_default)
    rh_cb.grid(row=4, column=1, padx=6, pady=(8,2), sticky="w")

    tk.Label(inputs, text="(Optional) Tdew (°C)").grid(row=5, column=0, sticky="e", padx=6, pady=2)
    tdew_cb = Combobox(inputs, values=cols, width=38)
    if tdew_default: tdew_cb.set(tdew_default)
    tdew_cb.grid(row=5, column=1, padx=6, pady=2, sticky="w")

    # -------- Options --------
    opts = tk.LabelFrame(win, text="Options")
    opts.grid(row=3, column=0, columnspan=10, sticky="we", padx=12, pady=6)

    do_fill_drivers = tk.IntVar(value=1)
    do_fill_nee     = tk.IntVar(value=1)
    do_partition    = tk.IntVar(value=1)
    save_csv_copy   = tk.IntVar(value=1)

    tk.Checkbutton(opts, text="Fill drivers (TA, VPD, RG)", variable=do_fill_drivers).grid(row=0, column=0, padx=8, sticky="w")
    tk.Checkbutton(opts, text="Fill NEE via MDS",          variable=do_fill_nee).grid(row=0, column=1, padx=8, sticky="w")
    tk.Checkbutton(opts, text="Nighttime partition (Reco/GPP)", variable=do_partition).grid(row=0, column=2, padx=8, sticky="w")
    tk.Checkbutton(opts, text="Save CSV copy",             variable=save_csv_copy).grid(row=0, column=3, padx=8, sticky="w")

    tk.Label(
        opts, fg="#555",
        text="Output uses “_F” for gap-filled variables (AmeriFlux 3.1.3). Saved CSV writes NaNs as -9999."
    ).grid(row=1, column=0, columnspan=4, sticky="w", padx=8, pady=(4, 0))

    # -------- Advanced --------
    adv = tk.LabelFrame(win, text="Advanced (nighttime definition & fitting)")
    adv.grid(row=4, column=0, columnspan=10, sticky="we", padx=12, pady=6)

    tk.Label(adv, text="Night RG threshold (W m⁻²):").grid(row=0, column=0, sticky="e", padx=6)
    tk.Label(adv, text="Window days:").grid(row=0, column=2, sticky="e", padx=6)
    tk.Label(adv, text="Min points/fit:").grid(row=0, column=4, sticky="e", padx=6)

    rg_thresh_var = tk.DoubleVar(value=20.0)
    win_days_var  = tk.IntVar(value=10)
    min_pts_var   = tk.IntVar(value=30)

    tk.Entry(adv, textvariable=rg_thresh_var, width=8).grid(row=0, column=1, sticky="w")
    tk.Entry(adv, textvariable=win_days_var,  width=8).grid(row=0, column=3, sticky="w")
    tk.Entry(adv, textvariable=min_pts_var,   width=8).grid(row=0, column=5, sticky="w")

    # -------- Preview & Summary --------
    preview = tk.LabelFrame(win, text="Preview / Summary")
    preview.grid(row=5, column=0, columnspan=10, sticky="we", padx=12, pady=6)

    txt = tk.Text(preview, height=12, width=130)
    txt.grid(row=0, column=0, columnspan=9, padx=6, pady=6, sticky="we")

    # Local run progress bar (this one shows while RUN is executing)
    run_pb = Progressbar(preview, orient="horizontal", mode="indeterminate", length=260)
    run_pb.grid(row=1, column=0, sticky="w", padx=6, pady=(0,6))

    def _write(msg: str):
        txt.configure(state="normal")
        txt.insert("end", msg + ("\n" if not msg.endswith("\n") else ""))
        txt.see("end")
        txt.configure(state="disabled")

    def _clear_preview():
        txt.configure(state="normal")
        txt.delete("1.0", "end")
        txt.configure(state="disabled")

    def _preview():
        _clear_preview()
        try:
            # Build working index
            if "TIMESTAMP_START" in df_in.columns:
                idx = pd.to_datetime(df_in["TIMESTAMP_START"], errors="coerce")
            else:
                idx = pd.to_datetime(df_in.index, errors="coerce")

            if idx.isna().any():
                _write("! Some timestamps could not be parsed (expect YYYYMMDDHHMM).")
            else:
                step = _time_step_minutes(pd.DatetimeIndex(idx))
                _write(f"• Median time step: {step:.2f} min" if step else "• Median time step: n/a")

            dup_ct = int(pd.Index(idx).duplicated().sum())
            _write(f"• Duplicate timestamps: {dup_ct}")

            # Column selections
            nee_col = nee_cb.get().strip()
            ta_col  = ta_cb.get().strip()
            vpd_sel = vpd_cb.get().strip()
            rg_col  = rg_cb.get().strip()

            # NaN counts
            if nee_col in df_in.columns: _write(f"NEE/FC NaN: {_nan_count(df_in[nee_col])}")
            if ta_col  in df_in.columns: _write(f"TA NaN:  {_nan_count(df_in[ta_col])}")
            if rg_col  in df_in.columns: _write(f"RG NaN:  {_nan_count(df_in[rg_col])}")

            # VPD plan
            if (vpd_sel not in ("", "<auto/estimate>")) and (vpd_sel in df_in.columns):
                _write(f"VPD column: '{vpd_sel}' (NaN: {_nan_count(df_in[vpd_sel])})")
            else:
                rh_col = rh_cb.get().strip(); tdew_col = tdew_cb.get().strip()
                plan = "TA+RH" if rh_col in df_in.columns else ("TA+Tdew" if tdew_col in df_in.columns else "unavailable (provide RH or Tdew)")
                _write(f"VPD will be estimated using: {plan}")

            _write(f"Night RG threshold: {rg_thresh_var.get()} W m⁻² | Window: {win_days_var.get()} d | Min pts: {min_pts_var.get()}")

        except Exception as e:
            _write(f"! Preview failed: {e}")

    # Single Preview button
    tk.Button(preview, text="Preview", command=_preview).grid(row=1, column=1, padx=8, pady=(0,6), sticky="w")

    status = tk.Label(win, text="", anchor="w", fg="#555")
    status.grid(row=6, column=0, columnspan=10, sticky="we", padx=12, pady=(0,10))

    # -------- Run action --------
    def _run():
        nee_col = nee_cb.get().strip()
        ta_col  = ta_cb.get().strip()
        vpd_sel = vpd_cb.get().strip()
        rg_col  = rg_cb.get().strip()

        # Validate required selections (NEE/FC, TA, RG). VPD can be estimated.
        for c, name in [(nee_col, "NEE/FC"), (ta_col, "TA"), (rg_col, "RG/SW_IN")]:
            if c not in df_in.columns:
                messagebox.showerror("Missing column", f"Column for {name} not found: '{c}'")
                return

        # Start both progress bars (local + shared, if provided)
        run_pb.start(8)
        if shared_progressbar:
            try:
                shared_progressbar.start(8)
            except Exception:
                pass
        _safe_config(win, cursor="watch"); _safe_config(parent, cursor="watch")
        status.config(text="Running ONEFlux (Py)...")
        _clear_preview()

        try:
            out  = df_in.copy()

            # Build working frame with DateTimeIndex
            if "TIMESTAMP_START" in out.columns:
                idx = pd.to_datetime(out["TIMESTAMP_START"], errors="coerce")
            else:
                idx = pd.to_datetime(out.index, errors="coerce")
            if idx.isna().any():
                raise ValueError("TIMESTAMP_START could not be parsed as datetime for some rows (expect YYYYMMDDHHMM).")

            work = out.copy()
            work.index = idx
            work = work.sort_index()

            # Ensure unique timestamps (prevents pandas reindex error)
            dup_ct = int(work.index.duplicated().sum())
            if dup_ct > 0:
                work.index = _make_unique_dtindex(work.index)

            # Ensure/estimate VPD [hPa]
            vpd_existing = None if (vpd_sel in ("", "<auto/estimate>")) else vpd_sel
            rh_col   = rh_cb.get().strip()
            tdew_col = tdew_cb.get().strip()

            # Method string for metadata
            if vpd_existing and vpd_existing in work.columns:
                vpd_method_str = f"existing column '{vpd_existing}'"
            elif rh_col in work.columns:
                vpd_method_str = f"estimate TA+RH (RH: '{rh_col}')"
            elif tdew_col in work.columns:
                vpd_method_str = f"estimate TA+Tdew (Tdew: '{tdew_col}')"
            else:
                vpd_method_str = "estimate (no RH/Tdew found; will attempt)"

            work, vpd_used = ensure_vpd_series(
                work,
                ta_col=ta_col,
                vpd_col=vpd_existing if vpd_existing in work.columns else None,
                rh_col=rh_col if rh_col in work.columns else None,
                tdew_col=tdew_col if tdew_col in work.columns else None,
                method="TA+RH",                 # prefer TA+RH, fall back to TA+Tdew
                out_col="VPD",
                persist_estimate_as="VPD_est_hPa",
                inplace=True
            )
            if "VPD_est_hPa" in work.columns:
                out["VPD_est_hPa"] = work["VPD_est_hPa"].values

            # Count pre-fill NaNs
            pre_neenan = _nan_count(work[nee_col])
            pre_tanan  = _nan_count(work[ta_col])
            pre_vpdnan = _nan_count(work["VPD"])
            pre_rgnan  = _nan_count(work[rg_col])

            # Fill drivers (use _F suffix)
            if do_fill_drivers.get():
                ta_F_col  = f"{ta_col}_F"
                vpd_F_col = f"{vpd_used}_F" if vpd_existing else "VPD_F"
                rg_F_col  = f"{rg_col}_F"

                work[ta_F_col]  = mds_fill_met(work[ta_col])
                work[vpd_F_col] = mds_fill_met(work["VPD"])
                work[rg_F_col]  = mds_fill_met(work[rg_col])

                out[ta_F_col]  = work[ta_F_col].values
                out[vpd_F_col] = work[vpd_F_col].values
                out[rg_F_col]  = work[rg_F_col].values

                ta_src, vpd_src, rg_src = ta_F_col, vpd_F_col, rg_F_col
            else:
                ta_src, vpd_src, rg_src = ta_col, "VPD", rg_col

            # Fill NEE (use _F suffix)
            if do_fill_nee.get():
                nee_F_col = f"{nee_col}_F"
                work[nee_F_col] = mds_fill_nee(work[nee_col], work[rg_src], work[ta_src], work[vpd_src])
                out[nee_F_col]  = work[nee_F_col].values
                nee_src = nee_F_col
            else:
                nee_src = nee_col

            # Post-fill counts
            post_tanan  = _nan_count(work.get(ta_src, work[ta_col]))
            post_vpdnan = _nan_count(work.get(vpd_src, work["VPD"]))
            post_rgnan  = _nan_count(work.get(rg_src, work[rg_col]))
            post_neenan = _nan_count(work.get(nee_src, work[nee_col]))

            # Nighttime partition
            did_partition = False
            if do_partition.get():
                part = partition_nee_nighttime(
                    work[nee_src], work[ta_src], work[rg_src],
                    rg_thresh=float(rg_thresh_var.get()),
                    window_days=int(win_days_var.get()),
                    min_pts=int(min_pts_var.get())
                )
                out["Reco_nt"] = part["Reco"].values
                out["GPP_nt"]  = part["GPP"].values
                out["Rref"]    = part["Rref"].values
                out["E0"]      = part["E0"].values
                out["qf_partition_nt"] = part["qf_partition_nt"].values
                did_partition = True

            # Decide output paths
            outdir = _results_dir(inputCSV)
            ts = datetime.now().strftime("%Y%m%d%H%M%S")
            base = (inputname_site or "site").replace(" ", "_")
            csv_path  = os.path.join(outdir, f"{base}_onefluxpy_{ts}.csv")
            meta_path = os.path.join(outdir, f"{base}_onefluxpy_{ts}_meta.txt")

            # Prepare CSV for saving:
            # - TIMESTAMP_START as YYYYMMDDHHMM based on the (possibly deduped) index
            # - Keep all columns, with _F gap-filled additions
            # - Write NaNs as -9999
            out_to_save = out.copy()

            # TIMESTAMP_START in YYYYMMDDHHMM
            ts_str = work.index.strftime("%Y%m%d%H%M")
            out_to_save["TIMESTAMP_START"] = ts_str

            # Put TIMESTAMP_START first
            cols_save = ["TIMESTAMP_START"] + [c for c in out_to_save.columns if c != "TIMESTAMP_START"]
            out_to_save = out_to_save[cols_save]

            # Replace NaNs with -9999 everywhere (per request)
            out_to_save = out_to_save.where(pd.notna(out_to_save), -9999)

            # Save CSV (if option enabled)
            if save_csv_copy.get():
                out_to_save.to_csv(csv_path, index=False)

            # Return updated dataframe to main via callback (keep numeric NaNs in-memory)
            on_update_df(out)

            # Build metadata & save
            step_min = _time_step_minutes(work.index)
            meta = {
                "Run timestamp": ts,
                "Label": "MeaningFlux",
                "Use": "tower team or network team",
                "Site": inputname_site or "",
                "Input CSV": inputCSV or "(not provided)",
                "Results folder": outdir,
                "Results CSV": csv_path if save_csv_copy.get() else "(not saved by option)",
                "Metadata TXT": meta_path,
                "ONEFlux docs": ONEFLUX_URL,
                "AmeriFlux variables": AMF_VARS_URL,
                "Note on _F": "_F indicates gap-filled variable (AmeriFlux 3.1.3).",
                "NaN policy (CSV)": "NaNs written as -9999",
                "Selections": f"NEE:{nee_col} | TA:{ta_col} | RG:{rg_col} | VPD:{(vpd_existing or 'VPD (estimated)')}",
                "VPD method": vpd_method_str,
                "Night RG threshold (W m-2)": float(rg_thresh_var.get()),
                "Window days": int(win_days_var.get()),
                "Min points/fit": int(min_pts_var.get()),
                "Median time step (min)": f"{step_min:.2f}" if step_min is not None else "n/a",
                "Duplicate timestamps fixed": int(dup_ct),
                "NaN before fill (NEE,TA,VPD,RG)": f"{pre_neenan},{pre_tanan},{pre_vpdnan},{pre_rgnan}",
                "NaN after fill  (NEE,TA,VPD,RG)": f"{post_neenan},{post_tanan},{post_vpdnan},{post_rgnan}",
                "Gap-filled drivers": bool(do_fill_drivers.get()),
                "Gap-filled NEE": bool(do_fill_nee.get()),
                "Nighttime partition run": bool(did_partition),
                "Partition outputs": "Reco_nt, GPP_nt, Rref, E0, qf_partition_nt" if did_partition else "(not run)",
                "Citation": "AmeriFlux/ONEFlux Processing; see URLs above.",
            }
            _write_metadata_txt(meta_path, meta)

            # Summarize to UI and message
            _write("=== ONEFlux summary ===")
            _write(f"Results folder: {outdir}")
            if save_csv_copy.get():
                _write(f"CSV saved: {csv_path}")
            _write(f"Metadata saved: {meta_path}")
            _write(f"Time step (median): {meta['Median time step (min)']}")
            _write(f"Duplicates fixed: {dup_ct}")
            _write(f"NEE NaN: {pre_neenan} → {post_neenan}")
            _write(f"TA  NaN: {pre_tanan} → {post_tanan}")
            _write(f"VPD NaN: {pre_vpdnan} → {post_vpdnan}")
            _write(f"RG  NaN: {pre_rgnan} → {post_rgnan}")
            if did_partition:
                _write("Partition outputs: Reco_nt, GPP_nt, Rref, E0, qf_partition_nt")

            # --- build a user message without backslashes inside f-string expressions
            if save_csv_copy.get():
                csv_line = f"CSV:\n{csv_path}\n\n"
            else:
                csv_line = ""

            msg = (
                "ONEFlux processing completed.\n\n"
                f"Results folder:\n{outdir}\n\n"
                f"{csv_line}"
                f"Metadata:\n{meta_path}"
            )
            messagebox.showinfo("Done", msg)

        except Exception as e:
            messagebox.showerror("Error", f"Processing failed:\n{e}")
        finally:
            # Stop progress bars & reset cursors
            try:
                run_pb.stop()
            except Exception:
                pass
            if shared_progressbar:
                try:
                    shared_progressbar.stop()
                except Exception:
                    pass
            _safe_config(parent, cursor="")
            _safe_config(win, cursor="")
            status.config(text="")

    # Bottom buttons (Preview is above; no duplication)
    btns = tk.Frame(win)
    btns.grid(row=7, column=0, columnspan=10, sticky="w", padx=12, pady=(0,12))
    tk.Button(btns, text="Run", width=14, command=_run).grid(row=0, column=0, padx=6)
    tk.Button(btns, text="Cancel", width=12, command=win.destroy).grid(row=0, column=1, padx=6)
