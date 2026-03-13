#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Leila Hernandez, LBNL
Date: Updated Oct 23, 2025

FFP GUI (satellite background) with:
 - Manual PBLH entry (default 700, user-editable)
 - Aerodynamic canopy height estimator (Pennypacker & Baldocchi, 2015)
 - Satellite basemap centered at (lat, lon) in the main window (big & fixed extent)
 - Aerodynamic canopy height plot opens in a separate Tk window (embedded canvas)
 - Draw red source-area contours for rs 10–90%
 - NEW: Option to draw contours for **every timestep** (not just stride)
 - NEW: Option to **accumulate** all contours (keep previous lines lighter)
 - Compute FETCH_70_M / FETCH_80_M / FETCH_90_M from contours (fallback to grid)
 - Compute FETCH_MAX_M (distance to peak contribution) from grid
 - Progress bar + ETA + Pause/Continue
 - Export PNGs (~10% stride) and a CSV with all per-timestep results
 - Uses calc_footprint_FFP_climatology.FFP_climatology (preferred) or calc_FFP_climatology.FFP_climatology
 - “_M” suffix means estimated with MeaningFlux
"""

import os
import io
import time
from datetime import datetime

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import Tk, Frame, LabelFrame, Button, Label, Entry, DoubleVar, Toplevel, Canvas
from tkinter import messagebox
from tkinter.ttk import Combobox, Progressbar, Scrollbar
from tkinter.scrolledtext import ScrolledText

# --- Matplotlib embedded in Tk (no external figure windows) ---
import matplotlib
matplotlib.use("TkAgg", force=True)
import matplotlib.pyplot as plt
plt.ioff()
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.dates import DateFormatter
from matplotlib.ticker import FuncFormatter
from matplotlib.figure import Figure

import contextily as ctx
from pyproj import Transformer

# Prefer calc_footprint_FFP_climatology; fallback to calc_FFP_climatology
try:
    import calc_footprint_FFP_climatology as ffp  # preferred
except Exception:
    try:
        import calc_FFP_climatology as ffp  # fallback
    except Exception as e:
        raise ImportError(
            "This script requires a module providing FFP_climatology(): "
            "calc_footprint_FFP_climatology or calc_FFP_climatology."
        ) from e

from contextlib import contextmanager, redirect_stdout, redirect_stderr

@contextmanager
def suppress_output():
    """Temporarily suppress stdout/stderr (for noisy library prints)."""
    with io.StringIO() as f_out, io.StringIO() as f_err:
        with redirect_stdout(f_out), redirect_stderr(f_err):
            yield

def format_hms(seconds: float) -> str:
    seconds = max(0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h > 0 else f"{m:d}:{s:02d}"

def degrees_to_dms(deg):
    d = int(deg); m = int((deg - d) * 60); s = (deg - d - m/60) * 3600
    return f"{d}°{m}'{s:.2f}\""

def _safe_slug(s: str) -> str:
    s = str(s).strip().replace(" ", "_")
    return "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in s)

def _stamp_compact(dt: datetime) -> str:
    # EXACTLY like 202108040000
    return dt.strftime("%Y%m%d%H%M")

def _ts_compact_from_any(x) -> str:
    """Return YYYYMMDDHHMM from a string like '202108040000' or a datetime-like."""
    try:
        s = str(x)
        if len(s) == 12 and s.isdigit():
            return s
        dt = pd.to_datetime(x)
        return dt.strftime("%Y%m%d%H%M")
    except Exception:
        return str(x)

# ---------- Missing-value policy: write -9999 at outputs ----------
MISSING_VAL = -9999.0
def _missing_if_nan(v):
    """Return float(v) if finite, else -9999.0."""
    try:
        return float(v) if np.isfinite(v) else MISSING_VAL
    except Exception:
        return MISSING_VAL

# ---------------------------------------------------------------------
# Helpers to compute Monin–Obukhov length (OL)
# ---------------------------------------------------------------------
def _to_pa(press_series: pd.Series | None) -> pd.Series | None:
    """Convert a pressure column to Pa using simple heuristics. If None, return None."""
    if press_series is None:
        return None
    p = pd.to_numeric(press_series, errors="coerce")
    med = float(np.nanmedian(p)) if np.isfinite(np.nanmedian(p)) else np.nan
    if not np.isfinite(med):
        return None
    # Heuristics: ~101 (kPa), ~1013 (hPa), or ~101325 (Pa)
    if med < 200:      # kPa
        return p * 1000.0
    elif med < 2000:   # hPa (mb)
        return p * 100.0
    else:              # assume Pa
        return p

def compute_monin_obukhov_length_from_columns(
    df: pd.DataFrame,
    ustar_col: str,
    H_col: str,
    TA_col: str,
    PRESS_col: str | None = None,
    out_col: str = "OL_M",
    rho_fallback: float = 1.2,
) -> pd.Series:
    """
    Compute Monin–Obukhov length L (m) and write to df[out_col].
    Formula: L = - (u_*^3 * T * ρ * c_p) / (k g H),
    with (w'θ') = H / (ρ c_p). Approximations: T≈air temperature (K),
    ρ≈p/(R_d T) if pressure available; else fallback ρ.
    """
    k = 0.4
    g = 9.81
    cp = 1004.67          # J kg⁻1 K⁻1
    Rd = 287.05           # J kg⁻1 K⁻¹

    ustar = pd.to_numeric(df[ustar_col], errors="coerce")
    H     = pd.to_numeric(df[H_col], errors="coerce")            # W m⁻2
    TA    = pd.to_numeric(df[TA_col], errors="coerce")

    # Temperature to Kelvin (heuristic: if looks like °C)
    T = TA.copy()
    T[T < 150] = T[T < 150] + 273.15   # assume °C if typical ambient temps

    # Air density
    rho = None
    if PRESS_col and (PRESS_col in df.columns):
        p_pa = _to_pa(df[PRESS_col])
        if p_pa is not None:
            rho = p_pa / (Rd * T)
    if rho is None:
        rho = pd.Series(rho_fallback, index=df.index)

    # Avoid divisions by ~0
    H_safe = H.where(np.abs(H) > 1e-9, np.nan)
    u3     = ustar**3

    # L = - (u_*^3 * T * ρ * c_p) / (k g H)
    L = - (u3 * T * rho * cp) / (k * g * H_safe)

    # Clean and clamp
    L = L.where(np.isfinite(L))
    L = L.clip(lower=-1e6, upper=1e6)

    df[out_col] = L
    return L

# ---------------------------------------------------------------------
# Shared default for manual PBLH
# ---------------------------------------------------------------------
class updated_pblh:
    default_pbhl = 700.0  # meters

# ---------------------------------------------------------------------
# FETCH metrics helpers
# ---------------------------------------------------------------------
def _compute_fetch_metrics(grid, x, y, _assumed_fetch):
    """Compute crosswind-integrated cumulative distances from a 2D grid."""
    if grid is None or x is None or y is None:
        return np.nan, np.nan, np.nan, np.nan
    g = np.asarray(grid, dtype=float)
    if g.ndim != 2 or g.size == 0 or not np.isfinite(g).any():
        return np.nan, np.nan, np.nan, np.nan

    xv = np.asarray(x)[0, :] if np.asarray(x).ndim == 2 else np.asarray(x)
    yv = np.asarray(y)[:, 0] if np.asarray(y).ndim == 2 else np.asarray(y)

    wy = np.gradient(yv)  # spacing in y
    fx = np.nansum(g * wy[:, None], axis=0)
    fx = np.maximum(fx, 0)
    total = np.nansum(fx)
    if not np.isfinite(total) or total <= 0:
        return np.nan, np.nan, np.nan, np.nan

    cx = np.cumsum(fx)

    def level(pct):
        idx = int(np.searchsorted(cx, (pct/100.0) * total, side='left'))
        idx = np.clip(idx, 0, len(xv)-1)
        return float(xv[idx])

    f70 = level(70); f80 = level(80); f90 = level(90)
    fmax = float(xv[int(np.nanargmax(fx))])
    return f70, f80, f90, fmax

def _fetch_from_contours(xr, yr, levels_pct, target_pct):
    """Distance (x) where the rs=target_pct contour crosses y=0."""
    if xr is None or yr is None:
        return np.nan

    def _as_list(a): return a if isinstance(a, (list, tuple)) else [a]
    xrL, yrL = _as_list(xr), _as_list(yr)
    try:
        idx = levels_pct.index(target_pct)
    except ValueError:
        return np.nan
    if idx >= len(xrL) or xrL[idx] is None or yrL[idx] is None:
        return np.nan

    x_seg = np.asarray(xrL[idx], dtype=float)
    y_seg = np.asarray(yrL[idx], dtype=float)
    if x_seg.size == 0 or y_seg.size == 0 or np.all(~np.isfinite(y_seg)):
        return np.nan

    crossings = np.where(np.diff(np.signbit(y_seg)))[0]
    if crossings.size == 0:
        j = int(np.nanargmin(np.abs(y_seg)))
        return float(x_seg[j])
    i = int(crossings[0])
    x0, y0 = x_seg[i], y_seg[i]
    x1, y1 = x_seg[i+1], y_seg[i+1]
    if not np.isfinite(y0) or not np.isfinite(y1) or (y1 - y0) == 0:
        return float(x0)
    r = -y0 / (y1 - y0)
    return float(x0 + r*(x1 - x0))

def _fetch_max_from_grid(grid, x, y):
    if grid is None or x is None or y is None:
        return np.nan
    g = np.asarray(grid, dtype=float)
    if g.ndim != 2 or g.size == 0:
        return np.nan
    xv = np.asarray(x)[0, :] if np.asarray(x).ndim == 2 else np.asarray(x)
    fx = np.nansum(g, axis=0)
    idx = int(np.nanargmax(fx))
    idx = np.clip(idx, 0, len(xv)-1)
    return float(xv[idx])

# ---------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------
class FFPCalculationsWindow:
    instance = None

    def __init__(self, df, updated_values, inputname_site):
        if FFPCalculationsWindow.instance is not None:
            FFPCalculationsWindow.instance.root.lift()
            return
        FFPCalculationsWindow.instance = self

        self.df = df
        self.updated_values = updated_values
        self.site_name = str(inputname_site).strip()  # <- explicit site passed in

        self.root = Tk()
        self.root.title(f'FFP calculations — {self.site_name}')
        self.root.geometry("1500x1200")  # BIG window (change size here)

        self.widgets = {}
        self.variables = {}
        self.abort_flag = False  # used as PAUSE flag now

        # --------- MAP FIGURE (big & fixed extent) ----------
        self.figure = Figure(figsize=(14, 10), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_aspect('equal')

        self.canvas = None
        self.rs_line_artists = []

        # Secondary HA plot window handle
        self.ha_window = None

        # Runtime options
        self.redraw_every = 5
        self.background_radius_factor = 3.0  # map radius = factor * assumed_fetch

        # NEW (Display toggles)
        self.accum_var = tk.BooleanVar(value=True)       # accumulate contours
        self.every_step_var = tk.BooleanVar(value=True)  # draw every timestep

        # Progress UI
        self.progress = None
        self.progress_label = None

        # Exports
        self.results_dir = os.path.join(os.getcwd(), "Results", "FFP_results")
        os.makedirs(self.results_dir, exist_ok=True)
        self.csv_rows = []
        self.saved_images = []
        self._last_start_dt = None
        self._last_end_dt = None

        self.setup_window()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    # ------------------- Layout -------------------
    def setup_window(self):

        # --- Header -----------------------------------------------------------
        hdr = tk.Frame(self.root)
        hdr.pack(side="top", fill="x", padx=12, pady=(12, 6))

        tk.Label(
            hdr,
            text=f"MeaningFlux • FFP (Flux Footprint & Fetch){' — ' + self.site_name if self.site_name else ''}",
            font=("TkDefaultFont", 13, "bold")
        ).pack(side="left")

        self.main_frame = Frame(self.root)
        self.main_frame.pack(padx=10, pady=10, fill='both', expand=True)

        
        # ------------------- SCROLLABLE LEFT PANEL (WIDER + RESPONSIVE) -------------------
        left_container = Frame(self.main_frame)
        left_container.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # make the left column take some width but not everything
        # (we'll control its min width via the canvas)
        self.main_frame.grid_columnconfigure(0, weight=0)  # left pane (fixed-ish)
        self.main_frame.grid_columnconfigure(1, weight=1)  # right pane grows
        self.main_frame.grid_rowconfigure(0, weight=1)
        
        LEFT_PANEL_WIDTH = 800  # <<—— widen here to taste (e.g., 560–700)
        
        self.left_canvas = Canvas(left_container, width=LEFT_PANEL_WIDTH, highlightthickness=0)
        left_scrollbar = Scrollbar(left_container, orient="vertical", command=self.left_canvas.yview)
        
        self.left_canvas.configure(yscrollcommand=left_scrollbar.set)
        self.left_canvas.grid(row=0, column=0, sticky="nsew")
        left_scrollbar.grid(row=0, column=1, sticky="ns")
        
        # allow the canvas to expand vertically within left_container
        left_container.grid_rowconfigure(0, weight=1)
        left_container.grid_columnconfigure(0, weight=1)
        
        self.left_frame = Frame(self.left_canvas)
        self.left_canvas.create_window((0, 0), window=self.left_frame, anchor="nw")
        
        def _on_left_configure(_event=None):
            # Update scroll region to encompass the inner frame
            self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))
        self.left_frame.bind("<Configure>", _on_left_configure)
        
        # Optional: keep the visible area at least LEFT_PANEL_WIDTH when the window resizes
        def _on_root_resize(_event=None):
            try:
                # Keep a minimum width; allow growing a bit if there’s space
                # (removing this preserves a fixed-width left pane)
                avail = min(self.main_frame.winfo_width(), LEFT_PANEL_WIDTH + 80)
                self.left_canvas.configure(width=max(LEFT_PANEL_WIDTH, avail//2))
            except Exception:
                pass
        self.root.bind("<Configure>", _on_root_resize)
        
        # Smooth mouse wheel behavior
        def _on_mousewheel(event):
            # Windows/Mac delta normalization
            if event.delta:     # Windows/Mac
                step = -1 * (event.delta // 120)
            else:               # X11 with Button-4/5 events
                step = 1 if getattr(event, 'num', 0) == 5 else -1
            self.left_canvas.yview_scroll(step, "units")
        
        self.left_frame.bind("<Enter>", lambda e: self.left_canvas.bind_all("<MouseWheel>", _on_mousewheel))
        self.left_frame.bind("<Leave>", lambda e: self.left_canvas.unbind_all("<MouseWheel>"))

        # ------------------- Plot area (right) -------------------
        self.plot_frame = Frame(self.main_frame)
        self.plot_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # let col 1 (plot area) take all extra space
        self.main_frame.grid_columnconfigure(0, weight=0)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        # Manual PBLH at top
        self.add_pblh_cell()

        # OL compute (optional) — goes right after PBLH
        self.add_ol_compute_cell()

        # Variable selection panel
        self.branch = LabelFrame(
            self.left_frame,
            text="Select variables for 2D Flux Footprint Prediction (FFP) (Kljun, 2015):"
        )
        self.branch.grid(row=2, column=0, padx=10, pady=10, sticky="n")

        self.variables = {
            'umean = Mean wind speed at zm [ms−1]': '',
            'pblh = Planetary boundary layer height [m] (see previous step)': '',
            'ol = Monin-Obukhov length [m]': '',
            'sigmav = Standard deviation of lateral velocity fluctuations [ms−1]': '',
            'ustar = Friction velocity [ms−1]': '',
            'wd = Wind direction [Decimal degrees]': ''
        }
        self.create_input_entries()

        # Date/time
        self.add_datetime_fields()

        # Buttons (estimate ha + run FFP)
        self.add_buttons()

        # Progress bar + ETA (+ Pause/Continue)
        self.add_progress_bar()

        # ------------------- NEW Display Options panel -------------------
        self.add_display_options()

        # Notes & Output box
        self.add_notes_box()

        # Plot canvas (map)
        self.create_plot_canvas()

        # Save mapping (silent)
        save_button = Button(self.branch, text="Save mapping", command=self.save_values)
        save_button.grid(row=len(self.variables), column=2, pady=10)

    # ------------------- Manual PBLH cell -------------------
    def add_pblh_cell(self):
        try:
            default_pbhl = float(getattr(updated_pblh, 'default_pbhl', 700.0))
        except Exception:
            default_pbhl = 700.0

        self.pblh_var = DoubleVar(value=default_pbhl)

        self.pblh_frame = LabelFrame(self.left_frame, text="Planetary Boundary Layer Height (manual)")
        self.pblh_frame.grid(row=0, column=0, padx=10, pady=(0, 10), sticky="ew")

        Label(self.pblh_frame, text="Assumed PBLH (m)").grid(row=0, column=0, padx=6, pady=6, sticky="w")
        self.pblh_entry = Entry(self.pblh_frame, textvariable=self.pblh_var, width=10)
        self.pblh_entry.grid(row=0, column=1, padx=6, pady=6, sticky="w")
        self.pblh_entry.bind("<FocusOut>", self._on_pblh_focus_out)

        Button(self.pblh_frame, text="Save PBLH to df",
               command=self.save_manual_pblh).grid(row=0, column=2, padx=8, pady=6)

        Label(self.pblh_frame,
              text="Tip: If you don’t have a PBLH column, save a manual value and select 'PBLH_assumed' below.",
              wraplength=360, justify="left").grid(row=1, column=0, columnspan=3, padx=6, pady=(0, 6), sticky="w")

    def _on_pblh_focus_out(self, _event=None):
        try:
            updated_pblh.default_pbhl = float(self.pblh_entry.get().strip())
        except Exception:
            pass

    def save_manual_pblh(self):
        try:
            val = float(self.pblh_entry.get().strip())
            if val <= 0:
                raise ValueError
        except ValueError:
            messagebox.showwarning("PBLH", "Please enter a positive numeric PBLH.")
            return
        self.df['PBLH_assumed'] = val

        pblh_label = 'pblh = Planetary boundary layer height [m] (see previous step)'
        cb = self.widgets.get(pblh_label)
        if cb is not None:
            existing = list(cb['values'])
            if 'PBLH_assumed' not in existing:
                cb['values'] = tuple(existing + ['PBLH_assumed'])
            cb.set('PBLH_assumed')
            self.variables[pblh_label] = 'PBLH_assumed'

    # ------------------- OL compute UI -------------------
    def add_ol_compute_cell(self):
        frm = LabelFrame(self.left_frame, text="Monin–Obukhov length (OL) — compute if missing")
        frm.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")
        frm.grid_columnconfigure(1, weight=1)

        Label(frm, text="USTAR [m s⁻1]").grid(row=0, column=0, padx=6, pady=4, sticky="w")
        cb_ustar = Combobox(frm, values=list(self.df.columns), width=26)
        Label(frm, text="H (sensible) [W m⁻2]").grid(row=1, column=0, padx=6, pady=4, sticky="w")
        cb_H = Combobox(frm, values=list(self.df.columns), width=26)
        Label(frm, text="TA [°C or K]").grid(row=2, column=0, padx=6, pady=4, sticky="w")
        cb_TA = Combobox(frm, values=list(self.df.columns), width=26)
        Label(frm, text="Pressure (optional)").grid(row=3, column=0, padx=6, pady=4, sticky="w")
        cb_P  = Combobox(frm, values=list(self.df.columns), width=26)

        def _guess(prefixes):
            for c in self.df.columns:
                if any(c.upper().startswith(p.upper()) for p in prefixes):
                    return c
            return ""

        cb_ustar.set(_guess(["USTAR"]))
        cb_H.set(_guess(["H_", "H", "SH", "H_S", "FSH"]))
        cb_TA.set(_guess(["TA", "T_AIR", "T_"]))
        cb_P.set(_guess(["P", "PA", "PRESS"]))

        cb_ustar.grid(row=0, column=1, padx=6, pady=4, sticky="ew")
        cb_H.grid(   row=1, column=1, padx=6, pady=4, sticky="ew")
        cb_TA.grid(  row=2, column=1, padx=6, pady=4, sticky="ew")
        cb_P.grid(   row=3, column=1, padx=6, pady=4, sticky="ew")

        info = (
            "L = - (u*^3 · T · ρ · c_p) / (k g H). ρ uses pressure if supplied; else ρ=1.2 kg m⁻³. "
            "T is auto-interpreted (°C→K when <150). Output column: OL_M."
        )
        Label(frm, text=info, wraplength=360, justify="left", fg="#444").grid(row=4, column=0, columnspan=2, padx=6, pady=(2, 6), sticky="w")

        def _run_compute_ol():
            u_col = cb_ustar.get().strip()
            h_col = cb_H.get().strip()
            t_col = cb_TA.get().strip()
            p_col = cb_P.get().strip() or None

            missing = [nm for nm,val in [("USTAR",u_col),("H",h_col),("TA",t_col)] if not val]
            if missing:
                messagebox.showwarning("Missing selection", f"Please select: {', '.join(missing)}.")
                return

            try:
                compute_monin_obukhov_length_from_columns(self.df, u_col, h_col, t_col, p_col, out_col="OL_M")
            except Exception as e:
                messagebox.showerror("OL compute failed", str(e))
                return

            ol_label = 'ol = Monin-Obukhov length [m]'
            cb = self.widgets.get(ol_label)
            if cb is not None:
                vals = list(cb['values'])
                if 'OL_M' not in vals:
                    cb['values'] = tuple(vals + ['OL_M'])
                cb.set('OL_M')
                self.variables[ol_label] = 'OL_M'

            messagebox.showinfo("Done", "Computed OL_M and added to the dataset.\nThe OL selector now points to OL_M.")

        Button(frm, text="Compute OL", command=_run_compute_ol).grid(row=5, column=0, columnspan=2, padx=6, pady=6, sticky="ew")

    # ------------------- Variable pickers -------------------
    def create_input_entries(self):
        for i, (label_text, _) in enumerate(self.variables.items()):
            label = Label(self.branch, text=label_text)
            label.grid(row=i, column=0, padx=10)

            self.widgets[label_text] = Combobox(self.branch)
            self.widgets[label_text]['values'] = list(self.df.columns)
            self.widgets[label_text].grid(row=i, column=1, padx=10, pady=5)
            self.widgets[label_text].bind("<<ComboboxSelected>>", self.on_combobox_select(label_text))

            self.predictive_select(label_text)

    def predictive_select(self, label_text):
        initial_chars_map = {
            'umean = Mean wind speed at zm [ms−1]': 'WS',
            'pblh = Planetary boundary layer height [m] (see previous step)': 'PBLH',
            'ol = Monin-Obukhov length [m]': 'MO',
            'sigmav = Standard deviation of lateral velocity fluctuations [ms−1]': 'V_',
            'ustar = Friction velocity [ms−1]': 'USTAR_',
            'wd = Wind direction [Decimal degrees]': 'WD'
        }
        if label_text in initial_chars_map:
            initial = initial_chars_map[label_text]
            for col in self.df.columns:
                if col.startswith(initial):
                    self.widgets[label_text].set(col)
                    self.variables[label_text] = col
                    break

    def on_combobox_select(self, label_text):
        def callback(_event):
            selected_variable = self.widgets[label_text].get()
            if selected_variable:
                self.variables[label_text] = selected_variable
        return callback

    # ------------------- Date/time -------------------
    def add_datetime_fields(self):
        dt_frame = LabelFrame(self.left_frame, text="Select Start and End Date/Time")
        dt_frame.grid(row=3, column=0, padx=10, pady=10)

        min_ts = pd.to_datetime(self.df['TIMESTAMP_START']).min()
        max_ts = pd.to_datetime(self.df['TIMESTAMP_START']).max()

        Label(dt_frame, text="Start Date (YYYY-MM-DD)").grid(row=0, column=0, padx=10, pady=5)
        self.start_date_entry = Entry(dt_frame); self.start_date_entry.insert(0, min_ts.strftime("%Y-%m-%d"))
        self.start_date_entry.grid(row=0, column=1, padx=10, pady=5)

        Label(dt_frame, text="Start Time (HH:MM)").grid(row=0, column=2, padx=10, pady=5)
        self.start_time_entry = Entry(dt_frame); self.start_time_entry.insert(0, min_ts.strftime("%H:%M"))
        self.start_time_entry.grid(row=0, column=3, padx=10, pady=5)

        Label(dt_frame, text="End Date (YYYY-MM-DD)").grid(row=1, column=0, padx=10, pady=5)
        self.end_date_entry = Entry(dt_frame); self.end_date_entry.insert(0, max_ts.strftime("%Y-%m-%d"))
        self.end_date_entry.grid(row=1, column=1, padx=10, pady=5)

        Label(dt_frame, text="End Time (HH:MM)").grid(row=1, column=2, padx=10, pady=5)
        self.end_time_entry = Entry(dt_frame); self.end_time_entry.insert(0, max_ts.strftime("%H:%M"))
        self.end_time_entry.grid(row=1, column=3, padx=10, pady=5)

    # ------------------- Buttons -------------------
    def add_buttons(self):
        est_ha_btn = Button(self.left_frame, text='Estimate ha (Pennypacker & Baldocchi, 2015)', command=self.estimate_ha)
        est_ha_btn.grid(row=4, column=0, columnspan=2, pady=(0, 6), sticky="ew")

        start_ffp_button = Button(self.left_frame, text='Start FFP Estimation (Kljun, 2015)', command=self.open_and_run_FFP_calc)
        start_ffp_button.grid(row=5, column=0, columnspan=2, pady=6, sticky="ew")

        export_btn = Button(self.left_frame, text='Export results now', command=self._export_csv_now)
        export_btn.grid(row=6, column=0, columnspan=2, pady=(0, 6), sticky="ew")

        self.start_ffp_button = start_ffp_button

    # ------------------- Progress bar + Pause/Continue -------------------
    def add_progress_bar(self):
        p = LabelFrame(self.left_frame, text="Processing")
        p.grid(row=7, column=0, padx=10, pady=10, sticky="ew")

        self.progress = Progressbar(p, mode='determinate', length=320, maximum=100)
        self.progress.grid(row=0, column=0, columnspan=3, padx=6, pady=6, sticky="ew")

        self.progress_label = Label(p, text="Idle")
        self.progress_label.grid(row=1, column=0, columnspan=3, padx=6, pady=(0, 6), sticky="w")

        def _pause():
            self.abort_flag = True
            try:
                self.progress_label.config(text="Paused…")
                cont_btn.config(state="normal")
                pause_btn.config(state="disabled")
            except Exception:
                pass

        def _continue():
            self.abort_flag = False
            try:
                self.progress_label.config(text="Resuming…")
                cont_btn.config(state="disabled")
                pause_btn.config(state="normal")
            except Exception:
                pass

        pause_btn = Button(p, text="Pause", command=_pause)
        pause_btn.grid(row=2, column=1, padx=6, pady=(0, 6), sticky="e")

        cont_btn = Button(p, text="Continue", command=_continue, state="disabled")
        cont_btn.grid(row=2, column=2, padx=6, pady=(0, 6), sticky="w")

        self._pause_btn = pause_btn
        self._cont_btn = cont_btn

    # ------------------- NEW: Display Options -------------------
    def add_display_options(self):
        disp = LabelFrame(self.left_frame, text="Display")
        disp.grid(row=8, column=0, padx=10, pady=10, sticky="ew")

        # Accumulate all contours (keeps previously drawn timesteps)
        tk.Checkbutton(
            disp, text="Accumulate contours (all timesteps)",
            variable=self.accum_var
        ).grid(row=0, column=0, sticky="w", padx=6, pady=4)

        # Draw a contour for every timestep (instead of stride-only)
        tk.Checkbutton(
            disp, text="Draw every timestep",
            variable=self.every_step_var
        ).grid(row=1, column=0, sticky="w", padx=6, pady=4)

    # ------------------- Notes box -------------------
    def add_notes_box(self):
        notes = LabelFrame(self.left_frame, text="Notes & Output")
        notes.grid(row=9, column=0, padx=10, pady=10, sticky="ew")
        notes.grid_columnconfigure(0, weight=1)

        folder = os.path.abspath(self.results_dir)
        msg = (
            " - The suffix “_M” indicates variables estimated with MeaningFlux.\n"
            " - FETCH_70_M / FETCH_80_M / FETCH_90_M are the downwind distances (x) at which the "
            "cross-wind–integrated cumulative footprint reaches 70%, 80%, and 90%, respectively.\n"
            " - FETCH_MAX_M is the downwind distance of maximum footprint contribution.\n"
            " - Fetch values are negative by convention (upwind); use |x| for distance.\n"
            " - The satellite map displays cumulative footprint (rs) contours at 0.10–0.90 (10%–90%).\n"
            " - Exported PNGs are saved at ~10% stride to avoid too many files; you can still draw every timestep in the GUI.\n"
            f" - Exports saved to:\n   {folder}\n"
            " - CSV name pattern: FFP_results_<site>_<YYYYMMDDHHMM>_<YYYYMMDDHHMM>.csv"
        )

        st = ScrolledText(notes, height=7, wrap="word")
        st.grid(row=0, column=0, sticky="ew")
        st.insert("1.0", msg)
        st.configure(state="disabled")

    # ------------------- Plot canvas (MAP) -------------------
    def create_plot_canvas(self):
        if self.canvas is None:
            self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
            self.canvas.get_tk_widget().pack(fill='both', expand=True)

    # ------------------- Aerodynamic plot in separate Tk window -------------------
    def estimate_ha(self):
        z_m = float(self.updated_values.z)
        daily_ha = compute_aerodynamic_canopy_height(self.df, z_m)

        if self.ha_window is not None and self.ha_window.winfo_exists():
            self.ha_window.lift()
            for child in self.ha_window.winfo_children():
                child.destroy()
        else:
            self.ha_window = Toplevel(self.root)
            self.ha_window.title(f'MeaningFlux • Aerodynamic Canopy Height — {self.site_name}')
            self.ha_window.geometry("980x420")

        fig = Figure(figsize=(9.6, 3.8), dpi=100)
        ax = fig.add_subplot(111)
        ax.scatter(daily_ha['TIMESTAMP_START'], daily_ha['ha_canopy'],
                   alpha=0.25, s=10, label='Daily mean')
        ax.plot(daily_ha['TIMESTAMP_START'], daily_ha['mov_avg'],
                linewidth=1.6, label='7-day MA')
        ax.xaxis.set_major_formatter(DateFormatter("%Y/%m/%d"))
        ax.set_ylabel('ha (m)')
        ax.grid(True, linewidth=0.5, alpha=0.3)
        ax.legend(loc='upper left', ncol=2, frameon=True, fontsize=9)
        fig.autofmt_xdate()

        canvas = FigureCanvasTkAgg(fig, master=self.ha_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        note_text = (
            "The aerodynamic canopy height (ha) is estimated via a Monin-Obukhov similarity approach "
            "(Pennypacker & Baldocchi, 2015; parameters: d = 0.6*h, z0 = 0.1*h). "
            "Note: any *_M variables in outputs indicate estimation with MeaningFlux."
        )
        tk.Label(self.ha_window, text=note_text, wraplength=900, justify="left").pack(pady=8)

    # ------------------- Helpers -------------------
    def save_values(self):
        pass

    def get_selected_variables(self):
        selected = {}
        for label_text, selected_variable in self.variables.items():
            if selected_variable:
                selected[label_text.split('=')[0].strip()] = selected_variable
        return selected

    def _dynamic_csv_path(self):
        site = _safe_slug(self.site_name)
        if self._last_start_dt is None or self._last_end_dt is None:
            name = f"FFP_results_{site}.csv"
        else:
            s = _stamp_compact(self._last_start_dt)
            e = _stamp_compact(self._last_end_dt)
            name = f"FFP_results_{site}_{s}_{e}.csv"
        return os.path.join(self.results_dir, name)

    def _export_csv_now(self):
        if not self.csv_rows:
            return
        df_csv = pd.DataFrame(self.csv_rows)
        if "TIMESTAMP_START" in df_csv.columns:
            df_csv["TIMESTAMP_START"] = df_csv["TIMESTAMP_START"].apply(_ts_compact_from_any)
        for col in ["FETCH_70_M", "FETCH_80_M", "FETCH_90_M", "FETCH_MAX_M"]:
            if col in df_csv.columns:
                df_csv[col] = pd.to_numeric(df_csv[col], errors="coerce").fillna(MISSING_VAL)
        out_csv = self._dynamic_csv_path()
        try:
            df_csv.to_csv(out_csv, index=False)
        except Exception:
            pass

    def update_rs_contours(self, xr_list, yr_list, center_xy, overlay=False, alpha=1.0, lw=2.6):
        """
        Draw rs contour(s) in red on the MAP axis.
        overlay=False removes prior lines; overlay=True keeps them (accumulation).
        NEW: alpha & lw allow lighter, thinner lines when accumulating.
        """
        if not overlay:
            for ln in getattr(self, "rs_line_artists", []):
                try:
                    ln.remove()
                except Exception:
                    pass
            self.rs_line_artists = []

        cx, cy = center_xy
        segments = []

        def _append(x, y):
            if x is None or y is None: return
            x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
            if x.size == 0 or y.size == 0: return
            segments.append((x + cx, y + cy))

        def _walk(xs, ys):
            if xs is None or ys is None: return
            if isinstance(xs, (list, tuple)) and len(xs) and isinstance(xs[0], (list, tuple, np.ndarray)):
                for a, b in zip(xs, ys):
                    _walk(a, b)
            else:
                _append(xs, ys)

        _walk(xr_list, yr_list)
        if not segments:
            return False

        for (xw, yw) in segments:
            ln, = self.ax.plot(xw, yw, color='red', lw=lw, alpha=alpha, zorder=20)
            self.rs_line_artists.append(ln)
        return True

    # ------------------- Run FFP -------------------
    def open_and_run_FFP_calc(self):
        # fresh run: make sure we're not paused
        self.abort_flag = False
        if hasattr(self, "_cont_btn"): self._cont_btn.config(state="disabled")
        if hasattr(self, "_pause_btn"): self._pause_btn.config(state="normal")

        try:
            if hasattr(self, "start_ffp_button"):
                self.start_ffp_button.config(state="disabled")
        except Exception:
            pass
        selected_variables = self.get_selected_variables()

        start_dt = datetime.strptime(f"{self.start_date_entry.get()} {self.start_time_entry.get()}", "%Y-%m-%d %H:%M")
        end_dt   = datetime.strptime(f"{self.end_date_entry.get()} {self.end_time_entry.get()}", "%Y-%m-%d %H:%M")

        # save for dynamic file naming
        self._last_start_dt = start_dt
        self._last_end_dt = end_dt

        saved_dt_vals = {
            'Start Year': start_dt.year, 'Start Month': start_dt.month, 'Start Day': start_dt.day,
            'Start Hour': start_dt.hour, 'Start Minute': start_dt.minute,
            'End Year': end_dt.year, 'End Month': end_dt.month, 'End Day': end_dt.day,
            'End Hour': end_dt.hour, 'End Minute': end_dt.minute,
        }

        _run_core(self.df, self.updated_values, saved_dt_vals, selected_variables, self)

    # ------------------- Mainloop -------------------
    def run(self):
        self.root.mainloop()

    def on_closing(self):
        try:
            self.abort_flag = True
        except Exception:
            pass
        FFPCalculationsWindow.instance = None
        self.root.destroy()

# ---------------------------------------------------------------------
# Window factory (requires explicit inputname_site)
# ---------------------------------------------------------------------
def calc_FFP_window(df, updated_values, inputname_site):
    if FFPCalculationsWindow.instance is None:
        FFPCalculationsWindow(df, updated_values, inputname_site)

# ---------------------------------------------------------------------
# Core runner: basemap + per-timestep FFP_climatology + exports
# ---------------------------------------------------------------------
def _setup_basemap(ax, lat, lon, radius, site_name=""):
    transformer_ll2wm = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    transformer_wm2ll = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    cx, cy = transformer_ll2wm.transform(lon, lat)

    ax.clear()
    ax.set_aspect('equal')
    ax.set_xlim(cx - radius, cx + radius)
    ax.set_ylim(cy - radius, cy + radius)
    try:
        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, zoom=14)
    except Exception:
        pass
    ax.plot(cx, cy, marker='+', ms=12, mew=2, color='cyan', zorder=15)

    def _fmt_lon(x, _pos, tr=transformer_wm2ll, cy_fixed=cy):
        return degrees_to_dms(tr.transform(x, cy_fixed)[0])
    def _fmt_lat(y, _pos, tr=transformer_wm2ll, cx_fixed=cx):
        return degrees_to_dms(tr.transform(cx_fixed, y)[1])

    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_lat))
    ax.tick_params(axis='x', labelrotation=30)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    title = "Flux Footprint Prediction — rs contours"
    if site_name:
        title += f" — {site_name}"
    ax.set_title(title)
    return (cx, cy)

def _run_core(df, updated_values, saved_FFP_datetime_values, selected_variables, window):
    # constants
    k = 0.4; d_h = 0.6; z0_h = 0.1

    lat, lon = float(updated_values.lat), float(updated_values.lon)
    tower_z = float(updated_values.z)
    assumed_fetch = tower_z * 100.0

    # time
    start_date = datetime(saved_FFP_datetime_values['Start Year'],
                          saved_FFP_datetime_values['Start Month'],
                          saved_FFP_datetime_values['Start Day'],
                          saved_FFP_datetime_values['Start Hour'],
                          saved_FFP_datetime_values['Start Minute'])
    end_date   = datetime(saved_FFP_datetime_values['End Year'],
                          saved_FFP_datetime_values['End Month'],
                          saved_FFP_datetime_values['End Day'],
                          saved_FFP_datetime_values['End Hour'],
                          saved_FFP_datetime_values['End Minute'])

    # keep original TIMESTAMP_START strings for CSV (e.g., 202108040000)
    ts_orig = df['TIMESTAMP_START'].copy()
    df['TIMESTAMP_START'] = pd.to_datetime(df['TIMESTAMP_START'])

    # filter
    mask = (df['TIMESTAMP_START'] >= start_date) & (df['TIMESTAMP_START'] <= end_date)
    if not mask.any():
        try:
            window.progress_label.config(text="No rows in the selected time range.")
        except Exception:
            pass
        if hasattr(window, "start_ffp_button"):
            window.start_ffp_button.config(state="normal")
        return None
    df_f = df.loc[mask].copy()

    # ensure ha_canopy (fallback estimator if not present)
