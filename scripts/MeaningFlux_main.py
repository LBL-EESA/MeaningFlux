# -*- coding: utf-8 -*-
"""
-------------------------------------------------------
*** MeaningFlux v1.0 ***

Open-source Python GUI implementing the MeaningFlux analytical framework for eddy covariance data: 
standardized visualization, gap-filling, footprint/fetch metrics, machine-learning predictability, and information-theoretic diagnostics.

Author: Leila C Hernandez Rodriguez 
Lawrence Berkeley National Laboratory, Berkeley, CA, USA (lchernandezrodriguez@lbl.gov)
ORCID: 0000-0001-8830-345X

-------------------------------------------------------
*** Copyright Notice ***

MeaningFlux Copyright (c) 2026, The Regents of the University of California, through Lawrence Berkeley National Laboratory 
(subject to receipt of any required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software, please contact Berkeley Lab's 
Intellectual Property Office at IPO@lbl.gov.

NOTICE. This Software was developed under funding from the U.S. Department of Energy and the U.S. Government consequently retains certain rights. 
As such, the U.S. Government has been granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license 
in the Software to reproduce, distribute copies to the public, prepare derivative works, and perform publicly and display publicly, and to permit others to do so.
-------------------------------------------------------
"""
from pathlib import Path
import sys
import tkinter as tk
from tkinter import messagebox, filedialog
from tkinter.ttk import Progressbar
from tkinter import LabelFrame, Button

from PIL import Image, ImageTk
import pandas as pd
import numpy as np
import os

# --- Paths / imports ---
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    # Running as a packaged application
    ROOT = Path(sys._MEIPASS)
else:
    # Running from the source repository
    ROOT = Path(__file__).resolve().parents[1]

SRC = ROOT / "src"

if not getattr(sys, "frozen", False):
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

# ===== Project-specific imports =====
from calc_plot_time_series import calc_plot_time_series
from calc_plot_daily_avg import calc_plot_daily_avg
from calc_plot_wind_rose import calc_plot_wind_rose
from calc_plot_density_and_scatter import calc_plot_density_and_scatter
from calc_plot_correlations import calc_plot_correlations
from calc_plot_budgets import calc_plot_budgets

from calc_standard_QAQC import calc_standard_QAQC
from calc_data_AMF_BASE_QAQC import calc_data_AMF_BASE_QAQC
from calc_data_availability import calc_data_availability
from calc_plot_directional_contribution_rose import calc_plot_directional_contribution_rose

from calc_gapfill_oneflux import open_oneflux_window
from calc_gapfill_N2O import calc_gapfill_N2O
from calc_gapfill_CH4 import calc_gapfill_CH4

from calc_FFP_window import calc_FFP_window
from calc_plot_fetch_rose import calc_plot_fetch_rose

from open_machine_learning_toolbox import open_meaningflux_ml as open_machine_learning_toolbox
from open_information_theory_toolbox import open_information_theory_toolbox

from open_BADM_window import open_BADM_window, UpdatedValues


# ===== Globals =====
df = None
inputname = None
inputname_site = None
inputCSV = None


# ----------------------------- Utilities -----------------------------
def _center(win, w, h):
    win.update_idletasks()
    x = (win.winfo_screenwidth() // 2) - (w // 2)
    y = (win.winfo_screenheight() // 2) - (h // 2)
    win.geometry(f"{w}x{h}+{x}+{y}")


def _ensure_df():
    global df
    if df is None or df.empty:
        messagebox.showwarning("Warning", "Load EC data first in Step 1.")
        return False
    return True


def _call(func, *args):
    if not _ensure_df():
        return
    try:
        func(*args)
    except Exception as e:
        messagebox.showerror("Error", str(e))


def _find_header_row(path: str, key: str = "TIMESTAMP_START", default_row: int = 2) -> int:
    """Find row index containing header key, fallback to default_row."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if key in line:
                    return i
    except Exception:
        pass
    return default_row


def _safe_to_datetime(series: pd.Series, fmt: str | None = "%Y%m%d%H%M") -> pd.Series:
    """Parse datetime using given format, fallback to automatic parsing."""
    try:
        return pd.to_datetime(series.astype(str), format=fmt, errors="coerce")
    except Exception:
        return pd.to_datetime(series, errors="coerce")


# ================== LICENSE =============
def show_about():
    about_text = (
        "MeaningFlux v1.0\n\n"
        "Author: Leila C. Hernandez Rodriguez\n"
        "Lawrence Berkeley National Laboratory (LBNL)\n\n"
        "MeaningFlux Copyright (c) 2026, "
        "The Regents of the University of California, "
        "through Lawrence Berkeley National Laboratory "
        "(subject to receipt of any required approvals "
        "from the U.S. Dept. of Energy). All rights reserved.\n\n"
        "NOTICE. This Software was developed under funding from "
        "the U.S. Department of Energy. The U.S. Government "
        "retains a paid-up, nonexclusive, irrevocable, "
        "worldwide license to reproduce, distribute, "
        "prepare derivative works, and publicly display "
        "the Software.\n\n"
        "Released under the LBNL modified BSD license.\n"
        "See LICENSE.txt for details.\n\n"
        "For questions about rights to use or distribute "
        "this software, contact: IPO@lbl.gov"
    )

    messagebox.showinfo("About MeaningFlux", about_text)


def show_workflow_guide():
    guide_text = (
        "MeaningFlux recommended workflow\n\n"
        "1. Load data and metadata\n"
        "   Start with a post-processed EC CSV and add BADM, NDVI, or LAI when available.\n\n"
        "2. Explore, QA/QC, and footprint context\n"
        "   Inspect temporal patterns, missingness, quality flags, and source-area representativeness before modeling.\n\n"
        "3. Gap-fill when needed\n"
        "   Use gap-filling only when a complete time series is required for budgets, prediction, or information diagnostics.\n\n"
        "4. Run AI-assisted analysis\n"
        "   Use ML first to quantify predictability and export the bridge table. Then use IT to evaluate nonlinear dependence, lagged information transfer, redundancy, synergy, and information fidelity.\n\n"
        "Tip: The workflow is site-level. Repeat the same steps across sites to create comparable outputs."
    )
    messagebox.showinfo("MeaningFlux Workflow Guide", guide_text)


# ===================== GUI =====================
def MeaningFlux_main_window():
    global df, inputname, inputname_site, inputCSV

    root = tk.Tk()
    root.title("MeaningFlux")
    _center(root, 900, 460)
    root.minsize(860, 420)
    root.resizable(True, True)
    root.configure(bg="#f4f4f4")

    # -------------------------------------------------------------------------
    # Menu bar
    # -------------------------------------------------------------------------
    menubar = tk.Menu(root)
    help_menu = tk.Menu(menubar, tearoff=0)
    help_menu.add_command(label="Workflow guide", command=show_workflow_guide)
    help_menu.add_command(label="About MeaningFlux", command=show_about)
    menubar.add_cascade(label="Help", menu=help_menu)
    root.config(menu=menubar)

    # -------------------------------------------------------------------------
    # Main layout: persistent workflow sidebar + content dashboard
    # -------------------------------------------------------------------------
    shell = tk.Frame(root, bg="#f4f4f4")
    shell.pack(fill="both", expand=True, padx=10, pady=10)
    shell.grid_columnconfigure(0, weight=0)
    shell.grid_columnconfigure(1, weight=1)
    shell.grid_rowconfigure(0, weight=1)

    sidebar = tk.Frame(shell, bg="#e9e9e9", width=185, relief="flat")
    sidebar.grid(row=0, column=0, sticky="nsw")
    sidebar.grid_propagate(False)

    content = tk.Frame(shell, bg="#ffffff", relief="solid", bd=1)
    content.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
    content.grid_columnconfigure(0, weight=1)
    content.grid_rowconfigure(0, weight=1)

    # Logo in persistent sidebar to save vertical space in every step.
    try:
        img = Image.open(ROOT / "docs" / "MeaningFlux_logo.png")
        orig_w, orig_h = img.size
        target_h = 74
        if orig_h > target_h:
            scale = target_h / orig_h
            img = img.resize((int(orig_w * scale), target_h), Image.LANCZOS)
        logo = ImageTk.PhotoImage(img)
        logo_label = tk.Label(sidebar, image=logo, bg="#e9e9e9")
        logo_label.image = logo
    except Exception:
        logo_label = tk.Label(sidebar, text="MeaningFlux", font=("Arial", 16, "bold"), bg="#e9e9e9")
    logo_label.grid(row=0, column=0, sticky="we", padx=8, pady=(8, 4))

    status = tk.Label(
        sidebar,
        text="Load EC data to begin.",
        bg="#e9e9e9",
        fg="#555555",
        wraplength=160,
        justify="center",
        font=("Arial", 9),
    )
    status.grid(row=1, column=0, sticky="we", padx=8, pady=(0, 8))

    workflow_label = tk.Label(
        sidebar,
        text="Workflow",
        bg="#e9e9e9",
        fg="#333333",
        font=("Arial", 10, "bold"),
        anchor="w",
    )
    workflow_label.grid(row=2, column=0, sticky="we", padx=10, pady=(4, 4))

    # Hidden progressbar kept for compatibility with other modules.
    pb = Progressbar(sidebar, mode="indeterminate", length=140)

    step_frames = {}
    nav_buttons = {}
    analysis_buttons = []

    def section_note(parent, text, row=0, column=0, columnspan=2, wraplength=360):
        lbl = tk.Label(parent, text=text, fg="#666666", justify="left", wraplength=wraplength)
        lbl.grid(row=row, column=column, columnspan=columnspan, sticky="w", padx=4, pady=(0, 5))
        return lbl

    def compact_button(parent, text, command, row, column=0, state="disabled", width=None, padx=4, pady=4):
        b = Button(parent, text=text, state=state, command=command, width=width)
        b.grid(row=row, column=column, padx=padx, pady=pady, sticky="w")
        return b

    def show(step):
        for f in step_frames.values():
            f.grid_forget()
        step_frames[step].grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        for k, b in nav_buttons.items():
            b.config(bg="#e9e9e9")
        nav_buttons[step].config(bg="#d0d0d0")

        if df is None or getattr(df, "empty", True):
            status.config(text="Load EC data to begin.")
        else:
            status.config(text=f"Loaded\n{inputname}.csv")

    steps = [
        ("step1", "① Data"),
        ("step2", "② QA/QC"),
        ("step3", "③ Gap-fill"),
        ("step4", "④ ML–IT"),
    ]

    for i, (key, text) in enumerate(steps):
        state = "normal" if key == "step1" else "disabled"
        b = tk.Button(
            sidebar,
            text=text,
            anchor="w",
            bg="#e9e9e9",
            relief="flat",
            padx=10,
            state=state,
            command=lambda k=key: show(k),
        )
        b.grid(row=i + 3, column=0, sticky="we", padx=6, pady=2)
        nav_buttons[key] = b

    sidebar.grid_rowconfigure(8, weight=1)

    guide_btn = tk.Button(
        sidebar,
        text="Workflow guide",
        anchor="w",
        bg="#e9e9e9",
        relief="flat",
        padx=10,
        command=show_workflow_guide,
    )
    guide_btn.grid(row=9, column=0, sticky="we", padx=6, pady=(8, 2))

    about_btn = tk.Button(
        sidebar,
        text="About / license",
        anchor="w",
        bg="#e9e9e9",
        relief="flat",
        padx=10,
        command=show_about,
    )
    about_btn.grid(row=10, column=0, sticky="we", padx=6, pady=(2, 8))

    # -------------------------------------------------------------------------
    # STEP 1 – Data
    # -------------------------------------------------------------------------
    step1 = tk.Frame(content, bg="#ffffff")
    step_frames["step1"] = step1
    step1.grid_columnconfigure(0, weight=1)
    step1.grid_columnconfigure(1, weight=1)

    tk.Label(step1, text="Step 1 – Load EC data", bg="#ffffff", font=("Arial", 12, "bold")).grid(
        row=0, column=0, columnspan=2, sticky="w", pady=(0, 6)
    )
    tk.Label(
        step1,
        text="Start with a post-processed eddy covariance CSV. Optional metadata and vegetation variables can be added before analysis.",
        bg="#ffffff",
        fg="#555555",
        wraplength=660,
        justify="left",
    ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 8))

    lf_required = LabelFrame(step1, text="Required input", padx=6, pady=6)
    lf_required.grid(row=2, column=0, sticky="nwe", padx=(0, 6), pady=6)
    section_note(lf_required, "EC CSV with TIMESTAMP_START.", row=0, columnspan=2, wraplength=300)

    path_var = tk.StringVar(value="No EC file loaded")
    tk.Label(lf_required, textvariable=path_var, fg="#444", justify="left", wraplength=300).grid(
        row=1, column=0, columnspan=2, sticky="w", padx=4, pady=(0, 4)
    )

    summary_var = tk.StringVar(value="Dataset summary\nNot loaded yet.")

    def _dataset_summary_text(df_):
        if df_ is None or getattr(df_, "empty", True):
            return "Dataset summary\nNot loaded yet."
        n_rows = len(df_)
        n_cols = len(df_.columns)
        if "TIMESTAMP_START" in df_.columns:
            t = pd.to_datetime(df_["TIMESTAMP_START"], errors="coerce")
            tmin = t.min()
            tmax = t.max()
            if pd.notna(tmin) and pd.notna(tmax):
                dt = t.sort_values().diff().dropna()
                step_txt = "unknown"
                if not dt.empty:
                    med = dt.median()
                    mins = med.total_seconds() / 60
                    if mins >= 60 and abs(mins % 60) < 1e-6:
                        step_txt = f"{mins/60:.0f} h"
                    else:
                        step_txt = f"{mins:.0f} min"
                return (
                    "Dataset summary\n"
                    f"Site/file: {inputname_site or inputname or 'loaded'}\n"
                    f"Time span: {tmin:%Y-%m-%d} to {tmax:%Y-%m-%d}\n"
                    f"Native step: {step_txt}\n"
                    f"Records: {n_rows:,}\n"
                    f"Variables: {n_cols:,}"
                )
        return f"Dataset summary\nRecords: {n_rows:,}\nVariables: {n_cols:,}"

    def load_csv():
        nonlocal path_var
        global df, inputCSV, inputname, inputname_site

        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not path:
            return

        header_row = _find_header_row(path)
        try:
            tmp = pd.read_csv(path, skiprows=header_row)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        if "TIMESTAMP_START" not in tmp.columns:
            messagebox.showerror("Error", "Missing TIMESTAMP_START column.")
            return

        tmp["TIMESTAMP_START"] = _safe_to_datetime(tmp["TIMESTAMP_START"])
        if tmp["TIMESTAMP_START"].isna().all():
            messagebox.showerror("Error", "Could not parse TIMESTAMP_START as datetime.")
            return

        tmp["DATESTAMP_START"] = tmp["TIMESTAMP_START"].dt.floor("D")
        tmp.replace(-9999, np.nan, inplace=True)

        df = tmp
        inputCSV = path
        inputname = os.path.basename(path).split(".")[0]
        inputname_site = inputname

        path_var.set(f"Loaded: {inputname}.csv")
        summary_var.set(_dataset_summary_text(df))
        status.config(text=f"Loaded\n{inputname}.csv")

        for btn in analysis_buttons:
            btn.config(state="normal")
        for key, btn in nav_buttons.items():
            if key != "step1":
                btn.config(state="normal")

    compact_button(lf_required, "Load EC CSV", load_csv, row=2, column=0, state="normal")

    lf_optional = LabelFrame(step1, text="Optional inputs", padx=6, pady=6)
    lf_optional.grid(row=2, column=1, sticky="nwe", padx=(6, 0), pady=6)
    section_note(lf_optional, "Add site metadata or vegetation proxies only when useful for interpretation.", row=0, columnspan=2, wraplength=310)

    btn_badm = compact_button(lf_optional, "Input BADM", open_BADM_window, row=1, column=0, state="disabled")
    analysis_buttons.append(btn_badm)

    def load_ndvi():
        if not _ensure_df():
            return
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not path:
            return
        mod = pd.read_csv(path)
        mod.rename(columns={c: c.strip() for c in mod.columns}, inplace=True)
        mod["DATESTAMP_START"] = pd.to_datetime(mod.get("dt"), errors="coerce").dt.date
        mod = mod[["DATESTAMP_START", "value_mean"]]
        df["NDVI"] = pd.merge(df[["DATESTAMP_START"]], mod, on="DATESTAMP_START", how="left")["value_mean"]
        df["NDVI_intp"] = df["NDVI"].interpolate()
        summary_var.set(_dataset_summary_text(df))
        messagebox.showinfo("Done", "NDVI loaded.")

    def load_lai():
        if not _ensure_df():
            return
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not path:
            return
        lai = pd.read_csv(path)
        if "LAI" not in lai.columns:
            messagebox.showerror("Error", "No 'LAI' column in file.")
            return
        df["LAI"] = lai["LAI"].values[: len(df)]
        summary_var.set(_dataset_summary_text(df))
        messagebox.showinfo("Done", "LAI loaded.")

    btn_ndvi = compact_button(lf_optional, "NDVI (MODIS)", load_ndvi, row=2, column=0, state="disabled")
    btn_lai = compact_button(lf_optional, "LAI field", load_lai, row=2, column=1, state="disabled")
    analysis_buttons.extend([btn_ndvi, btn_lai])

    lf_summary = LabelFrame(step1, text="Loaded dataset", padx=8, pady=8)
    lf_summary.grid(row=3, column=0, columnspan=2, sticky="we", pady=(8, 0))
    tk.Label(lf_summary, textvariable=summary_var, justify="left", fg="#333333", wraplength=660).grid(row=0, column=0, sticky="w")

    # -------------------------------------------------------------------------
    # STEP 2 – QA/QC dashboard
    # -------------------------------------------------------------------------
    step2 = tk.Frame(content, bg="#ffffff")
    step_frames["step2"] = step2
    step2.grid_columnconfigure(0, weight=1)
    step2.grid_columnconfigure(1, weight=1)

    tk.Label(step2, text="Step 2 – Explore, QA/QC, and footprint context", bg="#ffffff", font=("Arial", 12, "bold")).grid(
        row=0, column=0, columnspan=2, sticky="w", pady=(0, 6)
    )
    tk.Label(step2, text="Inspect data quality before gap-filling or AI analysis.", bg="#ffffff", fg="#555555").grid(
        row=1, column=0, columnspan=2, sticky="w", pady=(0, 8)
    )

    lf2_viz = LabelFrame(step2, text="2.1 Exploration", padx=6, pady=6)
    lf2_viz.grid(row=2, column=0, columnspan=2, sticky="we", pady=6)
    lf2_viz.grid_columnconfigure(0, weight=1)
    lf2_viz.grid_columnconfigure(1, weight=1)
    lf2_viz.grid_columnconfigure(2, weight=1)
    section_note(lf2_viz, "Check seasonality, gaps, outliers, driver ranges, and flux--driver relationships.", row=0, columnspan=3, wraplength=620)

    viz_buttons = [
        ("Time series", calc_plot_time_series),
        ("Daily averages", calc_plot_daily_avg),
        ("Wind rose", calc_plot_wind_rose),
        ("Density & scatter", calc_plot_density_and_scatter),
        ("Correlations", calc_plot_correlations),
        ("Budgets", calc_plot_budgets),
    ]
    for i, (text, func) in enumerate(viz_buttons):
        b = Button(lf2_viz, text=text, state="disabled", command=lambda f=func: _call(f, df, inputname_site))
        b.grid(row=(i // 3) + 1, column=i % 3, padx=6, pady=4, sticky="w")
        analysis_buttons.append(b)

    lf2_qaqc = LabelFrame(step2, text="2.2 QA/QC and completeness", padx=6, pady=6)
    lf2_qaqc.grid(row=3, column=0, sticky="nwe", padx=(0, 6), pady=6)
    section_note(lf2_qaqc, "Define analysis subset, apply flags, and document usable coverage.", row=0, wraplength=310)

    qaqc_buttons = [
        ("Guided Standard QA/QC", calc_standard_QAQC),
        ("Data availability", calc_data_availability),
        ("AmeriFlux BASE QA/QC", calc_data_AMF_BASE_QAQC),
    ]
    for i, (text, func) in enumerate(qaqc_buttons):
        b = Button(lf2_qaqc, text=text, state="disabled", command=lambda f=func: _call(f, df, inputname_site))
        b.grid(row=i + 1, column=0, padx=4, pady=4, sticky="w")
        analysis_buttons.append(b)

    lf2_fp = LabelFrame(step2, text="2.3 Footprint context", padx=6, pady=6)
    lf2_fp.grid(row=3, column=1, sticky="nwe", padx=(6, 0), pady=6)
    section_note(lf2_fp, "Use when source-area representativeness or fetch may affect interpretation.", row=0, wraplength=310)

    b_dir_rose = Button(
        lf2_fp,
        text="Directional contribution rose",
        state="disabled",
        command=lambda: _call(
            calc_plot_directional_contribution_rose,
            df,
            inputname_site,
            getattr(UpdatedValues, "lat", None),
            getattr(UpdatedValues, "lon", None),
        ),
    )
    b_dir_rose.grid(row=1, column=0, padx=4, pady=4, sticky="w")
    analysis_buttons.append(b_dir_rose)

    b_ffp = Button(lf2_fp, text="FFP calculations", state="disabled", command=lambda: _call(calc_FFP_window, df, UpdatedValues, inputname_site))
    b_ffp.grid(row=2, column=0, padx=4, pady=4, sticky="w")
    analysis_buttons.append(b_ffp)

    b_fetch = Button(lf2_fp, text="Fetch rose", state="disabled", command=lambda: _call(calc_plot_fetch_rose, df, inputname_site))
    b_fetch.grid(row=3, column=0, padx=4, pady=4, sticky="w")
    analysis_buttons.append(b_fetch)

    # -------------------------------------------------------------------------
    # STEP 3 – Gap filling
    # -------------------------------------------------------------------------
    step3 = tk.Frame(content, bg="#ffffff")
    step_frames["step3"] = step3
    step3.grid_columnconfigure(0, weight=1)
    step3.grid_columnconfigure(1, weight=1)

    tk.Label(step3, text="Step 3 – Gap-filling and flux completion", bg="#ffffff", font=("Arial", 12, "bold")).grid(
        row=0, column=0, columnspan=2, sticky="w", pady=(0, 6)
    )
    tk.Label(step3, text="Use gap-filling only when complete series are needed for budgets, ML, or IT diagnostics.", bg="#ffffff", fg="#555555").grid(
        row=1, column=0, columnspan=2, sticky="w", pady=(0, 8)
    )

    lf3_gap = LabelFrame(step3, text="3.1 Gap-filling methods", padx=6, pady=6)
    lf3_gap.grid(row=2, column=0, sticky="nwe", padx=(0, 6), pady=6)
    section_note(lf3_gap, "Original and gap-filled variables remain separate for provenance.", row=0, wraplength=310)

    def set_df(new):
        global df
        df = new
        summary_var.set(_dataset_summary_text(df))
        status.config(text="Dataset updated")

    b_oneflux = Button(lf3_gap, text="ONEFlux gap-fill + partition", state="disabled", command=lambda: open_oneflux_window(root, df, inputname_site, inputCSV, pb, set_df))
    b_oneflux.grid(row=1, column=0, padx=4, pady=4, sticky="w")
    analysis_buttons.append(b_oneflux)

    b_n2o = Button(lf3_gap, text="Gap-fill N₂O", state="disabled", command=lambda: calc_gapfill_N2O(parent=root, df_in=df, inputname_site=inputname_site, inputCSV=inputCSV, shared_progressbar=pb, on_update_df=set_df))
    b_n2o.grid(row=2, column=0, padx=4, pady=4, sticky="w")
    analysis_buttons.append(b_n2o)

    b_ch4 = Button(lf3_gap, text="Gap-fill CH₄", state="disabled", command=lambda: calc_gapfill_CH4(parent=root, df_in=df, inputname_site=inputname_site, inputCSV=inputCSV, shared_progressbar=pb, on_update_df=set_df))
    b_ch4.grid(row=3, column=0, padx=4, pady=4, sticky="w")
    analysis_buttons.append(b_ch4)

    lf3_note = LabelFrame(step3, text="Recommended use", padx=8, pady=8)
    lf3_note.grid(row=2, column=1, sticky="nwe", padx=(6, 0), pady=6)
    tk.Label(
        lf3_note,
        text="Use after QA/QC when:\n\n• cumulative budgets require complete records\n• ML needs a continuous predictor or target\n• IT diagnostics require aligned samples\n\nKeep raw and gap-filled variables separate.",
        justify="left",
        fg="#444444",
        wraplength=310,
    ).grid(row=0, column=0, sticky="w")

    # -------------------------------------------------------------------------
    # STEP 4 – ML to IT
    # -------------------------------------------------------------------------
    step4 = tk.Frame(content, bg="#ffffff")
    step_frames["step4"] = step4
    step4.grid_columnconfigure(0, weight=1)
    step4.grid_columnconfigure(1, weight=1)

    tk.Label(step4, text="Step 4 – Integrated ML–IT analysis", bg="#ffffff", font=("Arial", 12, "bold")).grid(
        row=0, column=0, columnspan=2, sticky="w", pady=(0, 6)
    )
    tk.Label(step4, text="Run ML first to predict the target flux; then use IT to evaluate information structure and model fidelity.", bg="#ffffff", fg="#555555").grid(
        row=1, column=0, columnspan=2, sticky="w", pady=(0, 8)
    )

    lf4_flow = LabelFrame(step4, text="4.1 Workflow", padx=8, pady=8)
    lf4_flow.grid(row=2, column=0, sticky="nwe", padx=(0, 6), pady=6)
    tk.Label(
        lf4_flow,
        text="① Predict target flux\n\n↓\n\n② Export ML-to-IT bridge\n\n↓\n\n③ Diagnose nonlinear dependence, lags, redundancy, synergy, and information fidelity",
        justify="left",
        fg="#333333",
        wraplength=310,
    ).grid(row=0, column=0, sticky="w")

    lf4_tools = LabelFrame(step4, text="4.2 Toolboxes", padx=8, pady=8)
    lf4_tools.grid(row=2, column=1, sticky="nwe", padx=(6, 0), pady=6)
    section_note(lf4_tools, "ML quantifies out-of-sample predictability. IT tests whether predictions preserve observed information structure.", row=0, wraplength=310)

    b_ml = Button(lf4_tools, text="1. Machine Learning Toolbox", state="disabled", command=lambda: _call(open_machine_learning_toolbox, df, inputname_site))
    b_ml.grid(row=1, column=0, padx=4, pady=6, sticky="we")
    analysis_buttons.append(b_ml)

    b_it = Button(lf4_tools, text="2. Information Theory Toolbox", state="disabled", command=lambda: _call(open_information_theory_toolbox, df, inputname_site))
    b_it.grid(row=2, column=0, padx=4, pady=6, sticky="we")
    analysis_buttons.append(b_it)

    show("step1")
    root.mainloop()


if __name__ == "__main__":
    MeaningFlux_main_window()
