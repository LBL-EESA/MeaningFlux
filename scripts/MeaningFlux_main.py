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
ROOT = Path(__file__).resolve().parents[1]   # .../meaningflux_code
SRC = ROOT / "src"
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


# ===================== GUI =====================
def MeaningFlux_main_window():
    global df, inputname, inputname_site, inputCSV

    root = tk.Tk()
    root.title("MeaningFlux")
    _center(root, 500, 400)
    root.configure(bg="#f4f4f4")

    # -------------------------------------------------------------------------
    # Menu bar (Help → About)
    # -------------------------------------------------------------------------
    menubar = tk.Menu(root)

    help_menu = tk.Menu(menubar, tearoff=0)
    help_menu.add_command(label="About MeaningFlux", command=show_about)

    menubar.add_cascade(label="Help", menu=help_menu)
    root.config(menu=menubar)

    # -------------------------------------------------------------------------
    # Header – centered logo, no visible wait bar
    # -------------------------------------------------------------------------
    header = tk.Frame(root, bg="#f4f4f4")
    header.grid(row=0, column=0, columnspan=2, sticky="we", padx=10, pady=(6, 2))
    header.grid_columnconfigure(0, weight=1)

    # Load logo without distortion; only shrink if too tall
    try:
        img = Image.open(ROOT / "docs" / "MeaningFlux_logo.png")
        orig_w, orig_h = img.size
        target_h = 120
        if orig_h > target_h:
            scale = target_h / orig_h
            new_w = int(orig_w * scale)
            img = img.resize((new_w, target_h), Image.LANCZOS)
        logo = ImageTk.PhotoImage(img)
        logo_label = tk.Label(header, image=logo, bg="#f4f4f4")
        logo_label.image = logo
    except Exception:
        logo_label = tk.Label(
            header,
            text="MeaningFlux",
            font=("Arial", 20, "bold"),
            bg="#f4f4f4"
        )

    # Center the logo
    logo_label.grid(row=0, column=0, pady=4)

    # Status (centered under logo)
    status = tk.Label(header, text="", bg="#f4f4f4", fg="#666")
    status.grid(row=1, column=0, sticky="we", pady=(2, 4))

    # Subtle Help / About button (top-right corner)
    help_btn = tk.Button(
        header,
        text="ⓘ",
        command=show_about,
        relief="flat",
        bg="#f4f4f4",
        fg="#888",
        font=("Arial", 12),
        cursor="hand2"
    )
    help_btn.grid(row=0, column=1, sticky="ne", padx=4, pady=4)

    header.grid_columnconfigure(1, weight=0)

    # (Optional) if your other modules need a Progressbar object, create one but do NOT grid it
    pb = Progressbar(header, mode="indeterminate", length=200)

    # -------------------------------------------------------------------------
    # Main layout: sidebar + content panel
    # -------------------------------------------------------------------------
    main = tk.Frame(root, bg="#f4f4f4")
    main.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=(0, 6))

    root.grid_rowconfigure(1, weight=1)
    root.grid_columnconfigure(0, weight=1)

    sidebar = tk.Frame(main, bg="#e9e9e9", width=150)
    sidebar.grid(row=0, column=0, sticky="nsw")
    sidebar.grid_propagate(False)

    content = tk.Frame(main, bg="#ffffff", relief="solid", bd=1)
    content.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
    main.grid_columnconfigure(1, weight=1)
    main.grid_rowconfigure(0, weight=1)

    # -------------------------------------------------------------------------
    # Step frames + navigation
    # -------------------------------------------------------------------------
    step_frames = {}
    nav_buttons = {}
    analysis_buttons = []  # all buttons that should be disabled until EC CSV is loaded

    def show(step):
        for k, f in step_frames.items():
            f.grid_forget()
        frame = step_frames[step]
        frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        for k, b in nav_buttons.items():
            b.config(bg="#e9e9e9")
        nav_buttons[step].config(bg="#d0d0d0")

    steps = [
        ("step1", "① Input data"),
        ("step2", "② Visualization"),
        ("step3", "③ QA/QC"),
        ("step4", "④ Gap-filling"),
        ("step5", "⑤ Flux Footprint"),
        ("step6", "⑥ AI / IT"),
    ]

    # Sidebar buttons – only Step 1 enabled initially
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
        b.grid(row=i, column=0, sticky="we", padx=5, pady=2)
        nav_buttons[key] = b

    sidebar.grid_rowconfigure(len(steps) + 1, weight=1)

    # -------------------------------------------------------------------------
    # STEP 1 – Load EC, BADM, NDVI/LAI
    # -------------------------------------------------------------------------
    step1 = tk.Frame(content, bg="#ffffff")
    step_frames["step1"] = step1

    tk.Label(
        step1,
        text="Step 1 – Load EC data and metadata",
        bg="#ffffff",
        font=("Arial", 12, "bold"),
    ).grid(row=0, column=0, sticky="w", pady=(0, 6))

    lf_ec = LabelFrame(step1, text="1.1 EC + BADM", padx=6, pady=6)
    lf_ec.grid(row=1, column=0, sticky="we")

    path_var = tk.StringVar(value="No EC file loaded")
    tk.Label(lf_ec, textvariable=path_var, fg="#444").grid(
        row=0, column=0, columnspan=3, sticky="w"
    )

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
            messagebox.showerror(
                "Error", "Could not parse TIMESTAMP_START as datetime."
            )
            return

        tmp["DATESTAMP_START"] = tmp["TIMESTAMP_START"].dt.floor("D")
        tmp.replace(-9999, np.nan, inplace=True)

        df = tmp
        inputCSV = path
        inputname = os.path.basename(path).split(".")[0]
        inputname_site = inputname

        path_var.set(f"Loaded: {inputname}.csv")
        status.config(text="EC data loaded. All tools unlocked.")
        status.config(text=f"Loaded: {inputname}.csv")

        # Enable all analysis buttons and other steps
        for btn in analysis_buttons:
            btn.config(state="normal")
        for key, btn in nav_buttons.items():
            if key != "step1":
                btn.config(state="normal")

    # Only enabled button at startup
    btn_load_ec = Button(lf_ec, text="Load EC CSV", command=load_csv)
    btn_load_ec.grid(row=1, column=0, padx=4, pady=4)

    btn_badm = Button(lf_ec, text="Input BADM", command=open_BADM_window, state="disabled")
    btn_badm.grid(row=1, column=1, padx=4, pady=4)
    analysis_buttons.append(btn_badm)

    # --- NDVI / LAI ---
    lf_lai = LabelFrame(step1, text="1.2 NDVI / LAI (optional)", padx=6, pady=6)
    lf_lai.grid(row=2, column=0, sticky="we", pady=6)

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
        df["NDVI"] = pd.merge(
            df[["DATESTAMP_START"]],
            mod,
            on="DATESTAMP_START",
            how="left",
        )["value_mean"]
        df["NDVI_intp"] = df["NDVI"].interpolate()
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
        messagebox.showinfo("Done", "LAI loaded.")

    btn_ndvi = Button(lf_lai, text="NDVI (MODIS)", command=load_ndvi, state="disabled")
    btn_ndvi.grid(row=0, column=0, padx=4, pady=4)
    analysis_buttons.append(btn_ndvi)

    btn_lai = Button(lf_lai, text="LAI field", command=load_lai, state="disabled")
    btn_lai.grid(row=0, column=1, padx=4, pady=4)
    analysis_buttons.append(btn_lai)


    # -------------------------------------------------------------------------
    # STEP 2 – Visualization (two columns)
    # -------------------------------------------------------------------------
    step2 = tk.Frame(content, bg="#ffffff")
    step_frames["step2"] = step2

    tk.Label(
        step2,
        text="Step 2 – Visualization tools",
        bg="#ffffff",
        font=("Arial", 12, "bold"),
    ).grid(row=0, column=0, sticky="w")

    lf2 = LabelFrame(step2, padx=6, pady=6)
    lf2.grid(row=1, column=0, sticky="nw", pady=6)

    viz_buttons = [
        ("Time series", calc_plot_time_series),
        ("Daily Avg", calc_plot_daily_avg),
        ("Wind Rose", calc_plot_wind_rose),
        ("Density & Scatter", calc_plot_density_and_scatter),
        ("Correlations", calc_plot_correlations),
        ("Budgets", calc_plot_budgets),
    ]

    for i, (text, func) in enumerate(viz_buttons):
        r = i // 2
        c = i % 2
        b = Button(
            lf2,
            text=text,
            state="disabled",
            command=lambda f=func: _call(f, df, inputname_site),
        )
        b.grid(row=r, column=c, padx=6, pady=4, sticky="w")
        analysis_buttons.append(b)

    # -------------------------------------------------------------------------
    # STEP 3 – QA/QC
    # -------------------------------------------------------------------------
    step3 = tk.Frame(content, bg="#ffffff")
    step_frames["step3"] = step3
    
    tk.Label(
        step3,
        text="Step 3 – QA/QC & diagnostics",
        bg="#ffffff",
        font=("Arial", 12, "bold"),
    ).grid(row=0, column=0, sticky="w")
    
    lf3 = LabelFrame(step3, padx=6, pady=6)
    lf3.grid(row=1, column=0, sticky="nw", pady=6)
    
    # Row 0
    b_standard_qaqc = Button(
        lf3,
        text="Standard QA/QC",
        state="disabled",
        command=lambda: _call(calc_standard_QAQC, df, inputname_site),
    )
    b_standard_qaqc.grid(row=0, column=0, padx=4, pady=4, sticky="w")
    analysis_buttons.append(b_standard_qaqc)
    
    # Row 1
    b_data_av = Button(
        lf3,
        text="Data availability",
        state="disabled",
        command=lambda: _call(calc_data_availability, df, inputname_site),
    )
    b_data_av.grid(row=1, column=0, padx=4, pady=4, sticky="w")
    analysis_buttons.append(b_data_av)
    
    # Row 2
    b_dir_rose = Button(
        lf3,
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
    b_dir_rose.grid(row=2, column=0, padx=4, pady=4, sticky="w")
    analysis_buttons.append(b_dir_rose)
    
    # Row 3 (LAST)
    b_qaqc = Button(
        lf3,
        text="AMERIFLUX BASE QAQC",
        state="disabled",
        command=lambda: _call(calc_data_AMF_BASE_QAQC, df, inputname_site),
    )
    b_qaqc.grid(row=3, column=0, padx=4, pady=4, sticky="w")
    analysis_buttons.append(b_qaqc)


    # -------------------------------------------------------------------------
    # STEP 4 – Gap-filling
    # -------------------------------------------------------------------------
    step4 = tk.Frame(content, bg="#ffffff")
    step_frames["step4"] = step4

    tk.Label(
        step4,
        text="Step 4 – Gap-filling methods",
        bg="#ffffff",
        font=("Arial", 12, "bold"),
    ).grid(row=0, column=0, sticky="w")

    lf4 = LabelFrame(step4, padx=6, pady=6)
    lf4.grid(row=1, column=0, sticky="nw", pady=6)

    def set_df(new):
        global df
        df = new
        status.config(text="Dataset updated by gap-filling.")

    # If your open_oneflux_window still expects a progressbar, pb is a dummy but valid object
    b_oneflux = Button(
        lf4,
        text="ONEFlux gap-fill + partition",
        state="disabled",
        command=lambda: open_oneflux_window(root, df, inputname_site, inputCSV, pb, set_df),
    )
    b_oneflux.grid(row=0, column=0, padx=4, pady=4, sticky="w")
    analysis_buttons.append(b_oneflux)

    b_n2o = Button(
        lf4,
        text="Gap-fill N₂O",
        state="disabled",
        command=lambda: calc_gapfill_N2O(
            parent=root,
            df_in=df,
            inputname_site=inputname_site,
            inputCSV=inputCSV,
            shared_progressbar=pb,
            on_update_df=set_df,
        ),
    )
    b_n2o.grid(row=1, column=0, padx=4, pady=4, sticky="w")
    analysis_buttons.append(b_n2o)

    b_ch4 = Button(
        lf4,
        text="Gap-fill CH₄",
        state="disabled",
        command=lambda: calc_gapfill_CH4(
            parent=root,
            df_in=df,
            inputname_site=inputname_site,
            inputCSV=inputCSV,
            shared_progressbar=pb,
            on_update_df=set_df,
        ),
    )
    b_ch4.grid(row=2, column=0, padx=4, pady=4, sticky="w")
    analysis_buttons.append(b_ch4)

    # -------------------------------------------------------------------------
    # STEP 5 – Footprint
    # -------------------------------------------------------------------------
    step5 = tk.Frame(content, bg="#ffffff")
    step_frames["step5"] = step5

    tk.Label(
        step5,
        text="Step 5 – Flux footprint",
        bg="#ffffff",
        font=("Arial", 12, "bold"),
    ).grid(row=0, column=0, sticky="w")

    lf5 = LabelFrame(step5, padx=6, pady=6)
    lf5.grid(row=1, column=0, sticky="nw", pady=6)

    b_ffp = Button(
        lf5,
        text="FFP calculations",
        state="disabled",
        command=lambda: _call(calc_FFP_window, df, UpdatedValues, inputname_site),
    )
    b_ffp.grid(row=0, column=0, padx=4, pady=4, sticky="w")
    analysis_buttons.append(b_ffp)

    b_fetch = Button(
        lf5,
        text="Fetch Rose",
        state="disabled",
        command=lambda: _call(calc_plot_fetch_rose, df, inputname_site),
    )
    b_fetch.grid(row=1, column=0, padx=4, pady=4, sticky="w")
    analysis_buttons.append(b_fetch)

    # -------------------------------------------------------------------------
    # STEP 6 – AI / IT
    # -------------------------------------------------------------------------
    step6 = tk.Frame(content, bg="#ffffff")
    step_frames["step6"] = step6

    tk.Label(
        step6,
        text="Step 6 – AI & Information Theory",
        bg="#ffffff",
        font=("Arial", 12, "bold"),
    ).grid(row=0, column=0, sticky="w")

    lf6 = LabelFrame(step6, padx=6, pady=6)
    lf6.grid(row=1, column=0, sticky="nw", pady=6)

    b_it = Button(
        lf6,
        text="Information Theory Toolbox",
        state="disabled",
        command=lambda: _call(open_information_theory_toolbox, df, inputname_site),
    )
    b_it.grid(row=0, column=0, padx=4, pady=4, sticky="w")
    analysis_buttons.append(b_it)

    b_ml = Button(
        lf6,
        text="Machine Learning Toolbox",
        state="disabled",
        command=lambda: _call(open_machine_learning_toolbox, df, inputname_site),
    )
    b_ml.grid(row=1, column=0, padx=4, pady=4, sticky="w")
    analysis_buttons.append(b_ml)

    # Start on Step 1
    show("step1")

    root.mainloop()


if __name__ == "__main__":
    MeaningFlux_main_window()
