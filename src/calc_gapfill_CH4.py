# -*- coding: utf-8 -*-
"""
MeaningFlux – CH₄ Gap-Fill (GUI Window)
---------------------------------------
Author: Leila C. Hernandez Rodriguez (LBNL) – with ChatGPT assistance

Save this file as **calc_gapfill_CH4.py** next to your main GUI script.

What this provides
- A toplevel window titled "MeaningFlux: CH₄ Gap-Fill" that mirrors your N₂O/ONEFlux UX.
- Runs multiple ML methods used for CH₄ gap-filling at FLUXNET-CH4 wetlands.
- Creates per-method filled columns: <TARGET>_F_<Method> and an ensemble <TARGET>_F.
- Keeps all original columns; converts NaN→-9999 only when exporting CSV.
- Uses the DataFrame you pass in (no separate file loading UI).

Citations (displayed in the window)
- Irvin, J., Zhou, S., McNicol, G., et al. (2021) Agricultural and Forest Meteorology 308–309, 108528.
  DOI: 10.1016/j.agrformet.2021.108528
- Stanford ML Group: methane-gapfill-ml (https://github.com/stanfordmlgroup/methane-gapfill-ml)

Notes
- -9999 values are mapped to NaN on load, and scikit-learn pipelines impute remaining NaNs.
- XGBoost is optional; if not installed, that method is automatically skipped.
- All tkinter mutations from the worker happen via `after(...)` to stay thread-safe.
"""

from __future__ import annotations

import os
import re
import threading
import tkinter as tk
from tkinter import messagebox, filedialog
from tkinter import LabelFrame, Button
from tkinter.ttk import Combobox, Progressbar

import numpy as np
import pandas as pd

# --- sklearn & pipelines ---
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Optional
try:
    from xgboost import XGBRegressor  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False


# ---------------------------- Public entry point ---------------------------- #

def calc_gapfill_CH4(
    parent: tk.Misc,
    df_in: pd.DataFrame,
    inputname_site: str | None = None,
    inputCSV: str | None = None,
    shared_progressbar=None,
    on_update_df=None,
):
    """
    Open the CH₄ gap-fill window and return immediately.
    """
    win = MethaneGapFillWindow(parent=parent, df_in=df_in, inputCSV=inputCSV,
                               shared_progressbar=shared_progressbar,
                               on_update_df=on_update_df)
    return win


# ---------------------------- Window implementation ---------------------------- #

class MethaneGapFillWindow(tk.Toplevel):
    def __init__(self, parent, df_in: pd.DataFrame, inputCSV: str | None,
                 shared_progressbar=None, on_update_df=None):
        super().__init__(parent)
        self.title("MeaningFlux: CH₄ Gap-Fill")
        self.geometry("1120x840")
        self.minsize(750, 900)

        # State
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._running = False

        # Data
        self.timestamp_col = "TIMESTAMP_START"
        self.inputCSV = inputCSV
        self.shared_pb = shared_progressbar
        self.on_update_df = on_update_df

        # Copy-in DF, standardize NaNs
        df = df_in.copy()
        df.replace(-9999, np.nan, inplace=True)
        # If TIMESTAMP_START exists as datetime or string, preserve it; keep a helper datetime for any internal need
        if self.timestamp_col in df.columns:
            if not np.issubdtype(df[self.timestamp_col].dtype, np.datetime64):
                # try to parse common AmeriFlux format; fall back to auto
                parsed = pd.to_datetime(df[self.timestamp_col].astype(str), format="%Y%m%d%H%M", errors="coerce")
                if parsed.notna().sum() < (0.5 * len(parsed)):
                    parsed = pd.to_datetime(df[self.timestamp_col], errors="coerce")
                df["__TS_DT__"] = parsed
            else:
                df["__TS_DT__"] = df[self.timestamp_col]
        self.df: pd.DataFrame = df

        # UI variables
        self.var_target = tk.StringVar()
        self.var_cv_folds = tk.IntVar(value=5)
        self.var_k = tk.IntVar(value=5)
        self.var_pls = tk.IntVar(value=3)

        # Build UI
        self._build_header_and_howto()
        self._build_vars_panel()
        self._build_methods_panel()
        self._build_controls()
        self._build_console()

        # Populate columns from df (with preselection of plausible drivers)
        self._populate_columns()

        self._log("Ready. DataFrame received from main GUI. (-9999 already mapped to NaN.)")

        # Safer close while threads may run
        self.protocol("WM_DELETE_WINDOW", self._on_close_request)

    # ------------------------------ UI builders ------------------------------ #

    def _build_header_and_howto(self):
        # Header
        tk.Label(self, text="MeaningFlux • CH₄ Gap-Fill",
                 font=("TkDefaultFont", 13, "bold")).pack(anchor="w", padx=12, pady=(10, 4))

        citation = (
            "Irvin, J., Zhou, S., McNicol, G., et al. (2021). Gap-filling eddy covariance methane fluxes: "
            "Comparison of machine learning model predictions and uncertainties at FLUXNET-CH4 wetlands. "
            "Agricultural and Forest Meteorology, 308–309, 108528. doi:10.1016/j.agrformet.2021.108528\n"
            "Code: Stanford ML Group – methane-gapfill-ml (https://github.com/stanfordmlgroup/methane-gapfill-ml)"
        )
        tk.Label(self, text=citation, wraplength=1050, fg="#444", justify="left").pack(anchor="w", padx=12, pady=(0, 8))

        how = tk.LabelFrame(self, text="How this works")
        how.pack(fill="x", padx=12, pady=(0, 8))
        tk.Label(
            how,
            justify="left", wraplength=1050, fg="#333",
            text=(
                "• Choose a CH₄ flux **target** (e.g., FCH4…) and one or more **driver** variables (preselected heuristically).\n"
                "• Each method does K-fold CV on observed rows, then fits on all observed data and predicts the gaps.\n"
                "• Outputs: per-method `<TARGET>_F_<Method>` and an ensemble `<TARGET>_F` (mean across successful methods at missing rows).\n"
                "• Methods: LinearRegression, PLSRegression, KNN, RandomForest, ExtraTrees, GradientBoosting, SVR, MLP, and optional XGBoost.\n"
                "• Pipelines include imputation (median) and scaling where appropriate; export converts NaN→-9999."
            )
        ).pack(anchor="w", padx=8, pady=6)

    def _build_vars_panel(self):
        lf = LabelFrame(self, text="Variables")
        lf.pack(fill="x", padx=12, pady=(0, 8))

        tk.Label(lf, text="Target (CH₄ flux variable)").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        self.cmb_target = Combobox(lf, textvariable=self.var_target, width=46, state="readonly")
        self.cmb_target.grid(row=0, column=1, sticky="w", padx=8, pady=6)

        tk.Label(lf, text="Driver variables (TIMESTAMP* hidden; plausible ones preselected)").grid(
            row=1, column=0, sticky="nw", padx=8, pady=6
        )
        self.lst_drivers = tk.Listbox(lf, selectmode=tk.MULTIPLE, width=56, height=10, exportselection=False)
        self.lst_drivers.grid(row=1, column=1, sticky="w", padx=8, pady=6)

        side = tk.Frame(lf)
        side.grid(row=1, column=2, sticky="nw", padx=8, pady=6)
        Button(side, text="Select all", command=lambda: self.lst_drivers.select_set(0, tk.END)).pack(fill="x", pady=(0, 6))
        Button(side, text="Clear", command=lambda: self.lst_drivers.selection_clear(0, tk.END)).pack(fill="x")

        self.lbl_info = tk.Label(lf, text="Columns populate from the DataFrame you passed in.")
        self.lbl_info.grid(row=2, column=0, columnspan=3, sticky="w", padx=8, pady=(6, 8))

    def _build_methods_panel(self):
        lf = LabelFrame(self, text="Methods & Parameters")
        lf.pack(fill="x", padx=12, pady=(0, 8))

        tk.Label(lf, text="Methods (multi-select)").grid(row=0, column=0, sticky="nw", padx=8, pady=6)
        self.lst_methods = tk.Listbox(lf, selectmode=tk.MULTIPLE, width=30, height=9, exportselection=False)
        self.lst_methods.grid(row=0, column=1, sticky="w", padx=8, pady=6)

        methods = [
            "LinearRegression",
            "PLSRegression",
            "KNN",
            "RandomForest",
            "ExtraTrees",
            "GradientBoosting",
            "SVR",
            "MLP",
        ]
        if HAS_XGB:
            methods.append("XGBoost")
        for m in methods:
            self.lst_methods.insert(tk.END, m)
        self.lst_methods.select_set(0, tk.END)

        prm = tk.Frame(lf)
        prm.grid(row=0, column=2, sticky="nw", padx=8, pady=6)

        tk.Label(prm, text="CV folds").grid(row=0, column=0, sticky="w")
        tk.Entry(prm, textvariable=self.var_cv_folds, width=8).grid(row=0, column=1, sticky="w", padx=(6, 0))

        tk.Label(prm, text="k (for KNN)").grid(row=1, column=0, sticky="w", pady=(8, 0))
        tk.Entry(prm, textvariable=self.var_k, width=8).grid(row=1, column=1, sticky="w", padx=(6, 0), pady=(8, 0))

        tk.Label(prm, text="PLS components").grid(row=2, column=0, sticky="w", pady=(8, 0))
        tk.Entry(prm, textvariable=self.var_pls, width=8).grid(row=2, column=1, sticky="w", padx=(6, 0), pady=(8, 0))

    def _build_controls(self):
        lf = LabelFrame(self, text="Run & Export")
        lf.pack(fill="x", padx=12, pady=(0, 8))

        self.prog = Progressbar(lf, orient="horizontal", mode="determinate", length=600, maximum=100)
        self.prog.grid(row=0, column=0, columnspan=6, sticky="we", padx=8, pady=(10, 6))
        lf.columnconfigure(0, weight=1)

        Button(lf, text="Run", command=self._on_run).grid(row=1, column=0, sticky="w", padx=8, pady=(4, 10))
        Button(lf, text="STOP", command=self._on_stop).grid(row=1, column=1, sticky="w", padx=8, pady=(4, 10))

        Button(lf, text="Save CSV…", command=self._on_save_csv).grid(row=1, column=5, sticky="e", padx=8, pady=(4, 10))

        self.var_status = tk.StringVar(value="Status: Idle")
        tk.Label(lf, textvariable=self.var_status).grid(row=1, column=4, sticky="e", padx=8)

    def _build_console(self):
        lf = LabelFrame(self, text="Log / Console")
        lf.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.txt = tk.Text(lf, height=12)
        self.txt.pack(fill="both", expand=True)

    # ------------------------------ Helpers ------------------------------ #

    def _log(self, msg: str):
        # Schedule on UI thread
        if self.txt and self.txt.winfo_exists():
            self.txt.after(0, lambda: (self.txt.insert(tk.END, msg + "\n"),
                                       self.txt.see(tk.END)))

    def _set_status(self, text: str):
        if self.txt and self.txt.winfo_exists():
            self.after(0, lambda: self.var_status.set(text))

    def _prog_set(self, value: float, maximum: float | None = None):
        if self.prog and self.prog.winfo_exists():
            def _upd():
                if maximum is not None:
                    self.prog.configure(maximum=maximum)
                self.prog['value'] = value
            self.prog.after(0, _upd)

        if self.shared_pb:
            try:
                self.shared_pb.after(0, lambda: self.shared_pb.step(1))
            except Exception:
                pass

    @staticmethod
    def _is_timestamp_like(name: str) -> bool:
        """Hide only columns that START WITH 'TIMESTAMP' (case-insensitive)."""
        return str(name).strip().lower().startswith("timestamp")

    @staticmethod
    def _is_plausible_ch4_driver_name(name: str) -> bool:
        """
        Heuristics for likely CH4 drivers (case-insensitive):
        water table, soil moisture/VWC/WFPS, TA/TS, VPD/RH, precip,
        radiation/energy, turbulence/wind (u*, WS), and GPP/NEE.
        """
        n = name.lower()
        if MethaneGapFillWindow._is_timestamp_like(name):
            return False

        # Water table / level
        if any(tok in n for tok in ["wtd", "water_table", "waterlevel", "wt_depth", "watertable", "wt_"]):
            return True

        # Soil moisture / water content
        if any(tok in n for tok in ["swc", "vwc", "theta", "wfps", "soil_moist", "soilmoist", "soilw", "sws"]):
            return True

        # Temperature (air/soil)
        if re.match(r"^(ta|tair|air[_]?temp|airtemp)\b", n):
            return True
        if re.match(r"^(ts|soil[_]?temp|soiltemp)\b", n):
            return True

        # Humidity / VPD
        if "vpd" in n or re.match(r"^rh(\b|_)", n) or "relative_humidity" in n:
            return True

        # Precipitation
        if any(tok in n for tok in ["precip", "rain", "ppt", "prcp"]):
            return True

        # Radiation / energy
        if any(tok in n for tok in ["sw_in", "ppfd", "par", "rnet", "shortwave", "netrad", "rad"]):
            return True
        if re.match(r"^rg(\b|_)", n) or "soil_heat" in n or re.match(r"^g(_|$)", n):
            return True

        # Turbulence / wind
        if "ustar" in n or "u*" in n or n.startswith("ws") or "wind_speed" in n or "windspd" in n:
            return True

        # Carbon flux proxies
        if re.match(r"^gpp(\b|_)", n) or re.match(r"^nee(\b|_)", n):
            return True

        return False

    def _populate_driver_list_only(self):
        """Populate the drivers list (hide TIMESTAMP*; preselect plausible drivers)."""
        self.lst_drivers.delete(0, tk.END)
        target = self.var_target.get().strip()
        candidates = [
            c for c in self.df.columns
            if c not in (self.timestamp_col, "__TS_DT__")
            and not self._is_timestamp_like(c)
            and c != target
        ]
        for c in candidates:
            self.lst_drivers.insert(tk.END, c)
        # auto-preselect likely CH4 drivers
        for i, name in enumerate(candidates):
            if self._is_plausible_ch4_driver_name(name):
                self.lst_drivers.select_set(i)

    def _populate_columns(self):
        """Fill the target combobox (preferring CH4-looking names) and populate drivers."""
        cols = [c for c in self.df.columns if c not in (self.timestamp_col, "__TS_DT__") and not self._is_timestamp_like(c)]
        target_candidates = sorted(cols, key=lambda c: (("CH4" not in c.upper()), c.lower()))
        self.cmb_target["values"] = target_candidates
        if target_candidates:
            current = self.var_target.get().strip()
            self.var_target.set(current if current in target_candidates else target_candidates[0])

        self._populate_driver_list_only()
        self.cmb_target.bind("<<ComboboxSelected>>", lambda e: self._populate_driver_list_only())

    # ------------------------------ Events ------------------------------ #

    def _on_run(self):
        if self._running:
            return
        target = self.var_target.get().strip()
        if not target:
            messagebox.showwarning("Run", "Pick a target variable.")
            return
        sel = self.lst_drivers.curselection()
        drivers = [self.lst_drivers.get(i) for i in sel if self.lst_drivers.get(i) != target]
        if not drivers:
            messagebox.showwarning("Run", "Pick at least one driver variable.")
            return
        msel = self.lst_methods.curselection()
        methods = [self.lst_methods.get(i) for i in msel]
        if not methods:
            messagebox.showwarning("Run", "Select at least one method.")
            return

        self._stop_event.clear()  # <-- fixed: call the method
        self._running = True
        self._set_status("Status: Running…")
        self._prog_set(0, maximum=max(1, len(methods) + 1))

        # Disable window close while running to avoid widget-destroy races
        self._old_close = self.protocol("WM_DELETE_WINDOW", self._on_close_request)

        # Launch worker
        args = (target, drivers, methods, int(max(2, self.var_cv_folds.get())))
        self._worker_thread = threading.Thread(target=self._worker, args=args, daemon=True)
        self._worker_thread.start()

    def _on_stop(self):
        if self._running:
            self._stop_event.set()
            self._set_status("Status: Stopping (will halt after current step)…")
            self._log("STOP pressed.")

    def _on_close_request(self):
        if self._running:
            if not messagebox.askyesno("Close", "A job is still running. Stop and close the window?"):
                return
            self._stop_event.set()
        try:
            self.destroy()
        except Exception:
            pass

    def _on_save_csv(self):
        # Allow export any time; if we have inputCSV we propose a default
        base = "gapfilled_CH4.csv"
        if self.inputCSV:
            stem = os.path.splitext(os.path.basename(self.inputCSV))[0]
            base = f"{stem}_gapfilled_CH4.csv"
            start_dir = os.path.dirname(self.inputCSV)
        else:
            start_dir = os.getcwd()

        path = filedialog.asksaveasfilename(
            title="Save gap-filled CSV",
            initialdir=start_dir,
            initialfile=base,
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            self._export_csv(path)
            messagebox.showinfo("Export", f"Saved: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Export error", str(e))

    # ------------------------------ Core worker ------------------------------ #

    def _worker(self, target: str, drivers: list[str], methods: list[str], cv_folds: int):
        try:
            # Prepare data
            df = self.df.copy()
            y = pd.to_numeric(df[target], errors="coerce")
            X = df[drivers].apply(pd.to_numeric, errors="coerce")

            mask_obs = y.notna()
            mask_miss = ~mask_obs
            n_obs = int(mask_obs.sum())
            n_miss = int(mask_miss.sum())

            if n_obs == 0:
                self._log("Target has no observed values.")
                self._set_status("Status: Idle")
                self._running = False
                return

            self._log(f"Target={target}; drivers={drivers}")
            self._log(f"Observed rows: {n_obs}; Missing rows to fill: {n_miss}")

            X_obs, y_obs = X[mask_obs].values, y[mask_obs].values
            X_miss = X[mask_miss].values if n_miss > 0 else None

            # Cross-validation
            splits = min(max(2, cv_folds), max(2, n_obs // 5))
            cv = KFold(n_splits=splits, shuffle=True, random_state=42)

            per_method_preds: dict[str, np.ndarray] = {}
            per_method_rmse: dict[str, float] = {}

            # Progress bookkeeping
            steps_total = len(methods) + (1 if n_miss > 0 else 0)
            step = 0
            self._prog_set(step, maximum=max(1, steps_total))

            for mname in methods:
                if self._stop_event.is_set():
                    self._log("Stopped by user.")
                    break

                est = self._make_pipeline(mname)
                if est is None:
                    self._log(f"[skip] {mname} not available (likely XGBoost missing).")
                    step += 1
                    self._prog_set(step)
                    continue

                self._log(f"=== {mname}: cross-validating… ===")
                rmses = []
                for tr, va in cv.split(X_obs):
                    if self._stop_event.is_set():
                        break
                    est.fit(X_obs[tr], y_obs[tr])
                    yp = est.predict(X_obs[va])
                    rmses.append(mean_squared_error(y_obs[va], yp, squared=False))
                if self._stop_event.is_set():
                    self._log("Stopped during CV.")
                    break

                rmse = float(np.mean(rmses)) if rmses else np.nan
                per_method_rmse[mname] = rmse
                self._log(f"    CV RMSE ≈ {rmse:.4g}")

                self._log(f"{mname}: fitting on all observed and predicting missing…")
                est.fit(X_obs, y_obs)
                if n_miss > 0:
                    yp_miss = est.predict(X_miss)
                    per_method_preds[mname] = yp_miss

                step += 1
                self._prog_set(step)

            # If we have predictions, write out columns & ensemble
            if (not self._stop_event.is_set()) and n_miss > 0 and per_method_preds:
                filled_cols = []
                for mname, preds in per_method_preds.items():
                    col = f"{target}_F_{mname}"
                    df[col] = df[target]
                    df.loc[mask_miss, col] = preds
                    filled_cols.append(col)

                ens_vals = np.vstack([df.loc[mask_miss, c].values for c in filled_cols])
                ens_mean = np.nanmean(ens_vals, axis=0)

                primary_F = f"{target}_F"
                df[primary_F] = df[target]
                df.loc[mask_miss, primary_F] = ens_mean

                # Update window df (so Save/Callback sees updates)
                self.df = df

                self._log("")
                self._log("Finished gap-filling. CV RMSE by method:")
                for mname, rm in per_method_rmse.items():
                    try:
                        self._log(f"  - {mname}: {rm:.4g}")
                    except Exception:
                        self._log(f"  - {mname}: {rm}")

                step += 1
                self._prog_set(step)
                self._set_status("Status: Done")

                # Notify parent (if provided)
                if callable(self.on_update_df):
                    try:
                        self.on_update_df(self.df.copy())
                    except Exception as e:
                        self._log(f"Callback error: {e}")
            else:
                if not self._stop_event.is_set():
                    self._log("Nothing to fill (no missing rows or no successful methods).")
                self._set_status("Status: Idle")

        except Exception as e:
            self._log(f"ERROR: {e}")
            messagebox.showerror("Gap-fill CH₄ error", str(e))
            self._set_status("Status: Error")
        finally:
            self._running = False
            try:
                self.protocol("WM_DELETE_WINDOW", self._on_close_request)
            except Exception:
                pass

    # ------------------------------ Pipelines ------------------------------ #

    def _make_pipeline(self, name: str) -> Pipeline | None:
        """
        Build an estimator pipeline with proper imputation and scaling.
        - Tree models: Imputer only (median)
        - PLS/KNN/SVR/MLP: Imputer + StandardScaler
        """
        name = name.strip()

        def pipe(est, scale: bool):
            steps = [("imputer", SimpleImputer(strategy="median"))]
            if scale:
                steps.append(("scaler", StandardScaler()))
            steps.append(("est", est))
            return Pipeline(steps)

        if name == "LinearRegression":
            return pipe(LinearRegression(), scale=True)
        if name == "PLSRegression":
            ncomp = max(1, int(self.var_pls.get()))
            return pipe(PLSRegression(n_components=ncomp), scale=True)
        if name == "KNN":
            k = max(1, int(self.var_k.get()))
            return pipe(KNeighborsRegressor(n_neighbors=k, weights="distance"), scale=True)
        if name == "SVR":
            return pipe(SVR(C=3.0, epsilon=0.05, kernel="rbf"), scale=True)
        if name == "MLP":
            return pipe(MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", max_iter=800, random_state=42), scale=True)
        if name == "RandomForest":
            return pipe(RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1), scale=False)
        if name == "ExtraTrees":
            return pipe(ExtraTreesRegressor(n_estimators=400, random_state=42, n_jobs=-1), scale=False)
        if name == "GradientBoosting":
            return pipe(GradientBoostingRegressor(random_state=42), scale=False)
        if name == "XGBoost":
            if not HAS_XGB:
                return None
            est = XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=0,
            )
            return pipe(est, scale=False)

        return None

    # ------------------------------ Export ------------------------------ #

    def _export_csv(self, path: str):
        exp = self.df.copy()
        # Preserve TIMESTAMP_START if present (format to %Y%m%d%H%M if datetime)
        if self.timestamp_col in exp.columns:
            if np.issubdtype(exp[self.timestamp_col].dtype, np.datetime64):
                try:
                    exp[self.timestamp_col] = pd.to_datetime(exp[self.timestamp_col]).dt.strftime("%Y%m%d%H%M")
                except Exception:
                    pass
        # Drop helper column
        if "__TS_DT__" in exp.columns:
            exp = exp.drop(columns=["__TS_DT__"])

        # Final export mapping NaN→-9999
        exp = exp.replace({np.nan: -9999})
        exp.to_csv(path, index=False)


# ------------------------------ Standalone smoke test ------------------------------ #
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    n = 1000
    rng = pd.date_range("2020-01-01", periods=n, freq="30min")
    df_demo = pd.DataFrame({
        "TIMESTAMP_START": [ts.strftime("%Y%m%d%H%M") for ts in rng],
        "FCH4_demo": np.sin(np.linspace(0, 40, n)) + np.random.normal(0, 0.2, n),
        "TA": 15 + 10*np.sin(np.linspace(0, 10, n)) + np.random.normal(0, 1, n),
        "SW_IN": np.clip(600*np.sin(np.linspace(0, 40, n)) + 50, 0, None),
        "WS": np.abs(np.random.normal(2, 0.6, n)),
        "VPD": np.abs(np.random.normal(1.2, 0.3, n)),
        "SWC": np.clip(np.random.normal(0.35, 0.05, n), 0.1, 0.6),
        "USTAR": np.abs(np.random.normal(0.3, 0.08, n)),
        "GPP": np.clip(8*np.sin(np.linspace(0, 10, n)) + 6, 0, None),
        "NEE": -np.clip(8*np.sin(np.linspace(0, 10, n)) + 6, 0, None) + np.random.normal(0, 0.5, n),
    })
    miss_idx = np.random.choice(n, size=n//3, replace=False)
    df_demo.loc[miss_idx, "FCH4_demo"] = np.nan
    df_demo.loc[np.random.choice(n, size=n//10, replace=False), "TA"] = -9999

    w = MethaneGapFillWindow(parent=root, df_in=df_demo, inputCSV=None,
                             shared_progressbar=None, on_update_df=None)
    w.mainloop()
