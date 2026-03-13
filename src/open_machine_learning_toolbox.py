#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MeaningFlux — Machine Learning Toolbox (EC-friendly, thread-safe, comparable)
Author: Leila C. Hernandez (LBNL) + assistant reorg
Updated: 2025-11-19

Key updates:
- Renamed "Flow-Gate LSTM" -> **Hysteresis-Gate LSTM (H-LSTM)**
- Robust timestamp handling (no reliance on index having 'TIMESTAMP_START')
- Plot x-axis uses datetime when available (else fallback to index)
- Compare → Overlay now always plots (legend + labels guaranteed)
- "Use fixed train/test split" checkbox (user decides)
- Training tab shows curves only for epoch-based models; others show a short note
- H-LSTM: simplified and robust timestamp + sequence construction
- Driver Discovery: multiple driver-ranking methods (RF importance, |Pearson r|, |LinReg coef|)
  and a Compare tab to visualize differences; clearer explanation of Hysteresis Explorer
"""

import time, math, queue, threading
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

# --- external Sequence class (optional, NOT used for H-LSTM now) ---
_HAS_EXT_SEQUENCE = False
try:
    from lstm_classes import Sequence
    _HAS_EXT_SEQUENCE = True
except Exception:
    Sequence = None

_stop_event = threading.Event()
_rng = np.random.RandomState(42)

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
    s.configure("TNotebook.Tab", background="#F6FBFD", padding=(10,6))
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
#                               UTILITIES
# =============================================================================
def _is_main_thread():
    return threading.current_thread() is threading.main_thread()

def ensure_on_main(widget, fn, *args, **kwargs):
    if _is_main_thread():
        return fn(*args, **kwargs)
    widget.after(0, lambda: fn(*args, **kwargs))

def on_main(widget, fn, *args, **kwargs):
    widget.after(0, lambda: fn(*args, **kwargs))

def _to_datetime_1d(obj):
    """
    Safely convert to datetime, even if `obj` is a DataFrame
    (e.g., duplicate timestamp columns). Always returns a 1-D Series-like.
    """
    if isinstance(obj, pd.DataFrame):
        obj = obj.iloc[:, 0]
    return pd.to_datetime(obj, errors="coerce")

def _resample_df(df, interval, ts_col):
    """Resample on ts_col using mean aggregation."""
    df = df.copy()
    df[ts_col] = _to_datetime_1d(df[ts_col])
    df = df.dropna(subset=[ts_col]).set_index(ts_col)
    return df.resample(interval).mean().dropna(how="any")

def _metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

def _resample_view(df_local, rs_mode, tscol):
    if rs_mode == 1:
        return _resample_df(df_local, "D", tscol), "Daily"
    if rs_mode == 2:
        return _resample_df(df_local, "W", tscol), "Weekly"
    # No resample; ensure datetime column exists and is valid
    dfw = df_local.copy()
    dfw[tscol] = _to_datetime_1d(dfw[tscol])
    dfw = dfw.dropna(subset=[tscol]).set_index(tscol, drop=False)
    dfw = dfw.dropna(how="any")
    return dfw, "No resample"

def _chronological_split(n, test_ratio=0.2):
    """Deterministic chronological split (train first, test last)."""
    n_test = int(round(test_ratio * n))
    idx = np.arange(n)
    return idx[:-n_test], idx[-n_test:]

def _random_fixed_split(n, test_ratio=0.2):
    idx = np.arange(n)
    _rng.shuffle(idx)
    n_test = int(round(test_ratio * n))
    return idx[n_test:], idx[:n_test]  # train, test

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
        self.parent.after(100, self._poll)

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
        self.parent.after(100, self._poll)

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
def run_linear(X, y, train_idx=None, test_idx=None):
    if train_idx is None or test_idx is None:
        X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_tr, X_ts = X[train_idx], X[test_idx]
        y_tr, y_ts = y[train_idx], y[test_idx]
    m = LinearRegression().fit(X_tr, y_tr)
    return y_ts, m.predict(X_ts), m, (train_idx, test_idx)

def run_rf(X, y, train_idx=None, test_idx=None):
    if train_idx is None or test_idx is None:
        X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_tr, X_ts = X[train_idx], X[test_idx]
        y_tr, y_ts = y[train_idx], y[test_idx]
    m = RandomForestRegressor(n_estimators=300, random_state=42).fit(X_tr, y_tr)
    return y_ts, m.predict(X_ts), m, (train_idx, test_idx)

def run_mlp(X, y, train_idx=None, test_idx=None):
    if train_idx is None or test_idx is None:
        X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_tr, X_ts = X[train_idx], X[test_idx]
        y_tr, y_ts = y[train_idx], y[test_idx]
    m = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42).fit(X_tr, y_tr)
    return y_ts, m.predict(X_ts), m, (train_idx, test_idx)

def run_keras_lstm(X, y, epochs=50, batch_size=32, train_idx=None, test_idx=None, progress_cb=None, info_cb=None):
    if Sequential is None:
        raise RuntimeError("TensorFlow/Keras not found. Install to enable LSTM.")
    if train_idx is None or test_idx is None:
        X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_tr, X_ts = X[train_idx], X[test_idx]
        y_tr, y_ts = y[train_idx], y[test_idx]
    # shape -> (samples, timesteps=1, features)
    X_tr = X_tr.reshape((X_tr.shape[0], 1, X_tr.shape[1]))
    X_ts = X_ts.reshape((X_ts.shape[0], 1, X_ts.shape[1]))
    model = Sequential([Input(shape=(1, X_tr.shape[2])), LSTM(50), Dense(1)])
    model.compile(optimizer="adam", loss="mse")

    class _CB(tf.keras.callbacks.Callback):
        def on_epoch_end(self, e, logs=None):
            if progress_cb and logs and "loss" in logs:
                progress_cb(e, float(logs["loss"]))
            if _stop_event.is_set():
                self.model.stop_training = True

    t0 = time.time()
    model.fit(X_tr, y_tr, epochs=int(epochs), batch_size=int(batch_size), verbose=0, callbacks=[_CB()])
    if info_cb: info_cb(f"{'Stopped' if _stop_event.is_set() else 'Training time'}: {time.time()-t0:.1f}s\n")
    yhat = model.predict(X_ts, verbose=0).flatten()
    return y_ts, yhat, None, (train_idx, test_idx)

# ----------------- Hysteresis-aware helpers -----------------
def _prepare_h_lstm_frame(df, predictors, target, gate_col, tscol, rs_mode):
    """
    Prepare a clean numeric DataFrame for H-LSTM:

    - Uses tscol to create a datetime index.
    - Optional resampling: None / Daily / Weekly.
    - Keeps only predictors + gate + target.
    - Drops NaNs.
    """
    cols_need = list(dict.fromkeys(predictors + [gate_col, target]))
    # Base frame with time
    base = df[cols_need + [tscol]].copy()
    base[tscol] = _to_datetime_1d(base[tscol])
    base = base.dropna(subset=[tscol]).set_index(tscol)

    if rs_mode == 1:
        base = base.resample("D").mean()
    elif rs_mode == 2:
        base = base.resample("W").mean()

    base = base.dropna(how="any")

    # Ensure numeric
    base = base.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna(how="any")

    return base  # index is datetime, columns = predictors + gate + target

def _make_seq_data(df_resampled, features, target, gate_col, seq_len):
    """
    Build LSTM sequences with an extra gradient (Δgate) feature.
    Returns X(N,T,F), y(N,), ts_out(N,) aligned to target times.

    Robust logic:
    - Use df_resampled.index as time axis.
    - Compute Δgate with .diff(), drop first row.
    - Drop NaNs AFTER computing Δgate.
    - Use dfw.index after dropna as timestamps (no fancy re-indexing).
    """
    # copy to avoid modifying caller
    dfw = df_resampled.copy()

    # Compute Δgate and drop the first row
    gate_vals = pd.to_numeric(dfw[gate_col], errors="coerce")
    dfw["__DGATE__"] = gate_vals.diff()
    dfw = dfw.iloc[1:, :]

    # columns: features + gate + Δgate + target
    cols_feat = list(dict.fromkeys(features + [gate_col]))
    cols = cols_feat + ["__DGATE__", target]

    for c in cols:
        if c not in dfw.columns:
            raise ValueError(f"Missing column: {c}")

    # Clean numeric / finite
    dfw = dfw[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna(how="any")

    # Time axis AFTER cleaning (ensures 1:1 alignment)
    ts_vec = pd.Index(dfw.index)

    from sklearn.preprocessing import MinMaxScaler
    arr_scaled = MinMaxScaler().fit_transform(dfw.values)

    Xs, Ys = [], []
    T = int(seq_len)
    for t in range(T, arr_scaled.shape[0]):
        Xs.append(arr_scaled[t - T:t, :-1])  # all except target
        Ys.append(arr_scaled[t, -1])         # target at time t

    ts_out = ts_vec[T:]
    X = np.asarray(Xs, dtype=float)
    y = np.asarray(Ys, dtype=float).reshape(-1)
    return X, y, pd.Index(ts_out)

# ----------------- Hysteresis-Gate LSTM (H-LSTM) runner -----------------
def run_hysteresis_lstm(df_resampled, features, target, gate_col, seq_len=20, hidden=96, epochs=100,
                        progress_cb=None, info_cb=None):
    """
    Returns: y_true_test (np.ndarray), y_pred_test (np.ndarray), ts_test (Index/DatetimeIndex)

    df_resampled:
      - Index is datetime
      - Columns: predictors + gate_col + target
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch not found. Install to enable H-LSTM.")

    X, y, ts_all = _make_seq_data(df_resampled, features, target, gate_col, int(seq_len))

    # chronologically split to preserve time order
    n = X.shape[0]
    n_tr = int(max(1, round(0.8 * n)))
    X_tr = torch.tensor(X[:n_tr], dtype=torch.float32)
    y_tr = torch.tensor(y[:n_tr], dtype=torch.float32).view(-1, 1)
    X_ts = torch.tensor(X[n_tr:], dtype=torch.float32)
    y_ts = torch.tensor(y[n_tr:], dtype=torch.float32).view(-1, 1)
    ts_test = ts_all[n_tr:]

    class HysteresisLSTM(nn.Module):
        """
        Simple LSTM head; the hysteresis signal (Δgate) is already part of inputs.
        """
        def __init__(self, in_size, hidden=96):
            super().__init__()
            self.lstm = nn.LSTM(in_size, hidden, batch_first=True)
            self.fc = nn.Linear(hidden, 1)
        def forward(self, x):
            y, _ = self.lstm(x)         # (B,T,H)
            return self.fc(y[:, -1, :]) # (B,1)

    model = HysteresisLSTM(in_size=X.shape[2], hidden=int(hidden))

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    if info_cb: info_cb(f"Data: N={n} | Train={n_tr} | Test={n-n_tr} | seq={seq_len} | F={X.shape[2]}\n")
    t0 = time.time()
    model.train()
    for e in range(int(epochs)):
        if _stop_event.is_set(): break
        opt.zero_grad()
        out = model(X_tr)              # (N_tr, 1)
        loss = loss_fn(out, y_tr)
        loss.backward(); opt.step()
        if progress_cb: progress_cb(e, float(loss.detach().cpu().numpy()))
    if info_cb:
        info_cb(f"{'Stopped' if _stop_event.is_set() else 'Training time'}: {time.time()-t0:.1f}s\n")

    model.eval()
    with torch.no_grad():
        y_pred = model(X_ts).cpu().numpy().reshape(-1)

    return y_ts.cpu().numpy().reshape(-1), y_pred, ts_test

# =============================================================================
#                        DRIVER DISCOVERY ANALYTICS
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
    parent_notebook.add(tab_predict, text="Predict")

    splitter = ttk.Panedwindow(tab_predict, orient="horizontal")
    splitter.pack(fill="both", expand=True)

    # LEFT controls
    controls = ttk.Frame(splitter, padding=(10, 10))
    controls.columnconfigure(0, weight=1)
    splitter.add(controls, weight=0)

    # RIGHT results
    results = ttk.Notebook(splitter)
    splitter.add(results, weight=1)

    Label(controls, text="Predictive Modeling", font=("TkDefaultFont", 11, "bold")).grid(sticky="w", pady=(0, 6))

    # Top action bar
    bar = ttk.Frame(controls); bar.grid(sticky="ew", pady=(0, 8))
    bar.columnconfigure(4, weight=1)
    btn_run = ttk.Button(bar, text="▶ Run"); btn_stop = ttk.Button(bar, text="■ Stop", state="disabled")
    btn_export = ttk.Button(bar, text="⤓ Export")
    btn_run.grid(row=0, column=0, padx=(0,6)); btn_stop.grid(row=0, column=1, padx=(0,6)); btn_export.grid(row=0, column=2, padx=(0,6))

    # Model selector
    box_model = ttk.LabelFrame(controls, text="Model"); box_model.grid(sticky="ew", pady=6)
    model = StringVar(controls, value="Random Forest")
    ttk.OptionMenu(box_model, model, model.get(),
                   "Linear Regression", "Random Forest", "Neural Network (MLP)",
                   "LSTM (Keras)", "Hysteresis-Gate LSTM (H-LSTM)").grid(sticky="ew", padx=6, pady=6)

    # Data selectors (de-duplicate column names for UI)
    seen = set(); cols = []
    for c in df.columns:
        if c not in seen:
            cols.append(c)
            seen.add(c)

    box_data = ttk.LabelFrame(controls, text="Data"); box_data.grid(sticky="ew", pady=6)
    ttk.Label(box_data, text="Predictors (multi-select)").grid(sticky="w", padx=6, pady=(6,0))
    lb_X = Listbox(box_data, selectmode=MULTIPLE, exportselection=False, height=8)
    for i, c in enumerate(cols): lb_X.insert(i, c)
    lb_X.grid(sticky="ew", padx=6, pady=4); box_data.columnconfigure(0, weight=1)

    ttk.Label(box_data, text="Target (y)").grid(sticky="w", padx=6)
    y_var = StringVar(box_data, value=cols[0]); ttk.OptionMenu(box_data, y_var, y_var.get(), *cols).grid(sticky="ew", padx=6, pady=(0,6))

    ttk.Label(box_data, text="Timestamp column").grid(sticky="w", padx=6)
    default_ts = next((c for c in cols if c.lower() in ["timestamp","time","datetime","timestamp_start","timestamp_end"]), cols[0])
    ts_var = StringVar(box_data, value=default_ts); ttk.OptionMenu(box_data, ts_var, ts_var.get(), *cols).grid(sticky="ew", padx=6, pady=(0,6))

    # Config
    box_cfg = ttk.LabelFrame(controls, text="Configuration"); box_cfg.grid(sticky="ew", pady=6)
    ttk.Label(box_cfg, text="Resampling").grid(row=0, column=0, sticky="w", padx=6, pady=(6,4))
    rs = IntVar(box_cfg, value=0)
    ttk.Radiobutton(box_cfg, text="None", variable=rs, value=0).grid(row=0, column=1, sticky="w")
    ttk.Radiobutton(box_cfg, text="Daily", variable=rs, value=1).grid(row=0, column=2, sticky="w")
    ttk.Radiobutton(box_cfg, text="Weekly", variable=rs, value=2).grid(row=0, column=3, sticky="w")

    # Splitting controls
    row_split = ttk.Frame(box_cfg); row_split.grid(row=1, column=0, columnspan=4, sticky="ew", padx=6, pady=(4,2))
    fixed_split_var = tk.BooleanVar(box_cfg, value=True)
    ttk.Checkbutton(row_split, text="Use fixed train/test split", variable=fixed_split_var).pack(side="left")
    chrono_var = tk.BooleanVar(box_cfg, value=True)
    ttk.Checkbutton(row_split, text="Chronological split (time-aware)", variable=chrono_var).pack(side="left", padx=(12,0))

    ttk.Label(box_cfg, text="Gate variable (for H-LSTM)").grid(row=2, column=0, sticky="w", padx=6, pady=(4,6))
    gate_var = StringVar(box_cfg, value=cols[0]); ttk.OptionMenu(box_cfg, gate_var, gate_var.get(), *cols).grid(row=2, column=1, columnspan=3, sticky="ew", padx=6, pady=(4,6))
    box_cfg.columnconfigure(4, weight=1)

    # Advanced
    adv = ttk.LabelFrame(controls, text="Advanced (sequence models)"); adv.grid(sticky="ew", pady=6)
    adv.columnconfigure(5, weight=1)
    ttk.Label(adv, text="Seq len").grid(row=0, column=0, sticky="w", padx=6, pady=(6,2))
    seq_len = StringVar(adv, value="20"); ttk.Entry(adv, textvariable=seq_len, width=6).grid(row=0, column=1, sticky="w")
    ttk.Label(adv, text="Hidden").grid(row=0, column=2, sticky="w", padx=(12,6))
    hidden  = StringVar(adv, value="96"); ttk.Entry(adv, textvariable=hidden, width=6).grid(row=0, column=3, sticky="w")
    ttk.Label(adv, text="Epochs").grid(row=0, column=4, sticky="w", padx=(12,6))
    epochs  = StringVar(adv, value="100"); ttk.Entry(adv, textvariable=epochs, width=6).grid(row=0, column=5, sticky="w")

    # Tips
    tk.Message(controls, width=320, text=(
        "Tips:\n"
        "• Start with Random Forest to gauge nonlinearity (and see feature importance).\n"
        "• Use LSTM/H-LSTM with daily or weekly resampling.\n"
        "• H-LSTM adds Δ(gate) to capture rising vs falling responses.\n"
        "• Keep 'fixed split' ON if you want comparable runs (non-LSTM models).\n"
        "• Chronological split preserves time order."
    ), fg="#444").grid(sticky="ew", pady=(6,0))

    # Results notebook tabs
    results_tabs = {}
    for name in ["Series", "Scatter", "Residuals", "Feature Importance", "Metrics", "Training", "Compare"]:
        fr = ttk.Frame(results)
        results.add(fr, text=name)
        results_tabs[name] = fr

    # Matplotlib canvases
    fig_series, ax_series = plt.subplots(figsize=(8.6, 3.2))
    can_series = FigureCanvasTkAgg(fig_series, master=results_tabs["Series"]); can_series.draw(); can_series.get_tk_widget().pack(fill="both", expand=True)

    fig_scat, ax_scat = plt.subplots(figsize=(8.6, 3.2))
    can_scat = FigureCanvasTkAgg(fig_scat, master=results_tabs["Scatter"]); can_scat.draw(); can_scat.get_tk_widget().pack(fill="both", expand=True)

    fig_resid, ax_resid = plt.subplots(figsize=(8.6, 3.2))
    can_resid = FigureCanvasTkAgg(fig_resid, master=results_tabs["Residuals"]); can_resid.draw(); can_resid.get_tk_widget().pack(fill="both", expand=True)

    fig_imp, ax_imp = plt.subplots(figsize=(8.6, 3.2))
    can_imp = FigureCanvasTkAgg(fig_imp, master=results_tabs["Feature Importance"]); can_imp.draw(); can_imp.get_tk_widget().pack(fill="both", expand=True)

    # Metrics labels
    box_metrics = ttk.Frame(results_tabs["Metrics"], padding=10); box_metrics.pack(fill="both", expand=True)
    lbl_rmse = ttk.Label(box_metrics, text="RMSE: —", font=("TkDefaultFont", 10, "bold"))
    lbl_mae  = ttk.Label(box_metrics, text="MAE: —",  font=("TkDefaultFont", 10, "bold"))
    lbl_r2   = ttk.Label(box_metrics, text="R²: —",   font=("TkDefaultFont", 10, "bold"))
    lbl_rmse.pack(anchor="w"); lbl_mae.pack(anchor="w"); lbl_r2.pack(anchor="w")

    # Training monitor
    monitor = TrainingMonitor(results_tabs["Training"])

    # ----------------------- Compare tab UI -----------------------
    tab_cmp = results_tabs["Compare"]
    row1 = ttk.Frame(tab_cmp); row1.pack(fill="x", pady=(6,4), padx=8)
    Label(row1, text="Metric:").pack(side="left")
    cmp_metric = StringVar(tab_cmp, value="RMSE")
    ttk.OptionMenu(row1, cmp_metric, "RMSE", "RMSE", "MAE", "R²").pack(side="left", padx=(6,18))
    btn_refresh = ttk.Button(row1, text="Refresh")
    btn_overlay = ttk.Button(row1, text="Overlay selected predictions")
    btn_clear   = ttk.Button(row1, text="Clear runs")
    btn_refresh.pack(side="left")
    btn_overlay.pack(side="left", padx=6)
    btn_clear.pack(side="left", padx=6)

    cols_cmp = ("when", "model", "resample", "target", "features", "rmse", "mae", "r2")
    tree = ttk.Treeview(tab_cmp, columns=cols_cmp, show="headings", height=6)
    for c in cols_cmp:
        tree.heading(c, text=c.upper())
        tree.column(c, width=110 if c not in ("features",) else 360, anchor="w")
    tree.pack(fill="x", padx=8, pady=(0,6))

    fig_cmp, ax_cmp = plt.subplots(figsize=(8.6, 3.0))
    can_cmp = FigureCanvasTkAgg(fig_cmp, master=tab_cmp); can_cmp.draw(); can_cmp.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0,6))

    fig_ov, ax_ov = plt.subplots(figsize=(8.6, 3.2))
    can_ov = FigureCanvasTkAgg(fig_ov, master=tab_cmp); can_ov.draw(); can_ov.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0,8))

    # ----------------------- handles -----------------------
    handles = dict(
        model=model, lb_X=lb_X, y_var=y_var, ts_var=ts_var,
        rs=rs, gate_var=gate_var, seq_len=seq_len, hidden=hidden, epochs=epochs,
        btn_run=btn_run, btn_stop=btn_stop, btn_export=btn_export,
        ax_series=ax_series, can_series=can_series,
        ax_scat=ax_scat, can_scat=can_scat,
        ax_resid=ax_resid, can_resid=can_resid,
        ax_imp=ax_imp, can_imp=can_imp,
        lbl_rmse=lbl_rmse, lbl_mae=lbl_mae, lbl_r2=lbl_r2,
        monitor=monitor, results_notebook=results, tab_feat=results_tabs["Feature Importance"],
        # compare
        runs=[], cmp_metric=cmp_metric, tree=tree,
        ax_cmp=ax_cmp, can_cmp=can_cmp, ax_ov=ax_ov, can_ov=can_ov,
        # config
        fixed_split=fixed_split_var, chrono_split=chrono_var
    )

    # ----------------------- Compare tab logic -----------------------
    def _update_compare_tab():
        for r in tree.get_children(): tree.delete(r)
        runs = handles["runs"]
        for i, r in enumerate(runs):
            tree.insert("", "end", iid=str(i), values=(
                str(r["when"]).split(".")[0],
                r["model"], r["resample"], r["target"],
                ", ".join(r["features"]),
                f'{r["rmse"]:.4g}', f'{r["mae"]:.4g}', f'{r["r2"]:.4g}'
            ))
        ax_cmp.cla()
        met = handles["cmp_metric"].get()
        if runs:
            labels = [f'{i}:{rr["model"].split()[0]}' for i, rr in enumerate(runs)]
            vals = [rr["rmse"] if met=="RMSE" else rr["mae"] if met=="MAE" else rr["r2"] for rr in runs]
            xpos = np.arange(len(vals))
            ax_cmp.bar(xpos, vals)
            ax_cmp.set_xticks(xpos); ax_cmp.set_xticklabels(labels, rotation=30, ha="right")
            ax_cmp.set_ylabel(met); ax_cmp.set_title(f"Model comparison by {met}")
        handles["can_cmp"].draw_idle()

    def _overlay_selected_preds():
        sel_items = tree.selection()
        if not sel_items:
            messagebox.showinfo("Compare", "Select one or more rows in the table.")
            return
        ax_ov.cla()
        drew_true = False
        for iid in sel_items:
            r = handles["runs"][int(iid)]
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
                ax_ov.plot(x, pdf["y_true"].values, label="True", alpha=0.8)
                drew_true = True
            # draw pred (give unique label)
            ax_ov.plot(x, pdf["y_pred"].values, "--", label=f'Pred: {r["model"]}')
        ax_ov.set_title("Overlay predictions")
        ax_ov.legend(loc="best")
        ax_ov.set_xlabel(xlab); ax_ov.set_ylabel(handles["y_var"].get())
        handles["can_ov"].draw_idle()

    btn_refresh.configure(command=_update_compare_tab)
    btn_overlay.configure(command=_overlay_selected_preds)
    btn_clear.configure(command=lambda: (handles["runs"].clear(), _update_compare_tab()))

    handles["compare_refresh"] = _update_compare_tab
    handles.update(dict(last_pred_df=None, last_importance_df=None))
    return tab_predict, handles

def _stash_exportables_predict(P, y_true, y_pred, timestamp, feature_names, importances):
    # predictions
    try:
        if timestamp is not None and len(timestamp) == len(y_true):
            pred_df = pd.DataFrame({"timestamp": _to_datetime_1d(timestamp), "y_true": y_true, "y_pred": y_pred})
        else:
            pred_df = pd.DataFrame({"index": np.arange(len(y_true)), "y_true": y_true, "y_pred": y_pred})
        P["last_pred_df"] = pred_df
    except Exception:
        P["last_pred_df"] = None
    # importance
    try:
        if feature_names is not None and importances is not None:
            fn = list(feature_names); imp = np.asarray(importances, dtype=float).ravel()
            L = min(len(fn), imp.shape[0]); fn = fn[:L]; imp = imp[:L]
            P["last_importance_df"] = pd.DataFrame({"feature": fn, "importance": imp})
        else:
            P["last_importance_df"] = None
    except Exception:
        P["last_importance_df"] = None

def _update_predict_plots(P, y_true, y_pred, title, y_var, timestamp=None, feature_names=None, importances=None):
    """Redraw Series / Scatter / Residuals / Metrics / Feature Importance."""
    _stash_exportables_predict(P, y_true, y_pred, timestamp, feature_names, importances)

    # Series
    ax = P["ax_series"]; ax.cla()
    if timestamp is not None and len(timestamp) == len(y_true):
        t_axis = _to_datetime_1d(timestamp)
        ax.plot(t_axis, y_true, label="True")
        ax.plot(t_axis, y_pred, "--", label="Pred")
        ax.set_xlabel("Time")
    else:
        ax.plot(y_true, label="True"); ax.plot(y_pred, "--", label="Pred"); ax.set_xlabel("Index")
    ax.set_title(title); ax.set_ylabel(y_var); ax.legend()
    P["can_series"].draw_idle()

    # Scatter
    ax = P["ax_scat"]; ax.cla()
    ax.scatter(y_true, y_pred, s=10, alpha=0.6)
    mn, mx = float(np.nanmin([y_true, y_pred])), float(np.nanmax([y_true, y_pred]))
    ax.plot([mn, mx], [mn, mx], "k--", lw=1)
    ax.set_title("True vs Predicted"); ax.set_xlabel("True"); ax.set_ylabel("Pred")
    P["can_scat"].draw_idle()

    # Residuals
    ax = P["ax_resid"]; ax.cla()
    resid = y_pred - y_true
    ax.hist(resid, bins=30, alpha=0.85)
    ax.set_title("Residuals"); ax.set_xlabel("Pred - True"); ax.set_ylabel("Count")
    P["can_resid"].draw_idle()

    # Metrics
    rmse, mae, r2 = _metrics(y_true, y_pred)
    P["lbl_rmse"].config(text=f"RMSE: {rmse:.4g}")
    P["lbl_mae"].config(text=f"MAE: {mae:.4g}")
    P["lbl_r2"].config(text=f"R²: {r2:.4g}")

    # Feature importance
    ax = P["ax_imp"]; ax.cla()
    if feature_names is not None and importances is not None:
        fn = list(feature_names)
        imp = np.asarray(importances, dtype=float).ravel()
        L = min(len(fn), imp.shape[0])
        fn = fn[:L]
        imp = np.nan_to_num(imp[:L], nan=0.0, posinf=0.0, neginf=0.0)
        xpos = np.arange(L)
        ax.bar(xpos, imp)
        ax.set_title("Feature Importance")
        ax.set_xticks(xpos)
        ax.set_xticklabels(fn, rotation=30, ha="right")
    else:
        ax.text(0.5, 0.5, "Not available for this model", ha="center", va="center", transform=ax.transAxes)
    P["can_imp"].draw_idle()

# =============================================================================
#               DRIVER DISCOVERY TAB: BUILD + PLOTTING + HYSTERESIS
# =============================================================================
def _build_drivers_tab(parent_notebook, df, inputname_site):
    tab = ttk.Frame(parent_notebook)
    parent_notebook.add(tab, text="Driver Discovery (statistical)")

    splitter = ttk.Panedwindow(tab, orient="horizontal")
    splitter.pack(fill="both", expand=True)

    # --- LEFT: scrollable controls panel ---
    scroll_container = ttk.Frame(splitter)
    scroll_container.columnconfigure(0, weight=1)

    # Canvas + vertical scrollbar
    scroll_canvas = tk.Canvas(
        scroll_container,
        borderwidth=0,
        highlightthickness=0
    )
    vscroll = ttk.Scrollbar(
        scroll_container,
        orient="vertical",
        command=scroll_canvas.yview
    )
    scroll_canvas.configure(yscrollcommand=vscroll.set)

    scroll_canvas.grid(row=0, column=0, sticky="nsew")
    vscroll.grid(row=0, column=1, sticky="ns")
    scroll_container.rowconfigure(0, weight=1)

    # Actual controls frame inside the canvas
    controls = ttk.Frame(scroll_canvas, padding=(10, 10))
    controls.columnconfigure(0, weight=1)

    # Put the controls frame into the canvas
    scroll_window = scroll_canvas.create_window(
        (0, 0),
        window=controls,
        anchor="nw"
    )

    # Update scrollregion whenever the size of 'controls' changes
    def _on_configure(event):
        scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))

    controls.bind("<Configure>", _on_configure)

    # Optional: make mouse wheel scroll the panel when cursor is over it
    def _on_mousewheel(event):
        # On Windows: event.delta is ±120; on macOS sometimes much smaller
        scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    controls.bind_all("<MouseWheel>", _on_mousewheel)

    splitter.add(scroll_container, weight=0)

    # --- RIGHT: results notebook (unchanged) ---
    results = ttk.Notebook(splitter)
    splitter.add(results, weight=1)


    Label(controls, text="Driver screening (not info theory)", font=("TkDefaultFont", 11, "bold")).grid(sticky="w", pady=(0,6))
    tk.Message(controls, width=320, text=(
        "Use this tab to screen likely drivers before modeling or causal analysis:\n"
        "• Random Forest permutation importance (ML)\n"
        "• Correlation (Pearson)\n"
        "• Linear regression coefficients\n"
        "• Partial Dependence Plots (PDP)\n"
        "• Hysteresis Explorer.\n\n"
        "For synergy/TE and other information-theory tools, use the separate Information Theory toolbox."
    ), fg="#444").grid(sticky="ew", pady=(0,8))

    # De-duplicate column names for UI
    seen = set(); cols = []
    for c in df.columns:
        if c not in seen:
            cols.append(c)
            seen.add(c)

    ttk.Label(controls, text="Candidate drivers (multi-select)").grid(sticky="w")
    lb_D = Listbox(controls, selectmode=MULTIPLE, exportselection=False, height=8)
    for i, c in enumerate(cols): lb_D.insert(i, c)
    lb_D.grid(sticky="ew", pady=4)

    ttk.Label(controls, text="Target (y)").grid(sticky="w")
    yD = StringVar(controls, value=cols[0]); ttk.OptionMenu(controls, yD, yD.get(), *cols).grid(sticky="ew", pady=(0,6))

    ttk.Label(controls, text="Timestamp column").grid(sticky="w")
    default_ts = next((c for c in cols if c.lower() in ["timestamp","time","datetime","timestamp_start","timestamp_end"]), cols[0])
    tsD = StringVar(controls, value=default_ts); ttk.OptionMenu(controls, tsD, tsD.get(), *cols).grid(sticky="ew", pady=(0,6))

    box_cfg = ttk.LabelFrame(controls, text="Configuration"); box_cfg.grid(sticky="ew", pady=6)
    ttk.Label(box_cfg, text="Resampling").grid(row=0, column=0, sticky="w", padx=6, pady=(6,4))
    rsD = IntVar(box_cfg, value=0)
    ttk.Radiobutton(box_cfg, text="None", variable=rsD, value=0).grid(row=0, column=1, sticky="w")
    ttk.Radiobutton(box_cfg, text="Daily", variable=rsD, value=1).grid(row=0, column=2, sticky="w")
    ttk.Radiobutton(box_cfg, text="Weekly", variable=rsD, value=2).grid(row=0, column=3, sticky="w")
    box_cfg.columnconfigure(4, weight=1)

    ttk.Button(controls, text="Analyze Drivers",
               command=lambda: threading.Thread(target=_run_drivers_worker,
                        args=(df, lb_D, yD, tsD, rsD, results), daemon=True).start()
               ).grid(sticky="ew", pady=8)

    # Hysteresis Explorer controls
    ttk.Separator(controls, orient="horizontal").grid(sticky="ew", pady=10)
    Label(controls, text="Hysteresis Explorer", font=("TkDefaultFont", 10, "bold")).grid(sticky="w")
    tk.Message(
        controls,
        width=320,
        text=(
            "Hysteresis Explorer lets you see whether the relationship between the target\n"
            "and a 'gate' variable is different when the gate is increasing vs decreasing.\n"
            "Points are colored by the sign of Δgate, and arrows show the direction of\n"
            "evolution in time. This helps detect memory effects and threshold behavior\n"
            "not visible in a simple scatter plot.\n\n"
            "Good gate candidates: u*, VPD, Rn, PBLH, soil moisture, LE, etc."
        ),
        fg="#444"
    ).grid(sticky="ew", pady=(0,6))

    gate_sel = StringVar(controls, value=cols[0]); ttk.Label(controls, text="Gate variable").grid(sticky="w")
    ttk.OptionMenu(controls, gate_sel, gate_sel.get(), *cols).grid(sticky="ew", pady=(0,6))
    row_h = ttk.Frame(controls); row_h.grid(sticky="ew")
    ttk.Label(row_h, text="Smoothing").grid(row=0, column=0, sticky="w")
    smooth_var = StringVar(row_h, value="3"); ttk.Entry(row_h, textvariable=smooth_var, width=6).grid(row=0, column=1, sticky="w", padx=(6,12))
    ttk.Label(row_h, text="Arrows every").grid(row=0, column=2, sticky="w")
    arrows_var = StringVar(row_h, value="20"); ttk.Entry(row_h, textvariable=arrows_var, width=6).grid(row=0, column=3, sticky="w", padx=(6,0))

    ttk.Button(controls, text="Open Hysteresis Explorer",
               command=lambda: threading.Thread(target=_run_hysteresis_worker,
                    args=(df, yD, gate_sel, tsD, rsD, smooth_var, arrows_var, results), daemon=True).start()
               ).grid(sticky="ew", pady=8)

    # Results notebook tabs
    tab_imp = ttk.Frame(results); results.add(tab_imp, text="Importance")
    tab_corr = ttk.Frame(results); results.add(tab_corr, text="Correlation")
    tab_pdp = ttk.Frame(results); results.add(tab_pdp, text="PDP")
    tab_hyst = ttk.Frame(results); results.add(tab_hyst, text="Hysteresis")
    tab_cmp = ttk.Frame(results); results.add(tab_cmp, text="Compare drivers")

    # Importance
    fig_imp, ax_imp = plt.subplots(figsize=(8.6, 3.2))
    can_imp = FigureCanvasTkAgg(fig_imp, master=tab_imp); can_imp.draw(); can_imp.get_tk_widget().pack(fill="both", expand=True)

    # Correlation
    fig_corr, ax_corr = plt.subplots(figsize=(8.6, 3.2))
    can_corr = FigureCanvasTkAgg(fig_corr, master=tab_corr); can_corr.draw(); can_corr.get_tk_widget().pack(fill="both", expand=True)

    # PDP
    fig_pdp, axs_pdp = plt.subplots(1, 3, figsize=(9.2, 3.2))
    can_pdp = FigureCanvasTkAgg(fig_pdp, master=tab_pdp); can_pdp.draw(); can_pdp.get_tk_widget().pack(fill="both", expand=True)

    # Hysteresis
    fig_hyst, ax_hyst = plt.subplots(figsize=(8.6, 4.0))
    can_hyst = FigureCanvasTkAgg(fig_hyst, master=tab_hyst); can_hyst.draw(); can_hyst.get_tk_widget().pack(fill="both", expand=True)

    # Compare drivers tab
    row_cmp = ttk.Frame(tab_cmp); row_cmp.pack(fill="x", padx=8, pady=(6,4))
    Label(row_cmp, text="Ranking metric:").pack(side="left")
    cmp_metric = StringVar(tab_cmp, value="RF Permutation Importance")
    ttk.OptionMenu(
        row_cmp,
        cmp_metric,
        "RF Permutation Importance",
        "RF Permutation Importance",
        "|Pearson r|",
        "|LinReg coef|"
    ).pack(side="left", padx=(6,18))
    tk.Message(
        tab_cmp,
        width=600,
        text=(
            "This view compares driver rankings obtained with three methods:\n"
            "• RF Permutation Importance (ML-based): change in score when each driver is shuffled.\n"
            "• |Pearson r|: absolute linear correlation with the target.\n"
            "• |LinReg coef|: absolute value of coefficients in a multivariate linear regression.\n"
            "Use it to see which drivers are consistently important across methods."
        ),
        fg="#444"
    ).pack(fill="x", padx=8, pady=(0,4))

    fig_cmp, ax_cmp = plt.subplots(figsize=(8.6, 3.2))
    can_cmp = FigureCanvasTkAgg(fig_cmp, master=tab_cmp); can_cmp.draw(); can_cmp.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0,8))

    handles = dict(
        lb_D=lb_D, yD=yD, tsD=tsD, rsD=rsD,
        gate_sel=gate_sel, smooth_var=smooth_var, arrows_var=arrows_var,
        ax_imp=ax_imp, can_imp=can_imp,
        ax_corr=ax_corr, can_corr=can_corr,
        axs_pdp=axs_pdp, can_pdp=can_pdp,
        ax_hyst=ax_hyst, can_hyst=can_hyst,
        results=results,
        last_importance_df=None,
        # compare drivers
        cmp_metric=cmp_metric, ax_cmp=ax_cmp, can_cmp=can_cmp,
        imp_rf=None, imp_corr=None, imp_lin=None
    )
    return tab, handles

def _refresh_driver_compare(D):
    """Update the 'Compare drivers' bar chart based on selected metric."""
    ax = D["ax_cmp"]; ax.cla()
    metric = D["cmp_metric"].get()

    if metric == "RF Permutation Importance":
        dfm = D.get("imp_rf")
        ylabel = "Permutation importance (Δ score)"
        title = "Driver ranking: RF Permutation Importance"
    elif metric == "|Pearson r|":
        dfm = D.get("imp_corr")
        ylabel = "|Pearson r|"
        title = "Driver ranking: |Pearson correlation|"
    else:  # |LinReg coef|
        dfm = D.get("imp_lin")
        ylabel = "|coefficient|"
        title = "Driver ranking: |Linear regression coef|"

    if dfm is None or getattr(dfm, "empty", True):
        ax.text(0.5, 0.5, "No rankings computed yet.\nClick 'Analyze Drivers' first.",
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
    # Permutation Importance (RF)
    pim = pis = None
    rf = None
    try:
        rf = RandomForestRegressor(n_estimators=300, random_state=42).fit(dfw[features].values, dfw[target].values)
        r = permutation_importance(rf, dfw[features].values, dfw[target].values, n_repeats=10, random_state=42)
        pim = np.asarray(r.importances_mean, dtype=float).ravel()
        pis = np.asarray(r.importances_std, dtype=float).ravel()
    except Exception:
        pim, pis, rf = None, None, None

    ax = D["ax_imp"]; ax.cla()
    if pim is not None and pim.size and len(features):
        L = min(len(features), pim.shape[0], pis.shape[0])
        fn = list(features)[:L]
        y = np.nan_to_num(pim[:L], nan=0.0, posinf=0.0, neginf=0.0)
        yerr = np.nan_to_num(pis[:L], nan=0.0, posinf=0.0, neginf=0.0)
        xpos = np.arange(L)
        ax.bar(xpos, y, yerr=yerr, capsize=3)
        ax.set_title("Permutation Importance (Random Forest)")
        ax.set_ylabel("Δ score")
        ax.set_xticks(xpos)
        ax.set_xticklabels(fn, rotation=30, ha="right")
        try:
            D["last_importance_df"] = pd.DataFrame({"feature": fn, "perm_importance": y, "perm_importance_std": yerr})
        except Exception:
            D["last_importance_df"] = None

        # Store RF-based ranking for compare tab
        try:
            D["imp_rf"] = pd.DataFrame({"feature": fn, "value": y})
        except Exception:
            D["imp_rf"] = None
    else:
        ax.text(0.5, 0.5, "Importance unavailable", ha="center", va="center", transform=ax.transAxes)
        D["imp_rf"] = None
    D["can_imp"].draw_idle()

    # Correlation heatmap + |r| rankings
    corr_abs_vals = None
    try:
        ax = D["ax_corr"]; ax.cla()
        cols = [*features, target]
        corr = dfw[cols].corr(method=corr_kind)
        im = ax.imshow(corr.values, aspect="auto")
        ax.set_title(f"Correlation heatmap ({corr_kind})")
        ax.set_xticks(range(corr.shape[1])); ax.set_yticks(range(corr.shape[0]))
        ax.set_xticklabels(corr.columns, rotation=30, ha="right"); ax.set_yticklabels(corr.index)
        fig = D["can_corr"].figure
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        D["can_corr"].draw_idle()

        # |r| with target for ranking
        corr_abs_vals = corr[target].loc[features].abs().values
        D["imp_corr"] = pd.DataFrame({"feature": features, "value": corr_abs_vals})
    except Exception:
        D["imp_corr"] = None

    # Linear regression coefficients (absolute) for ranking
    try:
        reg = LinearRegression().fit(dfw[features].values, dfw[target].values)
        coef_abs = np.abs(reg.coef_)
        D["imp_lin"] = pd.DataFrame({"feature": features, "value": coef_abs})
    except Exception:
        D["imp_lin"] = None

    # PDP (top-3)
    for a in D["axs_pdp"]: a.cla()
    try:
        from sklearn.inspection import PartialDependenceDisplay
        if rf is not None and pim is not None and len(features) > 0:
            Lp = min(len(features), pim.shape[0])
            if Lp > 0:
                order = np.argsort(pim[:Lp])[::-1][:3]
                for axp, i in zip(D["axs_pdp"], order):
                    PartialDependenceDisplay.from_estimator(rf, dfw[features], [int(i)], ax=axp, grid_resolution=20)
                    axp.set_title(f"PDP: {features[int(i)]}")
        else:
            for axp in D["axs_pdp"]:
                axp.text(0.5, 0.5, "PDP unavailable", ha="center", va="center", transform=axp.transAxes)
    except Exception:
        for axp in D["axs_pdp"]:
            axp.text(0.5, 0.5, "PDP unavailable", ha="center", va="center", transform=axp.transAxes)
    D["can_pdp"].draw_idle()

    # Update compare-drivers tab
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

def _pick_split_indices(P, n_samples):
    """Return (train_idx, test_idx) based on user choices."""
    if P["fixed_split"].get():
        if P["chrono_split"].get():
            tr, ts = _chronological_split(n_samples)
        else:
            tr, ts = _random_fixed_split(n_samples)
    else:
        tr, ts = None, None
    return tr, ts

def _run_predict_worker(P, df):
    try:
        # work with *unique* column names for selection
        all_cols = list(dict.fromkeys(df.columns))
        sel = [all_cols[i] for i in P["lb_X"].curselection()]
        tgt = P["y_var"].get(); tscol = P["ts_var"].get()
        if not sel:
            return on_main(P["can_series"].get_tk_widget(), messagebox.showerror, "Error", "Select at least one predictor.")
        if tgt not in df.columns:
            return on_main(P["can_series"].get_tk_widget(), messagebox.showerror, "Error", "Target not found.")
        if tscol not in df.columns:
            return on_main(P["can_series"].get_tk_widget(), messagebox.showerror, "Error", "Timestamp column not found.")

        method = P["model"].get()
        rs_mode = P["rs"].get()
        rs_label = {0: "No resample", 1: "Daily", 2: "Weekly"}.get(rs_mode, "No resample")

        # For non-H-LSTM models: shared resample logic
        imp = None
        maybe_ts = None
        tr = ts = None

        # H-LSTM: prepare its own clean frame (no train_test_split, no _resample_view)
        if method == "Hysteresis-Gate LSTM (H-LSTM)":
            g = P["gate_var"].get()
            # For safety, remove timestamp column from predictors if present
            if tscol in sel:
                sel = [c for c in sel if c != tscol]
                if not sel:
                    return on_main(
                        P["can_series"].get_tk_widget(),
                        messagebox.showerror,
                        "Error",
                        "For H-LSTM, the timestamp column cannot be the only predictor.\n"
                        "Please select at least one additional predictor variable."
                    )

            df_h = _prepare_h_lstm_frame(df, sel, tgt, g, tscol, rs_mode)

            sL = int(P["seq_len"].get()); hS = int(P["hidden"].get()); ep = int(P["epochs"].get())
            _stop_event.clear()
            P["monitor"].reset(ep, "Hysteresis-Gate LSTM (H-LSTM)",
                               f"Target={tgt} | X={sel} | Gate={g} | seq={sL}, hidden={hS}, epochs={ep}")

            y_true, y_pred, ts_test = run_hysteresis_lstm(
                df_h, [c for c in df_h.columns if c not in [tgt]], tgt, g,
                seq_len=sL, hidden=hS, epochs=ep,
                progress_cb=P["monitor"].push, info_cb=P["monitor"].info
            )
            maybe_ts = ts_test
            title = f"H-LSTM | {rs_label}"

        else:
            # Shared frame for non-H-LSTM models
            dfw, rs_label = _resample_view(df[[*sel, tgt, tscol]], rs_mode, tscol)
            X_all = dfw[sel].values; y_all = dfw[tgt].values
            tr, ts = _pick_split_indices(P, len(X_all))

            # If non-epoch model, show info in Training tab
            if method in ("Linear Regression", "Random Forest", "Neural Network (MLP)"):
                P["monitor"].reset(1, method, f"Target={tgt} | X={sel}")
                P["monitor"].info_note("No epoch curve for this model. Metrics and plots will update after fit.\n")

            if method == "Linear Regression":
                y_true, y_pred, _model, (tr, ts) = run_linear(X_all, y_all, tr, ts)
                title = f"Linear Regression | {rs_label}"

            elif method == "Random Forest":
                y_true, y_pred, model, (tr, ts) = run_rf(X_all, y_all, tr, ts)
                imp = getattr(model, "feature_importances_", None)
                title = f"Random Forest | {rs_label}"

            elif method == "Neural Network (MLP)":
                y_true, y_pred, _model, (tr, ts) = run_mlp(X_all, y_all, tr, ts)
                title = f"MLP | {rs_label}"

            elif method == "LSTM (Keras)":
                if Sequential is None:
                    return on_main(P["can_series"].get_tk_widget(), messagebox.showerror, "Missing dependency", "TensorFlow/Keras not found.")
                _stop_event.clear()
                P["monitor"].reset( int(P['epochs'].get() or 50), "LSTM (Keras)",
                                    f"Target={tgt} | X={sel} | tip: try daily resampling" )
                y_true, y_pred, _model, (tr, ts) = run_keras_lstm(
                    X_all, y_all,
                    epochs=int(P["epochs"].get() or 50),
                    batch_size=32,
                    train_idx=tr, test_idx=ts,
                    progress_cb=P["monitor"].push,
                    info_cb=P["monitor"].info
                )
                title = f"LSTM | {rs_label}"
            else:
                # just in case
                return

            # Build timestamp for non-sequence models
            if (tr is not None) and (ts is not None):
                if rs_mode in (1, 2):
                    idx = dfw.index.values
                    maybe_ts = _to_datetime_1d(idx[ts])
                else:
                    if tscol in dfw.columns:
                        maybe_ts = _to_datetime_1d(dfw.iloc[ts][tscol].values)
                    else:
                        maybe_ts = _to_datetime_1d(dfw.index.values[ts])

        # pick correct timestamp for plotting
        timestamp = maybe_ts

        # Update plots
        _update_predict_plots(P, y_true, y_pred, title, P["y_var"].get(), timestamp=timestamp,
                              feature_names=sel, importances=imp)

        # Update Compare registry
        rmse, mae, r2 = _metrics(y_true, y_pred)
        run_rec = dict(
            when=pd.Timestamp.utcnow(),
            model=method,
            resample=rs_label,
            target=tgt,
            features=sel,
            rmse=rmse, mae=mae, r2=r2,
            pred_df=P.get("last_pred_df")
        )
        P["runs"].append(run_rec)
        P["compare_refresh"]()

    except Exception as e:
        P["monitor"].error(str(e))
        on_main(P["can_series"].get_tk_widget(), messagebox.showerror, "Error", str(e))
    finally:
        on_main(P["btn_run"], P["btn_run"].configure, state="normal")
        on_main(P["btn_stop"], P["btn_stop"].configure, state="disabled")

def _run_drivers_worker(df, lb_D, yD, tsD, rsD, results_notebook):
    try:
        all_cols = list(dict.fromkeys(df.columns))
        sel = [all_cols[i] for i in lb_D.curselection()]
        tgt = yD.get(); tscol = tsD.get()
        if not sel: return on_main(results_notebook, messagebox.showerror, "Error", "Select at least one candidate driver.")
        if tgt not in df.columns: return on_main(results_notebook, messagebox.showerror, "Error", "Target not found.")
        if tscol not in df.columns: return on_main(results_notebook, messagebox.showerror, "Error", "Timestamp column not found.")

        D = getattr(results_notebook, "_handles", None)
        if D is None: return

        dfw, _ = _resample_view(df[[*sel, tgt, tscol]], rsD.get(), tscol)
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
        dfx = _make_hysteresis_view(df, tgt, gate, tscol, rsD.get(), smooth=int(smooth_var.get()))
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
    """Open MeaningFlux ML window with Predict & Driver Discovery tabs."""
    global _ml_window
    if _ml_window is not None and tk.Toplevel.winfo_exists(_ml_window):
        messagebox.showinfo("Info", "MeaningFlux ML window already open.")
        return

    _ml_window = tk.Toplevel()
    _ml_window.title(f"MeaningFlux — Machine Learning · {inputname_site}")
    _ml_window.geometry("1280x900")
    set_simple_theme(_ml_window)

    nb = ttk.Notebook(_ml_window); nb.pack(fill="both", expand=True)

    tab_predict, P = _build_predict_tab(nb, df, inputname_site)
    tab_drivers, D = _build_drivers_tab(nb, df, inputname_site)
    D["results"]._handles = D  # store handles for async updates

    def _run():
        _stop_event.clear()
        P["btn_run"].configure(state="disabled")
        P["btn_stop"].configure(state="normal")
        threading.Thread(target=_run_predict_worker, args=(P, df), daemon=True).start()

    def _stop():
        _stop_event.set()
        P["monitor"].stopped()
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
                        "rmse": r["rmse"], "mae": r["mae"], "r2": r["r2"],
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
        try: _ml_window.destroy()
        finally: _ml_window = None
    _ml_window.protocol("WM_DELETE_WINDOW", on_close)

def open_machine_learning_toolbox(df: pd.DataFrame, inputname_site: str):
    """Backwards-compatible wrapper."""
    return open_meaningflux_ml(df, inputname_site)


