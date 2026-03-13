# -*- coding: utf-8 -*-
"""
MeaningFlux → AMF-BASE-QAQC (local preflight) 

    calc_data_AMF_BASE_QAQC(df, inputname_site)
"""

import os
import sys
import tempfile
import subprocess
import zipfile
import shutil
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd

from tkinter import (
    Toplevel, Frame, Label, Text, Button, END, messagebox, BooleanVar, Checkbutton
)
from tkinter import filedialog

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# -------------------------------------------------------------------
# AMF-BASE auto-download / discovery
# -------------------------------------------------------------------

GITHUB_ZIP_URL = "https://github.com/AMF-FLX/AMF-BASE-QAQC/archive/refs/heads/main.zip"


def _ensure_qaqc_cfg_exists(repo_root: str) -> str:
    """
    Ensure AMF-BASE has a repo-root qaqc.cfg.
      template: <repo_root>/processing/qaqc_template.cfg
      cfg:      <repo_root>/qaqc.cfg
    """
    cfg_path = os.path.join(repo_root, "qaqc.cfg")
    template_path = os.path.join(repo_root, "processing", "qaqc_template.cfg")

    if os.path.isfile(cfg_path):
        return cfg_path

    if not os.path.isfile(template_path):
        raise RuntimeError(
            "AMF-BASE template config not found.\n"
            f"Expected: {template_path}"
        )

    shutil.copyfile(template_path, cfg_path)
    return cfg_path


def _ensure_amf_support_files_exist(repo_root: str) -> None:
    """
    AMF-BASE opens some files by bare filename relative to cwd.
    Since we run with cwd=repo_root, ensure these exist in repo root.
    """
    required = [
        (os.path.join(repo_root, "processing", "Check_messages.txt"),
         os.path.join(repo_root, "Check_messages.txt")),
    ]

    for src, dst in required:
        if os.path.isfile(dst):
            continue
        if not os.path.isfile(src):
            raise RuntimeError(
                "Missing AMF-BASE support file.\n"
                f"Expected source: {src}\n"
                f"Expected destination: {dst}"
            )
        shutil.copyfile(src, dst)


def ensure_amf_base_available() -> str:
    """
    Ensure AMF-BASE-QAQC repo is available locally next to this file.

    Returns: absolute path to processing/main.py
    """
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.join(here, "AMF-BASE-QAQC")
    main_py = os.path.join(repo_root, "processing", "main.py")

    if os.path.isfile(main_py):
        _ensure_qaqc_cfg_exists(repo_root)
        _ensure_amf_support_files_exist(repo_root)
        return main_py

    ok = messagebox.askyesno(
        "AmeriFlux QA/QC (AMF-BASE) not found",
        "AMF-BASE-QAQC is not installed locally.\n\n"
        "MeaningFlux can download the official version from GitHub:\n"
        "    https://github.com/AMF-FLX/AMF-BASE-QAQC\n\n"
        "Do you want to download it now?"
    )
    if not ok:
        raise RuntimeError("AMF-BASE-QAQC missing and user declined download.")

    try:
        with tempfile.TemporaryDirectory(prefix="amf_base_download_") as tmpdir:
            zip_path = os.path.join(tmpdir, "AMF-BASE-QAQC.zip")
            urllib.request.urlretrieve(GITHUB_ZIP_URL, zip_path)

            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmpdir)

            extracted_root = None
            for item in os.listdir(tmpdir):
                candidate = os.path.join(tmpdir, item)
                if os.path.isdir(candidate) and os.path.isfile(
                    os.path.join(candidate, "processing", "main.py")
                ):
                    extracted_root = candidate
                    break

            if extracted_root is None:
                raise RuntimeError("Downloaded ZIP but could not locate processing/main.py.")

            if os.path.isdir(repo_root):
                shutil.rmtree(repo_root)

            shutil.move(extracted_root, repo_root)

    except Exception as e:
        raise RuntimeError(f"Failed to download or extract AMF-BASE-QAQC:\n{e}")

    if not os.path.isfile(main_py):
        raise RuntimeError("AMF-BASE main.py not found after extraction.")

    _ensure_qaqc_cfg_exists(repo_root)
    _ensure_amf_support_files_exist(repo_root)
    return main_py


# -------------------------------------------------------------------
# SITE_ID support: read AMF-BASE site dictionary
# -------------------------------------------------------------------

def _load_amf_valid_site_ids(amf_main_py: str):
    """
    Load AMF-BASE site dictionary (SiteAttributes) to obtain valid site IDs.
    Returns a sorted list; empty list if import fails.
    """
    repo_root = os.path.dirname(os.path.dirname(amf_main_py))
    processing_dir = os.path.join(repo_root, "processing")

    if processing_dir not in sys.path:
        sys.path.insert(0, processing_dir)

    try:
        from site_attrs import SiteAttributes  # type: ignore
        d = SiteAttributes().get_site_dict()
        return sorted(list(d.keys()))
    except Exception:
        return []


def _pick_placeholder_site_id(amf_main_py: str) -> str:
    """
    Pick a placeholder SITE_ID from AMF's site dictionary.
    Uses first in sorted order; falls back to 'US-XXX' if list unavailable.
    """
    valid = _load_amf_valid_site_ids(amf_main_py)
    return valid[0] if valid else "US-XXX"


def _infer_site_id_from_filename(inputname_site):
    """
    Heuristic: find first token like 'US-IS3' from the uploaded filename.
    """
    base_name = os.path.basename(str(inputname_site))
    guess_site = "US-XXX"
    parts = base_name.replace(".", "_").split("_")
    for tok in parts:
        tok = tok.strip()
        if "-" in tok and len(tok) >= 6:
            guess_site = tok
            break
    return guess_site


# -------------------------------------------------------------------
# Resolution inference + FP-IN export (strict timestamp formatting)
# -------------------------------------------------------------------

def _infer_resolution_from_df(df):
    """
    Infer time resolution from TIMESTAMP_START spacing.
    Returns (resolution 'HH'/'HR', median_dt_minutes or None).
    """
    if "TIMESTAMP_START" not in df.columns:
        return "HH", None

    ts = pd.to_datetime(df["TIMESTAMP_START"], errors="coerce").dropna()
    if ts.size < 2:
        return "HH", None

    deltas = ts.diff().dt.total_seconds().div(60.0).dropna()
    deltas = deltas[deltas > 0]
    if deltas.empty:
        return "HH", None

    dt = float(deltas.median())
    if 20.0 <= dt < 45.0:
        return "HH", dt
    if 45.0 <= dt < 90.0:
        return "HR", dt
    return "HH", dt


def _export_df_to_amf_fp_in(df, site_id, resolution, tmpdir):
    """
    Export df as AmeriFlux FP-IN style CSV with strict timestamp formatting.

    Enforces:
      - TIMESTAMP_START, TIMESTAMP_END as YYYYMMDDHHMM (12 chars)
      - Adds TIMESTAMP_END if missing (median timestep; fallback 30/60)
      - Drops *_QC, DATESTAMP_START/END by default (these often confuse AMF-BASE)
      - NaN -> -9999
    """
    if "TIMESTAMP_START" not in df.columns:
        raise ValueError("TIMESTAMP_START column required for FP-IN export.")

    df_out = df.copy()

    # Parse and drop invalid TIMESTAMP_START rows
    ts_start = pd.to_datetime(df_out["TIMESTAMP_START"], errors="coerce")
    good = ~ts_start.isna()
    df_out = df_out.loc[good].copy()
    ts_start = ts_start.loc[good]

    if df_out.empty:
        raise ValueError("All TIMESTAMP_START values failed to parse (no rows to export).")

    # Infer timestep minutes
    deltas = ts_start.diff().dt.total_seconds().div(60.0).dropna()
    deltas = deltas[deltas > 0]
    if not deltas.empty:
        step_min = int(round(float(deltas.median())))
    else:
        step_min = 30 if resolution == "HH" else 60

    # Create/repair TIMESTAMP_END
    if "TIMESTAMP_END" in df_out.columns:
        ts_end = pd.to_datetime(df_out["TIMESTAMP_END"], errors="coerce")
        if ts_end.isna().all():
            ts_end = ts_start + pd.to_timedelta(step_min, unit="m")
        else:
            ts_end = ts_end.fillna(ts_start + pd.to_timedelta(step_min, unit="m"))
    else:
        ts_end = ts_start + pd.to_timedelta(step_min, unit="m")
        df_out["TIMESTAMP_END"] = ts_end

    # Strict formatting
    df_out["TIMESTAMP_START"] = ts_start.dt.strftime("%Y%m%d%H%M")
    df_out["TIMESTAMP_END"] = pd.to_datetime(df_out["TIMESTAMP_END"], errors="coerce").dt.strftime("%Y%m%d%H%M")

    # Validate lengths are 12 characters
    if df_out["TIMESTAMP_START"].astype(str).str.len().ne(12).any():
        bad = df_out.loc[df_out["TIMESTAMP_START"].astype(str).str.len().ne(12), "TIMESTAMP_START"].head(5).tolist()
        raise ValueError(f"TIMESTAMP_START formatting failed (examples: {bad})")
    if df_out["TIMESTAMP_END"].astype(str).str.len().ne(12).any():
        bad = df_out.loc[df_out["TIMESTAMP_END"].astype(str).str.len().ne(12), "TIMESTAMP_END"].head(5).tolist()
        raise ValueError(f"TIMESTAMP_END formatting failed (examples: {bad})")

    # Drop columns AMF often flags
    drop_cols = [c for c in df_out.columns if c.upper().endswith("_QC")]
    for c in ("DATESTAMP_START", "DATESTAMP_END"):
        if c in df_out.columns:
            drop_cols.append(c)
    drop_cols = [c for c in drop_cols if c not in ("TIMESTAMP_START", "TIMESTAMP_END")]
    if drop_cols:
        df_out.drop(columns=drop_cols, inplace=True, errors="ignore")

    # Build filename
    ts_start_str = str(df_out["TIMESTAMP_START"].iloc[0])
    ts_end_str = str(df_out["TIMESTAMP_END"].iloc[-1])

    fname = f"{site_id}_{resolution}_{ts_start_str}_{ts_end_str}.csv"
    full_path = os.path.join(tmpdir, fname)

    df_out = df_out.replace({np.nan: -9999})
    df_out.to_csv(full_path, index=False)

    return full_path, step_min


# -------------------------------------------------------------------
# Run AMF-BASE + bypass logic + classification + user-readable summary
# -------------------------------------------------------------------

def _run_amf_base_qaqc(amf_main_py, amf_input_file, site_id, resolution):
    repo_root = os.path.dirname(os.path.dirname(amf_main_py))

    cmd = [
        sys.executable,  # same env as MeaningFlux
        amf_main_py,
        site_id,
        resolution,
        "-t",
        "-fn", amf_input_file,
        "-np",
    ]

    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout or "", proc.stderr or "", cmd, repo_root


def _run_with_site_bypass(amf_main_py, amf_input_file, site_id, resolution, allow_bypass: bool):
    """
    Run AMF-BASE. If stdout contains "Invalid SITE_ID ... exiting." and allow_bypass=True,
    re-run with a placeholder SITE_ID from AMF's dictionary.

    Returns dict with keys:
      rc,out,err,cmd,repo_root,used_site_id,bypassed,bypass_note
    """
    rc, out, err, cmd, repo_root = _run_amf_base_qaqc(amf_main_py, amf_input_file, site_id, resolution)

    if allow_bypass and ("Invalid SITE_ID" in out):
        placeholder = _pick_placeholder_site_id(amf_main_py)
        bypass_note = (
            f"NOTE: AMF-BASE does not recognize SITE_ID '{site_id}'.\n"
            f"      Re-running in TEST MODE using placeholder SITE_ID '{placeholder}'.\n"
            f"      Structural checks remain useful; site-metadata-dependent checks may differ.\n"
        )
        rc2, out2, err2, cmd2, repo_root2 = _run_amf_base_qaqc(amf_main_py, amf_input_file, placeholder, resolution)
        return {
            "rc": rc2, "out": out2, "err": err2, "cmd": cmd2, "repo_root": repo_root2,
            "used_site_id": placeholder, "bypassed": True, "bypass_note": bypass_note
        }

    return {
        "rc": rc, "out": out, "err": err, "cmd": cmd, "repo_root": repo_root,
        "used_site_id": site_id, "bypassed": False, "bypass_note": ""
    }


def classify_amf_result(stdout: str, stderr: str, bypassed: bool):
    """
    PASS / WARN / FAIL
      FAIL: any [ERROR] or Traceback
      WARN: any [WARNING] or bypassed placeholder SITE_ID used
      PASS: none of the above
    """
    errors = []
    warnings = []

    for line in stderr.splitlines():
        if ("[ERROR]" in line) or ("Traceback (most recent call last):" in line):
            errors.append(line)
        elif "[WARNING]" in line:
            warnings.append(line)

    if bypassed:
        warnings.append("SITE_ID not recognized by AMF-BASE — placeholder SITE_ID used for the run.")

    if errors:
        return "FAIL", errors, warnings
    if warnings:
        return "WARN", errors, warnings
    return "PASS", errors, warnings


def summarize_issues(errors, warnings):
    """
    Convert raw AMF messages into concise, readable issues.
    Returns list of dicts: {severity, title, meaning, fix_hint}
    """
    out = []

    def add(sev, title, meaning, fix):
        out.append({"severity": sev, "title": title, "meaning": meaning, "fix_hint": fix})

    # Errors
    for e in errors:
        if "timestamp_format" in e or "Datetime string length" in e:
            add(
                "ERROR",
                "Timestamp format invalid",
                "TIMESTAMP_START / TIMESTAMP_END must be 12 characters: YYYYMMDDHHMM.",
                "Export timestamps to YYYYMMDDHHMM (no dashes/colons/timezones)."
            )
        elif "FileNotFoundError" in e and "Check_messages.txt" in e:
            add(
                "ERROR",
                "Missing AMF message file",
                "AMF-BASE requires Check_messages.txt in the repo root when running locally.",
                "Copy processing/Check_messages.txt into the AMF-BASE repo root."
            )
        else:
            add(
                "ERROR",
                "AMF-BASE reported an error",
                e.strip(),
                "Scroll to the first ERROR line in the raw log and fix it first."
            )

    # Warnings (group)
    unknown_vars = []
    dup_root = []
    other_warn = []

    for w in warnings:
        if "Unknown variable" in w:
            unknown_vars.append(w.strip())
        elif "duplicate root/qualifier" in w or "Found both root variable" in w:
            dup_root.append(w.strip())
        else:
            other_warn.append(w.strip())

    if unknown_vars:
        add(
            "WARN",
            "Non-standard variable names detected",
            f"AMF-BASE flagged {len(unknown_vars)} variable(s) as not in the AmeriFlux dictionary.",
            "Rename/map important columns to AmeriFlux names; drop helper columns like *_QC or DATESTAMP_*."
        )

    if dup_root:
        add(
            "WARN",
            "Duplicate sensor definitions",
            "AMF-BASE detected both a root variable and a qualified version (e.g., SWC and SWC_1_1_1).",
            "Keep either the root or the qualified version (not both)."
        )

    for w in other_warn:
        if "SITE_ID not recognized" in w:
            add(
                "WARN",
                "SITE_ID not recognized by AMF-BASE",
                "MeaningFlux used a placeholder SITE_ID so structural checks could run.",
                "This is OK for formatting checks; final submission should use the correct AmeriFlux SITE_ID."
            )
        else:
            add(
                "WARN",
                "AMF-BASE warning",
                w,
                "Review; may still be acceptable, but fixing reduces submission friction."
            )

    if not out:
        add(
            "INFO",
            "No issues detected",
            "AMF-BASE produced no warnings or errors.",
            "Proceed to submission."
        )

    return out


def status_text_and_color(status):
    if status == "PASS":
        return "✅ READY FOR AMERIFLUX SUBMISSION", "green"
    if status == "WARN":
        return "⚠️ FORMAT OK – REVIEW WARNINGS", "orange"
    return "❌ NOT READY FOR SUBMISSION", "red"


# -------------------------------------------------------------------
# Diagnostics helpers: Plot A and Plot C (plus evidence numbers)
# -------------------------------------------------------------------

def compute_offgrid_fraction(df, expected_step_min):
    """
    Fraction of Δt not equal to expected_step_min (ignores first row and NaT).
    """
    if expected_step_min is None or "TIMESTAMP_START" not in df.columns:
        return None

    ts = pd.to_datetime(df["TIMESTAMP_START"], errors="coerce")
    dt = ts.diff().dt.total_seconds() / 60.0
    dt = dt.dropna()
    dt = dt[dt > 0]
    if dt.empty:
        return 0.0

    off = (dt.round().astype(int) != int(expected_step_min)).mean() * 100.0
    return float(off)


def find_unknown_columns_from_amf_log(stderr_text: str):
    """
    Parse AMF log lines like:
      [WARNING] data_headers - Unknown variable [FCH4_QC] in column 8
    Returns a list of column names (unique, in order).
    """
    unknown = []
    for line in stderr_text.splitlines():
        if "Unknown variable [" in line:
            try:
                part = line.split("Unknown variable [", 1)[1]
                col = part.split("]", 1)[0].strip()
                if col and col not in unknown:
                    unknown.append(col)
            except Exception:
                pass
    return unknown


def build_unknown_columns_table(df, unknown_cols):
    """
    Build a small table for unknown columns:
      name, missing_frac, non_missing_frac
    """
    if not unknown_cols:
        return pd.DataFrame(columns=["col", "missing_frac", "non_missing_frac"])

    rows = []
    for c in unknown_cols:
        if c in df.columns:
            miss = float(df[c].isna().mean())
            rows.append({"col": c, "missing_frac": miss, "non_missing_frac": 1.0 - miss})
        else:
            rows.append({"col": c, "missing_frac": np.nan, "non_missing_frac": np.nan})

    out = pd.DataFrame(rows)
    return out.sort_values(by=["non_missing_frac"], ascending=False)


def make_plot_A_timestamp_continuity(df, expected_step_min=None):
    """
    Plot A: Δt (minutes) over time. Spikes show gaps/jumps/mixed resolution.
    """
    ts = pd.to_datetime(df["TIMESTAMP_START"], errors="coerce")
    dt_min = ts.diff().dt.total_seconds() / 60.0

    fig = plt.Figure(figsize=(7.2, 2.4), dpi=100)
    ax = fig.add_subplot(111)

    ax.plot(ts, dt_min, ".", alpha=0.4)
    ax.set_title("Plot A — Timestamp continuity (Δt in minutes)")
    ax.set_ylabel("Δt (min)")
    ax.set_xlabel("Time")

    if expected_step_min is not None:
        ax.axhline(expected_step_min, linestyle="--", alpha=0.5)

    fig.tight_layout()
    return fig


def make_plot_C_unknown_columns(unknown_df):
    """
    Plot C: Non-missing fraction for unknown/non-standard columns (top 25).
    This is "evidence" for what needs renaming/dropping.
    """
    fig = plt.Figure(figsize=(7.2, 3.0), dpi=100)
    ax = fig.add_subplot(111)

    if unknown_df is None or unknown_df.empty:
        ax.text(0.02, 0.5, "Plot C — No unknown variables detected.", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    d = unknown_df.copy()
    d = d.head(25)

    ax.bar(range(len(d)), d["non_missing_frac"].values)
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels(d["col"].tolist(), rotation=75, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Plot C — Unknown/non-standard columns (non-missing fraction, top 25)")
    ax.set_ylabel("Non-missing fraction")

    fig.tight_layout()
    return fig


# -------------------------------------------------------------------
# Turn signals into "What’s wrong with submission?"
# -------------------------------------------------------------------

def build_submission_diagnosis(status, errors, warnings, off_grid_pct, unknown_df, bypassed, expected_step_min):
    """
    Ranked list of "what is wrong" items:
      level: BLOCKER / WARNING / INFO
      title/why/evidence/fix
    """
    diag = []

    def add(level, title, why, evidence, fix):
        diag.append({"level": level, "title": title, "why": why, "evidence": evidence, "fix": fix})

    # Blockers from AMF errors
    if any(("timestamp_format" in e or "Datetime string length" in e) for e in errors):
        add(
            "BLOCKER",
            "Timestamps are not in AmeriFlux FP-IN format",
            "AmeriFlux FP-IN requires TIMESTAMP_START and TIMESTAMP_END formatted as YYYYMMDDHHMM (12 digits).",
            "AMF-BASE reports timestamp format errors.",
            "Export timestamps as 12-digit strings (YYYYMMDDHHMM)."
        )

    if any(("Check_messages.txt" in e) for e in errors):
        add(
            "BLOCKER",
            "AMF-BASE cannot run locally (missing Check_messages.txt)",
            "This is a local setup problem, not your data. AMF-BASE expects Check_messages.txt in repo root when run locally.",
            "FileNotFoundError: Check_messages.txt",
            "Copy processing/Check_messages.txt into the AMF-BASE repo root (MeaningFlux does this automatically)."
        )

    # SITE_ID bypass
    if bypassed:
        add(
            "WARNING",
            "SITE_ID not recognized by AMF-BASE dictionary (placeholder used)",
            "MeaningFlux ran AMF-BASE using a placeholder SITE_ID so structural checks could run. Site-specific checks may differ.",
            "AMF-BASE stdout: Invalid SITE_ID",
            "For real submission: use the correct AmeriFlux SITE_ID."
        )

    # Off-grid timestep signal
    if off_grid_pct is not None and expected_step_min is not None:
        if off_grid_pct > 5:
            add(
                "WARNING",
                "Many irregular timesteps detected",
                "A large off-grid Δt fraction often indicates mixed resolution concatenation, duplicates, or time parsing issues.",
                f"Off-grid Δt ≈ {off_grid_pct:.1f}% (expected {expected_step_min} min).",
                "Check timestamp generation/merging; remove duplicates; ensure consistent resolution and timezone."
            )
        elif off_grid_pct > 0:
            add(
                "INFO",
                "Minor timestep irregularities (gaps)",
                "Small off-grid Δt is usually just gaps and is common in EC data.",
                f"Off-grid Δt ≈ {off_grid_pct:.1f}% (expected {expected_step_min} min).",
                "Usually OK; review if spikes look like duplicates or merges."
            )

    # Unknown columns
    if unknown_df is not None and not unknown_df.empty:
        high = unknown_df[unknown_df["non_missing_frac"] > 0.2]
        add(
            "WARNING" if len(high) > 0 else "INFO",
            "Non-standard column names detected",
            "AmeriFlux uses a strict variable dictionary. Unknown names often need mapping or removal (e.g., helper QC flags).",
            f"{len(unknown_df)} unknown columns; {len(high)} have >20% non-missing data.",
            "Drop helper columns (e.g., *_QC, DATESTAMP_*) and map real variables to AmeriFlux standard names."
        )

    # Generic fail fallback
    if status == "FAIL" and not any(d["level"] == "BLOCKER" for d in diag):
        add(
            "BLOCKER",
            "AMF-BASE reported a critical error",
            "Fix the first ERROR in the log first; later errors may be cascading.",
            errors[0] if errors else "Unknown error",
            "Scroll to the first [ERROR] line in the raw log and fix it first."
        )

    if not diag:
        add(
            "INFO",
            "No submission problems detected",
            "This local preflight did not find blockers. AmeriFlux will still run official QA/QC upon submission.",
            "PASS (no errors/warnings).",
            "Proceed to submission."
        )

    order = {"BLOCKER": 0, "WARNING": 1, "INFO": 2}
    diag.sort(key=lambda x: order.get(x["level"], 9))
    return diag


# -------------------------------------------------------------------
# Fix-It: safe, non-destructive fixes applied to a copy of df
# -------------------------------------------------------------------

def _fix_sort_by_timestamp(df):
    df2 = df.copy()
    ts = pd.to_datetime(df2["TIMESTAMP_START"], errors="coerce")
    df2["_tmp_ts"] = ts
    df2 = df2.sort_values("_tmp_ts").drop(columns=["_tmp_ts"])
    return df2, "Sorted rows by TIMESTAMP_START."

def _fix_drop_duplicate_timestamps(df):
    df2 = df.copy()
    before = len(df2)
    df2 = df2.drop_duplicates(subset=["TIMESTAMP_START"], keep="first")
    after = len(df2)
    return df2, f"Dropped duplicate TIMESTAMP_START rows: {before-after} removed."

def _fix_drop_helper_columns(df):
    df2 = df.copy()
    drop_cols = [c for c in df2.columns if c.upper().endswith("_QC")]
    for c in ("DATESTAMP_START", "DATESTAMP_END"):
        if c in df2.columns:
            drop_cols.append(c)
    drop_cols = sorted(list(set(drop_cols)))
    if drop_cols:
        df2 = df2.drop(columns=drop_cols, errors="ignore")
        return df2, f"Dropped helper columns: {', '.join(drop_cols)}"
    return df2, "No helper columns found to drop."

def _fix_drop_root_when_qualified_exists(df):
    """
    If both SWC and SWC_* exist, drop SWC.
    Same for TS and TS_*.
    (Very conservative: only these two known common ones.)
    """
    df2 = df.copy()
    changes = []

    def has_qualified(prefix):
        return any(c.startswith(prefix + "_") for c in df2.columns)

    for root in ("SWC", "TS"):
        if root in df2.columns and has_qualified(root):
            df2 = df2.drop(columns=[root], errors="ignore")
            changes.append(root)

    if changes:
        return df2, f"Dropped root variables where qualified versions exist: {', '.join(changes)}"
    return df2, "No root/qualified duplicates (SWC/TS) detected to resolve."

def _fix_replace_neg9999_with_nan(df):
    df2 = df.copy()
    df2 = df2.replace(-9999, np.nan)
    return df2, "Replaced -9999 with NaN (internal working copy only)."


def open_fixit_window(parent, df_current, diagnosis, on_apply_callback):
    """
    Fix-It window: select safe fixes, preview, apply to a COPY of df_current.
    Calls on_apply_callback(df_fixed, summary_text) when user applies.
    """
    win = Toplevel(parent)
    win.title("MeaningFlux – Fix-It (safe preflight fixes)")
    win.geometry("900x650")

    top = Frame(win)
    top.pack(side="top", fill="x", padx=10, pady=10)

    Label(top, text="Fix-It (safe, non-destructive):", font=("Arial", 14, "bold")).pack(anchor="w")
    Label(
        top,
        text=("Select fixes below. MeaningFlux will apply them to a COPY of your dataset and re-run QA/QC.\n"
              "These fixes are conservative (format/cleanup only). Variable mapping and unit conversions are not applied here."),
        justify="left"
    ).pack(anchor="w", pady=(6, 0))

    # Show “what’s wrong” summary
    diag_box = Text(win, wrap="word", height=10)
    diag_box.pack(fill="x", expand=False, padx=10, pady=(0, 10))
    diag_box.insert(END, "What MeaningFlux thinks is wrong (from last run):\n\n")
    for d in diagnosis:
        diag_box.insert(END, f"[{d['level']}] {d['title']}\n")
        diag_box.insert(END, f"  Why: {d['why']}\n")
        diag_box.insert(END, f"  Fix: {d['fix']}\n\n")
    diag_box.config(state="disabled")

    # Fix options (checkboxes)
    opts_frame = Frame(win)
    opts_frame.pack(side="top", fill="x", padx=10)

    fixes = [
        ("Sort by TIMESTAMP_START", _fix_sort_by_timestamp, True),
        ("Drop duplicate TIMESTAMP_START rows", _fix_drop_duplicate_timestamps, True),
        ("Drop helper columns (*_QC, DATESTAMP_*)", _fix_drop_helper_columns, True),
        ("Resolve SWC/TS root vs qualified (drop root)", _fix_drop_root_when_qualified_exists, False),
    ]

    vars_ = []
    for i, (label, _, default_on) in enumerate(fixes):
        v = BooleanVar(value=default_on)
        vars_.append(v)
        Checkbutton(opts_frame, text=label, variable=v).grid(row=i, column=0, sticky="w", pady=2)

    # Preview area
    Label(win, text="Preview of applied changes:", font=("Arial", 11, "bold")).pack(anchor="w", padx=10, pady=(10, 0))
    preview = Text(win, wrap="word", height=10)
    preview.pack(fill="both", expand=True, padx=10, pady=(6, 10))

    def do_preview():
        df_tmp = df_current.copy()
        msgs = []
        for (label, fn, _), v in zip(fixes, vars_):
            if v.get():
                df_tmp, msg = fn(df_tmp)
                msgs.append(f"• {label}: {msg}")
        preview.delete("1.0", END)
        preview.insert(END, "\n".join(msgs) if msgs else "No fixes selected.")
        preview.insert(END, f"\n\nRows: {len(df_current)} → {len(df_tmp)}")
        preview.insert(END, f"\nColumns: {df_current.shape[1]} → {df_tmp.shape[1]}")

    def do_apply():
        df_fixed = df_current.copy()
        msgs = []
        for (label, fn, _), v in zip(fixes, vars_):
            if v.get():
                df_fixed, msg = fn(df_fixed)
                msgs.append(f"• {label}: {msg}")

        summary = "\n".join(msgs) if msgs else "No fixes applied."
        on_apply_callback(df_fixed, summary)
        win.destroy()

    btns = Frame(win)
    btns.pack(side="bottom", fill="x", padx=10, pady=10)

    Button(btns, text="Preview changes", command=do_preview).pack(side="left", padx=(0, 8))
    Button(btns, text="Apply selected fixes", command=do_apply).pack(side="left", padx=(0, 8))
    Button(btns, text="Cancel", command=win.destroy).pack(side="right")

    do_preview()


# -------------------------------------------------------------------
# Report export (HTML)
# -------------------------------------------------------------------

def generate_html_report(
    out_path: str,
    status: str,
    site_id_original: str,
    site_id_used: str,
    resolution: str,
    n_rows: int,
    time_start: str,
    time_end: str,
    bypassed: bool,
    diagnosis: list,
    issues: list
):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    badge_color = {"PASS": "green", "WARN": "orange", "FAIL": "red"}.get(status, "gray")

    # Diagnosis section
    diag_rows = []
    for d in diagnosis:
        diag_rows.append(
            f"<tr><td><b>{d['level']}</b></td><td>{d['title']}</td><td>{d['why']}</td><td>{d['fix']}</td></tr>"
        )
    diag_html = "\n".join(diag_rows) if diag_rows else ""

    # Issues section (from summarize_issues)
    issue_rows = []
    for it in issues:
        issue_rows.append(
            f"<tr><td><b>{it['severity']}</b></td><td>{it['title']}</td><td>{it['meaning']}</td><td>{it['fix_hint']}</td></tr>"
        )
    issues_html = "\n".join(issue_rows) if issue_rows else ""

    bypass_html = ""
    if bypassed:
        bypass_html = (
            "<p style='color:#b36b00;'><b>Note:</b> SITE_ID was not recognized by AMF-BASE. "
            "MeaningFlux re-ran AMF-BASE using a placeholder SITE_ID so structural checks could run. "
            "Site-metadata-dependent checks may differ from the final AmeriFlux submission.</p>"
        )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>MeaningFlux AmeriFlux QA/QC Preflight Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 36px; color: #222; }}
    h1 {{ margin-bottom: 6px; }}
    .badge {{
      display: inline-block; padding: 6px 10px; border-radius: 6px;
      color: white; background: {badge_color}; font-weight: 700;
    }}
    .meta {{ color: #555; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
    th, td {{ border: 1px solid #ddd; padding: 10px; vertical-align: top; }}
    th {{ background: #f5f5f5; }}
    .small {{ font-size: 0.93em; color: #555; }}
  </style>
</head>
<body>

<h1>AmeriFlux QA/QC Preflight Report</h1>
<div class="meta small">Generated: {now}</div>

<p><span class="badge">{status}</span></p>

{bypass_html}

<h2>Dataset summary</h2>
<ul>
  <li><b>Original SITE_ID:</b> {site_id_original}</li>
  <li><b>SITE_ID used for QA/QC:</b> {site_id_used}</li>
  <li><b>Resolution:</b> {resolution}</li>
  <li><b>Records checked:</b> {n_rows}</li>
  <li><b>Time range:</b> {time_start} → {time_end}</li>
</ul>

<h2>What’s wrong (MeaningFlux diagnosis)</h2>
<table>
  <tr><th style="width:12%;">Level</th><th style="width:20%;">Problem</th><th style="width:38%;">Why it matters</th><th style="width:30%;">Fix</th></tr>
  {diag_html}
</table>

<h2>AMF-BASE summarized issues</h2>
<table>
  <tr><th style="width:10%;">Severity</th><th style="width:20%;">Issue</th><th style="width:40%;">Meaning</th><th style="width:30%;">Suggested fix</th></tr>
  {issues_html}
</table>

<p class="small" style="margin-top: 28px;">
Generated by MeaningFlux using the official AMF-BASE-QAQC pipeline (test mode).
MeaningFlux provides a user-friendly preflight check; it does not replace AmeriFlux submission QA/QC.
</p>

</body>
</html>
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


# -------------------------------------------------------------------
# GUI ENTRY POINT
# -------------------------------------------------------------------

def calc_data_AMF_BASE_QAQC(df, inputname_site):
    """
    GUI to run AMF-BASE-QAQC locally (test mode), then show:
      - clear status banner
      - "What's wrong with my submission?" diagnosis box
      - top issues (human-readable)
      - Plot A + Plot C (optional; evidence)
      - raw AMF logs (debug)
      - Fix-It window (apply safe fixes, then re-run)
      - export HTML report
    """
    if df is None or df.empty:
        messagebox.showwarning("Warning", "Load EC data first.")
        return

    # Working copy: treat -9999 as missing for plotting + fix tools
    df_work = df.copy()
    df_work.replace(-9999, np.nan, inplace=True)

    # Infer site + resolution
    site_id_guess = _infer_site_id_from_filename(inputname_site)
    resolution, dt_minutes = _infer_resolution_from_df(df_work)

    try:
        amf_main_py = ensure_amf_base_available()
    except RuntimeError as e:
        messagebox.showerror("AMF-BASE-QAQC unavailable", str(e))
        return

    # Window
    win = Toplevel()
    win.title("MeaningFlux – AmeriFlux QA/QC (AMF-BASE Preflight)")
    win.geometry("1200x860")

    # -------------------------
    # Top summary panel
    # -------------------------
    top = Frame(win)
    top.pack(side="top", fill="x", padx=10, pady=10)

    status_label = Label(top, text="Not run yet", font=("Arial", 16, "bold"))
    status_label.grid(row=0, column=0, columnspan=6, sticky="w", pady=(0, 8))

    Label(top, text="SITE_ID (from filename):").grid(row=1, column=0, sticky="w")
    Label(top, text=site_id_guess, font=("TkDefaultFont", 10, "bold")).grid(row=1, column=1, sticky="w", padx=6)

    Label(top, text="Resolution (inferred):").grid(row=1, column=2, sticky="w", padx=(20, 0))
    res_txt = resolution + (f" (~{dt_minutes:.1f} min)" if dt_minutes is not None else "")
    Label(top, text=res_txt, font=("TkDefaultFont", 10, "bold")).grid(row=1, column=3, sticky="w", padx=6)

    allow_bypass_var = BooleanVar(value=True)
    Checkbutton(top, text="Allow SITE_ID bypass (placeholder for structural checks)", variable=allow_bypass_var).grid(
        row=1, column=4, sticky="w", padx=(20, 0)
    )

    show_plots_var = BooleanVar(value=True)
    Checkbutton(top, text="Show plots (A and C)", variable=show_plots_var).grid(
        row=1, column=5, sticky="w", padx=(20, 0)
    )

    info_label = Label(
        top,
        text=("Click “Run QA/QC” to export an AmeriFlux FP-IN file and run AMF-BASE in TEST MODE.\n"
              "MeaningFlux will explain what is wrong (if anything), show evidence, and help apply safe fixes."),
        justify="left"
    )
    info_label.grid(row=2, column=0, columnspan=6, sticky="w", pady=(8, 0))

    # -------------------------
    # Middle layout
    # -------------------------
    mid = Frame(win)
    mid.pack(side="top", fill="both", expand=True, padx=10, pady=(0, 10))

    left = Frame(mid)
    left.pack(side="left", fill="both", expand=True, padx=(0, 8))

    right = Frame(mid)
    right.pack(side="left", fill="both", expand=True, padx=(8, 0))

    # Left: Diagnosis + Issues + Logs
    Label(left, text="What’s wrong with my submission? (MeaningFlux diagnosis)", font=("Arial", 11, "bold")).pack(anchor="w")
    diagnosis_box = Text(left, wrap="word", height=12)
    diagnosis_box.pack(fill="x", expand=False, pady=(6, 10))

    Label(left, text="Top issues (AMF-BASE summarized):", font=("Arial", 11, "bold")).pack(anchor="w")
    issues_box = Text(left, wrap="word", height=12)
    issues_box.pack(fill="both", expand=True, pady=(6, 10))

    Label(left, text="Raw AMF-BASE output (debug):", font=("Arial", 11, "bold")).pack(anchor="w")
    log_box = Text(left, wrap="word", height=14)
    log_box.pack(fill="both", expand=True, pady=(6, 0))

    # Right: Plots (optional)
    Label(right, text="Evidence (plots):", font=("Arial", 11, "bold")).pack(anchor="w")
    plots_frame = Frame(right)
    plots_frame.pack(fill="both", expand=True, pady=(6, 0))

    # Save last run info for report export + fix-it
    last_run = {
        "status": None,
        "issues": None,
        "diagnosis": None,
        "bypassed": False,
        "site_id_original": site_id_guess,
        "site_id_used": site_id_guess,
        "resolution": resolution,
        "n_rows": None,
        "time_start": None,
        "time_end": None,
        "expected_step_min": None,
        "off_grid_pct": None,
        "unknown_df": None,
        "df_work": df_work,  # current working df (may be updated by Fix-It)
        "last_fix_summary": ""
    }

    def _render_banner(status):
        title, color = status_text_and_color(status)
        status_label.config(text=title, fg=color)

    def _render_diagnosis(diagnosis, bypass_note=""):
        diagnosis_box.delete("1.0", END)

        if bypass_note:
            diagnosis_box.insert(END, bypass_note + "\n\n")

        for d in diagnosis:
            diagnosis_box.insert(END, f"[{d['level']}] {d['title']}\n")
            diagnosis_box.insert(END, f"  Why: {d['why']}\n")
            diagnosis_box.insert(END, f"  Evidence: {d['evidence']}\n")
            diagnosis_box.insert(END, f"  Fix: {d['fix']}\n\n")

        if last_run["last_fix_summary"]:
            diagnosis_box.insert(END, "Fix-It changes applied (last time):\n")
            diagnosis_box.insert(END, last_run["last_fix_summary"] + "\n")

    def _render_issues(status, issues):
        issues_box.delete("1.0", END)

        if status == "PASS":
            issues_box.insert(END, "✅ Your file passed this local preflight.\n\n")
        elif status == "WARN":
            issues_box.insert(END, "⚠️ Your file is structurally valid, but warnings were detected.\n\n")
        else:
            issues_box.insert(END, "❌ Critical issues were detected.\n\n")

        for it in issues:
            sev = it["severity"]
            issues_box.insert(END, f"[{sev}] {it['title']}\n")
            issues_box.insert(END, f"  • Meaning: {it['meaning']}\n")
            issues_box.insert(END, f"  • Fix: {it['fix_hint']}\n\n")

    def _render_plots(df_for_plots, expected_step_min, unknown_df):
        # Clear
        for child in plots_frame.winfo_children():
            child.destroy()

        if not show_plots_var.get():
            Label(plots_frame, text="Plots are hidden (toggle 'Show plots' to display).").pack(anchor="w")
            return

        # Plot A
        if "TIMESTAMP_START" in df_for_plots.columns:
            figA = make_plot_A_timestamp_continuity(df_for_plots, expected_step_min=expected_step_min)
            canvasA = FigureCanvasTkAgg(figA, master=plots_frame)
            canvasA.draw()
            canvasA.get_tk_widget().pack(fill="x", expand=False, pady=(0, 12))
        else:
            Label(plots_frame, text="Plot A unavailable: TIMESTAMP_START not found.").pack(anchor="w")

        # Plot C
        figC = make_plot_C_unknown_columns(unknown_df)
        canvasC = FigureCanvasTkAgg(figC, master=plots_frame)
        canvasC.draw()
        canvasC.get_tk_widget().pack(fill="both", expand=True)

    def run_qaqc():
        try:
            df_current = last_run["df_work"]

            with tempfile.TemporaryDirectory(prefix="meaningflux_fp_in_") as tmpdir:
                # 1) Export FP-IN
                amf_input, step_min = _export_df_to_amf_fp_in(df_current, site_id_guess, resolution, tmpdir)

                preview_df = pd.read_csv(amf_input, nrows=1)
                preview_ts = preview_df["TIMESTAMP_START"].iloc[0] if "TIMESTAMP_START" in preview_df.columns else "N/A"

                # 2) Run AMF (optionally bypass SITE_ID)
                run = _run_with_site_bypass(
                    amf_main_py, amf_input, site_id_guess, resolution,
                    allow_bypass=bool(allow_bypass_var.get())
                )

                rc = run["rc"]
                out = run["out"]
                err = run["err"]
                cmd = run["cmd"]
                repo_root = run["repo_root"]
                used_site = run["used_site_id"]
                bypassed = run["bypassed"]
                bypass_note = run["bypass_note"]

                cfg_path = os.path.join(repo_root, "qaqc.cfg")

                # 3) Classify results
                status, errors, warnings = classify_amf_result(out, err, bypassed=bypassed)
                issues = summarize_issues(errors, warnings)

                # 4) Evidence numbers for diagnosis
                off_grid_pct = compute_offgrid_fraction(df_current, expected_step_min=step_min)

                unknown_cols = find_unknown_columns_from_amf_log(err)
                unknown_df = build_unknown_columns_table(df_current, unknown_cols)

                diagnosis = build_submission_diagnosis(
                    status=status,
                    errors=errors,
                    warnings=warnings,
                    off_grid_pct=off_grid_pct,
                    unknown_df=unknown_df,
                    bypassed=bypassed,
                    expected_step_min=step_min
                )

                # 5) Render banner + text
                _render_banner(status)
                _render_diagnosis(diagnosis, bypass_note=bypass_note)
                _render_issues(status, issues)

                # 6) Raw logs
                log_box.delete("1.0", END)
                log_box.insert(
                    END,
                    f"AMF-BASE Repo Root:\n  {repo_root}\n"
                    f"Config used:\n  {cfg_path}\n\n"
                    f"Command:\n  {' '.join(cmd)}\n\n"
                    f"Input File:\n  {amf_input}\n"
                    f"Preview TIMESTAMP_START[0]: {preview_ts}\n\n"
                    f"Return Code: {rc}\n\n"
                )
                if out:
                    log_box.insert(END, "──── STDOUT ────\n" + out + "\n")
                if err:
                    log_box.insert(END, "──── STDERR ────\n" + err + "\n")

                # 7) Plots
                _render_plots(df_current, expected_step_min=step_min, unknown_df=unknown_df)

                # 8) Save last run info for Fix-It + report export
                ts_start = pd.to_datetime(df_current["TIMESTAMP_START"], errors="coerce") if "TIMESTAMP_START" in df_current.columns else pd.Series([], dtype="datetime64[ns]")
                last_run.update({
                    "status": status,
                    "issues": issues,
                    "diagnosis": diagnosis,
                    "bypassed": bypassed,
                    "site_id_original": site_id_guess,
                    "site_id_used": used_site,
                    "resolution": resolution,
                    "n_rows": int(len(df_current)),
                    "time_start": str(ts_start.min()) if (len(ts_start) and not ts_start.isna().all()) else "N/A",
                    "time_end": str(ts_start.max()) if (len(ts_start) and not ts_start.isna().all()) else "N/A",
                    "expected_step_min": step_min,
                    "off_grid_pct": off_grid_pct,
                    "unknown_df": unknown_df,
                })

        except Exception as e:
            messagebox.showerror("Error running AMF-BASE-QAQC", str(e))

    def open_fixit():
        if not last_run["diagnosis"]:
            messagebox.showwarning("Fix-It not ready", "Run QA/QC first so MeaningFlux can detect issues.")
            return

        def on_apply(df_fixed, summary_text):
            # Update working df and note
            last_run["df_work"] = df_fixed
            last_run["last_fix_summary"] = summary_text
            # auto re-run after apply
            run_qaqc()

        open_fixit_window(win, last_run["df_work"], last_run["diagnosis"], on_apply_callback=on_apply)

    def export_report():
        if not last_run["status"] or not last_run["issues"] or not last_run["diagnosis"]:
            messagebox.showwarning("Report not available", "Run QA/QC first, then export the report.")
            return

        default_name = f"MeaningFlux_QAQC_Report_{last_run['site_id_original']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        out_path = filedialog.asksaveasfilename(
            title="Save QA/QC Report (HTML)",
            defaultextension=".html",
            initialfile=default_name,
            filetypes=[("HTML files", "*.html"), ("All files", "*.*")]
        )
        if not out_path:
            return

        try:
            generate_html_report(
                out_path=out_path,
                status=last_run["status"],
                site_id_original=last_run["site_id_original"],
                site_id_used=last_run["site_id_used"],
                resolution=last_run["resolution"],
                n_rows=last_run["n_rows"],
                time_start=last_run["time_start"],
                time_end=last_run["time_end"],
                bypassed=last_run["bypassed"],
                diagnosis=last_run["diagnosis"],
                issues=last_run["issues"]
            )
            messagebox.showinfo("Report saved", f"Saved:\n{out_path}")
        except Exception as e:
            messagebox.showerror("Report export failed", str(e))

    # -------------------------
    # Buttons
    # -------------------------
    buttons = Frame(win)
    buttons.pack(side="bottom", fill="x", padx=10, pady=10)

    Button(buttons, text="Run QA/QC (AMF-BASE Preflight)", command=run_qaqc).pack(side="left", padx=(0, 8))
    Button(buttons, text="Open Fix-It (apply safe fixes)", command=open_fixit).pack(side="left", padx=(0, 8))
    Button(buttons, text="Export Report (HTML)", command=export_report).pack(side="left", padx=(0, 8))
    Button(buttons, text="Close", command=win.destroy).pack(side="right")
