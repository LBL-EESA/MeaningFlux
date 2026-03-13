#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MeaningFlux — Information Theory Toolbox (EC-friendly, ML-consistent)
Author: Leila C. Hernandez (LBNL) + assistant
Updated: 2025-11-21

Key points:
- Uses the same DataFrame philosophy as the ML toolbox:
  * We trust the incoming df (already cleaned upstream).
  * We do NOT pre-filter columns as "numeric only".
  * When a variable is selected, we convert that column to numeric on-the-fly
    with pd.to_numeric(errors='coerce') and drop NaNs.

Features:
- Entropy H(X)
- Mutual Information I(X;Y) (histogram or KDE-based TIP)
- Conditional Mutual Information I(X;Y|Z)
- Lagged Mutual Information I(X_t;Y_{t+lag})
- PID (two sources -> one target):
    * histogram-based min-information PID
    * KDE-based TIP decomposition (Goodwell & Kumar style)
- Transfer Entropy TE(X->Y) (simple discrete, first-order)
- TE Network (matrix + heatmap, optional permutation test)

IMPORTANT: The public entry point is flexible:
  - Pattern A (MeaningFlux main app):
        open_information_theory_toolbox(df, inputname_site)
  - Pattern B (standalone demo / other GUIs):
        open_information_theory_toolbox(parent_widget, df)
"""

from __future__ import annotations

from typing import Iterable, List, Dict, Tuple, Optional, Union
import math
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import pandas as pd

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# =============================================================================
# Generic helpers
# =============================================================================

ArrayLike = Union[np.ndarray, Iterable[float]]

_it_window: Optional[tk.Toplevel] = None  # single window like ML toolbox


def _to_1d_array(x: ArrayLike) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim > 1:
        x = x.reshape(-1)
    return x


def _validate_same_length(*arrays: ArrayLike) -> List[np.ndarray]:
    arrs = [_to_1d_array(a) for a in arrays]
    lengths = {len(a) for a in arrs}
    if len(lengths) > 1:
        raise ValueError("All input arrays must have the same length.")
    return arrs


def _remove_nan(*arrays: np.ndarray) -> List[np.ndarray]:
    if not arrays:
        return []
    mask = np.ones_like(arrays[0], dtype=bool)
    for a in arrays:
        mask &= ~np.isnan(a)
    return [a[mask] for a in arrays]


def _align_dropna(*arrs: np.ndarray) -> List[np.ndarray]:
    """
    Trim arrays to common length and remove positions with any NaN.
    """
    if not arrs:
        return []
    L = min(len(a) for a in arrs)
    arrs = [np.asarray(a, dtype=float).reshape(-1)[:L] for a in arrs]
    mask = np.ones(L, dtype=bool)
    for a in arrs:
        mask &= ~np.isnan(a)
    return [a[mask] for a in arrs]


# =============================================================================
# Discretization
# =============================================================================

def discretize_equal_width(
    x: ArrayLike,
    n_bins: int = 10,
    return_edges: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Equal-width discretization. NaNs -> -1.
    """
    x = _to_1d_array(x)
    mask = ~np.isnan(x)
    x_valid = x[mask]

    if x_valid.size == 0:
        raise ValueError("Cannot discretize: all values are NaN.")

    xmin, xmax = np.min(x_valid), np.max(x_valid)
    if np.isclose(xmin, xmax):
        edges = np.array([xmin - 0.5, xmax + 0.5])
        labels = np.zeros_like(x, dtype=int)
    else:
        edges = np.linspace(xmin, xmax, n_bins + 1)
        labels = np.full_like(x, fill_value=-1, dtype=int)
        lab_valid = np.digitize(x_valid, edges[1:-1], right=False)
        lab_valid = np.clip(lab_valid, 0, n_bins - 1)
        labels[mask] = lab_valid

    if return_edges:
        return labels, edges
    return labels


def discretize_equal_frequency(
    x: ArrayLike,
    n_bins: int = 10,
    return_edges: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Equal-frequency (quantile) discretization. NaNs -> -1.
    """
    x = _to_1d_array(x)
    mask = ~np.isnan(x)
    x_valid = x[mask]

    if x_valid.size == 0:
        raise ValueError("Cannot discretize: all values are NaN.")

    qs = np.linspace(0, 100, n_bins + 1)
    edges = np.nanpercentile(x_valid, qs)
    edges = np.unique(edges)

    if edges.size < 2:
        edges = np.array([np.min(x_valid) - 0.5, np.max(x_valid) + 0.5])

    n_bins_eff = edges.size - 1
    labels = np.full_like(x, fill_value=-1, dtype=int)
    lab_valid = np.digitize(x_valid, edges[1:-1], right=False)
    lab_valid = np.clip(lab_valid, 0, n_bins_eff - 1)
    labels[mask] = lab_valid

    if return_edges:
        return labels, edges
    return labels


# =============================================================================
# Discrete entropy & mutual information
# =============================================================================

def _pmf_from_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    if labels.size == 0:
        raise ValueError("Cannot compute PMF: empty array.")
    max_label = labels.max()
    if max_label < 0:
        raise ValueError("Labels must be >= 0.")
    counts = np.bincount(labels, minlength=max_label + 1).astype(float)
    total = counts.sum()
    if total <= 0:
        raise ValueError("Counts sum to zero.")
    return counts / total


def entropy_discrete(labels: ArrayLike, base: float = 2.0) -> float:
    labels = _to_1d_array(labels)
    (labels,) = _remove_nan(labels)
    pmf = _pmf_from_labels(labels)
    pmf = pmf[pmf > 0]
    log_p = np.log(pmf) / np.log(base)
    return float(-np.sum(pmf * log_p))


def joint_entropy_discrete(
    x_labels: ArrayLike,
    y_labels: ArrayLike,
    base: float = 2.0,
) -> float:
    x, y = _validate_same_length(x_labels, y_labels)
    x, y = _remove_nan(x, y)
    x = x.astype(int)
    y = y.astype(int)

    if x.size == 0:
        raise ValueError("No valid samples after removing NaNs.")

    ny = int(y.max()) + 1
    joint_index = x * ny + y
    pmf = _pmf_from_labels(joint_index)
    pmf = pmf[pmf > 0]
    log_p = np.log(pmf) / np.log(base)
    return float(-np.sum(pmf * log_p))


def mutual_information_discrete(
    x_labels: ArrayLike,
    y_labels: ArrayLike,
    base: float = 2.0,
) -> float:
    x, y = _validate_same_length(x_labels, y_labels)
    x, y = _remove_nan(x, y)
    Hx = entropy_discrete(x, base=base)
    Hy = entropy_discrete(y, base=base)
    Hxy = joint_entropy_discrete(x, y, base=base)
    return float(Hx + Hy - Hxy)


def conditional_entropy_discrete(
    y_labels: ArrayLike,
    z_labels: ArrayLike,
    base: float = 2.0,
) -> float:
    y, z = _validate_same_length(y_labels, z_labels)
    y, z = _remove_nan(y, z)
    y = y.astype(int)
    z = z.astype(int)

    if y.size == 0:
        raise ValueError("No valid samples after removing NaNs.")

    z_unique, z_codes = np.unique(z, return_inverse=True)
    K = z_unique.size
    N = y.size
    H = 0.0
    base_log = np.log(base)

    for k in range(K):
        mask = (z_codes == k)
        yk = y[mask]
        if yk.size == 0:
            continue
        pmf_k = _pmf_from_labels(yk)
        pmf_k = pmf_k[pmf_k > 0]
        log_p_k = np.log(pmf_k) / base_log
        H_k = -np.sum(pmf_k * log_p_k)
        H += (yk.size / N) * H_k

    return float(H)


def conditional_mutual_information_discrete(
    x_labels: ArrayLike,
    y_labels: ArrayLike,
    z_labels: ArrayLike,
    base: float = 2.0,
) -> float:
    """
    I(X;Y|Z) = H(X|Z) + H(Y|Z) - H(X,Y|Z)
    """
    x, y, z = _validate_same_length(x_labels, y_labels, z_labels)
    x, y, z = _remove_nan(x, y, z)
    x = x.astype(int)
    y = y.astype(int)
    z = z.astype(int)

    if x.size == 0:
        raise ValueError("No valid samples after removing NaNs.")

    def H_given_Z(u: np.ndarray, zarr: np.ndarray) -> float:
        return conditional_entropy_discrete(u, zarr, base=base)

    ny = int(y.max()) + 1 if y.size > 0 else 1
    xy = x.astype(int) * ny + y.astype(int)

    Hx_z = H_given_Z(x, z)
    Hy_z = H_given_Z(y, z)
    Hxy_z = H_given_Z(xy, z)
    return float(Hx_z + Hy_z - Hxy_z)


# =============================================================================
# Continuous wrappers
# =============================================================================

def mutual_information(
    x: ArrayLike,
    y: ArrayLike,
    base: float = 2.0,
    method: str = "hist",
    n_bins: int = 10,
    disc: str = "equal_freq",
) -> float:
    if method != "hist":
        raise NotImplementedError("Only 'hist' method is implemented here.")

    x, y = _validate_same_length(x, y)
    x, y = _remove_nan(x, y)

    if disc == "equal_width":
        x_lab = discretize_equal_width(x, n_bins=n_bins)
        y_lab = discretize_equal_width(y, n_bins=n_bins)
    elif disc == "equal_freq":
        x_lab = discretize_equal_frequency(x, n_bins=n_bins)
        y_lab = discretize_equal_frequency(y, n_bins=n_bins)
    else:
        raise ValueError("disc must be 'equal_width' or 'equal_freq'.")

    return mutual_information_discrete(x_lab, y_lab, base=base)


def lagged_mutual_information(
    x: ArrayLike,
    y: ArrayLike,
    lags: Iterable[int],
    base: float = 2.0,
    method: str = "hist",
    n_bins: int = 10,
    disc: str = "equal_freq",
) -> Dict[int, float]:
    x = _to_1d_array(x)
    y = _to_1d_array(y)
    if x.size != y.size:
        raise ValueError("x and y must have the same length.")

    mi_by_lag: Dict[int, float] = {}
    for lag in lags:
        lag = int(lag)
        if lag > 0:
            x_l = x[:-lag]
            y_l = y[lag:]
        elif lag < 0:
            lag_abs = -lag
            x_l = x[lag_abs:]
            y_l = y[:-lag_abs]
        else:
            x_l = x.copy()
            y_l = y.copy()

        if x_l.size < 2:
            mi_by_lag[lag] = float("nan")
            continue

        mi_by_lag[lag] = mutual_information(
            x_l, y_l,
            base=base,
            method=method,
            n_bins=n_bins,
            disc=disc,
        )

    return mi_by_lag


# =============================================================================
# PID: min-information decomposition
# =============================================================================

def pid_min_information(
    x1: ArrayLike,
    x2: ArrayLike,
    y: ArrayLike,
    base: float = 2.0,
    method: str = "hist",
    n_bins: int = 10,
    disc: str = "equal_freq",
) -> Dict[str, float]:
    x1, x2, y = _validate_same_length(x1, x2, y)
    x1, x2, y = _remove_nan(x1, x2, y)

    I_x1_y = mutual_information(
        x1, y,
        base=base,
        method=method,
        n_bins=n_bins,
        disc=disc,
    )
    I_x2_y = mutual_information(
        x2, y,
        base=base,
        method=method,
        n_bins=n_bins,
        disc=disc,
    )

    if disc == "equal_width":
        x1_lab = discretize_equal_width(x1, n_bins=n_bins)
        x2_lab = discretize_equal_width(x2, n_bins=n_bins)
        y_lab = discretize_equal_width(y, n_bins=n_bins)
    else:
        x1_lab = discretize_equal_frequency(x1, n_bins=n_bins)
        x2_lab = discretize_equal_frequency(x2, n_bins=n_bins)
        y_lab = discretize_equal_frequency(y, n_bins=n_bins)

    nx2 = int(x2_lab.max()) + 1 if x2_lab.size > 0 else 1
    x12_lab = x1_lab.astype(int) * nx2 + x2_lab.astype(int)

    I_x12_y = mutual_information_discrete(x12_lab, y_lab, base=base)

    R = min(I_x1_y, I_x2_y)
    U1 = I_x1_y - R
    U2 = I_x2_y - R
    S = I_x12_y - R - U1 - U2

    return {
        "redundant": float(R),
        "unique_x1": float(U1),
        "unique_x2": float(U2),
        "synergy": float(S),
        "I_x1_y": float(I_x1_y),
        "I_x2_y": float(I_x2_y),
        "I_x1x2_y": float(I_x12_y),
        "base": float(base),
    }


# =============================================================================
# Transfer entropy (simple discrete, first-order)
# =============================================================================

def transfer_entropy(
    x: ArrayLike,
    y: ArrayLike,
    lag: int = 1,
    base: float = 2.0,
    n_bins: int = 10,
    disc: str = "equal_freq",
) -> float:
    """
    T_{X->Y} = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)
    Simple discrete TE with one-step past.
    """
    x = _to_1d_array(x)
    y = _to_1d_array(y)
    if x.size != y.size:
        raise ValueError("x and y must have the same length.")
    if lag < 1:
        raise ValueError("lag must be >= 1.")

    x_past = x[:-lag]
    y_past = y[:-lag]
    y_future = y[lag:]
    x_past, y_past, y_future = _remove_nan(x_past, y_past, y_future)

    if y_future.size < 2:
        return float("nan")

    if disc == "equal_width":
        x_p = discretize_equal_width(x_past, n_bins=n_bins)
        y_p = discretize_equal_width(y_past, n_bins=n_bins)
        y_f = discretize_equal_width(y_future, n_bins=n_bins)
    else:
        x_p = discretize_equal_frequency(x_past, n_bins=n_bins)
        y_p = discretize_equal_frequency(y_past, n_bins=n_bins)
        y_f = discretize_equal_frequency(y_future, n_bins=n_bins)

    H_yf_given_yp = conditional_entropy_discrete(y_f, y_p, base=base)

    nx = int(x_p.max()) + 1 if x_p.size > 0 else 1
    yp_xp = y_p.astype(int) * nx + x_p.astype(int)
    H_yf_given_yp_xp = conditional_entropy_discrete(y_f, yp_xp, base=base)

    te = H_yf_given_yp - H_yf_given_yp_xp
    return float(max(0.0, te))


# =============================================================================
# KDE / TIP backend (Goodwell & Kumar)
# =============================================================================

def calc_info_measures(pdf: np.ndarray) -> Dict[str, float]:
    """
    Compute information measures from a given PDF (1D, 2D, 3D)
    following Goodwell & Kumar (2017) TIP framework.
    """
    dim = len(pdf.shape)
    v = pdf.shape

    if np.sum(np.array(v) == 1) > 0:
        dim -= 1

    N = pdf.shape[0]
    info: Dict[str, float] = {}

    # 1D: entropy
    if dim == 1:
        Hvect = pdf * np.log2(1.0 / pdf)
        Hvect[np.isnan(Hvect)] = 0
        Hx = float(np.sum(Hvect))
        info["Hx"] = Hx

    # 2D: H(X), H(Y), H(X|Y), H(Y|X), I(X;Y)
    if dim == 2:
        H_xgy = 0.0
        H_ygx = 0.0

        m_i = np.sum(pdf, axis=1)
        Hivect = m_i * np.log2(1.0 / m_i)
        Hivect[np.isnan(Hivect)] = 0
        Hx = float(np.sum(Hivect))

        m_j = np.sum(pdf, axis=0)
        Hjvect = m_j * np.log2(1.0 / m_j)
        Hjvect[np.isnan(Hjvect)] = 0
        Hy = float(np.sum(Hjvect))

        for i in range(N):
            for j in range(N):
                m_ij = pdf[i, j]
                mj = m_j[j]
                mi = m_i[i]

                if m_ij > 0 and mi > 0:
                    H_ygx += m_ij * np.log2(mi / m_ij)
                if m_ij > 0 and mj > 0:
                    H_xgy += m_ij * np.log2(mj / m_ij)

        info["Hx1"] = Hx
        info["Hx2"] = Hy
        info["H_xgy"] = H_xgy
        info["H_ygx"] = H_ygx
        info["I"] = min(Hx - H_xgy, Hy - H_ygx)

    # 3D: TIP decomposition
    if dim == 3:
        I_x1y = 0.0
        I_x2y = 0.0
        I_x1x2 = 0.0
        T = 0.0

        m_jk = np.sum(pdf, axis=0)
        m_ij = np.sum(pdf, axis=2)
        m_ik = np.sum(pdf, axis=1)

        m_i = np.sum(m_ij, axis=1)
        m_j = np.sum(m_ij, axis=0)
        m_k = np.sum(m_jk, axis=0)

        Hivect = m_i * np.log2(1.0 / m_i)
        Hivect[np.isnan(Hivect)] = 0
        Hx1 = float(np.sum(Hivect))

        Hjvect = m_j * np.log2(1.0 / m_j)
        Hjvect[np.isnan(Hjvect)] = 0
        Hx2 = float(np.sum(Hjvect))

        Hkvect = m_k * np.log2(1.0 / m_k)
        Hkvect[np.isnan(Hkvect)] = 0
        Hy = float(np.sum(Hkvect))

        eps = np.finfo(float).eps

        for i in range(N):
            for j in range(N):
                for k in range(N):
                    m_ijk = pdf[i, j, k]

                    if (
                        m_ijk > eps
                        and m_ij[i, j] > eps
                        and m_jk[j, k] > eps
                        and m_j[j] > eps
                    ):
                        T_add = m_ijk * np.log2(
                            (m_ijk * m_j[j]) / (m_ij[i, j] * m_jk[j, k])
                        )
                        if T_add > 0:
                            T += T_add

                    if j == 1:
                        if m_ik[i, k] > eps and m_i[i] > eps and m_k[k] > eps:
                            I_tau_add = m_ik[i, k] * np.log2(
                                m_ik[i, k] / (m_i[i] * m_k[k])
                            )
                            if I_tau_add > 0:
                                I_x1y += I_tau_add

                    if i == 1:
                        if m_jk[j, k] > eps and m_j[j] > eps and m_k[k] > eps:
                            I_tau2_add = m_jk[j, k] * np.log2(
                                m_jk[j, k] / (m_j[j] * m_k[k])
                            )
                            if I_tau2_add > 0:
                                I_x2y += I_tau2_add

                    if k == 1:
                        if m_ij[i, j] > eps and m_i[i] > eps and m_j[j] > eps:
                            I_tau3_add = m_ij[i, j] * np.log2(
                                m_ij[i, j] / (m_i[i] * m_j[j])
                            )
                            if I_tau3_add > 0:
                                I_x1x2 += I_tau3_add

        dI = T - I_x1y
        I_tot = dI + I_x1y + I_x2y

        I_sourcenorm = I_x1x2 / min(Hx1, Hx2) if min(Hx1, Hx2) > 0 else 0.0
        if np.isnan(I_sourcenorm):
            I_sourcenorm = 0.0

        Rmax = min(I_x1y, I_x2y)
        Rmin = max(0.0, I_x1y + I_x2y - I_tot)
        dR = Rmax - Rmin
        R = Rmin + dR * I_sourcenorm

        U1 = I_x1y - R
        U2 = I_x2y - R
        S = I_tot - (U1 + U2 + R)

        info["Hx1"] = Hx1
        info["Hx2"] = Hx2
        info["Hy"] = Hy
        info["I_x1y"] = I_x1y
        info["I_x2y"] = I_x2y
        info["T"] = T
        info["dI"] = dI
        info["Itot"] = I_tot
        info["R"] = R
        info["S"] = S
        info["U1"] = U1
        info["U2"] = U2

    return info


def compute_pdfGUI(
    Data: np.ndarray,
    N: int,
    bin_scheme: str,
    Range: np.ndarray,
    method: str,
    zeffect,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 1D/2D/3D PDFs using KDE or histogram binning.
    Range must be shape (2, dim): Range[0,i]=min, Range[1,i]=max.
    """
    Data = np.asarray(Data, dtype=float)
    nTup, dim = Data.shape

    if dim == 1:
        pdf = np.zeros((N,))
    elif dim == 2:
        pdf = np.zeros((N, N))
    elif dim == 3:
        pdf = np.zeros((N, N, N))
    else:
        raise ValueError("compute_pdfGUI only supports dim = 1, 2, or 3.")

    Coords = np.zeros((dim, N))
    Edges = np.zeros((dim, N + 1))
    xo = np.zeros(dim)

    for i in range(dim):
        if bin_scheme == "local":
            lo = np.nanmin(Data[:, i])
            hi = np.nanmax(Data[:, i])
        elif bin_scheme == "global":
            lo = Range[0, i]
            hi = Range[1, i]
        else:
            raise ValueError("bin_scheme must be 'local' or 'global'.")

        if np.isclose(lo, hi):
            lo -= 1e-6
            hi += 1e-6

        Edges[i, :] = np.linspace(lo - 1e-8, hi, N + 1)
        xo[i] = lo
        Coords[i, :] = (Edges[i, :-1] + Edges[i, 1:]) / 2.0
        Coords[i, 0] = 0.0

    delta = Coords[:, -1] - Coords[:, -2]

    if method == "KDE":
        var = np.var(Data, axis=0)
        ptp = np.ptp(Data, axis=0)
        ptp[ptp == 0] = 1e-6

        h1D = 1.06 * (nTup ** (-1.0 / 5.0)) * var / ptp
        h2D = 1.77 * (nTup ** (-1.0 / 6.0)) * var / ptp
        h3D = 2.78 * (nTup ** (-1.0 / 7.0)) * var / ptp

        if dim == 1:
            h = h1D
        elif dim == 2:
            h = h2D
        else:
            h = h3D

        pdfcenter = np.zeros_like(pdf)

        for n in range(nTup):
            dat = Data[n, :]

            minind = np.maximum(
                np.floor((dat - xo - h) / delta).astype(int) - 1, 0
            )
            maxind = np.minimum(
                np.ceil((dat - xo + h) / delta).astype(int) + 1, N - 1
            )

            ICoords = Coords[0, minind[0]: maxind[0] + 1]
            if dim > 1:
                JCoords = Coords[1, minind[1]: maxind[1] + 1]
                if dim == 3:
                    KCoords = Coords[2, minind[2]: maxind[2] + 1]

            if dim == 1:
                pdfcenter[minind[0]: maxind[0] + 1] += np.exp(
                    -0.5 * ((ICoords - dat[0]) / h[0]) ** 2
                ) / (h[0] * np.sqrt(2 * np.pi))

            elif dim == 2:
                grid = np.array(np.meshgrid(ICoords, JCoords)).T.reshape(-1, 2)
                vals = np.exp(-0.5 * np.sum(((grid - dat) / h) ** 2, axis=1))
                vals = vals.reshape(
                    maxind[0] - minind[0] + 1,
                    maxind[1] - minind[1] + 1,
                )
                pdfcenter[
                    minind[0]: maxind[0] + 1,
                    minind[1]: maxind[1] + 1,
                ] += vals / (np.prod(h) * (2 * np.pi) ** (dim / 2.0))

            elif dim == 3:
                grid = np.array(np.meshgrid(ICoords, JCoords, KCoords)).T.reshape(
                    -1, 3
                )
                vals = np.exp(-0.5 * np.sum(((grid - dat) / h) ** 2, axis=1))
                vals = vals.reshape(
                    maxind[0] - minind[0] + 1,
                    maxind[1] - minind[1] + 1,
                    maxind[2] - minind[2] + 1,
                )
                pdfcenter[
                    minind[0]: maxind[0] + 1,
                    minind[1]: maxind[1] + 1,
                    minind[2]: maxind[2] + 1,
                ] += vals / (np.prod(h) * (2 * np.pi) ** (dim / 2.0))

        pdf = pdfcenter / np.sum(pdfcenter)

    else:
        BinData = np.zeros_like(Data, dtype=int)
        for i in range(dim):
            edges = Edges[i]
            b = np.digitize(Data[:, i], bins=edges) - 1
            b[b == N] = N - 1
            BinData[:, i] = b

        rng = [(0, N)] * dim
        pdf, _ = np.histogramdd(BinData, bins=N, range=rng, density=True)

    return pdf, Coords


def kde_mi_2d(
    x: np.ndarray,
    y: np.ndarray,
    N: int = 50,
    bin_scheme: str = "global",
    method: str = "KDE",
) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    L = min(len(x), len(y))
    x, y = x[:L], y[:L]

    Range = np.zeros((2, 2))
    Range[0, 0], Range[1, 0] = np.nanmin(x), np.nanmax(x)
    Range[0, 1], Range[1, 1] = np.nanmin(y), np.nanmax(y)

    Data = np.column_stack([x, y])
    zeffect = [0, 0]

    pdf, _ = compute_pdfGUI(Data, N, bin_scheme, Range, method, zeffect)
    info = calc_info_measures(pdf)
    return float(info.get("I", np.nan))


def kde_tip_pid_3d(
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    N: int = 30,
    bin_scheme: str = "global",
    method: str = "KDE",
) -> Dict[str, float]:
    x1 = np.asarray(x1, dtype=float).reshape(-1)
    x2 = np.asarray(x2, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    L = min(len(x1), len(x2), len(y))
    x1, x2, y = x1[:L], x2[:L], y[:L]

    Range = np.zeros((2, 3))
    Range[0, 0], Range[1, 0] = np.nanmin(x1), np.nanmax(x1)
    Range[0, 1], Range[1, 1] = np.nanmin(x2), np.nanmax(x2)
    Range[0, 2], Range[1, 2] = np.nanmin(y), np.nanmax(y)

    Data = np.column_stack([x1, x2, y])
    zeffect = [0, 0, 0]

    pdf, _ = compute_pdfGUI(Data, N, bin_scheme, Range, method, zeffect)
    info = calc_info_measures(pdf)
    return info


# =============================================================================
# Permutation test
# =============================================================================

def permutation_test(
    stat_fn,
    x: np.ndarray,
    y: np.ndarray,
    n_perm: int = 200,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    x, y = _align_dropna(x, y)

    stat_obs = float(stat_fn(x, y))
    count = 0
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        s = float(stat_fn(x, y_perm))
        if s >= stat_obs:
            count += 1
    p = (count + 1) / (n_perm + 1)
    return stat_obs, p


# =============================================================================
# Tkinter GUI (flexible entry point)
# =============================================================================

def open_information_theory_toolbox(
    arg1,
    arg2=None,
    title: str = "Information Theory Toolbox",
) -> Optional[tk.Toplevel]:
    """
    Flexible public entry point.

    Pattern A (MeaningFlux main app, like ML toolbox):
        open_information_theory_toolbox(df, inputname_site)

    Pattern B (standalone / other GUI):
        open_information_theory_toolbox(parent_widget, df)

    Parameters
    ----------
    arg1 : pandas.DataFrame or tk.Misc
        Either the DataFrame (Pattern A) or the parent widget (Pattern B).
    arg2 : str or pandas.DataFrame or None
        Either the site name (Pattern A) or the DataFrame (Pattern B).
    title : str
        Base window title (site name is appended if available).
    """
    global _it_window

    # Detect calling pattern
    parent: Optional[tk.Misc] = None
    df = None
    site_name = ""

    if isinstance(arg1, pd.DataFrame):
        # Pattern A: (df, site_name)
        df = arg1
        site_name = str(arg2) if arg2 is not None else ""
        parent = tk._default_root  # may be None; Toplevel can still be created
    else:
        # Pattern B: (parent, df)
        parent = arg1
        df = arg2
        site_name = ""

    # Coerce df into DataFrame
    if df is None:
        messagebox.showwarning("Warning", "Load the data first.")
        return None

    # If df is a string, try CSV path; otherwise attempt DataFrame()
    if isinstance(df, str):
        import os
        if os.path.isfile(df):
            try:
                df = pd.read_csv(df)
            except Exception as e:
                messagebox.showerror(
                    "Error",
                    f"Could not read data from file:\n{df}\n\n{e}",
                )
                return None
        else:
            messagebox.showerror(
                "Error",
                "The Information Theory Toolbox received a string instead of a "
                "data table, and it is not a valid file path.\n\n"
                "Please open/load a dataset in MeaningFlux first."
            )
            return None

    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Data must be convertible to a pandas DataFrame.\n"
                f"Got type: {type(df)}\n\n{e}",
            )
            return None

    if df.empty:
        messagebox.showwarning("Warning", "DataFrame is empty.")
        return None

    # Deduplicated column names for UI (same pattern as ML toolbox)
    all_cols = list(dict.fromkeys(df.columns))
    if not all_cols:
        messagebox.showerror("Error", "No columns available in the DataFrame.")
        return None

    # Single-window behavior (like ML toolbox)
    if _it_window is not None and tk.Toplevel.winfo_exists(_it_window):
        try:
            _it_window.lift()
        except Exception:
            pass
        return _it_window

    # ------------------------------------------------------------------
    # GUI layout
    # ------------------------------------------------------------------
    if parent is not None and isinstance(parent, tk.Misc):
        win = tk.Toplevel(parent)
    else:
        win = tk.Toplevel()

    _it_window = win

    full_title = "MeaningFlux — Information Theory"
    if site_name:
        full_title += f" · {site_name}"
    win.title(full_title)
    win.geometry("1180x760")
    win.minsize(980, 620)

    left = ttk.Frame(win, padding=10)
    left.grid(row=0, column=0, sticky="nsw")
    right = ttk.Frame(win, padding=10)
    right.grid(row=0, column=1, sticky="nsew")

    win.columnconfigure(1, weight=1)
    win.rowconfigure(0, weight=1)

    # --- Controls (left) ---
    ttk.Label(left, text="Measure", font=("", 10, "bold")).grid(
        row=0, column=0, sticky="w"
    )
    measure_var = tk.StringVar(value="Mutual Information I(X;Y)")
    measure_cb = ttk.Combobox(
        left,
        textvariable=measure_var,
        state="readonly",
        values=[
            "Entropy H(X)",
            "Mutual Information I(X;Y)",
            "Conditional MI I(X;Y|Z)",
            "Lagged MI I(X_t;Y_{t+lag})",
            "PID (X1,X2→Y)",
            "Transfer Entropy TE(X→Y)",
            "TE Network",
        ],
    )
    measure_cb.grid(row=1, column=0, sticky="ew", pady=(0, 8))

    ttk.Label(left, text="X / Driver 1 / Source").grid(row=2, column=0, sticky="w")
    x_var = tk.StringVar(value=all_cols[0])
    x_cb = ttk.Combobox(left, textvariable=x_var, values=all_cols, state="readonly")
    x_cb.grid(row=3, column=0, sticky="ew")

    ttk.Label(left, text="Y / Driver 2 / Target").grid(row=4, column=0, sticky="w")
    y_var = tk.StringVar(value=all_cols[1] if len(all_cols) > 1 else all_cols[0])
    y_cb = ttk.Combobox(left, textvariable=y_var, values=all_cols, state="readonly")
    y_cb.grid(row=5, column=0, sticky="ew")

    ttk.Label(left, text="Z / Target (for CMI & PID)").grid(row=6, column=0, sticky="w")
    z_var = tk.StringVar(value="")
    z_cb = ttk.Combobox(left, textvariable=z_var, values=[""] + all_cols, state="readonly")
    z_cb.grid(row=7, column=0, sticky="ew", pady=(0, 8))

    params = ttk.LabelFrame(left, text="Parameters")
    params.grid(row=8, column=0, sticky="ew", pady=(4, 8))

    ttk.Label(params, text="Estimator").grid(row=0, column=0, sticky="w")
    est_var = tk.StringVar(value="hist")
    est_cb = ttk.Combobox(
        params,
        textvariable=est_var,
        state="readonly",
        values=["hist", "kde-tip"],
    )
    est_cb.grid(row=0, column=1, sticky="ew", padx=(6, 0))

    ttk.Label(params, text="# bins / KDE N").grid(row=1, column=0, sticky="w")
    bins_var = tk.IntVar(value=10)
    ttk.Spinbox(
        params, from_=2, to=128, textvariable=bins_var, width=6
    ).grid(row=1, column=1, sticky="w", padx=(6, 0))

    ttk.Label(params, text="Discretizer (for hist)").grid(row=2, column=0, sticky="w")
    disc_var = tk.StringVar(value="freq")
    ttk.Combobox(
        params,
        textvariable=disc_var,
        state="readonly",
        values=["freq", "width"],
    ).grid(row=2, column=1, sticky="ew", padx=(6, 0))

    ttk.Label(params, text="Max |lag| (lagged MI)").grid(row=3, column=0, sticky="w")
    maxlag_var = tk.IntVar(value=24)
    ttk.Spinbox(
        params, from_=0, to=200, textvariable=maxlag_var, width=6
    ).grid(row=3, column=1, sticky="w", padx=(6, 0))

    ttk.Label(params, text="TE lag").grid(row=4, column=0, sticky="w")
    delay_var = tk.IntVar(value=1)
    ttk.Spinbox(
        params, from_=1, to=96, textvariable=delay_var, width=6
    ).grid(row=4, column=1, sticky="w", padx=(6, 0))

    ttk.Label(params, text="Permutations (TE / network)").grid(
        row=5, column=0, sticky="w"
    )
    perm_var = tk.IntVar(value=0)
    ttk.Spinbox(
        params, from_=0, to=5000, increment=50, textvariable=perm_var, width=6
    ).grid(row=5, column=1, sticky="w", padx=(6, 0))

    ttk.Label(left, text="Network variables (TE Network)").grid(
        row=9, column=0, sticky="w"
    )
    search_var = tk.StringVar()
    search_entry = ttk.Entry(left, textvariable=search_var)
    search_entry.grid(row=10, column=0, sticky="ew", pady=(0, 4))

    net_list = tk.Listbox(
        left, selectmode=tk.EXTENDED, height=10, exportselection=False
    )
    net_list.grid(row=11, column=0, sticky="nsew")
    left.rowconfigure(11, weight=1)

    for c in all_cols:
        net_list.insert(tk.END, c)

    def _filter_net(*_):
        q = search_var.get().strip().lower()
        net_list.delete(0, tk.END)
        for c in all_cols:
            if q in c.lower():
                net_list.insert(tk.END, c)

    search_var.trace_add("write", _filter_net)

    run_btn = ttk.Button(left, text="Run Analysis")
    run_btn.grid(row=12, column=0, sticky="ew", pady=(8, 4))
    
    saveplot_btn = ttk.Button(left, text="Save Plot", state=tk.NORMAL)
    saveplot_btn.grid(row=14, column=0, sticky="ew", pady=(8, 0))

    save_btn = ttk.Button(left, text="Save TE Network Matrix", state=tk.DISABLED)
    save_btn.grid(row=13, column=0, sticky="ew")

    # --- Results area (right) ---
    right_top = ttk.Frame(right)
    right_top.grid(row=0, column=0, sticky="new")
    right_bottom = ttk.Frame(right)
    right_bottom.grid(row=1, column=0, sticky="nsew", pady=(8, 0))

    right.rowconfigure(1, weight=1)
    right.columnconfigure(0, weight=1)

    ttk.Label(right_top, text="Results", font=("", 10, "bold")).grid(
        row=0, column=0, sticky="w"
    )
    txt = tk.Text(right_top, height=10, wrap="word")
    txt.grid(row=1, column=0, sticky="ew")
    right_top.columnconfigure(0, weight=1)

    fig = Figure(figsize=(7.5, 4.6), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_title("Plot")
    canvas = FigureCanvasTkAgg(fig, master=right_bottom)
    canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    right_bottom.rowconfigure(0, weight=1)
    right_bottom.columnconfigure(0, weight=1)

    last_matrix: Optional[np.ndarray] = None
    last_vars: Optional[List[str]] = None
    cbar = None  # track current colorbar so we can remove it on the next run


    # --- inner helpers (using df directly, ML-style) ---

    def _show(msg: str):
        txt.delete("1.0", tk.END)
        txt.insert(tk.END, msg)

    def _discretize(x: np.ndarray) -> np.ndarray:
        n_bins = int(bins_var.get())
        if disc_var.get() == "freq":
            return discretize_equal_frequency(x, n_bins=n_bins)
        return discretize_equal_width(x, n_bins=n_bins)

    def _col_as_numeric(col_name: str) -> np.ndarray:
        s = pd.to_numeric(df[col_name], errors="coerce")
        arr = s.to_numpy(dtype=float)
        (arr,) = _align_dropna(arr)
        if arr.size == 0:
            raise ValueError(
                f"Column '{col_name}' has no numeric values after conversion.\n"
                "Select a different variable or check your data."
            )
        return arr

    def _get_xy() -> Tuple[np.ndarray, np.ndarray, str, str]:
        xn, yn = x_var.get(), y_var.get()
        x = _col_as_numeric(xn)
        y = _col_as_numeric(yn)
        x, y = _align_dropna(x, y)
        if x.size == 0:
            raise ValueError(
                f"No overlapping numeric samples for {xn} and {yn} after removing NaNs."
            )
        return x, y, xn, yn

    def _get_xyz() -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, str, str]:
        xn, yn, zn = x_var.get(), y_var.get(), z_var.get()
        if not zn:
            raise ValueError("Select Z for this measure (target variable).")
        x = _col_as_numeric(xn)
        y = _col_as_numeric(yn)
        z = _col_as_numeric(zn)
        x, y, z = _align_dropna(x, y, z)
        if x.size == 0:
            raise ValueError(
                f"No overlapping numeric samples for {xn}, {yn}, {zn} "
                "after removing NaNs."
            )
        return x, y, z, xn, yn, zn

    # --- Run analysis ---

    def run_analysis():
        nonlocal last_matrix, last_vars, cbar, ax

        m = measure_var.get()
        n_bins = int(bins_var.get())
        disc = "equal_freq" if disc_var.get() == "freq" else "equal_width"
        nperm = int(perm_var.get())
        estimator = est_var.get()

        try:
            # --- hard reset of figure + axes + colorbar each run ---
            fig.clf()                     # remove ALL axes (main + old colorbars)
            ax = fig.add_subplot(111)     # new fresh axes taking full space
            cbar = None                   # forget any previous colorbar handle
            # -------------------------------------------------------

            if m == "Entropy H(X)":
                xn = x_var.get()
                x = _col_as_numeric(xn)
                x_lab = _discretize(x)
                H = entropy_discrete(x_lab, base=2.0)
                _show(f"H({xn}) ≈ {H:.6f} bits\n")

                ax.hist(x, bins=max(10, n_bins))
                ax.set_title(f"Histogram: {xn}")
                ax.set_xlabel(xn)
                ax.set_ylabel("Count")
                canvas.draw_idle()

            elif m == "Mutual Information I(X;Y)":
                x, y, xn, yn = _get_xy()
                if estimator == "kde-tip":
                    mi = kde_mi_2d(x, y, N=n_bins, bin_scheme="global", method="KDE")
                    used = f"kde-tip (N={n_bins})"
                else:
                    mi = mutual_information(
                        x, y,
                        base=2.0,
                        method="hist",
                        n_bins=n_bins,
                        disc=disc,
                    )
                    used = f"hist (bins={n_bins}, {disc})"

                _show(f"I({xn}; {yn}) ≈ {mi:.6f} bits  [{used}]\n")

                ax.cla()
                ax.scatter(x, y, s=6, alpha=0.6)
                ax.set_title(f"Scatter: {xn} vs {yn}")
                ax.set_xlabel(xn)
                ax.set_ylabel(yn)
                canvas.draw_idle()

            elif m == "Conditional MI I(X;Y|Z)":
                x, y, z, xn, yn, zn = _get_xyz()
                xi = _discretize(x)
                yi = _discretize(y)
                zi = _discretize(z)
                cmi = conditional_mutual_information_discrete(xi, yi, zi, base=2.0)
                _show(
                    f"I({xn}; {yn} | {zn}) ≈ {cmi:.6f} bits "
                    f"[discrete, bins={n_bins}, {disc}]\n"
                )

                ax.cla()
                ax.scatter(x, y, s=6, alpha=0.6)
                ax.set_title(f"Pairwise view: {xn} vs {yn}")
                ax.set_xlabel(xn)
                ax.set_ylabel(yn)
                canvas.draw_idle()

            elif m == "Lagged MI I(X_t;Y_{t+lag})":
                x, y, xn, yn = _get_xy()
                max_lag = int(maxlag_var.get())
                if max_lag <= 0:
                    raise ValueError("Max |lag| must be > 0.")
                lags = range(-max_lag, max_lag + 1)

                mi_by_lag = lagged_mutual_information(
                    x, y,
                    lags=lags,
                    base=2.0,
                    method="hist",
                    n_bins=n_bins,
                    disc=disc,
                )

                lines = [
                    f"Lagged MI I({xn}_t; {yn}_{{t+lag}}) in bits, "
                    f"bins={n_bins}, {disc}:"
                ]
                for L in sorted(mi_by_lag.keys()):
                    val = mi_by_lag[L]
                    if np.isnan(val):
                        lines.append(f"  lag={L}: nan")
                    else:
                        lines.append(f"  lag={L}: {val:.6f}")
                _show("\n".join(lines) + "\n")

                ax.cla()
                Ls = sorted(mi_by_lag.keys())
                vals = [mi_by_lag[L] for L in Ls]
                ax.plot(Ls, vals, marker="o")
                ax.axvline(0, linestyle="--", linewidth=1)
                ax.set_xlabel("Lag (time steps)")
                ax.set_ylabel("I(X_t; Y_{t+lag}) [bits]")
                ax.set_title(f"Lagged MI: {xn} → {yn}")
                canvas.draw_idle()

            elif m == "PID (X1,X2→Y)":
                x, y, z, xn, yn, zn = _get_xyz()
                if estimator == "kde-tip":
                    info_tip = kde_tip_pid_3d(
                        x, y, z,
                        N=n_bins,
                        bin_scheme="global",
                        method="KDE",
                    )
                    R = info_tip["R"]
                    S = info_tip["S"]
                    U1 = info_tip["U1"]
                    U2 = info_tip["U2"]
                    label_source = "TIP / KDE"
                else:
                    res = pid_min_information(
                        x, y, z,
                        base=2.0,
                        method="hist",
                        n_bins=n_bins,
                        disc=disc,
                    )
                    R = res["redundant"]
                    S = res["synergy"]
                    U1 = res["unique_x1"]
                    U2 = res["unique_x2"]
                    label_source = "Min-information PID (hist)"

                lines = [
                    f"PID for drivers ({xn}, {yn}) → target {zn}",
                    f"Estimator: {label_source}",
                    f"Redundant = {R:.6f}",
                    f"Unique_{xn} = {U1:.6f}",
                    f"Unique_{yn} = {U2:.6f}",
                    f"Synergy = {S:.6f}",
                ]
                _show("\n".join(lines) + "\n")

                ax.cla()
                labels = ["Redundant", f"Unique {xn}", f"Unique {yn}", "Synergy"]
                vals = [R, U1, U2, S]
                idx = np.arange(len(labels))
                ax.bar(idx, vals)
                ax.set_xticks(idx)
                ax.set_xticklabels(labels, rotation=20, ha="right")
                ax.set_ylabel("Bits")
                ax.set_title(f"PID: {xn}, {yn} → {zn}")
                canvas.draw_idle()

            elif m == "Transfer Entropy TE(X→Y)":
                x, y, xn, yn = _get_xy()
                lag = int(delay_var.get())
                if lag < 1:
                    raise ValueError("TE lag must be ≥ 1.")

                te_val = transfer_entropy(
                    x, y,
                    lag=lag,
                    base=2.0,
                    n_bins=n_bins,
                    disc=disc,
                )
                msg = (
                    f"TE({xn}→{yn}) [lag={lag}] ≈ {te_val:.6f} bits "
                    f"[discrete, bins={n_bins}, {disc}]\n"
                )

                if nperm > 0:
                    _, p = permutation_test(
                        lambda a, b: transfer_entropy(
                            a, b,
                            lag=lag,
                            base=2.0,
                            n_bins=n_bins,
                            disc=disc,
                        ),
                        x, y,
                        n_perm=nperm,
                    )
                    msg += f"Permutation p≈{p:.4f} (n={nperm})\n"

                _show(msg)

                ax.cla()
                ax.plot(y, label=yn, lw=1.0)
                ax.plot(x, label=xn, lw=1.0, alpha=0.7)
                ax.set_title(f"Series: target {yn} vs source {xn}")
                ax.set_xlabel("Index")
                ax.legend()
                canvas.draw_idle()

            elif m == "TE Network":
                sel = [net_list.get(i) for i in net_list.curselection()]
                vars_ = sel if sel else all_cols
                if len(vars_) < 2:
                    messagebox.showwarning(
                        "Need variables", "Select at least two variables."
                    )
                    return

                # Convert selected columns to numeric arrays
                data = df[list(vars_)].apply(pd.to_numeric, errors="coerce")
                n, p = data.shape
                mat = np.zeros((p, p), dtype=float)
                pmat = np.ones((p, p), dtype=float)
                lag = int(delay_var.get())
                if lag < 1:
                    raise ValueError("TE lag must be ≥ 1.")

                for j in range(p):
                    for i in range(p):
                        if i == j:
                            continue
                        xi = data.iloc[:, i].to_numpy(dtype=float)
                        yj = data.iloc[:, j].to_numpy(dtype=float)
                        xi, yj = _align_dropna(xi, yj)
                        if len(xi) < lag + 2:
                            mat[i, j] = np.nan
                            pmat[i, j] = 1.0
                            continue

                        te_ij = transfer_entropy(
                            xi, yj,
                            lag=lag,
                            base=2.0,
                            n_bins=n_bins,
                            disc=disc,
                        )
                        mat[i, j] = te_ij
                        if nperm > 0:
                            _, pval = permutation_test(
                                lambda a, b: transfer_entropy(
                                    a, b,
                                    lag=lag,
                                    base=2.0,
                                    n_bins=n_bins,
                                    disc=disc,
                                ),
                                xi, yj,
                                n_perm=nperm,
                            )
                            pmat[i, j] = pval

                last_matrix = mat.copy()
                last_vars = list(vars_)

                _show(
                    f"TE network computed. shape={mat.shape}, "
                    f"lag={lag}, bins={n_bins}, {disc}, permutations={nperm}.\n"
                )

                ax.cla()
                im = ax.imshow(mat, aspect="auto", origin="upper")
                ax.set_xticks(range(p))
                ax.set_yticks(range(p))
                ax.set_xticklabels(vars_, rotation=30, ha="right")
                ax.set_yticklabels(vars_)
                ax.set_xlabel("Target")
                ax.set_ylabel("Source")
                ax.set_title("TE Network (bits)")
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                canvas.draw_idle()

                save_btn.configure(state=tk.NORMAL)

            else:
                messagebox.showerror("Unknown measure", m)

        except Exception as e:
            messagebox.showerror("Analysis error", str(e))

    run_btn.configure(command=run_analysis)

    def save_matrix_csv():
        nonlocal last_matrix, last_vars
        if last_matrix is None or last_vars is None:
            messagebox.showinfo(
                "Nothing to save", "Run a TE Network first to create a matrix."
            )
            return
        fp = filedialog.asksaveasfilename(
            title="Save TE matrix CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if not fp:
            return
        try:
            pd.DataFrame(last_matrix, index=last_vars, columns=last_vars).to_csv(fp)
            messagebox.showinfo("Saved", f"Saved matrix to:\n{fp}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

    save_btn.configure(command=save_matrix_csv)
    
    def save_plot():
        fp = filedialog.asksaveasfilename(
            title="Save Plot",
            defaultextension=".png",
            filetypes=[
                ("PNG Image", "*.png"),
                ("PDF Document", "*.pdf"),
                ("All Files", "*.*"),
            ],
        )
        if not fp:
            return
        try:
            fig.savefig(fp, dpi=300, bbox_inches="tight")
            messagebox.showinfo("Saved", f"Plot saved to:\n{fp}")
        except Exception as e:
            messagebox.showerror("Save error", f"Could not save plot:\n{e}")

    saveplot_btn.configure(command=save_plot)


    def update_states(*_):
        m = measure_var.get()
        x_cb.configure(state=tk.NORMAL)
        y_cb.configure(state=tk.NORMAL)
        z_cb.configure(state=tk.NORMAL)
        save_btn.configure(state=tk.DISABLED)

        if m == "Entropy H(X)":
            y_cb.configure(state=tk.DISABLED)
            z_cb.configure(state=tk.DISABLED)
        elif m == "Mutual Information I(X;Y)":
            z_cb.configure(state=tk.DISABLED)
        elif m == "Conditional MI I(X;Y|Z)":
            pass
        elif m == "Lagged MI I(X_t;Y_{t+lag})":
            z_cb.configure(state=tk.DISABLED)
        elif m == "PID (X1,X2→Y)":
            pass
        elif m == "Transfer Entropy TE(X→Y)":
            z_cb.configure(state=tk.DISABLED)
        elif m == "TE Network":
            pass

    measure_cb.bind("<<ComboboxSelected>>", update_states)
    update_states()

    def _on_close():
        global _it_window
        try:
            win.destroy()
        finally:
            _it_window = None

    win.protocol("WM_DELETE_WINDOW", _on_close)
    return win


# Standalone demo (optional)
if __name__ == "__main__":
    root = tk.Tk()
    root.title("MeaningFlux Demo – IT Toolbox")
    n = 1000
    df_demo = pd.DataFrame({
        "FC": np.random.randn(n),
        "LE": np.random.randn(n) + 0.5,
        "VPD": np.random.rand(n) * 3.0,
        "SWC": np.random.rand(n),
        "TIMESTAMP_START": pd.date_range("2020-01-01", periods=n, freq="30min"),
        "STRING_COL": ["foo"] * n,  # non-numeric just to test robustness
    })
    ttk.Button(
        root,
        text="Open Information Theory Toolbox\n(pattern A: df, site_name)",
        command=lambda: open_information_theory_toolbox(df_demo, "DemoSite"),
    ).pack(padx=20, pady=(20, 10))
    ttk.Button(
        root,
        text="Open Information Theory Toolbox\n(pattern B: parent, df)",
        command=lambda: open_information_theory_toolbox(root, df_demo),
    ).pack(padx=20, pady=(0, 20))
    root.mainloop()



