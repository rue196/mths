#!/usr/bin/env python3
"""
elliptic_dual_regression.py

A dual‑axis linear regression that replaces ordinary least squares
with an **elliptic algebraic / transcendental split**.

Model
-----
Given data  (x_n, y_n),  n = 1 … N:

    algebraic side   u_n = 1 / x_n         (Möbius / inverse map)
                     v_n = 1 / y_n
                     fit  v ≈ a·u + b      → (a, b)

    transcendental   p_n = x_n^(i)          (imaginary plane)
                     q_n = y_n^(i)
                     residual captured by  (p, q)

    elliptic bridge  α = 1/(π − e),  projection Π ∈ [0, 1]

    final prediction ŷ(x) = 1 / (a·(1/x) + b)  +  Π · correction(x)

The projection Π is computed from the angle of the elliptic point
z_n = x_n + i y_n with periods ω₁ = π and ω₂ = e, bounded in [0,1]
(Figure 3.5 in the supply‑chain paper shows this curve).

The fit quality is measured by the **supertrace** of the residuals,
its entropy, and the invariant mass m = |S| · e^(−H) — the same
geometric‑thermodynamic quantities used in the supply‑chain model.
"""

from __future__ import annotations

import math
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ============================================================
#  Constants
# ============================================================
PI    = math.pi
E     = math.e
ALPHA = 1.0 / (PI - E)             # ≈ 2.362
W1    = PI                         # first period
W2    = E                          # second period


# ============================================================
#  Elliptic projection (bounded weight ∈ [0,1])
#  "Figure 3.5" — the linear/non-linear scaling curve
# ============================================================
def elliptic_projection(x: float, y: float) -> float:
    """
    Bounded weight from the spinor projection on the torus
        ℂ / (πℤ + eℤ).

    We use the fundamental parallelogram map:
        u = x / π       (mod 1)
        v = y / e       (mod 1)
    then a smooth positive kernel:
        Π(u, v) = (1 + cos(2π u)·cos(2π v)) / 2     ∈ [0, 1]
    """
    u = (x / W1) % 1.0
    v = (y / W2) % 1.0
    return 0.5 * (1.0 + math.cos(2 * math.pi * u) *
                        math.cos(2 * math.pi * v))


def elliptic_projection_vec(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    u = (x / W1) % 1.0
    v = (y / W2) % 1.0
    return 0.5 * (1.0 + np.cos(2 * np.pi * u) * np.cos(2 * np.pi * v))


# ============================================================
#  Algebraic side — inverse (Möbius) coordinates
# ============================================================
def inverse_coords(x: np.ndarray, y: np.ndarray, eps: float = 1e-12
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """u = 1/x,  v = 1/y  (with a small floor to avoid 1/0)."""
    u = 1.0 / np.maximum(np.abs(x), eps) * np.sign(x)
    v = 1.0 / np.maximum(np.abs(y), eps) * np.sign(y)
    return u, v


def algebraic_fit(u: np.ndarray, v: np.ndarray) -> Tuple[float, float]:
    """Ordinary least squares in (u, v) space.  v ≈ a·u + b."""
    A = np.vstack([u, np.ones_like(u)]).T
    a, b = np.linalg.lstsq(A, v, rcond=None)[0]
    return float(a), float(b)


def inverse_map(x: np.ndarray, a: float, b: float, eps: float = 1e-12
                ) -> np.ndarray:
    """Invert the algebraic fit:  ŷ_alg(x) = 1 / (a·(1/x) + b)."""
    u = 1.0 / np.maximum(np.abs(x), eps) * np.sign(x)
    denom = a * u + b
    return 1.0 / np.where(np.abs(denom) < eps, eps * np.sign(denom), denom)


# ============================================================
#  Transcendental side — imaginary plane (p, q)
# ============================================================
def transcendental_coords(x: np.ndarray, y: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    p_n = x_n^i,  q_n = y_n^i  (principal branch).

    For x > 0 this is exp(i · log x); for x < 0 we use |x| to
    stay on the principal sheet.
    """
    lx = np.log(np.maximum(np.abs(x), 1e-12))
    ly = np.log(np.maximum(np.abs(y), 1e-12))
    p = np.exp(1j * lx)                      # complex
    q = np.exp(1j * ly)
    return p, q


def transcendental_correction(x: np.ndarray,
                              p: np.ndarray,
                              q: np.ndarray,
                              alpha: float = ALPHA) -> np.ndarray:
    """
    Non‑linear correction in the imaginary plane:

        Δ(x) = α · Im( q̄ · p ) / (1 + |p|²)

    It is bounded and vanishes when the imaginary parts align,
    which happens exactly at the elliptic projection maximum.
    """
    num = np.imag(np.conj(q) * p)
    den = 1.0 + np.abs(p) ** 2
    return alpha * num / den


# ============================================================
#  Supertrace / entropy / mass of the residuals
# ============================================================
def supertrace(residuals: np.ndarray) -> float:
    S = 0.0
    for k, r in enumerate(residuals):
        sign = 1.0 if (k % 2 == 0) else -1.0
        S += sign * abs(r)
    return float(S)


def entropy(S: float, N: int, alpha: float = ALPHA) -> float:
    if N <= 0 or S == 0.0:
        return 0.0
    p = abs(S) / N
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -alpha * p * math.log(p)


def invariant_mass(S: float, N: int) -> float:
    H = entropy(S, N)
    return abs(S) * math.exp(-H) if H < 700 else 0.0


# ============================================================
#  Elliptic dual regression
# ============================================================
@dataclass
class EllipticRegressionResult:
    a: float                       # algebraic slope
    b: float                       # algebraic intercept
    pred_alg: np.ndarray           # algebraic prediction
    pred_final: np.ndarray         # algebraic + Π · transcendental
    projection: np.ndarray         # Π_k ∈ [0, 1]
    correction: np.ndarray         # Δ(x_k)
    residuals_alg: np.ndarray
    residuals_final: np.ndarray
    S_alg: float
    S_final: float
    H_alg: float
    H_final: float
    m_alg: float
    m_final: float
    rmse_alg: float
    rmse_final: float


def elliptic_dual_regression(x: np.ndarray,
                             y: np.ndarray,
                             alpha: float = ALPHA
                             ) -> EllipticRegressionResult:
    """
    Two‑axis regression:

        algebraic side (inverse / Möbius coords)
            u = 1/x, v = 1/y,  fit  v ≈ a·u + b
        transcendental side (imaginary plane)
            p = x^i,  q = y^i, correction Δ(x) = α·Im(q̄ p)/(1+|p|²)
        projection
            Π(x, y) ∈ [0,1]  → bounded linear/non-linear mix
    """
    # --- algebraic side ---
    u, v = inverse_coords(x, y)
    a, b = algebraic_fit(u, v)
    pred_alg = inverse_map(x, a, b)

    # --- transcendental side ---
    p, q = transcendental_coords(x, y)
    corr = transcendental_correction(x, p, q, alpha)

    # --- elliptic projection (bounded in [0,1]) ---
    proj = elliptic_projection_vec(x, y)

    # --- combine ---
    pred_final = pred_alg + proj * corr

    # --- residuals ---
    r_alg = y - pred_alg
    r_fin = y - pred_final

    # --- supertrace / entropy / mass ---
    S_alg = supertrace(r_alg)
    S_fin = supertrace(r_fin)
    H_alg = entropy(S_alg, len(y), alpha)
    H_fin = entropy(S_fin, len(y), alpha)
    m_alg = invariant_mass(S_alg, len(y))
    m_fin = invariant_mass(S_fin, len(y))

    rmse_alg = float(np.sqrt(np.mean(r_alg ** 2)))
    rmse_fin = float(np.sqrt(np.mean(r_fin ** 2)))

    return EllipticRegressionResult(
        a=a, b=b,
        pred_alg=pred_alg, pred_final=pred_final,
        projection=proj, correction=corr,
        residuals_alg=r_alg, residuals_final=r_fin,
        S_alg=S_alg, S_final=S_fin,
        H_alg=H_alg, H_final=H_fin,
        m_alg=m_alg, m_final=m_fin,
        rmse_alg=rmse_alg, rmse_final=rmse_fin,
    )


# ============================================================
#  Baseline — ordinary least squares
# ============================================================
def ols_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, np.ndarray]:
    A = np.vstack([x, np.ones_like(x)]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(m), float(c), m * x + c


# ============================================================
#  Figure 3.5 — the projection curve
# ============================================================
def figure_3_5(ax):
    """
    Draw the bounded elliptic projection curve
        Π(x, y) = ½(1 + cos(2π x/π) cos(2π y/e))
    on the fundamental parallelogram [0, π) × [0, e).
    This is the linear/non-linear scaling surface referenced
    in the supply‑chain paper (Figure 3.5).
    """
    xs = np.linspace(0, W1, 200)
    ys = np.linspace(0, W2, 200)
    X, Y = np.meshgrid(xs, ys)
    U = (X / W1) % 1.0
    V = (Y / W2) % 1.0
    P = 0.5 * (1.0 + np.cos(2 * np.pi * U) * np.cos(2 * np.pi * V))

    im = ax.imshow(P, extent=[0, W1, 0, W2], origin="lower",
                   cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xlabel("x   (mod π)")
    ax.set_ylabel("y   (mod e)")
    ax.set_title("Figure 3.5  ·  bounded elliptic projection Π ∈ [0,1]")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


# ============================================================
#  Demo
# ============================================================
def demo():
    rng = np.random.default_rng(7)

    # ---- synthetic data on a gently curved manifold ----
    N = 120
    x = np.linspace(0.6, 6.0, N)
    y_true = 1.0 / (0.35 / x + 0.18)                    # inverse‑Möbius shape
    y = y_true + 0.02 * rng.standard_normal(N)          # small noise

    # ---- baselines ----
    m_ols, c_ols, y_ols = ols_fit(x, y)
    rmse_ols = float(np.sqrt(np.mean((y - y_ols) ** 2)))

    # ---- elliptic dual regression ----
    res = elliptic_dual_regression(x, y)

    # ---- report ----
    print("=" * 74)
    print("Elliptic dual regression  ·  algebraic / transcendental split")
    print("=" * 74)
    print(f"α = 1/(π − e)  = {ALPHA:.12f}")
    print(f"periods        ω₁ = π = {W1:.6f}   ω₂ = e = {W2:.6f}")
    print()
    print("--- ordinary least squares (baseline) ---")
    print(f"  slope m        = {m_ols:+.6f}")
    print(f"  intercept c    = {c_ols:+.6f}")
    print(f"  RMSE           = {rmse_ols:.6e}")
    print()
    print("--- elliptic dual regression ---")
    print(f"  algebraic a    = {res.a:+.6f}    (fit in 1/y vs 1/x)")
    print(f"  algebraic b    = {res.b:+.6f}")
    print(f"  RMSE  algebraic side         = {res.rmse_alg:.6e}")
    print(f"  RMSE  algebraic + Π · Δ      = {res.rmse_final:.6e}")
    print(f"  improvement                  = "
          f"{100.0*(1 - res.rmse_final/res.rmse_alg):.2f} %")
    print()
    print("--- supertrace / entropy / mass ---")
    print(f"  S  (algebraic)      = {res.S_alg:+.6e}")
    print(f"  S  (algebraic+ΠΔ)   = {res.S_final:+.6e}")
    print(f"  H  (algebraic)      = {res.H_alg:.6f}")
    print(f"  H  (algebraic+ΠΔ)   = {res.H_final:.6f}")
    print(f"  m  (algebraic)      = {res.m_alg:.6e}")
    print(f"  m  (algebraic+ΠΔ)   = {res.m_final:.6e}")
    print()
    print(f"  Π range             = [{res.projection.min():.4f}, "
          f"{res.projection.max():.4f}]")
    print(f"  Δ range             = [{res.correction.min():+.4f}, "
          f"{res.correction.max():+.4f}]")

    # ---- plots ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) data + fits
    ax = axes[0, 0]
    ax.scatter(x, y, s=14, c="#3a7bd5", alpha=0.7, label="data")
    ax.plot(x, y_ols, "r--", lw=1.3, label=f"OLS (RMSE={rmse_ols:.2e})")
    ax.plot(x, res.pred_alg, "g-", lw=1.3,
            label=f"algebraic 1/(a/x+b)  (RMSE={res.rmse_alg:.2e})")
    ax.plot(x, res.pred_final, "k-", lw=1.6,
            label=f"algebraic + Π·Δ  (RMSE={res.rmse_final:.2e})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Data and the three fits")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) residuals
    ax = axes[0, 1]
    ax.axhline(0, color="k", lw=0.5)
    ax.plot(x, res.residuals_alg, "g-", lw=1.1, label="algebraic")
    ax.plot(x, res.residuals_final, "k-", lw=1.3, label="algebraic + Π·Δ")
    ax.set_xlabel("x")
    ax.set_ylabel("residual y − ŷ")
    ax.set_title("Residuals (Möbius side vs. combined)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (c) projection Π(x, y) along the data
    ax = axes[1, 0]
    ax.plot(x, res.projection, color="#8e44ad", lw=1.4, label="Π(x, y)")
    ax.plot(x, np.abs(res.correction), color="#e67e22", lw=1.2,
            label="|Δ(x)|")
    ax.set_xlabel("x")
    ax.set_ylabel("weight")
    ax.set_title("Bounded elliptic projection and transcendental correction")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (d) Figure 3.5 — the projection surface itself
    figure_3_5(axes[1, 1])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo()