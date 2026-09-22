#!/usr/bin/env python3
"""
elliptic_log_rank.py

Log‑rank test with an **elliptic algebraic / transcendental split**,
where the transcendental axis is *time* and the algebraic axis is
*rank*.

Model
-----
Given discrete event points  (t_n, r_n),  n = 1 … N,  where
    t_n = event time         (non‑decreasing, transcendental axis)
    r_n = rank / cumulative count  (algebraic axis)

    algebraic side   u_n = 1 / r_n              (Möbius inverse of rank)
                     v_n = 1 / t_n              (Möbius inverse of time)
                     fit  v ≈ a·u + b           → (a, b)

    transcendental   p_n = t_n^i  (time on the imaginary plane)
                     q_n = r_n^i
                     log‑rank residual  Δ(t) = α · Im( q̄ · p ) / (1 + |p|²)

    elliptic bridge  Π(t, r) ∈ [0, 1]  from Figure 3.5

    observed − expected at time n:
        O_n = Δr_n                            (observed rank increment)
        E_n = a · Δu_n + b · Δv_n             (algebraic expectation)
        Z_n = O_n − E_n                       (log‑rank residual)

    final log‑rank statistic:
        Z_ell = Σ_n Π(t_n, r_n) · Z_n
        var   = Σ_n Π(t_n, r_n)² · Var(Z_n)

The supertrace  S = Σ_n (−1)^n Z_n,  its entropy H, and the
invariant mass  m = |S| · e^{−H}  are reported exactly as in the
survival‑analysis chi‑square pipeline.
"""

from __future__ import annotations

import math
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Optional


# ============================================================
#  Constants
# ============================================================
PI    = math.pi
E     = math.e
ALPHA = 1.0 / (PI - E)             # ≈ 2.362
W1    = PI
W2    = E


# ============================================================
#  1. Figure 3.5 — bounded elliptic projection
# ============================================================
def elliptic_projection(t: float, r: float) -> float:
    """Π(t, r) ∈ [0, 1] on the torus ℂ / (πℤ + eℤ)."""
    u = (t / W1) % 1.0
    v = (r / W2) % 1.0
    return 0.5 * (1.0 + math.cos(2 * math.pi * u) *
                        math.cos(2 * math.pi * v))


def elliptic_projection_vec(t: np.ndarray, r: np.ndarray) -> np.ndarray:
    u = (t / W1) % 1.0
    v = (r / W2) % 1.0
    return 0.5 * (1.0 + np.cos(2 * np.pi * u) * np.cos(2 * np.pi * v))


# ============================================================
#  2. Algebraic side — rank / time inverse coordinates
# ============================================================
def inverse_coords(r: np.ndarray, t: np.ndarray,
                   eps: float = 1e-12
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """u = 1/r,  v = 1/t  (inverse Möbius coordinates)."""
    u = 1.0 / np.maximum(np.abs(r), eps) * np.sign(r)
    v = 1.0 / np.maximum(np.abs(t), eps) * np.sign(t)
    return u, v


def algebraic_fit(u: np.ndarray, v: np.ndarray) -> Tuple[float, float]:
    """Ordinary least squares:  v ≈ a·u + b  (Möbius tangent plane)."""
    A = np.vstack([u, np.ones_like(u)]).T
    a, b = np.linalg.lstsq(A, v, rcond=None)[0]
    return float(a), float(b)


# ============================================================
#  3. Transcendental side — time on the imaginary plane
# ============================================================
def transcendental_coords(t: np.ndarray, r: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Time is the transcendental axis:

        p_n = t_n^i = exp(i log t_n)      (time phase)
        q_n = r_n^i = exp(i log r_n)      (rank phase)

    For t = 0 we use a small floor to remain on the principal sheet.
    """
    lt = np.log(np.maximum(np.abs(t), 1e-12))
    lr = np.log(np.maximum(np.abs(r), 1e-12))
    p = np.exp(1j * lt)                       # time
    q = np.exp(1j * lr)                       # rank
    return p, q


def log_rank_residual(t: np.ndarray,
                      p: np.ndarray,
                      q: np.ndarray,
                      alpha: float = ALPHA) -> np.ndarray:
    """
    Bounded transcendental residual in the imaginary plane:

        Δ(t) = α · Im( q̄ · p ) / (1 + |p|²)

    Time enters through p; the rank enters through q.  When the
    time phase and rank phase align, the residual vanishes — the
    log‑rank test's "no difference" baseline.
    """
    num = np.imag(np.conj(q) * p)
    den = 1.0 + np.abs(p) ** 2
    return alpha * num / den


# ============================================================
#  4. Supertrace / entropy / mass of the log‑rank residuals
# ============================================================
def supertrace(z: np.ndarray) -> float:
    S = 0.0
    for k, v in enumerate(z):
        S += v if (k % 2 == 0) else -v
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
#  5. Log‑rank result container
# ============================================================
@dataclass
class EllipticLogRankResult:
    t: np.ndarray
    r: np.ndarray
    u: np.ndarray                  # 1/r
    v: np.ndarray                  # 1/t
    a: float                       # algebraic slope
    b: float                       # algebraic intercept
    O: np.ndarray                  # observed
    E: np.ndarray                  # expected (algebraic)
    Z: np.ndarray                  # raw residual
    Pi: np.ndarray                 # elliptic projection Π ∈ [0,1]
    delta: np.ndarray              # transcendental Δ(t)
    Z_ell: float                   # elliptic log‑rank statistic
    var_ell: float                 # variance
    chi2: float                    # Z_ell² / var_ell
    S_ell: float                   # supertrace
    H_ell: float                   # entropy
    m_ell: float                   # invariant mass


# ============================================================
#  6. Elliptic log‑rank test
# ============================================================
def elliptic_log_rank(t: np.ndarray, r: np.ndarray,
                      alpha: float = ALPHA
                      ) -> EllipticLogRankResult:
    """
    Compute the elliptic log‑rank statistic for discrete event
    points  (t_n, r_n).

    Steps
    -----
    1. Algebraic side:  u = 1/r,  v = 1/t,  fit  v ≈ a·u + b.
    2. Observed rank increment:  O_n = Δr_n.
    3. Expected rank increment:  E_n = a·Δu_n + b·Δv_n.
    4. Raw log‑rank residual:    Z_n = O_n − E_n.
    5. Transcendental side:      p = t^i,  q = r^i,
                                 Δ_n = α·Im(q̄ p)/(1+|p|²).
    6. Elliptic projection:      Π_n = Π(t_n, r_n) ∈ [0,1].
    7. Elliptic log‑rank:        Z_ell = Σ_n Π_n · Z_n
                                 var   = Σ_n Π_n² · Δ_n²
                                 chi²  = Z_ell² / var
    """
    t = np.asarray(t, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)

    # ---------- algebraic side ----------
    u, v = inverse_coords(r, t)
    a, b = algebraic_fit(u, v)

    # observed / expected rank increments
    O = np.diff(r, prepend=r[0])            # Δr_n
    dU = np.diff(u, prepend=u[0])           # Δu_n
    dV = np.diff(v, prepend=v[0])           # Δv_n
    E = a * dU + b * dV                     # algebraic expectation
    Z = O - E                               # raw residual

    # ---------- transcendental side ----------
    p, q = transcendental_coords(t, r)
    delta = log_rank_residual(t, p, q, alpha)

    # ---------- elliptic projection ----------
    Pi = elliptic_projection_vec(t, r)

    # ---------- elliptic log‑rank statistic ----------
    Z_ell = float(np.sum(Pi * Z))
    var_ell = float(np.sum((Pi * delta) ** 2))
    chi2 = (Z_ell ** 2) / var_ell if var_ell > 0 else 0.0

    # ---------- supertrace / entropy / mass ----------
    S_ell = supertrace(Pi * Z)
    H_ell = entropy(S_ell, len(Z), alpha)
    m_ell = invariant_mass(S_ell, len(Z))

    return EllipticLogRankResult(
        t=t, r=r, u=u, v=v,
        a=a, b=b,
        O=O, E=E, Z=Z,
        Pi=Pi, delta=delta,
        Z_ell=Z_ell, var_ell=var_ell, chi2=chi2,
        S_ell=S_ell, H_ell=H_ell, m_ell=m_ell,
    )


# ============================================================
#  7. Classical log‑rank baseline (for comparison)
# ============================================================
def classical_log_rank(t: np.ndarray, r: np.ndarray
                       ) -> Tuple[float, float, float]:
    """
    Standard log‑rank statistic on discrete event times:

        Z = Σ_n (O_n − E_n)
        var = Σ_n (O_n − E_n)² · (r_n / (N − r_n + 1))
        chi2 = Z² / var

    Here O_n = Δr_n and E_n is the mean of Δr under the null.
    """
    t = np.asarray(t, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    N = len(r)

    O = np.diff(r, prepend=r[0])
    # expected = mean Δr under H0 (uniform over remaining rank)
    E = np.full_like(O, O.mean()) if O.size else O

    Z = float(np.sum(O - E))
    var = float(np.sum((O - E) ** 2) * (1.0 / max(N, 1)))
    chi2 = (Z ** 2) / var if var > 0 else 0.0
    return Z, var, chi2


# ============================================================
#  8. Figure 3.5 — projection surface (unchanged)
# ============================================================
def figure_3_5(ax):
    xs = np.linspace(0, W1, 200)
    ys = np.linspace(0, W2, 200)
    X, Y = np.meshgrid(xs, ys)
    U = (X / W1) % 1.0
    V = (Y / W2) % 1.0
    P = 0.5 * (1.0 + np.cos(2 * np.pi * U) * np.cos(2 * np.pi * V))
    im = ax.imshow(P, extent=[0, W1, 0, W2], origin="lower",
                   cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xlabel("time  t  (mod π)")
    ax.set_ylabel("rank  r  (mod e)")
    ax.set_title("Figure 3.5  ·  Π(t, r) ∈ [0,1]")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


# ============================================================
#  9. Demo — synthetic survival data
# ============================================================
def demo():
    print("=" * 76)
    print("Elliptic log‑rank test  ·  transcendental axis = time")
    print("=" * 76)
    print(f"α       = 1/(π − e) = {ALPHA:.6f}")
    print(f"periods ω₁ = π      = {W1:.6f}   ω₂ = e = {W2:.6f}")

    # ---- two synthetic groups with different hazard shapes ----
    rng = np.random.default_rng(2024)

    # Group A: exponential hazard, well‑behaved
    tA = np.sort(rng.exponential(scale=2.0, size=80))
    tA = np.maximum(tA, 1e-3)
    rA = np.arange(1, len(tA) + 1, dtype=float)

    # Group B: heavy‑tailed hazard, late events
    tB = np.sort(rng.exponential(scale=4.0, size=80))
    tB = np.maximum(tB, 1e-3)
    rB = np.arange(1, len(tB) + 1, dtype=float)

    # ---- classical log‑rank on the merged timeline ----
    # simple aggregated version: use group A as the primary event stream
    Z_cl_A, var_cl_A, chi_cl_A = classical_log_rank(tA, rA)
    Z_cl_B, var_cl_B, chi_cl_B = classical_log_rank(tB, rB)

    # ---- elliptic log‑rank per group ----
    resA = elliptic_log_rank(tA, rA)
    resB = elliptic_log_rank(tB, rB)

    # ---- combined statistic on the union of event times ----
    t_all = np.concatenate([tA, tB])
    order = np.argsort(t_all)
    t_all = t_all[order]
    # rank = count of events up to each time
    r_all = np.arange(1, len(t_all) + 1, dtype=float)
    res_all = elliptic_log_rank(t_all, r_all)

    # ---- report ----
    def _report(name, res, chi_cl=None):
        print(f"\n--- {name} ---")
        print(f"  N events            : {len(res.t)}")
        print(f"  algebraic a         : {res.a:+.6f}")
        print(f"  algebraic b         : {res.b:+.6f}")
        print(f"  Z_ell               : {res.Z_ell:+.6f}")
        print(f"  var_ell             : {res.var_ell:.6e}")
        print(f"  chi² = Z² / var     : {res.chi2:.6e}")
        if chi_cl is not None:
            print(f"  classical chi²      : {chi_cl:.6e}")
        print(f"  S_ell               : {res.S_ell:+.6f}")
        print(f"  H_ell               : {res.H_ell:.6f}")
        print(f"  m_ell               : {res.m_ell:.6e}")
        print(f"  Π range             : [{res.Pi.min():.4f}, "
              f"{res.Pi.max():.4f}]")
        print(f"  Δ range             : [{res.delta.min():+.4f}, "
              f"{res.delta.max():+.4f}]")

    _report("Group A (exponential, scale 2.0)", resA, chi_cl_A)
    _report("Group B (exponential, scale 4.0)", resB, chi_cl_B)
    _report("All events (union)", res_all)

    # ---- plots ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) event times and ranks
    ax = axes[0, 0]
    ax.plot(resA.t, resA.r, "o-", color="#3a7bd5", ms=3,
            lw=1.0, label="Group A")
    ax.plot(resB.t, resB.r, "s-", color="#e74c3c", ms=3,
            lw=1.0, alpha=0.8, label="Group B")
    ax.set_xlabel("time  t")
    ax.set_ylabel("rank  r")
    ax.set_title("Discrete event points  (t, r)")
    ax.legend()
    ax.grid(alpha=0.3)

    # (b) raw log‑rank residuals
    ax = axes[0, 1]
    ax.axhline(0, color="k", lw=0.5)
    ax.plot(resA.t, resA.Z, color="#3a7bd5", lw=1.0, label="Group A  Z_n")
    ax.plot(resB.t, resB.Z, color="#e74c3c", lw=1.0, alpha=0.8,
            label="Group B  Z_n")
    ax.set_xlabel("time  t")
    ax.set_ylabel("Z_n = O_n − E_n")
    ax.set_title("Log‑rank residuals")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (c) elliptic projection along time
    ax = axes[1, 0]
    ax.plot(resA.t, resA.Pi, color="#8e44ad", lw=1.2,
            label="Π(t, r)  Group A")
    ax.plot(resB.t, resB.Pi, color="#e67e22", lw=1.2, alpha=0.8,
            label="Π(t, r)  Group B")
    ax.set_xlabel("time  t")
    ax.set_ylabel("Π ∈ [0, 1]")
    ax.set_title("Elliptic projection along the event stream")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (d) Figure 3.5 surface
    figure_3_5(axes[1, 1])

    plt.suptitle(
        f"Elliptic log‑rank test  ·  α = {ALPHA:.4f}  ·  "
        f"χ²_ell (union) = {res_all.chi2:.4f}",
        fontsize=13,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    print("\nDone.")


if __name__ == "__main__":
    demo()