#!/usr/bin/env python3
"""
spectral_harmonic_sum.py
========================

Spectral sum driven by the negative harmonic log.

    ζ(t) = Σ_{i=-K}^{K}  C[i] · exp(i · t · i / α_i)

    i = 0     :  C[0] = +H_1 = +1                    real positive DC
    i ≠ 0     :  C[i] = −H_{|i|+1} ≈ −ln(|i|+1) − γ  negative harmonic

    α_i bounded between the two alphas:

        α_min = 0.3628              (asymmetric, lower)
        α_max = 1/(π−e) ≈ 2.362     (symmetric,  upper)

        α_i  =  α_min + (α_max − α_min) · |i| / K

    → α_0 = α_min, α_K = α_max, linear interpolation between

Two-point spectral sum  ·  M(x, y) = [x^-1; y^-1] ⊗ [x^i, y^i]
----------------------------------------------------------------
    p_x    = x^(i-1) = x^-1 e^{i ln x}       complex transcendental
    p_y    = y^(i-1) = y^-1 e^{i ln y}       complex transcendental
    tr M   = p_x + p_y                       standard trace

    Q      = Re(tr M) = x^-1 cos ln x + y^-1 cos ln y    algebraic, real
              (negative when ln x, ln y fall in a cos < 0 band)
    P      = Im(tr M) = x^-1 sin ln x + y^-1 sin ln y    transcendental

    sym    =  Q   →  |Ci| positive half  (algebraic-real)
    asym   =  P   →  |Ci| negative half  (transcendental-imag)

Rationals = finite step Z
-------------------------
    Z_forward(t) = (ζ(t+A) − ζ(t)) / A
    Z_central(t) = (ζ(t+A) − ζ(t−A)) / (2A)

    A_STEP = α_max / α_min ≈ 6.511

Imports (unchanged parents)
---------------------------
    ugc_discrete_log.harmonic_numbers   H_n = 1 + 1/2 + … + 1/n
    CPU_Lin_Alg.ALPHA_ASYM = 0.3628     α_min
    CPU_Lin_Alg.ALPHA      = 1/(π−e)    α_max
    CPU_Lin_Alg.A_STEP     = α_max/α_min
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---- parent imports ----
from ugc_discrete_log import (
    harmonic_numbers,
    GAMMA,
)
from CPU_Lin_Alg import (
    ALPHA_ASYM,    # 0.3628
    ALPHA,         # 1/(π−e) ≈ 2.362
    A_STEP,        # ≈ 6.511
    PI, E,
)

ALPHA_MIN = ALPHA_ASYM
ALPHA_MAX = ALPHA


# ============================================================
#  Coefficients  ·  C[i]
# ============================================================
def build_spectral_coefficients(K: int) -> np.ndarray:
    """
    C of length 2K+1:

        C[K]   = +1                    (i = 0, real positive DC)
        C[K+i] = −H_{i+1}              (i > 0, negative harmonic)
        C[K−i] = −H_{i+1}              (i < 0, symmetric)
    """
    H = harmonic_numbers(K + 1)
    C = np.zeros(2 * K + 1, dtype=np.float64)
    C[K] = 1.0
    for i in range(1, K + 1):
        C[K + i] = -float(H[i + 1])
        C[K - i] = -float(H[i + 1])
    return C


# ============================================================
#  Bounded alpha
# ============================================================
def alpha_at(i: int, K: int) -> float:
    """α_i = α_min + (α_max − α_min)·|i|/K, clipped to [α_min, α_max]."""
    t = min(abs(i) / max(K, 1), 1.0)
    return ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * t


# ============================================================
#  Spectral sum
# ============================================================
def spectral_sum(t: float, C: np.ndarray,
                 K: Optional[int] = None) -> complex:
    """Direct evaluation:  ζ(t) = Σ C[i] · exp(i·t·i/α_i)."""
    if K is None:
        K = (len(C) - 1) // 2
    total = 0.0 + 0.0j
    for i in range(-K, K + 1):
        if i == 0:
            total += C[K] + 0.0j
            continue
        a_i = alpha_at(i, K)
        phase = t * i / a_i
        total += C[K + i] * complex(math.cos(phase), math.sin(phase))
    return total


def spectral_sum_vec(t: float, C: np.ndarray,
                     K: Optional[int] = None) -> complex:
    """Vectorized ζ(t) with per-index α."""
    if K is None:
        K = (len(C) - 1) // 2
    i_arr = np.arange(-K, K + 1, dtype=np.float64)
    ratio = np.abs(i_arr) / max(K, 1)
    alpha_arr = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * ratio
    alpha_arr[i_arr == 0] = 1.0
    phase = np.where(i_arr != 0, t * i_arr / alpha_arr, 0.0)
    return complex(np.sum(C * np.exp(1j * phase)))


# ============================================================
#  Two-point matrix
# ============================================================
@dataclass
class TwoPointMatrix:
    """
    M(x, y) = [x^-1; y^-1] ⊗ [x^i, y^i]   (i = imaginary unit)

        M = [x^(i-1),  x^-1 · y^i]
            [y^-1 · x^i,  y^(i-1)]

        p_x  = x^(i-1) = x^-1 · e^{i·ln x}   complex transcendental
        p_y  = y^(i-1) = y^-1 · e^{i·ln y}   complex transcendental
        tr M = p_x + p_y

        Q    = Re(tr M)   algebraic, real, sign depends on ln x, ln y
        P    = Im(tr M)   transcendental
    """
    x: float
    y: float

    # ---- the two transcendental diagonals ----
    def p_x(self) -> complex:
        if self.x <= 0:
            return 0j
        inv = 1.0 / self.x
        lnx = math.log(self.x)
        return complex(inv * math.cos(lnx), inv * math.sin(lnx))

    def p_y(self) -> complex:
        if self.y <= 0:
            return 0j
        inv = 1.0 / self.y
        lny = math.log(self.y)
        return complex(inv * math.cos(lny), inv * math.sin(lny))

    def trace(self) -> complex:
        """tr M = p_x + p_y."""
        return self.p_x() + self.p_y()

    def Q(self) -> float:
        """Algebraic part = Re(tr M)."""
        return self.trace().real

    def P(self) -> float:
        """Transcendental part = Im(tr M)."""
        return self.trace().imag

    def sym(self) -> float:
        """Real-plane projection  (algebraic, = Q)."""
        return self.Q()

    def asym(self) -> float:
        """Imag-plane projection  (transcendental, = P)."""
        return self.P()

    def Q_negative_reference(self) -> float:
        """
        Explicit negative algebraic reference:
            Q_ref = −(x^-1 + y^-1)  <  0  ∀ positive x, y
        """
        return -(1.0 / max(self.x, 1e-12)
                 + 1.0 / max(self.y, 1e-12))


# ============================================================
#  Finite step  ·  rationals as Z
# ============================================================
def Z_forward(t: float, C: np.ndarray,
              a: float = A_STEP,
              K: Optional[int] = None) -> complex:
    """Z(t) = (ζ(t+a) − ζ(t)) / a."""
    return (spectral_sum_vec(t + a, C, K)
            - spectral_sum_vec(t, C, K)) / a


def Z_central(t: float, C: np.ndarray,
              a: float = A_STEP,
              K: Optional[int] = None) -> complex:
    """Z(t) = (ζ(t+a) − ζ(t−a)) / (2a)."""
    return (spectral_sum_vec(t + a, C, K)
            - spectral_sum_vec(t - a, C, K)) / (2.0 * a)


# ============================================================
#  Full pipeline
# ============================================================
class SpectralHarmonicSum:
    """
    Ties together:
        · negative harmonic log      C[i] = −H_{|i|+1}
        · real positive DC           C[0] = +1
        · two-alpha bounded range    α_i ∈ [α_min, α_max]
        · two-point matrix           M(x, y) = [x^-1; y^-1] ⊗ [x^i, y^i]
        · finite-step rationals      Z(t) = dζ/dt
    """
    def __init__(self, K: int = 20,
                 alpha_min: float = ALPHA_MIN,
                 alpha_max: float = ALPHA_MAX):
        self.K = int(K)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.C = build_spectral_coefficients(self.K)

    # ---- alpha ----
    def alpha_at(self, i: int) -> float:
        t = min(abs(i) / max(self.K, 1), 1.0)
        return self.alpha_min + (self.alpha_max - self.alpha_min) * t

    # ---- ζ(t) ----
    def zeta(self, t: float) -> complex:
        return spectral_sum_vec(t, self.C, self.K)

    def power(self, t: float) -> float:
        return abs(self.zeta(t)) ** 2

    # ---- two-point ----
    def two_point(self, x: float, y: float) -> TwoPointMatrix:
        return TwoPointMatrix(x=x, y=y)

    # ---- finite step ----
    def Z_forward(self, t: float, a: float = A_STEP) -> complex:
        return Z_forward(t, self.C, a, self.K)

    def Z_central(self, t: float, a: float = A_STEP) -> complex:
        return Z_central(t, self.C, a, self.K)

    # ---- supertrace / entropy ----
    def supertrace(self, t_samples: np.ndarray) -> float:
        S = 0.0
        for k, t in enumerate(t_samples):
            v = self.zeta(float(t)).real
            S += v if (k % 2 == 0) else -v
        return float(S)

    def entropy(self, t_samples: np.ndarray) -> float:
        S = self.supertrace(t_samples)
        N = len(t_samples)
        if N <= 0 or S == 0.0:
            return 0.0
        p = abs(S) / N
        if p <= 0.0 or p >= 1.0:
            return 0.0
        return -self.alpha_max * p * math.log(p)

    def invariant_mass(self, t_samples: np.ndarray) -> float:
        H = self.entropy(t_samples)
        S = self.supertrace(t_samples)
        return abs(S) * math.exp(-H) if H < 700 else 0.0

    # ---- report ----
    def report(self) -> str:
       # NEW
        sym = bool(np.allclose(self.C, self.C[::-1]))
        lines = [
            "=" * 76,
            "Spectral harmonic sum",
            "=" * 76,
            f"  K              = {self.K}",
            f"  α_min          = {self.alpha_min:.6f}   (0.3628)",
            f"  α_max          = {self.alpha_max:.6f}   (1/(π−e))",
            f"  A_STEP         = {A_STEP:.6f}",
            "",
            "  coefficient array C[i]:",
            f"    C[0]     (i=0)    = {self.C[self.K]:+.6f}    real positive",
            f"    C[±1]             = {self.C[self.K + 1]:+.6f}    −H_2",
            f"    C[±2]             = {self.C[self.K + 2]:+.6f}    −H_3",
            f"    C[±K]             = {self.C[2*self.K]:+.6f}    −H_{{K+1}}",
            f"    symmetric         = {sym}",
            "",
            "  α_i bounded between α_min and α_max:",
        ]
        for i in (0, 1, self.K // 2, self.K):
            lines.append(f"    α_{i:>3d}            = {self.alpha_at(i):.6f}")
        return "\n".join(lines)


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("Spectral harmonic sum  ·  negative harmonic log drives ζ(t)")
    print("=" * 78)
    print(f"  α_min  = 0.3628       = {ALPHA_MIN:.6f}")
    print(f"  α_max  = 1/(π−e)      = {ALPHA_MAX:.6f}")
    print(f"  A_STEP = α_max/α_min  = {A_STEP:.6f}")
    print()

    K = 20
    shs = SpectralHarmonicSum(K=K)
    print(shs.report())
    print()

    # ---- 1 · ζ(t) ----
    print("=" * 78)
    print("1  ·  ζ(t) = Σ C[i]·exp(i·t·i/α_i)")
    print("=" * 78)
    print(f"  {'t':>8s}  {'Re ζ':>12s}  {'Im ζ':>12s}  {'|ζ|²':>10s}")
    print("  " + "-" * 50)
    for t in (0.0, 1.0, 2.0, 5.0, 10.0):
        z = shs.zeta(t)
        print(f"  {t:>8.3f}  {z.real:>+12.6f}  {z.imag:>+12.6f}  "
              f"{abs(z)**2:>10.6f}")
    print()

    # ---- 2 · i = 0 real positive ----
    print("=" * 78)
    print("2  ·  i = 0  ·  real positive contribution")
    print("=" * 78)
    z0 = shs.zeta(0.0)
    z0_no_dc = sum(
        shs.C[K + i] for i in range(-K, K + 1) if i != 0)
    print(f"  C[0]              = {shs.C[K]:+.6f}")
    print(f"  ζ(0)              = {z0.real:+.6f}  {z0.imag:+.6f}i")
    print(f"  ζ(0) − C[0]       = {z0.real - shs.C[K]:+.6f}  "
          f"(all non-DC terms)")
    print(f"  sum C[i≠0]        = {z0_no_dc:+.6f}")
    print()

    # ---- 3 · α bounded ----
    print("=" * 78)
    print("3  ·  natural numbers bounded between α_min and α_max")
    print("=" * 78)
    print(f"  {'i':>4s}  {'α_i':>10s}  {'position':>10s}")
    print("  " + "-" * 30)
    for i in (0, 1, 5, 10, 15, 20):
        a = shs.alpha_at(i)
        pos = (a - ALPHA_MIN) / (ALPHA_MAX - ALPHA_MIN)
        print(f"  {i:>4d}  {a:>10.6f}  {pos:>9.1%}")
    print()

    # ---- 4 · two-point matrix ----
    print("=" * 78)
    print("4  ·  two-point M(x, y) = [x^-1; y^-1] ⊗ [x^i, y^i]")
    print("=" * 78)
    print(f"  {'x':>5s}  {'y':>5s}  {'Q':>10s}  {'P':>10s}  "
          f"{'p_x (complex)':>24s}  {'p_y (complex)':>24s}")
    print("  " + "-" * 84)
    pairs = [(2.0, 3.0), (3.0, 5.0), (5.0, 7.0),
             (0.5, 2.0), (4.0, 6.0)]
    for x, y in pairs:
        m = shs.two_point(x, y)
        px, py = m.p_x(), m.p_y()
        Q = m.Q()
        P = m.P()
        # sign indicator
        qsign = "neg" if Q < 0 else "pos"
        print(f"  {x:>5.1f}  {y:>5.1f}  "
              f"{Q:>+10.6f}  {P:>+10.6f}  "
              f"{px.real:>+10.4f}{px.imag:>+9.4f}i  "
              f"{py.real:>+10.4f}{py.imag:>+9.4f}i")
    print()

    # ---- detailed one that shows negative Q ----
    print("  specific case with Q < 0  (algebraic part negative real):")
    m = shs.two_point(5.0, 7.0)
    print(f"    x = 5.0, y = 7.0")
    print(f"    ln x             = {math.log(5.0):.6f}")
    print(f"    ln y             = {math.log(7.0):.6f}")
    print(f"    cos(ln x)        = {math.cos(math.log(5.0)):.6f}")
    print(f"    cos(ln y)        = {math.cos(math.log(7.0)):.6f}")
    print(f"    Q = Re(tr M)     = {m.Q():+.6f}   ← negative real (algebraic)")
    print(f"    P = Im(tr M)     = {m.P():+.6f}   ← transcendental")
    print(f"    p_x              = {m.p_x().real:+.6f} "
          f"{m.p_x().imag:+.6f}i   (complex transcendental)")
    print(f"    p_y              = {m.p_y().real:+.6f} "
          f"{m.p_y().imag:+.6f}i   (complex transcendental)")
    print(f"    tr M = p_x + p_y = {m.trace().real:+.6f} "
          f"{m.trace().imag:+.6f}i")
    print(f"    Q_ref = −(x^-1+y^-1) = {m.Q_negative_reference():+.6f}")
    print()

    # ---- 5 · finite step ----
    print("=" * 78)
    print("5  ·  rationals  ·  finite step Z(t) = dζ/dt")
    print("=" * 78)
    print(f"  A_STEP = {A_STEP:.6f}")
    print(f"  {'t':>6s}  {'Z_forward':>24s}  {'Z_central':>24s}")
    print("  " + "-" * 56)
    for t in (0.5, 1.0, 2.0, 5.0):
        Zf = shs.Z_forward(t)
        Zc = shs.Z_central(t)
        print(f"  {t:>6.2f}  "
              f"{Zf.real:>+10.4f}{Zf.imag:>+11.4f}i  "
              f"{Zc.real:>+10.4f}{Zc.imag:>+11.4f}i")
    print()

    # ---- 6 · invariants ----
    print("=" * 78)
    print("6  ·  invariants over sampled t")
    print("=" * 78)
    t_samples = np.linspace(0.0, 10.0, 64)
    S = shs.supertrace(t_samples)
    H_ent = shs.entropy(t_samples)
    m_inv = shs.invariant_mass(t_samples)
    print(f"  t_samples      = {len(t_samples)}")
    print(f"  S              = {S:+.6f}")
    print(f"  H              = {H_ent:.6f}")
    print(f"  m              = {m_inv:.6e}")
    print()

    # ---- 7 · visualisation ----
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.36, wspace=0.30)

    # (a) C[i]
    ax = fig.add_subplot(gs[0, 0])
    idx = np.arange(-K, K + 1)
    colors = np.where(idx == 0, "#f1c40f", "#3498db")
    ax.bar(idx, shs.C, color=colors, edgecolor="k", linewidth=0.4)
    ax.axvline(0, color="k", lw=0.7, ls="--")
    ax.set_xlabel("i"); ax.set_ylabel("C[i]")
    ax.set_title("C[i]  ·  yellow = +1 (i=0), blue = −H_{|i|+1}")
    ax.grid(alpha=0.3, axis="y")

    # (b) α_i bounded
    ax = fig.add_subplot(gs[0, 1])
    ii = np.arange(-K, K + 1)
    aa = np.array([shs.alpha_at(int(i)) for i in ii])
    ax.plot(ii, aa, "o-", color="#8e44ad", ms=3, lw=1.2)
    ax.axhline(ALPHA_MIN, color="#16a085", ls="--",
               label=f"α_min = {ALPHA_MIN:.4f}")
    ax.axhline(ALPHA_MAX, color="#e74c3c", ls="--",
               label=f"α_max = {ALPHA_MAX:.4f}")
    ax.set_xlabel("i"); ax.set_ylabel("α_i")
    ax.set_title("α_i = α_min + (α_max − α_min)·|i|/K")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (c) ζ(t)
    ax = fig.add_subplot(gs[0, 2])
    t_grid = np.linspace(0.0, 10.0, 400)
    zetas = np.array([shs.zeta(float(t)) for t in t_grid])
    ax.plot(t_grid, zetas.real, color="#3498db", lw=1.2,
            label="Re ζ(t)")
    ax.plot(t_grid, zetas.imag, color="#e74c3c", lw=1.2,
            label="Im ζ(t)")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlabel("t"); ax.set_ylabel("ζ(t)")
    ax.set_title("ζ(t) = Σ C[i]·exp(i·t·i/α_i)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (d) |ζ|²
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t_grid, np.abs(zetas) ** 2, color="#8e44ad", lw=1.2)
    ax.set_xlabel("t"); ax.set_ylabel("|ζ|²")
    ax.set_title("power spectrum |ζ|²")
    ax.grid(alpha=0.3)

    # (e) two-point: Q and P across (x, y) grid
    ax = fig.add_subplot(gs[1, 1])
    xs = np.linspace(0.5, 8.0, 30)
    ys = np.linspace(0.5, 8.0, 30)
    Qgrid = np.zeros((len(xs), len(ys)))
    Pgrid = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            m = TwoPointMatrix(x, y)
            Qgrid[i, j] = m.Q()
            Pgrid[i, j] = m.P()
    im = ax.imshow(Qgrid, origin="lower",
                   extent=[ys.min(), ys.max(), xs.min(), xs.max()],
                   cmap="RdBu_r", aspect="auto", vmin=-0.5, vmax=0.5)
    ax.set_xlabel("y"); ax.set_ylabel("x")
    ax.set_title("Q = Re(tr M)  ·  negative when cos ln x, cos ln y < 0")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (f) finite step Z(t)
    ax = fig.add_subplot(gs[1, 2])
    Zf = np.array([shs.Z_forward(float(t)) for t in t_grid])
    Zc = np.array([shs.Z_central(float(t)) for t in t_grid])
    ax.plot(t_grid, Zf.real, color="#2980b9", lw=1.0,
            label="Re Z_forward")
    ax.plot(t_grid, Zc.real, color="#c0392b", lw=1.0, ls="--",
            label="Re Z_central")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlabel("t"); ax.set_ylabel("Z(t)")
    ax.set_title(f"finite step  Z = dζ/dt  (A = {A_STEP:.3f})")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.suptitle(
        "Spectral harmonic sum  ·  C[i] = −H_{|i|+1}  ·  "
        "α ∈ [0.3628, 1/(π−e)]  ·  Q = Re(tr M)  ·  "
        "finite-step rationals  ·  α = 2.362",
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # ---- summary ----
    print("=" * 78)
    print("Summary")
    print("=" * 78)
    print("  spectral sum ζ(t) = Σ_{i=-K}^{K} C[i]·exp(i·t·i/α_i)")
    print("  coefficient array:")
    print("    C[0]   = +H_1 = +1               real positive (i=0)")
    print("    C[i≠0] = −H_{|i|+1}               negative harmonic log")
    print("    v(i)   ≈ −ln(|i|+1) − γ           continuous log")
    print()
    print("  bounded alphas:")
    print(f"    α_min = {ALPHA_MIN:.6f}                (asymmetric)")
    print(f"    α_max = {ALPHA_MAX:.6f}                (symmetric)")
    print(f"    α_i linearly interpolates 0 → K")
    print()
    print("  two-point matrix M(x, y) = [x^-1; y^-1] ⊗ [x^i, y^i]:")
    print(f"    p_x    = x^(i-1) = x^-1·e^{{i·ln x}}    complex transcendental")
    print(f"    p_y    = y^(i-1) = y^-1·e^{{i·ln y}}    complex transcendental")
    print(f"    tr M   = p_x + p_y                    trace")
    print(f"    Q      = Re(tr M)                     algebraic real")
    print(f"             (Q < 0 when cos ln x, cos ln y < 0)")
    print(f"    P      = Im(tr M)                     transcendental imag")
    print(f"    sym    = Q                            |Ci| positive half")
    print(f"    asym   = P                            |Ci| negative half")
    print()
    print("  rationals as finite step:")
    print(f"    Z_forward(t) = (ζ(t+A) − ζ(t)) / A")
    print(f"    Z_central(t) = (ζ(t+A) − ζ(t−A)) / (2A)")
    print(f"    A_STEP       = α_max/α_min = {A_STEP:.6f}")
    print()
    print("Done.")


if __name__ == "__main__":
    demo()