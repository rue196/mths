#!/usr/bin/env python3
"""
tuple_quadratic_signal.py
=========================

Tuple quadratic  ·  tau × muon × electron  ·  square-free signal

For each index k = 1..K:

    a_k = σ_τ · c_{−k}       tau coefficient   (negative side)
    b_k = σ_μ · c_{+k}       muon coefficient  (positive side)
    c₀  = c_K                electron / DC term

The three coefficients form the tuple quadratic

    Q_k(x) = a_k · x² + b_k · x + c₀

The square-free signal carried by the tuple is

    sig_k = |μ(k)| ∈ {0, 1}

— whether k is square-free.  This is the localised square-free
factor from ζ(s)/ζ(2s) (the paper's density).

Pure Möbius (smooth=False):
    sig_k = 1  ⇒  a_k = ±1, b_k = ±1, c₀ = 1   →  QUADRATIC
    sig_k = 0  ⇒  a_k = 0,  b_k = 0,  c₀ = 1   →  TRIVIAL

Smoothed (smooth=True):
    a_k, b_k are fractional; sig_k still the raw Möbius indicator.

Density
-------
    ρ(t) = Σ_{k=1}^{K} sig_k · Q_k( e^{i·t·k/α_k} )

This is the Fourier–Mellin sum over the square-free-supported tuples.
As K → ∞, ⟨sig_k⟩ → 6/π² = 1/ζ(2) (Basel density).

Tuple invariants
----------------
    discriminant  Δ_k = b_k² − 4·a_k·c₀
    roots         x_± = (−b_k ± √Δ_k) / (2·a_k)
    Vieta sum     s_k = −b_k/a_k
    Vieta product p_k =  c₀/a_k
    kind          a_k ≠ 0            →  quadratic
                  a_k = 0, b_k ≠ 0   →  linear
                  a_k = b_k = 0      →  trivial

Complexity
----------
    build tuples   O(K)
    per tuple      O(1)
    spectral sum   O(K·N_t)
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ---- 512-bit clock + Möbius (parent) ----
from OK_conv_tau_muon_512 import (
    OKRealClock,
    mobius_sieve,
    spin_grade,
    PI, E, ALPHA, A_STEP, DENSITY,
)

# ---- α(k) for the spectral domain (parent) ----
from complete_planes_spectral_sum import (
    alpha_at,
    ALPHA_MIN, ALPHA_MAX,
)


# ============================================================
#  Square-free indicators
# ============================================================
def sf_indicator_of_int(n: int, mu: np.ndarray) -> int:
    """|μ(n)| ∈ {0, 1}  using a precomputed sieve."""
    n = abs(int(n))
    if n <= 0:
        return 0
    if n == 1:
        return 1
    if n >= len(mu):
        return 1          # beyond sieve → assume square-free
    return abs(int(mu[n]))


def sf_indicator_of_value(v: float, mu: np.ndarray) -> int:
    """|μ(round(|v|))|  — square-free indicator of the rounded magnitude."""
    return sf_indicator_of_int(int(round(abs(float(v)))), mu)


# ============================================================
#  QuadraticTuple
# ============================================================
@dataclass
class QuadraticTuple:
    """
    One tuple quadratic:  a·x² + b·x + c₀
    with the square-free signal sf_k and coefficient signals.
    """
    k: int
    a: float               # tau coefficient
    b: float               # muon coefficient
    c: float               # electron / DC
    sf_k: int              # |μ(k)|          index square-free
    sf_a: int              # |μ(round(|a|))| coefficient a square-free
    sf_b: int
    sf_c: int
    chirality_a: float     # σ_τ used
    chirality_b: float     # σ_μ used

    # --------------------------------------------------------
    @property
    def kind(self) -> str:
        if abs(self.a) > 1e-12:
            return "quadratic"
        if abs(self.b) > 1e-12:
            return "linear"
        return "trivial"

    @property
    def discriminant(self) -> float:
        return self.b * self.b - 4.0 * self.a * self.c

    @property
    def roots(self) -> Optional[Tuple[complex, complex]]:
        """Complex roots (x₊, x₋) or None if not quadratic."""
        if self.kind != "quadratic":
            return None
        d = self.discriminant
        if d >= 0:
            sq = math.sqrt(d)
            den = 2.0 * self.a
            return (complex((-self.b + sq) / den, 0.0),
                    complex((-self.b - sq) / den, 0.0))
        sq = math.sqrt(-d)
        den = 2.0 * self.a
        return (complex(-self.b / den, sq / den),
                complex(-self.b / den, -sq / den))

    @property
    def vieta_sum(self) -> Optional[float]:
        if abs(self.a) < 1e-12:
            return None
        return -self.b / self.a

    @property
    def vieta_product(self) -> Optional[float]:
        if abs(self.a) < 1e-12:
            return None
        return self.c / self.a

    def evaluate(self, x: complex) -> complex:
        return self.a * x * x + self.b * x + self.c

    def __repr__(self) -> str:
        r = self.roots
        root_s = ("none" if r is None
                  else f"({r[0].real:+.3f}{r[0].imag:+.3f}i, "
                       f"{r[1].real:+.3f}{r[1].imag:+.3f}i)")
        return (f"Q_{self.k}(x) = {self.a:+.3f}·x² "
                f"{self.b:+.3f}·x {self.c:+.3f}   "
                f"[sf_k={self.sf_k}  {self.kind:<9s}  "
                f"Δ={self.discriminant:+.3f}  roots={root_s}]")


# ============================================================
#  TupleQuadraticSignal
# ============================================================
class TupleQuadraticSignal:
    """
    Builds the K tuple quadratics from the OK muon/tau clock,
    carries the square-free signal sf_k = |μ(k)|, and computes
    the Fourier–Mellin density.
    """

    def __init__(self,
                 K: int = 64,
                 J_C: float = 1.0,
                 smooth: bool = False):
        self.K = int(K)
        self.J_C = float(J_C)
        self.smooth = bool(smooth)

        self.clock = OKRealClock(
            K=self.K, J_C=self.J_C,
            smooth=self.smooth, use_fft=False)

        self.mu = mobius_sieve(self.K)
        c = self.clock.c
        self.c_0 = float(c[self.K])
        self.sigma_mu  = self.clock.sigma_mu
        self.sigma_tau = self.clock.sigma_tau

        self.tuples: List[QuadraticTuple] = []
        for k in range(1, self.K + 1):
            a = self.sigma_tau * float(c[self.K - k])
            b = self.sigma_mu  * float(c[self.K + k])
            sf_k = sf_indicator_of_int(k, self.mu)
            sf_a = sf_indicator_of_value(a, self.mu)
            sf_b = sf_indicator_of_value(b, self.mu)
            sf_c = sf_indicator_of_value(self.c_0, self.mu)
            self.tuples.append(QuadraticTuple(
                k=k, a=a, b=b, c=self.c_0,
                sf_k=sf_k, sf_a=sf_a, sf_b=sf_b, sf_c=sf_c,
                chirality_a=self.sigma_tau,
                chirality_b=self.sigma_mu,
            ))

        self._build_aggregates()

    # --------------------------------------------------------
    def _build_aggregates(self):
        K = self.K
        self.sf_total   = sum(t.sf_k for t in self.tuples)
        self.sf_density = self.sf_total / max(K, 1)
        self.sf_theory  = DENSITY   # 6/π²

        self.n_quad = sum(1 for t in self.tuples if t.kind == "quadratic")
        self.n_lin  = sum(1 for t in self.tuples if t.kind == "linear")
        self.n_triv = sum(1 for t in self.tuples if t.kind == "trivial")

        # signal vector:  sf_k · (a_k, b_k, c₀)
        self.signal = np.array([
            t.sf_k * np.array([t.a, t.b, t.c], dtype=float)
            for t in self.tuples
        ])                                                          # (K, 3)
        self.sig_k = np.array([t.sf_k for t in self.tuples])         # (K,)
        self.a_vec = np.array([t.a for t in self.tuples])
        self.b_vec = np.array([t.b for t in self.tuples])
        self.disc  = np.array([t.discriminant for t in self.tuples])

    # --------------------------------------------------------
    def spectral_density(self, t_grid: np.ndarray) -> np.ndarray:
        """
        ρ(t) = Σ_k sf_k · Q_k( e^{i·t·k/α_k} )

        A complex Fourier–Mellin sum over the square-free-supported tuples.
        """
        out = np.zeros(t_grid.shape, dtype=complex)
        for k, tup in enumerate(self.tuples, start=1):
            if tup.sf_k == 0:
                continue
            alpha_k = alpha_at(k, self.K)
            phase = t_grid * k / alpha_k
            x = np.exp(1j * phase)
            out += tup.sf_k * (tup.a * x * x + tup.b * x + tup.c)
        return out

    # --------------------------------------------------------
    def signal_of_signals(self) -> np.ndarray:
        """sf_k · (a_k, b_k, c₀)  summed over k."""
        return self.signal.sum(axis=0)

    def sf_weighted_root_sum(self) -> complex:
        """Σ_k sf_k · (root₊ + root₋) over the quadratic tuples."""
        s = 0.0 + 0.0j
        for t in self.tuples:
            if t.sf_k == 0:
                continue
            r = t.roots
            if r is None:
                continue
            s += (r[0] + r[1])
        return s

    # --------------------------------------------------------
    def report(self) -> str:
        L = ["=" * 76,
             "Tuple quadratic  ·  tau × muon × electron  ·  square-free",
             "=" * 76]
        L.append(f"  K                 : {self.K}")
        L.append(f"  J_C               : {self.J_C}")
        L.append(f"  smooth            : {self.smooth}")
        L.append(f"  σ_μ(J_C)          : {self.sigma_mu:+.0f}")
        L.append(f"  σ_τ(J_C)          : {self.sigma_tau:+.0f}")
        L.append(f"  c₀ (electron)     : {self.c_0:+.6f}")
        L.append("")
        L.append(f"  square-free density  : {self.sf_density:.6f}")
        L.append(f"  6/π² (theory)        : {self.sf_theory:.6f}")
        L.append(f"  |Δ| / K              : "
                 f"{abs(self.sf_density - self.sf_theory):.6f}")
        L.append("")
        L.append(f"  tuple kinds:")
        L.append(f"    quadratic        : {self.n_quad}")
        L.append(f"    linear           : {self.n_lin}")
        L.append(f"    trivial          : {self.n_triv}")
        L.append("")
        L.append(f"  signal-of-signals   Σ_k sf_k · (a, b, c₀):")
        sos = self.signal_of_signals()
        L.append(f"    Σ a_k·sf_k       : {sos[0]:+.6f}")
        L.append(f"    Σ b_k·sf_k       : {sos[1]:+.6f}")
        L.append(f"    Σ c₀·sf_k        : {sos[2]:+.6f}")
        L.append("")
        L.append(f"  sf-weighted root sum: "
                 f"{self.sf_weighted_root_sum():+.6f}")
        return "\n".join(L)

    # --------------------------------------------------------
    def visualize(self) -> None:
        K = self.K
        k_arr = np.arange(1, K + 1)
        t_grid = np.linspace(0.0, 2.0 * PI, 512)
        rho = self.spectral_density(t_grid)

        fig = plt.figure(figsize=(16, 11))
        gs = GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.32)

        # (a) square-free signal sf_k
        ax = fig.add_subplot(gs[0, 0])
        ax.bar(k_arr, self.sig_k,
               color=np.where(self.sig_k == 1, "#16a085", "#e74c3c"),
               edgecolor="k", linewidth=0.3)
        ax.axhline(self.sf_density, color="k", ls="--", lw=1.0,
                   label=f"⟨sf_k⟩ = {self.sf_density:.4f}")
        ax.axhline(DENSITY, color="#8e44ad", ls=":", lw=1.0,
                   label=f"6/π² = {DENSITY:.4f}")
        ax.set_xlabel("k"); ax.set_ylabel("sf_k = |μ(k)|")
        ax.set_title("square-free signal  sf_k  ∈  {0, 1}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # (b) coefficients a_k (tau) and b_k (muon)
        ax = fig.add_subplot(gs[0, 1])
        ax.plot(k_arr, self.a_vec, "o-", ms=3, lw=1.0,
                color="#e74c3c", label="a_k = σ_τ · c_{−k}  (tau)")
        ax.plot(k_arr, self.b_vec, "s-", ms=3, lw=1.0,
                color="#3498db", label="b_k = σ_μ · c_{+k}  (muon)")
        ax.axhline(0, color="k", lw=0.4)
        ax.set_xlabel("k"); ax.set_ylabel("coefficient")
        ax.set_title("tau and muon coefficients")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # (c) tuple kind
        ax = fig.add_subplot(gs[0, 2])
        kinds = [t.kind for t in self.tuples]
        colors = {"quadratic": "#16a085",
                  "linear":    "#f1c40f",
                  "trivial":   "#95a5a6"}
        ax.bar(k_arr, [1] * K,
               color=[colors[kd] for kd in kinds],
               edgecolor="k", linewidth=0.3)
        ax.set_yticks([])
        ax.set_xlabel("k"); ax.set_ylabel("kind")
        ax.set_title(f"Q_k kind   "
                     f"(q={self.n_quad}, l={self.n_lin}, "
                     f"t={self.n_triv})")
        ax.grid(alpha=0.3, axis="x")

        # (d) discriminant Δ_k
        ax = fig.add_subplot(gs[1, 0])
        ax.plot(k_arr, self.disc, "o-", ms=3, lw=1.0,
                color="#8e44ad")
        ax.axhline(0, color="k", ls="--", lw=0.6,
                   label="Δ = 0 (double root)")
        ax.fill_between(k_arr, 0, self.disc, where=self.disc > 0,
                        color="#16a085", alpha=0.15,
                        label="Δ > 0 → 2 real roots")
        ax.fill_between(k_arr, 0, self.disc, where=self.disc < 0,
                        color="#e74c3c", alpha=0.15,
                        label="Δ < 0 → 2 complex roots")
        ax.set_xlabel("k"); ax.set_ylabel("Δ_k = b_k² − 4·a_k·c₀")
        ax.set_title("discriminant  ·  real vs complex roots")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # (e) |root| scatter  ·  sf_k colours
        ax = fig.add_subplot(gs[1, 1])
        for t in self.tuples:
            r = t.roots
            if r is None:
                continue
            color = "#16a085" if t.sf_k == 1 else "#95a5a6"
            ax.scatter(r[0].real, r[0].imag, color=color, s=30,
                       edgecolors="k", linewidths=0.3)
            ax.scatter(r[1].real, r[1].imag, color=color, s=30,
                       marker="s", edgecolors="k", linewidths=0.3)
        ax.axhline(0, color="k", lw=0.4)
        ax.axvline(0, color="k", lw=0.4)
        ax.set_xlabel("Re x"); ax.set_ylabel("Im x")
        ax.set_title("tuple roots  ·  green = square-free k")
        ax.grid(alpha=0.3)

        # (f) signal-of-signals
        ax = fig.add_subplot(gs[1, 2])
        sos = self.signal_of_signals()
        ax.bar(["Σ a_k·sf_k", "Σ b_k·sf_k", "Σ c₀·sf_k"],
               sos, color=["#e74c3c", "#3498db", "#f1c40f"],
               edgecolor="k")
        ax.axhline(0, color="k", lw=0.4)
        ax.set_ylabel("aggregate")
        ax.set_title("signal-of-signals  Σ_k sf_k · (a, b, c₀)")
        for i, v in enumerate(sos):
            ax.text(i, v, f"{v:+.3f}",
                    ha="center", va="bottom" if v > 0 else "top",
                    fontsize=8)
        ax.grid(alpha=0.3, axis="y")

        # (g) spectral density ρ(t)  ·  real + imaginary
        ax = fig.add_subplot(gs[2, :2])
        ax.plot(t_grid, rho.real, color="#3498db", lw=1.2,
                label="Re ρ(t)")
        ax.plot(t_grid, rho.imag, color="#e74c3c", lw=1.2,
                label="Im ρ(t)")
        ax.plot(t_grid, np.abs(rho), color="#2c3e50", lw=0.7,
                ls="--", alpha=0.6, label="|ρ(t)|")
        ax.axhline(0, color="k", lw=0.4)
        ax.set_xlabel("t"); ax.set_ylabel("ρ(t)")
        ax.set_title(
            "ρ(t) = Σ_k sf_k · Q_k( e^{i·t·k/α_k} )")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

        # (h) summary
        ax = fig.add_subplot(gs[2, 2])
        ax.axis("off")
        txt = (
            f"K                = {K}\n"
            f"sf density       = {self.sf_density:.6f}\n"
            f"6/π²             = {DENSITY:.6f}\n"
            f"|Δ density|      = {abs(self.sf_density - DENSITY):.6f}\n"
            f"\n"
            f"quadratic tuples = {self.n_quad}\n"
            f"linear tuples    = {self.n_lin}\n"
            f"trivial tuples   = {self.n_triv}\n"
            f"\n"
            f"Σ a_k·sf_k = {sos[0]:+.4f}\n"
            f"Σ b_k·sf_k = {sos[1]:+.4f}\n"
            f"Σ c₀·sf_k  = {sos[2]:+.4f}\n"
            f"\n"
            f"Σ roots    = {self.sf_weighted_root_sum():+.4f}\n"
        )
        ax.text(0.0, 1.0, txt, va="top", family="monospace",
                fontsize=9)

        plt.suptitle(
            f"Tuple quadratics  ·  tau × muon × electron  ·  "
            f"square-free signal  sf_k = |μ(k)|  ·  "
            f"α = {ALPHA:.4f}  ·  A_STEP = {A_STEP:.4f}",
            fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 76)
    print("Tuple quadratic  ·  tau × muon × electron  ·  square-free")
    print("=" * 76)
    print(f"  α      = 1/(π − e)    = {ALPHA:.6f}")
    print(f"  A_STEP                = {A_STEP:.6f}")
    print(f"  6/π²                  = {DENSITY:.6f}")
    print()

    # ---------- 1 · raw Möbius (sf_k ⇔ quadratic) ----------
    print("=" * 76)
    print("1  ·  RAW  ·  sf_k = 0 ⇔ trivial,  sf_k = 1 ⇔ quadratic")
    print("=" * 76)
    tq = TupleQuadraticSignal(K=64, J_C=1.0, smooth=False)
    print(tq.report())
    print()
    print("  first 16 tuples:")
    print(f"  {'k':>3}  {'a_k':>8}  {'b_k':>8}  {'c₀':>6}  "
          f"{'sf':>3}  {'kind':<9}  {'Δ_k':>10}  roots")
    print("  " + "-" * 90)
    for t in tq.tuples[:16]:
        r = t.roots
        root_s = ("—" if r is None
                  else f"{r[0].real:+.3f}{r[0].imag:+.3f}i")
        print(f"  {t.k:>3d}  {t.a:>+8.3f}  {t.b:>+8.3f}  "
              f"{t.c:>+6.3f}  {t.sf_k:>3d}  {t.kind:<9s}  "
              f"{t.discriminant:>+10.3f}  {root_s}")
    print()

    # ---------- 2 · signal-of-signals ----------
    print("=" * 76)
    print("2  ·  signal-of-signals  Σ_k sf_k · (a_k, b_k, c₀)")
    print("=" * 76)
    sos = tq.signal_of_signals()
    print(f"  Σ a_k·sf_k = {sos[0]:+.6f}   "
          f"(tau-weighted, square-free only)")
    print(f"  Σ b_k·sf_k = {sos[1]:+.6f}   "
          f"(muon-weighted, square-free only)")
    print(f"  Σ c₀·sf_k  = {sos[2]:+.6f}   "
          f"(electron × sf count)")
    print(f"  sf count   = {tq.sf_total}  "
          f"({tq.sf_density:.4f} of {tq.K})")
    print(f"  6/π²       = {DENSITY:.6f}   "
          f"(theory: K → ∞ limit)")
    print()

    # ---------- 3 · spectral density ρ(t) ----------
    print("=" * 76)
    print("3  ·  spectral density  ρ(t) = Σ_k sf_k · Q_k(e^{i t k / α_k})")
    print("=" * 76)
    print(f"  {'t':>8}  {'Re ρ':>12}  {'Im ρ':>12}  {'|ρ|':>10}")
    print("  " + "-" * 48)
    for t in (0.0, 1.0, 2.0, PI / 2, PI):
        rho = tq.spectral_density(np.array([t]))[0]
        print(f"  {t:>8.3f}  {rho.real:>+12.6f}  {rho.imag:>+12.6f}  "
              f"{abs(rho):>10.6f}")
    print()

    # ---------- 4 · smoothed variant ----------
    print("=" * 76)
    print("4  ·  SMOOTHED  ·  fractional coefficients, sf_k unchanged")
    print("=" * 76)
    tq_s = TupleQuadraticSignal(K=64, J_C=1.0, smooth=True)
    print(f"  sf density          : {tq_s.sf_density:.6f}")
    print(f"  n_quad / n_lin / n_triv : "
          f"{tq_s.n_quad} / {tq_s.n_lin} / {tq_s.n_triv}")
    print(f"  Σ a_k·sf_k          : {tq_s.signal_of_signals()[0]:+.6f}")
    print(f"  Σ b_k·sf_k          : {tq_s.signal_of_signals()[1]:+.6f}")
    print()
    print("  smoothed tuples (first 8):")
    print(f"  {'k':>3}  {'a_k':>9}  {'b_k':>9}  {'sf':>3}  "
          f"{'kind':<9}  {'Δ_k':>10}")
    print("  " + "-" * 56)
    for t in tq_s.tuples[:8]:
        print(f"  {t.k:>3d}  {t.a:>+9.4f}  {t.b:>+9.4f}  "
              f"{t.sf_k:>3d}  {t.kind:<9s}  "
              f"{t.discriminant:>+10.4f}")
    print()

    # ---------- 5 · K-convergence of sf density ----------
    print("=" * 76)
    print("5  ·  sf density convergence  ·  → 6/π² ≈ 0.607927")
    print("=" * 76)
    print(f"  {'K':>6}  {'sf_density':>12}  {'|Δ| vs 6/π²':>14}")
    print("  " + "-" * 38)
    for K in (16, 32, 64, 128, 256, 512, 1000):
        tq_k = TupleQuadraticSignal(K=K, smooth=False)
        d = tq_k.sf_density
        print(f"  {K:>6d}  {d:>12.6f}  {abs(d - DENSITY):>14.6f}")
    print()

    # ---------- 6 · summary ----------
    print("=" * 76)
    print("6  ·  summary")
    print("=" * 76)
    print("  tuple (a_k, b_k, c₀)  =  (σ_τ·c_{−k},  σ_μ·c_{+k},  c_K)")
    print("  quadratic              Q_k(x) = a_k·x² + b_k·x + c₀")
    print("  sf signal              sf_k = |μ(k)| ∈ {0, 1}")
    print("  pure Möbius            sf_k = 0 ⇔ trivial,  1 ⇔ quadratic")
    print("  density (Fourier)      ρ(t) = Σ_k sf_k · Q_k(e^{i·t·k/α_k})")
    print("  sf density → 6/π² = 1/ζ(2)  (Basel)")
    print("  discriminant           Δ_k = b_k² − 4·a_k·c₀")
    print("  roots (complex)        x_± = (−b_k ± √Δ_k)/(2·a_k)")
    print()

    tq.visualize()


if __name__ == "__main__":
    demo()