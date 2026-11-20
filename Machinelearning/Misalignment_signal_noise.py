#!/usr/bin/env python3
"""
bit_precision_drift_simulation.py
==================================

Signal-to-noise, floor-to-ceiling, and statistical drift as a
function of bit precision (8, 16, 32, 64, 512) and iteration count.

Physics
-------
    Δ_b      = 2·range / 2^b              quantization step
    σ_q(b)   = Δ_b / √12                  per-op rounding error
    σ_n(b,n) = √n · σ_q(b)                unbiased accumulation
    σ_n(b,n) = n · Δ_b / 2                biased accumulation

    SNR(b, n)              = signal / σ_n(b, n)
    floor_ceil_ratio(b)    = 2^b          (fixed by design)
    effective_bits(b, n)   = b − ½·log2(n)    unbiased
    effective_bits(b, n)   = b − log2(n)      biased

Cross-over  (SNR < 1 → irreversible sign flips)
    n_cross(b) = (1/σ_q(b))²               unbiased   ∝ 2^(2b)
    n_cross(b) = 2/Δ_b                     biased     ∝ 2^b

Bit budget  (adds up over iterations)
    total_bits_spent(n, b) = n · b
    useful_bits(n, b)      = b − ½·log2(n)   or   b − log2(n)
    efficiency(n, b)       = useful / total  →  1/n

Connection to misalignment
--------------------------
    P(misalign) = Φ(−SNR) = ½ · erfc(SNR/√2)

    Each token carried at b bits injects σ_q(b) of noise per
    iteration.  Over N tokens and n iterations the drift variance
    accumulates as N · n · σ_q(b)².  The effective NL-token
    equivalent of a bit-precision run of length n is:

        N_equiv(b, n)  =  n · (σ_q(b) / σ_nl)²

    At 8-bit vs 64-bit at n=100:
        σ_q(8) / σ_q(64)  ≈  4.7e18
        σ_q(8)²/ σ_q(64)² ≈  2.2e37

    → 8-bit carries ~37 decimal digits more drift than 64-bit at
       the same iteration count.
"""

from __future__ import annotations
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ---- reuse the α / density constants and exponential kernel ----
from agents_natural_language_mis_align import (
    ExponentialKernel,
    simulate_chain,
)

PI      = math.pi
E       = math.e
ALPHA   = 1.0 / (PI - E)          # ≈ 2.362338
DENSITY = 6.0 / (PI * PI)         # ≈ 0.607927

BITS_LIST = [8, 16, 32, 64, 512]


# ============================================================
#  Core physics
# ============================================================
def quantize(x: np.ndarray, bits: int,
             x_range: float = 1.0) -> np.ndarray:
    """Round x to `bits`-bit precision on [-x_range, +x_range]."""
    step = 2.0 * x_range / (2 ** bits)
    return np.round(x / step) * step


def step_size(bits: int, x_range: float = 1.0) -> float:
    return 2.0 * x_range / (2 ** bits)


def sigma_q(bits: int, x_range: float = 1.0) -> float:
    """Per-op rounding error (uniform quantization)."""
    return step_size(bits, x_range) / math.sqrt(12.0)


def noise_floor(bits: int, n_iter: int,
                x_range: float = 1.0,
                biased: bool = False) -> float:
    """Accumulated noise after n_iter iterations at `bits` precision."""
    step = step_size(bits, x_range)
    if biased:
        return n_iter * step / 2.0
    return math.sqrt(n_iter) * step / math.sqrt(12.0)


def snr(bits: int, n_iter: int,
        signal: float = 1.0,
        x_range: float = 1.0,
        biased: bool = False) -> float:
    return signal / max(noise_floor(bits, n_iter, x_range, biased), 1e-300)


def floor_ceiling_ratio(bits: int) -> float:
    """Ceiling / floor of the representation = 2^bits."""
    return float(2 ** bits)


def effective_bits(bits: int, n_iter: int,
                   biased: bool = False) -> float:
    """Precision remaining after n_iter iterations."""
    if n_iter <= 1:
        return float(bits)
    if biased:
        return bits - math.log2(n_iter)
    return bits - 0.5 * math.log2(n_iter)


def cross_over_iterations(bits: int,
                          threshold_snr: float = 1.0,
                          x_range: float = 1.0,
                          biased: bool = False) -> float:
    """n_iter at which SNR drops below threshold."""
    step = step_size(bits, x_range)
    if biased:
        return threshold_snr * 2.0 / step
    return (threshold_snr / (step / math.sqrt(12.0))) ** 2


def bit_budget(bits: int, n_iter: int,
               biased: bool = False) -> Dict[str, float]:
    """Total bits spent vs useful bits retained."""
    total = n_iter * bits
    useful = max(0.0, effective_bits(bits, n_iter, biased))
    wasted = total - useful * n_iter
    eff = useful / bits if bits > 0 else 0.0
    return dict(total_bits=total, useful_bits=useful,
                wasted_bits=wasted, efficiency=eff)


def misalign_probability(snr_value: float) -> float:
    """P(misalign) = Φ(−SNR) = ½·erfc(SNR/√2)."""
    return 0.5 * math.erfc(snr_value / math.sqrt(2.0))


def nl_token_equivalent(bits: int, n_iter: int,
                        sigma_nl: float = 0.5) -> float:
    """Equivalent NL-token count carrying the same drift variance."""
    s = sigma_q(bits)
    return n_iter * (s / max(sigma_nl, 1e-12)) ** 2


# ============================================================
#  Numerical experiment  ·  chaotic logistic-map drift
# ============================================================
def logistic_drift(bits: int, n_iter: int,
                   x0: float = 0.4, r: float = 3.9,
                   x_range: float = 1.0) -> np.ndarray:
    """
    Iterate x → r·x·(1−x) at `bits`-bit precision.
    Chaotic map amplifies any quantization error exponentially
    (Lyapunov exponent λ ≈ 0.5 for r = 3.9).
    """
    xs = np.zeros(n_iter + 1)
    xs[0] = x0
    step = 2.0 * x_range / (2 ** bits)
    for i in range(n_iter):
        x = r * xs[i] * (1.0 - xs[i])
        xs[i + 1] = round(x / step) * step
    return xs


def divergence_iteration(bits: int, ref: np.ndarray,
                         test: np.ndarray,
                         tol: float = 1e-2) -> int:
    """First iteration where |test − ref| > tol."""
    for i in range(len(ref)):
        if abs(test[i] - ref[i]) > tol:
            return i
    return -1


# ============================================================
#  Plot
# ============================================================
def make_plots(save: bool = False):
    n_grid = np.unique(np.round(np.logspace(0, 6, 200)).astype(int))

    fig = plt.figure(figsize=(16, 11))
    gs  = GridSpec(3, 3, figure=fig, hspace=0.44, wspace=0.32)

    colors = {8: "#e74c3c", 16: "#e67e22", 32: "#f1c40f",
              64: "#16a085", 512: "#2980b9"}

    # (a) SNR vs n_iter  ·  unbiased
    ax = fig.add_subplot(gs[0, 0])
    for b in BITS_LIST:
        sn = np.array([snr(b, int(n)) for n in n_grid])
        ax.loglog(n_grid, sn, color=colors[b], lw=1.5, label=f"{b}-bit")
    ax.axhline(1.0, color="k", ls=":", lw=0.9, label="SNR = 1")
    ax.set_xlabel("iteration count n")
    ax.set_ylabel("SNR  (unbiased rounding)")
    ax.set_title("(a) SNR ∝ n^(−1/2)  ·  floor grows as √n")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    # (b) SNR vs n_iter  ·  biased
    ax = fig.add_subplot(gs[0, 1])
    for b in BITS_LIST:
        sn = np.array([snr(b, int(n), biased=True) for n in n_grid])
        ax.loglog(n_grid, sn, color=colors[b], lw=1.5, label=f"{b}-bit")
    ax.axhline(1.0, color="k", ls=":", lw=0.9, label="SNR = 1")
    ax.set_xlabel("iteration count n")
    ax.set_ylabel("SNR  (biased truncation)")
    ax.set_title("(b) SNR ∝ n^(−1)  ·  floor grows as n")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    # (c) effective bits vs n_iter (unbiased)
    ax = fig.add_subplot(gs[0, 2])
    for b in BITS_LIST:
        eb = np.array([effective_bits(b, int(n)) for n in n_grid])
        ax.semilogx(n_grid, eb, color=colors[b], lw=1.5,
                    label=f"{b}-bit")
    ax.set_xlabel("iteration count n")
    ax.set_ylabel("effective bits")
    ax.set_title("(c) effective = b − ½·log2(n)")
    ax.set_ylim(-4, 520)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (d) floor-to-ceiling ratio vs bits (log-scale bar)
    ax = fig.add_subplot(gs[1, 0])
    ratios = [floor_ceiling_ratio(b) for b in BITS_LIST]
    ax.bar([str(b) for b in BITS_LIST], ratios,
           color=[colors[b] for b in BITS_LIST], edgecolor="k")
    ax.set_yscale("log")
    ax.set_ylabel("ceiling / floor  =  2^bits")
    ax.set_title("(d) floor-to-ceiling ratio scales as 2^b")
    for i, (b, r) in enumerate(zip(BITS_LIST, ratios)):
        ax.text(i, r, f"2^{b}\n≈{r:.0e}",
                ha="center", va="bottom", fontsize=8)
    ax.grid(alpha=0.3, axis="y", which="both")

    # (e) cross-over iteration count vs bits
    ax = fig.add_subplot(gs[1, 1])
    nc_unb = [cross_over_iterations(b, biased=False) for b in BITS_LIST]
    nc_bias = [cross_over_iterations(b, biased=True) for b in BITS_LIST]
    x = np.arange(len(BITS_LIST))
    ax.bar(x - 0.2, nc_unb, 0.4, label="unbiased", color="#16a085")
    ax.bar(x + 0.2, nc_bias, 0.4, label="biased", color="#e74c3c")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([str(b) for b in BITS_LIST])
    ax.set_xlabel("bit precision")
    ax.set_ylabel("iterations until SNR < 1")
    ax.set_title("(e) cross-over  ∝ 2^(2b) unbiased,  2^b biased")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y", which="both")

    # (f) efficiency: useful / total bits
    ax = fig.add_subplot(gs[1, 2])
    for b in [8, 32, 64]:
        eff = np.array([
            bit_budget(b, int(n))["efficiency"]
            for n in np.arange(1, 1001)
        ])
        ax.loglog(np.arange(1, 1001), eff, color=colors[b], lw=1.5,
                  label=f"{b}-bit")
    ax.set_xlabel("iteration count n")
    ax.set_ylabel("useful bits / total bits spent")
    ax.set_title("(f) efficiency → 1/n  ·  register budget wasted")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    # (g) numerical drift  ·  logistic map
    ax = fig.add_subplot(gs[2, 0])
    n_exp = 200
    ref = logistic_drift(64, n_exp)         # 64-bit as reference
    for b in [8, 16, 32]:
        traj = logistic_drift(b, n_exp)
        div = divergence_iteration(b, ref, traj, tol=1e-2)
        label = f"{b}-bit (div @ {div})" if div > 0 else f"{b}-bit (no div)"
        ax.semilogy(np.abs(traj - ref) + 1e-18, color=colors[b],
                    lw=1.2, label=label)
    ax.axhline(1e-2, color="k", ls=":", lw=0.9, label="drift threshold")
    ax.set_xlabel("iteration n")
    ax.set_ylabel("|x_b − x_64|")
    ax.set_title("(g) logistic map  ·  r=3.9  ·  chaotic amplification")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    # (h) misalignment probability vs iterations
    ax = fig.add_subplot(gs[2, 1])
    for b in [8, 16, 32, 64]:
        ps = np.array([misalign_probability(snr(b, int(n)))
                       for n in n_grid])
        ax.semilogx(n_grid, ps, color=colors[b], lw=1.5,
                    label=f"{b}-bit")
    ax.axhline(0.5, color="k", ls=":", lw=0.9)
    ax.set_xlabel("iteration count n")
    ax.set_ylabel("P(misalign) = Φ(−SNR)")
    ax.set_title("(h) misalignment probability")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (i) NL-token equivalent  ·  bit precision vs NL drift
    ax = fig.add_subplot(gs[2, 2])
    n_fix = 100
    eq = [nl_token_equivalent(b, n_fix) for b in BITS_LIST]
    ax.bar([str(b) for b in BITS_LIST], eq,
           color=[colors[b] for b in BITS_LIST], edgecolor="k")
    ax.set_yscale("log")
    ax.set_xlabel("bit precision")
    ax.set_ylabel("N_equiv  (NL tokens of drift)")
    ax.set_title(f"(i) drift in NL-token units  ·  n={n_fix}")
    for i, (b, v) in enumerate(zip(BITS_LIST, eq)):
        if v > 0:
            ax.text(i, v, f"{v:.2e}",
                    ha="center", va="bottom", fontsize=8)
    ax.grid(alpha=0.3, axis="y", which="both")

    plt.suptitle(
        "Bit-precision drift  ·  SNR = signal / √n·σ_q(b)  ·  "
        "floor/ceiling = 2^b  ·  cross-over ∝ 2^(2b)  ·  "
        f"α = {ALPHA:.4f}  ·  6/π² = {DENSITY:.4f}",
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save:
        plt.savefig("bit_precision_drift.png", dpi=140)
        print("saved bit_precision_drift.png")
    else:
        plt.show()


# ============================================================
#  Demo
# ============================================================
def demo(save_plot: bool = False):
    print("=" * 80)
    print("Bit-precision drift  ·  SNR, floor/ceiling, statistical divergence")
    print("=" * 80)
    print(f"  α = 1/(π − e)     = {ALPHA:.6f}")
    print(f"  6/π²              = {DENSITY:.6f}")
    print()

    # ---------------- 1. per-bit quanta ----------------
    print("=" * 80)
    print("1  ·  per-bit quanta  (range = 1.0 → Δ = 2/2^b)")
    print("=" * 80)
    print(f"  {'bits':>5s}  {'Δ (step)':>14s}  {'σ_q = Δ/√12':>16s}  "
          f"{'floor/ceil':>14s}  {'dB':>8s}")
    print("  " + "-" * 68)
    for b in BITS_LIST:
        d = step_size(b)
        s = sigma_q(b)
        r = floor_ceiling_ratio(b)
        db = 20.0 * math.log10(r) if r < 1e300 else 20.0 * b * math.log10(2)
        print(f"  {b:>5d}  {d:>14.6e}  {s:>16.6e}  "
              f"{r:>14.3e}  {db:>8.1f}")
    print()

    # ---------------- 2. SNR vs iterations ----------------
    print("=" * 80)
    print("2  ·  SNR(bits, n)  ·  unbiased accumulation (√n growth)")
    print("=" * 80)
    header = "  ".join(f"{b:>10d}-bit" for b in BITS_LIST)
    print(f"  {'n':>6s}  {header}")
    print("  " + "-" * 72)
    for n in [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000]:
        row = [snr(b, n) for b in BITS_LIST]
        print(f"  {n:>6d}  " +
              "  ".join(f"{v:>14.4e}" for v in row))
    print()

    # ---------------- 3. biased vs unbiased ----------------
    print("=" * 80)
    print("3  ·  biased (truncation) vs unbiased (round-to-nearest)")
    print("=" * 80)
    print(f"  {'bits':>5s}  {'SNR @n=1000':>16s}  {'SNR @n=1000':>16s}  "
          f"{'ratio':>10s}")
    print(f"  {'':>5s}  {'(unbiased)':>16s}  {'(biased)':>16s}  "
          f"{'unb/bia':>10s}")
    print("  " + "-" * 56)
    for b in BITS_LIST:
        s_u = snr(b, 1000, biased=False)
        s_b = snr(b, 1000, biased=True)
        print(f"  {b:>5d}  {s_u:>16.4e}  {s_b:>16.4e}  "
              f"{s_u/s_b:>10.4f}")
    print()

    # ---------------- 4. effective bits ----------------
    print("=" * 80)
    print("4  ·  effective bits remaining after n iterations")
    print("=" * 80)
    print(f"  {'n':>8s}  {'8-bit':>8s}  {'16-bit':>8s}  {'32-bit':>8s}  "
          f"{'64-bit':>8s}  {'512-bit':>8s}")
    print("  " + "-" * 62)
    for n in [1, 10, 100, 1_000, 10_000, 1_000_000]:
        row = [effective_bits(b, n) for b in BITS_LIST]
        print(f"  {n:>8d}  " +
              "  ".join(f"{v:>8.2f}" for v in row))
    print()

    # ---------------- 5. cross-over iterations ----------------
    print("=" * 80)
    print("5  ·  cross-over  ·  n until SNR < 1")
    print("=" * 80)
    print(f"  {'bits':>5s}  {'unbiased':>18s}  {'biased':>18s}")
    print("  " + "-" * 46)
    for b in BITS_LIST:
        nc_u = cross_over_iterations(b, biased=False)
        nc_b = cross_over_iterations(b, biased=True)
        print(f"  {b:>5d}  {nc_u:>18.4e}  {nc_b:>18.4e}")
    print()
    print("  scaling:")
    print("    unbiased  n_cross(b2)/n_cross(b1) = 2^(2·(b2−b1))")
    print("    biased    n_cross(b2)/n_cross(b1) = 2^(b2−b1)")
    print(f"    biased 8→64 ratio  = 2^56 = {2.0**56:.3e}")
    print(f"    unbiased 8→64 ratio = 2^112 = {2.0**112:.3e}")
    print()

    # ---------------- 6. bit budget ----------------
    print("=" * 80)
    print("6  ·  bit budget  ·  total bits spent vs useful bits retained")
    print("=" * 80)
    print(f"  {'n':>8s}  {'8-bit eff':>12s}  {'32-bit eff':>12s}  "
          f"{'64-bit eff':>12s}  {'512-bit eff':>12s}")
    print("  " + "-" * 62)
    for n in [1, 10, 100, 1_000, 10_000]:
        row = [bit_budget(b, n)["efficiency"] for b in [8, 32, 64, 512]]
        print(f"  {n:>8d}  " +
              "  ".join(f"{v:>12.4f}" for v in row))
    print()

    # ---------------- 7. misalignment probability ----------------
    print("=" * 80)
    print("7  ·  P(misalign) = Φ(−SNR)")
    print("=" * 80)
    print(f"  {'n':>6s}  " +
          "  ".join(f"{b:>10s}-bit" for b in [8, 16, 32, 64, 512]))
    print("  " + "-" * 72)
    for n in [10, 100, 1_000, 10_000, 100_000]:
        row = [misalign_probability(snr(b, n)) for b in BITS_LIST]
        print(f"  {n:>6d}  " +
              "  ".join(f"{v:>14.4e}" for v in row))
    print()

    # ---------------- 8. NL-token equivalent ----------------
    print("=" * 80)
    print("8  ·  NL-token equivalent of bit-precision drift")
    print("=" * 80)
    print(f"  {'bits':>5s}  {'σ_q(b)':>14s}  {'N_equiv @n=100':>20s}  "
          f"{'ratio vs 64-bit':>18s}")
    print("  " + "-" * 64)
    ref64 = nl_token_equivalent(64, 100)
    for b in BITS_LIST:
        s = sigma_q(b)
        neq = nl_token_equivalent(b, 100)
        ratio = neq / max(ref64, 1e-300)
        print(f"  {b:>5d}  {s:>14.4e}  {neq:>20.4e}  "
              f"{ratio:>18.4e}")
    print()

    # ---------------- 9. numerical chaos drift ----------------
    print("=" * 80)
    print("9  ·  logistic map  ·  x → 3.9·x·(1−x)  ·  64-bit reference")
    print("=" * 80)
    print(f"  {'bits':>5s}  {'divergence':>12s}  "
          f"{'|x_b − x_64| @n=100':>22s}")
    print("  " + "-" * 46)
    ref_traj = logistic_drift(64, 200)
    for b in [8, 16, 32]:
        traj = logistic_drift(b, 200)
        d = divergence_iteration(b, ref_traj, traj, tol=1e-2)
        err = abs(traj[100] - ref_traj[100])
        d_str = f"n = {d}" if d > 0 else "never"
        print(f"  {b:>5d}  {d_str:>12s}  {err:>22.4e}")
    print()
    print("  Lyapunov λ ≈ 0.5  ·  rounding error doubles every ~2 iterations")
    print("  once σ_q(b) · exp(λ·n) > 1, the trajectory is fully decorrelated")
    print()

    # ---------------- 10. interpretation ----------------
    print("=" * 80)
    print("10  ·  interpretation")
    print("=" * 80)
    print("  · Each bit doubles the floor/ceiling ratio  (2^b).")
    print("  · Unbiased accumulation: SNR ∝ n^(−1/2), cross-over ∝ 2^(2b).")
    print("  · Biased (truncation):   SNR ∝ n^(−1),   cross-over ∝ 2^b.")
    print("  · 8-bit biased diverges in ~256 iterations.")
    print("    64-bit biased diverges in ~2^63 ≈ 9.2e18 iterations.")
    print("  · Useful / total bits spent falls as 1/n  →  register budget")
    print("    is wasted linearly with iteration count.")
    print("  · Misalignment Φ(−SNR) reaches 0.5 once SNR < 1; after that,")
    print("    each next iteration amplifies the drift super-exponentially.")
    print()

    # ---------------- 11. plots ----------------
    make_plots(save=save_plot)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    demo(save_plot=False)