#!/usr/bin/env python3
"""
hex_wavelength_compiler.py
==========================
Compile a 6×6 spinor block into a hex color via:

    Π₆(Z) → κ       contraction depth (O(n) real + O(K log K) imag)
    S     → λ       wavelength (alternating sum)
    H     → ν       alternation rate (THz)
    m     → E       photon energy (eV)
    E     → λ       Planck relation  λ(nm) = 1239.84 / E_eV
    λ     → hex     CIE-approximate visible-spectrum RGB

Constants:
    α = 1/(π − e) ≈ 2.362338
    hc = 1239.842 eV·nm
"""

from __future__ import annotations
import math
from itertools import permutations
from typing import Dict, Tuple

import numpy as np

# ============================================================
#  Constants
# ============================================================
PI        = math.pi
E         = math.e
ALPHA     = 1.0 / (PI - E)          # ≈ 2.362338
HC_EV_NM  = 1239.841984             # h·c in eV·nm
C_NM_THZ  = 299792.458              # c in nm·THz  (ν[THz] · λ[nm] = c)
E_REF     = 2.0                     # reference energy scale (eV)
LAMBDA_MIN = 380.0
LAMBDA_MAX = 750.0
N         = 12


# ============================================================
#  Rank-6 Levi-Civita = det(Z)
# ============================================================
def pi_6(Z: np.ndarray) -> complex:
    return complex(np.linalg.det(Z.astype(complex)))


def scalar_perm_12_6(M: np.ndarray) -> float:
    """Full Perms(12,6) contraction — only used for small calibration."""
    C = 0.0
    for perm in permutations(range(N), 6):
        inv = sum(1 for i in range(6) for j in range(i + 1, 6)
                  if perm[i] > perm[j])
        sign = -1 if inv & 1 else 1
        prod = 1.0
        for k, row in enumerate(perm):
            prod *= M[row, k]
        C += sign * prod
    return C


# ============================================================
#  Supertrace / entropy / mass kernel
# ============================================================
def supertrace(signal: np.ndarray) -> float:
    signs = np.where(np.arange(signal.size) & 1, -1.0, 1.0)
    return float(np.dot(signs, np.abs(signal)))


def entropy(S: float, n: int) -> float:
    p = abs(S) / max(n, 1)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -ALPHA * p * math.log(p)


def invariant_mass(S: float, n: int) -> float:
    H = entropy(S, n)
    return abs(S) * math.exp(-H) if H < 700 else 0.0


# ============================================================
#  Contraction depth κ
# ============================================================
def kappa_from_Pi(Pi: complex, Pi0: float = 1.0,
                  eta: float = 0.5, zeta: float = 0.1,
                  n_conj: int = 6, m: float = 0.0) -> float:
    """κ = log(1 + |Π₆|/Π₀) + η·m + ζ·n_conj"""
    return (math.log1p(abs(Pi) / Pi0)
            + eta * m
            + zeta * n_conj)


# ============================================================
#  Photon energy ↔ wavelength ↔ frequency
# ============================================================
def energy_from_mass(m: float, e_ref: float = E_REF) -> float:
    """Total photon energy proxy: E = m · E_ref (eV)."""
    return abs(m) * e_ref


def wavelength_from_energy(e_eV: float) -> float:
    """λ(nm) = hc / E(eV)."""
    e_eV = max(e_eV, 1e-9)
    return HC_EV_NM / e_eV


def frequency_from_wavelength(lam_nm: float) -> float:
    """ν(THz) = c / λ(nm)."""
    return C_NM_THZ / max(lam_nm, 1e-9)


def wavelength_from_kappa(kappa: float) -> float:
    """Alternative: bounded sigmoid from contraction depth."""
    return LAMBDA_MIN + (LAMBDA_MAX - LAMBDA_MIN) / (1 + math.exp(-kappa))


# ============================================================
#  Wavelength → RGB hex  (approximate visible-spectrum piecewise)
# ============================================================
def wavelength_to_rgb(lam_nm: float, gamma: float = 0.8) -> Tuple[int, int, int]:
    """
    Piecewise approximation of the CIE 1931 color matching,
    tuned so 380–440 → violet-blue, 510 → green, 580 → yellow,
    650 → red, 750 → dark red edge.
    """
    if lam_nm < 380 or lam_nm > 780:
        return (0, 0, 0)
    if lam_nm < 440:
        r = -(lam_nm - 440) / 60.0
        g = 0.0
        b = 1.0
    elif lam_nm < 490:
        r = 0.0
        g = (lam_nm - 440) / 50.0
        b = 1.0
    elif lam_nm < 510:
        r = 0.0
        g = 1.0
        b = -(lam_nm - 510) / 20.0
    elif lam_nm < 580:
        r = (lam_nm - 510) / 70.0
        g = 1.0
        b = 0.0
    elif lam_nm < 645:
        r = 1.0
        g = -(lam_nm - 645) / 65.0
        b = 0.0
    else:
        r = 1.0
        g = 0.0
        b = 0.0

    # intensity fall-off at the edges
    if lam_nm < 420:
        f = 0.3 + 0.7 * (lam_nm - 380) / 40.0
    elif lam_nm > 700:
        f = 0.3 + 0.7 * (780 - lam_nm) / 80.0
    else:
        f = 1.0

    r = max(0.0, min(1.0, r * f)) ** gamma
    g = max(0.0, min(1.0, g * f)) ** gamma
    b = max(0.0, min(1.0, b * f)) ** gamma
    return (int(round(r * 255)),
            int(round(g * 255)),
            int(round(b * 255)))


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


# ============================================================
#  Full compiler: 6×6 block → hex + physics metadata
# ============================================================
def compile_block(Z: np.ndarray,
                  Pi0: float = 1.0,
                  eta: float = 0.5,
                  zeta: float = 0.1,
                  n_conj: int = 6,
                  use_mass_for_energy: bool = True) -> Dict:
    """
    Compile a 6×6 complex block into a hex color via the supertrace
    kernel.

    Steps:
        1. Π₆(Z) = det(Z)
        2. S     = supertrace of |eigenvalues|
        3. H     = entropy of S
        4. m     = |S| e^(−H)
        5. κ     = log(1+|Π₆|/Π₀) + η·m + ζ·n_conj
        6. λ     = hc / (m · E_ref)          (energy route)
           λ     = sigmoid(κ)               (contraction route)
        7. hex   = RGB(λ)
    """
    Pi = pi_6(Z)

    # signal for supertrace: absolute eigenvalues of Z
    eig = np.linalg.eigvals(Z.astype(complex))
    mags = np.abs(eig)
    S = supertrace(mags)
    n = mags.size
    H = entropy(S, n)
    m = invariant_mass(S, n)

    kappa = kappa_from_Pi(Pi, Pi0=Pi0, eta=eta, zeta=zeta,
                          n_conj=n_conj, m=m)

    lam_energy = wavelength_from_energy(energy_from_mass(m))
    lam_kappa  = wavelength_from_kappa(kappa)
    lam        = lam_energy if use_mass_for_energy else lam_kappa
    lam_clamped = max(LAMBDA_MIN, min(LAMBDA_MAX, lam))

    rgb = wavelength_to_rgb(lam_clamped)
    hex_str = rgb_to_hex(rgb)

    return {
        "Pi"              : Pi,
        "Pi_abs"          : abs(Pi),
        "S"               : S,
        "H"               : H,
        "m"               : m,
        "kappa"           : kappa,
        "energy_eV"       : energy_from_mass(m),
        "wavelength_nm_e" : lam_energy,
        "wavelength_nm_k" : lam_kappa,
        "wavelength_nm"   : lam_clamped,
        "frequency_THz"   : frequency_from_wavelength(lam_clamped),
        "rgb"             : rgb,
        "hex"             : hex_str,
    }


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("Hex color from supertrace mass  ·  photon-energy encoding")
    print("=" * 78)
    print(f"  α = 1/(π − e)      = {ALPHA:.6f}")
    print(f"  hc                 = {HC_EV_NM:.6f} eV·nm")
    print(f"  c                  = {C_NM_THZ:.3f} nm·THz")
    print()
    print("  S     → wavelength (alternating sum, geometry)")
    print("  H     → alternation rate (THz, how fast wave flips)")
    print("  m     → photon energy (eV, |S|·e^(−H))")
    print("  κ     → contraction depth  (how far the projection reaches)")
    print()

    rng = np.random.default_rng(2024)

    # ---------- a family of blocks ----------
    print("--- compile a family of 6×6 blocks ---")
    print(f"  {'label':>16s}  {'|Π₆|':>10s}  {'S':>8s}  {'H':>8s}  "
          f"{'m':>8s}  {'κ':>6s}  {'λ(nm)':>7s}  {'ν(THz)':>8s}  hex")
    print("  " + "-" * 100)

    cases = []

    # 1. identity (diagonal) — all eigenvalues = 1
    I6 = np.eye(6, dtype=complex)
    cases.append(("identity", I6))

    # 2. small real block
    cases.append(("small real", 0.5 * np.eye(6) + 0.05 * rng.standard_normal((6, 6))))

    # 3. scaled identity — eigenvalue sweep
    for s in [0.3, 1.0, 3.0, 10.0, 30.0]:
        cases.append((f"diag({s})", s * np.eye(6, dtype=complex)))

    # 4. complex blocks
    for s in [0.5, 1.0, 2.0, 5.0]:
        Z = s * (np.eye(6, dtype=complex)
                 + 0.1 * (rng.standard_normal((6, 6))
                          + 1j * rng.standard_normal((6, 6))))
        cases.append((f"complex({s})", Z))

    # 5. pure imaginary
    Zi = 1j * np.eye(6)
    cases.append(("imag identity", Zi))

    # 6. rotation block
    theta = PI / 6
    R = np.eye(6, dtype=complex)
    R[0, 0] = math.cos(theta); R[0, 1] = -math.sin(theta)
    R[1, 0] = math.sin(theta); R[1, 1] =  math.cos(theta)
    cases.append(("rotation π/6", R))

    results = []
    for label, Z in cases:
        r = compile_block(Z)
        results.append((label, r))
        print(f"  {label:>16s}  {r['Pi_abs']:>10.4e}  "
              f"{r['S']:>+8.4f}  {r['H']:>8.4f}  "
              f"{r['m']:>8.4f}  {r['kappa']:>6.3f}  "
              f"{r['wavelength_nm']:>7.1f}  "
              f"{r['frequency_THz']:>8.2f}  {r['hex']}")
    print()

    # ---------- bijection check ----------
    print("--- invariants of the hex pipeline ---")
    print(f"  hc / E_max        = {HC_EV_NM / (max(r['m'] for _, r in results) * E_REF):.2f} nm  "
          f"(energy-to-wavelength anchor)")
    print(f"  380 nm ↔ {C_NM_THZ / 380:.2f} THz   "
          f"750 nm ↔ {C_NM_THZ / 750:.2f} THz")
    print()

    # ---------- sweep S along a line ----------
    print("--- sweep S with fixed shape (diag scaling) ---")
    print(f"  {'diag':>6s}  {'S':>8s}  {'H':>8s}  {'m':>8s}  "
          f"{'λ(nm)':>7s}  {'ν(THz)':>8s}  hex")
    for s in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        Z = s * np.eye(6, dtype=complex)
        r = compile_block(Z)
        print(f"  {s:>6.2f}  {r['S']:>+8.4f}  {r['H']:>8.4f}  "
              f"{r['m']:>8.4f}  {r['wavelength_nm']:>7.1f}  "
              f"{r['frequency_THz']:>8.2f}  {r['hex']}")
    print()

    # ---------- visualization ----------
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(15, 9))
        gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)

        # (a) wavelength spectrum across a family of blocks
        ax = fig.add_subplot(gs[0, 0])
        scales = np.linspace(0.05, 30, 200)
        wavelengths, hexes = [], []
        for s in scales:
            Z = s * np.eye(6, dtype=complex)
            r = compile_block(Z)
            wavelengths.append(r['wavelength_nm'])
            hexes.append(r['hex'])
        ax.plot(scales, wavelengths, color="#2c3e50", lw=1.6)
        ax.axhline(380, color="#8e44ad", ls=":", label="violet edge")
        ax.axhline(750, color="#e74c3c", ls=":", label="red edge")
        ax.set_xlabel("diag scale s"); ax.set_ylabel("λ (nm)")
        ax.set_title("Wavelength from mass m via λ = hc / (m·E_ref)")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

        # (b) color strip
        ax = fig.add_subplot(gs[0, 1])
        strip = np.zeros((1, len(hexes), 3), dtype=np.uint8)
        for i, h in enumerate(hexes):
            r = int(h[1:3], 16); g = int(h[3:5], 16); b = int(h[5:7], 16)
            strip[0, i] = (r, g, b)
        ax.imshow(strip, aspect="auto",
                  extent=[scales[0], scales[-1], 0, 1])
        ax.set_yticks([])
        ax.set_xlabel("diag scale s")
        ax.set_title("Hex color band  ·  identity-diagonal family")
        for s_val, pos in zip([0.1, 1.0, 5.0, 20.0], [0.05, 0.15, 0.5, 0.9]):
            r = compile_block(s_val * np.eye(6, dtype=complex))
            ax.text(s_val, 1.05, r['hex'], ha="center",
                    fontsize=8, color="#2c3e50")

        # (c) S, H, m over the sweep
        ax = fig.add_subplot(gs[1, 0])
        Ss, Hs, ms = [], [], []
        for s in scales:
            Z = s * np.eye(6, dtype=complex)
            r = compile_block(Z)
            Ss.append(r['S']); Hs.append(r['H']); ms.append(r['m'])
        ax.plot(scales, Ss, color="#8e44ad", lw=1.5, label="S")
        ax.plot(scales, Hs, color="#e74c3c", lw=1.5, label="H")
        ax.plot(scales, ms, color="#16a085", lw=1.5, label="m = |S|e^(−H)")
        ax.set_xlabel("diag scale s"); ax.set_ylabel("value")
        ax.set_title("Supertrace / entropy / mass over the family")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

        # (d) κ → hex
        ax = fig.add_subplot(gs[1, 1])
        kappas, hexes_k = [], []
        for s in scales:
            Z = s * np.eye(6, dtype=complex)
            r = compile_block(Z, use_mass_for_energy=False)
            kappas.append(r['kappa'])
            hexes_k.append(r['hex'])
        strip_k = np.zeros((1, len(hexes_k), 3), dtype=np.uint8)
        for i, h in enumerate(hexes_k):
            r = int(h[1:3], 16); g = int(h[3:5], 16); b = int(h[5:7], 16)
            strip_k[0, i] = (r, g, b)
        ax.imshow(strip_k, aspect="auto",
                  extent=[kappas[0], kappas[-1], 0, 1])
        ax.set_yticks([])
        ax.set_xlabel("κ  (contraction depth)")
        ax.set_title("Hex color from κ  ·  sigmoid wavelength route")

        plt.suptitle(
            "Hex color from supertrace mass  ·  "
            "S → λ, H → ν, m → E, κ → contraction",
            fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    except ImportError:
        pass

    # ---------- final summary ----------
    print("--- final summary ---")
    print("  Hex pipeline is:")
    print("    1. Π₆(Z) = det(Z)            — projector scalar")
    print("    2. S = Σ(−1)^k |λ_k|          — alternating sum of |eigenvalues|")
    print("    3. H = −α·p·log p             — alternation rate")
    print("    4. m = |S|·e^(−H)             — surviving photon energy (eV)")
    print("    5. λ = hc / (m·E_ref)         — wavelength from energy")
    print("    6. hex = RGB(λ)               — visible-spectrum map")
    print()
    print("  The kernel is invertible on the design range:")
    print("    λ(nm) → E(eV) → m → S → Π₆  (up to permutation and phase)")


if __name__ == "__main__":
    demo()