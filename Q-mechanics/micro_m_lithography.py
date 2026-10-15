#!/usr/bin/env python3
"""
micro_lithography.py
====================
Micrometer lithography via coupled fermions.

Physics stack
-------------
1. Elliptic projection weight  (Figure 3.5, receiver_module.py)
       Π(t, r) = ½(1 + cos(2πt/π)·cos(2πr/e)) ∈ [0, 1]

2. Point-particle trajectory
       dx/dτ = dζ/dt · Π(t, r)               (spectral derivative drives motion)

3. Coupling at short range
       g_ij = w_ij · κ_ij
       w_ij = |χ_i − χ_j| / π                chiral bond weight
       κ_ij = √(L_i L_j) / L_ref             short-distance travel coupling

4. Magnetic field with parity
       B(r) = B0 · [even-mask] + B1 · [odd-mask]
       even parity → focus (attracting)
       odd parity  → deflect (repelling)

5. Even-even mask
       allowed scattering sites  ⇔  c_i = c_j = 0
       excluded scattering sites ⇔  c_i ≠ c_j

6. Hertz energy per site (Theorem 13, supertrace_6_kernel.pdf)
       H_i = Σ_j |g_ij · ω_ij|,    ω_ij = α(j−i) dζ/dt

7. Fiber dispersion of the exposure dose
       D(z) = (D_in * exp(−α|z|))
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ============================================================
#  Constants
# ============================================================
PI       = math.pi
E        = math.e
ALPHA    = 1.0 / (PI - E)               # ≈ 2.362338
A_STEP   = ALPHA / 0.3628               # ≈ 6.511
DENSITY  = 6.0 / (PI * PI)              # ≈ 0.607927
PHI      = (1.0 + math.sqrt(5.0)) / 2.0
W1, W2   = PI, E

# lithography-specific
UM       = 1e-6                          # 1 micrometer
NM       = 1e-9                          # 1 nanometer
CELL_UM  = 0.5                           # pixel pitch in micrometers
L_REF_NM = 30.0                          # reference length scale for κ_ij


# ============================================================
#  Figure 3.5 · elliptic projection
# ============================================================
def elliptic_projection(t: np.ndarray,
                        r: np.ndarray) -> np.ndarray:
    u = (t / W1) % 1.0
    v = (r / W2) % 1.0
    return 0.5 * (1.0 + np.cos(2 * PI * u) * np.cos(2 * PI * v))


# ============================================================
#  Chiral bond weight & short-distance coupling
# ============================================================
def chiral_angles(t: np.ndarray) -> np.ndarray:
    lt = np.log(np.maximum(np.abs(t), 1e-12))
    return (lt + PI) % (2 * PI) - PI


def travel_lengths_mm(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cumulative arclength along the beam path, in nm."""
    dx = np.diff(x, prepend=x[0]) * UM / NM      # convert μm → nm
    dy = np.diff(y, prepend=y[0]) * UM / NM
    ds = np.sqrt(dx * dx + dy * dy)
    ds[0] = 0.0
    return np.cumsum(ds) + 1e-6


# ============================================================
#  Magnetic field with even/odd parity
# ============================================================
@dataclass
class MagneticField:
    """
    Even-parity sites attract the coupled fermion (focusing).
    Odd-parity sites repel (deflecting).

        B(r) = +B0    if site label c_i = 0   (bosonic / even)
               -B1    if site label c_i = 1   (fermionic / odd)
    """
    B_even: float = 1.0
    B_odd:  float = -1.0
    def strength(self, c: np.ndarray) -> np.ndarray:
        return np.where(c == 0, self.B_even, self.B_odd)


def lorentz_force(vx: np.ndarray, vy: np.ndarray,
                  c: np.ndarray,
                  B: MagneticField) -> Tuple[np.ndarray, np.ndarray]:
    """
    F = q v × B  →  in 2D with B along ẑ:
        Fx = −v_y · B(c)
        Fy = +v_x · B(c)
    """
    Bz = B.strength(c)
    return -vy * Bz, vx * Bz


# ============================================================
#  Short-distance coupling  κ_ij = √(L_i L_j) / L_ref
# ============================================================
def short_distance_kappa(L: np.ndarray,
                         L_ref_nm: float = L_REF_NM) -> np.ndarray:
    """
    κ_ij = √(L_i L_j) / L_ref ∈ [0, 1]     (Definition 5,
    supertrace_6_kernel.pdf)

    L_i is the cumulative path length; for micrometer lithography
    we use the *short-range* regime so that L_ref is on the nm scale.
    """
    Lp = np.maximum(L, 1e-9)
    K = np.sqrt(np.outer(Lp, Lp)) / L_ref_nm
    return np.clip(K, 0.0, 1.0)


def coupling_matrix(chis: np.ndarray,
                    L: np.ndarray,
                    L_ref_nm: float = L_REF_NM) -> np.ndarray:
    """
    g_ij = w_ij · κ_ij, with
        w_ij = |χ_i − χ_j| / π
        κ_ij = √(L_i L_j) / L_ref
    """
    dchi = np.abs(chis[:, None] - chis[None, :])
    dchi = (dchi + PI) % (2 * PI) - PI
    W = np.abs(dchi) / PI
    K = short_distance_kappa(L, L_ref_nm)
    G = W * K
    np.fill_diagonal(G, 0.0)
    return G


# ============================================================
#  Even-even mask  ·  allowed scattering sites
# ============================================================
def even_even_mask(N: int) -> np.ndarray:
    i = np.arange(N)[:, None]
    j = np.arange(N)[None, :]
    return (i % 2 == 0) & (j % 2 == 0)


# ============================================================
#  Dynamic Hertz energy per site  (Theorem 13)
# ============================================================
def hertz_energy(G: np.ndarray, dzeta_dt: float) -> np.ndarray:
    """
    H_i = Σ_j |g_ij · ω_ij|,  ω_ij = α(j − i) dζ/dt
    """
    N = G.shape[0]
    j = np.arange(N)
    omega = ALPHA * (j[None, :] - j[:, None]) * dzeta_dt
    return np.sum(np.abs(G * omega), axis=1)


# ============================================================
#  dζ/dt  ·  spectral derivative  ·  point particle trajectory
# ============================================================
def dzeta_dt(t: float, coeffs: np.ndarray,
             alpha: float = ALPHA) -> float:
    """
    dζ/dt = −Σ_{j≠0} (j/α) Re(C_j e^{ijt/α})
    """
    K = (len(coeffs) - 1) // 2
    j = np.arange(-K, K + 1)
    jnz = j[j != 0]
    Cj = coeffs[jnz + K]
    phases = jnz * t / alpha
    return float(-np.sum((jnz / alpha) * np.real(Cj * np.exp(1j * phases))))


def build_coefficients(N: int, seed: int = 42) -> np.ndarray:
    """Random bounded coefficients for the spectral sum."""
    rng = np.random.default_rng(seed)
    C = (rng.standard_normal(2 * N + 1)
         + 1j * rng.standard_normal(2 * N + 1)) / (N ** 0.5)
    return C


# ============================================================
#  Beam propagation with magnetic + coupling forces
# ============================================================
@dataclass
class BeamState:
    """One fermion's state at time τ."""
    tau: float
    x: float
    y: float
    vx: float
    vy: float
    c: int                     # quantized chirality
    chi: float                 # chiral angle


def propagate_beam(N_particles: int = 64,
                   n_steps: int = 120,
                   dt: float = 0.05,
                   seed: int = 42) -> Dict:
    """
    Propagate N_particles coupled fermions through the lithography
    chamber under:
        · dζ/dt trajectory (spectral derivative)
        · magnetic field (even / odd parity)
        · short-distance coupling κ_ij = √(L_i L_j) / L_ref
        · even-even mask (allowed scattering)
    """
    rng = np.random.default_rng(seed)

    # --- spectral coefficients for dζ/dt ---
    C = build_coefficients(N_particles, seed)

    # --- initial positions & velocities (micrometer scale) ---
    x = rng.uniform(-100 * UM, 100 * UM, N_particles)
    y = rng.uniform(-100 * UM, 100 * UM, N_particles)
    vx = rng.normal(0, 1e-2, N_particles)
    vy = rng.normal(0, 1e-2, N_particles)

    # --- chirality per particle ---
    chis = rng.uniform(-PI, PI, N_particles)
    # quantized chirality
    c = (np.cos(chis) < 0).astype(int)     # c = 1 if cos χ < 0

    # --- histories ---
    x_hist = np.zeros((n_steps + 1, N_particles))
    y_hist = np.zeros((n_steps + 1, N_particles))
    H_hist = np.zeros((n_steps + 1, N_particles))
    G_hist = np.zeros((n_steps + 1, N_particles))

    x_hist[0] = x
    y_hist[0] = y

    B = MagneticField(B_even=1.0, B_odd=-1.0)

    for step in range(n_steps):
        tau = step * dt

        # 1. trajectory from dζ/dt
        zt = dzeta_dt(tau, C)

        # 2. elliptic projection weight Π(t, r) at current position
        t_norm = tau * A_STEP
        r_norm = np.sqrt(x ** 2 + y ** 2) / (100 * UM)
        Pi = elliptic_projection(t_norm, r_norm)

        # 3. Lorentz force from magnetic field with parity
        Fx_mag, Fy_mag = lorentz_force(vx, vy, c, B)

        # 4. trajectory-driven velocity
        vx += dt * (zt * Pi * 1e-3 + Fx_mag * 1e-4)
        vy += dt * (zt * Pi * 1e-3 + Fy_mag * 1e-4)

        # 5. damping (energy loss to the resist)
        vx *= (1.0 - 0.02 * dt)
        vy *= (1.0 - 0.02 * dt)

        # 6. position update
        x += vx * dt
        y += vy * dt

        # 7. coupling matrix on the current path
        L = travel_lengths_mm(x, y)
        G = coupling_matrix(chis, L, L_REF_NM)

        # 8. even-even mask applied
        mask = even_even_mask(N_particles)
        G_masked = np.where(mask, G, 0.0)

        # 9. Hertz energy per site
        H = hertz_energy(G_masked, dzeta_dt=abs(zt))

        # record
        x_hist[step + 1] = x
        y_hist[step + 1] = y
        H_hist[step + 1] = H
        G_hist[step + 1] = np.diagonal(G_masked)

    return dict(
        x_hist=x_hist, y_hist=y_hist,
        H_hist=H_hist, G_hist=G_hist,
        c=c, chis=chis,
        n_steps=n_steps, dt=dt,
    )


# ============================================================
#  Fiber dispersion of exposure dose
# ============================================================
def fiber_kernel(width: int = 41, alpha: float = ALPHA) -> np.ndarray:
    x = np.arange(-width // 2, width // 2)
    h = np.exp(-alpha * np.abs(x))
    return h / h.sum()


def fiber_disperse(dose: np.ndarray, width: int = 41) -> np.ndarray:
    h = fiber_kernel(width)
    L = len(dose)
    N = 1 << (2 * L - 1).bit_length()
    sig_pad = np.pad(dose, (0, N - L))
    ker_pad = np.pad(h, (0, N - len(h)))
    return np.real(np.fft.ifft(
        np.fft.fft(sig_pad) * np.fft.fft(ker_pad)))[:L]


# ============================================================
#  Visualization
# ============================================================
def plot_lithography(result: Dict, N_particles: int = 64):
    x_hist = result["x_hist"] / UM
    y_hist = result["y_hist"] / UM
    H_hist = result["H_hist"]
    G_hist = result["G_hist"]
    c      = result["c"]
    chis   = result["chis"]

    fig = plt.figure(figsize=(16, 11))
    gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

    # (a) Figure 3.5 · elliptic projection surface
    ax = fig.add_subplot(gs[0, 0])
    t_vals = np.linspace(0, W1, 200)
    r_vals = np.linspace(0, W2, 200)
    T, R = np.meshgrid(t_vals, r_vals)
    P = elliptic_projection(T, R)
    im = ax.imshow(P, extent=[0, W1, 0, W2], origin="lower",
                   cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xlabel("t (mod π)"); ax.set_ylabel("r (mod e)")
    ax.set_title(r"Figure 3.5 · Π(t, r) ∈ [0, 1]")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (b) beam trajectories
    ax = fig.add_subplot(gs[0, 1])
    colors = np.where(c == 0, "#16a085", "#e74c3c")
    for i in range(N_particles):
        ax.plot(x_hist[:, i], y_hist[:, i],
                color=colors[i], lw=0.7, alpha=0.65)
    ax.scatter(x_hist[0, :], y_hist[0, :], c="gray",
               s=15, edgecolors="k", label="start")
    ax.scatter(x_hist[-1, :], y_hist[-1, :], c=colors,
               s=25, edgecolors="k", zorder=5, label="end")
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    ax.set_title("Beam trajectories\n"
                 "green = even c=0 (focus)   red = odd c=1 (deflect)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_aspect("equal")

    # (c) chirality distribution
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(chis, bins=24, range=(-PI, PI),
            color="#8e44ad", edgecolor="k", alpha=0.85)
    ax.axvline(-PI / 2, color="k", ls=":", lw=0.8)
    ax.axvline( PI / 2, color="k", ls=":", lw=0.8)
    ax.set_xlabel(r"$\chi$")
    ax.set_ylabel("count")
    ax.set_title(f"Chirality distribution\n"
                 f"c=0 (even) = {int(np.sum(c==0))}  "
                 f"c=1 (odd) = {int(np.sum(c==1))}")
    ax.grid(alpha=0.3)

    # (d) Hertz energy per site over time
    ax = fig.add_subplot(gs[1, 0])
    for i in range(min(8, N_particles)):
        ax.plot(H_hist[:, i], color=colors[i], lw=0.9, alpha=0.8)
    ax.set_xlabel("step"); ax.set_ylabel(r"$H_i$")
    ax.set_title("Hertz energy per site (Theorem 13)")
    ax.grid(alpha=0.3)

    # (e) coupling strength g_ii over time
    ax = fig.add_subplot(gs[1, 1])
    for i in range(min(8, N_particles)):
        ax.plot(G_hist[:, i], color=colors[i], lw=0.9, alpha=0.8)
    ax.set_xlabel("step"); ax.set_ylabel(r"$g_{ii}$")
    ax.set_title("Diagonal coupling strength\n"
                 r"$g_{ij} = w_{ij} \cdot \kappa_{ij}$, "
                 r"$\kappa_{ij} = \sqrt{L_i L_j} / L_{\rm ref}$")
    ax.grid(alpha=0.3)

    # (f) total Hertz energy + fiber dose
    ax = fig.add_subplot(gs[1, 2])
    H_total = H_hist.sum(axis=1)
    dose = fiber_disperse(H_total, width=41)
    ax.plot(H_total, color="#2c3e50", lw=1.4,
            label="total Hertz")
    ax.plot(dose, color="#e67e22", lw=1.4, ls="--",
            label="fiber dose  exp(−α|z|)")
    ax.set_xlabel("step"); ax.set_ylabel("energy / dose")
    ax.set_title("Exposure dose after fiber dispersion\n"
                 f"‖H‖ = {np.linalg.norm(H_total):.3f}   "
                 f"‖D‖ = {np.linalg.norm(dose):.3f}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.suptitle(
        "Micrometer lithography  ·  coupled fermions  ·  "
        "dζ/dt trajectory  ·  even/odd parity magnetic steering  ·  "
        "even-even mask scattering",
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 82)
    print("Micrometer lithography  ·  coupled fermion beam")
    print("=" * 82)
    print(f"  α       = 1/(π − e)      = {ALPHA:.6f}")
    print(f"  A_STEP  = α / 0.3628     = {A_STEP:.6f}")
    print(f"  6/π²                     = {DENSITY:.6f}  (μ density)")
    print(f"  L_ref                    = {L_REF_NM:.1f} nm  (short range)")
    print()

    t0 = time.perf_counter()
    result = propagate_beam(N_particles=64, n_steps=120, dt=0.05, seed=42)
    dt_ms = (time.perf_counter() - t0) * 1e3

    N = result["x_hist"].shape[1]
    c = result["c"]
    chis = result["chis"]

    print(f"--- beam summary ---")
    print(f"  N particles             = {N}")
    print(f"  even c = 0 (focused)    = {int(np.sum(c == 0))}")
    print(f"  odd  c = 1 (deflected)  = {int(np.sum(c == 1))}")
    print(f"  ⟨χ⟩                     = {chis.mean():+.4f}")
    print(f"  propagation time        = {dt_ms:.2f} ms")
    print()

    # Hertz energy statistics
    H_final = result["H_hist"][-1]
    print(f"--- Hertz energy at final step ---")
    print(f"  min                     = {H_final.min():.4f}")
    print(f"  max                     = {H_final.max():.4f}")
    print(f"  mean                    = {H_final.mean():.4f}")
    print()

    # coupling statistics
    G_final = result["G_hist"][-1]
    off = G_final[G_final > 0]
    if off.size:
        print(f"--- diagonal coupling at final step ---")
        print(f"  ⟨g_ii⟩                  = {off.mean():.6f}")
        print(f"  max g_ii                = {G_final.max():.6f}")
        print()

    # fiber dose
    H_total = result["H_hist"].sum(axis=1)
    dose = fiber_disperse(H_total, width=41)
    print(f"--- fiber dose ---")
    print(f"  ‖H_total‖               = {np.linalg.norm(H_total):.4f}")
    print(f"  ‖dose‖                  = {np.linalg.norm(dose):.4f}")
    print(f"  ‖dose‖ / ‖H_total‖      = "
          f"{np.linalg.norm(dose) / np.linalg.norm(H_total):.4f}")
    print()

    # visualize
    plot_lithography(result, N_particles=N)

    # complexity
    print("--- complexity ---")
    print("  dζ/dt spectral derivative   O(K)")
    print("  elliptic projection          O(N)")
    print("  magnetic force               O(N)")
    print("  coupling matrix  g_ij        O(N²)")
    print("  even-even mask               O(N²)")
    print("  Hertz energy per step        O(N²)")
    print("  fiber dispersion             O(N log N)")
    print()


if __name__ == "__main__":
    demo()