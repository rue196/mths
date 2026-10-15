#!/usr/bin/env python3
"""
fermion_coupler_jet.py
======================

Extends chiral_jet_engine.py with:

  · Fermion force calculus (Theorem 13 of supertrace_6_kernel.pdf)
  · Discrete point particles in transcendental time
  · Vertical contraction (Re F) vs horizontal projection (Im F)
  · Coupler speed matching at the small entrance (apex)
  · Codimension scaling of density / viscosity / mass (Theorem 15)

Physical layout
---------------
                   ╱╲
                  ╱  ╲                    VERTICAL CONTRACTION
                 ╱    ╲                   spin up/down couples air
                ╱      ╲                  F_real = −Σ |g| cos φ
               ╱  ▲▲▲▲  ╲                 → fast, high-density
              ╱   ││││   ╲
             ╱    ▼▼▼▼    ╲
            ╱______________╲   ← coupler (z = 0, R = R_tip)
            │  ══════════  │              HORIZONTAL PROJECTION
            │              │              F_imag = Σ |g| sin φ
            │   EXHAUST    │              ω_ij(t) = α(j−i) dζ/dt
            ╰──────────────╯              → 6D → 3D reduction
                    ↓

Fermion force (Theorem 13)
    F_i = −∇_{Z_i} Σ_j |g_ij| cos(φ_ij) + i Σ_j |g_ij| sin(φ_ij)
    φ_ij = χ_i − χ_j
    ω_ij(t) = α (j − i) dζ/dt(t)
    ⟨ν_d⟩ = ν_0 + c_d Σ |g_ij|² ω_ij²          (Corollary 14)

Discrete point particles
    i       = discrete index
    j       = transcendental index
    x       = Re v(i) = −H_{i+1}                discrete position
    y       = Im v(j) = α (μ*H)(j)              transcendental time
    vx      = dRe/di                            finite-step velocity
    c       = 0 if cos χ ≥ 0 else 1              binary chiral spin

Coupler speed matching
    at z = z_c:  |v_z| = |v_r|                  (magnitude preserved)
    downstream:  v_r · A_jet = v_z · A_coupler  (flux conservation)

Codimension scaling (Theorem 15)
    ρ_d1/ρ_d2 = ν_d1/ν_d2 = m_d1/m_d2
              = ((6−d1)/(6−d2))² · V_d2/V_d1
    raised during contraction, projected horizontally
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ============================================================
#  Import the base jet engine
# ============================================================
try:
    from chiral_jet_engine import (
        PI, E, ALPHA_SYM, DENSITY,
        H_CONE, W_BASE, R_BASE, R_TIP, CONCAVE_POWER,
        N_BLADES, BLADE_R0, BLADE_R1, CYL_Z0, CYL_Z1,
        N_PISTONS, OMEGA, EXHAUST_L, EXHAUST_W,
        pi_6, chirality_angle, quantized_chirality,
        chiral_bond_weight, supertrace_from_spinors,
        entropy, invariant_mass,
        A_of_n, finite_step,
        r_wall, make_blade_spinor,
        make_12_vertex, pi_12_6,
        ChiralBladeArray, BladeConfig,
        hurricane_exhaust_field, piston_state,
    )
    _HAS_BASE = True
except ImportError:
    _HAS_BASE = False
    PI = math.pi; E = math.e
    ALPHA_SYM = 1.0 / (PI - E)
    DENSITY = 6.0 / (PI * PI)
    H_CONE = 3.0; W_BASE = 1.0; R_BASE = 0.5; R_TIP = 0.08
    CONCAVE_POWER = 0.6
    N_BLADES = 24; BLADE_R0 = 0.52; BLADE_R1 = 0.70
    CYL_Z0 = -0.55; CYL_Z1 = -0.05
    N_PISTONS = 8; OMEGA = 2 * PI
    EXHAUST_L = 1.6; EXHAUST_W = 0.35

    def pi_6(Z):
        return complex(np.linalg.det(Z))

    def chirality_angle(Z):
        Pi = pi_6(Z)
        return math.atan2(Pi.imag, Pi.real)

    def chiral_bond_weight(ZA, ZB):
        d = abs(chirality_angle(ZA) - chirality_angle(ZB))
        if d > PI:
            d = 2 * PI - d
        return d / PI

    def r_wall(z):
        z = np.asarray(z, dtype=float)
        zn = np.clip(z / H_CONE, 0.0, 1.0)
        return R_TIP + (R_BASE - R_TIP) * (1.0 - zn ** 2) ** CONCAVE_POWER

    class ChiralBladeArray:
        def __init__(self, N_blades=N_BLADES, pitch_max=PI/6, seed=0):
            self.N = N_blades
            rng = np.random.default_rng(seed)
            self.spinors = []
            for i in range(N_blades):
                Z = (rng.standard_normal((6, 6))
                     + 1j * rng.standard_normal((6, 6)))
                Z = Z / (np.max(np.abs(Z)) + 1e-12) * 0.5
                Z = Z - Z.mean()
                theta = 2 * PI * i / N_blades
                self.spinors.append(Z * np.exp(1j * theta))
            self.chiralities = np.array(
                [chirality_angle(Z) for Z in self.spinors])
            self.supertrace = sum(
                (1 if k % 2 == 0 else -1) * abs(pi_6(Z))
                for k, Z in enumerate(self.spinors))
            self.mass = abs(self.supertrace) * math.exp(-0.01)


# ============================================================
#  Möbius / harmonic tables
# ============================================================
def mobius_sieve(K: int) -> np.ndarray:
    mu = np.zeros(K + 1, dtype=np.int8)
    if K >= 1:
        mu[1] = 1
    primes, is_comp = [], np.zeros(K + 1, dtype=bool)
    for i in range(2, K + 1):
        if not is_comp[i]:
            primes.append(i); mu[i] = -1
        for p in primes:
            if i * p > K:
                break
            is_comp[i * p] = True
            if i % p == 0:
                mu[i * p] = 0; break
            mu[i * p] = -mu[i]
    return mu


def harmonic_numbers(K: int) -> np.ndarray:
    H = np.zeros(K + 1, dtype=float)
    if K >= 1:
        H[1] = 1.0
    for n in range(2, K + 1):
        H[n] = H[n - 1] + 1.0 / n
    return H


def mu_convolution_H(K: int):
    mu = mobius_sieve(K)
    H  = harmonic_numbers(K)
    F  = np.zeros(K + 1, dtype=float)
    for d in range(1, K + 1):
        if mu[d] == 0:
            continue
        for m in range(1, K // d + 1):
            F[d * m] += mu[d] * H[m]
    return F, mu, H


# ============================================================
#  Discrete point particles in transcendental time
# ============================================================
@dataclass
class DiscreteParticle:
    i: int                              # discrete index
    j: int                              # transcendental index
    x: float                            # Re v(i) = −H_{i+1}
    y: float                            # Im v(j) = α (μ*H)(j)
    vx: float                           # finite-step velocity
    spin: int                           # binary chiral c ∈ {0,1}
    chi: float                          # χ = arg Π₆(Z)
    F_real: float = 0.0                 # vertical contraction
    F_imag: float = 0.0                 # horizontal projection
    omega: float = 0.0                  # local fermionic frequency


def make_particle_cloud(n_particles: int, K: int = 256,
                        seed: int = 42) -> List[DiscreteParticle]:
    """Generate `n_particles` discrete particles at (i, j)."""
    rng = np.random.default_rng(seed)
    F, mu, H = mu_convolution_H(K)
    out = []
    for k in range(n_particles):
        i = (k % K) + 1
        j = ((k + 7) % K) + 1
        x = -float(H[i])
        y = ALPHA_SYM * float(F[j])
        vx = -(float(H[min(i + 1, K)]) - float(H[max(i - 1, 1)])) / 2.0
        Z = (rng.standard_normal((6, 6))
             + 1j * rng.standard_normal((6, 6)))
        Pi = pi_6(Z)
        chi = math.atan2(Pi.imag, Pi.real)
        spin = 0 if math.cos(chi) >= 0 else 1
        out.append(DiscreteParticle(
            i=i, j=j, x=x, y=y, vx=vx, spin=spin, chi=chi))
    return out


# ============================================================
#  Fermion force field  (Theorem 13)
# ============================================================
def fermion_force_field(particles: List[DiscreteParticle],
                        spinors: List[np.ndarray],
                        dzeta_dt: float = 0.5
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the fermionic force on each particle.

        F_i = −Σ_j |g_ij| cos(φ_ij) + i Σ_j |g_ij| sin(φ_ij)
        φ_ij = χ_i − χ_j
        ω_ij(t) = α (j − i) dζ/dt(t)

    Returns (F_real, F_imag, omega):
        Re F → vertical    (conservative contraction)
        Im F → horizontal  (oscillatory projection)
    """
    N = len(particles)
    F_real = np.zeros(N)
    F_imag = np.zeros(N)
    omega  = np.zeros(N)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            Zi = spinors[i % len(spinors)]
            Zj = spinors[j % len(spinors)]
            w = chiral_bond_weight(Zi, Zj)
            if w < 0.1:
                continue
            phi = particles[i].chi - particles[j].chi
            # conservative part  → vertical contraction
            F_real[i] -= w * math.cos(phi)
            # oscillatory part  → horizontal projection
            F_imag[i] += w * math.sin(phi)
            # local fermionic frequency
            omega[i]  += ALPHA_SYM * (j - i) * dzeta_dt
    return F_real, F_imag, omega


# ============================================================
#  Coupler at the small entrance  ·  speed matching
# ============================================================
@dataclass
class Coupler:
    """
    Coupler at the apex of the cone.

        z_c  = axial position
        R_c  = coupler radius (small = R_tip)
        R_j  = jet radius downstream
        L_j  = jet length
    """
    z_c: float = 0.0
    R_c: float = R_TIP
    R_j: float = R_BASE + 0.1
    L_j: float = EXHAUST_L

    @property
    def A_c(self) -> float:
        """Coupler cross-section (small, vertical)."""
        return PI * self.R_c ** 2

    @property
    def A_j(self) -> float:
        """Jet cross-section (large, horizontal)."""
        return 2.0 * PI * self.R_j * self.L_j

    def v_vertical(self, omega_mean: float) -> float:
        """
        Vertical speed through the coupler:
            v_z = α · ω / R_c  (smaller coupler → faster)
        """
        return ALPHA_SYM * abs(omega_mean) / max(self.R_c, 1e-6)

    def v_horizontal_coupler(self, v_z: float) -> float:
        """Horizontal speed at the coupler exit (magnitude preserved)."""
        return v_z                              # |v| conserved at 90° rotation

    def v_horizontal_jet(self, v_z: float) -> float:
        """Horizontal speed at the jet exit (flux conservation)."""
        return v_z * self.A_c / max(self.A_j, 1e-9)

    def match(self, omega_mean: float) -> Dict:
        v_z = self.v_vertical(omega_mean)
        return dict(
            v_z            = v_z,
            v_r_coupler    = self.v_horizontal_coupler(v_z),
            v_r_jet        = self.v_horizontal_jet(v_z),
            ratio_coupler  = 1.0,
            ratio_jet      = self.v_horizontal_jet(v_z) / max(v_z, 1e-9),
            A_c            = self.A_c,
            A_j            = self.A_j,
            area_ratio     = self.A_c / self.A_j,
        )


# ============================================================
#  Theorem 15  ·  codimension scaling
# ============================================================
def scaling_ratio(d1: float, d2: float,
                  V_d1: float = 1.0, V_d2: float = 1.0) -> float:
    """
    ρ_d1/ρ_d2 = ν_d1/ν_d2 = m_d1/m_d2
              = ((6−d1)/(6−d2))² · V_d2/V_d1

    Uses the (6−d1)/(6−d2) form so that ρ_3/ρ_6 → ∞ (Corollary 16).
    """
    if d2 == 6:
        return float("inf")
    return ((6.0 - d1) / (6.0 - d2)) ** 2 * (V_d2 / V_d1)


def unit_ball_volume(d: int) -> float:
    """Volume of the unit ball in R^d."""
    return math.pi ** (d / 2.0) / math.gamma(d / 2.0 + 1.0)


# ============================================================
#  Vertical contraction & horizontal projection
# ============================================================
def vertical_contraction(particles: List[DiscreteParticle],
                         F_real: np.ndarray) -> np.ndarray:
    """Vertical component of the fermion force (down the cone axis)."""
    return F_real.copy()


def horizontal_projection(particles: List[DiscreteParticle],
                          F_imag: np.ndarray) -> np.ndarray:
    """Horizontal component of the fermion force (radially outward)."""
    return F_imag.copy()


def propagate_particles(particles: List[DiscreteParticle],
                        F_real: np.ndarray,
                        F_imag: np.ndarray,
                        coupler: Coupler,
                        dt: float = 0.05) -> List[Dict]:
    """
    Propagate particles through the coupler.

    Below z_c: vertical contraction (F_real drives them in).
    At z_c:    90° velocity rotation, |v| preserved.
    Above z_c: horizontal projection (F_imag sprays them out).
    """
    trace = []
    for k, p in enumerate(particles):
        # vertical speed from the vertical force
        v_z = F_real[k] * dt
        # at the coupler: rotate 90°, magnitude preserved
        v_r = abs(v_z)  # v_r = v_z at coupler
        # downstream expansion (flux conservation)
        v_r_jet = coupler.v_horizontal_jet(abs(v_z))
        trace.append(dict(
            i=p.i, j=p.j, spin=p.spin,
            chi=p.chi, F_real=F_real[k], F_imag=F_imag[k],
            v_z=v_z, v_r_coupler=v_r, v_r_jet=v_r_jet,
        ))
    return trace


# ============================================================
#  Visualization
# ============================================================
def plot_coupler_jet(particles, F_real, F_imag, omega,
                     coupler, trace):
    fig = plt.figure(figsize=(16, 11))
    gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

    N = len(particles)
    x = np.array([p.x for p in particles])
    y = np.array([p.y for p in particles])
    spins = np.array([p.spin for p in particles])
    chis  = np.array([p.chi for p in particles])

    # (a) particle cloud · spin colouring
    ax = fig.add_subplot(gs[0, 0])
    colors = ["#16a085" if s == 0 else "#e74c3c" for s in spins]
    ax.scatter(x, y, c=colors, s=40, edgecolors="k", linewidth=0.4)
    ax.set_xlabel("x = Re v(i)   discrete position")
    ax.set_ylabel("y = Im v(j)   transcendental time")
    ax.set_title(f"Discrete particles   N = {N}\n"
                 f"green spin-up (c=0)   red spin-down (c=1)")
    ax.grid(alpha=0.3)

    # (b) fermion force field  ·  Re vs Im
    ax = fig.add_subplot(gs[0, 1])
    scale_r = 0.35 / (np.abs(F_real).max() + 1e-9)
    scale_i = 0.35 / (np.abs(F_imag).max() + 1e-9)
    for k in range(N):
        ax.arrow(x[k], y[k], scale_r * F_real[k], 0,
                 head_width=0.02, head_length=0.02,
                 color="#e67e22", alpha=0.75, length_includes_head=True)
        ax.arrow(x[k], y[k], 0, scale_i * F_imag[k],
                 head_width=0.02, head_length=0.02,
                 color="#3498db", alpha=0.75, length_includes_head=True)
    ax.scatter(x, y, c="k", s=4, alpha=0.5)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Fermion force field  (Theorem 13)\n"
                 "orange = Re F  vertical   ·   blue = Im F  horizontal")
    ax.grid(alpha=0.3)

    # (c) coupler speed matching
    ax = fig.add_subplot(gs[0, 2])
    z_axis = np.linspace(0.02, H_CONE, 200)
    R_axis = r_wall(z_axis)
    omega_mean = abs(omega).mean() + 1e-6
    v_z_axis = ALPHA_SYM * omega_mean / np.maximum(R_axis, 1e-6)
    v_r_axis = v_z_axis * coupler.A_c / coupler.A_j
    ax.loglog(z_axis, v_z_axis, color="#e67e22", lw=2.0,
              label="v_z   vertical (contraction)")
    ax.loglog(z_axis, v_r_axis, color="#3498db", lw=2.0,
              label="v_r   horizontal (projection)")
    ax.axvline(coupler.z_c + 1e-3, color="k", ls=":", lw=1.0,
               label=f"coupler  z={coupler.z_c}")
    ax.set_xlabel("z"); ax.set_ylabel("speed")
    ax.set_title("Coupler speed matching\n"
                 "v_z · A_c = v_r · A_j    (flux conservation)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    # (d) codimension scaling  (Theorem 15)
    ax = fig.add_subplot(gs[1, 0])
    d_vals = np.arange(1, 7)
    ratios = []
    for d in d_vals:
        V_3 = unit_ball_volume(3)
        V_d = unit_ball_volume(d)
        if d == 3:
            ratios.append(1.0)
        elif d == 6:
            ratios.append(np.nan)  # ρ_3/ρ_6 → ∞
        else:
            ratios.append(scaling_ratio(3, d, V_3, V_d))
    ax.plot(d_vals, ratios, "o-", color="#8e44ad", lw=1.8)
    ax.axhline(1.0, color="k", ls=":", lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("dimension d")
    ax.set_ylabel(r"$\rho_3 / \rho_d$  (scaling)")
    ax.set_title("Codimension scaling  (Theorem 15)\n"
                 r"$\rho_{d_1}/\rho_{d_2} = \left(\frac{6-d_1}{6-d_2}\right)^2 \frac{V_{d_2}}{V_{d_1}}$")
    ax.grid(alpha=0.3, which="both")

    # (e) chiral population
    ax = fig.add_subplot(gs[1, 1])
    n_up = int(np.sum(spins == 0))
    n_dn = int(np.sum(spins == 1))
    ax.bar([0, 1], [n_up, n_dn],
           color=["#16a085", "#e74c3c"], edgecolor="k", width=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["spin up\nc = 0", "spin down\nc = 1"])
    ax.set_ylabel("count")
    ax.set_title(f"Binary chiral population\n"
                     f"imbalance |c\u2080\u2212c\u2081| = {abs(n_up - n_dn)} / {N}")
    ax.grid(alpha=0.3, axis="y")

    # (f) chirality distribution
    ax = fig.add_subplot(gs[1, 2])
    ax.hist(chis, bins=32, range=(-PI, PI),
            color="#8e44ad", edgecolor="k", alpha=0.85)
    ax.axvline(-PI / 2, color="k", ls=":", lw=0.8)
    ax.axvline( PI / 2, color="k", ls=":", lw=0.8)
    ax.set_xlabel(r"$\chi = \arg \Pi_6(Z)$")
    ax.set_ylabel("count")
    ax.set_title(f"Chirality angle distribution\n"
                 f"mean = {chis.mean():+.4f}")
    ax.grid(alpha=0.3)

    plt.suptitle(
        "Fermion coupler jet  ·  vertical contraction (Re F)  →  "
        "coupler  →  horizontal projection (Im F)  ·  "
        f"α = {ALPHA_SYM:.4f}",
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 82)
    print("Fermion coupler jet  ·  contraction vs projection")
    print("=" * 82)
    print(f"  α        = 1/(π − e)    = {ALPHA_SYM:.6f}")
    print(f"  H_cone                  = {H_CONE}")
    print(f"  R_tip                   = {R_TIP}")
    print(f"  R_base                  = {R_BASE}")
    print()

    # ---------- 1. chiral blade array ----------
    blades = ChiralBladeArray(N_blades=N_BLADES, seed=0)
    print(f"[1] chiral blade array  ·  N = {N_BLADES}")
    print(f"      supertrace S       = {blades.supertrace:+.6f}")
    print(f"      invariant mass m   = {blades.mass:.6e}")
    print()

    # ---------- 2. discrete particles ----------
    particles = make_particle_cloud(n_particles=64, K=256)
    spins = np.array([p.spin for p in particles])
    chis  = np.array([p.chi for p in particles])
    print(f"[2] discrete particle cloud  ·  N = {len(particles)}")
    print(f"      spin up   (c = 0)  = {int(np.sum(spins == 0))}")
    print(f"      spin down (c = 1)  = {int(np.sum(spins == 1))}")
    print(f"      |c₀ − c₁|          = {abs(int(np.sum(spins == 0)) - int(np.sum(spins == 1)))}")
    print(f"      ⟨χ⟩                = {chis.mean():+.4f}")
    print()

    # ---------- 3. fermion force ----------
    F_real, F_imag, omega = fermion_force_field(
        particles, blades.spinors, dzeta_dt=0.5)
    for k, p in enumerate(particles):
        p.F_real = F_real[k]
        p.F_imag = F_imag[k]
        p.omega  = omega[k]
    print(f"[3] fermion force field  (Theorem 13)")
    print(f"      Re F  (vertical)   = [{F_real.min():+.4f}, {F_real.max():+.4f}]")
    print(f"      Im F  (horizontal) = [{F_imag.min():+.4f}, {F_imag.max():+.4f}]")
    print(f"      ω                  = [{omega.min():+.4f}, {omega.max():+.4f}]")
    print(f"      ⟨ν_d⟩ − ν_0        = "
          f"{ALPHA_SYM * np.sum((np.abs(F_real) + np.abs(F_imag)) ** 2 * omega ** 2):.4e}  "
          f"(Corollary 14)")
    print()

    # ---------- 4. coupler speed matching ----------
    coupler = Coupler(z_c=0.0, R_c=R_TIP, R_j=R_BASE + 0.1,
                      L_j=EXHAUST_L)
    omega_mean = float(np.abs(omega).mean())
    match = coupler.match(omega_mean)
    print(f"[4] coupler  ·  z = {coupler.z_c}   R_c = {coupler.R_c:.4f}")
    print(f"      A_c / A_j          = {match['area_ratio']:.6f}")
    print(f"      v_z  (vertical)    = {match['v_z']:.6f}")
    print(f"      v_r  (at coupler)  = {match['v_r_coupler']:.6f}   "
          f"(ratio = {match['ratio_coupler']:.6f})")
    print(f"      v_r  (at jet)      = {match['v_r_jet']:.6f}   "
          f"(ratio = {match['ratio_jet']:.6f})")
    print()

    # ---------- 5. codimension scaling ----------
    print(f"[5] codimension scaling  (Theorem 15)")
    V_3 = unit_ball_volume(3)
    for d in [3, 4, 5]:
        V_d = unit_ball_volume(d)
        r = scaling_ratio(3, d, V_3, V_d)
        print(f"      ρ_3/ρ_{d}  =  {r:.6f}")
    print(f"      ρ_3/ρ_6  →  ∞   (Corollary 16, degenerate limit)")
    print()

    # ---------- 6. propagation trace ----------
    trace = propagate_particles(particles, F_real, F_imag, coupler)
    v_z_arr = np.array([t["v_z"] for t in trace])
    v_r_arr = np.array([t["v_r_coupler"] for t in trace])
    print(f"[6] propagation through the coupler")
    print(f"      ⟨v_z⟩  = {v_z_arr.mean():+.6f}")
    print(f"      ⟨v_r⟩  = {v_r_arr.mean():+.6f}")
    print(f"      |⟨v_z⟩ − ⟨v_r⟩| = "
          f"{abs(v_z_arr.mean() - v_r_arr.mean()):.6e}   (≈ 0)")
    print()

    # ---------- 7. visualize ----------
    plot_coupler_jet(particles, F_real, F_imag, omega, coupler, trace)

    # ---------- 8. complexity ----------
    print("--- complexity ---")
    print("  Möbius / harmonic tables     O(K log log K) + O(K)")
    print("  particle cloud               O(N · K)")
    print("  fermion force (Theorem 13)   O(N² · K)   (sparse with threshold)")
    print("  coupler speed match          O(1)")
    print("  propagation trace            O(N)")
    print("  codimension scaling          O(1)")
    print()


if __name__ == "__main__":
    demo()