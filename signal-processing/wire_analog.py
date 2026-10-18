#!/usr/bin/env python3
"""
analog_wire_transmission.py
===========================

Analog transmission wire:

    L_i  = ∫_γi √(dt² + dr²)               travel length of wire point i
    κ_ij = √(L_i L_j) / L_ref ∈ [0, 1]     coupling kernel
    G_ij = M_ij · κ_ij · w_ij              even-even mask · chiral weight

    E-field  (Figure 2 of neutral.pdf):
        S_νℓ = Σ_n (−1)^n σ_ℓ(J_C) |Π_6(Z_n)|    ℓ ∈ {μ, τ}
        E(t) = (S_νμ + S_ντ) · cos(ω t)          alternating (AC)

    Electrons as point particles  (Figure 3 of neutral.pdf):
        Π_e(t, r) = ½(1 + cos 2t · cos 2πr/e) ∈ [0, 1]
        drives the electron cloud density along the wire
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from itertools import permutations
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ============================================================
#  Constants
# ============================================================
PI      = math.pi
E       = math.e
ALPHA   = 1.0 / (PI - E)               # ≈ 2.362
NORM    = 1.0 - math.exp(-ALPHA * (PI + E))
PHI     = (1.0 + math.sqrt(5.0)) / 2.0
A_STEP  = ALPHA / 0.3628               # ≈ 6.511


# ============================================================
#  1.  Wire geometry  ·  travel lengths  ·  coupling
# ============================================================
def wire_positions(N: int, length: float = 10.0,
                   wiggle: float = 0.25, turns: float = 3.0
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Parameterized wire (t, r) with sinusoidal wiggle."""
    s = np.linspace(0.0, length, N)
    t = s
    r = wiggle * np.sin(2 * PI * turns * s / length)
    return t, r


def travel_lengths(t: np.ndarray, r: np.ndarray) -> np.ndarray:
    """L_i = cumulative arclength  ds = √(dt² + dr²)."""
    dt = np.diff(t, prepend=t[0])
    dr = np.diff(r, prepend=r[0])
    ds = np.sqrt(dt * dt + dr * dr)
    ds[0] = 0.0
    return np.cumsum(ds) + 1e-9


def coupling_kernel(L: np.ndarray,
                    L_ref: float | None = None) -> np.ndarray:
    """κ_ij = √(L_i L_j) / L_ref ∈ [0, 1]."""
    if L_ref is None:
        L_ref = float(np.median(L[L > 0])) if np.any(L > 0) else 1.0
    K = np.sqrt(np.maximum(np.outer(L, L), 0.0)) / L_ref
    return np.clip(K, 0.0, 1.0)


def chiral_angles_1d(x: np.ndarray) -> np.ndarray:
    """χ_i = arg(x^i) = log x mod 2π."""
    lx = np.log(np.maximum(np.abs(x), 1e-12))
    return (lx + PI) % (2 * PI) - PI


def even_even_mask(N: int) -> np.ndarray:
    i = np.arange(N)[:, None]
    j = np.arange(N)[None, :]
    return (i % 2 == 0) & (j % 2 == 0)


# ============================================================
#  2.  Levi-Civita + kissing spinor  (for the E-field)
# ============================================================
_PERMS_6 = list(permutations(range(6)))
_SIGNS_6: List[int] = []
for _p in _PERMS_6:
    _inv = sum(1 for i in range(6) for j in range(i + 1, 6)
               if _p[i] > _p[j])
    _SIGNS_6.append((-1) ** _inv)


def levi_civita(Z: np.ndarray) -> complex:
    """720-term Levi-Civita contraction  Π_6(Z) = det(Z)."""
    total = 0.0 + 0.0j
    for perm, sign in zip(_PERMS_6, _SIGNS_6):
        prod = 1.0 + 0.0j
        for k, c in enumerate(perm):
            prod *= Z[k, c]
        total += sign * prod
    return total


def icosahedron_vertices() -> np.ndarray:
    verts = []
    for s1 in (-1.0, 1.0):
        for s2 in (-1.0, 1.0):
            verts.append((0.0,       s1,       s2 * PHI))
            verts.append((s1,        s2 * PHI, 0.0))
            verts.append((s2 * PHI,  0.0,      s1))
    V = np.array(verts, dtype=float)
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def lift_to_six_dim(V3: np.ndarray) -> np.ndarray:
    V6 = np.concatenate([V3, PHI * V3], axis=1)
    n = np.linalg.norm(V6, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return V6 / n


def event_spinor(t_n: float, r_n: float, V6: np.ndarray) -> np.ndarray:
    """Z_n[i,j] = V6[i,j] · exp(i·(t_n·i + r_n·j))."""
    idx = np.arange(6)
    phase = np.outer(t_n * idx, np.ones(6)) + \
            np.outer(np.ones(6), r_n * idx)
    return V6[:6, :] * np.exp(1j * phase)


# ============================================================
#  3.  Alternating E-field  ·  Figure 2 of neutral.pdf
# ============================================================
def spin_grade(J_C: float, delta: float) -> float:
    """σ_ℓ(J_C) = (−1)^{2 J_C + δ_ℓ}."""
    return float((-1.0) ** (2.0 * J_C + delta))


def alternating_E_field(t_samples: np.ndarray, r_samples: np.ndarray,
                        J_C: float = 1.0,
                        delta_mu: float = 0.0,
                        delta_tau: float = 1.0) -> Tuple[float, float]:
    """
    S_νℓ = Σ_n (−1)^{n+1} σ_ℓ(J_C) |Π_6(Z_n)|

    Returns (S_νμ, S_ντ).
    """
    V6 = lift_to_six_dim(icosahedron_vertices())
    sig_mu  = spin_grade(J_C, delta_mu)
    sig_tau = spin_grade(J_C, delta_tau)

    S_mu  = 0.0
    S_tau = 0.0
    for n, (tn, rn) in enumerate(zip(t_samples, r_samples), start=1):
        Z = event_spinor(tn, rn, V6)
        mag = abs(levi_civita(Z))
        alt = (-1.0) ** (n + 1)
        S_mu  += alt * sig_mu  * mag
        S_tau += alt * sig_tau * mag
    return float(S_mu), float(S_tau)


# ============================================================
#  4.  Bounded elliptic projection  ·  Figure 3 of neutral.pdf
# ============================================================
def pi_e(t: float, r: float) -> float:
    """Π_e(t,r) = ½(1 + cos 2t · cos 2πr/e) ∈ [0,1]."""
    return 0.5 * (1.0 + math.cos(2.0 * t) *
                        math.cos(2.0 * PI * r / E))


# ============================================================
#  5.  Analog wire with alternating field + electron cloud
# ============================================================
@dataclass
class AnalogWire:
    N: int = 64
    length: float = 10.0
    wiggle: float = 0.25
    turns: float = 3.0
    N_electrons: int = 20
    J_C: float = 1.0
    omega: float = 1.0                       # AC frequency

    # derived
    t_wire: np.ndarray = field(init=False)
    r_wire: np.ndarray = field(init=False)
    L: np.ndarray = field(init=False)
    K: np.ndarray = field(init=False)
    K_even: np.ndarray = field(init=False)
    mask: np.ndarray = field(init=False)
    E_mu: np.ndarray = field(init=False)
    E_tau: np.ndarray = field(init=False)

    # electron state
    e_pos: np.ndarray = field(init=False)
    e_vel: np.ndarray = field(init=False)
    history_pos: List[np.ndarray] = field(default_factory=list)
    history_pi: List[np.ndarray] = field(default_factory=list)
    history_E: List[float] = field(default_factory=list)
    history_rho: List[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        # ---- wire ----
        self.t_wire, self.r_wire = wire_positions(
            self.N, self.length, self.wiggle, self.turns)
        self.L = travel_lengths(self.t_wire, self.r_wire)
        self.K = coupling_kernel(self.L)
        self.mask = even_even_mask(self.N)
        self.K_even = np.where(self.mask, self.K, 0.0)
        np.fill_diagonal(self.K_even, 0.0)

        # ---- alternating E-field  (Figure 2) ----
        half = self.N // 2
        S_mu, S_tau = alternating_E_field(
            self.t_wire[:half], self.r_wire[:half], J_C=self.J_C)
        self.E_mu  = np.array([S_mu  * math.cos(self.omega * t)
                               for t in self.t_wire])
        self.E_tau = np.array([S_tau * math.cos(self.omega * t)
                               for t in self.t_wire])
        self.E_total = self.E_mu + self.E_tau

        # ---- electrons ----
        self.e_pos = np.linspace(0.5, self.length - 0.5, self.N_electrons)
        self.e_vel = np.zeros(self.N_electrons)

    # --------------------------------------------------------
    def wire_index(self, s: float) -> int:
        idx = int(s / self.length * (self.N - 1))
        return max(0, min(self.N - 1, idx))

    # --------------------------------------------------------
    def field_at_time(self, t: float) -> float:
        """AC field amplitude at time t."""
        half = self.N // 2
        S_mu, S_tau = alternating_E_field(
            self.t_wire[:half], self.r_wire[:half], J_C=self.J_C)
        return (S_mu + S_tau) * math.cos(self.omega * t)

    # --------------------------------------------------------
    def step(self, dt: float, t: float) -> None:
        E_field = self.field_at_time(t)

        accel = np.zeros(self.N_electrons)
        for i in range(self.N_electrons):
            s_i = self.e_pos[i]
            idx = self.wire_index(s_i)

            # 1. AC drive  (alternating E-field)
            a = E_field * 0.5

            # 2. coupling force from even-even subgraph
            f_couple = 0.0
            for j in range(self.N_electrons):
                if i == j:
                    continue
                s_j = self.e_pos[j]
                jdx = self.wire_index(s_j)
                k = self.K_even[idx, jdx]
                ds = s_j - s_i
                f_couple += k * ds / (ds * ds + 0.1)
            a += 0.15 * f_couple

            # 3. Π_e bounds the acceleration (cloud intensity)
            r_eff = s_i / self.length * PI
            pe = pi_e(t, r_eff)
            a *= (0.2 + 0.8 * pe)

            # 4. damping
            a -= 0.1 * self.e_vel[i]
            accel[i] = a

        self.e_vel += accel * dt
        self.e_vel *= 0.97
        self.e_pos += self.e_vel * dt

        # reflecting boundaries
        for i in range(self.N_electrons):
            if self.e_pos[i] > self.length:
                self.e_pos[i] = 2 * self.length - self.e_pos[i]
                self.e_vel[i] = -self.e_vel[i]
            elif self.e_pos[i] < 0.0:
                self.e_pos[i] = -self.e_pos[i]
                self.e_vel[i] = -self.e_vel[i]

        # record
        self.history_pos.append(self.e_pos.copy())
        pe_vals = np.array([pi_e(t, s / self.length * PI)
                            for s in self.e_pos])
        self.history_pi.append(pe_vals)
        self.history_E.append(E_field)
        rho = self.density()
        self.history_rho.append(rho)

    # --------------------------------------------------------
    def density(self) -> np.ndarray:
        """Electron number density along the wire (N_wire bins)."""
        rho = np.zeros(self.N)
        for s in self.e_pos:
            idx = self.wire_index(s)
            rho[idx] += 1.0
        return rho

    # --------------------------------------------------------
    def run(self, steps: int = 200, dt: float = 0.02) -> None:
        for n in range(steps):
            self.step(dt, n * dt)


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 76)
    print("Analog transmission wire  ·  alternating E-field  ·  point electrons")
    print("=" * 76)
    print(f"  α       = 1/(π − e)     = {ALPHA:.6f}")
    print(f"  A_STEP                  = {A_STEP:.6f}")
    print(f"  NORM                    = {NORM:.6f}")
    print(f"  Π_e(t,r)  ∈ [0, 1]      bounded cloud intensity")
    print()

    wire = AnalogWire(N=64, length=10.0, wiggle=0.25, turns=3.0,
                      N_electrons=20, J_C=1.0, omega=1.0)

    # ---- wire geometry ----
    print("--- wire geometry  ·  travel lengths ---")
    print(f"  N wire points           = {wire.N}")
    print(f"  wire length             = {wire.length}")
    print(f"  total arclength L_N     = {wire.L[-1]:.6f}")
    print(f"  L range                 = [{wire.L[1]:.6f}, "
          f"{wire.L[-1]:.6f}]")
    print(f"  κ range                 = [{wire.K[wire.K > 0].min():.4f}, "
          f"{wire.K.max():.4f}]")
    print(f"  even-even retained      = "
          f"{int(wire.mask.sum())} / {wire.N**2}  "
          f"(mask fraction = {wire.mask.mean():.4f})")
    print()

    # ---- E-field  (Figure 2) ----
    half = wire.N // 2
    S_mu, S_tau = alternating_E_field(
        wire.t_wire[:half], wire.r_wire[:half], J_C=wire.J_C)
    print("--- alternating E-field  (Figure 2 of neutral.pdf) ---")
    print(f"  J_C                     = {wire.J_C}")
    print(f"  σ_μ(J_C) = (−1)^{{2J_C+0}}   = "
          f"{spin_grade(wire.J_C, 0.0):+.0f}")
    print(f"  σ_τ(J_C) = (−1)^{{2J_C+1}}   = "
          f"{spin_grade(wire.J_C, 1.0):+.0f}")
    print(f"  S_νμ                    = {S_mu:+.6f}")
    print(f"  S_ντ                    = {S_tau:+.6f}")
    print(f"  S_νμ + S_ντ             = {S_mu + S_tau:+.6e}")
    print(f"  E amplitude             = "
          f"{abs(wire.E_total).max():.6e}")
    print()

    # ---- run simulation ----
    wire.run(steps=300, dt=0.02)
    print("--- electron dynamics ---")
    print(f"  N electrons             = {wire.N_electrons}")
    print(f"  steps                   = {len(wire.history_pos)}")
    print(f"  final |v| mean          = "
          f"{np.mean(np.abs(wire.e_vel)):.6f}")
    print(f"  final ⟨Π_e⟩             = "
          f"{np.mean(wire.history_pi[-1]):.4f}")
    print(f"  final density mass      = "
          f"{wire.history_rho[-1].sum():.2f}")
    print()

    # ============================================================
    #  Visualization
    # ============================================================
    fig = plt.figure(figsize=(16, 11))
    gs = GridSpec(3, 2, figure=fig, hspace=0.36, wspace=0.30)

    # (a) wire geometry
    ax = fig.add_subplot(gs[0, 0])
    sc = ax.scatter(wire.t_wire, wire.r_wire,
                    c=wire.L, cmap="viridis", s=30,
                    edgecolors="k", linewidths=0.5)
    ax.plot(wire.t_wire, wire.r_wire, color="gray", lw=0.8, alpha=0.5)
    ax.set_xlabel("t"); ax.set_ylabel("r")
    ax.set_title("Wire geometry  ·  colour = arclength L_i\n"
                 f"L_total = {wire.L[-1]:.4f}")
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    # (b) coupling kernel κ_ij
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(wire.K, origin="lower", cmap="inferno", aspect="equal")
    ax.set_xlabel("wire index j"); ax.set_ylabel("wire index i")
    ax.set_title("Coupling kernel  κ_ij = √(L_i L_j)/L_ref")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (c) alternating E-field  (Figure 2)
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(wire.t_wire, wire.E_mu, "b-", lw=1.2,
            label=r"$S_{\nu\mu}(t)$  (muon)")
    ax.plot(wire.t_wire, wire.E_tau, "r--", lw=1.2,
            label=r"$S_{\nu\tau}(t)$  (tau)")
    ax.plot(wire.t_wire, wire.E_total, "k-", lw=1.4,
            label=r"$E(t) = S_{\nu\mu} + S_{\nu\tau}$")
    ax.axhline(0, color="gray", lw=0.4)
    ax.set_xlabel("wire position s")
    ax.set_ylabel("E amplitude")
    ax.set_title("Alternating E-field  ·  Figure 2 of neutral.pdf\n"
                 "spin-graded alternating sum")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (d) electron trajectories on the wire (Figure 3 cloud)
    ax = fig.add_subplot(gs[1, 1])
    hist = np.array(wire.history_pos).T          # (N_e, T)
    for i in range(wire.N_electrons):
        ax.plot(hist[i], color=plt.cm.viridis(i / wire.N_electrons),
                lw=0.9, alpha=0.8)
    ax.set_xlabel("time step"); ax.set_ylabel("wire position s")
    ax.set_title("Electron trajectories on the wire\n"
                 "point particles bounded by Π_e")
    ax.grid(alpha=0.3)

    # (e) Π_e cloud projection  ·  Figure 3
    ax = fig.add_subplot(gs[2, 0])
    tt = np.linspace(0, 2 * PI, 128)
    rr = np.linspace(0, wire.length, 128)
    T, R = np.meshgrid(tt, rr, indexing="ij")
    PE = 0.5 * (1.0 + np.cos(2.0 * T)
                     * np.cos(2.0 * PI * R / E))
    im = ax.imshow(PE, origin="lower", cmap="viridis",
                   extent=[0, wire.length, 0, 2 * PI], aspect="auto",
                   vmin=0, vmax=1)
    ax.set_xlabel("wire position r")
    ax.set_ylabel("time t")
    ax.set_title("Π_e(t, r) = ½(1 + cos 2t · cos 2πr/e)  ∈ [0,1]\n"
                 "Figure 3 of neutral.pdf")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (f) electron density  ·  ρ(s)
    ax = fig.add_subplot(gs[2, 1])
    rho_final = wire.history_rho[-1]
    rho_first = wire.history_rho[0]
    ax.bar(np.arange(wire.N) - 0.2, rho_first, width=0.4,
           color="#3498db", alpha=0.7, label="initial ρ")
    ax.bar(np.arange(wire.N) + 0.2, rho_final, width=0.4,
           color="#e74c3c", alpha=0.7, label="final ρ")
    ax.set_xlabel("wire index")
    ax.set_ylabel("electron density ρ")
    ax.set_title("Point-electron density along the wire\n"
                 f"total electrons = {wire.N_electrons}")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

    plt.suptitle(
        f"Analog wire transmission  ·  κ_ij = √(L_i L_j)/L_ref  ·  "
        f"E = spin-graded alternating sum  ·  Π_e ∈ [0,1]  ·  "
        f"α = {ALPHA:.4f}",
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # ============================================================
    #  Summary
    # ============================================================
    print("=" * 76)
    print("Summary")
    print("=" * 76)
    print("  Wire geometry")
    print("      (t_i, r_i) parameterized with sinusoidal wiggle")
    print("      L_i = ∫_γi √(dt² + dr²)     (travel length)")
    print(f"      L_total = {wire.L[-1]:.4f}")
    print()
    print("  Coupling kernel")
    print("      κ_ij = √(L_i L_j) / L_ref ∈ [0, 1]")
    print(f"      κ_max = {wire.K.max():.4f}")
    print(f"      even-even retained = {int(wire.mask.sum())}/{wire.N**2}")
    print()
    print("  Alternating E-field  (Figure 2 of neutral.pdf)")
    print("      S_νℓ = Σ_n (−1)^n σ_ℓ(J_C) |Π_6(Z_n)|   ℓ ∈ {μ, τ}")
    print(f"      σ_μ(J_C) = {spin_grade(wire.J_C, 0.0):+.0f}")
    print(f"      σ_τ(J_C) = {spin_grade(wire.J_C, 1.0):+.0f}")
    print(f"      S_νμ = {S_mu:+.4f},  S_ντ = {S_tau:+.4f}")
    print(f"      E(t) = (S_νμ + S_ντ)·cos(ω t)")
    print()
    print("  Electrons as point particles  (Figure 3 of neutral.pdf)")
    print("      Π_e(t,r) = ½(1 + cos 2t · cos 2πr/e) ∈ [0,1]")
    print("      acceleration ∝ Π_e  ·  bounded cloud")
    print(f"      N_electrons = {wire.N_electrons}")
    print(f"      ⟨Π_e⟩ final = {np.mean(wire.history_pi[-1]):.4f}")
    print()
    print("Done.")


if __name__ == "__main__":
    demo()