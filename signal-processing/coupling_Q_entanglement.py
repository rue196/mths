#!/usr/bin/env python3
"""
coupling_Q_entanglement.py
==========================

Quantum geo-location through an even-even fermion coupling channel
with kissing-number spinor reconstruction.

Pipeline (updated)
------------------
1. Events (t_n, r_n) → standard coupling matrix, arc length is the
   travel coupling κ_ij = √(L_i L_j) / L_ref.

2. Coupling matrix modulated by the **coupling scalar point particle
   density** ρ_ij = √(ρ_i ρ_j) ∈ [0,1]:

       g_ij = w_ij · κ_ij · ρ_ij

   ρ_i is the point-particle density at event i (from odd-index
   interpolation).

3. Even-even entanglement mask (unchanged)
       g_ij retained iff i ≡ j ≡ 0 (mod 2)

4. Kissing-number reconstruction (unchanged)
       τ(3) = 12 icosahedron vertices lifted to ℝ⁶
       each even event → 6×6 spinor Z_n → Π(Z_n) = det(Z_n)

5. Signal propagation  ·  volt_scalar(t) modulation

       volt_scalar(t) = (1 + k_turns · L(t)) · ρ_elec(t)/max(ρ_elec)

       s_n^{(t+1)} = volt_scalar(t) · Σ_j G_nj s_j / Σ_j G_nj

   Only even-index slots hold signal; odd indices stay at 0 and act
   as the **electron point particle slots**.

6. 3D entanglement trajectory  ·  AC-carrier driven

   The 12 icosahedron fermion vertices are the **carriers for the
   electron point particles**.  They move through the AC field

       E(t) = (S_νμ + S_ντ) · cos(ω t)

   under three forces:
       · harmonic restoring force to the reference icosahedron
       · pairwise even-even coupling from G_even
       · AC-carrier drive E(t) pushing along the vertex radial direction

Imports (unchanged parents)
---------------------------
    DC_chirp.DCMobiusGate, mobius_sieve, even_even_mask
    wire_analog.alternating_E_field, pi_e, spin_grade
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from itertools import permutations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---- reference primitives ----
from DC_chirp import DCMobiusGate, mobius_sieve
from wire_analog import alternating_E_field, pi_e, spin_grade


# ============================================================
#  Constants
# ============================================================
PI    = math.pi
E     = math.e
ALPHA = 1.0 / (PI - E)                # ≈ 2.362
PHI   = (1.0 + math.sqrt(5.0)) / 2.0
W1    = PI
W2    = E


# ============================================================
#  Coupling matrix primitives
# ============================================================
def chiral_angles(t: np.ndarray) -> np.ndarray:
    """χ_n = arg(t_n^i) = log(t_n) mod 2π."""
    lt = np.log(np.maximum(np.abs(t), 1e-12))
    return (lt + PI) % (2 * PI) - PI


def travel_lengths(t: np.ndarray, r: np.ndarray) -> np.ndarray:
    """L_n = cumulative arclength along the (t, r) curve."""
    dt = np.diff(t, prepend=t[0])
    dr = np.diff(r, prepend=r[0])
    ds = np.sqrt(dt * dt + dr * dr)
    ds[0] = 0.0
    return np.cumsum(ds) + 1e-6


def coupling_matrix(t: np.ndarray, r: np.ndarray,
                    L_ref: Optional[float] = None,
                    density: Optional[np.ndarray] = None,
                    density_weight: float = 0.5) -> np.ndarray:
    """
    g_ij = w_ij · κ_ij · ρ_ij

        w_ij = |χ_i − χ_j| / π                chiral bond
        κ_ij = √(L_i L_j) / L_ref             arc-length travel coupling
        ρ_ij = √(ρ_i ρ_j)  ∈ [0,1]            point-particle density

    `density` is optional; if provided, it modulates the coupling
    between `(1 − density_weight) · G` and `density_weight · G · ρ_ij`.
    """
    N = len(t)
    chi = chiral_angles(t)
    dchi = np.abs(chi[:, None] - chi[None, :])
    dchi = (dchi + PI) % (2 * PI) - PI
    W = np.abs(dchi) / PI

    L = travel_lengths(t, r)
    if L_ref is None:
        L_ref = float(np.median(L[L > 0])) if np.any(L > 0) else 1.0
    K = np.sqrt(np.maximum(L[:, None] * L[None, :], 0.0)) / L_ref
    K = np.clip(K, 0.0, 1.0)

    G = W * K

    if density is not None and density_weight > 0:
        rho = np.asarray(density, dtype=float)
        rho_max = float(np.max(rho)) if rho.size else 1.0
        rho_norm = rho / (rho_max + 1e-12)
        density_coupling = np.sqrt(
            np.maximum(rho_norm[:, None] * rho_norm[None, :], 0.0))
        modulation = (1.0 - density_weight) + density_weight * density_coupling
        G = G * modulation

    np.fill_diagonal(G, 0.0)
    return G


# ============================================================
#  Even-even entanglement mask
# ============================================================
def even_even_mask(N: int) -> np.ndarray:
    i = np.arange(N)[:, None]
    j = np.arange(N)[None, :]
    return (i % 2 == 0) & (j % 2 == 0)


def entangled_coupling(G: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    N = G.shape[0]
    mask = even_even_mask(N)
    G_even = np.where(mask, G, 0.0)
    np.fill_diagonal(G_even, 0.0)
    return G_even, mask


def even_indices(N: int) -> np.ndarray:
    return np.arange(0, N, 2)


# ============================================================
#  Point-particle density from odd-index slots
# ============================================================
def point_particle_density(N: int,
                            signal: Optional[np.ndarray] = None,
                            base_density: float = 1.0) -> np.ndarray:
    """
    Electron point-particle density along the chain.

    Odd indices are the electron point-particle slots (they hold no
    signal because of the even-even mask).  Their density is
    interpolated from the adjacent even-signal amplitude:

        ρ[odd] = ½(|s[left_even]| + |s[right_even]|)

    Even indices contribute 0 (they are signal carriers, not
    particle carriers).  `base_density` scales the result.
    """
    rho = np.zeros(N, dtype=float)
    if signal is None:
        rho[np.arange(N) % 2 == 1] = base_density
        return rho

    s_abs = np.abs(np.asarray(signal, dtype=float))
    if s_abs.size < N:
        s_abs = np.pad(s_abs, (0, N - s_abs.size))

    for i in range(1, N, 2):
        left  = s_abs[i - 1] if i - 1 >= 0 else 0.0
        right = s_abs[i + 1] if i + 1 < N else 0.0
        rho[i] = 0.5 * (left + right)

    rho_max = float(np.max(rho)) if rho.size else 1.0
    if rho_max > 1e-12:
        rho = rho / rho_max * base_density
    return rho


# ============================================================
#  AC carrier  E(t) = (S_νμ + S_ντ) cos(ω t)
# ============================================================
def compute_S_mu_tau(t_samples: np.ndarray,
                     gate: DCMobiusGate,
                     N_events: int = 41,
                     J_C: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the muon (μ) and tau (τ) alternating sums at each t:

        S_νμ(t) = Σ_n (−1)^{n+1} σ_μ(J_C) |ζ(t + n·Δt)|
        S_ντ(t) = Σ_n (−1)^{n+1} σ_τ(J_C) |ζ(t + n·Δt)|
    """
    sig_mu  = spin_grade(J_C, 0.0)
    sig_tau = spin_grade(J_C, 1.0)
    dt_event = gate.period / N_events
    n_off = np.arange(N_events, dtype=float)
    alts  = (-1.0) ** (n_off + 1)

    S_mu  = np.zeros_like(t_samples, dtype=float)
    S_tau = np.zeros_like(t_samples, dtype=float)

    for i, t_now in enumerate(t_samples):
        taus = t_now + n_off * dt_event
        mags = np.abs(np.array([gate.zeta(z) for z in taus]))
        S_mu[i]  = sig_mu  * float(np.dot(alts, mags))
        S_tau[i] = sig_tau * float(np.dot(alts, mags))

    return S_mu, S_tau


def ac_carrier_field(t: np.ndarray,
                     S_mu: np.ndarray,
                     S_tau: np.ndarray,
                     omega: float = 1.0) -> np.ndarray:
    """
    E(t) = (S_νμ + S_ντ) · cos(ω t)
    """
    return (S_mu + S_tau) * np.cos(omega * np.asarray(t))


# ============================================================
#  Volt-level scalar
# ============================================================
def volt_scalar_profile(L: np.ndarray,
                        rho_elec: np.ndarray,
                        k_turns: float = 3.0) -> np.ndarray:
    """
    volt_scalar(t) = (1 + k_turns · L(t)) · ρ_elec(t)/max(ρ_elec)

    Combines arc-length turns with point-particle density.
    """
    L = np.asarray(L, dtype=float)
    rho = np.asarray(rho_elec, dtype=float)
    rho_max = float(np.max(np.abs(rho))) if rho.size else 1.0
    rho_norm = (np.abs(rho) / (rho_max + 1e-12)) if rho_max > 1e-12 else rho
    return (1.0 + k_turns * L) * rho_norm


# ============================================================
#  Kissing number  ·  τ(3) = 12 icosahedron vertices
# ============================================================
def icosahedron_vertices() -> np.ndarray:
    verts = []
    for s1 in (-1.0, 1.0):
        for s2 in (-1.0, 1.0):
            verts.append((0.0,         s1,        s2 * PHI))
            verts.append((s1,          s2 * PHI,  0.0))
            verts.append((s2 * PHI,    0.0,       s1))
    V = np.array(verts, dtype=float)
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def lift_to_six_dim(V3: np.ndarray) -> np.ndarray:
    V6 = np.concatenate([V3, PHI * V3], axis=1)
    norms = np.linalg.norm(V6, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return V6 / norms


_PERMS_6 = list(permutations(range(6)))
_SIGNS_6 = []
for _p in _PERMS_6:
    _inv = sum(1 for i in range(6) for j in range(i + 1, 6)
               if _p[i] > _p[j])
    _SIGNS_6.append((-1) ** _inv)


def levi_civita(Z: np.ndarray) -> complex:
    total = 0.0 + 0.0j
    for perm, sign in zip(_PERMS_6, _SIGNS_6):
        prod = 1.0 + 0.0j
        for k, c in enumerate(perm):
            prod *= Z[k, c]
        total += sign * prod
    return total


def event_spinor(t_n: float, r_n: float, V6: np.ndarray) -> np.ndarray:
    idx = np.arange(6)
    phase = np.outer(t_n * idx, np.ones(6)) + \
            np.outer(np.ones(6), r_n * idx)
    return V6[:6, :] * np.exp(1j * phase)


def spinor_reconstruct(t_even: np.ndarray, r_even: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray]:
    V3 = icosahedron_vertices()
    V6 = lift_to_six_dim(V3)
    Pis = np.zeros(len(t_even), dtype=complex)
    for n, (tn, rn) in enumerate(zip(t_even, r_even)):
        Z = event_spinor(tn, rn, V6)
        Pis[n] = levi_civita(Z)
    return Pis, np.abs(Pis)


# ============================================================
#  Signal propagation  ·  volt_scalar modulated
# ============================================================
def propagate_signal(G_even: np.ndarray,
                     s0: np.ndarray,
                     steps: int = 20,
                     decay: float = 0.05,
                     volt_scalar: Optional[np.ndarray] = None
                     ) -> np.ndarray:
    """
    Even-even diffusion with optional volt-level scalar:

        s_n^{(k+1)} = volt_scalar[n] · (Σ_j G_nj s_j / Σ_j G_nj) · (1 − decay)

    Odd indices stay at 0.
    """
    N = G_even.shape[0]
    s = s0.copy().astype(float)
    hist = [s.copy()]

    if volt_scalar is not None:
        vs = np.asarray(volt_scalar, dtype=float)
        if vs.size < N:
            vs = np.pad(vs, (0, N - vs.size), mode="edge")
        vs = vs[:N]
    else:
        vs = np.ones(N)

    row_sum = G_even.sum(axis=1) + 1e-12
    odd = np.arange(N) % 2 == 1

    for _ in range(steps):
        s_new = (G_even @ s) / row_sum
        s_new = s_new * vs
        s_new *= (1.0 - decay)
        s_new[odd] = 0.0
        s = s_new
        hist.append(s.copy())
    return np.array(hist)


# ============================================================
#  3D entanglement trajectory  ·  AC-carrier driven
# ============================================================
@dataclass
class TrajectorySim:
    """
    Simulate 12 fermion carriers (icosahedron vertices) in 3D.

    Each vertex *carries* electron point particles.  Motion is
    driven by three forces:

        F_restore = −k_spring · (p − V3)           icosahedron anchor
        F_couple  = pairwise even-even coupling
        F_ac      = k_ac · E(t) · radial direction  AC carrier

    The AC field E(t) = (S_νμ + S_ντ)·cos(ωt) pushes the carriers
    along the outward-radial direction of the reference icosahedron,
    oscillating at the carrier frequency.
    """
    G_even: np.ndarray
    V3: np.ndarray
    E_ac: Optional[np.ndarray] = None      # AC carrier field samples
    t_axis: Optional[np.ndarray] = None    # matching time axis
    positions: np.ndarray = field(default=None)
    velocities: np.ndarray = field(default=None)
    history: List[np.ndarray] = field(default_factory=list)
    ac_history: List[float] = field(default_factory=list)
    k_spring: float = 0.5
    k_ac: float = 0.3
    damping: float = 0.15
    dt: float = 0.05

    def __post_init__(self):
        rng = np.random.default_rng(0)
        if self.positions is None:
            self.positions = self.V3.copy() + \
                rng.standard_normal((12, 3)) * 0.05
        if self.velocities is None:
            self.velocities = rng.standard_normal((12, 3)) * 0.02
        self.history.append(self.positions.copy())
        self.ac_history.append(0.0)

    # ---------- pair coupling ----------
    def _pair_force(self) -> np.ndarray:
        N_even = self.G_even.shape[0] // 2
        slot = (np.arange(12) % max(N_even, 1)) * 2
        force = np.zeros_like(self.positions)
        for a in range(12):
            for b in range(12):
                if a == b:
                    continue
                g = self.G_even[slot[a], slot[b]]
                diff = self.positions[b] - self.positions[a]
                r = np.linalg.norm(diff) + 1e-6
                force[a] += g * diff / (r ** 2 + 0.1)
        return force

    # ---------- AC carrier drive ----------
    def _ac_force(self, t_now: float) -> Tuple[np.ndarray, float]:
        """
        AC field pushes along outward radial direction of the reference
        icosahedron.
        """
        if self.E_ac is None or self.t_axis is None:
            return np.zeros_like(self.positions), 0.0

        E_val = float(np.interp(t_now, self.t_axis, self.E_ac))

        # outward radial direction of the reference vertices
        radial = self.V3.copy()
        # radial components push along radial; the perpendicular
        # direction gives a swirl that keeps the motion bounded
        radial_amp = np.linalg.norm(radial, axis=1, keepdims=True)
        radial_amp[radial_amp < 1e-6] = 1.0
        radial = radial / radial_amp

        F = self.k_ac * E_val * radial
        return F, E_val

    # ---------- one step ----------
    def step(self, t_now: float) -> None:
        F_couple = self._pair_force()
        F_restore = -self.k_spring * (self.positions - self.V3)
        F_ac, E_val = self._ac_force(t_now)
        F_total = F_couple + F_restore + F_ac

        self.velocities += F_total * self.dt
        self.velocities *= (1.0 - self.damping * self.dt)
        self.positions += self.velocities * self.dt
        self.history.append(self.positions.copy())
        self.ac_history.append(E_val)

    def run(self, n_steps: int = 120) -> None:
        if self.t_axis is None:
            for k in range(n_steps):
                self.step(k * self.dt)
        else:
            T = float(self.t_axis[-1])
            for k in range(n_steps):
                # map step k to time in the field
                t_now = (k / max(n_steps - 1, 1)) * T
                self.step(t_now)

    @property
    def trajectory(self) -> np.ndarray:
        return np.array(self.history)


# ============================================================
#  Event generator
# ============================================================
def make_geo_events(n_events: int = 80,
                    seed: int = 2024) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.sort(rng.exponential(scale=3.0, size=n_events))
    t = np.maximum(t, 1e-3)
    r = np.arange(1, n_events + 1, dtype=float)
    return t, r


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("Quantum geo-location  ·  AC carrier + point-particle density")
    print("=" * 78)
    print(f"  α = 1/(π − e)     = {ALPHA:.6f}")
    print(f"  φ = (1+√5)/2      = {PHI:.6f}")
    print(f"  τ(3)              = 12 fermion carriers")
    print(f"  odd indices       = electron point-particle slots")
    print(f"  AC carrier        : E(t) = (S_νμ+S_ντ)·cos(ωt)")
    print(f"  coupling          : g_ij = w·κ·ρ")
    print(f"  propagation       : s ← volt_scalar · (G_even·s / row_sum)")
    print()

    # ---- events ----
    N = 80
    t, r = make_geo_events(n_events=N, seed=2024)
    print(f"--- events ---")
    print(f"  N events          = {N}")
    print(f"  t range           = [{t.min():.4f}, {t.max():.4f}]")
    print(f"  r range           = [{r.min():.1f}, {r.max():.1f}]")
    print()

    # ============================================================
    #  1 · base coupling matrix + arc length (travel coupling)
    # ============================================================
    print("=" * 78)
    print("1  ·  coupling matrix  ·  w·κ (arc length is travel coupling)")
    print("=" * 78)
    G_base = coupling_matrix(t, r)
    off = G_base[G_base > 0]
    print(f"  ⟨g_ij⟩            = {off.mean():.6f}")
    print(f"  max g_ij          = {G_base.max():.6f}")
    print(f"  sparsity          = "
          f"{1.0 - np.count_nonzero(G_base)/(N*N):.4f}")
    print()

    # ============================================================
    #  2 · AC carrier  E(t) = (S_νμ + S_ντ)·cos(ωt)
    # ============================================================
    print("=" * 78)
    print("2  ·  AC carrier  E(t) = (S_νμ + S_ντ)·cos(ωt)")
    print("=" * 78)
    gate = DCMobiusGate(K=20, smooth=True)
    T_gate = gate.period

    # time axis spanning the event range
    t_axis = np.linspace(0.0, max(t.max(), T_gate), 512)
    omega = 1.0

    S_mu, S_tau = compute_S_mu_tau(t_axis, gate,
                                    N_events=41, J_C=1.0)
    E_ac = ac_carrier_field(t_axis, S_mu, S_tau, omega)

    print(f"  S_νμ mean / rms   = "
          f"{np.mean(S_mu):+.4f} / {np.sqrt(np.mean(S_mu**2)):.4f}")
    print(f"  S_ντ mean / rms   = "
          f"{np.mean(S_tau):+.4f} / {np.sqrt(np.mean(S_tau**2)):.4f}")
    print(f"  E(t) mean / rms   = "
          f"{np.mean(E_ac):+.4e} / {np.sqrt(np.mean(E_ac**2)):.4f}")
    print(f"  E(t) range        = [{E_ac.min():+.4f}, {E_ac.max():+.4f}]")
    print()

    # ============================================================
    #  3 · point-particle density (odd-index slots)
    # ============================================================
    print("=" * 78)
    print("3  ·  point-particle density  ·  odd-index 0 slots")
    print("=" * 78)
    # interpolate density from a sample signal — here we use the
    # event times themselves as a proxy for signal distribution
    signal_proxy = np.zeros(N)
    signal_proxy[::2] = t[:N//2 + 1][:len(signal_proxy[::2])] \
        if len(t) >= len(signal_proxy[::2]) else 0
    # more natural: normalise the event times
    signal_proxy[::2] = t[::2][:len(signal_proxy[::2])]
    signal_proxy /= np.max(np.abs(signal_proxy)) + 1e-12

    rho_elec = point_particle_density(N, signal_proxy)
    odd_slots = np.arange(N) % 2 == 1
    print(f"  ρ range           = [{rho_elec.min():.4f}, "
          f"{rho_elec.max():.4f}]")
    print(f"  odd slots count   = {int(odd_slots.sum())}")
    print(f"  odd ρ mean        = {rho_elec[odd_slots].mean():.4f}")
    print(f"  even ρ (should=0) = {rho_elec[~odd_slots].sum():.4e}")
    print()

    # ============================================================
    #  4 · density-weighted coupling matrix
    # ============================================================
    print("=" * 78)
    print("4  ·  density-weighted coupling  ·  g_ij = w·κ·ρ_ij")
    print("=" * 78)
    G_dw = coupling_matrix(t, r, density=rho_elec, density_weight=0.5)
    off_dw = G_dw[G_dw > 0]
    print(f"  ⟨g_ij⟩            = {off_dw.mean():.6f}  "
          f"(base: {off.mean():.6f})")
    print(f"  max g_ij          = {G_dw.max():.6f}  "
          f"(base: {G_base.max():.6f})")
    ratio = off_dw.mean() / (off.mean() + 1e-12)
    print(f"  density modulation ratio = {ratio:.4f}")
    print()

    # ============================================================
    #  5 · even-even mask + kissing reconstruction
    # ============================================================
    print("=" * 78)
    print("5  ·  even-even mask + kissing reconstruction")
    print("=" * 78)
    G_even, mask = entangled_coupling(G_dw)
    kept = np.count_nonzero(G_even)
    print(f"  retained entries  = {kept} / {N*N}")

    even_idx = even_indices(N)
    t_even = t[even_idx]
    r_even = r[even_idx]
    Pis, mags = spinor_reconstruct(t_even, r_even)
    print(f"  n even events     = {len(t_even)}")
    print(f"  |Π| mean          = {mags.mean():.4f}")
    print()

    # ============================================================
    #  6 · volt_scalar propagation
    # ============================================================
    print("=" * 78)
    print("6  ·  volt_scalar propagation  ·  "
          "volt_scalar = (1+k_turns·L)·ρ/max(ρ)")
    print("=" * 78)
    # arc length = travel coupling L(t)
    L_travel = travel_lengths(t, r)
    # normalize to [0,1]
    L_norm = L_travel / (L_travel[-1] + 1e-12)

    k_turns = 3.0
    volt_scalar = volt_scalar_profile(L_norm, rho_elec, k_turns)
    print(f"  L range           = [{L_norm.min():.4f}, {L_norm.max():.4f}]")
    print(f"  volt_scalar range = [{volt_scalar.min():.4f}, "
          f"{volt_scalar.max():.4f}]")
    print(f"  volt_scalar mean  = {volt_scalar.mean():.4f}")

    # impulse at event 0
    s0 = np.zeros(N)
    s0[0] = 1.0
    s_hist = propagate_signal(
        G_even, s0, steps=30, decay=0.03,
        volt_scalar=volt_scalar)

    print(f"  final amplitude   = {s_hist[-1, 0]:.6e}")
    print(f"  final mass Σ|s|   = {np.sum(np.abs(s_hist[-1])):.6e}")
    even_mass = np.sum(np.abs(s_hist[-1][::2]))
    odd_mass  = np.sum(np.abs(s_hist[-1][1::2]))
    print(f"  even mass         = {even_mass:.6e}")
    print(f"  odd mass (0)      = {odd_mass:.6e}  (must be 0)")
    print()

    # ============================================================
    #  7 · 12 fermion carriers moving through E(t)
    # ============================================================
    print("=" * 78)
    print("7  ·  12 fermion carriers driven through E(t)")
    print("=" * 78)
    V3 = icosahedron_vertices()
    sim = TrajectorySim(
        G_even=G_even, V3=V3,
        E_ac=E_ac, t_axis=t_axis,
        k_spring=0.5, k_ac=0.3, damping=0.15,
    )
    sim.run(n_steps=150)
    traj = sim.trajectory

    print(f"  steps             = {traj.shape[0]}")
    print(f"  displacements     :")
    disp = np.linalg.norm(traj[-1] - V3, axis=1)
    print(f"    max             = {disp.max():.6f}")
    print(f"    mean            = {disp.mean():.6f}")
    print(f"    min             = {disp.min():.6f}")
    print(f"  E(t) sampled      = "
          f"[{min(sim.ac_history):+.4f}, {max(sim.ac_history):+.4f}]")
    print()

    # ============================================================
    #  Visualization
    # ============================================================
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.30)

    # (a) base coupling matrix
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(G_base, origin="lower", cmap="inferno", aspect="equal")
    ax.set_xlabel("event j"); ax.set_ylabel("event i")
    ax.set_title(f"base coupling w·κ\n"
                 f"⟨g⟩ = {off.mean():.4f}  max = {G_base.max():.4f}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (b) density-weighted coupling
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(G_dw, origin="lower", cmap="inferno", aspect="equal")
    ax.set_xlabel("event j"); ax.set_ylabel("event i")
    ax.set_title(f"density-weighted w·κ·ρ\n"
                 f"⟨g⟩ = {off_dw.mean():.4f}  max = {G_dw.max():.4f}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (c) even-even masked
    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(G_even, origin="lower", cmap="inferno",
                   aspect="equal")
    ax.set_xlabel("event j"); ax.set_ylabel("event i")
    ax.set_title(f"even-even mask\nkept = {kept}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (d) AC carrier E(t)
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t_axis, E_ac, color="#16a085", lw=1.2,
            label="E(t) = (S_νμ+S_ντ)cos(ωt)")
    ax.plot(t_axis, S_mu, color="#e74c3c", lw=0.7, ls="--",
            alpha=0.6, label="S_νμ")
    ax.plot(t_axis, S_tau, color="#3498db", lw=0.7, ls=":",
            alpha=0.6, label="S_ντ")
    ax.axhline(0, color="gray", lw=0.4)
    ax.set_xlabel("t"); ax.set_ylabel("field")
    ax.set_title("AC carrier  ·  drives the 12 fermion carriers")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (e) point-particle density
    ax = fig.add_subplot(gs[1, 1])
    ax.bar(np.arange(N), rho_elec,
           color=["#e74c3c" if i % 2 == 0 else "#3498db"
                  for i in range(N)],
           edgecolor="k", linewidth=0.2)
    ax.set_xlabel("event index")
    ax.set_ylabel("ρ (point-particle)")
    ax.set_title(f"point-particle density  ·  "
                 f"odd slots = electron slots  (mean = "
                 f"{rho_elec[odd_slots].mean():.3f})")
    ax.grid(alpha=0.3, axis="y")

    # (f) volt_scalar
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(np.arange(N), L_norm, color="#e67e22", lw=1.2,
            label="L(t)  arc length (travel)")
    ax.plot(np.arange(N), rho_elec / (rho_elec.max() + 1e-12),
            color="#2980b9", lw=1.0, ls="--",
            label="ρ / max ρ")
    ax.plot(np.arange(N), volt_scalar, color="#16a085", lw=1.4,
            label="volt_scalar")
    ax.set_xlabel("event index")
    ax.set_ylabel("value")
    ax.set_title("volt_scalar = (1+k_turns·L)·ρ/max(ρ)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (g) signal propagation
    ax = fig.add_subplot(gs[2, 0])
    for k in [0, 5, 10, 20, 30]:
        ax.plot(np.arange(N), s_hist[k], lw=1.0, alpha=0.85,
                label=f"step {k}")
    ax.scatter(even_idx, np.zeros_like(even_idx),
               marker="|", color="k", s=200, zorder=5,
               label="even indices")
    ax.scatter(np.arange(N)[1::2], np.zeros_like(rho_elec[1::2]),
               marker=".", color="#3498db", s=60, zorder=5,
               label="odd (e-) slots")
    ax.set_xlabel("event n"); ax.set_ylabel("signal s_n")
    ax.set_title("propagation  ·  volt_scalar-modulated even-even diffusion")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (h) 3D trajectories with E(t) drive
    ax = fig.add_subplot(gs[2, 1], projection="3d")
    for a in range(12):
        path = traj[:, a, :]
        ax.plot(path[:, 0], path[:, 1], path[:, 2],
                lw=0.9, alpha=0.7)
        ax.scatter(path[0, 0], path[0, 1], path[0, 2],
                   color="gray", s=20)
        ax.scatter(path[-1, 0], path[-1, 1], path[-1, 2],
                   color="crimson", s=40, edgecolors="k")
    for a in range(12):
        for b in range(a + 1, 12):
            if np.linalg.norm(V3[a] - V3[b]) < 1.2:
                ax.plot([V3[a, 0], V3[b, 0]],
                        [V3[a, 1], V3[b, 1]],
                        [V3[a, 2], V3[b, 2]],
                        color="gray", lw=0.4, alpha=0.4)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("12 fermion carriers\n"
                 "moving through E(t)")

    # (i) mass budget
    ax = fig.add_subplot(gs[2, 2])
    total_mass = np.sum(np.abs(s_hist), axis=1)
    even_mass_h = np.sum(np.abs(s_hist[:, ::2]), axis=1)
    odd_mass_h = np.sum(np.abs(s_hist[:, 1::2]), axis=1)
    ax.semilogy(total_mass, "k-", lw=1.4, label="total")
    ax.semilogy(even_mass_h, "b-", lw=1.4, label="even")
    ax.semilogy(odd_mass_h + 1e-18, "r:", lw=1.4, label="odd (=0)")
    ax.set_xlabel("step"); ax.set_ylabel(r"$\sum |s_n|$")
    ax.set_title("signal mass budget")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

    plt.suptitle(
        "Quantum geo-location  ·  12 fermion carriers  ·  "
        "AC carrier E(t) = (S_νμ+S_ντ)cos(ωt)  ·  "
        "odd slots = electron point particles  ·  "
        f"α = {ALPHA:.4f}",
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # ============================================================
    #  Summary
    # ============================================================
    print("=" * 78)
    print("Pipeline summary")
    print("=" * 78)
    print("  1. coupling matrix    g_ij = w_ij · κ_ij · ρ_ij")
    print("       w_ij = chiral bond    κ_ij = arc-length travel")
    print("       ρ_ij = √(ρ_i ρ_j)     point-particle density coupling")
    print("  2. even-even mask     retain only (i, j) both even")
    print("  3. kissing recon      12 icosahedron → 6×6 spinor")
    print("  4. AC carrier         E(t) = (S_νμ+S_ντ)·cos(ωt)")
    print("  5. propagation        s ← volt_scalar · (G_even · s / row_sum)")
    print("       volt_scalar = (1+k_turns·L(t)) · ρ_elec/max(ρ_elec)")
    print("  6. 12 fermion carriers move through E(t):")
    print("       F = F_restore + F_even-even + F_AC")
    print()
    print("  Key invariants:")
    print(f"    retained pairs     : {kept} / {N*N}")
    print(f"    |Π| mean           : {mags.mean():.4f}")
    print(f"    final signal mass  : "
          f"{np.sum(np.abs(s_hist[-1])):.6e}")
    print(f"    odd mass           : "
          f"{np.sum(np.abs(s_hist[-1][1::2])):.6e}  (must be 0)")
    print(f"    E(t) rms           : {np.sqrt(np.mean(E_ac**2)):.4f}")
    print(f"    volt_scalar mean   : {volt_scalar.mean():.4f}")
    print(f"    carrier disp mean  : {disp.mean():.6f}")
    print()
    print("  Odd-index 0 slots are the electron point particles — their")
    print("  density ρ interpolates from even-signal amplitude, and the")
    print("  density-weighted coupling g_ij = w·κ·ρ propagates through")
    print("  them under the volt_scalar modulation.")
    print()
    print("Done.")


if __name__ == "__main__":
    demo()