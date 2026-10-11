#!/usr/bin/env python3
"""
quantum_geo_location.py
=======================
Quantum geo-location through an even-even fermion coupling channel
with kissing-number spinor reconstruction.

Pipeline
--------
1. Events (t_n, r_n) with the standard coupling matrix
       g_ij = w_ij · κ_ij
       w_ij = |χ_i − χ_j| / π         chiral bond weight
       κ_ij = √(L_i L_j) / L_ref      travel coupling

2. Even-even entanglement mask
       g_ij retained  iff  i ≡ j ≡ 0 (mod 2)
       all odd-index pairs are zeroed — only even parity entangles

3. Kissing-number reconstruction
       τ(3) = 12 icosahedron vertices lifted to ℝ⁶
       each even event n maps to a 6×6 spinor Z_n
       Levi-Civita contraction Π(Z_n) = det(Z_n)

4. Signal transmission
       s_n(t+1) = Σ_j G_nj · s_j(t) / Σ_j G_nj
       the even-even channel is the propagation graph

5. 3D entanglement trajectory
       12 fermion vertices move under:
         · harmonic restoring force to the icosahedron
         · pairwise coupling from the even-even matrix
       track their 3D positions over simulation steps
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from itertools import permutations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


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
#  Coupling matrix primitives  (from test_color.py)
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
                    L_ref: Optional[float] = None) -> np.ndarray:
    """Standard g_ij = w_ij · κ_ij — no masking yet."""
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
    np.fill_diagonal(G, 0.0)
    return G


# ============================================================
#  Even-even entanglement mask
# ============================================================
def even_even_mask(N: int) -> np.ndarray:
    """
    Boolean mask: True only where both i and j are even (0-indexed).
    """
    i = np.arange(N)[:, None]
    j = np.arange(N)[None, :]
    return (i % 2 == 0) & (j % 2 == 0)


def entangled_coupling(G: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Zero out all odd-index pairs, keeping only the even-even subgraph.
    Returns (G_even, mask) where mask selects the retained entries.
    """
    N = G.shape[0]
    mask = even_even_mask(N)
    G_even = np.where(mask, G, 0.0)
    np.fill_diagonal(G_even, 0.0)
    return G_even, mask


def even_indices(N: int) -> np.ndarray:
    """List of even event indices."""
    return np.arange(0, N, 2)


# ============================================================
#  Kissing number  ·  τ(3) = 12 icosahedron vertices
# ============================================================
def icosahedron_vertices() -> np.ndarray:
    """12 icosahedron vertices in ℝ³ (golden ratio)."""
    verts = []
    for s1 in (-1.0, 1.0):
        for s2 in (-1.0, 1.0):
            verts.append((0.0,         s1,        s2 * PHI))
            verts.append((s1,          s2 * PHI,  0.0))
            verts.append((s2 * PHI,    0.0,       s1))
    V = np.array(verts, dtype=float)
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def lift_to_six_dim(V3: np.ndarray) -> np.ndarray:
    """V6[:, 0:3] = V3, V6[:, 3:6] = φ·V3, normalized."""
    V6 = np.concatenate([V3, PHI * V3], axis=1)
    norms = np.linalg.norm(V6, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return V6 / norms


# ---------- Levi-Civita  ·  720 terms ----------
_PERMS_6 = list(permutations(range(6)))
_SIGNS_6 = []
for _p in _PERMS_6:
    _inv = sum(1 for i in range(6) for j in range(i + 1, 6)
               if _p[i] > _p[j])
    _SIGNS_6.append((-1) ** _inv)


def levi_civita(Z: np.ndarray) -> complex:
    """Π(Z) = Σ_σ ε(σ) Π_k Z[k, σ(k)]  — O(720)."""
    total = 0.0 + 0.0j
    for perm, sign in zip(_PERMS_6, _SIGNS_6):
        prod = 1.0 + 0.0j
        for k, c in enumerate(perm):
            prod *= Z[k, c]
        total += sign * prod
    return total


# ============================================================
#  Kissing spinor per even event
# ============================================================
def event_spinor(t_n: float, r_n: float,
                 V6: np.ndarray) -> np.ndarray:
    """
    Map an even event to a 6×6 complex spinor Z_n:

        Z_n[i, j] = V6[i, j] · e^{i · (t_n·i + r_n·j)}

    The phase carries the event's location on the (t, r) plane.
    """
    idx = np.arange(6)
    phase = np.outer(t_n * idx, np.ones(6)) + \
            np.outer(np.ones(6), r_n * idx)
    return V6[:6, :] * np.exp(1j * phase)


def spinor_reconstruct(t_even: np.ndarray, r_even: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    For every even event, build the spinor and contract it.
    Returns (Pi_values, magnitudes).
    """
    V3 = icosahedron_vertices()
    V6 = lift_to_six_dim(V3)

    Pis = np.zeros(len(t_even), dtype=complex)
    for n, (tn, rn) in enumerate(zip(t_even, r_even)):
        Z = event_spinor(tn, rn, V6)
        Pis[n] = levi_civita(Z)
    return Pis, np.abs(Pis)


# ============================================================
#  Signal transmission through the even-even channel
# ============================================================
def propagate_signal(G_even: np.ndarray,
                     s0: np.ndarray,
                     steps: int = 20,
                     decay: float = 0.05) -> np.ndarray:
    """
    Even-even diffusion:
        s_n^{(k+1)} = (Σ_j G_nj s_j) / (Σ_j G_nj + ε) · (1 − decay)

    Only even indices carry signal. Odd indices stay zero.
    Returns history of shape (steps+1, N).
    """
    N = G_even.shape[0]
    s = s0.copy().astype(float)
    hist = [s.copy()]

    row_sum = G_even.sum(axis=1) + 1e-12
    for _ in range(steps):
        s_new = (G_even @ s) / row_sum
        s_new *= (1.0 - decay)
        # only even slots retain amplitude
        odd = np.arange(N) % 2 == 1
        s_new[odd] = 0.0
        s = s_new
        hist.append(s.copy())
    return np.array(hist)


# ============================================================
#  3D entanglement trajectory
# ============================================================
@dataclass
class TrajectorySim:
    """
    Simulate 12 kissing-number fermions in 3D under:
        · harmonic restoring force to their icosahedron position
        · pairwise coupling from the even-even matrix

    Positions are (12, 3).  The coupling strength is modulated by
    the mean row of G_even so that only coupled pairs interact.
    """
    G_even: np.ndarray
    V3: np.ndarray                     # (12, 3) reference
    positions: np.ndarray = field(default=None)
    velocities: np.ndarray = field(default=None)
    history: List[np.ndarray] = field(default_factory=list)
    k_spring: float = 0.5
    damping: float = 0.15
    dt: float = 0.05

    def __post_init__(self):
        rng = np.random.default_rng(0)
        if self.positions is None:
            self.positions = self.V3.copy() + rng.standard_normal((12, 3)) * 0.05
        if self.velocities is None:
            self.velocities = rng.standard_normal((12, 3)) * 0.02
        self.history.append(self.positions.copy())

    # ---------- pairwise coupling between fermions ----------
    def _pair_force(self) -> np.ndarray:
        """
        For each fermion a (vertex index 0..11), compute the force
        from all other fermions b, weighted by the even-even coupling
        strength at the corresponding event pair.
        """
        N_even = self.G_even.shape[0] // 2   # number of even events
        # map fermion index a to an even event slot via modulo
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

    # ---------- one step ----------
    def step(self) -> None:
        F_couple = self._pair_force()
        F_restore = -self.k_spring * (self.positions - self.V3)
        F_total = F_couple + F_restore

        self.velocities += F_total * self.dt
        self.velocities *= (1.0 - self.damping * self.dt)
        self.positions += self.velocities * self.dt
        self.history.append(self.positions.copy())

    def run(self, n_steps: int = 120) -> None:
        for _ in range(n_steps):
            self.step()

    @property
    def trajectory(self) -> np.ndarray:
        return np.array(self.history)      # (T, 12, 3)


# ============================================================
#  Event generator  ·  geo-location style
# ============================================================
def make_geo_events(n_events: int = 80,
                    seed: int = 2024) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthetic geo-location events:
        t_n = time-of-flight          (exponential, then sorted)
        r_n = pseudo-range rank       (monotone count)
    """
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
    print("Quantum geo-location  ·  even-even fermion coupling channel")
    print("=" * 78)
    print(f"  α = 1/(π − e)     = {ALPHA:.6f}")
    print(f"  φ = (1+√5)/2      = {PHI:.6f}")
    print(f"  τ(3)              = 12 icosahedron fermions")
    print(f"  entangle mask     : (i mod 2 == 0) AND (j mod 2 == 0)")
    print()

    # ---------- events ----------
    N = 80
    t, r = make_geo_events(n_events=N, seed=2024)
    print(f"--- events ---")
    print(f"  N events          = {N}")
    print(f"  t range           = [{t.min():.4f}, {t.max():.4f}]")
    print(f"  r range           = [{r.min():.1f}, {r.max():.1f}]")
    print()

    # ---------- standard coupling matrix ----------
    G = coupling_matrix(t, r)
    print(f"--- coupling matrix  g_ij = w_ij · κ_ij ---")
    off = G[G > 0]
    print(f"  ⟨g_ij⟩            = {off.mean():.6f}")
    print(f"  max g_ij          = {G.max():.6f}")
    print(f"  sparsity          = {1.0 - np.count_nonzero(G) / (N * N):.4f}")
    print()

    # ---------- even-even entanglement mask ----------
    G_even, mask = entangled_coupling(G)
    kept = np.count_nonzero(G_even)
    total = N * N
    print(f"--- even-even entanglement mask ---")
    print(f"  retained entries  = {kept} / {total}")
    print(f"  even subgraph     = {kept / (N//2 * (N//2 - 1)):.4f} of full even²")
    print(f"  ⟨g_even⟩          = {G_even[G_even > 0].mean() if kept else 0.0:.6f}")
    print(f"  max g_even        = {G_even.max():.6f}")
    print()

    # ---------- kissing reconstruction ----------
    even_idx = even_indices(N)
    t_even = t[even_idx]
    r_even = r[even_idx]
    Pis, mags = spinor_reconstruct(t_even, r_even)
    print(f"--- kissing reconstruction  ·  even events only ---")
    print(f"  n even events     = {len(t_even)}")
    print(f"  Π range (Re)      = [{Pis.real.min():+.4f}, {Pis.real.max():+.4f}]")
    print(f"  Π range (Im)      = [{Pis.imag.min():+.4f}, {Pis.imag.max():+.4f}]")
    print(f"  |Π| range         = [{mags.min():.4f}, {mags.max():.4f}]")
    print(f"  |Π| mean          = {mags.mean():.4f}")
    print()

    # ---------- signal transmission ----------
    s0 = np.zeros(N)
    s0[0] = 1.0                          # impulse at event 0 (even)
    s_hist = propagate_signal(G_even, s0, steps=30, decay=0.03)
    print(f"--- signal transmission  ·  even-even channel ---")
    print(f"  initial impulse   = 1.0 at event 0")
    print(f"  final amplitude   = {s_hist[-1, 0]:.6e}")
    print(f"  final mass        = Σ|s| = {np.sum(np.abs(s_hist[-1])):.6e}")
    print(f"  even mass         = {np.sum(np.abs(s_hist[-1][::2])):.6e}")
    print(f"  odd mass          = {np.sum(np.abs(s_hist[-1][1::2])):.6e}  (should be 0)")
    print()

    # ---------- 3D entanglement trajectory ----------
    V3 = icosahedron_vertices()
    sim = TrajectorySim(G_even=G_even, V3=V3)
    sim.run(n_steps=150)
    traj = sim.trajectory            # (T, 12, 3)
    print(f"--- 3D entanglement trajectory ---")
    print(f"  T frames          = {traj.shape[0]}")
    print(f"  displacement max  = "
          f"{np.max(np.linalg.norm(traj[-1] - V3, axis=1)):.6f}")
    print(f"  displacement mean = "
          f"{np.mean(np.linalg.norm(traj[-1] - V3, axis=1)):.6f}")
    print()

    # ============================================================
    #  Visualization
    # ============================================================
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.36, wspace=0.30)

    # (a) standard coupling matrix
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(G, origin="lower", cmap="inferno", aspect="equal")
    ax.set_xlabel("event j"); ax.set_ylabel("event i")
    ax.set_title(f"standard coupling\n"
                 f"⟨g⟩ = {off.mean():.4f}   max = {G.max():.4f}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (b) even-even masked matrix
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(G_even, origin="lower", cmap="inferno", aspect="equal")
    ax.set_xlabel("event j"); ax.set_ylabel("event i")
    ax.set_title(f"even-even entanglement mask\n"
                 f"kept = {kept}   ⟨g⟩ = "
                 f"{G_even[G_even>0].mean() if kept else 0.0:.4f}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (c) |Π| for even events
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(t_even, mags, "o-", color="#8e44ad", ms=4, lw=1.2)
    ax.set_xlabel("time t (even events)")
    ax.set_ylabel(r"$|\Pi(Z_n)|$")
    ax.set_title(f"Kissing reconstruction  |Π|\n"
                 f"mean = {mags.mean():.4f}")
    ax.grid(alpha=0.3)

    # (d) signal propagation
    ax = fig.add_subplot(gs[1, 0])
    for k in [0, 5, 10, 20, 30]:
        ax.plot(np.arange(N), s_hist[k], lw=1.0, alpha=0.8,
                label=f"step {k}")
    ax.scatter(even_idx, np.zeros(len(even_idx)),
               marker="|", color="k", s=200, zorder=5,
               label="even indices")
    ax.set_xlabel("event n"); ax.set_ylabel("signal s_n")
    ax.set_title("Signal diffusion through even-even channel")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (e) 3D entanglement trajectory
    ax = fig.add_subplot(gs[1, 1], projection="3d")
    for a in range(12):
        path = traj[:, a, :]
        ax.plot(path[:, 0], path[:, 1], path[:, 2],
                lw=0.9, alpha=0.7)
        ax.scatter(path[0, 0], path[0, 1], path[0, 2],
                   color="gray", s=20)
        ax.scatter(path[-1, 0], path[-1, 1], path[-1, 2],
                   color="crimson", s=40, edgecolors="k")
    # reference icosahedron
    for a in range(12):
        for b in range(a + 1, 12):
            if np.linalg.norm(V3[a] - V3[b]) < 1.2:
                ax.plot([V3[a, 0], V3[b, 0]],
                        [V3[a, 1], V3[b, 1]],
                        [V3[a, 2], V3[b, 2]],
                        color="gray", lw=0.4, alpha=0.4)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("3D fermion trajectory\n"
                 "grey = initial · red = final")

    # (f) mass budget
    ax = fig.add_subplot(gs[1, 2])
    total_mass_hist = np.sum(np.abs(s_hist), axis=1)
    even_mass_hist = np.sum(np.abs(s_hist[:, ::2]), axis=1)
    odd_mass_hist = np.sum(np.abs(s_hist[:, 1::2]), axis=1)
    ax.semilogy(total_mass_hist, "k-", lw=1.4, label="total")
    ax.semilogy(even_mass_hist, "b-", lw=1.4, label="even")
    ax.semilogy(odd_mass_hist + 1e-18, "r:", lw=1.4, label="odd (0)")
    ax.set_xlabel("step"); ax.set_ylabel(r"$\sum |s_n|$")
    ax.set_title("Signal mass budget")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

    plt.suptitle(
        "Quantum geo-location  ·  even-even entanglement  ·  "
        "kissing reconstruction τ(3) = 12  ·  "
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
    print("  1. standard coupling    g_ij = w_ij · κ_ij")
    print("  2. even-even mask       retain only (i, j) both even")
    print("  3. kissing reconstruction")
    print("       · τ(3) = 12 icosahedron fermions in ℝ⁶")
    print("       · each even event → 6×6 spinor Z_n")
    print("       · Π(Z_n) = det(Z_n) = 720-term Levi-Civita")
    print("  4. signal transmission  s_n^{(t+1)} = Σ_j G_nj s_j / Σ_j G_nj")
    print("  5. 3D trajectory        12 fermions under even-even forces")
    print()
    print("  Key invariants:")
    print(f"    retained pairs     : {kept} / {total}")
    print(f"    |Π| mean           : {mags.mean():.4f}")
    print(f"    final signal mass  : {np.sum(np.abs(s_hist[-1])):.6e}")
    print(f"    odd mass           : {np.sum(np.abs(s_hist[-1][1::2])):.6e}")
    print(f"    mean displacement  : "
          f"{np.mean(np.linalg.norm(traj[-1] - V3, axis=1)):.6f}")
    print()
    print("  Odd-index pairs are identically zero — by construction")
    print("  the entanglement channel lives only on the even subgraph.")
    print()
    print("Done.")


if __name__ == "__main__":
    demo()