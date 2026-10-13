#!/usr/bin/env python3
"""
quantum_geo_location.py
=======================
Quantum geo-location through an even-parity fermion channel
with kissing-number spinor reconstruction.

O(K) redesign
-------------
· coupling matrix       →  per-event μ-gated vectors  v_n ∈ ℝ^K
· g_ij = w_ij · κ_ij    →  cross_coupling(v_i, v_j) = |⟨v_i, v_j⟩| / (‖v_i‖·‖v_j‖)
· even-even mask        →  UGC label recovery  σ(v) ∈ {0..k-1}
· propagation           →  fiber kernel  H(z) = exp(−α|z|)

Kissing reconstruction and the 12-vertex trajectory are unchanged.

Pipeline
--------
1. Event signature  →  μ-gated vector
       v_n = μ(n) · cos(A_STEP · n² / K) · s(n mod L)
   where s(n) is the SHAKE256 signature of the event (t_n, r_n).

2. UGC label recovery  σ(v)  ∈ {0, …, k−1}
   even-parity events  ⇔  σ(v) ≡ 0 (mod 2)

3. Kissing-number reconstruction
       τ(3) = 12 icosahedron vertices lifted to ℝ⁶
       each even event → 6×6 spinor Z_n
       Π(Z_n) = det(Z_n) = 720-term Levi-Civita

4. Signal transmission through fiber kernel
       s_out = conv(s_in, H),  H(z) = exp(−α|z|) / Σ exp(−α|z|)

5. 3D entanglement trajectory
       12 fermion vertices move under:
         · harmonic restoring force to the icosahedron
         · pairwise coupling from the μ-gated cross-coupling
       (sparse: only pairs above a threshold interact)
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass, field
from itertools import permutations
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ============================================================
#  Constants
# ============================================================
PI         = math.pi
E          = math.e
ALPHA      = 1.0 / (PI - E)                # ≈ 2.362338
ALPHA_ASYM = 0.3628
A_STEP     = ALPHA / ALPHA_ASYM            # ≈ 6.511
PHI        = (1.0 + math.sqrt(5.0)) / 2.0
W1         = PI
W2         = E

K_VEC_DEF  = 256                          # length of the μ-gated vector
K_FILTER   = 2 ** 10 + 1
UINT64_MAX = 2 ** 64 - 1


# ============================================================
#  Möbius sieve  ·  O(K log log K)
# ============================================================
def mobius_sieve(K: int) -> np.ndarray:
    if K < 1:
        return np.zeros(K + 1, dtype=np.int8)
    mu = np.ones(K + 1, dtype=np.int8)
    is_comp = np.zeros(K + 1, dtype=bool)
    for i in range(2, K + 1):
        if not is_comp[i]:
            mu[i::i] = -mu[i::i]
            is_comp[i::i] = True
            i2 = i * i
            if i2 <= K:
                mu[i2::i2] = 0
    mu[0] = 0
    return mu


# ============================================================
#  μ-gated event vector  ·  O(K)
# ============================================================
def event_vector(t_n: float, r_n: float,
                 K: int, mu: np.ndarray) -> np.ndarray:
    """
    Length-K μ-gated vector for a single event:

        v(n) = μ(n) · cos(A_STEP · n² / K) · s(n mod L)

    where s(n) is the SHAKE256 signature of the event.
    """
    sig = hashlib.shake_256(
        struct.pack('>dd', t_n, r_n) + b"|evt"
    ).digest(32)
    s = np.array([b / 255.0 for b in sig], dtype=float)
    L = len(s)

    n = np.arange(K)
    mu_arr = np.array([float(mu[k]) if k < len(mu) else 0.0
                       for k in range(K)])
    chirp = np.cos(A_STEP * n * n / K)
    samp = s[n % L]
    return mu_arr * chirp * samp


def build_all_vectors(t: np.ndarray, r: np.ndarray,
                      K: int, mu: np.ndarray) -> np.ndarray:
    """Stack N event vectors into an (N, K) matrix."""
    N = len(t)
    V = np.zeros((N, K), dtype=float)
    for i in range(N):
        V[i] = event_vector(t[i], r[i], K, mu)
    return V


# ============================================================
#  Cross-coupling  ·  O(K)
# ============================================================
def cross_coupling(v_a: np.ndarray, v_b: np.ndarray) -> float:
    """
    Normalised cross-coupling ∈ [0, 1]:

        cross = |⟨v_a, v_b⟩| / (‖v_a‖ · ‖v_b‖)
    """
    na = float(np.linalg.norm(v_a))
    nb = float(np.linalg.norm(v_b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(abs(np.dot(v_a, v_b)) / (na * nb))


def cross_coupling_matrix(V: np.ndarray) -> np.ndarray:
    """
    N×N matrix of cross-couplings, for visualisation only.

    Never used to drive the pipeline — the pipeline uses sparse
    pairwise cross_coupling(v_i, v_j).
    """
    N = V.shape[0]
    norms = np.linalg.norm(V, axis=1) + 1e-12
    Vn = V / norms[:, None]
    G = np.abs(Vn @ Vn.T)
    np.fill_diagonal(G, 0.0)
    return G


# ============================================================
#  UGC label recovery
# ============================================================
def vector_phase(v: np.ndarray) -> float:
    """
    Phase of the μ-gated vector.

    Use the first two large-magnitude entries as (re, im).
    """
    nz = np.argsort(-np.abs(v))[:2]
    if len(nz) < 2:
        return 0.0
    re = v[nz[0]]
    im = v[nz[1]]
    return math.atan2(im, re)


def ugc_label_recovery(V: np.ndarray, k: int) -> np.ndarray:
    """
    Assign each event a label σ(v) ∈ {0, …, k−1}:

        σ(v) = round(k · φ(v) / 2π) mod k

    where φ(v) is the phase of the μ-gated vector.
    """
    N = V.shape[0]
    labels = np.zeros(N, dtype=int)
    for i in range(N):
        phi = vector_phase(V[i])
        labels[i] = int(round(k * phi / (2 * PI))) % k
    return labels


def even_parity_mask(labels: np.ndarray) -> np.ndarray:
    """Boolean mask: True where label is even."""
    return (labels % 2) == 0


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


# ---------- 720-term Levi-Civita ----------
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


def event_spinor(t_n: float, r_n: float,
                 V6: np.ndarray) -> np.ndarray:
    """
    Map an even event to a 6×6 complex spinor:

        Z_n[i, j] = V6[i, j] · e^{i · (t_n·i + r_n·j)}
    """
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
#  Fiber kernel  ·  exp(−α|z|)  ·  O(K log K)
# ============================================================
def fiber_kernel(width: int, alpha: float = ALPHA) -> np.ndarray:
    x = np.arange(-width // 2, width // 2)
    h = np.exp(-alpha * np.abs(x))
    return h / h.sum()


def fiber_propagate(signal: np.ndarray,
                    width: int = 41,
                    alpha: float = ALPHA) -> np.ndarray:
    """Convolve the signal with the exponential fiber kernel."""
    h = fiber_kernel(width, alpha)
    L = len(signal)
    N = 1 << (2 * L - 1).bit_length()
    sig_pad = np.pad(signal, (0, N - L))
    ker_pad = np.pad(h, (0, N - len(h)))
    y = np.fft.ifft(np.fft.fft(sig_pad) * np.fft.fft(ker_pad))[:L]
    return np.real(y)


# ============================================================
#  Signal transmission through the even-parity channel
# ============================================================
def propagate_signal(V: np.ndarray,
                     mask: np.ndarray,
                     s0: np.ndarray,
                     steps: int = 20,
                     decay: float = 0.05,
                     threshold: float = 0.3,
                     fiber_width: int = 41) -> np.ndarray:
    """
    Sparse even-parity diffusion:

        s_n^{(k+1)} = Σ_{j ∈ mask, g_nj ≥ τ} g_nj · s_j / Σ g_nj

    g_nj = cross_coupling(v_n, v_j)   (only computed for kept pairs)

    A final fiber-kernel pass adds dispersion.
    """
    N = V.shape[0]
    # precompute sparse even-parity neighbours above threshold
    neigh: List[List[Tuple[int, float]]] = [[] for _ in range(N)]
    for i in range(N):
        if not mask[i]:
            continue
        for j in range(N):
            if not mask[j] or i == j:
                continue
            g = cross_coupling(V[i], V[j])
            if g >= threshold:
                neigh[i].append((j, g))

    s = s0.copy().astype(float)
    # zero out odd-parity slots
    s[~mask] = 0.0
    hist = [s.copy()]

    for _ in range(steps):
        s_new = np.zeros_like(s)
        for i in range(N):
            if not mask[i] or not neigh[i]:
                continue
            total_w = sum(g for _, g in neigh[i])
            for j, g in neigh[i]:
                s_new[i] += g * s[j]
            s_new[i] /= (total_w + 1e-12)
        s_new *= (1.0 - decay)
        s_new[~mask] = 0.0
        s = s_new
        hist.append(s.copy())

    # final fiber dispersion pass
    s_fiber = fiber_propagate(s, width=fiber_width)
    hist.append(s_fiber)
    return np.array(hist)


# ============================================================
#  3D entanglement trajectory
# ============================================================
@dataclass
class TrajectorySim:
    """
    Simulate 12 kissing-number fermions in 3D under:

        · harmonic restoring force to the icosahedron
        · pairwise coupling from the μ-gated cross-coupling

    Only pairs above a threshold interact (sparse).
    """
    V: np.ndarray                     # (N, K) μ-gated vectors
    mask: np.ndarray                  # (N,) even-parity
    V3: np.ndarray                    # (12, 3) reference
    threshold: float = 0.3
    k_spring: float = 0.5
    damping: float = 0.15
    dt: float = 0.05
    positions: Optional[np.ndarray] = None
    velocities: Optional[np.ndarray] = None
    history: List[np.ndarray] = field(default_factory=list)

    def __post_init__(self):
        rng = np.random.default_rng(0)
        if self.positions is None:
            self.positions = (self.V3.copy()
                              + rng.standard_normal((12, 3)) * 0.05)
        if self.velocities is None:
            self.velocities = rng.standard_normal((12, 3)) * 0.02
        self.history.append(self.positions.copy())

        # map fermion index a ∈ {0..11} to an event index
        even_idx = np.where(self.mask)[0]
        if len(even_idx) == 0:
            even_idx = np.arange(self.V.shape[0])
        self.slot = np.array(
            [even_idx[a % len(even_idx)] for a in range(12)])

    # ---------- pairwise coupling between fermions ----------
    def _pair_force(self) -> np.ndarray:
        force = np.zeros_like(self.positions)
        for a in range(12):
            for b in range(12):
                if a == b:
                    continue
                i, j = self.slot[a], self.slot[b]
                g = cross_coupling(self.V[i], self.V[j])
                if g < self.threshold:
                    continue
                diff = self.positions[b] - self.positions[a]
                r = np.linalg.norm(diff) + 1e-6
                force[a] += g * diff / (r ** 2 + 0.1)
        return force

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
def demo(K_vec: int = K_VEC_DEF, k_labels: int = 4):
    print("=" * 78)
    print("Quantum geo-location  ·  O(K) μ-gated + UGC labels + fiber kernel")
    print("=" * 78)
    print(f"  α = 1/(π − e)     = {ALPHA:.6f}")
    print(f"  A_STEP            = {A_STEP:.6f}")
    print(f"  K_vec (μ-gate)    = {K_vec}")
    print(f"  k_labels          = {k_labels}")
    print()

    # ---------- events ----------
    N = 80
    t, r = make_geo_events(n_events=N, seed=2024)
    print(f"--- events ---")
    print(f"  N events          = {N}")
    print(f"  t range           = [{t.min():.4f}, {t.max():.4f}]")
    print()

    # ---------- μ-gated vectors  ·  O(N·K) ----------
    mu = mobius_sieve(K_vec)
    V = build_all_vectors(t, r, K_vec, mu)
    print(f"--- μ-gated vectors ---")
    print(f"  shape             = {V.shape}")
    print(f"  μ non-zero        = "
          f"{np.count_nonzero(mu[1:K_vec + 1])} / {K_vec} "
          f"= {np.count_nonzero(mu[1:K_vec + 1]) / K_vec:.4f}")
    print()

    # ---------- UGC labels  ·  even parity mask ----------
    labels = ugc_label_recovery(V, k_labels)
    mask = even_parity_mask(labels)
    n_even = int(np.count_nonzero(mask))
    print(f"--- UGC label recovery ---")
    print(f"  label range       = [{labels.min()}, {labels.max()}]")
    print(f"  even-parity events= {n_even} / {N}  "
          f"= {n_even / N:.4f}")
    print()

    # ---------- cross-coupling summary (for display only) ----------
    G = cross_coupling_matrix(V)
    off = G[G > 0]
    print(f"--- cross-coupling summary (sparse) ---")
    print(f"  ⟨g_ij⟩            = {off.mean():.6f}")
    print(f"  max g_ij          = {G.max():.6f}")
    print(f"  pairs ≥ 0.3       = "
          f"{int(np.sum((G >= 0.3) & (G > 0))) // 2}")
    print(f"  (matrix NOT stored in the pipeline)")
    print()

    # ---------- kissing reconstruction ----------
    even_idx = np.where(mask)[0]
    t_even = t[even_idx]
    r_even = r[even_idx]
    Pis, mags = spinor_reconstruct(t_even, r_even)
    print(f"--- kissing reconstruction  ·  even events only ---")
    print(f"  n even events     = {len(t_even)}")
    print(f"  Π range (Re)      = [{Pis.real.min():+.4f}, "
          f"{Pis.real.max():+.4f}]")
    print(f"  |Π| range         = [{mags.min():.4f}, {mags.max():.4f}]")
    print(f"  |Π| mean          = {mags.mean():.4f}")
    print()

    # ---------- signal transmission ----------
    s0 = np.zeros(N)
    s0[0] = 1.0
    s_hist = propagate_signal(V, mask, s0,
                              steps=20, decay=0.03,
                              threshold=0.3, fiber_width=41)
    print(f"--- signal transmission  ·  even-parity + fiber kernel ---")
    print(f"  input impulse     = 1.0 at event 0")
    print(f"  final mass        = Σ|s| = "
          f"{np.sum(np.abs(s_hist[-1])):.6e}")
    print(f"  even mass         = "
          f"{np.sum(np.abs(s_hist[-1][mask])):.6e}")
    print(f"  odd mass          = "
          f"{np.sum(np.abs(s_hist[-1][~mask])):.6e}")
    print()

    # ---------- 3D entanglement trajectory ----------
    V3 = icosahedron_vertices()
    sim = TrajectorySim(V=V, mask=mask, V3=V3, threshold=0.3)
    sim.run(n_steps=150)
    traj = sim.trajectory
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

    # (a) μ-gated vector stack  (N, K)
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(V, origin="lower", cmap="RdBu_r",
                   aspect="auto", vmin=-1, vmax=1)
    ax.set_xlabel("slot k"); ax.set_ylabel("event n")
    ax.set_title(f"μ-gated vectors  ·  shape = {V.shape}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (b) label histogram
    ax = fig.add_subplot(gs[0, 1])
    counts = [int(np.sum(labels == i)) for i in range(k_labels)]
    colors = ["#16a085" if i % 2 == 0 else "#e74c3c"
              for i in range(k_labels)]
    ax.bar(np.arange(k_labels), counts, color=colors, edgecolor="k")
    ax.set_xlabel("label"); ax.set_ylabel("count")
    ax.set_title(f"UGC labels  ·  even-parity events = {n_even}/{N}")
    ax.grid(alpha=0.3, axis="y")

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
    for k_idx in [0, 3, 6, 12, -1]:
        s_k = s_hist[k_idx]
        label = ("fiber" if k_idx == -1 else f"step {k_idx}")
        ax.plot(np.arange(N), s_k, lw=1.0, alpha=0.8, label=label)
    ax.scatter(even_idx, np.zeros(len(even_idx)),
               marker="|", color="k", s=200, zorder=5,
               label="even parity")
    ax.set_xlabel("event n"); ax.set_ylabel("signal s_n")
    ax.set_title("Even-parity diffusion  +  fiber dispersion")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (e) 3D trajectory
    ax = fig.add_subplot(gs[1, 1], projection="3d")
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
    ax.set_title("3D fermion trajectory\n"
                 "grey = initial · red = final")

    # (f) mass budget
    ax = fig.add_subplot(gs[1, 2])
    total_mass = np.sum(np.abs(s_hist), axis=1)
    even_mass = np.sum(np.abs(s_hist[:, mask]), axis=1)
    odd_mass = np.sum(np.abs(s_hist[:, ~mask]), axis=1)
    ax.semilogy(total_mass, "k-", lw=1.4, label="total")
    ax.semilogy(even_mass, "b-", lw=1.4, label="even")
    ax.semilogy(odd_mass + 1e-18, "r:", lw=1.4, label="odd")
    ax.axvline(len(s_hist) - 1, color="g", ls=":",
               label="fiber pass")
    ax.set_xlabel("step"); ax.set_ylabel(r"$\sum |s_n|$")
    ax.set_title("Signal mass budget")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")

    plt.suptitle(
        "Quantum geo-location  ·  O(K) μ-gate  ·  UGC labels  ·  "
        "fiber kernel  ·  kissing reconstruction τ(3) = 12",
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # ============================================================
    #  Summary
    # ============================================================
    print("=" * 78)
    print("Pipeline summary")
    print("=" * 78)
    print("  1. μ-gated vectors      v_n = μ(n) · cos(A_STEP · n² / K) · s_n")
    print("  2. UGC labels           σ(v) = round(k · φ(v) / 2π) mod k")
    print("  3. even-parity          σ(v) ≡ 0 (mod 2)")
    print("  4. kissing              Π(Z_n) = det(Z_n) = 720-term LC")
    print("  5. sparse cross-coupling g_ij = |⟨v_i, v_j⟩| / (‖·‖·‖·‖)")
    print("  6. fiber kernel         H(z) = exp(−α|z|),  O(K log K)")
    print()
    print("  Key invariants:")
    print(f"    μ density           : "
          f"{np.count_nonzero(mu[1:K_vec+1]) / K_vec:.4f}  "
          f"(target 6/π² = {6/PI**2:.4f})")
    print(f"    even-parity events  : {n_even} / {N}")
    print(f"    |Π| mean            : {mags.mean():.4f}")
    print(f"    final signal mass   : "
          f"{np.sum(np.abs(s_hist[-1])):.6e}")
    print(f"    mean displacement   : "
          f"{np.mean(np.linalg.norm(traj[-1] - V3, axis=1)):.6f}")
    print()
    print("  Complexity vs old N×N approach:")
    print("    old:  O(N²) memory  +  O(N²) build  +  O(N²) per step")
    print("    new:  O(N·K) build  +  O(E·K) per step  (E = kept pairs)")
    print()
    print("Done.")


if __name__ == "__main__":
    demo(K_vec=K_VEC_DEF, k_labels=4)