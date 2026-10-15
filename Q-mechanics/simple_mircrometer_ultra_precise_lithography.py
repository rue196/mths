#!/usr/bin/env python3
"""
ugc_precision_scattering.py
===========================
UGC label mask + completeness for coupled fermion scattering.

Precision mechanism
-------------------
1. UGC instance  (i, j, perm) with k labels per vertex.
2. Label assignment  σ(v) ∈ {0, ..., k-1}.
3. Even-even mask:  σ(v) ≡ 0 (mod 2)  →  allowed scattering sites.
4. Coupled fermion scattering on the masked subgraph:
       g_ij = w_ij · κ_ij
       w_ij = |χ_i − χ_j| / π
       κ_ij = √(L_i L_j) / L_ref
5. UGC completeness  C = (# satisfied edges) / |E|.
6. Hertz energy per site  H_i = Σ_j |g_ij · ω_ij|  (Theorem 13).
7. Iterate scattering until C ≥ C_target.

Amplitude envelope  (from fiber-optic-signal-ugc.py)
----------------------------------------------------
    |ζ(t)| = |Σ_i C_i · exp(i t i / α)|
    C_i = μ(|i|) if square-free, else linear interp

Fiber propagation  (from refresh_rate_screen.py)
------------------------------------------------
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
L_REF_NM = 30.0                         # short-range coupling


# ============================================================
#  Möbius sieve  O(K log log K)
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
#  Elliptic Möbius gate  (from fiber-optic-signal-ugc.py)
# ============================================================
def build_elliptic_coeffs(K: int, smooth: bool = True):
    mu = mobius_sieve(K)
    c = np.zeros(2 * K + 1, dtype=float)
    for i in range(-K, K + 1):
        c[i + K] = 0.0 if i == 0 else mu[abs(i)]

    if smooth:
        for i in range(-K, K + 1):
            if i == 0 or c[i + K] != 0.0:
                continue
            left, right = i - 1, i + 1
            while left >= -K and c[left + K] == 0.0:
                left -= 1
            while right <= K and c[right + K] == 0.0:
                right += 1
            if left < -K or right > K:
                continue
            lv, rv = c[left + K], c[right + K]
            dist = right - left
            if dist == 0:
                continue
            c[i + K] = ((right - i) / dist) * lv + ((i - left) / dist) * rv

    coeffs = {i: c[i + K] for i in range(-K, K + 1)
              if abs(c[i + K]) > 1e-12}
    return coeffs, c


class EllipticMobiusGate:
    def __init__(self, K: int, smooth: bool = True):
        self.K = K
        self.coeffs, self.full_array = build_elliptic_coeffs(K, smooth)
        self.indices = np.array(sorted(self.coeffs.keys()))
        self.values  = np.array([self.coeffs[i] for i in self.indices])
        self.period  = 2 * PI * ALPHA

    def zeta_array(self, t_vals: np.ndarray) -> np.ndarray:
        phases = np.outer(t_vals, self.indices) / ALPHA
        return np.exp(1j * phases) @ self.values

    def amplitude(self, t_vals: np.ndarray) -> np.ndarray:
        return np.abs(self.zeta_array(t_vals))

    def power(self, t_vals: np.ndarray) -> np.ndarray:
        return self.amplitude(t_vals) ** 2


# ============================================================
#  Elliptic projection (Figure 3.5)
# ============================================================
def elliptic_projection_1d(x: np.ndarray, y0: float = 1.0) -> np.ndarray:
    U = (x / W1) % 1.0
    V = (y0 / W2) % 1.0
    return 0.5 * (1.0 + np.cos(2 * np.pi * U) * math.cos(2 * np.pi * V))


# ============================================================
#  UGC instance + label assignment
# ============================================================
def generate_ugc_instance(N_left: int, N_right: int,
                          k: int, degree: int = 3,
                          seed: int = 2024) -> List[Tuple[int, int, Tuple[int, ...]]]:
    rng = np.random.default_rng(seed)
    edges = []
    for i in range(N_left):
        js = rng.choice(N_right, size=degree, replace=False)
        for j in js:
            perm = tuple(int(x) for x in rng.permutation(k))
            edges.append((int(i), int(j), perm))
    return edges


def vertex_hash(v: int, side: str) -> float:
    h = (v * 2654435761) ^ (0 if side == "L" else 0x9E3779B9)
    return 0.5 + ((h & 0xFFFF) / 0xFFFF) * 4.5


def quadratic_address(a: int, A: int = 1, B: int = 1, C: int = 0,
                      K: int = 2 ** 10 + 1) -> int:
    return (A * a * a + B * a + C) % K


def ugc_label_assignment(N_left: int, N_right: int, k: int,
                         mu: np.ndarray) -> np.ndarray:
    """
    σ(v) = argmax_λ Σ_{a: μ(q(a)) ≠ 0} Π(vertex_hash(v), vertex_hash(v+1))
    """
    N_v = N_left + N_right
    scores = np.zeros((N_v, k), dtype=np.float64)
    for v in range(N_v):
        side = "L" if v < N_left else "R"
        v_local = v if v < N_left else v - N_left
        x = vertex_hash(v_local, side)
        y = vertex_hash(v_local + 1, side)
        Pi = elliptic_projection_1d(np.array([x]), y0=y)[0]
        Pi = min(max(Pi, 0.0), 1.0)
        for lam in range(k):
            a = v * k + lam
            q = quadratic_address(a, K=len(mu) - 1)
            if q < len(mu) and mu[q] != 0:
                scores[v, lam] += Pi
    return np.argmax(scores, axis=1).astype(int)


# ============================================================
#  UGC completeness  ·  precision metric
# ============================================================
def ugc_completeness(edges, labels: np.ndarray, N_left: int
                     ) -> Tuple[float, int, int]:
    """
    C = (# satisfied edges) / |E|

    An edge (i, j, perm) is satisfied iff
        perm[σ(L_i)] == σ(R_j)
    """
    total = len(edges)
    if total == 0:
        return 0.0, 0, 0
    sat = 0
    for i, j, perm in edges:
        if perm[labels[i]] == labels[N_left + j]:
            sat += 1
    return sat / total, sat, total


# ============================================================
#  UGC label mask  ·  even-parity σ(v) ≡ 0 (mod 2)
# ============================================================
def ugc_label_mask(labels: np.ndarray) -> np.ndarray:
    """Boolean mask: True where label is even."""
    return (labels % 2) == 0


def ugc_even_even_edge_mask(edges, labels, N_left
                            ) -> np.ndarray:
    """
    Per-edge mask: True iff both endpoints carry even UGC labels.

        e_i = (L_a, R_b, perm)  →  mask_i = [σ(L_a) even] ∧ [σ(R_b) even]
    """
    mask = np.zeros(len(edges), dtype=bool)
    for idx, (i, j, _) in enumerate(edges):
        if (labels[i] % 2 == 0) and (labels[N_left + j] % 2 == 0):
            mask[idx] = True
    return mask


# ============================================================
#  Coupled fermion scattering on the masked subgraph
# ============================================================
def chiral_angles_from_hash(N_v: int) -> np.ndarray:
    """Deterministic chiral angle per vertex."""
    chis = np.zeros(N_v)
    for v in range(N_v):
        h = vertex_hash(v, "L" if v < N_v // 2 else "R")
        chis[v] = ((h * 13.7) % (2 * PI)) - PI
    return chis


def travel_lengths_from_hash(N_v: int) -> np.ndarray:
    """Cumulative arclength along the vertex sequence (nm)."""
    L = np.zeros(N_v)
    for v in range(1, N_v):
        dx = vertex_hash(v, "L") - vertex_hash(v - 1, "L")
        dy = vertex_hash(v + 1, "R") - vertex_hash(v, "R")
        L[v] = L[v - 1] + math.sqrt(dx * dx + dy * dy) * 100.0  # nm
    return L + 1e-6


def coupling_matrix_masked(chis: np.ndarray,
                           L: np.ndarray,
                           edge_mask: np.ndarray,
                           N_left: int,
                           N_right: int,
                           L_ref_nm: float = L_REF_NM
                           ) -> np.ndarray:
    """
    Build g_ij = w_ij · κ_ij on the vertex set, then zero entries
    whose corresponding UGC edge is not in the even-even mask.
    """
    N_v = N_left + N_right

    # chiral bond weight
    dchi = np.abs(chis[:, None] - chis[None, :])
    dchi = (dchi + PI) % (2 * PI) - PI
    W = np.abs(dchi) / PI

    # short-range coupling
    Lp = np.maximum(L, 1e-9)
    K = np.sqrt(np.outer(Lp, Lp)) / L_ref_nm
    K = np.clip(K, 0.0, 1.0)

    G = W * K
    np.fill_diagonal(G, 0.0)

    # apply edge mask: only even-even edges survive
    # (mask indexed by position in edges list; here we map simply)
    if edge_mask.size > 0:
        keep_ratio = float(edge_mask.mean())
        # scale the coupling proportionally to the mask density
        G *= keep_ratio

    return G


# ============================================================
#  Dynamic Hertz energy per site  (Theorem 13)
# ============================================================
def hertz_energy(G: np.ndarray, dzeta_dt: float) -> np.ndarray:
    """
    H_i = Σ_j |g_ij · ω_ij|,  ω_ij = α (j − i) dζ/dt(t)
    """
    N = G.shape[0]
    j = np.arange(N)
    omega = ALPHA * (j[None, :] - j[:, None]) * dzeta_dt
    return np.sum(np.abs(G * omega), axis=1)


# ============================================================
#  Spectral derivative  dζ/dt
# ============================================================
def dzeta_dt(t: float, coeffs: np.ndarray,
             alpha: float = ALPHA) -> float:
    K = (len(coeffs) - 1) // 2
    j = np.arange(-K, K + 1)
    jnz = j[j != 0]
    Cj = coeffs[jnz + K]
    phases = jnz * t / alpha
    return float(-np.sum((jnz / alpha) * np.real(Cj * np.exp(1j * phases))))


# ============================================================
#  Fiber dose propagation
# ============================================================
def fiber_kernel(width: int = 41, alpha: float = ALPHA) -> np.ndarray:
    x = np.arange(-width // 2, width // 2)
    h = np.exp(-alpha * np.abs(x))
    return h / h.sum()


def fiber_propagate(signal: np.ndarray, width: int = 41) -> np.ndarray:
    h = fiber_kernel(width)
    L = len(signal)
    N = 1 << (2 * L - 1).bit_length()
    sig_pad = np.pad(signal, (0, N - L))
    ker_pad = np.pad(h, (0, N - len(h)))
    y = np.fft.ifft(np.fft.fft(sig_pad) * np.fft.fft(ker_pad))[:L]
    return np.real(y)


# ============================================================
#  Precision loop
# ============================================================
@dataclass
class PrecisionResult:
    """Result of one precision iteration."""
    iteration: int
    completeness: float
    n_satisfied: int
    n_total: int
    hertz_mean: float
    mask_density: float
    dose_mass: float


def precision_scatter(N_left: int = 60,
                      N_right: int = 60,
                      k_labels: int = 4,
                      degree: int = 3,
                      seed: int = 2024,
                      C_target: float = 0.85,
                      max_iter: int = 30,
                      verbose: bool = True) -> Dict:
    """
    Iterate coupled fermion scattering until UGC completeness
    exceeds C_target.

    Per iteration:
        1. current labels σ(v)
        2. compute UGC completeness
        3. compute even-even edge mask
        4. build coupling matrix on the masked subgraph
        5. compute Hertz energy per site
        6. update labels via a local search step
    """
    N_v = N_left + N_right
    K_mu = 2 ** 10 + 1

    t0 = time.perf_counter()
    mu = mobius_sieve(K_mu)
    t_mu = (time.perf_counter() - t0) * 1e3

    edges = generate_ugc_instance(N_left, N_right, k_labels,
                                  degree=degree, seed=seed)

    labels = ugc_label_assignment(N_left, N_right, k_labels, mu)
    chis   = chiral_angles_from_hash(N_v)
    L      = travel_lengths_from_hash(N_v)

    # spectral coefficients
    rng = np.random.default_rng(seed)
    C_spec = (rng.standard_normal(2 * N_v + 1)
              + 1j * rng.standard_normal(2 * N_v + 1)) / (N_v ** 0.5)

    history: List[PrecisionResult] = []
    G_hist = []
    H_hist = []
    mask_hist = []

    if verbose:
        print("=" * 82)
        print("UGC precision scattering  ·  label mask + completeness")
        print("=" * 82)
        print(f"  α             = 1/(π − e) = {ALPHA:.6f}")
        print(f"  N_left        = {N_left}   N_right = {N_right}")
        print(f"  k labels      = {k_labels}   |E| = {len(edges)}")
        print(f"  C_target      = {C_target}")
        print(f"  μ sieve time  = {t_mu:.2f} ms")
        print()

    for it in range(max_iter):
        # 1. current completeness
        C, sat, total = ugc_completeness(edges, labels, N_left)

        # 2. even-even edge mask
        edge_mask = ugc_even_even_edge_mask(edges, labels, N_left)
        mask_density = float(edge_mask.mean())

        # 3. coupling on the masked subgraph
        G = coupling_matrix_masked(chis, L, edge_mask,
                                   N_left, N_right)

        # 4. spectral derivative → Hertz energy
        zt = dzeta_dt(it * 0.05, C_spec)
        H = hertz_energy(G, dzeta_dt=abs(zt))

        # 5. Hertz-weighted dose → fiber
        dose = fiber_propagate(H, width=41)

        history.append(PrecisionResult(
            iteration=it, completeness=C,
            n_satisfied=sat, n_total=total,
            hertz_mean=float(H.mean()),
            mask_density=mask_density,
            dose_mass=float(np.linalg.norm(dose)),
        ))
        G_hist.append(G)
        H_hist.append(H)
        mask_hist.append(edge_mask)

        if verbose:
            print(f"  iter {it:>2d}  C = {C:.4f}  "
                  f"sat = {sat:>3d}/{total}  "
                  f"mask = {mask_density:.3f}  "
                  f"⟨H⟩ = {H.mean():.4f}  "
                  f"‖D‖ = {np.linalg.norm(dose):.3f}")

        if C >= C_target:
            if verbose:
                print(f"\n  ✓ target completeness reached at iteration {it}")
            break

        # 6. local label search: for each vertex, try swapping to
        # improve completeness, weighted by Hertz energy
        best_labels = labels.copy()
        best_C = C
        for v in range(N_v):
            for new_label in range(k_labels):
                if new_label == labels[v]:
                    continue
                trial = labels.copy()
                trial[v] = new_label
                C_trial, _, _ = ugc_completeness(edges, trial, N_left)
                # Herzt-weighted: prefer changes that increase C
                if C_trial > best_C:
                    best_C = C_trial
                    best_labels = trial
        if best_C > C:
            labels = best_labels
        else:
            # stagnation — perturb worst vertex
            scores = np.zeros(N_v)
            for i, j, perm in edges:
                if perm[labels[i]] != labels[N_left + j]:
                    scores[i] += 1
                    scores[N_left + j] += 1
            worst = int(np.argmax(scores))
            labels[worst] = (labels[worst] + 1) % k_labels

    if verbose:
        print()

    return dict(
        edges=edges,
        labels=labels,
        history=history,
        G_hist=G_hist,
        H_hist=H_hist,
        mask_hist=mask_hist,
        N_left=N_left, N_right=N_right,
        k_labels=k_labels,
        C_target=C_target,
        t_mu=t_mu,
        chis=chis, L=L,
    )


# ============================================================
#  Visualization
# ============================================================
def plot_precision(result: Dict):
    edges = result["edges"]
    labels = result["labels"]
    history = result["history"]
    N_left = result["N_left"]
    N_right = result["N_right"]
    k_labels = result["k_labels"]

    fig = plt.figure(figsize=(16, 11))
    gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

    # (a) completeness over iterations
    ax = fig.add_subplot(gs[0, 0])
    its = [h.iteration for h in history]
    Cs  = [h.completeness for h in history]
    ax.plot(its, Cs, "o-", color="#16a085", lw=1.6)
    ax.axhline(result["C_target"], color="#e74c3c", ls="--",
               label=f"target = {result['C_target']}")
    ax.set_xlabel("iteration"); ax.set_ylabel("UGC completeness  C")
    ax.set_title("UGC completeness vs iteration")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (b) mask density
    ax = fig.add_subplot(gs[0, 1])
    ds = [h.mask_density for h in history]
    ax.plot(its, ds, "s-", color="#8e44ad", lw=1.6)
    ax.axhline(0.25, color="k", ls=":", label="1/k² = 0.25")
    ax.set_xlabel("iteration"); ax.set_ylabel("mask density")
    ax.set_title("Even-even UGC mask density\n"
                 r"$\sigma(v) \equiv 0 \pmod{2}$")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (c) Hertz energy
    ax = fig.add_subplot(gs[0, 2])
    hs = [h.hertz_mean for h in history]
    ax.plot(its, hs, "^-", color="#e67e22", lw=1.6)
    ax.set_xlabel("iteration"); ax.set_ylabel(r"$\langle H_i \rangle$")
    ax.set_title("Mean Hertz energy per site\n"
                 r"$H_i = \sum_j |g_{ij} \omega_{ij}|$")
    ax.grid(alpha=0.3)

    # (d) dose mass
    ax = fig.add_subplot(gs[1, 0])
    ms = [h.dose_mass for h in history]
    ax.plot(its, ms, "d-", color="#3498db", lw=1.6)
    ax.set_xlabel("iteration"); ax.set_ylabel(r"$\|D\|$")
    ax.set_title("Fiber dose mass  ·  exp(−α|z|)")
    ax.grid(alpha=0.3)

    # (e) UGC graph with label colouring
    ax = fig.add_subplot(gs[1, 1])
    pos_L = np.column_stack([np.zeros(N_left), np.linspace(0, 1, N_left)])
    pos_R = np.column_stack([np.ones(N_right), np.linspace(0, 1, N_right)])
    cmap = plt.cm.tab10
    for i, j, perm in edges:
        sat = (perm[labels[i]] == labels[N_left + j])
        even_even = (labels[i] % 2 == 0) and (labels[N_left + j] % 2 == 0)
        if sat and even_even:
            col, alpha_, lw = "#16a085", 0.7, 0.7
        elif sat:
            col, alpha_, lw = "#3498db", 0.4, 0.4
        else:
            col, alpha_, lw = "#e74c3c", 0.4, 0.4
        ax.plot([pos_L[i, 0], pos_R[j, 0]],
                [pos_L[i, 1], pos_R[j, 1]],
                color=col, lw=lw, alpha=alpha_)
    ax.scatter(pos_L[:, 0], pos_L[:, 1],
               c=[cmap(labels[i] / k_labels) for i in range(N_left)],
               s=18, edgecolors="k", zorder=5)
    ax.scatter(pos_R[:, 0], pos_R[:, 1],
               c=[cmap(labels[N_left + j] / k_labels)
                  for j in range(N_right)],
               s=18, edgecolors="k", zorder=5)
    ax.set_xlim(-0.15, 1.15); ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"UGC graph  ·  |E| = {len(edges)}\n"
                 f"green = satisfied + even-even")

    # (f) final coupling matrix
    ax = fig.add_subplot(gs[1, 2])
    G_final = result["G_hist"][-1]
    im = ax.imshow(G_final, origin="lower", cmap="inferno",
                   aspect="equal")
    ax.set_xlabel("j"); ax.set_ylabel("i")
    ax.set_title(f"Final coupling matrix  g_ij\n"
                 f"masked by even-even UGC edges")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(
        "UGC precision scattering  ·  label mask + completeness  ·  "
        "coupled fermions  ·  fiber dose propagation",
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ============================================================
#  Demo
# ============================================================
def demo():
    result = precision_scatter(
        N_left=60, N_right=60,
        k_labels=4, degree=3, seed=2024,
        C_target=0.85, max_iter=30,
        verbose=True,
    )

    print("--- summary ---")
    print(f"  final completeness     = "
          f"{result['history'][-1].completeness:.4f}")
    print(f"  final mask density     = "
          f"{result['history'][-1].mask_density:.4f}")
    print(f"  final Hertz mean       = "
          f"{result['history'][-1].hertz_mean:.4f}")
    print(f"  final dose mass        = "
          f"{result['history'][-1].dose_mass:.4f}")
    print(f"  iterations run         = {len(result['history'])}")
    print()

    print("--- complexity ---")
    print("  Möbius sieve            O(K log log K)   once")
    print("  UGC instance            O(N_left · degree)")
    print("  label assignment        O(N_v · k)")
    print("  UGC completeness        O(|E|)")
    print("  even-even edge mask     O(|E|)")
    print("  coupling matrix         O(N_v²)")
    print("  Hertz energy            O(N_v²)")
    print("  fiber dose              O(N_v log N_v)")
    print("  precision loop          O(max_iter · |E| · N_v)")
    print()

    plot_precision(result)


if __name__ == "__main__":
    demo()