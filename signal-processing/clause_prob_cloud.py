#!/usr/bin/env python3
"""
vertex_clause.py
================

A clause for the 6D / 12-vertex configuration.

    INPUT  V ∈ ℝ^{N×6}
        N = 12  →  REJECTED (not read, not picked up)
        N = 6   →  ACCEPTED  (3D-embedded 6-vertex)

    ACCEPTED 6-vertex drives:

        • electron probability cloud
              |ψ(t)|² = |E(t) + iB(t)|²
          where E(t) = Re ζ(t), B(t) = Im ζ(t) come from the
          Möbius spectral sum weighted by the Figure 3.5 projection

        • amplitude envelope of the set signal
              A_i  = ||V6[i, :]||   (row norms, normalised)
              A_full(t) = A_i · Π(t, r_i)

    Figure 3.5:
        Π(t, r) = ½ (1 + cos(2π t/π) · cos(2π r/e))  ∈ [0, 1]

    The clause never inspects, memoises, or returns anything from
    a 12-vertex input: it is discarded on the first check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import permutations
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ============================================================
#  Constants
# ============================================================
PI         = math.pi
E          = math.e
ALPHA_SYM  = 1.0 / (PI - E)             # ≈ 2.362
DENSITY    = 6.0 / (PI * PI)            # ≈ 0.6079
W1         = PI
W2         = E

N_ACCEPT   = 6                          # accepted vertex count
N_REJECT   = 12                         # rejected vertex count


# ============================================================
#  Clause: accept 6-vertex, reject everything else
# ============================================================
@dataclass
class ClauseVerdict:
    """
    The result of evaluating the clause on a candidate vertex array.
    `accepted == False` means the input was discarded and NOTHING
    further was read from it.
    """
    accepted: bool
    n_vertices: int
    reason: str
    V6: Optional[np.ndarray] = None      # only set when accepted


def vertex_clause(V) -> ClauseVerdict:
    """
    The clause.

    Rules
    -----
    • V must be a numpy array of shape (N, 6).
    • N must equal exactly 6 (the 3D-embedded 6-vertex config).
    • Any N ≠ 6 — and in particular N = 12 — is REJECTED with the
      input discarded on the spot.

    The clause is a *predicate*.  It never reads the contents of a
    rejected array; it only consults `V.shape[0]`.
    """
    V = np.asarray(V)

    # ---- shape guard: only the vertex axis is examined for rejection ----
    if V.ndim != 2 or V.shape[1] != 6:
        return ClauseVerdict(False, 0,
                             "shape is not (N, 6); discarded")

    n = V.shape[0]

    # ---- 12-vertex is rejected outright (never read) ----
    if n == N_REJECT:
        return ClauseVerdict(False, n,
                             "12-vertex configuration rejected: "
                             "not read, not picked up")

    # ---- anything other than 6 is rejected ----
    if n != N_ACCEPT:
        return ClauseVerdict(False, n,
                             f"{n}-vertex configuration rejected")

    # ---- 6-vertex: accepted ----
    # the only read of the *contents* of V happens here, on acceptance
    return ClauseVerdict(True, n, "6-vertex accepted", V6=V.copy())


# ============================================================
#  Figure 3.5  ·  bounded elliptic projection
# ============================================================
def elliptic_projection(t: float, r: float) -> float:
    u = (t / W1) % 1.0
    v = (r / W2) % 1.0
    return 0.5 * (1.0 + math.cos(2 * PI * u) * math.cos(2 * PI * v))


def elliptic_projection_vec(t: np.ndarray, r: np.ndarray) -> np.ndarray:
    u = (t / W1) % 1.0
    v = (r / W2) % 1.0
    return 0.5 * (1.0 + np.cos(2 * PI * u) * np.cos(2 * PI * v))


# ============================================================
#  6-vertex amplitudes (row norms, normalised)
# ============================================================
def vertex_amplitudes(V6: np.ndarray) -> np.ndarray:
    """
    A_i = ||V6[i, :]|| / max_i ||V6[i, :]||     ∈ [0, 1]^6
    """
    a = np.linalg.norm(V6, axis=1)
    m = a.max() if a.size else 1.0
    return a / (m + 1e-12)


# ============================================================
#  Möbius spectral sum  ζ(t)
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


def zeta_mobius(t: float, mu: np.ndarray, K: int,
                alpha: float = ALPHA_SYM) -> complex:
    n = np.arange(1, K + 1)
    m = mu[1:K + 1].astype(np.float64)
    theta = t * n / alpha
    return complex(np.sum(m * np.cos(theta)),
                   np.sum(m * np.sin(theta)))


# ============================================================
#  Electron probability cloud  ·  driven by Figure 3.5
# ============================================================
@dataclass
class CloudResult:
    t: np.ndarray                       # time samples
    Pi: np.ndarray                      # Π(t, r_i) per vertex, (6, T)
    E: np.ndarray                       # E(t) = Re ζ(t), (6, T)
    B: np.ndarray                       # B(t) = Im ζ(t), (6, T)
    psi: np.ndarray                     # E + iB, (6, T)
    prob: np.ndarray                    # |ψ|², (6, T)
    prob_total: np.ndarray              # Σ_i |ψ_i|², (T,)
    amplitude: np.ndarray               # A_full = A_i · Π_i, (6, T)


def electron_probability_cloud(V6: np.ndarray,
                               r_vals: Optional[np.ndarray] = None,
                               t_vals: Optional[np.ndarray] = None,
                               K_sieve: int = 64,
                               n_samples: int = 256
                               ) -> CloudResult:
    """
    Build the electron probability cloud for the 6-vertex config.

    Parameters
    ----------
    V6      : (6, 6) accepted 6-vertex matrix
    r_vals  : (6,) rank coordinates for the Figure 3.5 projection
              (default: normalised row norms)
    t_vals  : (T,) time samples over one period 2πα_sym
    K_sieve : Möbius sieve cutoff
    """
    # --- amplitude envelope of the set signal ---
    A = vertex_amplitudes(V6)                       # (6,)

    # --- rank coordinates for the projection ---
    if r_vals is None:
        r_vals = A                                   # ∈ (0, 1]^6

    # --- time samples over one period ---
    if t_vals is None:
        period = 2 * PI * ALPHA_SYM
        t_vals = np.linspace(0.0, period, n_samples)
    T = len(t_vals)

    # --- Möbius spectral sum, one ζ per vertex ---
    mu = mobius_sieve(K_sieve)
    E = np.zeros((6, T))
    B = np.zeros((6, T))
    for i in range(6):
        # per-vertex phase offset from the amplitude
        phase_offset = r_vals[i]
        for j, t in enumerate(t_vals):
            z = zeta_mobius(t + phase_offset, mu, K_sieve)
            E[i, j] = z.real
            B[i, j] = z.imag

    # --- Figure 3.5 elliptic projection Π(t, r_i) ---
    Pi = np.zeros((6, T))
    for i in range(6):
        Pi[i, :] = elliptic_projection_vec(t_vals, r_vals[i])

    # --- probability cloud: weight |E + iB|² by Π ---
    psi = E + 1j * B
    prob = (np.abs(psi) ** 2) * Pi                  # |ψ_i|² · Π_i
    prob_total = np.sum(prob, axis=0)               # (T,)

    # --- amplitude envelope of the set signal ---
    # A_full_i(t) = A_i · Π(t, r_i)
    amplitude = Pi * A[:, None]                     # (6, T)

    return CloudResult(
        t=t_vals, Pi=Pi, E=E, B=B, psi=psi,
        prob=prob, prob_total=prob_total,
        amplitude=amplitude,
    )


# ============================================================
#  Main entry: apply the clause and compute the cloud
# ============================================================
def apply_clause(V,
                 K_sieve: int = 64,
                 n_samples: int = 256) -> Dict:
    """
    Apply the clause to `V`, then — only if accepted — compute the
    electron probability cloud and the amplitude envelope of the
    set signal.

    Returns a dict with:
        accepted      bool
        reason        str
        n_vertices    int
        amplitudes    (6,)   only if accepted
        cloud         CloudResult   only if accepted
        n_rejected    0 (rejected inputs are never read further)
    """
    verdict = vertex_clause(V)
    out: Dict = dict(
        accepted=verdict.accepted,
        reason=verdict.reason,
        n_vertices=verdict.n_vertices,
    )
    if not verdict.accepted:
        # nothing further is computed from a rejected input
        out["amplitudes"] = None
        out["cloud"] = None
        return out

    V6 = verdict.V6
    A = vertex_amplitudes(V6)
    cloud = electron_probability_cloud(V6, K_sieve=K_sieve,
                                       n_samples=n_samples)
    out["amplitudes"] = A
    out["cloud"] = cloud
    return out


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("Vertex clause  ·  6-vertex accepted  ·  12-vertex rejected")
    print("=" * 78)
    print(f"  α_sym = 1/(π − e) = {ALPHA_SYM:.6f}")
    print(f"  Figure 3.5 Π(t,r) = ½(1 + cos(2πt/π) · cos(2πr/e))")
    print()

    rng = np.random.default_rng(42)

    # ---- 12-vertex (must be rejected) ----
    V12 = rng.standard_normal((12, 6))
    r12 = apply_clause(V12)
    print("--- 12-vertex configuration ---")
    print(f"  accepted      : {r12['accepted']}")
    print(f"  reason        : {r12['reason']}")
    print(f"  n_vertices    : {r12['n_vertices']}")
    print(f"  amplitudes    : {r12['amplitudes']}")
    print(f"  cloud         : {r12['cloud']}")
    print()

    # ---- 6-vertex (must be accepted) ----
    V6 = rng.standard_normal((6, 6))
    r6 = apply_clause(V6, K_sieve=48, n_samples=192)
    print("--- 6-vertex configuration ---")
    print(f"  accepted      : {r6['accepted']}")
    print(f"  reason        : {r6['reason']}")
    print(f"  n_vertices    : {r6['n_vertices']}")
    print(f"  amplitudes Aᵢ : " +
          ", ".join(f"{a:.4f}" for a in r6["amplitudes"]))
    cloud = r6["cloud"]
    print(f"  Π range       : [{cloud.Pi.min():.4f}, "
          f"{cloud.Pi.max():.4f}]")
    print(f"  |ψ|² range    : [{cloud.prob.min():.4e}, "
          f"{cloud.prob.max():.4e}]")
    print(f"  Σ_i|ψ_i|² mean: {cloud.prob_total.mean():.4e}")
    print(f"  A_full range  : [{cloud.amplitude.min():.4f}, "
          f"{cloud.amplitude.max():.4f}]")
    print()

    # ---- other shapes (also rejected) ----
    print("--- other shapes ---")
    for n in (1, 3, 5, 7, 8, 10):
        v = rng.standard_normal((n, 6))
        r = apply_clause(v)
        print(f"  ({n:>2d}, 6)  accepted={r['accepted']}  "
              f"reason='{r['reason']}'")
    print()

    # ---- visualise ----
    if HAS_MPL:
        fig = plt.figure(figsize=(15, 9))
        gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

        # (a) 6-vertex amplitudes
        ax = fig.add_subplot(gs[0, 0])
        ax.bar(np.arange(6), r6["amplitudes"], color="#16a085",
               edgecolor="k", width=0.6)
        ax.set_xlabel("vertex i"); ax.set_ylabel("A_i")
        ax.set_title("Accepted 6-vertex amplitudes\n(row norms)")
        ax.grid(alpha=0.3)

        # (b) Figure 3.5 surface
        ax = fig.add_subplot(gs[0, 1])
        ts = np.linspace(0, W1, 200)
        rs = np.linspace(0, W2, 200)
        TT, RR = np.meshgrid(ts, rs)
        PP = 0.5 * (1 + np.cos(2 * np.pi * TT / W1)
                       * np.cos(2 * np.pi * RR / W2))
        im = ax.imshow(PP, extent=[0, W1, 0, W2], origin="lower",
                       cmap="viridis", aspect="auto", vmin=0, vmax=1)
        ax.set_xlabel("t (mod π)"); ax.set_ylabel("r (mod e)")
        ax.set_title("Figure 3.5  ·  Π(t, r) ∈ [0, 1]")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # (c) rejection bar
        ax = fig.add_subplot(gs[0, 2])
        verdicts = ["12-vertex", "6-vertex"]
        accepted = [0, 1]
        colors   = ["#e74c3c", "#16a085"]
        ax.bar(verdicts, accepted, color=colors, edgecolor="k", width=0.5)
        ax.set_ylim(0, 1.2)
        ax.set_ylabel("accepted (0/1)")
        ax.set_title("Clause verdict")
        for i, v in enumerate(accepted):
            ax.text(i, v + 0.05, "✓" if v else "✗",
                    ha="center", fontsize=14)
        ax.grid(alpha=0.3)

        # (d) electron probability cloud  ·  per vertex
        ax = fig.add_subplot(gs[1, :2])
        for i in range(6):
            ax.plot(cloud.t, cloud.prob[i, :],
                    lw=1.0, alpha=0.75, label=f"vertex {i}")
        ax.plot(cloud.t, cloud.prob_total,
                color="k", lw=1.6, label="Σ_i |ψ_i|²")
        ax.set_xlabel("t"); ax.set_ylabel("|ψ|² · Π")
        ax.set_title("Electron probability cloud  ·  |ψ|² = |E + iB|² weighted by Π")
        ax.legend(fontsize=8, ncol=4); ax.grid(alpha=0.3)

        # (e) amplitude envelope
        ax = fig.add_subplot(gs[1, 2])
        ax.plot(cloud.t, cloud.amplitude.T, lw=1.0, alpha=0.7)
        ax.set_xlabel("t"); ax.set_ylabel("A_full(t) = A_i · Π(t, r_i)")
        ax.set_title("Amplitude envelope of the set signal")
        ax.grid(alpha=0.3)

        plt.suptitle(
            "Vertex clause  ·  6-vertex accepted  ·  12-vertex rejected  ·  "
            "Figure 3.5 drives cloud + amplitude",
            fontsize=13)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

    print("Done.")


if __name__ == "__main__":
    demo()