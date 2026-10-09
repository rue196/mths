#!/usr/bin/env python3
"""
Bond-fluid animation — O(K log K) build, O(K) per step, O(1) artist updates.

Key optimizations vs. jupyterflu6.py
------------------------------------
1. cKDTree.query(..., workers=-1)  → parallel O(K log K) neighbour search
2. I_FULL hoisted out of bond_weight() (was recomputed E times)
3. zeta / dzeta_dt / supertrace vectorized with numpy (no Python loops)
4. Bonds drawn as a single Line3DCollection artist → one set_segments per frame
5. Bond weights precomputed once (topology is fixed)
6. Positions updated in-place to avoid K×3 allocations per step
"""

from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial import cKDTree
from matplotlib.animation import FuncAnimation

# ============================================================
#  Constants
# ============================================================
PI    = math.pi
E     = math.e
ALPHA = 1.0 / (PI - E)                       # ≈ 2.362
I_FULL = (1.0 - math.exp(-ALPHA * (PI + E))) / ALPHA   # constant, hoisted

# ============================================================
#  Vectorized supertrace / entropy / invariant scalar
# ============================================================
def supertrace_from_coeffs(C: np.ndarray) -> float:
    """S = Σ (-1)^idx |C_idx|  — O(K), vectorized."""
    signs = np.where(np.arange(C.size) & 1, -1.0, 1.0)
    return float(np.dot(signs, np.abs(C)))

def entropy_from_supertrace(S: float, N: int, alpha: float = ALPHA) -> float:
    if S == 0.0:
        return 0.0
    p = abs(S) / N
    if p <= 0.0:
        return 0.0
    return -alpha * p * math.log(p)

def invariant_scalar(C: np.ndarray) -> float:
    S = supertrace_from_coeffs(C)
    return abs(S) * math.exp(-entropy_from_supertrace(S, C.size))

# ============================================================
#  Vectorized zeta and dzeta/dt
# ============================================================
def zeta(t: float, C: np.ndarray, alpha: float = ALPHA) -> float:
    """ζ(t) = Σ_i Re(C_i e^{i i t / α})   — O(K), vectorized."""
    K = (C.size - 1) // 2
    i = np.arange(C.size) - K
    phase = t * i / alpha
    return float(np.dot(C.real, np.cos(phase)) - np.dot(C.imag, np.sin(phase)))

def dzeta_dt(t: float, C: np.ndarray, alpha: float = ALPHA) -> float:
    """dζ/dt = −Σ_i (i/α)(Re C_i sin(θ) + Im C_i cos(θ))   — O(K)."""
    K = (C.size - 1) // 2
    i = np.arange(C.size) - K
    phase = t * i / alpha
    return float(-np.dot(i / alpha,
                         C.real * np.sin(phase) + C.imag * np.cos(phase)))

# ============================================================
#  Bond construction — O(K log K)
# ============================================================
def build_bonds(points: np.ndarray, k_neighbors: int = 4):
    """
    Return (edge_i, edge_j, weights) arrays.
    cKDTree query is O(K log K) and parallelised via workers=-1.
    """
    tree = cKDTree(points)
    # k_neighbors+1 because the first hit is the point itself
    dists, idxs = tree.query(points, k=k_neighbors + 1, workers=-1)
    dists, idxs = dists[:, 1:], idxs[:, 1:]          # drop self

    # keep only i < j edges (upper triangle), deduplicated
    K = points.shape[0]
    src = np.repeat(np.arange(K), k_neighbors)
    dst = idxs.reshape(-1)
    d   = dists.reshape(-1)
    keep = src < dst
    src, dst, d = src[keep], dst[keep], d[keep]

    # unique (i, j) pairs
    a = np.minimum(src, dst)
    b = np.maximum(src, dst)
    _, uniq = np.unique(np.stack([a, b], axis=1), axis=0, return_index=True)
    src, dst, d = src[uniq], dst[uniq], d[uniq]

    # analytic bond weight  w(d) = (1 − e^{−αd}) / I_full   — no per-call recompute
    w = (1.0 - np.exp(-ALPHA * d)) / I_FULL
    return src, dst, w

# ============================================================
#  Simulation
# ============================================================
def simulate(K=80, num_steps=50, dt=0.02, k_neighbors=4, seed=42,
             box=10.0):
    rng = np.random.default_rng(seed)

    # ---------- init ----------
    positions = rng.random((K, 3)) * box
    C = positions[:, 0] + 1j * positions[:, 1]     # complex, vectorized

    src, dst, weights = build_bonds(positions, k_neighbors)
    E_count = src.size                              # number of bonds

    # ---------- histories ----------
    pos_hist = np.empty((num_steps + 1, K, 3), dtype=np.float64)
    m_hist   = np.empty(num_steps + 1, dtype=np.float64)

    pos_hist[0] = positions
    m_hist[0]   = invariant_scalar(C)

    # ---------- time loop (O(K) per step) ----------
    for step in range(num_steps):
        t = step * dt
        dzet = dzeta_dt(t, C)

        centroid   = positions.mean(axis=0)
        directions = centroid - positions                    # (K, 3)
        norms      = np.linalg.norm(directions, axis=1, keepdims=True)
        directions /= (norms + 1e-12)

        positions += dt * dzet * directions * 0.1            # in-place
        C = positions[:, 0] + 1j * positions[:, 1]

        pos_hist[step + 1] = positions
        m_hist[step + 1]   = invariant_scalar(C)

    return pos_hist, m_hist, src, dst, weights, box, E_count

# ============================================================
#  Animation — one Line3DCollection, one scatter
# ============================================================
def animate_bonds(K=80, num_steps=50, dt=0.02, k_neighbors=4, seed=42):
    pos_hist, m_hist, src, dst, w, box, E_count = simulate(
        K=K, num_steps=num_steps, dt=dt,
        k_neighbors=k_neighbors, seed=seed,
    )

    fig = plt.figure(figsize=(13, 6))
    ax3d = fig.add_subplot(121, projection="3d")
    ax_s = fig.add_subplot(122)
    ax_s.set_xlabel("Frame"); ax_s.set_ylabel("Invariant scalar m")
    ax_s.set_title("m = |S| exp(−H)"); ax_s.grid(True)
    ax_s.set_xlim(0, num_steps)
    ax_s.set_ylim(m_hist.min() * 0.95, m_hist.max() * 1.05)
    line_s, = ax_s.plot([], [], "b-", lw=2)

    # ---- one scatter + one Line3DCollection ----
    colors = plt.cm.viridis(w)                          # (E, 4)
    scatter = ax3d.scatter(pos_hist[0][:, 0], pos_hist[0][:, 1],
                           pos_hist[0][:, 2], c="royalblue", s=24,
                           depthshade=True)
    segs0 = np.stack([
        pos_hist[0][src], pos_hist[0][dst]
    ], axis=1)                                          # (E, 2, 3)
    lc = Line3DCollection(segs0, colors=colors, linewidths=1.3, alpha=0.65)
    ax3d.add_collection3d(lc)

    title = ax3d.set_title("")
    ax3d.set_xlim(0, box); ax3d.set_ylim(0, box); ax3d.set_zlim(0, box)
    ax3d.set_xlabel("X"); ax3d.set_ylabel("Y"); ax3d.set_zlabel("Z")

    def update(frame):
        pos = pos_hist[frame]
        # update points (single artist)
        scatter._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
        # update all segments at once — O(E) memory write, no artist churn
        lc.set_segments(np.stack([pos[src], pos[dst]], axis=1))
        line_s.set_data(np.arange(frame + 1), m_hist[:frame + 1])
        title.set_text(f"Frame {frame}   K = {K}   E = {E_count}   "
                       f"m = {m_hist[frame]:.4f}")
        return scatter, lc, line_s, title

    ani = FuncAnimation(fig, update, frames=len(pos_hist),
                        interval=80, blit=False, repeat=True)
    plt.tight_layout()
    return ani, pos_hist, m_hist, w


# ============================================================
#  Main
# ============================================================
if __name__ == "__main__":
    import time

    K = 200
    num_steps = 60

    t0 = time.perf_counter()
    ani, pos_hist, m_hist, w = animate_bonds(
        K=K, num_steps=num_steps, dt=0.02, k_neighbors=4, seed=42,
    )
    t_build = (time.perf_counter() - t0) * 1e3

    print("=" * 70)
    print("Bond-fluid animation  ·  O(K log K) build  ·  O(K) per step")
    print("=" * 70)
    print(f"  K                     = {K}")
    print(f"  edges E               = {w.size}")
    print(f"  avg bond weight       = {w.mean():.4f}")
    print(f"  max bond weight       = {w.max():.4f}")
    print(f"  build + sim wall time = {t_build:.1f} ms")
    print(f"  m(0)                  = {m_hist[0]:.6f}")
    print(f"  m({num_steps})         = {m_hist[-1]:.6f}")
    print(f"  m range               = [{m_hist.min():.6f}, {m_hist.max():.6f}]")
    print()

    plt.show()
    # In Jupyter:
    # from IPython.display import HTML
    # HTML(ani.to_html5_video())