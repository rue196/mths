#!/usr/bin/env python3
"""
shortest_vector_graph.py

Shortest Vector Problem (SVP) on a graph lattice.

Instead of lattice vectors  v = Σ b_i · e_i  with a basis {b_i},
we take a graph  G = (V, E)  where each edge (i, j) has a weight

        w(i, j) = ‖ p_i − p_j ‖₂  /  I

and  I = ∫₀^{π+e} exp(−α x) dx,  α = 1/(π − e).

The lattice is the graph itself.  Its "vectors" are edges (or paths).
The SVP asks for the shortest non‑trivial vector, i.e. the shortest
edge (rank‑1 case) or the shortest closed walk (rank‑N case).

We report:

    1.  min edge weight             (rank‑1 SVP)
    2.  shortest cycle              (closed lattice vector)
    3.  shortest path between any two vertices (all‑pairs SVP)
    4.  the reduced basis           (the k shortest edges, k = dim)

All operations are O(V + E) for the edge scan and O(V² log V) for
all‑pairs Dijkstra in the dense graph.
"""

import math
import heapq
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D                       # noqa: F401
from scipy.integrate import quad


# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------
PI    = math.pi
E     = math.e
ALPHA = 1.0 / (PI - E)                 # ≈ 2.362


# ------------------------------------------------------------
# 1.  Point cloud in 3D
# ------------------------------------------------------------
def generate_points(N: int, seed: int = 42, spread: float = 1.0
                    ) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((N, 3)) * spread


# ------------------------------------------------------------
# 2.  Graph construction
# ------------------------------------------------------------
def build_cycle(points: np.ndarray):
    """Eulerian cycle graph — every vertex has even degree 2."""
    N = points.shape[0]
    edges = [(i, (i + 1) % N) for i in range(N)]
    return edges


def build_knn(points: np.ndarray, k: int = 3):
    """
    k‑nearest‑neighbour graph — a richer lattice that has more than
    one cycle and admits genuinely non‑trivial shortest vectors.
    """
    N = points.shape[0]
    edges = set()
    for i in range(N):
        d = np.linalg.norm(points - points[i], axis=1)
        order = np.argsort(d)
        for j in order[1:k + 1]:
            a, b = (i, int(j)) if i < j else (int(j), i)
            edges.add((a, b))
    return sorted(edges)


# ------------------------------------------------------------
# 3.  Edge weights = Euclidean distance / I
# ------------------------------------------------------------
def integral_g(alpha: float = ALPHA) -> float:
    I, _ = quad(lambda x: math.exp(-alpha * x), 0.0, PI + E)
    return I


def compute_distances(points: np.ndarray, edges):
    return np.array([np.linalg.norm(points[i] - points[j]) for i, j in edges])


# ------------------------------------------------------------
# 4.  SVP — shortest edge  (rank‑1)
# ------------------------------------------------------------
def svp_shortest_edge(edges, scaled_dists):
    idx = int(np.argmin(scaled_dists))
    return edges[idx], float(scaled_dists[idx]), idx


# ------------------------------------------------------------
# 5.  Shortest cycle  (closed lattice vector)
# ------------------------------------------------------------
def shortest_cycle(points, edges, scaled_dists):
    """
    For each edge (u, v), remove it and find the shortest u→v path
    in the remaining graph.  The minimum over all edges is the
    shortest cycle length (length = weight of edge + path).
    """
    # build adjacency
    N = points.shape[0]
    adj = [[] for _ in range(N)]
    for idx, (i, j) in enumerate(edges):
        w = scaled_dists[idx]
        adj[i].append((j, w, idx))
        adj[j].append((i, w, idx))

    best_cycle = (None, float("inf"), None)
    for skip_idx, (u, v) in enumerate(edges):
        w_edge = scaled_dists[skip_idx]
        # Dijkstra u → v without using edge skip_idx
        dist = {u: 0.0}
        pq = [(0.0, u, [u])]
        seen = set()
        while pq:
            d, node, path = heapq.heappop(pq)
            if node in seen:
                continue
            seen.add(node)
            if node == v:
                total = w_edge + d
                if total < best_cycle[1]:
                    best_cycle = (path + [u], total, skip_idx)
                break
            for (nb, w, eidx) in adj[node]:
                if eidx == skip_idx:
                    continue
                nd = d + w
                if nb not in dist or nd < dist[nb]:
                    dist[nb] = nd
                    heapq.heappush(pq, (nd, nb, path + [nb]))
    return best_cycle


# ------------------------------------------------------------
# 6.  All‑pairs shortest path  (SVP in the path metric)
# ------------------------------------------------------------
def all_pairs_dijkstra(points, edges, scaled_dists):
    N = points.shape[0]
    adj = [[] for _ in range(N)]
    for idx, (i, j) in enumerate(edges):
        w = scaled_dists[idx]
        adj[i].append((j, w))
        adj[j].append((i, w))

    def dijkstra(s):
        dist = [float("inf")] * N
        dist[s] = 0.0
        pq = [(0.0, s)]
        while pq:
            d, node = heapq.heappop(pq)
            if d > dist[node]:
                continue
            for (nb, w) in adj[node]:
                nd = d + w
                if nd < dist[nb]:
                    dist[nb] = nd
                    heapq.heappush(pq, (nd, nb))
        return dist

    D = np.array([dijkstra(s) for s in range(N)])
    return D


# ------------------------------------------------------------
# 7.  Lattice basis reduction  (k shortest edges)
# ------------------------------------------------------------
def reduced_basis(edges, scaled_dists, k: int):
    order = np.argsort(scaled_dists)[:k]
    return [(edges[i], float(scaled_dists[i])) for i in order]


# ------------------------------------------------------------
#  Main
# ------------------------------------------------------------
def main():
    N = 60
    points = generate_points(N, seed=42, spread=1.0)
    edges = build_knn(points, k=3)                 # richer lattice

    raw = compute_distances(points, edges)
    I   = integral_g()
    scaled_dists = raw / I

    print("=" * 72)
    print("Shortest Vector Problem on a graph lattice")
    print("=" * 72)
    print(f"  α                = 1/(π − e) ≈ {ALPHA:.6f}")
    print(f"  I                = ∫₀^(π+e) exp(−αx) dx = {I:.6f}")
    print(f"  N vertices       = {N}")
    print(f"  |E| edges        = {len(edges)}")
    print(f"  raw   mean/std   = {raw.mean():.4f}  /  {raw.std():.4f}")
    print(f"  scaled mean/std  = {scaled_dists.mean():.4f}  /  "
          f"{scaled_dists.std():.4f}")

    # ---------- rank‑1 SVP ----------
    edge_svp, w_svp, idx_svp = svp_shortest_edge(edges, scaled_dists)
    print()
    print("--- rank‑1 SVP  (shortest edge) ---")
    print(f"  shortest edge    = {edge_svp}")
    print(f"  scaled weight    = {w_svp:.6f}")
    print(f"  raw distance     = {raw[idx_svp]:.6f}")
    print(f"  |v| · I          = {w_svp * I:.6f}   (= raw distance)")

    # ---------- shortest cycle ----------
    cyc_path, cyc_len, cyc_edge = shortest_cycle(points, edges, scaled_dists)
    print()
    print("--- shortest cycle  (closed lattice vector) ---")
    if cyc_path is not None:
        print(f"  cycle length     = {cyc_len:.6f}")
        print(f"  path             = {cyc_path}")
    else:
        print("  no cycle found (graph is a tree)")

    # ---------- all‑pairs Dijkstra ----------
    D = all_pairs_dijkstra(points, edges, scaled_dists)
    np.fill_diagonal(D, np.inf)
    i_pair, j_pair = np.unravel_index(np.argmin(D), D.shape)
    d_pair = D[i_pair, j_pair]
    print()
    print("--- all‑pairs SVP  (shortest non‑zero path) ---")
    print(f"  vertices         = {i_pair} → {j_pair}")
    print(f"  path length      = {d_pair:.6f}")
    print(f"  (equals the rank‑1 edge if {i_pair},{j_pair} are adjacent)")

    # ---------- reduced basis ----------
    k = 4
    basis = reduced_basis(edges, scaled_dists, k)
    print()
    print(f"--- reduced basis  (k = {k} shortest edges) ---")
    for i, (e, w) in enumerate(basis, 1):
        print(f"  b{i}: {e}   w = {w:.6f}")

    # ============================================================
    #  Visualisation
    # ============================================================
    fig = plt.figure(figsize=(16, 6))

    # ---------- (a) 3D lattice graph ----------
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c="#3a7bd5", s=30, depthshade=False)

    for idx, (i, j) in enumerate(edges):
        p1, p2 = points[i], points[j]
        w = scaled_dists[idx]
        norm_w = (w - scaled_dists.min()) / (
            scaled_dists.max() - scaled_dists.min() + 1e-12)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color=plt.cm.viridis(1.0 - norm_w),
                alpha=0.6, linewidth=1.2)

    # highlight shortest vector
    i, j = edge_svp
    p1, p2 = points[i], points[j]
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
            color="#e74c3c", linewidth=3.5,
            label=f"rank‑1 SVP  |v| = {w_svp:.4f}")
    ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
               color="#e74c3c", s=90, edgecolor="k", zorder=5)

    # highlight shortest cycle
    if cyc_path is not None:
        cp = cyc_path
        for a, b in zip(cp[:-1], cp[1:]):
            p1, p2 = points[a], points[b]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    color="#f1c40f", linewidth=2.2, alpha=0.9)
        ax.scatter(points[cp[:-1], 0], points[cp[:-1], 1],
                   points[cp[:-1], 2],
                   color="#f1c40f", s=70, edgecolor="k",
                   zorder=6, label=f"shortest cycle  L = {cyc_len:.4f}")

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("Graph lattice  ·  k‑NN")
    ax.legend(fontsize=8, loc="upper left")

    # ---------- (b) distance distribution ----------
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.hist(scaled_dists, bins=30, color="#3a7bd5",
             edgecolor="white", alpha=0.85,
             label="scaled edge weights")
    ax2.axvline(w_svp, color="#e74c3c", lw=2.2,
                label=f"rank‑1 SVP = {w_svp:.4f}")
    if cyc_path is not None:
        ax2.axvline(cyc_len, color="#f1c40f", lw=2.2,
                    label=f"shortest cycle = {cyc_len:.4f}")
    ax2.set_xlabel("weight")
    ax2.set_ylabel("count")
    ax2.set_title("Edge weight distribution  ·  scaled by 1/I")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.suptitle(
        f"SVP on a graph lattice  ·  α = {ALPHA:.4f}  ·  I = {I:.4f}",
        fontsize=13,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # ---------- integral curve (unchanged from non-lineargraph.py) ----------
    x_vals = np.linspace(0, PI + E, 200)
    g_vals = np.exp(-ALPHA * x_vals)
    plt.figure(figsize=(7, 4))
    plt.plot(x_vals, g_vals, color="#8e44ad", lw=1.6)
    plt.fill_between(x_vals, 0, g_vals, color="#8e44ad", alpha=0.25)
    plt.axvline(PI + E, color="k", ls="--", lw=0.8)
    plt.xlabel("x")
    plt.ylabel("g(x) = exp(−αx)")
    plt.title(f"Lattice normalisation  ·  I = ∫₀^(π+e) exp(−αx) dx = {I:.6f}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()