#!/usr/bin/env python3
"""
elliptic_lattice_cvp.py
=======================

Closest vector problem on the Figure 3.5 elliptic lattice.

The lattice is a discrete sampling of the elliptic projection
    Π(t, r) = ½(1 + cos(2π t/π) · cos(2π r/e))  ∈ [0, 1]

at K_t × K_r points over [0, π) × [0, e).  Every lattice point
(i, j) carries a scalar value Π_ij and a finite-step gradient

    D_a Π = [Π(p + a·u) − Π(p)] / a

where  a = A_STEP × mean(dt, dr)  and  u = unit direction.

Given two lattice points p, q, the **closest vector** is the
shortest path where each edge is weighted by

    w(p → q) = ||q − p||_E · (1 + A_STEP · |D_a Π along p → q|)

So the path prefers flat regions of Π (valleys) and resists crossing
steep crests.  SVP = shortest such edge on the whole lattice;
CVP = closest lattice point to a continuous query.

Imports (both modules unchanged)
--------------------------------
    from log_rank        import elliptic_projection, PI, E
    from alu_finite_step import ALPHA, ALPHA_ASYM, A_STEP
"""

from __future__ import annotations
import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ---- log_rank.py  ·  Figure 3.5 scalar ----
from log_rank import (
    elliptic_projection,
    elliptic_projection_vec,
    PI,
    E,
)

# ---- alu_finite_step.py  ·  finite-step constants ----
from alu_finite_step import (
    ALPHA,
    ALPHA_ASYM,
    A_STEP,
)


# ============================================================
#  Result containers
# ============================================================
@dataclass
class SVPResult:
    edge: Tuple[Tuple[int, int], Tuple[int, int]]
    weight: float
    d_E: float
    d_Pi: float


@dataclass
class ClosestVectorResult:
    p_idx: Tuple[int, int]
    q_idx: Tuple[int, int]
    path: List[Tuple[int, int]]
    total_weight: float
    d_E_direct: float
    path_sum_E: float
    path_length: int
    stretch: float
    weight_ratio: float


@dataclass
class CVPResult:
    target: Tuple[float, float]
    lattice_point: Tuple[int, int]
    d_E: float
    d_elliptic: float
    Pi_target: float
    Pi_lattice: float


# ============================================================
#  Elliptic lattice
# ============================================================
class EllipticLattice:
    """
    Discrete lattice from Figure 3.5 with finite-step Euclidean metric.

    Lattice spacing is dt = π/K_t, dr = e/K_r.
    Finite step for derivatives is a = A_STEP × mean(dt, dr).
    """

    def __init__(self,
                 K_t: int = 16,
                 K_r: int = 16,
                 periodic: bool = True):
        self.K_t = int(K_t)
        self.K_r = int(K_r)
        self.periodic = bool(periodic)

        self.dt = PI / self.K_t
        self.dr = E / self.K_r

        self.t_axis = np.arange(self.K_t) * self.dt
        self.r_axis = np.arange(self.K_r) * self.dr

        # Scalar Π on the lattice  ·  Figure 3.5
        T, R = np.meshgrid(self.t_axis, self.r_axis, indexing='ij')
        self.Pi = 0.5 * (1.0 + np.cos(2.0 * math.pi * T / PI) *
                               np.cos(2.0 * math.pi * R / E))

        # Finite-step gradient (used for visualization)
        self._compute_finite_step_gradient()

    # --------------------------------------------------------
    def _compute_finite_step_gradient(self):
        h_mean = 0.5 * (self.dt + self.dr)
        a = A_STEP * h_mean

        shift_t = max(1, min(int(round(a / self.dt)), self.K_t // 2))
        shift_r = max(1, min(int(round(a / self.dr)), self.K_r // 2))

        if self.periodic:
            f_plus_t  = np.roll(self.Pi, -shift_t, axis=0)
            f_minus_t = np.roll(self.Pi,  shift_t, axis=0)
            f_plus_r  = np.roll(self.Pi, -shift_r, axis=1)
            f_minus_r = np.roll(self.Pi,  shift_r, axis=1)
        else:
            f_plus_t  = np.pad(self.Pi, ((shift_t, 0), (0, 0)),
                               mode='edge')[:self.K_t]
            f_minus_t = np.pad(self.Pi, ((0, shift_t), (0, 0)),
                               mode='edge')[-self.K_t:]
            f_plus_r  = np.pad(self.Pi, ((0, 0), (shift_r, 0)),
                               mode='edge')[:, :self.K_r]
            f_minus_r = np.pad(self.Pi, ((0, 0), (0, shift_r)),
                               mode='edge')[:, -self.K_r:]

        self.dPi_dt = (f_plus_t - f_minus_t) / (2.0 * shift_t * self.dt)
        self.dPi_dr = (f_plus_r - f_minus_r) / (2.0 * shift_r * self.dr)
        self.grad_mag = np.sqrt(self.dPi_dt ** 2 + self.dPi_dr ** 2)

    # --------------------------------------------------------
    def Pi_continuous(self, t: float, r: float) -> float:
        return 0.5 * (1.0 + math.cos(2.0 * math.pi * t / PI) *
                            math.cos(2.0 * math.pi * r / E))

    def point_coords(self, idx: Tuple[int, int]) -> Tuple[float, float]:
        i, j = idx
        return self.t_axis[i], self.r_axis[j]

    def nearest_lattice(self, t: float, r: float) -> Tuple[int, int]:
        if self.periodic:
            t = t % PI
            r = r % E
        i = int(round(t / self.dt)) % self.K_t
        j = int(round(r / self.dr)) % self.K_r
        return (i, j)

    # --------------------------------------------------------
    def neighbors(self, idx: Tuple[int, int]) -> List[Tuple[int, int]]:
        i, j = idx
        out = []
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni = i + di
            nj = j + dj
            if self.periodic:
                ni %= self.K_t
                nj %= self.K_r
                if (ni, nj) != (i, j):
                    out.append((ni, nj))
            else:
                if 0 <= ni < self.K_t and 0 <= nj < self.K_r:
                    out.append((ni, nj))
        return out

    # --------------------------------------------------------
    def finite_step_derivative_along(self,
                                     p_idx: Tuple[int, int],
                                     q_idx: Tuple[int, int]
                                     ) -> float:
        """D_a Π along p → q, step a = A_STEP × mean(dt, dr)."""
        t1, r1 = self.point_coords(p_idx)
        t2, r2 = self.point_coords(q_idx)

        # direction with torus unwrap
        dt_dir = t2 - t1
        dr_dir = r2 - r1
        if self.periodic:
            if dt_dir > PI / 2:  dt_dir -= PI
            elif dt_dir < -PI / 2: dt_dir += PI
            if dr_dir > E / 2:  dr_dir -= E
            elif dr_dir < -E / 2: dr_dir += E

        norm = math.hypot(dt_dir, dr_dir)
        if norm < 1e-12:
            return 0.0
        u_t = dt_dir / norm
        u_r = dr_dir / norm

        h_mean = 0.5 * (self.dt + self.dr)
        a = A_STEP * h_mean

        t_step = (t1 + a * u_t) % PI
        r_step = (r1 + a * u_r) % E

        Pi_p    = self.Pi_continuous(t1, r1)
        Pi_step = self.Pi_continuous(t_step, r_step)

        return (Pi_step - Pi_p) / a

    # --------------------------------------------------------
    def edge_weight(self,
                    p_idx: Tuple[int, int],
                    q_idx: Tuple[int, int]
                    ) -> float:
        """w = ||q − p||_E · (1 + A_STEP · |D_a Π|)."""
        t1, r1 = self.point_coords(p_idx)
        t2, r2 = self.point_coords(q_idx)
        d_E = math.hypot(t2 - t1, r2 - r1)
        d_Pi = self.finite_step_derivative_along(p_idx, q_idx)
        return d_E * (1.0 + A_STEP * abs(d_Pi))

    def direct_euclidean(self, p_idx, q_idx) -> float:
        t1, r1 = self.point_coords(p_idx)
        t2, r2 = self.point_coords(q_idx)
        return math.hypot(t2 - t1, r2 - r1)

    # --------------------------------------------------------
    def shortest_vector(self) -> SVPResult:
        """SVP: shortest weighted edge on the lattice."""
        best_w = float('inf')
        best = None
        for i in range(self.K_t):
            for j in range(self.K_r):
                for nb in self.neighbors((i, j)):
                    if nb <= (i, j):
                        continue
                    w = self.edge_weight((i, j), nb)
                    if w < best_w:
                        best_w = w
                        d_E = self.direct_euclidean((i, j), nb)
                        d_Pi = self.finite_step_derivative_along(
                            (i, j), nb)
                        best = SVPResult(
                            edge=((i, j), nb),
                            weight=w, d_E=d_E, d_Pi=d_Pi,
                        )
        return best

    # --------------------------------------------------------
    def closest_vector(self,
                       p_idx: Tuple[int, int],
                       q_idx: Tuple[int, int]
                       ) -> ClosestVectorResult:
        """Dijkstra shortest path p → q under elliptic edge weights."""
        dist = {p_idx: 0.0}
        prev: Dict[Tuple[int, int], Tuple[int, int]] = {}
        pq = [(0.0, p_idx)]
        done = set()

        while pq:
            d, cur = heapq.heappop(pq)
            if cur in done:
                continue
            done.add(cur)
            if cur == q_idx:
                break
            for nb in self.neighbors(cur):
                w = self.edge_weight(cur, nb)
                nd = d + w
                if nd < dist.get(nb, float('inf')):
                    dist[nb] = nd
                    prev[nb] = cur
                    heapq.heappush(pq, (nd, nb))

        if q_idx not in dist:
            raise RuntimeError(f"no path {p_idx} → {q_idx}")

        path = []
        cur = q_idx
        while cur != p_idx:
            path.append(cur)
            cur = prev[cur]
        path.append(p_idx)
        path.reverse()

        d_E = self.direct_euclidean(p_idx, q_idx)
        path_E = sum(self.direct_euclidean(path[k], path[k+1])
                     for k in range(len(path)-1))

        return ClosestVectorResult(
            p_idx=p_idx, q_idx=q_idx,
            path=path,
            total_weight=dist[q_idx],
            d_E_direct=d_E,
            path_sum_E=path_E,
            path_length=len(path) - 1,
            stretch=path_E / d_E if d_E > 1e-12 else 1.0,
            weight_ratio=dist[q_idx] / d_E if d_E > 1e-12 else 1.0,
        )

    # --------------------------------------------------------
    def closest_to_target(self, t_star: float, r_star: float) -> CVPResult:
        """CVP: nearest lattice point to a continuous query, weighted."""
        t_q = t_star % PI if self.periodic else t_star
        r_q = r_star % E if self.periodic else r_star
        seed = self.nearest_lattice(t_star, r_star)

        best = None
        best_score = float('inf')
        h_mean = 0.5 * (self.dt + self.dr)
        a = A_STEP * h_mean

        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ni = seed[0] + di
                nj = seed[1] + dj
                if self.periodic:
                    ni %= self.K_t
                    nj %= self.K_r
                elif not (0 <= ni < self.K_t and 0 <= nj < self.K_r):
                    continue

                t_c, r_c = self.point_coords((ni, nj))

                # Euclidean with torus wrap
                dtt = t_c - t_q
                drr = r_c - r_q
                if self.periodic:
                    if dtt > PI/2:  dtt -= PI
                    elif dtt < -PI/2: dtt += PI
                    if drr > E/2:   drr -= E
                    elif drr < -E/2: drr += E
                d_E = math.hypot(dtt, drr)

                if d_E < 1e-12:
                    d_Pi = 0.0
                else:
                    u_t = dtt / d_E
                    u_r = drr / d_E
                    t_s = (t_q + a * u_t) % PI
                    r_s = (r_q + a * u_r) % E
                    d_Pi = (self.Pi_continuous(t_s, r_s) -
                            self.Pi_continuous(t_q, r_q)) / a

                score = d_E * (1.0 + A_STEP * abs(d_Pi))
                if score < best_score:
                    best_score = score
                    best = (ni, nj, d_E)

        ni, nj, d_E = best
        Pi_lat = self.Pi_continuous(*self.point_coords((ni, nj)))
        Pi_tgt = self.Pi_continuous(t_q, r_q)

        return CVPResult(
            target=(t_star, r_star),
            lattice_point=(ni, nj),
            d_E=d_E, d_elliptic=best_score,
            Pi_target=Pi_tgt, Pi_lattice=Pi_lat,
        )


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("Closest vector problem  ·  Figure 3.5 elliptic lattice")
    print("=" * 78)
    print(f"  α         = 1/(π − e)   = {ALPHA:.6f}")
    print(f"  α_asym    = 0.3628      = {ALPHA_ASYM:.6f}")
    print(f"  A_STEP    = α / α_asym  = {A_STEP:.6f}")
    print(f"  π (t-period)            = {PI:.6f}")
    print(f"  e (r-period)            = {E:.6f}")
    print()

    K_t = K_r = 16
    lat = EllipticLattice(K_t=K_t, K_r=K_r, periodic=True)
    n_pts = lat.K_t * lat.K_r
    n_edges = sum(len(lat.neighbors((i, j)))
                  for i in range(lat.K_t)
                  for j in range(lat.K_r)) // 2
    a_finite = A_STEP * 0.5 * (lat.dt + lat.dr)

    print(f"--- lattice ---")
    print(f"  K_t × K_r           = {K_t} × {K_r} = {n_pts} points")
    print(f"  dt                  = {lat.dt:.6f}")
    print(f"  dr                  = {lat.dr:.6f}")
    print(f"  finite step a       = {a_finite:.6f}  "
          f"= A_STEP × mean(dt, dr)")
    print(f"  undirected edges    = {n_edges}")
    print(f"  Π range             = "
          f"[{lat.Pi.min():.4f}, {lat.Pi.max():.4f}]")
    print(f"  |D_a Π| range       = "
          f"[{lat.grad_mag.min():.4f}, {lat.grad_mag.max():.4f}]")
    print()

    # ============================================================
    #  SVP
    # ============================================================
    print("=" * 78)
    print("1  ·  SVP  ·  shortest weighted edge")
    print("=" * 78)
    svp = lat.shortest_vector()
    print(f"  edge              : {svp.edge[0]} → {svp.edge[1]}")
    print(f"  elliptic weight   : {svp.weight:.6f}")
    print(f"  d_E (Euclidean)   : {svp.d_E:.6f}")
    print(f"  D_a Π along edge  : {svp.d_Pi:+.6f}")
    print(f"  A_STEP · |D_a Π|  : {A_STEP * abs(svp.d_Pi):.6f}")
    print()

    # ============================================================
    #  Closest vector between two points
    # ============================================================
    print("=" * 78)
    print("2  ·  closest vector between lattice points  ·  Dijkstra")
    print("=" * 78)

    pairs = [
        ((0, 0), (K_t // 2, K_r // 2)),
        ((0, 0), (K_t - 1, 0)),
        ((0, 0), (0, K_r - 1)),
        ((0, 0), (K_t // 2, K_r - 1)),
    ]

    print(f"  {'p':>10s}  {'q':>10s}  {'Euclid':>8s}  "
          f"{'path_E':>8s}  {'weight':>8s}  "
          f"{'stretch':>8s}  {'w/d_E':>8s}  {'steps':>5s}")
    print("  " + "-" * 78)

    results = []
    for p, q in pairs:
        cv = lat.closest_vector(p, q)
        results.append(cv)
        print(f"  {str(p):>10s}  {str(q):>10s}  "
              f"{cv.d_E_direct:>8.4f}  {cv.path_sum_E:>8.4f}  "
              f"{cv.total_weight:>8.4f}  {cv.stretch:>8.4f}  "
              f"{cv.weight_ratio:>8.4f}  {cv.path_length:>5d}")
    print()

    diag = results[0]
    print(f"  Diagonal path  (0,0) → ({K_t//2},{K_r//2}):")
    print(f"    path  : {diag.path}")
    Pi_on_path = [round(lat.Pi[i, j], 3) for (i, j) in diag.path]
    print(f"    Π     : {Pi_on_path}")
    print(f"    Σ d_E : {diag.path_sum_E:.4f}  "
          f"(direct {diag.d_E_direct:.4f}, stretch {diag.stretch:.3f})")
    print()

    # ============================================================
    #  CVP
    # ============================================================
    print("=" * 78)
    print("3  ·  CVP  ·  closest lattice point to a continuous query")
    print("=" * 78)

    queries = [
        (0.50, 0.50),
        (PI / 4, E / 4),
        (PI / 2, E / 2),
        (2.50, 1.30),
    ]

    print(f"  {'query (t*, r*)':>22s}  {'lattice (i,j)':>15s}  "
          f"{'d_E':>8s}  {'d_ell':>8s}  "
          f"{'Π(t*,r*)':>10s}  {'Π(lattice)':>12s}")
    print("  " + "-" * 88)
    for ts, rs in queries:
        cv = lat.closest_to_target(ts, rs)
        print(f"  ({ts:7.3f}, {rs:7.3f})        "
              f"{str(cv.lattice_point):>15s}  "
              f"{cv.d_E:>8.4f}  {cv.d_elliptic:>8.4f}  "
              f"{cv.Pi_target:>10.4f}  {cv.Pi_lattice:>12.4f}")
    print()

    # ============================================================
    #  Visualization
    # ============================================================
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.36, wspace=0.30)
    extent = [0, PI, 0, E]

    # (a) Π + lattice points
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(lat.Pi.T, origin='lower', extent=extent,
                   cmap='viridis', aspect='auto', vmin=0, vmax=1)
    for i in range(lat.K_t):
        for j in range(lat.K_r):
            ax.scatter(lat.t_axis[i], lat.r_axis[j],
                       s=10, c='white', edgecolors='black',
                       linewidths=0.3, zorder=4)
    ax.set_xlabel("t  (mod π)"); ax.set_ylabel("r  (mod e)")
    ax.set_title(f"Π(t, r)  ·  {n_pts} lattice points")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (b) finite-step gradient magnitude
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(lat.grad_mag.T, origin='lower', extent=extent,
                   cmap='inferno', aspect='auto')
    ax.set_xlabel("t  (mod π)"); ax.set_ylabel("r  (mod e)")
    ax.set_title(f"|D_a Π|   (a = {a_finite:.3f})")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (c) SVP highlighted
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(lat.Pi.T, origin='lower', extent=extent,
              cmap='viridis', aspect='auto', vmin=0, vmax=1, alpha=0.5)
    p, q = svp.edge
    pt, pr = lat.point_coords(p)
    qt, qr = lat.point_coords(q)
    ax.plot([pt, qt], [pr, qr], '-o',
            color='#e74c3c', lw=3.5, ms=14,
            markeredgecolor='black', markeredgewidth=1.5,
            label=f'SVP  w = {svp.weight:.4f}')
    ax.set_xlabel("t  (mod π)"); ax.set_ylabel("r  (mod e)")
    ax.set_title("SVP  ·  shortest weighted edge")
    ax.legend(loc='upper right', fontsize=9)

    # (d) closest vector path
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(lat.Pi.T, origin='lower', extent=extent,
              cmap='viridis', aspect='auto', vmin=0, vmax=1, alpha=0.6)
    path_t = [lat.t_axis[i] for i, j in diag.path]
    path_r = [lat.r_axis[j] for i, j in diag.path]
    ax.plot(path_t, path_r, '-o', color='#e74c3c', lw=2.2, ms=5,
            markeredgecolor='black', markeredgewidth=0.7,
            label=f'weighted  w = {diag.total_weight:.3f}')
    pt, pr = lat.point_coords(diag.p_idx)
    qt, qr = lat.point_coords(diag.q_idx)
    ax.plot([pt, qt], [pr, qr], '--', color='white', lw=2.2, alpha=0.9,
            label=f'Euclidean  d = {diag.d_E_direct:.3f}')
    ax.scatter([pt, qt], [pr, qr], s=140, c='yellow',
               edgecolors='black', linewidths=1.5, zorder=5)
    ax.set_xlabel("t  (mod π)"); ax.set_ylabel("r  (mod e)")
    ax.set_title(f"closest vector  ·  (0,0) → ({K_t//2},{K_r//2})")
    ax.legend(loc='lower right', fontsize=8)

    # (e) stretch factor comparison
    ax = fig.add_subplot(gs[1, 1])
    labels = [f"({p[0]},{p[1]})→({q[0]},{q[1]})" for p, q in pairs]
    stretch = [r.stretch for r in results]
    ratio  = [r.weight_ratio for r in results]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, stretch, 0.4, color='#3498db',
           edgecolor='black', label='path_E / d_E_direct')
    ax.bar(x + 0.2, ratio, 0.4, color='#e74c3c',
           edgecolor='black', label='total_weight / d_E_direct')
    ax.axhline(1.0, color='black', lw=0.5, ls=':')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=7)
    ax.set_ylabel("ratio")
    ax.set_title("path vs direct  ·  stretch factor")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

    # (f) CVP queries
    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(lat.Pi.T, origin='lower', extent=extent,
              cmap='viridis', aspect='auto', vmin=0, vmax=1, alpha=0.55)
    for idx_q, (ts, rs) in enumerate(queries):
        cvq = lat.closest_to_target(ts, rs)
        li, lj = cvq.lattice_point
        lt, lr = lat.point_coords((li, lj))
        ax.scatter(ts, rs, s=140, marker='x', c='red', zorder=6,
                   label='query' if idx_q == 0 else None)
        ax.scatter(lt, lr, s=110, marker='o', c='yellow',
                   edgecolors='black', linewidths=1.2, zorder=5)
        ax.annotate('', xy=(lt, lr), xytext=(ts, rs),
                    arrowprops=dict(arrowstyle='->', color='white',
                                    lw=1.4, alpha=0.85))
    ax.set_xlabel("t  (mod π)"); ax.set_ylabel("r  (mod e)")
    ax.set_title("CVP  ·  nearest lattice point to query")
    ax.legend(loc='lower right', fontsize=9)

    plt.suptitle(
        f"Elliptic lattice CVP  ·  α = {ALPHA:.4f}  ·  "
        f"A_STEP = {A_STEP:.4f}  ·  "
        f"w(p→q) = ||q−p||_E · (1 + A_STEP·|D_a Π|)",
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # ============================================================
    #  Complexity
    # ============================================================
    print("=" * 78)
    print("4  ·  complexity")
    print("=" * 78)
    print(f"  lattice build              O(K_t · K_r)")
    print(f"  finite-step gradient       O(K_t · K_r)")
    print(f"  edge weight (single)       O(1)")
    print(f"  SVP (all edges)            O(K_t · K_r · d)")
    print(f"  closest_vector (Dijkstra)  O(K_t · K_r · log(K_t · K_r))")
    print(f"  closest_to_target (CVP)    O(1)   3×3 neighbourhood")
    print()
    print(f"  d = 4 (4-connected lattice)")
    print()
    print("Done.")


if __name__ == "__main__":
    demo()