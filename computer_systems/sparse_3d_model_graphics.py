#!/usr/bin/env python3
"""
sparse_neg_grid_3d.py
=====================
Sparse 3D model on the negative-index (real) grid.

Layout
------
  x, y   ←  negative integer indices −n  with  μ(n) ≠ 0     O(n) grid
  z      ←  A(4) + ΔA(4) · Re(tr M_n)                        finite step
  hue    ←  α · Im(tr M_n)                                   O(K log K)

Memory saving:  square-free density 6/π² ≈ 0.6079
    sparse slots per axis = M = #{n ≤ K : μ(n) ≠ 0}
    dense  grid  : K²  points
    sparse grid  : M²  points    →  ~37 % fewer per axis

Correspondence (MobiusLinAlg64 layout)
    sym  = data[K + n]  ↔  positive half   (real plane, z)
    asym = data[K − n]  ↔  negative half   (imag plane, hue)

But we use the NEGATIVE half (asym slots) as the grid.
"""

from __future__ import annotations
import math
import time
from typing import List, Tuple

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
PI        = math.pi
E         = math.e
ALPHA     = 1.0 / (PI - E)                  # ≈ 2.362338
GAMMA     = 0.5772156649015329
K_DEFAULT = 512
N_FIXED   = 4


# ============================================================
#  Finite-step derivative  A(k) = 10·a·sin(k²°)
# ============================================================
def A_of_k(k: int) -> float:
    return 10.0 * ALPHA * math.sin(math.radians(k * k))

def dA_forward(k: int) -> float:
    return A_of_k(k + 1) - A_of_k(k)

def dA_central(k: int) -> float:
    return 0.5 * (A_of_k(k + 1) - A_of_k(k - 1))

A_4    = A_of_k(N_FIXED)
DA_4   = dA_forward(N_FIXED)
DC_A_4 = dA_central(N_FIXED)     # ≈ π


# ============================================================
#  Möbius sieve  ·  O(K) linear
# ============================================================
def mobius_sieve(K: int) -> np.ndarray:
    mu = np.zeros(K + 1, dtype=np.int8)
    if K >= 1:
        mu[1] = 1
    primes, is_comp = [], np.zeros(K + 1, dtype=bool)
    for i in range(2, K + 1):
        if not is_comp[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K:
                break
            is_comp[i * p] = True
            if i % p == 0:
                mu[i * p] = 0
                break
            mu[i * p] = -mu[i]
    return mu


# ============================================================
#  Square-free index list  ·  minimal-memory grid
# ============================================================
def square_free_indices(K: int) -> np.ndarray:
    """All n ∈ [1, K] with μ(n) ≠ 0, ascending."""
    mu = mobius_sieve(K)
    return np.where(mu[1:] != 0)[0] + 1


# ============================================================
#  M-matrix trace parts
# ============================================================
def m_trace(n: int) -> complex:
    """
    tr M_n = n^(i−1) = n^{-1} (cos ln n + i sin ln n)
    real part:  cos(ln n) / n
    imag part:  sin(ln n) / n
    """
    ln_n = math.log(n)
    return complex(math.cos(ln_n) / n, math.sin(ln_n) / n)


def m_trace_batch(ns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized real/imag parts for an array of indices."""
    ln_n = np.log(ns.astype(np.float64))
    inv  = 1.0 / ns
    re   = np.cos(ln_n) * inv
    im   = np.sin(ln_n) * inv
    return re, im


# ============================================================
#  Sparse 3D grid  ·  negative-index square-free
# ============================================================
class SparseNegGrid3D:
    """
    Vertices of a 3D surface whose (x, y) live on a square-free
    negative-index lattice and whose z (real) and hue (imag)
    come from the M-matrix trace.

    Attributes
    ----------
    ns       : (M,)  square-free indices used per axis
    X, Y     : (M, M) grid of negative index coordinates
    Z        : (M, M) algebraic (real) part, finite-step scaled
    hue      : (M, M) imaginary (asymmetric) Möbius part
    mu_mass  : float  total |μ| mass (informational)
    """
    def __init__(self, K: int = K_DEFAULT, scale: float = 1.0):
        self.K = int(K)
        self.scale = float(scale)

        # --- square-free index set:  μ(n) ≠ 0 ---
        t0 = time.perf_counter()
        self.ns = square_free_indices(self.K)       # length M ≈ 0.6079 K
        self.M  = int(self.ns.size)
        t_sieve = (time.perf_counter() - t0) * 1e3

        # --- M-matrix trace parts (batched) ---
        t0 = time.perf_counter()
        re_n, im_n = m_trace_batch(self.ns)
        t_trace = (time.perf_counter() - t0) * 1e3

        # --- finite-step scaling (single scalar for the whole grid) ---
        # use the central finite step at n = 4 as the reference slope
        ref_step = DC_A_4 if DC_A_4 != 0.0 else 1.0
        step_gain = DA_4 / ref_step

        # --- negative-index coordinates:  x = y = −n ---
        # outer product to build the (M, M) grid
        neg = -self.ns.astype(np.float32)           # float32 → half the RAM
        self.X = np.outer(neg, np.ones_like(neg, dtype=np.float32))
        self.Y = np.outer(np.ones_like(neg, dtype=np.float32), neg)

        # --- z = A(4) + ΔA(4)·Re(tr M_n) as a rank-1 combination ---
        # we want z_ij = A4 + step_gain * (re_n[i] + re_n[j]) / 2
        re32 = re_n.astype(np.float32)
        diag = 0.5 * (np.outer(re32, np.ones_like(re32, dtype=np.float32))
                    + np.outer(np.ones_like(re32, dtype=np.float32), re32))
        self.Z = (A_4 + step_gain * diag).astype(np.float32)

        # --- hue = α · Im(tr M_n), rank-1 symmetric magnitude ---
        im32 = im_n.astype(np.float32)
        self.hue = (ALPHA * np.sqrt(
            np.outer(im32, np.ones_like(im32, dtype=np.float32)) ** 2
          + np.outer(np.ones_like(im32, dtype=np.float32), im32) ** 2
        )).astype(np.float32)

        # --- diagnostics ---
        self.t_sieve_ms = t_sieve
        self.t_trace_ms = t_trace
        self.bytes_X   = self.X.nbytes
        self.bytes_Y   = self.Y.nbytes
        self.bytes_Z   = self.Z.nbytes
        self.bytes_hue = self.hue.nbytes
        self.dense_K2  = self.K * self.K
        self.sparse_M2 = self.M * self.M

    # --------------------------------------------------------
    def summary(self) -> dict:
        return dict(
            K          = self.K,
            M          = self.M,
            square_frac= self.M / self.K,
            dense_K2   = self.dense_K2,
            sparse_M2  = self.sparse_M2,
            saving_pct = 100.0 * (1 - self.sparse_M2 / self.dense_K2),
            t_sieve_ms = self.t_sieve_ms,
            t_trace_ms = self.t_trace_ms,
            total_bytes= (self.bytes_X + self.bytes_Y
                          + self.bytes_Z + self.bytes_hue),
            A4         = A_4,
            DA4        = DA_4,
            DC_A4      = DC_A_4,
        )

    # --------------------------------------------------------
    def to_point_cloud(self, stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Flatten to (N, 3) points and (N,) hue, optionally strided."""
        X = self.X[::stride, ::stride].ravel()
        Y = self.Y[::stride, ::stride].ravel()
        Z = self.Z[::stride, ::stride].ravel()
        H = self.hue[::stride, ::stride].ravel()
        return np.column_stack([X, Y, Z]), H


# ============================================================
#  GLSL emitter  ·  same math, sparse Möbius gate
# ============================================================
GLSL_SPARSE = r"""
#version 450
layout(local_size_x = 64) in;

// ---- square-free index list (length M) ----
layout(std430, binding = 0) readonly buffer NsBlock {
    uint ns[M];
};

// ---- |Ci| trace array (length 2K+1, float32) ----
layout(std430, binding = 1) readonly buffer CiBlock {
    float c[2 * K + 1];
};

// ---- output: (x, y, z, hue) per vertex ----
layout(std430, binding = 2) writeonly buffer OutBlock {
    vec4 vert[M * M];
};

const int  K        = __K__;
const int  M        = __M__;
const float A_4     = __A_4__;
const float DA_4    = __DA_4__;
const float DC_A_4  = __DC_A_4__;
const float ALPHA   = __ALPHA__;
const float STEP_GAIN = DA_4 / (DC_A_4 > 0.0 ? DC_A_4 : 1.0);

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= uint(M * M)) return;
    uint i = gid / uint(M);
    uint j = gid % uint(M);

    uint ni = ns[i];
    uint nj = ns[j];

    // negative index grid
    float x = -float(ni);
    float y = -float(nj);

    // sym / asym trace parts
    float sym_i  = c[K + int(ni)];      // Re
    float sym_j  = c[K + int(nj)];
    float asym_i = c[K - int(ni)];      // Im
    float asym_j = c[K - int(nj)];

    float z   = A_4 + STEP_GAIN * 0.5 * (sym_i + sym_j);
    float hue = ALPHA * sqrt(asym_i * asym_i + asym_j * asym_j);

    vert[gid] = vec4(x, y, z, hue);
}
"""

def emit_glsl_sparse(grid: SparseNegGrid3D) -> str:
    return (GLSL_SPARSE
            .replace("__K__",     str(grid.K))
            .replace("__M__",     str(grid.M))
            .replace("__A_4__",   f"{A_4:.10f}")
            .replace("__DA_4__",  f"{DA_4:.10f}")
            .replace("__DC_A_4__",f"{DC_A_4:.10f}")
            .replace("__ALPHA__", f"{ALPHA:.10f}"))


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("Sparse 3D model  ·  negative-index grid  ·  square-free μ(n) ≠ 0")
    print("=" * 78)
    print(f"  α = 1/(π − e) = {ALPHA:.8f}")
    print(f"  A(4)          = {A_4:.8f}")
    print(f"  ΔA(4)         = {DA_4:.8f}")
    print(f"  Δ_c A(4)      = {DC_A_4:.8f}   (≈ π)")
    print()

    K = K_DEFAULT
    t0 = time.perf_counter()
    grid = SparseNegGrid3D(K=K, scale=1.0)
    t_build = (time.perf_counter() - t0) * 1e3
    s = grid.summary()

    print(f"--- build  ·  K = {K} ---")
    print(f"  square-free M        = {s['M']}   "
          f"({100*s['square_frac']:.2f} % of K)")
    print(f"  dense  grid K²       = {s['dense_K2']:,}")
    print(f"  sparse grid M²       = {s['sparse_M2']:,}")
    print(f"  saving               = {s['saving_pct']:.2f} %")
    print(f"  total build          = {t_build:.2f} ms")
    print(f"    · μ sieve          = {s['t_sieve_ms']:.2f} ms")
    print(f"    · trace batch      = {s['t_trace_ms']:.2f} ms")
    print()
    print(f"  float32 payload      = {s['total_bytes']:,} bytes "
          f"({s['total_bytes'] / 1e6:.2f} MB)")
    print()

    # ---------- verify the layout invariants ----------
    K_, M = grid.K, grid.M
    ns = grid.ns
    # pick three sample indices
    print("--- sample vertices ---")
    print(f"  {'i':>5}  {'j':>5}  {'n_i':>5}  {'n_j':>5}  "
          f"{'x':>7}  {'y':>7}  {'z':>10}  {'hue':>8}")
    for i, j in [(0, 0), (1, 2), (10, 20), (M//2, M//2), (M-1, M-1)]:
        print(f"  {i:>5}  {j:>5}  {ns[i]:>5}  {ns[j]:>5}  "
              f"{grid.X[i, j]:>7.2f}  {grid.Y[i, j]:>7.2f}  "
              f"{grid.Z[i, j]:>+10.5f}  {grid.hue[i, j]:>8.5f}")
    print()

    # ---------- supertrace over the negative grid ----------
    # S = Σ_n (−1)^n c[K−n]  (alternating sum over the imag half)
    im_n_vals = grid.hue.diagonal()
    signs     = np.where(ns & 1, -1.0, 1.0)
    S_neg     = float(np.sum(signs * im_n_vals))
    H_neg     = -ALPHA * (abs(S_neg) / M) * math.log(
                max(abs(S_neg) / M, 1e-30))
    m_neg     = abs(S_neg) * math.exp(-min(H_neg, 700.0))
    print(f"--- invariants over the negative-index diagonal ---")
    print(f"  supertrace S        = {S_neg:+.6f}")
    print(f"  entropy H           = {H_neg:.6f}")
    print(f"  invariant mass m    = {m_neg:.6e}")
    print()

    # ---------- GLSL ----------
    glsl = emit_glsl_sparse(grid)
    print("--- GLSL compute shader  ·  first 20 lines ---")
    for line in glsl.splitlines()[:20]:
        print(f"  {line}")
    print(f"  ...  ({len(glsl.splitlines())} lines total)")
    print()

    # ---------- complexity ----------
    print("--- complexity ---")
    print("  μ sieve              O(K)                     once")
    print("  square-free list     O(K)                     once")
    print("  trace batch          O(M)  vectorized          once")
    print("  grid outer products  O(M²) float32             once")
    print("  GLSL dispatch        O(M² / 64) workgroups")
    print("    · M² threads, O(1) work each")
    print()

    # ---------- visualization ----------
    if HAS_MPL:
        fig = plt.figure(figsize=(15, 9))
        gs = GridSpec(2, 3, figure=fig,
                      hspace=0.35, wspace=0.30)

        # (a) full sparse 3D surface
        ax = fig.add_subplot(gs[0, :2], projection="3d")
        stride = max(1, M // 128)
        ax.plot_surface(grid.X[::stride, ::stride],
                        grid.Y[::stride, ::stride],
                        grid.Z[::stride, ::stride],
                        facecolors=plt.cm.inferno(
                            grid.hue[::stride, ::stride] /
                            (grid.hue.max() + 1e-12)),
                        rstride=1, cstride=1,
                        linewidth=0, antialiased=True, alpha=0.9)
        ax.set_xlabel("−n_i"); ax.set_ylabel("−n_j"); ax.set_zlabel("z")
        ax.set_title(f"sparse 3D surface  ·  M = {M}  "
                     f"({100*s['saving_pct']:.1f} % smaller)")
        ax.view_init(elev=22, azim=-55)

        # (b) hue map on the negative grid
        ax = fig.add_subplot(gs[0, 2])
        im = ax.imshow(grid.hue, origin="lower", cmap="inferno",
                       extent=[-M, 0, -M, 0], aspect="equal")
        ax.set_xlabel("−n_i"); ax.set_ylabel("−n_j")
        ax.set_title("hue  ·  α·|Im(tr M_n)|")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # (c) M(k) real part — algebraic
        ax = fig.add_subplot(gs[1, 0])
        re_n, im_n = m_trace_batch(ns)
        ax.plot(ns, re_n, color="#2c3e50", lw=0.8,
                label="Re tr M_n")
        ax.set_xlabel("n"); ax.set_ylabel("value")
        ax.set_title("Algebraic (real) part  ·  O(n) grid")
        ax.grid(alpha=0.3); ax.legend(fontsize=9)

        # (d) M(k) imaginary part — Möbius
        ax = fig.add_subplot(gs[1, 1])
        ax.plot(ns, im_n, color="#e67e22", lw=0.8,
                label="Im tr M_n")
        ax.set_xlabel("n"); ax.set_ylabel("value")
        ax.set_title("Möbius (imaginary) part  ·  O(K log K)")
        ax.grid(alpha=0.3); ax.legend(fontsize=9)

        # (e) memory comparison
        ax = fig.add_subplot(gs[1, 2])
        ax.bar(["dense K²", "sparse M²"],
               [s["dense_K2"], s["sparse_M2"]],
               color=["#95a5a6", "#16a085"], edgecolor="k")
        for i, v in enumerate([s["dense_K2"], s["sparse_M2"]]):
            ax.text(i, v, f"  {v:,}", ha="center", va="bottom",
                    fontsize=9)
        ax.set_ylabel("vertex count")
        ax.set_title(f"Memory reduction  ·  {s['saving_pct']:.1f} %")
        ax.grid(alpha=0.3, axis="y")

        plt.suptitle(
            "Sparse 3D model  ·  negative-index grid  ·  square-free μ(n) ≠ 0  ·  "
            "A(k) = 10·a·sin(k²°)",
            fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    print("Done.")


if __name__ == "__main__":
    demo()