#!/usr/bin/env python3
"""
mobius_lin_gpu.py
=================

GPU linear algebra in the 64-bit |Ci| layout.

    M(x, y) = [x^-1; y^-1] ⊗ [x^i, y^i]

Trace:
    tr M(x, y) = x^(i-1) + y^(i-1)
               = x^-1 (cos ln x + i sin ln x)
               + y^-1 (cos ln y + i sin ln y)

|Ci| slot layout (length 2K+1, float64):

    c[K + k]   k ∈ [1, K]    symmetric  (real plane)   Re(tr M_k)
    c[K − k]   k ∈ [1, K]    asymmetric (imag plane)   Im(tr M_k)
    c[K]                     DC term                   Σ Re(tr M_k)

Quick invariants per slot k (two O(1) loads):

    sym  = c[K + k]
    asym = c[K − k]
    tr_re      = sym
    tr_im      = asym
    |tr|²      = sym² + asym²
    det_approx = sym² − asym²

The positive and negative halves are independent streams and can be
processed in parallel — one workgroup per slot, or K threads total.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Tuple

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
ALPHA      = 1.0 / (PI - E)             # ≈ 2.362
K_DEFAULT  = 256


# ============================================================
#  M-matrix primitives
# ============================================================
def m_matrix(x: float, y: float) -> np.ndarray:
    """M(x, y) = [x^-1; y^-1] ⊗ [x^i, y^i]  →  2×2 complex."""
    x_ = complex(abs(x) + 1e-9)
    y_ = complex(abs(y) + 1e-9)
    a = np.array([1.0 / x_, 1.0 / y_], dtype=complex)
    t = np.array([x_ ** 1j, y_ ** 1j], dtype=complex)
    return np.outer(a, t)


def m_trace(x: float, y: float) -> complex:
    """tr M(x, y) = x^(i−1) + y^(i−1)."""
    x_ = complex(abs(x) + 1e-9)
    y_ = complex(abs(y) + 1e-9)
    return x_ ** (1j - 1.0) + y_ ** (1j - 1.0)


# ============================================================
#  |Ci| array  ·  symmetric / asymmetric separation
# ============================================================
class MobiusLinAlg64:
    """
    Length-(2K+1) float64 |Ci| array.

        data[K + k]   symmetric  (real plane)     Re(tr M_k)
        data[K − k]   asymmetric (imag plane)     Im(tr M_k)
        data[K]       DC term
    """
    __slots__ = ("K", "data")

    def __init__(self, K: int = K_DEFAULT):
        self.K = int(K)
        self.data = np.zeros(2 * self.K + 1, dtype=np.float64)

    # ---------- O(1) writes ----------
    def set_matrix(self, k: int, x: float, y: float) -> None:
        if not (1 <= k <= self.K):
            raise ValueError(f"k = {k} out of [1, {self.K}]")
        tr = m_trace(x, y)
        self.data[self.K + k] = tr.real          # symmetric half
        self.data[self.K - k] = tr.imag          # asymmetric half

    def set_dc(self, v: float) -> None:
        self.data[self.K] = float(v)

    # ---------- O(1) reads ----------
    def sym(self, k: int) -> float:
        return float(self.data[self.K + k])

    def asym(self, k: int) -> float:
        return float(self.data[self.K - k])

    def quick_invariants(self, k: int) -> Dict[str, float]:
        """Two indexed loads, five scalars."""
        sym  = self.sym(k)
        asym = self.asym(k)
        return {
            "sym":          sym,
            "asym":         asym,
            "trace_re":     sym,
            "trace_im":     asym,
            "trace_abs2":   sym * sym + asym * asym,
            "det_approx":   sym * sym - asym * asym,
            "dc":           float(self.data[self.K]),
        }

    # ---------- O(K) aggregate ----------
    def supertrace(self) -> float:
        """S = Σ (−1)^i c_i over the (2K+1) slots."""
        S = 0.0
        for i, v in enumerate(self.data):
            S += v if (i % 2 == 0) else -v
        return float(S)

    def symmetric_energy(self) -> float:
        """Σ sym_k² over the positive half."""
        return float(np.sum(self.data[self.K + 1:2 * self.K + 1] ** 2))

    def asymmetric_energy(self) -> float:
        """Σ asym_k² over the negative half."""
        return float(np.sum(self.data[0:self.K] ** 2))

    # ---------- GPU payload ----------
    def as_bytes(self) -> bytes:
        return self.data.tobytes()


def build_lin_batch(pairs: List[Tuple[float, float]],
                    K: int | None = None) -> MobiusLinAlg64:
    """Build the |Ci| array from a list of (x, y) pairs."""
    if K is None:
        K = max(len(pairs), 1)
    linalg = MobiusLinAlg64(K)
    for k, (x, y) in enumerate(pairs, start=1):
        if k > K:
            break
        linalg.set_matrix(k, x, y)
    dc = sum(m_trace(x, y).real for x, y in pairs)
    linalg.set_dc(dc)
    return linalg


# ============================================================
#  GLSL emitter for the lin alg pass
# ============================================================
GLSL_LIN = r"""
#version 450
layout(local_size_x = 64) in;

// ---- |Ci| layout: 2K+1 doubles ----
layout(std430, binding = 0) readonly buffer CiBlock {
    double c[2 * K + 1];
};

// ---- output: 5 invariants per slot ----
layout(std430, binding = 1) writeonly buffer OutBlock {
    double inv[5 * K];
};

const int K = __K__;

void main() {
    uint k = gl_GlobalInvocationID.x;
    if (k >= uint(K)) return;

    int slot_pos = K + int(k + 1);   // positive half  → symmetric
    int slot_neg = K - int(k + 1);   // negative half  → asymmetric

    // two O(1) loads
    double sym  = c[slot_pos];
    double asym = c[slot_neg];

    // five quick invariants
    double tr_re     = sym;
    double tr_im     = asym;
    double tr_abs2   = sym * sym + asym * asym;
    double det_approx = sym * sym - asym * asym;

    inv[5u * k + 0u] = tr_re;
    inv[5u * k + 1u] = tr_im;
    inv[5u * k + 2u] = tr_abs2;
    inv[5u * k + 3u] = det_approx;
    inv[5u * k + 4u] = c[K];        // DC term
}
"""


def emit_glsl_lin(K: int) -> str:
    """Emit the lin alg shader with the K constant substituted."""
    return GLSL_LIN.replace("__K__", str(K))


# ============================================================
#  Reference CPU kernel  (mirrors the GLSL exactly)
# ============================================================
def cpu_quick_invariants(linalg: MobiusLinAlg64) -> np.ndarray:
    """
    Returns inv[k] = [tr_re, tr_im, tr_abs2, det_approx, dc]
    for k = 1 … K.
    """
    K = linalg.K
    out = np.zeros((K, 5), dtype=np.float64)
    dc = float(linalg.data[K])
    for k in range(1, K + 1):
        sym = linalg.data[K + k]
        asym = linalg.data[K - k]
        out[k - 1, 0] = sym
        out[k - 1, 1] = asym
        out[k - 1, 2] = sym * sym + asym * asym
        out[k - 1, 3] = sym * sym - asym * asym
        out[k - 1, 4] = dc
    return out


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("GPU linear algebra  ·  64-bit |Ci| layout  ·  symmetric / asymmetric")
    print("=" * 78)
    print(f"  α = 1/(π − e)  = {ALPHA:.8f}")
    print()

    K = K_DEFAULT

    # ---------- build a batch of M-matrices ----------
    pairs: List[Tuple[float, float]] = []
    for k in range(1, K + 1):
        # a smooth sweep through [0.5, 5.0] with a chirp on y
        x = 0.5 + (k - 1) / (K - 1) * 4.5
        y = 0.5 + (k - 1) / (K - 1) * 4.5
        pairs.append((x, y))

    t0 = time.perf_counter()
    linalg = build_lin_batch(pairs, K=K)
    t_build = (time.perf_counter() - t0) * 1e3

    print(f"--- |Ci| array  ·  K = {K} ---")
    print(f"  build time      = {t_build:.2f} ms")
    print(f"  length          = {len(linalg.data)} float64 "
          f"= {linalg.data.nbytes} bytes")
    print(f"  data[K]    DC   = {linalg.data[K]:+.6f}")
    print(f"  data[K+1]  sym  = {linalg.sym(1):+.6f}")
    print(f"  data[K-1]  asym = {linalg.asym(1):+.6f}")
    print(f"  data[K+K]  sym  = {linalg.sym(K):+.6f}")
    print(f"  data[0]    asym = {linalg.asym(K):+.6f}")
    print()

    # ---------- quick invariants for a few slots ----------
    print("--- quick invariants (O(1) per slot) ---")
    print(f"  {'k':>4}  {'sym':>10}  {'asym':>10}  "
          f"{'|tr|²':>10}  {'det_approx':>12}")
    for k in [1, 16, 64, 128, 192, 256]:
        inv = linalg.quick_invariants(k)
        print(f"  {k:>4}  {inv['sym']:>+10.5f}  {inv['asym']:>+10.5f}  "
              f"{inv['trace_abs2']:>10.5f}  {inv['det_approx']:>+12.5f}")
    print()

    # ---------- aggregate invariants ----------
    S        = linalg.supertrace()
    E_sym    = linalg.symmetric_energy()
    E_asym   = linalg.asymmetric_energy()
    print("--- aggregate invariants ---")
    print(f"  supertrace S        = {S:+.6f}")
    print(f"  symmetric energy    = {E_sym:.6f}    Σ sym²")
    print(f"  asymmetric energy   = {E_asym:.6f}    Σ asym²")
    print(f"  energy ratio        = "
          f"{E_sym / max(E_asym, 1e-12):.4f}")
    print()

    # ---------- CPU kernel benchmark ----------
    print("--- CPU kernel  ·  quick invariants per slot ---")
    N = 1000
    t0 = time.perf_counter()
    for _ in range(N):
        _ = cpu_quick_invariants(linalg)
    dt = (time.perf_counter() - t0) * 1e3
    per_slot = dt / N / K * 1e3
    print(f"  {N} passes × {K} slots  = {dt:.2f} ms total")
    print(f"  per pass                = {dt / N:.4f} ms")
    print(f"  per slot                = {per_slot:.6f} μs")
    print()

    # ---------- GPU payload ----------
    print("--- GPU payload ---")
    print(f"  |Ci| buffer     : {linalg.data.nbytes} bytes "
          f"({len(linalg.data)} float64)")
    print(f"  output buffer   : {K * 5 * 8} bytes "
          f"({K * 5} float64)  ·  5 invariants per slot")
    print()

    # ---------- GLSL ----------
    glsl = emit_glsl_lin(K)
    print("--- GLSL compute shader  ·  first 30 lines ---")
    for line in glsl.splitlines()[:30]:
        print(f"  {line}")
    print(f"  ...  ({len(glsl.splitlines())} lines total)")
    print()

    # ---------- complexity ----------
    print("--- complexity ---")
    print("  set_matrix(k, x, y)     O(1)             per slot")
    print("  quick_invariants(k)     O(1)             per slot")
    print("    · 2 loads, 3 mults, 1 add")
    print("  supertrace              O(K)             whole array")
    print("  symmetric_energy        O(K)")
    print("  asymmetric_energy       O(K)")
    print("  GPU dispatch            O(K / 64)        workgroups")
    print("    · K threads, each doing O(1) work")

    # ---------- plot ----------
    if HAS_MPL:
        fig = plt.figure(figsize=(15, 8))
        gs = GridSpec(2, 3, figure=fig)

        # (a) |Ci| layout with symmetric / asymmetric halves
        ax = fig.add_subplot(gs[0, :2])
        idx = np.arange(-K, K + 1)
        ax.plot(idx, linalg.data, color="#2c3e50", lw=0.7)
        ax.axvline(0, color="k", lw=0.5, ls=":")
        ax.axhspan(linalg.data[K + 1:2 * K + 1].min(),
                   linalg.data[K + 1:2 * K + 1].max(),
                   color="#3498db", alpha=0.12,
                   label="positive half  (symmetric, real)")
        ax.axhspan(linalg.data[0:K].min(),
                   linalg.data[0:K].max(),
                   color="#e74c3c", alpha=0.12,
                   label="negative half  (asymmetric, imag)")
        ax.set_xlabel("index  k")
        ax.set_ylabel("|Ci| value")
        ax.set_title(f"|Ci| array  ·  length {len(linalg.data)}  ·  "
                     f"DC = {linalg.data[K]:+.3f}")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # (b) invariants bar summary
        ax = fig.add_subplot(gs[0, 2])
        ax.axis("off")
        ax.text(0.0, 1.0,
                "aggregate invariants:\n\n"
                f"  supertrace S        = {S:+.4f}\n"
                f"  symmetric energy    = {E_sym:.4f}\n"
                f"  asymmetric energy   = {E_asym:.4f}\n"
                f"  ratio E_sym / E_asym= "
                f"{E_sym / max(E_asym, 1e-12):.4f}\n\n"
                f"GPU payload:\n"
                f"  |Ci| buffer  = {linalg.data.nbytes} B\n"
                f"  output       = {K * 5 * 8} B\n"
                f"  dispatch     = {K // 64} workgroups",
                va="top", family="monospace", fontsize=10)

        # (c) symmetric vs asymmetric halves
        ax = fig.add_subplot(gs[1, 0])
        ax.plot(np.arange(1, K + 1),
                linalg.data[K + 1:2 * K + 1],
                color="#3498db", lw=1.0, label="symmetric  c[K+k]")
        ax.set_xlabel("k"); ax.set_ylabel("value")
        ax.set_title("Positive half  ·  Re(tr M_k)")
        ax.grid(alpha=0.3); ax.legend(fontsize=9)

        ax = fig.add_subplot(gs[1, 1])
        ax.plot(np.arange(1, K + 1),
                linalg.data[K - 1::-1],
                color="#e74c3c", lw=1.0, label="asymmetric  c[K−k]")
        ax.set_xlabel("k"); ax.set_ylabel("value")
        ax.set_title("Negative half  ·  Im(tr M_k)")
        ax.grid(alpha=0.3); ax.legend(fontsize=9)

        # (d) quick invariants
        inv = cpu_quick_invariants(linalg)
        ax = fig.add_subplot(gs[1, 2])
        ax.plot(np.arange(1, K + 1), inv[:, 2],
                color="#2ecc71", lw=1.0, label="|tr|²")
        ax.plot(np.arange(1, K + 1), inv[:, 3],
                color="#8e44ad", lw=1.0, alpha=0.8,
                label="det_approx")
        ax.set_xlabel("k"); ax.set_ylabel("value")
        ax.set_title("Quick invariants  ·  O(1) per slot")
        ax.grid(alpha=0.3); ax.legend(fontsize=9)

        plt.suptitle(
            "GPU linear algebra  ·  64-bit |Ci| layout  ·  "
            "symmetric ↔ positive half,  asymmetric ↔ negative half",
            fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    print("Done.")


if __name__ == "__main__":
    demo()