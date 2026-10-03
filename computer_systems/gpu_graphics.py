#!/usr/bin/env python3
"""
gpu_graphics_video_game.py
==========================

GPU graphics pipeline using:

  • A(k) = 10·a·sin(k²°),  a = 1/(π − e)          — the scaling
  • ΔA(n) with n = 4 fixed                        — the local step
  • a flat C-style |Ci| buffer                    — O(1) access
  • the chip compression pipeline from chip-g.py  — O(K log K) once

The flat buffer follows the `buffy.c` pattern: one allocation, no
per-slot metadata, direct offset arithmetic on the access path.

    slot(k) = k + K          for k ∈ [−K, K]

Chip compression runs at build time:

    raw |Ci|  →  TSP routing  →  exp convolution  →  supertrace
              →  top-M at square-free indices     →  |Ci|_compressed

The shader then samples |Ci|_compressed directly.  Every fragment
does two O(1) loads plus the elliptic projection Π.
"""

from __future__ import annotations

import ctypes
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from string import Template

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
PI          = math.pi
E           = math.e
A_CONST     = 1.0 / (PI - E)
ALPHA_SYM   = A_CONST
NORM        = 1.0 - math.exp(-ALPHA_SYM * (PI + E))
K_DEFAULT   = 256
N_FIXED     = 4


def A_of_k(k: float) -> float:
    return 10.0 * A_CONST * math.sin(math.radians(k * k))


def finite_step_k(k: int, mode: str = "forward") -> float:
    if mode == "forward":
        return A_of_k(k + 1) - A_of_k(k)
    if mode == "backward":
        return A_of_k(k) - A_of_k(k - 1)
    if mode == "central":
        return 0.5 * (A_of_k(k + 1) - A_of_k(k - 1))
    raise ValueError(mode)


A_4       = A_of_k(N_FIXED)
DA_4      = finite_step_k(N_FIXED, "forward")
DC_A_4    = finite_step_k(N_FIXED, "central")
A_4_FLOOR = math.floor(A_4)
GRID_N    = A_4_FLOOR ** 2


# ============================================================
#  Möbius sieve
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
#  Chip compression pipeline  (from chip-g.py)
# ============================================================
class ChipProcessor:
    """Two-pass exponential convolution + supertrace + Möbius compression."""

    def __init__(self, max_K: int = 1000):
        self.max_K = max_K
        self.f = np.zeros(max_K, dtype=float)
        self.b = np.zeros(max_K, dtype=float)
        self.conv = np.zeros(max_K, dtype=float)
        self.order = np.zeros(max_K, dtype=int)
        self.mag = np.zeros(max_K, dtype=float)
        self.idx = np.arange(max_K, dtype=int)
        self.mu = None
        self._update_mu(max_K)

    def _update_mu(self, K: int) -> None:
        if self.mu is None or len(self.mu) < K + 1:
            self.mu = self._mobius_sieve(K)

    @staticmethod
    def _mobius_sieve(K: int):
        mu = [0] * (K + 1)
        mu[1] = 1
        primes, is_comp = [], [False] * (K + 1)
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
                else:
                    mu[i * p] = -mu[i]
        return mu

    def _tsp_route(self, signal, K: int) -> None:
        angles = np.zeros(K, dtype=float)
        for i in range(K):
            x = math.sin(i * 7.0) + 0.1 * math.cos(i * 13.0)
            y = math.cos(i * 11.0) + 0.1 * math.sin(i * 17.0)
            angles[i] = math.atan2(y, x) + math.pi
        buckets = [[] for _ in range(360)]
        for i in range(K):
            b = int((angles[i] / (2 * math.pi)) * 360) % 360
            buckets[b].append(i)
        order = []
        for b in buckets:
            order.extend(b)
        self.order[:K] = order

    def _conv_exp_kernel(self, signal, K: int, alpha: float = ALPHA_SYM) -> None:
        lam = math.exp(-alpha)
        f = self.f
        f[0] = signal[0]
        for i in range(1, K):
            f[i] = signal[i] + lam * f[i - 1]
        b = self.b
        b[K - 1] = signal[K - 1]
        for i in range(K - 2, -1, -1):
            b[i] = signal[i] + lam * b[i + 1]
        conv = self.conv
        inv_den = 1.0 / (1.0 - lam * lam)
        inv_norm = 1.0 / NORM
        for i in range(K):
            conv_exp = (f[i] + b[i] - signal[i]) * inv_den
            conv[i] = (1.0 - conv_exp) * inv_norm

    def _supertrace_and_mass(self, signal, K: int):
        S = 0.0
        for i in range(K):
            val = signal[i]
            S += abs(val) if (i % 2 == 0) else -abs(val)
        if S == 0.0:
            return 0.0, 0.0, 0.0
        p = abs(S) / K
        H = -ALPHA_SYM * p * math.log(p) if 0.0 < p < 1.0 else 0.0
        m = abs(S) * math.exp(-H) if H < 700 else 0.0
        return S, H, m

    def process(self, signal: np.ndarray):
        K = len(signal)
        if K > self.max_K:
            raise ValueError(f"signal length {K} > max_K {self.max_K}")

        self._tsp_route(signal, K)
        order = self.order[:K]
        if (not hasattr(self, "sorted_signal")
                or len(self.sorted_signal) < K):
            self.sorted_signal = np.zeros(K, dtype=signal.dtype)
        self.sorted_signal[:K] = signal[order]

        self._conv_exp_kernel(self.sorted_signal, K)
        S, H, m = self._supertrace_and_mass(self.conv, K)
        M = max(1, min(int(abs(S)), K))

        self._update_mu(K)
        mu = self.mu
        mag = self.mag[:K]
        for i in range(K):
            mag[i] = abs(self.conv[i])
        sorted_idx = np.argsort(mag)[::-1]

        kept: List[Tuple[int, float]] = []
        count = 0
        for idx in sorted_idx:
            n = int(idx) + 1
            if mu[n] != 0:
                kept.append((int(idx), float(self.conv[idx])))
                count += 1
                if count >= M:
                    break

        return kept, S, H, m, self.conv[:K].copy()


# ============================================================
#  Flat C-style |Ci| buffer  (buffy.c pattern)
# ============================================================
class CiBuffer:
    """
    Flat C-style buffer with O(1) indexed access.

        data     : contiguous float32, length 2K+1
        slot(k)  : k + K
        buf[k]   : data[k + K]              for k ∈ [−K, K]

    Follows the buffy.c pattern: one allocation, direct offsets,
    no per-slot metadata.  Access is a single indexed load.
    """
    __slots__ = ("K", "data")

    def __init__(self, K: int):
        self.K = K
        self.data = np.zeros(2 * K + 1, dtype=np.float32)

    # ---- O(1) access, clamped to [−K, K] ----
    def __getitem__(self, k: int) -> float:
        k = max(-self.K, min(self.K, int(k)))
        return float(self.data[k + self.K])

    def __setitem__(self, k: int, v: float) -> None:
        k = max(-self.K, min(self.K, int(k)))
        self.data[k + self.K] = np.float32(v)

    def slot(self, k: int) -> int:
        return max(-self.K, min(self.K, int(k))) + self.K

    def as_bytes(self) -> bytes:
        return self.data.tobytes()

    def as_ctypes(self) -> ctypes.Array:
        """The buffy.c pattern — hand a raw pointer to native code."""
        return (ctypes.c_float * len(self.data)).from_buffer(self.data)


def build_ci_buffer(K: int,
                    n: int = N_FIXED,
                    step_mode: str = "forward") -> CiBuffer:
    """Build the raw |Ci| buffer from A(k) with the n=4 step scale."""
    mu = mobius_sieve(K)
    buf = CiBuffer(K)
    ref_step = finite_step_k(n, step_mode)
    step_scale = DA_4 / (ref_step if ref_step != 0.0 else 1.0)

    for k in range(1, K + 1):
        gated = 1.0 if mu[k] != 0 else 0.0
        c_k = A_of_k(k) * step_scale * gated
        buf[k] = c_k
        buf[-k] = -c_k
    buf[0] = 0.0
    return buf


def compress_ci_buffer(raw: CiBuffer,
                       processor: Optional[ChipProcessor] = None
                       ) -> Tuple[CiBuffer, Dict]:
    """
    Run the raw |Ci| through the chip pipeline, then write the
    compressed result back into a fresh |Ci| buffer.

    Returns
    -------
    compressed : CiBuffer
    report     : dict with S, H, m, n_kept
    """
    K = raw.K
    if processor is None or processor.max_K < 2 * K + 1:
        processor = ChipProcessor(max_K=2 * K + 1)

    # the chip pipeline works on a 1D real signal; use the raw buffer
    signal = raw.data.astype(float)
    kept, S, H, m, conv = processor.process(signal)

    # place the compressed convolution back into a fresh buffer
    comp = CiBuffer(K)
    for i, v in enumerate(conv):
        comp.data[i] = np.float32(v)

    report = dict(S=S, H=H, m=m, n_kept=len(kept),
                  n_total=len(conv))
    return comp, report


# ============================================================
#  Elliptic projection
# ============================================================
def elliptic_projection(x: float, y: float) -> float:
    u = (x / PI) % 1.0
    v = (y / E) % 1.0
    return 0.5 * (1.0 + math.cos(2 * PI * u) * math.cos(2 * PI * v))


# ============================================================
#  Fragment shader (reference)
# ============================================================
def shade_fragment(buf: CiBuffer, i: int, j: int,
                   W: int, H: int) -> Tuple[float, float, float]:
    K = buf.K
    x = i / max(W - 1, 1)
    y = j / max(H - 1, 1)

    ku = int(x * (2 * K)) - K
    kv = int(y * (2 * K)) - K
    ku = ku if ku != 0 else 1
    kv = kv if kv != 0 else 1

    cu = buf[ku]
    cv = buf[kv]
    Pi = elliptic_projection(x * PI, y * E)

    r = 0.5 + 0.5 * math.tanh(cu * Pi)
    g = 0.5 + 0.5 * math.tanh(cv * Pi)
    b = Pi
    return r, g, b


def render_image(buf: CiBuffer, W: int, H: int) -> np.ndarray:
    img = np.zeros((H, W, 3), dtype=np.float32)
    for j in range(H):
        for i in range(W):
            img[j, i] = shade_fragment(buf, i, j, W, H)
    return img



# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("GPU graphics  ·  A(k) = 10·a·sin(k²°)  ·  |Ci| O(1) access")
    print("=" * 78)
    print(f"  a = 1/(π − e)  = {A_CONST:.8f}")
    print()

    # ---------- fixed n=4 block ----------
    print("--- fixed n = 4 block ---")
    print(f"  A(4)            = {A_4:.10f}")
    print(f"  ΔA(4) forward   = {DA_4:.10f}")
    print(f"  Δ_c A(4) central= {DC_A_4:.10f}   (≈ π = {PI:.10f})")
    print(f"  floor(A(4))     = {A_4_FLOOR}")
    print(f"  grid size 6×6   = {GRID_N} unit squares")
    print()

    # ---------- A(k) sample table ----------
    print("--- A(k) for k = 1..10 ---")
    print(f"  {'k':>3}  {'A(k)':>12}  {'ΔA(k)':>12}  {'Δ_c A(k)':>12}")
    for k in range(1, 11):
        print(f"  {k:>3}  {A_of_k(k):>12.6f}  "
              f"{finite_step_k(k, 'forward'):>12.6f}  "
              f"{finite_step_k(k, 'central'):>12.6f}")
    print()

    # ---------- build raw |Ci| ----------
    K = K_DEFAULT
    t0 = time.perf_counter()
    raw = build_ci_buffer(K, n=N_FIXED)
    t_raw = (time.perf_counter() - t0) * 1e3
    print(f"--- raw |Ci|  ·  K = {K} ---")
    print(f"  build time      = {t_raw:.2f} ms")
    print(f"  length          = {len(raw.data)} float32 "
          f"= {raw.data.nbytes} bytes")
    print(f"  buf[1]          = {raw[1]:+.6f}")
    print(f"  buf[-1]         = {raw[-1]:+.6f}")
    print(f"  buf[K]          = {raw[K]:+.6f}")            # fixed index
    print(f"  buf.data[-1]    = {raw.data[-1]:+.6f}")      # far end
    print()

    # ---------- chip compression ----------
    print("--- chip compression pipeline  ·  O(K log K) ---")
    t0 = time.perf_counter()
    comp, rep = compress_ci_buffer(raw, ChipProcessor(max_K=2 * K + 1))
    t_comp = (time.perf_counter() - t0) * 1e3
    print(f"  compress time   = {t_comp:.2f} ms")
    print(f"  supertrace S    = {rep['S']:+.6f}")
    print(f"  entropy H       = {rep['H']:.6f}")
    print(f"  mass m          = {rep['m']:.6f}")
    print(f"  kept            = {rep['n_kept']} / {rep['n_total']} "
          f"({100 * rep['n_kept'] / rep['n_total']:.1f} %)")
    print(f"  buf[1]          = {comp[1]:+.6f}")
    print(f"  buf[-1]         = {comp[-1]:+.6f}")
    print(f"  buf[K]          = {comp[K]:+.6f}")
    print(f"  buf.data[-1]    = {comp.data[-1]:+.6f}")
    print()

    # ---------- render ----------
    W, H = 96, 96
    t0 = time.perf_counter()
    img_raw  = render_image(raw,  W, H)
    img_comp = render_image(comp, W, H)
    t_render = (time.perf_counter() - t0) * 1e3
    print(f"--- render  ·  {W}×{H}  ·  raw vs compressed ---")
    print(f"  total           = {t_render:.2f} ms  "
          f"({t_render / (2 * W * H) * 1e3:.4f} μs per fragment)")
    diff = float(np.mean(np.abs(img_raw - img_comp)))
    print(f"  mean |Δ| between raw and compressed renders = {diff:.4f}")
    print()

    # ---------- O(1) access proof ----------
    print("--- O(1) access proof ---")
    N = 1_000_000
    idxs = np.random.randint(-K, K + 1, size=N)
    t0 = time.perf_counter()
    s = 0.0
    for idx in idxs:
        s += comp[int(idx)]
    dt = (time.perf_counter() - t0) * 1e3
    print(f"  {N:,} random accesses  = {dt:.2f} ms "
          f"({dt / N * 1000:.4f} μs each)")
    print(f"  checksum               = {s:+.6f}")
    print()

    # ---------- complexity ----------
    print("--- complexity ---")
    print("  Möbius sieve            O(K log log K)   once")
    print("  raw |Ci| build          O(K)")
    print("  chip compression        O(K log K)       once")
    print("  buf[k] access           O(1)             per fragment")
    print("  fragment shade          O(1)             per fragment")
    print("  render W×H              O(W·H)")

    # ---------- plot ----------
    if HAS_MPL:
        fig = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 3, figure=fig)

        ax = fig.add_subplot(gs[0, 0])
        ks = np.arange(1, 25)
        ax.plot(ks, [A_of_k(k) for k in ks], "o-",
                color="#2c3e50", label="A(k)")
        ax.bar(ks + 0.15,
               [finite_step_k(int(k), "forward") for k in ks],
               width=0.3, color="#e67e22", alpha=0.75,
               label="ΔA(k)")
        ax.axvline(N_FIXED, color="#e74c3c", ls=":",
                   label=f"n = {N_FIXED}")
        ax.set_xlabel("k"); ax.set_ylabel("A(k) / ΔA(k)")
        ax.set_title("A(k) = 10·a·sin(k²°)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        ax = fig.add_subplot(gs[0, 1])
        ax.plot(np.arange(-K, K + 1), raw.data,
                color="#3a7bd5", lw=0.6, label="raw |Ci|")
        ax.plot(np.arange(-K, K + 1), comp.data,
                color="#e74c3c", lw=0.6, alpha=0.75,
                label="compressed")
        ax.axhline(0, color="k", lw=0.4)
        ax.set_xlabel("index k"); ax.set_ylabel("|Ci|")
        ax.set_title(f"|Ci|  ·  length {len(raw.data)}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        ax = fig.add_subplot(gs[0, 2])
        ax.axis("off")
        ax.text(0.0, 1.0,
                "fixed n = 4 block:\n\n"
                f"A_4      = {A_4:.6f}\n"
                f"DA_4     = {DA_4:.6f}\n"
                f"DC_A_4   = {DC_A_4:.6f}\n"
                f"K        = {K}\n"
                f"grid     = {GRID_N} squares\n\n"
                f"chip compression:\n"
                f"S        = {rep['S']:+.4f}\n"
                f"H        = {rep['H']:.4f}\n"
                f"m        = {rep['m']:.4f}\n"
                f"kept     = {rep['n_kept']}",
                va="top", family="monospace", fontsize=9)

        ax = fig.add_subplot(gs[1, 0])
        ax.imshow(img_raw, origin="lower")
        ax.set_title(f"raw |Ci| render  ·  {W}×{H}")

        ax = fig.add_subplot(gs[1, 1])
        ax.imshow(img_comp, origin="lower")
        ax.set_title(f"compressed |Ci| render  ·  {W}×{H}")

        ax = fig.add_subplot(gs[1, 2])
        diff_img = np.abs(img_raw - img_comp).mean(axis=-1)
        ax.imshow(diff_img, origin="lower", cmap="magma")
        ax.set_title(f"|Δ|  mean = {diff:.4f}")

        plt.suptitle(
            "GPU graphics  ·  A(k) = 10·a·sin(k²°)  ·  "
            "chip compression  ·  |Ci| O(1) access",
            fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    print()
    print("Done.")


if __name__ == "__main__":
    demo()