#!/usr/bin/env python3
"""
gpu_ugc_reasoning_pipeline.py
=============================

GPU-accelerated UGC reasoning net.

CUDA kernels (Numba):
  · mobius_sieve_kernel      parallel μ  (Erato-style)
  · mobius_conv_kernel       F(n) = (μ*H)(n)  ·  scatter-add
  · encode_real_kernel       H-index binary search + 7-bit fraction
  · encode_imag_kernel       F-bucket + 12-bit quant + 9-bit trace
  · forward_pass_kernel      matvec + UGC activation
  · zeta_eval_kernel         ζ(t) = Σ C[i]·exp(i·t·i/α)

CPU fallback: pure NumPy.  Auto-detected via cuda.is_available().
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from numba import cuda
    HAS_CUDA = cuda.is_available()
except Exception:
    HAS_CUDA = False


# ============================================================
#  Constants
# ============================================================
PI       = math.pi
E        = math.e
ALPHA    = 1.0 / (PI - E)              # ≈ 2.362338
GAMMA    = 0.5772156649015329
DENSITY  = 6.0 / (PI * PI)             # ≈ 0.607927
BLOCK    = 6
N_PERM   = math.factorial(BLOCK)       # 720
A_STEP   = ALPHA / 0.3628              # ≈ 6.511
N_BUCKETS = 4096                        # |Im|-quantization buckets


# ============================================================
#  CPU reference  (fallback + validation)
# ============================================================
def mobius_sieve_cpu(K: int) -> np.ndarray:
    if K < 1:
        return np.zeros(K + 1, dtype=np.int8)
    mu = np.zeros(K + 1, dtype=np.int8)
    mu[1] = 1
    primes, is_comp = [], np.zeros(K + 1, dtype=bool)
    for i in range(2, K + 1):
        if not is_comp[i]:
            primes.append(i); mu[i] = -1
        for p in primes:
            if i * p > K: break
            is_comp[i * p] = True
            if i % p == 0:
                mu[i * p] = 0; break
            mu[i * p] = -mu[i]
    return mu


def harmonic_cpu(K: int) -> np.ndarray:
    H = np.zeros(K + 1, dtype=np.float64)
    if K >= 1:
        H[1] = 1.0
    for n in range(2, K + 1):
        H[n] = H[n - 1] + 1.0 / n
    return H


def mu_conv_cpu(K: int, mu: np.ndarray, H: np.ndarray) -> np.ndarray:
    F = np.zeros(K + 1, dtype=np.float64)
    for d in range(1, K + 1):
        if mu[d] == 0:
            continue
        md = float(mu[d])
        for m in range(1, K // d + 1):
            F[d * m] += md * H[m]
    return F


# ============================================================
#  GPU kernels
# ============================================================
if HAS_CUDA:

    @cuda.jit
    def _sieve_primes_kernel(is_prime):
        n = cuda.grid(1)
        if n >= is_prime.size:
            return
        if n < 2:
            is_prime[n] = False
        elif n < 4:
            is_prime[n] = True
        elif n % 2 == 0:
            is_prime[n] = False
        else:
            is_prime[n] = True
            d = 3
            while d * d <= n:
                if n % d == 0:
                    is_prime[n] = False
                    break
                d += 2

    @cuda.jit
    def _mu_toggle_primes_kernel(mu, is_prime):
        p = cuda.grid(1)
        if p >= is_prime.size or not is_prime[p]:
            return
        for k in range(p, mu.size, p):
            mu[k] = -mu[k]

    @cuda.jit
    def _mu_zero_squares_kernel(mu, is_prime):
        p = cuda.grid(1)
        if p >= is_prime.size or not is_prime[p]:
            return
        sq = p * p
        if sq >= mu.size:
            return
        for k in range(sq, mu.size, sq):
            mu[k] = 0

    @cuda.jit
    def _mobius_conv_kernel(mu, H, F, K):
        """
        Scatter-add:  F(d·k) += μ(d)·H(k)  for every square-free d.
        Total work O(K log K), parallel over d.
        """
        d = cuda.grid(1)
        if d < 1 or d >= mu.size:
            return
        md = mu[d]
        if md == 0:
            return
        md_f = float(md)
        for k in range(1, K // d + 1):
            cuda.atomic.add(F, d * k, md_f * H[k])

    @cuda.jit
    def _encode_real_kernel(values, H, K, words_re):
        """sign(1) | n(24) | fraction(7)  — from  |Re(Π₆)|."""
        i = cuda.grid(1)
        if i >= values.size:
            return
        v = values[i]
        sign = 1 if v < 0 else 0
        a = abs(v)
        if a <= 0.0:
            words_re[i] = 0
            return
        lo, hi = 1, K - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if H[mid] <= a:
                lo = mid
            else:
                hi = mid - 1
        n_re = lo
        lo_h = H[n_re]
        hi_h = H[n_re + 1] if n_re + 1 <= K else lo_h
        frac = (a - lo_h) / (hi_h - lo_h) if hi_h > lo_h else 0.0
        frac_q = int(min(127, max(0, round(frac * 127.0))))
        words_re[i] = ((sign & 1) << 31) | ((n_re & 0xFFFFFF) << 7) | frac_q

    @cuda.jit
    def _encode_imag_kernel(values, F_abs, F_abs_max, bucket_n, words_im):
        """sign(1) | σ*(10) | F_q(12) | trace(9)  — from  α·|F(n)| ≈ |Im|."""
        i = cuda.grid(1)
        if i >= values.size:
            return
        v = values[i]
        sign = 1 if v < 0 else 0
        a = abs(v)
        if a <= 0.0 or F_abs_max <= 0.0:
            words_im[i] = 0
            return
        q = int(min(N_BUCKETS - 1,
                    max(0, int(a / F_abs_max * N_BUCKETS))))
        n_im = bucket_n[q]
        F_q = int(min(0xFFF, max(0, round(a / F_abs_max * 0xFFF))))
        trace_q = int(min(0x1FF, max(0,
                round(F_abs[n_im] / F_abs_max * 0x1FF))))
        # Lehmer σ* = 0 (identity permutation, current CPU default)
        words_im[i] = ((sign & 1) << 31) | (F_q << 9) | trace_q

    @cuda.jit
    def _decode_weights_kernel(words_re, words_im, H, K, W_out):
        """Single-scalar reconstruction: ½(mag_re + mag_im)."""
        i = cuda.grid(1)
        if i >= words_re.size:
            return
        wr = words_re[i]
        wi = words_im[i]
        s_re = (wr >> 31) & 1
        n_re = (wr >> 7) & 0xFFFFFF
        frac = wr & 0x7F
        if n_re <= 0:
            mag = 0.0
        else:
            n_re = min(n_re, K - 1)
            lo = H[n_re]
            hi = H[n_re + 1] if n_re + 1 <= K else lo
            mag = lo + (frac / 0x7F) * (hi - lo)
        F_q = (wi >> 9) & 0xFFF
        mag_im = F_q / 0xFFF
        val = 0.5 * (mag + mag_im)
        if s_re:
            val = -val
        W_out[i] = val

    @cuda.jit
    def _forward_pass_kernel(x, W, cids, buffer_vals, symmetric,
                             H, K, d_in, d_out, y):
        """
        y[j] = UGC( Σ_i  W[i,j] · x[i] ).

        symmetric=1 → direct W (buffer bypassed)
        symmetric=0 → read buffer_vals[cids[i,j]]
        """
        j = cuda.grid(1)
        if j >= d_out:
            return
        s = 0.0
        for i in range(d_in):
            idx = i * d_out + j
            if symmetric == 1:
                w = W[idx]
            else:
                cid = cids[idx]
                w = buffer_vals[cid] if cid >= 0 else 0.0
            s += w * x[i]
        a = abs(s)
        n = int(min(K - 1, max(1, round(a))))
        log_val = -H[n]
        if s < 0:
            log_val = -abs(log_val)
        y[j] = log_val

    @cuda.jit
    def _zeta_eval_kernel(t_vals, C, alpha, K, results):
        """
        ζ(t) = Σ_{i=-K}^{K} C[i]·exp(i·t·i/α), take real part.
        One thread per t.  Direct cost O(K) per thread; total O(T·K)
        work, but O(K) wall-clock with T threads.
        """
        tid = cuda.grid(1)
        if tid >= t_vals.size:
            return
        t = t_vals[tid]
        total = 0.0
        for i in range(-K, K + 1):
            idx = i + K
            phase = t * i / alpha
            total += C[idx] * math.cos(phase)
        results[tid] = total


# ============================================================
#  GPU engine
# ============================================================
class GPUEngine:
    """
    GPU-side tables + encoding + forward + ζ(t).

    Falls back to NumPy if CUDA is unavailable.
    """
    def __init__(self, K: int = 4096, threads: int = 256):
        self.K = int(K)
        self.threads = int(threads)
        self.has_gpu = HAS_CUDA
        t0 = time.perf_counter()

        if self.has_gpu:
            self.mu_d, self.H_d, self.F_d = self._build_tables_gpu(K)
            self.mu = self.mu_d.copy_to_host()
            self.H  = self.H_d.copy_to_host()
            self.F  = self.F_d.copy_to_host()
        else:
            self.mu = mobius_sieve_cpu(K)
            self.H  = harmonic_cpu(K)
            self.F  = mu_conv_cpu(K, self.mu, self.H)

        self.F_abs = np.abs(self.F)
        self.F_abs_max = float(self.F_abs.max()) if self.F_abs.size else 1.0
        if self.has_gpu:
            self.F_abs_d = cuda.to_device(self.F_abs)
            self.bucket_n = self._build_bucket_lookup()
            self.bucket_n_d = cuda.to_device(self.bucket_n)

        self.t_tables_ms = (time.perf_counter() - t0) * 1e3

    # --------------------------------------------------------
    def _build_tables_gpu(self, K: int):
        n = K + 1
        gpus = cuda.gpus
        mu_d       = cuda.device_array(n, dtype=np.int8)
        is_prime_d = cuda.device_array(n, dtype=np.bool_)
        H_d        = cuda.to_device(harmonic_cpu(K))
        F_d        = cuda.device_array(n, dtype=np.float64)

        # μ
        mu_d.copy_to_device(np.ones(n, dtype=np.int8))
        # zero μ[0]
        mu_d[0] = 0
        blk = (n + self.threads - 1) // self.threads
        _sieve_primes_kernel[blk, self.threads](is_prime_d)
        _mu_toggle_primes_kernel[blk, self.threads](mu_d, is_prime_d)
        _mu_zero_squares_kernel[blk, self.threads](mu_d, is_prime_d)

        # F = (μ*H)
        F_d.copy_to_device(np.zeros(n, dtype=np.float64))
        _mobius_conv_kernel[blk, self.threads](mu_d, H_d, F_d, K)
        cuda.synchronize()
        return mu_d, H_d, F_d

    # --------------------------------------------------------
    def _build_bucket_lookup(self) -> np.ndarray:
        """For each |Im|-quantum q, nearest F-index n."""
        target_vals = np.linspace(0, self.F_abs_max, N_BUCKETS)
        buckets = np.zeros(N_BUCKETS, dtype=np.int32)
        for q, tgt in enumerate(target_vals):
            buckets[q] = int(np.argmin(np.abs(self.F_abs * ALPHA - tgt)))
        return buckets

    # --------------------------------------------------------
    def encode(self, values: np.ndarray
               ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (words_re, words_im) — 32 bits each, one per weight."""
        values = np.ascontiguousarray(values.astype(np.float64))
        n = values.size
        words_re_h = np.zeros(n, dtype=np.uint32)
        words_im_h = np.zeros(n, dtype=np.uint32)

        if self.has_gpu:
            values_d   = cuda.to_device(values)
            words_re_d = cuda.device_array(n, dtype=np.uint32)
            words_im_d = cuda.device_array(n, dtype=np.uint32)
            blk = (n + self.threads - 1) // self.threads
            _encode_real_kernel[blk, self.threads](
                values_d, self.H_d, self.K, words_re_d)
            _encode_imag_kernel[blk, self.threads](
                values_d, self.F_abs_d, self.F_abs_max,
                self.bucket_n_d, words_im_d)
            cuda.synchronize()
            words_re_d.copy_to_host(words_re_h)
            words_im_d.copy_to_host(words_im_h)
            return words_re_h, words_im_h

        # CPU fallback
        for i, v in enumerate(values):
            s_re = 1 if v < 0 else 0
            a = abs(v)
            if a <= 0:
                continue
            lo, hi = 1, self.K - 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if self.H[mid] <= a:
                    lo = mid
                else:
                    hi = mid - 1
            n_re = lo
            lo_h = self.H[n_re]
            hi_h = self.H[n_re + 1] if n_re + 1 <= self.K else lo_h
            frac = (a - lo_h) / (hi_h - lo_h) if hi_h > lo_h else 0.0
            frac_q = int(min(127, max(0, round(frac * 127.0))))
            words_re_h[i] = ((s_re & 1) << 31) | \
                            ((n_re & 0xFFFFFF) << 7) | frac_q
            q = int(min(N_BUCKETS - 1,
                        max(0, int(a / self.F_abs_max * N_BUCKETS))))
            n_im = self.bucket_n[q]
            F_q = int(min(0xFFF, max(0, round(a / self.F_abs_max * 0xFFF))))
            trace_q = int(min(0x1FF, max(0,
                    round(self.F_abs[n_im] / self.F_abs_max * 0x1FF))))
            words_im_h[i] = ((s_re & 1) << 31) | (F_q << 9) | trace_q
        return words_re_h, words_im_h

    # --------------------------------------------------------
    def decode(self, words_re: np.ndarray, words_im: np.ndarray
               ) -> np.ndarray:
        n = words_re.size
        out = np.zeros(n, dtype=np.float64)
        if self.has_gpu:
            wr_d = cuda.to_device(words_re.astype(np.uint32))
            wi_d = cuda.to_device(words_im.astype(np.uint32))
            out_d = cuda.device_array(n, dtype=np.float64)
            blk = (n + self.threads - 1) // self.threads
            _decode_weights_kernel[blk, self.threads](
                wr_d, wi_d, self.H_d, self.K, out_d)
            cuda.synchronize()
            out_d.copy_to_host(out)
            return out
        for i in range(n):
            wr = int(words_re[i]); wi = int(words_im[i])
            s_re = (wr >> 31) & 1
            n_re = (wr >> 7) & 0xFFFFFF
            frac = wr & 0x7F
            if n_re <= 0:
                mag = 0.0
            else:
                n_re = min(n_re, self.K - 1)
                lo = self.H[n_re]
                hi = self.H[n_re + 1] if n_re + 1 <= self.K else lo
                mag = lo + (frac / 0x7F) * (hi - lo)
            F_q = (wi >> 9) & 0xFFF
            mag_im = F_q / 0xFFF
            val = 0.5 * (mag + mag_im)
            out[i] = -val if s_re else val
        return out

    # --------------------------------------------------------
    def forward(self, x: np.ndarray, W: np.ndarray,
                cids: np.ndarray, buffer_vals: np.ndarray,
                symmetric: bool) -> np.ndarray:
        d_in, d_out = W.shape
        y = np.zeros(d_out, dtype=np.float64)
        if self.has_gpu:
            x_d  = cuda.to_device(x.astype(np.float64))
            W_d  = cuda.to_device(W.astype(np.float64).ravel())
            cids_d = cuda.to_device(cids.astype(np.int32).ravel())
            buf_d  = cuda.to_device(buffer_vals.astype(np.float64))
            y_d = cuda.device_array(d_out, dtype=np.float64)
            blk = (d_out + self.threads - 1) // self.threads
            _forward_pass_kernel[blk, self.threads](
                x_d, W_d, cids_d, buf_d, 1 if symmetric else 0,
                self.H_d, self.K, d_in, d_out, y_d)
            cuda.synchronize()
            y_d.copy_to_host(y)
            return y
        for j in range(d_out):
            s = 0.0
            for i in range(d_in):
                if symmetric:
                    w = W[i, j]
                else:
                    cid = int(cids[i, j])
                    w = buffer_vals[cid] if cid >= 0 else 0.0
                s += w * x[i]
            n = int(min(self.K - 1, max(1, round(abs(s)))))
            log_val = -self.H[n]
            y[j] = -abs(log_val) if s < 0 else log_val
        return y

    # --------------------------------------------------------
    def zeta(self, t_vals: np.ndarray, C: np.ndarray) -> np.ndarray:
        """
        ζ(t) = Σ_{i=-K}^{K} C[i]·exp(i·t·i/α), take real part.

        This is the linear equation with complex parts that grows
        super-polynomially with T·K.  GPU: O(K) wall-clock with T
        threads (each thread does the O(K) inner sum).
        """
        t_vals = np.ascontiguousarray(t_vals.astype(np.float64))
        C = np.ascontiguousarray(C.astype(np.float64))
        if C.size != 2 * self.K + 1:
            raise ValueError(f"C must have length {2*self.K+1}, "
                             f"got {C.size}")
        out = np.zeros(t_vals.size, dtype=np.float64)
        if self.has_gpu:
            t_d = cuda.to_device(t_vals)
            C_d = cuda.to_device(C)
            out_d = cuda.device_array(t_vals.size, dtype=np.float64)
            blk = (t_vals.size + self.threads - 1) // self.threads
            _zeta_eval_kernel[blk, self.threads](
                t_d, C_d, ALPHA, self.K, out_d)
            cuda.synchronize()
            out_d.copy_to_host(out)
            return out
        for k, t in enumerate(t_vals):
            total = 0.0
            for i in range(-self.K, self.K + 1):
                total += C[i + self.K] * math.cos(t * i / ALPHA)
            out[k] = total
        return out


# ============================================================
#  Split-channel layer  (asymmetric → MOSB, symmetric → direct)
# ============================================================
@dataclass
class Layer:
    W: np.ndarray                      # (d_in, d_out)
    words_re: np.ndarray               # (d_in*d_out,)
    words_im: np.ndarray
    W_decoded: np.ndarray              # (d_in, d_out)
    cids: np.ndarray                   # (d_in, d_out)
    symmetric: bool


class GPUReasoningNet:
    """
    Symmetric layer  → bypass buffer, solve directly
    Asymmetric layer → route weights through MOSB buffer (ids only on GPU)
    """
    def __init__(self, dims: List[int], K: int = 4096, seed: int = 0,
                 symmetric_layers: Optional[List[int]] = None):
        self.dims = list(dims)
        self.n_layers = len(dims) - 1
        self.engine = GPUEngine(K=K)
        self.symmetric_layers = symmetric_layers or []
        self.layers: List[Layer] = []
        self.buffer_vals: List[float] = []   # flat float buffer, cid → value
        self._build(seed)

    # --------------------------------------------------------
    def _build(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        for l in range(self.n_layers):
            d_in, d_out = self.dims[l], self.dims[l + 1]
            W = (rng.standard_normal((d_in, d_out)) * 0.5).astype(np.float64)

            sym = (l in self.symmetric_layers) and d_in == d_out
            if sym:
                W = 0.5 * (W + W.T)

            words_re, words_im = self.engine.encode(W.ravel())
            W_dec = self.engine.decode(words_re, words_im).reshape(d_in, d_out)

            cids = np.full((d_in, d_out), -1, dtype=np.int32)
            if not sym:
                for i in range(d_in):
                    for j in range(d_out):
                        cid = len(self.buffer_vals)
                        self.buffer_vals.append(float(W_dec[i, j]))
                        cids[i, j] = cid

            self.layers.append(Layer(
                W=W, words_re=words_re, words_im=words_im,
                W_decoded=W_dec, cids=cids, symmetric=sym))

        self.buffer_arr = np.array(self.buffer_vals, dtype=np.float64)

    # --------------------------------------------------------
    def forward(self, x: np.ndarray, verbose: bool = False) -> Dict:
        acts = [np.asarray(x, dtype=np.float64)]
        layer_info: List[Dict] = []
        for l, layer in enumerate(self.layers):
            path = "direct" if layer.symmetric else "buffer"
            if verbose:
                print(f"  layer {l}: {layer.W.shape[0]:>3d}→"
                      f"{layer.W.shape[1]:<3d}  "
                      f"symmetric={layer.symmetric}  path={path}")
            y = self.engine.forward(
                acts[-1], layer.W_decoded, layer.cids,
                self.buffer_arr, layer.symmetric)
            acts.append(y)
            layer_info.append(dict(layer=l, path=path,
                                   symmetric=layer.symmetric,
                                   n_weights=layer.W.size))
        return dict(activations=acts, layers=layer_info)

    # --------------------------------------------------------
    def solve(self, x: np.ndarray) -> np.ndarray:
        if not all(l.symmetric for l in self.layers):
            raise RuntimeError("solve() requires all layers symmetric")
        return self.forward(x)["activations"][-1]

    # --------------------------------------------------------
    def summary(self) -> Dict:
        n_params = sum(l.W.size for l in self.layers)
        n_bytes_packed = 8 * n_params           # two 32-bit words
        n_bytes_f32    = 4 * n_params
        return dict(
            dims=self.dims, n_layers=self.n_layers,
            n_params=n_params,
            n_bytes_packed=n_bytes_packed,
            n_bytes_float32=n_bytes_f32,
            compression=n_bytes_f32 / max(n_bytes_packed, 1),
            symmetric_layers=[l.symmetric for l in self.layers],
            buffer_slots=len(self.buffer_vals),
            gpu=HAS_CUDA,
            tables_ms=self.engine.t_tables_ms,
        )


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 80)
    print("GPU UGC reasoning pipeline  ·  split-channel weights  ·  MOSB buffer")
    print("=" * 80)
    print(f"  α = 1/(π − e)     = {ALPHA:.6f}")
    print(f"  A_STEP            = {A_STEP:.6f}")
    print(f"  6/π²              = {DENSITY:.6f}")
    print(f"  block             = {BLOCK}×{BLOCK}, perms = {N_PERM}")
    print(f"  GPU available     = {HAS_CUDA}")
    print()

    K = 4096
    print(f"--- tables  ·  K = {K} ---")
    t0 = time.perf_counter()
    engine = GPUEngine(K=K)
    t_tables = (time.perf_counter() - t0) * 1e3
    print(f"  build (μ, H, F)   = {t_tables:.2f} ms")
    print(f"  μ non-zero        = "
          f"{np.count_nonzero(engine.mu[1:])} / {K}  "
          f"= {np.count_nonzero(engine.mu[1:])/K:.4f}  "
          f"(target 6/π² = {DENSITY:.4f})")
    print(f"  F_max             = {engine.F_abs_max:.4f}")
    print(f"  H_max             = {engine.H[K]:.4f}")
    print()

    # ---------- validation: GPU vs CPU F ----------
    print("--- validation  (GPU vs CPU reference) ---")
    F_cpu = mu_conv_cpu(K, mobius_sieve_cpu(K), harmonic_cpu(K))
    diff = float(np.max(np.abs(engine.F - F_cpu)))
    print(f"  max |F_gpu − F_cpu| = {diff:.2e}")
    print()

    # ---------- net + forward ----------
    print("--- net build  ·  dims = [8, 8, 8, 8]  ·  layer 0 symmetric ---")
    net = GPUReasoningNet(dims=[8, 8, 8, 8], K=K, seed=1,
                          symmetric_layers=[0])
    for l, layer in enumerate(net.layers):
        print(f"  layer {l}: {layer.W.shape[0]:>3d}→{layer.W.shape[1]:<3d}  "
              f"symmetric={str(layer.symmetric):5s}  "
              f"params={layer.W.size:>3d}  "
              f"packed={8*layer.W.size:>4d} B")
    print()

    print("--- symmetry rule ---")
    for l, layer in enumerate(net.layers):
        if layer.symmetric:
            print(f"  layer {l}: symmetric → BYPASS buffer, direct solve")
        else:
            print(f"  layer {l}: asymmetric → READ from MOSB buffer")
    print()

    x = np.array([1.0, -0.5, 0.3, 0.8, -1.2, 0.6, 0.1, -0.9])
    print("--- forward pass ---")
    res = net.forward(x, verbose=True)
    out = res["activations"][-1]
    print(f"  output = {np.array2string(out, precision=4)}")
    print()

    # ---------- solve path ----------
    print("--- solve path ---")
    n_sym = sum(1 for l in net.layers if l.symmetric)
    print(f"  symmetric layers = {n_sym} / {net.n_layers}")
    if n_sym == net.n_layers:
        y = net.solve(x)
        print(f"  solved output    = {np.array2string(y, precision=4)}")
    else:
        print("  buffer path required for asymmetric layers")
    print()

    # ---------- encode/decode round-trip ----------
    print("--- encode / decode round-trip (first layer) ---")
    l0 = net.layers[0]
    W_hat = l0.W_decoded
    rel_err = float(np.linalg.norm(l0.W - W_hat)
                    / (np.linalg.norm(l0.W) + 1e-12))
    print(f"  rel err on layer 0 = {rel_err:.4f}")
    print()

    # ---------- ζ(t) evaluation ----------
    print("--- ζ(t) = Σ C[i]·exp(i·t·i/α)  (super-polynomial direct) ---")
    T = 512
    t_vals = np.linspace(0, 20, T)
    C = np.ones(2 * K + 1, dtype=np.float64)   # symmetric, all-ones

    # GPU
    t0 = time.perf_counter()
    z_gpu = engine.zeta(t_vals, C)
    t_gpu = (time.perf_counter() - t0) * 1e3

    # CPU reference (only first 8 samples for timing comparison)
    t0 = time.perf_counter()
    z_cpu_head = np.array([
        sum(C[i + K] * math.cos(t * i / ALPHA)
            for i in range(-K, K + 1))
        for t in t_vals[:8]
    ])
    t_cpu = (time.perf_counter() - t0) * 1e3

    err = float(np.max(np.abs(z_gpu[:8] - z_cpu_head)))
    print(f"  T samples          = {T}")
    print(f"  K half-width       = {K}")
    print(f"  direct work        = T·(2K+1) = {T * (2 * K + 1):,}")
    print(f"  GPU time           = {t_gpu:.3f} ms  "
          f"({T} threads in parallel)")
    print(f"  CPU time (8 samp)  = {t_cpu:.2f} ms")
    print(f"  max |GPU − CPU|    = {err:.2e}")
    print(f"  ζ(0)               = {z_gpu[0]:.4f}  "
          f"(expect 2K+1 = {2*K+1})")
    print()

    # ---------- compressed summary ----------
    print("--- net summary ---")
    s = net.summary()
    print(f"  n params          = {s['n_params']}")
    print(f"  packed bytes      = {s['n_bytes_packed']}  (64 b / weight)")
    print(f"  float32 bytes     = {s['n_bytes_float32']}")
    print(f"  compression       = {s['compression']:.2f}x")
    print(f"  symmetric layers  = {sum(s['symmetric_layers'])} / {s['n_layers']}")
    print(f"  buffer slots      = {s['buffer_slots']}")
    print(f"  gpu               = {s['gpu']}")
    print()

    # ---------- complexity ----------
    print("--- complexity ---")
    print("  μ sieve            O(K)  threads,  Erato-style")
    print("  F = (μ*H)          O(K log K)  total work  ·  scatter-add")
    print("  encode_real        O(log K)  per weight   ·  binary search")
    print("  encode_imag        O(1)      per weight   ·  bucket lookup")
    print("  decode             O(1)      per weight")
    print("  forward pass       O(d_in·d_out)  per layer")
    print("  ζ(t)               O(K) per t  ·  T threads in parallel")
    print()
    print("  The ζ(t) evaluation is the 'linear equation with complex")
    print("  parts' whose direct form grows super-polynomially in T·K;")
    print("  the GPU collapses it to O(K) wall-clock with T threads.")
    print()
    print("Done.")


if __name__ == "__main__":
    demo()