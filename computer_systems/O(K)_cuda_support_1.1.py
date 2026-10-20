#!/usr/bin/env python3
"""
llm_sparse_real_pipeline.py
===========================

LLM weights on the sparse 3D grid + real muon/tau clock.
Supersedes O(K)_cuda_support.py (which used complex ζ(t)).

Complex → Real replacement
--------------------------
    ζ(t) = Σ C[i]·exp(i·t·i/α)         →  S_μ[k], S_τ[k]   prefix sums
    |ζ(t)|² time sampling (T·K work)    →  M[k] = S_μ·S_τ    mirror (O(K))
    UGC activation from H[n]            →  K_sum[n] / (|S_μ|+|S_τ|)
    ζ_eval_kernel (super-poly T·K)      →  prefix_sums kernel (O(K))

Sparse LLM tensor layout
------------------------
    W(d_in, d_out)
      → 6×6 block partition (B_in × B_out)
      → Möbius gate μ(k_lin) ≠ 0    density 6/π² ≈ 0.6079
      → 32-bit spinor word per block (Lehmer σ* | mag* | trace)
      → per-block 3D coords (X, Y, Z, hue) from SparseNegGrid3D

Forward (real arithmetic only)
------------------------------
    y[j] = clock_activation( Σ_i W_dec[i,j]·x[i] )

    clock_activation(s) = K_sum[n] / (|S_μ[n]| + |S_τ[n]|)
                          with n = round(|s|) clamped to [1, K]

Symmetry rule preserved
-----------------------
    symmetric layers  → bypass buffer, direct spinor decode
    asymmetric layers → route weights through MOSB buffer
"""

from __future__ import annotations
import math
import time
import importlib.util
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---- CUDA ----
try:
    from numba import cuda
    HAS_CUDA = cuda.is_available()
except Exception:
    HAS_CUDA = False


# ============================================================
#  Sibling imports
# ============================================================
from weights_compression import (
    SpinorProjector, WeightCompressor,
    pack_word, unpack_word, rank_permutation, unrank_permutation,
    mobius_sieve, BLOCK as WC_BLOCK, N_PERM as WC_N_PERM,
)

from sparse_3d_model_graphics import (
    SparseNegGrid3D, square_free_indices, m_trace_batch,
    A_of_k, A_4, DA_4, DC_A_4,
)


# ---- load OKRealClock (filename contains parentheses) ----
def _load_okclock():
    for fname in ["O(K)_conv_tau_muon.py", "OK_muon_tau_real_clock.py"]:
        try:
            spec = importlib.util.spec_from_file_location("okclock", fname)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod
        except Exception:
            continue
    return None

_ok = _load_okclock()

if _ok is not None:
    OKRealClock = _ok.OKRealClock
    spin_grade = _ok.spin_grade
    smooth_zeros = _ok.smooth_zeros
else:
    def spin_grade(J_C, delta):
        return float((-1.0) ** (2.0 * J_C + delta))
    def smooth_zeros(c, K):
        for i in range(-K, K+1):
            idx = i + K
            if i == 0 or c[idx] != 0:
                continue
            l, r = i-1, i+1
            while l >= -K and c[l+K] == 0: l -= 1
            while r <= K and c[r+K] == 0: r += 1
            if l < -K or r > K or r == l: continue
            dist = r - l
            c[idx] = (r-i)/dist * c[l+K] + (i-l)/dist * c[r+K]


# ============================================================
#  Constants
# ============================================================
PI       = math.pi
E        = math.e
ALPHA    = 1.0 / (PI - E)
A_STEP   = ALPHA / 0.3628
DENSITY  = 6.0 / (PI * PI)
BLOCK    = 6


# ============================================================
#  GPU kernels  ·  REAL arithmetic only  ·  NO complex ζ
# ============================================================
if HAS_CUDA:

    @cuda.jit
    def _clock_signal_kernel(S_mu, S_tau, c0, K_sum, M, K):
        """Real K_sum and mirror product."""
        k = cuda.grid(1)
        if k > K:
            return
        K_sum[k] = S_mu[k] + c0 + S_tau[k]
        M[k] = S_mu[k] * S_tau[k]

    @cuda.jit
    def _forward_pass_kernel(x, W, cids, buffer_vals, symmetric,
                              S_mu, S_tau, K_sum, K, d_in, d_out, y):
        """
        y[j] = clock_activation( Σ_i W[i,j]·x[i] ).

        Pure real: no exp(), no complex, no 720-term Levi-Civita.
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
        n = int(min(K, max(1, round(abs(s)))))
        num = K_sum[n]
        den = abs(S_mu[n]) + abs(S_tau[n]) + 1e-9
        val = num / den
        if s < 0.0:
            val = -val
        y[j] = val


# ============================================================
#  Real clock engine  ·  replaces GPUEngine's complex ζ
# ============================================================
class RealClockEngine:
    """
    Real-arithmetic clock engine.

        · no complex ζ(t) = Σ C[i]·exp(i·t·i/α)
        · prefix sums S_μ[k], S_τ[k] computed once (O(K))
        · K_sum[k] = S_μ[k] + c_0 + S_τ[k]   (real)
        · M[k] = S_μ[k]·S_τ[k]               (real)
        · activation reads K_sum, not exp()
    """

    def __init__(self, K: int = 64, threads: int = 256):
        self.K = int(K)
        self.threads = int(threads)
        self.has_gpu = HAS_CUDA

        # coefficients c_k with zeros smoothed
        mu = mobius_sieve(K)
        self.mu = mu
        c = np.zeros(2 * K + 1)
        for i in range(-K, K + 1):
            if i != 0:
                c[i + K] = float(mu[abs(i)])
        smooth_zeros(c, K)
        self.c = c

        # spin grades (σ_μ = +1, σ_τ = −1)
        self.sigma_mu = spin_grade(1.0, 0.0)
        self.sigma_tau = spin_grade(1.0, 1.0)

        # prefix sums  ·  real, O(K)
        self.S_mu = np.zeros(K + 1)
        self.S_tau = np.zeros(K + 1)
        for k in range(1, K + 1):
            sign = 1.0 if (k % 2 == 1) else -1.0
            self.S_mu[k] = self.S_mu[k-1] + self.sigma_mu * sign * c[K + k]
            self.S_tau[k] = self.S_tau[k-1] + self.sigma_tau * sign * c[K - k]
        self.c0 = float(c[K])
        self.K_sum = self.S_mu + self.c0 + self.S_tau
        self.M = self.S_mu * self.S_tau
        self.speed = np.abs(np.diff(self.M, prepend=0.0))

        # GPU upload
        if self.has_gpu:
            self.S_mu_d = cuda.to_device(self.S_mu)
            self.S_tau_d = cuda.to_device(self.S_tau)
            self.K_sum_d = cuda.to_device(self.K_sum)
            self.M_d = cuda.to_device(self.M)

    # --------------------------------------------------------
    def activate(self, s: float) -> float:
        n = int(min(self.K, max(1, round(abs(s)))))
        num = float(self.K_sum[n])
        den = abs(float(self.S_mu[n])) + abs(float(self.S_tau[n])) + 1e-9
        val = num / den
        return -val if s < 0 else val

    # --------------------------------------------------------
    def forward(self, x: np.ndarray, W: np.ndarray,
                cids: np.ndarray, buffer_vals: np.ndarray,
                symmetric: bool) -> np.ndarray:
        d_in, d_out = W.shape
        y = np.zeros(d_out, dtype=np.float64)

        if self.has_gpu:
            x_d = cuda.to_device(x.astype(np.float64))
            W_d = cuda.to_device(W.astype(np.float64).ravel())
            cids_d = cuda.to_device(cids.astype(np.int32).ravel())
            buf_d = cuda.to_device(buffer_vals.astype(np.float64))
            y_d = cuda.device_array(d_out, dtype=np.float64)
            blk = (d_out + self.threads - 1) // self.threads
            _forward_pass_kernel[blk, self.threads](
                x_d, W_d, cids_d, buf_d, 1 if symmetric else 0,
                self.S_mu_d, self.S_tau_d, self.K_sum_d, self.K,
                d_in, d_out, y_d)
            cuda.synchronize()
            y_d.copy_to_host(y)
            return y

        # CPU fallback
        for j in range(d_out):
            s = 0.0
            for i in range(d_in):
                if symmetric:
                    w = W[i, j]
                else:
                    cid = int(cids[i, j])
                    w = buffer_vals[cid] if cid >= 0 else 0.0
                s += w * x[i]
            y[j] = self.activate(s)
        return y

    # --------------------------------------------------------
    def clock_snapshot(self) -> Dict:
        ticks_elec = [0]
        ticks_ksum = [k for k in range(1, self.K + 1)
                      if abs(self.K_sum[k]) < 1e-9]
        return dict(
            K=self.K,
            S_mu_range=(float(self.S_mu.min()), float(self.S_mu.max())),
            S_tau_range=(float(self.S_tau.min()), float(self.S_tau.max())),
            K_sum_range=(float(self.K_sum.min()), float(self.K_sum.max())),
            M_range=(float(self.M.min()), float(self.M.max())),
            speed_mean=float(self.speed.mean()),
            speed_peak=float(self.speed.max()),
            ticks_elec=ticks_elec,
            ticks_ksum=ticks_ksum,
        )


# ============================================================
#  Layer  ·  two-stage storage
# ============================================================
@dataclass
class Layer:
    W: np.ndarray
    words: np.ndarray
    positions: List[Tuple[int, int]]
    W_decoded: np.ndarray
    cids: np.ndarray
    symmetric: bool
    coords: Optional[np.ndarray] = None


# ============================================================
#  SparseRealLLM  ·  top-level net
# ============================================================
class SparseRealLLM:
    def __init__(self, dims: List[int], K_llm: int = 12,
                 seed: int = 0,
                 symmetric_layers: Optional[List[int]] = None):
        self.dims = list(dims)
        self.n_layers = len(dims) - 1
        self.K_llm = int(K_llm)
        self.symmetric_layers = symmetric_layers or []

        self.engine = RealClockEngine(K=max(2 * K_llm, 32))

        # 3D grid meta
        self.grid3d = SparseNegGrid3D(K=K_llm)
        self.M_grid = int(self.grid3d.M)

        self.layers: List[Layer] = []
        self.buffer_vals: List[float] = []
        self._build(seed)

    # --------------------------------------------------------
    def _compress(self, W: np.ndarray):
        """Per-6×6 block spinor compression with Möbius gating."""
        d_in, d_out = W.shape
        pad_in = (-d_in) % BLOCK
        pad_out = (-d_out) % BLOCK
        W_pad = np.pad(W, ((0, pad_in), (0, pad_out)))
        B_in = W_pad.shape[0] // BLOCK
        B_out = W_pad.shape[1] // BLOCK

        from scipy.optimize import linear_sum_assignment
        mags, trs = [], []
        for bi in range(B_in):
            for bj in range(B_out):
                Z = W_pad[bi*BLOCK:(bi+1)*BLOCK, bj*BLOCK:(bj+1)*BLOCK]
                if np.linalg.norm(Z) < 1e-12:
                    continue
                cost = -np.log(np.abs(Z) + 1e-30)
                row, col = linear_sum_assignment(cost)
                dom = float(np.prod(
                    [abs(Z[row[k], col[k]]) for k in range(BLOCK)]))
                mags.append(dom)
                trs.append(float(np.trace(Z)))

        proj = SpinorProjector(
            mag_max=max(np.percentile(mags, 99) if mags else 1.0, 1e-9),
            tr_max=max(np.percentile(np.abs(trs), 99) if trs else 1.0, 1e-9),
        )

        mu = mobius_sieve(B_in * B_out + 1)
        words: List[int] = []
        positions: List[Tuple[int, int]] = []
        W_dec = np.zeros_like(W_pad)
        perm_cache: Dict = {}

        for bi in range(B_in):
            for bj in range(B_out):
                k_lin = bi * B_out + bj
                if k_lin < mu.size and mu[k_lin] == 0:
                    continue
                Z = W_pad[bi*BLOCK:(bi+1)*BLOCK, bj*BLOCK:(bj+1)*BLOCK]
                if np.linalg.norm(Z) < 1e-12:
                    continue
                info = proj.project(Z)
                word = proj.encode(info)
                words.append(word)
                positions.append((bi, bj))
                W_dec[bi*BLOCK:(bi+1)*BLOCK,
                      bj*BLOCK:(bj+1)*BLOCK] = proj.decode(word, perm_cache)

        W_dec = W_dec[:d_in, :d_out]
        return (np.array(words, dtype=np.uint32),
                positions, W_dec, proj)

    # --------------------------------------------------------
    def _block_coords(self, positions: List[Tuple[int, int]]):
        if not positions:
            return None
        M = self.M_grid
        ns = self.grid3d.ns
        Z = self.grid3d.Z
        hue = self.grid3d.hue

        out = np.zeros((len(positions), 4), dtype=np.float32)
        for k, (bi, bj) in enumerate(positions):
            i = bi % M
            j = bj % M
            out[k, 0] = -float(ns[i])
            out[k, 1] = -float(ns[j])
            out[k, 2] = float(Z[i, j])
            out[k, 3] = float(hue[i, j])
        return out

    # --------------------------------------------------------
    def _build(self, seed: int) -> None:
        rng = np.random.default_rng(seed)

        for l in range(self.n_layers):
            d_in, d_out = self.dims[l], self.dims[l + 1]
            W = (rng.standard_normal((d_in, d_out)) * 0.5).astype(np.float64)

            sym = (l in self.symmetric_layers) and d_in == d_out
            if sym:
                W = 0.5 * (W + W.T)

            words, positions, W_dec, _ = self._compress(W)
            coords = self._block_coords(positions)

            cids = np.full((d_in, d_out), -1, dtype=np.int32)
            if not sym:
                for i in range(d_in):
                    for j in range(d_out):
                        cid = len(self.buffer_vals)
                        self.buffer_vals.append(float(W_dec[i, j]))
                        cids[i, j] = cid

            self.layers.append(Layer(
                W=W, words=words, positions=positions,
                W_decoded=W_dec, cids=cids, symmetric=sym,
                coords=coords))

        self.buffer_arr = np.array(self.buffer_vals, dtype=np.float64)

    # --------------------------------------------------------
    def forward(self, x: np.ndarray) -> List[np.ndarray]:
        acts = [np.asarray(x, dtype=np.float64)]
        for l, layer in enumerate(self.layers):
            y = self.engine.forward(
                acts[-1], layer.W_decoded, layer.cids,
                self.buffer_arr, layer.symmetric)
            acts.append(y)
        return acts

    def solve(self, x: np.ndarray) -> np.ndarray:
        if not all(l.symmetric for l in self.layers):
            raise RuntimeError("solve() requires all layers symmetric")
        return self.forward(x)[-1]

    # --------------------------------------------------------
    def summary(self) -> Dict:
        n_params = sum(l.W.size for l in self.layers)
        n_bytes_f32 = sum(l.W.nbytes for l in self.layers)
        n_words = sum(len(l.words) for l in self.layers)
        n_bytes_packed = n_words * 4
        n_blocks = sum(
            (l.W.shape[0] // BLOCK) * (l.W.shape[1] // BLOCK)
            for l in self.layers)
        return dict(
            dims=self.dims, n_layers=self.n_layers,
            n_params=n_params,
            n_blocks=n_blocks, n_words=n_words,
            density=n_words / max(n_blocks, 1),
            n_bytes_f32=n_bytes_f32,
            n_bytes_packed=n_bytes_packed,
            compression=n_bytes_f32 / max(n_bytes_packed, 1),
            symmetric_layers=[l.symmetric for l in self.layers],
            buffer_slots=len(self.buffer_vals),
            K_llm=self.K_llm,
            M_grid=self.M_grid,
            gpu=HAS_CUDA,
        )


# ============================================================
#  Reference forwards
# ============================================================
def reference_forward(net: SparseRealLLM, x: np.ndarray) -> np.ndarray:
    """Float32 tanh reference."""
    acts = x.copy()
    for l in net.layers:
        acts = np.tanh(l.W.T @ acts)
    return acts


def real_only_forward(net: SparseRealLLM, x: np.ndarray) -> np.ndarray:
    """Same real clock, pure float, for consistency check."""
    acts = x.copy()
    for l in net.layers:
        pre = l.W_decoded.T @ acts
        acts = np.array([net.engine.activate(s) for s in pre])
    return acts


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 80)
    print("LLM sparse real pipeline  ·  sparse 3D grid + real muon/tau clock")
    print("=" * 80)
    print(f"  α      = 1/(π − e)   = {ALPHA:.6f}")
    print(f"  a_step = α/0.3628    = {A_STEP:.6f}")
    print(f"  6/π²                 = {DENSITY:.6f}")
    print(f"  A_4 / DA_4 / DC_A_4  = {A_4:.4f} / {DA_4:.4f} / {DC_A_4:.4f}")
    print(f"  GPU available        = {HAS_CUDA}")
    print(f"  replacing            O(K)_cuda_support.py  (complex ζ)")
    print(f"  with                 O(K)_conv_tau_muon.py (real prefix sums)")
    print()

    # ---- configure dims from sparse 3D grid ----
    K_llm = 12
    ns = square_free_indices(K_llm)
    M = len(ns)
    d = 6 * M
    print(f"--- sparse 3D grid  ·  K_llm = {K_llm} ---")
    print(f"  square-free M        = {M}")
    print(f"  ns                   = {ns.tolist()}")
    print(f"  d = 6·M              = {d}")
    print(f"  density per axis     = {M/K_llm:.4f}  "
          f"(6/π² = {DENSITY:.4f})")
    print()

    # ---- build net ----
    dims = [d, d, d]
    net = SparseRealLLM(dims=dims, K_llm=K_llm, seed=1,
                        symmetric_layers=[0])
    print(f"--- net build  ·  dims = {dims} ---")
    for l, layer in enumerate(net.layers):
        n_blk = (layer.W.shape[0] // BLOCK) * (layer.W.shape[1] // BLOCK)
        print(f"  layer {l}: {layer.W.shape[0]}→{layer.W.shape[1]}  "
              f"symmetric={str(layer.symmetric):5s}  "
              f"stored={len(layer.words):>3d}/{n_blk}  "
              f"path={'direct' if layer.symmetric else 'buffer'}")
    print()

    # ---- sample block 3D coords ----
    print(f"--- sample block 3D positions (layer 0) ---")
    print(f"  {'block':>8}  {'x':>6}  {'y':>6}  {'z':>9}  {'hue':>8}")
    for k in range(min(5, len(net.layers[0].positions))):
        c = net.layers[0].coords[k]
        bi, bj = net.layers[0].positions[k]
        print(f"  ({bi:>2},{bj:>2})  {c[0]:>6.1f}  {c[1]:>6.1f}  "
              f"{c[2]:>+9.4f}  {c[3]:>8.4f}")
    print()

    # ---- real clock snapshot ----
    snap = net.engine.clock_snapshot()
    print(f"--- real clock  ·  K = {snap['K']} ---")
    print(f"  S_μ range       = [{snap['S_mu_range'][0]:+.4f}, "
          f"{snap['S_mu_range'][1]:+.4f}]")
    print(f"  S_τ range       = [{snap['S_tau_range'][0]:+.4f}, "
          f"{snap['S_tau_range'][1]:+.4f}]")
    print(f"  K_sum range     = [{snap['K_sum_range'][0]:+.4f}, "
          f"{snap['K_sum_range'][1]:+.4f}]")
    print(f"  M range         = [{snap['M_range'][0]:+.4f}, "
          f"{snap['M_range'][1]:+.4f}]")
    print(f"  speed mean/peak = {snap['speed_mean']:.4f} / "
          f"{snap['speed_peak']:.4f}")
    print(f"  electron ticks  = {snap['ticks_elec']}")
    print(f"  K_sum ticks     = {snap['ticks_ksum']}")
    print()

    # ---- forward ----
    rng = np.random.default_rng(42)
    x = rng.standard_normal(d) * 0.5

    print(f"--- forward pass ---")
    t0 = time.perf_counter()
    acts = net.forward(x)
    t_forward = (time.perf_counter() - t0) * 1e3
    y = acts[-1]
    print(f"  input |x|         = {np.linalg.norm(x):.4f}")
    print(f"  output |y|        = {np.linalg.norm(y):.4f}")
    print(f"  output range      = [{y.min():+.4f}, {y.max():+.4f}]")
    print(f"  forward time      = {t_forward:.3f} ms")
    print()

    # ---- consistency check ----
    y_cmp = real_only_forward(net, x)
    rel_err = float(np.linalg.norm(y - y_cmp)
                    / (np.linalg.norm(y_cmp) + 1e-12))
    cos_sim = float(np.dot(y, y_cmp)
                    / (np.linalg.norm(y) * np.linalg.norm(y_cmp) + 1e-12))
    print(f"--- consistency check (decoded weights, same activation) ---")
    print(f"  rel err           = {rel_err:.4e}")
    print(f"  cosine similarity = {cos_sim:.6f}")
    print()

    # ---- real-only verification ----
    print(f"--- real-arithmetic verification ---")
    dtypes = set()
    for l in net.layers:
        dtypes.add(str(l.W_decoded.dtype))
        dtypes.add(str(l.words.dtype))
    dtypes.add(str(net.buffer_arr.dtype))
    complex_present = any("complex" in s for s in dtypes)
    print(f"  dtypes            = {sorted(dtypes)}")
    print(f"  complex dtype     = {complex_present}")
    print(f"  ζ(t) complex eval = NONE (removed)")
    print(f"  real prefix sums  = YES")
    print(f"  exp(i·t·i/α)      = 0 calls")
    print()

    # ---- benchmark ----
    print(f"--- benchmark  ·  20 forwards ---")
    t0 = time.perf_counter()
    for _ in range(20):
        net.forward(x)
    t_batch = (time.perf_counter() - t0) * 1e3
    print(f"  total             = {t_batch:.2f} ms")
    print(f"  per forward       = {t_batch/20:.3f} ms")
    print()

    # ---- net summary ----
    s = net.summary()
    print(f"--- net summary ---")
    print(f"  dims              = {s['dims']}")
    print(f"  n params          = {s['n_params']:,}")
    print(f"  n blocks          = {s['n_blocks']:,}")
    print(f"  n words stored    = {s['n_words']:,}")
    print(f"  block density     = {s['density']:.4f}  "
          f"(target 6/π² = {DENSITY:.4f})")
    print(f"  bytes float32     = {s['n_bytes_f32']:,}")
    print(f"  bytes packed      = {s['n_bytes_packed']:,}")
    print(f"  compression       = {s['compression']:.2f}x")
    print(f"  symmetric layers  = "
          f"{sum(s['symmetric_layers'])} / {s['n_layers']}")
    print(f"  buffer slots      = {s['buffer_slots']}")
    print(f"  gpu               = {s['gpu']}")
    print()

    # ---- complex → real diff ----
    print(f"--- complex → real replacement ---")
    print(f"  {'OLD (complex ζ)':40s}  {'NEW (real clock)':26s}")
    print(f"  {'-'*40}  {'-'*26}")
    print(f"  {'ζ(t) = Σ C[i]·exp(i·t·i/α)':40s}  "
          f"{'S_μ[k], S_τ[k] prefix':26s}")
    print(f"  {'|ζ|² sampled at T times':40s}  "
          f"{'K_sum[k], M[k]':26s}")
    print(f"  {'UGC activation from H[n]':40s}  "
          f"{'K_sum[n] / (|S_μ|+|S_τ|)':26s}")
    print(f"  {'ζ_eval_kernel: T·(2K+1) work':40s}  "
          f"{'prefix_sums: O(K)':26s}")
    print(f"  {'complex dtype required':40s}  "
          f"{'real dtype only':26s}")
    print(f"  {'super-poly in T·K':40s}  "
          f"{'linear in K':26s}")
    print()

    # ---- complexity ----
    print(f"--- complexity ---")
    print(f"  μ sieve              O(K)                     once")
    print(f"  real prefix sums     O(K)                     once")
    print(f"  K_sum, M             O(K)                     once")
    print(f"  spinor compress      O(B² · 720)              once")
    print(f"  spinor decode        O(B² · 720)              once")
    print(f"  forward (CPU)        O(d_in · d_out)          per token")
    print(f"  forward (GPU)        O(d_in·d_out / 256)      wall-clock")
    print(f"  clock activation     O(1)                     per output")
    print(f"  ──────────────────────────────────────────────")
    print(f"  forward is real O(d_in·d_out).")
    print(f"  No complex exponentials anywhere.")
    print()
    print("Done.")


if __name__ == "__main__":
    demo()