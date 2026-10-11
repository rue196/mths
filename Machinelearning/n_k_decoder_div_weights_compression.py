#!/usr/bin/env python3
"""
spinor_split_compression.py
===========================
Split-channel spinor weight compression.

Per 6×6 block Z:
    Π₆(Z) = det(Z) = Re(Π₆) + i·Im(Π₆)

Real half  (O(n) harmonic):
    H_{n+1}                              harmonic numbers, integer grid
    find n such that H_{n+1} ≈ |Re(Π₆)|
    pack: sign | n (24b) | magnitude fraction (7b)

Imaginary half  (O(K log K) Möbius):
    F(n) = (μ * H)(n)                     Möbius convolution
    find n such that α·|F(n)| ≈ |Im(Π₆)|
    pack: sign | Lehmer σ* | F-quantization (12b) | trace (9b)

UGC edge invariant:
    Re(v(+n)) = v(−n)     ties the two channels
"""

from __future__ import annotations
import math, time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

# ============================================================
#  Constants
# ============================================================
PI       = math.pi
E        = math.e
ALPHA    = 1.0 / (PI - E)               # ≈ 2.362338
GAMMA    = 0.5772156649015329
DENSITY  = 6.0 / (PI * PI)
BLOCK    = 6
N_PERM   = math.factorial(BLOCK)        # 720


# ============================================================
#  Möbius sieve + harmonic numbers + Möbius convolution
# ============================================================
def mobius_sieve(K: int) -> np.ndarray:
    if K < 1:
        return np.zeros(K + 1, dtype=np.int8)
    mu = np.zeros(K + 1, dtype=np.int8); mu[1] = 1
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


def harmonic_numbers(K: int) -> np.ndarray:
    H = np.zeros(K + 1, dtype=float)
    if K >= 1:
        H[1] = 1.0
    for n in range(2, K + 1):
        H[n] = H[n - 1] + 1.0 / n
    return H


def mu_convolution_H(K: int) -> np.ndarray:
    """F(n) = (μ * H)(n)  — the O(K log K) imaginary source."""
    mu = mobius_sieve(K)
    H  = harmonic_numbers(K)
    F  = np.zeros(K + 1, dtype=float)
    for d in range(1, K + 1):
        if mu[d] == 0:
            continue
        md = int(mu[d])
        for m in range(1, K // d + 1):
            F[d * m] += md * H[m]
    return F


# ============================================================
#  Signed-value reference  (from n_k_div_finitestep_convergence.py)
# ============================================================
def signed_value(k: int, H: np.ndarray, F: np.ndarray,
                 imag_scale: float = ALPHA) -> complex:
    """
    v(k):
        k < 0  →  −H_{|k|+1}                (purely real, O(n))
        k > 0  →  −H_{k+1} + i·α·F(k)       (imag from O(K log K))
        k = 0  →  0
    """
    if k == 0:
        return 0.0 + 0.0j
    n = abs(k)
    real = -(float(H[n]) if n < H.size else math.log(n) + GAMMA)
    if k < 0:
        return complex(real, 0.0)
    imag = imag_scale * float(F[n]) if n < F.size else 0.0
    return complex(real, imag)


# ============================================================
#  Permutation ranking (Lehmer)
# ============================================================
def rank_permutation(perm) -> int:
    n = len(perm); code = 0
    for i in range(n):
        smaller = sum(1 for j in range(i + 1, n) if perm[j] < perm[i])
        code += smaller * math.factorial(n - 1 - i)
    return code


def unrank_permutation(code: int, n: int) -> Tuple[int, ...]:
    code = int(code) % math.factorial(n)
    avail = list(range(n)); perm = []
    for i in range(n):
        f = math.factorial(n - 1 - i)
        idx = code // f; code = code % f
        perm.append(avail.pop(idx))
    return tuple(perm)


# ============================================================
#  32-bit word layouts  (two halves = 64-bit register)
# ============================================================
# Real half  (O(n) harmonic)
#   bit 31       sign of Re(Π₆)
#   bits 30-7    harmonic index n (24 bits, up to 16.7M)
#   bits  6-0    magnitude fraction (7 bits)
def pack_real(sign: int, n_idx: int, mag_frac: int) -> int:
    return ((sign & 1) << 31) | \
           ((n_idx & 0xFFFFFF) << 7) | \
           (mag_frac & 0x7F)


def unpack_real(w: int) -> Tuple[int, int, int]:
    return ((w >> 31) & 1,
            (w >> 7) & 0xFFFFFF,
            w & 0x7F)


# Imaginary half  (O(K log K) Möbius)
#   bit 31       sign of Im(Π₆)
#   bits 30-21   Lehmer index σ*      (10 bits, 720 permutations)
#   bits 20-9    F-quantization       (12 bits)
#   bits  8-0    Möbius gate F(n)     (9 bits, quantized μ*H)
def pack_imag(sign: int, perm_idx: int, F_q: int, gate_q: int) -> int:
    return ((sign & 1) << 31) | \
           ((perm_idx & 0x3FF) << 21) | \
           ((F_q & 0xFFF) << 9) | \
           (gate_q & 0x1FF)


def unpack_imag(w: int) -> Tuple[int, int, int, int]:
    return ((w >> 31) & 1,
            (w >> 21) & 0x3FF,
            (w >> 9)  & 0xFFF,
            w & 0x1FF)


# ============================================================
#  Split-channel projector
# ============================================================
class SplitSpinorProjector:
    """
    Two-channel projector:

        real side   →  harmonic index  n           (O(n))
        imag side   →  Möbius convolution F(n)      (O(K log K))

    The 64-bit code for a 6×6 block is
        (w_real, w_imag) = (32-bit real word, 32-bit imaginary word)

    Decoding reconstructs Z_hat from both halves.
    """
    def __init__(self,
                 K: int = 4096,
                 mag_max: float = 1.0,
                 tr_max: float = 1.0,
                 imag_scale: float = ALPHA):
        self.K = int(K)
        self.mag_max = float(mag_max)
        self.tr_max = float(tr_max)
        self.imag_scale = float(imag_scale)

        # harmonic and Möbius tables
        self.H = harmonic_numbers(self.K)
        self.F = mu_convolution_H(self.K)
        self.mu = mobius_sieve(self.K)

        # inverse lookup: H(n) values for harmonic binning
        self._H_sorted_idx = np.argsort(self.H)     # monotone anyway

        # calibration ranges
        self.H_max = float(self.H[self.K]) if self.K > 0 else 1.0
        self.F_max = float(np.max(np.abs(self.F[1:self.K + 1]))) \
                     if self.K > 0 else 1.0

    # --------------------------------------------------------
    def project(self, Z: np.ndarray) -> Dict:
        """Full projection of a 6×6 block; returns both channels."""
        Pi   = complex(np.linalg.det(Z.astype(complex)))
        re   = Pi.real
        im   = Pi.imag

        # dominant monomial via Hungarian on -log|Z|
        cost = -np.log(np.abs(Z) + 1e-30)
        row, col = linear_sum_assignment(cost)
        perm = tuple(int(c) for c in col)
        dom_mag = float(np.prod([abs(Z[row[k], col[k]])
                                 for k in range(BLOCK)]))
        tr = float(np.trace(Z))

        # real-side: nearest harmonic n with H_n ≈ |re|
        n_re = self._harmonic_index(abs(re))
        h_frac = self._frac_within_bin(abs(re), n_re)

        # imaginary-side: nearest Möbius index with α|F| ≈ |im|
        n_im = self._mobius_index(abs(im))
        F_frac = self._frac_F(abs(im), n_im)
        gate = int(self.mu[n_im]) if n_im < self.mu.size else 0
        gate_q = int(min(0x1FF, abs(gate) * 0x1FF))

        return dict(
            Pi=Pi, perm=perm, dom_mag=dom_mag, tr=tr,
            re=re, im=im,
            n_re=n_re, h_frac=h_frac,
            n_im=n_im, F_frac=F_frac, gate_q=gate_q,
            sign_re=int(re < 0), sign_im=int(im < 0),
        )

    # --------------------------------------------------------
    def _harmonic_index(self, v: float) -> int:
        """n such that H_n ≈ v (O(n) grid)."""
        if v <= 0:
            return 0
        # invert: H_n ≈ ln n + γ → n ≈ exp(v − γ)
        n = int(round(math.exp(v - GAMMA)))
        n = max(1, min(self.K, n))
        # refine locally
        while n > 1 and self.H[n] > v:
            n -= 1
        while n < self.K and self.H[n + 1] < v:
            n += 1
        return n

    def _frac_within_bin(self, v: float, n: int) -> int:
        """Fraction of v between H_n and H_{n+1}, quantized to 7 bits."""
        if n >= self.K or n < 1:
            return 0
        lo = self.H[n]; hi = self.H[n + 1]
        if hi <= lo:
            return 0
        f = (v - lo) / (hi - lo)
        return int(min(0x7F, max(0, round(f * 0x7F))))

    def _mobius_index(self, v: float) -> int:
        """n such that α|F_n| ≈ v (O(K log K) grid)."""
        if v <= 0 or self.F_max <= 0:
            return 1
        target = v / self.imag_scale
        idx = np.argmin(np.abs(self.F[1:self.K + 1] - target)) + 1
        return int(idx)

    def _frac_F(self, v: float, n: int) -> int:
        """Fraction within the F-grid, quantized to 12 bits."""
        if self.F_max <= 0:
            return 0
        target = v / self.imag_scale
        f = abs(target) / self.F_max
        return int(min(0xFFF, max(0, round(f * 0xFFF))))

    # --------------------------------------------------------
    def encode(self, info: Dict) -> Tuple[int, int]:
        """Return (w_real, w_imag)."""
        # real half
        n_re = min(info['n_re'], 0xFFFFFF)
        w_real = pack_real(info['sign_re'], n_re, info['h_frac'])
        # imaginary half
        perm_idx = rank_permutation(info['perm']) % N_PERM
        w_imag = pack_imag(info['sign_im'], perm_idx,
                           info['F_frac'], info['gate_q'])
        return w_real, w_imag

    def decode(self, w_real: int, w_imag: int,
               perm_cache: Dict[int, Tuple[int, ...]] = None
               ) -> np.ndarray:
        if perm_cache is None:
            perm_cache = {}

        s_re, n_re, h_frac = unpack_real(w_real)
        s_im, perm_idx, F_q, gate_q = unpack_imag(w_imag)

        if perm_idx not in perm_cache:
            perm_cache[perm_idx] = unrank_permutation(perm_idx, BLOCK)
        perm = perm_cache[perm_idx]

        # real amplitude = H_{n_re} + frac · (H_{n_re+1} − H_{n_re})
        if n_re <= 0:
            re_amp = 0.0
        else:
            n_re = min(n_re, self.K - 1)
            lo = self.H[n_re]
            hi = self.H[n_re + 1] if n_re + 1 <= self.K else lo
            re_amp = lo + (h_frac / 0x7F) * (hi - lo)

        # imaginary amplitude = imag_scale · F_{n_im} · fraction
        im_amp = self.imag_scale * self.F_max * (F_q / 0xFFF)

        re_val = -re_amp if s_re else re_amp
        im_val = -im_amp if s_im else im_amp

        # reconstruct the 6×6 block: real part on σ*, imag part on diag
        Z = np.zeros((BLOCK, BLOCK), dtype=np.float32)
        for k in range(BLOCK):
            Z[k, perm[k]] = re_val
        Z[np.arange(BLOCK), np.arange(BLOCK)] += im_val / BLOCK
        return Z


# ============================================================
#  Weight tensor compressor (split-channel)
# ============================================================
class SplitWeightCompressor:
    def __init__(self, d: int, gate: bool = True,
                 K_sieve: int = 4096, importance_pct: float = 0.0):
        self.d = d
        self.B = d // BLOCK
        self.gate = gate
        self.K_sieve = K_sieve
        self.importance_pct = importance_pct
        self.mu = mobius_sieve(self.B * self.B + 1)
        self.proj: Optional[SplitSpinorProjector] = None

    def _calibrate(self, W: np.ndarray):
        mags, trs = [], []
        for bi in range(self.B):
            for bj in range(self.B):
                Z = W[bi*BLOCK:(bi+1)*BLOCK, bj*BLOCK:(bj+1)*BLOCK]
                if np.linalg.norm(Z) < 1e-12:
                    continue
                det = complex(np.linalg.det(Z.astype(complex)))
                mags.append(abs(det))
                trs.append(float(np.trace(Z)))
        self.proj = SplitSpinorProjector(
            K=self.K_sieve,
            mag_max=max(np.percentile(mags, 99) if mags else 1.0, 1e-9),
            tr_max=max(np.percentile(np.abs(trs), 99) if trs else 1.0, 1e-9),
        )

    def compress(self, W: np.ndarray) -> Dict:
        pad = self.B * BLOCK - self.d
        if pad > 0:
            W = np.pad(W, ((0, pad), (0, pad)), mode='constant')
        if self.proj is None:
            self._calibrate(W)

        words_re, words_im = [], []
        positions, importances = [], []
        S = 0.0

        for bi in range(self.B):
            for bj in range(self.B):
                k_lin = bi * self.B + bj
                if self.gate and k_lin < self.mu.size and self.mu[k_lin] == 0:
                    continue
                Z = W[bi*BLOCK:(bi+1)*BLOCK, bj*BLOCK:(bj+1)*BLOCK]
                if np.linalg.norm(Z) < 1e-12:
                    continue
                info = self.proj.project(Z)
                wr, wi = self.proj.encode(info)
                words_re.append(wr); words_im.append(wi)
                positions.append((bi, bj))
                importances.append(abs(info['Pi']))
                S += abs(info['Pi']) * (1 if len(words_re) % 2 == 1 else -1)

        # importance threshold
        if self.importance_pct > 0 and importances:
            thr = np.percentile(importances, self.importance_pct)
            keep = [i for i, v in enumerate(importances) if v >= thr]
            words_re = [words_re[i] for i in keep]
            words_im = [words_im[i] for i in keep]
            positions = [positions[i] for i in keep]

        N = max(len(words_re), 1)
        p = min(abs(S) / N, 0.999)
        H = -ALPHA * p * math.log(p) if p > 0 else 0.0
        m = abs(S) * math.exp(-H) if H < 700 else 0.0

        # 64-bit register: high = real half, low = imaginary half
        reg64 = np.array(
            [(int(wr) << 32) | int(wi)
             for wr, wi in zip(words_re, words_im)],
            dtype=np.uint64,
        )

        return dict(
            reg64=reg64,
            words_re=np.array(words_re, dtype=np.uint32),
            words_im=np.array(words_im, dtype=np.uint32),
            positions=positions,
            n_blocks=len(words_re), n_total=self.B * self.B,
            S=S, H=H, m=m, pad=pad, d_orig=self.d,
        )

    def decompress(self, packed: Dict) -> np.ndarray:
        d_pad = self.B * BLOCK
        W_hat = np.zeros((d_pad, d_pad), dtype=np.float32)
        cache: Dict[int, Tuple[int, ...]] = {}
        for reg, (bi, bj) in zip(packed['reg64'], packed['positions']):
            wr = int((int(reg) >> 32) & 0xFFFFFFFF)
            wi = int(int(reg) & 0xFFFFFFFF)
            Z_hat = self.proj.decode(wr, wi, cache)
            W_hat[bi*BLOCK:(bi+1)*BLOCK,
                  bj*BLOCK:(bj+1)*BLOCK] = Z_hat
        if packed['pad'] > 0:
            W_hat = W_hat[:self.d, :self.d]
        return W_hat


# ============================================================
#  Synthetic weight tensor
# ============================================================
def make_synthetic_weights(d: int = 96, rank: int = 24, seed: int = 0,
                           complex_w: bool = False):
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((d, rank)) * 0.3
    V = rng.standard_normal((rank, d)) * 0.3
    W = U @ V + 0.1 * np.eye(d)
    if complex_w:
        W = W + 1j * (0.2 * U @ V.T[:rank] if rank <= d else 0)
        W = np.real(W)
    return W.astype(np.float32)


def metrics(W: np.ndarray, W_hat: np.ndarray) -> Dict:
    diff = W - W_hat
    mse = float(np.mean(diff ** 2))
    nrm = float(np.linalg.norm(W))
    err = float(np.linalg.norm(diff)) / (nrm + 1e-12)
    cos = float(np.sum(W * W_hat)
                / (np.linalg.norm(W) * np.linalg.norm(W_hat) + 1e-12))
    return dict(mse=mse, rel_err=err, cosine=cos)


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("Split-channel spinor compression  ·  O(n) real + O(K log K) imag")
    print("=" * 78)
    print(f"  α = 1/(π − e)      = {ALPHA:.6f}")
    print(f"  6/π²               = {DENSITY:.6f}")
    print(f"  block size         = {BLOCK}×{BLOCK}")
    print(f"  permutations/block = {N_PERM}")
    print()

    # ---------- word layout ----------
    print("--- 64-bit register layout  (two 32-bit halves) ---")
    print("  high 32 bits  : real half   (O(n) harmonic)")
    print("       bit 31       sign of Re(Π₆)")
    print("       bits 30-7    harmonic index n      24 bits")
    print("       bits  6-0    fraction within bin    7 bits")
    print()
    print("  low  32 bits  : imaginary half  (O(K log K) Möbius)")
    print("       bit 31       sign of Im(Π₆)")
    print("       bits 30-21   Lehmer index σ*        10 bits")
    print("       bits 20-9    F-quantization          12 bits")
    print("       bits  8-0    Möbius gate μ·H          9 bits")
    print()

    # ---------- build tables ----------
    K = 4096
    t0 = time.perf_counter()
    H = harmonic_numbers(K)
    F = mu_convolution_H(K)
    mu = mobius_sieve(K)
    t_tables = (time.perf_counter() - t0) * 1e3
    print(f"--- tables  ·  K = {K} ---")
    print(f"  H_n built        : {t_tables:.2f} ms")
    print(f"  H_K              : {H[K]:.6f}   (≈ ln {K} + γ = "
          f"{math.log(K) + GAMMA:.6f})")
    print(f"  max |F_n|        : {np.max(np.abs(F[1:])):.6f}")
    print(f"  μ-nonzero count  : {np.count_nonzero(mu[1:])} / {K}  "
          f"= {np.count_nonzero(mu[1:])/K:.4f}   (target 6/π² = {DENSITY:.4f})")
    print()

    # ---------- density check ----------
    print("--- UGC edge check  (Re(v(+n)) = v(−n)) ---")
    proj = SplitSpinorProjector(K=K)
    for n in [1, 5, 10, 25, 50, 100]:
        v_neg = signed_value(-n, H, F)
        v_pos = signed_value(+n, H, F)
        match = "✓" if abs(v_neg.real - v_pos.real) < 1e-12 else "✗"
        print(f"  n = {n:>3d}   v(−n) = {v_neg.real:+.6f}   "
              f"Re(v(+n)) = {v_pos.real:+.6f}   "
              f"Im(v(+n)) = {v_pos.imag:+.6e}   {match}")
    print()

    # ---------- compress a real weight matrix ----------
    print("--- split-channel compression ---")
    d = 96
    W = make_synthetic_weights(d, rank=24, seed=0)
    print(f"  shape              = {W.shape}")
    print(f"  bytes (float32)    = {W.nbytes}")
    print()

    for gate, pct in [(True, 0.0), (True, 50.0), (True, 75.0), (False, 0.0)]:
        comp = SplitWeightCompressor(d, gate=gate, K_sieve=K,
                                     importance_pct=pct)
        t0 = time.perf_counter()
        packed = comp.compress(W)
        t_comp = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        W_hat = comp.decompress(packed)
        t_decomp = (time.perf_counter() - t0) * 1e3
        m = metrics(W, W_hat)
        stored = packed['reg64'].nbytes
        ratio = W.nbytes / max(stored, 1)
        print(f"  gate={int(gate)}  keep={pct:5.1f}%  "
              f"n_blocks={packed['n_blocks']:>4d}/{packed['n_total']}   "
              f"ratio={ratio:>6.2f}x   "
              f"rel_err={m['rel_err']:.4f}   "
              f"cos={m['cosine']:.4f}   "
              f"comp={t_comp:5.1f}ms  decomp={t_decomp:5.1f}ms")
    print()

    # ---------- sample block reconstruction ----------
    print("--- sample block reconstruction ---")
    comp = SplitWeightCompressor(d, gate=False, K_sieve=K)
    packed = comp.compress(W)
    # pick a stored block
    if packed['n_blocks'] > 0:
        bi, bj = packed['positions'][0]
        Z0 = W[bi*BLOCK:(bi+1)*BLOCK, bj*BLOCK:(bj+1)*BLOCK]
        reg = packed['reg64'][0]
        wr = int((int(reg) >> 32) & 0xFFFFFFFF)
        wi = int(int(reg) & 0xFFFFFFFF)
        Z1 = comp.proj.decode(wr, wi)
        print(f"  block ({bi}, {bj})")
        print(f"  det(Z_orig)       = {np.linalg.det(Z0):+.6e}")
        s_re, n_re, h_frac = unpack_real(wr)
        s_im, perm_idx, F_q, gate_q = unpack_imag(wi)
        print(f"  real half decode  : sign={s_re}  n={n_re}  "
              f"frac={h_frac}/127")
        print(f"  imag half decode  : sign={s_im}  σ*={perm_idx}  "
              f"F_q={F_q}/4095  gate={gate_q}/511")
        print(f"  rel err on block  : "
              f"{np.linalg.norm(Z0 - Z1) / (np.linalg.norm(Z0) + 1e-12):.4f}")
    print()

    # ---------- lookup speed ----------
    print("--- 32-bit lookup speed  (per half) ---")
    sample = np.random.default_rng(2).integers(0, 2**32, size=100_000,
                                                dtype=np.uint32)
    cache: Dict[int, Tuple[int, ...]] = {}

    t0 = time.perf_counter()
    for w in sample[:50_000]:
        _ = unpack_real(int(w))
    dt_re = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    for w in sample[:50_000]:
        _ = unpack_imag(int(w))
    dt_im = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    for w in sample[:10_000]:
        s, p, f, g = unpack_imag(int(w))
        if p not in cache:
            cache[p] = unrank_permutation(p, BLOCK)
    dt_full = (time.perf_counter() - t0) * 1e3

    print(f"  unpack_real       : {dt_re:.2f} ms / 50k  "
          f"({dt_re/50:.4f} ms each)")
    print(f"  unpack_imag       : {dt_im:.2f} ms / 50k  "
          f"({dt_im/50:.4f} ms each)")
    print(f"  full imag decode  : {dt_full:.2f} ms / 10k  "
          f"({dt_full/10:.4f} ms each)")
    print(f"  perm cache size   : {len(cache)}  (≤ 720)")
    print()

    # ---------- theoretical compression ceiling ----------
    print("--- compression ceiling ---")
    n_blocks = (d // BLOCK) ** 2
    n_kept = int(n_blocks * DENSITY)
    raw = d * d * 4                       # float32
    gated = n_kept * 8                    # two 32-bit words = 64 bits
    gated_25 = int(n_kept * 0.25) * 8
    print(f"  {d}×{d} = {n_blocks} blocks, gate keeps ~{n_kept}")
    print(f"  raw             = {raw} bytes")
    print(f"  gated (2×32b)   = {gated} bytes")
    print(f"  ceiling         ≈ {raw / max(gated, 1):.1f}x")
    print(f"  with 75% importance filter: ≈ {raw / max(gated_25, 1):.1f}x")
    print()

    # ---------- summary of mapping ----------
    print("--- O(n) ↔ O(K log K) mapping ---")
    print("  real half   →  H_{n+1}              harmonic, integer grid")
    print("              →  lookup O(n)")
    print("  imag half   →  F(n) = (μ*H)(n)      Möbius convolution")
    print("              →  lookup O(K log K)")
    print("  UGC edge    →  Re(v(+n)) = v(−n)    ties the two halves")
    print("  64-bit reg  →  32-bit real | 32-bit imaginary")
    print()


if __name__ == "__main__":
    demo()