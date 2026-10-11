#!/usr/bin/env python3
"""
spinor_weight_compression.py
============================
Compress open weights via the rank-6 Levi-Civita spinor projection.

Per 6×6 block Z:
    Π₆(Z) = det(Z)                     the scalar of the projector
    σ*    = argmax |Π Z[k,σ(k)]|       the dominant monomial
    mag*  = |Π Z[k,σ*(k)]|             its coefficient
    tr    = trace(Z)                   the diagonal correction

32-bit packed word:
    bit 31       sign of Π₆                    1
    bits 30-21   Lehmer index of σ*            10  (720 permutations)
    bits 20-9    quantized mag*                12
    bits  8-0    quantized trace (signed)       9

Sparse attention: only active 6×6 query-key blocks are stored,
Möbius-gated by μ(k) ≠ 0 at linear block index k.
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
ALPHA    = 1.0 / (PI - E)               # ≈ 2.362
DENSITY  = 6.0 / (PI * PI)              # ≈ 0.6079
BLOCK    = 6
N_PERM   = math.factorial(BLOCK)        # 720


# ============================================================
#  Möbius sieve  O(K)
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


# ============================================================
#  Permutation ranking  (Lehmer code)
# ============================================================
def rank_permutation(perm) -> int:
    n = len(perm); code = 0
    for i in range(n):
        smaller = sum(1 for j in range(i + 1, n) if perm[j] < perm[i])
        code += smaller * math.factorial(n - 1 - i)
    return code


def unrank_permutation(code: int, n: int) -> Tuple[int, ...]:
    code = int(code) % math.factorial(n)      # ← clamp to [0, n!−1]
    avail = list(range(n)); perm = []
    for i in range(n):
        f = math.factorial(n - 1 - i)
        idx = code // f; code = code % f
        perm.append(avail.pop(idx))
    return tuple(perm)

# ============================================================
#  32-bit word packing  (1 + 10 + 12 + 9 = 32)
# ============================================================
def pack_word(sign: int, perm_idx: int, mag_q: int, tr_q: int) -> int:
    return ((sign & 1) << 31) | \
           ((perm_idx & 0x3FF) << 21) | \
           ((mag_q & 0xFFF) << 9) | \
           (tr_q & 0x1FF)


def unpack_word(w: int) -> Tuple[int, int, int, int]:
    return ((w >> 31) & 1,
            (w >> 21) & 0x3FF,
            (w >> 9)  & 0xFFF,
            w & 0x1FF)


# ============================================================
#  Spinor projector  ·  per 6×6 block
# ============================================================
class SpinorProjector:
    """
    For a 6×6 real block Z:
        Pi       = det(Z)
        perm     = dominant monomial assignment
        dom_mag  = |Pi Z[k, perm[k]]|
        tr       = trace(Z)

    Reconstruction:
        Z_hat[k, perm[k]] = sign * dom_mag
        Z_hat[k, k]      += trace / 6
    """
    def __init__(self, mag_max: float = 1.0, tr_max: float = 1.0):
        self.mag_max = float(mag_max)
        self.tr_max  = float(tr_max)

    def project(self, Z: np.ndarray) -> Dict:
        Pi = float(np.linalg.det(Z))
        # Hungarian assignment on -log|Z| gives the dominant monomial
        cost = -np.log(np.abs(Z) + 1e-30)
        row, col = linear_sum_assignment(cost)
        dom_mag = float(np.prod([abs(Z[row[k], col[k]])
                                 for k in range(BLOCK)]))
        tr = float(np.trace(Z))
        importance = abs(Pi) * (1.0 + math.log1p(float(np.linalg.norm(Z))))
        return dict(
            Pi=Pi, perm=tuple(int(c) for c in col),
            dom_mag=dom_mag, tr=tr, importance=importance,
            sign=int(Pi < 0),
        )

    def encode(self, info: Dict) -> int:
        perm_idx = rank_permutation(info['perm'])
        mag_q = int(min(0xFFF, max(0,
                    round(info['dom_mag'] / self.mag_max * 0xFFF))))
        tr_q  = int(min(0x1FF, max(0,
                    round((info['tr'] / self.tr_max * 0.5 + 0.5) * 0x1FF))))
        return pack_word(info['sign'], perm_idx, mag_q, tr_q)

    def decode(self, w: int, perm_cache: dict) -> np.ndarray:
        sign, perm_idx, mag_q, tr_q = unpack_word(w)
        if perm_idx not in perm_cache:
            perm_cache[perm_idx] = unrank_permutation(perm_idx, BLOCK)
        perm = perm_cache[perm_idx]
        mag = mag_q / 0xFFF * self.mag_max
        tr  = (tr_q / 0x1FF * 2.0 - 1.0) * self.tr_max
        Z = np.zeros((BLOCK, BLOCK), dtype=np.float32)
        val = -mag if sign else mag
        for k in range(BLOCK):
            Z[k, perm[k]] = val
        Z[np.arange(BLOCK), np.arange(BLOCK)] += tr / BLOCK
        return Z


# ============================================================
#  Weight tensor compressor
# ============================================================
class WeightCompressor:
    def __init__(self, d: int, gate: bool = True,
                 importance_pct: float = 0.0):
        self.d = d
        self.B = d // BLOCK
        self.gate = gate
        self.importance_pct = importance_pct
        self.mu = mobius_sieve(self.B * self.B + 1)
        # calibrate magnitude scales on a first pass
        self.proj: Optional[SpinorProjector] = None

    def _calibrate(self, W: np.ndarray):
        mags, trs = [], []
        for bi in range(self.B):
            for bj in range(self.B):
                Z = W[bi*BLOCK:(bi+1)*BLOCK, bj*BLOCK:(bj+1)*BLOCK]
                if np.linalg.norm(Z) < 1e-12:
                    continue
                cost = -np.log(np.abs(Z) + 1e-30)
                row, col = linear_sum_assignment(cost)
                dom = float(np.prod([abs(Z[row[k], col[k]])
                                     for k in range(BLOCK)]))
                mags.append(dom); trs.append(float(np.trace(Z)))
        self.proj = SpinorProjector(
            mag_max=max(np.percentile(mags, 99) if mags else 1.0, 1e-9),
            tr_max =max(np.percentile(np.abs(trs), 99) if trs else 1.0, 1e-9),
        )

    def compress(self, W: np.ndarray) -> Dict:
        pad = self.B * BLOCK - self.d
        if pad > 0:
            W = np.pad(W, ((0, pad), (0, pad)), mode='constant')
        if self.proj is None:
            self._calibrate(W)

        words, positions, importances = [], [], []
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
                words.append(self.proj.encode(info))
                positions.append((bi, bj))
                importances.append(info['importance'])
                S += info['importance'] * (1 if len(words) % 2 == 1 else -1)

        # importance thresholding (percentile)
        if self.importance_pct > 0 and importances:
            thr = np.percentile(importances, self.importance_pct)
            keep = [i for i, v in enumerate(importances) if v >= thr]
            words     = [words[i] for i in keep]
            positions = [positions[i] for i in keep]

        N = max(len(words), 1)
        p = min(abs(S) / N, 0.999)
        H = -ALPHA * p * math.log(p) if p > 0 else 0.0
        m = abs(S) * math.exp(-H) if H < 700 else 0.0

        return dict(
            words=np.array(words, dtype=np.uint32),
            positions=positions,
            n_words=len(words),
            n_blocks=self.B * self.B,
            S=S, H=H, m=m, pad=pad,
            d_orig=self.d,
        )

    def decompress(self, packed: Dict) -> np.ndarray:
        d_pad = self.B * BLOCK
        W_hat = np.zeros((d_pad, d_pad), dtype=np.float32)
        perm_cache: Dict[int, Tuple[int, ...]] = {}
        for w, (bi, bj) in zip(packed['words'], packed['positions']):
            Z_hat = self.proj.decode(int(w), perm_cache)
            W_hat[bi*BLOCK:(bi+1)*BLOCK,
                  bj*BLOCK:(bj+1)*BLOCK] = Z_hat
        if packed['pad'] > 0:
            W_hat = W_hat[:self.d, :self.d]
        return W_hat


# ============================================================
#  Sparse attention compression
# ============================================================
def local_window_mask(d: int, window: int) -> np.ndarray:
    i = np.arange(d)[:, None]; j = np.arange(d)[None, :]
    return (np.abs(i - j) <= window).astype(np.float32)


def strided_mask(d: int, stride: int) -> np.ndarray:
    i = np.arange(d)[:, None]; j = np.arange(d)[None, :]
    return ((i - j) % stride == 0).astype(np.float32)


class SparseAttentionCompressor:
    """
    Sparse attention: only 6×6 query-key blocks in the pattern are
    stored, Möbius-gated by μ(k) ≠ 0 on the linear block index.
    """
    def __init__(self, d: int):
        self.d = d
        self.B = d // BLOCK
        self.mu = mobius_sieve(self.B * self.B + 1)
        self.proj: Optional[SpinorProjector] = None

    def _calibrate(self, A: np.ndarray, pattern: np.ndarray):
        mags, trs = [], []
        for bi in range(self.B):
            for bj in range(self.B):
                if pattern[bi*BLOCK:(bi+1)*BLOCK,
                           bj*BLOCK:(bj+1)*BLOCK].max() < 0.5:
                    continue
                Z = A[bi*BLOCK:(bi+1)*BLOCK, bj*BLOCK:(bj+1)*BLOCK]
                if np.linalg.norm(Z) < 1e-12:
                    continue
                cost = -np.log(np.abs(Z) + 1e-30)
                row, col = linear_sum_assignment(cost)
                dom = float(np.prod([abs(Z[row[k], col[k]])
                                     for k in range(BLOCK)]))
                mags.append(dom); trs.append(float(np.trace(Z)))
        self.proj = SpinorProjector(
            mag_max=max(np.percentile(mags, 99) if mags else 1.0, 1e-9),
            tr_max =max(np.percentile(np.abs(trs), 99) if trs else 1.0, 1e-9),
        )

    def compress(self, A: np.ndarray, pattern: np.ndarray) -> Dict:
        if self.proj is None:
            self._calibrate(A, pattern)
        words, positions = [], []
        active = 0
        S = 0.0
        for bi in range(self.B):
            for bj in range(self.B):
                if pattern[bi*BLOCK:(bi+1)*BLOCK,
                           bj*BLOCK:(bj+1)*BLOCK].max() < 0.5:
                    continue
                active += 1
                k_lin = bi * self.B + bj
                if k_lin < self.mu.size and self.mu[k_lin] == 0:
                    continue
                Z = A[bi*BLOCK:(bi+1)*BLOCK, bj*BLOCK:(bj+1)*BLOCK]
                if np.linalg.norm(Z) < 1e-12:
                    continue
                info = self.proj.project(Z)
                words.append(self.proj.encode(info))
                positions.append((bi, bj))
                S += info['importance'] * (1 if len(words) % 2 == 1 else -1)

        N = max(len(words), 1)
        p = min(abs(S) / N, 0.999)
        H = -ALPHA * p * math.log(p) if p > 0 else 0.0
        m = abs(S) * math.exp(-H) if H < 700 else 0.0

        return dict(words=np.array(words, dtype=np.uint32),
                    positions=positions, active_blocks=active,
                    n_words=len(words), S=S, H=H, m=m)

    def decompress(self, packed: Dict) -> np.ndarray:
        d_pad = self.B * BLOCK
        A_hat = np.zeros((d_pad, d_pad), dtype=np.float32)
        perm_cache: Dict[int, Tuple[int, ...]] = {}
        for w, (bi, bj) in zip(packed['words'], packed['positions']):
            A_hat[bi*BLOCK:(bi+1)*BLOCK,
                  bj*BLOCK:(bj+1)*BLOCK] = self.proj.decode(int(w), perm_cache)
        return A_hat[:self.d, :self.d]


# ============================================================
#  Synthetic open weights  (low-rank + diagonal)
# ============================================================
def make_synthetic_weights(d: int = 96, rank: int = 24, seed: int = 0):
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((d, rank)) * 0.3
    V = rng.standard_normal((rank, d)) * 0.3
    W = U @ V
    W += 0.1 * np.eye(d)
    return W.astype(np.float32)


def make_synthetic_attention(d: int = 96, seed: int = 0):
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((d, d // 4)) * 0.5
    K = rng.standard_normal((d, d // 4)) * 0.5
    A = Q @ K.T / math.sqrt(d // 4)
    return A.astype(np.float32)


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
    print("Spinor-projection weight compression  ·  rank-6 Levi-Civita scalar")
    print("=" * 78)
    print(f"  α = 1/(π − e)      = {ALPHA:.6f}")
    print(f"  6/π²               = {DENSITY:.6f}")
    print(f"  block size         = {BLOCK}×{BLOCK}")
    print(f"  permutations/block = {N_PERM}")
    print()

    # ---------- dense weight compression ----------
    print("--- dense weight compression ---")
    d = 96
    W = make_synthetic_weights(d, rank=24, seed=0)
    print(f"  shape              = {W.shape}")
    print(f"  bytes (float32)    = {W.nbytes}")

    for gate, pct in [(True, 0.0), (True, 50.0), (True, 75.0)]:
        comp = WeightCompressor(d, gate=gate, importance_pct=pct)
        t0 = time.perf_counter()
        packed = comp.compress(W)
        t_comp = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        W_hat = comp.decompress(packed)
        t_decomp = (time.perf_counter() - t0) * 1e3
        m = metrics(W, W_hat)
        stored = packed['n_words'] * 4
        ratio = W.nbytes / max(stored, 1)
        print(f"  gate={int(gate)}  keep={pct:5.1f}%  "
              f"n_words={packed['n_words']:>4d}/{packed['n_blocks']}   "
              f"ratio={ratio:>6.2f}x   "
              f"rel_err={m['rel_err']:.4f}   "
              f"cos={m['cosine']:.4f}   "
              f"comp={t_comp:5.1f}ms  decomp={t_decomp:5.1f}ms")
    print()

    # ---------- sparse attention compression ----------
    print("--- sparse attention compression ---")
    A = make_synthetic_attention(d, seed=1)
    print(f"  shape              = {A.shape}")
    print(f"  bytes (float32)    = {A.nbytes}")
    print()
    for name, mask in [
        ("local window w=8",  local_window_mask(d, 8)),
        ("local window w=16", local_window_mask(d, 16)),
        ("strided s=4",       strided_mask(d, 4)),
        ("strided s=8",       strided_mask(d, 8)),
    ]:
        comp = SparseAttentionCompressor(d)
        t0 = time.perf_counter()
        packed = comp.compress(A, mask)
        t_comp = (time.perf_counter() - t0) * 1e3
        A_hat = comp.decompress(packed)
        m = metrics(A * mask, A_hat * mask)
        stored = packed['n_words'] * 4
        # effective compression: compare to the whole A
        ratio = A.nbytes / max(stored, 1)
        active_density = packed['active_blocks'] / (comp.B * comp.B)
        print(f"  {name:20s}  active={packed['active_blocks']:>4d} "
              f"({active_density:.3f})  "
              f"stored={packed['n_words']:>4d}   "
              f"ratio={ratio:>6.2f}x   "
              f"rel_err={m['rel_err']:.4f}   "
              f"cos={m['cosine']:.4f}   "
              f"({t_comp:.1f}ms)")
    print()

    # ---------- 32-bit lookup speed ----------
    print("--- 32-bit lookup speed ---")
    proj = SpinorProjector(mag_max=0.5, tr_max=1.0)
    cache: Dict[int, Tuple[int, ...]] = {}
    sample = np.random.default_rng(2).integers(0, 2**32,
                                                size=100_000, dtype=np.uint32)
    t0 = time.perf_counter()
    for w in sample[:10_000]:
        _ = proj.decode(int(w), cache)
    dt = (time.perf_counter() - t0) * 1e3
    print(f"  10k decodes        = {dt:.2f} ms "
          f"({dt / 10:.4f} ms each, {10_000 / dt * 1000:.0f}/s)")
    print(f"  perm cache size    = {len(cache)}")
    print()

    # ---------- bit budget ----------
    print("--- 32-bit word layout ---")
    print(f"  sign of Π₆         : 1 bit")
    print(f"  Lehmer index σ*    : 10 bits   (0..719)")
    print(f"  magnitude of σ*    : 12 bits")
    print(f"  trace correction   : 9 bits")
    print(f"  total              : 32 bits = 4 bytes per 6×6 block")
    print()
    print("--- what the storage actually is ---")
    print("  · Π₆(Z) = det(Z)     — the projection scalar")
    print("  · σ*    = dominant monomial (which permutation)")
    print("  · mag*  = coefficient of that monomial")
    print("  · tr    = diagonal correction")
    print("  · Möbius gate μ(k) ≠ 0 decides which blocks get stored")
    print("  · importance threshold drops low-signal blocks")
    print()
    print("--- theoretical compression ceiling ---")
    n_blocks = (96 // 6) ** 2
    n_kept = int(n_blocks * DENSITY)
    print(f"  96×96 = {n_blocks} blocks, gate keeps ~{n_kept}")
    print(f"  raw       = {96*96*4} bytes")
    print(f"  gated     = {n_kept * 4} bytes")
    print(f"  ceiling   ≈ {96*96*4 / max(n_kept * 4, 1):.1f}x")
    print(f"  with 75% importance filter: ≈ "
          f"{96*96*4 / max(int(n_kept * 0.25) * 4, 1):.1f}x")
    print()


if __name__ == "__main__":
    demo()