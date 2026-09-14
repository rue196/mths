#!/usr/bin/env python3
"""
fft_mobius_compress.py

O(K log log K) Möbius memory compression using FFT separation of
symmetric and asymmetric indices, with exponential-kernel convolution
in reconstruction.

Pipeline
--------
  1. Möbius sieve μ(n)                  O(K log log K)   [dominant]
  2. Build |Ci| array of length 2K+1    O(K)
  3. Split into S (symmetric) and A     O(K)
  4. FFT of S and FFT of A              O(K log K)
  5. Keep top‑M coefficients per part   O(K)
  6. Reconstruct via:
        (a) inverse FFT of each part
        (b) exponential‑kernel convolution (chip.py)
        (c) recombine symmetric + asymmetric
"""

import math
import time
import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict


# ============================================================
#  Constants
# ============================================================
PI          = math.pi
E           = math.e
ALPHA       = 1.0 / (PI - E)             # ≈ 2.362
ALPHA_USER  = 0.3628
A_STEP      = ALPHA / ALPHA_USER         # ≈ 6.511
NORM        = 1.0 - math.exp(-ALPHA * (PI + E))
DENSITY     = 6.0 / (PI * PI)            # ≈ 0.6079


# ============================================================
#  1. Möbius sieve (Eratosthenes variant)  ·  O(K log log K)
# ============================================================
def mobius_sieve(K: int) -> List[int]:
    if K < 1:
        return [0] * (K + 1)
    mu = [1] * (K + 1)
    is_comp = [False] * (K + 1)
    for i in range(2, K + 1):
        if not is_comp[i]:
            for j in range(i, K + 1, i):
                mu[j] = -mu[j]
                is_comp[j] = True
            i2 = i * i
            if i2 <= K:
                for j in range(i2, K + 1, i2):
                    mu[j] = 0
    mu[0] = 0
    return mu


# ============================================================
#  2. |Ci| builder  ·  O(K)
# ============================================================
def build_k_sum(tokens, K: int, mu: Optional[List[int]] = None) -> np.ndarray:
    c = np.zeros(2 * K + 1, dtype=np.float64)
    for t in tokens:
        h = int(hashlib.md5(str(t).encode()).hexdigest()[:8], 16)
        i = (h % K) + 1
        if mu is not None and mu[i] == 0:
            continue
        c[K + i] += 1.0
        c[K - i] += 1.0
    return c


# ============================================================
#  3. Split into symmetric and asymmetric parts  ·  O(K)
# ============================================================
def split_symmetric_asymmetric(c: np.ndarray, K: int,
                               tol: float = 1e-6
                               ) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Split c (length 2K+1) into:
        c_sym : even (symmetric) part       c_sym[K+i] = c_sym[K−i]
        c_asym: odd  (asymmetric) part      c_asym[K+i] = −c_asym[K−i]

    Symmetric index  : |c[i]| ≈ |c[−i]|  → both go into c_sym
    Asymmetric index : otherwise         → both go into c_asym
    """
    n = 2 * K + 1
    c_sym = np.zeros(n, dtype=np.float64)
    c_asym = np.zeros(n, dtype=np.float64)
    n_sym = n_asym = 0

    for i in range(1, K + 1):
        pos, neg = K + i, K - i
        cp, cn = c[pos], c[neg]
        acp, acn = abs(cp), abs(cn)

        if acp > 0 and acn > 0 and abs(acp - acn) <= tol * (acp + acn + 1e-12):
            # symmetric: keep the average in c_sym
            avg = 0.5 * (cp + cn)
            c_sym[pos] = avg
            c_sym[neg] = avg
            n_sym += 2
        else:
            # asymmetric: keep both halves with opposite signs
            c_asym[pos] = cp
            c_asym[neg] = -cn
            n_asym += 2

    c_sym[K] = c[K]
    return c_sym, c_asym, n_sym, n_asym


# ============================================================
#  4. FFT compression  ·  O(K log K)
# ============================================================
@dataclass
class FFTPart:
    n_full: int
    kept_indices: np.ndarray
    kept_values: np.ndarray
    n_kept: int
    supertrace: float
    entropy: float
    mass: float


def _supertrace(sig: np.ndarray) -> float:
    S = 0.0
    for i, v in enumerate(sig):
        S += v if (i % 2 == 0) else -v
    return S


def _entropy(S: float, N: int) -> float:
    if N <= 0 or S == 0.0:
        return 0.0
    p = abs(S) / N
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -ALPHA * p * math.log(p)


def _mass(S: float, N: int) -> float:
    H = _entropy(S, N)
    return abs(S) * math.exp(-H) if H < 700 else 0.0


def fft_compress_part(part: np.ndarray, M: Optional[int] = None) -> FFTPart:
    """
    Compress one real signal via FFT.
    Keeps top‑M Fourier coefficients by magnitude.
    """
    n = len(part)
    S = _supertrace(part)
    H = _entropy(S, n)
    m = _mass(S, n)

    if M is None:
        M = max(1, min(n, int(abs(S))))
    M = max(1, min(M, n))

    spectrum = np.fft.rfft(part)
    mags = np.abs(spectrum)
    # always keep the DC term for exact reconstruction of the mean
    order = np.argsort(mags)[::-1]
    keep = np.unique(np.concatenate([[0], order[:M]]))
    kept_idx = keep.astype(np.int64)
    kept_val = spectrum[keep]

    return FFTPart(
        n_full=n,
        kept_indices=kept_idx,
        kept_values=kept_val,
        n_kept=len(kept_idx),
        supertrace=S,
        entropy=H,
        mass=m,
    )


def fft_decompress_part(part: FFTPart) -> np.ndarray:
    """Reconstruct a real signal from kept Fourier coefficients."""
    spectrum = np.zeros(part.n_full // 2 + 1, dtype=complex)
    spectrum[part.kept_indices] = part.kept_values
    return np.fft.irfft(spectrum, n=part.n_full)


# ============================================================
#  5. Exponential kernel convolution (chip.py)  ·  O(K)
# ============================================================
def conv_exp_kernel(signal: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    K = len(signal)
    if K == 0:
        return signal.copy()
    lam = math.exp(-alpha)
    f = np.zeros(K, dtype=signal.dtype)
    f[0] = signal[0]
    for i in range(1, K):
        f[i] = signal[i] + lam * f[i - 1]
    b = np.zeros(K, dtype=signal.dtype)
    b[K - 1] = signal[K - 1]
    for i in range(K - 2, -1, -1):
        b[i] = signal[i] + lam * b[i + 1]
    conv_exp = (f + b - signal) / (1 - lam * lam)
    return (1.0 - conv_exp) / NORM


def deconv_exp_kernel(signal: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """
    Inverse of the exponential kernel:
        conv[i] = (1 − conv_exp[i]) / NORM
        ⇒  conv_exp[i] = 1 − NORM·conv[i]
        conv_exp[i] = ((f+b−s)[i]) / (1−λ²)
        ⇒  f[i] + b[i] − s[i] = conv_exp[i]·(1−λ²)
    Solving for s:
        s[i] = (f[i] + b[i] − conv_exp[i]·(1−λ²))
    which is implicit.  We approximate by the direct inverse of the
    two‑pass filter using the same recurrence on the residual.
    """
    K = len(signal)
    if K == 0:
        return signal.copy()
    lam = math.exp(-alpha)
    conv_exp = 1.0 - NORM * signal
    rhs = conv_exp * (1.0 - lam * lam)
    # f + b − s = rhs  ⇒  s = f + b − rhs
    f = np.zeros(K, dtype=signal.dtype)
    f[0] = rhs[0]
    for i in range(1, K):
        f[i] = rhs[i] + lam * f[i - 1]
    b = np.zeros(K, dtype=signal.dtype)
    b[K - 1] = rhs[K - 1]
    for i in range(K - 2, -1, -1):
        b[i] = rhs[i] + lam * b[i + 1]
    # solve  s[i] = (f[i] + b[i] − rhs[i])  iteratively (fixed‑point)
    s = (f + b - rhs) / (1.0 + lam * lam)
    return s


# ============================================================
#  6. Full compressor
# ============================================================
@dataclass
class CompressedMemory:
    K: int
    sym_part: FFTPart
    asym_part: FFTPart
    n_sym_original: int
    n_asym_original: int
    fingerprint: str
    metadata: Dict = field(default_factory=dict)


class FFTMobiusMemory:
    """
    FFT memory compressor with symmetric / asymmetric separation.

    Compress:
        |Ci|  →  S + A  →  FFT compress each  →  CompressedMemory

    Decompress:
        CompressedMemory  →  inverse FFT of each part
                          →  exponential‑kernel convolution on S
                          →  recombine S + A
    """

    def __init__(self, K: int):
        self.K = K
        self.mu = mobius_sieve(K)

    # ---------- compress ----------
    def compress(self, tokens,
                 M_sym: Optional[int] = None,
                 M_asym: Optional[int] = None) -> CompressedMemory:
        c = build_k_sum(tokens, self.K, self.mu)
        c_sym, c_asym, n_sym, n_asym = split_symmetric_asymmetric(c, self.K)

        part_sym = fft_compress_part(c_sym, M_sym)
        part_asym = fft_compress_part(c_asym, M_asym)

        # fingerprint
        h = hashlib.sha256()
        h.update(struct.pack('>I', self.K))
        for p in (part_sym, part_asym):
            for idx, val in zip(p.kept_indices, p.kept_values):
                h.update(struct.pack('>I', int(idx)))
                h.update(struct.pack('>d', val.real))
                h.update(struct.pack('>d', val.imag))
        fp = h.hexdigest()

        return CompressedMemory(
            K=self.K,
            sym_part=part_sym,
            asym_part=part_asym,
            n_sym_original=n_sym,
            n_asym_original=n_asym,
            fingerprint=fp,
            metadata={
                'n_sym_kept':  part_sym.n_kept,
                'n_asym_kept': part_asym.n_kept,
                'S_sym':       part_sym.supertrace,
                'S_asym':      part_asym.supertrace,
            },
        )

    # ---------- decompress ----------
    def decompress(self, mem: CompressedMemory,
                   use_conv: bool = True) -> np.ndarray:
        """
        Reconstruct the length‑(2K+1) |Ci| array.

        Steps:
            (a) inverse FFT of each part
            (b) exponential‑kernel convolution on the symmetric part
            (c) recombine symmetric + asymmetric
        """
        c_sym = fft_decompress_part(mem.sym_part)
        c_asym = fft_decompress_part(mem.asym_part)

        if use_conv:
            # smooth the symmetric part (the "positive" one)
            c_sym = conv_exp_kernel(c_sym)

        # recombine: at positive indices S + A, at negative S − A
        n = 2 * mem.K + 1
        out = np.zeros(n, dtype=np.float64)
        K = mem.K
        for i in range(1, K + 1):
            pos, neg = K + i, K - i
            out[pos] = c_sym[pos] + c_asym[pos]
            out[neg] = c_sym[neg] - c_asym[neg]
        out[K] = c_sym[K]

        # enforce Möbius support: zero out non‑square‑free indices
        for i in range(1, K + 1):
            if self.mu[i] == 0:
                out[K + i] = 0.0
                out[K - i] = 0.0
        return out

    # ---------- direct reconstruction from convolution ----------
    def decompress_via_conv(self, mem: CompressedMemory) -> np.ndarray:
        """
        Reconstruction using convolution in the frequency domain:
            c_hat = S_hat + A_hat
            c = IFFT(c_hat)
            c_smooth = conv_exp_kernel(c)
        """
        c_sym = fft_decompress_part(mem.sym_part)
        c_asym = fft_decompress_part(mem.asym_part)
        c_full = c_sym + c_asym
        c_smooth = conv_exp_kernel(c_full)
        return c_smooth


# need struct for the fingerprint above
import struct


# ============================================================
#  7. Demo
# ============================================================
def demo():
    print("=" * 68)
    print("FFT Möbius memory compression  ·  symmetric / asymmetric split")
    print("=" * 68)

    # ---- timing the sieve ---------------------------------------------
    print("\n--- Sieve scaling ---")
    print(f"{'K':>8s} {'time (ms)':>12s} {'μ≠0 density':>14s}")
    for K in [10**3, 10**4, 10**5, 10**6]:
        t0 = time.perf_counter()
        mu = mobius_sieve(K)
        dt = (time.perf_counter() - t0) * 1e3
        density = sum(1 for n in range(1, K + 1) if mu[n] != 0) / K
        print(f"{K:>8d} {dt:>12.2f} {density:>14.6f}")
    print(f"(square‑free density → 6/π² = {DENSITY:.6f})")

    # ---- small demo ---------------------------------------------------
    K = 256
    comp = FFTMobiusMemory(K)

    import random
    random.seed(7)
    tokens = [f"tok_{random.randint(0, 10**6)}" for _ in range(500)]

    t0 = time.perf_counter()
    mem = comp.compress(tokens)
    t_c = time.perf_counter() - t0

    t0 = time.perf_counter()
    rec_conv = comp.decompress(mem, use_conv=True)
    t_d1 = time.perf_counter() - t0

    t0 = time.perf_counter()
    rec_fft = comp.decompress_via_conv(mem)
    t_d2 = time.perf_counter() - t0

    # original for reference
    c_orig = build_k_sum(tokens, K, comp.mu)

    def rel_err(a, b):
        denom = np.linalg.norm(b) + 1e-12
        return np.linalg.norm(a - b) / denom

    print(f"\n--- Small demo (K = {K}, {len(tokens)} tokens) ---")
    print(f"  symmetric entries  : {mem.n_sym_original}")
    print(f"  asymmetric entries : {mem.n_asym_original}")
    print(f"  kept (sym part)    : {mem.sym_part.n_kept} "
          f"of {mem.sym_part.n_full}")
    print(f"  kept (asym part)   : {mem.asym_part.n_kept} "
          f"of {mem.asym_part.n_full}")
    print(f"  |Ci| length        : {len(c_orig)}")
    print(f"  fingerprint        : {mem.fingerprint[:24]}…")
    print(f"\n  timings (ms):")
    print(f"    compress          : {t_c*1e3:8.3f}")
    print(f"    decompress + conv : {t_d1*1e3:8.3f}")
    print(f"    decompress FFT    : {t_d2*1e3:8.3f}")

    print(f"\n  relative L2 error:")
    print(f"    S + A + conv      : {rel_err(rec_conv, c_orig):.4e}")
    print(f"    full FFT + conv   : {rel_err(rec_fft, c_orig):.4e}")

    # ---- supertrace of reconstruction ---------------------------------
    def S_of(arr):
        s = 0.0
        for i, v in enumerate(arr):
            s += v if (i % 2 == 0) else -v
        return s

    print(f"\n  supertrace:")
    print(f"    original          : {S_of(c_orig):+.6f}")
    print(f"    reconstructed     : {S_of(rec_conv):+.6f}")
    print(f"    Δ                 : {S_of(rec_conv) - S_of(c_orig):+.3e}")

    # ---- batch pipeline demo ------------------------------------------
    print("\n--- Batch pipeline ---")
    batch_sizes = [100, 500, 1000, 2000]
    for n_tok in batch_sizes:
        toks = [f"t{i}" for i in range(n_tok)]
        t0 = time.perf_counter()
        m = comp.compress(toks)
        t1 = time.perf_counter() - t0
        r = comp.decompress(m)
        t2 = time.perf_counter() - t0
        err = rel_err(r, build_k_sum(toks, K, comp.mu))
        print(f"  tokens={n_tok:>5d}   "
              f"compress={t1*1e3:7.2f} ms   "
              f"total={t2*1e3:7.2f} ms   "
              f"err={err:.2e}")

    # ---- complexity summary --------------------------------------------
    print("\n--- Complexity ---")
    print("  Möbius sieve         O(K log log K)   ← dominant for large K")
    print("  |Ci| construction    O(K)")
    print("  S/A split            O(K)")
    print("  FFT of each part     O(K log K)")
    print("  keep top‑M           O(K)")
    print("  inverse FFT          O(K log K)")
    print("  exp‑kernel conv      O(K)")
    print("  -----------------------------------------")
    print("  Total                O(K log K)   (sieve is O(K log log K))")
    print("\nDone.")


if __name__ == "__main__":
    demo()