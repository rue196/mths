#!/usr/bin/env python3
"""
factor_dual_sat.py
==================

O(K) integer factorization via dual Möbius SAT.

    K = ⌊√N⌋ + 1       natural trial-division bound

Axes  (split by symmetry)
-------------------------
    SYMMETRIC     real axis        α = 0.3628
        odd i, μ(i) = −1   (odd number of prime factors)
        coefficient goes to  c_re[K ± i]

    ASYMMETRIC    imaginary axis   α = 1/(π−e) ≈ 2.362
        even i, OR  odd i with μ(i) = +1   (even number of prime factors)
        coefficient goes to  c_im[K ± i]     (Im > 0)

    DC            real axis        coefficient = N
        i = 0 slot   →  N  if μ(N) ≠ 0, else 0
        (the "square-free N" indicator)

Quadratic SAT gate
------------------
    q(i) = (i² + i) mod K       quadratic address
    μ(q(i)) = 0    →  gated, coefficient × 0.5   (still counted, attenuated)
    μ(q(i)) ≠ 0    →  full amplitude

Factor extraction
-----------------
    Real peak  =  argmax_{i∈[1,K]} c_re[K+i]      largest odd factor
    Imag peak  =  argmax_{i∈[1,K]} c_im[K+i]      largest even/Ω-even factor

    p = real peak if it has positive amplitude, else imag peak
    q = N // p

    p = 1   →  N is prime
    recurse →  full prime factorization

Complexity
----------
    sieve              O(K log log K)
    classify + build   O(K)
    dual supertrace    O(K)
    peak extraction    O(K)
    factor once        O(K)
    full recursion     O(K · Ω(N))     Ω = total prime factors
"""

from __future__ import annotations
import math
from typing import List, Tuple, Optional

import numpy as np


# ============================================================
#  Constants
# ============================================================
PI         = math.pi
E          = math.e
ALPHA_ODD  = 0.3628                  # symmetric side
ALPHA_EVEN = 1.0 / (PI - E)          # asymmetric side  ≈ 2.362
DENSITY    = 6.0 / (PI * PI)         # ≈ 0.6079


# ============================================================
#  Möbius sieve  ·  O(K log log K)
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
#  Candidate classification
# ============================================================
def classify(i: int, mu: List[int]) -> str:
    """
    'real' → symmetric axis, α = 0.3628
    'imag' → asymmetric axis, α = 1/(π-e), Im > 0
    """
    if i % 2 == 0:
        return 'imag'                      # even n → asymmetric
    if i == 0:
        return 'real'                      # DC
    m = mu[i] if i < len(mu) else 0
    if m == -1:
        return 'real'                      # odd Ω → symmetric
    if m == +1:
        return 'imag'                      # even Ω → asymmetric
    return 'imag'                          # μ=0 → fall through to asymmetric


def quadratic_address(i: int, K: int) -> int:
    """q(i) = (i² + i) mod K."""
    return (i * i + i) % K


# ============================================================
#  Dual |Ci| encoding
# ============================================================
def build_dual_ci(N: int, K: int,
                  mu: List[int]
                  ) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    """
    Build the (2K+1) complex |Ci| array for N.

    Returns (c_re, c_im, real_factors, imag_factors).
    Only factor candidates populate nonzero amplitudes.
    """
    c_re = np.zeros(2 * K + 1, dtype=np.float64)
    c_im = np.zeros(2 * K + 1, dtype=np.float64)
    real_factors: List[int] = []
    imag_factors: List[int] = []

    for i in range(1, K + 1):
        if N % i != 0:
            continue

        q = quadratic_address(i, K)
        gated = (q < len(mu)) and (mu[q] == 0)
        amp_gate = 0.5 if gated else 1.0

        side = classify(i, mu)
        if side == 'real':
            coef = ALPHA_ODD * i * amp_gate
            c_re[K + i] = coef
            c_re[K - i] = coef
            real_factors.append(i)
        else:
            coef = ALPHA_EVEN * i * amp_gate
            c_im[K + i] = coef
            c_im[K - i] = coef
            imag_factors.append(i)

    # DC term at i = 0
    if N % 2 == 1 and N <= K and N < len(mu) and mu[N] != 0:
        c_re[K] = float(N)

    return c_re, c_im, real_factors, imag_factors


# ============================================================
#  Dual supertrace
# ============================================================
def dual_supertrace(c_re: np.ndarray, c_im: np.ndarray,
                    K: int) -> Tuple[float, float, float]:
    """
    S_pos  = Σ_{i=1..K} (−1)^i · c_re[K+i]
    S_neg  = Σ_{i=1..K} (−1)^i · c_im[K+i]
    S_all  = S_pos + S_neg + c_re[K]   (DC)
    """
    S_pos = 0.0
    S_neg = 0.0
    for i in range(1, K + 1):
        sign = 1.0 if (i % 2 == 0) else -1.0
        S_pos += sign * c_re[K + i]
        S_neg += sign * c_im[K + i]
    S_all = S_pos + S_neg + float(c_re[K])
    return S_pos, S_neg, S_all


# ============================================================
#  Peak extraction
# ============================================================
def extract_factor(c_re: np.ndarray, c_im: np.ndarray,
                   K: int, N: int) -> int:
    """
    Real peak first; fall back to imag peak.  Returns p | N.
    p = 1 means N is prime (no non-trivial factor ≤ K).
    """
    # real side
    re_slice = c_re[K + 1: K + 1 + K]
    if re_slice.size and re_slice.max() > 0.0:
        return int(np.argmax(re_slice)) + 1

    # imag side
    im_slice = c_im[K + 1: K + 1 + K]
    if im_slice.size and im_slice.max() > 0.0:
        return int(np.argmax(im_slice)) + 1

    return 1


# ============================================================
#  Single-shot factorization
# ============================================================
def factor_once(N: int, verbose: bool = False
                ) -> Tuple[int, int, dict]:
    """
    Factor N in one pass.  Returns (p, q, debug).

        p = real peak if positive, else imag peak
        q = N // p

    If p = 1 then N is prime.
    """
    if N < 2:
        return 1, N, dict(note="trivial")

    K = math.isqrt(N) + 1
    mu = mobius_sieve(K)

    c_re, c_im, real_factors, imag_factors = \
        build_dual_ci(N, K, mu)
    S_pos, S_neg, S_all = dual_supertrace(c_re, c_im, K)
    p = extract_factor(c_re, c_im, K, N)
    q = N // p if p else N

    debug = dict(
        K=K, n_real=len(real_factors), n_imag=len(imag_factors),
        real_factors=real_factors, imag_factors=imag_factors,
        S_pos=S_pos, S_neg=S_neg, S_all=S_all,
        c_re_dc=float(c_re[K]),
    )

    if verbose:
        print(f"  N = {N}   K = {K}")
        print(f"  real factors (sym)  : {real_factors[:8]}"
              f"{'…' if len(real_factors) > 8 else ''}")
        print(f"  imag factors (asym) : {imag_factors[:8]}"
              f"{'…' if len(imag_factors) > 8 else ''}")
        print(f"  DC term c_re[0]     : {debug['c_re_dc']:.4f}")
        print(f"  S_pos = {S_pos:+.6f}")
        print(f"  S_neg = {S_neg:+.6f}")
        print(f"  S_all = {S_all:+.6f}")
        print(f"  peak p = {p}   q = {q}   p·q = {p*q}")

    return p, q, debug


# ============================================================
#  Full factorization (recursive)
# ============================================================
def factor_full(N: int, depth: int = 0) -> List[int]:
    """Full prime factorization via recursion on dual-SAT factor_once."""
    if N < 2:
        return []
    if N in (2, 3, 5, 7):
        return [N]

    p, q, _ = factor_once(N, verbose=False)

    if p <= 1 or p == N:
        # prime fallback (single-pass returned 1 or N)
        return [N]

    left  = factor_full(p, depth + 1)
    right = factor_full(q, depth + 1)
    return sorted(left + right)


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("Dual Möbius SAT factorization  ·  O(K)")
    print("=" * 78)
    print(f"  α_odd   (sym)    = {ALPHA_ODD:.6f}")
    print(f"  α_even  (asym)   = {ALPHA_EVEN:.6f}")
    print(f"  6/π²             = {DENSITY:.6f}")
    print()
    print("  Layout:")
    print("    i odd, μ=−1  →  real axis,      α = 0.3628")
    print("    i even / μ=+1→  imag axis,      α = 1/(π−e), Im > 0")
    print("    i = 0        →  DC,  = N if μ(N) ≠ 0")
    print("    gate: μ((i²+i) mod K) = 0 → amp × 0.5")
    print()

    # ---------- 1. show the |Ci| layout for one N ----------
    print("=" * 78)
    print("1  ·  |Ci| layout for N = 77 = 7 × 11")
    print("=" * 78)
    p, q, d = factor_once(77, verbose=True)
    print(f"  → factorization:  77 = {p} × {q}")
    print()

    # ---------- 2. direct examples ----------
    print("=" * 78)
    print("2  ·  one-shot factorizations")
    print("=" * 78)
    print(f"  {'N':>8}  {'K':>4}  {'p':>5}  {'q':>5}  "
          f"{'S_pos':>10}  {'S_neg':>10}  {'S_all':>10}  ok")
    for N in [15, 21, 35, 77, 143, 91, 169, 221, 323, 391]:
        p, q, d = factor_once(N, verbose=False)
        ok = (p * q == N) and p > 1 and q > 1
        print(f"  {N:>8d}  {d['K']:>4d}  {p:>5d}  {q:>5d}  "
              f"{d['S_pos']:>+10.4f}  {d['S_neg']:>+10.4f}  "
              f"{d['S_all']:>+10.4f}  {'✓' if ok else '✗'}")
    print()

    # ---------- 3. even and mixed parity ----------
    print("=" * 78)
    print("3  ·  even and mixed N  (asymmetric side carries the factor)")
    print("=" * 78)
    print(f"  {'N':>8}  {'K':>4}  {'p':>5}  {'q':>5}  "
          f"{'side':>6}  ok")
    for N in [6, 10, 14, 22, 26, 34, 38, 46, 58, 62]:
        p, q, d = factor_once(N, verbose=False)
        side = 'real' if any(x % 2 == 1 and x > 1 for x in
                             [p]) else 'imag'
        # simpler: report which side p came from
        if p % 2 == 1 and d['real_factors'] and p in d['real_factors']:
            side = 'real'
        elif d['imag_factors'] and p in d['imag_factors']:
            side = 'imag'
        else:
            side = 'triv'
        ok = (p * q == N) and p > 1 and q > 1
        print(f"  {N:>8d}  {d['K']:>4d}  {p:>5d}  {q:>5d}  "
              f"{side:>6s}  {'✓' if ok else '✗'}")
    print()

    # ---------- 4. primes ----------
    print("=" * 78)
    print("4  ·  primes  (single-shot returns p = 1 → N is prime)")
    print("=" * 78)
    print(f"  {'N':>8}  {'K':>4}  {'peak':>5}  verdict")
    for N in [7, 11, 13, 17, 19, 23, 29, 31, 37, 41]:
        p, q, d = factor_once(N, verbose=False)
        verdict = "prime" if p == 1 else f"factor {p}×{q}"
        print(f"  {N:>8d}  {d['K']:>4d}  {p:>5d}  {verdict}")
    print()

    # ---------- 5. full prime factorization ----------
    print("=" * 78)
    print("5  ·  full prime factorization  (recursion)")
    print("=" * 78)
    for N in [4, 8, 9, 12, 16, 25, 27, 36, 45, 72, 96, 143, 221, 3161 ]:
        factors = factor_full(N)
        prod = 1
        for f in factors:
            prod *= f
        ok = (prod == N)
        print(f"  {N:>6d}  →  {factors}  "
              f"({'✓' if ok else '✗'})")
    print()

    # ---------- 6. supertrace trace for a worked example ----------
    print("=" * 78)
    print("6  ·  full dual-supertrace decomposition  ·  N = 143 = 11 × 13")
    print("=" * 78)
    K = math.isqrt(143) + 1
    mu = mobius_sieve(K)
    c_re, c_im, real_f, imag_f = build_dual_ci(143, K, mu)
    print(f"  K = {K}")
    print(f"  candidate i's with N % i == 0:")
    print(f"    real side : {real_f}")
    print(f"    imag side : {imag_f}")
    print()
    print(f"  |Ci| non-zero entries:")
    for i in range(-K, K + 1):
        re = c_re[i + K]
        im = c_im[i + K]
        if abs(re) < 1e-12 and abs(im) < 1e-12:
            continue
        axis = "real" if abs(re) > 1e-12 else "imag"
        print(f"    C_{i:>+3d}  =  {re:>+8.4f}  "
              f"+ {im:>+8.4f}i    ({axis})")
    S_pos, S_neg, S_all = dual_supertrace(c_re, c_im, K)
    print()
    print(f"  S_pos = {S_pos:+.6f}")
    print(f"  S_neg = {S_neg:+.6f}")
    print(f"  S_all = {S_all:+.6f}")
    p, q, _ = factor_once(143, verbose=False)
    print(f"  peak p = {p}   q = {q}   ✓ = {p*q == 143}")
    print()

    # ---------- 7. complexity ----------
    print("=" * 78)
    print("7  ·  complexity")
    print("=" * 78)
    print("  sieve                O(K log log K)")
    print("  classify + |Ci|      O(K)")
    print("  quadratic gate       O(K)")
    print("  dual supertrace      O(K)")
    print("  peak extraction      O(K)")
    print("  ─────────────────────────────")
    print("  factor once          O(K log log K)  ≈  O(K)")
    print("  full recursion       O(K · Ω(N))")
    print()
    print("  K = ⌊√N⌋ + 1,  Ω(N) = total prime factors of N")
    print()
    print("Done.")


if __name__ == "__main__":
    demo()