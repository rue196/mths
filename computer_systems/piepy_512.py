#!/usr/bin/env python3
"""
piepy_512.py
============

512-bit extensions of piepy.py.

From piepy.py:

    exp_series(1.0, K) = Σ_{n=0}^{K} 1/n!        (O(K) terms)

gives e to ~log2((K+1)!) bits — but only if arithmetic is done
with ≥512-bit precision.  The original exp_series is float,
so it caps at 53 bits.

Fix for 512 bits
----------------
    e ≈ N_K / K!     N_K = Σ_{n=0}^{K} K!/n!  ∈  ℤ

Both N_K and K! are exact Python integers, so the sum has *zero*
rounding error.  Convert to Decimal at the end.

π via Machin's formula (also O(K) terms, ≥512 bits):

    π = 16·arctan(1/5) − 4·arctan(1/239)
    arctan(1/n) = Σ_{k≥0} (−1)^k / ((2k+1)·n^(2k+1))

α = 1/(π − e) inherits 512 bits from the two.

No mpmath, no external bignum library — only Python int + Decimal.
"""

from __future__ import annotations
import math
from decimal import Decimal, getcontext, localcontext


# ============================================================
#  Precision budget
# ============================================================
BITS       = 512
# 512 bits ≈ 154.13 decimal digits.  Keep 25 guard digits.
DEC_DIGITS = int(BITS * math.log10(2)) + 25   # = 180


# ============================================================
#  1  ·  O(K) exact-integer Taylor series for e
# ============================================================
def e_taylor_ratio(K: int) -> tuple[int, int]:
    """
    e ≈ N_K / K!

        N_K = Σ_{n=0}^{K} K!/n!        (exact Python int)

    Truncation error:   |e − N_K/K!| < 1/(K+1)! · (K+2)/(K+1)
                                    ≈ 1/(K+1)!

    so bits of precision ≈ log2((K+1)!).

    O(K) terms — one integer add and one integer divide per term.
    """
    fact = 1
    for n in range(1, K + 1):
        fact *= n                       # K!  (exact)
    N = 0
    ratio = fact                        # K!/0!
    for n in range(0, K + 1):
        N += ratio
        if n < K:
            ratio //= (n + 1)           # K!/(n+1)!  (exact division)
    return N, fact


def e_precision_bits(K: int) -> float:
    """log2((K+1)!)  —  bits of correctness from Taylor truncation."""
    return math.lgamma(K + 2) / math.log(2)


def minimal_K_for_bits(target: int) -> int:
    """Smallest K with e_precision_bits(K) ≥ target."""
    lo, hi = 1, 2 * target
    while lo < hi:
        mid = (lo + hi) // 2
        if e_precision_bits(mid) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def e_512bit(verbose: bool = False) -> Decimal:
    """e at ≥512 bits via the O(K) Taylor series from piepy.py."""
    K = minimal_K_for_bits(BITS)
    N, fact = e_taylor_ratio(K)
    with localcontext() as ctx:
        ctx.prec = DEC_DIGITS
        e_val = Decimal(N) / Decimal(fact)
    if verbose:
        print(f"  e:  Taylor order K   = {K}")
        print(f"      bits of precision = {e_precision_bits(K):.2f}")
        print(f"      Decimal digits    = {DEC_DIGITS}")
    return e_val


# ============================================================
#  2  ·  O(K) Machin series for π
# ============================================================
def arctan_inv_decimal(n: int, K: int) -> Decimal:
    """
    arctan(1/n)  via alternating series, O(K) terms, Decimal precision.

        arctan(1/n) = Σ_{k=0}^{K} (−1)^k / ((2k+1)·n^(2k+1))

    Truncation error < 1/((2K+3)·n^(2K+3)).
    """
    with localcontext() as ctx:
        ctx.prec = DEC_DIGITS
        n_dec = Decimal(n)
        n_sq  = n_dec * n_dec
        n_pow = n_dec                   # n^(2k+1) for k = 0
        total = Decimal(0)
        sign  = 1
        for k in range(K + 1):
            term = Decimal(1) / (Decimal(2 * k + 1) * n_pow)
            total += sign * term
            sign = -sign
            n_pow *= n_sq               # step to n^(2k+3)
    return total


def pi_precision_bits(K: int) -> float:
    """
    Bits of π from Machin's formula, dominated by arctan(1/5):

        |16·R_5(K)| < 16 / ((2K+3)·5^(2K+3))

        bits = log2((2K+3)·5^(2K+3)/16)
             = log2(2K+3) + (2K+3)·log2(5) − 4
    """
    return (math.log2(2 * K + 3)
            + (2 * K + 3) * math.log2(5.0)
            - 4.0)


def minimal_K_for_pi(target: int) -> int:
    lo, hi = 1, 2 * target
    while lo < hi:
        mid = (lo + hi) // 2
        if pi_precision_bits(mid) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def pi_512bit(verbose: bool = False) -> Decimal:
    """π at ≥512 bits via Machin's formula."""
    K = minimal_K_for_pi(BITS)
    with localcontext() as ctx:
        ctx.prec = DEC_DIGITS
        a5   = arctan_inv_decimal(5,   K)
        a239 = arctan_inv_decimal(239, K)
        pi_val = Decimal(16) * a5 - Decimal(4) * a239
    if verbose:
        print(f"  π:  Machin order K   = {K}")
        print(f"      bits of precision = {pi_precision_bits(K):.2f}")
    return pi_val


# ============================================================
#  3  ·  α = 1/(π − e)  at 512 bits
# ============================================================
def alpha_512bit(verbose: bool = False) -> Decimal:
    e_val  = e_512bit()
    pi_val = pi_512bit()
    with localcontext() as ctx:
        ctx.prec = DEC_DIGITS
        alpha = Decimal(1) / (pi_val - e_val)
    if verbose:
        print(f"  α = 1/(π − e)        = {alpha}")
    return alpha


# ============================================================
#  4  ·  Convergence verification
# ============================================================
def verify_e_convergence(verbose: bool = True) -> None:
    """Error vs K for the truncated Taylor series."""
    e_ref = e_512bit()
    if verbose:
        print("=" * 78)
        print("e  ·  Taylor series convergence  ·  O(K) integer exact")
        print("=" * 78)
        print(f"  {'K':>4s}  {'bits':>8s}  {'|e_K − e|':>14s}  "
              f"{'correct digits':>16s}")
        print("  " + "-" * 50)
    for K in (20, 40, 60, 70, 80, 90, 95, 100, 105, 110):
        N, fact = e_taylor_ratio(K)
        with localcontext() as ctx:
            ctx.prec = DEC_DIGITS
            e_K = Decimal(N) / Decimal(fact)
            err = abs(e_K - e_ref)
        # trailing zero-stripped error
        try:
            if err == 0:
                digits = "exact"
            else:
                digits = f"{-int(math.log10(float(err))):>6d}"
        except (OverflowError, ValueError):
            digits = "   >320"
        if verbose:
            print(f"  {K:>4d}  {e_precision_bits(K):>8.2f}  "
                  f"{float(err):>14.4e}  {digits}")


def verify_pi_convergence(verbose: bool = True) -> None:
    pi_ref = pi_512bit()
    if verbose:
        print()
        print("=" * 78)
        print("π  ·  Machin series convergence  ·  Decimal, O(K) terms")
        print("=" * 78)
        print(f"  {'K':>4s}  {'bits':>8s}  {'|π_K − π|':>14s}")
        print("  " + "-" * 32)
    for K in (60, 80, 90, 100, 105, 110, 115, 120):
        with localcontext() as ctx:
            ctx.prec = DEC_DIGITS
            a5   = arctan_inv_decimal(5,   K)
            a239 = arctan_inv_decimal(239, K)
            pi_K = Decimal(16) * a5 - Decimal(4) * a239
            err = abs(pi_K - pi_ref)
        if verbose:
            print(f"  {K:>4d}  {pi_precision_bits(K):>8.2f}  "
                  f"{float(err):>14.4e}")


# ============================================================
#  Demo
# ============================================================
def main():
    print("=" * 78)
    print("piepy_512  ·  512-bit constants from piepy.py's O(K) series")
    print("=" * 78)
    print(f"  BITS              = {BITS}")
    print(f"  DEC_DIGITS        = {DEC_DIGITS}")
    print(f"  BITS / log2(10)   = {BITS * math.log10(2):.2f} decimal digits")
    print()

    # ---- 512-bit constants ----
    print("=" * 78)
    print("1  ·  512-bit e, π, α")
    print("=" * 78)
    E_512     = e_512bit(verbose=True)
    PI_512    = pi_512bit(verbose=True)
    ALPHA_512 = alpha_512bit()
    print()
    print(f"  e_512       = {E_512}")
    print(f"  π_512       = {PI_512}")
    print(f"  α_512       = {ALPHA_512}")
    print()

    # ---- double precision comparison ----
    print("=" * 78)
    print("2  ·  Double-precision match  (float(E_512) == math.e ?)")
    print("=" * 78)
    print(f"  float(E_512)  = {float(E_512):.17g}")
    print(f"  math.e        = {math.e:.17g}")
    print(f"  E match       = {float(E_512) == math.e}")
    print(f"  float(PI_512) = {float(PI_512):.17g}")
    print(f"  math.pi       = {math.pi:.17g}")
    print(f"  PI match      = {float(PI_512) == math.pi}")
    print(f"  float(α_512)  = {float(ALPHA_512):.17g}")
    print(f"  1/(π−e) dbl   = {1.0 / (math.pi - math.e):.17g}")
    print(f"  ALPHA match   = {float(ALPHA_512) == 1.0 / (math.pi - math.e)}")
    print()

    # ---- convergence ----
    verify_e_convergence()
    verify_pi_convergence()
    print()

    # ---- 512-bit integrity check ----
    print("=" * 78)
    print("3  ·  512-bit integrity  (independent recomputation)")
    print("=" * 78)
    # recompute α from first principles and compare
    with localcontext() as ctx:
        ctx.prec = DEC_DIGITS
        alpha_check = Decimal(1) / (PI_512 - E_512)
    print(f"  recomputed 1/(π−e) == α_512  : "
          f"{alpha_check == ALPHA_512}")

    # verify e·e^{-1} == 1 at 512 bits (self-consistency)
    with localcontext() as ctx:
        ctx.prec = DEC_DIGITS
        one = (Decimal(1) / E_512) * E_512
    print(f"  (1/e)·e == 1 at 512 bits     : "
          f"{abs(one - Decimal(1)) < Decimal(10) ** (-DEC_DIGITS + 5)}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()