#!/usr/bin/env python3
"""
mobius_catalan_mersenne.py

Möbius sieve in O(K log log K) for identifying Catalan-Mersenne primes.

Catalan-Mersenne sequence
-------------------------
    c_0 = 2
    c_{n+1} = 2^{c_n} − 1

First few terms:
    c_0 = 2
    c_1 = 3
    c_2 = 7
    c_3 = 127
    c_4 = 2^127 − 1                ≈ 1.7 × 10^38
    c_5 = 2^(2^127 − 1) − 1        ≈ 10^(5 × 10^37)

Every c_n for n ≤ 4 is prime.  The sequence grows double‑exponentially,
so a sieve of size K only reaches c_n ≤ K for the first few n.

Two‑stage primality test
------------------------
1.  **Algebraic stage** — the Eratosthenes Möbius sieve runs in
    O(K log log K) and returns both μ(1..K) and the primality flag.
    Every Catalan-Mersenne term c_n ≤ K is looked up directly.

2.  **Transcendental stage** — Lucas-Lehmer runs on M_p = 2^p − 1 for
    prime exponents p found by the sieve.  This extends the reach to
    c_4 = 2^127 − 1 whose exponent p = 127 was confirmed prime by
    the sieve itself.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Tuple


# ============================================================
#  1. Eratosthenes Möbius sieve  ·  O(K log log K)
# ============================================================
def mobius_sieve_eratosthenes(K: int) -> Tuple[List[int], List[bool]]:
    """
    Eratosthenes variant of the Möbius sieve.

    Returns
    -------
    mu       : list of length K+1, mu[0] unused, mu[n] = μ(n)
    is_prime : list of length K+1, is_prime[n] True iff n is prime

    Time     : O(K log log K)
    Space    : O(K)

    The outer loop only fires when i is prime; the inner sums run
    over K/p + K/p² + … ≈ K/p · (1 + 1/p + …) ≈ K/p, so the total
    work is Σ_{p ≤ K} K/p = K · log log K + O(K).
    """
    if K < 2:
        return [0] * (K + 1), [False] * (K + 1)

    mu = [1] * (K + 1)
    is_prime = [True] * (K + 1)
    is_prime[0] = is_prime[1] = False

    for i in range(2, K + 1):
        if is_prime[i]:
            # prime i → flip sign of μ on every multiple of i
            for j in range(i, K + 1, i):
                mu[j] = -mu[j]
                if j != i:
                    is_prime[j] = False
            # zero μ on multiples of i² (non‑square‑free)
            i2 = i * i
            if i2 <= K:
                for j in range(i2, K + 1, i2):
                    mu[j] = 0

    mu[0] = 0
    return mu, is_prime


def mobius_sieve_linear(K: int) -> Tuple[List[int], List[int]]:
    """
    Linear sieve — O(K) time, O(K) space.  Useful as a fast
    alternative when K is very large.  Returns (μ, primes).
    """
    mu = [0] * (K + 1)
    mu[1] = 1
    primes: List[int] = []
    is_comp = [False] * (K + 1)
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
    return mu, primes


# ============================================================
#  2. Catalan-Mersenne sequence
# ============================================================
def catalan_mersenne_sequence(K: int) -> List[int]:
    """
    Generate the Catalan-Mersenne sequence c_0 = 2, c_{n+1} = 2^{c_n} − 1
    truncated to terms ≤ K.
    """
    seq: List[int] = []
    c = 2
    while c <= K:
        seq.append(c)
        c = 2 ** c - 1
    return seq


def catalan_mersenne_all(max_n: int = 5) -> List[int]:
    """First `max_n` terms with arbitrary precision."""
    seq = [2]
    for _ in range(max_n - 1):
        seq.append(2 ** seq[-1] - 1)
    return seq


# ============================================================
#  3. Deterministic Lucas-Lehmer test for M_p = 2^p − 1
# ============================================================
def lucas_lehmer(p: int, verbose: bool = False) -> bool:
    """
    Deterministic primality test for the Mersenne number M_p = 2^p − 1.

    Requires p to be an odd prime (or p = 2).  Runs in O(p) modular
    squarings, i.e. O(p log p) bit operations.
    """
    if p == 2:
        return True
    if p < 2 or p % 2 == 0:
        return False

    M = (1 << p) - 1
    s = 4
    for _ in range(p - 2):
        s = (s * s - 2) % M
    return s == 0


# ============================================================
#  4. Combined pipeline
# ============================================================
def find_catalan_mersenne_primes(K: int,
                                verbose: bool = True
                                ) -> Dict[str, object]:
    """
    Identify Catalan-Mersenne primes up to K using the Möbius sieve,
    and extend the reach with Lucas-Lehmer on the sieve-confirmed
    prime exponents.

    Parameters
    ----------
    K       : sieve size, O(K log log K)
    verbose : print progress

    Returns
    -------
    dict with sieve size, timings, terms, flags, small primes, and
    the Lucas-Lehmer results.
    """
    t0 = time.perf_counter()
    mu, is_prime = mobius_sieve_eratosthenes(K)
    sieve_ms = (time.perf_counter() - t0) * 1e3

    if verbose:
        n_primes = sum(1 for b in is_prime if b)
        print(f"Möbius sieve up to K = {K}   "
              f"({sieve_ms:.2f} ms, {n_primes} primes found)")

    # --- Catalan-Mersenne terms ≤ K ---
    terms = catalan_mersenne_sequence(K)
    prime_flags = [is_prime[c] for c in terms]

    small_primes = [c for c, f in zip(terms, prime_flags) if f]
    if verbose:
        print(f"Catalan-Mersenne terms ≤ {K}: {terms}")
        for c, f in zip(terms, prime_flags):
            tag = "prime" if f else "composite"
            print(f"   c = {c:>10d}   {tag}")

    # --- Lucas-Lehmer on the prime exponents found by the sieve ---
    ll_results: Dict[int, bool] = {}
    for c in terms:
        if c > 2 and c <= K and is_prime[c]:
            ll_results[c] = lucas_lehmer(c)

    if verbose and ll_results:
        print()
        print("Lucas-Lehmer extensions (sieve-found prime exponents):")
        for c, ok in ll_results.items():
            tag = "prime" if ok else "composite"
            print(f"   M_{c:<6d} = 2^{c} - 1   →  {tag}")

    return dict(
        K=K,
        sieve_ms=sieve_ms,
        terms=terms,
        prime_flags=prime_flags,
        small_primes=small_primes,
        lucas_lehmer=ll_results,
    )


# ============================================================
#  5. Empirical scaling test
# ============================================================
def sieve_scaling_test():
    print("=" * 72)
    print("Eratosthenes Möbius sieve scaling  ·  O(K log log K)")
    print("=" * 72)
    print(f"  {'K':>10s}  {'time (ms)':>12s}  {'primes':>10s}  "
          f"{'t / (K·loglog K)':>20s}")
    print("-" * 72)
    for K in [10**3, 10**4, 10**5, 10**6]:
        t0 = time.perf_counter()
        mu, is_prime = mobius_sieve_eratosthenes(K)
        dt = (time.perf_counter() - t0) * 1e3
        n_p = sum(1 for b in is_prime if b)
        ratio = dt / (K * math.log(math.log(K))) if K > 2 else 0.0
        print(f"  {K:>10d}  {dt:>12.2f}  {n_p:>10d}  "
              f"{ratio:>20.8f}")


# ============================================================
#  6. Verification  ·  μ(p) = −1 for every prime p
# ============================================================
def verify_mu_on_primes(K: int) -> Dict[str, object]:
    mu, is_prime = mobius_sieve_eratosthenes(K)
    ok = True
    bad: List[int] = []
    for n in range(2, K + 1):
        if is_prime[n] and mu[n] != -1:
            ok = False
            bad.append(n)
    return dict(K=K, ok=ok, failures=bad)


# ============================================================
#  7. Demo
# ============================================================
def demo():
    print("=" * 72)
    print("Catalan-Mersenne primes via Möbius sieve")
    print("=" * 72)
    print()
    print("Catalan-Mersenne sequence:")
    print("    c_0 = 2")
    print("    c_{n+1} = 2^{c_n} − 1")
    print()
    print("First terms:")
    print("    c_0 = 2")
    print("    c_1 = 3")
    print("    c_2 = 7")
    print("    c_3 = 127")
    print("    c_4 = 2^127 − 1                ≈ 1.7 × 10^38")
    print("    c_5 = 2^(2^127 − 1) − 1        ≈ 10^(5 × 10^37)")
    print()
    print("The sequence grows double‑exponentially, so any sieve of")
    print("size K only contains the first few terms.  The sieve handles")
    print("c_0…c_3 directly; Lucas-Lehmer extends to c_4 via exponent 127.")
    print()

    # ---------- Step 1: small sieve ----------
    print("=" * 72)
    print("Step 1: sieve up to K = 1000")
    print("=" * 72)
    find_catalan_mersenne_primes(1000, verbose=True)

    # ---------- Step 2: larger sieve ----------
    print()
    print("=" * 72)
    print("Step 2: sieve up to K = 10^6")
    print("=" * 72)
    res = find_catalan_mersenne_primes(10 ** 6, verbose=False)
    print(f"  K                : {res['K']}")
    print(f"  sieve time       : {res['sieve_ms']:.2f} ms")
    print(f"  terms ≤ K        : {res['terms']}")
    print(f"  prime flags      : {res['prime_flags']}")
    print(f"  small primes     : {res['small_primes']}")

    # ---------- Step 3: Lucas-Lehmer extensions ----------
    print()
    print("=" * 72)
    print("Step 3: Lucas-Lehmer on the first prime exponents")
    print("=" * 72)
    for c in [2, 3, 7, 127]:
        ok = lucas_lehmer(c)
        tag = "prime" if ok else "composite"
        print(f"  M_{c:<6d} = 2^{c} - 1   →  {tag}")
    print()
    print("  c_4 = 2^127 − 1 is confirmed prime by Lucas-Lehmer on")
    print("  the exponent p = 127, which the sieve itself certified.")

    # ---------- Step 4: empirical scaling ----------
    print()
    sieve_scaling_test()

    # ---------- Step 5: μ on primes check ----------
    print()
    print("=" * 72)
    print("Verification  ·  μ(p) = −1 for every prime p")
    print("=" * 72)
    for K in [1000, 10_000, 100_000]:
        v = verify_mu_on_primes(K)
        tag = "OK" if v["ok"] else "FAIL"
        print(f"  K = {K:>7d}   μ(p) = −1 for all primes ≤ K   →  {tag}")

    # ---------- Step 6: arbitrary precision terms ----------
    print()
    print("=" * 72)
    print("First 5 Catalan-Mersenne terms (arbitrary precision)")
    print("=" * 72)
    seq = catalan_mersenne_all(5)
    for i, c in enumerate(seq):
        s = str(c)
        disp = s if len(s) <= 44 else s[:20] + "..." + s[-20:]
        print(f"  c_{i} = {disp}   (digits: {len(s)})")

    print()
    print("Done.")


if __name__ == "__main__":
    demo()