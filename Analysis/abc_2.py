#!/usr/bin/env python3
"""
abc_mobius_density.py
=====================
Numerical exploration of the ABC density heuristic.

Computes:
  1. The Möbius Basel sum B(K) = Σ_{n≤K} μ(n)/n²  →  6/π²
  2. The square-free density Q(K)/K
  3. ABC triples up to a height H and their quality q = log c / log rad(abc)
  4. The empirical density of square-free a, b, c vs (6/π²)³
"""

from __future__ import annotations
import math
from typing import Dict, Iterator, List, Tuple
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ============================================================
#  Constants
# ============================================================
PI         = math.pi
E          = math.e
ALPHA_SYM  = 1.0 / (PI - E)              # ≈ 2.362338
DENSITY    = 6.0 / (PI * PI)             # ≈ 0.607927


# ============================================================
#  Möbius sieve  ·  O(K)
# ============================================================
def mobius_sieve(K: int) -> np.ndarray:
    if K < 1:
        return np.zeros(K + 1, dtype=np.int8)
    mu = np.zeros(K + 1, dtype=np.int8)
    mu[1] = 1
    primes, is_comp = [], np.zeros(K + 1, dtype=bool)
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
            mu[i * p] = -mu[i]
    return mu


# ============================================================
#  Radical
# ============================================================
def radical(n: int, smallest_prime: np.ndarray) -> int:
    """rad(n) = product of distinct primes dividing n."""
    r = 1
    while n > 1:
        p = int(smallest_prime[n])
        r *= p
        while n % p == 0:
            n //= p
    return r


def smallest_prime_factors(K: int) -> np.ndarray:
    spf = np.arange(K + 1)
    for i in range(2, int(K ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, K + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf


# ============================================================
#  Basel density
# ============================================================
def basel_partial(K: int, mu: np.ndarray) -> float:
    n = np.arange(1, K + 1)
    return float(np.sum(mu[1:K + 1] / (n * n)))


def square_free_density(K: int, mu: np.ndarray) -> float:
    return float(np.count_nonzero(mu[1:K + 1]) / K)


# ============================================================
#  ABC triples
# ============================================================
def abc_triples(H: int, spf: np.ndarray) -> Iterator[Tuple[int, int, int, int]]:
    """Yield (a, b, c, rad(abc)) for coprime triples a + b = c with c ≤ H."""
    for a in range(1, H // 2 + 1):
        for b in range(a + 1, H - a + 1):
            c = a + b
            if c > H:
                break
            if math.gcd(a, b) != 1:
                continue
            rad_abc = (radical(a, spf)
                       * radical(b, spf)
                       * radical(c, spf))
            yield a, b, c, rad_abc


def quality(c: int, rad_abc: int) -> float:
    if rad_abc <= 1:
        return float('inf')
    return math.log(c) / math.log(rad_abc)


# ============================================================
#  Main
# ============================================================
def main():
    print("=" * 78)
    print("ABC conjecture  ·  Möbius density heuristic  ·  numerical check")
    print("=" * 78)
    print(f"  α_sym = 1/(π − e)      = {ALPHA_SYM:.6f}")
    print(f"  6/π²                   = {DENSITY:.8f}")
    print()

    # ---------- 1. Basel density ----------
    print("--- Basel–Möbius sum ---")
    print(f"  {'K':>8s}   {'B(K)':>14s}   {'|B(K) − 6/π²|':>16s}   "
          f"{'Q(K)/K':>10s}   {'|Q/K − 6/π²|':>16s}")
    K_grid = [100, 1_000, 10_000, 100_000, 1_000_000]
    mu_large = mobius_sieve(max(K_grid))
    for K in K_grid:
        B = basel_partial(K, mu_large)
        QK = square_free_density(K, mu_large)
        print(f"  {K:>8d}   {B:>14.8f}   {abs(B - DENSITY):>16.3e}   "
              f"{QK:>10.6f}   {abs(QK - DENSITY):>16.3e}")
    print()

    # ---------- 2. ABC triples ----------
    H = 500
    print(f"--- ABC triples up to H = {H} ---")
    spf = smallest_prime_factors(H)
    triples = list(abc_triples(H, spf))
    print(f"  total triples  : {len(triples)}")

    # quality distribution
    qs = np.array([quality(c, r) for (a, b, c, r) in triples])
    print(f"  q range        : [{qs.min():.4f}, {qs.max():.4f}]")
    print(f"  q mean         : {qs.mean():.4f}")
    print(f"  q > 1.0        : {int(np.sum(qs > 1.0))}")
    print(f"  q > 1.2        : {int(np.sum(qs > 1.2))}")
    print(f"  q > 1.5        : {int(np.sum(qs > 1.5))}")
    print(f"  q > 2.0        : {int(np.sum(qs > 2.0))}")

    # top-5 quality triples
    order = np.argsort(-qs)[:5]
    print()
    print("  top-5 highest quality:")
    print(f"    {'a':>6s}  {'b':>6s}  {'c':>7s}  "
          f"{'rad(abc)':>10s}  {'q':>7s}")
    for i in order:
        a, b, c, r = triples[i]
        print(f"    {a:>6d}  {b:>6d}  {c:>7d}  {r:>10d}  {qs[i]:>7.4f}")

    # ---------- 3. square-free density of abc ----------
    print()
    print("--- square-free density of a, b, c ---")
    mu = mobius_sieve(H)
    sf_triples = 0
    sf_a = sf_b = sf_c = 0
    for (a, b, c, r) in triples:
        if mu[a] != 0:
            sf_a += 1
        if mu[b] != 0:
            sf_b += 1
        if mu[c] != 0:
            sf_c += 1
        if mu[a] != 0 and mu[b] != 0 and mu[c] != 0:
            sf_triples += 1
    n = len(triples)
    print(f"  total triples       : {n}")
    print(f"  square-free a       : {sf_a/n:.6f}   (expect 6/π² = {DENSITY:.6f})")
    print(f"  square-free b       : {sf_b/n:.6f}")
    print(f"  square-free c       : {sf_c/n:.6f}")
    print(f"  all three sf        : {sf_triples/n:.6f}   "
          f"(expect (6/π²)³ = {DENSITY**3:.6f})")

    # ---------- 4. visualization ----------
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.36, wspace=0.32)

    # (a) Basel sum convergence
    ax = fig.add_subplot(gs[0, 0])
    Ks = np.arange(1, 10_000)
    Bs = np.array([basel_partial(K, mu_large[:K+1]) for K in Ks])
    ax.plot(Ks, Bs, color="#2c3e50", lw=1.0, label="B(K)")
    ax.axhline(DENSITY, color="#e74c3c", ls="--",
               label=f"6/π² = {DENSITY:.6f}")
    ax.set_xscale("log")
    ax.set_xlabel("K"); ax.set_ylabel("B(K)")
    ax.set_title("Basel–Möbius sum  →  6/π²")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (b) square-free density
    ax = fig.add_subplot(gs[0, 1])
    QKs = np.array([square_free_density(K, mu_large[:K+1]) for K in Ks])
    ax.plot(Ks, QKs, color="#16a085", lw=1.0, label="Q(K)/K")
    ax.axhline(DENSITY, color="#e74c3c", ls="--",
               label=f"6/π² = {DENSITY:.6f}")
    ax.set_xscale("log")
    ax.set_xlabel("K"); ax.set_ylabel("Q(K)/K")
    ax.set_title("Square-free density")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (c) quality distribution
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(qs, bins=60, color="#8e44ad", edgecolor="k", alpha=0.85)
    ax.axvline(1.0, color="#e74c3c", ls="--", label="q = 1")
    ax.axvline(qs.mean(), color="#16a085", ls=":",
               label=f"mean = {qs.mean():.3f}")
    ax.set_xlabel("q = log c / log rad(abc)")
    ax.set_ylabel("count")
    ax.set_title(f"Quality distribution  (H = {H})")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (d) quality vs c
    ax = fig.add_subplot(gs[1, 0])
    cs = np.array([c for (_, _, c, _) in triples])
    ax.scatter(cs, qs, s=3, alpha=0.4, color="#2980b9")
    ax.axhline(1.0, color="#e74c3c", ls="--", lw=1.0)
    ax.axhline(1.2, color="#e67e22", ls=":", lw=0.8)
    ax.set_xlabel("c"); ax.set_ylabel("q")
    ax.set_title("Quality vs height")
    ax.grid(alpha=0.3)

    # (e) rad(abc)/c vs c
    ax = fig.add_subplot(gs[1, 1])
    rs = np.array([r for (_, _, _, r) in triples])
    ratio = rs / cs
    ax.scatter(cs, ratio, s=3, alpha=0.4, color="#e67e22")
    ax.set_yscale("log")
    ax.set_xlabel("c"); ax.set_ylabel("rad(abc)/c")
    ax.set_title("Radical fraction  ·  heuristic: mostly O(1)")
    ax.grid(alpha=0.3)

    # (f) density bars
    ax = fig.add_subplot(gs[1, 2])
    labels = ["a", "b", "c", "all three"]
    vals = [sf_a / n, sf_b / n, sf_c / n, sf_triples / n]
    expect = [DENSITY, DENSITY, DENSITY, DENSITY ** 3]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, vals, w, color="#16a085",
           edgecolor="k", label="empirical")
    ax.bar(x + w/2, expect, w, color="#c0392b",
           edgecolor="k", alpha=0.7, label="Möbius prediction")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("density")
    ax.set_title(f"Square-free density of ABC components  (H = {H})")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

    plt.suptitle(
        "ABC conjecture  ·  Möbius density heuristic  ·  "
        f"6/π² = {DENSITY:.6f}",
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # ---------- 5. summary ----------
    print()
    print("--- summary ---")
    print(f"  6/π²                          = {DENSITY:.8f}")
    print(f"  (6/π²)³                       = {DENSITY**3:.8f}")
    print(f"  empirical all-square-free     = {sf_triples/n:.8f}")
    print(f"  empirical q > 1               = {int(np.sum(qs > 1.0))}/{n}")
    print(f"  empirical q > 1.2             = {int(np.sum(qs > 1.2))}/{n}")
    print(f"  heuristic: count should grow like H^(1/2) → bounded for fixed H")
    print()
    print("  NOTE: this is a numerical exploration of the heuristic.")
    print("        It does not prove ABC. The gap is uncontrolled")
    print("        Möbius correlation along a + b = c.")


if __name__ == "__main__":
    main()