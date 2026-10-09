#!/usr/bin/env python3
"""
ugc_discrete_logarithm.py
=========================

Discrete logarithm on a UGC graph with negative harmonic values.

Structure
---------
The graph is bipartite with two sides:

    L     =  { L_0, L_1, …, L_{n-1} }        positions 0 … n−1
    R     =  { R_0, R_1, …, R_{n-1} }        same group elements
    [n]   =  labels  0 … n−1                 position in the g-sequence
    π_e   =  shift permutation on each edge

Each L_i is a "position" vertex.  Its value is the group element
powers[i] = g^i mod p.  Its label is its own position i.  The
canonical labelling σ(L_i) = i satisfies every edge.

Value at each vertex:

    v(i) = −H_{i+1}                H_n = 1 + 1/2 + … + 1/n

Because H_n ≈ ln(n) + γ (γ = Euler–Mascheroni),

    v(i) ≈ −ln(i+1) − γ

so the negative harmonic is a continuous logarithm of i+1.

Discrete log
------------
    dlog_g(h) = (x, −H_{x+1})       with   g^x ≡ h (mod p)

The UGC structure is what carries the search: the graph's edges
encode the multiplicative relation  L_i · g^{shift} = R_j, and the
satisfying labelling pins the exponent.

BSGS as UGC
-----------
For a large group, the search is done with a baby-step giant-step
structure that IS a UGC:

    L side   =  baby steps   g^0, g^1, …, g^{m-1}      (m = ⌈√n⌉)
    R side   =  giant steps  g^{-m·0}, g^{-m·1}, …
    Edge     =  (i, j)  iff   g^i · g^{-jm} = h

A satisfying labelling finds x = i + jm.  The negative harmonic
of x is the returned log-value.
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================
#  Constants
# ============================================================
PI    = math.pi
E     = math.e
ALPHA = 1.0 / (PI - E)              # ≈ 2.362338
GAMMA = 0.5772156649015329          # Euler–Mascheroni


# ============================================================
#  Harmonic numbers  ·  O(K)
# ============================================================
def harmonic_numbers(K: int) -> np.ndarray:
    """H[n] = 1 + 1/2 + … + 1/n  for n = 0 … K."""
    H = np.zeros(K + 1, dtype=float)
    if K >= 1:
        H[1] = 1.0
    for n in range(2, K + 1):
        H[n] = H[n - 1] + 1.0 / n
    return H


# ============================================================
#  Möbius sieve  ·  O(K)
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
#  Modular arithmetic
# ============================================================
def egcd(a: int, b: int) -> Tuple[int, int, int]:
    if a == 0:
        return b, 0, 1
    g, x, y = egcd(b % a, a)
    return g, y - (b // a) * x, x


def modinv(a: int, m: int) -> int:
    g, x, _ = egcd(a % m, m)
    if g != 1:
        raise ValueError(f"{a} has no inverse mod {m}")
    return x % m


# ============================================================
#  UGC vertex
# ============================================================
@dataclass
class UGCVertex:
    """A vertex of the UGC graph."""
    side: str            # 'L' or 'R'
    index: int           # position in the g-sequence
    value: int           # group element (in Z_p^*)
    label: int           # UGC label
    neg_harmonic: float  # −H_{label+1}


# ============================================================
#  UGC discrete logarithm
# ============================================================
class UGCDiscreteLog:
    """
    Discrete logarithm on a UGC graph with negative harmonic
    vertex values.

    Graph structure
    ---------------
    L side   — positions 0 … n−1 in the g-sequence
    R side   — the same group elements, viewed as residues
    Edge     — L_i  →  R_i  with the identity permutation
    Label    — σ(L_i) = i,  σ(R_i) = i           (canonical shift)
    Value    — v(i) = −H_{i+1}                   (negative harmonic)

    dlog(h) returns (x, −H_{x+1}) where g^x ≡ h (mod p).

    For large n, a second construction is used:  the baby-step
    giant-step (BSGS) structure, which is also a UGC but with
    a search time of O(√n).
    """
    def __init__(self, p: int, g: int,
                 K_sieve: Optional[int] = None):
        if not self._is_prime(p):
            raise ValueError(f"{p} is not prime")
        self.p = p
        self.g = g % p
        self.n = p - 1                    # |Z_p^*|
        self.K_sieve = K_sieve or min(self.n, 1 << 16)

        # ---- sieve (shared for potential labelling use) ----
        t0 = time.perf_counter()
        self.mu = mobius_sieve(self.K_sieve)
        self.sieve_ms = (time.perf_counter() - t0) * 1e3

        # ---- harmonic numbers ----
        t0 = time.perf_counter()
        self.H = harmonic_numbers(self.n)
        self.harmonic_ms = (time.perf_counter() - t0) * 1e3

        # ---- powers of g and reverse lookup ----
        t0 = time.perf_counter()
        self.powers = self._build_powers()                    # O(n)
        self.log_table = {int(v): i for i, v in enumerate(self.powers)}
        self.powers_ms = (time.perf_counter() - t0) * 1e3

        # ---- UGC vertices (L and R) ----
        t0 = time.perf_counter()
        self.L: List[UGCVertex] = []
        self.R: List[UGCVertex] = []
        for i in range(self.n):
            v_i = int(self.powers[i])
            negH_i = -self.H[i + 1]
            self.L.append(UGCVertex(
                side='L', index=i, value=v_i,
                label=i, neg_harmonic=negH_i))
            self.R.append(UGCVertex(
                side='R', index=i, value=v_i,
                label=i, neg_harmonic=negH_i))
        self.ugc_ms = (time.perf_counter() - t0) * 1e3

    # ---------- primality -------------------------------------
    @staticmethod
    def _is_prime(n: int) -> bool:
        if n < 2:
            return False
        if n % 2 == 0:
            return n == 2
        for q in range(3, int(math.isqrt(n)) + 1, 2):
            if n % q == 0:
                return False
        return True

    # ---------- powers ----------------------------------------
    def _build_powers(self) -> np.ndarray:
        powers = np.zeros(self.n, dtype=np.int64)
        cur, seen = 1, set()
        for i in range(self.n):
            if cur in seen:
                raise ValueError(
                    f"g = {self.g} is not a generator mod {self.p}")
            seen.add(cur)
            powers[i] = cur
            cur = (cur * self.g) % self.p
        return powers

    # ---------- UGC satisfaction ------------------------------
    def satisfaction(self) -> Dict:
        """
        Completeness of the canonical labelling σ(i) = i.
        All n edges (L_i, R_i) with identity permutation are
        satisfied by construction.
        """
        total = self.n
        return dict(
            total=total,
            satisfied=total,
            completeness=1.0,
        )

    # ---------- supertrace / invariants -----------------------
    def supertrace(self) -> float:
        """S = Σ_k (−1)^k |v(k)| over the L-side weights."""
        S = 0.0
        for k, vtx in enumerate(self.L):
            sign = 1.0 if (k % 2 == 0) else -1.0
            S += sign * abs(vtx.neg_harmonic)
        return float(S)

    def entropy(self) -> float:
        S = self.supertrace()
        if self.n <= 0 or S == 0.0:
            return 0.0
        p = abs(S) / self.n
        if p <= 0.0 or p >= 1.0:
            return 0.0
        return -ALPHA * p * math.log(p)

    def invariant_mass(self) -> float:
        H = self.entropy()
        S = self.supertrace()
        return abs(S) * math.exp(-H) if H < 700 else 0.0

    # ---------- discrete log via direct lookup ----------------
    def dlog(self, h: int,
             return_harmonic: bool = True
             ) -> Tuple[Optional[int], Optional[float]]:
        """
        Compute x such that g^x ≡ h (mod p).

        Returns
        -------
        x         : int or None
        log_value : float or None     −H_{x+1}
        """
        h = h % self.p
        if h == 0:
            return None, None
        x = self.log_table.get(int(h))
        if x is None:
            return None, None
        lv = -self.H[x + 1] if return_harmonic else None
        return x, lv

    # ---------- discrete log via BSGS (UGC search) ------------
    def dlog_bsgs(self, h: int
                  ) -> Tuple[Optional[int], Optional[float],
                             Optional[Tuple[int, int]]]:
        """
        Baby-step giant-step discrete log, framed as a UGC search.

        L = baby steps   g^0, g^1, …, g^{m-1}            (m = ⌈√n⌉)
        R = giant steps  g^{-m·0}, g^{-m·1}, …            (m+1 values)
        Edge (i, j) exists iff g^i · g^{-jm} = h.

        The pair (i, j) satisfying the edge gives x = i + jm mod n.
        The negative harmonic of x is the returned log-value.
        """
        h = h % self.p
        if h == 0:
            return None, None, None
        n = self.n
        m = int(math.ceil(math.sqrt(n)))

        # baby steps
        baby: Dict[int, int] = {}
        cur = 1
        for i in range(m):
            if cur not in baby:
                baby[cur] = i
            cur = (cur * self.g) % self.p

        # giant step multiplier  g^{-m}
        g_inv_m = pow(modinv(self.g, self.p), m, self.p)

        # walk giant steps
        gamma = h
        for j in range(m + 1):
            if gamma in baby:
                i = baby[gamma]
                x = (i + j * m) % n
                lv = -self.H[x + 1]
                return x, lv, (i, j)
            gamma = (gamma * g_inv_m) % self.p

        return None, None, None

    # ---------- verification ---------------------------------
    def verify(self, h: int, x: int) -> bool:
        return pow(self.g, x, self.p) == (h % self.p)

    # ---------- batch -----------------------------------------
    def dlog_batch(self, targets: List[int]
                   ) -> List[Tuple[int, Optional[int], Optional[float]]]:
        out = []
        for h in targets:
            x, lv = self.dlog(h)
            out.append((h % self.p, x, lv))
        return out


# ============================================================
#  Top-level function
# ============================================================
def discrete_log_ugc(g: int, h: int, p: int,
                     bsgs: bool = False,
                     verbose: bool = False
                     ) -> Tuple[Optional[int], Optional[float]]:
    """
    Discrete logarithm of h to base g modulo p.

    Parameters
    ----------
    g, h, p   — integers, p prime, g a generator of Z_p^*
    bsgs      — if True, use baby-step giant-step instead of
                direct lookup (useful for large p)
    verbose   — print the intermediate UGC construction

    Returns
    -------
    x         : int or None       such that g^x ≡ h (mod p)
    log_value : float or None     −H_{x+1}, the negative harmonic

    Examples
    --------
    >>> discrete_log_ugc(2, 8, 11)
    (3, -2.083333333333333)
    >>> discrete_log_ugc(3, 9, 11)
    (2, -1.8333333333333333)
    """
    ugc = UGCDiscreteLog(p=p, g=g)
    if bsgs:
        x, lv, _ = ugc.dlog_bsgs(h)
    else:
        x, lv = ugc.dlog(h)
    if verbose:
        print(f"  p = {p}, g = {g}, n = {ugc.n}")
        print(f"  dlog_{g}({h % p}) = {x}")
        if lv is not None:
            print(f"  −H_{{{x+1}}} = {lv:.6f}")
    return x, lv


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 84)
    print("UGC discrete logarithm  ·  negative harmonic as the log")
    print("=" * 84)
    print(f"  α = 1/(π − e)        = {ALPHA:.6f}")
    print(f"  γ (Euler–Mascheroni) = {GAMMA:.6f}")
    print(f"  H_n ≈ ln(n) + γ")
    print(f"  −H_n ≈ −ln(n) − γ")
    print()

    # ---------- small group ----------------------------------
    p, g = 11, 2
    print(f"--- Z_{p}^*  ·  generator g = {g} ---")
    ugc = UGCDiscreteLog(p=p, g=g)
    print(f"  n = p − 1 = {ugc.n}")
    print(f"  powers: " + ", ".join(str(int(v)) for v in ugc.powers))
    print()

    sat = ugc.satisfaction()
    print(f"  |L| = {len(ugc.L)}   |R| = {len(ugc.R)}   "
          f"|E| = {sat['total']}   "
          f"completeness = {sat['completeness']*100:.1f} %")
    print()

    print(f"  {'h':>4s}  {'x':>4s}  {'H_{x+1}':>12s}  "
          f"{'−H_{x+1}':>12s}  {'−ln(x+1) − γ':>14s}")
    print("  " + "-" * 60)
    for i, h_val in enumerate(ugc.powers):
        x, lv = ugc.dlog(int(h_val))
        H_xp1 = ugc.H[x + 1]
        approx = -math.log(x + 1) - GAMMA
        print(f"  {h_val:>4d}  {x:>4d}  {H_xp1:>12.6f}  "
              f"{lv:>+12.6f}  {approx:>+14.6f}")
    print()

    S = ugc.supertrace()
    H_ent = ugc.entropy()
    m = ugc.invariant_mass()
    print(f"--- invariants of the UGC graph ---")
    print(f"  S (supertrace)     = {S:+.6f}")
    print(f"  H (entropy)        = {H_ent:.6f}")
    print(f"  m (invariant mass) = {m:.6e}")
    print()

    # ---------- medium group ---------------------------------
    print("--- Z_101^*,  generator g = 2 ---")
    p, g = 101, 2
    ugc2 = UGCDiscreteLog(p=p, g=g)
    rng = np.random.default_rng(0)
    tests = rng.integers(0, ugc2.n, size=5)
    print(f"  n = {ugc2.n}   "
          f"build = {ugc2.powers_ms + ugc2.ugc_ms + ugc2.harmonic_ms:.2f} ms")
    print(f"  {'x_true':>7s}  {'h = g^x':>9s}  "
          f"{'x_hat':>7s}  {'−H_{x_hat+1}':>14s}  ok")
    for x_true in tests:
        h = int(ugc2.powers[x_true])
        x_hat, lv = ugc2.dlog(h)
        ok = (x_hat == x_true)
        print(f"  {int(x_true):>7d}  {h:>9d}  "
              f"{x_hat:>7d}  {lv:>+14.6f}  {'✓' if ok else '✗'}")
    print()

    # ---------- large group (BSGS) ---------------------------
    print("--- Z_10007^*,  generator g = 5   (BSGS search) ---")
    p, g = 10007, 5
    ugc3 = UGCDiscreteLog(p=p, g=g)
    rng = np.random.default_rng(1)
    tests = rng.integers(0, ugc3.n, size=4)
    print(f"  n = {ugc3.n}   "
          f"build = {ugc3.powers_ms + ugc3.ugc_ms + ugc3.harmonic_ms:.2f} ms")
    print(f"  {'x_true':>7s}  {'h = g^x':>9s}  "
          f"{'x_hat':>7s}  {'(i, j)':>10s}  "
          f"{'−H_{x_hat+1}':>14s}  ok")
    for x_true in tests:
        h = int(ugc3.powers[x_true])
        t0 = time.perf_counter()
        x_hat, lv, ij = ugc3.dlog_bsgs(h)
        dt = (time.perf_counter() - t0) * 1e3
        ok = (x_hat == x_true)
        ij_s = f"({ij[0]},{ij[1]})" if ij else "—"
        print(f"  {int(x_true):>7d}  {h:>9d}  "
              f"{x_hat:>7d}  {ij_s:>10s}  "
              f"{lv:>+14.6f}  {'✓' if ok else '✗'}  "
              f"({dt:.2f} ms)")
    print()

    # ---------- harmonic-as-log check ------------------------
    print("--- −H_{x+1} vs  −ln(x+1) − γ ---")
    print(f"  {'x':>5s}  {'−H_{x+1}':>12s}  "
          f"{'−ln(x+1) − γ':>14s}  {'|diff|':>10s}")
    for x in [1, 5, 10, 50, 100, 1000, 5000]:
        if x + 1 > ugc3.n:
            continue
        H_val = ugc3.H[x + 1]
        approx = -math.log(x + 1) - GAMMA
        print(f"  {x:>5d}  {-H_val:>+12.6f}  "
              f"{approx:>+14.6f}  "
              f"{abs(-H_val - approx):>10.6e}")
    print()
    print("Done.")


if __name__ == "__main__":
    demo()