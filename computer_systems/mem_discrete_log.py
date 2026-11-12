#!/usr/bin/env python3
"""
mobius_m_discrete_log.py  ·  balanced ternary edition
=====================================================

Balanced ternary {-1, 0, +1} packed via UGC discrete log with
spectral-sum interpretation.

Balanced ternary packing
------------------------
    N trits  →  V = Σ t_n · 3^(N-1-n)   V ∈ [-(3^N-1)/2, +(3^N-1)/2]
    Offset   →  V' = V + (3^N-1)/2      V' ∈ [0, 3^N - 1]
    Pack     →  h = g^V' mod p          via UGCDiscreteLog
    Unpack   →  V' = dlog_g(h), then V = V' - (3^N-1)/2

    Trit value 0 → square-free / DC slot  (C[0] = +1)
    Trit value ±1 → negative harmonic    (C[±i] = −H_{|i|+1})

Spectral interpretation
-----------------------
    Each trit position n corresponds to a spectral coefficient at
    index i_n and the product series weight:

        α_n = α_min + (α_max − α_min) · n / K

    with α_min = 0.3628, α_max = 1/(π − e).

    Coefficient mapping:
        n = 0,  trit =  0   →   i =  0,    C = +1       (DC, square-free)
        n > 0,  trit = +1   →   i = +n,    C = −H_{n+1}  (harmonic)
        n > 0,  trit = −1   →   i = −n,    C = −H_{n+1}  (harmonic)
        n > 0,  trit =  0   →   coefficient absent (0)

    The "product series" is the full sequence  [α_0, α_1, …, α_{N-1}].

Imports (unchanged parents)
---------------------------
    ugc_discrete_log.UGCDiscreteLog
    complete_planes_spectral_sum.build_spectral_coefficients, alpha_at,
                                 ALPHA_MIN, ALPHA_MAX, A_STEP
    CPU_Lin_Alg.ALPHA, ALPHA_ASYM
"""

from __future__ import annotations
import math
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np

# ---- parent imports ----
from ugc_discrete_log import (
    UGCDiscreteLog,
    harmonic_numbers,
    ALPHA as UGC_ALPHA,
    GAMMA,
)
from complete_planes_spectral_sum import (
    build_spectral_coefficients,
    alpha_at,
    ALPHA_MIN,
    ALPHA_MAX,
    A_STEP,
)
from CPU_Lin_Alg import (
    ALPHA as CPU_ALPHA,
    ALPHA_ASYM,
    PI,
    E,
    DENSITY,
)


# ============================================================
#  Primality + generator
# ============================================================
def _is_prime(n: int, k: int = 20) -> bool:
    if n < 2:
        return False
    small = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small:
        if n == p:
            return True
        if n % p == 0:
            return False
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for _ in range(k):
        a = random.randint(2, n - 2)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def _next_prime(n: int) -> int:
    if n < 2:
        return 2
    if n % 2 == 0:
        n += 1
    while not _is_prime(n):
        n += 2
    return n


def _trial_factor(n: int, limit: int = 200_000) -> Tuple[set, int]:
    factors: set = set()
    d = 2
    while d * d <= n and d <= limit:
        if n % d == 0:
            factors.add(d)
            while n % d == 0:
                n //= d
        d += 1
    return factors, n


def _find_generator(p: int) -> int:
    factors, cofactor = _trial_factor(p - 1)
    if cofactor > 1:
        factors.add(cofactor)
    for g in range(2, min(p, 10_000)):
        if all(pow(g, (p - 1) // q, p) != 1 for q in factors):
            return g
    return 2


# ============================================================
#  Balanced trit ↔ integer
# ============================================================
def trits_to_int_balanced(trits: List[int]) -> int:
    """
    Balanced ternary  {−1, 0, +1}^N  →  V ∈ [-(3^N-1)/2, +(3^N-1)/2].
    """
    V = 0
    for t in trits:
        if t not in (-1, 0, 1):
            raise ValueError(f"invalid balanced trit {t!r}")
        V = V * 3 + t
    return V


def int_to_trits_balanced(V: int, N: int) -> List[int]:
    """V → balanced ternary trits (big-endian)."""
    half = (3 ** N - 1) // 2
    if not (-half <= V <= half):
        raise ValueError(f"V={V} out of balanced range [{-half}, {half}]")
    V_shifted = V + half
    trits: List[int] = []
    for _ in range(N):
        trits.append(V_shifted % 3 - 1)     # 0→−1, 1→0, 2→+1
        V_shifted //= 3
    return list(reversed(trits))


# ============================================================
#  BSGS fallback for large p (n > UGCDiscreteLog threshold)
# ============================================================
def _bsgs(g: int, h: int, p: int, n: int) -> int:
    m = int(math.isqrt(n)) + 1
    baby: Dict[int, int] = {}
    cur = 1
    for i in range(m):
        if cur not in baby:
            baby[cur] = i
        cur = (cur * g) % p
    g_inv_m = pow(pow(g, p - 2, p), m, p)
    gamma = h % p
    for j in range(m + 1):
        if gamma in baby:
            x = (baby[gamma] + j * m) % n
            if pow(g, x, p) == h:
                return x
        gamma = (gamma * g_inv_m) % p
    raise ValueError(f"no discrete log for h={h}")


# ============================================================
#  Balanced ternary group
# ============================================================
@dataclass
class BalancedTernaryGroup:
    """
    Z_p^*  packing of balanced ternary.  p is prime, p > 3^N − 1.
    """
    n_trits:   int
    p:         int
    g:         int
    n:         int
    space:     int
    half:      int
    bytes_:    int
    ugc:       Optional[UGCDiscreteLog] = None

    UGC_THRESHOLD = 200_000     # build full UGC table only if n ≤ this

    @classmethod
    def build(cls, n_trits: int) -> "BalancedTernaryGroup":
        space = 3 ** n_trits
        half = (space - 1) // 2
        p = _next_prime(space)
        g = _find_generator(p)
        n = p - 1
        bytes_ = (p.bit_length() + 7) // 8

        ugc = None
        if n <= cls.UGC_THRESHOLD:
            ugc = UGCDiscreteLog(p=p, g=g)

        return cls(
            n_trits=n_trits, p=p, g=g, n=n,
            space=space, half=half, bytes_=bytes_,
            ugc=ugc,
        )

    # ---- pack / unpack ----
    def pack_int(self, V_balanced: int) -> int:
        """Balanced integer V ∈ [-(3^N-1)/2, +(3^N-1)/2] → group element."""
        if not (-self.half <= V_balanced <= self.half):
            raise ValueError(f"V={V_balanced} outside balanced range")
        V_shifted = V_balanced + self.half
        return pow(self.g, V_shifted % self.n, self.p)

    def unpack_int(self, h: int) -> int:
        """Group element → balanced integer."""
        h %= self.p
        if h == 0:
            raise ValueError("h = 0 not in Z_p^*")
        if self.ugc is not None:
            x, _ = self.ugc.dlog(h)
            if x is None:
                x, _, _ = self.ugc.dlog_bsgs(h)
            if x is None:
                raise ValueError(f"no dlog for h={h}")
        else:
            x = _bsgs(self.g, h, self.p, self.n)
        return int(x) - self.half

    def to_bytes(self, h: int) -> bytes:
        return h.to_bytes(self.bytes_, "big")

    def from_bytes(self, data: bytes) -> int:
        return int.from_bytes(data, "big")

    # ---- summary ----
    def summary(self) -> Dict:
        info_bits = self.n_trits * math.log2(3)
        return dict(
            n_trits=self.n_trits,
            space=self.space,
            half=self.half,
            p=self.p,
            g=self.g,
            n=self.n,
            bytes_=self.bytes_,
            info_bits=info_bits,
            has_ugc=self.ugc is not None,
        )


# ============================================================
#  α product series
# ============================================================
def alpha_series(N: int, K: int,
                 alpha_min: float = ALPHA_MIN,
                 alpha_max: float = ALPHA_MAX) -> np.ndarray:
    """
    Product series  α_n = α_min + (α_max − α_min) · n / K
    for n = 0, 1, …, N−1.

    Position 0 → α_0 = α_min         (asymmetric / DC)
    Position N−1 → near α_max         (symmetric / harmonic)
    """
    n_arr = np.arange(N, dtype=float)
    return alpha_min + (alpha_max - alpha_min) * n_arr / max(K, 1)


# ============================================================
#  Spectral interpretation of a trit vector
# ============================================================
def spectral_interpretation(trits: List[int], K: int) -> List[Dict]:
    """
    Interpret a balanced trit vector as spectral coefficients.

    Each trit position n ≥ 0 contributes one entry:
        n = 0, trit = 0   →  i = 0,   C = +1       (DC, square-free)
        n > 0, trit = +1  →  i = +n,  C = −H_{n+1}  (harmonic)
        n > 0, trit = −1  →  i = −n,  C = −H_{n+1}  (harmonic)
        n > 0, trit = 0   →  absent

    α_i = α_min + (α_max − α_min) · |i| / K
    """
    H = harmonic_numbers(K + 2)
    rows: List[Dict] = []

    for n, t in enumerate(trits):
        if n == 0:
            if t == 0:
                i = 0
                C = 1.0                                    # DC
                present = True
            elif t == +1:
                i = 1
                C = -float(H[2])                           # -H_2
                present = True
            else:                                          # t == -1
                i = -1
                C = -float(H[2])
                present = True
        else:
            if t == 0:
                i = 0
                C = 0.0
                present = False
            elif t == +1:
                i = n
                C = -float(H[n + 1]) if n + 1 < len(H) else 0.0
                present = True
            else:                                          # t == -1
                i = -n
                C = -float(H[n + 1]) if n + 1 < len(H) else 0.0
                present = True

        alpha = alpha_at(i, K) if abs(i) <= K else alpha_at(K, K)
        rows.append(dict(
            n=n, trit=t, i=i, C=C, alpha=alpha,
            present=present,
        ))
    return rows


# ============================================================
#  BalancedTernaryPacker  ·  pack + spectral view
# ============================================================
class BalancedTernaryPacker:
    """
    End-to-end balanced ternary packer with spectral interpretation.
    """
    def __init__(self, K: int):
        if K < 4:
            raise ValueError("K must be ≥ 4")
        self.K = K
        self.n_trits = K // 4
        self.group = BalancedTernaryGroup.build(self.n_trits)
        self.alpha_N = alpha_series(self.n_trits, K)

    # ---- integer path ----
    def pack_int(self, trits: List[int]) -> int:
        return self.group.pack_int(trits_to_int_balanced(trits))

    def unpack_int(self, h: int) -> List[int]:
        V = self.group.unpack_int(h)
        return int_to_trits_balanced(V, self.n_trits)

    # ---- byte path ----
    def pack_bytes(self, trits: List[int]) -> bytes:
        return self.group.to_bytes(self.pack_int(trits))

    def unpack_bytes(self, data: bytes) -> List[int]:
        return self.unpack_int(self.group.from_bytes(data))

    # ---- spectral helpers ----
    def spectral(self, trits: List[int]) -> List[Dict]:
        return spectral_interpretation(trits, self.K)

    def product_series(self) -> np.ndarray:
        return self.alpha_N.copy()

    def capacity(self) -> Dict:
        info = self.group.summary()
        return dict(
            K=self.K,
            n_trits=self.n_trits,
            p=self.group.p,
            g=self.group.g,
            space=self.group.space,
            half=self.group.half,
            bytes_=self.group.bytes_,
            info_bits=info["info_bits"],
            has_ugc=info["has_ugc"],
        )


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("Balanced ternary  ·  UGC discrete log  ·  spectral α series")
    print("=" * 78)
    print(f"  α_min           = 0.3628")
    print(f"  α_max           = 1/(π−e) = {ALPHA_MAX:.6f}")
    print(f"  A_STEP          = {A_STEP:.6f}")
    print(f"  6/π²            = {DENSITY:.6f}")
    print()

    # ---------------------------------------------------------
    #  1 · capacity table for balanced ternary
    # ---------------------------------------------------------
    print("=" * 78)
    print("1  ·  capacity  ·  N = K/4 balanced trits")
    print("=" * 78)
    print(f"  {'K':>4}  {'N':>3}  {'3^N':>20}  "
          f"{'info bits':>10}  {'bytes':>6}  {'UGC?':>6}")
    print("  " + "-" * 60)
    for K in [20, 40, 80, 160, 320]:
        N = K // 4
        space = 3 ** N
        info = N * math.log2(3)
        bytes_ = (space.bit_length() + 7) // 8 + 1
        ugc = "yes" if (space - 1) <= BalancedTernaryGroup.UGC_THRESHOLD else "BSGS"
        print(f"  {K:>4}  {N:>3}  {space:>20,}  "
              f"{info:>10.2f}  {bytes_:>6}  {ugc:>6}")
    print()

    # ---------------------------------------------------------
    #  2 · working packer  K = 40
    # ---------------------------------------------------------
    K = 40
    packer = BalancedTernaryPacker(K=K)
    g = packer.group
    cap = packer.capacity()
    print("=" * 78)
    print(f"2  ·  K = {K}  ·  N = {packer.n_trits} balanced trits")
    print("=" * 78)
    print(f"  p                = {cap['p']}")
    print(f"  g                = {cap['g']}")
    print(f"  p − 1            = {g.n}")
    print(f"  space (3^N)      = {cap['space']:,}")
    print(f"  half             = {cap['half']:,}")
    print(f"  balanced range   = [{-cap['half']}, +{cap['half']}]")
    print(f"  p > space        : {g.p > cap['space']}")
    print(f"  dlog bytes       = {cap['bytes_']}")
    print(f"  UGC table built  : {cap['has_ugc']}")
    print()

    # ---------------------------------------------------------
    #  3 · product series of α
    # ---------------------------------------------------------
    print("=" * 78)
    print(f"3  ·  product series  α_n = α_min + (α_max − α_min) · n / K")
    print("=" * 78)
    alphas = packer.product_series()
    print(f"  {'n':>3}  {'α_n':>10}")
    print("  " + "-" * 16)
    for n in range(packer.n_trits):
        print(f"  {n:>3d}  {alphas[n]:>10.6f}")
    print()

    # ---------------------------------------------------------
    #  4 · round-trip with balanced trits
    # ---------------------------------------------------------
    print("=" * 78)
    print("4  ·  round-trip  ·  balanced trits → h → balanced trits")
    print("=" * 78)
    test_vectors = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],          # all zero → V = 0
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],          # V = 3^9
        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0],         # V = -3^9
        [1, -1, 1, -1, 1, -1, 1, -1, 1, -1],     # alternating
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],          # V = (3^10 - 1)/2 (max)
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],# V = -(3^10 - 1)/2 (min)
    ]
    print(f"  {'V':>8}  {'trits':>28}  {'h = g^V':>12}  "
          f"{'round-trip':>10}")
    print("  " + "-" * 68)
    for trits in test_vectors:
        V = trits_to_int_balanced(trits)
        h = packer.pack_int(trits)
        trits_back = packer.unpack_int(h)
        ok = trits_back == trits
        tri_s = "".join(f"{t:+d}" for t in trits)
        print(f"  {V:>8d}  {tri_s:>28s}  {h:>12d}  "
              f"{'✓' if ok else '✗':>10}")
    print()

    # ---------------------------------------------------------
    #  5 · spectral interpretation
    # ---------------------------------------------------------
    print("=" * 78)
    print("5  ·  spectral interpretation  ·  C[0]=+1, C[±i]=−H_{|i|+1}")
    print("=" * 78)
    trits = [0, 1, -1, 0, 1, -1, 0, 1, -1, 0]
    print(f"  trits = {trits}")
    print()
    rows = packer.spectral(trits)
    print(f"  {'n':>3}  {'trit':>5}  {'i':>5}  {'C[i]':>10}  "
          f"{'α_i':>10}  {'present':>8}")
    print("  " + "-" * 48)
    for r in rows:
        print(f"  {r['n']:>3d}  {r['trit']:>+5d}  {r['i']:>+5d}  "
              f"{r['C']:>+10.6f}  {r['alpha']:>10.6f}  "
              f"{'yes' if r['present'] else 'no':>8}")
    present_C = [r['C'] for r in rows if r['present']]
    print()
    print(f"  sum C[i] over present: {sum(present_C):+.6f}")
    print(f"  i=0 present (DC):      "
          f"{any(r['i'] == 0 and r['present'] for r in rows)}")
    print()

    # ---------------------------------------------------------
    #  6 · byte round-trip
    # ---------------------------------------------------------
    print("=" * 78)
    print("6  ·  byte round-trip")
    print("=" * 78)
    trits = [1, -1, 0, 1, 0, -1, 1, 0, -1, 1]
    print(f"  trits in       = {trits}")
    h_bytes = packer.pack_bytes(trits)
    trits_out = packer.unpack_bytes(h_bytes)
    print(f"  packed bytes   = {h_bytes.hex()}  ({len(h_bytes)} B)")
    print(f"  trits out      = {trits_out}")
    print(f"  round-trip     : "
          f"{'OK' if trits_out == trits else 'FAIL'}")
    print()

    # ---------------------------------------------------------
    #  7 · larger K demonstration (no live UGC)
    # ---------------------------------------------------------
    print("=" * 78)
    print("7  ·  larger K  ·  UGC table becomes infeasible, BSGS only")
    print("=" * 78)
    for K in [80, 160, 320]:
        N = K // 4
        space = 3 ** N
        p = _next_prime(space)
        bytes_ = (p.bit_length() + 7) // 8
        info_bits = N * math.log2(3)
        print(f"  K = {K:>4d}  N = {N:>3d}  space = 3^{N} = {space:,.0f}")
        print(f"    p = {p:,.0f}  bytes = {bytes_}  "
              f"info = {info_bits:.1f} bits  "
              f"(UGC infeasible → BSGS O(√n))")
    print()

    # ---------------------------------------------------------
    #  Summary
    # ---------------------------------------------------------
    print("=" * 78)
    print("Summary")
    print("=" * 78)
    print("  Balanced ternary:")
    print("    trits ∈ {−1, 0, +1}^N")
    print("    V      = Σ t_n · 3^(N−1−n)")
    print("    V ∈ [−(3^N−1)/2, +(3^N−1)/2]")
    print("    V'     = V + (3^N−1)/2  ∈ [0, 3^N)")
    print("    pack   : h = g^V' mod p     (UGCDiscreteLog)")
    print("    unpack : V' = BSGS or table, V = V' − half")
    print()
    print("  Spectral coefficients (via complete_planes_spectral_sum):")
    print("    i =  0       →  C = +1         (real positive DC, square-free)")
    print("    i = ±n       →  C = −H_{n+1}   (negative harmonic)")
    print("    α_n          = α_min + (α_max − α_min) · n / K")
    print("    α_min = 0.3628,  α_max = 1/(π−e) ≈ 2.362")
    print()
    print("  Encoding uses UGCDiscreteLog when p ≤ 200,000;")
    print("  BSGS O(√n) otherwise.")
    print()
    print("  Byte capacity (K/4 trits → K/20 bytes):")
    for K in (40, 80, 160, 320):
        N = K // 4
        space = 3 ** N
        print(f"    K = {K:>3d}  →  {N:>3d} trits  →  "
              f"{(space.bit_length() + 7) // 8} bytes")
    print()
    print("Done.")


if __name__ == "__main__":
    demo()