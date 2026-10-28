#!/usr/bin/env python3
"""
mobius_ternary_pack.py
======================

Pack K/4 trits as a base-3 integer, normalize via finite-step log,
and encode/decode the normalized sum through a discrete-log group.

Pipeline
--------
    trits ∈ {0, 1, 2}^{K/4}
       │
       │  base-3 conversion
       ▼
    V = Σ trit_i · 3^i             V ∈ [0, 3^(K/4))
       │
       │  finite-step log
       ▼
    t = log(V) / log(A_STEP)       number of A_STEP multiplications
       │
       │  discrete-log encoding
       ▼
    h = g^V mod p                  group element  (p > 3^(K/4))

Capacity
--------
    N = K/4 trits holds  3^N distinct values  =  N · log₂(3) bits
    Direct trit packing:  5 trits / byte      →  ⌈N / 5⌉ bytes
    Discrete log:         ⌈log₂(p) / 8⌉ bytes where p > 3^N
                          ≈ ⌈N · log₂(3) / 8⌉  =  ⌈N / 5.05⌉

    Both give  ⌈K / 20⌉ bytes  for K trits.

    K=80  →  N=20  →  3^20 ≈ 3.49e9   →  4 bytes    (≈ IPv4)
    K=320 →  N=80  →  3^80 ≈ 1.48e38  →  16 bytes   (≈ IPv6)

Finite-step normalizer
----------------------
    A_STEP = ALPHA / 0.3628 ≈ 6.511
    t(V)   = log(V) / log(A_STEP)       "how many A_STEPs"
    V(t)   = A_STEP^t                    inverse

    t is a *continuous* normalized version of V, useful when the
    ternary sum is embedded in an analogue channel.  It does not
    change the byte count (the information is still the same) but
    it gives the value a stable numerical scale.

Discrete-log pack / unpack
--------------------------
    Given a prime p > 3^(K/4) and generator g of Z_p^*:

        pack:    h = g^V   (mod p)
        unpack:  V = BSGS_dlog(g, h, p)

    BSGS runs in O(√(3^N)) = O(1.732^N).  For N=20 that's ~60,000
    operations — small enough to verify the pipeline end-to-end.
"""

from __future__ import annotations
import math, random, struct
from dataclasses import dataclass
from typing import List, Tuple, Optional

# ---- project constants ----
PI      = math.pi
E       = math.e
ALPHA   = 1.0 / (PI - E)              # ≈ 2.362338
A_STEP  = ALPHA / 0.3628              # ≈ 6.511
DENSITY = 6.0 / (PI * PI)             # ≈ 0.607927


# ============================================================
#  Trit ↔ integer
# ============================================================
def trits_to_int(trits: List[int]) -> int:
    V = 0
    for t in trits:
        if t not in (0, 1, 2):
            raise ValueError(f"invalid trit {t}")
        V = V * 3 + t
    return V


def int_to_trits(V: int, n_trits: int) -> List[int]:
    if V < 0 or V >= 3 ** n_trits:
        raise ValueError(f"V={V} out of [0, 3^{n_trits})")
    trits = []
    for _ in range(n_trits):
        trits.append(V % 3)
        V //= 3
    return list(reversed(trits))


# ============================================================
#  Direct trit packing  ·  5 trits / byte
# ============================================================
def pack_trits_bytes(trits: List[int]) -> bytes:
    if not all(0 <= t <= 2 for t in trits):
        raise ValueError("trits must be in {0, 1, 2}")
    out = bytearray()
    for i in range(0, len(trits), 5):
        chunk = list(trits[i:i + 5])
        while len(chunk) < 5:
            chunk.append(0)
        v = 0
        for t in chunk:
            v = v * 3 + t
        out.append(v)                     # 0 ≤ v ≤ 242 < 256
    return bytes(out)


def unpack_trits_bytes(data: bytes, n_trits: int) -> List[int]:
    trits: List[int] = []
    for b in data:
        local: List[int] = []
        v = b
        for _ in range(5):
            local.append(v % 3)
            v //= 3
        trits.extend(reversed(local))
    return trits[:n_trits]


def bytes_for_trits(n_trits: int) -> int:
    """Direct trit packing bytes."""
    return (n_trits + 4) // 5


# ============================================================
#  Primality + generator finding  (bounded, works up to K≈80)
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
    """Trial divide by small primes; return (factors, cofactor)."""
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
    """Find generator of Z_p^* given partial factorization of p-1."""
    factors, cofactor = _trial_factor(p - 1)
    if cofactor > 1:
        factors.add(cofactor)
    for g in range(2, min(p, 10_000)):
        if all(pow(g, (p - 1) // q, p) != 1 for q in factors):
            return g
    return 2                                     # fallback


# ============================================================
#  Discrete-log group for ternary packing
# ============================================================
@dataclass
class TernaryDLGroup:
    """
    Discrete-log group over Z_p^* for ternary packing.

        n_trits  = K / 4
        space    = 3^n_trits
        p        = next prime > space
        g        = generator of Z_p^*
        n        = p − 1  (group order)
    """
    n_trits:   int
    p:         int
    g:         int
    n:         int
    space:     int
    bytes_:    int

    @classmethod
    def build(cls, n_trits: int) -> "TernaryDLGroup":
        space = 3 ** n_trits
        p = _next_prime(space)
        g = _find_generator(p)
        return cls(
            n_trits=n_trits, p=p, g=g, n=p - 1, space=space,
            bytes_=(p.bit_length() + 7) // 8,
        )

    # ---- pack / unpack ----
    def pack(self, V: int) -> int:
        if not (0 <= V < self.space):
            raise ValueError(f"V={V} outside [0, {self.space})")
        return pow(self.g, V % self.n, self.p)

    def unpack(self, h: int) -> int:
        return _bsgs(self.g, h % self.p, self.p, self.n)

    # ---- byte serialization of the group element ----
    def to_bytes(self, h: int) -> bytes:
        return h.to_bytes(self.bytes_, "big")

    def from_bytes(self, data: bytes) -> int:
        return int.from_bytes(data, "big")

    # ---- summary ----
    def summary(self) -> dict:
        info_bits = self.n_trits * math.log2(3)
        return dict(
            n_trits=self.n_trits,
            space=self.space,
            p=self.p,
            g=self.g,
            n=self.n,
            bytes_=self.bytes_,
            info_bits=info_bits,
            info_bytes_direct=bytes_for_trits(self.n_trits),
            info_bytes_dlog=self.bytes_,
        )


# ============================================================
#  BSGS discrete log  ·  O(√n)
# ============================================================
def _bsgs(g: int, h: int, p: int, n: int) -> int:
    """
    Baby-step giant-step: find x such that g^x ≡ h (mod p).

    Baby steps:  g^0, g^1, …, g^{m−1}          m = ⌈√n⌉
    Giant steps: h · g^{−mk}   for k = 0 … m
    """
    m = int(math.isqrt(n)) + 1

    baby: dict = {}
    cur = 1
    for i in range(m):
        if cur not in baby:
            baby[cur] = i
        cur = (cur * g) % p

    # g^{−m} = (g^{p−2})^m mod p   (Fermat inverse)
    g_inv = pow(g, p - 2, p)
    g_inv_m = pow(g_inv, m, p)

    gamma = h % p
    for j in range(m + 1):
        if gamma in baby:
            x = (baby[gamma] + j * m) % n
            if pow(g, x, p) == h:
                return x
        gamma = (gamma * g_inv_m) % p

    raise ValueError(f"no discrete log for h={h}")


# ============================================================
#  Finite-step normalizer
# ============================================================
def finite_step_normalize(V: int, a: float = A_STEP) -> float:
    """
    t(V) = log(V) / log(a)        "number of A_STEP multiplications"

    For V = 1, t = 0  (identity).
    For V = a^k, t = k.
    """
    if V <= 0:
        return 0.0
    return math.log(V) / math.log(a)


def finite_step_denormalize(t: float, a: float = A_STEP) -> float:
    """Inverse: V(t) = a^t  (continuous, not rounded)."""
    return a ** t


def finite_step_derivative(f, x: float, a: float = A_STEP) -> float:
    """D_a f(x) = (f(x+a) − f(x−a)) / (2a)  — for reference."""
    return (f(x + a) - f(x - a)) / (2 * a)


# ============================================================
#  Full pipeline wrapper
# ============================================================
class TernaryPacker:
    """
    End-to-end ternary packer for K trits → K/4 ternary digits.

        trits → V → finite-step-normalized t → dlog element h
        h → V → trits
    """
    def __init__(self, K: int):
        if K < 4:
            raise ValueError("K must be ≥ 4")
        self.K = K
        self.n_trits = K // 4
        self.group = TernaryDLGroup.build(self.n_trits)

    # ---- direct path ----
    def pack_direct(self, trits: List[int]) -> bytes:
        return pack_trits_bytes(trits)

    def unpack_direct(self, data: bytes) -> List[int]:
        return unpack_trits_bytes(data, self.n_trits)

    # ---- dlog path ----
    def pack_dlog(self, trits: List[int]) -> Tuple[int, float]:
        V = trits_to_int(trits)
        h = self.group.pack(V)
        t = finite_step_normalize(V)             # informational
        return h, t

    def unpack_dlog(self, h: int) -> Tuple[List[int], float]:
        V = self.group.unpack(h)
        trits = int_to_trits(V, self.n_trits)
        t = finite_step_normalize(V)
        return trits, t

    # ---- byte serialization for the wire ----
    def pack_dlog_bytes(self, trits: List[int]) -> bytes:
        h, _ = self.pack_dlog(trits)
        return self.group.to_bytes(h)

    def unpack_dlog_bytes(self, data: bytes) -> List[int]:
        h = self.group.from_bytes(data)
        trits, _ = self.unpack_dlog(h)
        return trits

    # ---- summary ----
    def capacity(self) -> dict:
        info = self.group.summary()
        return dict(
            K=self.K,
            n_trits=self.n_trits,
            trit_bytes=bytes_for_trits(self.n_trits),
            dlog_bytes=self.group.bytes_,
            info_bits=info["info_bits"],
            space=self.group.space,
        )


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("Möbius ternary packing  ·  K/4 trits  ·  finite-step log  ·  dlog")
    print("=" * 78)
    print(f"  α              = 1/(π − e)   = {ALPHA:.6f}")
    print(f"  A_STEP         = α / 0.3628  = {A_STEP:.6f}")
    print(f"  6/π²                          = {DENSITY:.6f}")
    print()

    # ---------- capacity table ----------
    print("--- capacity:  K/4 trits → bytes ---")
    print(f"  {'K':>4}  {'N=K/4':>5}  {'3^N':>15}  "
          f"{'info bits':>10}  {'trit bytes':>10}  "
          f"{'dlog bytes':>10}  {'note':>8}")
    notes = {80: "≈ IPv4", 320: "≈ IPv6"}
    for K in [20, 40, 80, 160, 320, 640]:
        N = K // 4
        space = 3 ** N
        info = N * math.log2(3)
        direct = bytes_for_trits(N)
        # dlog bytes = ceil(log2(p) / 8) where p > 3^N
        dlog = math.ceil(info / 8)                # lower bound ≈ direct
        print(f"  {K:>4}  {N:>5}  {space:>15}  "
              f"{info:>10.2f}  {direct:>10d}  {dlog:>10d}  "
              f"{notes.get(K, ''):>8}")
    print()

    print(f"  For all K, both packings give approximately  ⌈K/20⌉ bytes.")
    print(f"  A ternary digit carries log₂(3) ≈ 1.585 bits, so")
    print(f"  K/4 trits → (K/4)·1.585 = K·0.396 bits = ⌈K/20⌉ bytes.")
    print()

    # ---------- working example: K = 40 ----------
    K = 40
    packer = TernaryPacker(K=K)
    g = packer.group
    print(f"--- working example:  K = {K}  (N = {packer.n_trits} trits) ---")
    print(f"  p              = {g.p}")
    print(f"  g              = {g.g}")
    print(f"  p − 1          = {g.n}")
    print(f"  space (3^N)    = {g.space}")
    print(f"  p > 3^N        : {g.p > g.space}")
    print(f"  trit bytes     = {bytes_for_trits(packer.n_trits)}")
    print(f"  dlog bytes     = {g.bytes_}")
    print()

    # round-trip a few ternary values
    print(f"  {'V':>10}  {'trits (truncated)':>30}  "
          f"{'h = g^V mod p':>14}  {'unpacked V':>10}  ok")
    for V in [0, 1, 2, 100, 5000, g.space - 1]:
        trits = int_to_trits(V, packer.n_trits)
        h, t = packer.pack_dlog(trits)
        trits_back, t_back = packer.unpack_dlog(h)
        V_back = trits_to_int(trits_back)
        ok = (V_back == V) and (trits_back == trits)
        tri_s = "".join(str(t) for t in trits[:10])
        if packer.n_trits > 10:
            tri_s += "…"
        print(f"  {V:>10d}  {tri_s:>30s}  "
              f"{h:>14d}  {V_back:>10d}  {'✓' if ok else '✗'}")
    print()

    # finite-step normalization table
    print(f"--- finite-step normalization  (A_STEP = {A_STEP:.6f}) ---")
    print(f"  {'V':>10}  {'t = log(V)/log(A_STEP)':>24}  "
          f"{'A_STEP^t':>14}  {'round-trip err':>16}")
    for V in [1, 10, 100, 1000, 5000, 10000, 50000]:
        t = finite_step_normalize(V)
        V_back = finite_step_denormalize(t)
        err = abs(V_back - V)
        print(f"  {V:>10d}  {t:>24.6f}  {V_back:>14.4f}  {err:>16.6e}")
    print()

    # finite-step derivative of log
    print(f"--- finite-step derivative  D_a log(V) ---")
    print(f"  {'V':>8}  {'D_a log(V)':>14}  {'1/V':>14}  {'rel err':>12}")
    for V in [5, 50, 500, 5000]:
        D = finite_step_derivative(math.log, V)
        exact = 1.0 / V
        rel = abs(D - exact) / exact
        print(f"  {V:>8d}  {D:>14.6e}  {exact:>14.6e}  {rel:>12.4e}")
    print()

    # ---------- byte round-trip ----------
    print(f"--- byte round-trip  ·  K = {K} ---")
    trits = [2, 1, 0, 1, 2, 0, 1, 1, 2, 2]   # 10 trits
    h_bytes = packer.pack_dlog_bytes(trits)
    restored = packer.unpack_dlog_bytes(h_bytes)
    print(f"  trits in       = {trits}")
    print(f"  packed bytes   = {h_bytes.hex()}  ({len(h_bytes)} B)")
    print(f"  trits out      = {restored}")
    print(f"  round-trip     : {'OK' if restored == trits else 'FAIL'}")
    print()

    # ---------- larger K: show math only ----------
    print("--- larger K: byte capacity  (no live computation) ---")
    for K in [80, 160, 320, 640, 1280]:
        N = K // 4
        info_bits = N * math.log2(3)
        trit_bytes = bytes_for_trits(N)
        note = {80: "IPv4", 320: "IPv6"}.get(K, "")
        print(f"  K = {K:>5d}  N = {N:>3d}  "
              f"{info_bits:>8.1f} bits  →  {trit_bytes:>2d} B"
              f"{'  (' + note + ')' if note else ''}")
    print()

    # ---------- direct answer ----------
    print("=" * 78)
    print("ANSWER: how many bytes can be packed in a ternary sum?")
    print("=" * 78)
    print()
    print("  A ternary 'sum' of N = K/4 trits is a base-3 integer")
    print("  V ∈ [0, 3^N).  Its information content is N · log₂(3) bits.")
    print()
    print("  Direct ternary packing (5 trits per byte, since 3⁵ = 243 < 256):")
    print(f"      bytes = ⌈N / 5⌉ = ⌈K / 20⌉")
    print()
    print("  Discrete-log packing (h = g^V mod p with p ≈ 3^N):")
    print(f"      bytes = ⌈log₂(p) / 8⌉ = ⌈N · log₂(3) / 8⌉ ≈ ⌈N / 5.05⌉")
    print()
    print("  Both give the same integer answer  ⌈K / 20⌉ bytes.")
    print()
    print("  Concretely:")
    print(f"    K = 80   →  20 trits  →   4 bytes  ≈ IPv4 capacity")
    print(f"    K = 160  →  40 trits  →   8 bytes")
    print(f"    K = 320  →  80 trits  →  16 bytes  ≈ IPv6 capacity")
    print(f"    K = 640  → 160 trits  →  32 bytes")
    print()
    print("  The finite-step log  t = log(V) / log(A_STEP)  normalizes")
    print("  the sum to a continuous scale but does not change the")
    print("  byte count — the information is the same either way.")
    print()
    print("Done.")


if __name__ == "__main__":
    demo()