#!/usr/bin/env python3
"""
compress_dram.py
================

Memory compression for `DRAM.py` using `factor.py` as the extraction engine.

Uncompressed form       Compressed form                  Ratio
-----------------       -----------------               -----
G  ∈ ℝ^{n × n}          A = [v_i · v_{i+1}]  (n entries)   n : 1
G_ij = κ_ij v_i v_j     +  addrs  (n entries)
                        +  v_1    (1, only when n even)

Extraction via factor.py
------------------------
For odd  n :  v_1 = isqrt( Π_odd A_i / Π_even A_i )     — one factor.py call
For even n :  v_1 stored directly (1 extra integer)
Then     :  v_{i+1} = A_i / v_i   — walk around the cycle

The extraction is a **perfect-square test through factor.py**:
    factor.py factorises Π_odd / Π_even into primes
    if every exponent is even, the square root is v_1
    else the compressed form is corrupted / mismatched

Why this is a real compression
------------------------------
The full coupling matrix G is n² entries — that is what the DRAM
actually "carries".  The A signature is only n entries.  The
decompressor rebuilds G from A in O(n) after the O(√v_1²) factor.py
extraction, so total decompression is O(n + √ΠA).

Robustness
----------
Any single A_i corruption breaks:
    · integrality of Π_odd / Π_even           →  caught at division
    · perfect-square requirement for v_1      →  caught at factor.py
    · cycle closure A_{n−1} = v_{n−1}·v_0     →  caught at walk-around
"""

from __future__ import annotations
import math
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from DRAM import FermionDRAM, travel_lengths, mobius_sieve
from factor import factor_full


# ============================================================
#  isqrt via factor.py
# ============================================================
def isqrt_via_factor(n: int) -> Optional[int]:
    """
    Return ⌊√n⌋ if n is a perfect square, else None.

    Uses factor.py's `factor_full` to decompose n into primes, then
    checks that every exponent is even and takes half-exponents.
    """
    if n <= 0:
        return None
    factors = factor_full(n)
    if not factors:
        return None
    cnt = Counter(factors)
    result = 1
    for p, e in cnt.items():
        if e % 2 != 0:
            return None
        result *= p ** (e // 2)
    return result if result * result == n else None


# ============================================================
#  Coupling-factor compressor
# ============================================================
class CouplingFactorCompressor:
    """
    Compress the DRAM coupling matrix G into the cyclic signature A.

    See module docstring for the derivation.
    """

    def __init__(self, dram: FermionDRAM):
        self.dram = dram
        self.addrs: List[int] = dram.address_list()
        self.n = len(self.addrs)
        self.v_orig = np.array(
            [dram.cells[a].value for a in self.addrs], dtype=np.int64)
        self.kappa = self._compute_kappa()

    # --------------------------------------------------------
    def _compute_kappa(self) -> np.ndarray:
        t = np.array([self.dram.cells[a].t for a in self.addrs])
        r = np.array([self.dram.cells[a].r for a in self.addrs])
        L = travel_lengths(t, r)
        L_ref = float(np.median(L[L > 0])) if np.any(L > 0) else 1.0
        K = np.sqrt(np.maximum(np.outer(L, L), 0.0)) / L_ref
        K = np.clip(K, 0.0, 1.0)
        np.fill_diagonal(K, 0.0)
        return K

    # --------------------------------------------------------
    def compress(self) -> Dict:
        """
        Cyclic-product signature:
            A[i] = G_{i, i+1} / κ_{i, i+1} = v_i · v_{i+1}
        """
        n = self.n
        A = np.zeros(n, dtype=np.int64)
        for i in range(n):
            j = (i + 1) % n
            A[i] = int(self.v_orig[i]) * int(self.v_orig[j])
        v_1_extra = int(self.v_orig[0]) if n % 2 == 0 else None
        return dict(
            addrs=list(self.addrs),
            A=A.tolist(),
            v_1=v_1_extra,
            n=n,
        )

    # --------------------------------------------------------
    def decompress(self, compressed: Dict) -> List[int]:
        """
        Recover v from A, using factor.py for the perfect-square test.
        """
        addrs = compressed["addrs"]
        A = compressed["A"]
        n = compressed["n"]
        v_1_extra = compressed["v_1"]

        if n < 3:
            if n == 1:
                return [int(v_1_extra) if v_1_extra is not None else A[0]]
            # n == 2 : A[0] == A[1], both = v_0 v_1
            return [int(v_1_extra), A[0] // int(v_1_extra)]

        # ---- recover v_1 ----
        if n % 2 == 1:
            num = 1
            den = 1
            for i in range(n):
                if i % 2 == 0:
                    num *= A[i]
                else:
                    den *= A[i]
            if den == 0:
                raise ValueError("degenerate — zero cyclic product")
            if num % den != 0:
                raise ValueError(
                    f"cyclic ratio not integer: num={num}, den={den}")
            ratio = num // den
            v_1 = isqrt_via_factor(ratio)
            if v_1 is None:
                raise ValueError(
                    f"cyclic ratio {ratio} is not a perfect square")
        else:
            v_1 = v_1_extra
            if not v_1:
                raise ValueError("even n requires stored v_1")

        # ---- walk the cycle ----
        v = [int(v_1)]
        for i in range(n - 1):
            if v[-1] == 0:
                raise ValueError(f"zero divisor at step {i}")
            if A[i] % v[-1] != 0:
                raise ValueError(
                    f"A[{i}]={A[i]} not divisible by v[{i}]={v[-1]}")
            v.append(A[i] // v[-1])

        # ---- close the cycle ----
        if A[n - 1] != v[-1] * v[0]:
            raise ValueError(
                f"cycle closure failed: A[n-1]={A[n-1]}, "
                f"v[n-1]·v[0]={v[-1]*v[0]}")

        return v

    # --------------------------------------------------------
    def reconstruct_dram(self, v: List[int]) -> FermionDRAM:
        new_dram = FermionDRAM(
            K=self.dram.K, coupling_strength=self.dram.coupling_strength)
        for addr, val in zip(self.addrs, v):
            new_dram.write(addr, int(val))
        return new_dram

    # --------------------------------------------------------
    def stats(self, compressed: Dict) -> Dict:
        n = self.n
        full_entries = n * n
        comp_entries = n + (1 if compressed["v_1"] is not None else 0)
        return dict(
            n=n,
            full_G_entries=full_entries,
            compressed_entries=comp_entries,
            ratio=full_entries / comp_entries if comp_entries else 0,
            v_1_stored=(compressed["v_1"] is not None),
        )


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("DRAM compression via Coupling-Factor Encoding")
    print("=" * 78)

    # ---------- build DRAM ----------
    K = 60
    dram = FermionDRAM(K=K, coupling_strength=1)
    addrs = dram.address_list()
    n = len(addrs)
    print(f"  K                  = {K}")
    print(f"  live cells  n      = {n}")
    print(f"  density            = {n / K:.4f}  "
          f"(6/π² = {6 / math.pi**2:.4f})")
    print(f"  n odd/even         = {'odd' if n % 2 else 'even'}")
    print()

    # ---------- write values ----------
    rng = np.random.default_rng(7)
    for a in addrs:
        dram.write(a, int(rng.integers(1, 32)))
    print(f"  wrote random 1..31 to every live cell")
    print()

    # ---------- compress ----------
    comp = CouplingFactorCompressor(dram)
    compressed = comp.compress()
    A_list = compressed["A"]
    s = comp.stats(compressed)

    # ---------- examine the coupling signature via factor.py ----------
    print("--- coupling signature A_i = v_i · v_{i+1}  (first 8) ---")
    print(f"  {'i':>3}  {'A_i':>6}  {'v_i':>4}  {'v_i+1':>5}  "
          f"{'factor.py(A_i)':<24}")
    for i in range(min(n, 8)):
        v_here = int(comp.v_orig[i])
        v_next = int(comp.v_orig[(i + 1) % n])
        facts = factor_full(A_list[i])
        f_str = "·".join(map(str, facts)) if facts else "1"
        print(f"  {i:>3}  {A_list[i]:>6}  {v_here:>4}  {v_next:>5}  "
              f"{f_str:<24}")
    if n > 8:
        print(f"  ... ({n - 8} more)")
    print()

    # ---------- decompress ----------
    print("--- decompression ---")
    recovered = comp.decompress(compressed)
    match = all(int(recovered[i]) == int(comp.v_orig[i])
                for i in range(n))
    print(f"  recovered n        = {len(recovered)}")
    print(f"  v[:6] original     = {list(map(int, comp.v_orig[:6]))}")
    print(f"  v[:6] recovered    = {recovered[:6]}")
    print(f"  full match         = {match}")
    print()

    # ---------- explicit factor.py role for odd n ----------
    if n % 2 == 1:
        print("--- factor.py extraction  ·  odd n ---")
        num, den = 1, 1
        for i in range(n):
            if i % 2 == 0:
                num *= A_list[i]
            else:
                den *= A_list[i]
        ratio = num // den
        v_1 = isqrt_via_factor(ratio)
        print(f"  Π_odd  A_i         = {num}")
        print(f"  Π_even A_i         = {den}")
        print(f"  ratio (v_1²)       = {ratio}")
        print(f"  v_1 via factor.py  = {v_1}  "
              f"(original = {int(comp.v_orig[0])})")
        print(f"  match              = {v_1 == int(comp.v_orig[0])}")
        print()

    # ---------- compression statistics ----------
    print("--- compression statistics ---")
    print(f"  full coupling G    : {s['full_G_entries']} entries   (n²)")
    print(f"  compressed         : {s['compressed_entries']} entries   "
          f"(n = {s['n']}, extra v_1 = {s['v_1_stored']})")
    print(f"  ratio              : {s['ratio']:.1f} : 1")
    print()

    # ---------- end-to-end verification ----------
    print("--- end-to-end reconstruction ---")
    new_dram = comp.reconstruct_dram(recovered)
    G_orig, _ = dram.coupling_matrix()
    G_new, _ = new_dram.coupling_matrix()
    diff = float(np.abs(G_orig - G_new).max())
    print(f"  ‖G_orig‖_F          = {np.linalg.norm(G_orig):.4f}")
    print(f"  max |G_orig − G_new|= {diff:.4e}")
    print(f"  exact reconstruction: {diff < 1e-9}")
    print()

    # ---------- corruption detection ----------
    print("--- corruption detection ---")
    tests = [
        ("A[0] += 1", "first cyclic entry"),
        ("A[n//2] *= 2", "middle entry"),
        ("A[n-1] += 5", "last entry"),
    ]
    for label, desc in tests:
        bad = dict(compressed)
        bad_A = list(A_list)
        if "A[0]" in label:
            bad_A[0] += 1
        elif "A[n//2]" in label:
            bad_A[n // 2] *= 2
        else:
            bad_A[n - 1] += 5
        bad["A"] = bad_A
        try:
            comp.decompress(bad)
            print(f"  {label:<15s} ({desc:<20s}): NOT detected")
        except ValueError as e:
            msg = str(e)[:50]
            print(f"  {label:<15s} ({desc:<20s}): "
                  f"detected → {msg}")
    print()

    # ---------- factor.py cost ----------
    print("--- factor.py cost  ·  on the actual A_i  ---")
    total_entries = 0
    for a in A_list:
        total_entries += len(factor_full(a))
    print(f"  n A_i values       = {n}")
    print(f"  total prime factors = {total_entries}")
    print(f"  avg per A_i        = {total_entries / n:.2f}")
    print(f"  max A_i            = {max(A_list)}   "
          f"√ = {math.isqrt(max(A_list))}")
    print()

    # ---------- complexity ----------
    print("--- complexity ---")
    print(f"  compress            O(n)             (n multiplications)")
    print(f"  decompress          O(n + F)         F = factor.py cost")
    print(f"  factor.py on v_1²   O(√ΠA) = O(v_1)  via dual Möbius SAT")
    print(f"  reconstruct G       O(n²)")
    print()
    print(f"  For n={n} odd: extraction needs one isqrt_via_factor call")
    print(f"  on a number ≤ {max(A_list) ** (n // 2 + 1)}")
    print()
    print("Done.")


if __name__ == "__main__":
    demo()