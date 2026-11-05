#!/usr/bin/env python3
"""
compress_dram_finite.py
=======================

DRAM compression using **finite-step Newton** for the cyclic-ratio
perfect-square test.

Replaces `factor.py`'s `isqrt_via_factor` with a Newton iteration
whose derivative is the finite-step derivative

    D_a(f)(x) = (f(x + a) − f(x − a)) / (2a),    a = A_STEP ≈ 6.511

Key identity
------------
For f(x) = x² − n,  the finite-step derivative is *exact* for any a:

    D_a(f)(x) = ((x+a)² − (x−a)²) / (2a)  =  4ax / (2a)  =  2x

So Newton becomes

    x_{k+1} = x_k − (x_k² − n) / D_a(f)(x_k)
            = x_k − (x_k² − n) / (2 x_k)

which converges in ⌈log₂(digits)⌉ steps for perfect squares.

Initial estimate via finite-step logarithm
------------------------------------------
    k = log(n) / log(A_STEP)              "how many A_STEPs to reach n"
    x_0 = A_STEP^(k/2)                    ≃ √n

A_STEP is the structural unit of the whole architecture
(α / 0.3628 with α = 1/(π−e)).  Here it is the natural clock for the
Newton refinement: the initial guess counts finite steps, and the
refinement uses the finite-step derivative.

Compression pipeline (unchanged)
--------------------------------
    Uncompressed :  G ∈ ℝ^{n×n}        (n² entries)
    Compressed   :  A_i = v_i · v_{i+1}   (n entries) + optional v_1

    Cyclic ratio:  Π_odd A_i / Π_even A_i = v_1²   (odd n only)

    v_1 recovered via finite-step Newton → walk the cycle.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional

import numpy as np

from DRAM import FermionDRAM, travel_lengths


# ============================================================
#  Constants
# ============================================================
PI       = math.pi
E        = math.e
ALPHA    = 1.0 / (PI - E)             # ≈ 2.362338
A_STEP   = ALPHA / 0.3628             # ≈ 6.511

try:
    from DRAM_factor import isqrt_via_factor
    _HAS_FACTOR_COMPARE = True
except ImportError:
    isqrt_via_factor = None
    _HAS_FACTOR_COMPARE = False


# ============================================================
#  Finite-step derivative  (exact for quadratics)
# ============================================================
def finite_step_derivative(f, x: float, a: float = A_STEP) -> float:
    """D_a f(x) = (f(x+a) − f(x−a)) / (2a)."""
    return (f(x + a) - f(x - a)) / (2.0 * a)


# ============================================================
#  Integer sqrt via finite-step Newton
# ============================================================
def isqrt_via_finite_step(n: int,
                          max_iter: int = 80,
                          verbose: bool = False
                          ) -> Optional[int]:
    """
    Perfect-square root via Newton iteration with finite-step derivative.

    Returns x if x·x == n, else None.

    Initial estimate via finite-step logarithm:
        k    = log(n) / log(A_STEP)        number of A_STEPs
        x_0  = A_STEP^(k/2)                ≃ √n

    Newton refinement (finite-step derivative of x² − n is exactly 2x):
        x_{k+1} = x_k − (x_k² − n) / (2 x_k)
    """
    if n <= 0:
        return None
    if n == 1:
        return 1
    if n < 4:
        return None

    # ---- finite-step initial estimate ----
    log_k = math.log(n) / math.log(A_STEP)     # finite-step count
    half_k = log_k / 2.0
    if half_k > 700:
        x = max(1, int(math.isqrt(n)))
    else:
        x = max(1, int(round(A_STEP ** half_k)))
    trace = [x]

    # ---- Newton refinement ----
    for _ in range(max_iter):
        if x <= 0:
            x = 1
        f_val = x * x - n
        if f_val == 0:
            if verbose:
                print(f"    converged in {len(trace)} steps")
                print(f"    trace: {trace}")
            return x

        deriv = finite_step_derivative(lambda t: t * t - n, x, A_STEP)
        if deriv == 0:
            return None

        step = f_val / deriv
        x_new = int(round(x - step))
        if x_new <= 0:
            x_new = max(1, x // 2)
        trace.append(x_new)
        if x_new == x:
            break
        x = x_new

    if verbose:
        print(f"    trace: {trace[:12]}"
              f"{'…' if len(trace) > 12 else ''}")
        print(f"    x = {x}, x² = {x * x}, n = {n}")

    # ---- final verification ----
    if x * x == n:
        return x
    for dx in (-2, -1, 1, 2):
        y = x + dx
        if y > 0 and y * y == n:
            return y
    return None


# ============================================================
#  Coupling-factor compressor
# ============================================================
class CouplingFactorCompressor:
    """
    Same cyclic-product compression as `compress_dram.py`, but the
    perfect-square test now uses `isqrt_via_finite_step` instead of
    `factor.py`.
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
        n = self.n
        A = np.zeros(n, dtype=np.int64)
        for i in range(n):
            j = (i + 1) % n
            A[i] = int(self.v_orig[i]) * int(self.v_orig[j])
        v_1_extra = int(self.v_orig[0]) if n % 2 == 0 else None
        return dict(addrs=list(self.addrs),
                    A=A.tolist(), v_1=v_1_extra, n=n)

    # --------------------------------------------------------
    def decompress(self, compressed: Dict) -> List[int]:
        addrs = compressed["addrs"]
        A = compressed["A"]
        n = compressed["n"]
        v_1_extra = compressed["v_1"]

        if n < 3:
            if n == 1:
                return [int(v_1_extra) if v_1_extra is not None else A[0]]
            return [int(v_1_extra), A[0] // int(v_1_extra)]

        # ---- recover v_1 ----
        if n % 2 == 1:
            num, den = 1, 1
            for i in range(n):
                if i % 2 == 0:
                    num *= A[i]
                else:
                    den *= A[i]
            if den == 0:
                raise ValueError("degenerate cyclic product")
            if num % den != 0:
                raise ValueError(
                    f"non-integer ratio  num={num}  den={den}")
            ratio = num // den
            v_1 = isqrt_via_finite_step(ratio)
            if v_1 is None:
                raise ValueError(
                    f"ratio {ratio} not a perfect square "
                    f"(finite-step Newton rejected)")
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
                f"cycle closure failed: A[{n-1}]={A[n-1]}  "
                f"v[-1]·v[0]={v[-1] * v[0]}")

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
    print("DRAM compression  ·  finite-step Newton for perfect squares")
    print("=" * 78)
    print(f"  α       = 1/(π − e)     = {ALPHA:.6f}")
    print(f"  A_STEP  = α / 0.3628    = {A_STEP:.6f}")
    print(f"  D_a(x² − n) = 2x        (exact for quadratics, any a)")
    print()

    # ---------- 1 · standalone perfect-square tests ----------
    print("--- 1 · isqrt_via_finite_step  ·  standalone ---")
    tests = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144,
             169, 196, 225, 256, 289, 400, 441, 529, 676,
             841, 961, 1024, 1521, 12321,
             # non-squares
             2, 3, 5, 8, 15, 26, 50, 99, 101, 200, 12322]
    print(f"  {'n':>8}  {'isqrt':>8}  {'n is square':<14}")
    for n in tests:
        x = isqrt_via_finite_step(n)
        shown = str(x) if x is not None else '—'
        is_sq = x is not None
        print(f"  {n:>8}  {shown:>8}  {str(is_sq):<14}")
    print()

    # ---------- 2 · comparison with factor.py ----------
    if _HAS_FACTOR_COMPARE:
        print("--- 2 · comparison  ·  finite-step vs factor.py ---")
        mismatches = 0
        for n in range(1, 501):
            fs = isqrt_via_finite_step(n)
            fpy = isqrt_via_factor(n)
            if fs != fpy:
                mismatches += 1
                print(f"  mismatch on n={n}:  fs={fs}  factor.py={fpy}")
        print(f"  tested n = 1..500")
        print(f"  mismatches = {mismatches}")
        print()

    # ---------- 3 · convergence trace ----------
    print("--- 3 · convergence trace  ·  n = 12321 (= 111²) ---")
    print(f"  x_0 estimate via A_STEP: ")
    k = math.log(12321) / math.log(A_STEP)
    print(f"    k = log(n)/log(A_STEP) = {k:.4f}")
    print(f"    x_0 = A_STEP^(k/2)     ≃ {A_STEP ** (k/2):.4f}")
    print(f"  target √12321 = 111")
    print()
    x = isqrt_via_finite_step(12321, verbose=True)
    print(f"  result = {x}")
    print()

    # ---------- 4 · non-square rejection ----------
    print("--- 4 · non-square  ·  n = 12322 ---")
    x = isqrt_via_finite_step(12322, verbose=True)
    print(f"  result = {x}  (should be None)")
    print()

    # ---------- 5 · finite-step derivative is step-independent ----------
    print("--- 5 · D_a(x² − n) is step-independent ---")
    print(f"  f(x) = x² − 100,  x = 11")
    for a in (0.1, 1.0, A_STEP, 100.0):
        D = finite_step_derivative(lambda t: t * t - 100, 11.0, a)
        print(f"    a = {a:>8.4f}   D_a f(11) = {D:>10.6f}   "
              f"2·11 = 22")
    print()

    # ---------- 6 · end-to-end DRAM compression ----------
    print("--- 6 · full DRAM compression  ·  finite-step decompress ---")
    K = 60
    dram = FermionDRAM(K=K, coupling_strength=1)
    addrs = dram.address_list()
    n = len(addrs)
    rng = np.random.default_rng(7)
    for a in addrs:
        dram.write(a, int(rng.integers(1, 32)))

    comp = CouplingFactorCompressor(dram)
    compressed = comp.compress()
    s = comp.stats(compressed)
    print(f"  K = {K},  n = {n}  (n parity = "
          f"{'odd' if n % 2 else 'even'})")
    print(f"  full G      = {s['full_G_entries']} entries  (n²)")
    print(f"  compressed  = {s['compressed_entries']} entries  "
          f"(n = {s['n']}, extra v_1 = {s['v_1_stored']})")
    print(f"  ratio       = {s['ratio']:.1f} : 1")
    print()

    recovered = comp.decompress(compressed)
    match = all(int(recovered[i]) == int(comp.v_orig[i]) for i in range(n))
    print(f"  recovered[:6]  = {recovered[:6]}…")
    print(f"  original[:6]   = {[int(x) for x in comp.v_orig[:6]]}…")
    print(f"  full match     = {match}")
    print()

    # ---------- 7 · reconstruct G ----------
    new_dram = comp.reconstruct_dram(recovered)
    G_orig, _ = dram.coupling_matrix()
    G_new, _ = new_dram.coupling_matrix()
    diff = float(np.abs(G_orig - G_new).max())
    print("--- 7 · reconstruct coupling matrix G ---")
    print(f"  ‖G_orig‖_F        = {np.linalg.norm(G_orig):.4f}")
    print(f"  max |G_orig − G_new| = {diff:.4e}")
    print(f"  exact reconstruction = {diff < 1e-9}")
    print()

    # ---------- 8 · odd-n extraction trace ----------
    if n % 2 == 1:
        print("--- 8 · cyclic ratio extraction  ·  odd n ---")
        A_list = compressed["A"]
        num, den = 1, 1
        for i in range(n):
            if i % 2 == 0:
                num *= A_list[i]
            else:
                den *= A_list[i]
        ratio = num // den
        v_1 = isqrt_via_finite_step(ratio)
        print(f"  Π_odd  A_i               = {num}")
        print(f"  Π_even A_i               = {den}")
        print(f"  ratio  (v_1²)            = {ratio}")
        print(f"  √ratio  via finite step  = {v_1}")
        print(f"  original v_1             = {int(comp.v_orig[0])}")
        print(f"  match                    = {v_1 == int(comp.v_orig[0])}")
        print()

    # ---------- 9 · corruption detection ----------
    print("--- 9 · corruption detection ---")
    for i in [0, n // 2, n - 1]:
        bad = dict(compressed)
        bad["A"] = list(compressed["A"])
        bad["A"][i] += 1
        try:
            comp.decompress(bad)
            print(f"  A[{i}] += 1: NOT detected")
        except ValueError as e:
            msg = str(e)[:55]
            print(f"  A[{i}] += 1: detected → {msg}")
    print()

    # ---------- 10 · complexity ----------
    print("--- 10 · complexity ---")
    print(f"  initial estimate      O(1)  (finite-step log)")
    print(f"  Newton refinement     O(log₂ digits)  ≈ O(log n)")
    print(f"  factor.py isqrt       O(√n)  via dual Möbius SAT")
    print(f"  ratio detection       O(n)")
    print(f"  cycle walk            O(n)")
    print(f"  ─────────────────────────────")
    print(f"  total decompression   O(n + log n)  ≈ O(n)")
    print(f"  vs factor.py          O(n + √n)")
    print()
    print("Done.")


if __name__ == "__main__":
    demo()