#!/usr/bin/env python3
"""
mobius_sat_64bit.py

64-bit Möbius SAT architecture with (2K+1) double precision.

Layout
------
The |Ci| array is a length-(2K+1) 64-bit float vector:

    index i ∈ [-K, K]  ↔  slot  i + K

    positive half   i ∈ [1, K]     real plane   (symmetric entries)
    negative half   i ∈ [-K, -1]   imag plane   (asymmetric entries)
    zero            i = 0          DC term

Parallel separation
-------------------
Positive and negative halves are *streamed independently* (the
"parallel separation" of the 64-bit layout):

    positive stream : linear SAT envelope  env_pos[q]
    negative stream : quadratic chirp envelope env_neg[q]

The sieve μ(1..K) is shared by both streams and is the only
O(K log log K) step; each stream is O(K).  The outer SAT
enumeration is O(2^n).

Basel precision checker
-----------------------
The square-free density

    ρ(K) = (1/K) · Σ_{n≤K} μ(n)² = (1/K) · Σ_{n≤K} |μ(n)|

converges to 6/π² ≈ 0.607927…  We use |ρ(K) − 6/π²| as the
precision error bound.  If it exceeds the tolerance the sieve is
flagged and rebuilt with a larger K.

Envelope concurrency
--------------------
Positive half  → magnitude envelope      |env_pos[i]|
Negative half  → chirped phase envelope  env_neg[i]·cos(a i² + b i + c)

Chip angle sort
---------------
ChipProcessor._tsp_route from chip-g.py bucket-sorts each half by
deterministic pseudo-angle in O(K) time before the supertrace.
"""

from __future__ import annotations

import math
import time
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple

# ============================================================
#  Optional chip-g.py import (angle sort only)
# ============================================================
try:
    from chip_g import ChipProcessor
    _HAS_CHIP = True
except Exception:
    _HAS_CHIP = False

    class ChipProcessor:
        """Fallback: bucket-sort by deterministic pseudo-angle, O(K)."""
        def __init__(self, max_K: int = 1000):
            self.max_K = max_K
            self.order = np.zeros(max_K, dtype=int)

        def _tsp_route(self, signal, K):
            angles = np.zeros(K, dtype=float)
            for i in range(K):
                x = math.sin(i * 7.0) + 0.1 * math.cos(i * 13.0)
                y = math.cos(i * 11.0) + 0.1 * math.sin(i * 17.0)
                angles[i] = math.atan2(y, x) + math.pi
            buckets = [[] for _ in range(360)]
            for i in range(K):
                idx = int((angles[i] / (2 * math.pi)) * 360) % 360
                buckets[idx].append(i)
            order = []
            for b in buckets:
                order.extend(b)
            self.order[:K] = order


# ============================================================
#  Constants
# ============================================================
PI        = math.pi
E         = math.e
ALPHA     = 1.0 / (PI - E)             # ≈ 2.362
A_STEP    = ALPHA / 0.3628             # ≈ 6.511
DENSITY   = 6.0 / (PI * PI)            # ≈ 0.6079271018
BIT64_EPS = float(np.finfo(np.float64).eps)   # ≈ 2.220446e-16


# ============================================================
#  1. Möbius sieve (Eratosthenes variant)  O(K log log K)
# ============================================================
def mobius_sieve(K: int) -> List[int]:
    """Return μ(0..K).  Time O(K log log K), space O(K)."""
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
#  2. Quadratic harmonic (quadratic-harmonic.py)  O(K)
# ============================================================
def quadratic_chirp(n: int, a: float = 1e-4,
                    b: float = 0.0, c: float = 0.0) -> float:
    """Chirped harmonic φ(n) = cos(a n² + b n + c)."""
    return math.cos(a * n * n + b * n + c)


def instantaneous_frequency(n: int, a: float, b: float) -> float:
    """dφ/dn = 2a n + b — the chirp rate."""
    return 2.0 * a * n + b


# ============================================================
#  3. 64-bit Möbius SAT
# ============================================================
class MobiusSAT64Bit:
    """
    64-bit Möbius SAT with parallel positive/negative separation.

    Parameters
    ----------
    n_vars        : number of boolean variables
    clauses       : list of clauses, each clause a list of (var, sign)
    quad_A/B/C    : quadratic address  q(a) = A·a² + B·a + C  mod K
    K_filter      : sieve modulus (default 2ⁿ + 1)
    precision_tol : tolerance for the Basel density check
    chirp_a/b/c   : coefficients of the quadratic chirp used on
                    the negative (imaginary) half
    """

    def __init__(self,
                 n_vars: int,
                 clauses: List[List[Tuple[int, int]]],
                 quad_A: int = 1,
                 quad_B: int = 1,
                 quad_C: int = 0,
                 K_filter: Optional[int] = None,
                 precision_tol: float = 5e-3,
                 chirp_a: float = 1e-4,
                 chirp_b: float = 0.0,
                 chirp_c: float = 0.0):
        self.n = n_vars
        self.clauses = clauses
        self.A, self.B, self.C = quad_A, quad_B, quad_C
        self.K = K_filter if K_filter is not None else 2 ** n_vars + 1
        self.mu = mobius_sieve(self.K)
        self.precision_tol = precision_tol
        self.chirp_a, self.chirp_b, self.chirp_c = chirp_a, chirp_b, chirp_c
        self.chip = ChipProcessor(max_K=2 * self.K + 1)
        self.reset_stats()

    def reset_stats(self):
        self.stats = dict(
            total=0, passed_gate=0, satisfying=0,
            n_symmetric=0, n_asymmetric=0,
            basel_density=0.0, basel_error=0.0,
            precision_ok=True, recomputes=0,
        )

    # ---------- quadratic address & gate ----------
    def quadratic_address(self, a: int) -> int:
        return (self.A * a * a + self.B * a + self.C) % self.K

    def mobius_gate(self, a: int) -> bool:
        return self.mu[self.quadratic_address(a)] != 0

    # ---------- clause evaluation ----------
    @staticmethod
    def _literal_true(lit, a):
        var, sign = lit
        bit = (a >> var) & 1
        return bool(bit) if sign > 0 else not bool(bit)

    def _clause_true(self, clause, a):
        return any(self._literal_true(l, a) for l in clause)

    def formula_satisfied(self, a: int) -> bool:
        return all(self._clause_true(c, a) for c in self.clauses)

    # ============================================================
    #  Basel precision checker
    # ============================================================
    def basel_density(self) -> float:
        """
        ρ(K) = (1/K) · Σ_{n≤K} μ(n)²
             = (1/K) · Σ_{n≤K} |μ(n)|     →  6/π²
        """
        count = 0
        for n in range(1, self.K + 1):
            if self.mu[n] != 0:
                count += 1
        return count / self.K

    def basel_precision_check(self) -> Tuple[float, float, bool]:
        """Return (density, error, ok)."""
        density = self.basel_density()
        error = abs(density - DENSITY)
        ok = error <= self.precision_tol
        self.stats['basel_density'] = density
        self.stats['basel_error'] = error
        self.stats['precision_ok'] = ok
        return density, error, ok

    def basel_recompute(self, factor: float = 1.5
                        ) -> Tuple[float, float, bool]:
        """Grow K and rebuild the sieve if precision fails."""
        self.stats['recomputes'] += 1
        new_K = int(self.K * factor)
        self.mu = mobius_sieve(new_K)
        return self.basel_precision_check()

    # ============================================================
    #  Dual envelope (parallel positive / negative separation)
    # ============================================================
    def build_dual_envelope(self,
                            penalty_fn: Optional[Callable[[int], float]] = None
                            ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Build env_pos (positive half, real plane) and env_neg
        (negative half, imaginary plane) plus the DC term.

        Mapping:
            q = q(a) ∈ [0, K]     → env_pos[q]     += penalty(a)
            q = q(a) ∈ [K+1, 2K]  → env_neg[q−K]   += penalty(a)

        Both halves are independent (parallel separation).
        """
        K = self.K
        env_pos = np.zeros(K + 1, dtype=np.float64)
        env_neg = np.zeros(K + 1, dtype=np.float64)
        dc = 0.0

        if penalty_fn is None:
            def penalty_fn(a):
                return 1.0 if self.formula_satisfied(a) else 0.0

        n_sym = n_asym = 0
        for a in range(2 ** self.n):
            q = self.quadratic_address(a)
            p = penalty_fn(a)
            if q == 0:
                dc += p
            elif q <= K:
                env_pos[q] += p
                if p > 0:
                    n_sym += 1
            else:
                env_neg[q - K] += p
                if p > 0:
                    n_asym += 1

        self.stats['n_symmetric'] = n_sym
        self.stats['n_asymmetric'] = n_asym
        return env_pos, env_neg, dc

    # ============================================================
    #  Envelope concurrency
    # ============================================================
    def envelope_concurrent(self,
                            env_pos: np.ndarray,
                            env_neg: np.ndarray
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Positive half  →  magnitude envelope  |env_pos[i]|
        Negative half  →  chirped envelope    env_neg[i] · cos(a i² + b i + c)

        The two halves are computed independently; a true parallel
        implementation would place them on two SIMD lanes / threads.
        """
        K = self.K
        mag_env = np.abs(env_pos[1:K + 1])

        chirp_env = np.zeros(K, dtype=np.float64)
        for i in range(1, K + 1):
            phi = quadratic_chirp(i, self.chirp_a, self.chirp_b, self.chirp_c)
            chirp_env[i - 1] = env_neg[i] * phi

        return mag_env, chirp_env

    # ============================================================
    #  Chip angle sort  (chip-g.py _tsp_route)
    # ============================================================
    def chip_angle_sort(self, signal: np.ndarray) -> np.ndarray:
        """Bucket-sort the signal by deterministic pseudo-angle, O(K)."""
        K = len(signal)
        if K > self.chip.max_K:
            self.chip = ChipProcessor(max_K=K)
        self.chip._tsp_route(signal, K)
        order = self.chip.order[:K]
        return signal[order]

    # ============================================================
    #  Dual supertrace
    # ============================================================
    def supertrace_dual(self,
                        env_pos: np.ndarray,
                        env_neg: np.ndarray,
                        dc: float = 0.0
                        ) -> Tuple[float, float, float]:
        """
        S_pos  over the positive envelope
        S_neg  over the negative envelope
        S_all  = S_pos + S_neg + dc
        """
        S_pos = 0.0
        for i in range(1, len(env_pos)):
            S_pos += env_pos[i] if (i % 2 == 0) else -env_pos[i]

        S_neg = 0.0
        for i in range(1, len(env_neg)):
            S_neg += env_neg[i] if (i % 2 == 0) else -env_neg[i]

        return S_pos, S_neg, S_pos + S_neg + dc

    def supertrace_64(self,
                      env_pos: np.ndarray,
                      env_neg: np.ndarray,
                      dc: float = 0.0) -> float:
        """Single combined supertrace over the 64-bit layout."""
        _, _, S_all = self.supertrace_dual(env_pos, env_neg, dc)
        return S_all

    # ============================================================
    #  Entropy / mass
    # ============================================================
    @staticmethod
    def entropy(S: float, N: int) -> float:
        if N <= 0 or S == 0.0:
            return 0.0
        p = abs(S) / N
        if p <= 0.0 or p >= 1.0:
            return 0.0
        return -ALPHA * p * math.log(p)

    @staticmethod
    def mass(S: float, N: int) -> float:
        H = MobiusSAT64Bit.entropy(S, N)
        return abs(S) * math.exp(-H) if H < 700 else 0.0

    # ============================================================
    #  Full 64-bit solve
    # ============================================================
    def solve_64(self,
                 use_mobius_gate: bool = True,
                 check_precision: bool = True) -> Dict:
        t0 = time.perf_counter()
        self.reset_stats()

        # 1. Basel precision check
        if check_precision:
            _, _, ok = self.basel_precision_check()
            if not ok:
                self.basel_recompute()

        # 2. dual envelope
        env_pos, env_neg, dc = self.build_dual_envelope()

        # 3. envelope concurrency
        mag_env, chirp_env = self.envelope_concurrent(env_pos, env_neg)

        # 4. chip angle sort
        mag_sorted = self.chip_angle_sort(mag_env)
        chirp_sorted = self.chip_angle_sort(chirp_env)

        # 5. dual supertrace over the angle-sorted halves
        S_pos, S_neg, S_all = self.supertrace_dual(
            np.concatenate([[0.0], mag_sorted]),
            np.concatenate([[0.0], chirp_sorted]),
            dc,
        )
        H_all = self.entropy(S_all, 2 * self.K + 1)
        m_all = self.mass(S_all, 2 * self.K + 1)

        # 6. SAT evaluation
        sols = []
        for a in range(2 ** self.n):
            self.stats['total'] += 1
            if use_mobius_gate and not self.mobius_gate(a):
                continue
            self.stats['passed_gate'] += 1
            if self.formula_satisfied(a):
                self.stats['satisfying'] += 1
                sols.append(a)

        return dict(
            sols=sols,
            stats=dict(self.stats),
            env_pos=env_pos, env_neg=env_neg,
            mag_env=mag_env, chirp_env=chirp_env,
            mag_sorted=mag_sorted, chirp_sorted=chirp_sorted,
            S_pos=S_pos, S_neg=S_neg, S_all=S_all,
            H=H_all, m=m_all,
            elapsed_ms=(time.perf_counter() - t0) * 1e3,
        )


# ============================================================
#  4. Demo
# ============================================================
def demo():
    print("=" * 78)
    print("Möbius SAT 64-bit architecture  ·  (2K+1) double precision")
    print("=" * 78)

    n_vars = 5
    clauses = [
        [(0, +1), (1, +1)],
        [(0, -1), (2, +1)],
        [(1, -1), (2, -1)],
        [(2, +1), (3, +1)],
        [(3, -1), (4, +1)],
        [(4, -1), (0, +1)],
    ]
    print(f"\nVariables: {n_vars}   Clauses: {len(clauses)}")
    for c in clauses:
        print("   " + " ∨ ".join(
            ("" if s > 0 else "¬") + f"x{v}" for v, s in c))

    solver = MobiusSAT64Bit(
        n_vars, clauses,
        quad_A=1, quad_B=1, quad_C=0,
        K_filter=2 ** n_vars + 1,
        precision_tol=5e-3,
        chirp_a=1e-4, chirp_b=0.0, chirp_c=0.0,
    )
    print(f"\nK = {solver.K}    |Ci| length = 2K+1 = {2*solver.K+1}")

    # --- Basel precision check ---
    density, error, ok = solver.basel_precision_check()
    print("\n--- Basel precision check ---")
    print(f"  ρ(K)              = {density:.8f}")
    print(f"  6/π²              = {DENSITY:.8f}")
    print(f"  error             = {error:.6e}")
    print(f"  precision OK      = {ok}  (tol = {solver.precision_tol})")

    # --- 64-bit solve ---
    print("\n--- 64-bit solve ---")
    out = solver.solve_64()
    st = out['stats']
    print(f"  total assignments    : {st['total']}")
    print(f"  passed gate          : {st['passed_gate']}  "
          f"({100*st['passed_gate']/st['total']:.1f} %)")
    print(f"  expected density     : {100*DENSITY:.1f} %")
    print(f"  satisfying           : {st['satisfying']}")
    print(f"  symmetric entries    : {st['n_symmetric']}")
    print(f"  asymmetric entries   : {st['n_asymmetric']}")
    for s in out['sols']:
        bits = [(s >> i) & 1 for i in range(n_vars)]
        print(f"    {bits}")

    # --- dual envelope ---
    print("\n--- Dual envelope (parallel separation) ---")
    print(f"  env_pos range      : "
          f"[{out['env_pos'].min():.0f}, {out['env_pos'].max():.0f}]")
    print(f"  env_neg range      : "
          f"[{out['env_neg'].min():.0f}, {out['env_neg'].max():.0f}]")

    # --- envelope concurrency ---
    print("\n--- Envelope concurrency ---")
    print(f"  magnitude env  range : "
          f"[{out['mag_env'].min():.4f}, {out['mag_env'].max():.4f}]")
    print(f"  chirp env      range : "
          f"[{out['chirp_env'].min():.4f}, {out['chirp_env'].max():.4f}]")

    # --- chip angle sort ---
    print("\n--- Chip angle sort (TSP route) ---")
    print(f"  first 8 of mag_sorted  : "
          f"{out['mag_sorted'][:8].round(2).tolist()}")
    print(f"  first 8 of chirp_sorted: "
          f"{out['chirp_sorted'][:8].round(2).tolist()}")

    # --- dual supertrace ---
    print("\n--- Dual supertrace ---")
    print(f"  S_pos (real plane)  = {out['S_pos']:+.6f}")
    print(f"  S_neg (imag plane)  = {out['S_neg']:+.6f}")
    print(f"  S_all (combined)    = {out['S_all']:+.6f}")
    print(f"  H (entropy)         = {out['H']:.6f}")
    print(f"  m (mass)            = {out['m']:.6f}")

    # --- chirp sample ---
    print("\n--- Quadratic harmonic chirp (negative half) ---")
    for i in [1, 10, 50, 100]:
        phi = quadratic_chirp(i, solver.chirp_a, solver.chirp_b, solver.chirp_c)
        freq = instantaneous_frequency(i, solver.chirp_a, solver.chirp_b)
        print(f"  n = {i:>4d}   φ(n) = {phi:+.6f}   dφ/dn = {freq:.6e}")

    # --- complexity ---
    print("\n--- Complexity ---")
    print("  Möbius sieve (once)     O(K log log K)")
    print("  Basel density check     O(K)")
    print("  Dual envelope           O(2^n)")
    print("  Envelope concurrency    O(K)   (two halves, independent)")
    print("  Chip angle sort         O(K)   (bucket sort)")
    print("  Dual supertrace         O(K)")
    print("  SAT evaluation          O(2^n · |clauses|)")
    print(f"\n  elapsed: {out['elapsed_ms']:.2f} ms")

    print("\nDone.")


if __name__ == "__main__":
    demo()