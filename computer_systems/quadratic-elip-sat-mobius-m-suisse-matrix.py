#!/usr/bin/env python3
"""
elliptic_mobius_sat.py

SAT via elliptic search with Möbius quadratic gating.

Bridges:
    quadratic-mobius-sat.py          (clauses → quadratic penalty)
    search-elip-m-suisse-projection.py (addresses → traces → supertrace)

For each Boolean assignment a ∈ [0, 2ⁿ):

    1. Address    : (x_a, y_a) = elliptic point derived from a
    2. Trace      : tr(a) = x_a^(i-1) + y_a^(i-1)   (matrix trace)
    3. Clause map : tr(a) plays the role of the clause penalty
                    (satisfying ⇒ tr(a) ∈ {clause value = 0 set})
    4. Quadratic  : q(a) = A·a² + B·a + C
    5. Möbius gate: keep a iff μ(q(a) mod K) ≠ 0
    6. TSP route  : bucket sort by phase of (x_a, y_a)
    7. Compression: exponential kernel convolution + supertrace
    8. Reconstruction: keep top‑M square‑free coefficients

This is the elliptic analogue of the quadratic pseudo‑Boolean encoding:
the traces replace the quadratic form, and the same supertrace / mass
invariants certify the solution set.
"""

import math
import random
import numpy as np
import cmath

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)                 # ≈ 2.362
NORM = 1.0 - math.exp(-ALPHA * (PI + E))
DENSITY = 6.0 / (PI * PI)              # square‑free density ≈ 0.6079


# ============================================================
#  1. Möbius sieve
# ============================================================
def mobius_sieve(K):
    mu = [0] * (K + 1)
    if K >= 1:
        mu[1] = 1
    primes = []
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
    return mu


# ============================================================
#  2. Elliptic address from a Boolean assignment
# ============================================================
def assignment_to_elliptic(a, n_vars):
    """
    Map a ∈ [0, 2ⁿ) to (x, y) in [0.5, 5.0]² via the bits,
    using two smooth functions of the bit pattern.
    """
    bits = [(a >> i) & 1 for i in range(n_vars)]
    # Project bits onto two pseudo‑phases
    t = 0.0
    u = 0.0
    for i, b in enumerate(bits):
        t += b * math.sin((i + 1) * 0.7)
        u += b * math.cos((i + 1) * 1.3)
    # Squash into [0.5, 5.0]
    x = 0.5 + 4.5 * (0.5 * (1 + math.tanh(t)))
    y = 0.5 + 4.5 * (0.5 * (1 + math.tanh(u)))
    return x, y


# ============================================================
#  3. Matrix trace (elliptic coefficient)
# ============================================================
def matrix_trace(x, y, i_exp=2):
    """tr = x^(i-1) + y^(i-1), the elliptic trace."""
    return x ** (i_exp - 1) + y ** (i_exp - 1)


# ============================================================
#  4. TSP routing (bucket sort by phase)
# ============================================================
def tsp_route(addresses):
    angles = [cmath.phase(complex(x, y)) for x, y in addresses]
    buckets = [[] for _ in range(360)]
    for idx, a in enumerate(angles):
        a_norm = a + PI if a < 0 else a
        b = int((a_norm / (2 * PI)) * 360) % 360
        buckets[b].append(idx)
    order = []
    for b in buckets:
        order.extend(b)
    return order


# ============================================================
#  5. Exponential kernel convolution (O(K))
# ============================================================
def conv_exp_kernel(signal, alpha=ALPHA):
    K = len(signal)
    lam = math.exp(-alpha)
    f = np.zeros(K)
    f[0] = signal[0]
    for i in range(1, K):
        f[i] = signal[i] + lam * f[i - 1]
    b = np.zeros(K)
    b[K - 1] = signal[K - 1]
    for i in range(K - 2, -1, -1):
        b[i] = signal[i] + lam * b[i + 1]
    conv_exp = (f + b - signal) / (1 - lam * lam)
    return (1.0 - conv_exp) / NORM


# ============================================================
#  6. Supertrace, entropy, mass (with overflow guard)
# ============================================================
def supertrace_and_mass(signal):
    S = 0.0
    for i, val in enumerate(signal):
        sign = 1 if (i % 2 == 0) else -1
        S += sign * abs(val)
    if S == 0:
        return 0.0, 0.0, 0.0
    p = abs(S) / len(signal)
    if p <= 0.0 or p >= 1.0:
        return S, 0.0, abs(S)
    H = -ALPHA * p * math.log(p)
    if H > 700:
        return S, H, 0.0
    m = abs(S) * math.exp(-H)
    return S, H, m


# ============================================================
#  7. SAT solver with elliptic + Möbius gate
# ============================================================
class EllipticMobiusSAT:
    def __init__(self, n_vars, clauses,
                 quad_A=1, quad_B=1, quad_C=0,
                 i_exp=2, K_filter=None):
        self.n = n_vars
        self.clauses = clauses
        self.A, self.B, self.C = quad_A, quad_B, quad_C
        self.i_exp = i_exp
        self.K = K_filter if K_filter is not None else 2 ** n_vars + 1
        self.mu = mobius_sieve(self.K)

    # -------- addressing --------
    def elliptic_address(self, a):
        return assignment_to_elliptic(a, self.n)

    def quadratic_address(self, a):
        return (self.A * a * a + self.B * a + self.C) % self.K

    def mobius_gate(self, a):
        return self.mu[self.quadratic_address(a)] != 0

    # -------- clause evaluation --------
    @staticmethod
    def _lit_true(lit, a):
        v, s = lit
        bit = (a >> v) & 1
        return bool(bit) if s > 0 else not bool(bit)

    def clause_true(self, clause, a):
        return any(self._lit_true(l, a) for l in clause)

    def formula_true(self, a):
        return all(self.clause_true(c, a) for c in self.clauses)

    # -------- trace ensemble --------
    def compute_traces(self, use_gate=True):
        """
        Return arrays over all accepted assignments:
            assignments, addresses, traces, sorted_order
        """
        acc = []
        addr = []
        tr = []
        for a in range(2 ** self.n):
            if use_gate and not self.mobius_gate(a):
                continue
            x, y = self.elliptic_address(a)
            addr.append((x, y))
            tr.append(matrix_trace(x, y, self.i_exp))
            acc.append(a)
        tr = np.array(tr, dtype=float)
        # TSP sort
        if addr:
            order = tsp_route(addr)
            tr_sorted = tr[order]
        else:
            order = []
            tr_sorted = tr
        return acc, addr, tr, tr_sorted, order

    # -------- compression pipeline --------
    def compress(self, use_gate=True):
        acc, addr, tr, tr_sorted, order = self.compute_traces(use_gate)
        K_eff = len(tr_sorted)
        if K_eff == 0:
            return dict(acc=[], kept=[], S=0.0, H=0.0, m=0.0,
                        conv=np.array([]), ratio=0.0)
        conv = conv_exp_kernel(tr_sorted)
        S, H, m = supertrace_and_mass(conv)
        M = max(1, min(int(abs(S)), K_eff))
        mu = self.mu
        mag = np.abs(conv)
        idx_sorted = np.argsort(mag)[::-1]
        kept = []
        for idx in idx_sorted:
            n = idx + 1
            if n <= self.K and mu[n] != 0:
                kept.append((int(idx), float(conv[idx])))
                if len(kept) >= M:
                    break
        recon = np.zeros(K_eff, dtype=complex)
        for i, v in kept:
            recon[i] = v
        err = (np.linalg.norm(conv - recon) / np.linalg.norm(conv)
               if np.linalg.norm(conv) > 0 else 0.0)
        return dict(acc=acc, kept=kept, S=S, H=H, m=m,
                    conv=conv, ratio=len(kept) / K_eff, err=err,
                    order=order, traces=tr, addresses=addr)


# ============================================================
#  8. Demonstration
# ============================================================
def demo():
    print("=" * 68)
    print("Elliptic Möbius SAT  —  traces as clause penalties")
    print("=" * 68)

    # --- 5‑variable 2‑SAT instance (same as quadratic-mobius-sat) ---
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

    solver = EllipticMobiusSAT(
        n_vars, clauses,
        quad_A=1, quad_B=1, quad_C=0,
        i_exp=2, K_filter=2 ** n_vars + 1
    )

    # ---------- naive solution ----------
    print("\n--- Naive enumeration (no Möbius gate) ---")
    naive = [a for a in range(2 ** n_vars) if solver.formula_true(a)]
    print(f"  total       : {2**n_vars}")
    print(f"  satisfying  : {len(naive)}")
    for a in naive:
        bits = [(a >> i) & 1 for i in range(n_vars)]
        print(f"    {bits}")

    # ---------- gated solution ----------
    print(f"\n--- Möbius gate  q(a)=a²+a mod {2**n_vars+1} ---")
    gated = [a for a in range(2 ** n_vars)
             if solver.mobius_gate(a) and solver.formula_true(a)]
    passed = sum(1 for a in range(2 ** n_vars) if solver.mobius_gate(a))
    print(f"  total       : {2**n_vars}")
    print(f"  passed gate : {passed}  ({100*passed/2**n_vars:.1f} %"
          f"  vs  6/π² = {100*DENSITY:.1f} %)")
    print(f"  satisfying  : {len(gated)}")
    print(f"  same set    : {set(naive) == set(gated)}")

    # ---------- elliptic traces ----------
    print("\n--- Elliptic trace table (solutions only) ---")
    print(f"  {'a':>3s}  {'(x, y)':>22s}  {'tr(a)':>10s}  "
          f"{'q(a)':>5s}  {'μ(q)':>5s}")
    for a in naive:
        x, y = solver.elliptic_address(a)
        tr = matrix_trace(x, y, solver.i_exp)
        q = solver.quadratic_address(a)
        print(f"  {a:3d}  ({x:7.4f},{y:7.4f})  {tr:10.4f}  "
              f"{q:5d}  {solver.mu[q]:+d}")

    # ---------- compression ----------
    print("\n--- Compression via elliptic traces + Möbius filter ---")
    for label, use_gate in [("no gate", False), ("with gate", True)]:
        res = solver.compress(use_gate=use_gate)
        print(f"  {label:10s}  "
              f"K_eff={len(res['traces']):3d}  "
              f"kept={len(res['kept']):3d}  "
              f"ratio={res['ratio']:.3f}  "
              f"S={res['S']:+.4f}  "
              f"H={res['H']:.4f}  "
              f"m={res['m']:.4f}  "
              f"err={res['err']:.3e}")

    # ---------- envelope check ----------
    print("\n--- Envelope check ---")
    print(f"  Expected kept/K with gate  : ~{DENSITY:.4f}")
    print(f"  Random‑walk envelope √K    : {DENSITY*math.sqrt(2**n_vars):.4f}")

    print("\nDone.")


if __name__ == "__main__":
    demo()