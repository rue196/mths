#!/usr/bin/env python3
"""
quadratic_mobius_sat_32bit.py

32‑bit twin of quadratic_mobius_sat.py.

Everything that was float64 / int64 is now float32 / int32.
Assignment index and quadratic address are stored in uint32.
The K_filter is capped at 2³²−1 because q(a) must fit in uint32.

Precision
---------
float32  eps ≈ 1.19e-7   (float64 eps ≈ 2.22e-16)
int32    max  ≈ 2.15e9   (int64 max ≈ 9.22e18)

The Basel density tolerance is scaled to float32 precision.
"""

import math
import numpy as np

# ---------- Constants ----------
PI = math.pi
E  = math.e
ALPHA = 1.0 / (PI - E)

# float32/float64 discipline
FTYPE       = np.float32
FTYPE_NAME  = "float32"
EPS32       = float(np.finfo(FTYPE).eps)          # ≈ 1.19e-7
DENSITY     = FTYPE(6.0 / (PI * PI))              # 6/π² ≈ 0.6079271
DENSITY_F64 = 6.0 / (PI * PI)                     # for comparison only

# int32/uint32 discipline
ITYPE       = np.int32
UTYPE       = np.uint32
INT32_MAX   = 2**31 - 1
UINT32_MAX  = 2**32 - 1

# Basel tolerance is now scaled to float32 precision
BASEL_TOL_32 = 1e-5           # ≈ 84 × EPS32 — comfortable headroom


# ============================================================
#  1. Möbius sieve (Eratosthenes variant)  O(K log log K)
#     Stored as int8 (-1, 0, 1) — smallest possible width.
# ============================================================
def mobius_sieve32(K: int) -> np.ndarray:
    """
    Return μ(0..K) as an int8 array.
    μ ∈ {-1, 0, +1}, so int8 is exact and 8× smaller than int64.
    """
    if K < 1:
        return np.zeros(K + 1, dtype=np.int8)
    mu      = np.ones(K + 1, dtype=np.int8)
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
#  2. Quadratic Möbius SAT  (32‑bit)
# ============================================================
class QuadraticMobiusSAT32:
    """
    n_vars       : number of boolean variables
    clauses      : list of (var_idx, sign) lists
    quad_A/B/C   : coefficients of q(a) = A·a² + B·a + C  mod K
    K_filter     : sieve modulus (must be ≤ 2³²−1)
    """

    def __init__(self, n_vars, clauses,
                 quad_A=1, quad_B=1, quad_C=0, K_filter=None):
        self.n       = n_vars
        self.clauses = clauses
        self.A, self.B, self.C = quad_A, quad_B, quad_C

        if K_filter is None:
            K_filter = 2 ** n_vars + 1
        if K_filter > UINT32_MAX:
            raise ValueError(
                f"K_filter = {K_filter} exceeds uint32 max {UINT32_MAX}")
        self.K = int(K_filter)

        # μ(n) as int8 — half the register pressure of int16, 8× that of int64
        self.mu = mobius_sieve32(self.K)

        # Pre‑computed A·a² + B·a + C mod K as uint32 (avoids recomputation)
        N = 2 ** n_vars
        a_range = np.arange(N, dtype=np.uint64)          # use uint64 for a²
        q_all = ((quad_A * a_range * a_range +
                  quad_B * a_range +
                  quad_C) % self.K).astype(UTYPE)
        self.q_all = q_all

        # Envelope table as float32 (halves the memory of float64)
        self.env32 = np.zeros(self.K, dtype=FTYPE)

        self.reset_stats()

    def reset_stats(self):
        self.stats = dict(
            total=0, passed_gate=0, satisfying=0,
            basel_density=0.0, basel_error=0.0, precision_ok=True,
            recomputes=0, dtype=FTYPE_NAME,
        )

    # ---------- quadratic address & gate (vectorised) ----------
    def quadratic_address(self, a: int) -> int:
        return int(self.q_all[a])

    def mobius_gate(self, a: int) -> bool:
        return self.mu[self.q_all[a]] != 0

    # ---------- clause evaluation ----------
    @staticmethod
    def _literal_true(lit, a):
        var, sign = lit
        bit = (a >> var) & 1
        return bool(bit) if sign > 0 else not bool(bit)

    def _clause_true(self, clause, a):
        return any(self._literal_true(l, a) for l in clause)

    def formula_satisfied(self, a):
        return all(self._clause_true(c, a) for c in self.clauses)

    # ---------- solve ----------
    def solve(self, use_mobius_gate=True):
        self.reset_stats()
        sols = []
        for a in range(2 ** self.n):
            self.stats['total'] += 1
            if use_mobius_gate and not self.mobius_gate(a):
                continue
            self.stats['passed_gate'] += 1
            if self.formula_satisfied(a):
                self.stats['satisfying'] += 1
                sols.append(a)
        return sols

    # ============================================================
    #  Basel precision checker  (32‑bit threshold)
    # ============================================================
    def basel_density32(self) -> float:
        """
        ρ(K) = (1/K) · Σ_{n≤K} μ(n)²
        Computed in float32 to match the 32‑bit architecture.
        """
        count = int(np.count_nonzero(self.mu[1:self.K + 1]))
        return float(FTYPE(count) / FTYPE(self.K))

    def basel_precision_check32(self):
        """Compare to 6/π² with the float32 tolerance."""
        density = self.basel_density32()
        error   = abs(density - DENSITY_F64)
        ok      = error <= BASEL_TOL_32
        self.stats['basel_density'] = density
        self.stats['basel_error']   = error
        self.stats['precision_ok']  = ok
        return density, error, ok

    def basel_recompute32(self, factor=1.5):
        """Grow K (still ≤ 2³²−1) and rebuild μ as int8."""
        self.stats['recomputes'] += 1
        new_K = min(int(self.K * factor), UINT32_MAX)
        if new_K <= self.K:
            return self.basel_precision_check32()
        self.K = new_K
        self.mu = mobius_sieve32(self.K)
        # rebuild q_all and env32
        N = 2 ** self.n
        a_range = np.arange(N, dtype=np.uint64)
        self.q_all = ((self.A * a_range * a_range +
                       self.B * a_range +
                       self.C) % self.K).astype(UTYPE)
        self.env32 = np.zeros(self.K, dtype=FTYPE)
        return self.basel_precision_check32()

    # ============================================================
    #  Envelope (float32)
    # ============================================================
    def build_envelope32(self, penalty_fn=None):
        """
        env[q] = Σ_{a : q(a) = q} penalty(a)
        All arithmetic in float32.
        """
        env = np.zeros(self.K, dtype=FTYPE)
        if penalty_fn is None:
            def penalty_fn(a):
                return FTYPE(1.0) if self.formula_satisfied(a) else FTYPE(0.0)
        for a in range(2 ** self.n):
            env[self.q_all[a]] += penalty_fn(a)
        self.env32 = env
        return env

    def supertrace_via_quadratic32(self, env):
        """S = Σ_q (-1)^q · env[q]  in float32."""
        S = FTYPE(0.0)
        for q in range(self.K):
            v = env[q]
            if v == FTYPE(0.0):
                continue
            sign = FTYPE(1.0) if (q & 1) == 0 else FTYPE(-1.0)
            S += sign * v
        return float(S)

    def supertrace_via_quadratic_gated32(self, env):
        """Same as above but only over square‑free q."""
        S = FTYPE(0.0)
        mu = self.mu
        for q in range(self.K):
            v = env[q]
            if v == FTYPE(0.0):
                continue
            if mu[q] == 0:
                continue
            sign = FTYPE(1.0) if (q & 1) == 0 else FTYPE(-1.0)
            S += sign * v
        return float(S)

    def envelope_entropy32(self, env):
        S = self.supertrace_via_quadratic32(env)
        N = int(np.count_nonzero(env))
        if N <= 0 or S == 0.0:
            return 0.0
        p = abs(S) / N
        if p <= 0.0 or p >= 1.0:
            return 0.0
        return -ALPHA * p * math.log(p)

    def envelope_mass32(self, env):
        S = self.supertrace_via_quadratic32(env)
        H = self.envelope_entropy32(env)
        return abs(S) * math.exp(-H) if H < 700 else 0.0

    # ============================================================
    #  Quadratic pseudo‑Boolean form (float32)
    # ============================================================
    def build_quadratic_form32(self):
        """Q, b, c as float32 (halves memory of the float64 version)."""
        n = self.n
        Q = np.zeros((n, n), dtype=FTYPE)
        b = np.zeros(n, dtype=FTYPE)
        c = FTYPE(0.0)

        for clause in self.clauses:
            lits = []
            for var, sign in clause:
                if sign > 0:
                    lits.append((var, FTYPE(1), FTYPE(0)))
                else:
                    lits.append((var, FTYPE(-1), FTYPE(1)))

            c += FTYPE(1.0)
            for _, _, t in lits:
                c -= t
            for i in range(len(lits)):
                for j in range(i + 1, len(lits)):
                    c += lits[i][2] * lits[j][2]

            for i, (vi, si, ti) in enumerate(lits):
                b[vi] -= si
                for j, (vj, sj, tj) in enumerate(lits):
                    if i == j:
                        continue
                    b[vi] += si * tj

            for i in range(len(lits)):
                for j in range(i + 1, len(lits)):
                    vi, si, _ = lits[i]
                    vj, sj, _ = lits[j]
                    if vi == vj:
                        b[vi] += si * sj
                    else:
                        Q[vi, vj] += si * sj
        return Q, b, c

    def evaluate_quadratic32(self, a, Q, b, c):
        bits = np.array([(a >> i) & 1 for i in range(self.n)], dtype=FTYPE)
        return float(bits @ Q @ bits + b @ bits + c)


# ============================================================
#  3. Demonstration
# ============================================================
def demo32():
    print("=" * 72)
    print("Quadratic Möbius SAT  (32‑bit twin)")
    print(f"  float dtype = {FTYPE_NAME}   eps ≈ {EPS32:.3e}")
    print(f"  int dtype   = int8 (μ), uint32 (address)")
    print("=" * 72)

    n_vars = 5
    clauses = [
        [(0, +1), (1, +1)],
        [(0, -1), (2, +1)],
        [(1, -1), (2, -1)],
        [(2, +1), (3, +1)],
        [(3, -1), (4, +1)],
        [(4, -1), (0, +1)],
    ]

    solver = QuadraticMobiusSAT32(
        n_vars, clauses,
        quad_A=1, quad_B=1, quad_C=0,
        K_filter=2 ** n_vars + 1,
    )

    # --- Basel check ---
    density, err, ok = solver.basel_precision_check32()
    print(f"\nBasel ρ(K)   = {density:.8f}")
    print(f"6/π²         = {DENSITY_F64:.8f}")
    print(f"error        = {err:.3e}")
    print(f"precision OK = {ok}  (tol = {BASEL_TOL_32:.1e})")

    # --- naive vs gated solve ---
    naive = solver.solve(use_mobius_gate=False)
    gated = solver.solve(use_mobius_gate=True)
    print(f"\nnaive solutions : {len(naive)}")
    print(f"gated solutions : {len(gated)}")
    print(f"same set        : {set(naive) == set(gated)}")

    # --- envelope + supertrace ---
    env = solver.build_envelope32()
    S_all = solver.supertrace_via_quadratic32(env)
    S_mu  = solver.supertrace_via_quadratic_gated32(env)
    H_env = solver.envelope_entropy32(env)
    m_env = solver.envelope_mass32(env)
    print(f"\nS (all q)         = {S_all:+.6f}")
    print(f"S_μ (square‑free) = {S_mu:+.6f}")
    print(f"H (entropy)       = {H_env:.6f}")
    print(f"m (mass)          = {m_env:.6f}")

    # --- quadratic pseudo‑Boolean form ---
    Q, b, c = solver.build_quadratic_form32()
    print(f"\nQ dtype = {Q.dtype}, b dtype = {b.dtype}, c = {float(c):+.2f}")

    print("\nDone.")


if __name__ == "__main__":
    demo32()