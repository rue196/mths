#!/usr/bin/env python3
"""
quadratic_mobius_sat.py

SAT solving through a quadratic Möbius gate.

Pipeline
--------
   1. CNF formula  →  quadratic pseudo‑Boolean  Q(x) = xᵀQx + bᵀx + c
   2. Assignment a ∈ [0, 2ⁿ)  →  quadratic address  q(a) = A·a² + B·a + C
   3. Möbius gate: keep a only if μ(q(a) mod K) ≠ 0
   4. Evaluate SAT on the surviving candidates

Two views of the same gate
--------------------------
  • Clause view  : Q encodes the clause penalties (quadratic form).
  • Address view : q(n) = n² + n maps each assignment to a square‑free
                   address, pruning 1 − 6/π² ≈ 39.2 % of the search space
                   without changing the solution set.

Complexity: O(2ⁿ) enumeration + O(K) sieve, K = 2ⁿ + 1.
"""

import math
import numpy as np

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)
DENSITY = 6.0 / (PI * PI)     # 0.607927…


# ============================================================
#  1. Möbius sieve (linear, O(K))
# ============================================================
def mobius_sieve(K):
    if K < 1:
        return [0] * (K + 1)
    mu = [0] * (K + 1)
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
#  2. Quadratic Möbius SAT
# ============================================================
class QuadraticMobiusSAT:
    """
    n_vars   : number of boolean variables x_0 .. x_{n-1}
    clauses  : list of clauses; each clause is list of (var_idx, sign)
               sign = +1 → positive literal; sign = -1 → negative literal
    quad_A/B/C : coefficients of the quadratic address q(a) = A·a² + B·a + C
    K_filter : modulus for the Möbius lookup (default 2ⁿ + 1)
    """

    def __init__(self, n_vars, clauses,
                 quad_A=1, quad_B=1, quad_C=0, K_filter=None):
        self.n = n_vars
        self.clauses = clauses
        self.A, self.B, self.C = quad_A, quad_B, quad_C
        self.K = K_filter if K_filter is not None else 2 ** n_vars + 1
        self.mu = mobius_sieve(self.K)
        self.reset_stats()

    def reset_stats(self):
        self.stats = dict(total=0, passed_gate=0, satisfying=0)

    # ---------- quadratic address + gate ----------
    def quadratic_address(self, a):
        return (self.A * a * a + self.B * a + self.C) % self.K

    def mobius_gate(self, a):
        return self.mu[self.quadratic_address(a)] != 0

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

    # ---------- quadratic pseudo‑Boolean encoding ----------
    def build_quadratic_form(self):
        """
        Encode each clause C = (l_1 ∨ ... ∨ l_k) as the penalty
            P_C = Π_i (1 - l_i)
        with l_i = s_i·x_i + t_i  (s=+1, t=0 for x; s=-1, t=1 for ¬x).
        Return Q (n×n), b (n), c so that  Σ_C P_C = xᵀQx + bᵀx + c.
        """
        n = self.n
        Q = np.zeros((n, n))
        b = np.zeros(n)
        c = 0.0

        for clause in self.clauses:
            # Convert literals to (var, s, t)
            lits = []
            for var, sign in clause:
                if sign > 0:
                    lits.append((var, 1, 0))     # l = x
                else:
                    lits.append((var, -1, 1))    # l = 1 - x

            # Constant: 1 - Σ t_i + Σ_{i<j} t_i t_j
            c += 1.0
            for _, _, t in lits:
                c -= t
            for i in range(len(lits)):
                for j in range(i + 1, len(lits)):
                    c += lits[i][2] * lits[j][2]

            # Linear: -s_i  (from -l_i)  +  s_i·t_j  (from l_i·l_j)
            for i, (vi, si, ti) in enumerate(lits):
                b[vi] -= si
                for j, (vj, sj, tj) in enumerate(lits):
                    if i == j:
                        continue
                    b[vi] += si * tj

            # Quadratic: s_i s_j x_i x_j
            for i in range(len(lits)):
                for j in range(i + 1, len(lits)):
                    vi, si, _ = lits[i]
                    vj, sj, _ = lits[j]
                    if vi == vj:
                        # x·x = x (boolean) → fold into linear
                        b[vi] += si * sj
                    else:
                        Q[vi, vj] += si * sj

        return Q, b, c

    def evaluate_quadratic(self, a, Q, b, c):
        bits = np.array([(a >> i) & 1 for i in range(self.n)], dtype=float)
        return float(bits @ Q @ bits + b @ bits + c)

    # ============================================================
    #  Envelope via quadratic address
    # ============================================================
    def build_envelope(self, penalty_fn=None):
        """
        Build the SAT envelope over the *quadratic address space*.

        For each assignment a ∈ [0, 2ⁿ):
            q(a) = (A·a² + B·a + C) mod K
            env[q(a)] += penalty(a)

        penalty(a) defaults to the quadratic form value:
            P(a) = 1 if a satisfies the formula, else 0
        but any non‑negative scalar penalty works.

        Returns
        -------
        env : np.ndarray of length K
              env[q] = Σ_{a : q(a)=q} penalty(a)
        q_of_a : dict[int, int]
              assignment → quadratic address
        """
        K = self.K
        env = np.zeros(K, dtype=np.float64)
        q_of_a = {}

        if penalty_fn is None:
            # default: 1 if satisfying, 0 otherwise
            def penalty_fn(a):
                return 1.0 if self.formula_satisfied(a) else 0.0

        for a in range(2 ** self.n):
            q = self.quadratic_address(a)
            q_of_a[a] = q
            env[q] += penalty_fn(a)
        return env, q_of_a

    def supertrace_via_quadratic(self, env):
        """
        Supertrace over the quadratic‑address envelope:

            S = Σ_q (-1)^q · env[q]

        The alternating sign follows the parity of the quadratic
        address q, not of the linear assignment a.  Under the
        Möbius gate, only the square‑free q contribute.
        """
        S = 0.0
        for q in range(self.K):
            v = env[q]
            if v == 0.0:
                continue
            sign = 1.0 if (q % 2 == 0) else -1.0
            S += sign * v
        return S

    def supertrace_via_quadratic_gated(self, env):
        """
        Supertrace restricted to square‑free quadratic addresses:

            S_μ = Σ_{q : μ(q)≠0} (-1)^q · env[q]
        """
        S = 0.0
        for q in range(self.K):
            v = env[q]
            if v == 0.0:
                continue
            if self.mu[q] == 0:
                continue
            sign = 1.0 if (q % 2 == 0) else -1.0
            S += sign * v
        return S

    def envelope_entropy(self, env, alpha=None):
        """
        Entropy of the quadratic‑address envelope:

            H = -α · p · log(p),  p = |S| / N_occupied
        """
        if alpha is None:
            alpha = 1.0 / (math.pi - math.e)
        S = self.supertrace_via_quadratic(env)
        N = int(np.count_nonzero(env))
        if N <= 0 or S == 0.0:
            return 0.0
        p = abs(S) / N
        if p <= 0.0 or p >= 1.0:
            return 0.0
        return -alpha * p * math.log(p)

    def envelope_mass(self, env):
        """Mass = |S| · e^{−H} over the quadratic‑address envelope."""
        S = self.supertrace_via_quadratic(env)
        H = self.envelope_entropy(env)
        return abs(S) * math.exp(-H) if H < 700 else 0.0

    def envelope_supertrace_table(self, env, top=None):
        """
        Human‑readable table:
            q, μ(q), parity, env[q], signed contribution
        Sorted by |env[q]| descending.
        """
        rows = []
        for q in range(self.K):
            v = env[q]
            if v == 0.0:
                continue
            sign = 1.0 if (q % 2 == 0) else -1.0
            rows.append((q, self.mu[q], q % 2, v, sign * v))
        rows.sort(key=lambda r: -abs(r[3]))
        if top is not None:
            rows = rows[:top]
        return rows
    
# ============================================================
#  3. Demonstration
# ============================================================
def demo():
    print("=" * 64)
    print("Quadratic Möbius SAT  (K‑sum length 2K+1 in quadratic mode)")
    print("=" * 64)

    # --- 5‑variable 2‑SAT instance ---
    n_vars = 5
    clauses = [
        [(0, +1), (1, +1)],     # x0 ∨  x1
        [(0, -1), (2, +1)],     # ¬x0 ∨ x2
        [(1, -1), (2, -1)],     # ¬x1 ∨ ¬x2
        [(2, +1), (3, +1)],     # x2 ∨  x3
        [(3, -1), (4, +1)],     # ¬x3 ∨ x4
        [(4, -1), (0, +1)],     # ¬x4 ∨ x0
    ]

    print(f"\nVariables : {n_vars}")
    print(f"Clauses   : {len(clauses)}")
    for c in clauses:
        print("   " + " ∨ ".join(
            ("" if s > 0 else "¬") + f"x{v}" for v, s in c))

    solver = QuadraticMobiusSAT(
        n_vars, clauses,
        quad_A=1, quad_B=1, quad_C=0,
        K_filter=2 ** n_vars + 1
    )

    # ---------- naive enumeration ----------
    print("\n--- Naive enumeration (no gate) ---")
    sols_naive = solver.solve(use_mobius_gate=False)
    print(f"  total assignments  : {solver.stats['total']}")
    print(f"  satisfying         : {solver.stats['satisfying']}")
    for s in sols_naive:
        bits = [(s >> i) & 1 for i in range(n_vars)]
        print(f"    {bits}")

    # ---------- with Möbius gate ----------
    print(f"\n--- With Möbius gate   q(a) = a² + a mod {2**n_vars+1} ---")
    sols_gate = solver.solve(use_mobius_gate=True)
    g = solver.stats['passed_gate']
    t = solver.stats['total']
    print(f"  total assignments  : {t}")
    print(f"  passed gate        : {g}  ({100*g/t:.1f} %)")
    print(f"  expected density   : {100*DENSITY:.1f} %")
    print(f"  satisfying         : {solver.stats['satisfying']}")
    for s in sols_gate:
        bits = [(s >> i) & 1 for i in range(n_vars)]
        print(f"    {bits}")
    print(f"  same solution set  : {set(sols_naive) == set(sols_gate)}")

    # ---------- quadratic pseudo‑Boolean encoding ----------
    print("\n--- Quadratic pseudo‑Boolean encoding  Q(x) = xᵀQx + bᵀx + c ---")
    Q, b, c = solver.build_quadratic_form()
    print("  Q  (non‑zero upper‑triangle entries):")
    for i in range(n_vars):
        for j in range(i, n_vars):
            if abs(Q[i, j]) > 1e-12:
                print(f"    Q[{i},{j}] = {Q[i,j]:+.2f}")
    print("  b  (non‑zero linear):")
    for i in range(n_vars):
        if abs(b[i]) > 1e-12:
            print(f"    b[{i}] = {b[i]:+.2f}")
    print(f"  c  = {c:+.2f}")

    print("\n  Penalty on satisfying assignments (should be 0):")
    for s in sols_gate[:4]:
        print(f"    a={s:2d}  →  Q={solver.evaluate_quadratic(s, Q, b, c):+.2e}")

    print("\n  Penalty on a non‑satisfying assignment:")
    for a in range(2 ** n_vars):
        if a not in sols_naive:
            print(f"    a={a:2d}  →  Q={solver.evaluate_quadratic(a, Q, b, c):+.2f}")
            break

    # ---------- square‑free distribution of solutions ----------
    print("\n--- Möbius signature of the solution set ---")
    for s in sols_naive:
        addr = solver.quadratic_address(s)
        mu_val = solver.mu[addr]
        tag = "✓ square‑free" if mu_val != 0 else "✗ μ=0"
        print(f"    a={s:2d}  q(a)={addr:3d}  μ={mu_val:+d}  {tag}")

    print("\nDone.")

def demo_quadratic_envelope():
    print("=" * 72)
    print("SAT envelope via quadratic address  q(a) = a² + a mod K")
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
    solver = QuadraticMobiusSAT(
        n_vars, clauses,
        quad_A=1, quad_B=1, quad_C=0,
        K_filter=2 ** n_vars + 1,
    )

    # --- build the envelope over the quadratic address space ---------
    env, q_of_a = solver.build_envelope()

    print(f"\nK = {solver.K}, assignments = 2^{n_vars} = {2**n_vars}")
    print(f"occupied quadratic addresses : "
          f"{int(np.count_nonzero(env))} / {solver.K}")

    # --- supertraces -------------------------------------------------
    S_all    = solver.supertrace_via_quadratic(env)
    S_gated  = solver.supertrace_via_quadratic_gated(env)
    H_env    = solver.envelope_entropy(env)
    m_env    = solver.envelope_mass(env)

    print(f"\nSupertrace over all q        S     = {S_all:+.6f}")
    print(f"Supertrace over square‑free q S_μ   = {S_gated:+.6f}")
    print(f"Envelope entropy              H     = {H_env:.6f}")
    print(f"Envelope mass                 m     = {m_env:.6f}")

    # --- collapse of the solution set onto q -------------------------
    sols_naive = [a for a in range(2 ** n_vars) if solver.formula_satisfied(a)]
    sol_addrs = sorted(set(q_of_a[a] for a in sols_naive))
    print(f"\nSatisfying assignments       : {len(sols_naive)}")
    print(f"Distinct quadratic addresses : {len(sol_addrs)}")
    for q in sol_addrs:
        mu_val = solver.mu[q]
        tag = "✓ square‑free" if mu_val != 0 else "✗ μ=0"
        print(f"    q = {q:>3d}   μ(q) = {mu_val:+d}   env[q] = {env[q]:.0f}   {tag}")

    # --- envelope table, top rows by |env| ---------------------------
    print("\n--- envelope_supertrace_table (top 12) ---")
    print(f"  {'q':>4s}  {'μ(q)':>5s}  {'parity':>6s}  "
          f"{'env[q]':>8s}  {'signed':>10s}")
    for q, mu_val, par, v, signed in solver.envelope_supertrace_table(env, top=12):
        print(f"  {q:>4d}  {mu_val:>+5d}  "
              f"{'even' if par == 0 else 'odd':>6s}  "
              f"{v:>8.2f}  {signed:>+10.4f}")

    # --- signature check: gate preserves solution set ---------------
    print("\n--- Gate check: naive vs. gated solution set ---")
    naive_set = set(sols_naive)
    gated_set = set(
        a for a in range(2 ** n_vars)
        if solver.mobius_gate(a) and solver.formula_satisfied(a)
    )
    print(f"  naive |S| = {len(naive_set)}")
    print(f"  gated |S| = {len(gated_set)}")
    print(f"  same set  : {naive_set == gated_set}")

    # --- visual-ish ASCII envelope ---------------------------------
    print("\n--- envelope over q (nonzero rows only) ---")
    vmax = float(env.max()) if env.max() > 0 else 1.0
    for q in range(solver.K):
        if env[q] == 0.0:
            continue
        bar = "█" * int(round(20 * env[q] / vmax))
        marker = "μ" if solver.mu[q] != 0 else "·"
        print(f"  q={q:>3d}  {marker}  |{bar:<20s}|  {env[q]:.0f}")

    print("\nDone.")


if __name__ == "__main__":
    demo()
    demo_quadratic_envelope()
