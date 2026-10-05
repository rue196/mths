#!/usr/bin/env python3
"""
reasoning_buffer.py
===================

Long-term thought with an elliptic Möbius SAT buffer.

    ┌─────────────────────────────────────────────────────────┐
    │  PRESENTATION LAYER  (user-facing)                      │
    │    • clean prose                                        │
    │    • the SAT machinery is never mentioned               │
    └─────────────────────────────────────────────────────────┘
                          ▲
                          │  only accepted statements
                          │  are formatted for output
    ┌─────────────────────────────────────────────────────────┐
    │  REASONING LAYER  (hidden)                              │
    │    • elliptic traces  tr(a) = x^(i-1) + y^(i-1)         │
    │    • quadratic address  q(a) = A a² + B a + C  mod K    │
    │    • Möbius gate  μ(q(a)) ≠ 0                           │
    │    • supertrace  S = Σ (−1)^t |conv_t|                  │
    │    • mass  m = |S| e^{−H}                               │
    │                                                          │
    │  this is the buffer — it decides, then throws away      │
    │  every intermediate trace and keeps only the surviving  │
    │  statement's text + its arity                           │
    └─────────────────────────────────────────────────────────┘

Arity as a natural number
-------------------------
Every statement carries a list of arguments.  Its arity is

    k = len(args)     ∈ {1, 2, 3, …}

which is exactly the n/k in the elliptic SAT setting:

    k = 1  →  unary     "sunny"
    k = 2  →  binary    "sort(x, ascending)"
    k = 3  →  ternary   "choose(a, b, c)"
    ...

The SAT maps these arguments to a boolean assignment, evaluates the
query's requirements as clauses, and gates through the Möbius sieve.

Long-term thought
-----------------
The buffer persists across cycles.  Every candidate statement is
appended to a running state; the state is re-solved at each step;
the highest-mass surviving statement becomes the current thought.
"""

from __future__ import annotations

import math
import cmath
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================
#  Constants
# ============================================================
PI      = math.pi
E       = math.e
ALPHA   = 1.0 / (PI - E)                  # ≈ 2.362
NORM    = 1.0 - math.exp(-ALPHA * (PI + E))
DENSITY = 6.0 / (PI * PI)                 # ≈ 0.6079


# ============================================================
#  Möbius sieve
# ============================================================
def mobius_sieve(K: int) -> List[int]:
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
#  Statement and Query
# ============================================================
@dataclass
class Statement:
    """A candidate answer — a thought."""
    text: str                                  # the natural-language form
    args: List[str] = field(default_factory=list)  # the arguments
    source: str = "buffer"                     # tag for debugging

    @property
    def arity(self) -> int:
        """k = 1, 2, 3, … — the natural arity of the statement."""
        return len(self.args)


@dataclass
class Query:
    """A user query with a set of requirements."""
    text: str
    # requirements expressed as clauses over argument slots
    # each clause is a list of (slot_index, sign)
    clauses: List[List[Tuple[int, int]]] = field(default_factory=list)


# ============================================================
#  Elliptic Möbius SAT  —  the reasoning buffer's core
# ============================================================
class EllipticMobiusSAT:
    """
    Hidden reasoning layer.

    Maps a statement's arguments to a boolean assignment, evaluates
    the query's clauses, gates through the Möbius sieve, and returns
    a decision.

    Nothing in this class is ever surfaced to the user.
    """
    def __init__(self,
                 quad_A: int = 1,
                 quad_B: int = 1,
                 quad_C: int = 0,
                 i_exp: float = 2.0,
                 K_filter: int = 2 ** 8 + 1):
        self.A, self.B, self.C = quad_A, quad_B, quad_C
        self.i_exp = i_exp
        self.K = K_filter
        self.mu = mobius_sieve(self.K)

    # ---- argument vector → boolean assignment ----
    @staticmethod
    def args_to_bits(args: List[str], width: int = 8) -> int:
        """
        Hash the argument list to an integer of `width` bits.
        Each argument contributes one bit (present if non-empty).
        """
        bits = 0
        for i, a in enumerate(args[:width]):
            if a:
                bits |= (1 << i)
        return bits

    # ---- elliptic address ----
    def elliptic_address(self, a: int, n_bits: int) -> Tuple[float, float]:
        bits = [(a >> i) & 1 for i in range(n_bits)]
        t = sum(b * math.sin((i + 1) * 0.7) for i, b in enumerate(bits))
        u = sum(b * math.cos((i + 1) * 1.3) for i, b in enumerate(bits))
        x = 0.5 + 4.5 * (0.5 * (1.0 + math.tanh(t)))
        y = 0.5 + 4.5 * (0.5 * (1.0 + math.tanh(u)))
        return x, y

    # ---- matrix trace ----
    def trace_of(self, a: int, n_bits: int) -> float:
        x, y = self.elliptic_address(a, n_bits)
        return x ** (self.i_exp - 1) + y ** (self.i_exp - 1)

    # ---- quadratic address + Möbius gate ----
    def quadratic_address(self, a: int) -> int:
        return (self.A * a * a + self.B * a + self.C) % self.K

    def mobius_gate(self, a: int) -> bool:
        return self.mu[self.quadratic_address(a)] != 0

    # ---- clause evaluation ----
    @staticmethod
    def _lit(lit, a):
        slot, sign = lit
        bit = (a >> slot) & 1
        return bool(bit) if sign > 0 else not bool(bit)

    @classmethod
    def clause_true(cls, clause, a) -> bool:
        return any(cls._lit(l, a) for l in clause)

    @classmethod
    def formula_true(cls, clauses, a) -> bool:
        return all(cls.clause_true(c, a) for c in clauses)

    # ---- decision ----
    def decide(self, stmt: Statement, query: Query) -> Dict:
        """
        Run the buffer on one statement.
        Returns the internal record (never shown to the user).
        """
        k = stmt.arity
        n_bits = max(4, min(8, k + 2))
        a = self.args_to_bits(stmt.args, width=n_bits)
        q = self.quadratic_address(a)
        gate = self.mobius_gate(a)
        formula = self.formula_true(query.clauses, a)
        trace = self.trace_of(a, n_bits)
        accepted = gate and formula
        return dict(
            statement=stmt,
            arity=k,
            assignment=a,
            q_addr=q,
            mu_q=self.mu[q],
            gate=gate,
            formula=formula,
            trace=trace,
            accepted=accepted,
        )


# ============================================================
#  Exponential kernel  +  supertrace / mass
# ============================================================
def conv_exp_kernel(signal: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    K = len(signal)
    if K == 0:
        return signal
    lam = math.exp(-alpha)
    f = np.zeros(K); f[0] = signal[0]
    for i in range(1, K):
        f[i] = signal[i] + lam * f[i - 1]
    b = np.zeros(K); b[K - 1] = signal[-1]
    for i in range(K - 2, -1, -1):
        b[i] = signal[i] + lam * b[i + 1]
    conv_exp = (f + b - signal) / (1 - lam * lam)
    return (1.0 - conv_exp) / NORM


def supertrace_and_mass(signal: np.ndarray) -> Tuple[float, float, float]:
    S = 0.0
    for i, v in enumerate(signal):
        S += v if (i % 2 == 0) else -v
    if S == 0.0:
        return 0.0, 0.0, 0.0
    p = abs(S) / len(signal)
    if not (0.0 < p < 1.0):
        return S, 0.0, abs(S)
    H = -ALPHA * p * math.log(p)
    m = abs(S) * math.exp(-H) if H < 700 else 0.0
    return S, H, m


# ============================================================
#  Reasoning buffer  —  long-term thought state
# ============================================================
class ReasoningBuffer:
    """
    Persistent buffer for long-term thought.

    Each cycle:
        1. receive a batch of candidate statements
        2. run the hidden SAT on each
        3. keep the survivors (accepted == True)
        4. compress survivors' traces through the exponential kernel
        5. keep the top-M by magnitude
        6. update the running thought state

    The state is a list of accepted statements ordered by mass.
    Only the *text* of the current best statement ever leaves this
    class, and only when `emit()` is called.
    """
    def __init__(self, sat: Optional[EllipticMobiusSAT] = None,
                 max_survivors: int = 32):
        self.sat = sat or EllipticMobiusSAT()
        self.max_survivors = max_survivors
        self.accepted: List[Dict] = []       # accepted reasoning records
        self.rejected: List[Dict] = []       # rejected records (for stats)
        self.cycle = 0

    # ---------- one cycle of thought ----------
    def think(self, query: Query,
              candidates: List[Statement]) -> Dict:
        self.cycle += 1
        cycle_records = []
        for stmt in candidates:
            rec = self.sat.decide(stmt, query)
            cycle_records.append(rec)
            if rec["accepted"]:
                self.accepted.append(rec)
            else:
                self.rejected.append(rec)

        # ---- supertrace over the cycle's trace ensemble ----
        traces = np.array([r["trace"] for r in cycle_records], dtype=float)
        if len(traces) >= 2:
            conv = conv_exp_kernel(traces)
            S, H, m = supertrace_and_mass(conv)
        else:
            S, H, m = 0.0, 0.0, 0.0

        # ---- keep the top survivors by |trace| ----
        if self.accepted:
            self.accepted.sort(key=lambda r: -abs(r["trace"]))
            self.accepted = self.accepted[: self.max_survivors]

        return dict(
            cycle=self.cycle,
            n_candidates=len(candidates),
            n_accepted=sum(1 for r in cycle_records if r["accepted"]),
            n_rejected=sum(1 for r in cycle_records if not r["accepted"]),
            S=S, H=H, m=m,
            best=self.accepted[0] if self.accepted else None,
        )

    # ---------- the ONLY external output ----------
    def emit(self) -> Optional[str]:
        """
        Return the natural text of the current best statement,
        or None if the buffer is empty.

        The user sees nothing about SAT, Möbius gates, traces,
        supertraces, or masses — only clean prose.
        """
        if not self.accepted:
            return None
        return self.accepted[0]["statement"].text

    # ---------- debugging view (never called in production) ----------
    def debug_state(self) -> List[Dict]:
        return [
            dict(text=r["statement"].text,
                 arity=r["arity"],
                 trace=round(r["trace"], 4),
                 q=r["q_addr"],
                 mu_q=r["mu_q"])
            for r in self.accepted
        ]


# ============================================================
#  Presenter  —  the public face
# ============================================================
class Presenter:
    """
    Formats the reasoning buffer's output for the user.

    It receives only the winning statement's text.  It never
    sees the SAT internals.
    """
    def __init__(self):
        self.history: List[str] = []

    def present(self, text: Optional[str]) -> str:
        if text is None:
            reply = "I don't have a considered answer for that yet."
        else:
            reply = text
        self.history.append(reply)
        return reply


# ============================================================
#  Demo  —  long-term thought across multiple cycles
# ============================================================
def demo():
    print("=" * 74)
    print("Reasoning buffer  ·  elliptic Möbius SAT as internal if/else")
    print("=" * 74)
    print(f"  α = 1/(π − e) = {ALPHA:.6f}")
    print(f"  6/π²          = {DENSITY:.6f}  (gate density)")
    print()

    # ---- the query and its requirements ----
    query = Query(
        text="How do I sort a large list quickly?",
        clauses=[
            # requirements over argument slots
            [(0, +1)],            # arg 0 must be present
            [(1, +1)],            # arg 1 must be present
            [(2, +1)],            # arg 2 must be present
        ],
    )

    # ---- candidates with different arities ----
    candidates = [
        # k = 1 (unary)
        Statement("Sort the list.",           args=["list"]),
        # k = 2 (binary)
        Statement("Sort the list ascending.", args=["list", "asc"]),
        Statement("Sort the list descending.",args=["list", "desc"]),
        # k = 3 (ternary) — the fully-specified answers
        Statement("Use quicksort on the list with ascending order.",
                  args=["quicksort", "list", "asc"]),
        Statement("Use mergesort on the list for stable ascending output.",
                  args=["mergesort", "list", "asc"]),
        Statement("Use heapsort with in-place ascending order.",
                  args=["heapsort", "list", "asc"]),
        # k = 4 (quaternary) — over-specified
        Statement("Use radix sort on ints ascending, stable.",
                  args=["radix", "ints", "asc", "stable"]),
    ]

    # ---- the reasoning buffer ----
    buf = ReasoningBuffer(
        sat=EllipticMobiusSAT(
            quad_A=1, quad_B=1, quad_C=0,
            i_exp=2.0, K_filter=2 ** 8 + 1,
        ),
        max_survivors=16,
    )
    presenter = Presenter()

    # ---- cycle 1: broad exploration ----
    r1 = buf.think(query, candidates)
    print("--- cycle 1: broad exploration ---")
    print(f"  candidates   : {r1['n_candidates']}")
    print(f"  accepted     : {r1['n_accepted']}")
    print(f"  rejected     : {r1['n_rejected']}")
    print(f"  supertrace S : {r1['S']:+.4f}")
    print(f"  entropy   H  : {r1['H']:.4f}")
    print(f"  mass      m  : {r1['m']:.4f}")
    print()

    # ---- hidden internal state (debug only) ----
    print("--- internal buffer state (never shown to user) ---")
    print(f"  {'text':52s}  {'k':>2s}  {'tr(a)':>8s}  "
          f"{'q':>4s}  {'μ(q)':>4s}")
    for r in buf.debug_state():
        t = r['text'][:50] + (".." if len(r['text']) > 50 else "")
        print(f"  {t:52s}  {r['arity']:>2d}  {r['trace']:>+8.4f}  "
              f"{r['q']:>4d}  {r['mu_q']:>+4d}")
    print()

    # ---- cycle 2: refinement ----
    refinements = [
        Statement("Quicksort with median-of-three pivot, ascending.",
                  args=["quicksort", "pivot", "asc"]),
        Statement("Mergesort is optimal for stable large lists.",
                  args=["mergesort", "stable", "large"]),
    ]
    r2 = buf.think(query, refinements)
    print("--- cycle 2: refinement ---")
    print(f"  candidates   : {r2['n_candidates']}")
    print(f"  accepted     : {r2['n_accepted']}")
    print(f"  supertrace S : {r2['S']:+.4f}")
    print(f"  mass      m  : {r2['m']:.4f}")
    print()

    # ---- the ONLY user-facing output ----
    print("--- user-facing response ---")
    reply = presenter.present(buf.emit())
    print(f"  > {reply}")
    print()

    # ---- what was never said to the user ----
    print("--- what the user never sees ---")
    print("  · elliptic traces   tr(a) = x^(i−1) + y^(i−1)")
    print("  · quadratic address q(a) = a² + a mod K")
    print("  · Möbius gate       μ(q(a)) ≠ 0")
    print("  · supertrace        S = Σ (−1)^t |conv_t|")
    print("  · mass              m = |S| e^{−H}")
    print("  · arity k           1 (unary), 2 (binary), 3 (ternary), …")
    print("  · accepted / rejected counts, cycle numbers")
    print()

    # ---- statistics ----
    print("--- buffer statistics ---")
    print(f"  total cycles    : {buf.cycle}")
    print(f"  total accepted  : {len(buf.accepted)}")
    print(f"  total rejected  : {len(buf.rejected)}")
    gate_density = sum(1 for r in buf.accepted + buf.rejected
                       if r["gate"]) / max(1, len(buf.accepted + buf.rejected))
    print(f"  gate density    : {gate_density:.4f}  "
          f"(expected ≈ 6/π² = {DENSITY:.4f})")
    print()

    # ---- arity distribution ----
    arities = {}
    for r in buf.accepted + buf.rejected:
        k = r["arity"]
        arities[k] = arities.get(k, 0) + 1
    print("--- arity distribution (k = 1, 2, 3, …) ---")
    for k in sorted(arities):
        kind = {1: "unary", 2: "binary", 3: "ternary", 4: "quaternary"}.get(
            k, f"{k}-ary")
        print(f"  k = {k:>2d}  ({kind:<11s})  count = {arities[k]}")
    print()

    print("Done.")


if __name__ == "__main__":
    demo()