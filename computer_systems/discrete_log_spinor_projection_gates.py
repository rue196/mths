#!/usr/bin/env python3
"""
spinor_ugc_bsgs.py
==================

Spinor-projected BSGS over the UGC graph, plus boolean Möbius gate.

    Group structure  =  vertices of the spinor projection operator
                        12 base vertices, tiled by `scale`
    Baby step        =  6-contraction        (720-term Levi-Civita)
    Giant step       =  12 permutations      (cyclic shifts of vertices)
    Edge  (i, j)     =  binary perm          (baby i, giant j)
    UGC semantics    =  kept (discrete log, negative harmonic)
    Boolean          =  μ-gated logic gate embedded in search

Pipeline
--------
    For each weight x to encode / h = g^x to decode:

        1. spinor → 6 baby scalars, 12 giant scalars
        2. edges (i, j) ∈ {0..5} × {0..11}      72 total
        3. each edge's boolean value  b_ij  ∈ {0, 1}:
               a = i mod 2
               b = j mod 2
               b_ij  =  μ-gated AND / OR / XOR of (a, b)
        4. edge priority = |spinor_strength_ij| · (1 + b_ij)
        5. search: highest-priority edges first, BSGS fallback

Imports (unchanged parents)
---------------------------
    zt_projection_embedding   generate_vertices, epsilon_tensor,
                              spinor_projection
    ugc_discrete_log          UGCDiscreteLog, mobius_sieve, harmonic_numbers,
                              modinv, egcd, ALPHA, PI, E, GAMMA
"""

from __future__ import annotations
import hashlib
import math
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---- spinor projector (unchanged) ----
from zt_projection_embedding import (
    generate_vertices,
    epsilon_tensor,
    spinor_projection,
)

# ---- UGC discrete log (unchanged) ----
from ugc_discrete_log import (
    UGCDiscreteLog,
    UGCVertex,
    mobius_sieve,
    harmonic_numbers,
    modinv,
    egcd,
    ALPHA, PI, E, GAMMA,
)


# ============================================================
#  Deterministic hash for edge → exponent mapping
# ============================================================
def _det_hash(a: float, b: float) -> int:
    h = hashlib.sha256(struct.pack(">dd", a, b)).digest()
    return int.from_bytes(h[:8], "big")


# ============================================================
#  Boolean Möbius logic gate
# ============================================================
class MobiusLogicGate:
    """
    Boolean logic gate gated by μ(q) ≠ 0.

        AND(a, b)  = (a AND b) · [μ(q_ab) ≠ 0]
        OR(a, b)   = (a OR  b) · [μ(q_ab) ≠ 0]
        XOR(a, b)  = (a XOR b) · [μ(q_ab) ≠ 0]
        NOT(a)     = (NOT a)   · [μ(q_a)  ≠ 0]

        q_ab = (a² + a + b² + b) mod K_filter
        q_a  = (a² + a)          mod K_filter

    Gate open (μ ≠ 0)  →  passes the boolean result
    Gate closed (μ = 0) →  outputs 0

    Support:
        compose    —  layering gates
        eval_expr  —  parse small boolean expressions
    """
    def __init__(self, K_filter: int = 2048):
        self.K_filter = int(K_filter)
        self.mu = mobius_sieve(K_filter)

    def _gate(self, q: int) -> int:
        q = q % self.K_filter
        return 1 if self.mu[q] != 0 else 0

    # ---------- primitives ----------
    def AND(self, a: int, b: int) -> int:
        a, b = int(a) & 1, int(b) & 1
        q = (a * a + a + b * b + b) % self.K_filter
        return (a & b) * self._gate(q)

    def OR(self, a: int, b: int) -> int:
        a, b = int(a) & 1, int(b) & 1
        q = (a * a + a + b * b + b) % self.K_filter
        return (a | b) * self._gate(q)

    def XOR(self, a: int, b: int) -> int:
        a, b = int(a) & 1, int(b) & 1
        q = (a * a + a + b * b + b) % self.K_filter
        return (a ^ b) * self._gate(q)

    def NOT(self, a: int) -> int:
        a = int(a) & 1
        q = (a * a + a) % self.K_filter
        return (1 - a) * self._gate(q)

    # ---------- expression ----------
    def eval_expr(self, expr: str, inputs: Dict[str, int]) -> int:
        """Evaluate a small boolean expression with μ gating."""
        tokens = expr.replace("(", " ( ").replace(")", " ) ").split()
        pos = [0]

        def peek():
            return tokens[pos[0]] if pos[0] < len(tokens) else None

        def take():
            tok = tokens[pos[0]]
            pos[0] += 1
            return tok

        def primary():
            tok = peek()
            if tok == "(":
                take(); v = expr_or(); 
                if peek() == ")": take()
                return v
            if tok == "NOT":
                take(); return self.NOT(primary())
            if tok and tok[0].isalpha():
                take(); return int(inputs.get(tok, 0)) & 1
            return 0

        def expr_or():
            left = primary()
            while peek() in ("AND", "OR", "XOR"):
                op = take()
                right = primary()
                if op == "AND":   left = self.AND(left, right)
                elif op == "OR":  left = self.OR(left, right)
                else:             left = self.XOR(left, right)
            return left

        return expr_or()


# ============================================================
#  SpinorUGC  ·  group = 12-vertex projection operator
# ============================================================
@dataclass
class SpinorUGC:
    """
    BSGS over the UGC discrete log, structured by the spinor projector.

    Structure
    ---------
        vertices       12 base vertices (from projection operator)
                       tiled by `scale`
        epsilon        ε tensor  —  720 permutations

        baby_table     6 values    —  one per 6-contraction
        giant_table    12 values   —  one per cyclic shift

        edges          (i, j) ∈ {0..5} × {0..11}  =  72 binary perms
                       each edge → candidate exponent
                       each edge → boolean class b_ij ∈ {0, 1}

    Search
    ------
        Phase 1   highest-priority edges  (spinor · boolean)
        Phase 2   BSGS fallback
    """
    p: int
    g: int
    seed_vertices: int = 42
    scale: int = 1
    K_filter: int = 2048
    K_sieve: Optional[int] = None

    # derived
    ugc:     UGCDiscreteLog                 = field(init=False)
    gate:    MobiusLogicGate                = field(init=False)
    vertices: np.ndarray                    = field(init=False)
    epsilon: Dict                           = field(init=False)
    baby_table:    List[float]              = field(init=False)
    giant_table:   List[float]              = field(init=False)
    edges:         Dict[Tuple[int, int], int] = field(init=False)
    edge_bool:     Dict[Tuple[int, int], int] = field(init=False)
    edge_strength: Dict[Tuple[int, int], float] = field(init=False)
    edge_order:    List[Tuple[int, int]]    = field(init=False)
    m:             int                      = field(init=False)
    baby_lookup:   Dict[int, int]           = field(init=False)
    g_inv_m:       int                      = field(init=False)

    # --------------------------------------------------------
    def __post_init__(self):
        # 1. underlying UGC
        self.ugc = UGCDiscreteLog(p=self.p, g=self.g,
                                  K_sieve=self.K_sieve)
        self.gate = MobiusLogicGate(K_filter=self.K_filter)

        # 2. vertex structure
        n_total = 12 * int(self.scale)
        base = generate_vertices(self.seed_vertices)      # (12, 6)
        tiles = n_total // 12 + 1
        self.vertices = np.tile(base, (tiles, 1))[:n_total]
        self.epsilon = epsilon_tensor()

        # 3. baby table  ·  6 contractions
        self.baby_table = []
        for i in range(6):
            v6 = self._slice_6(i)
            self.baby_table.append(
                float(spinor_projection(v6, self.epsilon)))

        # 4. giant table  ·  12 cyclic permutations
        self.giant_table = []
        for j in range(12):
            v_shift = np.roll(self.vertices[:12], j, axis=0)
            self.giant_table.append(
                float(spinor_projection(v_shift[:6], self.epsilon)))

        # 5. edge tables
        self.edges = {}
        self.edge_bool = {}
        self.edge_strength = {}
        for i in range(6):
            for j in range(12):
                # candidate exponent
                h = _det_hash(round(self.baby_table[i], 6),
                              round(self.giant_table[j], 6))
                self.edges[(i, j)] = h % self.ugc.n

                # boolean class  ·  μ-gated AND of (i mod 2, j mod 2)
                a = i % 2
                b = j % 2
                self.edge_bool[(i, j)] = self.gate.AND(a, b)

                # spinor strength  ·  |baby + giant| (bounded)
                self.edge_strength[(i, j)] = float(
                    abs(self.baby_table[i] + self.giant_table[j]))

        # 6. edge priority order  (highest priority first)
        #    strength · (1 + bool) →  boolean-true edges fire first
        self.edge_order = sorted(
            self.edges.keys(),
            key=lambda ij: -(self.edge_strength[ij]
                             * (1 + self.edge_bool[ij])))

        # 7. BSGS fallback
        m = int(math.ceil(math.sqrt(self.ugc.n)))
        self.m = m
        self.baby_lookup = {}
        cur = 1
        for i in range(m):
            if cur not in self.baby_lookup:
                self.baby_lookup[cur] = i
            cur = (cur * self.g) % self.p
        g_inv = modinv(self.g, self.p)
        self.g_inv_m = pow(g_inv, m, self.p)

    # --------------------------------------------------------
    def _slice_6(self, offset: int) -> np.ndarray:
        """6×6 submatrix starting at row `offset` (wrapping)."""
        n = self.vertices.shape[0]
        idx = [(offset + k) % n for k in range(6)]
        return self.vertices[idx]

    # --------------------------------------------------------
    def _lv(self, x: int) -> float:
        """Negative harmonic  −H_{x+1}."""
        H = self.ugc.H
        if 0 <= x and x + 1 < len(H):
            return -float(H[x + 1])
        return 0.0

    # --------------------------------------------------------
    def dlog_spinor(self, h: int
                    ) -> Tuple[Optional[int], Optional[float],
                               Optional[Tuple[int, int]], str]:
        """
        Spinor-driven discrete log.

        Returns (x, −H_{x+1}, edge (i, j), method)
        method ∈ {'spinor', 'bsgs', 'none'}
        """
        h = h % self.p
        if h == 0:
            return None, None, None, 'none'

        # ---- phase 1 · spinor edges in priority order ----
        for ij in self.edge_order:
            x = self.edges[ij]
            if pow(self.g, x, self.p) == h:
                return x, self._lv(x), ij, 'spinor'

        # ---- phase 2 · BSGS fallback ----
        gamma = h
        for j in range(self.m + 1):
            if gamma in self.baby_lookup:
                i = self.baby_lookup[gamma]
                x = (i + j * self.m) % self.ugc.n
                edge = (i % 6, j % 12)
                return x, self._lv(x), edge, 'bsgs'
            gamma = (gamma * self.g_inv_m) % self.p

        return None, None, None, 'none'

    # --------------------------------------------------------
    #  UGC invariants  (unchanged)
    # --------------------------------------------------------
    def supertrace(self) -> float:
        return self.ugc.supertrace()

    def entropy(self) -> float:
        return self.ugc.entropy()

    def invariant_mass(self) -> float:
        return self.ugc.invariant_mass()

    # --------------------------------------------------------
    #  Boolean Möbius evaluation on an edge
    # --------------------------------------------------------
    def boolean_satisfied(self, i: int, j: int,
                          expr: str = "a AND b") -> int:
        """
        Evaluate the edge (i, j) as a boolean Möbius gate.

        a = i mod 2,   b = j mod 2
        """
        return self.gate.eval_expr(expr, {"a": i % 2, "b": j % 2})

    def composite_gate(self, i: int, j: int) -> bool:
        """
        Composite boolean: spinor strength AND mobius μ gate.

        spinor gate   open  if  |baby[i] + giant[j]|  >  threshold
        mobius gate   open  if  μ(q_ij) ≠ 0
        """
        threshold = np.median(list(self.edge_strength.values()))
        spinor_open = self.edge_strength[(i, j)] > threshold
        mobius_open = self.boolean_satisfied(i, j, "a AND b") == 1
        return bool(spinor_open and mobius_open)


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("Spinor UGC  ·  baby = 6-contraction  ·  giant = 12-perm  ·  "
          "boolean Möbius")
    print("=" * 78)
    print(f"  α      = 1/(π − e)   = {ALPHA:.6f}")
    print(f"  γ                    = {GAMMA:.6f}")
    print()

    # ============================================================
    #  1. SpinorUGC structure
    # ============================================================
    print("=" * 78)
    print("1  ·  SpinorUGC  ·  12 vertices × 6 contractions × 12 perms")
    print("=" * 78)
    spin = SpinorUGC(p=10007, g=5, seed_vertices=42, scale=1)
    n_eps_nz = sum(1 for v in spin.epsilon.values() if v != 0)

    print(f"  p                    = {spin.p}")
    print(f"  g                    = {spin.g}")
    print(f"  n (group order)      = {spin.ugc.n}")
    print(f"  vertices shape       = {spin.vertices.shape}")
    print(f"  |ε| non-zero         = {n_eps_nz} / 720")
    print(f"  BSGS m               = {spin.m}")
    print()

    print("--- baby table  ·  6 contractions ---")
    for i in range(6):
        print(f"  baby[{i}]  Π_6 = {spin.baby_table[i]:+.6f}")
    print()

    print("--- giant table  ·  12 cyclic perms ---")
    for j in range(12):
        print(f"  giant[{j:>2d}]  Π_6 = {spin.giant_table[j]:+.6f}")
    print()

    # ============================================================
    #  2. edges + boolean classes
    # ============================================================
    print("=" * 78)
    print("2  ·  edges (i, j)  ·  binary perms  ·  μ-gated boolean")
    print("=" * 78)
    print(f"  {'i':>2s}  {'j':>2s}  {'a=i%2':>5s}  {'b=j%2':>5s}  "
          f"{'μ-AND':>5s}  {'strength':>10s}  {'x_candidate':>11s}")
    print("  " + "-" * 60)
    for (i, j) in spin.edge_order[:12]:
        a = i % 2
        b = j % 2
        print(f"  {i:>2d}  {j:>2d}  {a:>5d}  {b:>5d}  "
              f"{spin.edge_bool[(i, j)]:>5d}  "
              f"{spin.edge_strength[(i, j)]:>10.4f}  "
              f"{spin.edges[(i, j)]:>11d}")
    print(f"  (showing 12 of 72 edges, priority-ordered)")
    print()

    # ============================================================
    #  3. dlog via spinor
    # ============================================================
    print("=" * 78)
    print("3  ·  discrete log  ·  spinor + BSGS fallback")
    print("=" * 78)
    rng = np.random.default_rng(4)
    tests = rng.integers(0, spin.ugc.n, size=8)
    print(f"  {'x_true':>7s}  {'h = g^x':>10s}  {'x_hat':>7s}  "
          f"{'−H_{x+1}':>12s}  {'(i,j)':>8s}  {'method':>7s}  ok")
    print("  " + "-" * 74)
    for x_true in tests:
        h = int(spin.ugc.powers[x_true])
        x_hat, lv, ij, method = spin.dlog_spinor(h)
        ok = (x_hat == x_true) if x_hat is not None else False
        ij_s = f"({ij[0]},{ij[1]})" if ij else "—"
        print(f"  {int(x_true):>7d}  {h:>10d}  "
              f"{(x_hat if x_hat is not None else '—'):>7}  "
              f"{(lv if lv is not None else 0):>+12.6f}  "
              f"{ij_s:>8s}  {method:>7s}  "
              f"{'✓' if ok else '✗'}")
    print()

    # ============================================================
    #  4. UGC invariants
    # ============================================================
    print("=" * 78)
    print("4  ·  UGC invariants  (unchanged)")
    print("=" * 78)
    S = spin.supertrace()
    H = spin.entropy()
    m = spin.invariant_mass()
    print(f"  S (supertrace)       = {S:+.6f}")
    print(f"  H (entropy)          = {H:.6f}")
    print(f"  m (invariant mass)   = {m:.6e}")
    print()

    # ============================================================
    #  5. Boolean Möbius gate  ·  truth tables + expressions
    # ============================================================
    print("=" * 78)
    print("5  ·  Boolean Möbius gate  ·  μ-gated AND / OR / XOR / NOT")
    print("=" * 78)
    gate = spin.gate
    mu_nz = int(np.count_nonzero(gate.mu))
    mu_total = len(gate.mu)
    print(f"  K_filter             = {gate.K_filter}")
    print(f"  μ density            = {mu_nz}/{mu_total} = "
          f"{mu_nz / mu_total:.4f}")
    print()

    print("--- truth table  ·  AND / OR / XOR ---")
    print(f"  {'a':>2s}  {'b':>2s}  {'q_ab':>5s}  {'μ(q_ab)':>7s}  "
          f"{'AND':>4s}  {'OR':>4s}  {'XOR':>4s}")
    print("  " + "-" * 36)
    for a in (0, 1):
        for b in (0, 1):
            q = (a*a + a + b*b + b) % gate.K_filter
            mu_v = int(gate.mu[q])
            print(f"  {a:>2d}  {b:>2d}  {q:>5d}  {mu_v:>+7d}  "
                  f"{gate.AND(a, b):>4d}  {gate.OR(a, b):>4d}  "
                  f"{gate.XOR(a, b):>4d}")
    print()

    print("--- truth table  ·  NOT ---")
    print(f"  {'a':>2s}  {'q_a':>4s}  {'μ(q_a)':>7s}  {'NOT':>5s}")
    print("  " + "-" * 22)
    for a in (0, 1):
        q = (a*a + a) % gate.K_filter
        mu_v = int(gate.mu[q])
        print(f"  {a:>2d}  {q:>4d}  {mu_v:>+7d}  {gate.NOT(a):>5d}")
    print()

    # expressions
    print("--- expressions ---")
    exprs = [
        ("a AND b",        {"a": 1, "b": 1}),
        ("a AND b",        {"a": 1, "b": 0}),
        ("a OR b",         {"a": 0, "b": 1}),
        ("a XOR b",        {"a": 1, "b": 1}),
        ("(a AND b) OR c", {"a": 1, "b": 1, "c": 0}),
        ("NOT a",          {"a": 0}),
        ("NOT (a AND b)",  {"a": 1, "b": 1}),
        ("a XOR (NOT b)",  {"a": 0, "b": 1}),
    ]
    for expr, inputs in exprs:
        try:
            v = gate.eval_expr(expr, inputs)
            ins = ", ".join(f"{k}={v}" for k, v in inputs.items())
            print(f"  {expr:<18s}  [{ins}]  →  {v}")
        except Exception as e:
            print(f"  {expr:<18s}  error: {e}")
    print()

    # ============================================================
    #  6. Combined  ·  edges + boolean class
    # ============================================================
    print("=" * 78)
    print("6  ·  combined  ·  spinor edges filtered by μ-gate")
    print("=" * 78)
    print("  For each edge (i, j):")
    print("    a = i mod 2,  b = j mod 2")
    print("    μ-gate  b_ij  = AND(a, b)")
    print("    spinor strength  s_ij  = |baby[i] + giant[j]|")
    print("    composite satisfied  ⇔  spinor_open(i,j) AND μ-AND(i,j)")
    print()
    print(f"  {'(i,j)':>8s}  {'a':>2s}  {'b':>2s}  {'μ-AND':>5s}  "
          f"{'strength':>10s}  {'composite':>10s}")
    print("  " + "-" * 46)
    for (i, j) in spin.edge_order[:12]:
        composite = spin.composite_gate(i, j)
        print(f"  ({i},{j})  {i % 2:>2d}  {j % 2:>2d}  "
              f"{spin.edge_bool[(i, j)]:>5d}  "
              f"{spin.edge_strength[(i, j)]:>10.4f}  "
              f"{'open' if composite else 'closed':>10s}")
    print()

    # ============================================================
    #  7. Complexity
    # ============================================================
    print("=" * 78)
    print("7  ·  complexity")
    print("=" * 78)
    print("  SpinorUGC.__init__:")
    print("    ε tensor              O(6!) = O(720)")
    print("    12 vertices           O(12 · 6) = O(1)")
    print("    baby table            O(6 · 720) = O(720)")
    print("    giant table           O(12 · 720) = O(8640)")
    print("    edge table            O(6 · 12) = O(72)")
    print("    BSGS tables           O(√n)")
    print()
    print("  dlog_spinor:")
    print("    phase 1  spinor       O(72 · log p) worst case")
    print("    phase 2  BSGS         O(√n · log p)")
    print("    →  amortised          O(√n · log p)")
    print()
    print("  MobiusLogicGate:")
    print("    sieve                 O(K log log K)")
    print("    gate lookup           O(1)")
    print("    expression parser     O(len(expr))")
    print()
    print("  Compound:")
    print("    search order          sorted by spinor · boolean")
    print("    →  boolean-true edges fire first")
    print("    →  high-strength edges fire first")
    print()
    print("Done.")


if __name__ == "__main__":
    demo()