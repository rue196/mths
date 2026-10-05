#!/usr/bin/env python3
"""
sparse_attn_sym_asym.py
=======================

Sparse attention with separated symmetric / asymmetric regimes.

Architecture
------------
    query  →  tokenize  →  classify each token
               │
               ├─ symmetric  (deterministic)  →  linear sieve    O(K log log K)
               └─ asymmetric (creative)       →  segmented sieve O(K log K)

    each head builds a sparse attention mask over its own subspace:

        sym × sym       masked by   μ(i·j mod K) ≠ 0     ← linear sieve
        asym × asym     masked by   μ((i² + j²) mod K) ≠ 0  ← segmented sieve
        sym × asym      gated by    elliptic Möbius SAT  ← cross-regime

    attention weights:

        A[i,j] = Π(q_i, k_j) · mask[i,j] · 1/(1 + |S_trans|)

    where Π is the Figure 3.5 elliptic projection and S_trans is the
    transcendental supertrace of the query block.

Everything is O(K log log K) for symmetric, O(K log K) for asymmetric.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================
#  Constants
# ============================================================
PI          = math.pi
E           = math.e
ALPHA_SYM   = 1.0 / (PI - E)              # ≈ 2.362
ALPHA_ASYM  = 0.3628
DENSITY     = 6.0 / (PI * PI)             # ≈ 0.6079
K_DEFAULT   = 64
SYM_THRESHOLD = 0.35


# ============================================================
#  Two Möbius sieves
# ============================================================
def mobius_sieve_linear(K: int) -> np.ndarray:
    """
    Linear sieve  ·  O(K log log K).
    Used for the symmetric (deterministic) regime.
    """
    if K < 1:
        return np.zeros(K + 1, dtype=np.int8)
    mu = [0] * (K + 1)
    mu[1] = 1
    primes: List[int] = []
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
    return np.array(mu, dtype=np.int8)


def mobius_sieve_segmented(K: int) -> np.ndarray:
    """
    Segmented sieve  ·  O(K log K).
    Used for the asymmetric (creative) regime.

    Slower than linear but block-friendlier: works on slices,
    which matches the fluid character of the asymmetric path.
    """
    if K < 1:
        return np.zeros(K + 1, dtype=np.int8)
    mu = np.zeros(K + 1, dtype=np.int8)
    mu[1] = 1
    for n in range(2, K + 1):
        mu[n] = 1

    limit = int(math.isqrt(K)) + 1
    base_primes: List[int] = []
    is_comp = [False] * (limit + 1)
    for i in range(2, limit + 1):
        if not is_comp[i]:
            base_primes.append(i)
            for j in range(i * i, limit + 1, i):
                is_comp[j] = True

    # mark p² multiples as 0
    for p in base_primes:
        p2 = p * p
        mu[p2::p2] = 0
    # flip sign for each prime factor
    for p in base_primes:
        mu[p::p] = -mu[p::p]
    return mu


# ============================================================
#  Elliptic projection  ·  Figure 3.5
# ============================================================
def elliptic_projection(x: float, y: float) -> float:
    u = (x / PI) % 1.0
    v = (y / E) % 1.0
    return 0.5 * (1.0 + math.cos(2 * PI * u) * math.cos(2 * PI * v))


# ============================================================
#  Deterministic / asymmetric classifier  (from the router)
# ============================================================
_ARITH_RE = re.compile(r"^[\s0-9\.\+\-\*/\(\)%eE]+$")
_COUNT_LETTER_RE = re.compile(
    r"how\s+many\s+['\"]?(?P<ch>\w)['\"]?\s+"
    r"(?:in|are\s+in|does)\s+['\"]?(?P<word>\w+)['\"]?",
    re.IGNORECASE | re.VERBOSE)
_COUNT_WORD_RE = re.compile(
    r"how\s+many\s+['\"]?(?P<w>\w+)['\"]?\s+"
    r"(?:in|are\s+in|does)\s+['\"]?(?P<text>.+?)['\"]?$",
    re.IGNORECASE | re.VERBOSE)
_LEN_RE = re.compile(
    r"(?:length|len)\s+of\s+['\"]?(?P<w>\w+)['\"]?",
    re.IGNORECASE)

SYM_WORDS = {
    "a", "an", "the", "in", "on", "at", "by", "for", "with", "from",
    "to", "of", "and", "or", "but", "nor", "so", "yet",
    "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those", "some", "any", "all",
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "do", "does", "did",
    "will", "would", "shall", "should", "can", "could",
    "may", "might", "must",
    "what", "when", "where", "why", "how", "who", "which",
    ".", ",", "?", "!", ";", ":", "-", "'", '"',
}


def elliptic_projection_sym(f: float, g: float) -> float:
    f = max(abs(f), 1e-12)
    g = max(abs(g), 1e-12)
    u = (1.0 / f) % 1.0
    v = (1.0 / g) % 1.0
    return 0.5 * (1.0 + math.cos(2 * PI * u) * math.cos(2 * PI * v))


def elliptic_projection_asym(f: float, g: float) -> float:
    f = max(abs(f), 1e-12)
    g = max(abs(g), 1e-12)
    p_re = math.cos(math.log(f))
    q_re = math.cos(math.log(g))
    return 0.5 * (1.0 + math.cos(2 * PI * p_re)
                       * math.cos(2 * PI * q_re))


def query_scalars(text: str) -> Tuple[float, float]:
    h = int(hashlib.md5(text.encode()).hexdigest()[:16], 16)
    f = 0.5 + ((h & 0xFFFF) / 0xFFFF) * 4.5
    g = 0.5 + (((h >> 16) & 0xFFFF) / 0xFFFF) * 4.5
    return f, g


def classify_token(tok: str) -> str:
    """
    Classify a single token as 'sym' or 'asym'.

    Rules
    -----
    · arithmetic symbols       → sym
    · function words            → sym
    · content words             → asym
    · a projection-based fallback decides when the token is not in
      SYM_WORDS and has no clear morphology
    """
    w = tok.lower().strip()
    if not w:
        return "sym"
    if _ARITH_RE.match(w):
        return "sym"
    if w in SYM_WORDS:
        return "sym"
    # short alphabetic words that aren't in the dictionary — projection test
    if len(w) <= 3:
        f, g = query_scalars(w)
        p_sym  = elliptic_projection_sym(f, g)
        p_asym = elliptic_projection_asym(f, g)
        gap = abs(p_sym - p_asym)
        return "sym" if gap < SYM_THRESHOLD else "asym"
    return "asym"


def classify_query(query: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Return (tokens, sym_flags, asym_flags) for a query.
    The two flags are disjoint boolean lists.
    """
    tokens = re.findall(r"\w+|[.,!?;:'\"]", query.lower().strip())
    sym_flags:  List[str] = []
    asym_flags: List[str] = []
    kinds = [classify_token(t) for t in tokens]
    for k in kinds:
        sym_flags.append(k == "sym")
        asym_flags.append(k == "asym")
    return tokens, sym_flags, asym_flags


# ============================================================
#  Sparse attention weights
# ============================================================
@dataclass
class SparseAttention:
    """
    Sparse attention weights for one query.

        A[i, j] = Π(q_i, k_j) · mask[i, j] · decay(S_trans)
        mask[i, j] = 1 iff regime(i, j) passes its Möbius gate

    Symmetric mask :  μ(i · j mod K) ≠ 0            linear sieve
    Asymmetric mask:  μ((i² + j²) mod K) ≠ 0        segmented sieve
    Cross mask     :  elliptic Möbius SAT           (see SATGate below)
    """
    K: int
    sym_mask:  np.ndarray    # (N, N) bool
    asym_mask: np.ndarray    # (N, N) bool
    cross_mask: np.ndarray   # (N, N) bool
    weights:   np.ndarray    # (N, N) float
    dense:     np.ndarray    # (N, N) float — for comparison

    def sparsity(self) -> Dict[str, float]:
        N = self.weights.shape[0]
        total = N * N
        nz    = int(np.count_nonzero(self.weights))
        return dict(
            n_total=total,
            n_nonzero=nz,
            sparsity=1.0 - nz / total,
            density=nz / total,
        )


class SymmetricSieveMask:
    """
    Linear sieve, O(K log log K).

    Gate:  μ(i · j mod K) ≠ 0
    """
    def __init__(self, K: int):
        self.K = K
        self.mu = mobius_sieve_linear(K)

    def build(self, N: int) -> np.ndarray:
        mask = np.zeros((N, N), dtype=bool)
        K = self.K
        for i in range(N):
            for j in range(N):
                q = (i * j) % K
                if self.mu[q] != 0:
                    mask[i, j] = True
        return mask


class AsymmetricSieveMask:
    """
    Segmented sieve, O(K log K).

    Gate:  μ((i² + j²) mod K) ≠ 0
    """
    def __init__(self, K: int):
        self.K = K
        self.mu = mobius_sieve_segmented(K)

    def build(self, N: int) -> np.ndarray:
        mask = np.zeros((N, N), dtype=bool)
        K = self.K
        for i in range(N):
            ii = (i * i) % K
            for j in range(N):
                q = (ii + (j * j)) % K
                if self.mu[q] != 0:
                    mask[i, j] = True
        return mask


class SATGate:
    """
    Cross-regime gate, built from the elliptic Möbius SAT.

    For a symmetric token at position i and an asymmetric token at j,
    the pair is admitted iff

        μ(q) ≠ 0     with   q = (i·K + j)² + (i·K + j)  mod K
    """
    def __init__(self, K: int):
        self.K = K
        self.mu = mobius_sieve_linear(K)             # shared table

    def build(self, N: int, sym_flags: List[bool],
              asym_flags: List[bool]) -> np.ndarray:
        K = self.K
        mask = np.zeros((N, N), dtype=bool)
        for i in range(N):
            if not sym_flags[i]:
                continue
            for j in range(N):
                if not asym_flags[j]:
                    continue
                a = (i * K + j) % (K * K)
                q = ((a * a) + a) % K
                if self.mu[q] != 0:
                    mask[i, j] = True
                    mask[j, i] = True                # symmetry
        return mask


# ============================================================
#  Query embedding (keeps the projection simple)
# ============================================================
def token_to_xy(tok: str, K: int) -> Tuple[float, float]:
    h = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)
    x = 0.5 + ((h & 0xFFFF) / 0xFFFF) * 4.5
    y = 0.5 + (((h >> 16) & 0xFFFF) / 0xFFFF) * 4.5
    return x, y


def transcendental_supertrace(tokens: List[str], K: int) -> float:
    """S_trans = Σ (−1)^n Re((x + i y)^i) over token embeddings."""
    S = 0.0
    for n, t in enumerate(tokens):
        x, y = token_to_xy(t, K)
        z = complex(x, y) ** 1j
        val = z.real
        S += val if (n % 2 == 0) else -val
    return float(S)


# ============================================================
#  Sparse attention layer
# ============================================================
class SparseSymAsymAttention:
    """
    The full sparse attention layer.

    Steps
    -----
    1. classify tokens   → sym / asym flags
    2. build sym mask    → linear sieve    O(K log log K)
    3. build asym mask   → segmented sieve O(K log K)
    4. build cross mask  → elliptic Möbius SAT
    5. combine masks     → per-pair gate
    6. weight            → Π(q_i, k_j) · gate · decay(S_trans)
    """
    def __init__(self, K: int = K_DEFAULT):
        self.K = K
        self.sym_sieve  = SymmetricSieveMask(K)
        self.asym_sieve = AsymmetricSieveMask(K)
        self.sat_gate   = SATGate(K)

    def build(self, query: str) -> SparseAttention:
        tokens, sym_flags, asym_flags = classify_query(query)
        N = len(tokens)
        if N == 0:
            empty = np.zeros((0, 0))
            return SparseAttention(self.K, empty, empty, empty,
                                   empty, empty)

        # ---- 1. symmetric mask  (linear sieve) ----
        sym_mask = self.sym_sieve.build(N)
        # restrict to positions whose tokens are actually symmetric
        for i in range(N):
            if not sym_flags[i]:
                sym_mask[i, :] = False
                sym_mask[:, i] = False

        # ---- 2. asymmetric mask  (segmented sieve) ----
        asym_mask = self.asym_sieve.build(N)
        for i in range(N):
            if not asym_flags[i]:
                asym_mask[i, :] = False
                asym_mask[:, i] = False

        # ---- 3. cross mask  (elliptic Möbius SAT) ----
        cross_mask = self.sat_gate.build(N, sym_flags, asym_flags)

        # ---- 4. total gate ----
        gate = sym_mask | asym_mask | cross_mask

        # ---- 5. weight from elliptic projection and decay ----
        S_trans = transcendental_supertrace(tokens, self.K)
        decay = 1.0 / (1.0 + abs(S_trans))

        weights = np.zeros((N, N), dtype=float)
        for i in range(N):
            xi, yi = token_to_xy(tokens[i], self.K)
            for j in range(N):
                if not gate[i, j]:
                    continue
                xj, yj = token_to_xy(tokens[j], self.K)
                Pi = elliptic_projection(xi + xj, yi + yj)
                weights[i, j] = Pi * decay

        # ---- 6. dense reference for comparison ----
        dense = np.zeros((N, N), dtype=float)
        for i in range(N):
            xi, yi = token_to_xy(tokens[i], self.K)
            for j in range(N):
                xj, yj = token_to_xy(tokens[j], self.K)
                dense[i, j] = elliptic_projection(xi + xj, yi + yj) * decay

        return SparseAttention(
            K=self.K,
            sym_mask=sym_mask,
            asym_mask=asym_mask,
            cross_mask=cross_mask,
            weights=weights,
            dense=dense,
        )


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 84)
    print("Sparse attention  ·  symmetric / asymmetric sieve separation")
    print("=" * 84)
    print(f"  α_sym   = 1/(π − e) = {ALPHA_SYM:.6f}   (linear sieve)")
    print(f"  α_asym  = 0.3628    = {ALPHA_ASYM:.6f}   (segmented sieve)")
    print(f"  K                    = {K_DEFAULT}")
    print(f"  gate density target  = 6/π² = {DENSITY:.6f}")
    print()

    # ---------- sieve comparison ----------
    print("--- sieve comparison (K = 32) ---")
    mu_lin = mobius_sieve_linear(32)
    mu_seg = mobius_sieve_segmented(32)
    print(f"  linear    μ(1..32) = "
          f"{[int(x) for x in mu_lin[1:]]}")
    print(f"  segmented μ(1..32) = "
          f"{[int(x) for x in mu_seg[1:]]}")
    print(f"  agree             = "
          f"{np.array_equal(mu_lin, mu_seg)}")
    print()

    # ---------- example queries ----------
    queries = [
        "2 + 2",
        "17 * 23 - 100",
        "how many r in strawberry",
        "what is love",
        "describe a quiet morning",
        "suggest a colour for autumn",
        "what does it feel like to be alone",
    ]

    attn = SparseSymAsymAttention(K=K_DEFAULT)

    print("--- attention sparsity per query ---")
    print(f"  {'query':38s}  {'N':>3s}  {'nz':>4s}  "
          f"{'sparse':>7s}  {'sym_nz':>6s}  {'asym_nz':>7s}  {'cross':>5s}")
    print("  " + "-" * 88)

    for q in queries:
        t0 = time.perf_counter()
        sa = attn.build(q)
        dt = (time.perf_counter() - t0) * 1e3
        sp = sa.sparsity()
        sym_nz   = int(np.count_nonzero(sa.sym_mask))
        asym_nz  = int(np.count_nonzero(sa.asym_mask))
        cross_nz = int(np.count_nonzero(sa.cross_mask))
        print(f"  {q[:38]:38s}  {len(q.split()):>3d}  "
              f"{sp['n_nonzero']:>4d}  "
              f"{sp['sparsity']:>7.3f}  {sym_nz:>6d}  "
              f"{asym_nz:>7d}  {cross_nz:>5d}")
    print()

    # ---------- detailed view of one query ----------
    print("--- detailed view: 'what is love' ---")
    q = "what is love"
    tokens, sym_f, asym_f = classify_query(q)
    print(f"  tokens    : {tokens}")
    print(f"  sym flags : {[int(b) for b in sym_f]}")
    print(f"  asym flags: {[int(b) for b in asym_f]}")
    print()

    sa = attn.build(q)
    N = len(tokens)

    print("  symmetric mask  μ(i·j mod K) ≠ 0:")
    for i in range(N):
        print(f"    {i}: " + " ".join(
            "1" if sa.sym_mask[i, j] else "." for j in range(N)))
    print()
    print("  asymmetric mask  μ((i²+j²) mod K) ≠ 0:")
    for i in range(N):
        print(f"    {i}: " + " ".join(
            "1" if sa.asym_mask[i, j] else "." for j in range(N)))
    print()
    print("  cross mask  (SAT):")
    for i in range(N):
        print(f"    {i}: " + " ".join(
            "1" if sa.cross_mask[i, j] else "." for j in range(N)))
    print()
    print("  final sparse attention weights:")
    for i in range(N):
        print(f"    {i}: " + " ".join(
            f"{sa.weights[i, j]:.3f}" if sa.weights[i, j] > 0 else "  ·  "
            for j in range(N)))
    print()
    print("  dense reference (no mask):")
    for i in range(N):
        print(f"    {i}: " + " ".join(
            f"{sa.dense[i, j]:.3f}" for j in range(N)))
    print()

    sp = sa.sparsity()
    print(f"  sparse  : {sp['n_nonzero']} / {sp['n_total']}  "
          f"(sparsity {sp['sparsity']:.3f})")
    print(f"  dense   : {sp['n_total']} / {sp['n_total']}  "
          f"(sparsity 0.000)")
    print()

    # ---------- aggregate ----------
    print("--- aggregate over the run ---")
    total_nz = 0
    total_n  = 0
    for q in queries:
        sa = attn.build(q)
        sp = sa.sparsity()
        total_nz += sp["n_nonzero"]
        total_n  = sp["n_total"]
    print(f"  total nonzero  : {total_nz}")
    print(f"  total possible : {total_n}")
    print(f"  overall sparse : {1.0 - total_nz / total_n:.4f}")
    print(f"  target 1 − 6/π²: {1.0 - DENSITY:.4f}")
    print()

    # ---------- timing ----------
    print("--- timing ---")
    N_test = 32
    t0 = time.perf_counter()
    for _ in range(100):
        _ = attn.sym_sieve.build(N_test)
    dt = (time.perf_counter() - t0) * 1e3 / 100
    print(f"  sym mask  (linear)    N={N_test}  = {dt:.3f} ms per build")

    t0 = time.perf_counter()
    for _ in range(100):
        _ = attn.asym_sieve.build(N_test)
    dt = (time.perf_counter() - t0) * 1e3 / 100
    print(f"  asym mask (segmented) N={N_test}  = {dt:.3f} ms per build")
    print()

    # ---------- complexity ----------
    print("--- complexity ---")
    print("  linear sieve            O(K log log K)    once")
    print("  segmented sieve         O(K log K)        once")
    print("  sym mask (N×N)          O(N²)             per query")
    print("  asym mask (N×N)         O(N²)")
    print("  cross mask (SAT)        O(N²)")
    print("  weight computation      O(N²)")
    print("  ──────────────────────────────────────")
    print("  per query total         O(N² + K log log K)")

    print()
    print("Done.")


if __name__ == "__main__":
    demo()