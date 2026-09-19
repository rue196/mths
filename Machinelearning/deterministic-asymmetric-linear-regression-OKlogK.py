#!/usr/bin/env python3
"""
mobius_ml_router.py

A Möbius ML algorithm that splits queries into two deterministic /
non‑deterministic regimes and routes each one to the corresponding
O(K) pipeline.

    symmetric  (deterministic)  →  α_sym = 1/(π − e) ≈ 2.362
        arithmetic, counting, closed‑form lookups — "2 + 2", "how
        many r's in strawberry", "len of 'hello'".  These have an
        exact answer independent of context.

    asymmetric (non‑deterministic)  →  α_asym = 0.3628
        opinion, feeling, creativity, ambiguous queries — "what is
        love", "suggest a colour".  There is no unique answer; the
        response is a chirped spectral blend of stored fragments.

Complexity
----------
    Möbius sieve            O(K log log K)        once
    ──────────────────────────────────────────────────────────────
    symmetric  per query    O(K)                  parse + solve
    asymmetric per query    O(K log K)            embed + inverse‑
                                                  score merge sort
    memory per query        O(K)                  |Ci| arrays

The classifier uses a structural check first (does the query parse
as a deterministic expression?) and falls back to the elliptic
projection of the query hash:

    Π_sym  from  u = 1/f, v = 1/g        (algebraic plane)
    Π_asym from  p = f^i, q = g^i        (imaginary plane)
    |Π_sym − Π_asym| < τ    →  symmetric
"""

from __future__ import annotations

import math
import re
import time
import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


# ============================================================
#  Constants — dual pair
# ============================================================
PI         = math.pi
E          = math.e
ALPHA_SYM  = 1.0 / (PI - E)          # ≈ 2.362
ALPHA_ASYM = 0.3628                  # asymmetric scale
DENSITY    = 6.0 / (PI * PI)         # ≈ 0.6079271018
K_DEFAULT  = 64
SYM_THRESHOLD = 0.35


# ============================================================
#  1. Möbius sieve (Eratosthenes variant)  O(K log log K)
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
#  2. Merge sort inverse score  O(K log K)
# ============================================================
def _merge_and_count(arr, tmp, l, m, r):
    i, j, k, inv = l, m + 1, l, 0
    while i <= m and j <= r:
        if arr[i] <= arr[j]:
            tmp[k] = arr[i]; i += 1
        else:
            tmp[k] = arr[j]; inv += m - i + 1; j += 1
        k += 1
    while i <= m: tmp[k] = arr[i]; i += 1; k += 1
    while j <= r: tmp[k] = arr[j]; j += 1; k += 1
    for i in range(l, r + 1):
        arr[i] = tmp[i]
    return inv


def _merge_sort_count(arr, tmp, l, r):
    inv = 0
    if l < r:
        m = (l + r) // 2
        inv += _merge_sort_count(arr, tmp, l, m)
        inv += _merge_sort_count(arr, tmp, m + 1, r)
        inv += _merge_and_count(arr, tmp, l, m, r)
    return inv


def inversion_count(arr) -> int:
    n = len(arr)
    if n < 2:
        return 0
    return _merge_sort_count(list(arr), [0] * n, 0, n - 1)


def inverse_score(a, b) -> float:
    if len(a) != len(b) or len(a) < 2:
        return 0.5
    pairs = sorted(zip(a, b), key=lambda p: p[0])
    b_sorted = [p[1] for p in pairs]
    inv = inversion_count(b_sorted)
    K = len(a)
    mx = K * (K - 1) // 2
    return inv / mx if mx > 0 else 0.0


# ============================================================
#  3. |Ci| embedding  O(K)
# ============================================================
def token_index(token: str, K: int) -> int:
    h = int(hashlib.md5(token.encode()).hexdigest()[:8], 16)
    return (h % K) + 1


def embed_tokens(tokens: List[str], K: int,
                 mu: Optional[List[int]] = None) -> np.ndarray:
    """Return length-(2K+1) |Ci| array."""
    c = np.zeros(2 * K + 1, dtype=np.float64)
    for t in tokens:
        i = token_index(t, K)
        if mu is not None and mu[i] == 0:
            continue
        c[K + i] += 1.0
        c[K - i] += 1.0
    return c


def embed_text(text: str, K: int, mu: Optional[List[int]] = None) -> np.ndarray:
    return embed_tokens(text.lower().split(), K, mu)


# ============================================================
#  4. Supertrace / entropy  (shared scalars)
# ============================================================
def supertrace(x: np.ndarray) -> float:
    S = 0.0
    for i, v in enumerate(x):
        S += v if (i % 2 == 0) else -v
    return float(S)


def entropy(S: float, N: int, alpha: float) -> float:
    if N <= 0 or S == 0.0:
        return 0.0
    p = abs(S) / N
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -alpha * p * math.log(p)


# ============================================================
#  5. Determinism classifier
# ============================================================
_ARITH_RE = re.compile(r"^[\s0-9\.\+\-\*/\(\)%eE]+$")
_COUNT_LETTER_RE = re.compile(
    r"""how\s+many\s+['"]?(?P<ch>\w)['"]?\s+
        (?:in|are\s+in|does)\s+['"]?(?P<word>\w+)['"]?""",
    re.VERBOSE | re.IGNORECASE,
)
_COUNT_WORD_RE = re.compile(
    r"""how\s+many\s+['"]?(?P<w>\w+)['"]?\s+
        (?:in|are\s+in|does)\s+['"]?(?P<text>.+?)['"]?$""",
    re.VERBOSE | re.IGNORECASE,
)
_LEN_RE = re.compile(r"(?:length|len)\s+of\s+['\"]?(?P<w>\w+)['\"]?",
                     re.IGNORECASE)


@dataclass
class Classification:
    deterministic: bool
    kind: str                 # 'arith' | 'count_letter' | 'count_word' | 'len' | 'fallback'
    projection_gap: float     # |Π_sym − Π_asym|
    matched: Optional[re.Match] = None


def elliptic_projection_sym(f: float, g: float) -> float:
    """
    Algebraic plane projection:
        u = 1/f, v = 1/g
        Π_sym = ½ (1 + cos(2π u/π) · cos(2π v/e))
    """
    f = max(abs(f), 1e-12)
    g = max(abs(g), 1e-12)
    u = (1.0 / f) % 1.0
    v = (1.0 / g) % 1.0
    return 0.5 * (1.0 + math.cos(2 * math.pi * u) * math.cos(2 * math.pi * v))


def elliptic_projection_asym(f: float, g: float) -> float:
    """
    Imaginary plane projection:
        p = f^i, q = g^i
        Π_asym = ½ (1 + cos(2π |Re p|) · cos(2π |Re q|))
    """
    f = max(abs(f), 1e-12)
    g = max(abs(g), 1e-12)
    p_re = math.cos(math.log(f))
    q_re = math.cos(math.log(g))
    return 0.5 * (1.0 + math.cos(2 * math.pi * p_re) *
                        math.cos(2 * math.pi * q_re))


def query_scalars(text: str) -> Tuple[float, float]:
    """Two scalars in [0.5, 5.0] derived from the query hash."""
    h = int(hashlib.md5(text.encode()).hexdigest()[:16], 16)
    f = 0.5 + ((h & 0xFFFF) / 0xFFFF) * 4.5
    g = 0.5 + (((h >> 16) & 0xFFFF) / 0xFFFF) * 4.5
    return f, g


def classify(query: str) -> Classification:
    """
    Structural first, projection fallback.

    Order:
      1. arithmetic  →  deterministic
      2. "how many X in Y"
      3. "length of X"
      4. projection gap < τ  →  deterministic (fallback)
                              else  asymmetric
    """
    q = query.strip()

    # -- 1. arithmetic --
    if _ARITH_RE.match(q) and any(op in q for op in "+-*/%"):
        return Classification(True, 'arith', 0.0)

    # -- 2. letter count --
    m = _COUNT_LETTER_RE.search(q)
    if m:
        return Classification(True, 'count_letter', 0.0, m)

    # -- 3. word count --
    m = _COUNT_WORD_RE.search(q)
    if m:
        return Classification(True, 'count_word', 0.0, m)

    # -- 4. length --
    m = _LEN_RE.search(q)
    if m:
        return Classification(True, 'len', 0.0, m)

    # -- 5. projection fallback --
    f, g = query_scalars(q)
    p_sym  = elliptic_projection_sym(f, g)
    p_asym = elliptic_projection_asym(f, g)
    gap = abs(p_sym - p_asym)
    det = gap < SYM_THRESHOLD
    return Classification(det, 'fallback', gap)


# ============================================================
#  6. Symmetric solver  ·  O(K)
# ============================================================
class SymmetricSolver:
    """
    Handles deterministic queries in O(K) time.
    Returns (answer_text, detail_dict).
    """

    def solve(self, query: str, cls: Classification) -> Tuple[str, Dict]:
        q = query.strip()
        if cls.kind == 'arith':
            try:
                val = eval(q, {'__builtins__': {}}, {})
                if isinstance(val, float) and val.is_integer():
                    val = int(val)
                return str(val), {'method': 'eval'}
            except Exception as exc:
                return f"arithmetic error: {exc}", {'method': 'eval', 'error': str(exc)}

        if cls.kind == 'count_letter':
            ch   = cls.matched.group('ch').lower()
            word = cls.matched.group('word').lower()
            return str(word.count(ch)), {
                'method': 'count_letter', 'char': ch, 'word': word}

        if cls.kind == 'count_word':
            w    = cls.matched.group('w').lower()
            text = cls.matched.group('text').lower()
            count = len(re.findall(r"\b" + re.escape(w) + r"\b", text))
            return str(count), {
                'method': 'count_word', 'word': w, 'text': text}

        if cls.kind == 'len':
            w = cls.matched.group('w')
            return str(len(w)), {'method': 'len', 'word': w}

        # fallback: no exact answer — return a 0‑entropy stub
        return "0", {'method': 'fallback_no_rule'}


# ============================================================
#  7. Asymmetric generator  ·  O(K log K)
# ============================================================
@dataclass
class AsymFragment:
    text: str
    embedding: np.ndarray


class AsymmetricGenerator:
    """
    Non‑deterministic responses via inverse‑score retrieval and
    a chirp‑weighted blend.  All steps in O(K log K) per query.
    """

    SEEDS = [
        "love is the quiet space where two silences agree",
        "a colour that feels like rain on a warm afternoon",
        "the answer drifts between memory and wanting",
        "like a wave that forgets the shore it came from",
        "the feeling of a door left open for no one",
        "a shape that only exists while you look at it",
        "somewhere between the echo and the song",
        "the hour when lamps begin to think for us",
        "a scent the room remembers but you cannot name",
        "the pause before a word you almost said",
    ]

    def __init__(self, K: int, mu: List[int],
                 chirp_a: float = 1e-3,
                 chirp_b: float = 1e-1):
        self.K = K
        self.mu = mu
        self.chirp_a = chirp_a
        self.chirp_b = chirp_b
        self.fragments: List[AsymFragment] = []
        for seed in self.SEEDS:
            emb = embed_text(seed, K, mu)
            self.fragments.append(AsymFragment(seed, emb))

    # ---------- chirped weight ----------
    def _chirp_weight(self, k: int) -> float:
        theta = self.chirp_a * k * k + self.chirp_b * k
        return math.cos(theta) * ALPHA_ASYM

    # ---------- response ----------
    def respond(self, query: str, top_k: int = 3) -> Tuple[str, Dict]:
        t0 = time.perf_counter()
        q_emb = embed_text(query, self.K, self.mu)

        # -- inverse score against each stored fragment  O(K log K) --
        scored: List[Tuple[float, AsymFragment]] = []
        for frag in self.fragments:
            s = inverse_score(q_emb.tolist(), frag.embedding.tolist())
            scored.append((s, frag))
        scored.sort(key=lambda p: p[0])

        # -- chirp‑weighted pick among the top‑k --
        top = scored[:top_k]
        weights = []
        for k_idx, (_s, _frag) in enumerate(top, start=1):
            weights.append(abs(self._chirp_weight(k_idx)))
        wsum = sum(weights) + 1e-12
        weights = [w / wsum for w in weights]

        # -- blend the texts (weighted concatenation) --
        picked = [f.text for w, (_s, f) in zip(weights, top) if w > 0.15]
        if not picked:
            picked = [top[0][1].text]

        answer = " / ".join(picked)
        detail = {
            'method': 'asym_blend',
            'top_scores': [round(s, 4) for s, _ in top],
            'weights': [round(w, 4) for w in weights],
            'supertrace_q': supertrace(q_emb),
            'supertrace_score': entropy(supertrace(q_emb), 2 * self.K + 1, ALPHA_ASYM),
            'elapsed_ms': (time.perf_counter() - t0) * 1e3,
        }
        return answer, detail


# ============================================================
#  8. The router
# ============================================================
@dataclass
class Answer:
    query: str
    deterministic: bool
    kind: str
    answer: str
    detail: Dict
    alpha_used: float
    elapsed_ms: float


class MobiusMLRouter:
    """
    Unified model:
        classify  →  symmetric solver  or  asymmetric generator
    All per‑query work is O(K) or O(K log K).
    """

    def __init__(self, K: int = K_DEFAULT):
        t0 = time.perf_counter()
        self.K = K
        self.mu = mobius_sieve(K)                      # O(K log log K)
        self.sieve_ms = (time.perf_counter() - t0) * 1e3
        self.solver = SymmetricSolver()
        self.generator = AsymmetricGenerator(K, self.mu)

    def ask(self, query: str) -> Answer:
        t0 = time.perf_counter()
        cls = classify(query)

        if cls.deterministic:
            text, detail = self.solver.solve(query, cls)
            alpha = ALPHA_SYM
        else:
            text, detail = self.generator.respond(query)
            alpha = ALPHA_ASYM

        return Answer(
            query=query,
            deterministic=cls.deterministic,
            kind=cls.kind,
            answer=text,
            detail=detail,
            alpha_used=alpha,
            elapsed_ms=(time.perf_counter() - t0) * 1e3,
        )


# ============================================================
#  9. Demo
# ============================================================
def demo():
    print("=" * 78)
    print("Möbius ML router  ·  symmetric (deterministic) vs asymmetric")
    print("=" * 78)
    print(f"α_sym  = 1/(π − e) ≈ {ALPHA_SYM:.6f}  (deterministic path)")
    print(f"α_asym = 0.3628               (non‑deterministic path)")

    router = MobiusMLRouter(K=64)
    print(f"Möbius sieve built once in {router.sieve_ms:.2f} ms  "
          f"(O(K log log K), K = {router.K})\n")

    queries = [
        # --- deterministic (symmetric) ---
        ("2 + 2",                                "arith"),
        ("17 * 23 - 100",                        "arith"),
        ("2 ** 10",                              "arith"),
        ("how many r in strawberry",             "count_letter"),
        ("how many a in banana",                 "count_letter"),
        ("how many cats in the cat catalogue",   "count_word"),
        ("length of hello",                      "len"),
        # --- non‑deterministic (asymmetric) ---
        ("what is love",                         "asym"),
        ("suggest a colour for autumn",          "asym"),
        ("describe a quiet morning",             "asym"),
        ("say something about rain",             "asym"),
        ("what does it feel like to be alone",   "asym"),
        # --- ambiguous fallback ---
        ("xqzywibble",                           "fallback"),
        ("the",                                  "fallback"),
    ]

    print(f"{'query':38s}  {'route':>6s}  {'kind':>13s}  "
          f"{'α':>7s}  {'time':>7s}  answer")
    print("-" * 100)

    sym_count = asym_count = 0
    for q, tag in queries:
        a = router.ask(q)
        route = "SYM" if a.deterministic else "ASYM"
        if a.deterministic:
            sym_count += 1
        else:
            asym_count += 1
        ans = a.answer if len(a.answer) <= 42 else a.answer[:39] + "…"
        print(f"{q[:38]:38s}  {route:>6s}  {a.kind:>13s}  "
              f"{a.alpha_used:7.4f}  {a.elapsed_ms:6.2f}ms  {ans}")

    print("-" * 100)
    print(f"\n  symmetric  queries : {sym_count}")
    print(f"  asymmetric queries : {asym_count}")

    # ---------- detail on a few ----------
    print("\n--- Detailed traces ---")
    for q in ("2 + 2",
              "how many r in strawberry",
              "what is love"):
        a = router.ask(q)
        print(f"\n  query : {q!r}")
        print(f"  route : {'symmetric' if a.deterministic else 'asymmetric'}"
              f"   (kind = {a.kind})")
        print(f"  α     : {a.alpha_used:.6f}")
        print(f"  answer: {a.answer}")
        for k, v in a.detail.items():
            print(f"    {k:18s} = {v}")

    # ---------- complexity ----------
    print("\n--- Complexity ---")
    print("  Möbius sieve                O(K log log K)   once")
    print("  classify                    O(1)             per query")
    print("  symmetric solve             O(K)             per query")
    print("  asymmetric embed + sort     O(K log K)       per query")
    print("  memory                      O(K)             per query")
    print("\n  The K‑dominant step is the Möbius sieve, shared across all")
    print("  queries.  Once it is built, each deterministic query is O(K)")
    print("  and each non‑deterministic query is O(K log K).")

    # ---------- throughput test ----------
    print("\n--- Throughput (1000 queries) ---")
    qs = [q for q, _ in queries] * 72
    t0 = time.perf_counter()
    for q in qs:
        router.ask(q)
    dt = (time.perf_counter() - t0) * 1e3
    print(f"  {len(qs)} queries in {dt:.1f} ms  "
          f"({dt/len(qs):.3f} ms/query)")

    print("\nDone.")


if __name__ == "__main__":
    demo()