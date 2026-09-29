#!/usr/bin/env python3
"""
vocab_reader_nlp.py
===================

NLP engine where vocabulary is evaluated with the StrikeFlagReader
from quick-reader-analyzer.py.

Pipeline
--------
    query  →  tokens  →  |Ci| array  →  StrikeFlagReader.read()
           →  profile {S, H, sym_score, asym_lb, H_em, strikes, flagged}

    each vocabulary entry → same profile path

    match score  = 0.35·S-proximity
                 + 0.25·H-proximity
                 + 0.20·symmetric-similarity
                 + 0.15·asymmetric-lb proximity
                 + 0.05·emotional proximity
                 + UGC-completeness gate (kept as a hard filter)

Routing
-------
    1. if reader flags the query (strikes ≥ threshold)  →  clarify
       and return the reader's deep PDE report
    2. else scan symmetric library, take best match score
       if score ≥ τ_match                                →  symmetric
    3. else scan asymmetric library, take best match score
       if score ≥ 0.5·τ_match                            →  asymmetric
    4. else                                              →  clarify

The M-matrix coupled PDE step is preserved from the earlier engine.
"""

from __future__ import annotations

import math
import re
import hashlib
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np


# ============================================================
#  Constants
# ============================================================
PI          = math.pi
E           = math.e
ALPHA_SYM   = 1.0 / (PI - E)
ALPHA_ASYM  = 0.3628
K_HIDDEN    = 64
K_READER    = 128                     # |Ci| length for the reader
K_FILTER    = 1024 + 1
EPS_DECAY   = 0.01
ALPHA_ANCH  = 0.10
TAU_ENTROPY = 0.35
TAU_SAT     = 0.55
TAU_MATCH   = 0.55                    # reader-based match threshold


# ============================================================
#  Möbius sieve
# ============================================================
def mobius_sieve(K: int) -> np.ndarray:
    if K < 1:
        return np.zeros(K + 1, dtype=np.int8)
    mu = np.ones(K + 1, dtype=np.int8)
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
#  Merge sort / inversion count  (from quick-reader-analyzer.py)
# ============================================================
def _merge_and_count(arr, temp, left, mid, right):
    i, j, k = left, mid + 1, left
    inv = 0
    while i <= mid and j <= right:
        if arr[i] <= arr[j]:
            temp[k] = arr[i]; i += 1
        else:
            temp[k] = arr[j]
            inv += (mid - i + 1)
            j += 1
        k += 1
    while i <= mid: temp[k] = arr[i]; i += 1; k += 1
    while j <= right: temp[k] = arr[j]; j += 1; k += 1
    for i in range(left, right + 1):
        arr[i] = temp[i]
    return inv


def _merge_sort_count(arr, temp, left, right):
    inv = 0
    if left < right:
        mid = (left + right) // 2
        inv += _merge_sort_count(arr, temp, left, mid)
        inv += _merge_sort_count(arr, temp, mid + 1, right)
        inv += _merge_and_count(arr, temp, left, mid, right)
    return inv


def inversion_count(arr):
    n = len(arr)
    if n < 2:
        return 0
    temp = [0] * n
    return _merge_sort_count(list(arr), temp, 0, n - 1)


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
#  Supertrace / entropy
# ============================================================
def supertrace(c) -> float:
    S = 0.0
    for i, v in enumerate(c):
        S += v if (i % 2 == 0) else -v
    return float(S)


def entropy_from_supertrace(S: float, K: int, alpha: float = ALPHA_SYM) -> float:
    if K <= 0 or S == 0.0:
        return 0.0
    p = abs(S) / K
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -alpha * p * math.log(p)


def emotional_entropy(hormone_vector, alpha: float = ALPHA_SYM) -> float:
    v = np.asarray(hormone_vector, dtype=float)
    if v.size == 0 or v.sum() <= 0.0:
        return 0.0
    p = v / v.sum()
    H = 0.0
    for pi in p:
        if pi > 1e-12:
            H -= pi * math.log(pi)
    return alpha * H


# ============================================================
#  StrikeFlagReader  (inline from quick-reader-analyzer.py)
# ============================================================
@dataclass
class QueryProfile:
    query_id: str
    k_sum: List[float]
    symmetric_score: float = 0.5
    supertrace_S: float = 0.0
    entropy_H: float = 0.0
    emotional_entropy: float = 0.0
    asymmetric_lb: float = 0.0
    strikes: int = 0
    flagged: bool = False
    deep_result: Optional[Dict] = None
    top_emotions: List[Tuple[str, int, float]] = field(default_factory=list)


class StrikeFlagReader:
    def __init__(self,
                 strike_threshold: int = 2,
                 entropy_tolerance: float = 0.15,
                 emotional_boost: float = 1.4,
                 K_pde: int = 128):
        self.strike_threshold = strike_threshold
        self.entropy_tolerance = entropy_tolerance
        self.emotional_boost = emotional_boost
        self.K_pde = K_pde
        self.profiles: List[QueryProfile] = []
        self.flags: deque = deque()
        self.reference: Optional[np.ndarray] = None

    def set_reference(self, ref_signal):
        self.reference = np.asarray(ref_signal, dtype=float)

    def read(self, query_id, k_sum,
             emotional_vector=None, symmetric_only=True) -> QueryProfile:
        c = np.asarray(k_sum, dtype=float)
        K = c.size
        S = supertrace(c)
        H = entropy_from_supertrace(S, K)

        H_em = 0.0
        top_emotions: List[Tuple[str, int, float]] = []
        if not symmetric_only and emotional_vector is not None:
            H_em = emotional_entropy(emotional_vector) * self.emotional_boost

        if self.reference is not None and self.reference.size == K:
            sym = inverse_score(c.tolist(), self.reference.tolist())
        else:
            sym = 0.5

        half = K // 2
        if half >= 2:
            S_h = supertrace(c[:half])
            asym_lb = entropy_from_supertrace(S_h, half)
        else:
            asym_lb = 0.0

        strikes = 0
        if abs(sym - 0.5) > self.entropy_tolerance:
            strikes += 1
        if H_em > 0.0 and abs(H - H_em) > self.entropy_tolerance:
            strikes += 1
        if asym_lb > H + self.entropy_tolerance:
            strikes += 1

        prof = QueryProfile(
            query_id=query_id, k_sum=c.tolist(),
            symmetric_score=sym, supertrace_S=S,
            entropy_H=H, emotional_entropy=H_em,
            asymmetric_lb=asym_lb, strikes=strikes,
            top_emotions=top_emotions,
        )
        if strikes >= self.strike_threshold:
            prof.flagged = True
            self.flags.append(prof)
            prof.deep_result = self._pde_investigate(prof)
        self.profiles.append(prof)
        return prof

    def _pde_investigate(self, prof: QueryProfile) -> Dict:
        K = self.K_pde
        c = np.asarray(prof.k_sum, dtype=float)
        n = c.size
        seed = int(hashlib.md5(prof.query_id.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        W_in = rng.standard_normal((K, max(1, min(n, K)))) * 0.01
        anchors = np.arange(0, K, max(1, K // 20))
        DT, ALPHA_PDE, EPS = 0.1, 0.1, 0.01
        h = np.zeros(K)
        for j in range(min(n, K)):
            h += W_in[:, j] * c[j]
            diff = np.zeros_like(h)
            diff[1:-1] = h[:-2] + h[2:] - 2.0 * h[1:-1]
            h += DT * (diff + ALPHA_PDE * np.sum(h[anchors]) - EPS * h)
        final_S = supertrace(h)
        final_H = entropy_from_supertrace(final_S, K)
        return dict(query_id=prof.query_id,
                    h_norm=float(np.linalg.norm(h)),
                    h_mean=float(np.mean(h)),
                    h_std=float(np.std(h)),
                    pde_S=final_S, pde_H=final_H,
                    strikes=prof.strikes, flagged=True)


# ============================================================
#  Tokens → |Ci| array for the reader
# ============================================================
def tokens_to_k_sum(text: str, K: int) -> np.ndarray:
    """
    Deterministic |Ci| array of length K from a text string:
        position i in the token stream
        h = md5(token) mod 10_000
        c[i] = (h / 10_000) + 0.1·sin(i · 0.7)
    """
    tokens = re.findall(r"\w+|[.,!?;:'\"]", text.lower().strip())
    c = np.zeros(K, dtype=float)
    for i, w in enumerate(tokens[:K]):
        h = int(hashlib.md5(w.encode()).hexdigest()[:8], 16)
        c[i] = ((h % 10_000) / 10_000.0) + 0.1 * math.sin(i * 0.7)
    return np.abs(c)


# ============================================================
#  M-matrix  ·  the PDE step for the semantic hidden state
# ============================================================
def m_diag(x: complex) -> complex:
    return (complex(x) + 1e-9) ** (1j - 1)


def m_laplacian(h: np.ndarray) -> np.ndarray:
    K = len(h)
    diff = np.zeros(K, dtype=complex)
    m_vals = np.array([m_diag(v) for v in h])
    for n in range(1, K - 1):
        diff[n] = m_vals[n - 1] + m_vals[n + 1] - 2.0 * m_vals[n]
    return diff


def word_to_ci(word: str, K: int, mu: np.ndarray) -> np.ndarray:
    c = np.zeros(2 * K + 1, dtype=complex)
    idx = (int(hashlib.md5(word.encode()).hexdigest()[:8], 16) % K) + 1
    if 0 < idx <= K and mu[idx] != 0:
        c[K + idx] += 1.0
        c[K - idx] += 1.0
    hh = int(hashlib.md5(word.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(hh)
    c.imag += rng.standard_normal(2 * K + 1) * 0.05
    return c


def ci_to_h(c: np.ndarray, K: int) -> np.ndarray:
    v = np.zeros(K, dtype=complex)
    for n in range(1, K + 1):
        v[n - 1] = 0.5 * (c[K + n] + c[K - n])
    return v


# ============================================================
#  Reader-based match score
# ============================================================
def reader_match_score(qp: QueryProfile, cp: QueryProfile,
                       sym_weight: float = 0.20) -> Dict[str, float]:
    """
    Five-signal similarity between two reader profiles:

        s_score   : 1 / (1 + |S_q − S_c|)
        h_score   : 1 / (1 + |H_q − H_c|)
        sym_score : 1 − inverse_score(q.k_sum, c.k_sum)
        lb_score  : 1 / (1 + |lb_q − lb_c|)
        emo_score : 1 / (1 + |H_em_q − H_em_c|)  (only when both > 0)

    Combined with weights
        0.35 · s_score + 0.25 · h_score + sym_weight · sym_score
        + 0.15 · lb_score + 0.05 · emo_score
    """
    s_score   = 1.0 / (1.0 + abs(qp.supertrace_S - cp.supertrace_S))
    h_score   = 1.0 / (1.0 + abs(qp.entropy_H - cp.entropy_H))
    sym_score = 1.0 - inverse_score(qp.k_sum, cp.k_sum)
    lb_score  = 1.0 / (1.0 + abs(qp.asymmetric_lb - cp.asymmetric_lb))
    emo_score = 0.0
    if qp.emotional_entropy > 0.0 and cp.emotional_entropy > 0.0:
        emo_score = 1.0 / (1.0 + abs(qp.emotional_entropy
                                    - cp.emotional_entropy))

    total = (0.35 * s_score
             + 0.25 * h_score
             + sym_weight * sym_score
             + 0.15 * lb_score
             + 0.05 * emo_score)
    return dict(total=total, s=s_score, h=h_score,
                sym=sym_score, lb=lb_score, emo=emo_score)


# ============================================================
#  The engine
# ============================================================
class ReaderVocabNLP:
    """
    NLP engine that evaluates vocabulary via the StrikeFlagReader
    and routes queries with a combined reader + M-matrix signal.
    """

    def __init__(self, K: int = K_HIDDEN,
                 K_reader: int = K_READER):
        self.K = K
        self.K_reader = K_reader
        self.mu = mobius_sieve(max(K, K_FILTER))
        self.anchors = list(range(0, K, max(1, K // 20)))

        # two readers:
        #   - one for the query itself (symmetric mode)
        #   - one with emotional boost for prompted queries
        self.reader = StrikeFlagReader(
            strike_threshold=2,
            entropy_tolerance=0.12,
            emotional_boost=1.4,
            K_pde=K_reader)
        # deterministic reference: constant |Ci|
        x = np.arange(K_reader)
        ref = np.abs(np.sin(x * 0.05) + 0.3 * np.cos(x * 0.13))
        self.reader.set_reference(ref)

        # hidden state for the M-coupled PDE
        self.h = np.zeros(K, dtype=complex)
        self.history: List[str] = []
        self.last_step_diagnostics: List[Dict] = []

        # libraries
        self.sym_library  = self._build_sym_library()
        self.asym_library = self._build_asym_library()

        # pre-computed candidate profiles (read once, reused)
        self._sym_profiles: Optional[List[Tuple[str, str, QueryProfile]]] = None
        self._asym_profiles: Optional[List[Tuple[str, QueryProfile]]] = None

    # ---------- libraries ----------
    def _build_sym_library(self) -> Dict[str, str]:
        return {
            "2 + 2":                              "4",
            "17 * 23 - 100":                      "291",
            "2 ** 10":                            "1024",
            "what is the capital of france":      "Paris",
            "how many r in strawberry":           "3",
            "how many a in banana":               "3",
            "how many cats in the cat catalogue": "2",
            "length of hello":                    "5",
        }

    def _build_asym_library(self) -> List[str]:
        return [
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

    # ---------- candidate profiling ----------
    def _profile_candidates(self):
        """Read every vocabulary entry once, cache the profiles."""
        if self._sym_profiles is None:
            self._sym_profiles = []
            for key, ans in self.sym_library.items():
                c = tokens_to_k_sum(key, self.K_reader)
                p = self.reader.read(f"sym:{key}", c,
                                     symmetric_only=True)
                self._sym_profiles.append((key, ans, p))
        if self._asym_profiles is None:
            self._asym_profiles = []
            for frag in self.asym_library:
                c = tokens_to_k_sum(frag, self.K_reader)
                p = self.reader.read(f"asym:{frag[:24]}", c,
                                     symmetric_only=True)
                self._asym_profiles.append((frag, p))

    # ---------- tokenisation ----------
    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r"\w+|[.,!?;:'\"]", text.lower().strip())

    # ---------- M-coupled PDE step ----------
    def step(self, word: str) -> Dict:
        c = word_to_ci(word, self.K, self.mu)
        v = ci_to_h(c, self.K)
        self.h += v
        diff = m_laplacian(self.h)
        anchor_sum = np.sum(self.h[self.anchors])
        nonlocal_term = ALPHA_ANCH * anchor_sum
        self.h += EPS_DECAY * (diff + nonlocal_term - EPS_DECAY * self.h)

        self.history.append(word)

        alg   = np.real(1.0 / (self.h + 1e-9))
        trans = np.real((self.h + 1e-9) ** 1j)
        S_alg   = self._supertrace(alg)
        S_trans = self._supertrace(trans)
        return dict(word=word, S_alg=S_alg, S_trans=S_trans,
                    disorder=abs(S_trans) / self.K)

    @staticmethod
    def _supertrace(v: np.ndarray) -> float:
        S = 0.0
        for n, x in enumerate(v):
            S += x if (n % 2 == 0) else -x
        return float(S)

    def process(self, text: str) -> None:
        self.h = np.zeros(self.K, dtype=complex)
        self.history = []
        self.last_step_diagnostics = []
        for w in self.tokenize(text):
            self.last_step_diagnostics.append(self.step(w))

    # ---------- router ----------
    def ask(self, query: str,
            emotional_vector: Optional[np.ndarray] = None) -> Dict:
        t0 = time.perf_counter()
        self._profile_candidates()

        # ---- 1. run M-coupled PDE on the query ----
        self.process(query)
        S_alg   = self._supertrace(np.real(1.0 / (self.h + 1e-9)))
        S_trans = self._supertrace(np.real((self.h + 1e-9) ** 1j))
        D       = abs(S_trans) / self.K

        # ---- 2. reader profile of the query ----
        q_c = tokens_to_k_sum(query, self.K_reader)
        qp  = self.reader.read(f"query:{query[:32]}", q_c,
                               emotional_vector=emotional_vector,
                               symmetric_only=(emotional_vector is None))

        # ---- 3. reader-flag the query ----
        if qp.flagged:
            deep = qp.deep_result or {}
            return self._reply(
                query, "clarify-by-reader",
                "Your query is too entropic — could you rephrase or "
                "add context?",
                D, S_alg, S_trans,
                dict(strikes=qp.strikes,
                     S_reader=qp.supertrace_S,
                     H_reader=qp.entropy_H,
                     sym=qp.symmetric_score,
                     asym_lb=qp.asymmetric_lb,
                     pde_S=deep.get("pde_S", 0.0),
                     pde_H=deep.get("pde_H", 0.0)),
                t0)

        # ---- 4. symmetric library via reader match ----
        best_sym = (0.0, None, None, None)
        for key, ans, cp in self._sym_profiles:
            m = reader_match_score(qp, cp)
            if m["total"] > best_sym[0]:
                best_sym = (m["total"], key, ans, m)
        if best_sym[0] >= TAU_MATCH:
            return self._reply(
                query, "symmetric-reader",
                best_sym[2], D, S_alg, S_trans,
                dict(match=best_sym[1],
                     reader_score=round(best_sym[0], 4),
                     signals={k: round(v, 4)
                              for k, v in best_sym[3].items()}),
                t0)

        # ---- 5. asymmetric library via reader match ----
        scored = []
        for frag, cp in self._asym_profiles:
            m = reader_match_score(qp, cp)
            scored.append((m["total"], frag, m))
        scored.sort(reverse=True, key=lambda p: p[0])

        if scored and scored[0][0] >= 0.5 * TAU_MATCH:
            top = scored[:3]
            answer = " / ".join(f for _, f, _ in top)
            return self._reply(
                query, "asymmetric-reader",
                answer, D, S_alg, S_trans,
                dict(top_scores=[round(s, 4) for s, _, _ in top],
                     reader_score=round(scored[0][0], 4),
                     signals={k: round(v, 4)
                              for k, v in scored[0][2].items()}),
                t0)

        # ---- 6. nothing matched ----
        return self._reply(
            query, "clarify-low-match",
            "I couldn't match your query to a known vocabulary entry. "
            "Could you rephrase?",
            D, S_alg, S_trans,
            dict(best_sym=round(best_sym[0], 4),
                 best_asym=round(scored[0][0], 4) if scored else 0.0,
                 reader_S=qp.supertrace_S,
                 reader_H=qp.entropy_H),
            t0)

    # ---------- reply ----------
    @staticmethod
    def _reply(query, route, answer, D, S_alg, S_trans,
               extra, t0) -> Dict:
        r = dict(
            query=query,
            route=route,
            answer=answer,
            disorder=round(D, 4),
            S_alg=round(S_alg, 4),
            S_trans=round(S_trans, 4),
            elapsed_ms=round((time.perf_counter() - t0) * 1e3, 3),
        )
        if extra:
            r.update(extra)
        return r


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 88)
    print("NLP with quick-reader vocabulary evaluation")
    print("=" * 88)
    print(f"  α_sym   = 1/(π − e) = {ALPHA_SYM:.6f}")
    print(f"  K_hidden            = {K_HIDDEN}")
    print(f"  K_reader            = {K_READER}")
    print(f"  τ_match             = {TAU_MATCH}")
    print()

    engine = ReaderVocabNLP()

    # ---------- vocabulary profiles ----------
    print("--- vocabulary reader profiles ---")
    print(f"  {'entry':40s}  {'S':>9s}  {'H':>8s}  {'sym':>7s}  "
          f"{'asym_lb':>8s}  {'strikes':>7s}  {'flag':>5s}")
    print("  " + "-" * 92)
    engine._profile_candidates()
    for key, ans, cp in engine._sym_profiles:
        print(f"  {key[:40]:40s}  {cp.supertrace_S:>+9.4f}  "
              f"{cp.entropy_H:>8.4f}  {cp.symmetric_score:>7.4f}  "
              f"{cp.asymmetric_lb:>8.4f}  {cp.strikes:>7d}  "
              f"{str(cp.flagged):>5s}  → {ans}")
    print()
    for frag, cp in engine._asym_profiles[:4]:
        print(f"  {frag[:40]:40s}  {cp.supertrace_S:>+9.4f}  "
              f"{cp.entropy_H:>8.4f}  {cp.symmetric_score:>7.4f}  "
              f"{cp.asymmetric_lb:>8.4f}  {cp.strikes:>7d}  "
              f"{str(cp.flagged):>5s}")
    print("  ...")
    print()

    # ---------- routing ----------
    queries = [
        "2 + 2",
        "17 * 23 - 100",
        "what is the capital of france",
        "how many r in strawberry",
        "length of hello",
        "what is love",
        "suggest a colour for autumn",
        "describe a quiet morning",
        "the",
        "xqzywibble",
    ]

    print("--- routing ---")
    print(f"{'query':38s}  {'route':>22s}  {'D':>7s}  "
          f"{'reader_S':>9s}  {'reader_H':>9s}  answer")
    print("-" * 124)

    for q in queries:
        r = engine.ask(q)
        ans = r["answer"] if len(r["answer"]) <= 26 \
              else r["answer"][:23] + "…"
        rs = r.get("S_reader", r.get("reader_S", 0.0))
        rh = r.get("H_reader", r.get("reader_H", 0.0))
        print(f"{q[:38]:38s}  {r['route']:>22s}  "
              f"{r['disorder']:>7.4f}  {rs:>+9.4f}  {rh:>9.4f}  {ans}")

    print()

    # ---------- full responses ----------
    print("--- full responses ---")
    for q in ("2 + 2", "what is love", "the"):
        r = engine.ask(q)
        print()
        print(f"  query  : {q!r}")
        print(f"  route  : {r['route']}")
        print(f"  answer : {r['answer']}")
        print(f"  D      : {r['disorder']}")
        print(f"  S_alg  : {r['S_alg']}")
        print(f"  S_trans: {r['S_trans']}")
        for key in ("S_reader", "H_reader", "sym", "asym_lb",
                    "strikes", "reader_score"):
            if key in r:
                print(f"  {key:7s}: {r[key]}")
        if "signals" in r:
            print(f"  signals: {r['signals']}")
        print(f"  time   : {r['elapsed_ms']} ms")

    print()

    # ---------- aggregate ----------
    print("--- aggregate over the run ---")
    routes: Dict[str, int] = {}
    for q in queries:
        r = engine.ask(q)
        routes[r["route"]] = routes.get(r["route"], 0) + 1
    for k, v in sorted(routes.items()):
        print(f"  {k:>22s} : {v}")

    print()
    print("--- Complexity ---")
    print("  Möbius sieve                   O(K log log K)   once")
    print("  reader.read                    O(K_reader log K_reader)")
    print("  vocabulary profiling (one-off) O(C · K_reader log K_reader)")
    print("  reader match score             O(K_reader log K_reader) "
          "  per candidate")
    print("  M-coupled PDE step             O(K)")
    print("  ─────────────────────────────────────────────")
    print("  per-query total                O(N·K + C·K_reader log K_reader)")

    print("\nDone.")


if __name__ == "__main__":
    demo()