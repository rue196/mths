#!/usr/bin/env python3
"""
learning_search_model.py

A model that learns from a query by searching and training in
O(K log log K), searching tokens in O(K log K) through the
semiotic‑differential PDE from # app.py, analyzing tokens like
quick-reader-analyzer.py, using quick-training.py for signed‑log
preprocessing, and ssd-ram-ui.py for SSD‑backed multi‑layer memory.

Pipeline (per query)
--------------------
    query
      → tokens                                 O(K)
      → |Ci| array (quick‑training)            O(K)
      → signed‑log transform                   O(K)
      → bucket sort                            O(K)
      → Möbius sieve (shared, cached)          O(K log log K)   ← once
      → PDE document vector (# app.py)         O(K)
      → inverse‑score search over memory       O(K log K)
      → if/else decision on the scalar
          • forward  (score < 0.4)  → reinforce
          • neutral  (0.4 ≤ score ≤ 0.6) → store
          • inverse  (score > 0.6)  → decay
      → write the kept coefficients to SSD     O(K)
      → derive imag layer in RAM (LRU cache)   O(K) on miss

The model keeps a symbolic "learnt scalar" per query, updated with
the classical reinforcement rule:
    w ← w · (1 + lr · (reward − baseline))
and returns the top‑ranked tokens to the user together with the
most important (large‑|Ci|) coefficients from the memory layer.
"""

from __future__ import annotations

import math
import os
import time
import hashlib
import struct
import numpy as np
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Iterator


# ============================================================
#  Constants
# ============================================================
PI          = math.pi
E           = math.e
ALPHA       = 1.0 / (PI - E)             # ≈ 2.362
ALPHA_USER  = 0.3628
A_STEP      = ALPHA / ALPHA_USER         # ≈ 6.511
DENSITY     = 6.0 / (PI * PI)            # ≈ 0.6079
NORM        = 1.0 - math.exp(-ALPHA * (PI + E))
LOG_MIN     = -30.0
LOG_MAX     = 30.0
N_BUCKETS   = 512
K_PDE       = 128


# ============================================================
#  1. Möbius sieve (Eratosthenes)  O(K log log K)
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
#  2. Token → |Ci|  (quick‑training.py)
# ============================================================
def build_k_sum(tokens, K: int, mu: Optional[List[int]] = None) -> np.ndarray:
    c = np.zeros(2 * K + 1, dtype=np.float64)
    for t in tokens:
        h = int(hashlib.md5(str(t).encode()).hexdigest()[:8], 16)
        i = (h % K) + 1
        if mu is not None and mu[i] == 0:
            continue
        c[K + i] += 1.0
        c[K - i] += 1.0
    return c


# ============================================================
#  3. Signed‑log transform  (quick‑training.py)
# ============================================================
@dataclass
class TransformResult:
    transformed: np.ndarray
    indices: np.ndarray
    symmetric_indices: List[int] = field(default_factory=list)
    asymmetric_indices: List[int] = field(default_factory=list)
    n_symmetric: int = 0
    n_asymmetric: int = 0


def classify_and_transform(c: np.ndarray, K: int,
                           tol: float = 1e-6) -> TransformResult:
    n = 2 * K + 1
    out = np.zeros(n, dtype=np.float64)
    sym_idx: List[int] = []
    asym_idx: List[int] = []

    for i in range(1, K + 1):
        pos, neg = K + i, K - i
        cp, cn = abs(c[pos]), abs(c[neg])
        both = (cp > 0) and (cn > 0)
        symmetric = both and (abs(cp - cn) <= tol * (cp + cn + 1e-12))
        if symmetric:
            v = -math.log(cp + 1e-12)
            v = max(LOG_MIN, min(LOG_MAX, v))
            out[pos] = v
            out[neg] = v
            sym_idx += [i, -i]
        else:
            if cp > 0:
                out[pos] = max(LOG_MIN, min(LOG_MAX, math.log(cp + 1e-12)))
            if cn > 0:
                out[neg] = max(LOG_MIN, min(LOG_MAX, math.log(cn + 1e-12)))
            asym_idx.append(i)
            if cn > 0 and not both:
                asym_idx.append(-i)

    if abs(c[K]) > 0:
        out[K] = max(LOG_MIN, min(LOG_MAX, math.log(abs(c[K]) + 1e-12)))

    return TransformResult(
        transformed=out,
        indices=np.arange(-K, K + 1),
        symmetric_indices=sym_idx,
        asymmetric_indices=asym_idx,
        n_symmetric=len(sym_idx),
        n_asymmetric=len(asym_idx),
    )


# ============================================================
#  4. Bucket sort  O(K)  (quick‑training.py)
# ============================================================
def bucket_sort(values: np.ndarray,
                order: Optional[np.ndarray] = None,
                n_buckets: int = N_BUCKETS
                ) -> Tuple[np.ndarray, np.ndarray]:
    n = len(values)
    if order is None:
        order = np.arange(n)
    if n < 2:
        return values.copy(), order.copy()

    lo, hi = LOG_MIN, LOG_MAX
    span = hi - lo
    buckets: List[List[int]] = [[] for _ in range(n_buckets)]
    for k in range(n):
        v = values[k]
        if v < lo:
            v = lo
        elif v > hi:
            v = hi
        b = int((v - lo) / span * (n_buckets - 1))
        buckets[b].append(k)

    out_vals = np.empty(n, dtype=values.dtype)
    out_order = np.empty(n, dtype=order.dtype)
    pos = 0
    for b in buckets:
        for k in b:
            out_vals[pos] = values[k]
            out_order[pos] = order[k]
            pos += 1
    return out_vals, out_order


# ============================================================
#  5. Supertrace / entropy  (shared)
# ============================================================
def supertrace(x) -> float:
    S = 0.0
    for i, v in enumerate(x):
        S += v if (i % 2 == 0) else -v
    return S


def entropy_from_supertrace(S: float, K: int, alpha: float = ALPHA) -> float:
    if K <= 0 or S == 0.0:
        return 0.0
    p = abs(S) / K
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -alpha * p * math.log(p)


# ============================================================
#  6. Merge sort + inverse score  O(K log K)
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
    while i <= mid:
        temp[k] = arr[i]; i += 1; k += 1
    while j <= right:
        temp[k] = arr[j]; j += 1; k += 1
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
#  7. PDE document vector (# app.py)  O(K)
# ============================================================
class PDEDocumentVector:
    """
    Simplified version of the semiotic‑differential PDE from # app.py.
    W_in and W_out are seeded per K so that the same vocab gives the
    same embedding deterministically.
    """
    def __init__(self, K: int = K_PDE, vocab_size: int = 1000, seed: int = 42):
        self.K = K
        self.V = vocab_size
        rng = np.random.default_rng(seed)
        self.W_in = rng.standard_normal((K, vocab_size)) * 0.01
        self.anchors = np.arange(0, K, max(1, K // 20))
        self.DT, self.ALPHA_PDE, self.EPS = 0.1, 0.1, 0.01

    def laplacian(self, h):
        diff = np.zeros_like(h)
        if self.K >= 3:
            diff[1:-1] = h[:-2] + h[2:] - 2.0 * h[1:-1]
        return diff

    def step(self, h, token):
        if token < self.V:
            h += self.W_in[:, token]
        diff = self.laplacian(h)
        an_sum = float(np.sum(h[self.anchors]))
        h += self.DT * (diff + self.ALPHA_PDE * an_sum - self.EPS * h)
        return h

    def vector(self, tokens) -> np.ndarray:
        h = np.zeros(self.K)
        for t in tokens:
            h = self.step(h, int(t) % self.V)
        return h


# ============================================================
#  8. SSD multi‑layer store (ssd-ram-ui.py, minimal)
# ============================================================
class SSDMultiLayer:
    """
    Single memmap for `real`, LRU for `imag` (derived),
    on‑the‑fly `envelope`.
    """
    def __init__(self, root: str, K: int,
                 chunk_size: int = 512, cache_chunks: int = 8):
        self.root = root
        self.K = K
        self.chunk_size = chunk_size
        self.cache_chunks = cache_chunks
        os.makedirs(root, exist_ok=True)

        path = os.path.join(root, "real.mm")
        self.real = np.memmap(path, dtype=np.float64, mode="w+", shape=(K,))
        self.imag_cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self.stats = dict(hits=0, misses=0, evictions=0)

    def _bounds(self, cid):
        s = cid * self.chunk_size
        e = min(s + self.chunk_size, self.K)
        return s, e

    def write_real(self, arr):
        arr = np.asarray(arr, dtype=np.float64)[:self.K]
        self.real[:arr.size] = arr
        if arr.size < self.K:
            self.real[arr.size:] = 0.0
        self.real.flush()

    def _imag_chunk(self, cid: int) -> np.ndarray:
        s, e = self._bounds(cid)
        n = e - s
        extra = min(e + 1, self.K)
        raw = np.asarray(self.real[s:extra], dtype=np.float64)
        imag = np.zeros(n, dtype=np.float64)
        if raw.size >= 2:
            d = (raw[1:] - raw[:-1]) / A_STEP
            m = min(d.size, n)
            imag[:m] = -d[:m]
            if m < n:
                imag[n - 1] = -(raw[-1] - raw[-2]) / A_STEP
        return imag

    def imag_chunk(self, cid: int) -> np.ndarray:
        if cid in self.imag_cache:
            self.imag_cache.move_to_end(cid)
            self.stats['hits'] += 1
            return self.imag_cache[cid]
        self.stats['misses'] += 1
        arr = self._imag_chunk(cid)
        self.imag_cache[cid] = arr
        while len(self.imag_cache) > self.cache_chunks:
            self.imag_cache.popitem(last=False)
            self.stats['evictions'] += 1
        return arr

    def real_array(self) -> np.ndarray:
        return np.asarray(self.real[:], dtype=np.float64)

    def envelope(self) -> np.ndarray:
        n_chunks = (self.K + self.chunk_size - 1) // self.chunk_size
        out = np.zeros(self.K)
        for cid in range(n_chunks):
            s, e = self._bounds(cid)
            real = np.asarray(self.real[s:e], dtype=np.float64)
            imag = self.imag_chunk(cid)
            m = min(real.size, imag.size)
            out[s:s + m] = np.hypot(real[:m], imag[:m])
        return out

    def cache_report(self) -> Dict:
        return dict(self.stats,
                    cache_size=len(self.imag_cache),
                    capacity=self.cache_chunks,
                    chunk_size=self.chunk_size)


# ============================================================
#  9. Memory entry
# ============================================================
@dataclass
class MemoryEntry:
    query: str
    scalar: float                     # learnt importance scalar
    supertrace: float
    entropy: float
    kept: List[Tuple[int, float]]     # (index, value) from signed‑log
    top_tokens: List[str]
    n_symmetric: int
    n_asymmetric: int
    timestamp: int


# ============================================================
#  10. The learning search model
# ============================================================
class LearningSearchModel:
    """
    Learns from each query, searches the SSD‑backed memory in
    O(K log K), analyses tokens like quick-reader-analyzer.py,
    and returns the top tokens plus the most important coefficients.

    Decision rule (if/else on the scalar)
    -------------------------------------
        score < 0.4   → forward  : reinforce  (scalar *= 1 + lr·r)
        0.4 ≤ s ≤ 0.6 → neutral  : store      (new memory entry)
        score > 0.6   → inverse  : decay      (scalar *= 1 − lr·r)

    Complexity
    ----------
        sieve            O(K log log K)   ← once
        per query:
          |Ci|, log, sort      O(K)
          PDE vector           O(K)
          inverse search       O(K log K)  (merge sort)
          SSD write + imag     O(K) on miss
    """

    def __init__(self,
                 K: int = 256,
                 vocab_size: int = 1000,
                 scratch: str = "lsm_scratch",
                 chunk_size: int = 256,
                 cache_chunks: int = 8,
                 lr: float = 0.05,
                 baseline: float = 0.0):
        self.K = K
        self.lr = lr
        self.baseline = baseline

        # shared sieve (built once)
        self.mu = mobius_sieve(K)

        # quick‑training transform tolerance
        self.tol = 1e-6

        # PDE document vector
        self.pde = PDEDocumentVector(K=K_PDE, vocab_size=vocab_size)

        # SSD multi‑layer memory
        self.store = SSDMultiLayer(scratch, K,
                                   chunk_size=chunk_size,
                                   cache_chunks=cache_chunks)

        # in‑RAM memory of past queries (small, LRU evicted)
        self.memory: List[MemoryEntry] = []
        self.max_memory = 256
        self.history: deque = deque(maxlen=512)

        # learnt weight per token (for the if/else rule)
        self.token_weights: Dict[str, float] = {}

    # ---------- summary ----------
    def summary(self) -> Dict:
        ...
    # ---------- token helpers ----------
    def _token_index(self, token: str) -> int:
        h = int(hashlib.md5(token.encode()).hexdigest()[:8], 16)
        return h % 10**6

    # ---------- quick‑training pipeline ----------
    def prepare(self, tokens) -> Dict:
        c = build_k_sum(tokens, self.K, self.mu)
        tr = classify_and_transform(c, self.K, tol=self.tol)
        sorted_vals, sorted_order = bucket_sort(tr.transformed)
        return dict(
            c=c, transformed=tr.transformed, indices=tr.indices,
            sorted_values=sorted_vals, sorted_order=sorted_order,
            n_symmetric=tr.n_symmetric, n_asymmetric=tr.n_asymmetric,
        )

    # ---------- search ----------       
    def search(self, query: str, top_k: int = 5) -> Dict:
        t0 = time.perf_counter()
        tokens = query.lower().split()

        # 1. quick‑training pipeline
        prepared = self.prepare(tokens)
        mag = np.abs(prepared['transformed'])
        idx_sorted = np.argsort(mag)[::-1][:max(1, top_k * 4)]
        kept = [(int(i), float(prepared['transformed'][i])) for i in idx_sorted]

        # 2. supertrace / entropy
        S = supertrace(prepared['transformed'])
        H = entropy_from_supertrace(S, 2 * self.K + 1)

        # 3. PDE document vector
        token_ids = [self._token_index(t) for t in tokens]
        pde_vec = self.pde.vector(token_ids)

        # 4. inverse‑score search over memory
        cur_kept_dict = {int(i): float(prepared['transformed'][i])
                         for i in idx_sorted}
        scored: List[Tuple[float, MemoryEntry]] = []
        for entry in self.memory:
            stored_dict = {int(k): float(v) for k, v in entry.kept}
            common = sorted(set(cur_kept_dict.keys()) & set(stored_dict.keys()))
            if len(common) >= 2:
                a = [cur_kept_dict[i] for i in common]
                b = [stored_dict[i] for i in common]
                sc = inverse_score(a, b)
            else:
                sc = 0.5
            scored.append((sc, entry))
        scored.sort(key=lambda p: p[0])
        top_matches = scored[:top_k]

        # 5. if/else decision
        if top_matches:
            best_score, best_entry = top_matches[0]
        else:
            best_score, best_entry = 0.5, None

        if best_score < 0.4:
            branch = 'forward'
            reward = 1.0 - best_score
            if best_entry:
                for t in best_entry.top_tokens:
                    self.token_weights[t] = (
                        self.token_weights.get(t, 0.0) *
                        (1.0 + self.lr * (reward - self.baseline))
                    )
        elif best_score <= 0.6:
            branch = 'neutral'
            reward = 0.5
        else:
            branch = 'inverse'
            reward = 1.0 - best_score
            for t in tokens:
                self.token_weights[t] = (
                    self.token_weights.get(t, 0.0) *
                    (1.0 - self.lr * (reward - self.baseline))
                )

        # 6. store the new query as memory
        entry = MemoryEntry(
            query=query,
            scalar=reward,
            supertrace=S,
            entropy=H,
            kept=kept,
            top_tokens=tokens[:8],
            n_symmetric=prepared['n_symmetric'],
            n_asymmetric=prepared['n_asymmetric'],
            timestamp=len(self.memory),
        )
        self.memory.append(entry)
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)

        # 7. write |Ci| to SSD
        self.store.write_real(prepared['c'])
        _ = self.store.imag_chunk(0)

        # 8. history
        elapsed = time.perf_counter() - t0
        self.history.append(dict(
            query=query, branch=branch, score=best_score,
            S=S, H=H, time=elapsed,
            cache=self.store.cache_report(),
        ))

        # 9. payload
        return dict(
            query=query,
            branch=branch,
            best_score=best_score,
            reward=reward,
            supertrace=S,
            entropy=H,
            n_symmetric=prepared['n_symmetric'],
            n_asymmetric=prepared['n_asymmetric'],
            top_kept=kept[:top_k],
            top_matches=[
                dict(query=e.query, score=s, scalar=e.scalar,
                     S=e.supertrace, H=e.entropy,
                     tokens=e.top_tokens)
                for s, e in top_matches
            ],
            pde_norm=float(np.linalg.norm(pde_vec)),
            token_weights=dict(sorted(
                ((t, self.token_weights[t]) for t in tokens
                 if t in self.token_weights),
                key=lambda p: -abs(p[1])
            )),
            cache=self.store.cache_report(),
            elapsed_ms=elapsed * 1e3,
        )
    # ---------- summary ----------
    def summary(self) -> Dict:
        if not self.history:
            return dict(n_queries=0)
        branches = {}
        for h in self.history:
            branches[h['branch']] = branches.get(h['branch'], 0) + 1
        return dict(
            n_queries=len(self.history),
            branches=branches,
            mean_time_ms=float(np.mean([h['time'] for h in self.history]) * 1e3),
            mean_S=float(np.mean([h['S'] for h in self.history])),
            mean_H=float(np.mean([h['H'] for h in self.history])),
            cache=self.store.cache_report(),
        )


# ============================================================
#  11. Demo
# ============================================================
def demo():
    print("=" * 72)
    print("Learning search model  ·  O(K log log K) learn / O(K log K) search")
    print("=" * 72)

    K = 256
    model = LearningSearchModel(
        K=K, vocab_size=1000,
        scratch="lsm_scratch", chunk_size=128, cache_chunks=4,
        lr=0.08, baseline=0.0,
    )

    # -- some seed queries ------------------------------------------
    seeds = [
        "quantum entanglement of photons",
        "Möbius function square free",
        "Basel problem 6 over pi squared",
        "pendulum chaos and entropy",
        "creative memory exploration",
    ]
    print("\n--- Seed queries ---")
    for q in seeds:
        out = model.search(q)
        print(f"  {q[:40]:42s}  "
              f"branch={out['branch']:8s}  "
              f"score={out['best_score']:.4f}  "
              f"S={out['supertrace']:+.4f}  H={out['entropy']:.4f}  "
              f"sym={out['n_symmetric']:>4d}  asym={out['n_asymmetric']:>4d}  "
              f"({out['elapsed_ms']:.1f} ms)")

    # -- test queries ----------------------------------------------
    print("\n--- Test queries ---")
    tests = [
        "photons and entanglement",
        "square free Möbius",
        "chaotic pendulum",
        "something completely unrelated",
    ]
    for q in tests:
        out = model.search(q)
        print(f"  {q[:40]:42s}  "
              f"branch={out['branch']:8s}  "
              f"score={out['best_score']:.4f}  "
              f"S={out['supertrace']:+.4f}  H={out['entropy']:.4f}  "
              f"({out['elapsed_ms']:.1f} ms)")
        if out['top_matches']:
            best = out['top_matches'][0]
            print(f"      ↳ best match: '{best['query'][:40]}'  "
                  f"score={best['score']:.4f}  scalar={best['scalar']:.4f}")

    # -- retrieve the most important coefficients --------------------
    print("\n--- Retrieval from SSD memory ---")
    r = model.retrieve(k=1)
    print(f"  envelope mean = {r['envelope_mean']:.4f}, "
          f"std = {r['envelope_std']:.4f}")
    print("  top kept (index, |Ci|):")
    for i, v in r['top_kept']:
        print(f"    idx={i:>4d}  v={v:+8.4f}")
    print(f"  cache: {r['cache']}")

    # -- summary ----------------------------------------------------
    print("\n--- Summary ---")
    s = model.summary()
    for k, v in s.items():
        if isinstance(v, float):
            print(f"  {k:>15s}: {v:.4f}")
        else:
            print(f"  {k:>15s}: {v}")

    # -- token weight table ----------------------------------------
    print("\n--- Learnt token weights (top 10) ---")
    tw = sorted(model.token_weights.items(),
                key=lambda p: -abs(p[1]))[:10]
    for t, w in tw:
        print(f"  {t:20s}  {w:+.4f}")

    # -- complexity ------------------------------------------------
    print("\n--- Complexity ---")
    print("  Möbius sieve (once)         O(K log log K)")
    print("  Per query:")
    print("    |Ci| + signed log + sort  O(K)")
    print("    PDE document vector       O(K)")
    print("    inverse search (merge)    O(K log K)")
    print("    SSD write + imag layer    O(K) on miss")
    print("\nDone.")
    
if __name__ == "__main__":
    demo()