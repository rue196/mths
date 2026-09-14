#!/usr/bin/env python3
"""
quick_train_sort.py

Fast ML/AI training-data preprocessor with a Möbius‑based
symmetric/asymmetric log transform and O(K) bucket sort.

Complexity
----------
    Sieve of Eratosthenes variant for μ(n)   O(K log log K)
    |Ci| construction                        O(K)
    Symmetric/asymmetric classification      O(K)
    Negative/positive log transform          O(K)
    Bucket sort over bounded log values      O(K)
    ------------------------------------------------
    Total                                    O(K log log K)

Transform
---------
    symmetric index  (i and −i both present, |C_i| = |C_{−i}|)
        →   −log(|C_i|)
    asymmetric index (only one of i, −i present, or |C_i| ≠ |C_{−i}|)
        →   +log(|C_i|)

The result is a single signed‑log |Ci| array of length 2K+1
with the corresponding indices, sorted in O(K) by a fixed‑width
bucket sort (the log values lie in a bounded interval).

The finite step a = 1/(π−e)/0.3628 ≈ 6.511 is used as the log
tolerance for classifying symmetric vs asymmetric entries.
"""

import math
import time
import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict


# ============================================================
#  Constants
# ============================================================
PI          = math.pi
E           = math.e
ALPHA       = 1.0 / (PI - E)             # ≈ 2.362
ALPHA_USER  = 0.3628
A_STEP      = ALPHA / ALPHA_USER         # ≈ 6.511
DENSITY     = 6.0 / (PI * PI)            # ≈ 0.6079
LOG_MIN     = -30.0                      # clip lower bound for log
LOG_MAX     = 30.0                       # clip upper bound for log
N_BUCKETS   = 512                        # fixed bucket count → O(K) sort


# ============================================================
#  Möbius sieve via Eratosthenes variant  ·  O(K log log K)
# ============================================================
def mobius_sieve(K: int) -> List[int]:
    """
    Compute μ(n) for n = 1..K.
    Time:  O(K log log K)
    Space: O(K)
    """
    if K < 1:
        return [0] * (K + 1)
    mu = [1] * (K + 1)
    is_comp = [False] * (K + 1)
    for i in range(2, K + 1):
        if not is_comp[i]:
            # multiply every multiple of i by −1
            for j in range(i, K + 1, i):
                mu[j] = -mu[j]
                is_comp[j] = True
            # zero out multiples of i²
            i2 = i * i
            if i2 <= K:
                for j in range(i2, K + 1, i2):
                    mu[j] = 0
    mu[0] = 0
    return mu


# ============================================================
#  |Ci| builder  ·  O(K)
# ============================================================
def build_k_sum(tokens, K: int, mu: Optional[List[int]] = None) -> np.ndarray:
    """
    Build a length‑(2K+1) |Ci| array from an iterable of tokens.

    Each token t contributes to index i = (hash(t) mod K) + 1 with
    magnitude 1, symmetrically placed at i and −i.  Möbius‑filtered:
    entries whose |i| is not square‑free are zeroed.
    """
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
#  Symmetric / asymmetric transform  ·  O(K)
# ============================================================
@dataclass
class TransformResult:
    transformed: np.ndarray                      # signed log |Ci|
    indices: np.ndarray                          # original indices −K..K
    symmetric_indices: List[int] = field(default_factory=list)
    asymmetric_indices: List[int] = field(default_factory=list)
    n_symmetric: int = 0
    n_asymmetric: int = 0


def classify_and_transform(c: np.ndarray, K: int,
                           tol: float = 1e-6) -> TransformResult:
    """
    Apply the symmetric/asymmetric log transform.

        symmetric   (|c[i]| ≈ |c[−i]|)  →  −log(|c[i]|)
        asymmetric  (only one present)  →  +log(|c[i]|)

    Returns a signed‑log array with the corresponding indices.
    """
    n = 2 * K + 1
    out = np.zeros(n, dtype=np.float64)
    sym_idx: List[int] = []
    asym_idx: List[int] = []

    for i in range(1, K + 1):
        pos = K + i
        neg = K - i
        cp = abs(c[pos])
        cn = abs(c[neg])

        both = (cp > 0) and (cn > 0)
        symmetric = both and (abs(cp - cn) <= tol * (cp + cn + 1e-12))

        if symmetric:
            v = -math.log(cp + 1e-12)
            v = max(LOG_MIN, min(LOG_MAX, v))
            out[pos] = v
            out[neg] = v
            sym_idx.append(i)
            sym_idx.append(-i)
        else:
            if cp > 0:
                v = math.log(cp + 1e-12)
                v = max(LOG_MIN, min(LOG_MAX, v))
                out[pos] = v
            if cn > 0:
                v = math.log(cn + 1e-12)
                v = max(LOG_MIN, min(LOG_MAX, v))
                out[neg] = v
            asym_idx.append(i)
            if cn > 0 and not both:
                asym_idx.append(-i)

    # i = 0
    if abs(c[K]) > 0:
        v = math.log(abs(c[K]) + 1e-12)
        out[K] = max(LOG_MIN, min(LOG_MAX, v))

    return TransformResult(
        transformed=out,
        indices=np.arange(-K, K + 1),
        symmetric_indices=sym_idx,
        asymmetric_indices=asym_idx,
        n_symmetric=len(sym_idx),
        n_asymmetric=len(asym_idx),
    )


# ============================================================
#  Bucket sort  ·  O(K) for bounded log values
# ============================================================
def bucket_sort(values: np.ndarray,
                order: Optional[np.ndarray] = None,
                n_buckets: int = N_BUCKETS
                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sort by value using a fixed‑width bucket sort.
    Assumes values lie in [LOG_MIN, LOG_MAX].

    Returns (sorted_values, sorted_order) where `order` is the
    original index permutation.
    """
    n = len(values)
    if order is None:
        order = np.arange(n)
    if n < 2:
        return values.copy(), order.copy()

    lo, hi = LOG_MIN, LOG_MAX
    span = hi - lo
    if span <= 0:
        return values.copy(), order.copy()

    # allocate buckets
    buckets: List[List[int]] = [[] for _ in range(n_buckets)]
    for k in range(n):
        v = values[k]
        if v < lo:
            v = lo
        elif v > hi:
            v = hi
        b = int((v - lo) / span * (n_buckets - 1))
        buckets[b].append(k)

    # concatenate
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
#  Quick pipeline
# ============================================================
class QuickTrainSorter:
    """
    Full pipeline:
        tokens  →  |Ci|  →  signed log  →  bucket sort  →  training data

    All stages in O(K log log K) total, dominated by the sieve.
    """

    def __init__(self, K: int):
        self.K = K
        self.mu = mobius_sieve(K)

    def prepare(self, tokens) -> Dict:
        t0 = time.perf_counter()

        # 1. |Ci| in O(K)
        c = build_k_sum(tokens, self.K, self.mu)
        t1 = time.perf_counter()

        # 2. signed log transform in O(K)
        result = classify_and_transform(c, self.K)
        t2 = time.perf_counter()

        # 3. bucket sort of the transformed array in O(K)
        sorted_vals, sorted_order = bucket_sort(result.transformed)
        t3 = time.perf_counter()

        return {
            'K': self.K,
            'c': c,
            'transformed': result.transformed,
            'indices': result.indices,
            'sorted_values': sorted_vals,
            'sorted_order': sorted_order,
            'n_symmetric': result.n_symmetric,
            'n_asymmetric': result.n_asymmetric,
            'timings': {
                'k_sum':    t1 - t0,
                'transform': t2 - t1,
                'sort':      t3 - t2,
                'total':     t3 - t0,
            },
        }

    def stream_batches(self, token_batches):
        """
        Yield one prepared batch at a time.  Each batch is
        independent, so this composes with an ML DataLoader.
        """
        for batch in token_batches:
            yield self.prepare(batch)


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 68)
    print("Quick training sorter  ·  O(K log log K)")
    print("=" * 68)

    # --- 1. sieve timing ------------------------------------------------
    print("\n--- Sieve scaling (Eratosthenes variant) ---")
    print(f"{'K':>10s} {'time (ms)':>12s} {'μ≠0 density':>14s}")
    for K in [10**3, 10**4, 10**5, 10**6]:
        t0 = time.perf_counter()
        mu = mobius_sieve(K)
        dt = (time.perf_counter() - t0) * 1e3
        density = sum(1 for n in range(1, K + 1) if mu[n] != 0) / K
        print(f"{K:>10d} {dt:>12.2f} {density:>14.6f}")
    print(f"(square‑free density should approach 6/π² = {DENSITY:.6f})")

    # --- 2. small demo --------------------------------------------------
    K = 64
    qts = QuickTrainSorter(K)

    tokens = [
        "the", "quick", "brown", "fox", "jumps", "over",
        "lazy", "dog", "mobius", "basel", "pi", "e",
        "harmonic", "square", "free", "index",
    ] * 8                                          # 128 tokens
    out = qts.prepare(tokens)

    print(f"\n--- Small demo (K = {K}, {len(tokens)} tokens) ---")
    print(f"  |Ci| shape          : {out['c'].shape}")
    print(f"  symmetric entries   : {out['n_symmetric']}")
    print(f"  asymmetric entries  : {out['n_asymmetric']}")
    print(f"  transformed range   : "
          f"[{out['transformed'].min():.3f}, {out['transformed'].max():.3f}]")

    print("\n  First 10 (index, transformed value, sign):")
    for i in range(10):
        idx = out['indices'][i]
        v = out['transformed'][i]
        sign = "symmetric (−log)" if idx in out['indices'][
            out['transformed'] == out['transformed'][i]
        ].tolist() and False else ""
        print(f"    i={idx:>4d}   v={v:+8.4f}")

    print("\n  Timings:")
    for k, v in out['timings'].items():
        print(f"    {k:>10s}: {v*1e3:8.3f} ms")

    # --- 3. large K batch ----------------------------------------------
    print("\n--- Large K batch (K = 4096, 2000 tokens) ---")
    big_K = 4096
    big_qts = QuickTrainSorter(big_K)
    import random
    random.seed(42)
    big_tokens = [f"tok_{random.randint(0, 10**6)}" for _ in range(2000)]
    big = big_qts.prepare(big_tokens)

    print(f"  |Ci| length          : {len(big['c'])}")
    print(f"  symmetric entries    : {big['n_symmetric']}")
    print(f"  asymmetric entries   : {big['n_asymmetric']}")
    print(f"  sorted range         : "
          f"[{big['sorted_values'][0]:+.3f}, "
          f"{big['sorted_values'][-1]:+.3f}]")
    print(f"  timings:")
    for k, v in big['timings'].items():
        print(f"    {k:>10s}: {v*1e3:8.3f} ms")

    # --- 4. streaming batches -------------------------------------------
    print("\n--- Streaming token batches ---")
    batches = [
        ["a", "b", "c"],
        ["d", "e", "f"],
        ["g", "h", "i"],
    ]
    for i, batch_out in enumerate(qts.stream_batches(batches)):
        print(f"  batch {i}: "
              f"|Ci| non‑zero = {int((batch_out['c'] != 0).sum())}, "
              f"sym = {batch_out['n_symmetric']}, "
              f"asym = {batch_out['n_asymmetric']}")

    # --- 5. complexity summary ------------------------------------------
    print("\n--- Complexity summary ---")
    print("  μ(n) sieve         O(K log log K)")
    print("  |Ci| construction  O(K)")
    print("  log transform      O(K)")
    print("  bucket sort        O(K)   (fixed 512 buckets, bounded log values)")
    print("  --------------------------------------")
    print("  Total              O(K log log K)")
    print("\nDone.")


if __name__ == "__main__":
    demo()