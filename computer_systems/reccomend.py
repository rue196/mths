#!/usr/bin/env python3
"""
recommendation_mobius.py

O(K log K) recommendation algorithm with two coupling rules.

Inputs
------
Each content item (word, video, photo) is represented by
  • a scalar sequence   s ∈ R^K          (Möbius-filtered hash)
  • a 2‑D position      (x, y) ∈ [0.5, 5.0]²
  • a timeline position p ∈ R

Rule 1 — Inverse‑score coupling (from ML.py)
--------------------------------------------
Similarity between a query and a candidate is the merge‑sort
inversion count of the candidate scalar reordered by the query
scalar.  Farther candidates decay on the scalar:

    score(q, c) = (1 − inverse_score(q, c)) · exp(−|p_q − p_c| / a)

with a = 1/(π−e)/0.3628 ≈ 6.511 (the finite‑step constant).

Rule 2 — M‑matrix PDE (replaces the Laplacian in Semiotic-differential.c)
-------------------------------------------------------------------------
For each item i the local operator is the trace of

        M_i = [[ x_prev⁻¹ ,  y_prev⁻¹ ],
               [ x_newⁱ  ,  y_newⁱ  ]]

  • algebraic row    (x_prev⁻¹, y_prev⁻¹)  →  previous (stored) entry
  • transcendental row (x_newⁱ, y_newⁱ)    →  new (candidate) entry

The Laplacian is replaced by

    diff[i] = tr(M_i) · h[i]

so the previous entry acts as the left‑neighbour influence and the
candidate recommendation acts as the right‑neighbour influence.
The rest of the PDE step is unchanged:

    h ← h + DT · (diff + α · anchor_sum(h) − ε · h)

All operations are O(K log K) (sieve + merge‑sort inversion).
"""

import math
import time
import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ============================================================
#  Constants
# ============================================================
PI         = math.pi
E          = math.e
ALPHA      = 1.0 / (PI - E)             # ≈ 2.362
ALPHA_USER = 0.3628
A_STEP     = ALPHA / ALPHA_USER         # ≈ 6.511  decay length
DT         = 0.1
EPS        = 0.01
ANCHOR_ALPHA = 0.1
I_EXP      = 2                          # exponent in transcendental row


# ============================================================
#  Möbius sieve (O(K))
# ============================================================
def mobius_sieve(K: int) -> List[int]:
    mu = [0] * (K + 1)
    if K >= 1:
        mu[1] = 1
    primes, is_comp = [], [False] * (K + 1)
    for i in range(2, K + 1):
        if not is_comp[i]:
            primes.append(i); mu[i] = -1
        for p in primes:
            if i * p > K:
                break
            is_comp[i * p] = True
            if i % p == 0:
                mu[i * p] = 0; break
            else:
                mu[i * p] = -mu[i]
    return mu


# ============================================================
#  Merge sort + inversion count (O(K log K))
# ============================================================
def _merge_and_count(a, tmp, l, m, r):
    i, j, k, inv = l, m + 1, l, 0
    while i <= m and j <= r:
        if a[i] <= a[j]:
            tmp[k] = a[i]; i += 1
        else:
            tmp[k] = a[j]; inv += m - i + 1; j += 1
        k += 1
    while i <= m:
        tmp[k] = a[i]; i += 1; k += 1
    while j <= r:
        tmp[k] = a[j]; j += 1; k += 1
    for i in range(l, r + 1):
        a[i] = tmp[i]
    return inv


def _merge_sort(a, tmp, l, r):
    inv = 0
    if l < r:
        m = (l + r) // 2
        inv += _merge_sort(a, tmp, l, m)
        inv += _merge_sort(a, tmp, m + 1, r)
        inv += _merge_and_count(a, tmp, l, m, r)
    return inv


def inversion_count(arr):
    n = len(arr)
    if n < 2:
        return 0
    return _merge_sort(list(arr), [0] * n, 0, n - 1)


def inverse_score(a, b) -> float:
    """
    Normalised inversion count ∈ [0, 1].
    0 → same ordering, 1 → opposite ordering.
    """
    if len(a) != len(b) or len(a) < 2:
        return 0.5
    pairs = sorted(zip(a, b), key=lambda p: p[0])
    b_sorted = [p[1] for p in pairs]
    inv = inversion_count(b_sorted)
    K = len(a)
    mx = K * (K - 1) // 2
    return inv / mx if mx else 0.0


# ============================================================
#  Content → scalar
# ============================================================
def content_scalar(text: str, K: int, mu: List[int]) -> np.ndarray:
    """
    Möbius‑filtered hash of a content string.
    Square‑free indices carry the signal; others are zero.
    """
    raw = hashlib.shake_256(text.encode("utf-8")).digest(K * 4)
    sig = np.frombuffer(raw, dtype=np.float32).astype(np.float64)
    sig = np.abs(sig)
    for i in range(K):
        if mu[i + 1] == 0:
            sig[i] = 0.0
    return sig


# ============================================================
#  Decay on the scalar
# ============================================================
def decay(distance: float, a: float = A_STEP) -> float:
    """exp(−|d|/a).  Further away ⇒ smaller weight."""
    return math.exp(-abs(distance) / a)


# ============================================================
#  M‑matrix primitives
# ============================================================
def m_trace(x_prev, y_prev, x_new, y_new, i=I_EXP) -> float:
    """
    Trace of  M = [[x_prev⁻¹, y_prev⁻¹],
                   [x_newⁱ ,  y_newⁱ ]].
    Algebraic row → previous (stored) entry.
    Transcendental row → new (candidate) entry.
    """
    xp = max(abs(x_prev), 1e-6)
    yp = max(abs(y_prev), 1e-6)
    xn = max(abs(x_new), 1e-6)
    yn = max(abs(y_new), 1e-6)
    return (xp ** -1) + (yn ** i)


def m_det(x_prev, y_prev, x_new, y_new, i=I_EXP) -> float:
    """det(M) = x_prev⁻¹ · y_newⁱ − y_prev⁻¹ · x_newⁱ."""
    xp = max(abs(x_prev), 1e-6)
    yp = max(abs(y_prev), 1e-6)
    xn = max(abs(x_new), 1e-6)
    yn = max(abs(y_new), 1e-6)
    return (xp ** -1) * (yn ** i) - (yp ** -1) * (xn ** i)


# ============================================================
#  Item
# ============================================================
@dataclass
class Item:
    id: str
    kind: str                # 'word' | 'video' | 'photo'
    text: str
    position: float          # timeline position
    x: float                 # algebraic coordinate
    y: float                 # transcendental coordinate
    scalar: Optional[np.ndarray] = None
    pde_state: Optional[np.ndarray] = None


# ============================================================
#  Algorithm 1 — Inverse‑score recommender
# ============================================================
class InverseScoreRecommender:
    def __init__(self, K: int, mu: List[int]):
        self.K = K
        self.mu = mu
        self.items: List[Item] = []

    def add(self, item: Item):
        item.scalar = content_scalar(item.text, self.K, self.mu)
        self.items.append(item)

    def score(self, q_scalar, q_pos, item: Item, a=A_STEP):
        sim = 1.0 - inverse_score(q_scalar, item.scalar)     # 0..1
        w   = sim * decay(item.position - q_pos, a)
        return w, sim

    def recommend(self, q_scalar, q_pos, top_k=5, threshold=0.0):
        scored = []
        for it in self.items:
            w, sim = self.score(q_scalar, q_pos, it)
            if w >= threshold:
                scored.append((w, sim, it))
        scored.sort(key=lambda x: -x[0])
        return scored[:top_k]


# ============================================================
#  Algorithm 2 — M‑matrix PDE recommender
# ============================================================
class MMatrixPDE:
    """
    Replaces the discrete Laplacian of semiotic-differential.c with
    the trace of the M‑matrix built from the previous item (algebraic)
    and the candidate (transcendental).

    Original:
        diff[i] = h[i-1] + h[i+1] - 2 h[i]

    New:
        diff[i] = tr(M) · h[i]
    where  M = [[x_prev⁻¹, y_prev⁻¹], [x_newⁱ, y_newⁱ]]
    and  x_prev,y_prev  come from the *previous* item,
         x_new, y_new    come from the *candidate*.
    """
    def __init__(self, K=64, n_anchors=20):
        self.K = K
        self.h = np.zeros(K)
        self.anchors = np.linspace(0, K - 1, n_anchors, dtype=int)
        self.history: List[np.ndarray] = []

    def _anchor_sum(self):
        return float(np.sum(self.h[self.anchors]))

    def step(self, x_prev, y_prev, x_new, y_new, i=I_EXP):
        tr = m_trace(x_prev, y_prev, x_new, y_new, i)
        local = tr * self.h                      # replaces Laplacian
        nonlocal_term = ANCHOR_ALPHA * self._anchor_sum()
        self.h = self.h + DT * (local + nonlocal_term - EPS * self.h)
        self.history.append(self.h.copy())
        return self.h

    def supertrace(self) -> float:
        S = 0.0
        for i, v in enumerate(self.h):
            S += v if (i % 2 == 0) else -v
        return S


# ============================================================
#  Combined recommender
# ============================================================
class MobiusRecommender:
    """
    Combines:
        (a) inverse‑score similarity on the content scalar
        (b) M‑matrix PDE drive on the algebraic (previous) and
            transcendental (candidate) rows.
    """

    def __init__(self, K: int = 64):
        self.K = K
        self.mu = mobius_sieve(K)
        self.inv = InverseScoreRecommender(K, self.mu)
        self.pde = MMatrixPDE(K)
        self.last_item: Optional[Item] = None

    # ---------- ingestion ----------
    def add(self, id, kind, text, position, x=None, y=None):
        if x is None or y is None:
            h = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            x = 0.5 + (h % 1000) / 1000.0 * 4.5
            y = 0.5 + ((h >> 10) % 1000) / 1000.0 * 4.5
        item = Item(id=id, kind=kind, text=text, position=float(position),
                    x=x, y=y)
        self.inv.add(item)
        return item

    # ---------- query ----------
    def recommend(self, query_text, query_pos,
                  top_k: int = 5, threshold: float = 0.0):
        q_scalar = content_scalar(query_text, self.K, self.mu)

        # 1) inverse‑score recommendations
        raw = self.inv.recommend(q_scalar, query_pos,
                                 top_k=top_k, threshold=threshold)

        # 2) M‑matrix PDE drive on the top recommendation
        pde_energy = 0.0
        if raw:
            best = raw[0][2]
            # previous item → algebraic row
            if self.last_item is not None:
                xp, yp = self.last_item.x, self.last_item.y
            else:
                xp, yp = 0.5, 0.5
            # candidate → transcendental row
            xn, yn = best.x, best.y
            self.pde.step(xp, yp, xn, yn)
            pde_energy = self.pde.supertrace()
            self.last_item = best

        return raw, pde_energy


# ============================================================
#  Demo — words, video captions, photo captions
# ============================================================
def demo():
    print("=" * 68)
    print("Möbius recommendation  ·  inverse score + M‑matrix PDE")
    print("=" * 68)

    K = 64
    rec = MobiusRecommender(K=K)

    # -- corpus (words / video / photo) ------------------------
    corpus = [
        ("w_quantum",   "word",  "quantum entanglement of photons",     0.0),
        ("w_entropy",   "word",  "entropy of a black hole horizon",     5.0),
        ("v_cat",       "video", "a cat playing with a ball of yarn",  12.0),
        ("p_sunset",    "photo", "golden sunset over the ocean",       20.0),
        ("v_rocket",    "video", "rocket launch at cape canaveral",    28.0),
        ("w_mobius",    "word",  "Möbius function and square-free",    33.0),
        ("p_mountain",  "photo", "snowy mountain peak at dawn",        40.0),
        ("v_dance",     "video", "contemporary dance performance",     47.0),
        ("w_basel",     "word",  "Basel problem and 6 over pi squared",52.0),
        ("p_city",      "photo", "neon city street at midnight",       58.0),
        ("v_recipe",    "video", "cooking pasta from scratch",         65.0),
        ("w_pendulum",  "word",  "pendulum motion and chaos theory",   72.0),
    ]
    for cid, kind, text, pos in corpus:
        rec.add(cid, kind, text, pos)
    print(f"Ingested {len(rec.inv.items)} items.\n")

    # -- queries ------------------------------------------------
    queries = [
        ("quantum physics and entanglement",  6.0),
        ("sunset over water",                22.0),
        ("space rocket launch",              30.0),
        ("number theory and Möbius",         35.0),
        ("chaotic pendulum dynamics",        74.0),
    ]

    print(f"{'query':40s} {'pos':>5s} "
          f"{'recommended':12s} {'score':>7s} {'sim':>7s} "
          f"{'decay':>7s} {'PDE S':>9s}")
    print("-" * 92)

    for q_text, q_pos in queries:
        raw, pde_S = rec.recommend(q_text, q_pos, top_k=3, threshold=0.0)
        if not raw:
            print(f"{q_text[:38]:40s} {q_pos:5.1f}  (no match)")
            continue
        for i, (w, sim, it) in enumerate(raw):
            d = decay(it.position - q_pos)
            head = q_text[:38] if i == 0 else ""
            print(f"{head:40s} {q_pos:5.1f} "
                  f"{it.id:12s} {w:7.4f} {sim:7.4f} {d:7.4f} "
                  f"{pde_S if i == 0 else 0.0:9.4f}")

    # -- M‑matrix trace / determinant on a toy pair ----------
    print("\n--- M‑matrix primitives on a toy pair ---")
    x_prev, y_prev = 1.2, 2.5
    x_new,  y_new  = 3.4, 1.1
    tr  = m_trace(x_prev, y_prev, x_new, y_new, I_EXP)
    det = m_det  (x_prev, y_prev, x_new, y_new, I_EXP)
    print(f"  previous (algebraic)  : x⁻¹={x_prev**-1:.4f}  "
          f"y⁻¹={y_prev**-1:.4f}")
    print(f"  candidate (transcend.) : x^i={x_new**I_EXP:.4f}  "
          f"y^i={y_new**I_EXP:.4f}")
    print(f"  tr(M) = x_prev⁻¹ + y_new^i = {tr:.6f}")
    print(f"  det(M)                     = {det:.6f}")

    # -- timing -------------------------------------------------
    print("\n--- complexity check ---")
    for n in [32, 64, 128, 256]:
        t0 = time.perf_counter()
        r = MobiusRecommender(K=n)
        for i in range(50):
            r.add(f"id_{i}", "word", f"content number {i} alpha beta", i * 1.0)
        for i in range(10):
            r.recommend("content alpha", 3.0, top_k=3)
        dt = time.perf_counter() - t0
        print(f"  K={n:4d}   50 items + 10 queries   {dt*1e3:8.2f} ms   "
              f"(≈ {dt/60*1e3:.3f} ms/op)")

    print("\nDone.")


if __name__ == "__main__":
    demo()