#!/usr/bin/env python3
"""
quick_reader_analyzer.py

Quick reader + analyzer for large K sums (|Ci| arrays),
with strike‑flagging and escalation to the PDE semiotic‑differential
investigator (# app.py) when a query profile becomes suspicious.

Pipeline (per query, O(K log K)):

    read |Ci| tokens
      → supertrace S = Σ (-1)^i |C_i|
      → entropy H = -α · p · log(p),  p = |S| / K
      → symmetric inverse score vs reference (merge sort)
      → asymmetric lower‑bound entropy (half‑window)
      → emotional entropy from align‑emotions.py (hormone vector)
      → strike count (symmetric deviation / H‑vs‑H_em / asym‑lb)
      → flag if strikes ≥ threshold
      → escalate to # app.py (PDE document_vector) for scalar report

Query profiles store:
    symmetric score, supertrace, entropy, emotional entropy,
    asymmetric lower bound, strikes, flag, deep‑result.
"""

import math
import heapq
import hashlib
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

# ============================================================
#  Constants
# ============================================================
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)             # ≈ 2.362
ALPHA_USER = 0.3628
A_STEP = ALPHA / ALPHA_USER        # ≈ 6.511  (finite‑derivative step)
DENSITY = 6.0 / (PI * PI)          # ≈ 0.6079 square‑free density

# ============================================================
#  Merge sort + inversion count  (O(K log K))
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
    temp = [0] * n
    return _merge_sort_count(list(arr), temp, 0, n - 1)


def inverse_score(a, b):
    """Symmetric inverse score ∈ [0,1]. O(K log K)."""
    if len(a) != len(b) or len(a) < 2:
        return 0.5
    pairs = sorted(zip(a, b), key=lambda p: p[0])
    b_sorted = [p[1] for p in pairs]
    inv = inversion_count(b_sorted)
    K = len(a)
    max_inv = K * (K - 1) // 2
    return inv / max_inv if max_inv > 0 else 0.0


# ============================================================
#  Supertrace entropy (semiotic‑differential style)
# ============================================================
def supertrace(c):
    """S = Σ (-1)^i |c_i|  —  O(K)."""
    S = 0.0
    for i, v in enumerate(c):
        S += v if (i % 2 == 0) else -v
    return S


def entropy_from_supertrace(S, K, alpha=ALPHA):
    """H = -α · p · log(p),  p = |S|/K, clamped to [0, 1]."""
    if K <= 0 or S == 0.0:
        return 0.0
    p = abs(S) / K
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -alpha * p * math.log(p)


# ============================================================
#  Emotional entropy (from align-emotions.py hormone vectors)
# ============================================================
def emotional_entropy(hormone_vector, alpha=ALPHA):
    """
    Shannon entropy of a normalised hormone vector.
    Higher entropy = more "activated" emotional state
    (used to boost the emotional score for high‑entropy emotions).
    """
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
#  Emotional memory (lightweight port of align‑emotions.py)
# ============================================================
class EmotionalMemory:
    """
    Stores (label, intensity, hormone_vector) triples.
    Query returns the top‑k closest hormones by inverse score
    of their |C_i| embeddings.
    """

    def __init__(self, K_embed=32):
        self.K = K_embed
        self.items = []   # list of dicts
        self.mu = self._mobius_sieve(self.K)

    @staticmethod
    def _mobius_sieve(K):
        mu = [0] * (K + 1)
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

    def _embed(self, hormone_vector):
        """Deterministic |C_i| embedding from a hormone vector."""
        h = np.asarray(hormone_vector, dtype=float)
        seed = int(hashlib.md5(h.tobytes()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        raw = rng.standard_normal(self.K) * 0.3 + h.mean()
        # Möbius filter: keep square‑free indices only
        emb = np.zeros(self.K)
        for i in range(self.K):
            if self.mu[i + 1] != 0:
                emb[i] = abs(raw[i])
        return emb

    def add(self, label, intensity, hormone_vector):
        self.items.append(dict(
            label=label,
            intensity=intensity,
            hormones=np.asarray(hormone_vector, dtype=float),
            embedding=self._embed(hormone_vector),
            entropy=emotional_entropy(hormone_vector),
        ))

    def query(self, hormone_vector, top_k=3):
        q_emb = self._embed(hormone_vector)
        scored = []
        for it in self.items:
            s = inverse_score(q_emb.tolist(), it['embedding'].tolist())
            scored.append((s, it))
        scored.sort(key=lambda x: x[0])
        return scored[:top_k]

    def build_default(self):
        """Populate with the 5 basic emotions from align‑emotions.py."""
        neutral = np.array([0.5] * 6)
        base = {
            'joy':     [0.8, 0.9, 0.8, 0.6, 0.5, 0.7],
            'sadness': [0.2, 0.3, 0.2, 0.5, 0.4, 0.3],
            'fear':    [0.9, 0.4, 0.3, 0.5, 0.9, 0.2],
            'anger':   [0.8, 0.3, 0.4, 0.5, 0.8, 0.1],
            'calm':    [0.4, 0.6, 0.7, 0.7, 0.3, 0.6],
        }
        for label, prof in base.items():
            for tier in (0.2, 0.5, 0.8):
                scaled = neutral + tier * (np.array(prof) - neutral)
                scaled = np.clip(scaled, 0.0, 1.0)
                self.add(label, int(tier * 10), scaled)


# ============================================================
#  Query profile
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


# ============================================================
#  Quick reader + strike‑flag analyzer
# ============================================================
class StrikeFlagReader:
    """
    Reads |Ci| arrays in O(K log K), computes entropies,
    accumulates strikes, and escalates to the PDE investigator
    when a profile is flagged.
    """

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
        self.emotions = EmotionalMemory(K_embed=32)
        self.emotions.build_default()

    # ---------- reference ----------
    def set_reference(self, ref_signal):
        """Reference |Ci| array (symmetric anchor)."""
        self.reference = np.asarray(ref_signal, dtype=float)

    # ---------- read one query ----------
    def read(self,
             query_id: str,
             k_sum,
             emotional_vector: Optional[np.ndarray] = None,
             symmetric_only: bool = True) -> QueryProfile:
        """
        Read one |Ci| array.

        symmetric_only=True  → symmetric inverse‑score search only (fast)
        symmetric_only=False → also queries align‑emotions emotional layer
        """
        c = np.asarray(k_sum, dtype=float)
        K = c.size

        # 1. Supertrace entropy
        S = supertrace(c)
        H = entropy_from_supertrace(S, K)

        # 2. Emotional layer (prompted when symmetric_only=False)
        H_em = 0.0
        top_emotions = []
        if not symmetric_only and emotional_vector is not None:
            H_em = emotional_entropy(emotional_vector) * self.emotional_boost
            top_emotions = [
                (it['label'], it['intensity'], float(s))
                for s, it in self.emotions.query(emotional_vector, top_k=3)
            ]

        # 3. Symmetric inverse score vs reference
        if self.reference is not None and self.reference.size == K:
            sym = inverse_score(c.tolist(), self.reference.tolist())
        else:
            sym = 0.5

        # 4. Asymmetric lower bound (half window)
        half = K // 2
        if half >= 2:
            S_h = supertrace(c[:half])
            asym_lb = entropy_from_supertrace(S_h, half)
        else:
            asym_lb = 0.0

        # 5. Strike accumulation
        strikes = 0
        if abs(sym - 0.5) > self.entropy_tolerance:
            strikes += 1                                  # symmetric anomaly
        if H_em > 0.0 and abs(H - H_em) > self.entropy_tolerance:
            strikes += 1                                  # H vs H_em anomaly
        if asym_lb > H + self.entropy_tolerance:
            strikes += 1                                  # asymmetric leak

        prof = QueryProfile(
            query_id=query_id,
            k_sum=c.tolist(),
            symmetric_score=sym,
            supertrace_S=S,
            entropy_H=H,
            emotional_entropy=H_em,
            asymmetric_lb=asym_lb,
            strikes=strikes,
            top_emotions=top_emotions,
        )

        # 6. Flag → escalate
        if strikes >= self.strike_threshold:
            prof.flagged = True
            self.flags.append(prof)
            prof.deep_result = self._pde_investigate(prof)

        self.profiles.append(prof)
        return prof

    # ---------- escalation: PDE semiotic‑differential ----------
    def _pde_investigate(self, prof: QueryProfile) -> Dict:
        """
        Deeper investigation on the flagged scalar using the
        Semiotic‑differential PDE (from # app.py).
        Returns a scalar report.
        """
        K = self.K_pde
        c = np.asarray(prof.k_sum, dtype=float)
        n = c.size

        # Deterministic per‑query weights
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

        # Scalar summary
        final_S = supertrace(h)
        final_H = entropy_from_supertrace(final_S, K)
        return {
            'query_id': prof.query_id,
            'h_norm': float(np.linalg.norm(h)),
            'h_mean': float(np.mean(h)),
            'h_std': float(np.std(h)),
            'pde_S': final_S,
            'pde_H': final_H,
            'strikes': prof.strikes,
            'flagged': True,
        }

    # ---------- batch read ----------
    def read_batch(self, batch: List[Tuple[str, List[float], Optional[np.ndarray]]],
                   symmetric_only: bool = True) -> List[QueryProfile]:
        out = []
        for qid, c, emo in batch:
            out.append(self.read(qid, c, emotional_vector=emo,
                                 symmetric_only=symmetric_only))
        return out

    # ---------- summary ----------
    def summary(self) -> Dict:
        n = len(self.profiles)
        flagged = [p for p in self.profiles if p.flagged]
        return {
            'n_queries': n,
            'n_flagged': len(flagged),
            'flagged_ids': [p.query_id for p in flagged],
            'mean_strikes': (sum(p.strikes for p in self.profiles) / n) if n else 0.0,
            'mean_H': (sum(p.entropy_H for p in self.profiles) / n) if n else 0.0,
            'mean_sym': (sum(p.symmetric_score for p in self.profiles) / n) if n else 0.0,
            'mean_H_em': (sum(p.emotional_entropy for p in self.profiles) / n) if n else 0.0,
            'mean_asym_lb': (sum(p.asymmetric_lb for p in self.profiles) / n) if n else 0.0,
        }


# ============================================================
#  Convenience: token → |Ci| reader
# ============================================================
def tokens_to_k_sum(tokens: List[int], K: int) -> np.ndarray:
    """
    Convert an integer token stream into a |Ci| array of length K.
    Values are hashed deterministically so equal tokens → equal |Ci|.
    """
    c = np.zeros(K)
    for i, t in enumerate(tokens[:K]):
        h = int(hashlib.md5(str(t).encode()).hexdigest()[:8], 16)
        c[i] = ((h % 10_000) / 10_000.0) + 0.1 * math.sin(i * 0.7)
    return np.abs(c)


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 68)
    print("Quick reader / analyzer with strike flagging")
    print("=" * 68)

    K = 512
    reader = StrikeFlagReader(strike_threshold=2,
                              entropy_tolerance=0.12,
                              emotional_boost=1.4)

    # Reference signal (a sine‑modulated |Ci|)
    x = np.arange(K)
    ref = np.abs(np.sin(x * 0.05) + 0.3 * np.cos(x * 0.13))
    reader.set_reference(ref)

    # Build a small batch of queries
    rng = np.random.default_rng(2024)
    batch = []
    for i in range(12):
        # Vary the "shape" of |Ci| to produce a range of entropy
        if i < 4:
            c = ref * (1.0 + 0.02 * rng.standard_normal(K))       # ~reference
            emo = np.array([0.5, 0.6, 0.7, 0.5, 0.4, 0.6])       # calm
        elif i < 8:
            c = np.abs(np.sin(x * (0.02 + 0.01 * i)))            # low entropy
            emo = np.array([0.3, 0.4, 0.3, 0.5, 0.5, 0.4])       # sad
        else:
            c = np.abs(rng.standard_normal(K)) * 3.0             # noisy
            emo = np.array([0.9, 0.3, 0.4, 0.5, 0.9, 0.2])       # fear
        batch.append((f"q{i:02d}", c, emo))

    # --- symmetric-only pass ---
    print("\n--- Pass 1: symmetric search only (fast, O(K log K)) ---")
    reader.read_batch(batch, symmetric_only=True)
    for p in reader.profiles:
        print(f"  {p.query_id}  sym={p.symmetric_score:.4f}  "
              f"S={p.supertrace_S:+8.4f}  H={p.entropy_H:.4f}  "
              f"strikes={p.strikes}  flagged={p.flagged}")

    # --- prompted pass (with emotional layer) ---
    print("\n--- Pass 2: entropy prompted + align‑emotions ---")
    reader2 = StrikeFlagReader(strike_threshold=2,
                               entropy_tolerance=0.12,
                               emotional_boost=1.4)
    reader2.set_reference(ref)
    reader2.read_batch(batch, symmetric_only=False)
    for p in reader2.profiles:
        emo_str = ", ".join(f"{lbl}({t})={s:.2f}"
                            for lbl, t, s in p.top_emotions) or "—"
        print(f"  {p.query_id}  sym={p.symmetric_score:.4f}  "
              f"H={p.entropy_H:.4f}  H_em={p.emotional_entropy:.4f}  "
              f"lb={p.asymmetric_lb:.4f}  strikes={p.strikes}  "
              f"flag={p.flagged}")
        if p.top_emotions:
            print(f"        ↳ top emotions: {emo_str}")

    # --- flags & deep results ---
    print("\n--- Flagged profiles (escalated to PDE investigator) ---")
    for p in reader2.flags:
        dr = p.deep_result or {}
        print(f"  {p.query_id}  strikes={p.strikes}  "
              f"pde_S={dr.get('pde_S', 0):+.4f}  "
              f"pde_H={dr.get('pde_H', 0):.4f}  "
              f"|h|={dr.get('h_norm', 0):.4f}  "
              f"h_mean={dr.get('h_mean', 0):+.4f}  "
              f"h_std={dr.get('h_std', 0):.4f}")

    # --- summary ---
    print("\n--- Summary (pass 2) ---")
    s = reader2.summary()
    for k, v in s.items():
        if isinstance(v, float):
            print(f"  {k:>15s}: {v:.4f}")
        else:
            print(f"  {k:>15s}: {v}")

    # --- token reader example ---
    print("\n--- Token → |Ci| reader ---")
    tokens = [42, 137, 5, 999, 0, 101, 202, 303]
    c = tokens_to_k_sum(tokens, K=64)
    prof = reader2.read("token_demo", c,
                        emotional_vector=np.array([0.2, 0.3, 0.2, 0.5, 0.4, 0.3]),
                        symmetric_only=False)
    print(f"  token_demo: sym={prof.symmetric_score:.4f}  "
          f"H={prof.entropy_H:.4f}  H_em={prof.emotional_entropy:.4f}  "
          f"strikes={prof.strikes}  flagged={prof.flagged}")


if __name__ == "__main__":
    demo()