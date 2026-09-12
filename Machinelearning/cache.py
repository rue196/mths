#!/usr/bin/env python3
"""
ssd_reader_analyzer.py

SSD‑backed reader + analyzer for large |Ci| arrays.

Storage model
-------------
    SSD   (memory‑mapped file)  :  real part   re[i] = |Ci|
    RAM   (LRU chunk cache)     :  imaginary part
                                    im[i] = -d(re)/dn / a
                                    a = 1/(π−e)/0.3628 ≈ 6.511

    Analytic envelope  env[i] = sqrt(re[i]² + im[i]²)
    Supertrace uses env, so the imaginary cache *directly*
    influences entropy and the strike decisions.

The imaginary part is only ever materialised in the cache:
it is recomputed on a cache miss (finite‑step derivative from
the SSD‑backed real part) and evicted when the LRU is full.

Everything else — supertrace, entropy, strike flagging,
escalation to the PDE investigator — is unchanged from the
in‑memory `quick‑reader‑analyzer.py`, but every read of a
sample now goes through the SSD → imaginary‑cache path.
"""

import math
import os
import hashlib
import numpy as np
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

# ============================================================
#  Constants
# ============================================================
PI  = math.pi
E   = math.e
ALPHA       = 1.0 / (PI - E)          # ≈ 2.362
ALPHA_USER  = 0.3628
A_STEP      = ALPHA / ALPHA_USER      # ≈ 6.511  finite‑derivative step
DENSITY     = 6.0 / (PI * PI)         # ≈ 0.6079 square‑free density

# ============================================================
#  SSD‑backed |Ci| store  with imaginary‑part cache
# ============================================================
class SSDCachedKSum:
    """
    Memory‑mapped real data on SSD, imaginary part in an LRU.

    Chunking
    --------
    The real array is divided into chunks of `chunk_size` samples.
    The imaginary part is computed per chunk and stored in `imag_cache`
    keyed by chunk index.  Eviction is LRU with capacity `cache_chunks`.
    """

    def __init__(self, path: str, K: int,
                 chunk_size: int = 4096,
                 cache_chunks: int = 16,
                 mode: str = 'w+',
                 initial_data: Optional[np.ndarray] = None):
        self.path         = path
        self.K            = K
        self.chunk_size   = chunk_size
        self.cache_chunks = cache_chunks

        # --- SSD‑backed real array (memory mapped) ---
        self.real_mmap = np.memmap(path, dtype=np.float64, mode=mode,
                                   shape=(K,))
        if initial_data is not None:
            if len(initial_data) != K:
                raise ValueError("initial_data length != K")
            self.real_mmap[:] = np.asarray(initial_data, dtype=np.float64)
            self.real_mmap.flush()

        # --- RAM cache for imaginary part ---
        self.imag_cache: "OrderedDict[int, np.ndarray]" = OrderedDict()

        # --- Stats ---
        self.stats = dict(sample_reads=0, cache_hits=0,
                          cache_misses=0, evictions=0)

    # ---------- chunk helpers ----------
    def _chunk_bounds(self, cid: int) -> Tuple[int, int]:
        start = cid * self.chunk_size
        end   = min(start + self.chunk_size, self.K)
        return start, end

    def _chunk_of(self, i: int) -> int:
        return i // self.chunk_size

    # ---------- imaginary‑chunk computation ----------
    def _compute_imag_chunk(self, cid: int) -> np.ndarray:
        """
        im[i] = -d(re)/dn / a  via forward finite difference.
        Needs one extra sample past the chunk end to close the last
        forward difference; if at end of file, falls back to a
        backward difference.
        """
        start, end = self._chunk_bounds(cid)
        length = end - start
        if length <= 0:
            return np.zeros(0)

        extra_end = min(end + 1, self.K)
        raw = np.asarray(self.real_mmap[start:extra_end], dtype=np.float64)

        imag = np.zeros(length, dtype=np.float64)
        if raw.size >= 2:
            deriv = (raw[1:] - raw[:-1]) / A_STEP
            n_copy = min(deriv.size, length)
            imag[:n_copy] = -deriv[:n_copy]
            # if forward diff didn't cover the last sample of the chunk:
            if n_copy < length:
                imag[length - 1] = -(raw[-1] - raw[-2]) / A_STEP
        return imag

    def imag_chunk(self, cid: int) -> np.ndarray:
        """Return imaginary chunk (cache hit or compute+insert+evict)."""
        if cid in self.imag_cache:
            self.imag_cache.move_to_end(cid)
            self.stats['cache_hits'] += 1
            return self.imag_cache[cid]

        self.stats['cache_misses'] += 1
        imag = self._compute_imag_chunk(cid)
        self.imag_cache[cid] = imag
        while len(self.imag_cache) > self.cache_chunks:
            self.imag_cache.popitem(last=False)
            self.stats['evictions'] += 1
        return imag

    # ---------- point access ----------
    def real(self, i: int) -> float:
        self.stats['sample_reads'] += 1
        return float(self.real_mmap[i])

    def imag(self, i: int) -> float:
        cid = self._chunk_of(i)
        chunk = self.imag_chunk(cid)
        return float(chunk[i - cid * self.chunk_size])

    def envelope(self, i: int) -> float:
        re = self.real(i)
        im = self.imag(i)
        return math.hypot(re, im)

    # ---------- streaming helpers ----------
    def iter_chunks(self):
        """Yield (start, real_chunk, imag_chunk) for each chunk."""
        n_chunks = (self.K + self.chunk_size - 1) // self.chunk_size
        for cid in range(n_chunks):
            start, end = self._chunk_bounds(cid)
            real = np.asarray(self.real_mmap[start:end], dtype=np.float64)
            imag = self.imag_chunk(cid)
            yield start, real, imag

    def real_array(self) -> np.ndarray:
        """Materialise the full real array (still memmapped, OS pages)."""
        return np.asarray(self.real_mmap[:], dtype=np.float64)

    def analytic_envelope(self) -> np.ndarray:
        """Full analytic envelope, one chunk at a time."""
        out = np.zeros(self.K, dtype=np.float64)
        for start, real, imag in self.iter_chunks():
            n = min(real.size, imag.size)
            out[start:start + n] = np.hypot(real[:n], imag[:n])
        return out

    def flush(self):
        self.real_mmap.flush()

    def cache_report(self) -> Dict:
        return dict(self.stats,
                    cache_size=len(self.imag_cache),
                    capacity=self.cache_chunks,
                    chunk_size=self.chunk_size)


# ============================================================
#  Merge sort + inversion count (RAM, small reference side)
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
    if len(a) != len(b) or len(a) < 2:
        return 0.5
    pairs = sorted(zip(a, b), key=lambda p: p[0])
    b_sorted = [p[1] for p in pairs]
    inv = inversion_count(b_sorted)
    K = len(a)
    max_inv = K * (K - 1) // 2
    return inv / max_inv if max_inv > 0 else 0.0


# ============================================================
#  Entropy helpers
# ============================================================
def entropy_from_supertrace(S, K, alpha=ALPHA):
    if K <= 0 or S == 0.0:
        return 0.0
    p = abs(S) / K
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -alpha * p * math.log(p)


def emotional_entropy(hormone_vector, alpha=ALPHA):
    v = np.asarray(hormone_vector, dtype=float)
    if v.size == 0 or v.sum() <= 0.0:
        return 0.0
    p = v / v.sum()
    return alpha * sum(-pi * math.log(pi) for pi in p if pi > 1e-12)


# ============================================================
#  Emotional memory (light port of align‑emotions.py)
# ============================================================
class EmotionalMemory:
    def __init__(self, K_embed=32):
        self.K = K_embed
        self.items = []
        self.mu = self._mobius_sieve(self.K)

    @staticmethod
    def _mobius_sieve(K):
        mu = [0] * (K + 1); mu[1] = 1
        primes, is_comp = [], [False] * (K + 1)
        for i in range(2, K + 1):
            if not is_comp[i]:
                primes.append(i); mu[i] = -1
            for p in primes:
                if i * p > K: break
                is_comp[i * p] = True
                if i % p == 0: mu[i * p] = 0; break
                else:          mu[i * p] = -mu[i]
        return mu

    def _embed(self, h):
        h = np.asarray(h, dtype=float)
        seed = int(hashlib.md5(h.tobytes()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        raw = rng.standard_normal(self.K) * 0.3 + h.mean()
        emb = np.zeros(self.K)
        for i in range(self.K):
            if self.mu[i + 1] != 0:
                emb[i] = abs(raw[i])
        return emb

    def add(self, label, intensity, h):
        self.items.append(dict(label=label, intensity=intensity,
                               hormones=np.asarray(h, dtype=float),
                               embedding=self._embed(h),
                               entropy=emotional_entropy(h)))

    def query(self, h, top_k=3):
        q = self._embed(h)
        scored = [(inverse_score(q.tolist(), it['embedding'].tolist()), it)
                  for it in self.items]
        scored.sort(key=lambda x: x[0])
        return scored[:top_k]

    def build_default(self):
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
    K: int
    symmetric_score: float = 0.5
    supertrace_S: float = 0.0
    entropy_H: float = 0.0
    emotional_entropy: float = 0.0
    asymmetric_lb: float = 0.0
    strikes: int = 0
    flagged: bool = False
    deep_result: Optional[Dict] = None
    top_emotions: List[Tuple[str, int, float]] = field(default_factory=list)
    cache_report: Dict = field(default_factory=dict)


# ============================================================
#  SSD reader with strike flagging
# ============================================================
class SSDStrikeFlagReader:
    """
    Reads |Ci| arrays from SSD, caches the imaginary part,
    accumulates strikes, and escalates flagged queries.
    """

    def __init__(self,
                 scratch_dir: str = ".",
                 chunk_size: int = 4096,
                 cache_chunks: int = 16,
                 strike_threshold: int = 2,
                 entropy_tolerance: float = 0.12,
                 emotional_boost: float = 1.4,
                 K_pde: int = 128):
        self.scratch_dir     = scratch_dir
        self.chunk_size      = chunk_size
        self.cache_chunks    = cache_chunks
        self.strike_threshold = strike_threshold
        self.entropy_tolerance = entropy_tolerance
        self.emotional_boost = emotional_boost
        self.K_pde           = K_pde

        self.profiles: List[QueryProfile] = []
        self.flags: deque = deque()
        self.reference: Optional[np.ndarray] = None
        self.emotions = EmotionalMemory(K_embed=32)
        self.emotions.build_default()

        os.makedirs(scratch_dir, exist_ok=True)

    def set_reference(self, ref_signal):
        self.reference = np.asarray(ref_signal, dtype=float)

    # ---------- supertrace over analytic envelope, streaming ----------
    def _stream_supertrace(self, store: SSDCachedKSum) -> float:
        """Σ (-1)^i · sqrt(re² + im²), computed chunk by chunk."""
        S = 0.0
        for start, real, imag in store.iter_chunks():
            n = min(real.size, imag.size)
            sign = 1.0 if (start % 2 == 0) else -1.0
            # alternate sign inside the chunk
            idx = np.arange(start, start + n)
            sgn = np.where(idx % 2 == 0, 1.0, -1.0)
            S += float(np.sum(sgn * np.hypot(real[:n], imag[:n])))
        return S

    def _asym_lb(self, store: SSDCachedKSum, fraction: float = 0.5) -> float:
        """Lower‑bound entropy from the first `fraction` of the array."""
        half = max(1, int(store.K * fraction))
        S = 0.0
        for start, real, imag in store.iter_chunks():
            end = start + real.size
            if start >= half:
                break
            take = min(real.size, half - start)
            idx = np.arange(start, start + take)
            sgn = np.where(idx % 2 == 0, 1.0, -1.0)
            S += float(np.sum(sgn * np.hypot(real[:take], imag[:take])))
        return entropy_from_supertrace(S, half)

    # ---------- read one query ----------
    def read(self,
             query_id: str,
             k_sum,
             emotional_vector: Optional[np.ndarray] = None,
             symmetric_only: bool = True,
             reuse_store: bool = True) -> QueryProfile:
        """
        Read one |Ci| array from SSD (via a freshly created memmap).
        The imaginary part is cached in the store's LRU, and the whole
        pipeline runs on the SSD‑backed representation.
        """
        arr = np.asarray(k_sum, dtype=np.float64)
        K = arr.size

        # --- SSD backing file per query (simple, safe) ---
        path = os.path.join(self.scratch_dir, f"k_{query_id}.mm")
        store = SSDCachedKSum(path, K,
                              chunk_size=self.chunk_size,
                              cache_chunks=self.cache_chunks,
                              mode='w+',
                              initial_data=arr)

        # 1. supertrace / entropy on the analytic envelope
        S = self._stream_supertrace(store)
        H = entropy_from_supertrace(S, K)

        # 2. emotional layer
        H_em = 0.0
        top_emotions = []
        if not symmetric_only and emotional_vector is not None:
            H_em = emotional_entropy(emotional_vector) * self.emotional_boost
            top_emotions = [(it['label'], it['intensity'], float(s))
                            for s, it in self.emotions.query(emotional_vector, top_k=3)]

        # 3. symmetric inverse score (real part vs reference)
        if self.reference is not None and self.reference.size == K:
            sym = inverse_score(store.real_array().tolist(),
                                self.reference.tolist())
        else:
            sym = 0.5

        # 4. asymmetric lower‑bound entropy (first half)
        asym_lb = self._asym_lb(store, fraction=0.5)

        # 5. strike accumulation
        strikes = 0
        if abs(sym - 0.5) > self.entropy_tolerance:
            strikes += 1
        if H_em > 0.0 and abs(H - H_em) > self.entropy_tolerance:
            strikes += 1
        if asym_lb > H + self.entropy_tolerance:
            strikes += 1

        prof = QueryProfile(
            query_id=query_id,
            K=K,
            symmetric_score=sym,
            supertrace_S=S,
            entropy_H=H,
            emotional_entropy=H_em,
            asymmetric_lb=asym_lb,
            strikes=strikes,
            top_emotions=top_emotions,
            cache_report=store.cache_report(),
        )

        # 6. flag → escalate
        if strikes >= self.strike_threshold:
            prof.flagged = True
            self.flags.append(prof)
            prof.deep_result = self._pde_investigate(prof, store)

        self.profiles.append(prof)
        store.flush()
        return prof

    # ---------- escalation ----------
    def _pde_investigate(self, prof: QueryProfile, store: SSDCachedKSum) -> Dict:
        """
        Deeper investigation on the flagged scalar using the
        Semiotic‑differential PDE.  Reads the analytic envelope
        back from SSD (via cache) so the investigation operates
        on the same imaginary‑part view.
        """
        K_pde = self.K_pde
        env = store.analytic_envelope()
        n = env.size

        seed = int(hashlib.md5(prof.query_id.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        W_in = rng.standard_normal((K_pde, min(n, K_pde))) * 0.01
        anchors = np.arange(0, K_pde, max(1, K_pde // 20))

        DT, ALPHA_PDE, EPS = 0.1, 0.1, 0.01
        h = np.zeros(K_pde)
        for j in range(min(n, K_pde)):
            h += W_in[:, j] * env[j]
            diff = np.zeros_like(h)
            diff[1:-1] = h[:-2] + h[2:] - 2.0 * h[1:-1]
            h += DT * (diff + ALPHA_PDE * np.sum(h[anchors]) - EPS * h)

        final_S = float(np.sum(np.where(np.arange(K_pde) % 2 == 0,
                                        1.0, -1.0) * np.abs(h)))
        final_H = entropy_from_supertrace(final_S, K_pde)
        return {
            'query_id': prof.query_id,
            'h_norm': float(np.linalg.norm(h)),
            'h_mean': float(np.mean(h)),
            'h_std':  float(np.std(h)),
            'pde_S':  final_S,
            'pde_H':  final_H,
            'strikes': prof.strikes,
            'flagged': True,
            'cache_hits':   prof.cache_report.get('cache_hits', 0),
            'cache_misses': prof.cache_report.get('cache_misses', 0),
            'evictions':    prof.cache_report.get('evictions', 0),
        }

    # ---------- batch ----------
    def read_batch(self, batch, symmetric_only=True):
        return [self.read(qid, c, emo, symmetric_only=symmetric_only)
                for qid, c, emo in batch]

    # ---------- summary ----------
    def summary(self) -> Dict:
        n = len(self.profiles)
        flagged = [p for p in self.profiles if p.flagged]
        tot_hits = sum(p.cache_report.get('cache_hits', 0) for p in self.profiles)
        tot_miss = sum(p.cache_report.get('cache_misses', 0) for p in self.profiles)
        tot_evic = sum(p.cache_report.get('evictions', 0) for p in self.profiles)
        return {
            'n_queries':   n,
            'n_flagged':   len(flagged),
            'flagged_ids': [p.query_id for p in flagged],
            'mean_strikes': (sum(p.strikes for p in self.profiles) / n) if n else 0.0,
            'mean_H':       (sum(p.entropy_H for p in self.profiles) / n) if n else 0.0,
            'mean_sym':     (sum(p.symmetric_score for p in self.profiles) / n) if n else 0.0,
            'mean_H_em':    (sum(p.emotional_entropy for p in self.profiles) / n) if n else 0.0,
            'mean_asym_lb': (sum(p.asymmetric_lb for p in self.profiles) / n) if n else 0.0,
            'cache_hits':   tot_hits,
            'cache_misses': tot_miss,
            'cache_evictions': tot_evic,
            'cache_hit_rate': tot_hits / (tot_hits + tot_miss) if (tot_hits + tot_miss) else 0.0,
        }


# ============================================================
#  Token → |Ci|  (same as before, kept for convenience)
# ============================================================
def tokens_to_k_sum(tokens: List[int], K: int) -> np.ndarray:
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
    print("SSD reader / analyzer  —  imaginary‑part cache")
    print("=" * 68)

    K = 4096          # 4096 samples; small enough for a short demo
    scratch = "ssd_scratch"
    reader = SSDStrikeFlagReader(
        scratch_dir=scratch,
        chunk_size=512,           # 8 chunks per query
        cache_chunks=3,           # deliberately small so we see eviction
        strike_threshold=2,
        entropy_tolerance=0.12,
        emotional_boost=1.4,
    )

    # Reference on SSD (small, but same interface)
    x = np.arange(K)
    ref = np.abs(np.sin(x * 0.02) + 0.3 * np.cos(x * 0.05))
    reader.set_reference(ref)

    rng = np.random.default_rng(7)
    batch = []
    for i in range(6):
        if i < 2:
            c = ref * (1.0 + 0.02 * rng.standard_normal(K))
            emo = np.array([0.5, 0.6, 0.7, 0.5, 0.4, 0.6])    # calm
        elif i < 4:
            c = np.abs(np.sin(x * (0.01 + 0.005 * i)))
            emo = np.array([0.3, 0.4, 0.3, 0.5, 0.5, 0.4])    # sad
        else:
            c = np.abs(rng.standard_normal(K)) * 3.0
            emo = np.array([0.9, 0.3, 0.4, 0.5, 0.9, 0.2])    # fear
        batch.append((f"q{i:02d}", c, emo))

    # --- Pass 1: symmetric only ---
    print("\n--- Pass 1: symmetric SSD search (imaginary cache active) ---")
    reader.read_batch(batch, symmetric_only=True)
    for p in reader.profiles:
        cr = p.cache_report
        print(f"  {p.query_id}  sym={p.symmetric_score:.4f}  "
              f"S={p.supertrace_S:+8.4f}  H={p.entropy_H:.4f}  "
              f"strikes={p.strikes}  flag={p.flagged}  "
              f"cache(h/m/e)={cr['cache_hits']}/{cr['cache_misses']}/{cr['evictions']}")

    # --- Pass 2: with emotional layer ---
    print("\n--- Pass 2: entropy prompted + align‑emotions ---")
    reader2 = SSDStrikeFlagReader(
        scratch_dir=scratch,
        chunk_size=512,
        cache_chunks=3,
        strike_threshold=2,
        entropy_tolerance=0.12,
        emotional_boost=1.4,
    )
    reader2.set_reference(ref)
    reader2.read_batch(batch, symmetric_only=False)
    for p in reader2.profiles:
        emo_str = ", ".join(f"{l}({t})={s:.2f}" for l, t, s in p.top_emotions) or "—"
        cr = p.cache_report
        print(f"  {p.query_id}  sym={p.symmetric_score:.4f}  "
              f"H={p.entropy_H:.4f}  H_em={p.emotional_entropy:.4f}  "
              f"lb={p.asymmetric_lb:.4f}  strikes={p.strikes}  "
              f"flag={p.flagged}  "
              f"cache(h/m/e)={cr['cache_hits']}/{cr['cache_misses']}/{cr['evictions']}")
        if p.top_emotions:
            print(f"        ↳ {emo_str}")

    # --- Flagged profiles ---
    print("\n--- Flagged profiles (escalated, SSD‑backed) ---")
    for p in reader2.flags:
        d = p.deep_result or {}
        print(f"  {p.query_id}  strikes={p.strikes}  "
              f"pde_S={d.get('pde_S', 0):+.4f}  "
              f"pde_H={d.get('pde_H', 0):.4f}  "
              f"|h|={d.get('h_norm', 0):.4f}  "
              f"h_std={d.get('h_std', 0):.4f}  "
              f"hits={d.get('cache_hits', 0)}  "
              f"miss={d.get('cache_misses', 0)}  "
              f"evict={d.get('evictions', 0)}")

    # --- Summary ---
    print("\n--- Summary (pass 2) ---")
    s = reader2.summary()
    for k, v in s.items():
        if isinstance(v, float):
            print(f"  {k:>15s}: {v:.4f}")
        else:
            print(f"  {k:>15s}: {v}")

    # --- Token reader ---
    print("\n--- Token → |Ci| → SSD reader ---")
    tokens = [42, 137, 5, 999, 0, 101, 202, 303, 404, 505]
    c = tokens_to_k_sum(tokens, K=1024)
    reader3 = SSDStrikeFlagReader(
        scratch_dir=scratch, chunk_size=256, cache_chunks=4,
        strike_threshold=2, entropy_tolerance=0.12,
    )
    reader3.set_reference(np.abs(np.sin(np.arange(1024) * 0.01)))
    prof = reader3.read("token_demo", c,
                        emotional_vector=np.array([0.2, 0.3, 0.2, 0.5, 0.4, 0.3]),
                        symmetric_only=False)
    print(f"  token_demo: sym={prof.symmetric_score:.4f}  "
          f"H={prof.entropy_H:.4f}  H_em={prof.emotional_entropy:.4f}  "
          f"strikes={prof.strikes}  flagged={prof.flagged}")
    print(f"  cache report: {prof.cache_report}")


if __name__ == "__main__":
    demo()