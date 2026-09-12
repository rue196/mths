#!/usr/bin/env python3
"""
ssd_multilayer_reader.py

SSD‑backed multi‑layer reader + analyzer.

The SSD holds *one* primary layer (`real` = the |Ci| array).
All other layers are **derived on demand** and cached in RAM,
so a user can query hidden layers without ever writing them
to disk.

Layers
------
    real      stored on SSD  (memmap)
    imag      derived        im[i] = -d(re)/dn / a,  a = 1/(π−e)/0.3628
    envelope  derived        env[i] = sqrt(re² + im²)
    hex       derived        per‑chunk SHA‑256 fingerprint from hex‑ok gate
    meta      stored (opt.)  sidecar file (labels, timestamps, …)

Everything the user asks for goes through `LayerReader`:

    read(layer, start, end)          → numpy slice
    ui_view(layer, page, size)       → paginated view (front‑end)
    stream(layer)                    → generator of (start, chunk)
    flagged_report(...)              → strike analysis on envelope

The hex‑ok gate from `hex-ok.py` is used both as a *reader* of the
derived hex layer and as the *key* for cached chunk fingerprints.
"""

import math
import os
import hashlib
import struct
import numpy as np
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Tuple

# ============================================================
#  Constants
# ============================================================
PI        = math.pi
E         = math.e
ALPHA     = 1.0 / (PI - E)             # ≈ 2.362
ALPHA_USER = 0.3628
A_STEP    = ALPHA / ALPHA_USER         # ≈ 6.511  finite‑derivative step
DENSITY   = 6.0 / (PI * PI)            # ≈ 0.6079 square‑free density

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
#  Minimal inline of the hex‑ok gate (numeric version)
# ============================================================
def _elliptic_permutation(K, w1=PI, w2=E):
    delta = w1 - w2
    angles = [(i * delta) % (2 * PI) for i in range(K)]
    return sorted(range(K), key=lambda i: angles[i])


def _conv_exp_kernel(signal, alpha=ALPHA):
    K = len(signal)
    if K == 0:
        return np.zeros(0)
    lam = math.exp(-alpha)
    f = np.zeros(K); f[0] = signal[0]
    for i in range(1, K):
        f[i] = signal[i] + lam * f[i - 1]
    b = np.zeros(K); b[K - 1] = signal[K - 1]
    for i in range(K - 2, -1, -1):
        b[i] = signal[i] + lam * b[i + 1]
    den = 1.0 - lam * lam
    norm = 1.0 - math.exp(-alpha * (PI + E))
    conv_exp = (f + b - signal) / den
    return (1.0 - conv_exp) / norm


def _supertrace(signal):
    S = 0.0
    for i, v in enumerate(signal):
        S += v if (i % 2 == 0) else -v
    return S


def _entropy(S, N, alpha=ALPHA):
    if N <= 0 or S == 0.0:
        return 0.0
    p = abs(S) / N
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -alpha * p * math.log(p)


class HexOKGate:
    """
    Numeric version of `hex-ok.py`:
    takes a numeric array (not ASCII), applies
    elliptic permutation → Möbius filter → logic gate →
    exponential convolution → supertrace compression → SHA‑256 hex.
    """
    def __init__(self, K: int = 256, logic_gate: str = 'log'):
        self.K = K
        self.logic_gate = logic_gate
        self.mu = mobius_sieve(K)

    def _logic(self, sig: np.ndarray) -> np.ndarray:
        if self.logic_gate == 'log':
            return np.log(np.maximum(np.abs(sig), 1e-12))
        if self.logic_gate == 'exp':
            return np.exp(np.clip(sig, -50, 50))
        if self.logic_gate == 'sin':
            return np.sin(sig)
        if self.logic_gate == 'cos':
            return np.cos(sig)
        return sig

    def fingerprint(self, signal: np.ndarray) -> str:
        sig = np.asarray(signal, dtype=np.float64).ravel()
        K = min(len(sig), self.K)
        if K == 0:
            return hashlib.sha256(b'').hexdigest()
        sig = sig[:K]

        # 1. elliptic permutation
        order = _elliptic_permutation(K)
        sig = sig[order]

        # 2. Möbius filter (keep square‑free indices)
        keep = np.array([i for i in range(K) if self.mu[i + 1] != 0])
        if keep.size == 0:
            return hashlib.sha256(b'').hexdigest()
        sig = sig[keep]

        # 3. logic gate
        sig = self._logic(sig)

        # 4. convolution + supertrace
        conv = _conv_exp_kernel(sig)
        S = _supertrace(conv)
        H = _entropy(S, len(conv))
        M = max(1, min(int(abs(S)), len(conv)))

        # 5. keep top‑M square‑free coefficients
        mag = np.abs(conv)
        idx_sorted = np.argsort(mag)[::-1]
        kept = []
        for idx in idx_sorted:
            n = int(idx) + 1
            if n <= len(self.mu) - 1 and self.mu[n] != 0:
                kept.append((int(idx), float(conv[idx])))
                if len(kept) >= M:
                    break

        # 6. pack + hash
        buf = bytearray()
        for idx, val in kept:
            buf.extend(struct.pack('>Hd', idx, val))
        return hashlib.sha256(buf).hexdigest()

# ============================================================
#  Layer specs + store
# ============================================================
@dataclass
class LayerSpec:
    name: str
    kind: str                              # 'stored' | 'derived'
    dtype: object = np.float64
    deriver: Optional[Callable] = None     # (store, start, end) -> array
    cache: bool = True
    per_chunk: bool = False                # deriver takes (store, cid)


class MultiLayerSSDStore:
    """
    A single SSD‑backed `real` memmap, plus any number of derived
    layers materialised through chunk‑level caches.
    """

    def __init__(self, root: str, K: int,
                 chunk_size: int = 4096,
                 cache_chunks: int = 16):
        self.root = root
        self.K = K
        self.chunk_size = chunk_size
        self.cache_chunks = cache_chunks
        os.makedirs(root, exist_ok=True)

        self.specs: Dict[str, LayerSpec] = {}
        self.memmaps: Dict[str, np.memmap] = {}
        self.chunk_cache: Dict[str, "OrderedDict[int, np.ndarray]"] = {}
        self.stats = dict(reads=0, cache_hits=0, cache_misses=0, evictions=0)
        self.hex_gate = HexOKGate(K=min(256, chunk_size))

    # ---------- registration ----------
    def register_stored(self, name: str, initial: Optional[np.ndarray] = None,
                        dtype=np.float64):
        path = os.path.join(self.root, f"{name}.mm")
        mm = np.memmap(path, dtype=dtype, mode='w+', shape=(self.K,))
        if initial is not None:
            mm[:] = np.asarray(initial, dtype=dtype)
            mm.flush()
        self.memmaps[name] = mm
        self.specs[name] = LayerSpec(name=name, kind='stored', dtype=dtype,
                                     cache=False)

    def register_derived(self, name: str, deriver: Callable,
                         dtype=np.float64, cache: bool = True,
                         per_chunk: bool = False):
        self.specs[name] = LayerSpec(name=name, kind='derived', dtype=dtype,
                                     deriver=deriver, cache=cache,
                                     per_chunk=per_chunk)
        if cache:
            self.chunk_cache[name] = OrderedDict()

    # ---------- chunk helpers ----------
    def _chunk_bounds(self, cid: int) -> Tuple[int, int]:
        s = cid * self.chunk_size
        e = min(s + self.chunk_size, self.K)
        return s, e

    def _cid(self, i: int) -> int:
        return i // self.chunk_size

    # ---------- layer access ----------
    def _raw_chunk(self, layer: str, cid: int) -> np.ndarray:
        spec = self.specs[layer]
        start, end = self._chunk_bounds(cid)
        n = end - start

        if spec.kind == 'stored':
            return np.asarray(self.memmaps[layer][start:end],
                              dtype=np.asarray(self.memmaps[layer]).dtype)

        # derived
        if spec.cache and cid in self.chunk_cache[layer]:
            self.chunk_cache[layer].move_to_end(cid)
            self.stats['cache_hits'] += 1
            return self.chunk_cache[layer][cid]

        self.stats['cache_misses'] += 1
        if spec.per_chunk:
            arr = spec.deriver(self, cid)
        else:
            arr = spec.deriver(self, start, end)
        arr = np.asarray(arr)
        if arr.dtype != np.float64 and spec.dtype is not None:
            try:
                arr = arr.astype(spec.dtype)
            except Exception:
                pass
        if spec.cache:
            self.chunk_cache[layer][cid] = arr
            while len(self.chunk_cache[layer]) > self.cache_chunks:
                self.chunk_cache[layer].popitem(last=False)
                self.stats['evictions'] += 1
        return arr

    def get_chunk(self, layer: str, cid: int) -> np.ndarray:
        return self._raw_chunk(layer, cid)

    def get(self, layer: str, i: int):
        self.stats['reads'] += 1
        if i < 0 or i >= self.K:
            raise IndexError(i)
        cid = self._cid(i)
        chunk = self._raw_chunk(layer, cid)
        return chunk[i - cid * self.chunk_size]

    def get_range(self, layer: str, start: int, end: Optional[int] = None):
        if end is None:
            end = self.K
        start = max(0, start); end = min(self.K, end)
        out = np.empty(end - start, dtype=object) if self.specs[layer].dtype is object \
              else np.empty(end - start, dtype=self.specs[layer].dtype)
        pos = 0
        cid0, cid1 = self._cid(start), self._cid(end - 1)
        for cid in range(cid0, cid1 + 1):
            cs, ce = self._chunk_bounds(cid)
            chunk = self._raw_chunk(layer, cid)
            lo = max(start, cs) - cs
            hi = min(end, ce) - cs
            take = hi - lo
            if take > 0:
                out[pos:pos + take] = chunk[lo:hi]
                pos += take
        return out

    def stream(self, layer: str) -> Iterator[Tuple[int, np.ndarray]]:
        n_chunks = (self.K + self.chunk_size - 1) // self.chunk_size
        for cid in range(n_chunks):
            s, _ = self._chunk_bounds(cid)
            yield s, self._raw_chunk(layer, cid)

    def cache_report(self) -> Dict:
        return dict(self.stats,
                    cache_sizes={k: len(v) for k, v in self.chunk_cache.items()},
                    cache_capacity=self.cache_chunks,
                    chunk_size=self.chunk_size)

# ============================================================
#  Default derivers
# ============================================================
def deriver_imag(store: MultiLayerSSDStore, start: int, end: int) -> np.ndarray:
    """
    im[i] = -d(re)/dn / a  via forward difference.
    Needs one sample past `end` to close the last difference.
    """
    n = end - start
    extra_end = min(end + 1, store.K)
    raw = np.asarray(store.memmaps['real'][start:extra_end], dtype=np.float64)
    imag = np.zeros(n, dtype=np.float64)
    if raw.size >= 2:
        deriv = (raw[1:] - raw[:-1]) / A_STEP
        m = min(deriv.size, n)
        imag[:m] = -deriv[:m]
        if m < n:
            imag[n - 1] = -(raw[-1] - raw[-2]) / A_STEP
    return imag


def deriver_envelope(store: MultiLayerSSDStore, start: int, end: int) -> np.ndarray:
    re = np.asarray(store.memmaps['real'][start:end], dtype=np.float64)
    im = deriver_imag(store, start, end)
    n = min(re.size, im.size)
    return np.hypot(re[:n], im[:n])


def deriver_hex_chunk(store: MultiLayerSSDStore, cid: int) -> np.ndarray:
    """
    One hex fingerprint per chunk (from the hex‑ok gate on the real data).
    Returned as a length‑`n` array of the same string (so it fits the
    column model).
    """
    s, e = store._chunk_bounds(cid)
    real = np.asarray(store.memmaps['real'][s:e], dtype=np.float64)
    fp = store.hex_gate.fingerprint(real)
    return np.array([fp] * (e - s), dtype=object)

# ============================================================
#  Reader with UI enumeration + strike analysis
# ============================================================
@dataclass
class UiPage:
    layer: str
    page: int
    page_size: int
    start: int
    end: int
    values: List


class LayerReader:
    """
    Front‑facing API over a `MultiLayerSSDStore`.

    Typical front‑end calls:

        reader.ui_view('real', page=0, page_size=64)
        reader.ui_view('hex',  page=2, page_size=64)
        reader.read('envelope', start=1000, end=1100)
        reader.flagged_report()
    """

    def __init__(self, store: MultiLayerSSDStore):
        self.store = store

    # ---------- basic reads ----------
    def read(self, layer: str, start: int = 0, end: Optional[int] = None) -> np.ndarray:
        return self.store.get_range(layer, start, end)

    def stream(self, layer: str):
        return self.store.stream(layer)

    # ---------- UI pagination ----------
    def ui_view(self, layer: str, page: int = 0, page_size: int = 64) -> UiPage:
        start = page * page_size
        end = min(start + page_size, self.store.K)
        vals = self.read(layer, start, end)
        return UiPage(layer=layer, page=page, page_size=page_size,
                      start=start, end=end,
                      values=vals.tolist())

    def ui_layers(self) -> List[str]:
        return list(self.store.specs.keys())

    # ---------- strike analysis on the envelope layer ----------
    def flagged_report(self,
                       strike_threshold: int = 2,
                       entropy_tolerance: float = 0.12,
                       reference_layer: Optional[str] = None) -> List[Dict]:
        """
        Compute supertrace / entropy / asymmetric lower bound on the
        envelope layer, flag profiles, and return the flagged ones.
        """
        env = self.read('envelope')
        K = env.size
        S = _supertrace(env)
        H = _entropy(S, K)

        # asymmetric lower bound (first half)
        half = K // 2
        S_h = _supertrace(env[:half])
        asym_lb = _entropy(S_h, half)

        # symmetric inverse score against a reference layer (optional)
        sym = 0.5
        if reference_layer is not None and reference_layer in self.store.specs:
            ref = self.read(reference_layer)
            if ref.size == K:
                pairs = sorted(zip(env.tolist(), ref.tolist()),
                               key=lambda p: p[0])
                b_sorted = [p[1] for p in pairs]
                inv = _inversion_count(b_sorted)
                mx = K * (K - 1) // 2
                sym = inv / mx if mx > 0 else 0.5

        strikes = 0
        if abs(sym - 0.5) > entropy_tolerance:
            strikes += 1
        if asym_lb > H + entropy_tolerance:
            strikes += 1

        report = dict(
            K=K, S=S, H=H, asym_lb=asym_lb, sym=sym,
            strikes=strikes,
            flagged=strikes >= strike_threshold,
            cache=self.store.cache_report(),
        )
        return [report] if report['flagged'] else []

    # ---------- per‑query profiles (for the strike-flag reader) ----------
    def profile(self, query_id: str,
                emotional_entropy: float = 0.0,
                entropy_tolerance: float = 0.12,
                emotional_boost: float = 1.4,
                strike_threshold: int = 2) -> Dict:
        env = self.read('envelope')
        K = env.size
        S = _supertrace(env)
        H = _entropy(S, K)
        H_em = emotional_entropy * emotional_boost
        half = K // 2
        S_h = _supertrace(env[:half])
        asym_lb = _entropy(S_h, half)

        strikes = 0
        if H_em > 0.0 and abs(H - H_em) > entropy_tolerance:
            strikes += 1
        if asym_lb > H + entropy_tolerance:
            strikes += 1

        return dict(
            query_id=query_id,
            K=K, S=S, H=H, H_em=H_em, asym_lb=asym_lb,
            strikes=strikes,
            flagged=strikes >= strike_threshold,
            cache=self.store.cache_report(),
        )

# ============================================================
#  Helpers
# ============================================================
def _inversion_count(arr):
    n = len(arr)
    if n < 2:
        return 0
    return _merge_count(list(arr), [0] * n, 0, n - 1)


def _merge_count(a, tmp, l, r):
    if l >= r:
        return 0
    m = (l + r) // 2
    inv = _merge_count(a, tmp, l, m) + _merge_count(a, tmp, m + 1, r)
    i, j, k = l, m + 1, l
    while i <= m and j <= r:
        if a[i] <= a[j]:
            tmp[k] = a[i]; i += 1
        else:
            tmp[k] = a[j]
            inv += m - i + 1
            j += 1
        k += 1
    while i <= m: tmp[k] = a[i]; i += 1; k += 1
    while j <= r: tmp[k] = a[j]; j += 1; k += 1
    for i in range(l, r + 1):
        a[i] = tmp[i]
    return inv

# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 68)
    print("SSD multi‑layer reader  (real + imag + envelope + hex)")
    print("=" * 68)

    K = 4096
    scratch = "ssd_ml_scratch"
    store = MultiLayerSSDStore(scratch, K, chunk_size=512, cache_chunks=3)

    # --- primary data: real |Ci| on SSD ---
    x = np.arange(K)
    real = np.abs(np.sin(x * 0.02) + 0.3 * np.cos(x * 0.05))
    store.register_stored('real', initial=real)

    # --- derived layers ---
    store.register_derived('imag',     deriver_imag,     cache=True)
    store.register_derived('envelope', deriver_envelope, cache=True)
    store.register_derived('hex',      deriver_hex_chunk, dtype=object,
                           cache=True, per_chunk=True)

    reader = LayerReader(store)

    # --- layer overview ---
    print(f"\nLayers registered: {reader.ui_layers()}")
    print(f"Primary layer size: K = {K}, chunk_size = {store.chunk_size}")

    # --- UI pagination ---
    for layer in ['real', 'imag', 'envelope']:
        page = reader.ui_view(layer, page=0, page_size=8)
        print(f"\nUI page 0 of '{layer}'  [{page.start}:{page.end}]:")
        print("   " + ", ".join(f"{v:+.4f}" for v in page.values))

    # --- hex layer (per‑chunk fingerprint) ---
    print("\nUI page 0 of 'hex'  (first 3 fingerprints):")
    hex_page = reader.ui_view('hex', page=0, page_size=3)
    for v in hex_page.values[:3]:
        print(f"   {v[:16]}…")

    # --- read a range ---
    s, e = 1000, 1010
    env_slice = reader.read('envelope', start=s, end=e)
    print(f"\nEnvelope slice [{s}:{e}]:")
    print("   " + ", ".join(f"{v:.4f}" for v in env_slice))

    # --- stream layer ---
    print("\nStreaming 'real' (first 3 chunks):")
    for i, (start, chunk) in enumerate(reader.stream('real')):
        if i >= 3:
            break
        print(f"   chunk@{start:5d}: mean={chunk.mean():.4f}  "
              f"std={chunk.std():.4f}  n={chunk.size}")

    # --- strike analysis on envelope ---
    print("\nFlagged report on 'envelope':")
    flags = reader.flagged_report(strike_threshold=2,
                                  entropy_tolerance=0.12,
                                  reference_layer='real')
    if flags:
        for f in flags:
            print(f"   flagged: strikes={f['strikes']}  "
                  f"S={f['S']:+.4f}  H={f['H']:.4f}  "
                  f"lb={f['asym_lb']:.4f}  sym={f['sym']:.4f}")
    else:
        print("   no flagged profiles")

    # --- per‑query profile with emotional entropy ---
    print("\nPer‑query profile with emotional entropy boost:")
    prof = reader.profile('q_demo', emotional_entropy=0.35,
                          entropy_tolerance=0.12, emotional_boost=1.4)
    for k, v in prof.items():
        if isinstance(v, dict):
            print(f"   {k}: {v}")
        elif isinstance(v, float):
            print(f"   {k}: {v:.4f}")
        else:
            print(f"   {k}: {v}")

    # --- cache report ---
    print("\nCache report:")
    cr = store.cache_report()
    for k, v in cr.items():
        print(f"   {k}: {v}")

    print("\nDone.")


if __name__ == "__main__":
    demo()