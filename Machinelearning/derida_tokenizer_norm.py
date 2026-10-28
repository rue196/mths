#!/usr/bin/env python3
"""
spectral_derivative_tokenizer.py
================================

A tokenizer whose response to a user query **is a spectral derivative**,
with decoding weighted by the finite-step derivative, asymmetric tail
attention cross-checked by the dispersal RL policies, and |Ci| arrays
stored as JSON chunks in a read-only MOSB buffer.

Pipeline
--------
    user query (natural language)
        │
        ▼
    1. tokenize into words
        │
        ▼
    2. each word → complex |Ci| array of length 2K+1
       (μ-signed base + hash phase + elliptic smoothing for μ=0)
        │
        ▼
    3. finite-step weight per index
         w[i] = |(C[i+1] − C[i−1]) / (2·A_STEP)|
       normalized to ∑ w[i] = 1  (continuous decoding weights)
        │
        ▼
    4. attention regions (Möbius elliptic spectrum)
         center:  |i| small → Gaussian peak → NORMAL (symmetric)
         tails:   |i| large → below threshold → ASYMMETRIC
        │
        ▼
    5. dispersal cross-check (from dispersal_information_RL.py)
         BLIND:       trust unconditionally
         CROSS_CHECK: bound commitment by check_eff
         RL_ADMIT:    if commitment > check_eff → block, admit, reset
        │
        ▼
    6. store each word as a JSON chunk in the read-only MOSB buffer
       (cid → JSON → |Ci|, weights, attention, asym flag, commitment)
        │
        ▼
    7. derivative query at position t
         response = Σ_i (i·i/α) · C_i · e^{i·t·i/α} · w[i]
         = spectral derivative dζ/dt(t), finite-step weighted
         asymmetric blocks contribute 0 to the response

Key formula
-----------
    Real symmetric case (derida.rb):
        dζ/dt = −2 Σ_{i>0} C_i · (i/α) · sin(t·i/α)

    General complex case:
        dζ/dt = Σ_i (i·i/α) · C_i · e^{i·t·i/α}

The first is used when C_{-i} = C_i (pure real).  The second is used
here because |Ci| arrays are complex.

Storage layout
--------------
    MOSB chunk = [MAGIC(4) | K(4) | n(8) | cid(8) | S(8) | H(8) | m(8)
                  | JSON payload]

    JSON payload:
    {
      "word":             str,
      "K":                int,
      "ci_re":            [float] × (2K+1),
      "ci_im":            [float] × (2K+1),
      "weights_fs":       [float] × (2K+1),         # finite-step normalized
      "attention":        [float] × (2K+1),         # Gaussian kernel
      "asym_indices":     [int],
      "asym_commitment":  float,
      "accepted":         bool,
      "commitment_after": float,
      "corrected":        bool,
      "signature":        hex16
    }
"""

from __future__ import annotations
import hashlib, json, math, struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================
#  Constants  (derida.rb + negative-log-mobius-gate)
# ============================================================
PI       = math.pi
E        = math.e
ALPHA    = 1.0 / (PI - E)             # ≈ 2.362338
A_STEP   = ALPHA / 0.3628             # ≈ 6.511  (finite step)
GAMMA    = 0.5772156649015329

DEFAULT_K = 20
DEFAULT_SIGMA_FRAC = 0.25             # attention Gaussian width / K
DEFAULT_TAIL_THRESHOLD = 0.05          # attention below → asymmetric


# ============================================================
#  Möbius sieve  ·  O(K)
# ============================================================
def mobius_sieve(K: int) -> List[int]:
    mu = [0] * (K + 1)
    if K >= 1:
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
            mu[i * p] = -mu[i]
    return mu


def smooth_zeros_1d(c: np.ndarray, K: int) -> None:
    """Fill μ=0 slots with linear interpolation (in place)."""
    for i in range(-K, K + 1):
        idx = i + K
        if i == 0 or abs(c[idx]) > 1e-12:
            continue
        l, r = i - 1, i + 1
        while l >= -K and abs(c[l + K]) < 1e-12:
            l -= 1
        while r <= K and abs(c[r + K]) < 1e-12:
            r += 1
        if l < -K or r > K or r == l:
            continue
        d = r - l
        c[idx] = (r - i) / d * c[l + K] + (i - l) / d * c[r + K]


# ============================================================
#  Spectral derivative  (derida.rb, generalized to complex)
# ============================================================
def spectral_derivative_real(t: float, c_re: np.ndarray,
                              alpha: float = ALPHA) -> float:
    """
    Real derivative for symmetric real coefficients C_{-i} = C_i.

        dζ/dt = −2 Σ_{i>0} C_i · (i/α) · sin(t·i/α)
    """
    K = (len(c_re) - 1) // 2
    acc = 0.0
    for i in range(1, K + 1):
        theta = t * i / alpha
        acc += -2.0 * c_re[K + i] * (i / alpha) * math.sin(theta)
    return acc


def spectral_derivative_complex(t: float, c_re: np.ndarray,
                                 c_im: np.ndarray,
                                 alpha: float = ALPHA) -> complex:
    """
    General derivative for complex coefficients.

        dζ/dt = Σ_i (i·i/α) · C_i · e^{i·t·i/α}
    """
    K = (len(c_re) - 1) // 2
    total = 0.0 + 0.0j
    for i in range(-K, K + 1):
        if i == 0:
            continue
        C = complex(c_re[K + i], c_im[K + i])
        theta = t * i / alpha
        total += (1j * i / alpha) * C * complex(
            math.cos(theta), math.sin(theta))
    return total


def spectral_derivative_weighted(t: float,
                                  c_re: np.ndarray,
                                  c_im: np.ndarray,
                                  w: np.ndarray,
                                  alpha: float = ALPHA
                                  ) -> Tuple[float, float]:
    """
    Complex derivative with per-index finite-step weights.

        dζ/dt = Σ_i (i·i/α) · C_i · e^{i·t·i/α} · w[i]

    Returns (Re, Im).
    """
    K = (len(c_re) - 1) // 2
    d_re = 0.0
    d_im = 0.0
    for i in range(-K, K + 1):
        if i == 0:
            continue
        C = complex(c_re[K + i], c_im[K + i])
        theta = t * i / alpha
        term = (1j * i / alpha) * C * complex(
            math.cos(theta), math.sin(theta))
        wi = float(w[K + i])
        d_re += term.real * wi
        d_im += term.imag * wi
    return d_re, d_im


# ============================================================
#  Finite-step continuous weighting
# ============================================================
def finite_step_weight(c_re: np.ndarray, c_im: np.ndarray,
                        a: float = A_STEP) -> np.ndarray:
    """
    Continuous normalized encoding:
        w[i] = |(C[i+1] − C[i−1]) / (2a)|
    normalized so that ∑ w[i] = 1.
    """
    n = len(c_re)
    w = np.zeros(n)
    for i in range(1, n - 1):
        d_re = (c_re[i + 1] - c_re[i - 1]) / (2.0 * a)
        d_im = (c_im[i + 1] - c_im[i - 1]) / (2.0 * a)
        w[i] = math.sqrt(d_re * d_re + d_im * d_im)
    s = float(w.sum())
    if s > 1e-12:
        w /= s
    return w


# ============================================================
#  Attention regions  ·  Gaussian peak at center, asymmetric tails
# ============================================================
def attention_regions(K: int,
                      sigma_frac: float = DEFAULT_SIGMA_FRAC,
                      threshold: float = DEFAULT_TAIL_THRESHOLD
                      ) -> Tuple[List[int], List[int], np.ndarray]:
    """
    Attention kernel on the Möbius elliptic spectrum:

        attn[i] = exp(-i² / (2σ²))      σ = sigma_frac · K

    i = 0 has maximum attention (statistically "normal", peak).
    Tails with attn[i] < threshold are marked ASYMMETRIC.
    """
    sigma = max(1.0, sigma_frac * K)
    weights = np.zeros(2 * K + 1)
    center: List[int] = []
    tail: List[int] = []
    for i in range(-K, K + 1):
        w = math.exp(-(i * i) / (2.0 * sigma * sigma))
        weights[i + K] = w
        if w >= threshold:
            center.append(i)
        else:
            tail.append(i)
    return center, tail, weights


# ============================================================
#  Dispersal cross-check  (dispersal_information_RL.py)
# ============================================================
class DispersalGate:
    """
    Cross-check asymmetric tokens using the dispersal RL policies.

        BLIND       — no cross-check, commit as-is
        CROSS_CHECK — bound by check_eff
        RL_ADMIT    — detect + admit + reset
    """
    def __init__(self, policy: str = "rl_admit",
                 check_strength: float = 0.3,
                 n_data_points: int = 3):
        self.policy = policy
        self.check_strength = check_strength
        self.n_data_points = n_data_points
        self.check_eff = 1.0 - (1.0 - check_strength) ** max(1, n_data_points)

    def verify(self, commitment: float) -> Tuple[bool, float, bool]:
        """Return (accepted, commitment_after, corrected)."""
        if self.policy == "blind":
            return True, commitment, False
        if self.policy == "cross_check":
            caught = self.check_eff * min(commitment, 1.0)
            c_new = max(0.0, commitment - caught)
            return True, c_new, commitment > c_new
        if self.policy == "rl_admit":
            if commitment > self.check_eff:
                return False, 0.0, True
            caught = self.check_eff * min(commitment, 1.0)
            c_new = max(0.0, commitment - caught)
            return True, c_new, commitment > c_new
        return True, commitment, False


# ============================================================
#  MOSB read-only buffer
# ============================================================
MAGIC       = b"MOSB"
HEADER_FMT  = ">4sIQQddd"
HEADER_SIZE = struct.calcsize(HEADER_FMT)


class MOSBReadOnlyBuffer:
    def __init__(self, K: int = 512):
        self.K = K
        self.data = bytearray()
        self.chunks: List[Tuple[int, int, float, float, float]] = []
        self._cid = 0

    @staticmethod
    def _supertrace(payload: bytes) -> float:
        S = 0.0
        for t, b in enumerate(payload):
            v = (b - 128) / 128.0
            S += v if (t % 2 == 0) else -v
        return S

    @staticmethod
    def _entropy_mass(S: float, n: int) -> Tuple[float, float]:
        if S == 0 or n <= 0:
            return 0.0, 0.0
        p = min(abs(S) / n, 0.9999)
        H = -ALPHA * p * math.log(p) if p > 0 else 0.0
        return H, abs(S) * math.exp(-H)

    def append(self, payload: bytes) -> int:
        n = len(payload)
        S = self._supertrace(payload)
        H, m = self._entropy_mass(S, n)
        cid = self._cid
        self._cid += 1
        hdr = struct.pack(HEADER_FMT, MAGIC, self.K, n, cid, S, H, m)
        start = len(self.data)
        self.data.extend(hdr + payload)
        self.chunks.append((start, start + HEADER_SIZE + n, S, H, m))
        return cid

    def read_chunk(self, cid: int
                   ) -> Optional[Tuple[bytes, float, float, float]]:
        if cid < 0 or cid >= len(self.chunks):
            return None
        start, end, *_ = self.chunks[cid]
        blob = bytes(self.data[start:end])
        magic, K, n, cid_stored, S, H, m = struct.unpack(
            HEADER_FMT, blob[:HEADER_SIZE])
        if magic != MAGIC or K != self.K or cid_stored != cid:
            return None
        payload = blob[HEADER_SIZE:HEADER_SIZE + n]
        if abs(self._supertrace(payload) - S) > 1e-9:
            return None
        return payload, S, H, m

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)

    @property
    def size_bytes(self) -> int:
        return len(self.data)


# ============================================================
#  Word → complex |Ci| array
# ============================================================
def word_to_ci(word: str, K: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deterministic complex |Ci| array from the word hash.

        C_i = μ(|i|) · hash_phase_i
        smoothed at μ=0 slots (elliptic Möbius gate)
    """
    mu = mobius_sieve(K)
    h = hashlib.shake_256(word.lower().encode()).digest(2 * (2 * K + 1))
    c_re = np.zeros(2 * K + 1)
    c_im = np.zeros(2 * K + 1)
    for i in range(-K, K + 1):
        if i == 0:
            continue
        m = float(mu[abs(i)])
        b1 = h[(i + K) * 2] / 255.0
        b2 = h[(i + K) * 2 + 1] / 255.0
        c_re[i + K] = m * (0.5 + 0.5 * b1)
        c_im[i + K] = m * (0.5 + 0.5 * b2) * ALPHA
    smooth_zeros_1d(c_re, K)
    smooth_zeros_1d(c_im, K)
    return c_re, c_im


# ============================================================
#  The tokenizer
# ============================================================
class SpectralDerivativeTokenizer:
    """
    Tokenizer with a derivative-as-query contract and asymmetry-aware
    read path.
    """
    def __init__(self, K: int = DEFAULT_K,
                 policy: str = "rl_admit",
                 n_data_points: int = 3,
                 check_strength: float = 0.3):
        self.K = K
        self.mosb = MOSBReadOnlyBuffer(K=K)
        self.gate = DispersalGate(policy, check_strength, n_data_points)
        self.cids: Dict[str, int] = {}
        self.chunks: Dict[str, dict] = {}
        center, tail, weights = attention_regions(K)
        self.center = center
        self.tail = tail
        self.tail_set = set(tail)
        self.attention = weights

    # ---- tokenize ----
    def tokenize(self, query: str) -> List[str]:
        words = [w for w in query.lower().split() if w]
        for w in words:
            if w in self.chunks:
                continue
            c_re, c_im = word_to_ci(w, self.K)
            w_fs = finite_step_weight(c_re, c_im)

            asym_indices: List[int] = []
            for i in range(-self.K, self.K + 1):
                if i == 0:
                    continue
                if i not in self.tail_set:
                    continue
                if abs(c_re[i + self.K]) < 1e-9 and abs(c_im[i + self.K]) < 1e-9:
                    continue
                asym_indices.append(i)

            if asym_indices:
                asym_commit = float(np.mean(
                    [w_fs[i + self.K] for i in asym_indices]))
            else:
                asym_commit = 0.0

            accepted, c_after, corrected = self.gate.verify(asym_commit)

            chunk = dict(
                word=w, K=self.K,
                ci_re=c_re.tolist(),
                ci_im=c_im.tolist(),
                weights_fs=w_fs.tolist(),
                attention=self.attention.tolist(),
                asym_indices=asym_indices,
                asym_commitment=asym_commit,
                accepted=accepted,
                commitment_after=c_after,
                corrected=corrected,
                signature=hashlib.sha256(
                    w.encode() + c_re.tobytes()).hexdigest()[:16],
            )
            self.chunks[w] = chunk
            payload = json.dumps(chunk, separators=(",", ":")).encode()
            self.cids[w] = self.mosb.append(payload)
        return words

    # ---- read via MOSB (with asymmetric gate) ----
    def _read(self, word: str) -> Optional[dict]:
        cid = self.cids.get(word)
        if cid is None:
            return None
        chunk = self.mosb.read_chunk(cid)
        if chunk is None:
            return None
        return json.loads(chunk[0].decode())

    # ---- derivative query ----
    def derivative_query(self, t: float) -> dict:
        """
        Response to the user at spectral position t:
            dζ/dt(t) with finite-step-weighted coefficients,
            asymmetric-blocked tokens contribute 0.
        """
        contributions: List[dict] = []
        total_re = 0.0
        total_im = 0.0
        n_blocked = 0
        n_accepted = 0

        for word, cid in self.cids.items():
            rec = self._read(word)
            if rec is None:
                continue
            if rec['asym_indices'] and not rec['accepted']:
                n_blocked += 1
                contributions.append(dict(
                    word=word, status='blocked',
                    commitment=rec['asym_commitment'],
                    corrected=rec['corrected'],
                ))
                continue

            c_re = np.asarray(rec['ci_re'])
            c_im = np.asarray(rec['ci_im'])
            w_fs = np.asarray(rec['weights_fs'])

            # weighted derivative per word
            d_re, d_im = spectral_derivative_weighted(
                t, c_re, c_im, w_fs)

            total_re += d_re
            total_im += d_im
            n_accepted += 1

            contributions.append(dict(
                word=word, status='accepted',
                d_re=d_re, d_im=d_im,
                mag=math.sqrt(d_re * d_re + d_im * d_im),
            ))

        return dict(
            t=t,
            n_tokens=len(self.cids),
            n_accepted=n_accepted,
            n_blocked=n_blocked,
            d_re=total_re,
            d_im=total_im,
            magnitude=math.sqrt(total_re * total_re + total_im * total_im),
            contributions=contributions,
        )

    # ---- JSON export ----
    def to_json(self) -> dict:
        return dict(
            K=self.K,
            policy=self.gate.policy,
            center=self.center,
            tail=self.tail,
            n_tokens=len(self.cids),
            chunks=list(self.chunks.values()),
        )

    # ---- summary ----
    def summary(self) -> dict:
        n_asym = sum(1 for c in self.chunks.values() if c['asym_indices'])
        n_blocked = sum(1 for c in self.chunks.values()
                        if c['asym_indices'] and not c['accepted'])
        return dict(
            K=self.K,
            n_tokens=len(self.cids),
            n_center=len(self.center),
            n_tail=len(self.tail),
            n_asym_tokens=n_asym,
            n_blocked=n_blocked,
            mosb_chunks=self.mosb.n_chunks,
            mosb_bytes=self.mosb.size_bytes,
            policy=self.gate.policy,
            check_eff=self.gate.check_eff,
        )


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("Spectral-derivative tokenizer  ·  asymmetric-tail attention")
    print("=" * 78)
    print(f"  α            = 1/(π − e)   = {ALPHA:.6f}")
    print(f"  A_STEP                      = {A_STEP:.6f}")
    print(f"  attention    σ               = {DEFAULT_SIGMA_FRAC} · K")
    print(f"  tail thresh                  = {DEFAULT_TAIL_THRESHOLD}")
    print()

    # ---------- sanity check: derida.rb formula ----------
    print("--- derida.rb sanity check ---")
    K = 20
    c_re = np.zeros(2 * K + 1)
    for i in range(-K, K + 1):
        c_re[i + K] = 1.0
    t0 = 1.5
    d_general = spectral_derivative_real(t0, c_re, ALPHA)
    # Direct complex sum for comparison
    c_im_zeros = np.zeros(2 * K + 1)
    d_full = spectral_derivative_complex(t0, c_re, c_im_zeros, ALPHA)
    print(f"  t0 = {t0}")
    print(f"  real-form   dζ/dt = {d_general:+.6f}")
    print(f"  complex-form Re   = {d_full.real:+.6f}")
    print(f"  match             = "
          f"{abs(d_general - d_full.real) < 1e-9}")
    print()

    # ---------- tokenize a query ----------
    for policy in ("blind", "cross_check", "rl_admit"):
        print(f"=== policy: {policy} ===")
        tok = SpectralDerivativeTokenizer(
            K=DEFAULT_K, policy=policy, n_data_points=3)
        query = "sort the database by user id"
        words = tok.tokenize(query)
        print(f"  query              = {query!r}")
        print(f"  words              = {words}")

        s = tok.summary()
        print(f"  K                  = {s['K']}")
        print(f"  n_center           = {s['n_center']}")
        print(f"  n_tail             = {s['n_tail']}")
        print(f"  asymmetric tokens  = {s['n_asym_tokens']} / {s['n_tokens']}")
        print(f"  blocked            = {s['n_blocked']}")
        print(f"  MOSB chunks        = {s['mosb_chunks']}")
        print(f"  MOSB bytes         = {s['mosb_bytes']}")
        print(f"  check_eff          = {s['check_eff']:.4f}")
        print()

        # derive at several t
        for t_val in (0.0, 1.5, 3.0):
            r = tok.derivative_query(t_val)
            print(f"  dζ/dt({t_val:>4.1f})  "
                  f"Re={r['d_re']:>+10.4f}  "
                  f"Im={r['d_im']:>+10.4f}  "
                  f"|.|={r['magnitude']:>8.4f}  "
                  f"acc={r['n_accepted']}  blk={r['n_blocked']}")
        print()

        # show per-word contributions
        r = tok.derivative_query(1.5)
        print(f"  per-word contributions at t = 1.5:")
        print(f"    {'word':<12s}  {'status':<10s}  "
              f"{'d_re':>10s}  {'d_im':>10s}  {'|.|':>8s}")
        for c in r['contributions']:
            if c['status'] == 'accepted':
                print(f"    {c['word']:<12s}  {'accepted':<10s}  "
                      f"{c['d_re']:>+10.4f}  {c['d_im']:>+10.4f}  "
                      f"{c['mag']:>8.4f}")
            else:
                print(f"    {c['word']:<12s}  {'BLOCKED':<10s}  "
                      f"{'—':>10s}  {'—':>10s}  "
                      f"commit={c['commitment']:.3f}")
        print()

    # ---------- JSON export ----------
    print("=== JSON export ===")
    tok = SpectralDerivativeTokenizer(K=12, policy="rl_admit")
    tok.tokenize("hello world")
    payload = tok.to_json()
    jstr = json.dumps(payload, separators=(",", ":"))
    print(f"  export size        = {len(jstr)} bytes")
    print(f"  K                  = {payload['K']}")
    print(f"  n_tokens           = {payload['n_tokens']}")
    print(f"  center indices     = {payload['center']}")
    print(f"  tail indices       = {payload['tail']}")
    print(f"  first chunk words  = "
          f"{[c['word'] for c in payload['chunks'][:2]]}")
    print()

    # ---------- complexity ----------
    print("--- complexity ---")
    print(f"  tokenize per word       O(K)         (μ + hash + smooth)")
    print(f"  finite-step weight      O(K)")
    print(f"  asym classification     O(K)")
    print(f"  MOSB append             O(K)         (JSON size)")
    print(f"  derivative query        O(n_tokens · K)")
    print()
    print(f"  memory per word         ~{4 * (2 * DEFAULT_K + 1) * 3} bytes"
          f"  (ci_re + ci_im + weights_fs as float32)")
    print()
    print("Done.")


if __name__ == "__main__":
    demo()