#!/usr/bin/env python3
"""
token_coeff_dual.py
===================

Dual-coefficient tokenizer.

    c[K+i] = c_re[K+i] + i·c_im[K+i]

        c_re  →  ORIGINAL token         α = 0.3628          (real axis)
        c_im  →  ASSOCIATED token(s)    α = 1/(π−e) ≈ 2.362 (imag axis)

Axis classification  (from factor.py)
-------------------------------------
    i odd, μ(i) = −1            →  REAL     (self coefficient)
    i even / μ(i) = +1 / μ(i)=0 →  IMAG     (associated token)

Quadratic SAT gate  (from factor.py)
------------------------------------
    q(i) = (i² + i) mod K
    μ(q(i)) = 0  →  coefficient × 0.5   (attenuated, still recoverable)

Decoding a |Ci| array
---------------------
    original token       =  argmax_{i>0} c_re[K+i]
    associated tokens    =  top-k { i : c_im[K+i] } by amplitude
                            (each mapped back to its word via the hash map)

Storage  (MOSB read-only buffer, JSON chunk per token)
-----------------------------------------------------
    {
      "word":           str,
      "K":              int,
      "ci_re":          [float] × (2K+1),
      "ci_im":          [float] × (2K+1),
      "i_self":         int,                    # original token's spectrum index
      "assoc_index":    [(word, i, coef)],      # associated tokens + positions
      "weights_fs":     [float] × (2K+1),
      "attention":      [float] × (2K+1),
      "asym_indices":   [int],                  # tail positions with |c_im| > 0
      "asym_commitment":float,
      "accepted":       bool,
      "corrected":      bool,
      "signature":      hex16
    }
"""

from __future__ import annotations
import hashlib
import json
import math
import struct
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---- reuse factor.py primitives ----
from factor import (
    mobius_sieve,
    classify,
    quadratic_address,
    ALPHA_ODD,
    ALPHA_EVEN,
    DENSITY,
)

# ---- reuse derida-tokenizer primitives ----
from derida_tokenizer_norm import (
    ALPHA,
    A_STEP,
    GAMMA,
    DispersalGate,
    MOSBReadOnlyBuffer,
    finite_step_weight,
    attention_regions,
    spectral_derivative_weighted,
    DEFAULT_K,
    DEFAULT_SIGMA_FRAC,
    DEFAULT_TAIL_THRESHOLD,
)


# ============================================================
#  Hash projection  ·  word → spectrum index
# ============================================================
def word_to_index(word: str, K: int) -> int:
    """Deterministic projection of a word onto [-K, K], excluding 0."""
    h = int(hashlib.sha256(word.lower().encode()).hexdigest()[:8], 16)
    i = (h % (2 * K)) - K
    return i if i != 0 else 1


def _amp_from_hash(word: str, salt: bytes, K: int, i: int) -> float:
    """Deterministic per-(word, i) amplitude in [0, 1]."""
    digest = hashlib.shake_256(
        word.lower().encode() + salt + i.to_bytes(4, "big", signed=True)
    ).digest(4)
    return int.from_bytes(digest, "big") / (2 ** 32 - 1)


# ============================================================
#  Build a dual coefficient array for one token
# ============================================================
def build_token_coeff(word: str,
                      context: List[str],
                      K: int = DEFAULT_K
                      ) -> Tuple[np.ndarray, np.ndarray,
                                 int, List[Tuple[str, int, float]]]:
    """
    Build the dual |Ci| array for one token.

        real part  = c_re  →  the ORIGINAL token's coefficient
        imag part  = c_im  →  ASSOCIATED tokens' coefficients

    Returns
    -------
    c_re, c_im, i_self, assoc_index
    """
    mu = mobius_sieve(K)
    c_re = np.zeros(2 * K + 1, dtype=np.float64)
    c_im = np.zeros(2 * K + 1, dtype=np.float64)

    # ---- 1. original token on the REAL axis ----
    i_self = word_to_index(word, K)
    q_self = quadratic_address(abs(i_self), K)
    gated_self = 0.5 if (q_self < len(mu) and mu[q_self] == 0) else 1.0
    amp_self = 0.5 + 0.5 * _amp_from_hash(word, b"self", K, i_self)

    # Original token's own amplitude uses α_odd regardless of classify —
    # the real axis is reserved for "self" by definition of the layout.
    c_re[K + i_self] += ALPHA_ODD * amp_self * gated_self
    # symmetric mirror on the negative side
    c_re[K - i_self] += ALPHA_ODD * amp_self * gated_self

    # ---- 2. associated tokens on the IMAGINARY axis ----
    assoc_index: List[Tuple[str, int, float]] = []
    seen_indices: set = {i_self, -i_self}

    for w_a in context:
        if not w_a or w_a.lower() == word.lower():
            continue
        i_a = word_to_index(w_a, K)
        # collision → nudge into an open spectrum slot
        while i_a in seen_indices and len(seen_indices) < 2 * K - 2:
            i_a = (i_a + 1) % (2 * K)
            if i_a == 0:
                i_a = 1
            if i_a > K:
                i_a -= 2 * K
        if i_a in seen_indices:
            continue
        seen_indices.add(i_a)
        seen_indices.add(-i_a)

        q_a = quadratic_address(abs(i_a), K)
        gated_a = 0.5 if (q_a < len(mu) and mu[q_a] == 0) else 1.0
        amp_a = 0.5 + 0.5 * _amp_from_hash(w_a, b"assoc", K, i_a)

        coef = ALPHA_EVEN * amp_a * gated_a
        c_im[K + i_a] += coef
        c_im[K - i_a] += coef
        assoc_index.append((w_a, i_a, coef))

    return c_re, c_im, i_self, assoc_index


# ============================================================
#  Decode a |Ci| array back to tokens
# ============================================================
def decode_ci(c_re: np.ndarray,
              c_im: np.ndarray,
              K: int,
              top_k: int = 5
              ) -> Dict:
    """
    Reverse mapping from |Ci| to tokens.

        original token index = argmax_{i>0} c_re[K+i]
        associated indices   = top-k { i>0 : c_im[K+i] } by amplitude
    """
    n = 2 * K + 1

    re_slice = c_re[K + 1: K + 1 + K]
    im_slice = c_im[K + 1: K + 1 + K]

    i_self = int(np.argmax(re_slice)) + 1 if re_slice.size else 0
    self_coef = float(c_re[K + i_self]) if i_self else 0.0

    assoc: List[Tuple[int, float]] = []
    if im_slice.size:
        order = np.argsort(im_slice)[::-1][:top_k]
        for idx in order:
            val = float(im_slice[idx])
            if val > 1e-9:
                assoc.append((int(idx) + 1, val))

    return dict(
        i_self=i_self,
        self_coef=self_coef,
        assoc_positions=assoc,
        n_assoc=len(assoc),
    )


# ============================================================
#  The dual-coefficient tokenizer
# ============================================================
class DualCoeffTokenizer:
    def __init__(self,
                 K: int = DEFAULT_K,
                 policy: str = "rl_admit",
                 n_data_points: int = 3,
                 check_strength: float = 0.3):
        self.K = K
        self.gate = DispersalGate(policy, check_strength, n_data_points)
        self.mosb = MOSBReadOnlyBuffer(K=K)

        center, tail, weights = attention_regions(K)
        self.center = center
        self.tail = tail
        self.tail_set = set(tail)
        self.attention = weights

        self.cids: Dict[str, int] = {}
        self.chunks: Dict[str, dict] = {}
        self.index_to_word: Dict[int, str] = {}

    # ---- tokenize with context neighbours as embeddings ----
    def tokenize(self, query: str,
                 window: int = 3) -> List[str]:
        """
        Split into words, then for each word use its neighbours
        in the query as the "associated / embedding" set.
        """
        words = [w for w in query.lower().split() if w]
        if not words:
            return []

        for idx, w in enumerate(words):
            if w in self.chunks:
                continue
            lo = max(0, idx - window)
            hi = min(len(words), idx + window + 1)
            context = [words[j] for j in range(lo, hi) if j != idx]

            c_re, c_im, i_self, assoc_index = build_token_coeff(
                w, context, self.K)

            w_fs = finite_step_weight(c_re, c_im)

            # ---- asymmetric tail classification ----
            asym_indices: List[int] = []
            for i in range(-self.K, self.K + 1):
                if i == 0 or i not in self.tail_set:
                    continue
                if abs(c_im[i + self.K]) > 1e-9:
                    asym_indices.append(i)

            asym_commit = float(np.mean(
                [w_fs[i + self.K] for i in asym_indices]
            )) if asym_indices else 0.0

            accepted, c_after, corrected = self.gate.verify(asym_commit)

            rec = dict(
                word=w, K=self.K,
                i_self=i_self,
                ci_re=c_re.tolist(),
                ci_im=c_im.tolist(),
                assoc_index=[[wa, ia, ca] for wa, ia, ca in assoc_index],
                weights_fs=w_fs.tolist(),
                attention=self.attention.tolist(),
                asym_indices=asym_indices,
                asym_commitment=asym_commit,
                accepted=accepted,
                commitment_after=c_after,
                corrected=corrected,
                signature=hashlib.sha256(
                    w.encode() + c_re.tobytes()
                ).hexdigest()[:16],
            )
            self.chunks[w] = rec
            self.cids[w] = self.mosb.append(
                json.dumps(rec, separators=(",", ":")).encode())
            self.index_to_word[i_self] = w

    # ---- MOSB read ----
    def _read(self, word: str) -> Optional[dict]:
        cid = self.cids.get(word)
        if cid is None:
            return None
        chunk = self.mosb.read_chunk(cid)
        if chunk is None:
            return None
        return json.loads(chunk[0].decode())

    # ---- decode one token back to (self, assoc) ----
    def decode_token(self, word: str) -> Optional[Dict]:
        rec = self._read(word)
        if rec is None:
            return None
        c_re = np.asarray(rec["ci_re"])
        c_im = np.asarray(rec["ci_im"])
        info = decode_ci(c_re, c_im, self.K, top_k=5)
        return dict(
            word=word,
            i_self=info["i_self"],
            self_coef=info["self_coef"],
            assoc_positions=info["assoc_positions"],
            associated_words=[wa for wa, _ia, _ca in rec["assoc_index"]],
            asym_commitment=rec["asym_commitment"],
            accepted=rec["accepted"],
        )

    # ---- derivative query ----
    def derivative_query(self, t: float) -> Dict:
        """Same contract as derida-tokenizer, but now the real side
        carries the token identity and the imag side carries its
        associated neighbourhood."""
        total_re = 0.0
        total_im = 0.0
        per_word: List[Dict] = []
        n_blocked = 0
        n_accepted = 0

        for word in self.cids:
            rec = self._read(word)
            if rec is None:
                continue
            if rec["asym_indices"] and not rec["accepted"]:
                n_blocked += 1
                per_word.append(dict(
                    word=word, status="blocked",
                    commitment=rec["asym_commitment"],
                ))
                continue

            c_re = np.asarray(rec["ci_re"])
            c_im = np.asarray(rec["ci_im"])
            w_fs = np.asarray(rec["weights_fs"])

            d_re, d_im = spectral_derivative_weighted(
                t, c_re, c_im, w_fs)

            total_re += d_re
            total_im += d_im
            n_accepted += 1
            per_word.append(dict(
                word=word, status="accepted",
                d_re=d_re, d_im=d_im,
                mag=math.sqrt(d_re * d_re + d_im * d_im),
            ))

        return dict(
            t=t,
            n_tokens=len(self.cids),
            n_accepted=n_accepted, n_blocked=n_blocked,
            d_re=total_re, d_im=total_im,
            magnitude=math.sqrt(total_re * total_re + total_im * total_im),
            contributions=per_word,
        )

    # ---- summary / JSON export ----
    def summary(self) -> Dict:
        n_asym = sum(1 for c in self.chunks.values() if c["asym_indices"])
        n_blocked = sum(1 for c in self.chunks.values()
                        if c["asym_indices"] and not c["accepted"])
        return dict(
            K=self.K, n_tokens=len(self.cids),
            n_center=len(self.center), n_tail=len(self.tail),
            n_asym_tokens=n_asym, n_blocked=n_blocked,
            mosb_chunks=self.mosb.n_chunks,
            mosb_bytes=self.mosb.size_bytes,
            policy=self.gate.policy,
            check_eff=self.gate.check_eff,
        )

    def to_json(self) -> Dict:
        return dict(
            K=self.K, policy=self.gate.policy,
            center=self.center, tail=self.tail,
            n_tokens=len(self.cids),
            chunks=list(self.chunks.values()),
        )


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("Dual-coefficient tokenizer")
    print("  real  →  ORIGINAL token   (α = 0.3628)")
    print("  imag  →  ASSOCIATED tokens (α = 1/(π−e) ≈ 2.362)")
    print("=" * 78)
    print(f"  α_odd   = {ALPHA_ODD:.6f}")
    print(f"  α_even  = {ALPHA_EVEN:.6f}")
    print(f"  6/π²    = {DENSITY:.6f}")
    print()

    # ---------- 1. build one token's coefficient ----------
    print("--- 1 · single-token coefficient array ---")
    word = "sort"
    context = ["database", "by", "user", "id"]
    c_re, c_im, i_self, assoc = build_token_coeff(word, context, K=20)
    K = 20
    print(f"  original token   : {word!r}   i_self = {i_self:+d}")
    print(f"  c_re[{i_self:+d}]      = {c_re[K + i_self]:+.6f}    "
          f"(real, α = 0.3628)")
    print(f"  associated tokens and their imaginary coefficients:")
    for w_a, i_a, ca in assoc:
        print(f"    {w_a:<10s}  i = {i_a:+d}   "
              f"c_im[{i_a:+d}] = {ca:+.6f}")
    print()

    # decode it back
    info = decode_ci(c_re, c_im, K, top_k=5)
    print(f"  decode check:")
    print(f"    recovered self peak i = {info['i_self']:+d}   "
          f"(= {word!r}? {info['i_self'] == i_self})")
    print(f"    recovered assoc peaks = "
          f"{[p for p, _v in info['assoc_positions']]}")
    print(f"    expected assoc peaks  = "
          f"{[ia for _wa, ia, _ca in assoc]}")
    print()

    # ---------- 2. tokenize a query across all policies ----------
    for policy in ("blind", "cross_check", "rl_admit"):
        print(f"=== 2 · query 'sort the database by user id' "
              f"·  policy={policy} ===")
        tok = DualCoeffTokenizer(K=20, policy=policy, n_data_points=3)
        tok.tokenize("sort the database by user id", window=3)
        s = tok.summary()
        print(f"  K={s['K']}   tokens={s['n_tokens']}   "
              f"asym_tokens={s['n_asym_tokens']}   "
              f"blocked={s['n_blocked']}")
        print(f"  MOSB chunks={s['mosb_chunks']}   "
              f"bytes={s['mosb_bytes']}   check_eff={s['check_eff']:.4f}")
        print()

        # per-token dual decode
        print(f"  {'word':<12s}  {'i_self':>6s}  {'assoc words':<30s}  "
              f"{'accepted':>8s}")
        for w in tok.cids:
            d = tok.decode_token(w)
            assoc_w = ",".join(d["associated_words"][:4])
            if len(d["associated_words"]) > 4:
                assoc_w += "…"
            print(f"  {w:<12s}  {d['i_self']:>+6d}  "
                  f"{assoc_w:<30s}  {str(d['accepted']):>8s}")
        print()

        # derivative at three positions
        for t_val in (0.0, 1.5, 3.0):
            r = tok.derivative_query(t_val)
            print(f"  dζ/dt({t_val:>4.1f})  "
                  f"Re={r['d_re']:>+10.4f}  "
                  f"Im={r['d_im']:>+10.4f}  "
                  f"|.|={r['magnitude']:>8.4f}  "
                  f"acc={r['n_accepted']}  blk={r['n_blocked']}")
        print()

    # ---------- 3. explicit dual-layout for one word ----------
    print("--- 3 · |Ci| dual layout  ·  word = 'index'  K = 10 ---")
    K2 = 10
    c_re, c_im, i_self, assoc = build_token_coeff(
        "index", ["database", "user", "id"], K=K2)
    print(f"  {'i':>4s}  {'c_re':>10s}  {'c_im':>10s}  role")
    for i in range(-K2, K2 + 1):
        re = c_re[K2 + i]
        im = c_im[K2 + i]
        if abs(re) < 1e-12 and abs(im) < 1e-12:
            continue
        if i == i_self:
            role = "self (real)"
        elif abs(re) > 1e-12:
            role = "self-mirror"
        else:
            wa = next((wa for wa, ia, _ in assoc if ia == i), "?")
            role = f"assoc '{wa}'"
        print(f"  {i:>+4d}  {re:>+10.6f}  {im:>+10.6f}  {role}")
    print()

    # ---------- 4. JSON export ----------
    print("--- 4 · JSON export ---")
    tok = DualCoeffTokenizer(K=12, policy="rl_admit")
    tok.tokenize("hello world", window=2)
    payload = tok.to_json()
    jstr = json.dumps(payload, separators=(",", ":"))
    print(f"  export size    = {len(jstr)} bytes")
    print(f"  K              = {payload['K']}")
    print(f"  n_tokens       = {payload['n_tokens']}")
    print(f"  chunks[0] word = {payload['chunks'][0]['word']!r}   "
          f"i_self = {payload['chunks'][0]['i_self']:+d}")
    print(f"  chunks[0] assoc= "
          f"{[a[0] for a in payload['chunks'][0]['assoc_index']]}")
    print()

    # ---------- 5. complexity ----------
    print("--- 5 · complexity ---")
    print(f"  build_token_coeff       O(K)         (μ + hash + gate per i)")
    print(f"  finite_step_weight      O(K)")
    print(f"  asym classification     O(K)")
    print(f"  decode_ci               O(K)         (argmax + top-k)")
    print(f"  derivative query        O(n_tokens · K)")
    print(f"  MOSB append             O(K)         (JSON chunk)")
    print()
    print(f"  memory per token        ~{4 * (2 * DEFAULT_K + 1) * 2} bytes"
          f"  (ci_re + ci_im as float32, K = {DEFAULT_K})")
    print()
    print("Done.")


if __name__ == "__main__":
    demo()