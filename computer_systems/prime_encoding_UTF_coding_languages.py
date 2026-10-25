#!/usr/bin/env python3
"""
coding_language_utf8.py
=======================

UTF-8-style encoding for coding languages with a function/modular split.

    Function tokens   →  prime index p,  real only  (i = 0),
                         α_func = 0.3628   (implicit in  -H_{p+1})
    Modular tokens    →  composite index n with μ(n) ≠ 0,
                         complex  -H_{n+1} + i·α_asym·(μ*H)(n),
                         α_asym = 1/(π − e)

    |Ci| array  (length 2K+1)
        c[K − p]   real     ← functions      (left / symmetric side)
        c[K + n]   complex  ← modular parts  (right / asymmetric side)

    Finite-step normalisation
        A_STEP = α_asym / 0.3628 ≈ 6.511
        c_norm.imag = c.imag / A_STEP
        normalises the asymmetric (modular) code to the function scale

    Stored vector for user input
        mod_vec = c_norm[K+1:].imag
        this is the "modular part" buffer
"""

from __future__ import annotations
import hashlib
import math
import re
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---- reuse the UTF-8 table machinery ----
from UTF_8_dzeta_dt import (
    mobius_sieve,
    harmonic_numbers,
    mu_convolution_H,
    utf8_byte_to_signed_index,
    build_particle_cloud,
    ugc_survivors,
    UTF8Table,
    ReadOnlyBuffer,
    ReadOnlyALU,
    ALPHA as ALPHA_MOD,          # 1/(π-e) ≈ 2.362
    ALPHA_ASYM as ALPHA_FUNC,    # 0.3628
    A_STEP,                      # ≈ 6.511
    K_DEFAULT as K_UTF8_DEFAULT,
)

PI = math.pi
E = math.e


# ============================================================
#  Tokenizer
# ============================================================
FUNCTION_KEYWORDS = frozenset({
    'def', 'class', 'lambda', 'function', 'fn', 'func',
    'return', 'yield',
    'if', 'elif', 'else', 'for', 'while',
    'try', 'except', 'finally', 'with', 'as',
    'import', 'from', 'async', 'await',
    'match', 'case', 'switch', 'do', 'in', 'of',
    'struct', 'enum', 'interface', 'impl', 'trait',
    'public', 'private', 'protected', 'static',
    'new', 'delete', 'void', 'int', 'float', 'string', 'bool',
})

_TOKEN_PATTERN = re.compile(
    r'"(?:[^"\\]|\\.)*"'            # double-quoted string literal
    r"|'(?:[^'\\]|\\.)*'"            # single-quoted string literal
    r'|[A-Za-z_][A-Za-z0-9_]*'       # identifier / keyword
    r'|\d+\.?\d*(?:[eE][+-]?\d+)?'   # numeric literal
    r'|[^\s\w]',                     # punctuation / operator
    re.UNICODE,
)


def _classify(token: str, next_token: str) -> str:
    t = token.strip()
    if not t:
        return 'modular'
    if t in FUNCTION_KEYWORDS:
        return 'function'
    # identifier followed by '(' is a call
    if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', t) and next_token == '(':
        return 'function'
    return 'modular'


def tokenize_source(source: str) -> List[Tuple[str, str]]:
    """Return list of (token, 'function'|'modular')."""
    matches = list(_TOKEN_PATTERN.finditer(source))
    tokens: List[Tuple[str, str]] = []
    for i, m in enumerate(matches):
        tok = m.group()
        nxt = matches[i + 1].group() if i + 1 < len(matches) else ''
        tokens.append((tok, _classify(tok, nxt)))
    return tokens


# ============================================================
#  Coding-language UTF-8 table
# ============================================================
class CodingLanguageTable:
    """
    Precomputed lookup tables:
        primes        :  p ∈ [2, K]  with  μ(p) = −1
        composites_sf :  n ∈ [4, K]  with  μ(n) ≠ 0
        H, F, mu      :  shared with UTF-8 pipeline
    """

    def __init__(self, K: int = K_UTF8_DEFAULT):
        self.K = int(K)
        self.F, self.mu, self.H = mu_convolution_H(K + 2)

        self.primes = np.array(
            [n for n in range(2, K + 1) if self.mu[n] == -1],
            dtype=np.int64)
        self.composites_sf = np.array(
            [n for n in range(4, K + 1) if self.mu[n] != 0],
            dtype=np.int64)

        if self.primes.size == 0 or self.composites_sf.size == 0:
            raise ValueError(f"K={K} too small to populate prime/composite sets")

    # ---- deterministic token → index mappings ----
    def prime_index(self, token: str) -> int:
        h = int(hashlib.md5(token.encode()).hexdigest()[:8], 16)
        return int(self.primes[h % self.primes.size])

    def composite_index(self, token: str) -> int:
        h = int(hashlib.md5(token.encode()).hexdigest()[:8], 16)
        return int(self.composites_sf[h % self.composites_sf.size])

    # ---- table accessors ----
    def H_at(self, n: int) -> float:
        if 0 <= n < len(self.H):
            return float(self.H[n])
        return 0.0

    def F_at(self, n: int) -> float:
        if 0 <= n < len(self.F):
            return float(self.F[n])
        return 0.0


# ============================================================
#  Encoding result
# ============================================================
@dataclass
class CodingEncoding:
    c:           np.ndarray          # raw |Ci|, complex, length 2K+1
    c_norm:      np.ndarray          # imag normalised by A_STEP
    n_function:  int
    n_modular:   int
    func_vec:    np.ndarray          # real function encoding  c[:K].real
    mod_vec:     np.ndarray          # stored modular vector   c_norm[K+1:].imag
    source_hash: str

    def func_energy(self) -> float:
        return float(np.sum(self.func_vec ** 2))

    def mod_energy(self) -> float:
        return float(np.sum(self.mod_vec ** 2))

    def func_to_mod_ratio(self) -> float:
        me = self.mod_energy()
        if me < 1e-12:
            return float('inf')
        return math.sqrt(self.func_energy() / me)

    def supertrace(self) -> float:
        S = 0.0
        for i, v in enumerate(self.c_norm):
            sign = 1.0 if (i % 2 == 0) else -1.0
            S += sign * float(v.real)
        return S

    def to_buffer(self) -> Tuple[ReadOnlyBuffer, int]:
        """Store the modular vector in a MOSB read-only buffer."""
        buf = ReadOnlyBuffer(K=max(len(self.mod_vec), 1))
        cid = buf.append(self.mod_vec.astype(np.float64).tobytes())
        return buf, cid


# ============================================================
#  Encoder
# ============================================================
def encode_program(source: str,
                   table: Optional[CodingLanguageTable] = None,
                   normalize_by_astep: bool = True) -> CodingEncoding:
    """
    Encode a coding-language program.

        function token t   →  c[K − p] += −H_{p+1}               (real)
        modular  token t   →  c[K + n] += −H_{n+1} + i·α·F[n]    (complex)

    The modular imaginary part is then divided by A_STEP, which
    places it on the same scale as the function real part
    (α_func = 0.3628).
    """
    tbl = table or CodingLanguageTable()
    K = tbl.K
    tokens = tokenize_source(source)

    c = np.zeros(2 * K + 1, dtype=np.complex128)
    n_func = n_mod = 0

    for tok, cat in tokens:
        if cat == 'function':
            p = tbl.prime_index(tok)
            v = -tbl.H_at(p + 1)                 # real, α_func implicit
            c[K - p] += v                         # left side (symmetric)
            n_func += 1
        else:
            n = tbl.composite_index(tok)
            v_re = -tbl.H_at(n + 1)
            v_im = ALPHA_MOD * tbl.F_at(n)        # α_asym = 1/(π-e)
            c[K + n] += complex(v_re, v_im)       # right side (asymmetric)
            n_mod += 1

    # finite-step normalisation of the asymmetric code
    c_norm = c.copy()
    if normalize_by_astep:
        c_norm.imag /= A_STEP

    func_vec = c[:K].real.copy()
    mod_vec = c_norm[K + 1:].imag.copy()

    return CodingEncoding(
        c=c, c_norm=c_norm,
        n_function=n_func, n_modular=n_mod,
        func_vec=func_vec, mod_vec=mod_vec,
        source_hash=hashlib.sha256(source.encode()).hexdigest(),
    )


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("Coding-language UTF-8  ·  function(prime,real) / modular(composite,imag)")
    print("=" * 78)
    print(f"  α_func  = 0.3628                      (functions,  i = 0)")
    print(f"  α_asym  = 1/(π − e) = {ALPHA_MOD:.6f}    (modular,  complex)")
    print(f"  A_STEP  = α_asym / 0.3628 = {A_STEP:.6f}  (finite-step normaliser)")
    print()

    tbl = CodingLanguageTable(K=K_UTF8_DEFAULT)
    print(f"--- table  ·  K = {tbl.K} ---")
    print(f"  primes           = {tbl.primes.size:>3d}  "
          f"(first: {tbl.primes[:8].tolist()})")
    print(f"  composites μ≠0   = {tbl.composites_sf.size:>3d}  "
          f"(first: {tbl.composites_sf[:8].tolist()})")
    print()

    # ---- 4 test programs ----
    programs = {
        "function only":
            "def compute(x): return x + 1",
        "modular only":
            '"hello world" 42 3.14',
        "mixed":
            'def greet(name):\n'
            '    return "Hello, " + name\n'
            'greet("Alice")',
        "python class":
            'class Net:\n'
            '    def forward(self, x):\n'
            '        return x * 2',
    }

    for label, src in programs.items():
        print("=" * 70)
        print(f"--- {label}  ·  {len(src)} chars ---")
        print(f"  source: {src[:70]!r}{'…' if len(src) > 70 else ''}")
        enc = encode_program(src, table=tbl)

        n_tok = enc.n_function + enc.n_modular
        print(f"  tokens            = {n_tok:>3d}  "
              f"(func={enc.n_function:>2d}, mod={enc.n_modular:>2d})")
        print(f"  func energy       = {enc.func_energy():>12.6f}")
        print(f"  mod energy        = {enc.mod_energy():>12.6f}  "
              f"(after ÷A_STEP)")
        ratio = enc.func_to_mod_ratio()
        ratio_s = "∞" if ratio == float('inf') else f"{ratio:.4f}"
        print(f"  func/mod ratio    = {ratio_s}")
        print(f"  supertrace        = {enc.supertrace():>+12.4f}")
        print(f"  |c.real| max      = {np.abs(enc.c.real).max():>12.4f}")
        print(f"  |c.imag| max      = {np.abs(enc.c.imag).max():>12.4f}")
        print()

    # ---- inspect mixed |Ci| array ----
    print("=" * 70)
    print("--- mixed example  ·  |Ci| array inspection ---")
    src = programs["mixed"]
    enc = encode_program(src, table=tbl)
    K = tbl.K

    nz = np.where(np.abs(enc.c) > 1e-9)[0]
    print(f"  non-zero |Ci| entries  = {len(nz)} / {len(enc.c)}")
    print()
    print(f"  {'idx':>4}  {'n':>4}  {'Re(c)':>12}  {'Im(c)':>12}  "
          f"{'kind':>9}")
    print(f"  {'-'*4}  {'-'*4}  {'-'*12}  {'-'*12}  {'-'*9}")
    for idx in nz[:14]:
        n = idx - K
        re = float(enc.c[idx].real)
        im = float(enc.c[idx].imag)
        kind = 'function' if n < 0 else ('modular' if n > 0 else 'zero')
        print(f"  {idx:>4d}  {n:>+4d}  {re:>+12.4f}  {im:>+12.4f}  "
              f"{kind:>9s}")

    # ---- finite-step normalisation ----
    print()
    print("--- finite-step normalisation  ·  A_STEP ---")
    print(f"  A_STEP                  = {A_STEP:.6f}")
    print(f"  |c.imag| max (before)   = {np.abs(enc.c.imag).max():.4f}")
    print(f"  |c.imag| max (after)    = {np.abs(enc.c_norm.imag).max():.4f}")
    scale = (np.abs(enc.c.imag).max()
             / max(np.abs(enc.c_norm.imag).max(), 1e-12))
    print(f"  observed scale factor   = {scale:.4f}")
    print(f"  = α_asym / α_func       = {ALPHA_MOD / 0.3628:.4f}  ✓")

    # ---- store modular vector in ReadOnlyBuffer ----
    print()
    print("--- store modular vector in a MOSB buffer ---")
    buf, cid = enc.to_buffer()
    chunk = buf.read(cid)
    if chunk is not None:
        payload, S, H, m = chunk
        print(f"  cid                     = {cid}")
        print(f"  payload bytes           = {len(payload)}")
        print(f"  supertrace S            = {S:+.6f}")
        print(f"  entropy H               = {H:.6f}")
        print(f"  mass m                  = {m:.4e}")

    # ---- UGC boolean on the encoding ----
    print()
    print("--- UGC boolean on the |Ci| array ---")
    table_utf8 = UTF8Table.build(K=K)
    particles = build_particle_cloud(table_utf8, src.encode("utf-8"))
    surv, non = ugc_survivors(particles, K, table_utf8.mu, 4, 0.3)
    rate = len(surv) / max(1, len(surv) + len(non))
    print(f"  particles              = {len(particles)}")
    print(f"  UGC survivors          = {len(surv)}")
    print(f"  UGC non-survivors      = {len(non)}")
    print(f"  survival rate          = {rate:.4f}")

    # ---- summary table ----
    print()
    print("=" * 70)
    print("--- summary ---")
    print(f"  {'program':<20}  {'tokens':>6s}  {'func':>5s}  {'mod':>5s}  "
          f"{'F/M':>8s}")
    for label, src in programs.items():
        enc = encode_program(src, table=tbl)
        ratio = enc.func_to_mod_ratio()
        ratio_s = "∞" if ratio == float('inf') else f"{ratio:.3f}"
        print(f"  {label:<20}  {enc.n_function+enc.n_modular:>6d}  "
              f"{enc.n_function:>5d}  {enc.n_modular:>5d}  "
              f"{ratio_s:>8s}")
    print()

    # ---- complexity ----
    print("--- complexity ---")
    print("  tokenize              O(|source|)")
    print("  prime/composite map   O(1)   hash → index")
    print("  |Ci| build            O(|tokens|)")
    print("  finite-step norm      O(K)")
    print("  UGC boolean           O(|tokens| · K)")
    print("  ─────────────────────────────")
    print("  total                 O(|source| · K)")
    print()
    print("Done.")


if __name__ == "__main__":
    demo()