#!/usr/bin/env python3
"""
latex_mobius_compiler.py

A Möbius cross‑compiler for LaTeX source.

Pipeline  (O(K log K) overall)
------------------------------
    LaTeX source
      → tokenize            (commands, braces, brackets, math, &, \\, …)
      → vocabulary          (unique (kind, text) → integer index)
      → signal              sig[i] = (idx+0.5)/V + 0.1·depth + 0.05·kind_w
      → logic gate          ('log', 'exp', 'sin', 'cos', 'derivative', 'none')
      → elliptic TSP route  (bucket sort by pseudo‑angle i·(π−e) mod 2π)
      → exponential kernel  conv = (f + b − sig)/(1 − λ²) → normalise
      → supertrace          S = Σ (−1)^i conv[i]
      → entropy + mass      H = −α·p·log(p),  m = |S|·e^(−H),  p = |S|/K
      → Möbius filter       keep top‑M coefficients with μ(idx) ≠ 0
      → fingerprint         SHA‑256 over the lossless sidecar
      → artifact (.mob)     JSON with vocab, kept coeffs, lossless payload

Decompilation
-------------
    lossless  : exact reconstruction from the zlib‑packed token stream
    lossy     : nearest‑neighbour reconstruction from kept coefficients

CLI
---
    latex_mobius_compiler.py compile   <file.tex> [out.mob]
    latex_mobius_compiler.py decompile <file.mob> [out.tex] [--lossy]
    latex_mobius_compiler.py info      <file.mob>
    latex_mobius_compiler.py demo
    latex_mobius_compiler.py interactive
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import struct
import sys
import zlib
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ============================================================
#  0. Constants
# ============================================================
PI          = math.pi
E           = math.e
ALPHA       = 1.0 / (PI - E)          # ≈ 2.362
ALPHA_USER  = 0.3628
A_STEP      = ALPHA / ALPHA_USER      # ≈ 6.511  finite‑derivative step
DENSITY     = 6.0 / (PI * PI)         # ≈ 0.6079  square‑free density
MAGIC       = b"MOBLATX1"


# ============================================================
#  1. Möbius sieve  (linear, O(K))
# ============================================================
def mobius_sieve(K: int) -> List[int]:
    """Return μ(0..K).  μ[0] is unused."""
    mu = [0] * (K + 1)
    if K >= 1:
        mu[1] = 1
    primes, is_comp = [], [False] * (K + 1)
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
            else:
                mu[i * p] = -mu[i]
    return mu


# ============================================================
#  2. LaTeX tokenizer
# ============================================================
@dataclass
class Token:
    kind: str
    text: str
    depth: int = 0


_CMD_RE = re.compile(r'\\([A-Za-z@]+|.)')


def tokenize_latex(src: str) -> List[Token]:
    """Tokenize LaTeX into a flat list of Tokens.  O(len(src))."""
    toks: List[Token] = []
    i, n, depth = 0, len(src), 0
    while i < n:
        c = src[i]

        # comment
        if c == '%':
            j = src.find('\n', i)
            if j < 0:
                j = n
            toks.append(Token('comment', src[i:j], depth))
            i = j
            continue

        # command / \\ newline
        if c == '\\':
            if i + 1 < n and src[i + 1] == '\\':
                toks.append(Token('newline', '\\\\', depth))
                i += 2
                continue
            m = _CMD_RE.match(src, i)
            if m:
                toks.append(Token('command', m.group(0), depth))
                i = m.end()
                continue

        # braces
        if c == '{':
            toks.append(Token('brace_open', '{', depth))
            depth += 1
            i += 1
            continue
        if c == '}':
            depth = max(0, depth - 1)
            toks.append(Token('brace_close', '}', depth))
            i += 1
            continue

        # brackets
        if c == '[':
            toks.append(Token('bracket_open', '[', depth)); i += 1; continue
        if c == ']':
            toks.append(Token('bracket_close', ']', depth)); i += 1; continue

        # math
        if c == '$':
            if i + 1 < n and src[i + 1] == '$':
                toks.append(Token('math_dd', '$$', depth)); i += 2
            else:
                toks.append(Token('math_d', '$', depth)); i += 1
            continue

        # alignment
        if c == '&':
            toks.append(Token('amp', '&', depth)); i += 1; continue

        # whitespace
        if c.isspace():
            j = i
            while j < n and src[j].isspace():
                j += 1
            toks.append(Token('space', src[i:j], depth)); i = j; continue

        # word
        if c.isalpha():
            j = i
            while j < n and src[j].isalpha():
                j += 1
            toks.append(Token('word', src[i:j], depth)); i = j; continue

        # number
        if c.isdigit():
            j = i
            while j < n and (src[j].isdigit() or src[j] == '.'):
                j += 1
            toks.append(Token('number', src[i:j], depth)); i = j; continue

        # other
        toks.append(Token('other', c, depth)); i += 1

    return toks


# ============================================================
#  3. Vocabulary + signal encoding
# ============================================================
KIND_WEIGHT = {
    'command': 1.00, 'brace_open': 0.85, 'brace_close': 0.85,
    'bracket_open': 0.70, 'bracket_close': 0.70,
    'math_d': 0.95, 'math_dd': 0.95, 'amp': 0.60, 'newline': 0.55,
    'space': 0.20, 'word': 0.40, 'number': 0.45,
    'comment': 0.30, 'other': 0.15,
}
_KIND_CODES = list(KIND_WEIGHT.keys())
_KIND_TO_CODE = {k: i for i, k in enumerate(_KIND_CODES)}


class TokenVocab:
    def __init__(self) -> None:
        self._idx: Dict[Tuple[str, str], int] = {}
        self._inv: List[Tuple[str, str]] = []

    def add(self, tok: Token) -> int:
        key = (tok.kind, tok.text)
        i = self._idx.get(key)
        if i is None:
            i = len(self._inv)
            self._idx[key] = i
            self._inv.append(key)
        return i

    def lookup(self, tok: Token) -> int:
        return self._idx.get((tok.kind, tok.text), -1)

    def get(self, i: int) -> Optional[Tuple[str, str]]:
        return self._inv[i] if 0 <= i < len(self._inv) else None

    def __len__(self) -> int:
        return len(self._inv)

    def to_list(self) -> List[List[str]]:
        return [[k, t] for k, t in self._inv]

    @classmethod
    def from_list(cls, lst: List[List[str]]) -> "TokenVocab":
        v = cls()
        for k, t in lst:
            v._idx[(k, t)] = len(v._inv)
            v._inv.append((k, t))
        return v


def encode_signal(tokens: List[Token], vocab: TokenVocab) -> np.ndarray:
    K, V = len(tokens), max(1, len(vocab))
    sig = np.empty(K, dtype=np.float64)
    for i, t in enumerate(tokens):
        idx = vocab.lookup(t)
        w = KIND_WEIGHT.get(t.kind, 0.5)
        sig[i] = (idx + 0.5) / V + 0.1 * t.depth + 0.05 * w
    return sig


def decode_signal(sig: np.ndarray, vocab: TokenVocab,
                  meta: List[Tuple[str, int]]) -> List[Token]:
    V = max(1, len(vocab))
    out: List[Token] = []
    for i, val in enumerate(sig):
        if i >= len(meta):
            break
        kind, depth = meta[i]
        w = KIND_WEIGHT.get(kind, 0.5)
        residual = val - 0.1 * depth - 0.05 * w
        idx = int(round(residual * V - 0.5))
        idx = max(0, min(idx, V - 1))
        _, text = vocab.get(idx) or ('other', '')
        out.append(Token(kind, text, depth))
    return out


# ============================================================
#  4. Chip pipeline
# ============================================================
def elliptic_permutation(K: int, w1: float = PI, w2: float = E) -> np.ndarray:
    """TSP routing by pseudo‑angle.  O(K log K)."""
    if K <= 1:
        return np.arange(K)
    idx = np.arange(K)
    angles = (idx * (w1 - w2)) % (2 * PI)
    return np.argsort(angles)


def conv_exp_kernel(sig: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """Two‑pass exponential convolution.  O(K)."""
    K = len(sig)
    if K == 0:
        return sig.copy()
    lam = math.exp(-alpha)
    f = np.empty(K, dtype=np.float64)
    f[0] = sig[0]
    for i in range(1, K):
        f[i] = sig[i] + lam * f[i - 1]
    b = np.empty(K, dtype=np.float64)
    b[K - 1] = sig[K - 1]
    for i in range(K - 2, -1, -1):
        b[i] = sig[i] + lam * b[i + 1]
    den = 1.0 - lam * lam
    norm = 1.0 - math.exp(-alpha * (PI + E))
    conv = (f + b - sig) / den
    return (1.0 - conv) / norm


def supertrace_and_entropy(sig: np.ndarray) -> Tuple[float, float, float]:
    K = len(sig)
    S = 0.0
    for i, v in enumerate(sig):
        S += v if (i % 2 == 0) else -v
    if K == 0 or S == 0.0:
        return S, 0.0, 0.0
    p = abs(S) / K
    if p <= 0.0 or p >= 1.0:
        return S, 0.0, abs(S)
    H = -ALPHA * p * math.log(p)
    m = abs(S) * math.exp(-H) if H < 700 else 0.0
    return S, H, m


def apply_gate(sig: np.ndarray, gate: str) -> np.ndarray:
    if gate == 'log':
        return np.log(np.maximum(np.abs(sig), 1e-12))
    if gate == 'exp':
        return np.exp(np.clip(sig, -50, 50))
    if gate == 'sin':
        return np.sin(sig)
    if gate == 'cos':
        return np.cos(sig)
    if gate == 'derivative':
        out = np.zeros_like(sig)
        if len(sig) >= 2:
            out[:-1] = (sig[1:] - sig[:-1]) / A_STEP
        return out
    return sig.copy()


def compress(sig: np.ndarray, mu: List[int], use_elliptic: bool = True
             ) -> Tuple[List[Tuple[int, float]], float, float, float, np.ndarray]:
    K = len(sig)
    if K == 0:
        return [], 0.0, 0.0, 0.0, sig.copy()

    order = elliptic_permutation(K) if use_elliptic else np.arange(K)
    sig_r = sig[order]
    conv = conv_exp_kernel(sig_r)
    S, H, m = supertrace_and_entropy(conv)

    M = max(1, min(int(abs(S)), K))
    mag = np.abs(conv)
    order_mag = np.argsort(mag)[::-1]
    kept: List[Tuple[int, float]] = []
    for idx in order_mag:
        n = int(idx) + 1
        if n <= len(mu) - 1 and mu[n] != 0:
            kept.append((int(idx), float(conv[idx])))
            if len(kept) >= M:
                break
    return kept, S, H, m, conv


# ============================================================
#  5. Artifact
# ============================================================
@dataclass
class CompiledArtifact:
    fingerprint: str
    K: int
    gate: str
    use_elliptic: bool
    S: float
    H: float
    m: float
    vocab: List[List[str]]
    token_kinds: List[str]
    token_depths: List[int]
    kept: List[List[float]]
    lossless_b64: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "CompiledArtifact":
        d = json.loads(s)
        obj = cls(**d)
        subs = obj.extra.get('sub_artifacts')
        if subs:
            obj.extra['sub_artifacts'] = [cls(**sub) for sub in subs]
        return obj

    def size_bytes(self) -> int:
        return len(self.to_json().encode('utf-8'))


def compute_fingerprint(vocab: List[List[str]],
                        token_indices: List[int],
                        token_kinds: List[str],
                        token_depths: List[int]) -> str:
    h = hashlib.sha256()
    h.update(MAGIC)
    for k, t in vocab:
        h.update(k.encode()); h.update(b'\x00')
        h.update(t.encode()); h.update(b'\x00')
    for idx, k, d in zip(token_indices, token_kinds, token_depths):
        h.update(struct.pack('>I', idx))
        h.update(k.encode()); h.update(b'\x00')
        h.update(struct.pack('>H', d))
    return h.hexdigest()


def lossless_pack(idxs: List[int], kinds: List[str], depths: List[int]) -> str:
    buf = bytearray()
    for i, k, d in zip(idxs, kinds, depths):
        v = i
        while True:
            b = v & 0x7F
            v >>= 7
            if v:
                buf.append(b | 0x80)
            else:
                buf.append(b)
                break
        buf.append(_KIND_TO_CODE.get(k, 255))
        buf.append(d & 0xFF)
    return base64.b64encode(zlib.compress(bytes(buf), 9)).decode('ascii')


def lossless_unpack(payload_b64: str) -> Tuple[List[int], List[str], List[int]]:
    raw = zlib.decompress(base64.b64decode(payload_b64))
    idxs, kinds, depths = [], [], []
    i, n = 0, len(raw)
    while i < n:
        shift, v = 0, 0
        while True:
            b = raw[i]; i += 1
            v |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7
        kc = raw[i]; i += 1
        kind = _KIND_CODES[kc] if kc < len(_KIND_CODES) else 'other'
        d = raw[i]; i += 1
        idxs.append(v); kinds.append(kind); depths.append(d)
    return idxs, kinds, depths


# ============================================================
#  6. Compiler
# ============================================================
class LatexMobiusCompiler:
    def __init__(self, max_K: int = 8192, gate: str = 'log',
                 use_elliptic: bool = True, lossless: bool = True) -> None:
        self.max_K = max_K
        self.gate = gate
        self.use_elliptic = use_elliptic
        self.lossless = lossless
        self.mu = mobius_sieve(max_K)

    # ---------- one chunk ----------
    def _compile_chunk(self, tokens: List[Token]) -> CompiledArtifact:
        vocab = TokenVocab()
        for t in tokens:
            vocab.add(t)

        sig = encode_signal(tokens, vocab)
        sig_g = apply_gate(sig, self.gate)
        kept, S, H, m, _ = compress(sig_g, self.mu, use_elliptic=self.use_elliptic)

        idxs = [vocab.lookup(t) for t in tokens]
        kinds = [t.kind for t in tokens]
        depths = [t.depth for t in tokens]
        fp = compute_fingerprint(vocab.to_list(), idxs, kinds, depths)
        payload = lossless_pack(idxs, kinds, depths) if self.lossless else None

        return CompiledArtifact(
            fingerprint=fp, K=len(tokens), gate=self.gate,
            use_elliptic=self.use_elliptic,
            S=S, H=H, m=m,
            vocab=vocab.to_list(),
            token_kinds=kinds, token_depths=depths,
            kept=[[i, v] for i, v in kept],
            lossless_b64=payload,
            extra={'n_vocab': len(vocab), 'kept_count': len(kept),
                   'density': DENSITY, 'a_step': A_STEP},
        )

    # ---------- public ----------
    def compile(self, latex_src: str) -> CompiledArtifact:
        tokens = tokenize_latex(latex_src)
        if len(tokens) <= self.max_K:
            return self._compile_chunk(tokens)
        chunks = [tokens[i:i + self.max_K]
                  for i in range(0, len(tokens), self.max_K)]
        art = self._compile_chunk(chunks[0])
        art.extra['sub_artifacts'] = [self._compile_chunk(c) for c in chunks[1:]]
        return art

    def compile_file(self, path: str) -> CompiledArtifact:
        with open(path, 'r', encoding='utf-8') as f:
            return self.compile(f.read())

    def decompile(self, art: CompiledArtifact, lossless: Optional[bool] = None) -> str:
        use_lossless = self.lossless if lossless is None else lossless

        if use_lossless and art.lossless_b64:
            idxs, kinds, depths = lossless_unpack(art.lossless_b64)
            vocab = TokenVocab.from_list(art.vocab)
            parts = []
            for i, k, d in zip(idxs, kinds, depths):
                entry = vocab.get(i)
                parts.append(entry[1] if entry else '')
            src = ''.join(parts)
            for sub in art.extra.get('sub_artifacts', []):
                src += self.decompile(sub, lossless=True)
            return src

        # ---- lossy ----
        K = art.K
        sig_r = np.zeros(K, dtype=np.float64)
        for i, v in art.kept:
            if 0 <= i < K:
                sig_r[i] = v
        if art.use_elliptic:
            order = elliptic_permutation(K)
            sig = np.zeros(K, dtype=np.float64)
            sig[order] = sig_r
        else:
            sig = sig_r
        if art.gate == 'log':
            sig = np.exp(sig)
        elif art.gate == 'exp':
            sig = np.log(np.maximum(np.abs(sig), 1e-12))
        vocab = TokenVocab.from_list(art.vocab)
        meta = list(zip(art.token_kinds, art.token_depths))
        toks = decode_signal(sig, vocab, meta)
        return ''.join(t.text for t in toks)

    # ---------- artifact I/O ----------
    def write_artifact(self, art: CompiledArtifact, path: str) -> int:
        s = art.to_json().encode('utf-8')
        with open(path, 'wb') as f:
            f.write(s)
        return len(s)

    def read_artifact(self, path: str) -> CompiledArtifact:
        with open(path, 'r', encoding='utf-8') as f:
            return CompiledArtifact.from_json(f.read())


# ============================================================
#  7. CLI helpers
# ============================================================
def _print_stats(art: CompiledArtifact, src_bytes: int) -> None:
    art_bytes = art.size_bytes()
    ratio = art_bytes / max(1, src_bytes)
    tag = 'compressed' if ratio < 1 else 'expanded'
    print(f"  K               : {art.K}")
    print(f"  Gate            : {art.gate}")
    print(f"  Elliptic TSP    : {art.use_elliptic}")
    print(f"  Supertrace S    : {art.S:+.6f}")
    print(f"  Entropy H       : {art.H:.6f}")
    print(f"  Mass m          : {art.m:.6f}")
    print(f"  Vocabulary size : {len(art.vocab)}")
    print(f"  Kept coeffs     : {len(art.kept)} "
          f"(≈ {100*len(art.kept)/max(1,art.K):.1f} % of K)")
    print(f"  Fingerprint     : {art.fingerprint}")
    print(f"  Source bytes    : {src_bytes:,}")
    print(f"  Artifact bytes  : {art_bytes:,}")
    print(f"  Ratio           : {ratio:.3f}×  ({tag})")


def _cmd_compile(args: List[str]) -> None:
    if not args:
        print("usage: compile <file.tex> [out.mob]"); return
    src, out = args[0], (args[1] if len(args) > 1 else args[0] + '.mob')
    comp = LatexMobiusCompiler(max_K=8192, gate='log', lossless=True)
    art = comp.compile_file(src)
    comp.write_artifact(art, out)
    print(f"Compiled {src} → {out}")
    _print_stats(art, os.path.getsize(src))


def _cmd_decompile(args: List[str]) -> None:
    if not args:
        print("usage: decompile <file.mob> [out.tex] [--lossy]"); return
    src, lossy = args[0], '--lossy' in args
    out = next((a for a in args[1:] if not a.startswith('--')),
               src + ('.lossy.tex' if lossy else '.tex'))
    comp = LatexMobiusCompiler()
    art = comp.read_artifact(src)
    text = comp.decompile(art, lossless=not lossy)
    with open(out, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Decompiled {src} → {out}  "
          f"({'lossy' if lossy else 'lossless'})")


def _cmd_info(args: List[str]) -> None:
    if not args:
        print("usage: info <file.mob>"); return
    comp = LatexMobiusCompiler()
    art = comp.read_artifact(args[0])
    _print_stats(art, src_bytes=art.extra.get('src_bytes', 0) or art.size_bytes())


def _cmd_demo(_args: List[str]) -> None:
    src = r"""
\documentclass{article}
\usepackage{amsmath, amssymb}
\title{A Möbius Compiler for \LaTeX}
\author{Compiler Demo}
\date{\today}

\begin{document}
\maketitle

\section{Introduction}
The Möbius sieve $\mu(n)$ classifies each index as
square-free, prime-squared, or higher-power.
We use it to compress a \LaTeX{} token stream.

\section{Math}
\begin{equation}
  \zeta(s) = \sum_{n=1}^{\infty} \frac{1}{n^s}
  \quad\text{and}\quad
  \sum_{n=1}^{\infty} \frac{\mu(n)}{n^2} = \frac{6}{\pi^2}.
\end{equation}

\section{Repetition}
\begin{itemize}
  \item \textbf{First} item.
  \item \textbf{Second} item.
  \item \textbf{Third} item.
\end{itemize}

\end{document}
""".strip()

    print("=" * 68)
    print("LaTeX Möbius compiler  ·  demo")
    print("=" * 68)
    print(f"\nSource ({len(src)} bytes):")
    print("-" * 68)
    print(src)
    print("-" * 68)

    comp = LatexMobiusCompiler(max_K=8192, gate='log', lossless=True)
    art = comp.compile(src)

    print("\nCompilation stats:")
    _print_stats(art, src_bytes=len(src.encode('utf-8')))

    out_l = comp.decompile(art, lossless=True)
    print("\nLossless roundtrip:",
          "OK" if out_l == src else "MISMATCH")
    if out_l != src:
        for i, (a, b) in enumerate(zip(src, out_l)):
            if a != b:
                print(f"  first diff at char {i}: {a!r} vs {b!r}")
                break

    out_y = comp.decompile(art, lossless=False)
    common = sum(1 for a, b in zip(src, out_y) if a == b)
    print(f"Lossy reconstruction: {common}/{len(src)} chars match "
          f"({100*common/len(src):.1f} %)")
    print("\nLossy preview (first 200 chars):")
    print("-" * 68)
    print(out_y[:200])
    print("-" * 68)


def _cmd_interactive(_args: List[str]) -> None:
    comp = LatexMobiusCompiler(max_K=8192, gate='log', lossless=True)
    print("Interactive LaTeX Möbius compiler.")
    print("Commands: compile <file> | decompile <file.mob> [out.tex] [--lossy] |")
    print("          info <file.mob> | demo | quit")
    while True:
        try:
            line = input("\nmob> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not line:
            continue
        parts = line.split()
        cmd, rest = parts[0], parts[1:]
        if cmd in ('quit', 'exit', 'q'):
            break
        if cmd == 'compile':
            _cmd_compile(rest)
        elif cmd == 'decompile':
            _cmd_decompile(rest)
        elif cmd == 'info':
            _cmd_info(rest)
        elif cmd == 'demo':
            _cmd_demo(rest)
        else:
            print(f"unknown command: {cmd}")


def main(argv: List[str]) -> None:
    if not argv:
        _cmd_demo([])
        return
    cmd, rest = argv[0], argv[1:]
    dispatch = {
        'compile':   _cmd_compile,
        'decompile': _cmd_decompile,
        'info':      _cmd_info,
        'demo':      _cmd_demo,
        'interactive': _cmd_interactive,
    }
    fn = dispatch.get(cmd)
    if fn is None:
        print(f"unknown command: {cmd}")
        print("usage: latex_mobius_compiler.py "
              "{compile|decompile|info|demo|interactive} ...")
        return
    fn(rest)


if __name__ == '__main__':
    main(sys.argv[1:])