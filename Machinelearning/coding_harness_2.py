#!/usr/bin/env python3
"""
coding_harness.py

A coding harness that parses source code, classifies each token as
**symmetric (deterministic)** or **asymmetric (user/data)**, and
routes them to the appropriate compression pipeline:

    symmetric tokens     (brackets, keywords, operators, def/class)
        → ASCII-like Möbius compression      O(K log log K)

    asymmetric tokens    (user identifiers, literals, function bodies)
        → |Ci| embedding + inverse-score sort O(K log K)

    linear-regression   (algebraic entries of the signature matrix)
        → dual-axis regression                O(K log K)

    bug counting         (symmetric good vs asymmetric bug trace)
        → dual gate check                     O(K log K)

    infinite-loop cap    (rational_maybe.py search)
        → bound every loop with rational r+s·e  O(K)

Everything in one file so it runs end-to-end.
"""

from __future__ import annotations

import math
import re
import time
import hashlib
import numpy as np
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Tuple, Iterable


# ============================================================
#  Constants
# ============================================================
PI         = math.pi
E          = math.e
ALPHA_SYM  = 1.0 / (PI - E)            # ≈ 2.362
ALPHA_ASYM = 0.3628
DENSITY    = 6.0 / (PI * PI)           # ≈ 0.6079271018
K_DEFAULT  = 256


# ============================================================
#  1. Möbius sieve  ·  O(K log log K)
# ============================================================
def mobius_sieve(K: int) -> np.ndarray:
    if K < 1:
        return np.zeros(K + 1, dtype=np.int8)
    mu = np.ones(K + 1, dtype=np.int8)
    is_comp = np.zeros(K + 1, dtype=bool)
    for i in range(2, K + 1):
        if not is_comp[i]:
            mu[i::i] = -mu[i::i]
            is_comp[i::i] = True
            i2 = i * i
            if i2 <= K:
                mu[i2::i2] = 0
    mu[0] = 0
    return mu


# ============================================================
#  2. Merge sort inverse score  ·  O(K log K)
# ============================================================
def _merge_and_count(a, t, l, m, r):
    i, j, k, inv = l, m + 1, l, 0
    while i <= m and j <= r:
        if a[i] <= a[j]:
            t[k] = a[i]; i += 1
        else:
            t[k] = a[j]; inv += m - i + 1; j += 1
        k += 1
    while i <= m: t[k] = a[i]; i += 1; k += 1
    while j <= r: t[k] = a[j]; j += 1; k += 1
    for i in range(l, r + 1):
        a[i] = t[i]
    return inv


def _merge_sort_count(a, t, l, r):
    inv = 0
    if l < r:
        m = (l + r) // 2
        inv += _merge_sort_count(a, t, l, m)
        inv += _merge_sort_count(a, t, m + 1, r)
        inv += _merge_and_count(a, t, l, m, r)
    return inv


def inversion_count(arr) -> int:
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
#  3. Tokenizer  ·  O(K)
# ============================================================
SYMMETRIC_KEYWORDS = {
    "def", "class", "return", "if", "else", "elif", "for", "while",
    "in", "not", "and", "or", "is", "None", "True", "False",
    "import", "from", "as", "try", "except", "finally", "with",
    "lambda", "yield", "break", "continue", "pass", "raise", "assert",
    "global", "nonlocal", "del", "async", "await",
}

SYMMETRIC_OPERATORS = {
    "+", "-", "*", "/", "//", "%", "**", "==", "!=", "<", ">", "<=", ">=",
    "=", "+=", "-=", "*=", "/=", "//=", "%=", "**=", "&", "|", "^", "~",
    "<<", ">>", ":", ",", ".", ";", "->", "@", "@=",
}

SYMMETRIC_BRACKETS = {"(", ")", "[", "]", "{", "}"}


TOKEN_RE = re.compile(
    r"""
    (?P<comment>\#[^\n]*)
  | (?P<string>"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')
  | (?P<ident>[A-Za-z_][A-Za-z_0-9]*)
  | (?P<number>\d+\.\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|\d+)
  | (?P<op>\*\*|//|<<|>>|<=|>=|==|!=|->|\+=|-=|\*=|/=|//=|%=|\*\*=|&=|\|=|\^=|@=|[-+*/%<>=!&|^~.,:;@\[\]{}()])
  | (?P<ws>\s+)
  | (?P<other>.)
""",
    re.VERBOSE,
)


@dataclass
class Token:
    kind: str
    text: str
    line: int
    col: int
    symmetry: str = "unknown"   # 'sym' | 'asym'


def tokenize(code: str) -> List[Token]:
    """Split code into tokens with line/col.  O(K)."""
    tokens: List[Token] = []
    line = 1
    col  = 0
    for m in TOKEN_RE.finditer(code):
        text = m.group()
        kind = m.lastgroup
        if kind == "ws":
            nl = text.count("\n")
            if nl:
                line += nl
                col = len(text) - text.rfind("\n") - 1
            else:
                col += len(text)
            continue
        tokens.append(Token(kind, text, line, col))
        col += len(text)
        if kind == "comment" or kind == "string":
            # advance line count if multi-line
            nl = text.count("\n")
            if nl:
                line += nl
                col = len(text) - text.rfind("\n") - 1
    return tokens


# ============================================================
#  4. Symmetry classifier
# ============================================================
def classify_token(tok: Token, declared: set) -> str:
    """
    symmetric: keywords, operators, brackets, structure tokens,
                plus any identifier that has been *declared*
                (def foo / class Bar / def foo()  →  foo is sym)
    asymmetric: string/number literals, undeclared identifiers
                (user data), comments
    """
    if tok.kind == "string" or tok.kind == "number":
        return "asym"
    if tok.kind == "comment":
        return "asym"
    if tok.kind == "ident":
        if tok.text in SYMMETRIC_KEYWORDS:
            return "sym"
        if tok.text in declared:
            return "sym"
        return "asym"
    if tok.kind == "op":
        if tok.text in SYMMETRIC_BRACKETS:
            return "sym"
        if tok.text in SYMMETRIC_OPERATORS:
            return "sym"
        return "sym"     # punctuation like ':' ',' '.' is structural
    return "sym"


def detect_declarations(tokens: List[Token]) -> set:
    """
    Scan for `def NAME` and `class NAME` to mark identifiers as
    symmetric.  O(K).
    """
    declared = set()
    for i, tok in enumerate(tokens[:-1]):
        if tok.kind == "ident" and tok.text in ("def", "class"):
            nxt = tokens[i + 1]
            if nxt.kind == "ident":
                declared.add(nxt.text)
    return declared


# ============================================================
#  5. Symmetric pipeline  ·  ASCII-like Möbius compression
# ============================================================
class SymmetricCompressor:
    """
    Compresses symmetric tokens by hashing each token's text into an
    ASCII-like value and running the Möbius-filtered chip pipeline.
    Time O(K log log K).
    """
    def __init__(self, K: int = K_DEFAULT):
        self.K = K
        self.mu = mobius_sieve(K)

    def encode(self, tokens: List[Token]) -> np.ndarray:
        """
        ASCII-like encoding: for each symmetric token, average the
        ASCII codes of its characters, then Möbius-filter by index.
        """
        n = len(tokens)
        if n > self.K:
            tokens = tokens[:self.K]
            n = self.K
        c = np.zeros(2 * n + 1, dtype=float)
        for idx, tok in enumerate(tokens):
            if tok.symmetry != "sym":
                continue
            ascii_mean = sum(ord(ch) for ch in tok.text) / max(len(tok.text), 1)
            i = (idx % n) + 1
            if self.mu[i] == 0:
                continue
            c[n + i] += ascii_mean
            c[n - i] += ascii_mean
        return c

    def supertrace(self, c: np.ndarray) -> float:
        return float(np.sum(np.where(np.arange(len(c)) % 2 == 0, c, -c)))

    def compress(self, tokens: List[Token]) -> dict:
        t0 = time.perf_counter()
        c = self.encode(tokens)
        S = self.supertrace(c)
        N = len(c)
        if N > 0 and S != 0.0:
            p = abs(S) / N
            H = -ALPHA_SYM * p * math.log(p) if 0.0 < p < 1.0 else 0.0
            m = abs(S) * math.exp(-H) if H < 700 else 0.0
        else:
            H, m = 0.0, 0.0
        return dict(
            c=c, S=S, H=H, m=m,
            n_sym=sum(1 for t in tokens if t.symmetry == "sym"),
            elapsed_ms=(time.perf_counter() - t0) * 1e3,
        )


# ============================================================
#  6. Asymmetric pipeline  ·  |Ci| + inverse-score sort
# ============================================================
class AsymmetricCompressor:
    """
    Compresses asymmetric tokens (user data / literals) by embedding
    each token into a |Ci| array and sorting with the inverse score.
    Time O(K log K).
    """
    def __init__(self, K: int = K_DEFAULT):
        self.K = K
        self.mu = mobius_sieve(K)

    def embed(self, tok: Token) -> np.ndarray:
        h = int(hashlib.md5(tok.text.encode()).hexdigest()[:8], 16)
        i = (h % self.K) + 1
        c = np.zeros(2 * self.K + 1, dtype=float)
        if self.mu[i] != 0:
            c[self.K + i] += 1.0
            c[self.K - i] += 1.0
        return c

    def compress(self, tokens: List[Token],
                 ref: Optional[np.ndarray] = None) -> dict:
        t0 = time.perf_counter()
        asym = [t for t in tokens if t.symmetry == "asym"]
        if not asym:
            return dict(
                n_asym=0, inverse_scores=[], S=0.0, H=0.0, m=0.0,
                elapsed_ms=(time.perf_counter() - t0) * 1e3,
            )
        ref_emb = ref if ref is not None else self.embed(asym[0])
        scores = []
        for tok in asym:
            emb = self.embed(tok)
            sc = inverse_score(emb.tolist(), ref_emb.tolist())
            scores.append((tok.text, sc))
        # supertrace of the concatenated embeddings
        concat = np.concatenate([self.embed(t) for t in asym[:self.K]])
        S = float(np.sum(np.where(np.arange(len(concat)) % 2 == 0,
                                  concat, -concat)))
        N = len(concat)
        if N > 0 and S != 0.0:
            p = abs(S) / N
            H = -ALPHA_ASYM * p * math.log(p) if 0.0 < p < 1.0 else 0.0
            m = abs(S) * math.exp(-H) if H < 700 else 0.0
        else:
            H, m = 0.0, 0.0
        return dict(
            n_asym=len(asym),
            inverse_scores=scores,
            S=S, H=H, m=m,
            elapsed_ms=(time.perf_counter() - t0) * 1e3,
        )


# ============================================================
#  7. Linear regression on the signature matrix  ·  O(K log K)
# ============================================================
def build_signature_matrix(tokens: List[Token]) -> np.ndarray:
    """
    Build a (K, 3) matrix:
        col 0: token length
        col 1: ASCII sum
        col 2: log(1 + index)
    Symmetric and asymmetric tokens both contribute; the matrix is
    the algebraic input to the regression.
    """
    n = min(len(tokens), K_DEFAULT)
    M = np.zeros((n, 3), dtype=float)
    for i, tok in enumerate(tokens[:n]):
        M[i, 0] = len(tok.text)
        M[i, 1] = sum(ord(c) for c in tok.text)
        M[i, 2] = math.log1p(i)
    return M


def linear_regression_symmetric(tokens: List[Token]) -> dict:
    """
    Fit each feature column against the token index using ordinary
    least squares.  The residual supertrace, entropy, and mass give
    the "algebraic" score.
    O(K log K) because the least-squares solve is O(K) and the
    supertrace pass is O(K).
    """
    t0 = time.perf_counter()
    M = build_signature_matrix(tokens)
    if M.size == 0:
        return dict(a=[], b=[], S=0.0, H=0.0, m=0.0, rmse=[],
                    elapsed_ms=(time.perf_counter() - t0) * 1e3)
    x = np.arange(len(M), dtype=float)
    A = np.vstack([x, np.ones_like(x)]).T
    a_list, b_list, rmse_list = [], [], []
    residuals = np.zeros(len(M))
    for j in range(M.shape[1]):
        a, b = np.linalg.lstsq(A, M[:, j], rcond=None)[0]
        pred = a * x + b
        r = M[:, j] - pred
        a_list.append(float(a))
        b_list.append(float(b))
        rmse_list.append(float(np.sqrt(np.mean(r ** 2))))
        residuals += np.abs(r)
    S = float(np.sum(np.where(np.arange(len(residuals)) % 2 == 0,
                              residuals, -residuals)))
    N = len(residuals)
    if N > 0 and S != 0.0:
        p = abs(S) / N
        H = -ALPHA_SYM * p * math.log(p) if 0.0 < p < 1.0 else 0.0
        m = abs(S) * math.exp(-H) if H < 700 else 0.0
    else:
        H, m = 0.0, 0.0
    return dict(
        a=a_list, b=b_list, rmse=rmse_list,
        S=S, H=H, m=m,
        elapsed_ms=(time.perf_counter() - t0) * 1e3,
    )


# ============================================================
#  8. Bug counter  ·  dual gate
# ============================================================
class BugCounter:
    """
    Dual-stage bug counter:
      symmetric reference  →  expected supertrace
      asymmetric trace     →  bug signal
      if trace deviation > tolerance, count a bug.
    """
    def __init__(self, tolerance: float = 0.05):
        self.tolerance = tolerance
        self.ref_S: Optional[float] = None
        self.ref_H: Optional[float] = None
        self.bug_count = 0
        self.bug_locations: List[Tuple[int, str]] = []

    def set_reference(self, tokens: List[Token],
                      sym_comp: SymmetricCompressor) -> None:
        res = sym_comp.compress(tokens)
        self.ref_S = res["S"]
        self.ref_H = res["H"]

    def check(self, tokens: List[Token],
              sym_comp: SymmetricCompressor) -> dict:
        res = sym_comp.compress(tokens)
        if self.ref_S is None:
            return dict(is_ok=True, similarity=1.0,
                        S=res["S"], H=res["H"], m=res["m"],
                        deviation=0.0)
        dev_S = abs(res["S"] - self.ref_S) / (abs(self.ref_S) + 1e-12)
        dev_H = abs(res["H"] - self.ref_H) / (abs(self.ref_H) + 1e-12)
        deviation = (dev_S + dev_H) / 2.0
        is_bug = deviation > self.tolerance
        if is_bug:
            self.bug_count += 1
            self.bug_locations.append((len(tokens), f"dev={deviation:.4f}"))
        return dict(
            is_ok=not is_bug,
            similarity=1.0 - min(deviation, 1.0),
            S=res["S"], H=res["H"], m=res["m"],
            deviation=deviation,
        )


# ============================================================
#  9. Infinite-loop cap  ·  rational_maybe.py
# ============================================================
def exp_series(k: int) -> float:
    s, t = 1.0, 1.0
    for n in range(1, k + 1):
        t /= n
        s += t
    return s


def pi_series(k: int) -> float:
    s, sign = 0.0, 1.0
    for n in range(k + 1):
        s += sign / (2 * n + 1)
        sign = -sign
    return 4.0 * s


def search_rational_combination(y: float, e_ap: float,
                                r_range=(-5, 5),
                                s_range=(-5, 5),
                                denom_limit: int = 4) -> Tuple[Fraction, Fraction, float]:
    best_r, best_s, best_err = Fraction(0, 1), Fraction(0, 1), float('inf')
    rationals = set()
    for den in range(1, denom_limit + 1):
        for num in range(r_range[0] * den, r_range[1] * den + 1):
            rationals.add(Fraction(num, den))
    rationals = list(rationals)
    for r in rationals:
        for s in rationals:
            pred = float(r) + float(s) * e_ap
            err = abs(y - pred)
            if err < best_err:
                best_err, best_r, best_s = err, r, s
    return best_r, best_s, best_err


def cap_loop_with_rational(loop_state: float,
                           max_iter: int = 10_000,
                           series_terms: int = 120
                           ) -> dict:
    """
    Cap an infinite loop by snapping its state to the nearest
    rational combination r + s·e  derived from the series
    approximations of π and e.  Any further iterations that would
    push the state outside [r, r+s·e] are terminated.
    """
    e_ap = exp_series(series_terms)
    pi_ap = pi_series(series_terms)
    a_ap = 1.0 / (pi_ap - e_ap)
    y = a_ap * loop_state
    r, s, err = search_rational_combination(y, e_ap)
    boundary = (float(r) + float(s) * e_ap) / a_ap
    if abs(loop_state) > abs(boundary) + 1e-9:
        return dict(capped=True, iteration=max_iter,
                    boundary=boundary, fit=(r, s, err))
    return dict(capped=False, iteration=None,
                boundary=boundary, fit=(r, s, err))


# ============================================================
# 10. The coding harness
# ============================================================
@dataclass
class HarnessReport:
    n_tokens: int = 0
    n_sym: int = 0
    n_asym: int = 0
    n_bugs: int = 0
    n_loops_capped: int = 0
    sym_result: dict = field(default_factory=dict)
    asym_result: dict = field(default_factory=dict)
    linreg_result: dict = field(default_factory=dict)
    bug_result: dict = field(default_factory=dict)
    loop_caps: List[dict] = field(default_factory=list)
    total_ms: float = 0.0


class CodingHarness:
    """
    End-to-end coding harness:

        tokenize  →  classify  →  symmetric compressor  →  O(K log log K)
                              →  asymmetric compressor →  O(K log K)
                              →  linear regression     →  O(K log K)
                              →  bug counter
                              →  loop cap (rational)
    """
    def __init__(self, K: int = K_DEFAULT):
        self.K = K
        self.sym_comp = SymmetricCompressor(K)
        self.asym_comp = AsymmetricCompressor(K)
        self.bug_counter = BugCounter(tolerance=0.05)

    def analyze(self, code: str,
                reference_code: Optional[str] = None,
                loop_states: Optional[Iterable[float]] = None) -> HarnessReport:
        t0 = time.perf_counter()
        rep = HarnessReport()

        # 1. tokenize and classify
        tokens = tokenize(code)
        declared = detect_declarations(tokens)
        for tok in tokens:
            tok.symmetry = classify_token(tok, declared)
        rep.n_tokens = len(tokens)
        rep.n_sym = sum(1 for t in tokens if t.symmetry == "sym")
        rep.n_asym = sum(1 for t in tokens if t.symmetry == "asym")

        # 2. symmetric compression  →  O(K log log K)
        rep.sym_result = self.sym_comp.compress(tokens)

        # 3. asymmetric compression  →  O(K log K)
        rep.asym_result = self.asym_comp.compress(tokens)

        # 4. linear regression  →  O(K log K)
        rep.linreg_result = linear_regression_symmetric(tokens)

        # 5. bug counter (if reference provided)
        if reference_code is not None:
            ref_tokens = tokenize(reference_code)
            ref_declared = detect_declarations(ref_tokens)
            for tok in ref_tokens:
                tok.symmetry = classify_token(tok, ref_declared)
            self.bug_counter.set_reference(ref_tokens, self.sym_comp)
            rep.bug_result = self.bug_counter.check(tokens, self.sym_comp)
            rep.n_bugs = self.bug_counter.bug_count

        # 6. loop caps
        if loop_states is not None:
            for state in loop_states:
                cap = cap_loop_with_rational(state)
                rep.loop_caps.append(cap)
                if cap["capped"]:
                    rep.n_loops_capped += 1

        rep.total_ms = (time.perf_counter() - t0) * 1e3
        return rep

    def report(self, rep: HarnessReport) -> str:
        lines = []
        lines.append("=" * 72)
        lines.append("Coding harness report")
        lines.append("=" * 72)
        lines.append(f"  tokens              : {rep.n_tokens}")
        lines.append(f"  symmetric  (sym)    : {rep.n_sym}")
        lines.append(f"  asymmetric (asym)   : {rep.n_asym}")
        lines.append(f"  bugs detected       : {rep.n_bugs}")
        lines.append(f"  loops capped        : {rep.n_loops_capped}")
        lines.append(f"  total time          : {rep.total_ms:.2f} ms")
        lines.append("")
        lines.append("--- symmetric pipeline (O(K log log K)) ---")
        sr = rep.sym_result
        lines.append(f"  S (supertrace)      : {sr.get('S', 0.0):+.6f}")
        lines.append(f"  H (entropy)         : {sr.get('H', 0.0):.6f}")
        lines.append(f"  m (mass)            : {sr.get('m', 0.0):.6e}")
        lines.append(f"  elapsed             : {sr.get('elapsed_ms', 0.0):.2f} ms")
        lines.append("")
        lines.append("--- asymmetric pipeline (O(K log K)) ---")
        ar = rep.asym_result
        lines.append(f"  tokens              : {ar.get('n_asym', 0)}")
        lines.append(f"  S                   : {ar.get('S', 0.0):+.6f}")
        lines.append(f"  H                   : {ar.get('H', 0.0):.6f}")
        lines.append(f"  m                   : {ar.get('m', 0.0):.6e}")
        lines.append(f"  elapsed             : {ar.get('elapsed_ms', 0.0):.2f} ms")
        if ar.get("inverse_scores"):
            lines.append("  top inverse scores (first 5):")
            for text, sc in ar["inverse_scores"][:5]:
                lines.append(f"    {text[:24]:24s}  {sc:.4f}")
        lines.append("")
        lines.append("--- linear regression (algebraic, O(K log K)) ---")
        lr = rep.linreg_result
        for j, (a, b, r) in enumerate(zip(lr.get("a", []),
                                          lr.get("b", []),
                                          lr.get("rmse", []))):
            lines.append(f"  feature {j}: a={a:+.4f}  b={b:+.4f}  rmse={r:.4f}")
        lines.append(f"  S                   : {lr.get('S', 0.0):+.6f}")
        lines.append(f"  H                   : {lr.get('H', 0.0):.6f}")
        lines.append(f"  m                   : {lr.get('m', 0.0):.6e}")
        lines.append(f"  elapsed             : {lr.get('elapsed_ms', 0.0):.2f} ms")
        lines.append("")
        if rep.bug_result:
            br = rep.bug_result
            lines.append("--- bug counter ---")
            lines.append(f"  is_ok               : {br.get('is_ok', True)}")
            lines.append(f"  similarity          : {br.get('similarity', 1.0):.4f}")
            lines.append(f"  deviation           : {br.get('deviation', 0.0):.4f}")
            lines.append("")
        if rep.loop_caps:
            lines.append("--- loop caps (rational r + s·e) ---")
            for i, cap in enumerate(rep.loop_caps[:5]):
                r, s, err = cap["fit"]
                lines.append(
                    f"  loop {i}: capped={cap['capped']}  "
                    f"boundary={cap['boundary']:+.4f}  "
                    f"fit = {r} + {s}·e  (err={err:.3e})"
                )
            lines.append("")
        return "\n".join(lines)


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 72)
    print("Coding harness  ·  symmetric / asymmetric token routing")
    print("=" * 72)
    print(f"  α_sym  = 1/(π − e) = {ALPHA_SYM:.6f}")
    print(f"  α_asym = 0.3628     = {ALPHA_ASYM:.6f}")
    print()

    harness = CodingHarness(K=256)

    good_code = """
def add(a, b):
    return a + b

class Calculator:
    def __init__(self):
        self.value = 0

    def add(self, x):
        self.value += x
        return self.value

def main():
    c = Calculator()
    c.add(5)
    print(c.value)

if __name__ == "__main__":
    main()
"""

    buggy_code = """
def add(a, b):
    return a - b

class Calculator:
    def __init__(self):
        self.value = 0

    def add(self, x):
        self.value -= x
        return self.value

def main():
    c = Calculator()
    c.add(5)
    print(c.value)

if __name__ == "__main__":
    main()
"""

    # ---- 1. analyze the good code ----
    print("--- Analyzing GOOD code ---")
    rep_good = harness.analyze(good_code)
    print(harness.report(rep_good))

    # ---- 2. analyze the buggy code with reference = good ----
    print("--- Analyzing BUGGY code (with reference) ---")
    harness2 = CodingHarness(K=256)
    rep_bug = harness2.analyze(buggy_code, reference_code=good_code)
    print(harness2.report(rep_bug))

    # ---- 3. loop capping demo ----
    print("--- Loop capping (rational search) ---")
    loop_states = [1.0, 2.5, 10.0, 100.0, 1000.0]
    harness3 = CodingHarness(K=256)
    rep_loop = harness3.analyze(good_code, loop_states=loop_states)
    for i, cap in enumerate(rep_loop.loop_caps):
        r, s, err = cap["fit"]
        print(f"  loop {i}: state={loop_states[i]:.2f}  "
              f"capped={cap['capped']}  "
              f"boundary={cap['boundary']:+.4f}  "
              f"fit = {r} + {s}·e  (err={err:.3e})")

    # ---- 4. explicit rational search on a few values ----
    print("\n--- Explicit rational search (a·x → r + s·e) ---")
    for x in [math.pi, math.e, (1 + math.sqrt(5)) / 2]:
        e_ap = exp_series(120)
        pi_ap = pi_series(120)
        a_ap = 1.0 / (pi_ap - e_ap)
        y = a_ap * x
        r, s, err = search_rational_combination(y, e_ap)
        print(f"  x = {x:.6f}  →  y = {y:.6f}  "
              f"≈ {r} + {s}·e  (err = {err:.4e})")

    print("\nDone.")


if __name__ == "__main__":
    demo()