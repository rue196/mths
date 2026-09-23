#!/usr/bin/env python3
"""
code_database_reader.py

Reads a large code database and routes every file through two
pipelines:

    symmetric    (good code + language tokens)
        → Möbius sieve + ASCII embedding → supertrace/entropy
        → O(K log log K)  per file

    asymmetric   (user data, literals, suspected bugs)
        → |Ci| embedding + inverse-score sort vs reference
        → strike-flag → O(K log K)  per file
        → escalated to the PDE investigator if strikes ≥ threshold

The reader builds a per‑file profile, aggregates directory stats,
and lists the top‑flagged files for further investigation.
"""

from __future__ import annotations

import os
import re
import math
import time
import hashlib
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Iterator


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
    while i <= mid: temp[k] = arr[i]; i += 1; k += 1
    while j <= right: temp[k] = arr[j]; j += 1; k += 1
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


def inversion_count(arr) -> int:
    n = len(arr)
    if n < 2:
        return 0
    temp = [0] * n
    return _merge_sort_count(list(arr), temp, 0, n - 1)


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
#  3. Supertrace / entropy  ·  O(K)
# ============================================================
def supertrace(c) -> float:
    S = 0.0
    for i, v in enumerate(c):
        S += v if (i % 2 == 0) else -v
    return float(S)


def entropy_from_supertrace(S: float, K: int, alpha: float = ALPHA_SYM) -> float:
    if K <= 0 or S == 0.0:
        return 0.0
    p = abs(S) / K
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -alpha * p * math.log(p)


# ============================================================
#  4. Tokenizer + symmetry classifier (from coding_harness.py)
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
    "(", ")", "[", "]", "{", "}",
}


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
    symmetry: str = "unknown"


def tokenize(code: str) -> List[Token]:
    tokens: List[Token] = []
    line, col = 1, 0
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
        if kind in ("comment", "string"):
            nl = text.count("\n")
            if nl:
                line += nl
                col = len(text) - text.rfind("\n") - 1
    return tokens


def detect_declarations(tokens: List[Token]) -> set:
    declared = set()
    for i, tok in enumerate(tokens[:-1]):
        if tok.kind == "ident" and tok.text in ("def", "class"):
            nxt = tokens[i + 1]
            if nxt.kind == "ident":
                declared.add(nxt.text)
    return declared


def classify_token(tok: Token, declared: set) -> str:
    if tok.kind in ("string", "number", "comment"):
        return "asym"
    if tok.kind == "ident":
        if tok.text in SYMMETRIC_KEYWORDS or tok.text in declared:
            return "sym"
        return "asym"
    return "sym"


# ============================================================
#  5. Symmetric pipeline  ·  O(K log log K)
# ============================================================
class SymmetricPipeline:
    """
    ASCII-like embedding of symmetric tokens, supertrace/entropy
    computed over the Möbius-filtered array.
    """
    def __init__(self, K: int = K_DEFAULT):
        self.K = K
        self.mu = mobius_sieve(K)

    def embed(self, tokens: List[Token]) -> np.ndarray:
        n = min(len(tokens), self.K)
        c = np.zeros(2 * n + 1, dtype=float)
        for idx, tok in enumerate(tokens[:n]):
            if tok.symmetry != "sym":
                continue
            ascii_mean = sum(ord(ch) for ch in tok.text) / max(len(tok.text), 1)
            i = (idx % n) + 1 if n > 0 else 1
            if self.mu[i] == 0:
                continue
            c[n + i] += ascii_mean
            c[n - i] += ascii_mean
        return c

    def analyze(self, tokens: List[Token]) -> dict:
        t0 = time.perf_counter()
        c = self.embed(tokens)
        S = supertrace(c)
        N = len(c)
        H = entropy_from_supertrace(S, N, ALPHA_SYM)
        m = abs(S) * math.exp(-H) if 0.0 < H < 700 else 0.0
        return dict(
            c=c, S=S, H=H, m=m, N=N,
            n_sym=sum(1 for t in tokens if t.symmetry == "sym"),
            elapsed_ms=(time.perf_counter() - t0) * 1e3,
        )


# ============================================================
#  6. Asymmetric pipeline  ·  O(K log K)
# ============================================================
class AsymmetricPipeline:
    """
    |Ci| embedding of asymmetric tokens, inverse-score sort against
    a reference embedding, supertrace/entropy of the concatenation.
    """
    def __init__(self, K: int = K_DEFAULT):
        self.K = K
        self.mu = mobius_sieve(K)

    def embed(self, text: str) -> np.ndarray:
        h = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        i = (h % self.K) + 1
        c = np.zeros(2 * self.K + 1, dtype=float)
        if self.mu[i] != 0:
            c[self.K + i] += 1.0
            c[self.K - i] += 1.0
        return c

    def analyze(self, tokens: List[Token],
                reference: Optional[np.ndarray] = None) -> dict:
        t0 = time.perf_counter()
        asym = [t for t in tokens if t.symmetry == "asym"]
        n = len(asym)
        if n == 0:
            return dict(
                n_asym=0, inverse_scores=[], S=0.0, H=0.0, m=0.0,
                elapsed_ms=(time.perf_counter() - t0) * 1e3,
            )
        # reference embedding = first asymmetric token if none given
        ref_emb = reference if reference is not None else self.embed(asym[0].text)
        scores = []
        for tok in asym:
            emb = self.embed(tok.text)
            sc = inverse_score(emb.tolist(), ref_emb.tolist())
            scores.append((tok.text, sc, tok.line))
        # concatenated supertrace
        concat = np.concatenate([self.embed(t.text) for t in asym[:self.K]])
        S = supertrace(concat)
        N = len(concat)
        H = entropy_from_supertrace(S, N, ALPHA_ASYM)
        m = abs(S) * math.exp(-H) if 0.0 < H < 700 else 0.0
        return dict(
            n_asym=n,
            inverse_scores=scores,
            S=S, H=H, m=m, N=N,
            elapsed_ms=(time.perf_counter() - t0) * 1e3,
        )


# ============================================================
#  7. File profile
# ============================================================
@dataclass
class FileProfile:
    path: str
    size_bytes: int
    n_tokens: int
    n_sym: int
    n_asym: int
    sym_S: float = 0.0
    sym_H: float = 0.0
    sym_m: float = 0.0
    asym_S: float = 0.0
    asym_H: float = 0.0
    asym_m: float = 0.0
    mean_inverse_score: float = 0.5
    strikes: int = 0
    flagged: bool = False
    deep_result: Optional[Dict] = None
    top_asym: List[Tuple[str, float, int]] = field(default_factory=list)
    sym_ms: float = 0.0
    asym_ms: float = 0.0


# ============================================================
#  8. Code-database reader
# ============================================================
class CodeDatabaseReader:
    """
    Reads a directory of source files and routes each through the
    symmetric pipeline (O(K log log K)) and the asymmetric pipeline
    (O(K log K)).

    Strike accumulation per file:
        • symmetric anomaly:  |S_sym| deviation from reference
        • asymmetric anomaly: mean inverse score > threshold
        • asymmetry leak:     |S_asym| > |S_sym| + tolerance

    Flagged files are escalated to the PDE investigator.
    """
    def __init__(self,
                 K: int = K_DEFAULT,
                 strike_threshold: int = 2,
                 entropy_tolerance: float = 0.15,
                 K_pde: int = 128):
        self.K = K
        self.strike_threshold = strike_threshold
        self.entropy_tolerance = entropy_tolerance
        self.K_pde = K_pde

        self.sym_pipe = SymmetricPipeline(K)
        self.asym_pipe = AsymmetricPipeline(K)

        self.profiles: List[FileProfile] = []
        self.flags: deque = deque()
        self.ref_S_sym: Optional[float] = None
        self.ref_H_sym: Optional[float] = None

    # ---------- reference calibration ----------
    def calibrate(self, ref_code: str) -> dict:
        """Use a reference file to set symmetric baselines."""
        tokens = tokenize(ref_code)
        declared = detect_declarations(tokens)
        for t in tokens:
            t.symmetry = classify_token(t, declared)
        res = self.sym_pipe.analyze(tokens)
        self.ref_S_sym = res["S"]
        self.ref_H_sym = res["H"]
        return res

    # ---------- read one file ----------
    def read_file(self, path: str,
                  code: Optional[str] = None) -> FileProfile:
        if code is None:
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    code = f.read()
            except OSError as e:
                # unreadable — return a stub
                return FileProfile(path=path, size_bytes=0, n_tokens=0,
                                   n_sym=0, n_asym=0)

        tokens = tokenize(code)
        declared = detect_declarations(tokens)
        for t in tokens:
            t.symmetry = classify_token(t, declared)

        # ---- symmetric pipeline: O(K log log K) ----
        sym_res = self.sym_pipe.analyze(tokens)

        # ---- asymmetric pipeline: O(K log K) ----
        asym_res = self.asym_pipe.analyze(tokens)

        # ---- build profile ----
        prof = FileProfile(
            path=path,
            size_bytes=len(code.encode("utf-8")),
            n_tokens=len(tokens),
            n_sym=sym_res["n_sym"],
            n_asym=asym_res["n_asym"],
            sym_S=sym_res["S"],
            sym_H=sym_res["H"],
            sym_m=sym_res["m"],
            asym_S=asym_res["S"],
            asym_H=asym_res["H"],
            asym_m=asym_res["m"],
            sym_ms=sym_res["elapsed_ms"],
            asym_ms=asym_res["elapsed_ms"],
            top_asym=asym_res["inverse_scores"][:5],
        )
        if prof.top_asym:
            prof.mean_inverse_score = sum(s for _, s, _ in prof.top_asym) / len(prof.top_asym)

        # ---- strike accumulation ----
        strikes = 0
        # 1. symmetric anomaly: |S_sym − S_ref| / |S_ref| > tol
        if self.ref_S_sym is not None:
            dev = abs(prof.sym_S - self.ref_S_sym) / (abs(self.ref_S_sym) + 1e-12)
            if dev > self.entropy_tolerance:
                strikes += 1
        # 2. asymmetric anomaly: mean inverse score exceeds 0.5 + tol
        if abs(prof.mean_inverse_score - 0.5) > self.entropy_tolerance:
            strikes += 1
        # 3. asymmetry leak: |S_asym| > |S_sym| + tol
        if abs(prof.asym_S) > abs(prof.sym_S) + self.entropy_tolerance:
            strikes += 1
        prof.strikes = strikes

        # ---- flag → escalate ----
        if strikes >= self.strike_threshold:
            prof.flagged = True
            self.flags.append(prof)
            prof.deep_result = self._pde_investigate(prof, tokens)

        self.profiles.append(prof)
        return prof

    # ---------- PDE investigator  ·  O(K) ----
    def _pde_investigate(self, prof: FileProfile,
                         tokens: List[Token]) -> Dict:
        K = self.K_pde
        n = len(tokens)
        if n == 0:
            return dict(path=prof.path, h_norm=0.0, h_mean=0.0,
                        h_std=0.0, pde_S=0.0, pde_H=0.0)

        seed = int(hashlib.md5(prof.path.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        W_in = rng.standard_normal((K, min(n, K))) * 0.01
        anchors = np.arange(0, K, max(1, K // 20))

        c = np.array([sum(ord(ch) for ch in t.text) for t in tokens[:K]],
                     dtype=float)
        h = np.zeros(K)
        for j in range(min(n, K)):
            h += W_in[:, j] * c[j]
            diff = np.zeros_like(h)
            diff[1:-1] = h[:-2] + h[2:] - 2.0 * h[1:-1]
            h += 0.1 * (diff + 0.1 * np.sum(h[anchors]) - 0.01 * h)

        final_S = supertrace(h)
        final_H = entropy_from_supertrace(final_S, K, ALPHA_SYM)
        return dict(
            path=prof.path,
            h_norm=float(np.linalg.norm(h)),
            h_mean=float(np.mean(h)),
            h_std=float(np.std(h)),
            pde_S=final_S,
            pde_H=final_H,
        )

    # ---------- walk directory ----------
    def read_directory(self, root: str,
                       extensions: Tuple[str, ...] = (".py", ".c", ".h",
                                                     ".cpp", ".js", ".ts",
                                                     ".rb", ".go", ".rs"),
                       max_files: Optional[int] = None
                       ) -> Iterator[FileProfile]:
        n = 0
        for dirpath, _, files in os.walk(root):
            for fname in files:
                if not fname.endswith(extensions):
                    continue
                if max_files is not None and n >= max_files:
                    return
                yield self.read_file(os.path.join(dirpath, fname))
                n += 1

    # ---------- aggregate summary ----------
    def summary(self) -> Dict:
        n = len(self.profiles)
        if n == 0:
            return dict(n_files=0)
        flagged = [p for p in self.profiles if p.flagged]
        return dict(
            n_files=n,
            n_flagged=len(flagged),
            flagged_paths=[p.path for p in flagged[:10]],
            total_bytes=sum(p.size_bytes for p in self.profiles),
            total_tokens=sum(p.n_tokens for p in self.profiles),
            total_sym=sum(p.n_sym for p in self.profiles),
            total_asym=sum(p.n_asym for p in self.profiles),
            mean_sym_S=float(np.mean([p.sym_S for p in self.profiles])),
            mean_sym_H=float(np.mean([p.sym_H for p in self.profiles])),
            mean_asym_S=float(np.mean([p.asym_S for p in self.profiles])),
            mean_strikes=float(np.mean([p.strikes for p in self.profiles])),
            sym_ms=float(np.mean([p.sym_ms for p in self.profiles])),
            asym_ms=float(np.mean([p.asym_ms for p in self.profiles])),
        )

    def report(self, top_n: int = 10) -> str:
        lines = []
        lines.append("=" * 76)
        lines.append("Code database reader  ·  symmetric + asymmetric pipelines")
        lines.append("=" * 76)
        lines.append(f"  α_sym   = 1/(π − e) = {ALPHA_SYM:.6f}")
        lines.append(f"  α_asym  = 0.3628     = {ALPHA_ASYM:.6f}")
        lines.append(f"  K       = {self.K}")
        lines.append("")
        s = self.summary()
        for k, v in s.items():
            if isinstance(v, float):
                lines.append(f"  {k:>18s}: {v:.4f}")
            elif isinstance(v, list):
                lines.append(f"  {k:>18s}: {v}")
            else:
                lines.append(f"  {k:>18s}: {v}")
        lines.append("")
        lines.append(f"--- Top {top_n} flagged files ---")
        flagged = sorted(self.flags, key=lambda p: -p.strikes)
        for p in flagged[:top_n]:
            dr = p.deep_result or {}
            lines.append(
                f"  {p.path[:44]:44s}  strikes={p.strikes}  "
                f"S_sym={p.sym_S:+.3f}  S_asym={p.asym_S:+.3f}  "
                f"pde_S={dr.get('pde_S', 0.0):+.3f}"
            )
        lines.append("")
        lines.append("--- Pipeline complexity ---")
        lines.append("  symmetric  pipeline     O(K log log K)   per file")
        lines.append("  asymmetric pipeline     O(K log K)       per file")
        lines.append("  escalations (PDE)       O(K)             per flagged file")
        return "\n".join(lines)


# ============================================================
#  9. Demo
# ============================================================
def demo():
    print("=" * 76)
    print("Code database reader  ·  demo")
    print("=" * 76)
    print()

    reader = CodeDatabaseReader(K=256,
                                strike_threshold=2,
                                entropy_tolerance=0.15)

    # ---------- reference (good code) ----------
    reference = """
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
    reader.calibrate(reference)
    print(f"Reference calibrated:  S_sym = {reader.ref_S_sym:+.6f}  "
          f"H_sym = {reader.ref_H_sym:.6f}")

    # ---------- a small in-memory database ----------
    db = {
        "good_add.py": """
def add(a, b):
    return a + b

if __name__ == "__main__":
    print(add(2, 3))
""",
        "buggy_sub.py": """
def add(a, b):
    return a - b       # bug: should be +

if __name__ == "__main__":
    print(add(2, 3))
""",
        "clean_loop.py": """
def sum_upto(n):
    total = 0
    for i in range(1, n + 1):
        total += i
    return total

print(sum_upto(10))
""",
        "buggy_loop.py": """
def sum_upto(n):
    total = 0
    for i in range(1, n + 1):
        total -= i     # bug: should be +=
    return total

print(sum_upto(10))
""",
        "noisy_input.py": """
user_data = 'x7y2z9wq1'
tokens_2384 = 'aaaa bbbb cccc dddd'
raw_payload = '9f2c-a1b4-77e0-1234'
flag_xyzzy = True
value_9999 = 12345.678
print(user_data, tokens_2384, raw_payload, flag_xyzzy, value_9999)
""",
    }

    print(f"\nReading {len(db)} files from in‑memory database ...\n")
    for name, code in db.items():
        prof = reader.read_file(name, code)
        marker = "FLAG" if prof.flagged else "    "
        print(f"  [{marker}] {name:20s}  "
              f"tokens={prof.n_tokens:>4d}  "
              f"sym={prof.n_sym:>4d}  asym={prof.n_asym:>4d}  "
              f"S_sym={prof.sym_S:+7.3f}  S_asym={prof.asym_S:+7.3f}  "
              f"strikes={prof.strikes}")

    print()
    print(reader.report(top_n=5))

    # ---------- per-file detail for a flagged file ----------
    print()
    print("--- Detail: top asymmetric tokens of flagged files ---")
    for p in reader.flags:
        print(f"\n  {p.path}")
        print(f"    sym_S   = {p.sym_S:+.4f}   sym_H = {p.sym_H:.4f}  "
              f"sym_m = {p.sym_m:.4e}")
        print(f"    asym_S  = {p.asym_S:+.4f}  asym_H = {p.asym_H:.4f}  "
              f"asym_m = {p.asym_m:.4e}")
        print(f"    mean inverse score = {p.mean_inverse_score:.4f}")
        print(f"    top asymmetric tokens:")
        for text, score, line in p.top_asym:
            snippet = text if len(text) <= 24 else text[:21] + "..."
            print(f"      line {line:>4d}  {snippet:26s}  inv={score:.4f}")
        dr = p.deep_result or {}
        print(f"    PDE:  |h|={dr.get('h_norm', 0.0):.4f}  "
              f"pde_S={dr.get('pde_S', 0.0):+.4f}  "
              f"pde_H={dr.get('pde_H', 0.0):.4f}")

    # ---------- directory-walking mode ----------
    print()
    print("--- Directory walk test (current directory, .py files) ---")
    reader2 = CodeDatabaseReader(K=256)
    reader2.calibrate(reference)
    n_read = 0
    try:
        for prof in reader2.read_directory(".", max_files=20):
            n_read += 1
    except Exception as e:
        print(f"  directory walk stopped: {e}")
    print(f"  files read: {n_read}")

    print()
    print("Done.")


if __name__ == "__main__":
    demo()