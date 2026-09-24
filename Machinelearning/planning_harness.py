#!/usr/bin/env python3
"""
planning_harness.py

A planning harness that uses an **elliptic log-rank test** to compare
query requirements against an existing codebase, and then projects the
log-rank residual onto the token space to rank "what to do next".

Model
-----
Two streams:

    requirement stream    r_n = tokens of the task description
    code stream           c_n = tokens of the existing code

Both are sorted and classified (sym / asym).  The algebraic side
builds the tangent-plane fit between the two streams; the
transcendental side carries the temporal ordering of tokens.

For every token type u we compute

    Z_u = O_u − E_u        (log-rank residual for token u)

where
    O_u = observed count of u in the code
    E_u = expected count of u under the requirement stream

The residual is weighted by the elliptic projection

    Π_u = Π(pos_u, rank_u) ∈ [0, 1]

giving the final **weighted likelihood**

    L_u = Π_u · |Z_u| · sign_needed(u)

Tokens with large positive L_u are *missing and likely needed next*;
tokens with large negative L_u are *present but not required*.

The algebraic part of the log-rank uses the deterministic symmetric
tokens (keywords, brackets, operators).  The transcendental part uses
the asymmetric tokens (identifiers, literals, user data) — those
carry the semantic weight of the task.

Complexity
----------
    tokenize + classify       O(K)
    log-rank accumulation     O(K)
    elliptic projection       O(K)
    residual projection       O(K log K)      (merge-sort ranking)
"""

from __future__ import annotations

import math
import re
import time
import numpy as np
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ============================================================
#  Constants
# ============================================================
PI         = math.pi
E          = math.e
ALPHA_SYM  = 1.0 / (PI - E)            # ≈ 2.362
ALPHA_ASYM = 0.3628
W1         = PI
W2         = E
K_DEFAULT  = 512


# ============================================================
#  1. Möbius sieve
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
#  2. Merge-sort ranking  ·  O(K log K)
# ============================================================
def merge_sort_desc(arr: List[Tuple[str, float]]
                    ) -> List[Tuple[str, float]]:
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort_desc(arr[:mid])
    right = merge_sort_desc(arr[mid:])
    out = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i][1] >= right[j][1]:
            out.append(left[i]); i += 1
        else:
            out.append(right[j]); j += 1
    out.extend(left[i:])
    out.extend(right[j:])
    return out


# ============================================================
#  3. Tokenizer (same style as the coding harness)
# ============================================================
SYMMETRIC_KEYWORDS = {
    "def", "class", "return", "if", "else", "elif", "for", "while",
    "in", "not", "and", "or", "is", "None", "True", "False",
    "import", "from", "as", "try", "except", "finally", "with",
    "lambda", "yield", "break", "continue", "pass", "raise", "assert",
    "global", "nonlocal", "del", "async", "await",
    # natural-language symmetric tokens that often appear in tasks
    "the", "a", "an", "of", "to", "for", "with", "and", "or",
    "should", "must", "will", "can", "be", "is", "are", "was",
    "in", "on", "at", "by", "as", "this", "that", "these", "those",
    "it", "its", "from",
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
    line: int = 0
    col: int = 0
    symmetry: str = "unknown"


def tokenize(text: str) -> List[Token]:
    tokens: List[Token] = []
    line, col = 1, 0
    for m in TOKEN_RE.finditer(text):
        text_m = m.group()
        kind = m.lastgroup
        if kind == "ws":
            nl = text_m.count("\n")
            if nl:
                line += nl
                col = len(text_m) - text_m.rfind("\n") - 1
            else:
                col += len(text_m)
            continue
        tokens.append(Token(kind, text_m, line, col))
        col += len(text_m)
        if kind in ("comment", "string"):
            nl = text_m.count("\n")
            if nl:
                line += nl
                col = len(text_m) - text_m.rfind("\n") - 1
    return tokens


def classify_token(tok: Token) -> str:
    if tok.kind in ("string", "number", "comment"):
        return "asym"
    if tok.kind == "ident":
        if tok.text.lower() in SYMMETRIC_KEYWORDS:
            return "sym"
        return "asym"
    if tok.kind == "op":
        return "sym"
    return "sym"


# ============================================================
#  4. Figure 3.5 — bounded elliptic projection
# ============================================================
def elliptic_projection(t: float, r: float) -> float:
    u = (t / W1) % 1.0
    v = (r / W2) % 1.0
    return 0.5 * (1.0 + math.cos(2 * math.pi * u) *
                        math.cos(2 * math.pi * v))


# ============================================================
#  5. Log-rank test between requirement and code streams
# ============================================================
@dataclass
class TokenLogRank:
    token: str
    O: int            # observed count in code
    R: int            # required count in requirements
    Z: float          # log-rank residual Z = O − E
    E: float          # expected count under requirement stream
    position: float   # mean normalized position in the requirement stream
    rank: float       # mean normalized rank in the code stream
    Pi: float         # elliptic projection at (position, rank)
    L: float          # weighted likelihood = Π · Z

    def as_dict(self) -> dict:
        return dict(token=self.token, O=self.O, R=self.R,
                    Z=round(self.Z, 4), E=round(self.E, 4),
                    position=round(self.position, 4),
                    rank=round(self.rank, 4),
                    Pi=round(self.Pi, 4),
                    L=round(self.L, 4))


def log_rank_between(req_tokens: List[Token],
                     code_tokens: List[Token]) -> List[TokenLogRank]:
    """
    Compute the log-rank residual per token type.

    For each token type u:
        O_u = count in code
        E_u = R_u · (|code| / |req|)      expected under requirement
        Z_u = O_u − E_u                   residual

    Then project with the elliptic weight Π(position_u, rank_u):

        position_u = mean normalized index of u in the requirement stream
        rank_u     = mean normalized index of u in the code stream
        Π_u        = Π(position_u, rank_u) ∈ [0, 1]

    Final weighted likelihood:
        L_u = Π_u · Z_u
    Positive L → token is underused relative to requirements (needed)
    Negative L → token is overused relative to requirements (redundant)
    """
    n_req  = max(len(req_tokens), 1)
    n_code = max(len(code_tokens), 1)

    # observed count in code
    O_counts = Counter(t.text for t in code_tokens)

    # required count in requirements
    R_counts = Counter(t.text for t in req_tokens)

    # position in requirement stream (mean normalized index)
    pos_sum = defaultdict(float)
    for idx, tok in enumerate(req_tokens):
        pos_sum[tok.text] += (idx + 1) / n_req
    position = {u: pos_sum[u] / max(R_counts[u], 1) for u in R_counts}

    # rank in code stream (mean normalized index)
    rank_sum = defaultdict(float)
    for idx, tok in enumerate(code_tokens):
        rank_sum[tok.text] += (idx + 1) / n_code
    rank = {u: rank_sum[u] / max(O_counts[u], 1) for u in O_counts}

    # union of tokens
    universe = set(R_counts) | set(O_counts)

    results: List[TokenLogRank] = []
    scale = n_code / n_req
    for u in universe:
        R_u = R_counts.get(u, 0)
        O_u = O_counts.get(u, 0)
        E_u = R_u * scale
        Z_u = O_u - E_u
        pos_u = position.get(u, 0.5)
        rnk_u = rank.get(u, 0.5)
        Pi_u  = elliptic_projection(pos_u, rnk_u)
        L_u   = Pi_u * Z_u
        results.append(TokenLogRank(
            token=u, O=O_u, R=R_u, Z=Z_u, E=E_u,
            position=pos_u, rank=rnk_u,
            Pi=Pi_u, L=L_u,
        ))
    return results


# ============================================================
#  6. Algebraic vs transcendental split of the residual
# ============================================================
def split_residual(results: List[TokenLogRank]) -> Dict[str, dict]:
    """
    Split the log-rank residual into:
        algebraic part       → symmetric tokens (keywords, operators)
        transcendental part  → asymmetric tokens (identifiers, data)

    Algebraic tokens carry the *structure* of the plan.
    Asymmetric tokens carry the *content* of the plan.
    """
    algebraic = [r for r in results
                 if classify_token(Token("ident", r.token)) == "sym"]
    trans     = [r for r in results
                 if classify_token(Token("ident", r.token)) == "asym"]

    def _stats(rows):
        if not rows:
            return dict(n=0, Z_sum=0.0, L_pos=0.0, L_neg=0.0)
        Z_sum = sum(r.Z for r in rows)
        L_pos = sum(r.L for r in rows if r.L > 0)
        L_neg = sum(r.L for r in rows if r.L < 0)
        return dict(n=len(rows), Z_sum=Z_sum, L_pos=L_pos, L_neg=L_neg)

    return dict(algebraic=_stats(algebraic),
                transcendental=_stats(trans))


# ============================================================
#  7. Next-token projection  ·  "what to write next"
# ============================================================
def project_next_tokens(results: List[TokenLogRank],
                        top_k: int = 12) -> Tuple[List[TokenLogRank],
                                                  List[TokenLogRank]]:
    """
    Rank tokens by weighted likelihood L.
    Returns (needed, redundant):
        needed    — tokens with highest positive L (should be added)
        redundant — tokens with most negative L (should be removed)
    """
    ranked = merge_sort_desc([(r, r.L) for r in results])
    needed    = [r for r, L in ranked if L >  0][:top_k]
    redundant = [r for r, L in ranked if L <  0][-top_k:][::-1]
    return needed, redundant


# ============================================================
#  8. The planning harness
# ============================================================
@dataclass
class PlanReport:
    n_req_tokens: int = 0
    n_code_tokens: int = 0
    n_sym_req: int = 0
    n_asym_req: int = 0
    n_sym_code: int = 0
    n_asym_code: int = 0
    Z_sum: float = 0.0
    L_sum: float = 0.0
    Pi_mean: float = 0.0
    algebraic_stats: Dict = field(default_factory=dict)
    transcendental_stats: Dict = field(default_factory=dict)
    needed: List[TokenLogRank] = field(default_factory=list)
    redundant: List[TokenLogRank] = field(default_factory=list)
    elapsed_ms: float = 0.0


class PlanningHarness:
    """
    Compare a query (task requirement) against an existing codebase
    using the elliptic log-rank test, and project the residual onto
    the token space to rank "what to do next".
    """
    def __init__(self, K: int = K_DEFAULT):
        self.K = K
        self.mu = mobius_sieve(K)

    def plan(self, requirement: str, code: str,
             top_k: int = 12) -> PlanReport:
        t0 = time.perf_counter()
        rep = PlanReport()

        req_tokens  = tokenize(requirement)
        code_tokens = tokenize(code)

        for t in req_tokens:
            t.symmetry = classify_token(t)
        for t in code_tokens:
            t.symmetry = classify_token(t)

        rep.n_req_tokens  = len(req_tokens)
        rep.n_code_tokens = len(code_tokens)
        rep.n_sym_req  = sum(1 for t in req_tokens if t.symmetry == "sym")
        rep.n_asym_req = sum(1 for t in req_tokens if t.symmetry == "asym")
        rep.n_sym_code  = sum(1 for t in code_tokens if t.symmetry == "sym")
        rep.n_asym_code = sum(1 for t in code_tokens if t.symmetry == "asym")

        # ---- log-rank residuals ----
        results = log_rank_between(req_tokens, code_tokens)
        rep.Z_sum  = float(np.sum([r.Z for r in results]))
        rep.L_sum  = float(np.sum([r.L for r in results]))
        rep.Pi_mean = float(np.mean([r.Pi for r in results])) if results else 0.0

        # ---- split algebraic / transcendental ----
        split = split_residual(results)
        rep.algebraic_stats = split["algebraic"]
        rep.transcendental_stats = split["transcendental"]

        # ---- projection: what to add / remove next ----
        needed, redundant = project_next_tokens(results, top_k=top_k)
        rep.needed    = needed
        rep.redundant = redundant

        rep.elapsed_ms = (time.perf_counter() - t0) * 1e3
        return rep

    def report(self, rep: PlanReport) -> str:
        lines = []
        lines.append("=" * 76)
        lines.append("Planning harness  ·  elliptic log-rank vs. requirements")
        lines.append("=" * 76)
        lines.append(f"  α_sym   = 1/(π − e) = {ALPHA_SYM:.6f}")
        lines.append(f"  α_asym  = 0.3628     = {ALPHA_ASYM:.6f}")
        lines.append("")
        lines.append(f"  requirement tokens  : {rep.n_req_tokens}"
                     f"   (sym={rep.n_sym_req}, asym={rep.n_asym_req})")
        lines.append(f"  code tokens         : {rep.n_code_tokens}"
                     f"   (sym={rep.n_sym_code}, asym={rep.n_asym_code})")
        lines.append(f"  Σ Z (log-rank)      : {rep.Z_sum:+.4f}")
        lines.append(f"  Σ L (weighted)      : {rep.L_sum:+.4f}")
        lines.append(f"  mean Π              : {rep.Pi_mean:.4f}")
        lines.append(f"  elapsed             : {rep.elapsed_ms:.2f} ms")
        lines.append("")

        # ---- algebraic vs transcendental ----
        lines.append("--- algebraic split (symmetric tokens) ---")
        a = rep.algebraic_stats
        lines.append(f"  n tokens            : {a.get('n', 0)}")
        lines.append(f"  Σ Z                 : {a.get('Z_sum', 0.0):+.4f}")
        lines.append(f"  Σ L⁺ (needed)       : {a.get('L_pos', 0.0):+.4f}")
        lines.append(f"  Σ L⁻ (redundant)    : {a.get('L_neg', 0.0):+.4f}")
        lines.append("")
        lines.append("--- transcendental split (asymmetric tokens) ---")
        t = rep.transcendental_stats
        lines.append(f"  n tokens            : {t.get('n', 0)}")
        lines.append(f"  Σ Z                 : {t.get('Z_sum', 0.0):+.4f}")
        lines.append(f"  Σ L⁺ (needed)       : {t.get('L_pos', 0.0):+.4f}")
        lines.append(f"  Σ L⁻ (redundant)    : {t.get('L_neg', 0.0):+.4f}")
        lines.append("")

        # ---- next tokens ----
        lines.append("--- Likelier tokens to add (positive weighted residual) ---")
        lines.append(f"  {'token':<24s}  {'O':>4s}  {'R':>4s}  "
                     f"{'Z':>8s}  {'Π':>6s}  {'L':>8s}")
        for r in rep.needed:
            lines.append(f"  {r.token[:24]:<24s}  {r.O:>4d}  {r.R:>4d}  "
                         f"{r.Z:>+8.3f}  {r.Pi:>6.3f}  {r.L:>+8.3f}")

        lines.append("")
        lines.append("--- Tokens to remove (negative weighted residual) ---")
        lines.append(f"  {'token':<24s}  {'O':>4s}  {'R':>4s}  "
                     f"{'Z':>8s}  {'Π':>6s}  {'L':>8s}")
        for r in rep.redundant:
            lines.append(f"  {r.token[:24]:<24s}  {r.O:>4d}  {r.R:>4d}  "
                         f"{r.Z:>+8.3f}  {r.Pi:>6.3f}  {r.L:>+8.3f}")

        lines.append("")
        lines.append("--- Reading the output ---")
        lines.append("  O      : count of the token in the code")
        lines.append("  R      : count of the token in the requirements")
        lines.append("  Z      : log-rank residual = O − E   (E = R·|code|/|req|)")
        lines.append("  Π      : elliptic projection Π(pos, rank) ∈ [0,1]")
        lines.append("  L      : weighted likelihood = Π · Z")
        lines.append("  L > 0  : token is underused → likely needed next")
        lines.append("  L < 0  : token is overused → likely redundant")
        return "\n".join(lines)


# ============================================================
#  9. Demo
# ============================================================
def demo():
    print("=" * 76)
    print("Planning harness  ·  log-rank projection for next-token planning")
    print("=" * 76)
    print()

    harness = PlanningHarness(K=K_DEFAULT)

    # ---------- existing code (a partial solution) ----------
    existing_code = """
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

    # ---------- tasks with different requirements ----------
    tasks = [
        # Task 1: the code mostly satisfies this
        "Write a small calculator with add, multiply, subtract operations.",

        # Task 2: code is missing multiply / subtract / divide
        "Write a calculator class with add, multiply, subtract and "
        "divide methods. It should also support reset and history.",

        # Task 3: code has nothing about this
        "Implement a binary search tree with insert, search and delete.",

        # Task 4: code is over-engineered for this simple task
        "Print the number 42.",
    ]

    for i, task in enumerate(tasks, 1):
        print("=" * 76)
        print(f"TASK {i}: {task}")
        print("=" * 76)
        rep = harness.plan(task, existing_code, top_k=10)
        print(harness.report(rep))
        print()

    # ---------- another scenario: incremental addition ----------
    print("=" * 76)
    print("INCREMENTAL SCENARIO")
    print("=" * 76)
    print("Existing code implements `add`.  Requirements now ask for")
    print("`multiply` and `subtract` in addition.  The harness should")
    print("rank `multiply` and `subtract` as the likeliest next tokens.")
    print()

    partial_code = """
def add(a, b):
    return a + b
"""
    next_task = """
Add multiply and subtract functions to the calculator.
Both should take two arguments a and b.
"""
    rep = harness.plan(next_task, partial_code, top_k=10)
    print(harness.report(rep))

    print()
    print("Done.")


if __name__ == "__main__":
    demo()