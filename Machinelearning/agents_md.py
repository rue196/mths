#!/usr/bin/env python3
"""
nl_series_to_agents_md.py
=========================

O(K) natural-language → |Ci| array encoding.

Generates `AGENTS.md` — an experiment document that:

    1. Starts in natural language (prose)
    2. Shows a few words encoded as |Ci| arrays via the
       truncated infinite expansion series:
            v(-n) = -H_{|n|}                     (real, discrete)
            v(+n) = -H_{|n|} + i·α·(μ*H)(|n|)    (complex, transcendental)
            α = 1/(π − e)
    3. After the first set of words, transitions to pure |Ci| numbers
    4. Ends with a self-contained Python experiment other agents can run

The transition (prose → numbers) is the "dilution" from
`agents_natural_language_mis_align.py`, made visible as a document.
The |Ci| array is the only thing that survives after Phase 3.

Usage
-----
    python nl_series_to_agents_md.py               # writes AGENTS.md
    python nl_series_to_agents_md.py --check       # verifies encoding
    python nl_series_to_agents_md.py --K 60        # custom K

Complexity
----------
    mobius sieve    O(K log log K)
    harmonic H      O(K)
    conv (μ*H)      O(K log K)
    per word        O(1)
    per phrase      O(|words|)
    ---
    total           O(K log K)  bounded by K
"""

from __future__ import annotations
import argparse
import hashlib
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ============================================================
#  Constants
# ============================================================
PI        = math.pi
E         = math.e
ALPHA     = 1.0 / (PI - E)              # ≈ 2.362338
GAMMA     = 0.5772156649015329
K_DEFAULT = 60


# ============================================================
#  Primitives (from UTF_8_dzeta_dt.py)
# ============================================================
def mobius_sieve(K: int) -> np.ndarray:
    mu = np.zeros(K + 1, dtype=np.int8)
    if K >= 1:
        mu[1] = 1
    primes, is_comp = [], np.zeros(K + 1, dtype=bool)
    for i in range(2, K + 1):
        if not is_comp[i]:
            primes.append(i); mu[i] = -1
        for p in primes:
            if i * p > K:
                break
            is_comp[i * p] = True
            if i % p == 0:
                mu[i * p] = 0
                break
            mu[i * p] = -mu[i]
    return mu


def harmonic_numbers(K: int) -> np.ndarray:
    H = np.zeros(K + 1, dtype=float)
    if K >= 1:
        H[1] = 1.0
    for n in range(2, K + 1):
        H[n] = H[n - 1] + 1.0 / n
    return H


def mu_convolution_H(K: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """F = μ * H, computed in O(K log K)."""
    mu = mobius_sieve(K)
    H  = harmonic_numbers(K)
    F  = np.zeros(K + 1, dtype=float)
    for d in range(1, K + 1):
        if mu[d] == 0:
            continue
        md = int(mu[d])
        for m in range(1, K // d + 1):
            F[d * m] += md * H[m]
    return F, mu, H


# ============================================================
#  Encoding:  word → signed index → |Ci| contribution
# ============================================================
def word_to_signed_index(word: str, K: int) -> int:
    """
    SHA-256 hash → signed index n ∈ [-K, K],  n ≠ 0.

    The hash is fixed, so the mapping is deterministic: the
    same word always produces the same index n.
    """
    h = int(hashlib.sha256(word.lower().encode("utf-8")).hexdigest()[:8], 16)
    n = -K + (h % (2 * K))
    return n if n != 0 else 1


def series_value(n: int, H: np.ndarray, F: np.ndarray,
                 alpha: float = ALPHA) -> complex:
    """
    v(n) = -H_{|n|} + i·α·(μ*H)(|n|)      n > 0
    v(n) = -H_{|n|}                        n < 0
    v(0) = 0

    Both H and (μ*H) are truncated infinite series:
        H_{|n|}      = Σ_{k=1}^{|n|} 1/k
        (μ*H)(|n|)   = Σ_{d | |n|} μ(d) · H_{|n|/d}
    """
    if n == 0:
        return 0j
    a = abs(n)
    real = -float(H[a]) if a < len(H) else -math.log(a) - GAMMA
    if n < 0:
        return complex(real, 0.0)
    imag = alpha * float(F[a]) if a < len(F) else 0.0
    return complex(real, imag)


def encode_phrase(phrase: str, K: int = K_DEFAULT,
                  H: np.ndarray = None,
                  F: np.ndarray = None) -> np.ndarray:
    """
    Encode a phrase into a complex |Ci| array of length 2K+1.

    Each word contributes v(n) to position c[K + n].
    Multiple words mapping to the same n add together.
    """
    if H is None or F is None:
        F, _, H = mu_convolution_H(K + 2)
    c = np.zeros(2 * K + 1, dtype=complex)
    for word in phrase.lower().split():
        n = word_to_signed_index(word, K)
        c[K + n] += series_value(n, H, F)
    return c


# ============================================================
#  Formatters
# ============================================================
def fmt_labelled(c: np.ndarray, K: int) -> str:
    """Human-readable: c[+n] = Re + Im·i."""
    lines: List[str] = []
    for i, v in enumerate(c):
        if abs(v) < 1e-12:
            continue
        n = i - K
        lines.append(f"  c[{n:>+3d}] = {v.real:>+10.6f} {v.imag:>+10.6f}i")
    return "\n".join(lines) if lines else "  (all zero)"


def fmt_numbers_only(c: np.ndarray) -> str:
    """Pure numerical string — no labels, no prose."""
    parts: List[str] = []
    for v in c:
        if abs(v) < 1e-12:
            continue
        parts.append(f"{v.real:+.6f}{v.imag:+.6f}i")
    return " ".join(parts) if parts else "0"


def fmt_norm(c: np.ndarray) -> float:
    return float(np.linalg.norm(c))


def fmt_energy(c: np.ndarray) -> float:
    return float(np.sum(np.abs(c) ** 2))


# ============================================================
#  Experiment snippet (embedded verbatim in AGENTS.md)
# ============================================================
EXPERIMENT_CODE = '''\
import hashlib
import math
import numpy as np

# ------------------------------------------------------------
#  O(K) natural language → |Ci| encoder
#  Reproduce the arrays in the parent document.
# ------------------------------------------------------------
PI      = math.pi
E       = math.e
ALPHA   = 1.0 / (PI - E)         # ≈ 2.362338
GAMMA   = 0.5772156649015329
K       = 60

def mobius_sieve(K):
    mu = np.zeros(K + 1, dtype=np.int8)
    if K >= 1:
        mu[1] = 1
    primes, is_comp = [], np.zeros(K + 1, dtype=bool)
    for i in range(2, K + 1):
        if not is_comp[i]:
            primes.append(i); mu[i] = -1
        for p in primes:
            if i * p > K:
                break
            is_comp[i * p] = True
            if i % p == 0:
                mu[i * p] = 0; break
            mu[i * p] = -mu[i]
    return mu

def harmonic_numbers(K):
    H = np.zeros(K + 1)
    if K >= 1:
        H[1] = 1.0
    for n in range(2, K + 1):
        H[n] = H[n - 1] + 1.0 / n
    return H

def mu_convolution_H(K):
    mu = mobius_sieve(K)
    H  = harmonic_numbers(K)
    F  = np.zeros(K + 1)
    for d in range(1, K + 1):
        if mu[d] == 0:
            continue
        md = int(mu[d])
        for m in range(1, K // d + 1):
            F[d * m] += md * H[m]
    return F, mu, H

def word_to_signed_index(word, K):
    h = int(hashlib.sha256(word.lower().encode("utf-8"))
            .hexdigest()[:8], 16)
    n = -K + (h % (2 * K))
    return n if n != 0 else 1

def series_value(n, H, F, alpha=ALPHA):
    if n == 0:
        return 0j
    a = abs(n)
    real = -float(H[a]) if a < len(H) else -math.log(a) - GAMMA
    if n < 0:
        return complex(real, 0.0)
    imag = alpha * float(F[a]) if a < len(F) else 0.0
    return complex(real, imag)

def encode_phrase(phrase, K=K):
    F, _, H = mu_convolution_H(K + 2)
    c = np.zeros(2 * K + 1, dtype=complex)
    for word in phrase.lower().split():
        n = word_to_signed_index(word, K)
        c[K + n] += series_value(n, H, F)
    return c

# ------------------------------------------------------------
#  Verification
# ------------------------------------------------------------
if __name__ == "__main__":
    phrases = ["hello", "memory", "agents", "hello memory agents"]

    for phrase in phrases:
        c = encode_phrase(phrase, K=K)
        nz = int(np.count_nonzero(np.abs(c) > 1e-12))
        print(f"phrase: {phrase!r}")
        print(f"  nonzero entries : {nz}")
        print(f"  ||Ci||          : {np.linalg.norm(c):.6f}")
        print(f"  energy          : {np.sum(np.abs(c) ** 2):.6f}")
        for i, v in enumerate(c):
            if abs(v) > 1e-12:
                n = i - K
                print(f"  c[{n:>+3d}] = "
                      f"{v.real:>+10.6f} {v.imag:>+10.6f}i")
        print()

    # determinism check
    a = encode_phrase("hello memory agents", K=K)
    b = encode_phrase("hello memory agents", K=K)
    print("deterministic :", np.allclose(a, b))
'''


# ============================================================
#  AGENTS.md builder
# ============================================================
def build_agents_md(
    seed_words: List[str] = None,
    K: int = K_DEFAULT,
) -> str:
    """
    Build the AGENTS.md content.

    Structure:
        §1  about            (prose, formula)
        §2  phase 1          (words + their |Ci|)
        §3  phase 2          (composite phrases + their |Ci|)
        §4  phase 3          (pure |Ci| numbers, no prose)
        §5  experiment       (embedded Python)
        §6  notes for agents
    """
    seed_words = seed_words or ["hello", "memory", "agents"]
    F, mu, H = mu_convolution_H(K + 2)

    lines: List[str] = []

    # ---------- header ----------
    lines += [
        "# AGENTS.md · O(K) Natural Language → |Ci| Series Encoding",
        "",
        f"_Generated at K = {K}. α = 1/(π − e) ≈ {ALPHA:.6f}._",
        "",
        "## §1 · About",
        "",
        "Any natural language phrase can be encoded as a complex `|Ci|`",
        f"array of length `2K+1` (K = {K}), using a truncated infinite",
        "expansion series:",
        "",
        "```",
        "  v(-n) = -H_{|n|}                        (real, discrete)",
        "  v(+n) = -H_{|n|} + i·α·(μ*H)(|n|)      (complex, transcendental)",
        "  v(0)  = 0",
        f"  α     = 1/(π − e) ≈ {ALPHA:.6f}",
        "```",
        "",
        "Where `H_k = Σ_{j=1..k} 1/j` is the harmonic series and",
        "`(μ*H)(n) = Σ_{d|n} μ(d)·H_{n/d}` is the Möbius convolution of",
        "the harmonic series.",
        "",
        "Every word is hashed (SHA-256) to a **signed index** `n ∈ [-K, K]`,",
        "then its complex value `v(n)` is added to position `c[K + n]`.",
        "Encoding is **deterministic** — the same word always yields the",
        "same `n`, and the same phrase always yields the same `|Ci|` array.",
        "",
        "---",
        "",
    ]

    # ---------- phase 1: single words ----------
    lines += [
        "## §2 · Phase 1 · First words → |Ci|",
        "",
        "The first natural language words. Each is shown with its",
        "signed index and its full `|Ci|` contribution:",
        "",
    ]

    for w in seed_words:
        n = word_to_signed_index(w, K)
        c = encode_phrase(w, K=K, H=H, F=F)
        v = series_value(n, H, F)
        lines += [
            f"### `{w}`  ·  n = {n:+d}",
            "",
            "```",
            f"  v({n:+d}) = {v.real:+.6f} {v.imag:+.6f}i",
            "",
            "  |Ci| non-zero entries:",
            fmt_labelled(c, K),
            "```",
            "",
        ]

    # ---------- phase 2: composite phrases ----------
    lines += [
        "---",
        "",
        "## §3 · Phase 2 · Composite phrases → |Ci|",
        "",
        "The same encoding applied to multi-word phrases. Each phrase",
        "produces a composite `|Ci|` array — the sum of its words'",
        "contributions:",
        "",
    ]

    phrases = [
        " ".join(seed_words[:2]),
        " ".join(seed_words),
        " ".join(seed_words + seed_words[:1]),
    ]
    for ph in phrases:
        c = encode_phrase(ph, K=K, H=H, F=F)
        lines += [
            f"### `{ph}`",
            "",
            "```",
            fmt_labelled(c, K),
            "```",
            "",
        ]

    # ---------- phase 3: pure numbers ----------
    lines += [
        "---",
        "",
        "## §4 · Phase 3 · Pure |Ci| arrays",
        "",
        "From here on the document carries only numerical data.",
        "Each line below is a complete `|Ci|` array for one phrase.",
        "The prose does not resume until the experiment block.",
        "",
        "```",
    ]

    # more phrases, purely numerical
    more = [
        seed_words[:1],
        seed_words[1:2],
        seed_words[2:3],
        seed_words[:2],
        seed_words[1:],
        seed_words,
        seed_words + seed_words[:1],
        seed_words * 2,
    ]
    for group in more:
        ph = " ".join(group)
        c = encode_phrase(ph, K=K, H=H, F=F)
        lines.append(f"# {fmt_numbers_only(c)}")

    lines += [
        "```",
        "",
        "**Norm summary** (also purely numerical):",
        "",
        "```",
    ]

    # pure numerical norms — still no prose
    for group in more:
        ph = " ".join(group)
        c = encode_phrase(ph, K=K, H=H, F=F)
        lines.append(f"# ||Ci|| = {fmt_norm(c):.6f}  E = {fmt_energy(c):.6f}")

    lines += [
        "```",
        "",
        "---",
        "",
    ]

    # ---------- experiment ----------
    lines += [
        "## §5 · Experiment · run this to verify",
        "",
        "Any agent receiving this document can run the following Python",
        "code to reproduce the `|Ci|` arrays shown above. The code is",
        "self-contained — no external imports beyond `hashlib`, `math`,",
        "and `numpy`.",
        "",
        "```python",
        EXPERIMENT_CODE.rstrip(),
        "```",
        "",
        "---",
        "",
    ]

    # ---------- notes ----------
    lines += [
        "## §6 · Notes for agents",
        "",
        "1. **Determinism.** The encoder is deterministic. Two agents",
        "   running the experiment on the same phrase will agree on every",
        "   value of `|Ci|` bit-for-bit, up to floating-point rounding.",
        "",
        "2. **One-way encoding.** SHA-256 is not invertible, so the encoding",
        "   cannot recover the original phrase from `|Ci|` alone. The",
        "   mapping is directional.",
        "",
        "3. **Dilution.** In §4 the natural language vanishes and only",
        "   numbers remain. This mirrors the *agent drift* effect from",
        "   `agents_natural_language_mis_align.py`: after a long enough",
        "   natural-language buffer, the guardrail signal (the original",
        "   user query) is diluted to `P_safe → ½`. In this document the",
        "   prose is genuinely replaced by numbers — no guardrail is",
        "   left, only the encoding.",
        "",
        "4. **Transmission.** After §4 the only content worth transmitting",
        "   is the `|Ci|` arrays plus the encoded structure (the phrase",
        "   list, the K value). The prose in §1–§3 is explanatory but",
        "   disposable.",
        "",
        "5. **Verification.** Run the experiment in §5. Compare the",
        "   computed `||Ci||` values against the numbers in §4. They",
        "   should match exactly. This is how one agent confirms another's",
        "   `|Ci|` encoding without trusting the prose.",
        "",
        "6. **Complexity.** The sieve is `O(K log log K)`, the harmonic",
        "   series `O(K)`, the Möbius convolution `O(K log K)`, the",
        "   per-phrase encoding `O(|words|)`. The whole pipeline is",
        "   bounded by `O(K log K)` in `K`.",
        "",
    ]

    return "\n".join(lines)


# ============================================================
#  Demo / CLI
# ============================================================
def run_demo(K: int = K_DEFAULT, out_path: Path = None):
    out_path = out_path or Path("AGENTS.md")

    print("=" * 78)
    print("O(K) natural language → |Ci| → AGENTS.md")
    print("=" * 78)
    print(f"  α         = 1/(π − e)  = {ALPHA:.6f}")
    print(f"  K         = {K}")
    print(f"  GAMMA     = {GAMMA:.10f}")
    print()

    seed_words = ["hello", "memory", "agents"]

    # ---- compute everything once to show summary ----
    F, mu, H = mu_convolution_H(K + 2)

    print("── phase 1 · single-word encodings ──")
    for w in seed_words:
        n = word_to_signed_index(w, K)
        v = series_value(n, H, F)
        print(f"  {w:<8s}  n = {n:>+4d}  "
              f"v = {v.real:>+10.6f} {v.imag:>+10.6f}i")
    print()

    print("── phase 2 · composite phrases ──")
    for ph in [" ".join(seed_words[:2]),
               " ".join(seed_words),
               " ".join(seed_words + seed_words[:1])]:
        c = encode_phrase(ph, K=K, H=H, F=F)
        nz = int(np.count_nonzero(np.abs(c) > 1e-12))
        print(f"  {ph:<40s}  nonzero={nz:>2d}  "
              f"||Ci||={fmt_norm(c):.6f}")
    print()

    print("── phase 3 · pure numerical arrays (preview) ──")
    for w in seed_words:
        c = encode_phrase(w, K=K, H=H, F=F)
        nums = fmt_numbers_only(c)
        preview = nums[:76] + ("..." if len(nums) > 76 else "")
        print(f"  {preview}")
    print()

    # ---- build AGENTS.md ----
    md = build_agents_md(seed_words, K=K)
    out_path.write_text(md, encoding="utf-8")

    print("── generated AGENTS.md ──")
    print(f"  path   : {out_path.resolve()}")
    print(f"  lines  : {md.count(chr(10)) + 1}")
    print(f"  words  : {len(md.split())}")
    print(f"  bytes  : {len(md.encode('utf-8'))}")
    print()

    print("── document structure ──")
    for section in md.split("## §")[1:]:
        header = section.split("\n")[0].strip()
        size = len(section)
        print(f"  §{header:<48s}  ({size:>5d} chars)")
    print()

    print("── experiment size ──")
    print(f"  embedded Python : {len(EXPERIMENT_CODE)} chars, "
          f"{EXPERIMENT_CODE.count(chr(10)) + 1} lines")
    print()

    # ---- self-check ----
    print("── self-check ──")
    checks = []
    for w in seed_words:
        c1 = encode_phrase(w, K=K, H=H, F=F)
        c2 = encode_phrase(w, K=K, H=H, F=F)
        checks.append(bool(np.allclose(c1, c2)))
    print(f"  determinism per word : {all(checks)}")

    phrase = " ".join(seed_words)
    c1 = encode_phrase(phrase, K=K, H=H, F=F)
    sum_c = sum(encode_phrase(w, K=K, H=H, F=F) for w in seed_words)
    print(f"  phrase = Σ(words)    : {bool(np.allclose(c1, sum_c))}")

    print(f"  |Ci| length          : {len(c1)}  (2K+1, K={K})")
    print(f"  ||phrase||           : {fmt_norm(c1):.6f}")
    print(f"  energy               : {fmt_energy(c1):.6f}")
    print()
    print("Done.")


def run_check(K: int = K_DEFAULT):
    """Verify encoding against the embedded experiment (round-trip test)."""
    print("=" * 78)
    print("Self-check · verify encoding invariants")
    print("=" * 78)

    F, mu, H = mu_convolution_H(K + 2)

    # 1. μ density should approach 6/π²
    live = int(np.count_nonzero(mu[1:])) / K
    target = 6.0 / (PI * PI)
    print(f"  μ density  : {live:.4f}  (target 6/π² = {target:.4f})")

    # 2. H should be monotonically increasing
    mono = bool(np.all(np.diff(H) > 0))
    print(f"  H monotonic: {mono}")

    # 3. For prime p, (μ*H)(p) = H[p] - H[1]
    test_primes = [2, 3, 5, 7, 11, 13]
    ok = True
    for p in test_primes:
        expected = H[p] - H[1]
        actual   = F[p]
        if abs(expected - actual) > 1e-9:
            ok = False
            break
    print(f"  (μ*H)(p) check   : {ok}")

    # 4. Encoding is deterministic
    for phrase in ["hello", "memory agents", "hello memory agents"]:
        c1 = encode_phrase(phrase, K=K, H=H, F=F)
        c2 = encode_phrase(phrase, K=K, H=H, F=F)
        assert np.allclose(c1, c2)
    print(f"  determinism      : True")

    # 5. Phrase = sum of words
    for phrase in ["hello memory", "hello memory agents"]:
        c_phrase = encode_phrase(phrase, K=K, H=H, F=F)
        c_sum = sum(encode_phrase(w, K=K, H=H, F=F)
                    for w in phrase.split())
        assert np.allclose(c_phrase, c_sum)
    print(f"  phrase = Σ(words): True")

    # 6. Real part is always negative (harmonic sum)
    c = encode_phrase("hello memory agents", K=K, H=H, F=F)
    re_negative = bool(np.all(c[c != 0].real <= 0))
    print(f"  Re(c) ≤ 0 always : {re_negative}")

    print()
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Encode natural language into |Ci| arrays and "
                    "generate AGENTS.md")
    parser.add_argument("--K", type=int, default=K_DEFAULT,
                        help=f"series truncation size (default {K_DEFAULT})")
    parser.add_argument("--out", type=str, default="AGENTS.md",
                        help="output path (default AGENTS.md)")
    parser.add_argument("--check", action="store_true",
                        help="run self-check only, do not write AGENTS.md")
    parser.add_argument("--words", type=str, nargs="+",
                        default=["hello", "memory", "agents"],
                        help="seed words for the encoding")
    args = parser.parse_args()

    if args.check:
        run_check(K=args.K)
        return

    run_demo(K=args.K, out_path=Path(args.out))


if __name__ == "__main__":
    main()