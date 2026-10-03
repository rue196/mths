#!/usr/bin/env python3
"""
cpu_llm_inference.py
====================

CPU inference with a split 32/64-bit architecture.

  32-bit 1D path     → command execution             O(K)
  64-bit symmetric   → memory search (deterministic) O(K log log K)
  64-bit asymmetric  → prediction (creative)         O(K log K)
  M-matrix + Π       → bounded linear algebra        O(K)

The 32-bit path is the LLM's command tape.  A tape is a list of
32-bit words, each packing (opcode, a, b, c).  Walking the tape in
order is O(K) in the number of words.

Command word (32 bits):
    31..24  opcode   (uint8)
    23..16  arg a    (uint8)
    15..8   arg b    (uint8)
     7..0   arg c    (uint8)

Opcodes:
    0x00 NOP
    0x01 ADD      state += a
    0x02 SUB      state -= a
    0x03 MUL      state *= a
    0x04 MOD      state %= (a | 1)
    0x05 XOR      state ^= a
    0x06 ROTL     rotl32(state, a % 32)
    0x07 ROTR     rotr32(state, a % 32)
    0x08 MIX      state = (state * 2654435761 + a) & 0xFFFFFFFF
    0x10 EMIT     push state to output
    0x20 CALL_MEM state ← memory.search(state)       O(K log log K)
    0x21 CALL_PRED state ← predictor.predict(state)  O(K log K)
    0x22 CALL_LIN state ← lin.m_scalar(state)        O(K)
    0xFF HALT

Linear algebra operator (M-matrix + elliptic bounded algebraic):

    M(x, y) = [x⁻¹; y⁻¹] ⊗ [xⁱ, yⁱ]  +  Π(x, y) · I

    tr M(x, y) = x^(iΠ(x,y) − 1) + y^(iΠ(x,y) − 1) + 2Π(x, y)

Π(x, y) ∈ [0, 1] is the Figure 3.5 elliptic projection.  The
Π-bounded exponent keeps the trace finite for every bounded input.
"""

from __future__ import annotations
import hashlib
import math
import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================
#  Constants
# ============================================================
PI          = math.pi
E           = math.e
ALPHA_SYM   = 1.0 / (PI - E)              # ≈ 2.362
ALPHA_ASYM  = 0.3628
DENSITY     = 6.0 / (PI * PI)             # ≈ 0.6079
UINT32_MAX  = 0xFFFFFFFF
K_DEFAULT   = 128


# ============================================================
#  Möbius sieve  ·  O(K log log K)  once
# ============================================================
def mobius_sieve(K: int) -> List[int]:
    if K < 1:
        return [0] * (K + 1)
    mu = [1] * (K + 1)
    is_comp = [False] * (K + 1)
    for i in range(2, K + 1):
        if not is_comp[i]:
            for j in range(i, K + 1, i):
                mu[j] = -mu[j]
                is_comp[j] = True
            i2 = i * i
            if i2 <= K:
                for j in range(i2, K + 1, i2):
                    mu[j] = 0
    mu[0] = 0
    return mu


# ============================================================
#  Elliptic projection  ·  Figure 3.5  ∈ [0, 1]
# ============================================================
def elliptic_projection(x: float, y: float) -> float:
    u = (x / PI) % 1.0
    v = (y / E) % 1.0
    return 0.5 * (1.0 + math.cos(2 * PI * u) * math.cos(2 * PI * v))


# ============================================================
#  32-bit command word
# ============================================================
def pack_word32(op: int, a: int = 0, b: int = 0, c: int = 0) -> int:
    return (((op & 0xFF) << 24) | ((a & 0xFF) << 16) |
            ((b & 0xFF) << 8)  | (c & 0xFF))


def unpack_word32(word: int) -> Tuple[int, int, int, int]:
    return ((word >> 24) & 0xFF,
            (word >> 16) & 0xFF,
            (word >> 8)  & 0xFF,
            word & 0xFF)


OP_NOP       = 0x00
OP_ADD       = 0x01
OP_SUB       = 0x02
OP_MUL       = 0x03
OP_MOD       = 0x04
OP_XOR       = 0x05
OP_ROTL      = 0x06
OP_ROTR      = 0x07
OP_MIX       = 0x08
OP_EMIT      = 0x10
OP_CALL_MEM  = 0x20
OP_CALL_PRED = 0x21
OP_CALL_LIN  = 0x22
OP_HALT      = 0xFF

OP_NAMES = {
    OP_NOP: "NOP", OP_ADD: "ADD", OP_SUB: "SUB", OP_MUL: "MUL",
    OP_MOD: "MOD", OP_XOR: "XOR", OP_ROTL: "ROTL", OP_ROTR: "ROTR",
    OP_MIX: "MIX", OP_EMIT: "EMIT", OP_CALL_MEM: "CALL_MEM",
    OP_CALL_PRED: "CALL_PRED", OP_CALL_LIN: "CALL_LIN",
    OP_HALT: "HALT",
}


# ============================================================
#  M-matrix linear algebra  ·  O(K) per apply
# ============================================================
class MobiusLinear:
    """
    M(x, y) = [x⁻¹; y⁻¹] ⊗ [xⁱ, yⁱ]  +  Π(x, y) · I

    Diagonal trace:
        tr M(x, y) = x^(iΠ − 1) + y^(iΠ − 1) + 2Π
    """
    def __init__(self, K: int):
        self.K = K

    def m_trace(self, x: float, y: float) -> complex:
        x_ = complex(max(abs(x), 1e-9))
        y_ = complex(max(abs(y), 1e-9))
        Pi = elliptic_projection(x, y)
        exp = 1j * Pi - 1.0
        return x_ ** exp + y_ ** exp + 2.0 * Pi

    def m_apply(self, v: np.ndarray) -> np.ndarray:
        K = self.K
        out = np.zeros(K, dtype=complex)
        for i in range(K - 1):
            out[i] = self.m_trace(float(v[i]), float(v[i + 1]))
        return out

    def m_scalar(self, v: np.ndarray) -> float:
        out = self.m_apply(v)
        return float(np.abs(np.sum(out)))


# ============================================================
#  Symmetric memory  ·  64-bit  ·  O(K log log K)
# ============================================================
class SymmetricMemory64:
    """
    Deterministic |Ci| store at 64-bit precision, indexed by the
    Möbius sieve.  Construction of the sieve is O(K log log K).
    A lookup is O(1) by index.
    """
    def __init__(self, K: int, mu: List[int]):
        self.K = K
        self.mu = mu
        self.slots: Dict[int, np.ndarray] = {}

    def embed(self, tokens: List[str]) -> np.ndarray:
        K = self.K
        c = np.zeros(2 * K + 1, dtype=np.float64)
        for t in tokens:
            h = int(hashlib.md5(str(t).encode()).hexdigest()[:8], 16)
            i = (h % K) + 1
            if self.mu[i] != 0:                          # square-free only
                c[K + i] += 1.0
                c[K - i] += 1.0
        return c

    def store(self, key: str, tokens: List[str]) -> None:
        h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
        idx = h % self.K
        self.slots[idx] = self.embed(tokens)

    def search(self, state: int) -> int:
        idx = state % self.K
        c = self.slots.get(idx)
        if c is None:
            return state
        S = 0.0
        for k, val in enumerate(c):
            S += val if (k % 2 == 0) else -val
        return int(abs(S) * 1000) & UINT32_MAX


# ============================================================
#  Asymmetric predictor  ·  64-bit  ·  O(K log K)
# ============================================================
class AsymmetricPredictor64:
    """
    Creative prediction via inverse-score merge sort over a small
    library of stored fragments.  Each prediction is O(F · K log K)
    where F is the number of fragments and K the embedding size.
    """
    def __init__(self, K: int, mu: List[int]):
        self.K = K
        self.mu = mu
        self.fragments: List[Tuple[str, np.ndarray]] = []

    def add(self, text: str) -> None:
        self.fragments.append((text, self._embed(text)))

    def predict(self, state: int, query: str = "") -> int:
        if not self.fragments:
            return state
        q = self._embed(query if query else str(state))
        scored = []
        for i, (_, frag) in enumerate(self.fragments):
            scored.append((self._inverse_score(q, frag), i))
        scored.sort()
        return int(abs(scored[0][0]) * UINT32_MAX) & UINT32_MAX

    def _embed(self, text: str) -> np.ndarray:
        K = self.K
        c = np.zeros(2 * K + 1, dtype=np.float64)
        for t in text.lower().split():
            h = int(hashlib.md5(t.encode()).hexdigest()[:8], 16)
            i = (h % K) + 1
            if self.mu[i] != 0:
                c[K + i] += 1.0
                c[K - i] += 1.0
        return c

    @staticmethod
    def _inverse_score(a: np.ndarray, b: np.ndarray) -> float:
        n = len(a)
        if n != len(b) or n < 2:
            return 0.5
        pairs = sorted(zip(a.tolist(), b.tolist()), key=lambda p: p[0])
        b_sorted = [p[1] for p in pairs]
        inv = _merge_sort_inv(b_sorted)
        mx = n * (n - 1) // 2
        return inv / mx if mx > 0 else 0.0


def _merge_sort_inv(arr: List[float]) -> int:
    n = len(arr)
    if n < 2:
        return 0
    return _ms_count(arr, [0] * n, 0, n - 1)


def _ms_count(arr, tmp, l, r):
    inv = 0
    if l < r:
        m = (l + r) // 2
        inv += _ms_count(arr, tmp, l, m)
        inv += _ms_count(arr, tmp, m + 1, r)
        inv += _ms_merge(arr, tmp, l, m, r)
    return inv


def _ms_merge(arr, tmp, l, m, r):
    i, j, k, inv = l, m + 1, l, 0
    while i <= m and j <= r:
        if arr[i] <= arr[j]:
            tmp[k] = arr[i]; i += 1
        else:
            tmp[k] = arr[j]; inv += m - i + 1; j += 1
        k += 1
    while i <= m: tmp[k] = arr[i]; i += 1; k += 1
    while j <= r: tmp[k] = arr[j]; j += 1; k += 1
    for i in range(l, r + 1):
        arr[i] = tmp[i]
    return inv


# ============================================================
#  32-bit command path  ·  O(K) to walk
# ============================================================
class CommandPath32:
    """A 32-bit command tape.  Walking is O(K) in the words."""
    def __init__(self, words: List[int], engine: "CPUInferenceEngine"):
        self.words = list(words)
        self.engine = engine
        self.state = 0
        self.output: List[int] = []
        self.trace: List[Tuple[int, int]] = []
        self.halted = False

    def step(self) -> bool:
        if self.halted or not self.words:
            return False
        w = self.words.pop(0)
        op, a, b, c = unpack_word32(w)
        self._exec(op, a, b, c)
        self.trace.append((op, self.state))
        return True

    def _exec(self, op: int, a: int, b: int, c: int) -> None:
        s = self.state
        if   op == OP_NOP:       pass
        elif op == OP_ADD:       s = (s + a) & UINT32_MAX
        elif op == OP_SUB:       s = (s - a) & UINT32_MAX
        elif op == OP_MUL:       s = (s * a) & UINT32_MAX
        elif op == OP_MOD:       s = s % (a | 1)
        elif op == OP_XOR:       s = (s ^ a) & UINT32_MAX
        elif op == OP_ROTL:
            k = a & 31
            s = ((s << k) | (s >> (32 - k))) & UINT32_MAX
        elif op == OP_ROTR:
            k = a & 31
            s = ((s >> k) | (s << (32 - k))) & UINT32_MAX
        elif op == OP_MIX:
            s = (s * 2654435761 + a) & UINT32_MAX
        elif op == OP_EMIT:
            self.output.append(s)
        elif op == OP_CALL_MEM:
            s = self.engine.memory.search(s)
        elif op == OP_CALL_PRED:
            s = self.engine.predictor.predict(s, self.engine.last_query)
        elif op == OP_CALL_LIN:
            K = self.engine.K
            v = np.array([(s % 1000) / 1000.0 + 0.5] * K)
            scal = self.engine.lin.m_scalar(v)
            s = int(scal * 1000) & UINT32_MAX
        elif op == OP_HALT:
            self.halted = True
        self.state = s

    def run(self, max_steps: Optional[int] = None) -> List[int]:
        limit = max_steps if max_steps is not None else len(self.words) + 8
        steps = 0
        while not self.halted and self.words and steps < limit:
            if not self.step():
                break
            steps += 1
        return self.output


# ============================================================
#  CPU inference engine
# ============================================================
class CPUInferenceEngine:
    """
    Routes a query through the split 32/64-bit architecture.

        deterministic  →  32-bit command tape          O(K)
        memory lookup  →  SymmetricMemory64            O(K log log K)
        creative       →  AsymmetricPredictor64        O(K log K)
        linear algebra →  MobiusLinear                 O(K)
    """
    def __init__(self, K: int = K_DEFAULT):
        self.K = K
        t0 = time.perf_counter()
        self.mu = mobius_sieve(K)                        # O(K log log K)
        self.sieve_ms = (time.perf_counter() - t0) * 1e3
        self.memory    = SymmetricMemory64(K, self.mu)
        self.predictor = AsymmetricPredictor64(K, self.mu)
        self.lin       = MobiusLinear(K)
        self.last_query = ""

        # seed memory
        self.memory.store("greeting", ["hello", "world"])
        self.memory.store("capital",  ["paris", "france"])
        self.memory.store("answer",   ["42"])

        # seed predictor
        for frag in [
            "love is the quiet space where two silences agree",
            "a colour that feels like rain on a warm afternoon",
            "the answer drifts between memory and wanting",
            "the hour when lamps begin to think for us",
            "a scent the room remembers but you cannot name",
        ]:
            self.predictor.add(frag)

    # ---------- classification ----------
    def classify(self, query: str) -> str:
        q = query.strip().lower()
        if re.match(r"^[\s0-9\.\+\-\*/\(\)%]+$", q) and \
           any(op in q for op in "+-*/%"):
            return "arith"
        if "how many" in q or "length" in q or q.startswith("len "):
            return "count"
        return "asym"

    # ---------- build a 32-bit command tape ----------
    def build_path(self, query: str) -> CommandPath32:
        kind = self.classify(query)
        self.last_query = query
        words: List[int] = []

        if kind == "arith":
            try:
                val = eval(query, {"__builtins__": {}}, {})
                if isinstance(val, float) and val.is_integer():
                    val = int(val)
            except Exception:
                val = 0
            v = val & UINT32_MAX
            words.append(pack_word32(OP_XOR,  v & 0xFF))
            words.append(pack_word32(OP_MIX, (v >> 8) & 0xFF))
            words.append(pack_word32(OP_EMIT))
            words.append(pack_word32(OP_HALT))

        elif kind == "count":
            words.append(pack_word32(OP_MIX, 0x07))
            words.append(pack_word32(OP_CALL_MEM))
            words.append(pack_word32(OP_EMIT))
            words.append(pack_word32(OP_HALT))

        else:
            words.append(pack_word32(OP_MIX, 0x07))
            words.append(pack_word32(OP_CALL_PRED))
            words.append(pack_word32(OP_CALL_LIN))
            words.append(pack_word32(OP_EMIT))
            words.append(pack_word32(OP_HALT))

        return CommandPath32(words, self)

    # ---------- infer ----------
    def infer(self, query: str) -> Dict:
        path = self.build_path(query)
        t0 = time.perf_counter()
        out = path.run()
        dt = (time.perf_counter() - t0) * 1e3
        return dict(
            query=query,
            kind=self.classify(query),
            output=out,
            final_state=path.state,
            steps=len(path.trace),
            elapsed_ms=dt,
        )


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("CPU inference  ·  32-bit command tape + 64-bit numerics")
    print("=" * 78)

    engine = CPUInferenceEngine(K=K_DEFAULT)
    print(f"  sieve build (O(K log log K))  = {engine.sieve_ms:.2f} ms")
    print(f"  K                             = {engine.K}")
    print(f"  α_sym  = 1/(π − e)            = {ALPHA_SYM:.6f}")
    print(f"  α_asym = 0.3628               = {ALPHA_ASYM:.6f}")
    print()

    # ---------- 32-bit tape step by step ----------
    print("--- 32-bit command tape  ·  O(K) per tape ---")
    path = engine.build_path("2 + 2")
    print(f"  tape length : {len(path.words)}")
    for w in path.words:
        op, a, b, c = unpack_word32(w)
        print(f"    word 0x{w:08x}  op={OP_NAMES.get(op, '?'):>10s}  "
              f"a={a:3d}  b={b:3d}  c={c:3d}")
    out = path.run()
    print(f"  output      : {out}")
    print(f"  final state : 0x{path.state:08x}")
    print()

    # ---------- inference over queries ----------
    queries = [
        "2 + 2",
        "17 * 23 - 100",
        "how many r in strawberry",
        "length of hello",
        "what is love",
        "describe a quiet morning",
        "suggest a colour for autumn",
    ]

    print("--- inference ---")
    print(f"{'query':34s}  {'kind':>6s}  {'steps':>5s}  "
          f"{'state':>12s}  {'ms':>8s}  output")
    print("-" * 96)

    for q in queries:
        r = engine.infer(q)
        out_str = str(r["output"]) if r["output"] else "—"
        if len(out_str) > 20:
            out_str = out_str[:17] + "…"
        print(f"{q[:34]:34s}  {r['kind']:>6s}  {r['steps']:>5d}  "
              f"0x{r['final_state']:08x}  {r['elapsed_ms']:>8.3f}  {out_str}")
    print()

    # ---------- timing per layer ----------
    print("--- per-layer timings ---")

    t0 = time.perf_counter()
    for i in range(1000):
        _ = engine.memory.search(i)
    dt = (time.perf_counter() - t0) * 1e3
    print(f"  memory    1000 lookups        = {dt:8.2f} ms  "
          f"({dt/1000:.4f} ms each)")

    t0 = time.perf_counter()
    for i in range(100):
        _ = engine.predictor.predict(i, "what is love")
    dt = (time.perf_counter() - t0) * 1e3
    print(f"  predictor 100 predictions     = {dt:8.2f} ms  "
          f"({dt/100:.4f} ms each)")

    v = np.linspace(0.5, 5.0, engine.K)
    t0 = time.perf_counter()
    for _ in range(100):
        _ = engine.lin.m_scalar(v)
    dt = (time.perf_counter() - t0) * 1e3
    print(f"  lin alg   100 M-applications  = {dt:8.2f} ms  "
          f"({dt/100:.4f} ms each)")
    print()

    # ---------- lin alg samples ----------
    print("--- M-matrix trace samples (Π-bounded) ---")
    print(f"  {'x':>6s}  {'y':>6s}  {'Π(x,y)':>8s}  "
          f"{'Re tr':>9s}  {'Im tr':>9s}  {'|tr|':>9s}")
    for (x, y) in [(0.5, 0.5), (1.0, 1.0), (2.0, 3.0),
                   (PI / 2, E / 2), (4.0, 4.0)]:
        tr = engine.lin.m_trace(x, y)
        Pi = elliptic_projection(x, y)
        print(f"  {x:6.3f}  {y:6.3f}  {Pi:8.4f}  "
              f"{tr.real:9.4f}  {tr.imag:9.4f}  {abs(tr):9.4f}")
    print()

    # ---------- complexity ----------
    print("--- complexity ---")
    print("  Möbius sieve               O(K log log K)   once")
    print("  32-bit command tape walk   O(K)             per tape")
    print("  symmetric memory lookup    O(1)             per index")
    print("  asymmetric prediction      O(F · K log K)   per query")
    print("       (F = number of stored fragments; F ≈ K → O(K² log K))")
    print("  M-matrix linear algebra    O(K)             per apply")
    print()
    print("Done.")


if __name__ == "__main__":
    demo()