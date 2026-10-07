#!/usr/bin/env python3
"""
mobius_dataset_decompiler.py
============================

Five-stage pipeline with the elliptic Möbius SAT chain of thought:

  dataset / source
    │
    ▼  stage 1  ·  32-bit decompile            O(K)
    ▼  stage 2  ·  32-bit Möbius memory        O(K log log K)
    ▼  stage 3  ·  32-bit sparse weight lookup O(N²)
    ▼  stage 4  ·  64-bit unpack               O(K)
    ▼  stage 5  ·  elliptic Möbius SAT chain   O(N · K)
    │
    ▼  user-facing response (clean prose only)

The chain of thought is imported from
    reasoning_chain_of_thought_tokenizer.py
and driven by:
    elliptic trace      tr(a) = x^(i−1) + y^(i−1)
    quadratic address   q(a) = A a² + B a + C  mod K
    Möbius gate         μ(q(a)) ≠ 0
    supertrace          S = Σ (−1)^t |conv_t|
    mass                m = |S| e^{−H}
"""

from __future__ import annotations

import ast
import hashlib
import math
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np
import struct

# ============================================================
#  Constants
# ============================================================
PI          = math.pi
E           = math.e
ALPHA       = 1.0 / (PI - E)             # ≈ 2.362
NORM        = 1.0 - math.exp(-ALPHA * (PI + E))
DENSITY     = 6.0 / (PI * PI)            # ≈ 0.6079

K_MEM       = 512
K_FILTER    = 2 ** 8 + 1
N_SPARSE    = 64
UINT32_MAX  = 0xFFFFFFFF
UINT64_MAX  = 0xFFFFFFFFFFFFFFFF


# ============================================================
#  Möbius sieve
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
#  32-bit command word
# ============================================================
class Op(IntEnum):
    LOAD_CONST = 0x01
    LOAD_NAME  = 0x02
    STORE_NAME = 0x03
    BINOP      = 0x04
    UNARYOP    = 0x05
    CALL       = 0x06
    ATTRIBUTE  = 0x07
    SUBSCRIPT  = 0x08
    COMPARE    = 0x09
    JUMP       = 0x0A
    JUMP_IF    = 0x0B
    RETURN     = 0x0C
    IMPORT     = 0x0D
    MAKE_FUNC  = 0x0E
    MAKE_CLASS = 0x0F
    YIELD      = 0x10
    LOOP_START = 0x11
    LOOP_END   = 0x12
    HALT       = 0xFF


def pack32(op: int, a: int = 0, b: int = 0, c: int = 0) -> int:
    return (((op & 0xFF) << 24) |
            ((a & 0xFF) << 16) |
            ((b & 0xFF) << 8)  |
            (c & 0xFF)) & UINT32_MAX


def unpack32(w: int) -> Tuple[int, int, int, int]:
    return ((w >> 24) & 0xFF,
            (w >> 16) & 0xFF,
            (w >> 8)  & 0xFF,
            w & 0xFF)


OP_NAMES = {int(o): o.name for o in Op}


# ============================================================
#  Stage 1  ·  32-bit decompile
# ============================================================
class TapeBuilder(ast.NodeVisitor):
    def __init__(self):
        self.words: List[int] = []
        self.names: List[str] = []
        self.consts: List[object] = []
        self._name_idx: Dict[str, int] = {}

    def _name(self, n: str) -> int:
        if n not in self._name_idx:
            self._name_idx[n] = len(self.names)
            self.names.append(n)
        return self._name_idx[n] & 0xFF

    def _const(self, v) -> int:
        try:
            hash(v)
        except TypeError:
            v = repr(v)
        self.consts.append(v)
        return (len(self.consts) - 1) & 0xFF

    def _emit(self, op, a=0, b=0, c=0) -> None:
        self.words.append(pack32(op, a, b, c))

    def visit_Module(self, node):
        for child in node.body:
            self.visit(child)
        self._emit(Op.HALT)

    def visit_FunctionDef(self, node):
        self._emit(Op.MAKE_FUNC, self._name(node.name))
        for child in node.body:
            self.visit(child)
        self._emit(Op.RETURN)

    def visit_ClassDef(self, node):
        self._emit(Op.MAKE_CLASS, self._name(node.name))
        for child in node.body:
            self.visit(child)

    def visit_Return(self, node):
        if node.value is not None:
            self.visit(node.value)
        self._emit(Op.RETURN)

    def visit_Assign(self, node):
        self.visit(node.value)
        for t in node.targets:
            if isinstance(t, ast.Name):
                self._emit(Op.STORE_NAME, self._name(t.id))

    def visit_AugAssign(self, node):
        self.visit(node.value)
        self._emit(Op.BINOP, 0x01)
        if isinstance(node.target, ast.Name):
            self._emit(Op.STORE_NAME, self._name(node.target.id))

    def visit_Expr(self, node):
        self.visit(node.value)

    def visit_If(self, node):
        self.visit(node.test)
        self._emit(Op.JUMP_IF, 0)
        for child in node.body:
            self.visit(child)
        if node.orelse:
            self._emit(Op.JUMP, 0)
            for child in node.orelse:
                self.visit(child)

    def visit_For(self, node):
        self._emit(Op.LOOP_START, 0)
        self.visit(node.iter)
        for child in node.body:
            self.visit(child)
        self._emit(Op.LOOP_END, 0)

    def visit_While(self, node):
        self._emit(Op.LOOP_START, 1)
        self.visit(node.test)
        self._emit(Op.JUMP_IF, 0)
        for child in node.body:
            self.visit(child)
        self._emit(Op.LOOP_END, 1)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            self._emit(Op.LOAD_NAME, self._name(node.func.id))
        elif isinstance(node.func, ast.Attribute):
            self.visit(node.func.value)
            self._emit(Op.ATTRIBUTE, self._name(node.func.attr))
        else:
            self.visit(node.func)
        for arg in node.args:
            self.visit(arg)
        self._emit(Op.CALL, len(node.args) & 0xFF, 0)

    def visit_Attribute(self, node):
        self.visit(node.value)
        self._emit(Op.ATTRIBUTE, self._name(node.attr))

    def visit_Subscript(self, node):
        self.visit(node.value)
        self.visit(node.slice)
        self._emit(Op.SUBSCRIPT)

    def visit_BinOp(self, node):
        self.visit(node.left)
        self.visit(node.right)
        self._emit(Op.BINOP, 0x01)

    def visit_UnaryOp(self, node):
        self.visit(node.operand)
        self._emit(Op.UNARYOP, 0x02)

    def visit_Compare(self, node):
        self.visit(node.left)
        for c in node.comparators:
            self.visit(c)
        self._emit(Op.COMPARE, 0x01)

    def visit_Import(self, node):
        for alias in node.names:
            self._emit(Op.IMPORT, self._name(alias.name.split(".")[0]))

    def visit_ImportFrom(self, node):
        self._emit(Op.IMPORT, self._name((node.module or "").split(".")[0]))

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self._emit(Op.LOAD_NAME, self._name(node.id))
        elif isinstance(node.ctx, ast.Store):
            self._emit(Op.STORE_NAME, self._name(node.id))

    def visit_Constant(self, node):
        self._emit(Op.LOAD_CONST, self._const(node.value))

    def visit_List(self, node):
        for e in node.elts:
            self.visit(e)

    def visit_Tuple(self, node):
        for e in node.elts:
            self.visit(e)

    def visit_Dict(self, node):
        for k, v in zip(node.keys, node.values):
            self.visit(k)
            self.visit(v)


def decompile_32bit(source: str, name: str = "<source>") -> Dict:
    t0 = time.perf_counter()
    tree = ast.parse(source)
    builder = TapeBuilder()
    builder.visit(tree)

    h32 = hashlib.sha256(source.encode()).digest()[:4]
    addr32 = int.from_bytes(h32, "big") & UINT32_MAX

    return dict(
        source_name=name, tree=tree,
        words=builder.words, names=builder.names,
        consts=builder.consts,
        n_words=len(builder.words), addr32=addr32,
        elapsed_ms=(time.perf_counter() - t0) * 1e3,
    )


# ============================================================
#  Stage 2  ·  32-bit Möbius memory compress
# ============================================================
@dataclass
class MobiusMemory32:
    K: int
    mu: np.ndarray
    slots: List[Tuple[int, int]]
    q_addr: List[int]
    S: float
    H: float
    m: float
    addr32: int
    n_raw: int

    @classmethod
    def compress(cls, words: List[int], K: int = K_MEM) -> "MobiusMemory32":
        mu = mobius_sieve(K)
        kept_indices = [k for k in range(1, K + 1) if mu[k] != 0]
        slots: List[Tuple[int, int]] = []
        for pos, w in enumerate(words):
            k = kept_indices[pos % len(kept_indices)]
            slots.append((k, int(w) & UINT32_MAX))
        q_addr = [((k * k + k) % K) for k, _ in slots]
        S = 0.0
        for t, (_, w) in enumerate(slots):
            val = abs((w & 0xFFFF) - 0x8000) / 0x8000
            S += val if (t % 2 == 0) else -val
        N = max(len(slots), 1)
        p = abs(S) / N
        H = -ALPHA * p * math.log(p) if 0.0 < p < 1.0 else 0.0
        m = abs(S) * math.exp(-H) if H < 700 else 0.0
        h = hashlib.sha256()
        for k, w in slots:
            h.update(bytes([k & 0xFF, (k >> 8) & 0xFF]))
            h.update(w.to_bytes(4, "big"))
        addr32 = int.from_bytes(h.digest()[:4], "big") & UINT32_MAX
        return cls(K=K, mu=mu, slots=slots, q_addr=q_addr,
                   S=float(S), H=float(H), m=float(m),
                   addr32=addr32, n_raw=len(words))


# ============================================================
#  Stage 3  ·  32-bit sparse weight lookup
# ============================================================
@dataclass
class SparseEntry:
    idx: int
    mobius_k: int
    word32: int
    q_addr: int
    weight: float
    sym_gate: bool
    asym_gate: bool
    cross_gate: bool


class SparseWeight32:
    def __init__(self, K: int):
        self.K = K
        self.mu = mobius_sieve(K)

    def _proj(self, x: float, y: float) -> float:
        u = (x / PI) % 1.0
        v = (y / E) % 1.0
        return 0.5 * (1.0 + math.cos(2 * PI * u) * math.cos(2 * PI * v))

    def lookup(self, mem: MobiusMemory32,
               N: int = N_SPARSE) -> List[SparseEntry]:
        K = self.K
        mu = self.mu
        decay = 1.0 / (1.0 + abs(mem.S))
        entries: List[SparseEntry] = []
        for i, (k, w) in enumerate(mem.slots[:N]):
            q_sym = (i * (i + 1)) % K
            q_asym = (i * i + (i + 1) * (i + 1)) % K
            a = (i * K + k) % (K * K)
            q_cross = ((a * a) + a) % K
            g_sym = mu[q_sym] != 0
            g_asym = mu[q_asym] != 0
            g_cross = mu[q_cross] != 0
            if not (g_sym or g_asym or g_cross):
                continue
            x = 0.5 + (k / max(K, 1)) * 4.5
            y = 0.5 + ((w & 0xFFFF) / 0xFFFF) * 4.5
            entries.append(SparseEntry(
                idx=len(entries), mobius_k=k, word32=w,
                q_addr=mem.q_addr[i],
                weight=self._proj(x, y) * decay,
                sym_gate=g_sym, asym_gate=g_asym,
                cross_gate=g_cross,
            ))
        return entries


# ============================================================
#  Stage 4  ·  64-bit unpack
# ============================================================
@dataclass
class Unpacked64:
    lines: List[str]
    n_words: int
    addr64: int
    n_steps: int
    elapsed_ms: float


class Unpack64:
    def __init__(self, names: List[str], consts: List[object]):
        self.names = names
        self.consts = consts

    def _name(self, i: int) -> str:
        return self.names[i] if i < len(self.names) else f"<n{i}>"

    def _const(self, i: int) -> str:
        return repr(self.consts[i]) if i < len(self.consts) else "<c>"

    def walk(self, entries: List[SparseEntry],
             mem: MobiusMemory32) -> Unpacked64:
        t0 = time.perf_counter()
        lines: List[str] = []
        for e in entries:
            op, a, b, c = unpack32(e.word32)
            opname = OP_NAMES.get(op, f"OP{op:#x}")
            if op == Op.LOAD_CONST:
                lines.append(f"{opname} {self._const(a)}")
            elif op == Op.LOAD_NAME:
                lines.append(f"{opname} {self._name(a)}")
            elif op == Op.STORE_NAME:
                lines.append(f"{opname} {self._name(a)}")
            elif op == Op.MAKE_FUNC:
                lines.append(f"def {self._name(a)}():")
            elif op == Op.MAKE_CLASS:
                lines.append(f"class {self._name(a)}:")
            elif op == Op.IMPORT:
                lines.append(f"import {self._name(a)}")
            elif op == Op.CALL:
                lines.append(f"call  a={a}  b={b}")
            elif op == Op.BINOP:
                lines.append(f"binop  a={a}")
            elif op == Op.RETURN:
                lines.append("return")
            elif op == Op.JUMP_IF:
                lines.append("if <test>:")
            elif op == Op.JUMP:
                lines.append("else:")
            elif op == Op.LOOP_START:
                lines.append(f"loop {a} start")
            elif op == Op.LOOP_END:
                lines.append(f"loop {a} end")
            elif op == Op.HALT:
                lines.append("halt")
            else:
                lines.append(f"{opname}  a={a}  b={b}  c={c}")
        h = hashlib.sha256()
        for line in lines:
            h.update(line.encode("utf-8"))
        h.update(struct.pack(">d", mem.S))
        addr64 = int.from_bytes(h.digest()[:8], "big") & UINT64_MAX
        return Unpacked64(lines=lines, n_words=len(entries),
                          addr64=addr64, n_steps=len(entries),
                          elapsed_ms=(time.perf_counter() - t0) * 1e3)


# ============================================================
#  Stage 5  ·  Elliptic Möbius SAT chain of thought
#             (verbatim from reasoning_chain_of_thought_tokenizer.py)
# ============================================================
@dataclass
class Statement:
    text: str
    args: List[str] = field(default_factory=list)
    source: str = "buffer"

    @property
    def arity(self) -> int:
        return len(self.args)


@dataclass
class Query:
    text: str
    clauses: List[List[Tuple[int, int]]] = field(default_factory=list)


class EllipticMobiusSAT:
    def __init__(self, quad_A: int = 1, quad_B: int = 1, quad_C: int = 0,
                 i_exp: float = 2.0, K_filter: int = K_FILTER):
        self.A, self.B, self.C = quad_A, quad_B, quad_C
        self.i_exp = i_exp
        self.K = K_filter
        self.mu = mobius_sieve(self.K)

    @staticmethod
    def args_to_bits(args: List[str], width: int = 8) -> int:
        bits = 0
        for i, a in enumerate(args[:width]):
            if a:
                bits |= (1 << i)
        return bits

    def elliptic_address(self, a: int, n_bits: int) -> Tuple[float, float]:
        bits = [(a >> i) & 1 for i in range(n_bits)]
        t = sum(b * math.sin((i + 1) * 0.7) for i, b in enumerate(bits))
        u = sum(b * math.cos((i + 1) * 1.3) for i, b in enumerate(bits))
        x = 0.5 + 4.5 * (0.5 * (1.0 + math.tanh(t)))
        y = 0.5 + 4.5 * (0.5 * (1.0 + math.tanh(u)))
        return x, y

    def trace_of(self, a: int, n_bits: int) -> float:
        x, y = self.elliptic_address(a, n_bits)
        return x ** (self.i_exp - 1) + y ** (self.i_exp - 1)

    def quadratic_address(self, a: int) -> int:
        return (self.A * a * a + self.B * a + self.C) % self.K

    def mobius_gate(self, a: int) -> bool:
        return self.mu[self.quadratic_address(a)] != 0

    @staticmethod
    def _lit(lit, a):
        slot, sign = lit
        bit = (a >> slot) & 1
        return bool(bit) if sign > 0 else not bool(bit)

    @classmethod
    def clause_true(cls, clause, a) -> bool:
        return any(cls._lit(l, a) for l in clause)

    @classmethod
    def formula_true(cls, clauses, a) -> bool:
        return all(cls.clause_true(c, a) for c in clauses)

    def decide(self, stmt: Statement, query: Query) -> Dict:
        k = stmt.arity
        n_bits = max(4, min(8, k + 2))
        a = self.args_to_bits(stmt.args, width=n_bits)
        q = self.quadratic_address(a)
        gate = self.mobius_gate(a)
        formula = self.formula_true(query.clauses, a)
        trace = self.trace_of(a, n_bits)
        accepted = gate and formula
        return dict(
            statement=stmt, arity=k, assignment=a,
            q_addr=q, mu_q=self.mu[q],
            gate=gate, formula=formula,
            trace=trace, accepted=accepted,
        )


def conv_exp_kernel(signal: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    K = len(signal)
    if K == 0:
        return signal
    lam = math.exp(-alpha)
    f = np.zeros(K); f[0] = signal[0]
    for i in range(1, K):
        f[i] = signal[i] + lam * f[i - 1]
    b = np.zeros(K); b[K - 1] = signal[-1]
    for i in range(K - 2, -1, -1):
        b[i] = signal[i] + lam * b[i + 1]
    conv_exp = (f + b - signal) / (1 - lam * lam)
    return (1.0 - conv_exp) / NORM


def supertrace_and_mass(signal: np.ndarray) -> Tuple[float, float, float]:
    S = 0.0
    for i, v in enumerate(signal):
        S += v if (i % 2 == 0) else -v
    if S == 0.0:
        return 0.0, 0.0, 0.0
    p = abs(S) / len(signal)
    if not (0.0 < p < 1.0):
        return S, 0.0, abs(S)
    H = -ALPHA * p * math.log(p)
    m = abs(S) * math.exp(-H) if H < 700 else 0.0
    return S, H, m


class ReasoningBuffer:
    def __init__(self, sat: Optional[EllipticMobiusSAT] = None,
                 max_survivors: int = 32):
        self.sat = sat or EllipticMobiusSAT()
        self.max_survivors = max_survivors
        self.accepted: List[Dict] = []
        self.rejected: List[Dict] = []
        self.cycle = 0

    def think(self, query: Query,
              candidates: List[Statement]) -> Dict:
        self.cycle += 1
        cycle_records = []
        for stmt in candidates:
            rec = self.sat.decide(stmt, query)
            cycle_records.append(rec)
            if rec["accepted"]:
                self.accepted.append(rec)
            else:
                self.rejected.append(rec)

        traces = np.array([r["trace"] for r in cycle_records], dtype=float)
        if len(traces) >= 2:
            conv = conv_exp_kernel(traces)
            S, H, m = supertrace_and_mass(conv)
        else:
            S, H, m = 0.0, 0.0, 0.0

        if self.accepted:
            self.accepted.sort(key=lambda r: -abs(r["trace"]))
            self.accepted = self.accepted[: self.max_survivors]

        return dict(
            cycle=self.cycle,
            n_candidates=len(candidates),
            n_accepted=sum(1 for r in cycle_records if r["accepted"]),
            n_rejected=sum(1 for r in cycle_records if not r["accepted"]),
            S=S, H=H, m=m,
            best=self.accepted[0] if self.accepted else None,
            records=cycle_records,
        )

    def emit(self) -> Optional[str]:
        if not self.accepted:
            return None
        return self.accepted[0]["statement"].text

    def debug_state(self) -> List[Dict]:
        return [
            dict(text=r["statement"].text, arity=r["arity"],
                 trace=round(r["trace"], 4),
                 q=r["q_addr"], mu_q=r["mu_q"])
            for r in self.accepted
        ]


class Presenter:
    def __init__(self):
        self.history: List[str] = []

    def present(self, text: Optional[str]) -> str:
        reply = text if text is not None else \
                "I don't have a considered answer for that yet."
        self.history.append(reply)
        return reply


# ============================================================
#  Full pipeline
# ============================================================
def pipeline(sources: List[Tuple[str, str]],
             query: Query,
             K_mem: int = K_MEM,
             N_sparse: int = N_SPARSE) -> Dict:
    print("=" * 78)
    print("Möbius dataset decompiler  ·  elliptic Möbius SAT chain of thought")
    print("=" * 78)
    print(f"  α = 1/(π − e)       = {ALPHA:.6f}")
    print(f"  K_mem               = {K_mem}")
    print(f"  N_sparse            = {N_sparse}")
    print(f"  gate density (6/π²) = {DENSITY:.6f}")
    print()

    # ---------- stage 1 ----------
    print("--- stage 1  ·  32-bit decompile ---")
    all_words: List[int] = []
    all_names: List[str] = []
    all_consts: List[object] = []
    for name, src in sources:
        dec = decompile_32bit(src, name)
        all_words.extend(dec["words"])
        all_names.extend(dec["names"])
        all_consts.extend(dec["consts"])
        print(f"  {name:20s}  words={dec['n_words']:>4d}  "
              f"addr32=0x{dec['addr32']:08x}  "
              f"{dec['elapsed_ms']:.2f} ms")
    print(f"  total words         = {len(all_words)}")
    print()

    # ---------- stage 2 ----------
    print("--- stage 2  ·  32-bit Möbius memory compress ---")
    mem = MobiusMemory32.compress(all_words, K=K_mem)
    print(f"  n_raw words          = {mem.n_raw}")
    print(f"  n_kept slots         = {len(mem.slots)}")
    print(f"  density              = "
          f"{len(mem.slots) / mem.K:.4f}  "
          f"(target 6/π² = {DENSITY:.4f})")
    print(f"  supertrace S         = {mem.S:+.6f}")
    print(f"  entropy H            = {mem.H:.6f}")
    print(f"  mass m               = {mem.m:.6f}")
    print(f"  compressed addr32    = 0x{mem.addr32:08x}")
    print()

    # ---------- stage 3 ----------
    print("--- stage 3  ·  32-bit sparse weight lookup ---")
    sparse = SparseWeight32(K=K_mem)
    entries = sparse.lookup(mem, N=N_sparse)
    print(f"  entries kept         = {len(entries)}")
    n_sym   = sum(1 for e in entries if e.sym_gate)
    n_asym  = sum(1 for e in entries if e.asym_gate)
    n_cross = sum(1 for e in entries if e.cross_gate)
    print(f"  sym gate hits        = {n_sym}")
    print(f"  asym gate hits       = {n_asym}")
    print(f"  cross gate hits      = {n_cross}")
    print(f"  mean weight          = "
          f"{np.mean([e.weight for e in entries]) if entries else 0.0:.4f}")
    print()

    # ---------- stage 4 ----------
    print("--- stage 4  ·  64-bit unpack ---")
    unpacker = Unpack64(all_names, all_consts)
    unpacked = unpacker.walk(entries, mem)
    print(f"  words walked         = {unpacked.n_words}")
    print(f"  unpacked addr64      = 0x{unpacked.addr64:016x}")
    print(f"  elapsed              = {unpacked.elapsed_ms:.2f} ms")
    print()
    print("  first 12 unpacked lines:")
    for line in unpacked.lines[:12]:
        print(f"    {line}")
    if len(unpacked.lines) > 12:
        print(f"    ... ({len(unpacked.lines) - 12} more lines)")
    print()

    # ---------- stage 5  ·  chain of thought ----------
    print("--- stage 5  ·  elliptic Möbius SAT chain of thought ---")
    candidates = []
    for line in unpacked.lines[:16]:
        tokens = line.split()
        args = tokens[:min(len(tokens), 4)]
        candidates.append(Statement(text=line, args=args))

    buf = ReasoningBuffer(
        sat=EllipticMobiusSAT(quad_A=1, quad_B=1, quad_C=0,
                              i_exp=2.0, K_filter=K_FILTER),
        max_survivors=16,
    )
    r = buf.think(query, candidates)

    print(f"  candidates fed        = {len(candidates)}")
    print(f"  accepted              = {r['n_accepted']}")
    print(f"  rejected              = {r['n_rejected']}")
    print(f"  supertrace S          = {r['S']:+.4f}")
    print(f"  entropy   H           = {r['H']:.4f}")
    print(f"  mass      m           = {r['m']:.4f}")
    print()

    print("  per-candidate elliptic trace  (hidden):")
    print(f"  {'text':44s}  {'k':>2s}  "
          f"{'tr(a)':>8s}  {'q':>4s}  {'μ(q)':>4s}  "
          f"{'gate':>4s}  {'fml':>3s}  {'acc':>3s}")
    for rec in r["records"]:
        t = rec["statement"].text[:42] + \
            (".." if len(rec["statement"].text) > 42 else "")
        print(f"  {t:44s}  {rec['arity']:>2d}  "
              f"{rec['trace']:>+8.4f}  "
              f"{rec['q_addr']:>4d}  {rec['mu_q']:>+4d}  "
              f"{int(rec['gate']):>4d}  "
              f"{int(rec['formula']):>3d}  "
              f"{int(rec['accepted']):>3d}")
    print()

    # ---------- user-facing response ----------
    presenter = Presenter()
    response = presenter.present(buf.emit())
    print("--- user-facing response ---")
    print(f"  > {response}")
    print()
    print("--- what the user never sees ---")
    print("  · elliptic traces   tr(a) = x^(i−1) + y^(i−1)")
    print("  · quadratic address q(a) = a² + a mod K")
    print("  · Möbius gate       μ(q(a)) ≠ 0")
    print("  · supertrace        S = Σ (−1)^t |conv_t|")
    print("  · mass              m = |S| e^{−H}")
    print("  · arity k           1, 2, 3, …")
    print("  · accepted / rejected counts, cycle numbers")
    print()

    return dict(
        sources=sources, memory=mem, sparse_entries=entries,
        unpacked=unpacked, reasoning=r, response=response,
    )


# ============================================================
#  Demo
# ============================================================
def demo():
    sources = [
        ("add.py", '''
def add(a, b):
    return a + b
'''),
        ("greet.py", '''
def greet(name):
    return 'hello ' + name
'''),
        ("total.py", '''
def total(xs):
    s = 0
    for x in xs:
        s += x
    return s
'''),
        ("safe_div.py", '''
def safe_div(a, b):
    if b == 0:
        return 0
    return a / b
'''),
    ]

    query = Query(
        text="Reconstruct the dataset through the Möbius + sparse pipeline.",
        clauses=[
            [(0, +1)],      # slot 0 must be present
            [(1, +1)],      # slot 1 must be present
        ],
    )

    result = pipeline(sources, query, K_mem=K_MEM, N_sparse=N_SPARSE)

    mem = result["memory"]
    entries = result["sparse_entries"]
    r = result["reasoning"]
    print("--- summary ---")
    print(f"  32-bit words         : {mem.n_raw}")
    print(f"  μ-slot placements    : {len(mem.slots)}")
    print(f"  sparse entries       : {len(entries)}")
    print(f"  64-bit unpacked      : {result['unpacked'].n_words}")
    print(f"  chain of thought     : {len(r['records'])} candidates, "
          f"{r['n_accepted']} accepted")
    print(f"  final S / m          : {r['S']:+.4f} / {r['m']:.4f}")
    print()
    print("--- what the user sees ---")
    print(f"  {result['response']}")
    print()


if __name__ == "__main__":
    demo()