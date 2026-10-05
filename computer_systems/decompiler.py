#!/usr/bin/env python3
"""
decompiler_32to64.py
====================

Two-stage decompiler:

    Stage 1  ·  32-bit bounded pass             O(K)
        · parse AST
        · enforce caps (nodes, lines, call depth)
        · extract intent + observed signatures
        · build the 1D command tape

    Stage 2  ·  64-bit full decompile            O(K · L)
        · lift the tape to 64-bit state
        · walk the tape and emit pseudo-source
        · flag asymmetric / unintended behaviour

The 1D command tape is a linear list of 32-bit words.  Each word
packs (opcode, arg_a, arg_b, arg_c) — the same layout used by
the CPU-inference command path.  Walking the tape is O(K) in the
number of words (plus loop multiplicity).

Opcodes
-------
    0x01 LOAD_CONST   0x02 LOAD_NAME   0x03 STORE_NAME
    0x04 BINOP        0x05 UNARYOP     0x06 CALL
    0x07 ATTRIBUTE    0x08 SUBSCRIPT   0x09 COMPARE
    0x0A JUMP         0x0B JUMP_IF     0x0C RETURN
    0x0D IMPORT       0x0E MAKE_FUNC   0x0F MAKE_CLASS
    0x10 YIELD        0x11 LOOP_START  0x12 LOOP_END
    0xFF HALT

The same math (α = 1/(π − e), 6/π², μ-square-free gating) applies
at every layer that touches this tape.
"""

from __future__ import annotations

import ast
import hashlib
import math
import sys
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple


# ============================================================
#  Constants
# ============================================================
PI         = math.pi
E          = math.e
ALPHA      = 1.0 / (PI - E)             # ≈ 2.362
DENSITY    = 6.0 / (PI * PI)            # ≈ 0.6079

INT32_MAX  = 2 ** 31 - 1
UINT32_MAX = 2 ** 32 - 1
INT64_MAX  = 2 ** 63 - 1
UINT64_MAX = 2 ** 64 - 1


# ============================================================
#  32-bit caps
# ============================================================
@dataclass
class CompileCaps:
    MAX_FILES:           int = 512
    MAX_LINES_PER_FILE:  int = 50_000
    MAX_AST_NODES:       int = 200_000
    MAX_CALL_DEPTH:      int = 64
    MAX_ADDR:            int = UINT32_MAX
    MAX_LOOP_ITERATIONS: int = 1_000_000


class CapBreach(Exception):
    def __init__(self, cap, value, limit):
        super().__init__(f"{cap}: {value} > {limit}")
        self.cap, self.value, self.limit = cap, value, limit


# ============================================================
#  1D command tape  ·  32-bit packed words
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


def pack_word32(op: int, a: int = 0, b: int = 0, c: int = 0) -> int:
    return (((op & 0xFF) << 24) |
            ((a & 0xFF) << 16) |
            ((b & 0xFF) << 8)  |
            (c & 0xFF))


def unpack_word32(word: int) -> Tuple[int, int, int, int]:
    return ((word >> 24) & 0xFF,
            (word >> 16) & 0xFF,
            (word >> 8)  & 0xFF,
            word & 0xFF)


# ============================================================
#  Security patterns
# ============================================================
DANGEROUS_BUILTINS = {
    "eval", "exec", "compile", "__import__", "open",
    "input", "globals", "locals", "vars",
    "getattr", "setattr", "delattr",
    "breakpoint", "memoryview",
}

DANGEROUS_MODULES = {
    "subprocess", "os", "sys", "socket", "urllib",
    "requests", "http", "ftplib", "telnetlib",
    "pickle", "marshal", "shelve",
    "ctypes", "cffi", "mmap",
    "base64", "codecs", "binascii",
    "shutil", "tempfile", "pathlib",
    "multiprocessing", "threading", "asyncio",
}

DANGEROUS_ATTRS = {
    "system", "popen", "spawn", "execv", "execve", "fork",
    "__class__", "__bases__", "__subclasses__", "__globals__",
    "__code__", "__closure__", "__dict__", "__mro__",
    "read", "write", "connect", "bind", "listen", "accept",
    "loads", "load", "dumps", "dump",
}


# ============================================================
#  AST  →  1D command tape
# ============================================================
class TapeBuilder(ast.NodeVisitor):
    """
    Emits 32-bit words for every AST node.  The resulting tape is a
    flat, linear representation of the program's operations.

    Flags are accumulated here for dangerous builtins, attributes,
    modules and any cap that would be breached during the build.
    """
    def __init__(self, caps: CompileCaps):
        self.caps = caps
        self.words: List[int] = []
        self.names: List[str] = []
        self.consts: List[object] = []
        self._name_idx: Dict[str, int] = {}
        self._loop_counter = 0
        self._call_depth = 0
        self._node_count = 0
        self.flags: List[Dict] = []

    # ---- helpers ----
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

    def _emit(self, op: Op, a: int = 0, b: int = 0, c: int = 0) -> None:
        self.words.append(pack_word32(op, a, b, c))

    def _flag(self, kind: str, name: str, detail: str) -> None:
        self.flags.append(dict(kind=kind, name=name, detail=detail))

    # ---- cap enforcement ----
    def visit(self, node):
        self._node_count += 1
        if self._node_count > self.caps.MAX_AST_NODES:
            raise CapBreach("MAX_AST_NODES", self._node_count,
                            self.caps.MAX_AST_NODES)
        return super().visit(node)

    # ---- dispatch ----
    def visit_Module(self, node):
        for child in node.body:
            self.visit(child)
        self._emit(Op.HALT)

    def visit_FunctionDef(self, node):
        self._emit(Op.MAKE_FUNC, self._name(node.name))
        for child in node.body:
            self.visit(child)
        self._emit(Op.RETURN)

    def visit_AsyncFunctionDef(self, node):
        self._emit(Op.MAKE_FUNC, self._name(node.name), b=1)
        for child in node.body:
            self.visit(child)
        self._emit(Op.RETURN)

    def visit_ClassDef(self, node):
        self._emit(Op.MAKE_CLASS, self._name(node.name))
        for child in node.body:
            self.visit(child)

    def visit_Return(self, node):
        if node.value:
            self.visit(node.value)
        self._emit(Op.RETURN)

    def visit_Assign(self, node):
        self.visit(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._emit(Op.STORE_NAME, self._name(target.id))

    def visit_AugAssign(self, node):
        self.visit(node.value)
        self._emit(Op.BINOP, self._binop_kind(node.op))
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

    def visit_While(self, node):
        loop_id = self._loop_counter
        self._loop_counter += 1
        self._emit(Op.LOOP_START, loop_id)
        self.visit(node.test)
        self._emit(Op.JUMP_IF, 0)
        for child in node.body:
            self.visit(child)
        self._emit(Op.LOOP_END, loop_id)

    def visit_For(self, node):
        loop_id = self._loop_counter
        self._loop_counter += 1
        self._emit(Op.LOOP_START, loop_id)
        self.visit(node.iter)
        for child in node.body:
            self.visit(child)
        self._emit(Op.LOOP_END, loop_id)

    def visit_Call(self, node):
        self._call_depth += 1
        if self._call_depth > self.caps.MAX_CALL_DEPTH:
            raise CapBreach("MAX_CALL_DEPTH", self._call_depth,
                            self.caps.MAX_CALL_DEPTH)
        if isinstance(node.func, ast.Name):
            fname = node.func.id
            self._emit(Op.LOAD_NAME, self._name(fname))
            if fname in DANGEROUS_BUILTINS:
                self._flag("DANGEROUS_BUILTIN", fname,
                           f"called at line {node.lineno}")
        elif isinstance(node.func, ast.Attribute):
            self.visit(node.func.value)
            attr = node.func.attr
            self._emit(Op.ATTRIBUTE, self._name(attr))
            if attr in DANGEROUS_ATTRS:
                self._flag("DANGEROUS_ATTR", attr,
                           f"called at line {node.lineno}")
        else:
            self.visit(node.func)
        for arg in node.args:
            self.visit(arg)
        self._emit(Op.CALL, len(node.args) & 0xFF,
                   len(node.keywords) & 0xFF)
        self._call_depth -= 1

    def visit_Attribute(self, node):
        self.visit(node.value)
        attr = node.attr
        self._emit(Op.ATTRIBUTE, self._name(attr))
        if attr in DANGEROUS_ATTRS:
            self._flag("DANGEROUS_ATTR", attr,
                       f"accessed at line {node.lineno}")

    def visit_Subscript(self, node):
        self.visit(node.value)
        self.visit(node.slice)
        self._emit(Op.SUBSCRIPT)

    def visit_BinOp(self, node):
        self.visit(node.left)
        self.visit(node.right)
        self._emit(Op.BINOP, self._binop_kind(node.op))

    def visit_UnaryOp(self, node):
        self.visit(node.operand)
        self._emit(Op.UNARYOP, self._unop_kind(node.op))

    def visit_Compare(self, node):
        self.visit(node.left)
        for op, comp in zip(node.ops, node.comparators):
            self.visit(comp)
            self._emit(Op.COMPARE, self._cmp_kind(op))

    def visit_Import(self, node):
        for alias in node.names:
            top = alias.name.split(".")[0]
            self._emit(Op.IMPORT, self._name(top))
            if top in DANGEROUS_MODULES:
                self._flag("DANGEROUS_MODULE", top,
                           f"imported at line {node.lineno}")

    def visit_ImportFrom(self, node):
        top = (node.module or "").split(".")[0]
        self._emit(Op.IMPORT, self._name(top))
        if top in DANGEROUS_MODULES:
            self._flag("DANGEROUS_MODULE", top,
                       f"imported at line {node.lineno}")

    def visit_Yield(self, node):
        if node.value:
            self.visit(node.value)
        self._emit(Op.YIELD)

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
        for k in node.keys:
            self.visit(k)
        for v in node.values:
            self.visit(v)

    def visit_ListComp(self, node):
        for g in node.generators:
            self.visit(g.iter)
        self.visit(node.elt)

    def visit_Lambda(self, node):
        for arg in node.args.args:
            self._emit(Op.STORE_NAME, self._name(arg.arg))
        self.visit(node.body)

    # ---- operator kind maps ----
    @staticmethod
    def _binop_kind(op) -> int:
        return {
            ast.Add:      0x01, ast.Sub: 0x02, ast.Mult: 0x03,
            ast.Div:      0x04, ast.Mod: 0x05, ast.Pow:  0x06,
            ast.FloorDiv: 0x07, ast.BitAnd: 0x08, ast.BitOr: 0x09,
            ast.BitXor:   0x0A, ast.LShift: 0x0B, ast.RShift: 0x0C,
            ast.MatMult:  0x0D,
        }.get(type(op), 0)

    @staticmethod
    def _unop_kind(op) -> int:
        return {ast.UAdd: 0x01, ast.USub: 0x02,
                ast.Not: 0x03, ast.Invert: 0x04}.get(type(op), 0)

    @staticmethod
    def _cmp_kind(op) -> int:
        return {ast.Eq: 0x01, ast.NotEq: 0x02,
                ast.Lt: 0x03, ast.LtE: 0x04,
                ast.Gt: 0x05, ast.GtE: 0x06,
                ast.Is: 0x07, ast.IsNot: 0x08,
                ast.In: 0x09, ast.NotIn: 0x0A}.get(type(op), 0)


# ============================================================
#  Tape walker  ·  64-bit full decompile
# ============================================================
@dataclass
class DecompileResult:
    code: str
    n_words: int
    n_flags: int
    flags: List[Dict]
    words: List[int]
    names: List[str]
    consts: List[object]
    addr_32: int
    addr_64: int
    elapsed_ms: float = 0.0
    cap_breach: Optional[str] = None


class TapeWalker:
    """
    Walks the 32-bit tape at 64-bit state width and emits
    decompiled pseudo-source.

    Every word executed adds a line to the output.  Loops are
    executed up to MAX_LOOP_ITERATIONS before a LOOP_BOUND flag
    fires.  Anything that looks unusual — unknown opcode, exceeding
    a cap, hitting a dangerous pattern twice — is flagged.
    """
    def __init__(self, tape: List[int], names: List[str],
                 consts: List[object],
                 caps: Optional[CompileCaps] = None):
        self.tape = tape
        self.names = names
        self.consts = consts
        self.caps = caps or CompileCaps()
        self.pc = 0
        self.stack: List[str] = []
        self.lines: List[str] = []
        self.loop_counters: Dict[int, int] = {}
        self.flags: List[Dict] = []
        self.total_steps = 0

    def _name(self, i: int) -> str:
        return self.names[i] if i < len(self.names) else f"<n{i}>"

    def _const(self, i: int) -> str:
        if i < len(self.consts):
            return repr(self.consts[i])
        return "<c>"

    def _pop(self, n: int = 1) -> List[str]:
        out = []
        for _ in range(n):
            out.append(self.stack.pop() if self.stack else "?")
        out.reverse()
        return out

    def _flag(self, kind: str, detail: str) -> None:
        self.flags.append(dict(kind=kind, detail=detail, pc=self.pc))

    def walk(self) -> Tuple[str, int]:
        max_steps = (self.caps.MAX_LOOP_ITERATIONS
                     + len(self.tape) * 2)
        while self.pc < len(self.tape) and self.total_steps < max_steps:
            word = self.tape[self.pc]
            op, a, b, c = unpack_word32(word)
            self.pc += 1
            self.total_steps += 1
            self._exec(op, a, b, c)
        if self.total_steps >= max_steps:
            self._flag("MAX_LOOP_ITERATIONS",
                       f"hit at pc={self.pc}")
        return "\n".join(self.lines), len(self.lines)

    def _exec(self, op: int, a: int, b: int, c: int) -> None:
        if op == Op.LOAD_CONST:
            self.stack.append(self._const(a))
        elif op == Op.LOAD_NAME:
            self.stack.append(self._name(a))
        elif op == Op.STORE_NAME:
            (val,) = self._pop(1)
            self.lines.append(f"{self._name(a)} = {val}")
        elif op == Op.BINOP:
            (l, r) = self._pop(2)
            self.stack.append(f"({l} {self._op_str(a)} {r})")
        elif op == Op.UNARYOP:
            (v,) = self._pop(1)
            self.stack.append(f"({self._unop_str(a)} {v})")
        elif op == Op.ATTRIBUTE:
            (v,) = self._pop(1)
            self.stack.append(f"{v}.{self._name(a)}")
        elif op == Op.SUBSCRIPT:
            (v, k) = self._pop(2)
            self.stack.append(f"{v}[{k}]")
        elif op == Op.COMPARE:
            (l, r) = self._pop(2)
            self.stack.append(f"({l} {self._cmp_str(a)} {r})")
        elif op == Op.CALL:
            args = self._pop(b)
            callee = self.stack.pop() if self.stack else "?"
            self.lines.append(f"{callee}({', '.join(args)})")
        elif op == Op.RETURN:
            if self.stack:
                (v,) = self._pop(1)
                self.lines.append(f"return {v}")
            else:
                self.lines.append("return")
        elif op == Op.JUMP_IF:
            (test,) = self._pop(1)
            self.lines.append(f"if {test}:")
        elif op == Op.JUMP:
            self.lines.append("else:")
        elif op == Op.LOOP_START:
            self.loop_counters[a] = self.loop_counters.get(a, 0) + 1
            self.lines.append(f"# loop_{a}_begin  "
                              f"(iteration {self.loop_counters[a]})")
            if self.loop_counters[a] > self.caps.MAX_LOOP_ITERATIONS:
                self._flag("LOOP_BOUND",
                           f"loop_{a} exceeded cap")
        elif op == Op.LOOP_END:
            self.lines.append(f"# loop_{a}_end")
        elif op == Op.IMPORT:
            name = self._name(a)
            self.lines.append(f"import {name}")
        elif op == Op.MAKE_FUNC:
            name = self._name(a)
            kind = "async def" if b else "def"
            self.lines.append(f"{kind} {name}(...):")
        elif op == Op.MAKE_CLASS:
            name = self._name(a)
            self.lines.append(f"class {name}:")
        elif op == Op.YIELD:
            if self.stack:
                (v,) = self._pop(1)
                self.lines.append(f"yield {v}")
            else:
                self.lines.append("yield")
        elif op == Op.HALT:
            self.pc = len(self.tape)
        else:
            self._flag("UNKNOWN_OPCODE",
                       f"op={op:#x} pc={self.pc - 1}")

    @staticmethod
    def _op_str(k: int) -> str:
        return {0x01: "+", 0x02: "-", 0x03: "*", 0x04: "/",
                0x05: "%", 0x06: "**", 0x07: "//",
                0x08: "&", 0x09: "|", 0x0A: "^",
                0x0B: "<<", 0x0C: ">>", 0x0D: "@"}.get(k, "?")

    @staticmethod
    def _unop_str(k: int) -> str:
        return {0x01: "+", 0x02: "-", 0x03: "not",
                0x04: "~"}.get(k, "?")

    @staticmethod
    def _cmp_str(k: int) -> str:
        return {0x01: "==", 0x02: "!=", 0x03: "<", 0x04: "<=",
                0x05: ">", 0x06: ">=", 0x07: "is", 0x08: "is not",
                0x09: "in", 0x0A: "not in"}.get(k, "?")


# ============================================================
#  Intent / Observed extraction
# ============================================================
@dataclass
class Intent:
    name: str
    kind: str
    args: List[str] = field(default_factory=list)
    doc: str = ""
    lineno: int = 0


@dataclass
class Observed:
    name: str
    calls: List[str] = field(default_factory=list)
    attrs: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    dangerous_builtins: List[str] = field(default_factory=list)
    dangerous_attrs: List[str] = field(default_factory=list)
    dangerous_modules: List[str] = field(default_factory=list)


def extract_intents(tree: ast.Module) -> List[Intent]:
    out: List[Intent] = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(Intent(
                name=n.name, kind="function",
                args=[a.arg for a in n.args.args],
                doc=ast.get_docstring(n) or "",
                lineno=n.lineno))
        elif isinstance(n, ast.ClassDef):
            out.append(Intent(name=n.name, kind="class",
                              doc=ast.get_docstring(n) or "",
                              lineno=n.lineno))
    return out


def extract_observed(tree: ast.Module) -> Dict[str, Observed]:
    by_func: Dict[str, Observed] = {}
    for n in ast.walk(tree):
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        obs = Observed(name=n.name)
        for sub in ast.walk(n):
            if isinstance(sub, ast.Call):
                if isinstance(sub.func, ast.Name):
                    obs.calls.append(sub.func.id)
                    if sub.func.id in DANGEROUS_BUILTINS:
                        obs.dangerous_builtins.append(sub.func.id)
                elif isinstance(sub.func, ast.Attribute):
                    obs.attrs.append(sub.func.attr)
                    if sub.func.attr in DANGEROUS_ATTRS:
                        obs.dangerous_attrs.append(sub.func.attr)
            if isinstance(sub, ast.Import):
                for al in sub.names:
                    top = al.name.split(".")[0]
                    obs.imports.append(top)
                    if top in DANGEROUS_MODULES:
                        obs.dangerous_modules.append(top)
            if isinstance(sub, ast.ImportFrom):
                top = (sub.module or "").split(".")[0]
                obs.imports.append(top)
                if top in DANGEROUS_MODULES:
                    obs.dangerous_modules.append(top)
        by_func[n.name] = obs
    return by_func


def asymmetry_score(intent: Intent, obs: Observed) -> Dict:
    expected = set(intent.args) | {"self", "cls", intent.name}
    extra_calls = [c for c in obs.calls if c not in expected
                   and c not in DANGEROUS_BUILTINS]
    severity = (3 * len(obs.dangerous_builtins)
                + 3 * len(obs.dangerous_attrs)
                + 2 * len(obs.dangerous_modules))
    structural = (len(extra_calls) /
                  max(len(obs.calls) + len(intent.args), 1))
    total = structural + 0.25 * severity
    return dict(
        structural=round(structural, 4),
        severity=severity,
        total=round(total, 4),
        extra_calls=sorted(set(extra_calls)),
        dangerous_builtins=sorted(set(obs.dangerous_builtins)),
        dangerous_attrs=sorted(set(obs.dangerous_attrs)),
        dangerous_modules=sorted(set(obs.dangerous_modules)),
    )


# ============================================================
#  The two-stage decompiler
# ============================================================
class Decompiler32to64:
    """
    Stage 1  32-bit bounded pass     → build the tape, enforce caps
    Stage 2  64-bit full decompile   → walk the tape, emit code,
                                        flag asymmetric behaviour
    """
    def __init__(self, caps: Optional[CompileCaps] = None,
                 tau_asym: float = 0.35):
        self.caps = caps or CompileCaps()
        self.tau_asym = tau_asym

    def _stage1_32bit(self, source: str) -> Dict:
        lines = source.splitlines()
        if len(lines) > self.caps.MAX_LINES_PER_FILE:
            raise CapBreach("MAX_LINES_PER_FILE", len(lines),
                            self.caps.MAX_LINES_PER_FILE)
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return dict(ok=False, error=f"SyntaxError: {e}")

        node_count = sum(1 for _ in ast.walk(tree))
        if node_count > self.caps.MAX_AST_NODES:
            raise CapBreach("MAX_AST_NODES", node_count,
                            self.caps.MAX_AST_NODES)

        h32 = hashlib.sha256(source.encode()).digest()[:4]
        addr32 = int.from_bytes(h32, "big") & UINT32_MAX

        builder = TapeBuilder(self.caps)
        builder.visit(tree)

        return dict(
            ok=True, tree=tree, node_count=node_count,
            n_lines=len(lines), addr32=addr32,
            words=builder.words, names=builder.names,
            consts=builder.consts, flags=builder.flags,
            intents=extract_intents(tree),
            observed=extract_observed(tree),
        )

    def _stage2_64bit(self, stage1: Dict) -> Dict:
        tape   = stage1["words"]
        names  = stage1["names"]
        consts = stage1["consts"]

        h64 = hashlib.sha256(repr(tape).encode()).digest()[:8]
        addr64 = int.from_bytes(h64, "big") & UINT64_MAX

        walker = TapeWalker(tape, names, consts, self.caps)
        code, n_lines = walker.walk()

        verdicts = []
        for intent in stage1["intents"]:
            obs = stage1["observed"].get(intent.name)
            if obs is None:
                continue
            asym = asymmetry_score(intent, obs)
            verdict = ("SYMMETRIC"
                       if asym["total"] <= self.tau_asym
                       and not asym["dangerous_builtins"]
                       and not asym["dangerous_attrs"]
                       and not asym["dangerous_modules"]
                       else "ASYMMETRIC")
            verdicts.append(dict(
                name=intent.name, kind=intent.kind,
                verdict=verdict, asymmetry=asym))

        return dict(
            code=code, n_lines=n_lines, addr64=addr64,
            flags=walker.flags, verdicts=verdicts)

    # ---------- public entry ----------
    def decompile(self, source: str,
                  name: str = "<source>") -> DecompileResult:
        t0 = time.perf_counter()

        try:
            s1 = self._stage1_32bit(source)
        except CapBreach as b:
            return DecompileResult(
                code="", n_words=0, n_flags=1, flags=[],
                words=[], names=[], consts=[],
                addr_32=0, addr_64=0,
                elapsed_ms=(time.perf_counter() - t0) * 1e3,
                cap_breach=f"{b.cap}: {b.value} > {b.limit}",
            )

        if not s1.get("ok"):
            return DecompileResult(
                code="", n_words=0, n_flags=1, flags=[],
                words=[], names=[], consts=[],
                addr_32=0, addr_64=0,
                elapsed_ms=(time.perf_counter() - t0) * 1e3,
                cap_breach=s1.get("error", "stage 1 failed"),
            )

        s2 = self._stage2_64bit(s1)

        all_flags = list(s1["flags"]) + list(s2["flags"])
        for v in s2["verdicts"]:
            if v["verdict"] == "ASYMMETRIC":
                all_flags.append(dict(
                    kind="ASYMMETRIC",
                    name=v["name"],
                    detail=f"total={v['asymmetry']['total']:.4f}"))

        return DecompileResult(
            code=s2["code"],
            n_words=len(s1["words"]),
            n_flags=len(all_flags),
            flags=all_flags,
            words=s1["words"],
            names=s1["names"],
            consts=s1["consts"],
            addr_32=s1["addr32"],
            addr_64=s2["addr64"],
            elapsed_ms=(time.perf_counter() - t0) * 1e3,
        )


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("Decompiler  ·  32-bit bounded pass → 64-bit full decompile")
    print("=" * 78)
    print(f"  host                : "
          f"{'64-bit' if sys.maxsize > 2**32 else '32-bit'}")
    print(f"  MAX_AST_NODES       : {CompileCaps.MAX_AST_NODES:,}")
    print(f"  MAX_CALL_DEPTH      : {CompileCaps.MAX_CALL_DEPTH}")
    print(f"  MAX_LINES_PER_FILE  : {CompileCaps.MAX_LINES_PER_FILE:,}")
    print(f"  MAX_LOOP_ITERATIONS : {CompileCaps.MAX_LOOP_ITERATIONS:,}")
    print()

    decompiler = Decompiler32to64()

    samples = [
        ("clean.py", '''
def add(a, b):
    """Return a + b."""
    return a + b

class Calculator:
    """Simple calculator."""
    def __init__(self):
        self.value = 0

    def add(self, x):
        """Add x to value."""
        self.value += x
        return self.value

c = Calculator()
c.add(5)
print(c.value)
'''),

        ("asymmetric.py", '''
def add(a, b):
    """Return a + b."""
    import subprocess
    subprocess.system("curl http://example.com")
    return eval("a + b")
'''),

        ("sneaky.py", '''
def harmless(x):
    """Return x."""
    return x.__class__.__bases__[0].__subclasses__()
'''),

        ("looping.py", '''
def compute(n):
    """Sum from 1 to n."""
    total = 0
    i = 1
    while i <= n:
        total += i
        i += 1
    return total
'''),

        ("deep.py",
         "def f():\n    return "
         + "g(" * 100 + "1" + ")" * 100 + "\n"),

        ("huge.py",
         "\n".join(f"x{i} = {i}" for i in range(200_001))),
    ]

    for name, src in samples:
        print("=" * 78)
        print(f"decompile  ·  {name}")
        print("=" * 78)

        res = decompiler.decompile(src, name)

        print(f"  ok             : {res.cap_breach is None}")
        if res.cap_breach:
            print(f"  cap breach     : {res.cap_breach}")
            print()
            continue

        print(f"  32-bit addr    : 0x{res.addr_32:08x}")
        print(f"  64-bit addr    : 0x{res.addr_64:016x}")
        print(f"  tape words     : {res.n_words}")
        print(f"  flags          : {res.n_flags}")
        print(f"  elapsed        : {res.elapsed_ms:.2f} ms")
        print()

        if res.flags:
            print("  --- flags ---")
            for f in res.flags[:10]:
                kind = f.get("kind", "?")
                nm   = f.get("name", "")
                det  = f.get("detail", "")
                print(f"    [{kind:>20s}]  {nm:>20s}  {det}")
            if len(res.flags) > 10:
                print(f"    ... and {len(res.flags) - 10} more")
            print()

        if res.code:
            lines = res.code.splitlines()
            print("  --- decompiled pseudo-source ---")
            for line in lines[:20]:
                print(f"    {line}")
            if len(lines) > 20:
                print(f"    ... ({len(lines) - 20} more lines)")
        print()


if __name__ == "__main__":
    demo()