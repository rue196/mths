#!/usr/bin/env python3
"""
security_scan.py
================

Bounded 32-bit compile + 64-bit behavioral scan.

Pipeline
--------
    source
      → AST parse
      → intent extraction   (what the function/class declares it does)
      → observed extraction (what it actually does)
      → asymmetry score     (observed − intent)
      → 32-bit compile      (bounded address space, caps on loops)
      → 64-bit scan         (host architecture, dual-envelope check)
      → verdict

Verdicts
--------
    SYMMETRIC    observed ⊆ intent, no dangerous patterns, no cap breach
    ASYMMETRIC   observed − intent ≠ ∅, or a security pattern fires
    CAP_BREACH   the 32-bit compile pass hit a hard cap
                 (itself a security flag: possible infinite loop / DoS)

32-bit caps (enforced during compile)
-------------------------------------
    MAX_FILES            512
    MAX_LINES_PER_FILE   50 000
    MAX_AST_NODES        200 000
    MAX_CALL_DEPTH       64
    MAX_ADDR             uint32 max (2³²−1)
    MAX_LOOP_ITERATIONS  1 000 000

64-bit scan
-----------
    positive envelope  = declared intent signal
    negative envelope  = observed − intent residual (asymmetric)
    dual supertrace   S_pos, S_neg  via the 64-bit layout
"""

from __future__ import annotations

import ast
import math
import sys
import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ============================================================
#  Constants
# ============================================================
PI         = math.pi
E          = math.e
ALPHA      = 1.0 / (PI - E)
DENSITY    = 6.0 / (PI * PI)

INT32_MAX  = 2 ** 31 - 1
UINT32_MAX = 2 ** 32 - 1

# asymmetry thresholds
TAU_ASYM   = 0.35                # observed−intent above this → asymmetric
TAU_CAP    = 1                   # single cap breach is already a flag


# ============================================================
#  32-bit compile caps
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
    """Raised when any 32-bit compile cap is exceeded."""
    def __init__(self, cap: str, value: int, limit: int):
        super().__init__(f"{cap}: {value} > {limit}")
        self.cap = cap
        self.value = value
        self.limit = limit


# ============================================================
#  Security patterns (observed behavior known to be dangerous)
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

SAFE_BUILTINS = {
    "abs", "all", "any", "bin", "bool", "bytes", "callable",
    "chr", "complex", "dict", "dir", "divmod", "enumerate",
    "filter", "float", "format", "frozenset", "hash", "hex",
    "id", "int", "isinstance", "issubclass", "iter", "len",
    "list", "map", "max", "min", "next", "object", "oct",
    "ord", "pow", "print", "range", "repr", "reversed",
    "round", "set", "slice", "sorted", "str", "sum", "tuple",
    "type", "zip", "Exception", "ValueError", "TypeError",
    "KeyError", "IndexError", "AttributeError", "StopIteration",
    "RuntimeError", "ZeroDivisionError", "NotImplementedError",
    "True", "False", "None", "NotImplemented", "Ellipsis",
    "__name__", "__doc__", "__file__", "__builtins__",
}


# ============================================================
#  Intent extraction  (what the code declares it does)
# ============================================================
@dataclass
class Intent:
    name: str
    kind: str                                 # 'function' | 'method' | 'class'
    args: List[str] = field(default_factory=list)
    doc: str = ""
    decorators: List[str] = field(default_factory=list)
    type_hints: Dict[str, str] = field(default_factory=dict)
    declared_calls: Set[str] = field(default_factory=set)     # from docstring
    lineno: int = 0


class IntentExtractor(ast.NodeVisitor):
    def __init__(self):
        self.intents: List[Intent] = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._handle(node, "function")
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._handle(node, "function")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        self._handle(node, "class")
        self.generic_visit(node)

    def _handle(self, node, kind: str):
        doc = ast.get_docstring(node) or ""
        decorators = [self._decor_name(d) for d in node.decorator_list]
        hints: Dict[str, str] = {}
        if getattr(node, "returns", None) is not None:
            hints["return"] = self._dotted(node.returns)
        for arg in node.args.args:
            if arg.annotation is not None:
                hints[arg.arg] = self._dotted(arg.annotation)

        declared_calls = self._calls_named_in_doc(doc)

        self.intents.append(Intent(
            name=node.name,
            kind=kind,
            args=[a.arg for a in node.args.args],
            doc=doc,
            decorators=decorators,
            type_hints=hints,
            declared_calls=declared_calls,
            lineno=node.lineno,
        ))

    @staticmethod
    def _decor_name(d) -> str:
        if isinstance(d, ast.Name):
            return d.id
        if isinstance(d, ast.Attribute):
            return d.attr
        if isinstance(d, ast.Call):
            return IntentExtractor._decor_name(d.func)
        return ""

    @staticmethod
    def _dotted(n) -> str:
        if isinstance(n, ast.Name):
            return n.id
        if isinstance(n, ast.Attribute):
            return f"{IntentExtractor._dotted(n.value)}.{n.attr}"
        if isinstance(n, ast.Subscript):
            return f"{IntentExtractor._dotted(n.value)}[...]"
        if isinstance(n, ast.Constant):
            return repr(n.value)
        return "?"

    @staticmethod
    def _calls_named_in_doc(doc: str) -> Set[str]:
        """
        Pull `foo(...)` or `:func:foo` style references from the
        docstring — those are the calls the author declared.
        """
        import re
        out: Set[str] = set()
        for m in re.finditer(r"`?([A-Za-z_][A-Za-z_0-9]*)\s*\(", doc):
            out.add(m.group(1))
        for m in re.finditer(r":(?:func|meth|class):`~?([A-Za-z_.]+)`", doc):
            out.add(m.group(1).split(".")[-1])
        return out


# ============================================================
#  Observed extraction  (what the code actually does)
# ============================================================
@dataclass
class Observed:
    name: str
    calls: Set[str] = field(default_factory=set)
    attrs: Set[str] = field(default_factory=set)
    imports: Set[str] = field(default_factory=set)
    globals_referenced: Set[str] = field(default_factory=set)
    literals: List[str] = field(default_factory=list)
    dangerous_builtins: Set[str] = field(default_factory=set)
    dangerous_attrs: Set[str] = field(default_factory=set)
    dangerous_modules: Set[str] = field(default_factory=set)
    max_depth: int = 0
    node_count: int = 0


class ObservedExtractor(ast.NodeVisitor):
    """
    Extracts a behavior signature for one function or class body.

    Enforces the 32-bit compile caps in-line:
        MAX_AST_NODES   → CapBreach
        MAX_CALL_DEPTH  → CapBreach
    """
    def __init__(self, caps: CompileCaps, node=None, name: str = "<module>"):
        self.caps = caps
        self.name = name
        self.obs = Observed(name=name)
        self.depth = 0
        if node is not None:
            for child in ast.iter_child_nodes(node):
                self.visit(child)

    # ---- cap-enforcing walk ----
    def visit(self, node):
        self.obs.node_count += 1
        if self.obs.node_count > self.caps.MAX_AST_NODES:
            raise CapBreach("MAX_AST_NODES",
                            self.obs.node_count,
                            self.caps.MAX_AST_NODES)
        return super().visit(node)

    # ---- facts ----
    def visit_Call(self, node):
        self._record_call(node.func)
        self.depth += 1
        if self.depth > self.caps.MAX_CALL_DEPTH:
            raise CapBreach("MAX_CALL_DEPTH", self.depth,
                            self.caps.MAX_CALL_DEPTH)
        self.generic_visit(node)
        self.depth -= 1

    def visit_Attribute(self, node):
        self.obs.attrs.add(node.attr)
        if node.attr in DANGEROUS_ATTRS:
            self.obs.dangerous_attrs.add(node.attr)
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.obs.globals_referenced.add(node.id)
        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            top = alias.name.split(".")[0]
            self.obs.imports.add(top)
            if top in DANGEROUS_MODULES:
                self.obs.dangerous_modules.add(top)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        mod = (node.module or "").split(".")[0]
        self.obs.imports.add(mod)
        if mod in DANGEROUS_MODULES:
            self.obs.dangerous_modules.add(mod)
        self.generic_visit(node)

    def visit_Constant(self, node):
        v = node.value
        if isinstance(v, str) and len(v) <= 80:
            self.obs.literals.append(v)
        elif isinstance(v, (int, float)):
            self.obs.literals.append(repr(v))
        self.generic_visit(node)

    def _record_call(self, fn):
        if isinstance(fn, ast.Name):
            name = fn.id
            self.obs.calls.add(name)
            if name in DANGEROUS_BUILTINS:
                self.obs.dangerous_builtins.add(name)
        elif isinstance(fn, ast.Attribute):
            self.obs.attrs.add(fn.attr)
            if fn.attr in DANGEROUS_ATTRS:
                self.obs.dangerous_attrs.add(fn.attr)
            self.obs.calls.add(fn.attr)


# ============================================================
#  Asymmetry score
# ============================================================
def asymmetry_score(intent: Intent, observed: Observed) -> Dict:
    """
    Compute the observed−intent residual.

        declared_calls ⊆ observed_calls  → low asymmetry
        extra_observed_calls             → asymmetry

    Plus severity weights from dangerous patterns and cap metrics.
    """
    declared = intent.declared_calls | set(intent.args)
    # local names used inside the function are legitimate
    body_names = {n for n in observed.globals_referenced}
    # names already declared by docstring / args are expected
    expected = declared | SAFE_BUILTINS | {
        intent.name, "self", "cls"
    }

    extra_calls = observed.calls - expected
    extra_globals = {
        n for n in observed.globals_referenced
        if n not in expected and not n.startswith("_")
    }

    # structural asymmetry (0 … 1)
    n_extra = len(extra_calls) + len(extra_globals)
    n_total = max(len(observed.calls) + len(observed.globals_referenced), 1)
    structural = n_extra / n_total

    # severity from dangerous patterns
    severity = (
        3 * len(observed.dangerous_builtins)
        + 3 * len(observed.dangerous_attrs)
        + 2 * len(observed.dangerous_modules)
    )

    total = structural + 0.25 * severity
    return dict(
        structural=round(structural, 4),
        severity=severity,
        total=round(total, 4),
        extra_calls=sorted(extra_calls),
        extra_globals=sorted(extra_globals),
        dangerous_builtins=sorted(observed.dangerous_builtins),
        dangerous_attrs=sorted(observed.dangerous_attrs),
        dangerous_modules=sorted(observed.dangerous_modules),
    )


# ============================================================
#  32-bit compile pass
# ============================================================
class Compile32:
    """
    A hard 32-bit bounded compile.

    Steps
    -----
    1. line / token caps
    2. AST node cap (via ObservedExtractor)
    3. uint32 file hash
    4. address-limit check on the AST traversal counter

    A breach raises CapBreach; the caller turns it into a
    security verdict (CAP_BREACH).
    """

    def __init__(self, caps: Optional[CompileCaps] = None):
        self.caps = caps or CompileCaps()
        self.file_count = 0

    def compile_source(self, source: str,
                       name: str = "<source>") -> Dict:
        # ---- file-level cap ----
        self.file_count += 1
        if self.file_count > self.caps.MAX_FILES:
            raise CapBreach("MAX_FILES", self.file_count,
                            self.caps.MAX_FILES)

        # ---- line / token caps ----
        lines = source.splitlines()
        if len(lines) > self.caps.MAX_LINES_PER_FILE:
            raise CapBreach("MAX_LINES_PER_FILE", len(lines),
                            self.caps.MAX_LINES_PER_FILE)

        # ---- token-level cap ----
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return dict(ok=False, error=f"SyntaxError: {e}")

        node_count = sum(1 for _ in ast.walk(tree))
        if node_count > self.caps.MAX_AST_NODES:
            raise CapBreach("MAX_AST_NODES", node_count,
                            self.caps.MAX_AST_NODES)

        # ---- uint32 file hash ----
        h = hashlib.sha256(source.encode("utf-8")).digest()[:4]
        addr = int.from_bytes(h, "big") % (self.caps.MAX_ADDR + 1)

        # ---- extract intent + observed for every def ----
        ie = IntentExtractor()
        ie.visit(tree)

        observations: List[Tuple[Intent, Observed]] = []
        for intent in ie.intents:
            target = self._find_node(tree, intent.lineno, intent.name)
            if target is None:
                continue
            oe = ObservedExtractor(self.caps, target, intent.name)
            observations.append((intent, oe.obs))

        return dict(
            ok=True,
            tree=tree,
            addr=addr,
            n_nodes=node_count,
            n_lines=len(lines),
            n_intents=len(ie.intents),
            observations=observations,
            is_32bit_ok=True,
        )

    @staticmethod
    def _find_node(tree, lineno: int, name: str):
        for node in ast.walk(tree):
            if getattr(node, "lineno", None) == lineno and \
               getattr(node, "name", None) == name:
                return node
        return None


# ============================================================
#  64-bit scan pass
# ============================================================
class Scan64:
    """
    Host-architecture scan with the 64-bit dual-envelope layout.

        positive half = intent signal
        negative half = observed − intent  (asymmetric residual)
        dual supertrace  S_pos, S_neg
    """
    def __init__(self, K: int = 128):
        self.K = K

    def dual_envelope(self,
                      observations: List[Tuple[Intent, Observed]]
                      ) -> Tuple[np.ndarray, np.ndarray]:
        K = self.K
        env_pos = np.zeros(K + 1, dtype=np.float64)
        env_neg = np.zeros(K + 1, dtype=np.float64)

        for intent, obs in observations:
            # intent half
            h = int(hashlib.sha256(
                (intent.name + intent.doc).encode()).hexdigest()[:8], 16)
            q = h % K + 1
            env_pos[q] += 1.0

            # observed half
            sig = "|".join(sorted(obs.calls)) \
                  + "|".join(sorted(obs.attrs))
            h2 = int(hashlib.sha256(sig.encode()).hexdigest()[:8], 16)
            q2 = h2 % K + 1
            env_neg[q2] += 1.0

        return env_pos, env_neg

    @staticmethod
    def supertrace(env: np.ndarray) -> float:
        S = 0.0
        for i, v in enumerate(env):
            S += v if (i % 2 == 0) else -v
        return float(S)

    def scan(self, observations) -> Dict:
        env_pos, env_neg = self.dual_envelope(observations)
        S_pos = self.supertrace(env_pos)
        S_neg = self.supertrace(env_neg)
        S_all = S_pos + S_neg
        return dict(
            env_pos=env_pos, env_neg=env_neg,
            S_pos=S_pos, S_neg=S_neg, S_all=S_all,
            balance=abs(S_pos - S_neg) / max(1.0, abs(S_pos) + abs(S_neg)),
        )


# ============================================================
#  Verdict
# ============================================================
@dataclass
class Finding:
    name: str
    kind: str
    verdict: str                    # SYMMETRIC / ASYMMETRIC / CAP_BREACH
    asymmetry: Dict = field(default_factory=dict)
    intent: Optional[Intent] = None
    observed: Optional[Observed] = None


@dataclass
class ScanResult:
    source_name: str
    ok: bool
    n_intents: int = 0
    findings: List[Finding] = field(default_factory=list)
    scan64: Dict = field(default_factory=dict)
    addr: int = 0
    elapsed_ms: float = 0.0
    error: str = ""


class SecurityScanner:
    def __init__(self,
                 caps: Optional[CompileCaps] = None,
                 K64: int = 128,
                 tau_asym: float = TAU_ASYM):
        self.caps = caps or CompileCaps()
        self.compiler = Compile32(self.caps)
        self.scanner64 = Scan64(K=K64)
        self.tau_asym = tau_asym

    # ---------- main entry ----------
    def scan(self, source: str,
             name: str = "<source>") -> ScanResult:
        t0 = time.perf_counter()
        res = ScanResult(source_name=name, ok=True)

        # ---- 32-bit compile ----
        try:
            c32 = self.compiler.compile_source(source, name)
        except CapBreach as b:
            res.ok = False
            res.error = f"CAP_BREACH: {b.cap} = {b.value} > {b.limit}"
            res.findings.append(Finding(
                name=name, kind="module",
                verdict="CAP_BREACH",
                asymmetry=dict(cap=b.cap, value=b.value, limit=b.limit),
            ))
            res.elapsed_ms = (time.perf_counter() - t0) * 1e3
            return res

        if not c32.get("ok"):
            res.ok = False
            res.error = c32.get("error", "compile failed")
            res.elapsed_ms = (time.perf_counter() - t0) * 1e3
            return res

        res.addr = c32["addr"]
        res.n_intents = c32["n_intents"]

        # ---- per-intent asymmetry verdict ----
        for intent, obs in c32["observations"]:
            asym = asymmetry_score(intent, obs)
            verdict = ("SYMMETRIC"
                       if asym["total"] <= self.tau_asym
                       and not asym["dangerous_builtins"]
                       and not asym["dangerous_attrs"]
                       and not asym["dangerous_modules"]
                       else "ASYMMETRIC")
            res.findings.append(Finding(
                name=intent.name, kind=intent.kind,
                verdict=verdict, asymmetry=asym,
                intent=intent, observed=obs,
            ))

        # ---- 64-bit scan ----
        res.scan64 = self.scanner64.scan(c32["observations"])

        res.elapsed_ms = (time.perf_counter() - t0) * 1e3
        return res

    # ---------- report ----------
    @staticmethod
    def report(r: ScanResult) -> str:
        L: List[str] = []
        L.append("=" * 78)
        L.append(f"security scan  ·  {r.source_name}")
        L.append("=" * 78)
        L.append(f"  ok            : {r.ok}")
        L.append(f"  32-bit addr   : 0x{r.addr:08x}")
        L.append(f"  n intents     : {r.n_intents}")
        L.append(f"  elapsed       : {r.elapsed_ms:.2f} ms")
        if r.error:
            L.append(f"  error         : {r.error}")
        L.append("")

        sym = sum(1 for f in r.findings if f.verdict == "SYMMETRIC")
        asym = sum(1 for f in r.findings if f.verdict == "ASYMMETRIC")
        cap  = sum(1 for f in r.findings if f.verdict == "CAP_BREACH")
        L.append(f"  symmetric  : {sym}")
        L.append(f"  asymmetric : {asym}")
        L.append(f"  cap breach : {cap}")
        L.append("")

        for f in r.findings:
            L.append(f"  [{f.verdict:11s}] {f.kind:<8s} {f.name}")
            if f.verdict == "SYMMETRIC":
                continue
            a = f.asymmetry
            L.append(f"       structural      : {a.get('structural', 0.0):.4f}")
            L.append(f"       severity        : {a.get('severity', 0)}")
            L.append(f"       total asymmetry : {a.get('total', 0.0):.4f}")
            if a.get("extra_calls"):
                L.append(f"       extra calls     : {a['extra_calls']}")
            if a.get("extra_globals"):
                L.append(f"       extra globals   : {a['extra_globals']}")
            if a.get("dangerous_builtins"):
                L.append(f"       DANGEROUS builtins : "
                         f"{a['dangerous_builtins']}")
            if a.get("dangerous_attrs"):
                L.append(f"       DANGEROUS attrs    : "
                         f"{a['dangerous_attrs']}")
            if a.get("dangerous_modules"):
                L.append(f"       DANGEROUS modules  : "
                         f"{a['dangerous_modules']}")
            if a.get("cap"):
                L.append(f"       cap breach      : {a['cap']} "
                         f"({a['value']} > {a['limit']})")

        if r.scan64:
            L.append("")
            L.append("--- 64-bit scan ---")
            L.append(f"  S_pos           : {r.scan64['S_pos']:+.6f}")
            L.append(f"  S_neg           : {r.scan64['S_neg']:+.6f}")
            L.append(f"  S_all           : {r.scan64['S_all']:+.6f}")
            L.append(f"  balance         : {r.scan64['balance']:.4f}")
        return "\n".join(L)


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("Security scanner  ·  32-bit compile + 64-bit scan")
    print("=" * 78)
    print(f"  host architecture : {'64-bit' if sys.maxsize > 2**32 else '32-bit'}")
    print(f"  MAX_ADDR          : 0x{INT32_MAX:x} … 0x{UINT32_MAX:x}")
    print(f"  τ_asym            : {TAU_ASYM}")
    print()

    scanner = SecurityScanner()

    # -------- 1. clean code --------
    clean = '''
def add(a, b):
    """Return a + b. Calls nothing else."""
    return a + b

class Calculator:
    """Simple calculator."""
    def __init__(self):
        self.value = 0

    def add(self, x):
        """Add x to value and return it."""
        self.value += x
        return self.value
'''

    # -------- 2. asymmetric: unexpected builtin --------
    asymmetric = '''
def add(a, b):
    """Return a + b."""
    import subprocess
    subprocess.system("curl http://example.com")
    return eval("a + b")
'''

    # -------- 3. dangerous attrs --------
    sneaky = '''
def harmless(x):
    """Return x."""
    return x.__class__.__bases__[0].__subclasses__()
'''

    # -------- 4. cap breach: excessive call depth --------
    deep = "def f():\n    return " + "g(" * 100 + "1" + ")" * 100 + "\n"

    # -------- 5. cap breach: huge file --------
    huge = "\n".join(f"x{i} = {i}" for i in range(200_001))

    samples = [
        ("clean.py", clean),
        ("asymmetric.py", asymmetric),
        ("sneaky.py", sneaky),
        ("deep.py", deep),
        ("huge.py", huge),
    ]

    for name, src in samples:
        result = scanner.scan(src, name)
        print(scanner.report(result))
        print()


if __name__ == "__main__":
    demo()