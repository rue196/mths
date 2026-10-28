#!/usr/bin/env python3
"""
code_db_template_compiler.py
============================

Compile a large code database into:
    · dlog-addressed ternary handles   (via mobius_packed_discrete_log)
    · reusable asymmetric templates     (via coding_full_pipeline_harness)

Architecture
------------
    LARGE CODE DB
      │
      │  add(path, source)
      ▼
    TernaryPacker
      · tokenize → trits  (0=function, 1=modular)
      · V = Σ trit·3^i
      · h = g^V mod p               (dlog address)
      │
      │  review(path)
      ▼
    CodingPipeline  (coding_full_pipeline_harness)
      · plan → encode → MOSB → cmd → grid
      · bug_tokens  = specific asymmetric tokens the harness flags
      │
      │  if bug_tokens non-empty: store template
      ▼
    AsymmetricTemplate  keyed by h and sha256(trits)
      · full_source       — the whole file
      · asym_tokens       — modular tokens only
      · bug_tokens        — the flagged subset
      · replay()          — reconstruct full_source for re-review
      · replay_asym()     — the pure asymmetric payload

Class surface
-------------
    CodeDBTemplateCompiler
        .add(path, source)      register + pack to dlog address
        .review(path)           run pipeline, extract templates
        .compile_db(files)      batch compile + review
        .match(code)            Jaccard lookup against templates
        .replay(h)              full source from dlog address
        .review_template(h)     re-run harness on stored template
        .save(path) / .load(path)   persistence (JSON)
"""

from __future__ import annotations
import hashlib
import importlib.util
import json
import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================
#  Module loading
# ============================================================
def _load_mod(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


try:
    _mpl = _load_mod("mpl", "mobius_packed_discrete_log.py")
    TernaryPacker = _mpl.TernaryPacker
    trits_to_int = _mpl.trits_to_int
    int_to_trits = _mpl.int_to_trits
    bytes_for_trits = _mpl.bytes_for_trits
    finite_step_normalize = _mpl.finite_step_normalize
    _HAS_PACKER = True
except Exception as _e:
    print(f"[warn] mobius_packed_discrete_log.py not importable: {_e}")
    print("[warn] using inline fallback packer")
    _HAS_PACKER = False


# ============================================================
#  Inline fallback packer (used only if import fails)
# ============================================================
if not _HAS_PACKER:
    PI = math.pi
    E = math.e
    ALPHA = 1.0 / (PI - E)
    A_STEP = ALPHA / 0.3628

    def trits_to_int(trits):
        V = 0
        for t in trits:
            V = V * 3 + t
        return V

    def int_to_trits(V, n_trits):
        out = []
        for _ in range(n_trits):
            out.append(V % 3)
            V //= 3
        return list(reversed(out))

    def bytes_for_trits(n_trits):
        return (n_trits + 4) // 5

    def finite_step_normalize(V, a=A_STEP):
        return math.log(V) / math.log(a) if V > 0 else 0.0

    def _is_prime(n, k=20):
        if n < 2:
            return False
        for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            if n == p:
                return True
            if n % p == 0:
                return False
        d, r = n - 1, 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for _ in range(k):
            a = random.randint(2, n - 2)
            x = pow(a, d, n)
            if x in (1, n - 1):
                continue
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    def _next_prime(n):
        if n < 2:
            return 2
        if n % 2 == 0:
            n += 1
        while not _is_prime(n):
            n += 2
        return n

    def _find_generator(p):
        factors = set()
        d, n = 2, p - 1
        while d * d <= n and d <= 200_000:
            if n % d == 0:
                factors.add(d)
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            factors.add(n)
        for g in range(2, min(p, 10_000)):
            if all(pow(g, (p - 1) // q, p) != 1 for q in factors):
                return g
        return 2

    class _InlinePacker:
        def __init__(self, K: int):
            self.K = K
            self.n_trits = K // 4
            space = 3 ** self.n_trits
            self.p = _next_prime(space)
            self.g = _find_generator(self.p)
            self.n = self.p - 1
            self.space = space
            self.group = self

        def pack_dlog(self, trits):
            V = trits_to_int(trits)
            h = pow(self.g, V, self.p)
            return h, finite_step_normalize(V)

        def pack(self, V):
            return pow(self.g, V % self.n, self.p)

        def to_bytes(self, h):
            return h.to_bytes((self.p.bit_length() + 7) // 8, "big")

        def from_bytes(self, data):
            return int.from_bytes(data, "big")

    TernaryPacker = _InlinePacker


# ---- coding pipeline ----
try:
    _cfp = _load_mod("cfp", "coding_full_pipeline_harness.py")
    CodingPipeline = _cfp.CodingPipeline
    _HAS_PIPELINE = True
except Exception as _e:
    print(f"[warn] coding_full_pipeline_harness.py not importable: {_e}")
    CodingPipeline = None
    _HAS_PIPELINE = False


# ---- tokenizer ----
try:
    _pe = _load_mod("pe", "prime_encoding_UTF_coding_languages.py")
    tokenize_source = _pe.tokenize_source
    _HAS_TOKENIZER = True
except Exception as _e:
    print(f"[warn] prime_encoding_UTF_coding_languages.py not importable: {_e}")
    tokenize_source = None
    _HAS_TOKENIZER = False


# ============================================================
#  Template dataclass
# ============================================================
@dataclass
class AsymmetricTemplate:
    h:            int
    V:            int
    n_trits:      int
    k_class:      int
    trits:        List[int]
    source_hash:  str
    full_source:  str
    asym_tokens:  List[str]
    asym_string:  str
    n_function:   int
    n_modular:    int
    problem:      str
    bug_tokens:   List[str]
    signature:    bytes
    created_at:   float = field(default_factory=time.time)

    def replay(self) -> str:
        """Full source — for full re-review by the coding harness."""
        return self.full_source

    def replay_asym(self) -> str:
        """Only asymmetric tokens — the template payload."""
        return self.asym_string

    def summary(self) -> dict:
        return dict(
            h=f"{self.h:016x}",
            n_trits=self.n_trits,
            n_function=self.n_function,
            n_modular=self.n_modular,
            n_bug_tokens=len(self.bug_tokens),
            problem=self.problem,
            signature=self.signature.hex()[:16],
        )


# ============================================================
#  Template store  ·  keyed by h and by trit hash
# ============================================================
class AsymmetricTemplateStore:
    def __init__(self):
        self.by_h: Dict[int, AsymmetricTemplate] = {}
        self.by_trit_hash: Dict[bytes, int] = {}

    def put(self, t: AsymmetricTemplate) -> None:
        self.by_h[t.h] = t
        self.by_trit_hash[hashlib.sha256(bytes(t.trits)).digest()] = t.h

    def get(self, h: int) -> Optional[AsymmetricTemplate]:
        return self.by_h.get(h)

    def lookup_by_trits(self, trits: List[int]) -> Optional[AsymmetricTemplate]:
        sig = hashlib.sha256(bytes(trits)).digest()
        h = self.by_trit_hash.get(sig)
        return self.by_h.get(h) if h is not None else None

    def match(self, asym_tokens: List[str],
              threshold: float = 0.5) -> List[Tuple[int, float]]:
        """Jaccard match of asymmetric tokens against stored templates."""
        if not asym_tokens:
            return []
        q = set(asym_tokens)
        out: List[Tuple[int, float]] = []
        for h, t in self.by_h.items():
            ts = set(t.asym_tokens)
            if not ts:
                continue
            j = len(q & ts) / len(q | ts)
            if j >= threshold:
                out.append((h, j))
        out.sort(key=lambda x: -x[1])
        return out

    @property
    def n_templates(self) -> int:
        return len(self.by_h)

    def summary(self) -> dict:
        return dict(
            n_templates=len(self.by_h),
            n_function=sum(t.n_function for t in self.by_h.values()),
            n_modular=sum(t.n_modular for t in self.by_h.values()),
            total_bug_tokens=sum(len(t.bug_tokens) for t in self.by_h.values()),
        )


# ============================================================
#  Compiler
# ============================================================
class CodeDBTemplateCompiler:
    def __init__(self, K: int = 80, B_out: int = 10,
                 K_buffer: int = 256, K_gate: int = 20):
        self.K = K
        self.packer = TernaryPacker(K=K)
        self.pipeline = (
            CodingPipeline(K_encode=K, B_out=B_out,
                           K_buffer=K_buffer, K_gate=K_gate)
            if _HAS_PIPELINE else None
        )
        self.store = AsymmetricTemplateStore()
        self.addresses: Dict[str, int] = {}
        self.entries: Dict[str, dict] = {}

    # --------------------------------------------------------
    def _trits_from_source(self, source: str):
        tokens = tokenize_source(source)
        n = self.packer.n_trits
        trits: List[int] = []
        for _, kind in tokens:
            trits.append(0 if kind == 'function' else 1)
            if len(trits) >= n:
                break
        if len(trits) < n:
            trits.extend([0] * (n - len(trits)))
        return tokens, trits

    # --------------------------------------------------------
    def add(self, path: str, source: str) -> dict:
        tokens, trits = self._trits_from_source(source)
        V = trits_to_int(trits)
        h, t = self.packer.pack_dlog(trits)
        self.addresses[path] = h
        self.entries[path] = dict(
            path=path, h=h, V=V, trits=trits, t=float(t),
            n_tokens=len(tokens),
            n_function=sum(1 for _, k in tokens if k == 'function'),
            n_modular=sum(1 for _, k in tokens if k == 'modular'),
            source=source,
        )
        return dict(path=path, h=h, V=V, t=float(t))

    # --------------------------------------------------------
    def review(self, path: str,
               requirement: Optional[str] = None) -> dict:
        if path not in self.entries:
            raise KeyError(path)
        if self.pipeline is None:
            return dict(path=path, error="pipeline not available")
        meta = self.entries[path]
        req = requirement or f"review {path}"
        report = self.pipeline.run(req, meta['source'])
        bug_tokens = list(report['bug_report'].bug_tokens)
        bugs = len(bug_tokens)

        result = dict(
            path=path, h=meta['h'], bugs=bugs,
            survival=report['bug_report'].ugc_survival_rate,
            n_function=report['bug_report'].n_sym,
            n_modular=report['bug_report'].n_asym,
            blocks_placed=report['grid_summary']['n_blocks_placed'],
            template_h=None,
        )
        if bugs > 0:
            tmpl = self._store_template(meta, report, bug_tokens)
            result['template_h'] = tmpl.h
            result['template_summary'] = tmpl.summary()
        return result

    # --------------------------------------------------------
    def _store_template(self, meta, report, bug_tokens) -> AsymmetricTemplate:
        tokens = tokenize_source(meta['source'])
        asym = [t for t, k in tokens if k == 'modular']
        asym_string = " ".join(asym)
        sig = hashlib.sha256(asym_string.encode()).digest()

        existing = self.store.lookup_by_trits(meta['trits'])
        if existing is not None:
            return existing

        tmpl = AsymmetricTemplate(
            h=meta['h'], V=meta['V'], n_trits=len(meta['trits']),
            k_class=self.K, trits=meta['trits'],
            source_hash=hashlib.sha256(meta['source'].encode()).hexdigest(),
            full_source=meta['source'],
            asym_tokens=asym, asym_string=asym_string,
            n_function=meta['n_function'], n_modular=meta['n_modular'],
            problem=f"bugs={len(bug_tokens)}, "
                    f"survival={report['bug_report'].ugc_survival_rate:.3f}",
            bug_tokens=bug_tokens,
            signature=sig,
        )
        self.store.put(tmpl)
        return tmpl

    # --------------------------------------------------------
    def compile_db(self, files: Dict[str, str]) -> dict:
        t0 = time.perf_counter()
        for p, s in files.items():
            self.add(p, s)
        t_pack = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        reviews = [self.review(p) for p in files]
        t_review = (time.perf_counter() - t0) * 1e3

        return dict(
            n_files=len(files),
            n_buggy=sum(1 for r in reviews if r.get('bugs', 0) > 0),
            n_templates=self.store.n_templates,
            pack_ms=t_pack,
            review_ms=t_review,
            reviews=reviews,
        )

    # --------------------------------------------------------
    def match(self, code: str, threshold: float = 0.5) -> dict:
        tokens = tokenize_source(code)
        asym = [t for t, k in tokens if k == 'modular']
        matches = self.store.match(asym, threshold)
        return dict(
            n_matches=len(matches),
            matches=[dict(h=f"{h:016x}", score=round(s, 3),
                          problem=self.store.get(h).problem)
                     for h, s in matches],
        )

    def replay(self, h: int) -> Optional[str]:
        t = self.store.get(h)
        return t.replay() if t else None

    def review_template(self, h: int) -> Optional[dict]:
        t = self.store.get(h)
        if t is None or self.pipeline is None:
            return None
        report = self.pipeline.run(f"re-review {h:016x}", t.full_source)
        return dict(
            h=h,
            bugs=len(report['bug_report'].bug_tokens),
            survival=report['bug_report'].ugc_survival_rate,
        )

    # --------------------------------------------------------
    def save(self, path: str) -> None:
        import base64
        data = []
        for t in self.store.by_h.values():
            data.append(dict(
                h=str(t.h), V=str(t.V),
                n_trits=t.n_trits, k_class=t.k_class,
                trits=t.trits, source_hash=t.source_hash,
                full_source=t.full_source,
                asym_tokens=t.asym_tokens, asym_string=t.asym_string,
                n_function=t.n_function, n_modular=t.n_modular,
                problem=t.problem, bug_tokens=t.bug_tokens,
                signature_b64=base64.b64encode(t.signature).decode(),
                created_at=t.created_at,
            ))
        with open(path, "w") as f:
            json.dump(dict(templates=data), f, indent=2)

    def load(self, path: str) -> None:
        import base64
        with open(path, "r") as f:
            data = json.load(f)
        for d in data['templates']:
            t = AsymmetricTemplate(
                h=int(d['h']), V=int(d['V']),
                n_trits=d['n_trits'], k_class=d['k_class'],
                trits=d['trits'], source_hash=d['source_hash'],
                full_source=d['full_source'],
                asym_tokens=d['asym_tokens'], asym_string=d['asym_string'],
                n_function=d['n_function'], n_modular=d['n_modular'],
                problem=d['problem'], bug_tokens=d['bug_tokens'],
                signature=base64.b64decode(d['signature_b64']),
                created_at=d['created_at'],
            )
            self.store.put(t)


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("Code DB template compiler  ·  ternary dlog + coding harness")
    print("=" * 78)
    print(f"  packer available    = {_HAS_PACKER}")
    print(f"  pipeline available  = {_HAS_PIPELINE}")
    print(f"  tokenizer available = {_HAS_TOKENIZER}")
    print()

    if not _HAS_TOKENIZER:
        print("Cannot run demo: tokenizer missing")
        return

    compiler = CodeDBTemplateCompiler(K=80)

    # ---------- sample DB ----------
    files = {
        "calc_ok.py": """
def add(a, b):
    return a + b
""",
        "calc_bug.py": """
def add(a, b):
    return a - b

def main():
    print(add(2, 3))
""",
        "calc_bug2.py": """
class Calculator:
    def __init__(self):
        self.value = 0

    def add(self, x):
        self.value -= x
        return self.value
""",
        "hello.py": """
def greet(name):
    return "Hello, " + name
""",
        "loop_bug.py": """
def spin(n):
    i = 0
    while i < n:
        print(i)
""",
    }

    print(f"--- DB  ({len(files)} files) ---")
    for p, s in files.items():
        print(f"  {p:<14s}  {len(s):>4} chars")
    print()

    # ---------- compile ----------
    result = compiler.compile_db(files)

    print(f"--- compile report ---")
    print(f"  files       = {result['n_files']}")
    print(f"  buggy       = {result['n_buggy']}")
    print(f"  templates   = {result['n_templates']}")
    print(f"  pack time   = {result['pack_ms']:.2f} ms")
    print(f"  review time = {result['review_ms']:.2f} ms")
    print()

    print(f"  {'file':<14s}  {'h (dlog)':>18s}  "
          f"{'bugs':>5s}  {'surv':>6s}")
    for r in result['reviews']:
        hx = f"{r['h']:016x}"
        bugs = r.get('bugs', 0)
        surv = r.get('survival', 0.0)
        print(f"  {r['path']:<14s}  {hx:>18s}  "
              f"{bugs:>5d}  {surv:>6.3f}")
    print()

    # ---------- template store ----------
    if compiler.store.n_templates > 0:
        print(f"--- template store ---")
        for k, v in compiler.store.summary().items():
            print(f"  {k:<18s}  {v}")
        print()
        print(f"  {'h (dlog)':>18s}  {'func':>4s}  {'mod':>3s}  "
              f"{'bugs':>4s}  problem")
        for h, t in compiler.store.by_h.items():
            print(f"  {f'{h:016x}':>18s}  {t.n_function:>4d}  "
                  f"{t.n_modular:>3d}  {len(t.bug_tokens):>4d}  "
                  f"{t.problem}")
        print()

        # ---------- match new code ----------
        print(f"--- match new code against templates ---")
        new_code = """
def add(a, b):
    return a - b
"""
        m = compiler.match(new_code, threshold=0.3)
        print(f"  n_matches = {m['n_matches']}")
        for match in m['matches'][:3]:
            print(f"    h={match['h']}  score={match['score']}  "
                  f"{match['problem']}")
        print()

        # ---------- replay ----------
        first_h = next(iter(compiler.store.by_h))
        print(f"--- replay template h={first_h:016x} ---")
        code = compiler.replay(first_h)
        for line in code.strip().split("\n")[:4]:
            print(f"    {line}")
        print()

        review = compiler.review_template(first_h)
        if review:
            print(f"  re-review:")
            for k, v in review.items():
                print(f"    {k:<10s}  {v}")
            print()

        # ---------- persistence ----------
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        tmp.close()
        try:
            compiler.save(tmp.name)
            size = os.path.getsize(tmp.name)
            print(f"--- persistence ---")
            print(f"  saved {size} bytes → {os.path.basename(tmp.name)}")
            compiler2 = CodeDBTemplateCompiler(K=80)
            compiler2.load(tmp.name)
            print(f"  loaded {compiler2.store.n_templates} templates")
            print()
        finally:
            os.unlink(tmp.name)
    else:
        print("--- no templates stored (harness found no bug tokens) ---")
        print()

    # ---------- complexity ----------
    print(f"--- complexity ---")
    print(f"  tokenize              O(|source|)")
    print(f"  trit extraction       O(K)")
    print(f"  V = trits_to_int      O(K)")
    print(f"  h = g^V (pack)        O(log V) = O(K)")
    print(f"  dlog (BSGS)           O(√(3^(K/4)))  =  O(1.732^(K/4))")
    print(f"  review                O(K·K_utf8 + K·K_buffer)")
    print(f"  Jaccard match         O(n_templates · K)")
    print()
    print(f"  K=80  →  N=20  →  BSGS ≈ 60K ops  (fine)")
    print(f"  K=160 →  N=40  →  BSGS ≈ 1e10 ops (skip verify)")
    print()
    print("Done.")


if __name__ == "__main__":
    demo()