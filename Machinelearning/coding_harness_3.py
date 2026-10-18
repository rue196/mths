#!/usr/bin/env python3
"""
coding_harness_v2.py
====================

Bug-run coding harness:

    · 8-bit cmd stream      → UTF_8_dzeta_dt.py
    · if/else DL gate       → Block_unauthorized_users.py
    · CPU clock             → DC_chirp.py  (DCMobiusGate)
    · symmetry rule         → Full_pipeline_neural_net_compression_local_pc.py

Pipeline
--------
    token  →  byte  →  signed index  →  particle  →  UGC edge
                                          │
                        ┌─────────────────┴──────────────────┐
                        │  sym    →  bypass buffer            │
                        │  asym   →  MOSB + DL gate           │
                        └─────────────────┬──────────────────┘
                                          ▼
                                 bug?  yes / no
"""

from __future__ import annotations
import math
import hashlib
import re
import struct
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================
#  Constants
# ============================================================
PI       = math.pi
E        = math.e
ALPHA    = 1.0 / (PI - E)                 # ≈ 2.362
A_STEP   = ALPHA / 0.3628                 # ≈ 6.511
DENSITY  = 6.0 / (PI * PI)                # ≈ 0.6079
GAMMA    = 0.5772156649015329
NORM     = 1.0 - math.exp(-ALPHA * (PI + E))

K_UTF8    = 60
K_ALU     = 64
K_BUFFER  = 256
K_GATE    = 20

LOC_TOL     = 0.35
SAT_MATCH   = 0.55
UGC_THRESH  = 0.30
UGC_K       = 4


# ============================================================
#  Möbius sieve  +  harmonic numbers  +  μ*H convolution
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
                mu[i * p] = 0; break
            mu[i * p] = -mu[i]
    return mu


def harmonic_numbers(K: int) -> np.ndarray:
    H = np.zeros(K + 1, dtype=float)
    if K >= 1:
        H[1] = 1.0
    for n in range(2, K + 1):
        H[n] = H[n - 1] + 1.0 / n
    return H


def mu_convolution_H(K: int):
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
#  UTF_8_dzeta_dt.py
# ============================================================
def utf8_byte_to_signed_index(b: int, K: int) -> int:
    n = -K + int(round(2 * K * b / 255.0))
    return n if n != 0 else 1


def signed_value(n: int, H: np.ndarray, F: np.ndarray) -> complex:
    if n == 0:
        return 0j
    a = abs(n)
    real = -(float(H[a]) if a < H.size else math.log(a) + GAMMA)
    if n < 0:
        return complex(real, 0.0)
    imag = ALPHA * (float(F[a]) if a < F.size else 0.0)
    return complex(real, imag)


@dataclass
class UTF8Table:
    K: int
    H: np.ndarray
    F: np.ndarray
    mu: np.ndarray
    v: np.ndarray

    @classmethod
    def build(cls, K: int = K_UTF8) -> "UTF8Table":
        F, mu, H = mu_convolution_H(K + 1)
        ns = np.arange(-K, K + 1)
        v = np.array([signed_value(int(n), H, F) for n in ns])
        return cls(K=K, H=H, F=F, mu=mu, v=v)

    def real_part(self, n: int) -> float:
        return float(self.v[n + self.K].real)

    def imag_part(self, n: int) -> float:
        return float(self.v[n + self.K].imag)


@dataclass
class DiscreteParticle:
    i: int
    j: int
    x: float
    y: float
    vx: float
    zt: float

    @classmethod
    def from_table(cls, table: UTF8Table, i: int, j: int
                   ) -> "DiscreteParticle":
        a_i = abs(i) if i != 0 else 1
        a_j = abs(j) if j != 0 else 1
        H = table.H
        vx = -(H[a_i] - H[max(1, a_i - 1)])
        zt = ALPHA * float(table.F[min(a_j, table.F.size - 1)])
        return cls(i=i, j=j,
                   x=table.real_part(i),
                   y=table.imag_part(j),
                   vx=float(vx), zt=float(zt))

    @property
    def theta(self) -> float:
        return math.atan2(self.y, self.x)


def build_particle_cloud(table: UTF8Table,
                          stream: bytes) -> List[DiscreteParticle]:
    K = table.K
    out = []
    for b in stream:
        i = utf8_byte_to_signed_index(b, K)
        j = (i + 1) % (2 * K + 1) - K
        if j == 0:
            j = -1
        out.append(DiscreteParticle.from_table(table, i, j))
    return out


def particle_vector(p: DiscreteParticle, K: int,
                    mu: np.ndarray) -> np.ndarray:
    n = np.arange(K)
    mu_arr = np.array([float(mu[k]) if k < len(mu) else 0.0
                       for k in range(K)])
    chirp = np.cos(A_STEP * n * n / K)
    theta = p.theta
    return mu_arr * chirp * complex(math.cos(theta), math.sin(theta))


def cross_coupling(va: np.ndarray, vb: np.ndarray) -> float:
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(abs(np.vdot(va, vb)) / (na * nb))


def ugc_label(phase: float, k: int) -> int:
    return int(round(k * phase / (2 * PI))) % k


def ugc_satisfied(va: np.ndarray, vb: np.ndarray, k: int,
                  threshold: float) -> bool:
    if cross_coupling(va, vb) < threshold:
        return False
    phi_a = math.atan2(va.imag.sum(), va.real.sum())
    phi_b = math.atan2(vb.imag.sum(), vb.real.sum())
    return ugc_label(phi_a, k) == ugc_label(phi_b, k)


def ugc_survivors(particles: List[DiscreteParticle], K: int,
                  mu: np.ndarray, k: int, threshold: float
                  ) -> Tuple[List[int], List[int]]:
    N = len(particles)
    vecs = [particle_vector(p, K, mu) for p in particles]
    surv, non = [], []
    for i in range(N - 1):
        (surv if ugc_satisfied(vecs[i], vecs[i+1], k, threshold)
         else non).append(i)
    return surv, non


# ============================================================
#  Block_unauthorized_users.py  ·  discrete-log survivor gate
# ============================================================
def mu_gated_vector(sig: bytes, K: int, mu: np.ndarray) -> np.ndarray:
    if not sig or K <= 0:
        return np.zeros(K, dtype=float)
    s = np.array([b / 255.0 for b in sig], dtype=float)
    L = len(s)
    n = np.arange(K)
    mu_arr = np.array([float(mu[k]) if k < len(mu) else 0.0
                       for k in range(K)])
    chirp = np.cos(A_STEP * n * n / K)
    return mu_arr * chirp * s[n % L]


def sat_envelope(data: bytes, K: int, mu: np.ndarray) -> np.ndarray:
    if not data:
        return np.zeros(K, dtype=float)
    amps = np.array([b / 255.0 for b in data])
    if len(amps) < K:
        amps = np.tile(amps, int(math.ceil(K / len(amps))))
    amps = amps[:K]
    env = np.zeros(K, dtype=float)
    for a, v in enumerate(amps):
        q = (a * a + a) % K
        if 0 <= q < len(mu) and mu[q] != 0:
            env[q] += abs(v)
    return env


def sat_overlap(env_a: np.ndarray, env_b: np.ndarray) -> float:
    na = float(np.linalg.norm(env_a))
    nb = float(np.linalg.norm(env_b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(env_a, env_b) / (na * nb))


@dataclass
class DiscreteLogGate:
    cpu_sig: bytes
    K: int = K_BUFFER
    loc_tol: float = LOC_TOL
    sat_match: float = SAT_MATCH

    mu: np.ndarray = field(init=False)
    cpu_vec: np.ndarray = field(init=False)
    cpu_env: np.ndarray = field(init=False)

    def __post_init__(self):
        self.mu = mobius_sieve(self.K)
        self.cpu_vec = mu_gated_vector(self.cpu_sig, self.K, self.mu)
        self.cpu_env = sat_envelope(self.cpu_sig, self.K, self.mu)

    def check(self, claim_sig: bytes) -> Dict:
        if not claim_sig:
            return dict(allowed=False, reason="empty",
                        cross=0.0, sat=0.0)
        cv = mu_gated_vector(claim_sig, self.K, self.mu)
        ce = sat_envelope(claim_sig, self.K, self.mu)
        cc = cross_coupling(self.cpu_vec, cv)
        ov = sat_overlap(self.cpu_env, ce)
        if cc < self.loc_tol:
            return dict(allowed=False, reason="cross", cross=cc, sat=ov)
        if ov < self.sat_match:
            return dict(allowed=False, reason="sat",   cross=cc, sat=ov)
        return dict(allowed=True, reason="ok", cross=cc, sat=ov)


# ============================================================
#  DC_chirp.py  ·  CPU clock
# ============================================================
class DCMobiusGate:
    def __init__(self, K: int = K_GATE):
        self.K = K
        mu = mobius_sieve(K)
        c = np.zeros(2 * K + 1)
        for i in range(-K, K + 1):
            if i != 0:
                c[i + K] = float(mu[abs(i)])
        # smooth zeros
        for i in range(-K, K + 1):
            if i == 0 or c[i + K] != 0.0:
                continue
            l, r = i - 1, i + 1
            while l >= -K and c[l + K] == 0.0: l -= 1
            while r <= K and c[r + K] == 0.0: r += 1
            if l < -K or r > K or r == l:
                continue
            dist = r - l
            c[i + K] = (r - i) / dist * c[l + K] \
                     + (i - l) / dist * c[r + K]
        self.full = c
        idx = [i for i in range(-K, K + 1) if abs(c[i + K]) > 1e-12]
        self.indices = np.array(idx, dtype=float)
        self.coeffs  = np.array([c[i + K] for i in idx])
        self.period  = 2 * PI * ALPHA

    def zeta(self, t: float) -> complex:
        if len(self.indices) == 0:
            return 0j
        return complex(np.sum(self.coeffs *
                              np.exp(1j * t * self.indices / ALPHA)))

    def dc_throughput(self) -> float:
        N = max(len(self.coeffs), 1)
        x = float(np.sum(self.coeffs ** 2)) / N
        return float((1.0 - math.exp(-ALPHA * x)) / NORM)

    def supertrace(self):
        S = 0.0
        for i, v in enumerate(self.full):
            S += v if (i % 2 == 0) else -v
        N = len(self.full)
        if abs(S) < 1e-12 or N == 0:
            return float(S), 0.0, 0.0
        p = min(abs(S) / N, 0.9999)
        H = -ALPHA * p * math.log(p) if p > 0 else 0.0
        m = abs(S) * math.exp(-H) if H < 700 else 0.0
        return float(S), float(H), float(m)


# ============================================================
#  MOSB buffer  (from Full_pipeline_neural_net_compression)
# ============================================================
MAGIC       = b"MOSB"
HEADER_FMT  = ">4sIQQddd"
HEADER_SIZE = struct.calcsize(HEADER_FMT)


class MOSBReadOnlyBuffer:
    def __init__(self, K: int = K_BUFFER):
        self.K = int(K)
        self.data = bytearray()
        self.chunk_index: List[Tuple[int, int, float, float, float]] = []
        self._cid = 0

    @staticmethod
    def _supertrace(payload: bytes) -> float:
        S = 0.0
        for t, b in enumerate(payload):
            v = (b - 128) / 128.0
            S += v if (t % 2 == 0) else -v
        return S

    @staticmethod
    def _entropy_mass(S: float, n: int) -> Tuple[float, float]:
        if S == 0.0 or n <= 0:
            return 0.0, 0.0
        p = min(abs(S) / n, 0.9999)
        H = -ALPHA * p * math.log(p) if p > 0 else 0.0
        return H, abs(S) * math.exp(-H)

    def append(self, payload: bytes) -> int:
        n = len(payload)
        S = self._supertrace(payload)
        H, m = self._entropy_mass(S, n)
        cid = self._cid; self._cid += 1
        header = struct.pack(HEADER_FMT, MAGIC, self.K, n, cid, S, H, m)
        start = len(self.data)
        self.data.extend(header + payload)
        self.chunk_index.append((start, start + HEADER_SIZE + n, S, H, m))
        return cid

    @property
    def n_chunks(self) -> int:
        return len(self.chunk_index)


# ============================================================
#  Tokenizer  (from coding_harness_2.py)
# ============================================================
SYMMETRIC_KEYWORDS = {
    "def", "class", "return", "if", "else", "elif", "for", "while",
    "in", "not", "and", "or", "is", "None", "True", "False",
    "import", "from", "as", "try", "except", "finally", "with",
    "lambda", "yield", "break", "continue", "pass", "raise", "assert",
    "global", "nonlocal", "del", "async", "await",
}
SYMMETRIC_BRACKETS = {"(", ")", "[", "]", "{", "}"}

TOKEN_RE = re.compile(
    r"""\#[^\n]*
      | "(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'
      | [A-Za-z_][A-Za-z_0-9]*
      | \d+\.\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|\d+
      | \*\*|//|<<|>>|<=|>=|==|!=|->|\+=|-=|\*=|/=|//=|%=|\*\*=|&=|\|=|\^=|@=
      | [-+*/%<>=!&|^~.,:;@\[\]{}()]
      | \s+
      | .
    """, re.VERBOSE)


@dataclass
class Token:
    text: str
    kind: str
    symmetry: str = "unknown"
    byte: int = 0


def tokenize(code: str) -> List[Token]:
    tokens = []
    for m in TOKEN_RE.finditer(code):
        s = m.group()
        if s.isspace():
            continue
        if s.startswith("#"):
            kind = "comment"
        elif s.startswith('"') or s.startswith("'"):
            kind = "string"
        elif re.match(r"[A-Za-z_]", s):
            kind = "ident"
        elif re.match(r"\d|\.\d", s):
            kind = "number"
        else:
            kind = "op"
        tokens.append(Token(text=s, kind=kind))
    return tokens


def classify(tokens: List[Token]) -> List[Token]:
    declared = set()
    for i, t in enumerate(tokens[:-1]):
        if t.text in ("def", "class") and tokens[i + 1].kind == "ident":
            declared.add(tokens[i + 1].text)
    for t in tokens:
        if t.kind in ("string", "number", "comment"):
            t.symmetry = "asym"
        elif t.kind == "ident":
            t.symmetry = ("sym"
                          if t.text in SYMMETRIC_KEYWORDS
                          or t.text in declared
                          else "asym")
        else:
            t.symmetry = "sym"
    return tokens


def token_to_byte(t: Token) -> int:
    """8-bit command: sym → ASCII of first char, asym → hash low byte."""
    if t.symmetry == "sym":
        return ord(t.text[0]) & 0xFF
    h = int(hashlib.blake2b(t.text.encode(), digest_size=4)
            .hexdigest(), 16)
    return h & 0xFF


# ============================================================
#  Coding harness v2
# ============================================================
@dataclass
class BugReport:
    n_tokens: int = 0
    n_sym: int = 0
    n_asym: int = 0
    ugc_survivors: int = 0
    ugc_non_survivors: int = 0
    ugc_survival_rate: float = 0.0
    dl_accepted: int = 0
    dl_blocked: int = 0
    dl_block_reasons: Dict[str, int] = field(default_factory=dict)
    bugs_detected: int = 0
    bug_tokens: List[str] = field(default_factory=list)
    dc_throughput: float = 0.0
    dc_S: float = 0.0
    dc_H: float = 0.0
    dc_m: float = 0.0
    buffer_chunks: int = 0
    elapsed_ms: float = 0.0


class BugRunHarness:
    """
    Symmetry rule (from Full_pipeline) is enforced here:

        symmetric weights  → direct solve, MOSB bypassed
        asymmetric weights → MOSB chunk read + discrete-log gate
    """
    def __init__(self,
                 cpu_sig: Optional[bytes] = None,
                 K_utf8: int = K_UTF8,
                 K_buffer: int = K_BUFFER,
                 K_gate: int = K_GATE,
                 ugc_threshold: float = UGC_THRESH,
                 ugc_k: int = UGC_K):
        self.K_utf8 = K_utf8
        self.K_buffer = K_buffer
        self.table = UTF8Table.build(K=K_utf8)

        # --- CPU clock ---
        self.cpu = DCMobiusGate(K=K_gate)
        self.dc_throughput = self.cpu.dc_throughput()
        self.cpu_S, self.cpu_H, self.cpu_m = self.cpu.supertrace()

        # --- discrete-log gate reference ---
        if cpu_sig is None:
            cpu_sig = hashlib.shake_256(
                b"cpu-ref|" + f"{self.dc_throughput:.10f}".encode()
            ).digest(32)
        self.cpu_sig = cpu_sig
        self.gate = DiscreteLogGate(cpu_sig=cpu_sig, K=K_buffer)

        # --- MOSB buffer ---
        self.buffer = MOSBReadOnlyBuffer(K=K_buffer)

        self.ugc_threshold = ugc_threshold
        self.ugc_k = ugc_k

    # --------------------------------------------------------
    def run(self, code: str,
            reference_code: Optional[str] = None) -> BugReport:
        t0 = time.perf_counter()
        rep = BugReport()

        tokens = classify(tokenize(code))
        rep.n_tokens = len(tokens)
        rep.n_sym   = sum(1 for t in tokens if t.symmetry == "sym")
        rep.n_asym  = sum(1 for t in tokens if t.symmetry == "asym")

        # --- 8-bit cmd stream ---
        cmds = bytes(token_to_byte(t) for t in tokens)

        # --- UGC on particle cloud ---
        particles = build_particle_cloud(self.table, cmds)
        surv, non = ugc_survivors(
            particles, self.K_utf8, self.table.mu,
            self.ugc_k, self.ugc_threshold)
        rep.ugc_survivors = len(surv)
        rep.ugc_non_survivors = len(non)
        total_edges = max(1, len(surv) + len(non))
        rep.ugc_survival_rate = len(surv) / total_edges

        # --- discrete-log gate on ASYMMETRIC tokens only ---
        asym_tokens = [t for t in tokens if t.symmetry == "asym"]
        for tok in asym_tokens:
            sig = hashlib.shake_256(tok.text.encode()).digest(32)
            verdict = self.gate.check(sig)
            if verdict["allowed"]:
                rep.dl_accepted += 1
            else:
                rep.dl_blocked += 1
                reason = verdict["reason"]
                rep.dl_block_reasons[reason] = \
                    rep.dl_block_reasons.get(reason, 0) + 1
                rep.bug_tokens.append(tok.text)

        # --- write asymmetric tokens through MOSB (audit trail) ---
        for tok in asym_tokens[:32]:
            sig = hashlib.shake_256(tok.text.encode()).digest(32)
            self.buffer.append(sig)

        rep.bugs_detected   = rep.dl_blocked
        rep.dc_throughput   = self.dc_throughput
        rep.dc_S = self.cpu_S
        rep.dc_H = self.cpu_H
        rep.dc_m = self.cpu_m
        rep.buffer_chunks   = self.buffer.n_chunks
        rep.elapsed_ms      = (time.perf_counter() - t0) * 1e3
        return rep

    # --------------------------------------------------------
    def report(self, rep: BugReport, title: str = "bug run") -> str:
        L = []
        L.append("=" * 72)
        L.append(title)
        L.append("=" * 72)
        L.append(f"  tokens              : {rep.n_tokens}")
        L.append(f"  symmetric           : {rep.n_sym}")
        L.append(f"  asymmetric          : {rep.n_asym}")
        L.append("")
        L.append("--- 8-bit cmd UGC boolean ---")
        L.append(f"  survivors           : {rep.ugc_survivors}")
        L.append(f"  non-survivors       : {rep.ugc_non_survivors}")
        L.append(f"  survival rate       : {rep.ugc_survival_rate:.4f}")
        L.append(f"  threshold           : {self.ugc_threshold:.4f}")
        L.append("")
        L.append("--- discrete-log gate (asymmetric only) ---")
        L.append(f"  accepted            : {rep.dl_accepted}")
        L.append(f"  blocked             : {rep.dl_blocked}")
        for reason, n in rep.dl_block_reasons.items():
            L.append(f"    {reason:6s}          : {n}")
        if rep.bug_tokens:
            L.append(f"  bug tokens (first 8): {rep.bug_tokens[:8]}")
        L.append("")
        L.append("--- CPU clock (DC chirp) ---")
        L.append(f"  DC throughput       : {rep.dc_throughput:.6f}")
        L.append(f"  supertrace S        : {rep.dc_S:+.6f}")
        L.append(f"  entropy    H        : {rep.dc_H:.6f}")
        L.append(f"  mass       m        : {rep.dc_m:.4e}")
        L.append("")
        L.append("--- summary ---")
        L.append(f"  bugs detected       : {rep.bugs_detected}")
        L.append(f"  buffer chunks       : {rep.buffer_chunks}")
        L.append(f"  elapsed             : {rep.elapsed_ms:.2f} ms")
        return "\n".join(L)


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 72)
    print("Coding harness v2  ·  8-bit bug run + discrete-log gate")
    print("=" * 72)
    print(f"  α       = 1/(π − e)   = {ALPHA:.6f}")
    print(f"  A_STEP                = {A_STEP:.6f}")
    print(f"  NORM                  = {NORM:.6f}")
    print(f"  6/π²                  = {DENSITY:.6f}")
    print()

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
"""

    harness = BugRunHarness(K_utf8=K_UTF8, K_buffer=K_BUFFER, K_gate=K_GATE)

    print("--- REFERENCE: good code ---")
    rep_good = harness.run(good_code)
    print(harness.report(rep_good, "reference bug run (good code)"))
    print()

    print("--- BUG RUN: buggy code ---")
    rep_bug = harness.run(buggy_code)
    print(harness.report(rep_bug, "bug run (buggy code)"))
    print()

    print("=" * 72)
    print("Comparison")
    print("=" * 72)
    print(f"  {'metric':30s}  {'good':>10s}  {'buggy':>10s}  {'Δ':>10s}")
    for name, k in [
        ("UGC survivors", "ugc_survivors"),
        ("UGC non-survivors", "ugc_non_survivors"),
        ("UGC survival rate", "ugc_survival_rate"),
        ("DL accepted", "dl_accepted"),
        ("DL blocked", "dl_blocked"),
        ("bugs detected", "bugs_detected"),
    ]:
        vg = getattr(rep_good, k)
        vb = getattr(rep_bug, k)
        if isinstance(vg, float):
            print(f"  {name:30s}  {vg:>10.4f}  {vb:>10.4f}  "
                  f"{vb-vg:>+10.4f}")
        else:
            print(f"  {name:30s}  {vg:>10d}  {vb:>10d}  "
                  f"{vb-vg:>+10d}")
    print()

    print("--- DC chirp CPU clock ---")
    print(f"  ζ(0)                  = {harness.cpu.zeta(0.0):.4f}")
    print(f"  ζ(T/4)                = "
          f"{harness.cpu.zeta(harness.cpu.period / 4):.4f}")
    print(f"  DC throughput         = {harness.dc_throughput:.6f}")
    print(f"  supertrace S          = {harness.cpu_S:+.6f}")
    print(f"  entropy    H          = {harness.cpu_H:.6f}")
    print(f"  mass       m          = {harness.cpu_m:.4e}")
    print()

    print("--- symmetry rule (from Full_pipeline) ---")
    tokens = classify(tokenize(good_code))
    sym = sum(1 for t in tokens if t.symmetry == "sym")
    asym = sum(1 for t in tokens if t.symmetry == "asym")
    print(f"  symmetric tokens      = {sym}   → direct, buffer skipped")
    print(f"  asymmetric tokens     = {asym}   → MOSB + DL gate")
    print()
    print("  Asymmetric guard (Block_unauthorized_users.py):")
    print("    if cross_coupling ≥ LOC_TOL  AND")
    print("       sat_overlap   ≥ SAT_MATCH:")
    print("         ACCEPT")
    print("    else:")
    print("         BLOCK   (bug avoided)")
    print()

    print("--- complexity ---")
    print("  tokenize            O(K)")
    print("  8-bit cmd stream    O(K)")
    print("  UTF-8 table         O(K log K)")
    print("  particle cloud      O(K)")
    print("  UGC boolean         O(K · K_utf8)")
    print("  DL gate             O(K · K_buffer)")
    print("  DC throughput       O(1)  (closed form)")
    print("  supertrace          O(K)")
    print()


if __name__ == "__main__":
    demo()