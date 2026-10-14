#!/usr/bin/env python3
"""
reasoning_chain_of_thought_tokenizer.py
=======================================

Long-term thought with an elliptic Möbius SAT buffer.

UPDATED INTEGRATION
-------------------
Decision layer   →  UTF_8_dzeta_dt.py
     · UTF-8 table  v(n) = −H_{n+1} + i·α·(μ*H)(n)
     · discrete point particles at (x, y, vx, zt)
     · UGC boolean   σ(a) = σ(b)  AND  g(a,b) ≥ τ
     · read-only ALU via reconstruction coupling

Search layer     →  cpu_inference.py
     · symmetric memory search       O(K log log K)
     · asymmetric predictor          O(K log K)
     · M-matrix bounded linear alg   O(K)

Presentation     →  clean prose, nothing else.

Per-statement decision:
    text → UTF-8 bytes → particles → μ-gated vectors
         → UGC boolean → ALU arithmetic → trace → accept/reject
"""

from __future__ import annotations

import math
import struct
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================
#  Import shims
# ============================================================
try:
    from UTF_8_dzeta_dt import (
        UTF8Table, DiscreteParticle, build_particle_cloud,
        particle_vector, cross_coupling, ugc_label, ugc_satisfied,
        ugc_survivors, ReadOnlyBuffer, ReadOnlyALU, fiber_propagate,
        mobius_sieve as mobius_sieve_utf8,
        mu_convolution_H,
        ALPHA as ALPHA_UTF8,
        A_STEP as A_STEP_UTF8,
        K_DEFAULT as K_UTF8,
    )
    _HAS_UTF8 = True
except ImportError:
    _HAS_UTF8 = False

try:
    from cpu_inference import (
        CPUInferenceEngine, SymmetricMemory64,
        AsymmetricPredictor64, MobiusLinear,
        elliptic_projection,
        mobius_sieve as mobius_sieve_cpu,
        ALPHA_SYM, ALPHA_ASYM, DENSITY,
        UINT32_MAX, K_DEFAULT as K_CPU,
    )
    _HAS_CPU = True
except ImportError:
    _HAS_CPU = False


# ============================================================
#  Constants (unified)
# ============================================================
PI       = math.pi
E        = math.e
ALPHA    = 1.0 / (PI - E)               # ≈ 2.362338
A_STEP   = ALPHA / 0.3628               # ≈ 6.511
DENSITY  = 6.0 / (PI * PI)              # ≈ 0.607927
K_UTF8_DEF  = 60
K_ALU_DEF   = 64
K_CPU_DEF   = 128


# ============================================================
#  Fallback primitives if modules are missing
# ============================================================
def _fallback_mobius(K: int) -> np.ndarray:
    mu = np.zeros(K + 1, dtype=np.int8)
    if K >= 1:
        mu[1] = 1
    primes, is_comp = [], np.zeros(K + 1, dtype=bool)
    for i in range(2, K + 1):
        if not is_comp[i]:
            primes.append(i); mu[i] = -1
        for p in primes:
            if i * p > K: break
            is_comp[i * p] = True
            if i % p == 0:
                mu[i * p] = 0; break
            mu[i * p] = -mu[i]
    return mu


# ============================================================
#  Statement and Query (unchanged)
# ============================================================
@dataclass
class Statement:
    """A candidate answer — a thought."""
    text: str
    args: List[str] = field(default_factory=list)
    source: str = "buffer"

    @property
    def arity(self) -> int:
        return len(self.args)

    def payload(self) -> bytes:
        """Encode statement + args as UTF-8 bytes."""
        return (self.text.encode("utf-8")
                + b"|"
                + b"|".join(a.encode("utf-8") for a in self.args))


@dataclass
class Query:
    """A user query."""
    text: str
    clauses: List[List[Tuple[int, int]]] = field(default_factory=list)


# ============================================================
#  UGC + ALU Reasoner  ·  the decision layer
# ============================================================
class MobiusUGCReasoner:
    """
    Decision layer for the reasoning buffer.

    Pipeline per statement:

        1. text → UTF-8 bytes
        2. bytes → particle cloud  (from UTF_8_dzeta_dt)
        3. per-consecutive-pair UGC boolean decision
             cross_coupling ≥ τ
             σ(a) = σ(b)   with  σ(v) = round(k · φ / 2π) mod k
        4. read-only ALU on the survivors  (ADD / MUL)
        5. combined trace → accept/reject

    No SAT, no matrix, no Möbius gate as a separate filter — the
    UTF-8 table *is* the Möbius gate.
    """
    def __init__(self,
                 K_utf8: int = K_UTF8_DEF,
                 K_alu: int = K_ALU_DEF,
                 k_labels: int = 4,
                 threshold: float = 0.3,
                 survival_floor: float = 0.25,
                 alu_floor: float = 1e-9):
        if _HAS_UTF8:
            self.table = UTF8Table.build(K=K_utf8)
        else:
            # minimal standalone table
            mu = _fallback_mobius(K_utf8 + 1)
            H = np.zeros(K_utf8 + 2); H[1] = 1.0
            for n in range(2, K_utf8 + 2):
                H[n] = H[n - 1] + 1.0 / n
            F = np.zeros(K_utf8 + 2)
            self.table = type("T", (), dict(
                K=K_utf8, H=H, F=F, mu=mu))()
        self.K = K_utf8
        self.K_alu = K_alu
        self.k_labels = k_labels
        self.threshold = threshold
        self.survival_floor = survival_floor
        self.alu_floor = alu_floor

        if _HAS_UTF8:
            self.alu = ReadOnlyALU(K=K_alu)
            self.buffer_cls = ReadOnlyBuffer
        else:
            self.alu = None
            self.buffer_cls = None

    # --------------------------------------------------------
    def decide(self, stmt: Statement, query: Query) -> Dict:
        """Run UGC + ALU on one statement."""
        if not _HAS_UTF8:
            # graceful fallback: accept everything with uniform trace
            return dict(statement=stmt, arity=stmt.arity,
                        accepted=True, survival_rate=1.0,
                        alu_scalar=0.0, trace=float(stmt.arity),
                        n_particles=0, n_survivors=0)

        payload = stmt.payload() + b"|" + query.text.encode("utf-8")
        particles = build_particle_cloud(self.table, payload)
        n = len(particles)
        if n < 2:
            return dict(statement=stmt, arity=stmt.arity,
                        accepted=False, survival_rate=0.0,
                        alu_scalar=0.0, trace=0.0,
                        n_particles=n, n_survivors=0)

        # ---- 1. UGC boolean decision ----
        mu = self.table.mu
        surv, non = ugc_survivors(particles, self.K, mu,
                                  self.k_labels, self.threshold)
        total_edges = max(1, len(surv) + len(non))
        survival_rate = len(surv) / total_edges

        # ---- 2. read-only ALU on the survivors ----
        buf = self.buffer_cls(K=self.K)
        for p in particles:
            buf.append(struct.pack(">dd", p.x, p.y))

        alu_zs = []
        for i in surv[:8]:
            if i + 1 < n:
                for op in ("ADD", "MUL"):
                    alu_zs.append(self.alu.dispatch(buf, i, i + 1, op)["z"])
        alu_scalar = float(np.mean(np.abs(alu_zs))) if alu_zs else 0.0

        # ---- 3. combined trace ----
        trace = survival_rate * (1.0 + alu_scalar)

        # ---- 4. decision ----
        accepted = (survival_rate >= self.survival_floor
                    and alu_scalar > self.alu_floor)

        return dict(
            statement=stmt,
            arity=stmt.arity,
            accepted=accepted,
            survival_rate=survival_rate,
            alu_scalar=alu_scalar,
            trace=trace,
            n_particles=n,
            n_survivors=len(surv),
        )


# ============================================================
#  Supertrace / mass over a cycle's traces
# ============================================================
def _conv_exp(signal: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
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
    return (f + b - signal) / (1 - lam * lam)


def supertrace_mass(traces: np.ndarray) -> Tuple[float, float, float]:
    """Supertrace, entropy, and mass over the trace ensemble."""
    if len(traces) == 0:
        return 0.0, 0.0, 0.0
    conv = _conv_exp(traces)
    S = 0.0
    for i, v in enumerate(conv):
        S += v if (i % 2 == 0) else -v
    if S == 0.0:
        return 0.0, 0.0, 0.0
    p = abs(S) / len(conv)
    if not (0.0 < p < 1.0):
        return S, 0.0, abs(S)
    H = -ALPHA * p * math.log(p)
    m = abs(S) * math.exp(-H)
    return float(S), float(H), float(m)


# ============================================================
#  Reasoning buffer  ·  UGC/ALU decision + CPU search
# ============================================================
class ReasoningBuffer:
    """
    Persistent long-term thought.

    Per cycle:
        1. candidate statements (either supplied or generated by
           the CPU asymmetric predictor)
        2. UGC boolean + ALU decision on each
        3. accepted statements stored in the CPU symmetric memory
        4. supertrace mass over the cycle's traces
        5. best accepted statement = current thought
    """
    def __init__(self,
                 reasoner: Optional[MobiusUGCReasoner] = None,
                 cpu: Optional[object] = None,
                 max_survivors: int = 32):
        self.reasoner = reasoner or MobiusUGCReasoner()
        self.cpu = cpu
        self.max_survivors = max_survivors
        self.accepted: List[Dict] = []
        self.rejected: List[Dict] = []
        self.cycle = 0

    # --------------------------------------------------------
    def think(self,
              query: Query,
              candidates: Optional[List[Statement]] = None,
              generate: bool = False,
              n_generated: int = 3) -> Dict:
        self.cycle += 1

        # ---------- candidate set ----------
        cands: List[Statement] = list(candidates or [])
        if generate and self.cpu is not None:
            # CPU asymmetric predictor produces new candidates
            for i in range(n_generated):
                state = hash(query.text) & UINT32_MAX
                state = self.cpu.memory.search(state + i)
                seed = self.cpu.predictor.predict(state, query.text)
                cands.append(Statement(
                    text=f"hypothesis-{i+1}-{seed & 0xFFFF:04x}",
                    args=[f"h{i}", "auto"],
                    source="cpu_predictor"))

        # ---------- UGC + ALU decision per candidate ----------
        cycle_records = []
        for stmt in cands:
            rec = self.reasoner.decide(stmt, query)
            cycle_records.append(rec)
            if rec["accepted"]:
                self.accepted.append(rec)
                if self.cpu is not None:
                    self.cpu.memory.store(
                        f"stmt:{hash(stmt.text) & 0xFFFF:x}",
                        [stmt.text] + stmt.args)
            else:
                self.rejected.append(rec)

        # ---------- supertrace mass over the cycle ----------
        traces = np.array([r["trace"] for r in cycle_records], dtype=float)
        S, H, m = supertrace_mass(traces)

        # ---------- keep top survivors ----------
        if self.accepted:
            self.accepted.sort(key=lambda r: -r["trace"])
            self.accepted = self.accepted[: self.max_survivors]

        return dict(
            cycle=self.cycle,
            n_candidates=len(cands),
            n_accepted=sum(1 for r in cycle_records if r["accepted"]),
            n_rejected=sum(1 for r in cycle_records if not r["accepted"]),
            S=S, H=H, m=m,
            best=self.accepted[0] if self.accepted else None,
            records=cycle_records,
        )

    # --------------------------------------------------------
    def emit(self) -> Optional[str]:
        """The only external output — clean prose."""
        if not self.accepted:
            return None
        return self.accepted[0]["statement"].text

    # --------------------------------------------------------
    def debug_state(self) -> List[Dict]:
        return [dict(text=r["statement"].text,
                     arity=r["arity"],
                     trace=round(r["trace"], 4),
                     survival=round(r["survival_rate"], 4),
                     alu=round(r["alu_scalar"], 4))
                for r in self.accepted]


# ============================================================
#  Presenter  ·  the public face
# ============================================================
class Presenter:
    def __init__(self):
        self.history: List[str] = []

    def present(self, text: Optional[str]) -> str:
        reply = text if text is not None else \
            "I don't have a considered answer for that yet."
        self.history.append(reply)
        return reply


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("Reasoning chain  ·  UGC boolean + ALU decision + CPU search")
    print("=" * 78)
    print(f"  α = 1/(π − e)     = {ALPHA:.6f}")
    print(f"  A_STEP            = {A_STEP:.6f}")
    print(f"  6/π²              = {DENSITY:.6f}  (μ density)")
    print()

    # ---------- CPU engine (symmetric + asymmetric search) ----------
    if _HAS_CPU:
        cpu = CPUInferenceEngine(K=K_CPU_DEF)
        print(f"  CPU engine        K = {cpu.K}  "
              f"sieve = {cpu.sieve_ms:.2f} ms")
    else:
        cpu = None
        print("  CPU engine        fallback (no module)")

    # ---------- UGC + ALU reasoner ----------
    reasoner = MobiusUGCReasoner(
        K_utf8=K_UTF8_DEF, K_alu=K_ALU_DEF,
        k_labels=4, threshold=0.3,
        survival_floor=0.25)
    print(f"  UTF-8 table       K = {reasoner.K}")
    print(f"  ALU               K = {reasoner.K_alu}")
    print()

    buf = ReasoningBuffer(reasoner=reasoner, cpu=cpu, max_survivors=16)
    presenter = Presenter()

    # ---------- query ----------
    query = Query(
        text="How do I sort a large list quickly?",
        clauses=[[(0, +1)], [(1, +1)], [(2, +1)]],
    )

    # ---------- cycle 1: given candidates ----------
    candidates = [
        Statement("Sort the list.", args=["list"]),
        Statement("Sort the list ascending.", args=["list", "asc"]),
        Statement("Sort the list descending.", args=["list", "desc"]),
        Statement("Use quicksort on the list with ascending order.",
                  args=["quicksort", "list", "asc"]),
        Statement("Use mergesort on the list for stable ascending output.",
                  args=["mergesort", "list", "asc"]),
        Statement("Use heapsort with in-place ascending order.",
                  args=["heapsort", "list", "asc"]),
        Statement("Use radix sort on ints ascending, stable.",
                  args=["radix", "ints", "asc", "stable"]),
    ]

    r1 = buf.think(query, candidates=candidates)
    print("--- cycle 1  ·  UGC + ALU decision on given candidates ---")
    print(f"  candidates   : {r1['n_candidates']}")
    print(f"  accepted     : {r1['n_accepted']}")
    print(f"  rejected     : {r1['n_rejected']}")
    print(f"  supertrace S : {r1['S']:+.4f}")
    print(f"  entropy   H  : {r1['H']:.4f}")
    print(f"  mass      m  : {r1['m']:.4f}")
    print()
    print(f"  {'text':52s}  {'k':>2s}  {'surv':>6s}  "
          f"{'alu':>8s}  {'trace':>8s}")
    for rec in r1["records"]:
        t = rec["statement"].text
        t = t[:50] + (".." if len(t) > 50 else "")
        flag = "✓" if rec["accepted"] else "✗"
        print(f"  {t:52s}  {rec['arity']:>2d}  "
              f"{rec['survival_rate']:>6.3f}  "
              f"{rec['alu_scalar']:>8.4f}  "
              f"{rec['trace']:>8.4f}  {flag}")
    print()

    # ---------- cycle 2: CPU predictor generates new candidates ----------
    r2 = buf.think(query, generate=True, n_generated=4)
    print("--- cycle 2  ·  CPU predictor generates new candidates ---")
    print(f"  candidates   : {r2['n_candidates']}")
    print(f"  accepted     : {r2['n_accepted']}")
    print(f"  mass      m  : {r2['m']:.4f}")
    print()
    for rec in r2["records"]:
        t = rec["statement"].text
        flag = "✓" if rec["accepted"] else "✗"
        print(f"  {t:52s}  trace={rec['trace']:.4f}  {flag}")
    print()

    # ---------- user-facing response ----------
    print("--- user-facing response ---")
    reply = presenter.present(buf.emit())
    print(f"  > {reply}")
    print()

    # ---------- what the user never sees ----------
    print("--- what the user never sees ---")
    print("  · UTF-8 table       v(n) = −H_{n+1} + i·α·(μ*H)(n)")
    print("  · UGC boolean       σ(a) = σ(b)  AND  g(a,b) ≥ τ")
    print("  · cross-coupling    g = |⟨v_a, v_b⟩| / (‖v_a‖·‖v_b‖)")
    print("  · read-only ALU     ADD / MUL via reconstruction coupling")
    print("  · supertrace mass   m = |S| e^{−H}")
    print("  · CPU search        symmetric memory + asymmetric predictor")
    print("  · arity k, survival rate, alu scalar, trace value")
    print()

    # ---------- statistics ----------
    total = len(buf.accepted) + len(buf.rejected)
    print("--- buffer statistics ---")
    print(f"  cycles           : {buf.cycle}")
    print(f"  total decisions  : {total}")
    print(f"  accepted         : {len(buf.accepted)}")
    print(f"  rejected         : {len(buf.rejected)}")
    if total > 0:
        print(f"  acceptance rate  : {len(buf.accepted) / total:.4f}")
    if _HAS_CPU and cpu is not None:
        print(f"  cpu memory slots : {len(cpu.memory.slots)}")
        print(f"  cpu fragments    : {len(cpu.predictor.fragments)}")
    print()

    # ---------- complexity ----------
    print("--- complexity per statement decision ---")
    print("  UTF-8 encoding               O(|text|)")
    print("  particle cloud               O(|bytes|)")
    print("  μ-gated vector per particle  O(K_utf8)")
    print("  UGC boolean per edge         O(K_utf8)")
    print("  ALU dispatch (ADD/MUL)       O(K_alu²)")
    print("  ───────────────────────────────────────")
    print("  total per statement          O(|bytes| · K_utf8 + K_alu²)")
    print()
    print("  CPU symmetric search         O(K_cpu log log K_cpu)")
    print("  CPU asymmetric predictor     O(F · K_cpu log K_cpu)")
    print()


if __name__ == "__main__":
    demo()