#!/usr/bin/env python3
"""
reasoning_chain_of_thought_tokenizer.py
=======================================

Long-term thought with an elliptic Möbius SAT buffer whose if/else
decision is driven by the **binary chirality scalar** of the
rank-6 Levi-Civita projection:

    Z ∈ ℂ^{6×6}                    the spinor projector
    q ∈ [0, K)                     quadratic addresses, one per slot
    Π₆(Z) = Σ_σ ε(σ) Π_k Z_{k,σ(k)}   rank-6 contraction  (720 terms)
    χ(Z)  = arg Π₆(Z)              chirality angle
    c(Z)  = 0 if cos χ ≥ 0 else 1  binary chirality  ∈ {0, 1}
    survivor = 1 − c               inverse scalar    ∈ {0, 1}

The [0, 1] range is exactly if/else:
    0  →  eliminated
    1  →  survivor

    ┌───────────────────────────────────────────────────────┐
    │  PRESENTATION LAYER  (user-facing)                    │
    │    clean prose                                        │
    └───────────────────────────────────────────────────────┘
                          ▲
    ┌───────────────────────────────────────────────────────┐
    │  REASONING LAYER  (hidden)                            │
    │    • spinor projector Z[6,6]  —  holds the q slots    │
    │    • rank-6 Levi-Civita       →  Π₆(Z)                │
    │    • chirality                →  c ∈ {0, 1}           │
    │    • inverse scalar           →  survivor ∈ {0, 1}    │
    │    • elliptic trace           →  secondary score      │
    │    • supertrace / mass        →  ensemble invariants  │
    └───────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from itertools import permutations
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================
#  Constants
# ============================================================
PI      = math.pi
E       = math.e
ALPHA   = 1.0 / (PI - E)                  # ≈ 2.362
NORM    = 1.0 - math.exp(-ALPHA * (PI + E))
DENSITY = 6.0 / (PI * PI)                 # ≈ 0.6079


# ============================================================
#  Rank-6 Levi-Civita sign table  (720 entries, computed once)
# ============================================================
_LEVI_TERMS_6: List[Tuple[Tuple[int, ...], int]] = []
for _p in permutations(range(6)):
    _inv = sum(1 for i in range(6) for j in range(i + 1, 6)
               if _p[i] > _p[j])
    _LEVI_TERMS_6.append((_p, (-1) ** _inv))


def levi_civita_6(Z: np.ndarray) -> complex:
    """Π₆(Z) = Σ_σ ε(σ) Π_k Z[k, σ(k)]."""
    total = 0.0 + 0.0j
    for perm, sign in _LEVI_TERMS_6:
        prod = 1.0 + 0.0j
        for k, c in enumerate(perm):
            prod *= Z[k, c]
        total += sign * prod
    return total


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
#  Deterministic hash  (avoids Python's randomised hash)
# ============================================================
def _dh(obj) -> int:
    s = repr(obj).encode("utf-8")
    return int(hashlib.md5(s).hexdigest()[:8], 16)


# ============================================================
#  Statement and Query
# ============================================================
@dataclass
class Statement:
    """A candidate answer — a thought."""
    text: str
    args: List[str] = field(default_factory=list)
    source: str = "buffer"

    @property
    def arity(self) -> int:
        """k = 1, 2, 3, … — the natural arity of the statement."""
        return len(self.args)


@dataclass
class Query:
    """A user query with a set of requirements."""
    text: str
    clauses: List[List[Tuple[int, int]]] = field(default_factory=list)


# ============================================================
#  Spinor projector  ·  stores the quadratic addresses
# ============================================================
class ChiralProjector:
    """
    The 6×6 complex spinor whose entries are derived from the
    quadratic addresses of a candidate.

    Each slot of the 6×6 grid holds exactly one address:

        q_slots[i, j] ∈ [0, K)       integer address
        Z[i, j]       = r · e^{i φ}  complex entry

        r  = q / K                    (bounded by 1)
        φ  = 2π q / K                 (phase over [0, 2π))

    The rank-6 contraction Π₆(Z) gives the chirality angle, and
    the q_slots grid gives the Möbius gate input.
    """
    def __init__(self, K: int,
                 quad_A: int = 1, quad_B: int = 1, quad_C: int = 0):
        self.K = int(K)
        self.A, self.B, self.C = quad_A, quad_B, quad_C
        self.Z = np.zeros((6, 6), dtype=complex)
        self.q_slots = np.zeros((6, 6), dtype=int)

    def quadratic_address(self, a: int) -> int:
        return (self.A * a * a + self.B * a + self.C) % self.K

    def clear(self) -> None:
        self.Z[:] = 0
        self.q_slots[:] = 0

    def store_slot(self, i: int, j: int, q: int) -> None:
        """Store a quadratic address in slot (i, j)."""
        self.q_slots[i, j] = int(q)
        r = (q / self.K) if self.K > 0 else 0.0
        phi = 2.0 * PI * q / max(self.K, 1)
        self.Z[i, j] = r * (math.cos(phi) + 1j * math.sin(phi))

    def fill(self, addresses: List[int]) -> None:
        """Fill all 36 slots with the given address stream (cycled)."""
        self.clear()
        if not addresses:
            addresses = [0]
        for idx in range(36):
            i, j = divmod(idx, 6)
            q = addresses[idx % len(addresses)]
            self.store_slot(i, j, q)

    def contract(self) -> complex:
        return levi_civita_6(self.Z)


# ============================================================
#  Chiral SAT  ·  the if/else tokenizer driven by chirality
# ============================================================
class ChiralSAT:
    """
    Decision layer:

        • build the address stream from arguments + clauses
        • store them in the spinor projector
        • contract rank-6                →  Π₆
        • chirality  c = 0 if cos χ ≥ 0 else 1
        • survivor = 1 − c               the inverse scalar
        • Möbius gate on the same slots  (secondary check)

    The final decision:
        survivor_bit ∈ {0, 1}   →  if/else
    """
    def __init__(self,
                 quad_A: int = 1,
                 quad_B: int = 1,
                 quad_C: int = 0,
                 i_exp: float = 2.0,
                 K_filter: int = 2 ** 8 + 1):
        self.A, self.B, self.C = quad_A, quad_B, quad_C
        self.i_exp = i_exp
        self.K = int(K_filter)
        self.mu = mobius_sieve(self.K)
        self.projector = ChiralProjector(
            self.K, quad_A=quad_A, quad_B=quad_B, quad_C=quad_C)

    # ---- address stream ----
    def build_addresses(self, stmt: Statement,
                        query: Query) -> List[int]:
        """
        Argument slots → addresses.
        Clause slots   → addresses.
        Each address is  q(a) = A a² + B a + C  mod K
        with  a  a deterministic hash of (slot, content).
        """
        addrs: List[int] = []
        for slot, arg in enumerate(stmt.args[:6]):
            a = (slot + 1) * 6 + (_dh((arg, slot)) & 0xFF)
            addrs.append(self.projector.quadratic_address(a))
        for slot, clause in enumerate(query.clauses[:6]):
            h = _dh((tuple(sorted(clause)), slot))
            a = (len(stmt.args) + slot + 1) * 6 + (h & 0xFF)
            addrs.append(self.projector.quadratic_address(a))
        if not addrs:
            addrs = [self.projector.quadratic_address(1)]
        return addrs

    # ---- chirality ----
    @staticmethod
    def chirality_of(Pi: complex) -> Tuple[float, int]:
        chi = math.atan2(Pi.imag, Pi.real)
        c = 0 if math.cos(chi) >= 0.0 else 1
        return chi, c

    # ---- Möbius gate on the same address stream ----
    def gate_pass(self, addresses: List[int]) -> Tuple[bool, float]:
        if not addresses:
            return False, 0.0
        hits = sum(1 for q in addresses
                   if 0 <= q < len(self.mu) and self.mu[q] != 0)
        density = hits / len(addresses)
        return density >= 0.5, density

    # ---- elliptic trace (secondary) ----
    def elliptic_trace(self, stmt: Statement) -> float:
        a = 0
        for i, arg in enumerate(stmt.args[:8]):
            if arg:
                a |= (1 << i)
        n_bits = max(4, min(8, stmt.arity + 2))
        bits = [(a >> i) & 1 for i in range(n_bits)]
        t = sum(b * math.sin((i + 1) * 0.7) for i, b in enumerate(bits))
        u = sum(b * math.cos((i + 1) * 1.3) for i, b in enumerate(bits))
        x = 0.5 + 4.5 * (0.5 * (1.0 + math.tanh(t)))
        y = 0.5 + 4.5 * (0.5 * (1.0 + math.tanh(u)))
        return x ** (self.i_exp - 1) + y ** (self.i_exp - 1)

    # ---- decision ----
    def decide(self, stmt: Statement, query: Query) -> Dict:
        # 1. build addresses from arguments + clauses
        addrs = self.build_addresses(stmt, query)

        # 2. store addresses in the spinor projector
        self.projector.fill(addrs)

        # 3. rank-6 contraction
        Pi = self.projector.contract()

        # 4. chirality and inverse scalar
        chi, c = self.chirality_of(Pi)
        survivor = 1 - c                    # inverse scalar ∈ {0, 1}

        # 5. secondary Möbius gate on the same slots
        gate_ok, gate_density = self.gate_pass(addrs)

        # 6. combined decision (chirality primary, gate secondary)
        accepted = (survivor == 1) and gate_ok

        # 7. elliptic trace for the ensemble
        trace = self.elliptic_trace(stmt)

        # 8. chirality-weighted trace  (0 if eliminated)
        weighted_trace = trace * survivor

        return dict(
            statement=stmt,
            arity=stmt.arity,
            addresses=addrs,
            q_slots=self.projector.q_slots.copy(),
            Z=self.projector.Z.copy(),
            Pi=Pi,
            Pi_abs=abs(Pi),
            chi=chi,
            c=c,
            survivor=survivor,                # 0 = eliminated, 1 = survivor
            gate_ok=gate_ok,
            gate_density=gate_density,
            trace=trace,
            weighted_trace=weighted_trace,
            accepted=accepted,
        )


# ============================================================
#  Exponential kernel  +  supertrace / mass
# ============================================================
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


# ============================================================
#  Reasoning buffer
# ============================================================
class ReasoningBuffer:
    """
    Persistent buffer for long-term thought.

    Every cycle runs the ChiralSAT on each candidate, keeps the
    survivors, and re-solves the ensemble supertrace from the
    chirality-weighted traces.
    """
    def __init__(self, sat: Optional[ChiralSAT] = None,
                 max_survivors: int = 32):
        self.sat = sat or ChiralSAT()
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

        # supertrace over the chirality-weighted traces
        traces = np.array([r["weighted_trace"] for r in cycle_records],
                          dtype=float)
        if len(traces) >= 2:
            conv = conv_exp_kernel(traces)
            S, H, m = supertrace_and_mass(conv)
        else:
            S, H, m = 0.0, 0.0, 0.0

        if self.accepted:
            self.accepted.sort(key=lambda r: -abs(r["weighted_trace"]))
            self.accepted = self.accepted[: self.max_survivors]

        return dict(
            cycle=self.cycle,
            n_candidates=len(candidates),
            n_accepted=sum(1 for r in cycle_records if r["accepted"]),
            n_rejected=sum(1 for r in cycle_records if not r["accepted"]),
            n_survivor_bit=sum(r["survivor"] for r in cycle_records),
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
            dict(text=r["statement"].text,
                 arity=r["arity"],
                 chi=r["chi"],
                 c=r["c"],
                 survivor=r["survivor"],
                 gate=r["gate_ok"],
                 Pi_abs=r["Pi_abs"],
                 trace=r["trace"],
                 weighted=r["weighted_trace"])
            for r in self.accepted
        ]


# ============================================================
#  Presenter
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
#  Demo  ·  chirality-driven if/else over multiple cycles
# ============================================================
def _print_spinor_grid(q_slots: np.ndarray, title: str) -> None:
    print(f"  {title}")
    for row in q_slots:
        print("    " + " ".join(f"{int(q):>4d}" for q in row))


def demo():
    print("=" * 78)
    print("Chiral SAT  ·  binary chirality as the inverse if/else scalar")
    print("=" * 78)
    print(f"  α = 1/(π − e)         = {ALPHA:.6f}")
    print(f"  gate density (6/π²)   = {DENSITY:.6f}")
    print(f"  survivor = 1 − c      ∈ {{0, 1}}")
    print(f"    c = 0  →  survivor (cos χ ≥ 0)")
    print(f"    c = 1  →  eliminated (cos χ < 0)")
    print()

    query = Query(
        text="How do I sort a large list quickly?",
        clauses=[
            [(0, +1)],
            [(1, +1)],
            [(2, +1)],
        ],
    )

    candidates = [
        Statement("Sort the list.",                             args=["list"]),
        Statement("Sort the list ascending.",                   args=["list", "asc"]),
        Statement("Sort the list descending.",                  args=["list", "desc"]),
        Statement("Use quicksort on the list, ascending.",      args=["quicksort", "list", "asc"]),
        Statement("Use mergesort on the list, stable ascending.", args=["mergesort", "list", "asc"]),
        Statement("Use heapsort with in-place ascending order.", args=["heapsort", "list", "asc"]),
        Statement("Use radix sort on ints, ascending, stable.",  args=["radix", "ints", "asc", "stable"]),
    ]

    buf = ReasoningBuffer(
        sat=ChiralSAT(quad_A=1, quad_B=1, quad_C=0,
                      i_exp=2.0, K_filter=2 ** 8 + 1),
        max_survivors=16,
    )
    presenter = Presenter()

    # ---------- cycle 1 ----------
    r1 = buf.think(query, candidates)
    print("--- cycle 1  ·  candidates through the chiral SAT ---")
    print(f"  {'text':48s}  {'k':>2s}  {'|Π₆|':>7s}  "
          f"{'χ':>7s}  {'c':>2s}  {'surv':>4s}  {'μ':>4s}  {'acc':>4s}")
    print("  " + "-" * 86)
    for rec in r1["records"]:
        t = rec["statement"].text
        t = t[:46] + (".." if len(t) > 46 else "")
        print(f"  {t:48s}  {rec['arity']:>2d}  "
              f"{rec['Pi_abs']:>7.3f}  "
              f"{rec['chi']:>+7.3f}  {rec['c']:>2d}  "
              f"{rec['survivor']:>4d}  "
              f"{int(rec['gate_ok']):>4d}  "
              f"{int(rec['accepted']):>4d}")
    print()

    # show the spinor projector's q-slots for one candidate
    sample = r1["records"][3]      # the quicksort ternary
    print("--- spinor projector  ·  quadratic addresses in the 6×6 grid ---")
    _print_spinor_grid(sample["q_slots"],
                       "candidate: 'Use quicksort on the list, ascending.'")
    print(f"    Π₆(Z) = {sample['Pi'].real:+.4e} "
          f"{sample['Pi'].imag:+.4e}i")
    print(f"    |Π₆|  = {sample['Pi_abs']:.6e}")
    print(f"    χ     = {sample['chi']:+.6f} rad")
    print(f"    c     = {sample['c']}")
    print(f"    surv  = {sample['survivor']}")
    print()

    # ---------- cycle 2 ----------
    refinements = [
        Statement("Quicksort with median-of-three pivot, ascending.",
                  args=["quicksort", "pivot", "asc"]),
        Statement("Mergesort is optimal for stable large lists.",
                  args=["mergesort", "stable", "large"]),
    ]
    r2 = buf.think(query, refinements)
    print("--- cycle 2  ·  refinements through the chiral SAT ---")
    for rec in r2["records"]:
        print(f"  {rec['statement'].text[:56]:58s}  "
              f"c={rec['c']}  surv={rec['survivor']}  "
              f"|Π₆|={rec['Pi_abs']:.4f}")
    print()

    # ---------- the only user-facing line ----------
    print("--- user-facing response ---")
    print(f"  > {presenter.present(buf.emit())}")
    print()

    # ---------- statistics ----------
    total = len(buf.accepted) + len(buf.rejected)
    c0 = sum(1 for r in buf.accepted + buf.rejected if r["c"] == 0)
    c1 = total - c0
    print("--- chirality statistics ---")
    print(f"  total candidates    : {total}")
    print(f"  c = 0 (survivor)    : {c0}   ({100*c0/total:.1f} %)")
    print(f"  c = 1 (eliminated)  : {c1}   ({100*c1/total:.1f} %)")
    print(f"  accepted (both ok)  : {len(buf.accepted)}")
    print(f"  rejected            : {len(buf.rejected)}")
    print(f"  final S             : {r1['S']:+.6f}")
    print(f"  final mass m        : {r1['m']:.6f}")
    print()

    # ---------- arity distribution ----------
    arities: Dict[int, int] = {}
    for r in buf.accepted + buf.rejected:
        arities[r["arity"]] = arities.get(r["arity"], 0) + 1
    print("--- arity distribution ---")
    for k in sorted(arities):
        kind = {1: "unary", 2: "binary", 3: "ternary",
                4: "quaternary"}.get(k, f"{k}-ary")
        print(f"  k = {k:>2d}  ({kind:<11s})  count = {arities[k]}")
    print()

    print("--- what the user never sees ---")
    print("  · spinor projector Z[6,6] with q_slots")
    print("  · rank-6 Levi-Civita Π₆(Z)  (720 terms)")
    print("  · chirality angle χ = arg Π₆(Z)")
    print("  · binary chirality c ∈ {0, 1}")
    print("  · inverse scalar survivor = 1 − c")
    print("  · Möbius gate on the same q slots")
    print("  · supertrace / mass on the weighted traces")
    print()


if __name__ == "__main__":
    demo()