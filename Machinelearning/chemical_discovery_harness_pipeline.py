#!/usr/bin/env python3
"""
elliptic_chemical_discovery.py
==============================

Chemical discovery harness on the elliptic algebraic / transcendental split.

    known chemicals        →  algebraic side       (1/mass vs 1/S)
    unknown candidates     →  transcendental side  (asymmetric μ-gated search)
    chirality              →  bounded elliptic Π ∈ [0,1], periods (π, e)
    data                   →  element supertraces  (elements-supertrace.py)

Pipeline
--------
  1. every chemical has (mass, S, shells)
  2. build its 6×6 spinor Z from shells
     · Π₆(Z) = det(Z),  χ = arg Π₆,  c ∈ {0,1}  (chirality)
  3. project (mass, S) onto the elliptic curve
        Π_e(mass, S) = ½(1 + cos(2·mass)·cos(2π·S/e))   ∈ [0,1]
  4. fit known chemicals on the algebraic side
        u = 1/mass, v = 1/S,   v ≈ a·u + b
  5. asymmetric search over μ-square-free candidates
        for each candidate k:  μ(k) ≠ 0
          · predict S_alg(k) = 1 / (a/mass_k + b)
          · add elliptic correction Π_e(k) · Δ(k)
          · score against chirality c(k)
  6. accept if the supertrace trace of the residuals exceeds threshold
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ============================================================
#  Constants
# ============================================================
PI       = math.pi
E        = math.e
ALPHA    = 1.0 / (PI - E)          # ≈ 2.362338
W1       = PI                       # elliptic period 1
W2       = E                        # elliptic period 2
RHO      = 7.77e-16                 # wave-expanse density
BLOCK    = 6
DENSITY  = 6.0 / (PI * PI)


# ============================================================
#  Element data  (from elements-supertrace.py)
# ============================================================
ELEMENTS = {
    1:  (1, 1, 1.008, 'H'),
    2:  (18,1, 4.003, 'He'),
    3:  (1, 2, 6.941, 'Li'),
    4:  (2, 2, 9.012, 'Be'),
    5:  (13,2, 10.81, 'B'),
    6:  (14,2, 12.011, 'C'),
    7:  (15,2, 14.007, 'N'),
    8:  (16,2, 15.999, 'O'),
    9:  (17,2, 18.998, 'F'),
    10: (18,2, 20.180, 'Ne'),
    11: (1, 3, 22.990, 'Na'),
    12: (2, 3, 24.305, 'Mg'),
    13: (13,3, 26.982, 'Al'),
    14: (14,3, 28.086, 'Si'),
    15: (15,3, 30.974, 'P'),
    16: (16,3, 32.065, 'S'),
    17: (17,3, 35.453, 'Cl'),
    18: (18,3, 39.948, 'Ar'),
    19: (1, 4, 39.098, 'K'),
    20: (2, 4, 40.078, 'Ca'),
    21: (3, 4, 44.956, 'Sc'),
    22: (4, 4, 47.867, 'Ti'),
    23: (5, 4, 50.942, 'V'),
    24: (6, 4, 51.996, 'Cr'),
    25: (7, 4, 54.938, 'Mn'),
    26: (8, 4, 55.845, 'Fe'),
    27: (9, 4, 58.933, 'Co'),
    28: (10,4, 58.693, 'Ni'),
    29: (11,4, 63.546, 'Cu'),
    30: (12,4, 65.380, 'Zn'),
    31: (13,4, 69.723, 'Ga'),
    32: (14,4, 72.630, 'Ge'),
    33: (15,4, 74.922, 'As'),
    34: (16,4, 78.960, 'Se'),
    35: (17,4, 79.904, 'Br'),
    36: (18,4, 83.798, 'Kr'),
}


# ============================================================
#  Shell filling + supertrace  (elements-supertrace.py)
# ============================================================
_SUBSHELLS = [
    (1,0,2),(2,0,2),(2,1,6),(3,0,2),(3,1,6),(4,0,2),
    (3,2,10),(4,1,6),(5,0,2),(4,2,10),(5,1,6),(6,0,2),
    (4,3,14),(5,2,10),(6,1,6),(7,0,2),(5,3,14),(6,2,10),(7,1,6),
]


def electron_shells(Z: int) -> List[int]:
    shells = [0] * 8
    remaining = Z
    for n, l, cap in _SUBSHELLS:
        if remaining <= 0:
            break
        take = min(cap, remaining)
        shells[n] += take
        remaining -= take
    return shells[1:]                    # index 0..6 → n = 1..7


def supertrace_from_shells(shells: List[int]) -> float:
    S = 0.0
    for n, e in enumerate(shells, start=1):
        sign = 1 if (n % 2 == 1) else -1
        S += sign * e
    return float(S)


def wave_radius(mass: float, rho: float = RHO) -> float:
    if rho <= 0:
        return 1.0
    V = mass / rho
    return (3.0 * V / (4.0 * PI)) ** (1.0 / 3.0)


# ============================================================
#  Chemical
# ============================================================
@dataclass
class Chemical:
    """A chemical species or candidate configuration."""
    symbol: str
    Z: int
    group: int
    period: int
    mass: float
    shells: List[int]

    # derived
    supertrace: float = field(init=False)
    chirality: int = field(init=False)          # 0 or 1
    chi_angle: float = field(init=False)        # ∈ (-π, π]
    ell_proj: float = field(init=False)         # Π_e ∈ [0,1]
    spinor: np.ndarray = field(init=False)

    def __post_init__(self):
        self.supertrace = supertrace_from_shells(self.shells)
        self.spinor = self._build_spinor()
        Pi = complex(np.linalg.det(self.spinor))
        self.chi_angle = float(np.angle(Pi))
        self.chirality = 0 if math.cos(self.chi_angle) >= 0 else 1
        self.ell_proj = elliptic_projection(self.mass, self.supertrace)

    # --------------------------------------------------------
    def _build_spinor(self) -> np.ndarray:
        """
        6×6 spinor from the first six shells:

            Z[i,j] = e_i · δ_ij + ε_ij
            ε_ij   = cos(α · (i-j) · S / mass)          (chirality chirp)

        The diagonal carries the shell population.  The off-diagonal
        carries the chiral coupling induced by the supertrace.
        """
        Z = np.zeros((BLOCK, BLOCK), dtype=complex)
        e = self.shells[:BLOCK]
        for i in range(BLOCK):
            Z[i, i] = complex(e[i], 0.0)
            for j in range(BLOCK):
                if i == j:
                    continue
                w = math.cos(ALPHA * (i - j) * self.supertrace
                             / max(self.mass, 1e-9))
                Z[i, j] = complex(w, 0.0)
        return Z


# ============================================================
#  Bounded elliptic projection  (Figure 3.5, periods π and e)
# ============================================================
def elliptic_projection(x: float, y: float) -> float:
    """
    Π(x, y) = ½(1 + cos(2π x/π) cos(2π y/e))  ∈ [0, 1].
    """
    u = (x / W1) % 1.0
    v = (y / W2) % 1.0
    return 0.5 * (1.0 + math.cos(2 * math.pi * u) *
                        math.cos(2 * math.pi * v))


def elliptic_projection_vec(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    u = (x / W1) % 1.0
    v = (y / W2) % 1.0
    return 0.5 * (1.0 + np.cos(2 * np.pi * u) * np.cos(2 * np.pi * v))


# ============================================================
#  Algebraic side  ·  1/mass  vs  1/S  (Möbius fit)
# ============================================================
def mobius_fit(mass: np.ndarray, S: np.ndarray) -> Tuple[float, float]:
    u = 1.0 / np.maximum(np.abs(mass), 1e-12)
    v = 1.0 / np.where(np.abs(S) < 1e-12, 1e-12, S)
    A = np.vstack([u, np.ones_like(u)]).T
    a, b = np.linalg.lstsq(A, v, rcond=None)[0]
    return float(a), float(b)


def mobius_predict(mass: float, a: float, b: float) -> float:
    u = 1.0 / max(abs(mass), 1e-12)
    denom = a * u + b
    if abs(denom) < 1e-12:
        return 0.0
    return 1.0 / denom


# ============================================================
#  Transcendental side  ·  imaginary-plane correction
# ============================================================
def transcendental_correction(mass: float, S: float,
                              alpha: float = ALPHA) -> float:
    """
    Δ(mass, S) = α · Im( q̄ · p ) / (1 + |p|²)
        p = mass^i,  q = S^i.
    """
    p = complex(math.cos(math.log(max(mass, 1e-12))),
                math.sin(math.log(max(mass, 1e-12))))
    q = complex(math.cos(math.log(max(abs(S), 1e-12))),
                math.sin(math.log(max(abs(S), 1e-12))))
    num = float(np.imag(np.conj(q) * p))
    den = 1.0 + abs(p) ** 2
    return alpha * num / den


# ============================================================
#  Möbius sieve  ·  asymmetric candidate gating
# ============================================================
def mobius_sieve(K: int) -> np.ndarray:
    if K < 1:
        return np.zeros(K + 1, dtype=np.int8)
    mu = np.zeros(K + 1, dtype=np.int8)
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


# ============================================================
#  Supertrace of a residual ensemble
# ============================================================
def residual_supertrace(residuals: np.ndarray) -> float:
    S = 0.0
    for k, r in enumerate(residuals):
        sign = 1.0 if (k % 2 == 0) else -1.0
        S += sign * abs(r)
    return float(S)


def entropy_mass(S: float, N: int) -> Tuple[float, float]:
    if N <= 0 or S == 0.0:
        return 0.0, 0.0
    p = min(abs(S) / N, 0.9999)
    H = -ALPHA * p * math.log(p) if p > 0 else 0.0
    return float(H), float(abs(S) * math.exp(-H)) if H < 700 else 0.0


# ============================================================
#  Discovery result
# ============================================================
@dataclass
class DiscoveryCandidate:
    """A candidate chemical plus its elliptic decision trace."""
    chemical: Chemical
    S_alg: float                    # algebraic prediction
    Delta: float                    # transcendental correction
    S_final: float                  # algebraic + Π · Δ
    residual: float                 # |S_known − S_final|
    ell_proj: float                 # Π_e ∈ [0,1]
    chirality: int                  # 0 or 1
    mu_k: int                       # Möbius gate value at the address
    score: float                    # chirality-weighted trace
    accepted: bool


class EllipticChemicalDiscovery:
    """
    Chemical discovery harness:

        known chemicals   →  algebraic fit
        candidate pool    →  μ-square-free asymmetric search
        elliptic curve    →  Π ∈ [0,1] with periods (π, e)
        chirality         →  from det(Z_shells)
        accept            →  supertrace trace above threshold
    """
    def __init__(self,
                 K_sieve: int = 512,
                 acceptance_floor: float = 0.15,
                 chirality_target: Optional[int] = None):
        self.known: List[Chemical] = []
        self.a = 0.0
        self.b = 0.0
        self.K_sieve = int(K_sieve)
        self.mu = mobius_sieve(self.K_sieve)
        self.acceptance_floor = float(acceptance_floor)
        self.chirality_target = chirality_target
        self.candidates: List[DiscoveryCandidate] = []

    # --------------------------------------------------------
    def load_known_elements(self) -> None:
        for Z, (g, p, m, sym) in ELEMENTS.items():
            shells = electron_shells(Z)
            self.known.append(Chemical(
                symbol=sym, Z=Z, group=g, period=p,
                mass=m, shells=shells))
        m_arr = np.array([c.mass for c in self.known])
        S_arr = np.array([c.supertrace for c in self.known])
        self.a, self.b = mobius_fit(m_arr, S_arr)

    # --------------------------------------------------------
    def _score(self, ch: Chemical,
               S_alg: float, Delta: float,
               ell: float,
               mu_k: int) -> Tuple[float, float]:
        S_final = S_alg + ell * Delta
        residual = abs(ch.supertrace - S_final)

        # chirality weight
        if self.chirality_target is None:
            chi_w = 1.0
        elif ch.chirality == self.chirality_target:
            chi_w = 1.0
        else:
            chi_w = 0.25              # down-weight the wrong chirality

        # μ gate: square-free indices are physical
        mu_w = 1.0 if mu_k != 0 else 0.0

        # combined score
        score = chi_w * mu_w / (1.0 + residual)
        return S_final, score

    # --------------------------------------------------------
    def search(self,
               mass_range: Tuple[float, float] = (1.0, 90.0),
               group_range: Tuple[int, int] = (1, 18),
               period_range: Tuple[int, int] = (1, 6),
               max_candidates: int = 400,
               seed: int = 2024) -> List[DiscoveryCandidate]:
        """
        Asymmetric search over μ-square-free candidate addresses.

        Each address k ∈ {1, …, K_sieve} with μ(k) ≠ 0 seeds a
        candidate.  The transcendental plane picks the mass; the
        algebraic Möbius fit predicts the supertrace; the elliptic
        projection Π and the chirality decide acceptance.
        """
        rng = np.random.default_rng(seed)
        self.candidates = []
        live = [k for k in range(1, self.K_sieve + 1) if self.mu[k] != 0]
        if not live:
            return []

        for _ in range(max_candidates):
            k = int(rng.choice(live))
            mu_k = int(self.mu[k])

            # seed the mass by the transcendental flow  k → mass
            # mass = exp(log(k) · (1 + Δ_small))  keeps the mass
            # tied to the square-free address
            base_mass = math.exp(math.log(k + 1.0) *
                                  (0.3 + 0.7 * (k % 7) / 7.0))
            mass = float(np.clip(base_mass,
                                 mass_range[0], mass_range[1]))
            group  = int(rng.integers(group_range[0], group_range[1] + 1))
            period = int(rng.integers(period_range[0], period_range[1] + 1))

            # fake Z and shells for a candidate: shells are derived
            # from the mass via the inverse harmonic map
            n_e = max(1, int(round(mass / 4)))
            shells = electron_shells(min(36, n_e * 2))
            if not any(shells):
                shells = [1] + [0] * 6

            ch = Chemical(
                symbol=f"X{k}",
                Z=int(k), group=group, period=period,
                mass=mass, shells=shells)

            S_alg = mobius_predict(mass, self.a, self.b)
            Delta = transcendental_correction(mass, ch.supertrace)
            ell = ch.ell_proj

            S_final, score = self._score(ch, S_alg, Delta, ell, mu_k)
            residual = abs(ch.supertrace - S_final)
            accepted = (score >= self.acceptance_floor)

            self.candidates.append(DiscoveryCandidate(
                chemical=ch,
                S_alg=S_alg, Delta=Delta, S_final=S_final,
                residual=residual, ell_proj=ell,
                chirality=ch.chirality, mu_k=mu_k,
                score=score, accepted=accepted,
            ))

        self.candidates.sort(key=lambda c: -c.score)
        return self.candidates

    # --------------------------------------------------------
    def best(self, n: int = 5,
             require_chirality: Optional[int] = None
             ) -> List[DiscoveryCandidate]:
        pool = self.candidates
        if require_chirality is not None:
            pool = [c for c in pool if c.chirality == require_chirality]
        return pool[:n]

    # --------------------------------------------------------
    def supertrace_report(self) -> Dict:
        if not self.candidates:
            return dict(S=0.0, H=0.0, m=0.0, n=0)
        residuals = np.array([c.residual for c in self.candidates])
        S = residual_supertrace(residuals)
        H, m = entropy_mass(S, len(residuals))
        return dict(S=S, H=H, m=m, n=len(residuals),
                    n_accepted=sum(1 for c in self.candidates if c.accepted))


# ============================================================
#  Visualization
# ============================================================
def visualize(harness: EllipticChemicalDiscovery):
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.30)

    # (a) elliptic projection surface with known chemicals
    ax = fig.add_subplot(gs[0, 0])
    xs = np.linspace(0, W1, 160)
    ys = np.linspace(0, W2, 160)
    X, Y = np.meshgrid(xs, ys)
    P = elliptic_projection_vec(X, Y)
    im = ax.imshow(P, extent=[0, W1, 0, W2], origin="lower",
                   cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.scatter([c.mass % W1 for c in harness.known],
               [c.supertrace % W2 for c in harness.known],
               c="red", s=30, edgecolors="w", linewidths=0.5,
               label="known elements")
    ax.set_xlabel("mass (mod π)")
    ax.set_ylabel("S (mod e)")
    ax.set_title("Elliptic projection Π(mass, S) ∈ [0,1]")
    ax.legend(fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (b) chirality of known elements
    ax = fig.add_subplot(gs[0, 1])
    chi0 = [c for c in harness.known if c.chirality == 0]
    chi1 = [c for c in harness.known if c.chirality == 1]
    ax.scatter([c.mass for c in chi0], [c.supertrace for c in chi0],
               c="#16a085", s=36, label=f"c=0  ({len(chi0)})")
    ax.scatter([c.mass for c in chi1], [c.supertrace for c in chi1],
               c="#e74c3c", s=36, label=f"c=1  ({len(chi1)})")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlabel("mass"); ax.set_ylabel("supertrace S")
    ax.set_title("Chirality of known chemicals")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (c) algebraic fit
    ax = fig.add_subplot(gs[0, 2])
    m_arr = np.array([c.mass for c in harness.known])
    S_arr = np.array([c.supertrace for c in harness.known])
    u = 1.0 / m_arr
    v = 1.0 / np.where(np.abs(S_arr) < 1e-9, 1e-9, S_arr)
    ax.scatter(u, v, c="#3a7bd5", s=28, label="known")
    uu = np.linspace(u.min(), u.max(), 100)
    ax.plot(uu, harness.a * uu + harness.b, "r--", lw=1.3,
            label=f"v = {harness.a:+.3f}·u {harness.b:+.3f}")
    ax.set_xlabel("u = 1/mass"); ax.set_ylabel("v = 1/S")
    ax.set_title("Algebraic Möbius fit")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (d) candidate scores
    ax = fig.add_subplot(gs[1, 0])
    if harness.candidates:
        sc = [c.score for c in harness.candidates]
        ch = [c.chirality for c in harness.candidates]
        ax.hist([s for s, c in zip(sc, ch) if c == 0],
                bins=30, color="#16a085", alpha=0.7, label="c=0")
        ax.hist([s for s, c in zip(sc, ch) if c == 1],
                bins=30, color="#e74c3c", alpha=0.7, label="c=1")
        ax.axvline(harness.acceptance_floor, color="k", ls=":",
                   lw=1.2, label=f"floor={harness.acceptance_floor}")
    ax.set_xlabel("score"); ax.set_ylabel("count")
    ax.set_title("Candidate scores (chirality split)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (e) residual vs Π
    ax = fig.add_subplot(gs[1, 1])
    if harness.candidates:
        ell = [c.ell_proj for c in harness.candidates]
        res = [c.residual for c in harness.candidates]
        acc = [c.accepted for c in harness.candidates]
        ax.scatter([e for e, a in zip(ell, acc) if not a],
                   [r for r, a in zip(res, acc) if not a],
                   c="gray", s=18, alpha=0.6, label="rejected")
        ax.scatter([e for e, a in zip(ell, acc) if a],
                   [r for r, a in zip(res, acc) if a],
                   c="#e74c3c", s=36, edgecolors="k",
                   linewidths=0.5, label="accepted")
    ax.set_xlabel("Π (elliptic projection)")
    ax.set_ylabel("|S_known − S_final|")
    ax.set_title("Elliptic projection vs residual")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (f) best candidates in the (mass, S) plane
    ax = fig.add_subplot(gs[1, 2], projection="3d")
    if harness.candidates:
        best = harness.best(20)
        ax.scatter([c.chemical.mass for c in best],
                   [c.chemical.supertrace for c in best],
                   [c.score for c in best],
                   c=[c.chirality for c in best],
                   cmap="coolwarm", s=50, edgecolors="k",
                   linewidths=0.5)
    ax.set_xlabel("mass"); ax.set_ylabel("S")
    ax.set_zlabel("score")
    ax.set_title("Top 20 accepted candidates")

    plt.suptitle(
        f"Elliptic chemical discovery  ·  α = {ALPHA:.4f}  ·  "
        f"periods (π, e)  ·  μ-gated asymmetric search",
        fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("Elliptic chemical discovery harness")
    print("=" * 78)
    print(f"  α = 1/(π − e)       = {ALPHA:.6f}")
    print(f"  ω₁ = π              = {W1:.6f}")
    print(f"  ω₂ = e              = {W2:.6f}")
    print(f"  6/π²                = {DENSITY:.6f}  (μ density)")
    print()

    harness = EllipticChemicalDiscovery(
        K_sieve=512, acceptance_floor=0.15,
        chirality_target=None)
    harness.load_known_elements()

    print(f"--- known chemicals (algebraic side) ---")
    print(f"  loaded        = {len(harness.known)}")
    print(f"  algebraic a   = {harness.a:+.6f}")
    print(f"  algebraic b   = {harness.b:+.6f}")
    chi0 = sum(1 for c in harness.known if c.chirality == 0)
    chi1 = sum(1 for c in harness.known if c.chirality == 1)
    print(f"  chirality c=0 : {chi0}    c=1 : {chi1}")
    print()

    print(f"--- known chemicals (sample) ---")
    print(f"  {'Z':>3s}  {'sym':>4s}  {'mass':>8s}  {'S':>6s}  "
          f"{'χ':>8s}  {'c':>2s}  {'Π':>6s}")
    for c in harness.known[:12]:
        print(f"  {c.Z:>3d}  {c.symbol:>4s}  {c.mass:>8.3f}  "
              f"{c.supertrace:>+6.1f}  {c.chi_angle:>+8.4f}  "
              f"{c.chirality:>2d}  {c.ell_proj:>6.4f}")
    print()

    print("--- asymmetric search over μ-square-free candidates ---")
    cands = harness.search(mass_range=(1.0, 90.0), max_candidates=400)
    print(f"  candidates searched : {len(cands)}")
    print(f"  accepted            : "
          f"{sum(1 for c in cands if c.accepted)}")
    print(f"  μ = +1              : "
          f"{sum(1 for c in cands if c.mu_k == 1)}")
    print(f"  μ = -1              : "
          f"{sum(1 for c in cands if c.mu_k == -1)}")
    print()

    print("--- top 10 candidates ---")
    print(f"  {'sym':>5s}  {'mass':>8s}  {'S_known':>8s}  "
          f"{'S_alg':>8s}  {'Δ':>8s}  {'S_final':>8s}  "
          f"{'Π':>6s}  {'c':>2s}  {'μ':>3s}  {'score':>6s}")
    for c in harness.best(10):
        print(f"  {c.chemical.symbol:>5s}  {c.chemical.mass:>8.3f}  "
              f"{c.chemical.supertrace:>+8.1f}  "
              f"{c.S_alg:>+8.4f}  {c.Delta:>+8.4f}  "
              f"{c.S_final:>+8.4f}  "
              f"{c.ell_proj:>6.4f}  {c.chirality:>2d}  "
              f"{c.mu_k:>+3d}  {c.score:>6.4f}")
    print()

    rep = harness.supertrace_report()
    print("--- residual supertrace / entropy / mass ---")
    print(f"  S (supertrace)      = {rep['S']:+.6e}")
    print(f"  H (entropy)         = {rep['H']:.6f}")
    print(f"  m (invariant mass)  = {rep['m']:.6e}")
    print(f"  n candidates        = {rep['n']}")
    print(f"  n accepted          = {rep['n_accepted']}")
    print()

    # chirality-restricted search
    print("--- chirality-restricted search (target c = 0) ---")
    h2 = EllipticChemicalDiscovery(
        K_sieve=256, acceptance_floor=0.15, chirality_target=0)
    h2.load_known_elements()
    h2.search(mass_range=(1.0, 60.0), max_candidates=200)
    best0 = h2.best(5, require_chirality=0)
    print(f"  accepted c=0        : "
          f"{sum(1 for c in h2.candidates if c.accepted)}")
    for c in best0:
        print(f"    {c.chemical.symbol:>6s}  mass={c.chemical.mass:>7.3f}  "
              f"score={c.score:.4f}  Π={c.ell_proj:.4f}")
    print()

    print("--- pipeline summary ---")
    print("  1. known chemicals    →  algebraic fit  v ≈ a·u + b")
    print("  2. shell population   →  6×6 spinor  Z_ij")
    print("  3. chirality          →  c = [cos(arg det Z) < 0]")
    print("  4. elliptic Π         →  ½(1+cos(2π x/π)cos(2π y/e))")
    print("  5. transcendental Δ   →  α·Im(q̄ p)/(1+|p|²)")
    print("  6. μ gate             →  square-free address only")
    print("  7. accept if          →  chirality-weighted score ≥ floor")
    print()

    visualize(harness)


if __name__ == "__main__":
    demo()