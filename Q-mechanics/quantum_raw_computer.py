#!/usr/bin/env python3
"""
fault_tolerant_quantum_spinor.py

Fault-tolerant quantum simulation on the kissing-number spinor
embedding of the boson-fermion framework (boson-fermion-div.pdf).

Physical picture
----------------
    τ(3) = 12 kissing-number vertices   →  12 fermions in ℝ⁶
    each vertex is a spinor                z_ij = x̂_ij + i·ŷ_ij ∈ ℂ
    spin phase Ψθ : z_ij → e^{iθ} z_ij     →  logical encoding
    Levi-Civita contraction Π(z)           →  O(6!) = O(720) per step
    raw reconstruction                     →  phase-sensitive
    frozen reconstruction                  →  magnitude-only

The kissing number τ(3) = 12 is the icosahedron. Its 12 vertices are
lifted to ℝ⁶ and complexified into a 6×6 spinor matrix Z. The
Levi-Civita contraction

    Π(Z) = Σ_{σ ∈ S₆} ε(σ) · Π_k z_{k,σ(k)}

is the logical invariant, and the spin phase Ψθ acts on every entry:

    Π(Z · e^{iθ}) = e^{i·6θ} · Π(Z).

Therefore |Π(Z)| is invariant under Ψθ, while arg Π(Z) rotates by 6θ.

Fault tolerance
---------------
The **raw** reconstruction uses the full complex Π(Z), which drifts
with the spin phase and accumulates errors. The **frozen**
reconstruction uses only the magnitude |Π(Z)|, which is invariant
under Ψθ and therefore immune to phase scattering.

Every step costs O(6!) = O(720) for the Levi-Civita contraction —
this is the raw reconstruction cost of the projection operator.
"""

from __future__ import annotations

import math
import cmath
import time
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
from dataclasses import dataclass
from typing import List, Tuple, Dict


# ============================================================
#  Constants
# ============================================================
PI         = math.pi
E          = math.e
ALPHA_SYM  = 1.0 / (PI - E)            # ≈ 2.362
ALPHA_ASYM = 0.3628
PHI        = (1.0 + math.sqrt(5.0)) / 2.0


# ============================================================
#  1. Precompute the 720 permutations of S₆ and their signs
# ============================================================
_PERMS_6: List[Tuple[int, ...]] = list(permutations(range(6)))
_SIGNS_6: List[int] = []
for _perm in _PERMS_6:
    _inv = sum(1 for i in range(6) for j in range(i + 1, 6)
               if _perm[i] > _perm[j])
    _SIGNS_6.append((-1) ** _inv)


def levi_civita_complex(Z: np.ndarray) -> complex:
    """
    Π(Z) = Σ_σ ε(σ) · Π_k z_{k, σ(k)}     — O(6!) = O(720).

    Z is a 6×6 complex matrix of spinors.
    This is the raw reconstruction cost of the projection operator.
    """
    total = 0.0 + 0.0j
    for perm, sign in zip(_PERMS_6, _SIGNS_6):
        prod = 1.0 + 0.0j
        for k, c in enumerate(perm):
            prod *= Z[k, c]
        total += sign * prod
    return total


# ============================================================
#  2. Icosahedron  ·  τ(3) = 12 fermions in ℝ³ → ℝ⁶
# ============================================================
def icosahedron_vertices() -> np.ndarray:
    """The 12 icosahedron vertices in ℝ³ (golden-ratio embedding)."""
    verts = []
    for s1 in (-1.0, 1.0):
        for s2 in (-1.0, 1.0):
            verts.append((0.0,         s1,        s2 * PHI))
            verts.append((s1,          s2 * PHI,  0.0))
            verts.append((s2 * PHI,    0.0,       s1))
    return np.array(verts, dtype=float)


def lift_to_six_dim(vertices_3d: np.ndarray) -> np.ndarray:
    """
    Lift the 12 3D vertices to ℝ⁶ by golden-ratio duplication:

        V6[:, 0:3] = V3
        V6[:, 3:6] = φ · V3

    then normalise each row to unit length.
    """
    n = vertices_3d.shape[0]
    V6 = np.zeros((n, 6), dtype=float)
    V6[:, 0:3] = vertices_3d
    V6[:, 3:6] = PHI * vertices_3d
    norms = np.linalg.norm(V6, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return V6 / norms


# ============================================================
#  3. Spinor state  ·  6×6 complex Z from the 12 fermions
# ============================================================
class SpinorKissingState:
    """
    The 12 kissing-number vertices are 12 fermions in ℝ⁶.
    Each coordinate pair is complexified:

        z_ij = x_ij + i · y_ij                i, j = 1, …, 6

    The first 6 rows of V12 form the x-block, the last 6 rows form the
    y-block, and their combination gives the 6×6 complex spinor
    matrix Z.

    The spin phase Ψθ : Z → e^{iθ} Z is the logical encoding.
    """
    def __init__(self, theta: float = 0.0):
        self.vertices_3d = icosahedron_vertices()
        self.V12 = lift_to_six_dim(self.vertices_3d)          # 12 × 6
        self.Z0  = self.V12[:6, :] + 1j * self.V12[6:, :]     # 6 × 6 complex
        self.theta = theta
        self.Z = self.Z0 * np.exp(1j * theta)

    def apply_spin_phase(self, theta: float) -> None:
        """Ψθ : Z → e^{iθ} Z."""
        self.theta = theta
        self.Z = self.Z0 * np.exp(1j * theta)

    def contract(self) -> complex:
        """Π(Z) = Σ_σ ε(σ) Π_k z_{k,σ(k)}   — O(720)."""
        return levi_civita_complex(self.Z)

    def invariant_magnitude(self) -> float:
        """|Π(Z)| — invariant under Ψθ."""
        return abs(self.contract())

    def rotating_phase(self) -> float:
        """arg Π(Z) — rotates by 6θ under Ψθ."""
        return cmath.phase(self.contract())

    def kissing_number(self) -> int:
        """τ(3) = 12 for the icosahedron."""
        return self.V12.shape[0]


# ============================================================
#  4. Raw vs frozen reconstruction
# ============================================================
@dataclass
class ReconstructionStep:
    step: int
    theta: float
    Pi_re: float
    Pi_im: float
    Pi_mag: float
    Pi_phase: float
    raw_error: float
    frozen_error: float
    elapsed_ms: float


class RawFrozenSimulator:
    """
    Compare raw (phase-sensitive) and frozen (magnitude-only)
    reconstructions under boson-fermion scattering.

    Raw:    track full complex Π(Z(θ_t))       →  phase errors accumulate
    Frozen: track |Π(Z(θ_t))| = |Π(Z₀)|        →  phase errors cancelled

    Every step costs O(6!) = O(720) for the Levi-Civita contraction.
    """
    def __init__(self,
                 n_steps: int = 200,
                 dt: float = 0.1,
                 phase_noise: float = 0.15,
                 amplitude_noise: float = 0.003):
        self.n_steps = n_steps
        self.dt = dt
        self.phase_noise = phase_noise
        self.amplitude_noise = amplitude_noise

        # reference state
        self.state = SpinorKissingState(theta=0.0)
        self.Pi0 = self.state.contract()
        self.mag0 = abs(self.Pi0)
        self.phase0 = cmath.phase(self.Pi0)

        # history
        self.history: List[ReconstructionStep] = []
        self.raw_errors: List[float] = []
        self.frozen_errors: List[float] = []
        self.thetas: List[float] = []

    # ---------- one step ----------
    def step(self, i: int, freeze_phase: bool = False) -> ReconstructionStep:
        t0 = time.perf_counter()

        # 1. spin phase: drifts for raw, locked for frozen
        if freeze_phase:
            theta_new = 0.0
        else:
            theta_new = self.state.theta + np.random.normal(0.0, self.phase_noise) * self.dt

        # 2. amplitude noise on V12 (both regimes)
        amp_noise = np.random.normal(0.0, self.amplitude_noise,
                                     self.state.V12.shape)
        V12_noisy = self.state.V12 * (1.0 + amp_noise)
        Z_noisy = V12_noisy[:6, :] + 1j * V12_noisy[6:, :]

        # 3. apply the spin phase
        Z_rot = Z_noisy * np.exp(1j * theta_new)

        # 4. raw contraction — O(6!) = O(720)
        Pi_raw = levi_civita_complex(Z_rot)

        # 5. errors
        raw_err = abs(Pi_raw - self.Pi0) / (abs(self.Pi0) + 1e-12)
        mag = abs(Pi_raw)
        frozen_err = abs(mag - self.mag0) / (self.mag0 + 1e-12)

        # 6. record
        rec = ReconstructionStep(
            step=i,
            theta=theta_new,
            Pi_re=Pi_raw.real,
            Pi_im=Pi_raw.imag,
            Pi_mag=mag,
            Pi_phase=cmath.phase(Pi_raw),
            raw_error=raw_err,
            frozen_error=frozen_err,
            elapsed_ms=(time.perf_counter() - t0) * 1e3,
        )
        self.history.append(rec)
        self.raw_errors.append(raw_err)
        self.frozen_errors.append(frozen_err)
        self.thetas.append(theta_new)

        # advance the raw phase state
        if not freeze_phase:
            self.state.apply_spin_phase(theta_new)

        return rec

    # ---------- full runs ----------
    def run_raw(self) -> None:
        self.state = SpinorKissingState(theta=0.0)
        self.history.clear()
        self.raw_errors.clear()
        self.frozen_errors.clear()
        self.thetas.clear()
        for i in range(self.n_steps):
            self.step(i, freeze_phase=False)

    def run_frozen(self) -> None:
        self.state = SpinorKissingState(theta=0.0)
        self.history.clear()
        self.raw_errors.clear()
        self.frozen_errors.clear()
        self.thetas.clear()
        for i in range(self.n_steps):
            self.step(i, freeze_phase=True)

    # ---------- summary ----------
    def summary(self) -> Dict:
        raw = np.array(self.raw_errors)
        frz = np.array(self.frozen_errors)
        return dict(
            n_steps=self.n_steps,
            mean_raw_error=float(raw.mean()),
            mean_frozen_error=float(frz.mean()),
            max_raw_error=float(raw.max()),
            max_frozen_error=float(frz.max()),
            final_raw_error=float(raw[-1]),
            final_frozen_error=float(frz[-1]),
            reduction=float(raw.mean() / max(frz.mean(), 1e-12)),
            total_time_ms=float(sum(s.elapsed_ms for s in self.history)),
        )


# ============================================================
#  5. Benchmark
# ============================================================
def run_benchmark(n_steps: int = 200,
                  phase_noise: float = 0.15,
                  amplitude_noise: float = 0.003) -> Dict:
    np.random.seed(42)
    sim_raw = RawFrozenSimulator(n_steps=n_steps,
                                 phase_noise=phase_noise,
                                 amplitude_noise=amplitude_noise)
    sim_raw.run_raw()
    raw_summary = sim_raw.summary()

    np.random.seed(1234)
    sim_frz = RawFrozenSimulator(n_steps=n_steps,
                                 phase_noise=phase_noise,
                                 amplitude_noise=amplitude_noise)
    sim_frz.run_frozen()
    frozen_summary = sim_frz.summary()

    return dict(
        raw_summary=raw_summary,
        frozen_summary=frozen_summary,
        raw_errors=list(sim_raw.raw_errors),
        frozen_errors=list(sim_frz.frozen_errors),
        raw_thetas=list(sim_raw.thetas),
        raw_mag=[s.Pi_mag for s in sim_raw.history],
        frozen_mag=[s.Pi_mag for s in sim_frz.history],
        raw_phase=[s.Pi_phase for s in sim_raw.history],
        frozen_phase=[s.Pi_phase for s in sim_frz.history],
    )


# ============================================================
#  6. Reporting
# ============================================================
def report_benchmark(bench: Dict) -> str:
    lines = []
    lines.append("=" * 82)
    lines.append("Fault-tolerant quantum simulation  ·  kissing-number spinor")
    lines.append("=" * 82)
    lines.append(f"  α_sym        = 1/(π − e) = {ALPHA_SYM:.6f}")
    lines.append(f"  τ(3)                      = 12 fermions (icosahedron)")
    lines.append(f"  spinors z_ij = x_ij + i·y_ij,  i,j = 1,…,6")
    lines.append(f"  spin phase  Ψθ : z → e^{{iθ}} z")
    lines.append(f"  contraction cost           = O(6!) = O(720) per step")
    lines.append("")

    lines.append("--- Raw reconstruction (phase-sensitive) ---")
    for k, v in bench["raw_summary"].items():
        if isinstance(v, float):
            lines.append(f"  {k:>22s}: {v:.6f}")
        else:
            lines.append(f"  {k:>22s}: {v}")
    lines.append("")

    lines.append("--- Frozen reconstruction (magnitude-only) ---")
    for k, v in bench["frozen_summary"].items():
        if isinstance(v, float):
            lines.append(f"  {k:>22s}: {v:.6f}")
        else:
            lines.append(f"  {k:>22s}: {v}")
    lines.append("")

    red = bench["raw_summary"]["mean_raw_error"] / max(
        bench["frozen_summary"]["mean_frozen_error"], 1e-12)
    lines.append("--- Improvement from the frozen projection ---")
    lines.append(f"  mean error reduction factor : {red:.3f}")
    lines.append(f"  max  raw  /  frozen         : "
                 f"{bench['raw_summary']['max_raw_error']:.6f}  /  "
                 f"{bench['frozen_summary']['max_frozen_error']:.6f}")
    lines.append(f"  final raw / frozen          : "
                 f"{bench['raw_summary']['final_raw_error']:.6f}  /  "
                 f"{bench['frozen_summary']['final_frozen_error']:.6f}")
    return "\n".join(lines)


# ============================================================
#  7. Plot
# ============================================================
def plot_benchmark(bench: Dict):
    steps = np.arange(len(bench["raw_errors"]))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # (a) reconstruction error
    ax = axes[0, 0]
    ax.plot(steps, bench["raw_errors"], color="#e74c3c",
            lw=1.0, alpha=0.75, label="raw error (phase drifts)")
    ax.plot(steps, bench["frozen_errors"], color="#3a7bd5",
            lw=1.0, alpha=0.75, label="frozen error (|Π| only)")
    ax.set_xlabel("step")
    ax.set_ylabel("|Π(Z_t) − Π(Z_0)| / |Π(Z_0)|")
    ax.set_title("Reconstruction error")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (b) spin phase drift
    ax = axes[0, 1]
    ax.plot(steps, bench["raw_thetas"], color="#8e44ad",
            lw=1.0, label="θ_t  (raw)")
    ax.axhline(0.0, color="#3a7bd5", lw=1.4, ls="--",
               label="θ = 0  (frozen)")
    ax.set_xlabel("step")
    ax.set_ylabel("spin phase θ")
    ax.set_title("Spin phase drift  Ψθ : z → e^{iθ} z")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (c) magnitude |Π|
    ax = axes[1, 0]
    ax.plot(steps, bench["raw_mag"], color="#e74c3c",
            lw=1.0, alpha=0.75, label="raw |Π|")
    ax.plot(steps, bench["frozen_mag"], color="#3a7bd5",
            lw=1.2, label="frozen |Π|")
    ax.set_xlabel("step")
    ax.set_ylabel("|Π(Z_t)|")
    ax.set_title("Contraction magnitude — invariant under Ψθ")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (d) cumulative error
    ax = axes[1, 1]
    ax.plot(steps, np.cumsum(bench["raw_errors"]),
            color="#e74c3c", lw=1.2, label="raw cumulative")
    ax.plot(steps, np.cumsum(bench["frozen_errors"]),
            color="#3a7bd5", lw=1.2, label="frozen cumulative")
    ax.set_xlabel("step")
    ax.set_ylabel("cumulative error")
    ax.set_title("Error accumulation")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.suptitle(
        "Kissing-number spinor  ·  τ(3) = 12 fermions in ℝ⁶  ·  "
        "O(720) contraction per step",
        fontsize=13,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# ============================================================
#  8. Demo
# ============================================================
def demo():
    print("=" * 82)
    print("Fault-tolerant quantum simulation  ·  kissing-number spinor")
    print("=" * 82)
    print(f"  α_sym  = 1/(π − e) = {ALPHA_SYM:.6f}")
    print(f"  φ      = (1+√5)/2  = {PHI:.6f}")
    print()

    # --- spinor setup ---
    state = SpinorKissingState(theta=0.0)
    print("--- Spinor configuration ---")
    print(f"  12 icosahedron vertices        → τ(3) = 12 fermions")
    print(f"  lifted to ℝ⁶                   → V12 shape {state.V12.shape}")
    print(f"  complexified to spinors         → Z shape {state.Z0.shape}")
    print(f"  z_ij = x_ij + i·y_ij              i, j = 1,…,6")
    print()
    print("  first row of Z (6 spinors):")
    for j, z in enumerate(state.Z0[0, :]):
        print(f"    z_0{j} = {z.real:+.6f} {z.imag:+.6f}i")
    print()

    # --- spin phase action ---
    Pi0 = state.contract()
    print("--- Spin phase action Ψθ ---")
    print(f"  Π(Z_0)            = {Pi0.real:+.6f} {Pi0.imag:+.6f}i")
    print(f"  |Π(Z_0)|          = {abs(Pi0):.6f}")
    print()
    for theta in [0.1, 0.5, 1.0, 2.0]:
        state.apply_spin_phase(theta)
        Pi = state.contract()
        mag = abs(Pi)
        print(f"  Π(Z_0 · e^{{i·{theta:.1f}}})  = "
              f"{Pi.real:+.6f} {Pi.imag:+.6f}i   "
              f"|Π| = {mag:.6f}   "
              f"(ratio to |Π_0| = {mag/abs(Pi0):.6f})")
    print()
    print("  → |Π(Z)| is invariant, the phase rotates by 6θ.")
    print("  → Frozen reconstruction uses |Π| only; raw uses full Π.")
    print()

    # --- benchmark ---
    bench = run_benchmark(n_steps=200,
                          phase_noise=0.15,
                          amplitude_noise=0.003)
    print(report_benchmark(bench))

    # --- plot ---
    plot_benchmark(bench)

    print()
    print("Done.")


if __name__ == "__main__":
    demo()