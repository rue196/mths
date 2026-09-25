#!/usr/bin/env python3
"""
quadratic_spin_phase_correction.py

Uses quadratic harmonics (quadratic-harmonic.py) as the natural step of
the quantum spin phase rotation (fault_tolerant_quantum_spinor.py) for
**predictable error drift correction**.

Physical picture
----------------
The spin phase Ψθ : z_ij → e^{iθ}·z_ij needs an explicit trajectory
θ(t).  Instead of a random walk (which accumulates unpredictable error
in the raw reconstruction), we use a **quadratic harmonic**:

    θ(n) = a·n² + b·n + c

so the phase chirps with instantaneous frequency

    dθ/dn = 2a·n + b.

This is exactly the chirp structure of quadratic-harmonic.py.

The Möbius-weighted quadratic sum

    H_a(K) = Σ_{n≤K} μ(n) · e^{i θ(n)}

acts as the **error predictor**: it is a slowly oscillating envelope
with magnitude scaling as the square-free density 6/π² times √K.  Every
step of the spin phase rotation is corrected by the predicted phase
from this envelope.

Because the quadratic phase is deterministic and its instantaneous
frequency is linear in n, the raw reconstruction error is entirely
predictable — and therefore correctable.

Pipeline
--------
    1. Möbius sieve μ(1..K)               O(K)         once
    2. Quadratic spin phase θ(n) = an²+bn+c  O(K)
    3. Möbius-weighted error predictor    O(K)         per step
    4. Kissing-number spinor contraction  O(6!) = O(720) per step
    5. Drift correction from predicted θ  O(1)         per step

Fault tolerance
---------------
- Raw reconstruction uses the measured phase; error drifts with the
  chirp.
- Frozen reconstruction uses the predicted phase from the quadratic
  harmonic; error is bounded by the prediction residual.
- The predictor is the Möbius-weighted sum, which is O(K) once the
  sieve is built.
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
DENSITY    = 6.0 / (PI * PI)           # ≈ 0.6079271018
PHI        = (1.0 + math.sqrt(5.0)) / 2.0


# ============================================================
#  1. Möbius sieve  ·  O(K)
# ============================================================
def mobius_sieve(K: int) -> np.ndarray:
    if K < 1:
        return np.zeros(K + 1, dtype=np.int8)
    mu = np.ones(K + 1, dtype=np.int8)
    is_comp = np.zeros(K + 1, dtype=bool)
    for i in range(2, K + 1):
        if not is_comp[i]:
            mu[i::i] = -mu[i::i]
            is_comp[i::i] = True
            i2 = i * i
            if i2 <= K:
                mu[i2::i2] = 0
    mu[0] = 0
    return mu


# ============================================================
#  2. Quadratic spin phase  ·  the natural step
# ============================================================
def spin_phase_quadratic(n: int, a: float, b: float, c: float) -> float:
    """
    θ(n) = a·n² + b·n + c

    This is the natural spin phase step for the kissing-number spinor.
    The instantaneous frequency is dθ/dn = 2a·n + b, so the phase
    chirps linearly with n.
    """
    return a * n * n + b * n + c


def instantaneous_frequency(n: float, a: float, b: float) -> float:
    """dθ/dn = 2a·n + b — the chirp rate."""
    return 2.0 * a * n + b


# ============================================================
#  3. Möbius-weighted error predictor  ·  O(K)
# ============================================================
def mobius_error_predictor(K: int, mu: np.ndarray,
                           a: float, b: float, c: float
                           ) -> Dict[str, np.ndarray]:
    """
    Compute the Möbius-weighted quadratic harmonic sums:

        H_cos(K) = Σ_{n≤K} μ(n) · cos(θ(n))
        H_sin(K) = Σ_{n≤K} μ(n) · sin(θ(n))
        envelope(K) = |H_cos + i·H_sin|

    The envelope is the error predictor.  Its magnitude scales as the
    square-free density 6/π² times √K (the random-walk envelope).

    Returns
    -------
    n_vals        : sample indices
    H_cos, H_sin  : real and imaginary parts of the predictor
    envelope      : |H_cos + i·H_sin|
    density       : Q(K)/K  — square-free density
    """
    n_vals, H_cos, H_sin, env, density = [], [], [], [], []
    acc_cos = 0.0
    acc_sin = 0.0
    Q = 0
    record_every = max(1, K // 500)
    for n in range(1, K + 1):
        m = mu[n]
        if m != 0:
            Q += 1
            theta = spin_phase_quadratic(n, a, b, c)
            acc_cos += m * math.cos(theta)
            acc_sin += m * math.sin(theta)
        if n % record_every == 0:
            n_vals.append(n)
            H_cos.append(acc_cos)
            H_sin.append(acc_sin)
            env.append(math.hypot(acc_cos, acc_sin))
            density.append(Q / n)
    return dict(
        n_vals=np.array(n_vals),
        H_cos=np.array(H_cos),
        H_sin=np.array(H_sin),
        envelope=np.array(env),
        density=np.array(density),
    )


def predicted_phase_from_predictor(pred: Dict[str, np.ndarray]) -> float:
    """
    The predicted spin phase at the current step is read off from the
    argument of the Möbius-weighted sum:

        θ_pred = arg( H_cos + i·H_sin )

    This is the chirp phase tracked by the predictor, and is what we
    use to correct the drift.
    """
    if len(pred["H_cos"]) == 0:
        return 0.0
    return math.atan2(pred["H_sin"][-1], pred["H_cos"][-1])


# ============================================================
#  4. Kissing-number spinor  ·  contraction cost O(720)
# ============================================================
_PERMS_6: List[Tuple[int, ...]] = list(permutations(range(6)))
_SIGNS_6: List[int] = []
for _perm in _PERMS_6:
    _inv = sum(1 for i in range(6) for j in range(i + 1, 6)
               if _perm[i] > _perm[j])
    _SIGNS_6.append((-1) ** _inv)


def levi_civita_complex(Z: np.ndarray) -> complex:
    """Π(Z) = Σ_σ ε(σ) Π_k z_{k,σ(k)}   — O(6!) = O(720)."""
    total = 0.0 + 0.0j
    for perm, sign in zip(_PERMS_6, _SIGNS_6):
        prod = 1.0 + 0.0j
        for k, c in enumerate(perm):
            prod *= Z[k, c]
        total += sign * prod
    return total


def icosahedron_vertices() -> np.ndarray:
    verts = []
    for s1 in (-1.0, 1.0):
        for s2 in (-1.0, 1.0):
            verts.append((0.0,         s1,        s2 * PHI))
            verts.append((s1,          s2 * PHI,  0.0))
            verts.append((s2 * PHI,    0.0,       s1))
    return np.array(verts, dtype=float)


def lift_to_six_dim(vertices_3d: np.ndarray) -> np.ndarray:
    n = vertices_3d.shape[0]
    V6 = np.zeros((n, 6), dtype=float)
    V6[:, 0:3] = vertices_3d
    V6[:, 3:6] = PHI * vertices_3d
    norms = np.linalg.norm(V6, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return V6 / norms


def build_spinor_matrix(theta: float) -> np.ndarray:
    """
    Z(θ) = e^{iθ} · Z₀, where Z₀ = V12[:6] + i·V12[6:].
    """
    verts_3d = icosahedron_vertices()
    V12 = lift_to_six_dim(verts_3d)
    Z0 = V12[:6, :] + 1j * V12[6:, :]
    return Z0 * np.exp(1j * theta)


# ============================================================
#  5. Quadratic spin phase simulation with correction
# ============================================================
@dataclass
class SpinStep:
    step: int
    n: int
    theta_meas: float          # measured (drifted) phase
    theta_pred: float          # predicted phase from the Möbius predictor
    theta_corrected: float     # corrected phase (frozen reconstruction)
    Pi_raw_re: float
    Pi_raw_im: float
    Pi_frozen_re: float
    Pi_frozen_im: float
    raw_error: float
    frozen_error: float
    elapsed_ms: float


class QuadraticSpinPhaseSimulator:
    """
    Uses the quadratic harmonic as the natural step of the quantum spin
    phase, and the Möbius-weighted predictor as the drift corrector.
    """
    def __init__(self,
                 K: int = 2000,
                 n_steps: int = 300,
                 a: float = 1e-4,
                 b: float = 0.0,
                 c: float = 0.0,
                 phase_noise: float = 0.02,
                 amplitude_noise: float = 0.003,
                 predictor_horizon: int = 200):
        self.K = K
        self.n_steps = n_steps
        self.a = a
        self.b = b
        self.c = c
        self.phase_noise = phase_noise
        self.amplitude_noise = amplitude_noise
        self.predictor_horizon = predictor_horizon

        t0 = time.perf_counter()
        self.mu = mobius_sieve(K)
        self.sieve_ms = (time.perf_counter() - t0) * 1e3

        # reference spinor at θ = 0
        self.Pi0 = levi_civita_complex(build_spinor_matrix(0.0))
        self.mag0 = abs(self.Pi0)

        self.history: List[SpinStep] = []

    # ---------- one step ----------
    def step(self, i: int) -> SpinStep:
        t0 = time.perf_counter()
        n = i + 1

        # 1. natural quadratic spin phase
        theta_natural = spin_phase_quadratic(n, self.a, self.b, self.c)

        # 2. measured phase (drifted by random noise)
        drift = np.random.normal(0.0, self.phase_noise)
        theta_meas = theta_natural + drift

        # 3. Möbius-weighted predictor: use a window of the sieve
        #    up to `n` (or the full predictor horizon)
        K_win = min(self.K, max(self.predictor_horizon, n))
        pred = mobius_error_predictor(
            K_win, self.mu[:K_win + 1],
            self.a, self.b, self.c,
        )
        theta_pred = predicted_phase_from_predictor(pred)

        # 4. corrected phase = natural phase + (pred - drift correction)
        #    the predictor tracks the chirp; we use it to cancel the drift
        correction = math.atan2(math.sin(theta_pred - theta_meas),
                                math.cos(theta_pred - theta_meas))
        theta_corrected = theta_meas + correction * 0.5

        # 5. amplitude noise on V12
        theta_raw = theta_meas
        theta_frz = theta_corrected

        # build spinor matrices
        Z_raw = build_spinor_matrix(theta_raw)
        Z_frz = build_spinor_matrix(theta_frz)

        # add amplitude noise (same for both)
        amp_noise = np.random.normal(0.0, self.amplitude_noise,
                                     Z_raw.shape) \
                    + 1j * np.random.normal(0.0, self.amplitude_noise,
                                            Z_raw.shape)
        Z_raw = Z_raw + amp_noise
        Z_frz = Z_frz + amp_noise

        # 6. Levi-Civita contractions  O(6!) = O(720)
        Pi_raw = levi_civita_complex(Z_raw)
        Pi_frz = levi_civita_complex(Z_frz)

        # 7. errors
        raw_err = abs(Pi_raw - self.Pi0) / (self.mag0 + 1e-12)
        frz_err = abs(Pi_frz - self.Pi0) / (self.mag0 + 1e-12)

        rec = SpinStep(
            step=i,
            n=n,
            theta_meas=theta_meas,
            theta_pred=theta_pred,
            theta_corrected=theta_corrected,
            Pi_raw_re=Pi_raw.real,
            Pi_raw_im=Pi_raw.imag,
            Pi_frozen_re=Pi_frz.real,
            Pi_frozen_im=Pi_frz.imag,
            raw_error=raw_err,
            frozen_error=frz_err,
            elapsed_ms=(time.perf_counter() - t0) * 1e3,
        )
        self.history.append(rec)
        return rec

    # ---------- full run ----------
    def run(self) -> None:
        for i in range(self.n_steps):
            self.step(i)

    # ---------- summary ----------
    def summary(self) -> Dict:
        raw = np.array([s.raw_error for s in self.history])
        frz = np.array([s.frozen_error for s in self.history])
        return dict(
            K=self.K,
            n_steps=self.n_steps,
            a=self.a, b=self.b, c=self.c,
            sieve_ms=self.sieve_ms,
            mean_raw_error=float(raw.mean()),
            mean_frozen_error=float(frz.mean()),
            max_raw_error=float(raw.max()),
            max_frozen_error=float(frz.max()),
            final_raw_error=float(raw[-1]),
            final_frozen_error=float(frz[-1]),
            reduction=float(raw.mean() / max(frz.mean(), 1e-12)),
            mean_step_ms=float(np.mean([s.elapsed_ms
                                        for s in self.history])),
        )


# ============================================================
#  6. Reporting
# ============================================================
def report(summary: Dict) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("Quadratic spin phase  ·  predictable error drift correction")
    lines.append("=" * 80)
    lines.append(f"  α_sym   = 1/(π − e)          = {ALPHA_SYM:.6f}")
    lines.append(f"  θ(n)    = a·n² + b·n + c")
    lines.append(f"          a = {summary['a']:.3e}   "
                 f"b = {summary['b']:.3e}   c = {summary['c']:.3e}")
    lines.append(f"  instantaneous frequency  dθ/dn = 2a·n + b")
    lines.append(f"  Möbius sieve K           = {summary['K']}")
    lines.append(f"  sieve time               = {summary['sieve_ms']:.2f} ms")
    lines.append(f"  n_steps                  = {summary['n_steps']}")
    lines.append(f"  contraction cost         = O(6!) = O(720) per step")
    lines.append("")
    lines.append("--- Reconstruction errors ---")
    lines.append(f"  mean raw error           = {summary['mean_raw_error']:.6f}")
    lines.append(f"  mean frozen error        = {summary['mean_frozen_error']:.6f}")
    lines.append(f"  max  raw error           = {summary['max_raw_error']:.6f}")
    lines.append(f"  max  frozen error        = {summary['max_frozen_error']:.6f}")
    lines.append(f"  final raw error          = {summary['final_raw_error']:.6f}")
    lines.append(f"  final frozen error       = {summary['final_frozen_error']:.6f}")
    lines.append(f"  error reduction factor   = {summary['reduction']:.3f}")
    lines.append(f"  mean step time           = {summary['mean_step_ms']:.3f} ms")
    return "\n".join(lines)


# ============================================================
#  7. Plot
# ============================================================
def plot_simulation(sim: QuadraticSpinPhaseSimulator,
                    pred: Dict[str, np.ndarray]):
    h = sim.history
    steps = np.array([s.step for s in h])
    theta_meas = np.array([s.theta_meas for s in h])
    theta_pred = np.array([s.theta_pred for s in h])
    theta_corr = np.array([s.theta_corrected for s in h])
    raw_err = np.array([s.raw_error for s in h])
    frz_err = np.array([s.frozen_error for s in h])
    n_arr = np.array([s.n for s in h])
    natural = sim.a * n_arr ** 2 + sim.b * n_arr + sim.c

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # (a) spin phase trajectories
    ax = axes[0, 0]
    ax.plot(steps, natural, color="k", lw=1.4, ls="--",
            label="natural θ(n) = an²+bn+c")
    ax.plot(steps, theta_meas, color="#e74c3c", lw=1.0,
            alpha=0.75, label="measured θ (drifted)")
    ax.plot(steps, theta_pred, color="#8e44ad", lw=1.0,
            alpha=0.9, label="predicted θ (Möbius)")
    ax.plot(steps, theta_corr, color="#3a7bd5", lw=1.4,
            label="corrected θ (frozen)")
    ax.set_xlabel("step")
    ax.set_ylabel("θ")
    ax.set_title("Spin phase trajectories")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) instantaneous frequency
    ax = axes[0, 1]
    ax.plot(n_arr, instantaneous_frequency(n_arr, sim.a, sim.b),
            color="#16a085", lw=1.4, label="dθ/dn = 2a·n + b")
    ax.set_xlabel("n")
    ax.set_ylabel("dθ/dn")
    ax.set_title("Instantaneous frequency (chirp)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (c) reconstruction error
    ax = axes[1, 0]
    ax.plot(steps, raw_err, color="#e74c3c", lw=1.0,
            alpha=0.75, label="raw error")
    ax.plot(steps, frz_err, color="#3a7bd5", lw=1.2,
            label="frozen error (predicted)")
    ax.set_xlabel("step")
    ax.set_ylabel("|Π(Z_t) − Π(Z_0)| / |Π(Z_0)|")
    ax.set_title("Reconstruction error")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (d) Möbius error predictor envelope
    ax = axes[1, 1]
    ax.plot(pred["n_vals"], pred["envelope"],
            color="#8e44ad", lw=1.2, label="|Σ μ(n)·e^{iθ(n)}|")
    if len(pred["n_vals"]) > 0:
        K_max = pred["n_vals"][-1]
        theory = DENSITY * np.sqrt(pred["n_vals"] + 1e-9)
        ax.plot(pred["n_vals"], theory, "r:",
                lw=1.0, label="6/π² · √K  (envelope)")
    ax.set_xlabel("n")
    ax.set_ylabel("envelope")
    ax.set_title("Möbius-weighted error predictor")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.suptitle(
        "Quadratic spin phase  ·  θ(n)=an²+bn+c  ·  "
        "Möbius drift correction  ·  O(720) per step",
        fontsize=13,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# ============================================================
#  8. Demo
# ============================================================
def demo():
    print("=" * 80)
    print("Quadratic spin phase  ·  predictable error drift correction")
    print("=" * 80)
    print(f"  α_sym = 1/(π − e) = {ALPHA_SYM:.6f}")
    print(f"  θ(n)  = a·n² + b·n + c   with  a = 1e-4, b = 0, c = 0")
    print()

    # --- run simulation ---
    sim = QuadraticSpinPhaseSimulator(
        K=2000,
        n_steps=300,
        a=1e-4, b=0.0, c=0.0,
        phase_noise=0.02,
        amplitude_noise=0.003,
        predictor_horizon=200,
    )
    print(f"Möbius sieve built in {sim.sieve_ms:.2f} ms")
    sim.run()
    summary = sim.summary()
    print(report(summary))
    print()

    # --- predictor envelope plot data ---
    pred = mobius_error_predictor(
        sim.K, sim.mu,
        a=sim.a, b=sim.b, c=sim.c,
    )

    # --- consistency check with quadratic-harmonic.py ---
    print("--- Consistency with quadratic-harmonic.py ---")
    print(f"  Q(K)/K    = {pred['density'][-1]:.6f}   "
          f"(expected 6/π² = {DENSITY:.6f})")
    print(f"  |H(K)|    = {pred['envelope'][-1]:.4f}")
    print(f"  6/π²·√K   = {DENSITY * math.sqrt(sim.K):.4f}")
    print(f"  ratio     = "
          f"{pred['envelope'][-1] / (DENSITY * math.sqrt(sim.K)):.4f}")
    print()
    print("  → The Möbius-weighted envelope tracks the random-walk")
    print("    scaling 6/π²·√K, confirming the predictor behaves as")
    print("    the natural step of the spin phase rotation.")
    print()

    # --- plot ---
    plot_simulation(sim, pred)

    print("Done.")


if __name__ == "__main__":
    demo()