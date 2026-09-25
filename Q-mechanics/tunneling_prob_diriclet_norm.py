#!/usr/bin/env python3
"""
quadratic_harmonic_computer_simulation.py
=========================================

Computer simulation of the quadratic harmonic spin phase with
NORM-based tunneling, merging

  • quadratic_harmonic_quantum_computer_mobius.py
        - quadratic spin phase  θ(n) = a·n² + b·n + c
        - Möbius-weighted error predictor  Σ_{n≤K} μ(n) e^{iθ(n)}
        - kissing-number spinor + Levi-Civita contraction (O(6!)=O(720))

  • convolution-dirichlet-harmonic.py
        - exponential convolution normalization
              ALPHA = 1/(π − e)
              NORM  = 1 − exp(−ALPHA·(π + e))
        - this NORM is reused here as the tunneling normalization

Tunneling model
---------------
In the convolution pipeline the chip response is

    conv[i] = (1 − conv_exp[i]) / NORM

so NORM is the natural unit of the transmission.  Reading this as a
barrier traversal:

    NORM        = confinement probability      (amplitude √NORM)
    1 − NORM    = tunneling probability        (amplitude √(1−NORM))
    ε_tunnel    = exp(−ALPHA·(π + e))          = 1 − NORM

The effective spinor at each step is the coherent superposition

    Z_eff(θ) = √NORM · Z(θ) + √(1 − NORM) · e^{iπ/2} · Z(θ)

of the classically confined channel and a π/2-phase-shifted tunneled
channel.

Three reconstructions are tracked:

    raw        Z(θ_meas)                        (no tunneling)
    tunneled   Z_eff(θ_meas)                    (tunneling only)
    corrected  Z_eff(θ_corr)                    (tunneling + Möbius predictor)

where θ_corr = θ_meas + ½·atan2( sin(θ_pred − θ_meas),
                                 cos(θ_pred − θ_meas) )
and θ_pred = arg Σ_{n≤K} μ(n) e^{iθ(n)}.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
#  Spinor / predictor machinery
# ------------------------------------------------------------------
from quadratic_harmonic_quantum_computer_mobius import (
    PI, E,
    ALPHA_SYM, DENSITY, PHI,
    mobius_sieve,
    spin_phase_quadratic,
    instantaneous_frequency,
    mobius_error_predictor,
    predicted_phase_from_predictor,
    levi_civita_complex,
    build_spinor_matrix,
)

# ------------------------------------------------------------------
#  Convolution constants (ALPHA, NORM)
#  filename contains hyphens → load via importlib
# ------------------------------------------------------------------
_cdh_path = pathlib.Path(__file__).with_name("convolution-dirichlet-harmonic.py")
if _cdh_path.exists():
    _spec = importlib.util.spec_from_file_location(
        "convolution_dirichlet_harmonic", str(_cdh_path))
    _cdh = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_cdh)
    ALPHA = float(_cdh.ALPHA)
    NORM = float(_cdh.NORM)
else:
    # fall back to explicit formulas
    ALPHA = 1.0 / (PI - E)
    NORM = 1.0 - math.exp(-ALPHA * (PI + E))

# ------------------------------------------------------------------
#  Tunneling constants derived from NORM
# ------------------------------------------------------------------
TUNNEL_PROB  = 1.0 - NORM               # ≈ 9.73e-7  (leakage)
TUNNEL_AMP   = math.sqrt(TUNNEL_PROB)   # ≈ 9.86e-4
CONFINE_PROB = NORM                     # ≈ 0.99999903
CONFINE_AMP  = math.sqrt(CONFINE_PROB)  # ≈ 0.99999951
TUNNEL_PHASE = PI / 2.0                 # tunneled-channel phase shift

# ------------------------------------------------------------------
@dataclass
class TunnelingStep:
    step: int
    n: int
    theta_meas: float
    theta_pred: float
    theta_corrected: float
    Pi_raw_re: float
    Pi_raw_im: float
    Pi_tunneled_re: float
    Pi_tunneled_im: float
    Pi_corrected_re: float
    Pi_corrected_im: float
    raw_error: float
    tunneled_error: float
    corrected_error: float
    tunnel_weight: float
    elapsed_ms: float


# ==================================================================
class QuadraticHarmonicTunnelingSimulator:
    """
    Quadratic harmonic spin phase with NORM-derived tunneling.

    The convolution normalization NORM is the tunneling normalization:

        NORM        - confinement probability
        1 - NORM    - tunneling probability

    The spinor is a coherent superposition of the confined channel
    and a π/2-phase-shifted tunneled channel.
    """

    def __init__(self,
                 K: int = 2000,
                 n_steps: int = 300,
                 a: float = 1e-4,
                 b: float = 0.0,
                 c: float = 0.0,
                 phase_noise: float = 0.02,
                 amplitude_noise: float = 0.003,
                 predictor_horizon: int = 200,
                 tunnel_coupling: float = 1.0):
        self.K = int(K)
        self.n_steps = int(n_steps)
        self.a, self.b, self.c = float(a), float(b), float(c)
        self.phase_noise = float(phase_noise)
        self.amplitude_noise = float(amplitude_noise)
        self.predictor_horizon = int(predictor_horizon)
        # coupling > 1 amplifies tunneling for visualization;
        # coupling = 1 uses the physical NORM-derived value.
        self.tunnel_coupling = float(tunnel_coupling)

        t0 = time.perf_counter()
        self.mu = mobius_sieve(self.K)
        self.sieve_ms = (time.perf_counter() - t0) * 1e3

        self.Pi0 = levi_civita_complex(build_spinor_matrix(0.0))
        self.mag0 = abs(self.Pi0)

        # --- NORM-derived tunneling amplitudes -------------------
        tc = self.tunnel_coupling
        t_amp = TUNNEL_AMP * tc
        c_amp = math.sqrt(max(0.0, 1.0 - t_amp * t_amp))
        norm = math.hypot(t_amp, c_amp)
        if norm > 0.0:
            t_amp /= norm
            c_amp /= norm
        self.tunnel_amp = t_amp
        self.confine_amp = c_amp

        self.history: List[TunnelingStep] = []

    # --------------------------------------------------------------
    def _spinor_with_tunneling(self, theta: float) -> np.ndarray:
        """
        Z_eff(θ) = c_conf · Z(θ) + c_tun · e^{iπ/2} · Z(θ)

        The tunneling channel carries a π/2 phase shift relative to
        the confined channel (typical of barrier traversal).
        """
        Z_conf = build_spinor_matrix(theta)
        Z_tun = build_spinor_matrix(theta)
        return self.confine_amp * Z_conf + self.tunnel_amp * 1j * Z_tun

    # --------------------------------------------------------------
    def step(self, i: int) -> TunnelingStep:
        t0 = time.perf_counter()
        n = i + 1

        # 1. natural quadratic spin phase
        theta_natural = spin_phase_quadratic(n, self.a, self.b, self.c)

        # 2. measured phase (drifted)
        drift = np.random.normal(0.0, self.phase_noise)
        theta_meas = theta_natural + drift

        # 3. Möbius-weighted predictor over a window up to n
        K_win = min(self.K, max(self.predictor_horizon, n))
        pred = mobius_error_predictor(
            K_win, self.mu[:K_win + 1],
            self.a, self.b, self.c,
        )
        theta_pred = predicted_phase_from_predictor(pred)

        # 4. phase correction (half of the wrapped residual)
        delta = math.atan2(math.sin(theta_pred - theta_meas),
                           math.cos(theta_pred - theta_meas))
        theta_corr = theta_meas + 0.5 * delta

        # 5. amplitude noise, shared by all channels
        amp = (np.random.normal(0.0, self.amplitude_noise, (6, 6))
               + 1j * np.random.normal(0.0, self.amplitude_noise, (6, 6)))

        # 6. three reconstructions
        #   (a) raw       : no tunneling
        Z_raw = build_spinor_matrix(theta_meas) + amp
        Pi_raw = levi_civita_complex(Z_raw)

        #   (b) tunneled  : NORM superposition at θ_meas
        Z_tun = self._spinor_with_tunneling(theta_meas) + amp
        Pi_tun = levi_civita_complex(Z_tun)

        #   (c) corrected : NORM superposition at θ_corr
        Z_corr = self._spinor_with_tunneling(theta_corr) + amp
        Pi_corr = levi_civita_complex(Z_corr)

        # 7. relative errors vs reference Π(Z0)
        inv = 1.0 / (self.mag0 + 1e-12)
        raw_err = abs(Pi_raw - self.Pi0) * inv
        tun_err = abs(Pi_tun - self.Pi0) * inv
        corr_err = abs(Pi_corr - self.Pi0) * inv

        rec = TunnelingStep(
            step=i, n=n,
            theta_meas=theta_meas,
            theta_pred=theta_pred,
            theta_corrected=theta_corr,
            Pi_raw_re=Pi_raw.real, Pi_raw_im=Pi_raw.imag,
            Pi_tunneled_re=Pi_tun.real, Pi_tunneled_im=Pi_tun.imag,
            Pi_corrected_re=Pi_corr.real, Pi_corrected_im=Pi_corr.imag,
            raw_error=raw_err,
            tunneled_error=tun_err,
            corrected_error=corr_err,
            tunnel_weight=self.tunnel_amp ** 2,
            elapsed_ms=(time.perf_counter() - t0) * 1e3,
        )
        self.history.append(rec)
        return rec

    # --------------------------------------------------------------
    def run(self) -> None:
        for i in range(self.n_steps):
            self.step(i)

    # --------------------------------------------------------------
    def summary(self) -> Dict:
        raw = np.array([s.raw_error for s in self.history])
        tun = np.array([s.tunneled_error for s in self.history])
        cor = np.array([s.corrected_error for s in self.history])
        return dict(
            K=self.K, n_steps=self.n_steps,
            a=self.a, b=self.b, c=self.c,
            sieve_ms=self.sieve_ms,
            ALPHA=ALPHA, NORM=NORM,
            tunnel_prob=TUNNEL_PROB,
            tunnel_amp_phys=TUNNEL_AMP,
            confine_prob=CONFINE_PROB,
            confine_amp_phys=CONFINE_AMP,
            tunnel_coupling=self.tunnel_coupling,
            tunnel_amp_used=self.tunnel_amp,
            confine_amp_used=self.confine_amp,
            mean_raw_error=float(raw.mean()),
            mean_tunneled_error=float(tun.mean()),
            mean_corrected_error=float(cor.mean()),
            max_raw_error=float(raw.max()),
            max_tunneled_error=float(tun.max()),
            max_corrected_error=float(cor.max()),
            final_corrected_error=float(cor[-1]),
            tunneling_gain=float(raw.mean() / max(tun.mean(), 1e-12)),
            correction_gain=float(tun.mean() / max(cor.mean(), 1e-12)),
            mean_step_ms=float(np.mean([s.elapsed_ms for s in self.history])),
        )


# ==================================================================
#  Reporting
# ==================================================================
def report(summary: Dict) -> str:
    L = []
    L.append("=" * 80)
    L.append("Quadratic harmonic spin phase  ·  NORM-based tunneling")
    L.append("=" * 80)
    L.append(f"  α          = 1/(π − e)            = {summary['ALPHA']:.6f}")
    L.append(f"  NORM       = 1 − e^(−α(π+e))      = {summary['NORM']:.10f}")
    L.append(f"  τ_tunnel   = 1 − NORM             = {summary['tunnel_prob']:.6e}")
    L.append(f"  A_tunnel   = √(1 − NORM)          = {summary['tunnel_amp_phys']:.6e}")
    L.append(f"  A_confine  = √NORM                = {summary['confine_amp_phys']:.10f}")
    L.append(f"  tunnel coupling                   = {summary['tunnel_coupling']:.3f}")
    L.append(f"  A_tunnel (used)                   = {summary['tunnel_amp_used']:.6e}")
    L.append(f"  A_confine (used)                  = {summary['confine_amp_used']:.6e}")
    L.append("")
    L.append(f"  Möbius sieve K                    = {summary['K']}")
    L.append(f"  sieve time                        = {summary['sieve_ms']:.2f} ms")
    L.append(f"  n_steps                           = {summary['n_steps']}")
    L.append(f"  θ(n) = a n² + b n + c   "
             f"a={summary['a']:.3e}  b={summary['b']:.3e}  c={summary['c']:.3e}")
    L.append(f"  contraction cost                  = O(6!) = O(720) per step")
    L.append("")
    L.append("--- Reconstruction errors  |Π(Z) − Π(Z0)| / |Π(Z0)| ---")
    L.append(f"  mean raw       (no tunneling)     = {summary['mean_raw_error']:.6f}")
    L.append(f"  mean tunneled  (NORM superpos.)   = {summary['mean_tunneled_error']:.6f}")
    L.append(f"  mean corrected (NORM + predictor) = {summary['mean_corrected_error']:.6f}")
    L.append(f"  max  raw                          = {summary['max_raw_error']:.6f}")
    L.append(f"  max  tunneled                     = {summary['max_tunneled_error']:.6f}")
    L.append(f"  max  corrected                    = {summary['max_corrected_error']:.6f}")
    L.append(f"  final corrected                   = {summary['final_corrected_error']:.6f}")
    L.append(f"  tunneling gain  (raw / tunneled)  = {summary['tunneling_gain']:.4f}")
    L.append(f"  correction gain (tunneled / corr) = {summary['correction_gain']:.4f}")
    L.append(f"  mean step time                    = {summary['mean_step_ms']:.3f} ms")
    return "\n".join(L)


# ==================================================================
#  Plot
# ==================================================================
def plot_simulation(sim: QuadraticHarmonicTunnelingSimulator,
                    pred: Dict[str, np.ndarray]) -> None:
    h = sim.history
    steps = np.array([s.step for s in h])
    theta_meas = np.array([s.theta_meas for s in h])
    theta_pred = np.array([s.theta_pred for s in h])
    theta_corr = np.array([s.theta_corrected for s in h])
    raw_err = np.array([s.raw_error for s in h])
    tun_err = np.array([s.tunneled_error for s in h])
    cor_err = np.array([s.corrected_error for s in h])
    n_arr = np.array([s.n for s in h])
    natural = sim.a * n_arr ** 2 + sim.b * n_arr + sim.c

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # (a) spin-phase trajectories
    ax = axes[0, 0]
    ax.plot(steps, natural, "k--", lw=1.4, label="natural θ(n)")
    ax.plot(steps, theta_meas, color="#e74c3c", lw=1.0, alpha=0.8,
            label="measured θ (drifted)")
    ax.plot(steps, theta_pred, color="#8e44ad", lw=1.0, alpha=0.9,
            label="predicted θ (Möbius)")
    ax.plot(steps, theta_corr, color="#3a7bd5", lw=1.4,
            label="corrected θ")
    ax.set_xlabel("step"); ax.set_ylabel("θ")
    ax.set_title("Spin-phase trajectories")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (b) NORM tunneling weights
    ax = axes[0, 1]
    ax.plot(steps, np.full_like(steps, NORM, dtype=float),
            color="#16a085", lw=1.6,
            label=f"NORM = {NORM:.8f}  (confinement)")
    ax.plot(steps, np.full_like(steps, 1.0 - NORM, dtype=float),
            color="#c0392b", lw=1.6,
            label=f"1 − NORM = {1-NORM:.3e}  (tunneling)")
    ax.plot(steps, np.full_like(steps, sim.tunnel_amp ** 2, dtype=float),
            color="#8e44ad", lw=1.0, ls=":",
            label=f"A_tun² (used) = {sim.tunnel_amp**2:.3e}")
    ax.set_yscale("log")
    ax.set_xlabel("step"); ax.set_ylabel("probability")
    ax.set_title("NORM-derived tunneling weights")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    # (c) reconstruction errors
    ax = axes[1, 0]
    ax.plot(steps, raw_err, color="#e74c3c", lw=1.0, alpha=0.8,
            label="raw (no tunneling)")
    ax.plot(steps, tun_err, color="#8e44ad", lw=1.1,
            label="tunneled (NORM)")
    ax.plot(steps, cor_err, color="#3a7bd5", lw=1.4,
            label="corrected (NORM + predictor)")
    ax.set_xlabel("step")
    ax.set_ylabel("|Π(Z) − Π(Z0)| / |Π(Z0)|")
    ax.set_title("Reconstruction error")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (d) Möbius predictor envelope
    ax = axes[1, 1]
    ax.plot(pred["n_vals"], pred["envelope"],
            color="#8e44ad", lw=1.2,
            label="|Σ μ(n) e^{iθ(n)}|")
    if len(pred["n_vals"]) > 0:
        theory = DENSITY * np.sqrt(pred["n_vals"] + 1e-9)
        ax.plot(pred["n_vals"], theory, "r:", lw=1.0,
                label="6/π² · √K  (random-walk)")
    ax.set_xlabel("n"); ax.set_ylabel("envelope")
    ax.set_title("Möbius-weighted error predictor")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.suptitle(
        "Quadratic harmonic spin phase  ·  "
        "NORM = 1 − exp(−α(π+e)) as tunneling  ·  O(720) per step",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# ==================================================================
#  Demo
# ==================================================================
def demo():
    print("=" * 80)
    print("Quadratic harmonic computer simulation  ·  NORM-based tunneling")
    print("=" * 80)
    print(f"  α     = 1/(π − e)        = {ALPHA:.6f}")
    print(f"  NORM  = 1 − e^(−α(π+e))  = {NORM:.10f}")
    print(f"  τ_tun = 1 − NORM         = {TUNNEL_PROB:.6e}")
    print(f"  A_tun = √(1 − NORM)      = {TUNNEL_AMP:.6e}")
    print()

    # Physical tunneling: coupling = 1.0.  Raise the coupling (e.g. 1000)
    # to amplify the tunneling contribution for visualization.
    sim = QuadraticHarmonicTunnelingSimulator(
        K=2000,
        n_steps=300,
        a=1e-4, b=0.0, c=0.0,
        phase_noise=0.02,
        amplitude_noise=0.003,
        predictor_horizon=200,
        tunnel_coupling=1.0,
    )
    print(f"Möbius sieve built in {sim.sieve_ms:.2f} ms")
    sim.run()
    summary = sim.summary()
    print(report(summary))
    print()

    # predictor envelope (for the (d) subplot)
    pred = mobius_error_predictor(
        sim.K, sim.mu, a=sim.a, b=sim.b, c=sim.c)

    # consistency check with quadratic-harmonic.py
    print("--- Consistency with quadratic-harmonic.py ---")
    print(f"  Q(K)/K    = {pred['density'][-1]:.6f}   "
          f"(expected 6/π² = {DENSITY:.6f})")
    print(f"  |H(K)|    = {pred['envelope'][-1]:.4f}")
    print(f"  6/π²·√K   = {DENSITY * math.sqrt(sim.K):.4f}")
    print(f"  ratio     = "
          f"{pred['envelope'][-1] / (DENSITY * math.sqrt(sim.K)):.4f}")
    print()
    print("  → The convolution NORM enters the spinor as a tiny")
    print("    confinement correction (A_tun = √(1−NORM) ≈ 1e−3), so the")
    print("    tunneling channel is a small, controlled perturbation of")
    print("    the raw reconstruction.  Increasing tunnel_coupling")
    print("    amplifies the effect for visualization.")
    print()

    plot_simulation(sim, pred)
    print("Done.")


if __name__ == "__main__":
    demo()