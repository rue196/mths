#!/usr/bin/env python3
"""
transformer_throughput.py
=========================

Transformer throughput driven by the elliptic Möbius gate.

Signal chain
------------
    |ζ(t)|²                 amplitude envelope      primary winding drive
    L(t) = ∫ √(dRe²+dIm²)   arc length of ζ          →  turns ratio
    ρ_elec(t)               point-particle density   →  coupling scalar
    E(t) = (S_νμ+S_ντ)cos(ωt)  AC carrier            →  alternating drive
    P(f)                    spectral density         →  |H(f)|² transfer

Voltage model
-------------
    volt_scalar(t)   =  (1 + k_turns · L(t)) · ρ_elec(t)/max(ρ_elec)
                        ↑ turns from arc length        ↑ coupling from
                                                          point particles

    V(t)  =  V_ref · envelope(t) · volt_scalar(t) · cos(ω t)
    I(t)  =  V(t) / R_load
    P(t)  =  V(t) · I(t)  =  V(t)² / R_load

Frequency model
---------------
    |H(f)|²  =  P_spec(f) / max(P_spec)          from est.spec_P
    P_out(f) =  P_in · |H(f)|²

Everything (|ζ|², L(t), ρ_elec, E(t), P(f)) is taken verbatim from the
SpectralDensityMobius estimator — this module adds only the electric
layer on top.

Imports (unchanged)
-------------------
    from AC_DC_elliptic_mobius_tau_muon import SpectralDensityMobius, ...
    from wire_analog import alternating_E_field, pi_e, spin_grade
    from DC_chirp import DCMobiusGate, mobius_sieve
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ---- engine: SpectralDensityMobius already wires DC_chirp + wire_analog ----
from AC_DC_elliptic_mobius_tau_muon import (
    SpectralDensityMobius,
    PI, E, ALPHA, A_STEP, DENSITY,
)

# ---- reference primitives (used for verification, not pipeline) ----
from wire_analog import (
    alternating_E_field,
    pi_e,
    spin_grade,
    AnalogWire,
)
from DC_chirp import (
    DCMobiusGate,
    mobius_sieve,
    even_even_mask,
    coupling_matrix_1d,
    propagate_signal,
)


# ============================================================
#  TransformerThroughput
# ============================================================
@dataclass
class TransformerThroughput:
    """
    Electric layer on top of SpectralDensityMobius.

    Parameters
    ----------
    K, J_C, omega, N_samples, smooth  →  forwarded to SpectralDensityMobius
    V_ref      peak reference voltage [V] at envelope=1, scalar=1
    R_load     resistive load [Ω]
    k_turns    turns sensitivity to arc length L(t)  (turns = 1 + k_turns·L)
    """

    K:         int   = 20
    J_C:       float = 1.0
    omega:     float = 1.0
    N_samples: int   = 512
    smooth:    bool  = True
    V_ref:     float = 120.0
    R_load:    float = 10.0
    k_turns:   float = 3.0

    # derived (filled in __post_init__)
    est:           SpectralDensityMobius = field(init=False)
    t:             np.ndarray            = field(init=False)
    T:             float                 = field(init=False)
    envelope:      np.ndarray            = field(init=False)  # |ζ|² / max
    L:             np.ndarray            = field(init=False)  # arc length
    turns_ratio:   np.ndarray            = field(init=False)  # 1 + k·L
    density:       np.ndarray            = field(init=False)  # ρ_elec / max
    carrier:       np.ndarray            = field(init=False)  # cos(ωt)
    volt_scalar:   np.ndarray            = field(init=False)  # turns · density
    V:             np.ndarray            = field(init=False)  # V(t)
    I:             np.ndarray            = field(init=False)  # I(t)
    P:             np.ndarray            = field(init=False)  # P(t)

    # --------------------------------------------------------
    def __post_init__(self):
        if self.R_load <= 0:
            raise ValueError(f"R_load must be positive; got {self.R_load}")
        if self.V_ref <= 0:
            raise ValueError(f"V_ref must be positive; got {self.V_ref}")

        # ---- engine ----
        self.est = SpectralDensityMobius(
            K=self.K, J_C=self.J_C, omega=self.omega,
            N_samples=self.N_samples, smooth=self.smooth,
        )
        self.t = self.est.t
        self.T = self.est.T

        self._build_profiles()

    # --------------------------------------------------------
    def _build_profiles(self):
        est = self.est

        # 1. amplitude envelope |ζ|², normalized to [0,1]
        p = est.power
        self.envelope = p / (p.max() + 1e-12)

        # 2. arc length L(t) ∈ [0,1]  (est.arc already normalized)
        self.L = est.arc.copy()

        # 3. turns ratio from arc length
        self.turns_ratio = 1.0 + self.k_turns * self.L

        # 4. point particle density ρ_elec, normalized to [0,1]
        rho = est.electron_density
        rho_max = rho.max()
        self.density = (rho / (rho_max + 1e-12)) if rho_max > 0 else rho

        # 5. AC carrier  cos(ω t)
        self.carrier = np.cos(self.omega * self.t)

        # 6. volt-level scalar — arc length AND density
        self.volt_scalar = self.turns_ratio * self.density

        # 7. secondary voltage
        self.V = (self.V_ref
                  * self.envelope
                  * self.volt_scalar
                  * self.carrier)

        # 8. current  (resistive load)
        self.I = self.V / self.R_load

        # 9. instantaneous power
        self.P = self.V * self.I                 # = V² / R_load

    # --------------------------------------------------------
    #  Time-domain metrics
    # --------------------------------------------------------
    def rms_voltage(self) -> float:
        return float(np.sqrt(np.mean(self.V ** 2)))

    def rms_current(self) -> float:
        return float(np.sqrt(np.mean(self.I ** 2)))

    def avg_power(self) -> float:
        return float(np.mean(self.P))

    def peak_voltage(self) -> float:
        return float(np.max(np.abs(self.V)))

    def peak_power(self) -> float:
        return float(np.max(self.P))

    def apparent_power(self) -> float:
        """VA = V_rms · I_rms."""
        return self.rms_voltage() * self.rms_current()

    def power_factor(self) -> float:
        """
        Real power / apparent power.

        Given I(t) = V(t)/R_load the load is purely resistive so
        PF = 1 by construction.  Reported for completeness.
        """
        ap = self.apparent_power()
        return self.avg_power() / ap if ap > 1e-12 else 0.0

    def throughput_energy(self) -> float:
        """∫ P(t) dt over one period (Joules per period)."""
        if len(self.t) < 2:
            return 0.0
        dt = self.t[1] - self.t[0]
        return float(np.trapezoid(self.P, dx=dt))

    # --------------------------------------------------------
    #  Frequency-domain throughput  (P(f))
    # --------------------------------------------------------
    def spectral_transfer(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transfer function at arc-length resonances:

            |H(f)|²  =  P_spec(f) / max(P_spec)
        """
        est = self.est
        if est.spec_f.size == 0:
            return np.array([]), np.array([])
        P = est.spec_P
        H2 = P / (P.max() + 1e-12)
        return est.spec_f.copy(), H2

    def spectral_throughput(self, P_in: float = 1.0
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """P_out(f) = P_in · |H(f)|²."""
        f, H2 = self.spectral_transfer()
        return f, P_in * H2

    def integrated_throughput(self) -> float:
        """Sum of P_out(f) over the discrete resonances."""
        _, P_out = self.spectral_throughput()
        return float(P_out.sum())

    # --------------------------------------------------------
    #  Chain decomposition — what each stage contributed
    # --------------------------------------------------------
    def stage_contributions(self) -> Dict[str, float]:
        return dict(
            envelope_mean    = float(self.envelope.mean()),
            envelope_max     = float(self.envelope.max()),
            L_mean           = float(self.L.mean()),
            L_max            = float(self.L.max()),
            turns_mean       = float(self.turns_ratio.mean()),
            turns_max        = float(self.turns_ratio.max()),
            density_mean     = float(self.density.mean()),
            density_max      = float(self.density.max()),
            volt_scalar_mean = float(self.volt_scalar.mean()),
            volt_scalar_max  = float(self.volt_scalar.max()),
            carrier_peak     = float(np.max(np.abs(self.carrier))),
        )

    # --------------------------------------------------------
    def report(self) -> str:
        s = self.stage_contributions()
        lines = [
            "=" * 76,
            "Transformer throughput  ·  elliptic Möbius gate",
            "=" * 76,
            f"  V_ref                  = {self.V_ref:.4f} V",
            f"  R_load                 = {self.R_load:.4f} Ω",
            f"  k_turns                = {self.k_turns:.4f}",
            f"  ω                      = {self.omega:.4f}",
            f"  T                      = {self.T:.6f}",
            f"  N_samples              = {self.N_samples}",
            "",
            "--- primary side:  |ζ(t)|²  amplitude envelope ---",
            f"  mean                   = {s['envelope_mean']:.6f}",
            f"  max                    = {s['envelope_max']:.6f}",
            "",
            "--- turns ratio:  L(t) = ∫ √(dRe² + dIm²) ---",
            f"  L mean / max           = {s['L_mean']:.4f} / {s['L_max']:.4f}",
            f"  turns  = 1 + k·L       = {s['turns_mean']:.4f} / "
            f"{s['turns_max']:.4f}",
            "",
            "--- coupling scalar:  ρ_elec(t) point-particle density ---",
            f"  mean / max             = {s['density_mean']:.4f} / "
            f"{s['density_max']:.4f}",
            f"  middle band            = [{0.25 * self.T:.3f}, "
            f"{0.75 * self.T:.3f}]",
            "",
            "--- AC carrier:  E(t) = (S_νμ + S_ντ) cos(ω t) ---",
            f"  S_νμ                   = {self.est.S_mu:+.6f}",
            f"  S_ντ                   = {self.est.S_tau:+.6f}",
            f"  S_νμ + S_ντ            = "
            f"{self.est.S_mu + self.est.S_tau:+.4e}",
            f"  carrier peak           = {s['carrier_peak']:.4f}",
            "",
            "--- volt-level scalar  (turns · density) ---",
            f"  mean                   = {s['volt_scalar_mean']:.6f}",
            f"  max                    = {s['volt_scalar_max']:.6f}",
            "",
            "--- time-domain throughput ---",
            f"  V_rms                  = {self.rms_voltage():.4f} V",
            f"  V_peak                 = {self.peak_voltage():.4f} V",
            f"  I_rms                  = {self.rms_current():.4f} A",
            f"  avg power              = {self.avg_power():.4f} W",
            f"  peak power             = {self.peak_power():.4f} W",
            f"  apparent power         = {self.apparent_power():.4f} VA",
            f"  power factor           = {self.power_factor():.4f}",
            f"  energy / period        = "
            f"{self.throughput_energy():.4e} J",
            "",
        ]

        f, H2 = self.spectral_transfer()
        if len(f) > 0:
            lines += [
                "--- spectral throughput  ·  P(f) at arc resonances ---",
                f"  resonances             = {len(f)}",
                f"  f range                = [{f.min():.3f}, {f.max():.3f}]",
                f"  |H(f)|² range          = [{H2.min():.4e}, "
                f"{H2.max():.4e}]",
                f"  integrated P_out(f)    = "
                f"{self.integrated_throughput():.4e}",
            ]
        else:
            lines.append("--- spectral throughput ---")
            lines.append("  (no arc-length resonances)")
        return "\n".join(lines)

    # --------------------------------------------------------
    def visualize(self, save_path: Optional[str] = None) -> None:
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.30)

        # (a) |ζ|² amplitude envelope
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(self.t, self.est.power, color="#3498db", lw=1.4)
        ax.set_xlabel("t"); ax.set_ylabel("|ζ|²")
        ax.set_title("Primary  ·  |ζ(t)|²  amplitude envelope")
        ax.grid(alpha=0.3)

        # (b) L(t) → turns ratio
        ax = fig.add_subplot(gs[0, 1])
        ax.plot(self.t, self.L, color="#e67e22", lw=1.4,
                label="L(t)  arc length")
        ax2 = ax.twinx()
        ax2.plot(self.t, self.turns_ratio, color="#8e44ad", lw=1.2,
                 ls="--", label=f"turns = 1 + {self.k_turns}·L")
        ax.set_xlabel("t"); ax.set_ylabel("L(t)", color="#e67e22")
        ax2.set_ylabel("turns ratio", color="#8e44ad")
        ax.set_title("Arc length  →  turns ratio")
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

        # (c) point-particle density
        ax = fig.add_subplot(gs[1, 0])
        ax.fill_between(self.t, 0, self.density,
                        color="#8e44ad", alpha=0.4,
                        label="ρ_elec(t)")
        ew = self.est.electron_weight
        ew_max = ew.max() + 1e-12
        ax.scatter(self.est.electron_t, ew / ew_max,
                   color="#c0392b", s=8, alpha=0.6,
                   label=f"point particles "
                         f"({self.est.electron_t.size})")
        ax.axvspan(0.25 * self.T, 0.75 * self.T,
                   color="gray", alpha=0.08,
                   label="middle band")
        ax.set_xlabel("t"); ax.set_ylabel("ρ_elec (normalized)")
        ax.set_title("Coupling scalar  ·  point-particle density")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # (d) AC carrier E(t)
        ax = fig.add_subplot(gs[1, 1])
        ax.plot(self.t, self.est.E_total, color="#16a085", lw=1.4,
                label="E(t) = (S_νμ + S_ντ) cos(ω t)")
        ax.plot(self.t, self.est.E_mu, color="#e74c3c",
                lw=0.8, ls="--", alpha=0.65, label="E_μ")
        ax.plot(self.t, self.est.E_tau, color="#3498db",
                lw=0.8, ls=":",  alpha=0.65, label="E_τ")
        ax.axhline(0, color="gray", lw=0.4)
        ax.set_xlabel("t"); ax.set_ylabel("E(t)")
        ax.set_title("AC carrier  ·  alternating E-field")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # (e) V(t) and I(t)
        ax = fig.add_subplot(gs[2, 0])
        ax.plot(self.t, self.V, color="#e74c3c", lw=1.2,
                label=f"V(t)  V_rms = {self.rms_voltage():.2f} V")
        ax2 = ax.twinx()
        ax2.plot(self.t, self.I, color="#16a085", lw=1.0,
                 ls="--", alpha=0.85,
                 label=f"I(t)  I_rms = {self.rms_current():.3f} A")
        ax.axhline(0, color="gray", lw=0.4)
        ax.set_xlabel("t"); ax.set_ylabel("V(t) [V]", color="#e74c3c")
        ax2.set_ylabel("I(t) [A]", color="#16a085")
        ax.set_title(
            f"Secondary voltage / current  ·  "
            f"avg power = {self.avg_power():.2f} W")
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

        # (f) spectral throughput P_out(f)
        ax = fig.add_subplot(gs[2, 1])
        f, P_out = self.spectral_throughput()
        if len(f) > 0:
            f = self.est.spec_f
            P = self.est.spec_P
            mu_mask = self.est.spec_muon_mask
            ta_mask = self.est.spec_tau_mask
            if mu_mask.any():
                ax.scatter(f[mu_mask], P_out[mu_mask],
                           color="#e74c3c", s=55, marker="o",
                           edgecolors="k", linewidths=0.5,
                           label=f"μ  ({int(mu_mask.sum())})")
            if ta_mask.any():
                ax.scatter(f[ta_mask], P_out[ta_mask],
                           color="#3498db", s=55, marker="s",
                           edgecolors="k", linewidths=0.5,
                           label=f"τ  ({int(ta_mask.sum())})")
            for f0, P0 in zip(f, P_out):
                ax.plot([f0, f0], [0, P0],
                        color="gray", lw=0.4, alpha=0.4)
            ax.set_xscale("log")
            ax.set_xlabel("frequency  f = 1/λ")
            ax.set_ylabel("|H(f)|²  throughput")
            ax.set_title(
                f"Spectral throughput  ·  {len(f)} arc-length resonances  ·  "
                f"∫ P_out = {self.integrated_throughput():.3e}")
            ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")

        plt.suptitle(
            "Transformer throughput  ·  |ζ|² envelope  ×  L arc turns  ×  "
            f"ρ_elec coupling  ×  E(t) carrier  ×  P(f) transfer  ·  "
            f"α = {ALPHA:.4f}",
            fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save_path:
            plt.savefig(save_path, dpi=140)
        plt.show()


# ============================================================
#  Convenience constructor
# ============================================================
def make_transformer(K: int = 20,
                     J_C: float = 1.0,
                     omega: float = 1.0,
                     N_samples: int = 512,
                     V_ref: float = 120.0,
                     R_load: float = 10.0,
                     k_turns: float = 3.0,
                     smooth: bool = True) -> TransformerThroughput:
    return TransformerThroughput(
        K=K, J_C=J_C, omega=omega, N_samples=N_samples,
        smooth=smooth, V_ref=V_ref, R_load=R_load, k_turns=k_turns,
    )


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("Transformer throughput  ·  elliptic Möbius gate")
    print("=" * 78)
    print(f"  α         = 1/(π − e)    = {ALPHA:.6f}")
    print(f"  A_STEP                    = {A_STEP:.6f}")
    print(f"  6/π²                      = {DENSITY:.6f}")
    print()

    tr = make_transformer(
        K=20, J_C=1.0, omega=1.0,
        N_samples=512,
        V_ref=120.0,
        R_load=10.0,
        k_turns=3.0,
        smooth=True,
    )
    print(tr.report())
    print()

    # ---- instantaneous sample table ----
    print("--- instantaneous values at 8 sample points ---")
    print(f"  {'t':>8s}  {'|ζ|²':>10s}  {'L(t)':>8s}  "
          f"{'turns':>8s}  {'ρ':>8s}  "
          f"{'E(t)':>10s}  {'V(t)':>10s}  {'I(t)':>10s}")
    print("  " + "-" * 92)
    step = max(1, tr.N_samples // 8)
    for i in range(0, tr.N_samples, step):
        print(f"  {tr.t[i]:>8.4f}  {tr.est.power[i]:>10.4f}  "
              f"{tr.L[i]:>8.4f}  {tr.turns_ratio[i]:>8.4f}  "
              f"{tr.density[i]:>8.4f}  {tr.est.E_total[i]:>+10.4f}  "
              f"{tr.V[i]:>+10.4f}  {tr.I[i]:>+10.4f}")
    print()

    # ---- chain decomposition ----
    print("--- stage decomposition ---")
    sc = tr.stage_contributions()
    for k, v in sc.items():
        print(f"  {k:<18s} = {v:.6f}")
    print()

    # ---- spectral throughput at resonances ----
    f, P_out = tr.spectral_throughput()
    if len(f) > 0:
        print("--- spectral throughput  (first 8 resonances) ---")
        print(f"  {'k':>3s}  {'f_k':>10s}  {'P_spec':>12s}  "
              f"{'|H(f)|²':>10s}  {'P_out':>10s}")
        print("  " + "-" * 60)
        for k in range(min(8, len(f))):
            print(f"  {k:>3d}  {f[k]:>10.4f}  "
                  f"{tr.est.spec_P[k]:>12.4e}  "
                  f"{P_out[k]:>10.4e}  "
                  f"{P_out[k]:>10.4e}")
        print()
        print(f"  integrated P_out(f) = {tr.integrated_throughput():.4e}")
        print()

    # ---- report which wire_analog primitives align ----
    print("--- cross-check against wire_analog ---")
    # Recompute S_νμ, S_ντ on the same sample window the estimator used
    half = tr.N_samples // 2
    r_samp = np.arange(1, half + 1, dtype=float)
    S_mu, S_tau = alternating_E_field(
        tr.t[:half], r_samp, J_C=tr.J_C)
    print(f"  wire_analog.alternating_E_field:")
    print(f"    S_νμ                 = {S_mu:+.6f}  "
          f"(est: {tr.est.S_mu:+.6f})")
    print(f"    S_ντ                 = {S_tau:+.6f}  "
          f"(est: {tr.est.S_tau:+.6f})")
    print(f"  wire_analog.spin_grade:")
    print(f"    σ_μ(J_C)             = {spin_grade(tr.J_C, 0.0):+.0f}")
    print(f"    σ_τ(J_C)             = {spin_grade(tr.J_C, 1.0):+.0f}")
    print()

    tr.visualize()


if __name__ == "__main__":
    demo()