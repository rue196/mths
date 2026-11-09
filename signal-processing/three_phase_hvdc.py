#!/usr/bin/env python3
"""
transformer_3phase.py
=====================

Three-phase electric power system on the elliptic Möbius gate.

Signal model
------------
    E(t) = (S_νμ + S_ντ) · cos(ω t)                ← reference rail / phase 1
    alternating field  =  S_νμ + S_ντ              ← AC amplitude scalar
    volt_scalar(t)     =  (1 + k_turns · L(t)) · ρ_elec(t)/max(ρ_elec)
                            ↑ arc length → turns    ↑ point-particle coupling

Per-phase:
    signal_k(t) = DC_offset + S_amp · cos(ω t − φ_k)
    V_k(t)      = V_ref · volt_scalar(t) · signal_k(t)
    I_k(t)      = (V_k(t) / R_load) · (1 + k_turns · L(t))    ← arc-length current

Modes
-----
    three_phase   φ = (0, 2π/3, 4π/3),  DC_offset = 0    (balanced AC)
    hvdc          φ = 0,                 no cos, phases are pure DC rails
    hybrid        φ = (0, 2π/3, 4π/3),  DC_offset = S_amp
                  (HVDC carried on top of 3-phase AC)

Outputs
-------
    phase voltages V_1, V_2, V_3
    line-to-line  V_12, V_23, V_31
    neutral        I_N = I_1 + I_2 + I_3
    DC bus         V_dc_bus (rectified 3φ line-line)
    HVDC rail      E(t) mean (hvdc mode)

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

# ---- engine (wires DC_chirp + wire_analog internally) ----
from AC_DC_elliptic_mobius_tau_muon import (
    SpectralDensityMobius,
    PI, E, ALPHA, A_STEP, DENSITY,
)

# ---- reference primitives for cross-check ----
from wire_analog import alternating_E_field, pi_e, spin_grade
from DC_chirp  import DCMobiusGate, mobius_sieve, even_even_mask


VALID_MODES = ('three_phase', 'hvdc', 'hybrid')


# ============================================================
#  ThreePhaseTransformer
# ============================================================
@dataclass
class ThreePhaseTransformer:
    """
    Three-phase (or HVDC / hybrid) electric model.

    Parameters
    ----------
    K, J_C, omega, N_samples, smooth  →  forwarded to SpectralDensityMobius
    V_ref           peak reference voltage [V]
    R_load          resistive load per phase [Ω]
    k_turns         turns sensitivity to arc length L(t)
                    (turns(t) = 1 + k_turns · L(t))
    mode            'three_phase' | 'hvdc' | 'hybrid'
    dc_offset_factor  hybrid DC offset in units of S_amp  (1.0 = full)
    """

    K:               int   = 20
    J_C:             float = 1.0
    omega:           float = 1.0
    N_samples:       int   = 512
    smooth:          bool  = True
    V_ref:           float = 120.0
    R_load:          float = 10.0
    k_turns:         float = 3.0
    mode:            str   = 'three_phase'
    dc_offset_factor: float = 1.0

    # derived
    est:          SpectralDensityMobius = field(init=False)
    t:            np.ndarray            = field(init=False)
    T:            float                 = field(init=False)
    S_amp:        float                 = field(init=False)  # S_νμ + S_ντ
    E_ref:        np.ndarray            = field(init=False)  # = S_amp·cos(ωt)
    L:            np.ndarray            = field(init=False)
    turns:        np.ndarray            = field(init=False)
    density:      np.ndarray            = field(init=False)
    volt_scalar:  np.ndarray            = field(init=False)
    phase_offsets: Tuple[float, float, float] = field(init=False)

    # per phase: V, I, P, signal, carrier
    V_phase:  List[np.ndarray] = field(init=False)
    I_phase:  List[np.ndarray] = field(init=False)
    P_phase:  List[np.ndarray] = field(init=False)
    sig_phase: List[np.ndarray] = field(init=False)
    car_phase: List[np.ndarray] = field(init=False)

    # aggregate quantities
    V_12: np.ndarray = field(init=False)
    V_23: np.ndarray = field(init=False)
    V_31: np.ndarray = field(init=False)
    I_N:  np.ndarray = field(init=False)
    V_dc_bus: np.ndarray = field(init=False)
    P_total:  np.ndarray = field(init=False)

    # --------------------------------------------------------
    def __post_init__(self):
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"mode must be one of {VALID_MODES}; got {self.mode!r}")
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

        # ---- phase offsets per mode ----
        if self.mode == 'hvdc':
            self.phase_offsets = (0.0, 0.0, 0.0)
        else:
            self.phase_offsets = (0.0, 2.0 * PI / 3.0, 4.0 * PI / 3.0)

        self._build()

    # --------------------------------------------------------
    def _build(self):
        est = self.est

        # ---- base signals ----
        self.S_amp = est.S_mu + est.S_tau        # alternating field amplitude
        self.E_ref = est.E_total                 # E(t) = S_amp · cos(ω t)

        # ---- modulation scalars ----
        self.L = est.arc.copy()                  # arc length, normalized [0,1]
        self.turns = 1.0 + self.k_turns * self.L

        rho = est.electron_density
        rho_max = rho.max()
        self.density = (rho / (rho_max + 1e-12)) if rho_max > 0 else rho

        self.volt_scalar = self.turns * self.density

        # ---- per-phase construction ----
        # DC offset only in hybrid mode (in units of S_amp)
        if self.mode == 'hybrid':
            dc_off = self.dc_offset_factor * self.S_amp
        else:
            dc_off = 0.0

        self.V_phase, self.I_phase, self.P_phase = [], [], []
        self.sig_phase, self.car_phase = [], []

        for k, phi in enumerate(self.phase_offsets):
            if self.mode == 'hvdc':
                # pure DC: no cos modulation, all phases share the same
                # constant level = S_amp
                sig_k = np.full_like(self.t, self.S_amp)
                car_k = np.ones_like(self.t)
            else:
                car_k = np.cos(self.omega * self.t - phi)
                sig_k = dc_off + self.S_amp * car_k

            V_k = self.V_ref * self.volt_scalar * sig_k
            # variable current scaled by arc-length → turns ratio
            I_k = (V_k / self.R_load) * self.turns
            P_k = V_k * I_k

            self.sig_phase.append(sig_k)
            self.car_phase.append(car_k)
            self.V_phase.append(V_k)
            self.I_phase.append(I_k)
            self.P_phase.append(P_k)

        # ---- line-to-line voltages ----
        self.V_12 = self.V_phase[0] - self.V_phase[1]
        self.V_23 = self.V_phase[1] - self.V_phase[2]
        self.V_31 = self.V_phase[2] - self.V_phase[0]

        # ---- neutral current ----
        self.I_N = self.I_phase[0] + self.I_phase[1] + self.I_phase[2]

        # ---- DC bus from rectified 3φ line-line ----
        # 6-pulse style: average of the magnitudes of the three
        # line-to-line voltages at each instant
        self.V_dc_bus = (
            np.abs(self.V_12) + np.abs(self.V_23) + np.abs(self.V_31)
        ) / 3.0

        # ---- total power ----
        self.P_total = self.P_phase[0] + self.P_phase[1] + self.P_phase[2]

    # --------------------------------------------------------
    #  Metrics
    # --------------------------------------------------------
    @staticmethod
    def _rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(x ** 2)))

    def rms_phase_voltage(self, k: int) -> float:
        return self._rms(self.V_phase[k])

    def peak_phase_voltage(self, k: int) -> float:
        return float(np.max(np.abs(self.V_phase[k])))

    def rms_phase_current(self, k: int) -> float:
        return self._rms(self.I_phase[k])

    def avg_phase_power(self, k: int) -> float:
        return float(np.mean(self.P_phase[k]))

    def rms_line_voltage(self) -> float:
        return self._rms(self.V_12)

    def rms_neutral_current(self) -> float:
        return self._rms(self.I_N)

    def avg_total_power(self) -> float:
        return float(np.mean(self.P_total))

    def hvdc_rail(self) -> float:
        """
        HVDC rail:
            hvdc mode → mean of E_ref (= S_amp · cos(ωt) averaged)
            others    → mean of rectified DC bus
        """
        if self.mode == 'hvdc':
            return float(np.mean(self.E_ref))
        return float(np.mean(self.V_dc_bus))

    def dc_bus_mean(self) -> float:
        return float(np.mean(self.V_dc_bus))

    def dc_bus_ripple(self) -> float:
        """RMS of (V_dc_bus − mean) / mean — ripple ratio."""
        mu = self.dc_bus_mean()
        if abs(mu) < 1e-12:
            return 0.0
        return float(np.sqrt(np.mean((self.V_dc_bus - mu) ** 2)) / abs(mu))

    def power_factor_phase(self, k: int) -> float:
        s = self.rms_phase_voltage(k) * self.rms_phase_current(k)
        if s < 1e-12:
            return 0.0
        return self.avg_phase_power(k) / s

    def balance_ratio(self) -> float:
        """
        For a balanced 3φ system: |I_1 + I_2 + I_3|_rms / |I_k|_rms should
        be ~0.  Reported as the residual.
        """
        In = self.rms_neutral_current()
        Imean = (self.rms_phase_current(0)
                 + self.rms_phase_current(1)
                 + self.rms_phase_current(2)) / 3.0
        if Imean < 1e-12:
            return 0.0
        return In / Imean

    # --------------------------------------------------------
    def report(self) -> str:
        lines = [
            "=" * 78,
            f"Three-phase transformer  ·  mode = {self.mode}",
            "=" * 78,
            f"  V_ref                    = {self.V_ref:.4f} V",
            f"  R_load (per phase)       = {self.R_load:.4f} Ω",
            f"  k_turns                  = {self.k_turns:.4f}",
            f"  ω                        = {self.omega:.4f}",
            f"  T                        = {self.T:.6f}",
            f"  N_samples                = {self.N_samples}",
            f"  phase offsets            = "
            f"{tuple(round(p, 4) for p in self.phase_offsets)}",
            "",
            "--- reference rail  E(t) = (S_νμ+S_ντ) cos(ω t) ---",
            f"  S_νμ                     = {self.est.S_mu:+.6f}",
            f"  S_ντ                     = {self.est.S_tau:+.6f}",
            f"  S_amp = S_νμ + S_ντ      = {self.S_amp:+.6e}",
            f"  E(t) peak                = "
            f"{float(np.max(np.abs(self.E_ref))):.4e}",
            f"  E(t) mean                = "
            f"{float(np.mean(self.E_ref)):+.4e}",
            "",
            "--- modulation scalars ---",
            f"  L mean / max             = "
            f"{self.L.mean():.4f} / {self.L.max():.4f}",
            f"  turns = 1 + k·L          = "
            f"{self.turns.mean():.4f} / {self.turns.max():.4f}",
            f"  ρ_elec mean / max        = "
            f"{self.density.mean():.4f} / {self.density.max():.4f}",
            f"  volt_scalar mean / max   = "
            f"{self.volt_scalar.mean():.4f} / {self.volt_scalar.max():.4f}",
            "",
            "--- per-phase voltage / current / power ---",
        ]
        for k in range(3):
            phi_deg = math.degrees(self.phase_offsets[k])
            lines += [
                f"  Phase {k+1}  φ = {phi_deg:+7.1f}°",
                f"    V_rms                  = "
                f"{self.rms_phase_voltage(k):.4f} V",
                f"    V_peak                 = "
                f"{self.peak_phase_voltage(k):.4f} V",
                f"    I_rms                  = "
                f"{self.rms_phase_current(k):.4f} A",
                f"    P_avg                  = "
                f"{self.avg_phase_power(k):.4f} W",
                f"    PF                     = "
                f"{self.power_factor_phase(k):.4f}",
            ]
        lines += [
            "",
            "--- aggregate ---",
            f"  V_ll rms (12)            = "
            f"{self.rms_line_voltage():.4f} V",
            f"  I_neutral rms            = "
            f"{self.rms_neutral_current():.4e} A",
            f"  balance ratio |I_N|/⟨|I_k|⟩ = "
            f"{self.balance_ratio():.4e}",
            f"  P_total avg              = "
            f"{self.avg_total_power():.4f} W",
            f"  V_dc_bus mean            = "
            f"{self.dc_bus_mean():.4f} V",
            f"  V_dc_bus ripple          = "
            f"{self.dc_bus_ripple()*100:.2f} %",
            f"  HVDC rail                = "
            f"{self.hvdc_rail():.4f} V",
        ]
        return "\n".join(lines)

    # --------------------------------------------------------
    def visualize(self, save_path: Optional[str] = None) -> None:
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.30)

        colors = ["#e74c3c", "#27ae60", "#2980b9"]
        phase_labels = ["Phase 1 (E(t))", "Phase 2", "Phase 3"]

        # (a) per-phase voltages
        ax = fig.add_subplot(gs[0, 0])
        for k in range(3):
            ax.plot(self.t, self.V_phase[k],
                    color=colors[k], lw=1.2,
                    label=f"{phase_labels[k]}  "
                          f"V_rms={self.rms_phase_voltage(k):.1f}")
        ax.axhline(0, color="gray", lw=0.4)
        ax.set_xlabel("t"); ax.set_ylabel("V_phase [V]")
        ax.set_title(f"Phase voltages  ·  mode = {self.mode}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # (b) per-phase currents (arc-length scaled)
        ax = fig.add_subplot(gs[0, 1])
        for k in range(3):
            ax.plot(self.t, self.I_phase[k],
                    color=colors[k], lw=1.0, alpha=0.9,
                    label=f"I_{k+1}  "
                          f"I_rms={self.rms_phase_current(k):.3f}")
        ax.axhline(0, color="gray", lw=0.4)
        ax.set_xlabel("t"); ax.set_ylabel("I_phase [A]")
        ax.set_title("Phase currents  ·  arc-length → turns scaling")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # (c) line-to-line voltages
        ax = fig.add_subplot(gs[1, 0])
        ax.plot(self.t, self.V_12, color="#8e44ad", lw=1.2,
                label=f"V_12  rms={self._rms(self.V_12):.1f}")
        ax.plot(self.t, self.V_23, color="#16a085", lw=1.2,
                label=f"V_23  rms={self._rms(self.V_23):.1f}")
        ax.plot(self.t, self.V_31, color="#d35400", lw=1.2,
                label=f"V_31  rms={self._rms(self.V_31):.1f}")
        ax.axhline(0, color="gray", lw=0.4)
        ax.set_xlabel("t"); ax.set_ylabel("V_ll [V]")
        ax.set_title("Line-to-line voltages")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # (d) neutral current
        ax = fig.add_subplot(gs[1, 1])
        ax.plot(self.t, self.I_N, color="#c0392b", lw=1.3,
                label=f"I_N  rms={self.rms_neutral_current():.4e}")
        ax.axhline(0, color="gray", lw=0.4)
        ax.set_xlabel("t"); ax.set_ylabel("I_N [A]")
        ax.set_title(
            f"Neutral current  ·  balance ratio = "
            f"{self.balance_ratio():.4e}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # (e) DC bus / HVDC rail
        ax = fig.add_subplot(gs[2, 0])
        ax.plot(self.t, self.V_dc_bus, color="#2c3e50", lw=1.3,
                label=f"V_dc_bus  mean={self.dc_bus_mean():.1f} V  "
                      f"ripple={self.dc_bus_ripple()*100:.1f}%")
        ax.axhline(self.dc_bus_mean(), color="#e74c3c",
                   ls="--", lw=0.9,
                   label=f"mean = {self.dc_bus_mean():.1f} V")
        ax.set_xlabel("t"); ax.set_ylabel("V_dc [V]")
        ax.set_title(f"Rectified DC bus  ·  HVDC rail = "
                     f"{self.hvdc_rail():.1f} V")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # (f) modulation scalars: L, turns, density, volt_scalar
        ax = fig.add_subplot(gs[2, 1])
        ax.plot(self.t, self.L, color="#e67e22", lw=1.2, label="L(t)")
        ax.plot(self.t, self.turns / self.turns.max(),
                color="#8e44ad", lw=1.0, ls="--",
                label="turns / max")
        ax.plot(self.t, self.density, color="#2980b9", lw=1.0,
                ls=":", label="ρ_elec(t)")
        ax.plot(self.t, self.volt_scalar / self.volt_scalar.max(),
                color="#16a085", lw=1.4,
                label="volt_scalar / max")
        ax.set_xlabel("t"); ax.set_ylabel("normalized scale")
        ax.set_title("Modulation chain  ·  L → turns → volt_scalar")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        plt.suptitle(
            f"3-phase transformer  ·  mode = {self.mode}  ·  "
            f"E(t) = (S_νμ+S_ντ)cos(ωt)  ·  "
            f"V_k = V_ref·volt_scalar·sig_k  ·  "
            f"I_k = V_k/R · turns  ·  α = {ALPHA:.4f}",
            fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if save_path:
            plt.savefig(save_path, dpi=140)
        plt.show()


# ============================================================
#  Convenience constructor
# ============================================================
def make_3phase(mode: str = 'three_phase',
                dc_offset_factor: float = 1.0,
                **kwargs) -> ThreePhaseTransformer:
    return ThreePhaseTransformer(mode=mode,
                                  dc_offset_factor=dc_offset_factor,
                                  **kwargs)


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 78)
    print("Three-phase transformer  ·  elliptic Möbius gate")
    print("=" * 78)
    print(f"  α         = 1/(π − e)    = {ALPHA:.6f}")
    print(f"  A_STEP                    = {A_STEP:.6f}")
    print(f"  6/π²                      = {DENSITY:.6f}")
    print()

    # ---- compare all three modes ----
    modes = [
        ('three_phase', 0.0, "balanced 3φ AC"),
        ('hvdc',        0.0, "pure DC rail (all phases aligned)"),
        ('hybrid',      1.0, "HVDC carried on 3φ AC"),
    ]

    results = {}
    for mode, dc_off, desc in modes:
        print("=" * 78)
        print(f"MODE  ·  {mode}   ({desc})")
        print("=" * 78)
        tr = make_3phase(
            mode=mode, dc_offset_factor=dc_off,
            K=20, J_C=1.0, omega=1.0,
            N_samples=512,
            V_ref=120.0, R_load=10.0, k_turns=3.0,
            smooth=True,
        )
        results[mode] = tr
        print(tr.report())
        print()

    # ---- comparison table across modes ----
    print("=" * 78)
    print("MODE COMPARISON")
    print("=" * 78)
    print(f"  {'mode':<14s}  {'V1_rms':>10s}  {'V_ll':>10s}  "
          f"{'I_N_rms':>10s}  {'P_total':>10s}  "
          f"{'V_dc':>10s}  {'ripple%':>8s}")
    print("  " + "-" * 90)
    for mode, _, _ in modes:
        tr = results[mode]
        print(f"  {mode:<14s}  "
              f"{tr.rms_phase_voltage(0):>10.3f}  "
              f"{tr.rms_line_voltage():>10.3f}  "
              f"{tr.rms_neutral_current():>10.3e}  "
              f"{tr.avg_total_power():>10.2f}  "
              f"{tr.dc_bus_mean():>10.3f}  "
              f"{tr.dc_bus_ripple()*100:>8.2f}")
    print()
    print("  note:")
    print("    three_phase: |I_N| ≈ 0 (balanced), V_ll = √3·V_ph")
    print("    hvdc:        |I_N| = 3·|I_ph| (all currents DC, sum to 3×),")
    print("                 DC bus = constant per-phase DC level")
    print("    hybrid:      DC offset + 3φ AC → V_ll has large DC offset")
    print()

    # ---- cross-check E_ref against wire_analog primitives ----
    print("--- cross-check against wire_analog ---")
    tr0 = results['three_phase']
    half = tr0.N_samples // 2
    r_samp = np.arange(1, half + 1, dtype=float)
    S_mu, S_tau = alternating_E_field(tr0.t[:half], r_samp, J_C=tr0.J_C)
    print(f"  wire_analog.alternating_E_field:")
    print(f"    S_νμ     = {S_mu:+.6f}    (est: {tr0.est.S_mu:+.6f})")
    print(f"    S_ντ     = {S_tau:+.6f}    (est: {tr0.est.S_tau:+.6f})")
    print(f"    S_amp    = {S_mu + S_tau:+.6f}    "
          f"(tr.S_amp: {tr0.S_amp:+.6e})")
    print(f"  wire_analog.spin_grade:")
    print(f"    σ_μ(J_C) = {spin_grade(tr0.J_C, 0.0):+.0f}")
    print(f"    σ_τ(J_C) = {spin_grade(tr0.J_C, 1.0):+.0f}")
    print()

    # ---- visualize each mode ----
    for mode, _, _ in modes:
        results[mode].visualize()


if __name__ == "__main__":
    demo()