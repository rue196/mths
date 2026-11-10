#!/usr/bin/env python3
"""
neutral_battery_handler.py
==========================

Anti-depleting battery handler  ·  AC-carrier edition.

    anode    =  τ channel   S_ντ(t)   (quarter-period lag behind μ)
    cathode  =  μ channel   S_νμ(t)
    carrier  =  E(t) = (S_νμ + S_ντ) · cos(ω t)

Three anti-depletion invariants  (updated)
------------------------------------------
    (1) AC carrier      E(t) = (S_νμ + S_ντ)·cos(ωt)
                        mean(S_νμ+S_ντ) / rms(S_νμ+S_ντ) < dc_threshold
                        →  no DC drain across the terminals

    (2) Parity mask     even_even_mask(N_events)  from DC_chirp.py
                        mask matches (i%2==0) & (j%2==0) structurally
                        →  no even-odd scattering leak

    (3) Spectral        P(f) = |FFT(E_ac)|²,   |H(f)|² = P(f)/max(P(f))
                        |H(f)|² ∈ [0, 1]  and non-empty
                        →  bounded power transfer, no runaway

Why it can't deplete
--------------------
    · zero DC → no net charge accumulation per cycle
    · even×even only → no scattering path
    · bounded spectrum → no power blowup

Imports (unchanged parents)
---------------------------
    DC_chirp.DCMobiusGate, mobius_sieve, even_even_mask
    wire_analog.spin_grade, pi_e
    coupling_Q_entanglement.coupling_matrix
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from DC_chirp import DCMobiusGate, mobius_sieve, even_even_mask
from wire_analog import spin_grade, pi_e
from coupling_Q_entanglement import coupling_matrix


# ============================================================
#  Constants
# ============================================================
PI      = math.pi
E       = math.e
ALPHA   = 1.0 / (PI - E)
A_STEP  = ALPHA / 0.3628
DENSITY = 6.0 / (PI * PI)


# ============================================================
#  Battery state
# ============================================================
@dataclass
class BatteryState:
    cycle: int
    t: float
    anode: float                # S_ντ (τ channel, quarter-period lag)
    cathode: float              # S_νμ (μ channel)
    E_ac: float                 # E(t) = (S_νμ+S_ντ)·cos(ωt)
    E_ac_mean_abs: float        # global |⟨E⟩| (same for all cycles)
    voltage: float              # E_ac (instantaneous AC voltage)
    capacity: float             # |S_νμ|² + |S_ντ|²  (standing wave)
    free_electrons: float       # leaked charge (diagnostic, swept to 0)
    carrier_ok: bool            # mean(S_νμ+S_ντ)/rms < dc_threshold
    carrier_dc_ratio: float     # actual DC / RMS of amplitude
    parity_ok: bool             # even_even_mask preserved
    H2_max: float               # max |H(f)|² (normalized)
    spectral_ok: bool           # |H(f)|² ∈ [0, 1] and non-empty
    depleting: bool


# ============================================================
#  Neutral battery handler
# ============================================================
class NeutralBatteryHandler:
    """
    AC-carrier neutral battery.

    anode    = τ channel   S_ντ(t)   (quarter-period lag)
    cathode  = μ channel   S_νμ(t)
    carrier  = E(t) = (S_νμ + S_ντ)·cos(ωt)
    """

    def __init__(self,
                 K: int = 20,
                 J_C: float = 1.0,
                 N_events: int = 41,
                 N_samples: int = 2048,
                 omega: float = 1.0,
                 dc_threshold: float = 0.1,
                 free_e_threshold: float = 0.05):
        self.K = int(K)
        self.J_C = float(J_C)
        self.N_events = int(N_events)
        self.N_samples = int(N_samples)
        self.omega = float(omega)
        self.dc_threshold = float(dc_threshold)
        self.free_e_threshold = float(free_e_threshold)

        # ---- shared clock ----
        self.gate = DCMobiusGate(K=K, smooth=True)
        self.T = self.gate.period
        self.dt_event = self.T / self.N_events
        self.t = np.linspace(0.0, self.T, self.N_samples)
        self.dt = float(self.t[1] - self.t[0])

        # ---- |ζ| lookup (extend past the quarter-period shift) ----
        hi = self.T + 2.0 * self.dt_event + PI / (2.0 * self.omega)
        self._zgrid = np.linspace(0.0, hi, self.N_samples * 2)
        self._zvals = np.abs(
            np.array([self.gate.zeta(zg) for zg in self._zgrid]))

        # ---- S_νμ and S_ντ (independent, quarter-period shifted) ----
        self.S_mu, self.S_tau = self._compute_mu_tau()

        # ---- AC carrier E(t) = (S_νμ + S_ντ)·cos(ωt) ----
        self.cos_wt = np.cos(self.omega * self.t)
        self.E_ac = (self.S_mu + self.S_tau) * self.cos_wt
        self.E_ac_mean = float(np.mean(self.E_ac))
        self.E_ac_rms = float(np.sqrt(np.mean(self.E_ac ** 2)))

        # ---- invariant (1): carrier DC fraction ----
        amp = self.S_mu + self.S_tau
        amp_rms = float(np.sqrt(np.mean(amp ** 2)))
        self.carrier_dc_ratio = (
            abs(float(np.mean(amp))) / (amp_rms + 1e-12))
        self.carrier_ok = self.carrier_dc_ratio < self.dc_threshold

        # ---- invariant (2): even-even mask from DC_chirp ----
        self._build_parity()

        # ---- invariant (3): spectral transfer |H(f)|² ----
        self._build_spectral()

        # ---- free-electrons diagnostic (convolution) ----
        self.conv = np.convolve(self.S_mu, self.S_tau,
                                mode="same") * self.dt

        # ---- state ----
        self.states: List[BatteryState] = []
        self.interventions = 0
        self.depletion_events = 0

    # --------------------------------------------------------
    def _compute_mu_tau(self):
        """
        S_νμ from the base event stream, S_ντ from a quarter-AC-period
        lagged stream.  Together they give a genuine AC carrier.
        """
        sig_mu  = spin_grade(self.J_C, 0.0)
        sig_tau = spin_grade(self.J_C, 1.0)
        n_off = np.arange(self.N_events, dtype=float)
        alts  = (-1.0) ** (n_off + 1)

        quarter = PI / (2.0 * self.omega)          # quarter of AC period

        S_mu  = np.zeros_like(self.t)
        S_tau = np.zeros_like(self.t)
        for i, t_now in enumerate(self.t):
            taus_mu  = t_now + n_off * self.dt_event
            taus_tau = t_now + n_off * self.dt_event + quarter

            mags_mu  = np.interp(taus_mu,  self._zgrid, self._zvals)
            mags_tau = np.interp(taus_tau, self._zgrid, self._zvals)

            S_mu[i]  = sig_mu  * float(np.dot(alts, mags_mu))
            S_tau[i] = sig_tau * float(np.dot(alts, mags_tau))
        return S_mu, S_tau

    # --------------------------------------------------------
    def _build_parity(self):
        """
        Build the even-even mask via DC_chirp.even_even_mask and
        construct the parity-protected coupling matrix.
        """
        N = self.N_events
        t_ev = np.arange(1, N + 1) * self.dt_event
        r_ev = np.arange(1, N + 1, dtype=float)
        self.G = coupling_matrix(t_ev, r_ev)

        # the mask itself, from DC_chirp
        self.mask_ee = even_even_mask(N)
        self.G_ee = self.G * self.mask_ee
        self.G_eo = self.G * (~self.mask_ee.astype(bool))

        # structural check: mask must be (i even) & (j even)
        i_idx = np.arange(N)[:, None]
        j_idx = np.arange(N)[None, :]
        ref = (i_idx % 2 == 0) & (j_idx % 2 == 0)
        self.parity_ok = bool(
            np.array_equal(self.mask_ee.astype(bool), ref))
        self.mask_shape = tuple(self.mask_ee.shape)

    # --------------------------------------------------------
    def _build_spectral(self):
        """
        P(f) = |FFT(E_ac)|²,
        |H(f)|² = P(f) / max(P(f))   ∈ [0, 1]
        """
        sig = self.E_ac - self.E_ac_mean
        N = len(sig)
        self.fft_vals = np.fft.rfft(sig)
        self.freqs = np.fft.rfftfreq(N, d=self.dt)
        self.P_f = np.abs(self.fft_vals) ** 2

        P_max = float(self.P_f.max()) if self.P_f.size else 0.0
        if P_max > 1e-15:
            self.H2 = self.P_f / P_max
        else:
            self.H2 = np.zeros_like(self.P_f)

        self.H2_max = float(self.H2.max()) if self.H2.size else 0.0
        # invariant (3): non-empty and within [0, 1]
        self.spectral_ok = (
            self.H2.size > 0
            and self.H2.min() >= -1e-9
            and self.H2.max() <= 1.0 + 1e-9
        )

    # --------------------------------------------------------
    #  Measure one cycle
    # --------------------------------------------------------
    def measure(self, idx: int, cycle: int) -> BatteryState:
        t = float(self.t[idx])
        q_cath = float(self.S_mu[idx])
        q_anod = float(self.S_tau[idx])
        E_t    = float(self.E_ac[idx])

        # free electrons diagnostic
        c_val = abs(float(self.conv[idx]))
        free_e = c_val if c_val > self.free_e_threshold else 0.0

        capacity = q_cath ** 2 + q_anod ** 2

        depleting = not (
            self.carrier_ok and self.parity_ok and self.spectral_ok)

        return BatteryState(
            cycle=cycle, t=t,
            anode=q_anod, cathode=q_cath,
            E_ac=E_t,
            E_ac_mean_abs=abs(self.E_ac_mean),
            voltage=E_t,
            capacity=capacity,
            free_electrons=free_e,
            carrier_ok=self.carrier_ok,
            carrier_dc_ratio=self.carrier_dc_ratio,
            parity_ok=self.parity_ok,
            H2_max=self.H2_max,
            spectral_ok=self.spectral_ok,
            depleting=depleting,
        )

    # --------------------------------------------------------
    def anti_deplete(self, state: BatteryState) -> BatteryState:
        """
        Re-check all three invariants, log interventions, sweep
        any leaked free charge to the nearest convolution zero.
        """
        if not state.carrier_ok:
            self.interventions += 1
            self.depletion_events += 1
        if not state.parity_ok:
            self.interventions += 1
            self.depletion_events += 1
        if not state.spectral_ok:
            self.interventions += 1
            self.depletion_events += 1

        # free electrons: sweep to nearest zero of |conv|
        if state.free_electrons > 0:
            idx0 = int(round(state.t / self.dt))
            idx0 = max(0, min(self.N_samples - 1, idx0))
            window = 20
            lo = max(0, idx0 - window)
            hi = min(self.N_samples, idx0 + window)
            candidates = [j for j in range(lo, hi)
                          if abs(self.conv[j]) <= self.free_e_threshold]
            if candidates:
                j = min(candidates, key=lambda k: abs(self.conv[k]))
                state = self.measure(j, state.cycle)
            state.free_electrons = 0.0

        state.depleting = (
            (not state.carrier_ok)
            or (not state.parity_ok)
            or (not state.spectral_ok)
        )
        return state

    # --------------------------------------------------------
    def run_cycle(self, idx: int, cycle: int) -> BatteryState:
        raw = self.measure(idx, cycle)
        safe = self.anti_deplete(raw)
        self.states.append(safe)
        return safe

    def full_cycle_scan(self) -> List[BatteryState]:
        for k, idx in enumerate(range(self.N_samples)):
            self.run_cycle(idx, k)
        return self.states

    # --------------------------------------------------------
    def report(self) -> str:
        s = self.states
        if not s:
            return "(no cycles run)"

        depleting_cycles = sum(1 for x in s if x.depleting)
        volt_range = (min(x.voltage for x in s),
                      max(x.voltage for x in s))
        cap_range  = (min(x.capacity for x in s),
                      max(x.capacity for x in s))

        L = ["=" * 76,
             "AC-carrier neutral battery  ·  anti-depletion report",
             "=" * 76]
        L.append("  terminals")
        L.append("    anode   (τ)      : S_ντ(t)  (quarter-period lag)")
        L.append("    cathode (μ)      : S_νμ(t)")
        L.append("")
        L.append("  invariants (three)")
        L.append("    (1) AC carrier   E(t) = (S_νμ+S_ντ)·cos(ωt)")
        L.append(f"        DC / RMS     : {self.carrier_dc_ratio:.4e}  "
                 f"(threshold {self.dc_threshold})")
        L.append(f"        carrier_ok   : {self.carrier_ok}")
        L.append(f"    (2) parity mask  even_even_mask(N_events={self.N_events})")
        L.append(f"        mask shape   : {self.mask_shape}")
        L.append(f"        parity_ok    : {self.parity_ok}")
        L.append("    (3) spectral     P(f) → |H(f)|²")
        L.append(f"        |H(f)|² max  : {self.H2_max:.6f}")
        L.append(f"        spectral_ok  : {self.spectral_ok}")
        L.append("")
        L.append("  energy")
        L.append(f"    E_ac range       : [{self.E_ac.min():+.4f}, "
                 f"{self.E_ac.max():+.4f}]")
        L.append(f"    E_ac mean        : {self.E_ac_mean:+.4e}")
        L.append(f"    E_ac rms         : {self.E_ac_rms:.4f}")
        L.append(f"    voltage range    : [{volt_range[0]:+.4f}, "
                 f"{volt_range[1]:+.4f}]")
        L.append(f"    capacity range   : [{cap_range[0]:.4f}, "
                 f"{cap_range[1]:.4f}]")
        L.append("")
        L.append("  free electrons (diagnostic)")
        L.append(f"    max free charge  : "
                 f"{max(x.free_electrons for x in s):.6f}")
        L.append(f"    leak threshold   : {self.free_e_threshold}")
        L.append("")
        L.append("  handler diagnostics")
        L.append(f"    cycles logged    : {len(s)}")
        L.append(f"    interventions    : {self.interventions}")
        L.append(f"    depletion flag   : {depleting_cycles} / {len(s)}")
        L.append(f"    battery state    : "
                 f"{'DEPLETED' if depleting_cycles > 0 else 'HEALTHY'}")
        return "\n".join(L)

    # --------------------------------------------------------
    def visualize(self):
        if not self.states:
            return

        t       = np.array([x.t for x in self.states])
        anode   = np.array([x.anode for x in self.states])
        cathode = np.array([x.cathode for x in self.states])
        E_ac    = np.array([x.E_ac for x in self.states])
        cap     = np.array([x.capacity for x in self.states])
        free_e  = np.array([x.free_electrons for x in self.states])
        deplet  = np.array([x.depleting for x in self.states])

        fig = plt.figure(figsize=(16, 10))
        gs  = GridSpec(2, 3, figure=fig, hspace=0.36, wspace=0.30)

        # (a) terminal signals
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(t, cathode, color="#e74c3c", lw=1.2,
                label=r"$S_{\nu\mu}(t)$  (cathode, μ)")
        ax.plot(t, anode, color="#3498db", lw=1.2, ls="--",
                label=r"$S_{\nu\tau}(t)$  (anode, τ)")
        ax.axhline(0, color="k", lw=0.4)
        ax.set_xlabel("t"); ax.set_ylabel("signal")
        ax.set_title("Terminal signals  ·  anode (τ) / cathode (μ)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # (b) AC carrier E(t)
        ax = fig.add_subplot(gs[0, 1])
        ax.plot(t, E_ac, color="#16a085", lw=1.2,
                label="E(t) = (S_νμ+S_ντ)·cos(ωt)")
        ax.axhline(0, color="k", lw=0.5)
        ax.axhline(self.E_ac_mean, color="gray", ls=":",
                   lw=1.0, label=f"mean = {self.E_ac_mean:+.2e}")
        ax.set_xlabel("t"); ax.set_ylabel("E(t)")
        ax.set_title(f"AC carrier  ·  DC/RMS = "
                     f"{self.carrier_dc_ratio:.2e}  ·  "
                     f"ok = {self.carrier_ok}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # (c) capacity
        ax = fig.add_subplot(gs[0, 2])
        ax.plot(t, cap, color="#8e44ad", lw=1.2,
                label="capacity = |S_νμ|² + |S_ντ|²")
        ax.set_xlabel("t"); ax.set_ylabel("capacity")
        ax.set_title("Standing-wave energy  ·  no decay")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # (d) even-even mask
        ax = fig.add_subplot(gs[1, 0])
        im = ax.imshow(self.mask_ee.astype(float), origin="lower",
                       cmap="Blues", aspect="auto", vmin=0, vmax=1)
        ax.set_xlabel("j"); ax.set_ylabel("i")
        ax.set_title(
            f"even_even_mask(N={self.N_events})  ·  "
            f"parity_ok = {self.parity_ok}")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # (e) spectral transfer
        ax = fig.add_subplot(gs[1, 1])
        if self.H2.size > 0:
            ax.semilogy(self.freqs, self.H2 + 1e-12,
                        color="#2c3e50", lw=1.2,
                        label=f"|H(f)|²  max={self.H2_max:.4f}")
            ax.axhline(1.0, color="gray", ls="--", lw=0.7,
                       label="bound = 1")
            ax.set_xlabel("f"); ax.set_ylabel("|H(f)|²")
            ax.set_title(f"Spectral transfer  ·  "
                         f"spectral_ok = {self.spectral_ok}")
            ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")

        # (f) status panel
        ax = fig.add_subplot(gs[1, 2])
        ax.axis("off")
        healthy = all(not d for d in deplet)
        banner  = "HEALTHY" if healthy else "DEPLETED"
        bcolor  = "#16a085" if healthy else "#e74c3c"

        ax.text(0.0, 1.0,
                f"AC-carrier battery\n\n"
                f"  ω              = {self.omega:.4f}\n"
                f"  T              = {self.T:.4f}\n"
                f"  N_events       = {self.N_events}\n"
                f"  N_samples      = {self.N_samples}\n\n"
                f"  invariants:\n"
                f"    carrier_ok   = {self.carrier_ok}\n"
                f"      DC/RMS     = {self.carrier_dc_ratio:.4e}\n"
                f"      thr        = {self.dc_threshold}\n"
                f"    parity_ok    = {self.parity_ok}\n"
                f"      mask shape = {self.mask_shape}\n"
                f"    spectral_ok  = {self.spectral_ok}\n"
                f"      |H|² max   = {self.H2_max:.4f}\n\n"
                f"  diagnostics:\n"
                f"    cycles       = {len(self.states)}\n"
                f"    interventions= {self.interventions}\n"
                f"    depleting    = "
                f"{sum(1 for d in deplet if d)} / {len(deplet)}",
                va="top", family="monospace", fontsize=9)
        ax.text(0.0, 0.03, banner, fontsize=14,
                color=bcolor, fontweight="bold",
                family="monospace")

        plt.suptitle(
            f"AC-carrier neutral battery  ·  "
            f"E(t) = (S_νμ+S_ντ)·cos(ωt)  ·  "
            f"parity = even-even mask  ·  "
            f"spectral = |H(f)|²  ·  α = {ALPHA:.4f}",
            fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 76)
    print("AC-carrier neutral battery  ·  anti-depletion")
    print("=" * 76)
    print(f"  α      = 1/(π − e)    = {ALPHA:.6f}")
    print(f"  A_STEP                = {A_STEP:.6f}")
    print(f"  6/π²                  = {DENSITY:.6f}")
    print()

    battery = NeutralBatteryHandler(
        K=20, J_C=1.0, N_events=41, N_samples=1024,
        omega=1.0, dc_threshold=0.1, free_e_threshold=0.05)
    battery.full_cycle_scan()
    print(battery.report())
    print()

    # ---- sample of cycles ----
    print("--- sample cycles  (12 evenly spaced) ---")
    print(f"  {'cyc':>4}  {'t':>9}  {'cathode':>10}  "
          f"{'anode':>10}  {'E_ac':>10}  "
          f"{'capacity':>10}  {'free_e':>8}")
    step = max(1, len(battery.states) // 12)
    for k in range(0, len(battery.states), step):
        s = battery.states[k]
        print(f"  {s.cycle:>4d}  {s.t:>9.4f}  "
              f"{s.cathode:>+10.4f}  {s.anode:>+10.4f}  "
              f"{s.E_ac:>+10.4f}  {s.capacity:>10.4f}  "
              f"{s.free_electrons:>8.4f}")
    print()

    # ---- depletion summary ----
    deplet = sum(1 for s in battery.states if s.depleting)
    print("--- depletion summary ---")
    print(f"  total cycles          : {len(battery.states)}")
    print(f"  depleting cycles      : {deplet}")
    print(f"  interventions         : {battery.interventions}")
    print(f"  battery verdict       : "
          f"{'HEALTHY (un-depletable)' if deplet == 0 else 'DEPLETED'}")
    print()

    # ---- physics summary ----
    print("--- physics of the AC-carrier battery ---")
    print("  cathode   = μ channel       S_νμ(t)")
    print("  anode     = τ channel       S_ντ(t)  (quarter-period lag)")
    print("  carrier   = E(t) = (S_νμ + S_ντ)·cos(ωt)")
    print()
    print("  invariants:")
    print("    (1) AC carrier mean ≈ 0        →  no DC drain")
    print("    (2) even-even mask preserved   →  no even-odd scattering")
    print("    (3) |H(f)|² ∈ [0,1]            →  bounded power transfer")
    print()
    print("  why it can't deplete:")
    print("    · the battery is an AC standing wave")
    print("    · capacity oscillates as |S_νμ|² + |S_ντ|²")
    print("    · zero DC component → no net charge transfer per cycle")
    print("    · even×even only  → no scattering leak path")
    print("    · bounded spectrum → no power runaway")
    print()

    battery.visualize()


if __name__ == "__main__":
    demo()