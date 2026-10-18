#!/usr/bin/env python3
"""
spectral_density_mobius.py
==========================

Spectral density estimation on the elliptic Möbius gate.

Model
-----
    |ζ(t)|²               amplitude envelope  across time
    L(t) = ∫ √(dRe²+dIm²) arc length of ζ in the complex plane
    peaks(L)              resonances
    muon  (cosine split)  peaks where Re(ζ) > 0     (σ_μ = +1)
    tau   (sine   split)  peaks where Im(ζ) > 0     (σ_τ = −1)
    smoothing             across zeros via DCMobiusGate(smooth=True)
    middle band           electron point-particle density  ρ_elec(t)
    E(t)                  alternating AC field  (wire_analog.py)
    P(f)                  spectral density at arc-length resonances
"""

from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from DC_chirp import DCMobiusGate, mobius_sieve
from wire_analog import alternating_E_field, pi_e, spin_grade


# ============================================================
#  Constants
# ============================================================
PI      = math.pi
E       = math.e
ALPHA   = 1.0 / (PI - E)
A_STEP  = ALPHA / 0.3628
DENSITY = 6.0 / (PI * PI)


# ============================================================
#  SpectralDensityMobius
# ============================================================
class SpectralDensityMobius:
    """
    Spectral density estimator built on the elliptic Möbius gate.

        |ζ(t)|²     amplitude    → power envelope in time
        L(t)        arc length   → resonance coordinate
        peaks(L)                 → discrete spectral peaks
        cos/sin     chirality    → μ / τ channel split
        middle band              → electron point particle density
        E(t)        AC drive     → alternating (ω) carrier
    """

    def __init__(self,
                 K: int = 20,
                 J_C: float = 1.0,
                 omega: float = 1.0,
                 N_samples: int = 512,
                 smooth: bool = True):
        self.K = int(K)
        self.J_C = float(J_C)
        self.omega = float(omega)
        self.N_samples = int(N_samples)

        # ---- 1. elliptic Möbius gate  (smoothed across zeros) ----
        self.gate = DCMobiusGate(K=K, smooth=smooth)
        self.T = self.gate.period

        # ---- 2. sample ζ(t) over one period ----
        self.t = np.linspace(0.0, self.T, self.N_samples)
        self.zeta = np.array([self.gate.zeta(ti) for ti in self.t])

        # ---- 3. |ζ(t)|²  amplitude envelope ----
        self.power = np.abs(self.zeta) ** 2

        # ---- 4. arc length L(t) along ζ in the complex plane ----
        dRe = np.diff(self.zeta.real, prepend=self.zeta.real[0])
        dIm = np.diff(self.zeta.imag, prepend=self.zeta.imag[0])
        ds = np.sqrt(dRe ** 2 + dIm ** 2)
        self.arc = np.cumsum(ds)
        self.arc /= (self.arc[-1] + 1e-12)               # normalised [0, 1]

        # ---- 5. peaks of the arc length ----
        self.peak_idx = self._find_peaks(self.arc)

        # ---- 6. muon / tau split of peaks ----
        self.muon_idx, self.tau_idx = self._split_peaks()

        # ---- 7. alternating AC E-field  (wire_analog) ----
        self._build_efield()

        # ---- 8. electron point-particle cloud in middle band ----
        self._build_electron_cloud()

        # ---- 9. spectral density at arc-length resonances ----
        self._build_spectrum()

    # --------------------------------------------------------
    def _find_peaks(self, arr: np.ndarray) -> np.ndarray:
        """Simple local-maximum detector on a smoothed profile."""
        # smooth slightly so we pick the envelope peaks
        ker = np.ones(5) / 5.0
        smooth = np.convolve(arr, ker, mode="same")
        peaks = []
        for i in range(2, len(smooth) - 2):
            if (smooth[i] > smooth[i - 1] and smooth[i] > smooth[i + 1]
                    and smooth[i] > smooth[i - 2]
                    and smooth[i] > smooth[i + 2]):
                peaks.append(i)
        return np.array(peaks, dtype=int)

    # --------------------------------------------------------
    def _split_peaks(self):
        """Classify each peak as muon (cos-dominant) or tau (sin-dominant)."""
        mu, ta = [], []
        for pk in self.peak_idx:
            z = self.zeta[pk]
            if abs(z) < 1e-12:
                continue
            # phase: cos-dominant → muon, sin-dominant → tau
            if z.real >= z.imag:
                mu.append(int(pk))
            else:
                ta.append(int(pk))
        return np.array(mu, dtype=int), np.array(ta, dtype=int)

    # --------------------------------------------------------
    def _build_efield(self):
        """Alternating AC E-field from wire_analog.py."""
        half = self.N_samples // 2
        r_samp = np.arange(1, half + 1, dtype=float)
        S_mu, S_tau = alternating_E_field(
            self.t[:half], r_samp, J_C=self.J_C)
        self.S_mu = float(S_mu)
        self.S_tau = float(S_tau)
        self.E_mu = S_mu * np.cos(self.omega * self.t)
        self.E_tau = S_tau * np.cos(self.omega * self.t)
        self.E_total = self.E_mu + self.E_tau

    # --------------------------------------------------------
    def _build_electron_cloud(self):
        """
        Point-particle density in the middle band:

            t ∈ [T/4, 3T/4]

            ρ_elec(t) = |ζ(t)|² · Π_e(t, r_eff) · middle(t)
        """
        T = self.T
        in_middle = (self.t > 0.25 * T) & (self.t < 0.75 * T)
        rho = np.zeros_like(self.power)
        for i in range(self.N_samples):
            if not in_middle[i]:
                continue
            r_eff = self.t[i] / T * PI
            pe = pi_e(self.t[i], r_eff)
            rho[i] = self.power[i] * pe

        self.electron_density = rho
        # sample point particles from the density
        prob = rho / (rho.sum() + 1e-12)
        rng = np.random.default_rng(2024)
        n_pts = 200
        self.electron_t = rng.choice(
            self.t, size=n_pts, p=prob)
        self.electron_weight = np.interp(
            self.electron_t, self.t, rho)

    # --------------------------------------------------------
    def _build_spectrum(self):
        """
        Spectral density at arc-length resonances.

            f_k = 1 / (λ_k + ε)       λ_k = local arc wavelength
            w_k = chirality weight    muon: cos(phase), tau: sin(phase)
            P_k = |ζ(t_k)|² · w_k · E_k
        """
        if self.peak_idx.size == 0:
            self.spec_f = np.array([])
            self.spec_P = np.array([])
            self.spec_channel = np.array([], dtype=object)
            self.psd_f = np.array([])
            self.psd_P = np.array([])
            return

        s_mu = spin_grade(self.J_C, 0.0)
        s_tau = spin_grade(self.J_C, 1.0)

        f_list, P_list, ch_list = [], [], []
        prev_arc = 0.0
        for pk in self.peak_idx:
            lam = float(self.arc[pk] - prev_arc)
            prev_arc = float(self.arc[pk])
            f_k = 1.0 / (lam + 1e-6)

            z = self.zeta[pk]
            mag = abs(z) + 1e-12
            w_mu = max(0.0, z.real / mag) * abs(s_mu)
            w_tau = max(0.0, z.imag / mag) * abs(s_tau)

            P_k = float(self.power[pk]) * (w_mu + w_tau)
            channel = "muon" if w_mu >= w_tau else "tau"

            f_list.append(f_k)
            P_list.append(P_k)
            ch_list.append(channel)

        self.spec_f = np.array(f_list)
        self.spec_P = np.array(P_list)
        self.spec_channel = np.array(ch_list, dtype=object)
        self.spec_muon_mask = (self.spec_channel == "muon")
        self.spec_tau_mask  = (self.spec_channel == "tau")

        # Gaussian-broadened PSD on log-frequency
        if self.spec_f.size > 0:
            fmin = max(self.spec_f.min(), 1e-3)
            fmax = max(self.spec_f.max(), fmin * 2)
            self.psd_f = np.logspace(
                math.log10(fmin),
                math.log10(fmax), 256)
            self.psd_P = np.zeros_like(self.psd_f)
            sigma = 0.15
            for f0, P0 in zip(self.spec_f, self.spec_P):
                self.psd_P += P0 * np.exp(
                    -0.5 * ((np.log10(self.psd_f)
                             - math.log10(max(f0, 1e-9))) / sigma) ** 2)

    # --------------------------------------------------------
    def report(self) -> str:
        L = ["=" * 76,
             "Spectral density  ·  elliptic Möbius gate",
             "=" * 76]
        L.append(f"  K                  : {self.K}")
        L.append(f"  J_C                : {self.J_C}")
        L.append(f"  σ_μ(J_C)           : {spin_grade(self.J_C, 0.0):+.0f}")
        L.append(f"  σ_τ(J_C)           : {spin_grade(self.J_C, 1.0):+.0f}")
        L.append(f"  ω (carrier)        : {self.omega:.4f}")
        L.append(f"  period             : {self.T:.6f}")
        L.append(f"  samples            : {self.N_samples}")
        L.append("")
        L.append(f"  DC throughput      : {self.gate.dc_throughput():.6f}")
        S, H, m = self.gate.supertrace()
        L.append(f"  supertrace S       : {S:+.6f}")
        L.append(f"  entropy    H       : {H:.6f}")
        L.append(f"  mass       m       : {m:.4e}")
        L.append("")
        L.append(f"  arc length peaks   : {self.peak_idx.size}")
        L.append(f"    muon (cosine)    : {self.muon_idx.size}")
        L.append(f"    tau  (sine)      : {self.tau_idx.size}")
        L.append("")
        L.append(f"  ⟨|ζ|²⟩             : {self.power.mean():.6f}")
        L.append(f"  |ζ|² max           : {self.power.max():.6f}")
        L.append("")
        L.append(f"  E amplitude        : {abs(self.E_total).max():.4e}")
        L.append(f"  S_νμ + S_ντ        : {self.S_mu + self.S_tau:+.4e}")
        L.append("")
        L.append(f"  electron cloud     : {self.electron_t.size} particles")
        L.append(f"  middle band        : "
                 f"[{0.25 * self.T:.3f}, {0.75 * self.T:.3f}]")
        L.append(f"  ⟨ρ_elec⟩ (middle)  : "
                 f"{self.electron_density.mean():.6e}")
        L.append("")
        if self.spec_f.size > 0:
            L.append(f"  spectral peaks     : {self.spec_f.size}")
            L.append(f"    f range          : "
                     f"[{self.spec_f.min():.3f}, {self.spec_f.max():.3f}]")
            L.append(f"    P range          : "
                     f"[{self.spec_P.min():.4e}, {self.spec_P.max():.4e}]")
            L.append(f"    muon peaks       : {int(self.spec_muon_mask.sum())}")
            L.append(f"    tau peaks        : {int(self.spec_tau_mask.sum())}")
        return "\n".join(L)

    # --------------------------------------------------------
    def visualize(self) -> None:
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.36, wspace=0.30)

        # (a) amplitude |ζ|² + arc length
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(self.t, self.power, color="#3498db", lw=1.2,
                label=r"$|\zeta(t)|^2$  amplitude")
        ax2 = ax.twinx()
        ax2.plot(self.t, self.arc, color="#e67e22", lw=1.0,
                 ls="--", alpha=0.8, label="arc length L(t)")
        ax2.set_ylabel("L(t)", color="#e67e22")
        ax.set_xlabel("t"); ax.set_ylabel("|ζ|²", color="#3498db")
        ax.set_title("Amplitude envelope  +  arc length")
        ax.grid(alpha=0.3)

        # (b) arc length peaks split into muon/tau
        ax = fig.add_subplot(gs[0, 1])
        ax.plot(self.t, self.arc, color="#2c3e50", lw=1.1,
                alpha=0.7, label="arc length")
        if self.muon_idx.size:
            ax.scatter(self.t[self.muon_idx], self.arc[self.muon_idx],
                       color="#e74c3c", s=55, zorder=5,
                       label=f"μ peaks ({self.muon_idx.size})",
                       edgecolors="k", linewidths=0.5)
        if self.tau_idx.size:
            ax.scatter(self.t[self.tau_idx], self.arc[self.tau_idx],
                       color="#3498db", s=55, zorder=5,
                       label=f"τ peaks ({self.tau_idx.size})",
                       marker="s", edgecolors="k", linewidths=0.5)
        ax.set_xlabel("t"); ax.set_ylabel("arc length L(t)")
        ax.set_title("Arc length peaks  ·  μ / τ split")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # (c) electron point particles in the middle band
        ax = fig.add_subplot(gs[0, 2])
        mid_mask = (self.t > 0.25 * self.T) & (self.t < 0.75 * self.T)
        ax.fill_between(self.t, 0, self.electron_density,
                        color="#8e44ad", alpha=0.25,
                        label="ρ_elec(t) density")
        ax.scatter(self.electron_t, self.electron_weight,
                   color="#8e44ad", s=12, alpha=0.65,
                   edgecolors="k", linewidths=0.3,
                   label=f"point particles ({self.electron_t.size})")
        ax.axvspan(0.25 * self.T, 0.75 * self.T,
                   color="gray", alpha=0.08, zorder=0)
        ax.set_xlabel("t"); ax.set_ylabel("electron density")
        ax.set_title("Electron point-particle cloud  ·  middle band")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # (d) alternating AC E-field
        ax = fig.add_subplot(gs[1, 0])
        ax.plot(self.t, self.E_mu, "b-", lw=1.0, alpha=0.7,
                label=rf"$S_{{\nu\mu}}\cos(\omega t)$")
        ax.plot(self.t, self.E_tau, "r--", lw=1.0, alpha=0.7,
                label=rf"$S_{{\nu\tau}}\cos(\omega t)$")
        ax.plot(self.t, self.E_total, "k-", lw=1.4,
                label=r"$E(t)$")
        ax.axhline(0, color="gray", lw=0.4)
        ax.set_xlabel("t"); ax.set_ylabel("E(t)")
        ax.set_title("Alternating AC E-field  (wire_analog)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # (e) discrete spectral peaks coloured by μ / τ
        ax = fig.add_subplot(gs[1, 1])
        if self.spec_f.size > 0:
            f = self.spec_f
            P = self.spec_P
            if self.spec_muon_mask.any():
                ax.scatter(f[self.spec_muon_mask],
                           P[self.spec_muon_mask],
                           color="#e74c3c", s=55, marker="o",
                           edgecolors="k", linewidths=0.5,
                           label=f"μ ({int(self.spec_muon_mask.sum())})")
            if self.spec_tau_mask.any():
                ax.scatter(f[self.spec_tau_mask],
                           P[self.spec_tau_mask],
                           color="#3498db", s=55, marker="s",
                           edgecolors="k", linewidths=0.5,
                           label=f"τ ({int(self.spec_tau_mask.sum())})")
            for f0, P0 in zip(f, P):
                ax.plot([f0, f0], [0, P0],
                        color="gray", lw=0.4, alpha=0.4)
            ax.set_xscale("log")
        ax.set_xlabel("frequency  f = 1/λ")
        ax.set_ylabel("spectral density  P(f)")
        ax.set_title("Spectral peaks  ·  arc-length resonances")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

        # (f) smooth PSD (Gaussian-broadened, log-log)
        ax = fig.add_subplot(gs[1, 2])
        if self.psd_f.size > 0:
            ax.loglog(self.psd_f, self.psd_P + 1e-12,
                      color="#2c3e50", lw=1.4,
                      label="PSD (broadened)")
            if self.spec_f.size > 0:
                ax.scatter(self.spec_f, self.spec_P + 1e-12,
                           color="#e74c3c", s=20, zorder=5,
                           label="peak locations")
        ax.set_xlabel("frequency  f")
        ax.set_ylabel("PSD  P(f)")
        ax.set_title("Smooth spectral density")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

        plt.suptitle(
            f"Spectral density on the elliptic Möbius gate  ·  "
            f"|ζ|² amplitude  ·  μ/τ chirality  ·  electron cloud  ·  "
            f"AC E-field  ·  α = {ALPHA:.4f}",
            fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 76)
    print("Spectral density  ·  elliptic Möbius gate")
    print("=" * 76)
    print(f"  α      = 1/(π − e)    = {ALPHA:.6f}")
    print(f"  A_STEP                = {A_STEP:.6f}")
    print(f"  6/π²                  = {DENSITY:.6f}")
    print()

    est = SpectralDensityMobius(
        K=20, J_C=1.0, omega=1.0,
        N_samples=512, smooth=True)
    print(est.report())
    print()

    # ---- print a few peaks with channel labels ----
    if est.spec_f.size > 0:
        print("--- spectral peaks  (first 10) ---")
        print(f"  {'k':>3}  {'f_k':>10}  {'P_k':>12}  {'ch':>5}")
        for k in range(min(10, est.spec_f.size)):
            print(f"  {k:>3d}  {est.spec_f[k]:>10.4f}  "
                  f"{est.spec_P[k]:>12.4e}  "
                  f"{est.spec_channel[k]:>5s}")
        print()

    # ---- summary ----
    print("--- summary ---")
    print("  |ζ(t)|²               amplitude envelope in time")
    print("  L(t) = ∫ √(dRe²+dIm²) arc length of ζ")
    print("  peaks(L)              resonances")
    print("  muon peaks            Re(ζ) ≥ Im(ζ)   (cosine split)")
    print("  tau  peaks            Im(ζ) >  Re(ζ)  (sine split)")
    print("  smoothing             DCMobiusGate smooth=True")
    print("  electron band         t ∈ [T/4, 3T/4]")
    print("  ρ_elec(t)             |ζ|² · Π_e(t, r_eff)")
    print("  E(t)                  (S_νμ + S_ντ) cos(ω t)")
    print("  P(f)                  |ζ(t_k)|² · chirality · E_k")
    print()

    est.visualize()


if __name__ == "__main__":
    demo()