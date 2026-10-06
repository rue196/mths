#!/usr/bin/env python3
"""
mobius_audio_sampler.py
=======================

Möbius sampling bit rate for audio.

    Nominal rate      fs  = 44 100 Hz   (the "kHz" reference)
    Sample slots      K   = 1024        (one frame of the wire)
    Bits per slot     bps ∈ {8, 16, 24, 32}
    Finite-step step  A   = α / α_user = 1/(π−e) / 0.3628 ≈ 6.511
    Möbius gate       keep slot k iff  μ(k) ≠ 0
    Throughput        τ_k = Π(x_k, y0) ∈ [0, 1]  on the 1D wire path

Effective bit rate
------------------

    B = fs · bps · ρ · τ̄

where

    ρ  = (#kept slots) / K                ∈ [0, 1]   → 6/π² as K grows
    τ̄ = mean τ_k over kept slots          ∈ [0, 1]

Three reference points:

    uniform PCM       ρ = 1,   τ̄ = 1     B = fs · bps
    Möbius only       ρ ≈ 6/π², τ̄ = 1     B ≈ fs · bps · 6/π²
    Möbius + wire     ρ ≈ 6/π², τ̄ ≈ 0.5   B ≈ fs · bps · 3/π²

The finite-step derivative carries the local slope, so the receiver
integrates to reconstruct.  Fewer bits per kept slot are needed
than in a uniform PCM stream because the derivative is predictive.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ============================================================
#  Constants
# ============================================================
PI         = math.pi
E          = math.e
ALPHA_SYM  = 1.0 / (PI - E)            # ≈ 2.362
ALPHA_USER = 0.3628
A_STEP     = ALPHA_SYM / ALPHA_USER    # ≈ 6.511   finite-step step
DENSITY    = 6.0 / (PI * PI)           # ≈ 0.6079  Möbius density
W1         = PI
W2         = E


# ============================================================
#  Möbius sieve  ·  O(K log log K)
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
#  Wire throughput path  (wire-throughput-sat.py)
# ============================================================
def elliptic_projection(x: np.ndarray, y0: float = 1.0) -> np.ndarray:
    """Π(x, y0) ∈ [0, 1] on the torus ℂ/(πℤ + eℤ)."""
    u = (x / W1) % 1.0
    v = (y0 / W2) % 1.0
    return 0.5 * (1.0 + np.cos(2 * np.pi * u) * math.cos(2 * np.pi * v))


def throughput_1d(K: int) -> np.ndarray:
    """1D throughput τ_k along the wire, sampled from the elliptic projection."""
    x = np.linspace(0, W1, K, endpoint=False)
    return elliptic_projection(x, y0=0.5 * W2)


# ============================================================
#  Finite-step derivative  (fft-eliptic-finite-step-audio.py)
# ============================================================
def finite_derivative(signal: np.ndarray, step: float = A_STEP) -> np.ndarray:
    """Forward finite difference with step `step`."""
    if len(signal) < 2:
        return np.zeros_like(signal)
    diff = np.zeros_like(signal, dtype=float)
    diff[:-1] = (signal[1:] - signal[:-1]) / step
    return diff


def integrate_signal(deriv: np.ndarray, first_sample: float,
                     step: float = A_STEP) -> np.ndarray:
    """Reconstruct signal from derivative using cumulative sum."""
    return first_sample + step * np.cumsum(deriv)


# ============================================================
#  Sampling config
# ============================================================
@dataclass
class SamplingConfig:
    fs: int = 44_100                # Hz
    bps: int = 16                   # bits per sample
    K: int = 1024                   # slots per frame
    A: float = A_STEP               # finite-step derivative step
    use_mobius: bool = True         # gate slots by μ(k) ≠ 0
    use_throughput: bool = True     # weight slots by τ_k


# ============================================================
#  Möbius audio sampler
# ============================================================
class MobiusAudioSampler:
    """
    Sample audio using:
        • finite-step derivative as the construction primitive
        • Möbius sieve as the slot gate
        • wire throughput τ_k as the per-slot weight

    Bit rate:
        B = fs · bps · ρ · τ̄
    """
    def __init__(self, cfg: SamplingConfig = SamplingConfig()):
        self.cfg = cfg
        self.mu = mobius_sieve(cfg.K)
        self.tau = throughput_1d(cfg.K)

        # slot mask: all slots, or μ(k) ≠ 0
        if cfg.use_mobius:
            self.kept = [k for k in range(1, cfg.K + 1) if self.mu[k] != 0]
        else:
            self.kept = list(range(1, cfg.K + 1))
        self.kept_idx = np.array([k - 1 for k in self.kept], dtype=int)

        # throughput per slot, default 1.0 if not used
        tau_kept = (self.tau[self.kept_idx]
                    if cfg.use_throughput
                    else np.ones(len(self.kept)))
        self.tau_kept = tau_kept

    # ---------- density and throughput ----------
    def density(self) -> float:
        """ρ = |kept| / K."""
        return len(self.kept) / self.cfg.K

    def throughput_mean(self) -> float:
        """τ̄ = mean τ_k over kept slots."""
        if not len(self.tau_kept):
            return 0.0
        return float(np.mean(self.tau_kept))

    def effective_sample_rate(self) -> float:
        """fs · ρ (Hz) — the Möbius density-adjusted rate."""
        return self.cfg.fs * self.density()

    def effective_sample_rate_tau(self) -> float:
        """fs · ρ · τ̄ (Hz) — the throughput-weighted effective rate."""
        return self.cfg.fs * self.density() * self.throughput_mean()

    # ---------- bit rate ----------
    def bit_rate(self) -> float:
        """
        B = fs · bps · ρ · τ̄  (bits per second).

        When use_throughput=False, τ̄ = 1 and B = fs·bps·ρ.
        When use_mobius=False, ρ = 1.
        """
        return (self.cfg.fs
                * self.cfg.bps
                * self.density()
                * self.throughput_mean())

    def bit_rate_uniform(self) -> float:
        """Reference: uniform PCM  B = fs · bps."""
        return self.cfg.fs * self.cfg.bps

    def compression_ratio(self) -> float:
        """Bits saved relative to uniform PCM."""
        return 1.0 - self.bit_rate() / self.bit_rate_uniform()

    # ---------- sampling ----------
    def sample(self, signal: np.ndarray) -> Dict:
        """
        Sample the audio:

        1. finite-step derivative of the first K samples
        2. multiply by throughput τ_k at each kept slot
        3. discard the non-kept slots

        Returns a dict with the sampled array (sparse), the
        derivative, and the first sample (used by the integrator).
        """
        K = self.cfg.K
        sig = signal[:K]
        if len(sig) < K:
            sig = np.pad(sig, (0, K - len(sig)))

        deriv = finite_derivative(sig, self.cfg.A)
        samples = np.zeros(K, dtype=float)
        samples[self.kept_idx] = deriv[self.kept_idx] * self.tau_kept

        return dict(
            first_sample=float(sig[0]),
            samples=samples,
            deriv=deriv,
            original=sig,
        )

    # ---------- reconstruction ----------
    def reconstruct(self, sample_dict: Dict) -> np.ndarray:
        """
        Reconstruct the audio:

        1. divide by τ_k to undo the throughput weighting
        2. linear-interpolate the derivative at non-kept slots
        3. integrate with the finite-step step A
        """
        K = self.cfg.K
        samples = sample_dict["samples"]

        # deweight
        deriv_hat = np.zeros(K, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            deweighted = samples[self.kept_idx] / np.maximum(
                self.tau_kept, 1e-9)
        deriv_hat[self.kept_idx] = deweighted

        # interpolate over the missing slots
        full_idx = np.arange(K)
        if len(self.kept_idx) >= 2:
            interpolated = np.interp(full_idx,
                                     self.kept_idx,
                                     deweighted)
        else:
            interpolated = np.zeros(K)
        missing = np.setdiff1d(full_idx, self.kept_idx)
        deriv_hat[missing] = interpolated[missing]

        return integrate_signal(
            deriv_hat,
            sample_dict["first_sample"],
            self.cfg.A,
        )

    # ---------- summary ----------
    def summary(self) -> Dict:
        return dict(
            fs=self.cfg.fs,
            bps=self.cfg.bps,
            K=self.cfg.K,
            density=self.density(),
            density_theory=DENSITY,
            tau_mean=self.throughput_mean(),
            eff_fs=self.effective_sample_rate(),
            eff_fs_tau=self.effective_sample_rate_tau(),
            bit_rate=self.bit_rate(),
            bit_rate_uniform=self.bit_rate_uniform(),
            compression_ratio=self.compression_ratio(),
        )


# ============================================================
#  Test signal
# ============================================================
def make_audio(K: int, fs: int) -> np.ndarray:
    """A short piece of synthetic audio: sum of two tones + envelope."""
    t = np.arange(K) / fs
    s = (0.6 * np.sin(2 * np.pi * 440.0 * t)
         + 0.3 * np.sin(2 * np.pi * 880.0 * t + 0.7)
         + 0.1 * np.sin(2 * np.pi * 1320.0 * t + 1.1))
    env = 0.5 * (1.0 - np.cos(2 * np.pi * t / (K / fs)))
    return s * env


def snr(orig: np.ndarray, recon: np.ndarray) -> float:
    err = orig - recon
    num = float(np.sum(orig ** 2))
    den = float(np.sum(err ** 2)) + 1e-20
    return 10.0 * math.log10(num / den)


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=" * 80)
    print("Möbius sampling bit rate  ·  finite-step derivative + wire throughput")
    print("=" * 80)
    print(f"  α_sym       = 1/(π − e)   = {ALPHA_SYM:.6f}")
    print(f"  α_user      = 0.3628      = {ALPHA_USER:.4f}")
    print(f"  A (step)    = α/α_user    = {A_STEP:.6f}")
    print(f"  Möbius density target     = 6/π² = {DENSITY:.6f}")
    print()

    # ---------- configurations ----------
    configs = [
        ("uniform PCM",      SamplingConfig(use_mobius=False, use_throughput=False)),
        ("Möbius only",      SamplingConfig(use_mobius=True,  use_throughput=False)),
        ("Möbius + wire",    SamplingConfig(use_mobius=True,  use_throughput=True)),
    ]

    print("--- bit rate at 44.1 kHz, 16 bps, K = 1024 ---")
    print(f"  {'scheme':16s}  {'ρ':>7s}  {'τ̄':>7s}  "
          f"{'eff_fs (Hz)':>12s}  {'B (kbps)':>10s}  {'compress':>9s}")
    print("  " + "-" * 72)

    results = {}
    for name, cfg in configs:
        smp = MobiusAudioSampler(cfg)
        s = smp.summary()
        results[name] = (smp, s)
        print(f"  {name:16s}  {s['density']:>7.4f}  "
              f"{s['tau_mean']:>7.4f}  {s['eff_fs_tau']:>12.1f}  "
              f"{s['bit_rate']/1e3:>10.2f}  "
              f"{100 * s['compression_ratio']:>8.2f} %")
    print()

    # ---------- bit rate across bps ----------
    print("--- bit rate vs bits-per-sample (Möbius + wire) ---")
    print(f"  {'bps':>4s}  {'B (kbps)':>10s}  "
          f"{'B_uniform (kbps)':>16s}  {'ratio':>7s}")
    print("  " + "-" * 46)
    smp_wire = MobiusAudioSampler(
        SamplingConfig(use_mobius=True, use_throughput=True))
    for bps in (8, 16, 24, 32):
        cfg = SamplingConfig(bps=bps, use_mobius=True, use_throughput=True)
        smp = MobiusAudioSampler(cfg)
        B = smp.bit_rate()
        B_uniform = smp.bit_rate_uniform()
        print(f"  {bps:>4d}  {B/1e3:>10.2f}  {B_uniform/1e3:>16.2f}  "
              f"{B/B_uniform:>7.4f}")
    print()

    # ---------- bit rate across K ----------
    print("--- bit rate vs frame size K (Möbius + wire, 16 bps) ---")
    print(f"  {'K':>6s}  {'ρ':>7s}  {'τ̄':>7s}  "
          f"{'B (kbps)':>10s}  {'ρ/6·π⁻²':>10s}")
    print("  " + "-" * 52)
    for K in (128, 256, 512, 1024, 2048, 4096):
        cfg = SamplingConfig(K=K, use_mobius=True, use_throughput=True)
        smp = MobiusAudioSampler(cfg)
        s = smp.summary()
        print(f"  {K:>6d}  {s['density']:>7.4f}  {s['tau_mean']:>7.4f}  "
              f"{s['bit_rate']/1e3:>10.2f}  "
              f"{s['density']/DENSITY:>10.4f}")
    print()

    # ---------- reconstruction ----------
    print("--- reconstruction: SNR at matched bit rate ---")
    K = 1024
    cfg_uni = SamplingConfig(K=K, bps=16,
                             use_mobius=False, use_throughput=False)
    cfg_mob = SamplingConfig(K=K, bps=16,
                             use_mobius=True,  use_throughput=True)
    smp_uni = MobiusAudioSampler(cfg_uni)
    smp_mob = MobiusAudioSampler(cfg_mob)

    sig = make_audio(K, cfg_uni.fs)

    r_uni = smp_uni.sample(sig)
    r_mob = smp_mob.sample(sig)

    rec_uni = smp_uni.reconstruct(r_uni)
    rec_mob = smp_mob.reconstruct(r_mob)

    snr_uni = snr(sig, rec_uni)
    snr_mob = snr(sig, rec_mob)

    print(f"  scheme           SNR (dB)   B (kbps)")
    print(f"  uniform PCM      {snr_uni:>8.2f}   "
          f"{smp_uni.bit_rate()/1e3:>8.2f}")
    print(f"  Möbius + wire    {snr_mob:>8.2f}   "
          f"{smp_mob.bit_rate()/1e3:>8.2f}")
    print(f"  Δ                {snr_mob - snr_uni:>+8.2f}   "
          f"{(smp_mob.bit_rate() - smp_uni.bit_rate())/1e3:>+8.2f}")
    print()

    # ---------- finite-step derivative statistics ----------
    deriv = r_mob["deriv"]
    print("--- finite-step derivative stats (K = 1024) ---")
    print(f"  |deriv|  mean      = {np.mean(np.abs(deriv)):.6e}")
    print(f"  |deriv|  max       = {np.max(np.abs(deriv)):.6e}")
    print(f"  deriv    RMS       = {np.sqrt(np.mean(deriv**2)):.6e}")
    print(f"  original RMS       = {np.sqrt(np.mean(sig**2)):.6e}")
    print(f"  step A             = {A_STEP:.6f}")
    print()

    # ---------- complexity ----------
    print("--- complexity ---")
    print("  Möbius sieve            O(K log log K)   once")
    print("  Throughput path         O(K)")
    print("  Finite-step derivative  O(K)             per frame")
    print("  Sample                  O(|kept|)")
    print("  Reconstruct             O(K)")
    print("  ─────────────────────────────────────")
    print("  per frame total         O(K)")

    # ---------- plot ----------
    if HAS_MPL:
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.35)

        # (a) original signal
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(sig[:256], color="#2c3e50", lw=1.2)
        ax.set_xlabel("slot k"); ax.set_ylabel("amplitude")
        ax.set_title("Original audio (first 256 slots)")
        ax.grid(alpha=0.3)

        # (b) finite-step derivative
        ax = fig.add_subplot(gs[0, 1])
        ax.plot(r_mob["deriv"][:256], color="#c0392b", lw=1.0)
        ax.set_xlabel("slot k"); ax.set_ylabel("d/dt")
        ax.set_title(f"Finite-step derivative (step = {A_STEP:.3f})")
        ax.grid(alpha=0.3)

        # (c) throughput path
        ax = fig.add_subplot(gs[0, 2])
        ax.plot(smp_mob.tau, color="#16a085", lw=1.2)
        ax.fill_between(np.arange(K), 0, smp_mob.tau,
                        color="#16a085", alpha=0.2)
        ax.set_xlabel("slot k"); ax.set_ylabel("τ_k")
        ax.set_title("Wire throughput  τ_k = Π(x_k, y₀)")
        ax.grid(alpha=0.3)

        # (d) Möbius slot mask
        ax = fig.add_subplot(gs[1, 0])
        mask = np.zeros(K)
        mask[smp_mob.kept_idx] = 1.0
        ax.plot(mask[:256], color="#8e44ad", lw=0.8)
        ax.set_xlabel("slot k"); ax.set_ylabel("μ(k) ≠ 0 ?")
        ax.set_title(f"Möbius slot gate  (density = "
                     f"{smp_mob.density():.4f})")
        ax.grid(alpha=0.3)

        # (e) samples (Möbius + throughput)
        ax = fig.add_subplot(gs[1, 1])
        ax.stem(np.arange(256), r_mob["samples"][:256],
                linefmt="C3-", markerfmt="C3.", basefmt="k-")
        ax.set_xlabel("slot k"); ax.set_ylabel("sample value")
        ax.set_title(f"Möbius-sampled derivative  "
                     f"({len(smp_mob.kept)} kept of {K})")
        ax.grid(alpha=0.3)

        # (f) reconstruction overlay
        ax = fig.add_subplot(gs[1, 2])
        ax.plot(sig[:256], color="#2c3e50", lw=1.2, label="original")
        ax.plot(rec_mob[:256], color="#e74c3c", lw=1.0, ls="--",
                label="recon (Möbius+wire)")
        ax.plot(rec_uni[:256], color="#3498db", lw=0.8, ls=":",
                label="recon (uniform)")
        ax.set_xlabel("slot k"); ax.set_ylabel("amplitude")
        ax.set_title(f"Reconstruction  ·  SNR_mob = {snr_mob:.1f} dB, "
                     f"SNR_uni = {snr_uni:.1f} dB")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # (g) bit rate vs bps
        ax = fig.add_subplot(gs[2, 0])
        bps_list = [8, 16, 24, 32]
        B_mob = [MobiusAudioSampler(
                    SamplingConfig(bps=b, use_mobius=True,
                                   use_throughput=True)).bit_rate()/1e3
                 for b in bps_list]
        B_uni = [fs * b / 1e3
                 for b in bps_list for fs in [44_100]]
        B_uni = [44_100 * b / 1e3 for b in bps_list]
        ax.plot(bps_list, B_uni, "o-", color="#3498db", label="uniform")
        ax.plot(bps_list, B_mob, "s-", color="#e74c3c",
                label="Möbius + wire")
        ax.set_xlabel("bits per sample"); ax.set_ylabel("B (kbps)")
        ax.set_title("Bit rate vs bps")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

        # (h) bit rate vs K
        ax = fig.add_subplot(gs[2, 1])
        Ks = [64, 128, 256, 512, 1024, 2048, 4096]
        rhos = []
        for K in Ks:
            cfg = SamplingConfig(K=K, use_mobius=True,
                                 use_throughput=True)
            rhos.append(MobiusAudioSampler(cfg).density())
        ax.plot(Ks, rhos, "o-", color="#16a085")
        ax.axhline(DENSITY, color="k", ls=":",
                   label=f"6/π² = {DENSITY:.4f}")
        ax.set_xscale("log")
        ax.set_xlabel("K (slots)"); ax.set_ylabel("ρ")
        ax.set_title("Möbius density → 6/π²")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

        # (i) bit rate breakdown
        ax = fig.add_subplot(gs[2, 2])
        ax.axis("off")
        s = smp_mob.summary()
        ax.text(0.0, 1.0,
                f"Möbius sampler  (K = {s['K']})\n\n"
                f"  fs            = {s['fs']} Hz\n"
                f"  bps           = {s['bps']}\n"
                f"  A (step)      = {A_STEP:.4f}\n\n"
                f"  ρ  (density)  = {s['density']:.4f}\n"
                f"  τ̄  (through.) = {s['tau_mean']:.4f}\n"
                f"  eff fs (ρ)    = {s['eff_fs']:.1f} Hz\n"
                f"  eff fs (ρ·τ̄)  = {s['eff_fs_tau']:.1f} Hz\n\n"
                f"  B uniform     = "
                f"{s['bit_rate_uniform']/1e3:.2f} kbps\n"
                f"  B Möbius      = "
                f"{s['bit_rate']/1e3:.2f} kbps\n"
                f"  compression   = "
                f"{100*s['compression_ratio']:.2f} %",
                va="top", family="monospace", fontsize=10)

        plt.suptitle(
            "Möbius audio sampler  ·  finite-step derivative  ·  "
            "wire throughput  ·  44.1 kHz",
            fontsize=13)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

    print()
    print("Done.")


if __name__ == "__main__":
    demo()