#!/usr/bin/env python3
"""
elliptic_mobius_fourier.py

Elliptic Möbius power spectrum as a Fourier-like series.
Computes ζ(t) = Σ C_i * exp(i * t * i / α) for many t,
and then analyses |ζ(t)|² via FFT.
Includes Basel checksum error detection.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)          # ≈ 2.362

# ---------- Möbius sieve ----------
def mobius_sieve(K):
    mu = [0] * (K + 1)
    mu[1] = 1
    primes = []
    is_comp = [False] * (K + 1)
    for i in range(2, K + 1):
        if not is_comp[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K:
                break
            is_comp[i * p] = True
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    return mu

def build_elliptic_coeffs(K, smooth=True):
    """
    Build C_i for i = -K..K.
    If smooth=True, fill zero indices via linear interpolation.
    Returns: (coeffs, full_array)
    """
    mu = mobius_sieve(K)
    c = np.zeros(2*K + 1, dtype=float)
    # First assign the non‑zero μ values
    for i in range(-K, K+1):
        if i == 0:
            c[i + K] = 0.0
        else:
            c[i + K] = mu[abs(i)]

    if smooth:
        # For each zero index, interpolate between nearest non‑zero neighbours
        for i in range(-K, K+1):
            if i == 0:
                continue
            if c[i + K] == 0.0:
                left = i - 1
                right = i + 1
                while left >= -K and c[left + K] == 0.0:
                    left -= 1
                while right <= K and c[right + K] == 0.0:
                    right += 1
                if left < -K or right > K:
                    continue
                left_val = c[left + K]
                right_val = c[right + K]
                dist = right - left
                if dist == 0:
                    continue
                weight_left = (right - i) / dist
                weight_right = (i - left) / dist
                c[i + K] = weight_left * left_val + weight_right * right_val

    coeffs = {i: c[i + K] for i in range(-K, K+1) if abs(c[i + K]) > 1e-12}
    return coeffs, c

# ---------- Elliptic gate ----------
class EllipticMobiusGate:
    def __init__(self, K, smooth=True):
        self.K = K
        self.coeffs, self.full_array = build_elliptic_coeffs(K, smooth)
        self.indices = np.array(sorted(self.coeffs.keys()))
        self.period = 2 * PI * ALPHA
        self.smooth = smooth

    def zeta(self, t):
        """ζ(t) = Σ C_i * exp(i * t * i / α)."""
        if len(self.indices) == 0:
            return 0.0 + 0.0j
        phases = t * self.indices / ALPHA
        vals = np.array([self.coeffs[i] for i in self.indices])
        return np.sum(vals * np.exp(1j * phases))

    def power_spectrum(self, t):
        return np.abs(self.zeta(t))**2

    def power_spectrum_series(self, N_points=1000):
        """Return t_vals and power_vals for N_points equally spaced t in [0, period]."""
        t_vals = np.linspace(0, self.period, N_points)
        power_vals = np.array([self.power_spectrum(t) for t in t_vals])
        return t_vals, power_vals

    def integrate_power_spectrum(self, N=1000):
        t_vals, power = self.power_spectrum_series(N)
        try:
            integral = np.trapezoid(power, t_vals)
        except AttributeError:
            integral = np.trapz(power, t_vals)
        avg = integral / self.period
        return integral, avg

    def basel_theoretical(self):
        return 2 * (6 / (PI * PI)) * self.K

    def basel_error(self, N=1000):
        _, avg = self.integrate_power_spectrum(N)
        theo = self.basel_theoretical()
        return abs(avg - theo) / theo if theo != 0 else 0.0

    def check_basel(self, tolerance=1e-2, N=1000):
        err = self.basel_error(N)
        return err < tolerance, err

    def fourier_analysis(self, N_points=1000):
        """
        Compute the FFT of the power spectrum and return frequencies and magnitudes.
        """
        t_vals, power = self.power_spectrum_series(N_points)
        dt = t_vals[1] - t_vals[0]
        # FFT (real)
        fft_vals = np.fft.rfft(power)
        freqs = np.fft.rfftfreq(N_points, dt)
        magnitudes = np.abs(fft_vals)
        return freqs, magnitudes, t_vals, power

# ---------- Demo ----------
def demo():
    K = 40
    print("Elliptic Möbius Fourier Series (smooth = True)")
    gate = EllipticMobiusGate(K, smooth=True)

    # Basel check
    ok, err = gate.check_basel(tolerance=0.02, N=2000)
    print(f"Basel error: {err:.6f}  -> {'OK' if ok else 'FAIL'}")

    # Power spectrum
    t_vals, power = gate.power_spectrum_series(N_points=500)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(t_vals, power)
    plt.xlabel('t')
    plt.ylabel('|ζ(t)|²')
    plt.title('Power spectrum (time domain)')
    plt.grid(True)

    # FFT analysis
    freqs, mags, _, _ = gate.fourier_analysis(N_points=500)
    plt.subplot(1, 2, 2)
    plt.plot(freqs, mags)
    plt.xlabel('Frequency (1/t)')
    plt.ylabel('FFT magnitude')
    plt.title('Fourier spectrum of |ζ(t)|²')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Also print first few Fourier frequencies
    print("\nFirst 5 Fourier frequencies and magnitudes:")
    for f, m in zip(freqs[:5], mags[:5]):
        print(f"  f = {f:.4f}, mag = {m:.4f}")

if __name__ == "__main__":
    demo()