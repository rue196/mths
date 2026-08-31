#!/usr/bin/env python3
"""
elliptic_mobius_gate.py

Möbius elliptic curve gate with smooth projection for μ=0.
Coefficients:
  - For i where μ(|i|) ≠ 0: C_i = μ(|i|)
  - For i where μ(|i|) = 0: C_i = linear interpolation from nearest neighbours
Uses α = 1/(π−e) in the spectral sum.
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
                # find left and right nearest non‑zero
                left = i - 1
                right = i + 1
                while left >= -K and c[left + K] == 0.0:
                    left -= 1
                while right <= K and c[right + K] == 0.0:
                    right += 1
                if left < -K or right > K:
                    # if no neighbour, keep 0
                    continue
                # linear interpolation
                left_val = c[left + K]
                right_val = c[right + K]
                # weight by distance
                dist = right - left
                if dist == 0:
                    continue
                weight_left = (right - i) / dist
                weight_right = (i - left) / dist
                c[i + K] = weight_left * left_val + weight_right * right_val

    # collect only non‑zero coefficients (including smoothed zeros that may become non‑zero)
    coeffs = {i: c[i + K] for i in range(-K, K+1) if abs(c[i + K]) > 1e-12}
    return coeffs, c

# ---------- Elliptic gate ----------
class EllipticMobiusGate:
    def __init__(self, K, smooth=True):
        self.K = K
        self.coeffs, self.full_array = build_elliptic_coeffs(K, smooth)
        self.indices = np.array(sorted(self.coeffs.keys()))
        self.period = 2 * PI * ALPHA

    def zeta(self, t):
        """ζ(t) = Σ C_i * exp(i * t * i / α)."""
        if len(self.indices) == 0:
            return 0.0 + 0.0j
        phases = t * self.indices / ALPHA
        vals = np.array([self.coeffs[i] for i in self.indices])
        return np.sum(vals * np.exp(1j * phases))

    def power_spectrum(self, t):
        z = self.zeta(t)
        return np.abs(z)**2

    def integrate_power_spectrum(self, N=1000):
        t_vals = np.linspace(0, self.period, N)
        power = np.array([self.power_spectrum(t) for t in t_vals])
        # Use np.trapezoid (new name) or fallback to np.trapz
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

# ---------- Demo ----------
def demo():
    K = 20
    print("Elliptic Möbius Gate (smooth projection for μ=0)")
    gate = EllipticMobiusGate(K, smooth=True)

    # Show coefficients (only non‑zero)
    print(f"K = {K}, number of non‑zero coeffs = {len(gate.indices)}")
    print("First 10 non‑zero coefficients (i -> C_i):")
    for i in list(gate.indices)[:10]:
        print(f"  C_{i} = {gate.coeffs[i]:.4f}")

    # Integrate
    integral, avg = gate.integrate_power_spectrum(N=2000)
    print(f"\nIntegral over [0, {gate.period:.3f}] : {integral:.6f}")
    print(f"Average |ζ(t)|² : {avg:.6f}")

    # Basel
    theo = gate.basel_theoretical()
    print(f"Theoretical average (2*(6/π²)*K): {theo:.6f}")
    ok, err = gate.check_basel(tolerance=0.02, N=2000)
    print(f"Basel error: {err:.6f}  -> {'OK' if ok else 'FAIL'}")

    # Plot spectrum
    t_vals = np.linspace(0, gate.period, 500)
    power_vals = [gate.power_spectrum(t) for t in t_vals]
    plt.figure(figsize=(10,4))
    plt.plot(t_vals, power_vals)
    plt.axhline(avg, color='r', linestyle='--', label=f'Average = {avg:.4f}')
    plt.axhline(theo, color='g', linestyle=':', label=f'Basel = {theo:.4f}')
    plt.xlabel('t')
    plt.ylabel('|ζ(t)|²')
    plt.title(f'Elliptic Möbius Power Spectrum (K={K})')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    demo()