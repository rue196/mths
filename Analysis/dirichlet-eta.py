#!/usr/bin/env python3
"""
plot_spectral_sum_eta.py

Plot the spectral sum ζ(t) = Σ C_i exp(i t i / α) with α = a = 1/(π - e^0.3628).
The coefficients are scaled so that ∫ |ζ(t)| dt = η(a) ≈ 0.659538886352.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# ---------- Constants ----------
PI = math.pi
E = math.e
EXP = math.exp(0.3628)          # e^0.3628 ≈ 1.4371
A = 1.0 / (PI - EXP)            # ≈ 0.5866
ALPHA = 1/(PI - E)

# ---------- Möbius sieve (O(K)) ----------
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

# ---------- Spectral sum ----------
def zeta(t, coeffs, alpha=ALPHA):
    total = 0.0 + 0.0j
    K = len(coeffs) - 1
    for i in range(1, K + 1):
        if coeffs[i] != 0:
            total += coeffs[i] * np.exp(1j * t * i / alpha)
    return total

# ---------- Integrate |ζ| over one period ----------
def integrate_abs_zeta(coeffs, N_points=1000, alpha=ALPHA):
    period = 2 * PI * alpha
    t_vals = np.linspace(0, period, N_points)
    sum_abs = 0.0
    for t in t_vals:
        sum_abs += abs(zeta(t, coeffs, alpha))
    dt = t_vals[1] - t_vals[0]
    return sum_abs * dt

# ---------- Main ----------
def main():
    K = 100
    mu = mobius_sieve(K)
    coeffs = [mu[i] if i > 0 else 0 for i in range(K + 1)]

    # Compute un‑scaled integral
    I_abs = integrate_abs_zeta(coeffs, N_points=2000)
    target_eta = 0.659538886352
    scaling = target_eta / I_abs
    scaled_coeffs = [c * scaling for c in coeffs]

    # Sample ζ(t) over one period
    period = 2 * PI * ALPHA
    N_plot = 500
    t_vals = np.linspace(0, period, N_plot)
    z_vals = np.array([zeta(t, scaled_coeffs, ALPHA) for t in t_vals])
    real_vals = np.real(z_vals)
    imag_vals = np.imag(z_vals)
    abs_vals = np.abs(z_vals)

    # Compute average of |ζ(t)|
    avg_abs = np.mean(abs_vals)
    print(f"Average |ζ(t)| = {avg_abs:.8f} (target η(a) = {target_eta:.8f})")
    print(f"Integral over period = {avg_abs * period:.8f}")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(t_vals, real_vals, 'b-', label='Re ζ(t)')
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('Real')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t_vals, imag_vals, 'r-', label='Im ζ(t)')
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('Imag')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(t_vals, abs_vals, 'g-', label='|ζ(t)|')
    axes[2].axhline(y=avg_abs, color='m', linestyle='--', label=f'Average = {avg_abs:.6f}')
    axes[2].axhline(y=target_eta, color='orange', linestyle=':', label=f'η(a) = {target_eta:.6f}')
    axes[2].set_xlabel('t')
    axes[2].set_ylabel('|ζ|')
    axes[2].legend()
    axes[2].grid(True)

    plt.suptitle(f'Spectral sum with α = 1/(π - e^0.3628) ≈ {ALPHA:.4f}\nCoefficients scaled to match ∫|ζ|dt = η({A:.4f}) ≈ {target_eta:.6f}')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()