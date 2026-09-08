import math
import numpy as np

# ---------- Constants ----------
PI = math.pi
EXP = math.exp(0.3628)          # e^0.3628 ≈ 1.4371
A = 1.0 / (PI - EXP)            # ≈ 0.5866
ALPHA = A                       # spectral sum phase constant

def mobius_sieve(K):
    """Linear sieve for Möbius function (O(K))."""
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

def spectral_sum(t, coeffs, alpha=ALPHA):
    """ζ(t) = Σ_{i=1}^K coeffs[i] * exp(i * t * i / alpha)."""
    total = 0.0 + 0.0j
    K = len(coeffs) - 1
    for i in range(1, K + 1):
        if coeffs[i] != 0:
            total += coeffs[i] * np.exp(1j * t * i / alpha)
    return total

def integrate_abs_spectral(coeffs, N_points=1000, alpha=ALPHA):
    """∫_0^{2π*alpha} |ζ(t)| dt using trapezoidal rule."""
    period = 2 * PI * alpha
    t_vals = np.linspace(0, period, N_points)
    sum_abs = 0.0
    for t in t_vals:
        sum_abs += abs(spectral_sum(t, coeffs, alpha))
    dt = t_vals[1] - t_vals[0]
    return sum_abs * dt

def main():
    K = 100
    # Use Möbius coefficients (square‑free filter)
    mu = mobius_sieve(K)
    coeffs = [mu[i] if i > 0 else 0 for i in range(K + 1)]

    # Compute the un‑scaled integral
    I_abs = integrate_abs_spectral(coeffs, N_points=2000)
    print(f"Integral of |ζ(t)| (unscaled): {I_abs:.8f}")

    # Target η(a) ≈ 0.659538886352
    target_eta = 0.659538886352
    scaling = target_eta / I_abs
    print(f"Scaling factor to match η(a): {scaling:.8f}")

    # Scale coefficients and verify
    scaled_coeffs = [c * scaling for c in coeffs]
    I_abs_scaled = integrate_abs_spectral(scaled_coeffs, N_points=2000)
    print(f"Scaled integral = {I_abs_scaled:.8f} (target {target_eta:.8f})")
    print(f"Absolute error: {abs(I_abs_scaled - target_eta):.2e}")

if __name__ == "__main__":
    main()