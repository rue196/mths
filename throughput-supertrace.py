#!/usr/bin/env python3
"""
mobius_m_matrix.py

Maps EllipticMobiusGate coefficients to an M‑matrix and verifies
the supertrace integrals using only NumPy (no SciPy).
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)          # ≈ 2.362
NORM = 1.0 - math.exp(-ALPHA * (PI + E))   # ≈ 1.0 (used in convolution)

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
    """Build C_i = μ(|i|) for i != 0, with smooth interpolation for μ=0."""
    mu = mobius_sieve(K)
    c = np.zeros(2*K + 1, dtype=float)
    for i in range(-K, K+1):
        if i == 0:
            c[i + K] = 0.0
        else:
            c[i + K] = mu[abs(i)]
    if smooth:
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

class EllipticMobiusGate:
    def __init__(self, K, smooth=True):
        self.K = K
        self.coeffs, self.full_array = build_elliptic_coeffs(K, smooth)
        self.indices = np.array(sorted(self.coeffs.keys()))
        self.period = 2 * PI * ALPHA

    def zeta(self, t):
        if len(self.indices) == 0:
            return 0.0 + 0.0j
        phases = t * self.indices / ALPHA
        vals = np.array([self.coeffs[i] for i in self.indices])
        return np.sum(vals * np.exp(1j * phases))

    def power_spectrum(self, t):
        return np.abs(self.zeta(t))**2

    def amplitude(self, t):
        return np.abs(self.zeta(t))

    def integrate(self, func, N=1000):
        """
        Numerically integrate func(t) over one period using the trapezoidal rule.
        No SciPy dependency; uses numpy.trapz / numpy.trapezoid.
        """
        t_vals = np.linspace(0, self.period, N)
        vals = np.array([func(t) for t in t_vals])
        # Trapezoidal integration (compatible with old and new NumPy)
        try:
            integral = np.trapezoid(vals, t_vals)
        except AttributeError:
            integral = np.trapz(vals, t_vals)
        # NORM is used in convolution but not needed here; we keep it as a comment.
        return integral

# ---------- M‑matrix mapping ----------
def coeffs_to_M_matrix(coeffs, K):
    """Build a 2×2 matrix from positive (real) and negative (imag) coefficients."""
    real_sum = 0.0
    imag_sum = 0.0
    for i, val in coeffs.items():
        if i > 0:
            real_sum += val
        else:
            imag_sum += val
    M = np.array([[real_sum, -imag_sum], [imag_sum, real_sum]], dtype=complex)
    return M

def supertrace(M):
    """Alternating sum of diagonal elements."""
    S = 0.0
    for idx, val in enumerate(np.diag(M)):
        sign = 1 if (idx % 2 == 0) else -1
        S += sign * abs(val)
    return S

def entropy(S, N=2):
    if S == 0:
        return 0.0
    p = abs(S) / N
    if p <= 0 or p >= 1:
        return 0.0
    return -ALPHA * p * math.log(p)

def mass(S, H):
    return abs(S) * math.exp(-H)

# ---------- Main ----------
def main():
    K = 20
    gate = EllipticMobiusGate(K, smooth=True)
    print(f"K = {K}, number of non‑zero coeffs: {len(gate.indices)}")

    # 1. Integrals (using only NumPy)
    integral_power = gate.integrate(gate.power_spectrum, N=2000)
    integral_amplitude = gate.integrate(gate.amplitude, N=2000)

    bosonic_theory = 2 * (6 / (PI * PI)) * K
    fermionic_theory = 2 * (6 / PI) * K

    print(f"\n12‑vertex integral (|ζ|²): {integral_power:.6f}  (theory: {bosonic_theory:.6f})")
    print(f"6‑vertex integral  (|ζ|)  : {integral_amplitude:.6f}  (theory: {fermionic_theory:.6f})")
    print(f"Error (|ζ|²): {abs(integral_power - bosonic_theory) / bosonic_theory:.2e}")
    print(f"Error (|ζ|)  : {abs(integral_amplitude - fermionic_theory) / fermionic_theory:.2e}")

    # 2. Build M‑matrix and compute invariants
    M = coeffs_to_M_matrix(gate.coeffs, K)
    print(f"\nM‑matrix:\n{M}")

    S = supertrace(M)
    H = entropy(S, N=2)
    m = mass(S, H)
    print(f"Supertrace S = {S:.4f}")
    print(f"Entropy H = {H:.4f}")
    print(f"Mass m = {m:.4f}")

    # 3. Plot trajectory
    t_vals = np.linspace(0, gate.period, 300)
    z_vals = np.array([gate.zeta(t) for t in t_vals])
    plt.figure(figsize=(8, 6))
    plt.plot(z_vals.real, z_vals.imag, 'b-', alpha=0.7, label='ζ(t)')
    plt.scatter(z_vals.real[0], z_vals.imag[0], color='red', s=50, label='start')
    plt.xlabel('Re(ζ)')
    plt.ylabel('Im(ζ)')
    plt.title(f'Trajectory of ζ(t) (K={K})')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()