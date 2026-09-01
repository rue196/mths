#!/usr/bin/env python3
"""
mobius_maxwell_m_matrix.py

Replaces Maxwell PDE with M‑matrix dynamics.
Electric field E = real part of ζ(t) = diagonal entries of M.
Magnetic field B = imaginary part of ζ(t) = off‑diagonal entries.
Supertrace S = alternating sum of diagonal entries.
Time evolution: derivative of S = wave equation in spectral basis.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)          # ≈ 2.362
NORM = 1.0 - math.exp(-ALPHA * (PI + E))

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

    # ---------- M‑matrix mapping ----------
    def M_matrix(self, t):
        """Construct M = [[E, -B], [B, E]] where E=Re(ζ), B=Im(ζ)."""
        z = self.zeta(t)
        E_field = z.real
        B_field = z.imag
        M = np.array([[E_field, -B_field], [B_field, E_field]], dtype=complex)
        return M

    def supertrace(self, t):
        """Supertrace of M: S = diag[0] - diag[1] = E - E = 0 for this symmetric form? Actually we need to use the alternating sum of diagonal entries.
        For a 2x2 matrix, S = M[0,0] - M[1,1] = E - E = 0.
        But in the general flow matrix, we sum over t indices; here we only have two diagonal entries.
        To get a non‑zero S, we need to consider the coefficients themselves.
        We'll compute S from the coefficients: S = Σ_{i=1}^K (-1)^i C_i (real) for positive indices.
        """
        S = 0.0
        for i in range(1, self.K+1):
            if self.coeffs.get(i, 0) != 0:
                val = self.coeffs[i] * np.cos(t * i / ALPHA)  # real part at time t
                sign = 1 if (i % 2 == 0) else -1
                S += sign * val
        return S

    def entropy(self, t):
        S = self.supertrace(t)
        if S == 0:
            return 0.0
        N = self.K
        p = abs(S) / N
        if p <= 0 or p >= 1:
            return 0.0
        return -ALPHA * p * math.log(p)

    def mass(self, t):
        S = self.supertrace(t)
        H = self.entropy(t)
        return abs(S) * math.exp(-H)

    def derivative_supertrace(self, t, dt=1e-6):
        """Numerical derivative of supertrace using central difference."""
        S1 = self.supertrace(t + dt)
        S0 = self.supertrace(t - dt)
        return (S1 - S0) / (2 * dt)

    def wave_equation_residual(self, t, dt=1e-6):
        """Check if the supertrace satisfies the wave equation: ∂²S/∂t² = -ω² S."""
        S = self.supertrace(t)
        # Second derivative
        S2 = self.supertrace(t + dt)
        S1 = self.supertrace(t)
        S0 = self.supertrace(t - dt)
        d2S = (S2 - 2*S1 + S0) / (dt**2)
        # Compute frequency from the spectral sum: ω = i/α
        # For the dominant mode, we take the average of i/α weighted by coefficients.
        # We'll just compute the expected ω² = (1/α²) * average of i².
        # For simplicity, we use the first non‑zero i.
        if self.K > 0:
            i0 = min([abs(i) for i in self.indices if i != 0], default=1)
            omega2 = (i0 / ALPHA)**2
            residual = d2S + omega2 * S
            return residual
        else:
            return 0.0

# ---------- Demonstration ----------
def main():
    K = 10
    gate = EllipticMobiusGate(K, smooth=True)
    print(f"K = {K}, number of non‑zero coeffs: {len(gate.indices)}")

    t_vals = np.linspace(0, gate.period, 200)
    S_vals = []
    H_vals = []
    m_vals = []
    residual_vals = []

    for t in t_vals:
        S_vals.append(gate.supertrace(t))
        H_vals.append(gate.entropy(t))
        m_vals.append(gate.mass(t))
        residual_vals.append(gate.wave_equation_residual(t))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0,0].plot(t_vals, S_vals, label='Supertrace S(t)')
    axes[0,0].set_title('Supertrace')
    axes[0,0].set_xlabel('t')
    axes[0,0].grid(True)

    axes[0,1].plot(t_vals, H_vals, label='Entropy H(t)', color='orange')
    axes[0,1].set_title('Entropy')
    axes[0,1].set_xlabel('t')
    axes[0,1].grid(True)

    axes[1,0].plot(t_vals, m_vals, label='Mass m(t)', color='green')
    axes[1,0].set_title('Mass')
    axes[1,0].set_xlabel('t')
    axes[1,0].grid(True)

    axes[1,1].plot(t_vals, residual_vals, label='Wave equation residual', color='red')
    axes[1,1].set_title('Wave equation residual')
    axes[1,1].set_xlabel('t')
    axes[1,1].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()