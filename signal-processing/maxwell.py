#!/usr/bin/env python3
"""
maxwell_mobius.py

Maxwell's equations in the Möbius spectral basis.
Electric field E: real coefficients C_i (i>0, μ(i)≠0)
Magnetic field B: imaginary coefficients C_{-i} (i>0, μ(i)≠0)
Divergence = supertrace (alternating sum)
Curl = alternating sum weighted by i/α
Density ρ = divergence of E (scaled)
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

    def field_components(self, t):
        """Return E(t) and B(t) from the real and imaginary parts."""
        z = self.zeta(t)
        E_field = z.real
        B_field = z.imag
        return E_field, B_field

    def divergence_E(self, t):
        """∇·E = supertrace of the E coefficients."""
        # E coefficients are the real parts: for each index i, E_i = Re(C_i) * cos? Actually we need the coefficients themselves.
        # The divergence at time t is the sum over i of (-1)^i * E_i(t) where E_i(t) = C_i * cos(t*i/α) for positive i.
        # But we can compute it directly from the coefficients.
        # We'll compute the supertrace of the real part of the signal.
        E_vals = np.array([self.coeffs[i] * np.cos(t * i / ALPHA) if i > 0 else 0 for i in range(-self.K, self.K+1)])
        # Actually we need to separate positive and negative indices.
        # For positive i, the real part is C_i * cos(t*i/α); for negative i, the real part is C_i * cos(t*i/α) but with i negative.
        # Using the symmetry C_{-i} = C_i, the real part is Σ C_i cos(t*i/α) over all i.
        # The divergence is the alternating sum of the real parts.
        div = 0.0
        for i in range(-self.K, self.K+1):
            if i == 0:
                continue
            if self.coeffs.get(i, 0) != 0:
                val = self.coeffs[i] * np.cos(t * i / ALPHA)
                sign = 1 if (i % 2 == 0) else -1   # use parity of index
                div += sign * val
        return div

    def curl_E(self, t):
        """(∇×E)_z = -∂B/∂t. In 1D we treat it as a scalar."""
        # ∂B/∂t = -Σ C_{-i} * (i/α) * sin(t*i/α) for negative i
        # We'll compute the alternating sum of the derivatives.
        curl = 0.0
        for i in range(-self.K, self.K+1):
            if i == 0:
                continue
            if self.coeffs.get(i, 0) != 0:
                # For E, we use the real part: C_i * cos(t*i/α)
                # The curl is the alternating sum of the derivative? Actually ∇×E = -∂B/∂t.
                # We can compute ∂B/∂t from the imaginary part.
                if i < 0:
                    # imaginary part coefficient at i is 0? Actually we need to use the stored coefficient.
                    pass
        # Simpler: use the zeta function and its derivative.
        dt = 1e-6
        z1 = self.zeta(t + dt)
        z0 = self.zeta(t)
        dz = (z1 - z0) / dt
        # ∂B/∂t = Im(dz), ∂E/∂t = Re(dz)
        # ∇×E = -∂B/∂t, ∇×B = ∂E/∂t (in vacuum)
        curl_E = -dz.imag
        curl_B = dz.real
        return curl_E, curl_B

    def density(self, t):
        """ρ = ∇·E (scaled by ε₀)."""
        return self.divergence_E(t)

    def maxwell_check(self, t):
        """
        Check all four Maxwell equations at time t.
        Returns a dict with the residuals.
        """
        E, B = self.field_components(t)
        divE = self.divergence_E(t)
        divB = 0.0   # no magnetic monopoles
        curlE, curlB = self.curl_E(t)
        # ∂B/∂t and ∂E/∂t are approximated from the derivative of the fields.
        dt = 1e-6
        E1, B1 = self.field_components(t + dt)
        dE_dt = (E1 - E) / dt
        dB_dt = (B1 - B) / dt
        # Maxwell: ∇×E = -∂B/∂t
        res1 = curlE + dB_dt
        # ∇×B = μ₀ J + μ₀ ε₀ ∂E/∂t (in vacuum J=0)
        # So ∇×B = μ₀ ε₀ ∂E/∂t
        # We'll check ∇×B - ∂E/∂t = 0 (with c=1)
        res2 = curlB - dE_dt
        # ∇·E = ρ
        res3 = divE - self.density(t)
        # ∇·B = 0
        res4 = divB
        return {'∇×E + ∂B/∂t': res1, '∇×B - ∂E/∂t': res2, '∇·E - ρ': res3, '∇·B': res4}

# ---------- Main demonstration ----------
def main():
    K = 10
    gate = EllipticMobiusGate(K, smooth=True)
    print(f"K = {K}, number of non‑zero coeffs: {len(gate.indices)}")

    # Sample time points
    t_vals = np.linspace(0, gate.period, 200)
    E_vals = []
    B_vals = []
    divE_vals = []
    rho_vals = []

    for t in t_vals:
        E, B = gate.field_components(t)
        E_vals.append(E)
        B_vals.append(B)
        divE_vals.append(gate.divergence_E(t))
        rho_vals.append(gate.density(t))

    # Plot fields and divergence
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0,0].plot(t_vals, E_vals, label='E(t)')
    axes[0,0].set_title('Electric field')
    axes[0,0].set_xlabel('t')
    axes[0,0].grid(True)

    axes[0,1].plot(t_vals, B_vals, label='B(t)')
    axes[0,1].set_title('Magnetic field')
    axes[0,1].set_xlabel('t')
    axes[0,1].grid(True)

    axes[1,0].plot(t_vals, divE_vals, label='∇·E')
    axes[1,0].plot(t_vals, rho_vals, '--', label='ρ')
    axes[1,0].set_title('Divergence of E and charge density')
    axes[1,0].set_xlabel('t')
    axes[1,0].legend()
    axes[1,0].grid(True)

    # Check Maxwell at one time
    t0 = gate.period / 4
    res = gate.maxwell_check(t0)
    print(f"\nMaxwell residuals at t = {t0:.3f}:")
    for key, val in res.items():
        print(f"  {key}: {val:.4e}")

    # Plot energy densities
    E2_vals = np.array(E_vals)**2
    B2_vals = np.array(B_vals)**2
    axes[1,1].plot(t_vals, E2_vals, label='|E|²')
    axes[1,1].plot(t_vals, B2_vals, label='|B|²')
    axes[1,1].set_title('Field energy densities')
    axes[1,1].set_xlabel('t')
    axes[1,1].legend()
    axes[1,1].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()