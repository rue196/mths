#!/usr/bin/env python3
"""

Voltage spike tunneling using Maxwell's equations in Möbius spectral basis.
For each voltage spike (amplitude + phase), compute the electric field E(t)
and its divergence (∇·E), then apply an exponential convolution to simulate
tunneling through a potential barrier.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

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
        z = self.zeta(t)
        return z.real, z.imag   # E(t), B(t)

    def divergence_E(self, t):
        """∇·E = alternating sum of the real parts of the coefficients at time t."""
        div = 0.0
        for i in range(-self.K, self.K+1):
            if i == 0:
                continue
            if self.coeffs.get(i, 0) != 0:
                val = self.coeffs[i] * np.cos(t * i / ALPHA)
                sign = 1 if (i % 2 == 0) else -1
                div += sign * val
        return div

# ---------- Voltage spike handler ----------
def voltage_tunnel(events, K=10, smooth=True, barrier_width=10):
    """
    events: list of (amplitude, phase) for voltage spikes.
    Returns: processed signal (tunneled), S, H, m, and divergence values.
    """
    # 1. Create gate and sample field over time
    gate = EllipticMobiusGate(K, smooth=True)
    N = len(events)
    times = np.linspace(0, gate.period, N)   # sample N time points

    # 2. Compute electric field and divergence at each time
    E_vals = []
    div_vals = []
    for t in times:
        E, _ = gate.field_components(t)
        div = gate.divergence_E(t)
        E_vals.append(E)
        div_vals.append(div)

    # 3. Apply convolution to simulate tunneling: convolution of E with a barrier kernel
    # The barrier kernel: Gaussian or exponential (like a potential barrier)
    x = np.arange(-barrier_width//2, barrier_width//2)
    barrier = np.exp(-ALPHA * np.abs(x))   # exponential decay
    barrier = barrier / barrier.sum()      # normalise

    # Convolve E with barrier (tunneling through finite barrier)
    tunneled = convolve(E_vals, barrier, mode='same')

    # 4. Also apply the chip pipeline (optional compression) to tunneled signal
    # We'll compute supertrace and mass from the tunneled signal (as a diagnostic)
    # For simplicity, compute S, H, m directly from tunneled array
    S_t = 0.0
    for i, v in enumerate(tunneled):
        sign = 1 if (i % 2 == 0) else -1
        S_t += sign * abs(v)
    N_t = len(tunneled)
    if S_t != 0 and N_t > 0:
        p = abs(S_t) / N_t
        H_t = -ALPHA * p * math.log(p) if 0 < p < 1 else 0.0
        m_t = abs(S_t) * math.exp(-H_t)
    else:
        H_t = 0.0; m_t = 0.0

    # 5. Return results
    return times, E_vals, div_vals, tunneled, S_t, H_t, m_t

# ---------- Demo ----------
def main():
    np.random.seed(42)
    # Generate random voltage events (amplitude, phase)
    N_events = 100
    amps = np.random.uniform(0.1, 1.0, N_events)
    phases = np.random.uniform(0, 2*PI, N_events)
    events = list(zip(amps, phases))

    # Process with tunneling
    times, E, div, tunneled, S, H, m = voltage_tunnel(events, K=10, smooth=True, barrier_width=10)

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    axes[0].plot(times, E, 'b-', label='Electric field E(t)')
    axes[0].set_ylabel('E')
    axes[0].set_title('Electric field from Mobius spectral sum')
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(times, div, 'r-', label='∇·E (divergence)')
    axes[1].set_ylabel('∇·E')
    axes[1].set_title('Divergence (charge density)')
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(times, tunneled, 'g-', label='Tunneled signal (convolved)')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Amplitude')
    axes[2].set_title(f'Tunneled signal (S={S:.3f}, H={H:.3f}, m={m:.3f})')
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout()
    plt.show()

    print(f"Tunneled signal: Supertrace S = {S:.4f}, Entropy H = {H:.4f}, Mass m = {m:.4f}")

if __name__ == "__main__":
    main()