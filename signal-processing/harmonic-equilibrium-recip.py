import math
import numpy as np
import matplotlib.pyplot as plt

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)          # ≈ 2.362

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

# ---------- Harmonic numbers (O(K)) ----------
def harmonic_numbers(N):
    H = np.zeros(N + 1, dtype=float)
    H[1] = 1.0
    for n in range(2, N + 1):
        H[n] = H[n-1] + 1.0 / n
    return H

# ---------- Build coefficient array C_i from Möbius ----------
def build_coeffs(K):
    mu = mobius_sieve(K)
    c = np.zeros(2*K + 1, dtype=complex)
    for n in range(1, K + 1):
        c[K + n] = mu[n]          # i = n
        c[K - n] = mu[n]          # i = -n
    c[K] = 0.0                    # i = 0
    return c

# ---------- Spectral sum at a single t (O(K)) ----------
def zeta_at(t, c, alpha):
    K = (len(c) - 1) // 2
    total = 0.0 + 0.0j
    for i in range(-K, K + 1):
        total += c[i + K] * np.exp(1j * t * i / alpha)
    return total

# ---------- Scarcity multiplier ----------
def scarcity_multiplier(p, n):
    return (1 - p) ** (-n)

# ---------- Main simulation using harmonic points ----------
def equilibrium_with_harmonics(K_modes=25, N_harmonics=100):
    """
    Run the equilibrium model where time steps are the harmonic numbers H_n.
    """
    # 1. Build Möbius coefficients
    c = build_coeffs(K_modes)
    H = harmonic_numbers(N_harmonics)
    time = H[1:]   # first harmonic is H_1 = 1.0

    # 2. Precompute zeta at each harmonic point (real part only)
    zeta_vals = np.zeros(N_harmonics)
    for n in range(1, N_harmonics + 1):
        t = H[n]
        z = zeta_at(t, c, ALPHA)
        zeta_vals[n-1] = z.real   # use real part as driving signal

    # 3. Initial conditions (same as equilibrium.py)
    x_inv0 = 1.0
    y_inv0 = 0.8
    x_i0 = 1.0
    y_i0 = 1.2

    # Binomial parameters
    p0_x, p_amp_x, freq_x, n_x = 0.2, 0.15, 0.5, 2
    p0_y, p_amp_y, freq_y, n_y = 0.3, 0.10, 0.3, 3

    # Storage
    M = np.zeros((N_harmonics, 2, 2))
    M[0, :, :] = [[x_inv0, y_inv0], [x_i0, y_i0]]

    # 4. Time evolution over harmonic points
    for idx in range(1, N_harmonics):
        t = time[idx]          # H_{idx+1}
        t_prev = time[idx-1]   # H_idx

        # Driving signal: zeta at this harmonic point (scalar)
        d_zeta = zeta_vals[idx]   # use the value of zeta, not derivative

        # Binomial oscillations
        p_x = p0_x + p_amp_x * np.sin(2 * np.pi * freq_x * t)
        p_y = p0_y + p_amp_y * np.sin(2 * np.pi * freq_y * t)
        p_x = np.clip(p_x, 0.0, 0.99)
        p_y = np.clip(p_y, 0.0, 0.99)

        Sx = scarcity_multiplier(p_x, n_x)
        Sy = scarcity_multiplier(p_y, n_y)

        # Update scarcity (x^-1, y^-1) using zeta as growth rate
        x_inv = M[idx-1, 0, 0] * (1 + d_zeta * (t - t_prev)) + (Sx - M[idx-1, 0, 0]) * 0.01
        y_inv = M[idx-1, 0, 1] * (1 + d_zeta * (t - t_prev)) + (Sy - M[idx-1, 0, 1]) * 0.01
        x_inv = max(x_inv, 0.1)
        y_inv = max(y_inv, 0.1)

        # Update demand (x^i, y^i) driven by zeta
        x_i = M[idx-1, 1, 0] * (1 + 0.02 * d_zeta * (t - t_prev)) + 0.01 * np.sin(2 * np.pi * 0.1 * t)
        y_i = M[idx-1, 1, 1] * (1 + 0.02 * d_zeta * (t - t_prev)) + 0.01 * np.cos(2 * np.pi * 0.08 * t)
        x_i = np.clip(x_i, 0.1, 5.0)
        y_i = np.clip(y_i, 0.1, 5.0)

        M[idx, :, :] = [[x_inv, y_inv], [x_i, y_i]]

    # 5. Derived quantities
    det = M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]
    trace = M[:, 0, 0] + M[:, 1, 1]
    price_x = M[:, 0, 0] * M[:, 1, 0]
    price_y = M[:, 0, 1] * M[:, 1, 1]

    return time, zeta_vals, M, det, trace, price_x, price_y

# ---------- Plotting ----------
def main():
    K_modes = 25          # number of spectral modes
    N_harmonics = 100     # number of harmonic points
    time, zeta_vals, M, det, trace, price_x, price_y = equilibrium_with_harmonics(K_modes, N_harmonics)

    print(f"Number of harmonic points: {N_harmonics}")
    print(f"Final scarcity: x^-1 = {M[-1,0,0]:.3f}, y^-1 = {M[-1,0,1]:.3f}")
    print(f"Final demand:   x^i  = {M[-1,1,0]:.3f}, y^i  = {M[-1,1,1]:.3f}")

    # Plot
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))

    # Zeta driving signal
    ax = axes[0, 0]
    ax.plot(time, zeta_vals, color='purple')
    ax.set_title('Spectral sum $\zeta(H_n)$ (real part)')
    ax.set_xlabel('Harmonic time $H_n$')
    ax.grid(True, alpha=0.3)

    # Scarcity
    ax = axes[0, 1]
    ax.plot(time, M[:, 0, 0], label='$x^{-1}$', color='red')
    ax.plot(time, M[:, 0, 1], label='$y^{-1}$', color='blue')
    ax.set_title('Scarcity Factors')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Demand
    ax = axes[1, 0]
    ax.plot(time, M[:, 1, 0], label='$x^{i}$', color='orange')
    ax.plot(time, M[:, 1, 1], label='$y^{i}$', color='green')
    ax.set_title('Demand Factors')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Prices
    ax = axes[1, 1]
    ax.plot(time, price_x, label='Price x', color='crimson')
    ax.plot(time, price_y, label='Price y', color='navy')
    ax.set_title('Composite Prices')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Determinant
    ax = axes[2, 0]
    ax.plot(time, det, color='black')
    ax.axhline(y=0, linestyle='--', color='gray')
    ax.set_title('Determinant of M')
    ax.grid(True, alpha=0.3)

    # Trace
    ax = axes[2, 1]
    ax.plot(time, trace, color='darkgreen')
    ax.set_title('Trace of M')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()