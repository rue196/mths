import math
import numpy as np
import matplotlib.pyplot as plt

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)          # ≈ 2.362

# ---------- Linear sieve for Möbius (O(K)) ----------
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
def harmonic_numbers(K):
    H = np.zeros(K + 1, dtype=float)
    if K >= 1:
        H[1] = 1.0
    for n in range(2, K + 1):
        H[n] = H[n-1] + 1.0 / n
    return H

# ---------- Build coefficient array C_i from Möbius ----------
def build_coeffs(K):
    mu = mobius_sieve(K)
    c = np.zeros(2*K + 1, dtype=complex)
    # C_i = μ(|i|) for i != 0, and C_0 = 0
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

# ---------- Evaluate at harmonic points ----------
def evaluate_at_harmonics(K_max, N_harmonics=None):
    """
    Evaluate ζ(H_n) for n = 1..N_harmonics.
    K_max: size of the coefficient array (half‑length).
    N_harmonics: number of harmonic points (if None, use K_max).
    """
    if N_harmonics is None:
        N_harmonics = K_max

    c = build_coeffs(K_max)
    H = harmonic_numbers(N_harmonics)
    results = []
    for n in range(1, N_harmonics + 1):
        t = H[n]
        z = zeta_at(t, c, ALPHA)
        results.append((t, z.real, z.imag, abs(z)))
    return results

# ---------- Main ----------
def main():
    K_max = 60
    N_harmonics = 100

    print(f"Using K_max = {K_max}, evaluating for n = 1..{N_harmonics}")
    data = evaluate_at_harmonics(K_max, N_harmonics)

    t_vals = [d[0] for d in data]
    real_vals = [d[1] for d in data]
    imag_vals = [d[2] for d in data]
    abs_vals = [d[3] for d in data]

    print("\nFirst few harmonic points and ζ(H_n):")
    for i in range(min(5, len(data))):
        n = i + 1
        print(f"H_{n} = {t_vals[i]:.4f}, ζ = {real_vals[i]:.4f} + {imag_vals[i]:.4f}i, |ζ| = {abs_vals[i]:.4f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(range(1, N_harmonics+1), real_vals, 'b-', label='Re ζ')
    ax1.plot(range(1, N_harmonics+1), imag_vals, 'r-', label='Im ζ')
    ax1.set_xlabel('n (harmonic index)')
    ax1.set_ylabel('ζ(H_n)')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title('Spectral sum at harmonic points (real and imaginary parts)')

    ax2.plot(range(1, N_harmonics+1), abs_vals, 'g-', label='|ζ(H_n)|')
    ax2.set_xlabel('n (harmonic index)')
    ax2.set_ylabel('|ζ(H_n)|')
    ax2.legend()
    ax2.grid(True)
    ax2.set_title('Magnitude of spectral sum at harmonic points')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()