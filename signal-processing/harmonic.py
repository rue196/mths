# harmonic.py
import numpy as np
import math
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)          # ≈ 2.362

# ---------- Möbius sieve (O(K)) ----------
def mobius_sieve(K):
    """Linear sieve for μ(1..K). Returns list mu[0..K] (mu[0]=0)."""
    if K < 1:
        return [0] * (K + 1)
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

# ---------- Build coefficient array C_i from Möbius ----------
def build_coeffs(K):
    """
    Build symmetric coefficient array C_i = μ(|i|) for i=-K..K,
    with C_0 = 0.
    Returns: numpy array of length 2K+1, dtype=complex.
    """
    mu = mobius_sieve(K)
    c = np.zeros(2*K + 1, dtype=complex)
    for n in range(1, K + 1):
        c[K + n] = mu[n]          # i = n
        c[K - n] = mu[n]          # i = -n
    c[K] = 0.0                    # i = 0
    return c

# ---------- Spectral sum at a single t (O(K)) ----------
def zeta_at(t, c, alpha):
    """
    Evaluate ζ(t) = Σ_{i=-K}^{K} C_i * exp(i * t * i / α).
    c: array of length 2K+1, built by build_coeffs.
    Returns complex value.
    """
    K = (len(c) - 1) // 2
    total = 0.0 + 0.0j
    for i in range(-K, K + 1):
        total += c[i + K] * np.exp(1j * t * i / alpha)
    return total

# ---------- Harmonic numbers (O(K)) ----------
def harmonic_numbers(K):
    """
    Return array H[1..K] where H[n] = 1 + 1/2 + ... + 1/n.
    Time: O(K), Memory: O(K).
    """
    H = np.zeros(K + 1, dtype=float)
    if K >= 1:
        H[1] = 1.0
    for n in range(2, K + 1):
        H[n] = H[n-1] + 1.0 / n
    return H

# ---------- Supertrace and entropy ----------
def supertrace_from_coeffs(C):
    S = 0.0
    for idx, coeff in enumerate(C):
        sign = 1 if (idx % 2 == 0) else -1
        S += sign * abs(coeff)
    return S

def entropy_from_supertrace(S, N, alpha=ALPHA):
    if S == 0:
        return 0.0
    p = abs(S) / N
    if p <= 0:
        return 0.0
    return -alpha * p * math.log(p)

def invariant_scalar(C):
    S = supertrace_from_coeffs(C)
    H = entropy_from_supertrace(S, len(C))
    return abs(S) * math.exp(-H)

# ---------- Integral kernel (Toeplitz) ----------
def integral_kernel(K, alpha=ALPHA):
    norm = 1.0 - math.exp(-alpha * (PI + E))
    kernel = np.zeros(2*K - 1, dtype=float)
    for d in range(-(K-1), K):
        val = (1.0 - math.exp(-alpha * abs(d))) / norm
        kernel[d + (K-1)] = val
    return kernel

# ---------- FFT convolution ----------
def apply_convolution(signal, kernel):
    L = len(signal)
    N = 1 << (2*L - 1).bit_length()
    sig_pad = np.pad(signal, (0, N - L), mode='constant')
    ker_pad = np.pad(kernel, (0, N - len(kernel)), mode='constant')
    conv = ifft(fft(sig_pad) * fft(ker_pad))[:L]
    return conv

# ---------- Compression with supertrace ----------
def compress_with_supertrace(signal, kernel, alpha=ALPHA):
    K = len(signal)
    conv = apply_convolution(signal, kernel)

    S = supertrace_from_coeffs(conv)
    H = entropy_from_supertrace(S, K, alpha)
    m = invariant_scalar(conv)

    M = max(1, int(abs(S)))
    if M > K:
        M = K

    idx_sorted = np.argsort(np.abs(conv))[::-1]
    kept_indices = idx_sorted[:M]
    kept_values = conv[kept_indices]

    recon = np.zeros(K, dtype=complex)
    recon[kept_indices] = kept_values
    error = np.linalg.norm(conv - recon)

    return kept_indices, kept_values, M, error, S, H, m, conv, recon

# ---------- Main (demo) ----------
def main():
    K = 200
    print(f"Computing harmonic numbers H[1..{K}] (O(K) time)...")
    H = harmonic_numbers(K)
    signal = H[1:]   # length K, values H[1],...,H[K]
    print(f"Signal length: {len(signal)}")
    print(f"First 10 harmonic numbers: {signal[:10]}")

    # Build integral kernel
    kernel = integral_kernel(K, ALPHA)

    # Compress
    kept_indices, kept_values, M, error, S, H_ent, m, conv, recon = compress_with_supertrace(
        signal, kernel, ALPHA
    )

    print(f"\nOriginal length: {K}")
    print(f"Kept coefficients: {M} (storage ratio {M/K:.2f})")
    print(f"Supertrace S = {S:.4f}")
    print(f"Entropy H = {H_ent:.4f}")
    print(f"Invariant mass m = {m:.4f}")
    print(f"Reconstruction L2 error = {error:.4e}")

    # Plot magnitude spectra
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.abs(conv), 'b-', label='Convolved signal')
    plt.plot(kept_indices, np.abs(kept_values), 'ro', markersize=3, label='Kept')
    plt.xlabel('Coefficient index')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.title('Spectral coefficients (magnitude)')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(np.abs(recon), 'r-', label='Reconstructed')
    plt.plot(np.abs(conv), 'b--', label='Original (convolved)')
    plt.xlabel('Coefficient index')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.title('Reconstruction vs original')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()