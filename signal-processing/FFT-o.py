import numpy as np
import math
from numpy.fft import fft, ifft

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)          # ≈ 2.362
A = ALPHA

# ---------- Prime sieve (O(N log log N)) ----------
def sieve_primes(limit):
    """Return list of primes up to limit."""
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0:2] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i:limit+1:i] = False
    return np.nonzero(is_prime)[0]

def nth_prime(n):
    """Return the n-th prime (1‑based)."""
    # rough upper bound for n-th prime: n*(log n + log log n) for n>=6
    if n < 6:
        limit = 20
    else:
        limit = int(n * (math.log(n) + math.log(math.log(n)))) + 10
    primes = sieve_primes(limit)
    while len(primes) < n:
        limit *= 2
        primes = sieve_primes(limit)
    return primes[n-1]

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

# ---------- Spectral derivative dζ/dt ----------
def dzeta_dt(t, C, alpha=ALPHA):
    K = (len(C) - 1) // 2
    deriv = 0.0
    for i in range(1, K + 1):
        theta = t * i / alpha
        deriv += -2.0 * C[K + i] * (i / alpha) * math.sin(theta)
    return deriv

# ---------- Integral operator kernel (Toeplitz) ----------
def integral_kernel(K, alpha=ALPHA):
    """
    Build a Toeplitz kernel of length 2K-1:
    k[d] = (1 - exp(-alpha * |d|)) / (1 - exp(-alpha * (PI + E)))
    """
    norm = 1.0 - math.exp(-alpha * (PI + E))
    kernel = np.zeros(2*K - 1, dtype=float)
    for d in range(-(K-1), K):
        val = (1.0 - math.exp(-alpha * abs(d))) / norm
        kernel[d + (K-1)] = val
    return kernel

# ---------- FFT convolution ----------
def apply_convolution(signal, kernel):
    L = len(signal)
    N = 1 << (2*L - 1).bit_length()   # next power of two
    sig_pad = np.pad(signal, (0, N - L), mode='constant')
    ker_pad = np.pad(kernel, (0, N - len(kernel)), mode='constant')
    conv = ifft(fft(sig_pad) * fft(ker_pad))[:L]
    return conv

# ---------- Compression using supertrace ----------
def compress_with_supertrace(signal, kernel, alpha=ALPHA):
    """
    Signal: 1D array (length K).
    Kernel: Toeplitz kernel (length 2K-1).
    Returns: compressed representation (indices and coefficients) and metadata.
    """
    K = len(signal)
    # 1. Convolve with integral kernel (smoothing)
    conv = apply_convolution(signal, kernel)

    # 2. Compute supertrace of the convolved signal (treated as coefficients)
    S = supertrace_from_coeffs(conv)
    H = entropy_from_supertrace(S, K, alpha)
    m = invariant_scalar(conv)

    # 3. Determine M = number of coefficients to keep (based on |S|, rounded)
    M = max(1, int(abs(S)))   # keep at least 1
    if M > K:
        M = K

    # 4. Keep the M largest magnitude coefficients in the convolved spectrum
    idx_sorted = np.argsort(np.abs(conv))[::-1]
    kept_indices = idx_sorted[:M]
    kept_values = conv[kept_indices]

    # 5. Reconstruct (zero out all other coefficients)
    recon = np.zeros(K, dtype=complex)
    recon[kept_indices] = kept_values

    # 6. Error (L2 norm of difference between original and reconstructed)
    error = np.linalg.norm(conv - recon)

    return kept_indices, kept_values, M, error, S, H, m, conv, recon

# ---------- Demonstration ----------
def main():
    # Choose K as a prime (e.g., the 50th prime = 229)
    prime_index = 50
    K = nth_prime(prime_index)
    print(f"K = {K} (prime index {prime_index})")

    # Generate symmetric coefficients C_i (all ones)
    C = np.ones(2*K + 1)

    # Time points for evaluating dzeta/dt
    t = np.linspace(0, 20, K)   # K points

    # Compute signal: spectral derivative at each time
    signal = np.array([dzeta_dt(ti, C, ALPHA) for ti in t])

    # Build the integral kernel
    kernel = integral_kernel(K, ALPHA)

    # Compress
    kept_indices, kept_values, M, error, S, H, m, conv, recon = compress_with_supertrace(
        signal, kernel, ALPHA
    )

    # Statistics
    print(f"Original length: {K}")
    print(f"Kept coefficients: {M} (storage ratio {M/K:.2f})")
    print(f"Supertrace S = {S:.4f}")
    print(f"Entropy H = {H:.4f}")
    print(f"Invariant mass m = {m:.4f}")
    print(f"Reconstruction L2 error = {error:.4e}")

    # Optionally plot the first few coefficients
    import matplotlib.pyplot as plt
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