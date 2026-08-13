import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)          # ≈ 2.362
A = ALPHA
NORM = 1.0 - math.exp(-ALPHA * (PI + E))

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

# ---------- Prime sieve (for K selection) ----------
def sieve_primes(limit):
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0:2] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i*i:limit+1:i] = False
    return np.nonzero(is_prime)[0]

def nth_prime(n):
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

# ---------- Compression using supertrace and Möbius ----------
def compress_with_supertrace(signal, kernel, mu, alpha=ALPHA):
    """
    Signal: 1D array (length K).
    Kernel: Toeplitz kernel (length 2K-1).
    mu: Möbius array (length K+1) for square‑free selection.
    Returns: kept_indices, kept_values, M, error, S, H, m, conv, recon
    """
    K = len(signal)
    conv = apply_convolution(signal, kernel)

    S = supertrace_from_coeffs(conv)
    H = entropy_from_supertrace(S, K, alpha)
    m = invariant_scalar(conv)

    # M = number to keep: based on |S|, but at least 1
    M = max(1, int(abs(S)))
    if M > K:
        M = K

    # Select top M magnitudes, but only square‑free indices (mu[idx+1] != 0)
    mag = np.abs(conv)
    idx_sorted = np.argsort(mag)[::-1]
    kept_indices = []
    kept_values = []
    for idx in idx_sorted:
        if mu[idx+1] != 0:   # square‑free
            kept_indices.append(idx)
            kept_values.append(conv[idx])
            if len(kept_indices) >= M:
                break

    # Reconstruct (zero out non‑kept)
    recon = np.zeros(K, dtype=complex)
    for idx, val in zip(kept_indices, kept_values):
        recon[idx] = val

    error = np.linalg.norm(conv - recon)

    return np.array(kept_indices), np.array(kept_values), M, error, S, H, m, conv, recon

# ---------- MobiusMemoryCache ----------
class MobiusMemoryCache:
    """
    Stores states (signals) along with their supertrace traces.
    Allows reconstruction from compressed coefficients.
    """
    def __init__(self, K, alpha=ALPHA):
        self.K = K
        self.alpha = alpha
        self.mu = mobius_sieve(K)
        self.kernel = integral_kernel(K, alpha)
        # Storage
        self.states = []          # list of original signals (full arrays)
        self.compressed = []      # list of (kept_indices, kept_values)
        self.traces = []          # list of (S, H, m)
        self.timestamps = []      # optional time labels

    def add_state(self, signal, timestamp=None):
        """
        Add a new state to the cache.
        signal: 1D numpy array (real or complex) of length K.
        Returns: (S, H, m) of the state.
        """
        # Compute traces and compress
        kept_idx, kept_vals, M, err, S, H, m, conv, recon = compress_with_supertrace(
            signal, self.kernel, self.mu, self.alpha
        )
        self.states.append(signal.copy())
        self.compressed.append((kept_idx, kept_vals))
        self.traces.append((S, H, m))
        if timestamp is not None:
            self.timestamps.append(timestamp)
        else:
            self.timestamps.append(len(self.states)-1)
        return S, H, m

    def reconstruct(self, index):
        """
        Reconstruct the full signal from the compressed representation at 'index'.
        Returns: reconstructed signal (array) and the (S, H, m) trace.
        """
        if index >= len(self.compressed):
            raise IndexError("Index out of range")
        kept_idx, kept_vals = self.compressed[index]
        recon = np.zeros(self.K, dtype=complex)
        recon[kept_idx] = kept_vals
        # The reconstructed signal is the convolved version; we could also
        # optionally apply deconvolution (not implemented).
        return recon, self.traces[index]

    def get_trace(self, index):
        return self.traces[index]

    def get_all_traces(self):
        return np.array(self.traces)  # shape (N, 3)

    def get_all_states(self):
        return np.array(self.states)

    def clear(self):
        self.states.clear()
        self.compressed.clear()
        self.traces.clear()
        self.timestamps.clear()

# ---------- Example: simulate pendulum and store states ----------
def simulate_pendulum_states(N_steps=50, K=64, dt=0.05):
    """
    Generate a sequence of pendulum states (4‑D M‑matrix) and store them as signals.
    Each state is a 4‑vector; we'll treat each component as a separate signal or
    combine them into a complex signal. For simplicity, we'll store the angle θ
    as the signal (a scalar time series). But we need a 1D array of length K.
    We'll take the angle over a sliding window of K time steps.
    Actually, we need a signal of length K for each state. So we'll take a window
    of the pendulum's angle of length K as the signal.
    """
    from pendulum import run_pendulum_simulation
    # Get the full trajectory
    time_vals, state_hist, zeta_vals, S_hist, mass_hist = run_pendulum_simulation(
        N_harmonics=N_steps + K, K_max=60
    )
    # Extract angle: θ = (x_i - y_i)/2
    theta = (state_hist[:, 2] - state_hist[:, 3]) / 2.0
    # Now create sliding windows of length K
    signals = []
    for i in range(N_steps):
        window = theta[i:i+K]
        if len(window) == K:
            signals.append(window)
    return np.array(signals)

# ---------- Main ----------
def main():
    # Choose K as a prime (for Möbius)
    prime_index = 30
    K = nth_prime(prime_index)
    print(f"K = {K} (prime index {prime_index})")

    # Create cache
    cache = MobiusMemoryCache(K)

    # Generate some states (e.g., pendulum trajectory)
    print("Generating pendulum states...")
    signals = simulate_pendulum_states(N_steps=50, K=K, dt=0.05)
    print(f"Generated {len(signals)} states.")

    # Add states to cache
    for i, sig in enumerate(signals):
        # Convert to complex (real part only)
        sig_c = sig.astype(complex)
        S, H, m = cache.add_state(sig_c, timestamp=i)
        if i % 10 == 0:
            print(f"State {i}: S={S:.4f}, H={H:.4f}, m={m:.4f}")

    # Reconstruct a state
    idx = 5
    recon, (S, H, m) = cache.reconstruct(idx)
    original = cache.states[idx]

    print(f"\nReconstruction of state {idx}:")
    print(f"  Original S={S:.4f}, H={H:.4f}, m={m:.4f}")
    print(f"  Reconstructed shape: {recon.shape}")

    # Plot original vs reconstructed
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.abs(original), label='Original (abs)')
    plt.plot(np.abs(recon), 'r--', label='Reconstructed (abs)')
    plt.xlabel('Coefficient index')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.title('Original vs Reconstructed (magnitude)')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(np.real(original), label='Original (real)')
    plt.plot(np.real(recon), 'r--', label='Reconstructed (real)')
    plt.xlabel('Coefficient index')
    plt.ylabel('Real part')
    plt.legend()
    plt.title('Original vs Reconstructed (real)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Show all traces
    traces = cache.get_all_traces()
    plt.figure()
    plt.plot(traces[:, 0], label='S')
    plt.plot(traces[:, 1], label='H')
    plt.plot(traces[:, 2], label='m')
    plt.xlabel('State index')
    plt.ylabel('Trace value')
    plt.legend()
    plt.title('Supertrace, Entropy, and Mass over time')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()