import math
import random
import numpy as np
from numpy.fft import fft, ifft
import cmath

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)   # ≈ 2.362

# ---------- Prime sieve (O(N log log N)) ----------
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

# ---------- Integral operator kernel (Toeplitz) ----------
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

# ---------- Compression using supertrace ----------
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
    return kept_indices, kept_values, M, S, H, m, conv

# ---------- Server class ----------
class Server:
    def __init__(self, prime_index=50, seed=42):
        """
        K is chosen as the prime_index-th prime.
        """
        self.K = nth_prime(prime_index)
        self.prime_index = prime_index
        self.points, self.weights = self._generate_data(seed)
        self.client = None
        # compressed storage (dictionary of index->complex)
        self.compressed = {}
        self.compressed_info = {}

    def _generate_data(self, seed):
        random.seed(seed)
        points = [(random.uniform(-5,5), random.uniform(-5,5)) for _ in range(self.K)]
        weights = [complex(random.gauss(0,1), random.gauss(0,1)) for _ in range(self.K)]
        return points, weights

    def set_client(self, client):
        self.client = client

    def get_algebraic_data(self):
        return self.points, self.weights

    def run_compression(self, sigma=None, method='supertrace'):
        """
        New compression pipeline:
          - sort by angle
          - client computes the integral kernel (or any kernel)
          - apply convolution via FFT
          - compress using supertrace (keep M largest)
        """
        points_sorted, weights_sorted = self._sort_by_angle(self.points, self.weights)
        K = len(weights_sorted)

        # Client computes the kernel (now integral kernel)
        if self.client is None:
            raise ValueError("Client not set.")
        kernel = self.client.compute_kernel(points_sorted, sigma)

        if method == 'supertrace':
            # Use the supertrace-based compression
            kept_indices, kept_values, M, S, H, m, conv = compress_with_supertrace(
                np.array(weights_sorted), kernel, ALPHA
            )
            # Store compressed as dictionary
            self.compressed = {int(i): v for i, v in zip(kept_indices, kept_values)}
            self.compressed_info = {
                'M': M, 'S': S, 'H': H, 'm': m,
                'conv_full': conv, 'kept_indices': kept_indices
            }
            return self.compressed, points_sorted, weights_sorted, M, S, H, m
        else:
            # Fallback to magnitude-based compression
            M = self.K // 2
            conv = apply_convolution(np.array(weights_sorted), kernel)
            idx_sorted = np.argsort(np.abs(conv))[::-1][:M]
            self.compressed = {int(i): conv[i] for i in idx_sorted}
            self.compressed_info = {'M': M, 'conv_full': conv}
            return self.compressed, points_sorted, weights_sorted, M, None, None, None

    def _sort_by_angle(self, points, weights):
        data = sorted(zip(points, weights), key=lambda p: cmath.phase(complex(p[0][0], p[0][1])))
        points_sorted = [p for p, w in data]
        weights_sorted = [w for p, w in data]
        return points_sorted, weights_sorted

    def reconstruct(self):
        """
        Reconstruct the full convolved signal from the compressed representation.
        """
        if not self.compressed:
            raise ValueError("No compressed data. Run compression first.")
        K = self.K
        recon = np.zeros(K, dtype=complex)
        for i, val in self.compressed.items():
            recon[i] = val
        return recon

    def get_compression_ratio(self):
        return len(self.compressed) / self.K if self.K > 0 else 0.0

# ---------- Client class ----------
class Client:
    def __init__(self):
        pass

    def compute_kernel(self, points_sorted, sigma=None):
        """
        Compute the integral kernel (transcendental part).
        The kernel does not depend on the points; it only depends on K (length of points).
        """
        K = len(points_sorted)
        return integral_kernel(K, ALPHA)

# ---------- Example usage ----------
def main():
    prime_index = 50   # 50th prime = 229
    server = Server(prime_index=prime_index, seed=42)
    client = Client()
    server.set_client(client)

    print(f"Server K = {server.K} (prime index {prime_index})")

    compressed, pts, wts, M, S, H, m = server.run_compression(method='supertrace')
    print(f"Compressed: kept {M} coefficients out of {server.K} (ratio {M/server.K:.3f})")
    print(f"Supertrace S = {S:.4f}, Entropy H = {H:.4f}, Mass m = {m:.4f}")

    # Reconstruct and compute error
    recon = server.reconstruct()
    conv_full = server.compressed_info['conv_full']
    error = np.linalg.norm(conv_full - recon) / np.linalg.norm(conv_full)
    print(f"Reconstruction relative L2 error = {error:.4e}")

    # Show compressed dictionary (first few)
    items = list(compressed.items())[:5]
    print("First 5 compressed entries:", items)

if __name__ == "__main__":
    main()