import math
import random
import numpy as np
from numpy.fft import fft, ifft
import cmath

# ------------------------------------------------------------
# 1. Server: holds algebraic data (points, weights)
# ------------------------------------------------------------
class Server:
    def __init__(self, K, seed=42):
        self.K = K
        self.points, self.weights = self._generate_data(seed)
        self.client = None   # will be set later

    def _generate_data(self, seed):
        random.seed(seed)
        points = [(random.uniform(-5,5), random.uniform(-5,5)) for _ in range(self.K)]
        weights = [complex(random.gauss(0,1), random.gauss(0,1)) for _ in range(self.K)]
        return points, weights

    def set_client(self, client):
        self.client = client

    def get_algebraic_data(self):
        """Return points and weights (the algebraic part)."""
        return self.points, self.weights

    def run_compression(self, sigma=2.0, M=None, method='magnitude'):
        """Full pipeline: sort, get kernel from client, apply Toeplitz, compress."""
        # Sort by angle (O(K log K))
        points_sorted, weights_sorted = self._sort_by_angle(self.points, self.weights)
        K = len(weights_sorted)

        # Client computes the transcendental kernel (O(K) or O(K log K) if needed)
        kernel = self.client.compute_kernel(points_sorted, sigma)

        # Apply Toeplitz (convolution) using FFT (O(K log K))
        conv_result = self._apply_toeplitz(weights_sorted, kernel)

        # Compress
        if M is None:
            M = K // 2
        if method == 'magnitude':
            compressed = self._compress_by_magnitude(conv_result, M)
        elif method == 'lowpass':
            compressed = self._compress_by_lowpass(conv_result, M)
        else:
            raise ValueError("Unknown compression method.")

        return compressed, points_sorted, weights_sorted

    def _sort_by_angle(self, points, weights):
        data = sorted(zip(points, weights), key=lambda p: cmath.phase(complex(p[0][0], p[0][1])))
        points_sorted = [p for p, w in data]
        weights_sorted = [w for p, w in data]
        return points_sorted, weights_sorted

    def _apply_toeplitz(self, weights, kernel):
        K = len(weights)
        w_pad = np.concatenate([weights, np.zeros(K-1, dtype=complex)])
        W = fft(w_pad)
        K_fft = fft(kernel)
        Y = W * K_fft
        y = ifft(Y)
        return y[:K]

    def _compress_by_magnitude(self, weights, M):
        idx_sorted = sorted(range(len(weights)), key=lambda i: abs(weights[i]), reverse=True)
        compressed = {i: weights[i] for i in idx_sorted[:M]}
        return compressed

    def _compress_by_lowpass(self, weights, M):
        W = fft(weights)
        W_comp = np.zeros_like(W)
        W_comp[:M] = W[:M]
        return ifft(W_comp)

# ------------------------------------------------------------
# 2. Client: computes transcendental kernel from algebraic points
#    Client pulls the algebraic data (points) and pushes back the kernel.
# ------------------------------------------------------------
class Client:
    def __init__(self):
        pass

    def compute_kernel(self, points_sorted, sigma):
        """
        Compute a Toeplitz kernel (transcendental part) based on the sorted points.
        Here we use a Gaussian kernel (depends on differences in index, not on actual coordinates).
        But the client could also compute x^i, y^i if needed.
        """
        K = len(points_sorted)
        kernel = np.zeros(2*K-1, dtype=complex)
        for d in range(-(K-1), K):
            # Example: kernel based on index difference (Gaussian)
            kernel[d + (K-1)] = math.exp(-(d*d) / (2*sigma*sigma))
        return kernel

# ------------------------------------------------------------
# 3. Example usage
# ------------------------------------------------------------
def main():
    K = 100
    server = Server(K)
    client = Client()
    server.set_client(client)

    sigma = 2.0
    M = 30
    compressed, points_sorted, weights_sorted = server.run_compression(sigma, M, method='magnitude')

    # Full convolution for comparison
    kernel = client.compute_kernel(points_sorted, sigma)
    conv_full = server._apply_toeplitz(weights_sorted, kernel)

    # Reconstruct from compressed
    conv_recon = np.zeros(K, dtype=complex)
    for i, val in compressed.items():
        conv_recon[i] = val

    error = np.linalg.norm(conv_full - conv_recon) / np.linalg.norm(conv_full)
    print(f"K = {K}, M = {M}")
    print(f"Relative error (magnitude compression): {error:.4f}")
    print(f"Compressed size: {len(compressed)} (out of {K})")

if __name__ == "__main__":
    main()

def compute_kernel(self, points_sorted, sigma):
    K = len(points_sorted)
    kernel = np.zeros(2*K-1, dtype=complex)
    # For each lag d, kernel[d] = (1/K) * Σ_i x_i^{?} ... 
    # This would be O(K^2) if naive; we can make it O(K log K) using FFT if the kernel is translation‑invariant.
    # Simpler: we'll just use a Gaussian in index for demonstration.

input('Press ENTER to exit')