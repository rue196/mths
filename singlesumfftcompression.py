import math
import random
import numpy as np
from numpy.fft import fft, ifft
import cmath

# ------------------------------------------------------------
# 1. Generate K random points (x,y) and weights w (complex)
# ------------------------------------------------------------
def generate_data(K, seed=42):
    random.seed(seed)
    points = [(random.uniform(-5,5), random.uniform(-5,5)) for _ in range(K)]
    weights = [complex(random.gauss(0,1), random.gauss(0,1)) for _ in range(K)]
    return points, weights

# ------------------------------------------------------------
# 2. Sort by angle (arg) to get a cyclic order
# ------------------------------------------------------------
def sort_by_angle(points, weights):
    # Convert to complex, compute angle, sort
    data = sorted(zip(points, weights), key=lambda p: cmath.phase(complex(p[0][0], p[0][1])))
    points_sorted = [p for p, w in data]
    weights_sorted = [w for p, w in data]
    return points_sorted, weights_sorted

# ------------------------------------------------------------
# 3. Build a Toeplitz kernel (Gaussian in index difference)
#    This kernel is translation‑invariant in the sorted order.
# ------------------------------------------------------------
def toeplitz_kernel(K, sigma):
    # Returns an array of length 2K-1 representing the kernel k[d] = exp(-d^2/(2*sigma^2))
    kernel = np.zeros(2*K-1)
    for d in range(-(K-1), K):
        kernel[d + (K-1)] = math.exp(-(d*d) / (2*sigma*sigma))
    return kernel

# ------------------------------------------------------------
# 4. Apply the Toeplitz operator via convolution (FFT)
#    result[i] = Σ_j kernel[i-j] * weight[j]
#    This is O(K log K) using FFT.
# ------------------------------------------------------------
def apply_toeplitz(weights, kernel):
    K = len(weights)
    # Use FFT convolution
    # Pad weights to length 2K-1
    w_pad = np.concatenate([weights, np.zeros(K-1, dtype=complex)])
    # FFT of weights and kernel
    W = fft(w_pad)
    K_fft = fft(kernel)
    Y = W * K_fft
    y = ifft(Y)
    # Extract first K elements (the convolution result)
    return y[:K]

# ------------------------------------------------------------
# 5. Compress by keeping only the largest magnitude coefficients
#    (or low‑frequency components). Here we keep top M.
# ------------------------------------------------------------
def compress_by_magnitude(weights, M):
    # Get indices sorted by absolute value descending
    idx_sorted = sorted(range(len(weights)), key=lambda i: abs(weights[i]), reverse=True)
    # Keep the first M indices and their values
    compressed = {i: weights[i] for i in idx_sorted[:M]}
    return compressed

def compress_by_lowpass(weights, M):
    # Keep only first M frequency components (low‑pass)
    # Transform to frequency domain
    W = fft(weights)
    # Keep only first M and last M-1? Actually low‑pass: keep first M/2 and symmetric.
    # For simplicity, keep first M components (0..M-1)
    W_comp = np.zeros_like(W)
    W_comp[:M] = W[:M]
    # Inverse transform
    weights_comp = ifft(W_comp)
    return weights_comp

# ------------------------------------------------------------
# 6. Full compression pipeline
# ------------------------------------------------------------
def compress_sum(points, weights, sigma=1.0, M=None, method='magnitude'):
    # Sort by angle (O(K log K))
    points_sorted, weights_sorted = sort_by_angle(points, weights)
    K = len(weights_sorted)
    # Build kernel (O(K))
    kernel = toeplitz_kernel(K, sigma)
    # Apply Toeplitz operator (convolution) (O(K log K))
    conv_result = apply_toeplitz(weights_sorted, kernel)
    # Compress: choose M = K//2 if not specified
    if M is None:
        M = K // 2
    if method == 'magnitude':
        compressed = compress_by_magnitude(conv_result, M)
        # Reconstructed sequence: we need to reconstruct an approximation of the original sum.
        # For simplicity, we can reconstruct by setting all other indices to zero and inverse transform?
        # Actually we want to compress the sum itself, i.e., represent the function by M coefficients.
        # We can store the compressed indices and values, and for reconstruction we can evaluate the sum
        # using only those terms. The compressed representation is just a sparse set of coefficients.
        # For demonstration, we'll return the compressed dict.
        return compressed, points_sorted, weights_sorted
    elif method == 'lowpass':
        weights_comp = compress_by_lowpass(conv_result, M)
        # We can store the compressed weights (complex array) and reconstruction is inverse transform.
        return weights_comp, points_sorted, weights_sorted
    else:
        raise ValueError("Unknown compression method.")

# ------------------------------------------------------------
# 7. Example usage and error measurement
# ------------------------------------------------------------
def main():
    K = 100
    points, weights = generate_data(K)
    # Original sum (without compression) – we can define a target function evaluation.
    # For demonstration, we'll evaluate the sum at a fixed point (x0, y0) using the Toeplitz kernel.
    # Actually we can just compare the convolution result before and after compression.
    # Compute full convolution result (original).
    sigma = 2.0
    kernel = toeplitz_kernel(K, sigma)
    points_sorted, weights_sorted = sort_by_angle(points, weights)
    conv_full = apply_toeplitz(weights_sorted, kernel)

    # Compress using magnitude threshold (keep top M)
    M = 30
    compressed, _, _ = compress_sum(points, weights, sigma, M, method='magnitude')
    # Reconstruct the convolution result from compressed coefficients (set others to zero)
    conv_recon = np.zeros(K, dtype=complex)
    for i, val in compressed.items():
        conv_recon[i] = val

    # Compute error
    error = np.linalg.norm(conv_full - conv_recon) / np.linalg.norm(conv_full)
    print(f"K = {K}, M = {M}")
    print(f"Relative error (magnitude compression): {error:.4f}")

    # Low‑pass compression
    weights_comp, _, _ = compress_sum(points, weights, sigma, M, method='lowpass')
    conv_recon_lp = apply_toeplitz(weights_comp, kernel)  # but we already have compressed weights; we need to reconstruct the conv result.
    # Actually low‑pass compresses the convolution result itself, not the weights.
    # We'll use the compressed convolution result directly.
    # For low‑pass, we stored the compressed convolution sequence.
    conv_lp = compress_by_lowpass(conv_full, M)
    error_lp = np.linalg.norm(conv_full - conv_lp) / np.linalg.norm(conv_full)
    print(f"Relative error (low‑pass compression): {error_lp:.4f}")

    # Additional: show the number of non‑zero coefficients
    print(f"Compressed size: {len(compressed)} (out of {K})")

if __name__ == "__main__":
    main()

input('Press ENTER to exit')