import numpy as np
import math
import cmath
from numpy.fft import fft, ifft

# ---------- 1. Generate synthetic data for Δ(x,y) ----------
def generate_data(K, seed=42):
    np.random.seed(seed)
    # random points in complex plane
    points = np.random.uniform(-5, 5, size=(K, 2))
    # Δ = x^{-1} y^i - y^{-1} x^i, with i = 1j (imag unit)
    x = points[:,0]; y = points[:,1]
    # Avoid singularities
    x = np.where(np.abs(x) < 1e-8, 1e-8, x)
    y = np.where(np.abs(y) < 1e-8, 1e-8, y)
    # i = 1j
    Delta = (1/x) * np.exp(1j * np.log(y)) - (1/y) * np.exp(1j * np.log(x))
    return points, Delta

# ---------- 2. Sort by angle ----------
def sort_by_angle(points, values):
    angles = np.angle(points[:,0] + 1j * points[:,1])
    idx = np.argsort(angles)
    return points[idx], values[idx]

# ---------- 3. Gaussian Toeplitz kernel ----------
def toeplitz_kernel(K, sigma=1.0):
    kernel = np.zeros(2*K-1, dtype=float)
    for d in range(-(K-1), K):
        kernel[d + (K-1)] = math.exp(-(d*d) / (2*sigma*sigma))
    return kernel

# ---------- 4. FFT convolution ----------
def apply_convolution(signal, kernel):
    K = len(signal)
    N = 1 << (2*K - 1).bit_length()
    sig_pad = np.pad(signal, (0, N - K), mode='constant')
    ker_pad = np.pad(kernel, (0, N - (2*K - 1)), mode='constant')
    return ifft(fft(sig_pad) * fft(ker_pad))[:K]

# ---------- 5. Compression ----------
def compress_spectral(points, values, M, sigma=1.0):
    # Sort by angle
    pts_sorted, vals_sorted = sort_by_angle(points, values)
    K = len(vals_sorted)
    # Convolve
    kernel = toeplitz_kernel(K, sigma)
    conv = apply_convolution(vals_sorted, kernel)
    # Keep M largest magnitudes
    idx = np.argsort(np.abs(conv))[::-1][:M]
    compressed = {int(i): conv[i] for i in idx}
    return compressed, pts_sorted, vals_sorted

# ---------- 6. Example ----------
K = 1000
M = 300   # keep 30% → storage O(M) = O(K)
sigma = 2.0
points, values = generate_data(K)

compressed, pts_sorted, vals_sorted = compress_spectral(points, values, M, sigma)

# Reconstruct for error check
recon = np.zeros(K, dtype=complex)
for i, v in compressed.items():
    recon[i] = v
error = np.linalg.norm(recon - apply_convolution(vals_sorted, toeplitz_kernel(K, sigma)))
print(f"K={K}, M={M}, storage ratio={M/K:.2f}, recon error={error:.4e}")


input('Press ENTER to exit')
