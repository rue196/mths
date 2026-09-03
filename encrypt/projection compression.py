import numpy as np
import math
import cmath
from numpy.fft import fft, ifft

# ------------------------------------------------------------
# 1. Generate 12 random vertices in R^6
# ------------------------------------------------------------
def generate_vertices(seed=42):
    np.random.seed(seed)
    # 12 vertices, each in R^6 (6 coordinates)
    vertices = np.random.randn(12, 6)
    return vertices

# ------------------------------------------------------------
# 2. Build the rank‑6 Levi‑Civita tensor (fully antisymmetric)
# ------------------------------------------------------------
def epsilon_tensor():
    # For 6D, the Levi-Civita tensor has 6^6 = 46656 entries.
    # We'll implement it as a function that returns the sign of a permutation.
    # For simplicity, we compute the contraction on the fly for a given set of indices.
    from itertools import permutations
    # Precompute a dictionary for all permutations of (0,1,2,3,4,5)
    eps = {}
    for perm in permutations(range(6)):
        eps[perm] = 1 if len(set(perm)) == 6 else 0  # only full permutations
        # sign: parity of permutation
        # We'll compute sign using inversion count
        inv = 0
        for i in range(6):
            for j in range(i+1,6):
                if perm[i] > perm[j]:
                    inv += 1
        eps[perm] = (-1)**inv if len(set(perm)) == 6 else 0
    return eps

# ------------------------------------------------------------
# 3. Spinor projection operator Π
#    Contract the 12 vertices with the Levi-Civita tensor
#    to produce a scalar invariant.
# ------------------------------------------------------------
def spinor_projection(vertices, eps):
    # We need to choose a contraction pattern. For simplicity,
    # we take the product of the 12 vertices in a certain order,
    # but the actual contraction is more complex.
    # Here we compute a scalar invariant by contracting the vertices
    # with the epsilon tensor: e_{i1...i6} * v1_{i1} * v2_{i2} * ... * v6_{i6}
    # We'll take the first 6 vertices and contract.
    # This is a simplified version; the full projection would involve all 12 vertices.
    # For demonstration, we compute a single scalar.
    # We'll use the first 6 vertices for the contraction.
    v = vertices[:6]  # shape (6,6)
    # Compute sum over all index permutations
    scalar = 0.0
    for perm, sign in eps.items():
        if sign == 0:
            continue
        prod = 1.0
        for idx, coord_idx in enumerate(perm):
            prod *= v[idx][coord_idx]
        scalar += sign * prod
    return scalar

# ------------------------------------------------------------
# 4. Generate a dataset of projections under spinor rotations
# ------------------------------------------------------------
def generate_dataset(num_samples=1000, seed=42):
    # Generate num_samples different configurations of 12 vertices
    # and apply spinor rotations e^{iθ} to each.
    np.random.seed(seed)
    dataset = []
    for _ in range(num_samples):
        vertices = generate_vertices(seed=None)  # random each time
        eps = epsilon_tensor()
        # For each sample, compute the projection at several θ values?
        # Alternatively, we can compute a feature vector per sample.
        # We'll compute the projection for a fixed θ=0 (no rotation) and store.
        scalar = spinor_projection(vertices, eps)
        # We'll also compute the projection after rotating each coordinate pair by θ.
        # For simplicity, we just store the vertices and the scalar.
        dataset.append((vertices, scalar))
    return dataset

# ------------------------------------------------------------
# 5. Spectral compression pipeline (from earlier)
# ------------------------------------------------------------
def sort_by_angle(points, values):
    angles = np.angle(points[:,0] + 1j * points[:,1])
    idx = np.argsort(angles)
    return points[idx], values[idx]

def toeplitz_kernel(K, sigma=1.0):
    kernel = np.zeros(2*K-1, dtype=float)
    for d in range(-(K-1), K):
        kernel[d + (K-1)] = math.exp(-(d*d) / (2*sigma*sigma))
    return kernel

def apply_convolution(signal, kernel):
    K = len(signal)
    N = 1 << (2*K - 1).bit_length()
    sig_pad = np.pad(signal, (0, N - K), mode='constant')
    ker_pad = np.pad(kernel, (0, N - (2*K - 1)), mode='constant')
    return ifft(fft(sig_pad) * fft(ker_pad))[:K]

def compress_spectral(points, values, M, sigma=1.0):
    pts_sorted, vals_sorted = sort_by_angle(points, values)
    K = len(vals_sorted)
    kernel = toeplitz_kernel(K, sigma)
    conv = apply_convolution(vals_sorted, kernel)
    idx = np.argsort(np.abs(conv))[::-1][:M]
    compressed = {int(i): conv[i] for i in idx}
    return compressed, pts_sorted, vals_sorted

# ------------------------------------------------------------
# 6. Main demonstration
# ------------------------------------------------------------
def main():
    # Generate a dataset of 1000 samples of 12 vertices in R^6
    # For each sample, we compute a feature: e.g., the scalar projection.
    # We'll treat the 12 vertices as 12 "points" in 6D, but we need a 2D projection for compression.
    # We'll map each vertex to a complex number by taking the first two coordinates.
    # Then we compress the sequence of these complex numbers (length 12) using spectral compression.
    
    num_samples = 1000
    # Instead of generating 1000 datasets, we'll generate one dataset of 12 vertices
    # and apply the spinor projection at many θ angles.
    # This yields a sequence of scalars (one per θ) which we can compress.
    
    # Choose a fixed set of vertices
    vertices = generate_vertices(seed=42)
    eps = epsilon_tensor()
    
    # Generate θ values from 0 to 2π
    theta_values = np.linspace(0, 2*np.pi, 1000)
    projections = []
    for theta in theta_values:
        # Rotate each coordinate pair (x,y) by θ: (x,y) -> (x cosθ - y sinθ, x sinθ + y cosθ)
        # Apply to all 12 vertices
        rotated = vertices.copy()
        # For each vertex, apply rotation to each pair (0,1), (2,3), (4,5)
        for v in rotated:
            # pair 0-1
            x, y = v[0], v[1]
            v[0] = x * math.cos(theta) - y * math.sin(theta)
            v[1] = x * math.sin(theta) + y * math.cos(theta)
            # pair 2-3
            x, y = v[2], v[3]
            v[2] = x * math.cos(theta) - y * math.sin(theta)
            v[3] = x * math.sin(theta) + y * math.cos(theta)
            # pair 4-5
            x, y = v[4], v[5]
            v[4] = x * math.cos(theta) - y * math.sin(theta)
            v[5] = x * math.sin(theta) + y * math.cos(theta)
        # Compute projection (scalar) for this rotated set
        scalar = spinor_projection(rotated, eps)
        projections.append(scalar)
    
    # Now we have a sequence of 1000 scalars (projections vs θ).
    # We treat this sequence as a signal to compress.
    # We'll create points (x,y) for each θ: (θ, 0) but we need 2D points.
    # For compression, we'll use θ as the x-coordinate and 0 as y.
    # Actually, we can just use the index as the "angle" for sorting.
    # We'll sort by the angle which is already θ.
    # So we set points = (θ, 0) for each sample.
    points = np.array([[theta, 0] for theta in theta_values])
    values = np.array(projections, dtype=complex)  # they are real, but treat as complex
    
    # Now compress
    M = 300   # keep 30%
    sigma = 2.0
    compressed, pts_sorted, vals_sorted = compress_spectral(points, values, M, sigma)
    K = len(values)
    # Reconstruct
    recon = np.zeros(K, dtype=complex)
    for i, v in compressed.items():
        recon[i] = v
    error = np.linalg.norm(recon - apply_convolution(vals_sorted, toeplitz_kernel(K, sigma)))
    print(f"K={K}, M={M}, storage ratio={M/K:.2f}, recon error={error:.4e}")
    
    # Also verify that the projection is invariant under global spinor rotation?
    # The theorem states that Π is invariant under global rotation e^{iθ}.
    # Our projections array is constant? Actually, if we rotate all vertices globally,
    # the contraction should be invariant, so all projections should be the same.
    # Let's check the variance of the projections.
    mean_proj = np.mean(projections)
    var_proj = np.var(projections)
    print(f"Mean projection: {mean_proj:.6f}, variance: {var_proj:.6e}")
    print("If variance is near zero, the projection is indeed invariant under global rotation.")
    # The variance should be very small (close to machine epsilon).

if __name__ == "__main__":
    main()

input('Press ENTER to exit')