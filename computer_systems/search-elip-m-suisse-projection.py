import math
import random
import numpy as np
import cmath

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)          # ≈ 2.362
NORM = 1.0 - math.exp(-ALPHA * (PI + E))

# ---------- 1. Address generation (elliptic‑curve points simulated) ----------
def generate_addresses(K, seed=42):
    random.seed(seed)
    # Random (x,y) in [0.5, 5.0] to avoid singularities
    addresses = [(random.uniform(0.5, 5.0), random.uniform(0.5, 5.0)) for _ in range(K)]
    return addresses

# ---------- 2. Matrix trace (scalar) ----------
def matrix_trace(x, y, i=2):
    """
    Trace of M = [[x^(i-1), x^(-1)*y^i],
                  [y^(-1)*x^i, y^(i-1)]]
    tr = x^(i-1) + y^(i-1)
    """
    return x ** (i-1) + y ** (i-1)

# ---------- 3. Möbius sieve (linear, O(K)) ----------
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

# ---------- 4. TSP routing (bucket sort by angle) ----------
def tsp_route(addresses):
    # Convert to complex, get phase, bucket sort
    angles = [cmath.phase(complex(x, y)) for x, y in addresses]
    # Bucket sort: 360 buckets
    buckets = [[] for _ in range(360)]
    for idx, a in enumerate(angles):
        # map a from [-π, π] to [0, 2π)
        a_norm = a + PI if a < 0 else a
        b = int((a_norm / (2 * PI)) * 360) % 360
        buckets[b].append(idx)
    order = []
    for b in buckets:
        order.extend(b)
    return order

# ---------- 5. Integral kernel convolution (exponential filter, O(K)) ----------
def conv_exp_kernel(signal, alpha=ALPHA):
    K = len(signal)
    lam = math.exp(-alpha)
    # Forward pass
    f = np.zeros(K)
    f[0] = signal[0]
    for i in range(1, K):
        f[i] = signal[i] + lam * f[i-1]
    # Backward pass
    b = np.zeros(K)
    b[K-1] = signal[K-1]
    for i in range(K-2, -1, -1):
        b[i] = signal[i] + lam * b[i+1]
    # Convolution with exp(-alpha|i-j|)
    conv_exp = (f + b - signal) / (1 - lam * lam)
    # Integral kernel: (1 - conv_exp) / NORM
    conv = (1.0 - conv_exp) / NORM
    return conv

# ---------- 6. Supertrace and mass ----------
def supertrace_and_mass(signal):
    S = 0.0
    for i, val in enumerate(signal):
        sign = 1 if (i % 2 == 0) else -1
        S += sign * abs(val)
    if S == 0:
        H = 0.0
        m = 0.0
    else:
        p = abs(S) / len(signal)
        H = -ALPHA * p * math.log(p) if p > 0 else 0.0
        m = abs(S) * math.exp(-H)
    return S, H, m

# ---------- 7. Main compression pipeline ----------
def compress_scalar_polynomial(addresses, i_exp=2):
    K = len(addresses)
    print(f"Number of addresses: {K}")

    # 1. Compute coefficients (traces)
    coeffs = np.array([matrix_trace(x, y, i_exp) for x, y in addresses], dtype=float)
    print(f"Coefficients (first 5): {coeffs[:5]}")

    # 2. TSP routing (reorder addresses and coefficients)
    order = tsp_route(addresses)
    coeffs_sorted = coeffs[order]

    # 3. Convolution with integral kernel
    conv = conv_exp_kernel(coeffs_sorted)

    # 4. Supertrace
    S, H, m = supertrace_and_mass(conv)
    M = max(1, int(abs(S)))
    if M > K:
        M = K
    print(f"Supertrace S = {S:.4f}, Entropy H = {H:.4f}, Mass m = {m:.4f}")
    print(f"Keeping M = {M} coefficients (based on |S|)")

    # 5. Möbius sieve (square‑free indices)
    mu = mobius_sieve(K)   # mu[0] unused

    # 6. Compression: keep top M magnitudes with μ(index) != 0
    mag = np.abs(conv)
    idx_sorted = np.argsort(mag)[::-1]
    kept = []
    count = 0
    for idx in idx_sorted:
        n = idx + 1   # 1‑based for μ
        if mu[n] != 0:
            kept.append((idx, conv[idx]))
            count += 1
            if count >= M:
                break
    print(f"Compressed size: {len(kept)} (ratio {len(kept)/K:.3f})")

    # 7. Reconstruct (zero out non‑kept)
    recon = np.zeros(K, dtype=complex)
    for idx, val in kept:
        recon[idx] = val
    error = np.linalg.norm(conv - recon) / np.linalg.norm(conv)
    print(f"Reconstruction relative L2 error: {error:.4e}")

    return kept, S, H, m, conv, coeffs, order

# ---------- Demonstration ----------
def main():
    K = 200
    addresses = generate_addresses(K, seed=42)
    kept, S, H, m, conv, coeffs, order = compress_scalar_polynomial(addresses, i_exp=2)

    # Print first few kept entries
    print("\nFirst 5 kept (index, coefficient):")
    for idx, val in kept[:5]:
        print(f"  {idx}: {val:.4f}")

if __name__ == "__main__":
    main()