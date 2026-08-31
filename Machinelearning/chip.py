import math
import random
import numpy as np

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)          # ≈ 2.362
NORM = 1.0 - math.exp(-ALPHA * (PI + E))

# ---------- Möbius sieve (linear, O(K)) ----------
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

# ---------- TSP routing (bucket sort, O(K)) ----------
def tsp_route(signal):
    K = len(signal)
    # Use deterministic pseudo‑angles
    angles = np.zeros(K)
    for i in range(K):
        x = math.sin(i * 7.0) + 0.1 * math.cos(i * 13.0)
        y = math.cos(i * 11.0) + 0.1 * math.sin(i * 17.0)
        angles[i] = math.atan2(y, x) + math.pi
    # Bucket sort: 360 buckets
    buckets = [[] for _ in range(360)]
    for i, a in enumerate(angles):
        idx = int((a / (2 * math.pi)) * 360) % 360
        buckets[idx].append(i)
    order = []
    for b in buckets:
        order.extend(b)
    return order

# ---------- Exponential convolution (two‑pass, O(K)) ----------
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
    # Combine: convolution with exp(-alpha|i-j|) = (f[i] + b[i] - signal[i]) / (1 - lam^2)
    # Then integral kernel = (1 - conv_exp) / NORM
    conv_exp = (f + b - signal) / (1 - lam * lam)
    conv = (1.0 - conv_exp) / NORM
    return conv

# ---------- Supertrace and entropy ----------
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

# ---------- Chip simulation ----------
def chip_pipeline(signal):
    K = len(signal)
    print(f"Input signal length: {K}")

    # 1. TSP routing (reorder)
    order = tsp_route(signal)
    signal_sorted = signal[order]   # reorder in place (simulated)

    # 2. Convolution
    conv = conv_exp_kernel(signal_sorted)

    # 3. Supertrace
    S, H, m = supertrace_and_mass(conv)
    M = max(1, int(abs(S)))
    if M > K:
        M = K
    print(f"Supertrace S = {S:.4f}, Entropy H = {H:.4f}, Mass m = {m:.4f}")
    print(f"Keeping M = {M} coefficients")

    # 4. Möbius sieve (pre‑compiled on chip)
    mu = mobius_sieve(K)   # μ for indices 0..K; μ[0] unused

    # 5. Compression: keep top M with square‑free index (μ(n) != 0)
    # We'll get the magnitudes, then select
    mag = np.abs(conv)
    # Get indices sorted by magnitude descending
    sorted_idx = np.argsort(mag)[::-1]
    kept = []
    count = 0
    for idx in sorted_idx:
        n = idx + 1   # 1‑based for μ
        if mu[n] != 0:
            kept.append((idx, conv[idx]))
            count += 1
            if count >= M:
                break

    # 6. Output compressed data
    print(f"Compressed size: {len(kept)} (ratio {len(kept)/K:.3f})")
    return kept, S, H, m, conv

# ---------- Test ----------
def main():
    K = 200
    # Generate a test signal: harmonic numbers (smooth)
    signal = np.array([math.log(i+1) for i in range(K)])
    kept, S, H, m, conv = chip_pipeline(signal)

    # Reconstruct (zero out non‑kept)
    recon = np.zeros(K, dtype=complex)
    for idx, val in kept:
        recon[idx] = val
    error = np.linalg.norm(conv - recon) / np.linalg.norm(conv)
    print(f"Reconstruction relative L2 error = {error:.4e}")

    # Print first few kept entries
    print("\nFirst 5 kept (index, value):")
    for idx, val in kept[:5]:
        print(f"  {idx}: {val:.4f}")

if __name__ == "__main__":
    main()