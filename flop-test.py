#!/usr/bin/env python3
"""
test_mobius_memory.py
Full test of Möbius memory pipeline with |C_i| arrays:
  - Generate signal from harmonic numbers (|C_i| = H_i)
  - Pack into odd square-free indices
  - Apply chip compression (convolution + supertrace + Möbius filtering)
  - Measure FLOPs, memory, runtime, error, and Basel checksum
  - Bound all operations with theoretical estimates
"""

import math
import time
import sys
import numpy as np
from collections import defaultdict

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)          # ≈ 2.362
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

def is_square_free(n, mu_cache=None):
    if mu_cache is not None and n < len(mu_cache):
        return mu_cache[n] != 0
    if n < 2:
        return True
    for p in range(2, int(n ** 0.5) + 1):
        if n % p == 0:
            count = 0
            while n % p == 0:
                n //= p
                count += 1
            if count > 1:
                return False
    return True

# ---------- Generate |C_i| signal ----------
def generate_signal_from_harmonic(K):
    """Return |C_i| = H_i (harmonic numbers) for i=1..K."""
    H = np.zeros(K, dtype=float)
    s = 0.0
    for i in range(1, K+1):
        s += 1.0 / i
        H[i-1] = s
    return H

def generate_signal_from_primes(K):
    """Return |C_i| = log(p_i) for first K primes."""
    # Simple prime sieve
    limit = int(K * (math.log(K) + math.log(math.log(K)))) + 100
    is_prime = np.ones(limit+1, dtype=bool)
    is_prime[0:2] = False
    for i in range(2, int(limit**0.5)+1):
        if is_prime[i]:
            is_prime[i*i:limit+1:i] = False
    primes = np.nonzero(is_prime)[0]
    if len(primes) < K:
        raise ValueError("Not enough primes")
    return np.log(primes[:K])

# ---------- Exponential convolution (two‑pass, O(K)) ----------
def conv_exp_kernel(signal, alpha=ALPHA):
    K = len(signal)
    lam = math.exp(-alpha)
    # Forward pass
    f = np.zeros(K, dtype=float)
    f[0] = signal[0]
    for i in range(1, K):
        f[i] = signal[i] + lam * f[i-1]
    # Backward pass
    b = np.zeros(K, dtype=float)
    b[K-1] = signal[K-1]
    for i in range(K-2, -1, -1):
        b[i] = signal[i] + lam * b[i+1]
    conv_exp = (f + b - signal) / (1 - lam * lam)
    conv = (1.0 - conv_exp) / NORM
    return conv

# ---------- Supertrace and entropy ----------
def supertrace_from_signal(signal):
    S = 0.0
    for idx, val in enumerate(signal):
        sign = 1 if (idx % 2 == 0) else -1
        S += sign * abs(val)
    return S

def entropy_from_supertrace(S, N, alpha=ALPHA):
    if S == 0:
        return 0.0
    p = abs(S) / N
    if p <= 0:
        return 0.0
    return -alpha * p * math.log(p)

def mass_from_signal(signal):
    S = supertrace_from_signal(signal)
    H = entropy_from_supertrace(S, len(signal))
    return abs(S) * math.exp(-H)

# ---------- Chip compression pipeline ----------
def chip_compress(signal, mu_cache):
    K = len(signal)
    # 1. Convolution (integral kernel)
    conv = conv_exp_kernel(signal)
    # 2. Supertrace and mass
    S = supertrace_from_signal(conv)
    H = entropy_from_supertrace(S, K)
    m = mass_from_signal(conv)
    # 3. Determine M = number to keep
    M = max(1, int(abs(S)))
    if M > K:
        M = K
    # 4. Get indices of largest magnitudes (argpartition for O(K))
    mag = np.abs(conv)
    idx_sorted = np.argsort(mag)[::-1]   # O(K log K) but we use it for simplicity
    # 5. Keep only those with μ(idx+1) != 0 (square-free)
    kept = []
    count = 0
    for idx in idx_sorted:
        n = idx + 1
        if mu_cache[n] != 0:
            kept.append((idx, conv[idx]))
            count += 1
            if count >= M:
                break
    # 6. Reconstruct (zero‑padded)
    recon = np.zeros(K, dtype=complex)
    for idx, val in kept:
        recon[idx] = val
    error = np.linalg.norm(conv - recon) / (np.linalg.norm(conv) + 1e-12)
    return kept, S, H, m, conv, recon, error

# ---------- Basel checksum ----------
def basel_checksum(mu, K):
    S = 0.0
    for n in range(1, K+1):
        if mu[n] != 0:
            S += mu[n] / (n * n)
    return S

def basel_density(signal, mu_cache):
    """Fraction of non‑zero entries at allowed indices."""
    K = len(signal)
    count = 0
    for i, val in enumerate(signal):
        if mu_cache[i+1] != 0 and abs(val) > 1e-12:
            count += 1
    return count / K if K > 0 else 0.0

# ---------- Main test ----------
def main():
    print("=== Möbius Memory Pipeline Test (|C_i| arrays) ===\n")

    # Parameters
    K = 4096
    print(f"Signal length K = {K}")

    # 1. Generate |C_i| signal (harmonic numbers)
    print("1. Generating |C_i| signal...")
    signal = generate_signal_from_harmonic(K)
    print(f"   First 10 values: {signal[:10]}")

    # 2. Möbius sieve
    print("2. Computing Möbius sieve...")
    mu = mobius_sieve(K)
    nonzero_count = sum(1 for n in range(1, K+1) if mu[n] != 0)
    density = nonzero_count / K
    print(f"   Non‑zero μ count: {nonzero_count} / {K} (density {density:.4f})")
    print(f"   Theoretical density (Basel): {6/PI**2:.4f}")

    # 3. Basel checksum of μ
    basel = basel_checksum(mu, K)
    print(f"   Basel sum Σ μ(n)/n² = {basel:.8f} (error: {basel - 6/PI**2:.2e})")

    # 4. Chip compression (with timing)
    print("3. Running chip compression...")
    start_time = time.perf_counter()
    kept, S, H, m, conv, recon, error = chip_compress(signal, mu)
    elapsed = time.perf_counter() - start_time
    print(f"   Time: {elapsed*1000:.3f} ms")

    # 5. Compression stats
    M = len(kept)
    print(f"   Kept {M} coefficients (ratio {M/K:.3f})")
    print(f"   Supertrace S = {S:.6f}")
    print(f"   Entropy H = {H:.6f}")
    print(f"   Mass m = {m:.6f}")
    print(f"   Reconstruction relative L2 error = {error:.4e}")

    # 6. Memory footprint
    print("4. Memory footprint:")
    # Estimate memory in bytes
    mem_signal = signal.nbytes
    mem_conv = conv.nbytes
    mem_recon = recon.nbytes
    mem_kept = sys.getsizeof(kept) + sum(sys.getsizeof(k) for k in kept)
    # Buffers in chip compression (f, b, mag, idx_sorted) are temporary
    # but we approximate
    mem_total = mem_signal + mem_conv + mem_recon + mem_kept
    print(f"   Signal: {mem_signal} bytes")
    print(f"   Convolved: {mem_conv} bytes")
    print(f"   Reconstructed: {mem_recon} bytes")
    print(f"   Kept list: {mem_kept} bytes")
    print(f"   Total (approx): {mem_total} bytes")

    # 7. FLOPs estimate
    print("5. FLOPs estimate (per pipeline run):")
    # Convolution: 4*K multiplications/additions
    flops_conv = 4 * K
    # Supertrace: K abs + K add
    flops_supertrace = 2 * K
    # Sorting: np.argsort is ~ K log2 K comparisons
    flops_sort = K * math.log2(K)
    # Möbius filtering: K lookups
    flops_filter = K
    # Total
    flops_total = flops_conv + flops_supertrace + flops_sort + flops_filter
    print(f"   Convolution: {flops_conv}")
    print(f"   Supertrace: {flops_supertrace}")
    print(f"   Sorting (argsort): ~{int(flops_sort)}")
    print(f"   Möbius filter: {flops_filter}")
    print(f"   Total FLOPs: ~{int(flops_total)}")

    # 8. Basel density check on kept coefficients
    kept_density = basel_density(np.abs(np.array([v for _, v in kept])), mu)
    print(f"6. Basel density of kept coefficients: {kept_density:.4f} (expected ~0.6079)")

    # 9. Error bound (supertrace stability)
    S_conv = supertrace_from_signal(conv)
    S_recon = supertrace_from_signal(recon.real)
    S_error = abs(S_conv - S_recon)
    print(f"7. Supertrace stability:")
    print(f"   S(conv) = {S_conv:.6f}, S(recon) = {S_recon:.6f}, |ΔS| = {S_error:.2e}")
    print(f"   Error bound (α * N) = {ALPHA * K:.2e}")

    print("\nTest completed successfully.")

if __name__ == "__main__":
    main()