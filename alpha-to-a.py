import math
import time

# ---------- Constants ----------
ALPHA_USER = 0.3628                     # user-provided
A = 1.0 / (math.pi - math.e)            # ≈ 2.362
ALPHA_PI_E = A                          # the original constant

# ---------- Sieve of Eratosthenes (O(N log log N)) ----------
def sieve_primes(N):
    is_prime = bytearray(b'\x01') * (N + 1)
    is_prime[0:2] = b'\x00\x00'
    limit = int(N ** 0.5)
    for i in range(2, limit + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start:N+1:step] = b'\x00' * ((N - start)//step + 1)
    return [i for i in range(2, N+1) if is_prime[i]]

# ---------- Supertrace and entropy ----------
def compute_invariants(primes, alpha):
    K = len(primes)
    if K == 0:
        return 0.0, 0.0, 0.0
    # Alternating sum of logs
    S = 0.0
    for i, p in enumerate(primes):
        S += (-1) ** i * math.log(p)
    # Entropy H = -alpha * (|S|/K) * log(|S|/K)
    if S == 0:
        H = 0.0
    else:
        p_norm = abs(S) / K
        H = -alpha * p_norm * math.log(p_norm) if p_norm > 0 else 0.0
    # Mass m = |S| * exp(-H)
    m = abs(S) * math.exp(-H)
    return S, H, m

# ---------- Main ----------
def main(N):
    start = time.time()
    primes = sieve_primes(N)
    K = len(primes)
    elapsed = time.time() - start
    if K == 0:
        print("No primes found.")
        return

    print(f"Sieve up to N = {N}: {K} primes, last prime = {primes[-1]}")
    print(f"Sieve time: {elapsed:.3f}s (O(N log log N))")

    # Using alpha = 0.3628
    S, H1, m1 = compute_invariants(primes, ALPHA_USER)
    # Using alpha = a = 1/(π-e)
    _, H2, m2 = compute_invariants(primes, ALPHA_PI_E)

    print("\n--- Supertrace and invariants ---")
    print(f"Supertrace S = {S:.6f}")
    print(f"Entropy with alpha={ALPHA_USER:.4f}: H1 = {H1:.6f}, mass m1 = {m1:.6f}")
    print(f"Entropy with alpha={ALPHA_PI_E:.4f}: H2 = {H2:.6f}, mass m2 = {m2:.6f}")

    # Simple bound check: |S| ≤ A * K ?
    bound = A * K
    print(f"\nBound check: |S| ≤ a * K = {A:.4f} * {K} = {bound:.2f}")
    print(f"  |S| = {abs(S):.6f}  →  {'within' if abs(S) <= bound else 'exceeds'} bound")

    # Twin prime count (optional)
    twin_count = sum(1 for i in range(K-1) if primes[i+1] - primes[i] == 2)
    print(f"Twin prime pairs (p, p+2) within range: {twin_count}")

if __name__ == "__main__":
    N = 4_000_000   # adjust as needed
    main(N)