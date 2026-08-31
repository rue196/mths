import math
import struct

def mobius_sieve(K):
    """Linear sieve for μ(1..K). Returns list mu[0..K] (mu[0]=0)."""
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

def pack_mobius(mu):
    """
    Pack mu[1..K] into bytes (2 bits per value).
    Returns bytes object.
    """
    K = len(mu) - 1
    if K < 1:
        return b''
    bit_len = 2 * K
    num_bytes = (bit_len + 7) // 8
    packed = bytearray(num_bytes)
    bit_pos = 0
    for n in range(1, K + 1):
        val = mu[n]
        code = 0 if val == 0 else (1 if val == 1 else 2)  # -1 -> 2
        # store code in 2 bits (MSB first) at bit_pos
        byte_idx = bit_pos // 8
        bit_offset = 6 - (bit_pos % 8)  # because 2 bits, we shift into MSB
        packed[byte_idx] |= (code << bit_offset)
        bit_pos += 2
    return bytes(packed)

def unpack_mobius(data, K):
    """Unpack bytes back to mu[1..K]."""
    if K < 1:
        return [0] * (K + 1)
    mu = [0] * (K + 1)
    bit_pos = 0
    for n in range(1, K + 1):
        byte_idx = bit_pos // 8
        bit_offset = 6 - (bit_pos % 8)
        if byte_idx >= len(data):
            raise ValueError("Data too short")
        code = (data[byte_idx] >> bit_offset) & 0b11
        if code == 0:
            mu[n] = 0
        elif code == 1:
            mu[n] = 1
        elif code == 2:
            mu[n] = -1
        else:
            raise ValueError(f"Invalid code {code} at position {n}")
        bit_pos += 2
    return mu

def basel_checksum(mu, K):
    """Compute Σ μ(n)/n² for n=1..K."""
    S = 0.0
    for n in range(1, K + 1):
        if mu[n] != 0:
            S += mu[n] / (n * n)
    return S

def compress_and_verify(K):
    """Generate, pack, unpack, and verify with Basel checksum."""
    mu = mobius_sieve(K)
    packed = pack_mobius(mu)
    print(f"K = {K}")
    print(f"Packed size: {len(packed)} bytes (was {K} values)")
    print(f"Compression ratio: {len(packed)*8 / K:.2f} bits per value")

    # Unpack and verify
    mu2 = unpack_mobius(packed, K)
    assert mu == mu2, "Unpack mismatch"
    S = basel_checksum(mu, K)
    S2 = basel_checksum(mu2, K)
    print(f"Basel sum (μ(n)/n²): {S:.8f}")
    print(f"Error from 6/π²: {S - 6/math.pi**2:.2e}")
    assert abs(S - S2) < 1e-12

    # Show non‑zero density
    nonzero = sum(1 for n in range(1, K+1) if mu[n] != 0)
    print(f"Non‑zero count: {nonzero} (density {nonzero/K:.4f})")
    print(f"Theoretical density (Basel): {6/math.pi**2:.4f}")

# Example
if __name__ == "__main__":
    compress_and_verify(200000)