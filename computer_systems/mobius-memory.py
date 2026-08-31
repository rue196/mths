import math
import struct

def mobius_sieve(K):
    """Linear sieve for Möbius function, returns list mu[0..K]."""
    if K < 1:
        return [0] * (K + 1)
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

# ---------- Compression ----------
def compress_mobius(mu):
    """
    Compress Möbius array into bytes.
    Format:
      - 4 bytes: K (length-1)
      - 4 bytes: abs_sum = Σ|μ(i)|  (key)
      - bit-packed data: 2 bits per entry (i=1..K), packed big-endian.
    Returns bytes object.
    """
    K = len(mu) - 1
    if K == 0:
        return struct.pack('>II', 0, 0)

    # Compute absolute sum (key)
    abs_sum = sum(abs(mu[i]) for i in range(1, K+1))

    # Pack bits: 2 bits per value, MSB first
    # Map: 0 -> 0b00, 1 -> 0b01, -1 -> 0b10
    bit_length = 2 * K
    num_bytes = (bit_length + 7) // 8
    packed = bytearray(num_bytes)

    for i in range(1, K+1):
        val = mu[i]
        # encode: 0->0, 1->1, -1->2
        code = 0 if val == 0 else (1 if val == 1 else 2)
        # bit position: (i-1)*2
        bit_pos = (i - 1) * 2
        byte_idx = bit_pos // 8
        bit_offset = bit_pos % 8
        # store code in bits bit_offset and bit_offset+1
        packed[byte_idx] |= (code << (6 - bit_offset))  # because 2 bits, MSB aligned

    # Build header: K and abs_sum
    header = struct.pack('>II', K, abs_sum)
    return header + bytes(packed)

# ---------- Decompression ----------
def decompress_mobius(data):
    """
    Decompress bytes back to Möbius array.
    Raises ValueError if key does not match.
    """
    if len(data) < 8:
        raise ValueError("Data too short")
    K, abs_sum = struct.unpack('>II', data[:8])
    if K == 0:
        return [0] * (K + 1)

    packed = data[8:]
    mu = [0] * (K + 1)
    mu[1] = 1  # will be overwritten, but safe

    # Decode bits
    bit_length = 2 * K
    for i in range(1, K+1):
        bit_pos = (i - 1) * 2
        byte_idx = bit_pos // 8
        bit_offset = bit_pos % 8
        if byte_idx >= len(packed):
            raise ValueError("Insufficient data")
        # extract 2 bits from the byte
        code = (packed[byte_idx] >> (6 - bit_offset)) & 0b11
        if code == 0:
            mu[i] = 0
        elif code == 1:
            mu[i] = 1
        elif code == 2:
            mu[i] = -1
        else:
            raise ValueError(f"Invalid code {code} at position {i}")

    # Verify key
    computed_sum = sum(abs(mu[i]) for i in range(1, K+1))
    if computed_sum != abs_sum:
        raise ValueError(f"Integrity check failed: expected abs_sum={abs_sum}, got {computed_sum}")
    return mu


# ---------- Example usage ----------
def main():
    K = 200
    mu_original = mobius_sieve(K)
    print(f"Original array (first 20): {mu_original[1:21]}")

    # Compress
    compressed = compress_mobius(mu_original)
    print(f"Compressed size: {len(compressed)} bytes (was {len(mu_original)*8} bytes as Python ints)")

    # Decompress
    mu_reconstructed = decompress_mobius(compressed)
    print(f"Reconstructed (first 20): {mu_reconstructed[1:21]}")

    # Check equality
    assert mu_original == mu_reconstructed, "Reconstruction mismatch"
    print("Success: original and reconstructed match.")

    # Demonstrate the key
    abs_sum = sum(abs(mu_original[i]) for i in range(1, K+1))
    print(f"Absolute sum (key) = {abs_sum}")

if __name__ == "__main__":
    main()