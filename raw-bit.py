import math
import struct

def mobius_sieve(K):
    """Linear sieve for μ(n), returns list mu[0..K] in O(K)."""
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

def pack_mobius_bits(mu, bit_width=64):
    """
    Pack the symmetric array μ(-K)..μ(K) into a list of integers.
    Each value uses 2 bits: 00=0, 01=1, 10=-1.
    Returns: bytes object (big‑endian packed bits).
    """
    K = len(mu) - 1
    total_values = 2 * K + 1          # indices -K .. K
    bits_needed = total_values * 2
    num_bytes = (bits_needed + 7) // 8
    if bit_width == 64:
        num_ints = (num_bytes + 7) // 8
    elif bit_width == 32:
        num_ints = (num_bytes + 3) // 4
    else:
        raise ValueError("bit_width must be 32 or 64")

    # We'll pack into a bytearray first, then convert to integers
    packed = bytearray(num_bytes)
    for i, val in enumerate(mu):      # i corresponds to μ(i) for i=0..K
        # We need to write μ(+i) and μ(-i) for i>0, and μ(0) for i=0.
        # Order: μ(-K), μ(-K+1), ..., μ(-1), μ(0), μ(+1), ..., μ(+K)
        # We'll write in that order.
        pass

    # Actually easier: build a list of values in the desired order
    values = []
    for i in range(K, 0, -1):
        values.append(mu[i])   # μ(-i)
    values.append(mu[0])       # μ(0)
    for i in range(1, K+1):
        values.append(mu[i])   # μ(+i)

    # Now pack each value into 2 bits
    bit_pos = 0
    for v in values:
        code = 0 if v == 0 else (1 if v == 1 else 2)  # 2 for -1
        # pack code into 2 bits (MSB first)
        shift = 6 - (bit_pos % 8)   # because 2 bits per value, we pack from MSB
        if shift < 0:
            # not possible with 2-bit codes, but keep for safety
            shift = 0
        # We need to write code into the byte at bit_pos//8
        byte_idx = bit_pos // 8
        if byte_idx >= len(packed):
            # should not happen
            break
        packed[byte_idx] |= (code << shift)
        bit_pos += 2

    # Now convert to list of integers of the chosen bit width
    if bit_width == 64:
        fmt = '>{}Q'.format(num_ints)
    else:
        fmt = '>{}I'.format(num_ints)
    # pad bytes to multiple of 8/4
    if bit_width == 64:
        extra = (8 - (len(packed) % 8)) % 8
    else:
        extra = (4 - (len(packed) % 4)) % 4
    packed += b'\x00' * extra
    ints = struct.unpack(fmt, bytes(packed))
    return ints, packed

# Example: K = 10
K = 10
mu = mobius_sieve(K)
ints, raw_bytes = pack_mobius_bits(mu, bit_width=64)

print(f"K = {K}, 2K+1 = {2*K+1} values")
print(f"μ(0..K): {mu}")
print("Packed as 64-bit integers (hex):")
for i, val in enumerate(ints):
    print(f"  [{i}] 0x{val:016X}")

# Show raw bits as binary for the first few values (optional)
print("\nRaw bits (first 64 bits):")
bits = bin(ints[0])[2:].zfill(64)
print(bits)