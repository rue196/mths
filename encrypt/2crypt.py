import math
import hashlib
import struct
from typing import List, Tuple, Dict

# ------------------------------------------------------------
# 1. Compute spectral coefficients and entropy
# ------------------------------------------------------------
def spectral_entropy_and_coeffs(data: List[int], alpha: float = 0.3628, collatz_even: bool = True):
    if not data:
        return 0.0, [], 0

    freq = {}
    for ch in data:
        freq[ch] = freq.get(ch, 0) + 1
    total = len(data)
    probs = {ch: cnt / total for ch, cnt in freq.items()}
    
    letters = sorted(probs.keys())
    K = len(letters)
    C = [0j] * (2 * K + 1)
    for idx, ch in enumerate(letters):
        i = idx + 1
        p = probs[ch]
        if collatz_even and i % 2 != 0:
            C[K + i] = 0j
        else:
            C[K + i] = -alpha * p * math.log(p) / i
    sum_iCi = sum(i * C[K + i].real for i in range(1, K + 1))
    H = sum_iCi / alpha if alpha != 0 else 0.0
    return H, C, K

# ------------------------------------------------------------
# 2. Collatz transform on double
# ------------------------------------------------------------
def collatz_transform_double(value: float, steps: int = 5) -> float:
    bits = struct.unpack('>Q', struct.pack('>d', value))[0]
    for _ in range(steps):
        if bits & 1 == 0:
            bits >>= 1
        else:
            bits = (bits << 1) + bits + 1
    return struct.unpack('>d', struct.pack('>Q', bits))[0]

# ------------------------------------------------------------
# 3. Deterministic PRNG using SHAKE
# ------------------------------------------------------------
def shake_prng(seed: bytes, length: int) -> bytes:
    return hashlib.shake_256(seed).digest(length)

def deterministic_shuffle(n: int, seed: bytes) -> List[int]:
    if n <= 1:
        return list(range(n))
    rand_bytes = shake_prng(seed, n * 4)
    indices = list(range(n))
    pos = 0
    for i in range(n-1, 0, -1):
        r = int.from_bytes(rand_bytes[pos:pos+4], 'big')
        pos += 4
        j = r % (i + 1)
        indices[i], indices[j] = indices[j], indices[i]
    return indices

# ------------------------------------------------------------
# 4. Encryption (stores indices and key_stream in metadata)
# ------------------------------------------------------------
def encrypt(data: List[int], alpha: float = 0.3628, collatz_even: bool = True,
            collatz_steps: int = 0) -> Tuple[List[int], Dict]:
    K = len(data)
    if K == 0:
        return [], {'K': 0, 'alpha': alpha, 'collatz_even': collatz_even,
                    'collatz_steps': collatz_steps, 'indices': [], 'key_stream': []}

    H, C, K_letters = spectral_entropy_and_coeffs(data, alpha, collatz_even)
    
    # Apply Collatz mixing
    if collatz_steps > 0:
        H = collatz_transform_double(H, steps=collatz_steps)
    
    # Build seed from H, alpha, collatz_even, and K
    seed = struct.pack('>d', H) + str(alpha).encode() + str(collatz_even).encode() + str(K).encode()
    
    # Generate permutation
    indices = deterministic_shuffle(K, seed + b'perm')
    scrambled = [data[indices[i]] for i in range(K)]
    
    # Generate key stream from coefficients (real parts)
    coeff_bytes = b''.join(struct.pack('>d', C[K_letters + i].real) for i in range(1, K_letters + 1))
    if not coeff_bytes:
        coeff_bytes = struct.pack('>d', H)
    key_stream_bytes = shake_prng(coeff_bytes + seed + b'keystream', K)
    key_stream = list(key_stream_bytes)
    
    encrypted = [scrambled[i] ^ key_stream[i] for i in range(K)]
    
    meta = {
        'K': K,
        'alpha': alpha,
        'collatz_even': collatz_even,
        'collatz_steps': collatz_steps,
        'indices': indices,
        'key_stream': key_stream
    }
    return encrypted, meta

# ------------------------------------------------------------
# 5. Decryption (uses stored indices and key_stream)
# ------------------------------------------------------------
def decrypt(encrypted: List[int], meta: Dict) -> List[int]:
    K = meta['K']
    if K == 0:
        return []
    indices = meta['indices']
    key_stream = meta['key_stream']
    
    unscrambled = [encrypted[i] ^ key_stream[i] for i in range(K)]
    inverse_perm = [0] * K
    for i, pos in enumerate(indices):
        inverse_perm[pos] = i
    decrypted = [unscrambled[inverse_perm[i]] for i in range(K)]
    return decrypted

# ------------------------------------------------------------
# Demonstration
# ------------------------------------------------------------
if __name__ == "__main__":
    plaintext = [ord(c) for c in "Hello, spectral encryption!"]
    print("Plaintext:", plaintext)
    
    alpha = 0.3628
    encrypted, meta = encrypt(plaintext, alpha, collatz_steps=3)   # with Collatz mixing
    print("Encrypted:", encrypted)
    
    decrypted = decrypt(encrypted, meta)
    print("Decrypted:", decrypted)
    print("Decrypted string:", ''.join(chr(x) for x in decrypted))

input('Press ENTER to exit')
