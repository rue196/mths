import math
import hashlib
import struct
from typing import List, Tuple, Union

# ------------------------------------------------------------
# 0. Spectral constant: alpha = 1/(π - e)
# ------------------------------------------------------------
ALPHA = 1.0 / (math.pi - math.e)   # ≈ 2.362

# ------------------------------------------------------------
# 1. Deterministic PRNG for address generation
# ------------------------------------------------------------
def prng_bytes(seed: bytes, length: int) -> bytes:
    """Generate 'length' pseudo‑random bytes using SHAKE‑256."""
    return hashlib.shake_256(seed).digest(length)

def generate_addresses(K: int, seed: bytes) -> List[Tuple[float, float]]:
    """
    Generate K points (x, y) in [0.5, 5.0] from the seed.
    Uses 8 bytes per coordinate (two doubles) to avoid bias.
    """
    # Need 16 bytes per point (8 for x, 8 for y)
    raw = prng_bytes(seed, K * 16)
    points = []
    for i in range(K):
        # Unpack two doubles from the raw bytes
        x = struct.unpack('>d', raw[16*i:16*i+8])[0]
        y = struct.unpack('>d', raw[16*i+8:16*i+16])[0]
        # Map to [0.5, 5.0] by taking fractional part and scaling
        x = 0.5 + (x - math.floor(x)) * 4.5
        y = 0.5 + (y - math.floor(y)) * 4.5
        points.append((x, y))
    return points

# ------------------------------------------------------------
# 2. Matrix M(x,y) for a given integer exponent i
#    Returns the trace (real number) – we only need the trace.
# ------------------------------------------------------------
def matrix_trace(x: float, y: float, i: int) -> float:
    """
    M = [[x^(i-1), x^(-1)*y^i],
         [y^(-1)*x^i, y^(i-1)]]
    trace = x^(i-1) + y^(i-1)
    """
    return x ** (i-1) + y ** (i-1)

# ------------------------------------------------------------
# 3. Spectral coefficients from a list of traces
#    Uses alpha = 1/(π−e) and Collatz even‑index masking.
# ------------------------------------------------------------
def spectral_coeffs_from_traces(traces: List[float], alpha: float = ALPHA,
                                collatz_even: bool = True) -> Tuple[float, List[float]]:
    """
    Given a list of (positive) traces, compute:
      - probabilities p_i = trace_i / sum(traces)
      - coefficients C_i = -alpha * p_i * log(p_i) / i  (if i even or collatz_even=False)
      - entropy H = (1/alpha) * sum_i i*C_i  (which cancels alpha)
    Returns (H, C_list) where C_list[i-1] corresponds to index i.
    """
    K = len(traces)
    if K == 0:
        return 0.0, []
    total = sum(traces)
    if total == 0:
        return 0.0, [0.0] * K

    C = [0.0] * K
    sum_iCi = 0.0
    for idx, tr in enumerate(traces, start=1):
        p = tr / total
        # Avoid log(0)
        if p <= 0:
            continue
        if collatz_even and idx % 2 != 0:
            c_i = 0.0
        else:
            c_i = -alpha * p * math.log(p) / idx
        C[idx-1] = c_i
        sum_iCi += idx * c_i

    H = sum_iCi / alpha if alpha != 0 else 0.0
    return H, C

# ------------------------------------------------------------
# 4. Key derivation using spectral coefficients + Collatz mixing
# ------------------------------------------------------------
def derive_key(password: Union[str, bytes], salt: bytes = b'',
               key_len: int = 32, K: int = 256,
               i_exp: int = 2, collatz_even: bool = True,
               collatz_steps: int = 5) -> bytes:
    """
    Derive a cryptographic key from a password.
    Steps:
      1. Seed = SHAKE(password + salt) to generate K addresses (x,y).
      2. Compute traces of M(x,y) for exponent i_exp.
      3. Compute spectral coefficients C_i and entropy H.
      4. Apply optional Collatz transform to H (mixing).
      5. Build a seed from H, alpha, and parameters.
      6. Generate a deterministic permutation (Fisher‑Yates) from the seed.
      7. Generate key stream from the real parts of C_i and the seed via SHAKE.
    All arrays are O(K) memory.
    """
    if isinstance(password, str):
        password = password.encode('utf-8')
    seed_material = hashlib.shake_256(password + salt).digest(32)

    # ---- Step 1: generate K addresses ----
    addresses = generate_addresses(K, seed_material)

    # ---- Step 2: compute traces ----
    traces = [matrix_trace(x, y, i_exp) for x, y in addresses]
    # Ensure positivity (absolute value)
    traces = [abs(t) + 1e-12 for t in traces]

    # ---- Step 3: spectral coefficients and entropy ----
    H, C = spectral_coeffs_from_traces(traces, alpha=ALPHA,
                                       collatz_even=collatz_even)

    # ---- Step 4: optional Collatz mixing on H (double space) ----
    def collatz_transform_double(value: float, steps: int) -> float:
        bits = struct.unpack('>Q', struct.pack('>d', value))[0]
        for _ in range(steps):
            if bits & 1 == 0:
                bits >>= 1
            else:
                bits = (bits << 1) + bits + 1
        return struct.unpack('>d', struct.pack('>Q', bits))[0]

    if collatz_steps > 0:
        H = collatz_transform_double(H, collatz_steps)

    # ---- Step 5: seed for permutation and key stream ----
    # Include all parameters to ensure uniqueness
    param_seed = (struct.pack('>d', H) +
                  str(ALPHA).encode() + str(collatz_even).encode() +
                  str(i_exp).encode() + str(K).encode() +
                  password + salt)

    # ---- Step 6: generate a deterministic permutation (cycle cover) ----
    # We don't actually need the permutation for key stream,
    # but we include it to satisfy the "cycle double cover" concept.
    # The permutation can be used to permute something else if needed.
    # For the key stream we directly use the coefficients.
    # However, we generate the permutation to illustrate O(K) memory.
    # We'll generate it and discard (or we can use it to reorder C).
    # For simplicity, we skip storing the permutation and just use the coefficients.
    # But we still need a seed for the key stream; we already have param_seed.
    # We'll build key stream from the real parts of C_i.

    # ---- Step 7: generate key stream ----
    # Pack the real parts of C_i into bytes (as doubles)
    coeff_bytes = b''.join(struct.pack('>d', c) for c in C)
    if not coeff_bytes:
        coeff_bytes = struct.pack('>d', H)
    # Expand to desired key length using SHAKE
    key_stream = hashlib.shake_256(coeff_bytes + param_seed).digest(key_len)
    return key_stream

# ------------------------------------------------------------
# 5. Encryption / Decryption (XOR stream)
# ------------------------------------------------------------
def encrypt(data: bytes, key: bytes) -> bytes:
    """XOR encrypt data with the key (repeated if key shorter)."""
    return bytes([data[i] ^ key[i % len(key)] for i in range(len(data))])

def decrypt(data: bytes, key: bytes) -> bytes:
    """XOR decrypt (same as encrypt)."""
    return encrypt(data, key)

# ------------------------------------------------------------
# 6. Demonstration
# ------------------------------------------------------------
def main():
    password = "Jq:RuvoM0s2KM4qigG$YnUC'2K}VOh8?Dgfks=G<R)n~0OO"
    salt = b"saltysalt"
    key_len = 32
    K =  160               # number of points
    i_exp = 150             # matrix exponent

    # Derive key
    key = derive_key(password, salt, key_len, K, i_exp)
    print(f"Derived key (hex): {key.hex()}")

    # Test with a message
    plaintext = b"Hello, spectral encryption !"
    ciphertext = encrypt(plaintext, key)
    decrypted = decrypt(ciphertext, key)

    print(f"Plaintext:  {plaintext}")
    print(f"Ciphertext (hex): {ciphertext.hex()}")
    print(f"Decrypted:  {decrypted}")
    print(f"Success: {plaintext == decrypted}")

    # Check reproducibility
    key2 = derive_key(password, salt, key_len, K, i_exp)
    print(f"Key matches: {key == key2}")

if __name__ == "__main__":
    main()

input('Press ENTER to exit')