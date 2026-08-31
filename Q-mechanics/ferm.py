import math
import hashlib
import struct
import numpy as np
from typing import List, Tuple, Union, Optional
from itertools import permutations
import os

# ---------- spinor projection (unchanged) ----------
def generate_vertices(seed=42):
    np.random.seed(seed)
    return np.random.randn(12, 6)

def epsilon_tensor():
    eps = {}
    for perm in permutations(range(6)):
        if len(set(perm)) != 6:
            eps[perm] = 0
            continue
        inv = sum(1 for i in range(6) for j in range(i+1,6) if perm[i] > perm[j])
        eps[perm] = (-1)**inv
    return eps

def spinor_projection(vertices, eps):
    v = vertices[:6]
    scalar = 0.0
    for perm, sign in eps.items():
        if sign == 0:
            continue
        prod = 1.0
        for idx, coord_idx in enumerate(perm):
            prod *= v[idx][coord_idx]
        scalar += sign * prod
    return scalar

def rotate_vertices(vertices, theta):
    rotated = vertices.copy()
    for v in rotated:
        for pair in [(0,1), (2,3), (4,5)]:
            x, y = v[pair[0]], v[pair[1]]
            v[pair[0]] = x * math.cos(theta) - y * math.sin(theta)
            v[pair[1]] = x * math.sin(theta) + y * math.cos(theta)
    return rotated

def projection_features(vertices, num_theta=256):
    eps = epsilon_tensor()
    theta_vals = np.linspace(0, 2*np.pi, num_theta)
    proj = []
    for theta in theta_vals:
        rot = rotate_vertices(vertices, theta)
        proj.append(spinor_projection(rot, eps))
    proj = np.array(proj)
    var = np.var(proj)
    mean = np.mean(proj)
    # spectral compression (unchanged)
    K = len(proj)
    sigma = 2.0
    kernel = np.array([math.exp(-(d*d)/(2*sigma*sigma)) for d in range(-(K-1), K)])
    N = 1 << (2*K - 1).bit_length()
    sig_pad = np.pad(proj, (0, N - K))
    ker_pad = np.pad(kernel, (0, N - (2*K - 1)))
    conv = np.fft.ifft(np.fft.fft(sig_pad) * np.fft.fft(ker_pad))[:K]
    M = int(0.3 * K)
    idx = np.argsort(np.abs(conv))[::-1][:M]
    recon = np.zeros(K, dtype=complex)
    for i in idx:
        recon[i] = conv[i]
    error = np.linalg.norm(recon - conv)
    return var, mean, error

# ---------- new matrix‑based flow entropy ----------
ALPHA = 1.0 / (math.pi - math.e)

def prng_bytes(seed: bytes, length: int) -> bytes:
    return hashlib.shake_256(seed).digest(length)

def generate_addresses(K: int, seed: bytes) -> List[Tuple[float, float]]:
    raw = prng_bytes(seed, K * 16)
    points = []
    for i in range(K):
        x = struct.unpack('>d', raw[16*i:16*i+8])[0]
        y = struct.unpack('>d', raw[16*i+8:16*i+16])[0]
        x = 0.5 + (x - math.floor(x)) * 4.5
        y = 0.5 + (y - math.floor(y)) * 4.5
        points.append((x, y))
    return points

def build_supermatrix(points: List[Tuple[float, float]], N: int) -> np.ndarray:
    """
    Construct an N×N super‑matrix M whose entry (t, j) is:
        M[t, j] = (x_j)^t + (y_j)^t   for t = 1..N, j = 1..N
    This encodes the time evolution of each particle's coordinates.
    The inverse coordinates x_j^{-1}, y_j^{-1} are not directly used,
    but they define the initial spatial configuration; the powers
    represent the flow through time.
    """
    M = np.zeros((N, N))
    for t in range(1, N+1):
        for j, (x, y) in enumerate(points[:N]):
            M[t-1, j] = (x ** t) + (y ** t)
    return M

def supertrace(M: np.ndarray, even_weight: float = 1.0, odd_weight: float = -1.0) -> float:
    """
    Compute the supertrace: sum over even t (bosonic) with weight +1,
    and over odd t (fermionic) with weight -1.
    This enforces a supersymmetry between even and odd time slices.
    """
    N = M.shape[0]
    trace = 0.0
    for t in range(N):
        val = M[t, t]  # diagonal elements (could also use full trace, but diagonal suffices)
        if (t+1) % 2 == 0:   # even time index (1‑based)
            trace += even_weight * val
        else:
            trace += odd_weight * val
    return trace

def matrix_entropy(M: np.ndarray, alpha: float = ALPHA) -> float:
    """
    Convert the supertrace into an entropy‑like scalar H.
    We first compute the absolute value of the supertrace to avoid negatives,
    then apply a logarithmic measure similar to the original spectral coefficient.
    """
    S = abs(supertrace(M))
    if S == 0:
        return 0.0
    # Use a Shannon‑like entropy: -alpha * S * log(S) / N
    H = -alpha * S * math.log(S) / M.shape[0]
    return H

# ---------- updated key derivation ----------
def derive_key(password: Union[str, bytes], salt: bytes = b'',
               key_len: int = 32, K: int = 256,
               i_exp: int = 2, collatz_even: bool = True,
               collatz_steps: int = 7,
               public_vertices: Optional[np.ndarray] = None) -> bytes:
    """
    Derive a cryptographic key. If public_vertices is given, the spinor
    projection features (var, mean, error) are added to the entropy pool.
    Additionally, a super‑matrix is built from the addresses and its
    supertrace entropy is used as the main mixing scalar.
    """
    if isinstance(password, str):
        password = password.encode('utf-8')
    
    # ---- extra entropy from public vertices (unchanged) ----
    extra = b''
    if public_vertices is not None:
        var, mean, err = projection_features(public_vertices, num_theta=256)
        extra = struct.pack('>ddd', var, mean, err)
        extra += public_vertices.tobytes()
    
    # ---- base seed ----
    seed_material = hashlib.shake_256(password + salt + extra).digest(32)

    # ---- generate K addresses ----
    addresses = generate_addresses(K, seed_material)

    # ---- build the super‑matrix of size N = K ----
    M = build_supermatrix(addresses, K)

    # ---- compute entropy H from the supertrace ----
    H = matrix_entropy(M, alpha=ALPHA)

    # ---- apply Collatz transformation to H (as before) ----
    if collatz_steps > 0:
        H = collatz_transform_double(H, collatz_steps)

    # ---- prepare parameter seed (includes H, alpha, collatz_even, i_exp, etc.) ----
    param_seed = (struct.pack('>d', H) +
                  str(ALPHA).encode() + str(collatz_even).encode() +
                  str(i_exp).encode() + str(K).encode() +
                  password + salt + extra)

    # ---- generate key stream (unchanged) ----
    # We also include the flattened super‑matrix as extra mixing
    matrix_bytes = M.tobytes()
    key_stream = hashlib.shake_256(matrix_bytes + param_seed).digest(key_len)
    return key_stream

# ---------- Collatz transform (unchanged) ----------
def collatz_transform_double(value: float, steps: int) -> float:
    bits = struct.unpack('>Q', struct.pack('>d', value))[0]
    for _ in range(steps):
        if bits & 1 == 0:
            bits >>= 1
        else:
            bits = (bits << 1) + bits + 1
    return struct.unpack('>d', struct.pack('>Q', bits))[0]

# ---------- Encryption / Decryption (unchanged) ----------
def encrypt(data: bytes, key: bytes) -> bytes:
    return bytes([data[i] ^ key[i % len(key)] for i in range(len(data))])

def decrypt(data: bytes, key: bytes) -> bytes:
    return encrypt(data, key)

# ---------- Demonstration ----------
def main():
    public = generate_vertices(seed=42)
    password = "mysecretpassword"
    salt = b"saltysalt"
    key_len = 12
    K = 1  # N = 8 particles (must be >0)

    key = derive_key(password, salt, key_len, K, i_exp=2,
                     public_vertices=public)
    print(f"Derived key (hex): {key.hex()}")

    plaintext = b"Hello, integrated supermatrix encryption with spinors!"
    ciphertext = encrypt(plaintext, key)
    decrypted = decrypt(ciphertext, key)
    print(f"Plaintext:  {plaintext}")
    print(f"Ciphertext (hex): {ciphertext.hex()}")
    print(f"Decrypted:  {decrypted}")
    print(f"Success: {plaintext == decrypted}")

    key2 = derive_key(password, salt, key_len, K, i_exp=2,
                      public_vertices=public)
    print(f"Key matches: {key == key2}")

if __name__ == "__main__":
    main()
    input('Press ENTER to exit')