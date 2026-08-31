import math
import hashlib
import struct
import numpy as np
from typing import List, Tuple, Union, Optional
from itertools import permutations
import os
import hmac

# ---------- Copy projection‑suisse‑encrypt functions (minimal) ----------
ALPHA = 1.0 / (math.pi - math.e)

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

def matrix_trace(x: float, y: float, i: int) -> float:
    return x ** (i-1) + y ** (i-1)

def spectral_coeffs_from_traces(traces: List[float], alpha: float = ALPHA,
                                collatz_even: bool = True) -> Tuple[float, List[float]]:
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

def collatz_transform_double(value: float, steps: int) -> float:
    bits = struct.unpack('>Q', struct.pack('>d', value))[0]
    for _ in range(steps):
        if bits & 1 == 0:
            bits >>= 1
        else:
            bits = (bits << 1) + bits + 1
    return struct.unpack('>d', struct.pack('>Q', bits))[0]

def derive_key(password: Union[str, bytes], salt: bytes = b'',
               key_len: int = 32, K: int = 256,
               i_exp: int = 2, collatz_even: bool = True,
               collatz_steps: int = 7,
               public_vertices: Optional[np.ndarray] = None) -> bytes:
    if isinstance(password, str):
        password = password.encode('utf-8')
    extra = b''
    if public_vertices is not None:
        var, mean, err = projection_features(public_vertices, num_theta=256)
        extra = struct.pack('>ddd', var, mean, err)
        extra += public_vertices.tobytes()
    seed_material = hashlib.shake_256(password + salt + extra).digest(32)
    addresses = generate_addresses(K, seed_material)
    traces = [matrix_trace(x, y, i_exp) for x, y in addresses]
    traces = [abs(t) + 1e-12 for t in traces]
    H, C = spectral_coeffs_from_traces(traces, alpha=ALPHA,
                                       collatz_even=collatz_even)
    if collatz_steps > 0:
        H = collatz_transform_double(H, collatz_steps)
    param_seed = (struct.pack('>d', H) +
                  str(ALPHA).encode() + str(collatz_even).encode() +
                  str(i_exp).encode() + str(K).encode() +
                  password + salt + extra)
    coeff_bytes = b''.join(struct.pack('>d', c) for c in C)
    if not coeff_bytes:
        coeff_bytes = struct.pack('>d', H)
    key_stream = hashlib.shake_256(coeff_bytes + param_seed).digest(key_len)
    return key_stream

# ---------- Fingerprint function ----------
def fingerprint(key: bytes, length: int = 16, algo: str = 'hmac') -> bytes:
    """
    Generate a short fingerprint from a cryptographic key.
    - algo='hmac': use HMAC-SHA256 with a fixed domain separator.
    - algo='hash': use SHA256 (truncated).
    Returns bytes of length `length`.
    """
    if algo == 'hmac':
        # Use a fixed domain separator to prevent collisions with other uses
        domain = b"projection-suisse-fingerprint"
        h = hmac.new(key, domain, hashlib.sha256)
        digest = h.digest()
    else:
        h = hashlib.sha256(key)
        digest = h.digest()
    return digest[:length]

# ---------- Example ----------
def main():
    # Generate a public key (12 vertices in R^6) – shared secret between parties
    public = generate_vertices(seed=42)

    password = "mysecretpassword"
    salt = b"saltysalt"

    # Derive a small key (16 bytes) suitable for online use
    key_len = 16
    K = 64  # smaller memory footprint for speed
    i_exp = 2

    key = derive_key(password, salt, key_len, K, i_exp,
                     public_vertices=public, collatz_steps=3)
    print(f"Derived key (hex, {len(key)} bytes): {key.hex()}")

    # Generate a fingerprint (also 16 bytes)
    fp = fingerprint(key, length=16)
    print(f"Fingerprint (hex): {fp.hex()}")

    # Show that the fingerprint is stable
    key2 = derive_key(password, salt, key_len, K, i_exp,
                      public_vertices=public, collatz_steps=3)
    fp2 = fingerprint(key2, length=16)
    print(f"Fingerprint matches: {fp == fp2}")

if __name__ == "__main__":
    main()