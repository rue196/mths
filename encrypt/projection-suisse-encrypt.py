#!/usr/bin/env python3
"""
projection-suisse-encrypt-3_5.py

Extended projection‑Suisse encryption with:

  • Figure 3.5 elliptic projection   Π(x,y) = ½(1 + cos(2πx/π)·cos(2πy/e))
    used as the *modular degree scaler* — the fixed i_exp is replaced by
    a degree that is modulated by the torus projection of the point.

  • Hash‑based quadratic coordinates
    q(x,y) = A·h² + B·h + C  mod K   where  h = SHAKE256(x,y)

  • SAT envelope inserted between address generation and key derivation
    env[q] = Σ_{a : q(a)=q} |trace_a|  filtered by μ(q) ≠ 0

  • Projection operator Π and Levi‑Civita contraction kept as the main
    encryption objects — the spinor projection and its contraction
    invariants still gate the final key stream.
"""

import math
import hashlib
import struct
import numpy as np
from typing import List, Tuple, Union, Optional
from itertools import permutations


# ============================================================
#  Constants
# ============================================================
PI    = math.pi
E     = math.e
ALPHA = 1.0 / (PI - E)                # ≈ 2.362


# ============================================================
#  1. Projection primitives  (unchanged — the encryption core)
# ============================================================
def generate_vertices(seed: int = 40) -> np.ndarray:
    np.random.seed(seed)
    return np.random.randn(12, 6)


def epsilon_tensor() -> dict:
    """Levi‑Civita rank‑6 tensor as a dict of permutations."""
    eps = {}
    for perm in permutations(range(6)):
        if len(set(perm)) != 6:
            eps[perm] = 0
            continue
        inv = sum(1 for i in range(6) for j in range(i + 1, 6)
                  if perm[i] > perm[j])
        eps[perm] = (-1) ** inv
    return eps


def spinor_projection(vertices: np.ndarray, eps: dict) -> float:
    """Rank‑6 contraction  Π(z) = Σ_σ ε(σ) Π_k z_{k,σ(k)}."""
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


def rotate_vertices(vertices: np.ndarray, theta: float) -> np.ndarray:
    rotated = vertices.copy()
    for v in rotated:
        for pair in [(0, 1), (2, 3), (4, 5)]:
            x, y = v[pair[0]], v[pair[1]]
            v[pair[0]] = x * math.cos(theta) - y * math.sin(theta)
            v[pair[1]] = x * math.sin(theta) + y * math.cos(theta)
    return rotated


def projection_features(vertices: np.ndarray, num_theta: int = 256
                        ) -> Tuple[float, float, float]:
    """Sweep the projection over one rotation and return (var, mean, error)."""
    eps = epsilon_tensor()
    theta_vals = np.linspace(0, 2 * np.pi, num_theta)
    proj = np.array([
        spinor_projection(rotate_vertices(vertices, theta), eps)
        for theta in theta_vals
    ])
    var  = float(np.var(proj))
    mean = float(np.mean(proj))

    K = len(proj)
    sigma = 2.0
    kernel = np.array([math.exp(-(d * d) / (2 * sigma * sigma))
                       for d in range(-(K - 1), K)])
    N = 1 << (2 * K - 1).bit_length()
    sig_pad = np.pad(proj, (0, N - K))
    ker_pad = np.pad(kernel, (0, N - (2 * K - 1)))
    conv = np.fft.ifft(np.fft.fft(sig_pad) * np.fft.fft(ker_pad))[:K]

    M = int(0.3 * K)
    idx = np.argsort(np.abs(conv))[::-1][:M]
    recon = np.zeros(K, dtype=complex)
    for i in idx:
        recon[i] = conv[i]
    error = float(np.linalg.norm(recon - conv))
    return var, mean, error


# ============================================================
#  2. Figure 3.5 — modular scaling elliptic curve
# ============================================================
def elliptic_projection_figure_3_5(x: float, y: float) -> float:
    """
    Bounded elliptic projection on the fundamental parallelogram
    [0, π) × [0, e):

        Π(x, y) = ½ (1 + cos(2π x/π) · cos(2π y/e))  ∈ [0, 1]

    This is Figure 3.5 from the supply‑chain paper.  It replaces the
    fixed degree i_exp with a point‑dependent modular degree.
    """
    u = (x / PI) % 1.0
    v = (y / E) % 1.0
    return 0.5 * (1.0 + math.cos(2.0 * PI * u) * math.cos(2.0 * PI * v))


# ============================================================
#  3. Hash‑based quadratic coordinates
# ============================================================
def hash_quadratic_coordinate(x: float, y: float, K: int,
                              A: int = 1, B: int = 1, C: int = 0) -> int:
    """
    Hash (x, y) → 64‑bit integer h, then map to the quadratic address

        q(x, y) = A·h² + B·h + C  mod K
    """
    h = int.from_bytes(
        hashlib.shake_256(struct.pack('>dd', x, y)).digest(8), 'big')
    return (A * h * h + B * h + C) % K


# ============================================================
#  4. Möbius sieve  (for the SAT gate)
# ============================================================
def mobius_sieve(K: int) -> List[int]:
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


# ============================================================
#  5. Degree modulator — replaces fixed i_exp
# ============================================================
def matrix_trace_3_5(x: float, y: float, i_exp: float) -> float:
    """
    Matrix trace  tr_i = x^d + y^d  with the degree d replaced by

        d = i_exp · Π(x, y)

    where Π is the Figure 3.5 elliptic projection.  Points near the
    torus maximum use the full degree; points near the minimum use
    a degree close to zero.
    """
    Pi = elliptic_projection_figure_3_5(x, y)
    d  = i_exp * Pi
    if abs(d) < 1e-9:
        return 2.0                          # degenerate case → tr = 2
    x_ = max(abs(x), 1e-6)
    y_ = max(abs(y), 1e-6)
    return x_ ** d + y_ ** d


# ============================================================
#  6. SAT envelope over the hash‑quadratic addresses
# ============================================================
def sat_envelope(addresses: List[Tuple[float, float]],
                 traces: List[float],
                 K: int,
                 mu: List[int],
                 A: int = 1, B: int = 1, C: int = 0
                 ) -> Tuple[np.ndarray, List[int]]:
    """
    For each address a = (x, y):
        q = hash_quadratic_coordinate(x, y, K, A, B, C)
        keep a iff  μ(q) ≠ 0
        env[q] += |tr_a|

    Returns
    -------
    env     : np.ndarray of length K
    kept_q  : list of surviving quadratic addresses (in order of a)
    """
    env = np.zeros(K, dtype=np.float64)
    kept_q: List[int] = []
    for (x, y), tr in zip(addresses, traces):
        q = hash_quadratic_coordinate(x, y, K, A, B, C)
        if q < len(mu) and mu[q] != 0:
            env[q] += abs(tr)
            kept_q.append(q)
    return env, kept_q


def envelope_supertrace(env: np.ndarray, mu: List[int]) -> float:
    """S = Σ_q (−1)^q · env[q] over square‑free q."""
    S = 0.0
    for q in range(len(env)):
        if env[q] == 0.0:
            continue
        if mu[q] == 0:
            continue
        sign = 1.0 if (q % 2 == 0) else -1.0
        S += sign * env[q]
    return S


# ============================================================
#  7. Spectral coefficients from the SAT envelope
# ============================================================
def spectral_coeffs_from_envelope(env: np.ndarray,
                                  mu: List[int],
                                  alpha: float = ALPHA,
                                  keep_top: Optional[int] = None
                                  ) -> Tuple[float, List[float]]:
    """
    Turn the SAT envelope into the spectral coefficient vector.

    C_q = −α · p_q · log(p_q) / (q + 1)     (q square‑free)
    where p_q = env[q] / Σ env.
    """
    K = len(env)
    total = float(env.sum())
    if total <= 0.0:
        return 0.0, [0.0] * K

    # rank the occupied square‑free addresses by env
    occupied = [(q, env[q]) for q in range(K)
                if env[q] > 0.0 and mu[q] != 0]
    occupied.sort(key=lambda p: -p[1])

    if keep_top is not None:
        occupied = occupied[:keep_top]

    C = [0.0] * K
    sum_iCi = 0.0
    for q, v in occupied:
        p = v / total
        c = -alpha * p * math.log(p) / (q + 1)
        C[q] = c
        sum_iCi += (q + 1) * c
    H = sum_iCi / alpha if alpha != 0.0 else 0.0
    return H, C


# ============================================================
#  8. Address generation  (unchanged)
# ============================================================
def prng_bytes(seed: bytes, length: int) -> bytes:
    return hashlib.shake_256(seed).digest(length)


def generate_addresses(K: int, seed: bytes
                       ) -> List[Tuple[float, float]]:
    raw = prng_bytes(seed, K * 16)
    points = []
    for i in range(K):
        x = struct.unpack('>d', raw[16 * i:16 * i + 8])[0]
        y = struct.unpack('>d', raw[16 * i + 8:16 * i + 16])[0]
        x = 0.5 + (x - math.floor(x)) * 4.5
        y = 0.5 + (y - math.floor(y)) * 4.5
        points.append((x, y))
    return points


# ============================================================
#  9. Key derivation — projection + contraction + SAT envelope
# ============================================================
def derive_key(password: Union[str, bytes],
               salt: bytes = b'',
               key_len: int = 32,
               K: int = 256,
               i_exp: float = 20.0,
               collatz_even: bool = True,
               collatz_steps: int = 7,
               quad_A: int = 1,
               quad_B: int = 1,
               quad_C: int = 0,
               keep_top: Optional[int] = None,
               public_vertices: Optional[np.ndarray] = None) -> bytes:
    """
    Full pipeline:

        password + salt + projection_features(vertices)
            → base seed
            → K addresses (x, y)
            → Figure 3.5 degree  d = i_exp · Π(x,y)
            → trace_3_5 = x^d + y^d
            → hash‑quadratic coordinate  q(x,y)
            → SAT envelope  env[q]  (Möbius gate)
            → spectral coefficients C
            → key stream (SHAKE256)

    The projection operator Π and the Levi‑Civita contraction remain
    the main encryption objects: they control the extra entropy in the
    seed and they define the degree modulation of every trace.
    """
    if isinstance(password, str):
        password = password.encode('utf-8')

    # ---- 1. projection entropy (main encryption object) ----
    extra = b''
    if public_vertices is not None:
        var, mean, err = projection_features(public_vertices, num_theta=256)
        extra  = struct.pack('>ddd', var, mean, err)
        extra += public_vertices.tobytes()

    # ---- 2. base seed ----
    seed_material = hashlib.shake_256(password + salt + extra).digest(32)

    # ---- 3. addresses ----
    addresses = generate_addresses(K, seed_material)

    # ---- 4. Figure 3.5 modulated traces ----
    traces = [matrix_trace_3_5(x, y, i_exp) for x, y in addresses]
    traces = [abs(t) + 1e-12 for t in traces]

    # ---- 5. SAT envelope over the hash‑quadratic addresses ----
    mu = mobius_sieve(K)
    env, kept_q = sat_envelope(addresses, traces, K, mu,
                               A=quad_A, B=quad_B, C=quad_C)
    S_env = envelope_supertrace(env, mu)

    # ---- 6. spectral coefficients from the envelope ----
    H, C = spectral_coeffs_from_envelope(env, mu,
                                         alpha=ALPHA,
                                         keep_top=keep_top)

    # ---- 7. Collatz on H  (unchanged) ----
    if collatz_steps > 0:
        H = collatz_transform_double(H, collatz_steps)

    # ---- 8. derive the key stream ----
    param_seed = (struct.pack('>d', H) +
                  struct.pack('>d', S_env) +
                  str(ALPHA).encode() +
                  str(collatz_even).encode() +
                  str(i_exp).encode() +
                  str(K).encode() +
                  str(quad_A).encode() +
                  str(quad_B).encode() +
                  str(quad_C).encode() +
                  password + salt + extra)

    coeff_bytes = b''.join(struct.pack('>d', c) for c in C)
    if not coeff_bytes:
        coeff_bytes = struct.pack('>d', H)
    key_stream = hashlib.shake_256(coeff_bytes + param_seed).digest(key_len)
    return key_stream


# ============================================================
#  10. Collatz on IEEE‑754 bit pattern  (unchanged)
# ============================================================
def collatz_transform_double(value: float, steps: int) -> float:
    bits = struct.unpack('>Q', struct.pack('>d', value))[0]
    for _ in range(steps):
        if bits & 1 == 0:
            bits >>= 1
        else:
            bits = (bits << 1) + bits + 1
    return struct.unpack('>d', struct.pack('>Q', bits))[0]


# ============================================================
#  11. Encryption / decryption  (unchanged)
# ============================================================
def encrypt(data: bytes, key: bytes) -> bytes:
    return bytes([data[i] ^ key[i % len(key)] for i in range(len(data))])


def decrypt(data: bytes, key: bytes) -> bytes:
    return encrypt(data, key)


# ============================================================
#  12. Demonstration
# ============================================================
def main():
    print("=" * 72)
    print("Projection‑Suisse encryption  ·  Figure 3.5 + SAT envelope")
    print("=" * 72)

    # --- public vertices  (projection operator source) ---
    public = generate_vertices(seed=42)
    var, mean, err = projection_features(public, num_theta=256)
    print("\n[1] projection features (main encryption object)")
    print(f"    variance of Π over sweep  = {var:.6e}")
    print(f"    mean of Π                 = {mean:.6e}")
    print(f"    reconstruction error      = {err:.6e}")

    # --- Figure 3.5 sample ------------------------------------------------
    print("\n[2] Figure 3.5 modular projection samples")
    for (x, y) in [(0.5, 0.5), (PI / 2, E / 2), (PI, E), (1.0, 1.0)]:
        Pi = elliptic_projection_figure_3_5(x, y)
        print(f"    Π({x:5.3f}, {y:5.3f}) = {Pi:.6f}")

    # --- full derivation --------------------------------------------------
    password = "mysecretpassword"
    salt = b"saltysalt"
    key_len = 32
    K = 256
    i_exp = 20.0

    key = derive_key(password, salt, key_len, K, i_exp,
                     quad_A=1, quad_B=1, quad_C=0,
                     keep_top=None,
                     public_vertices=public)

    print("\n[3] derived key")
    print(f"    key (hex) = {key.hex()}")

    # --- encryption -------------------------------------------------------
    plaintext = b"multivariate encryption "
    ciphertext = encrypt(plaintext, key)
    decrypted = decrypt(ciphertext, key)

    print("\n[4] encryption round trip")
    print(f"    plaintext  : {plaintext}")
    print(f"    ciphertext : {ciphertext.hex()}")
    print(f"    decrypted  : {decrypted}")
    print(f"    success    : {plaintext == decrypted}")

    # --- reproducibility --------------------------------------------------
    key2 = derive_key(password, salt, key_len, K, i_exp,
                      quad_A=1, quad_B=1, quad_C=0,
                      keep_top=None,
                      public_vertices=public)
    print(f"\n[5] key reproducibility: {key == key2}")

    # --- comparison: fixed degree vs Figure 3.5 degree -------------------
    print("\n[6] fixed i_exp vs Figure 3.5 modulated degree")
    addrs = generate_addresses(32, hashlib.shake_256(b"demo").digest(32))
    print(f"    {'x':>8s}  {'y':>8s}  {'Π(x,y)':>8s}  "
          f"{'tr_fixed':>12s}  {'tr_3_5':>12s}")
    for (x, y) in addrs[:8]:
        Pi = elliptic_projection_figure_3_5(x, y)
        tr_fixed = x ** (i_exp - 1) + y ** (i_exp - 1)
        tr_35    = matrix_trace_3_5(x, y, i_exp - 1)
        print(f"    {x:8.4f}  {y:8.4f}  {Pi:8.4f}  "
              f"{tr_fixed:12.4e}  {tr_35:12.4e}")

    # --- SAT envelope density --------------------------------------------
    mu = mobius_sieve(K)
    addrs = generate_addresses(K,
                               hashlib.shake_256(b"density-check").digest(32))
    traces = [matrix_trace_3_5(x, y, i_exp) for x, y in addrs]
    env, kept = sat_envelope(addrs, traces, K, mu)
    density = len(kept) / K
    print(f"\n[7] SAT envelope density over K = {K}")
    print(f"    kept addresses   : {len(kept)} / {K}  "
          f"(density = {density:.4f})")
    print(f"    expected 6/π²    : {6.0 / (PI * PI):.4f}")
    print(f"    envelope S       : {envelope_supertrace(env, mu):+.6f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
    input("Press ENTER to exit")