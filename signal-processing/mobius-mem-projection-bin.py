#!/usr/bin/env python3
"""
mobius_binary_contraction_compress.py

Möbius compression scheme using the **binary contraction** of the
spinor projection operator Π from `projection-operator.pdf`, and
**reconstruction through the fiber-optic UGC pipeline** from
`fiber-optic-signal-ugc.py`.

Binary contraction
------------------
The projection operator acts on a 12-vertex configuration in ℝ⁶ via
the rank-6 Levi-Civita tensor.  The invariance theorem reduces the
effective degrees of freedom through the covering map  SU(2) → SO(3).
We implement that reduction as a **binary halving** at each step:

    step 1:  12 → 6      pairwise sum of the two 6-vertex blocks
    step 2:   6 → 3      pair each row into a complex spinor
                         and sum adjacent spinors

The 3 surviving complex values (plus the Levi-Civita scalar Π) form
the **compressed signature** of the input.

Compression
-----------
    μ(1..K)   →   binary contraction  →   signature (3 amp + Π)
              →   2-bit Möbius pack with abs_sum key
                                                (mobius-memory.py)

Reconstruction
--------------
    signature  →  6-vertex set signal            (fiber-optic-signal-ugc)
               →  elliptic Möbius gate           (amplitude envelope)
               →  fiber propagation              (dispersion kernel)
               →  decoded amplitudes
               →  Möbius unpack with abs_sum check
"""

from __future__ import annotations

import math
import struct
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from itertools import permutations
from scipy.signal import convolve


# ============================================================
#  Constants
# ============================================================
PI         = math.pi
E          = math.e
ALPHA_SYM  = 1.0 / (PI - E)            # ≈ 2.362
ALPHA_ASYM = 0.3628
W1         = PI
W2         = E


# ============================================================
#  1. Möbius sieve
# ============================================================
def mobius_sieve(K: int) -> np.ndarray:
    if K < 1:
        return np.zeros(K + 1, dtype=np.int8)
    mu = np.ones(K + 1, dtype=np.int8)
    is_comp = np.zeros(K + 1, dtype=bool)
    for i in range(2, K + 1):
        if not is_comp[i]:
            mu[i::i] = -mu[i::i]
            is_comp[i::i] = True
            i2 = i * i
            if i2 <= K:
                mu[i2::i2] = 0
    mu[0] = 0
    return mu


# ============================================================
#  2. Projection operator primitives  (projection-operator.pdf)
# ============================================================
def levi_civita_contraction(V6: np.ndarray) -> float:
    """
    Rank-6 Levi-Civita contraction on a 6×6 vertex matrix:

        Π(z) = Σ_σ ε(σ) · Π_k z_{k, σ(k)}

    This is the orientation-consistent scalar invariant of the
    6-vertex sub-configuration.
    """
    scalar = 0.0
    for perm in permutations(range(6)):
        inv = sum(1 for i in range(6) for j in range(i + 1, 6)
                  if perm[i] > perm[j])
        sign = (-1) ** inv
        prod = 1.0
        for k, c in enumerate(perm):
            prod *= V6[k, c]
        scalar += sign * prod
    return scalar


def binary_contraction(vertices_12x6: np.ndarray) -> dict:
    """
    Binary halving through the covering map  SU(2) → SO(3):

        step 1:  12 → 6      pairwise sum of the two 6-vertex blocks
        step 2:   6 → 3      pair each row into a complex spinor
                             and sum adjacent spinors

    Returns
    -------
    V6       : 6×6 contracted matrix (before the SU(2)→SO(3) step)
    spinors  : length-6 complex vector of spinor pairs
    reduced  : length-3 complex vector after the SO(3) reduction
    amplitudes : |reduced| — the 3 real amplitudes of the signature
    Pi_scalar  : Levi-Civita scalar of V6 (global invariant)
    """
    V12 = np.asarray(vertices_12x6, dtype=float)

    # ---- step 1: 12 → 6 ----
    V6 = V12[:6, :] + V12[6:, :]              # element-wise (6, 6)

    # ---- step 2: 6 → 3 ----
    # each row is collapsed to a complex spinor  z = x + iy
    spinors = np.array([complex(V6[i, 0], V6[i, 1]) for i in range(6)])
    # pair adjacent spinors and sum  →  3 complex values
    reduced = np.array([
        spinors[0] + spinors[1],
        spinors[2] + spinors[3],
        spinors[4] + spinors[5],
    ])
    amplitudes = np.abs(reduced)               # 3 real amplitudes

    Pi_scalar = levi_civita_contraction(V6)

    return dict(
        V6=V6,
        spinors=spinors,
        reduced=reduced,
        amplitudes=amplitudes,
        Pi_scalar=Pi_scalar,
    )


# ============================================================
#  3. Möbius 2-bit pack  (mobius-memory.py)
# ============================================================
def compress_mobius(mu: np.ndarray) -> bytes:
    """
    2-bit packing of the Möbius array with abs_sum integrity key.

    Header  : (K, abs_sum) as two big-endian uint32
    Payload : 2 bits per entry (0→00, 1→01, −1→10), MSB-aligned
    """
    K = len(mu) - 1
    if K == 0:
        return struct.pack('>II', 0, 0)
    abs_sum = int(np.sum(np.abs(mu[1:K + 1])))
    bit_length = 2 * K
    num_bytes = (bit_length + 7) // 8
    packed = bytearray(num_bytes)
    for i in range(1, K + 1):
        val = int(mu[i])
        code = 0 if val == 0 else (1 if val == 1 else 2)
        bit_pos = (i - 1) * 2
        byte_idx = bit_pos // 8
        bit_offset = bit_pos % 8
        packed[byte_idx] |= (code << (6 - bit_offset))
    return struct.pack('>II', K, abs_sum) + bytes(packed)


def decompress_mobius(data: bytes) -> np.ndarray:
    """Reconstruct the Möbius array and verify the abs_sum key."""
    if len(data) < 8:
        raise ValueError("Data too short")
    K, abs_sum = struct.unpack('>II', data[:8])
    if K == 0:
        return np.zeros(1, dtype=np.int8)
    packed = data[8:]
    mu = np.zeros(K + 1, dtype=np.int8)
    for i in range(1, K + 1):
        bit_pos = (i - 1) * 2
        byte_idx = bit_pos // 8
        bit_offset = bit_pos % 8
        if byte_idx >= len(packed):
            raise ValueError("Insufficient data")
        code = (packed[byte_idx] >> (6 - bit_offset)) & 0b11
        if code == 0:
            mu[i] = 0
        elif code == 1:
            mu[i] = 1
        elif code == 2:
            mu[i] = -1
        else:
            raise ValueError(f"Invalid code {code} at position {i}")
    computed = int(np.sum(np.abs(mu[1:K + 1])))
    if computed != abs_sum:
        raise ValueError(
            f"Integrity check failed: expected abs_sum={abs_sum}, "
            f"got {computed}")
    return mu


# ============================================================
#  4. Elliptic Möbius gate  (from eliptic-finite-step-mobius.py)
# ============================================================
def build_elliptic_coeffs(K: int, smooth: bool = True):
    mu = mobius_sieve(K)
    c = np.zeros(2 * K + 1, dtype=float)
    for i in range(-K, K + 1):
        c[i + K] = 0.0 if i == 0 else mu[abs(i)]

    if smooth:
        for i in range(-K, K + 1):
            if i == 0 or c[i + K] != 0.0:
                continue
            left, right = i - 1, i + 1
            while left >= -K and c[left + K] == 0.0:
                left -= 1
            while right <= K and c[right + K] == 0.0:
                right += 1
            if left < -K or right > K:
                continue
            dist = right - left
            if dist == 0:
                continue
            wl = (right - i) / dist
            wr = (i - left) / dist
            c[i + K] = wl * c[left + K] + wr * c[right + K]

    coeffs = {i: c[i + K] for i in range(-K, K + 1)
              if abs(c[i + K]) > 1e-12}
    return coeffs, c


class EllipticMobiusGate:
    def __init__(self, K: int, smooth: bool = True):
        self.K = K
        self.coeffs, self.full_array = build_elliptic_coeffs(K, smooth)
        self.indices = np.array(sorted(self.coeffs.keys()))
        self.values = np.array([self.coeffs[i] for i in self.indices])
        self.period = 2 * PI * ALPHA_SYM

    def zeta(self, t: float) -> complex:
        if len(self.indices) == 0:
            return 0.0 + 0.0j
        phases = t * self.indices / ALPHA_SYM
        return complex(np.sum(self.values * np.cos(phases)),
                       np.sum(self.values * np.sin(phases)))

    def zeta_array(self, t_vals: np.ndarray) -> np.ndarray:
        phases = np.outer(t_vals, self.indices) / ALPHA_SYM
        return np.exp(1j * phases) @ self.values

    def amplitude(self, t_vals: np.ndarray) -> np.ndarray:
        return np.abs(self.zeta_array(t_vals))

    def power(self, t_vals: np.ndarray) -> np.ndarray:
        return self.amplitude(t_vals) ** 2

    def basel_theoretical(self) -> float:
        return 2.0 * (6.0 / (PI * PI)) * self.K

    def basel_error(self, N: int = 2000) -> float:
        t_vals = np.linspace(0, self.period, N)
        power = self.power(t_vals)
        try:
            integral = np.trapezoid(power, t_vals)
        except AttributeError:
            integral = np.trapz(power, t_vals)
        avg = integral / self.period
        theo = self.basel_theoretical()
        return abs(avg - theo) / theo if theo != 0 else 0.0


# ============================================================
#  5. Figure 3.5 — bounded elliptic projection
# ============================================================
def elliptic_projection_1d(x: np.ndarray, y0: float = 1.0) -> np.ndarray:
    U = (x / W1) % 1.0
    V = (y0 / W2) % 1.0
    return 0.5 * (1.0 + np.cos(2 * np.pi * U) * math.cos(2 * np.pi * V))


def elliptic_projection_2d(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    U = (X / W1) % 1.0
    V = (Y / W2) % 1.0
    return 0.5 * (1.0 + np.cos(2 * np.pi * U) * np.cos(2 * np.pi * V))


# ============================================================
#  6. Signature packing  (compression payload)
# ============================================================


class CompressedSignature:
    """
    The full compressed payload:

      • K               : length of the Möbius array
      • abs_sum         : integrity key  Σ|μ(i)|
      • amplitudes      : 3 real amplitudes from the binary contraction
      • Pi_scalar       : Levi-Civita scalar of the 6-vertex block
      • mu_packed       : the 2-bit packed Möbius array
      • signature_bytes : total byte count
    """
    def __init__(self, K: int, abs_sum: int,
                 amplitudes: np.ndarray, Pi_scalar: float,
                 mu_packed: bytes):
        self.K = K
        self.abs_sum = abs_sum
        self.amplitudes = amplitudes
        self.Pi_scalar = Pi_scalar
        self.mu_packed = mu_packed

    def to_bytes(self) -> bytes:
        """Header: K, abs_sum, 3 amplitudes, Π, then packed μ."""
        header = struct.pack(
            '>II3dd', self.K, self.abs_sum,
            float(self.amplitudes[0]),
            float(self.amplitudes[1]),
            float(self.amplitudes[2]),
            float(self.Pi_scalar),
        )
        return header + self.mu_packed

    @classmethod
    def from_bytes(cls, data: bytes) -> "CompressedSignature":
        if len(data) < struct.calcsize('>II3dd'):
            raise ValueError("Payload too short")
        K, abs_sum, a0, a1, a2, Pi_scalar = struct.unpack(
            '>II3dd', data[:struct.calcsize('>II3dd')])
        amplitudes = np.array([a0, a1, a2])
        mu_packed = data[struct.calcsize('>II3dd'):]
        return cls(K, abs_sum, amplitudes, Pi_scalar, mu_packed)

    def size_bytes(self) -> int:
        return len(self.to_bytes())


# ============================================================
#  7. Compression pipeline
# ============================================================
def compress(mu: np.ndarray,
             seed_vertices: int = 42,
             K_gate: int = 20) -> CompressedSignature:
    """
    Full compression pipeline:

        1. seed 12 vertices in ℝ⁶ from the hash of μ
        2. binary contraction 12 → 6 → 3
        3. return 3 amplitudes + Π + 2-bit packed μ
    """
    K = len(mu) - 1
    # ---- seed 12 vertices deterministically from μ ----
    h = hash(tuple(int(x) for x in mu[:min(K, 32)]))
    rng = np.random.default_rng(seed_vertices ^ (h & 0xFFFFFFFF))
    V12 = rng.standard_normal((12, 6))

    # ---- binary contraction ----
    contract = binary_contraction(V12)
    amplitudes = contract["amplitudes"]
    Pi_scalar = contract["Pi_scalar"]

    # ---- pack μ ----
    mu_packed = compress_mobius(mu)
    abs_sum = int(np.sum(np.abs(mu[1:K + 1])))

    return CompressedSignature(K, abs_sum, amplitudes, Pi_scalar, mu_packed)


# ============================================================
#  8. Fiber-optic reconstruction  (fiber-optic-signal-ugc.py)
# ============================================================
def reconstruct_via_fiber(sig: CompressedSignature,
                          K_gate: int = 20,
                          N_samples: int = 512,
                          barrier_width: int = 41) -> dict:
    """
    Reconstruct the amplitudes through the fiber-optic UGC pipeline:

        1. build the 6-vertex set signal from the amplitudes + Π
        2. elliptic Möbius gate → amplitude envelope |ζ(t)|
        3. encode amplitudes as phase modulation on the envelope
        4. propagate through the fiber (exponential dispersion kernel)
        5. decode amplitudes from the received field
        6. unpack the Möbius array with the abs_sum check
    """
    K = sig.K
    amps = sig.amplitudes                     # 3 real amplitudes
    Pi_scalar = sig.Pi_scalar

    # ---- 1. build 6-vertex set signal (tile 3 amps → 6 modes) ----
    six_amps = np.concatenate([amps, amps])  # 6 modes
    six_amps = six_amps / (six_amps.max() + 1e-12)
    phase0 = 0.0 if Pi_scalar >= 0 else math.pi
    envelope_scale = math.tanh(abs(Pi_scalar))

    # ---- 2. elliptic Möbius gate ----
    gate = EllipticMobiusGate(K_gate, smooth=True)
    t_vals = np.linspace(0, gate.period, N_samples)
    zeta_mag = gate.amplitude(t_vals)
    amp_env = 0.3 + 0.7 * (zeta_mag / (zeta_mag.max() + 1e-12))

    # ---- 3. encode as phase modulation across N_v vertices ----
    N_v = 6
    phi = 2.0 * PI * np.arange(N_v) / N_v
    amp_at_v = np.interp(np.linspace(0, 1, N_v),
                         np.linspace(0, 1, N_samples), amp_env)
    A_full = six_amps * amp_at_v * envelope_scale
    E0 = A_full * np.cos(phi + phase0)
    B0 = A_full * np.sin(phi + phase0)

    # ---- 4. fiber propagation ----
    z_grid = np.linspace(0, 1, N_samples)
    E_input = np.interp(z_grid, np.linspace(0, 1, N_v), E0)
    B_input = np.interp(z_grid, np.linspace(0, 1, N_v), B0)
    x = np.arange(-barrier_width // 2, barrier_width // 2)
    kernel = np.exp(-ALPHA_SYM * np.abs(x))
    kernel = kernel / kernel.sum()
    E_out = convolve(E_input, kernel, mode="same")
    B_out = convolve(B_input, kernel, mode="same")

    # ---- 5. decode amplitudes ----
    # sample the received intensity at the 6 mode positions
    samples_E = np.interp(np.linspace(0, 1, N_v),
                          np.linspace(0, 1, N_samples), E_out)
    samples_B = np.interp(np.linspace(0, 1, N_v),
                          np.linspace(0, 1, N_samples), B_out)
    # normalise by the known envelope
    ref = np.maximum(amp_at_v, 1e-6)
    samples_E_n = samples_E / ref
    samples_B_n = samples_B / ref
    # recovered amplitudes = |E + iB| on the unit circle
    recovered_complex = samples_E_n + 1j * samples_B_n
    recovered_six = np.abs(recovered_complex)
    # average the first 3 and last 3 to recover the 3 original amplitudes
    recovered_amps = 0.5 * (recovered_six[:3] + recovered_six[3:])

    # ---- 6. unpack the Möbius array ----
    mu_recovered = decompress_mobius(sig.mu_packed)
    # check the recovered amplitudes are within tolerance
    amp_err = np.max(np.abs(recovered_amps - amps)) / (np.max(amps) + 1e-12)

    return dict(
        gate=gate,
        set_signal=dict(six_amps=six_amps, phase0=phase0,
                        Pi_scalar=Pi_scalar,
                        envelope_scale=envelope_scale),
        amp_env=amp_env,
        A_full=A_full, E0=E0, B0=B0,
        E_out=E_out, B_out=B_out,
        recovered_amps=recovered_amps,
        amp_error=amp_err,
        mu_recovered=mu_recovered,
    )


# ============================================================
#  9. Full demo
# ============================================================
def demo(K: int = 200, K_gate: int = 20,
         N_samples: int = 512, barrier_width: int = 41,
         seed_vertices: int = 42):
    print("=" * 80)
    print("Möbius binary contraction  ·  fiber-optic reconstruction")
    print("=" * 80)
    print(f"  K              = {K}")
    print(f"  K_gate         = {K_gate}")
    print(f"  N_samples      = {N_samples}")
    print(f"  barrier_width  = {barrier_width}")
    print()

    # ---------- 1. build the Möbius array ----------
    t0 = time.perf_counter()
    mu = mobius_sieve(K)
    t_sieve = (time.perf_counter() - t0) * 1e3
    print(f"[1] Möbius sieve        {t_sieve:8.2f} ms")

    # ---------- 2. compress ----------
    t0 = time.perf_counter()
    sig = compress(mu, seed_vertices=seed_vertices, K_gate=K_gate)
    t_compress = (time.perf_counter() - t0) * 1e3
    payload = sig.to_bytes()
    print(f"[2] Binary compression  {t_compress:8.2f} ms  "
          f"({len(payload)} bytes)")
    print(f"      3 amplitudes   : {np.round(sig.amplitudes, 6)}")
    print(f"      Π scalar       : {sig.Pi_scalar:+.6f}")
    print(f"      abs_sum key    : {sig.abs_sum}")
    print(f"      packed μ bytes : {len(sig.mu_packed)}")

    # ---------- 3. round-trip through bytes ----------
    sig2 = CompressedSignature.from_bytes(payload)
    assert np.allclose(sig.amplitudes, sig2.amplitudes)
    assert sig.abs_sum == sig2.abs_sum
    print(f"      round-trip     : OK")

    # ---------- 4. reconstruct via fiber ----------
    t0 = time.perf_counter()
    recon = reconstruct_via_fiber(
        sig2, K_gate=K_gate,
        N_samples=N_samples, barrier_width=barrier_width)
    t_recon = (time.perf_counter() - t0) * 1e3
    print(f"[3] Fiber reconstruction {t_recon:8.2f} ms")
    print(f"      recovered amps : {np.round(recon['recovered_amps'], 6)}")
    print(f"      amp error      : {recon['amp_error']:.6e}")

    # ---------- 5. verify Möbius reconstruction ----------
    mu_rec = recon["mu_recovered"]
    match = bool(np.array_equal(mu, mu_rec))
    print(f"      μ match        : {match}")

    # ---------- 6. compression ratio ----------
    raw_bytes = K * 8                        # 64-bit integers
    ratio = len(payload) / raw_bytes
    print(f"\n--- Compression ratio ---")
    print(f"  raw μ (64-bit ints)  : {raw_bytes:>8} bytes")
    print(f"  compressed payload   : {len(payload):>8} bytes")
    print(f"  ratio                : {ratio:.4f}  "
          f"(saving {100*(1-ratio):.2f} %)")

    # ---------- 7. plot ----------
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.4)

    # (a) Möbius array
    ax = fig.add_subplot(gs[0, 0])
    ax.step(np.arange(1, K + 1), mu[1:K + 1],
            where='mid', color="#3a7bd5", lw=0.7)
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlabel("n")
    ax.set_ylabel("μ(n)")
    ax.set_title(f"Möbius array  (K = {K})")
    ax.grid(True, alpha=0.3)

    # (b) 3 amplitudes
    ax = fig.add_subplot(gs[0, 1])
    ax.bar(np.arange(3), sig.amplitudes, color="#16a085", width=0.6)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["α₀", "α₁", "α₂"])
    ax.set_ylabel("amplitude")
    ax.set_title(f"Binary contraction  ·  Π = {sig.Pi_scalar:+.4f}")
    ax.grid(True, alpha=0.3)

    # (c) Figure 3.5 heat map
    xs = np.linspace(0, W1, 128)
    ys = np.linspace(0, W2, 128)
    X, Y = np.meshgrid(xs, ys)
    P2D = elliptic_projection_2d(X, Y)
    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(P2D, extent=[0, W1, 0, W2], origin="lower",
                   cmap="inferno", aspect="auto", vmin=0, vmax=1)
    ax.set_xlabel("x (mod π)")
    ax.set_ylabel("y (mod e)")
    ax.set_title("Figure 3.5")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (d) elliptic Möbius amplitude envelope
    ax = fig.add_subplot(gs[1, :2])
    t_env = np.linspace(0, recon["gate"].period, N_samples)
    ax.plot(t_env, recon["gate"].amplitude(t_env),
            color="#3a7bd5", lw=1.4, label="|ζ(t)|")
    ax.plot(t_env, recon["amp_env"] * recon["gate"].amplitude(t_env).max(),
            color="#e67e22", lw=1.0, alpha=0.7,
            label="lifted envelope [0.3, 1.0]")
    ax.set_xlabel("t")
    ax.set_ylabel("amplitude")
    ax.set_title("Elliptic Möbius gate — amplitude carrier")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (e) fiber kernel
    ax = fig.add_subplot(gs[1, 2])
    x_k = np.arange(-barrier_width // 2, barrier_width // 2)
    kernel_disp = np.exp(-ALPHA_SYM * np.abs(x_k))
    kernel_disp = kernel_disp / kernel_disp.sum()
    ax.plot(x_k, kernel_disp, color="#8e44ad", lw=1.5)
    ax.set_xlabel("z")
    ax.set_ylabel("H(z)")
    ax.set_title("Fiber kernel  exp(−α|z|)")
    ax.grid(True, alpha=0.3)

    # (f) transmitted signal
    ax = fig.add_subplot(gs[2, :2])
    z_v = np.linspace(0, 1, 6)
    ax.plot(z_v, recon["A_full"], "o-", color="#8e44ad", lw=1.4,
            label="A_full = amp × |ζ| × Π")
    ax.plot(z_v, recon["E0"], "s-", color="#2ecc71", lw=1.0,
            alpha=0.7, label="E₀")
    ax.plot(z_v, recon["B0"], "^-", color="#3498db", lw=1.0,
            alpha=0.7, label="B₀")
    ax.set_xlabel("mode index")
    ax.set_ylabel("field")
    ax.set_title("Transmitter: 6 modes with elliptic amplitude")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (g) received field
    ax = fig.add_subplot(gs[2, 2])
    ax.plot(np.linspace(0, 1, N_samples), recon["E_out"],
            color="#e74c3c", lw=1.0, label="E_out")
    ax.plot(np.linspace(0, 1, N_samples), recon["B_out"],
            color="#8e44ad", lw=1.0, alpha=0.7, label="B_out")
    ax.set_xlabel("position")
    ax.set_ylabel("field")
    ax.set_title("Received field")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (h) amplitude recovery
    ax = fig.add_subplot(gs[3, :2])
    x3 = np.arange(3)
    ax.bar(x3 - 0.18, sig.amplitudes, width=0.34,
           color="#3a7bd5", label="true")
    ax.bar(x3 + 0.18, recon["recovered_amps"], width=0.34,
           color="#e74c3c", label="recovered")
    ax.set_xticks(x3)
    ax.set_xticklabels(["α₀", "α₁", "α₂"])
    ax.set_ylabel("amplitude")
    ax.set_title(f"Amplitude recovery  ·  error = "
                 f"{recon['amp_error']:.3e}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (i) reconstruction correctness
    ax = fig.add_subplot(gs[3, 2])
    ax.step(np.arange(1, K + 1), mu[1:K + 1],
            where='mid', color="#3a7bd5", lw=0.7, label="original")
    ax.step(np.arange(1, K + 1), mu_rec[1:K + 1] + 0.05,
            where='mid', color="#e74c3c", lw=0.5, alpha=0.6,
            label="reconstructed + 0.05")
    ax.set_xlabel("n")
    ax.set_ylabel("μ(n)")
    ax.set_title(f"μ reconstruction  ·  match = {match}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Möbius binary contraction + fiber-optic reconstruction  ·  "
        f"ratio = {ratio:.4f},  amp error = {recon['amp_error']:.3e},  "
        f"μ match = {match}",
        fontsize=13, y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

    return dict(
        mu=mu, sig=sig, recon=recon,
        ratio=ratio, mu_match=match,
        payload_bytes=len(payload), raw_bytes=raw_bytes,
    )


if __name__ == "__main__":
    demo(K=200, K_gate=20,
         N_samples=512, barrier_width=41,
         seed_vertices=42)