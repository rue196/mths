#!/usr/bin/env python3
"""
audio_to_packing.py

Compute supertrace mass from an audio signal (using finite‑step derivative),
then pack that many unit squares into a grid with spacing A = 6.511.

The mass m from the audio trace sets the number of squares:
    n_squares = int(round(sqrt(m)))  or using log scaling.
For large m, we take log10 to get a manageable count.
"""

import math
import numpy as np
import soundfile as sf
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)          # ≈ 2.362
NORM = 1.0 - math.exp(-ALPHA * (PI + E))
ALPHA_USER = 0.3628
A = ALPHA / ALPHA_USER          # ≈ 6.511 (finite derivative step, also grid spacing)

# ---------- Audio compression functions (simplified) ----------
def mobius_sieve(K):
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

def integral_kernel(K, alpha=ALPHA):
    norm = 1.0 - math.exp(-alpha * (PI + E))
    kernel = np.zeros(2*K - 1, dtype=float)
    for d in range(-(K-1), K):
        val = (1.0 - math.exp(-alpha * abs(d))) / norm
        kernel[d + (K-1)] = val
    return kernel

def apply_convolution(signal, kernel):
    L = len(signal)
    N = 1 << (2*L - 1).bit_length()
    sig_pad = np.pad(signal, (0, N - L), mode='constant')
    ker_pad = np.pad(kernel, (0, N - len(kernel)), mode='constant')
    conv = ifft(fft(sig_pad) * fft(ker_pad))[:L]
    return conv

def supertrace_from_coeffs(C):
    S = 0.0
    for idx, coeff in enumerate(C):
        sign = 1 if (idx % 2 == 0) else -1
        S += sign * abs(coeff)
    return S

def entropy_from_supertrace(S, N, alpha=ALPHA):
    if S == 0:
        return 0.0
    p = abs(S) / N
    if p <= 0:
        return 0.0
    return -alpha * p * math.log(p)

def invariant_scalar(C):
    S = supertrace_from_coeffs(C)
    H = entropy_from_supertrace(S, len(C))
    return abs(S) * math.exp(-H)

def tsp_route_complex(z):
    angles = np.angle(z)
    buckets = [[] for _ in range(360)]
    for idx, a in enumerate(angles):
        a_norm = a + PI if a < 0 else a
        b = int((a_norm / (2 * PI)) * 360) % 360
        buckets[b].append(idx)
    order = []
    for b in buckets:
        order.extend(b)
    return np.array(order)

def finite_derivative(signal, step=A):
    if len(signal) < 2:
        return np.zeros_like(signal)
    diff = np.zeros_like(signal, dtype=float)
    diff[:-1] = (signal[1:] - signal[:-1]) / step
    return diff

def stereo_trace(left, right, i_exp=2):
    return left ** (i_exp - 1) + right ** (i_exp - 1)

def compress_stereo_derivative(left, right, i_exp=2, use_mobius=True):
    K = len(left)
    assert len(right) == K, "Channels must have same length"
    orig_trace = stereo_trace(left, right, i_exp)
    trace_deriv = finite_derivative(orig_trace, A)
    first_sample = orig_trace[0]
    z = left + 1j * right
    order = tsp_route_complex(z)
    coeffs_sorted = trace_deriv[order]
    kernel = integral_kernel(K, ALPHA)
    conv = apply_convolution(coeffs_sorted, kernel)
    S = supertrace_from_coeffs(conv)
    H = entropy_from_supertrace(S, K, ALPHA)
    m = invariant_scalar(conv)
    mu = mobius_sieve(K) if use_mobius else None
    M = max(1, int(abs(S)))
    if M > K:
        M = K
    mag = np.abs(conv)
    idx_sorted = np.argsort(mag)[::-1]
    kept = []
    count = 0
    for idx in idx_sorted:
        if use_mobius:
            n = idx + 1
            if mu[n] == 0:
                continue
        kept.append((idx, conv[idx]))
        count += 1
        if count >= M:
            break
    info = {
        'S': S, 'H': H, 'm': m, 'M': M,
        'order': order, 'kernel': kernel,
        'use_mobius': use_mobius, 'i_exp': i_exp,
        'K': K, 'first_sample': first_sample
    }
    return kept, info

# ---------- Square packing using the mass ----------
def pack_squares_from_mass(mass, spacing=A):
    """
    Pack squares in a grid using the mass to determine the number.
    Number of squares = int(round(sqrt(mass))) for moderate mass,
    but for large mass we take log10 to get a reasonable count.
    """
    # For very large mass, take log10 to compress
    if mass > 1e6:
        n = int(round(10 * math.log10(mass)))   # e.g., mass=2.7e11 -> log10=11.4 -> n≈114
    else:
        n = int(round(math.sqrt(mass)))
    n = max(1, n)
    # Grid side: number of squares per side
    side = int(math.ceil(math.sqrt(n)))
    # Total squares in grid
    total = side * side
    # Container side = side * spacing
    container_side = side * spacing
    print(f"Mass = {mass:.2e} -> n_squares = {total}, container side = {container_side:.3f}")
    return total, container_side, side

def draw_packing(total, container_side, side, spacing):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, container_side)
    ax.set_ylim(0, container_side)
    ax.set_aspect('equal')
    ax.set_title(f'Packing {total} squares (grid {side}x{side}), spacing = {spacing:.3f}')
    half = spacing / 2.0
    for ix in range(side):
        for iy in range(side):
            x = ix * spacing + half/2
            y = iy * spacing + half/2
            rect = Rectangle((x - half/2, y - half/2), half, half, fc='blue', ec='black', alpha=0.7)
            ax.add_patch(rect)
    plt.show()

# ---------- Main ----------
def main():
    # 1. Generate or load audio (synthetic stereo)
    fs = 44100
    t = np.linspace(0, 1, fs)
    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    right = 0.5 * np.sin(2 * np.pi * 440 * t + 0.5)

    # 2. Compute supertrace mass from audio
    i_exp = 2
    use_mobius = True
    kept, info = compress_stereo_derivative(left, right, i_exp, use_mobius)
    mass = info['m']
    print(f"Audio supertrace mass m = {mass:.4e}")

    # 3. Pack squares using the mass
    total, container_side, side = pack_squares_from_mass(mass, spacing=A)

    # 4. Draw the packing
    draw_packing(total, container_side, side, spacing=A)

if __name__ == "__main__":
    main()