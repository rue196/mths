#!/usr/bin/env python3
"""
fft-audio-derivative.py

Audio compression using M‑matrix trace, finite‑step derivative,
and supertrace compression. The derivative is applied before
compression; reconstruction integrates the derivative back.
"""

import numpy as np
import math
import soundfile as sf
from scipy.signal import hilbert
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
import os

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)          # ≈ 2.362
NORM = 1.0 - math.exp(-ALPHA * (PI + E))
ALPHA_USER = 0.3628
A = ALPHA / ALPHA_USER          # ≈ 6.511 (finite derivative step)

# ---------- Finite derivative ----------
def finite_derivative(signal, step=A):
    """Forward finite difference with step `step`."""
    if len(signal) < 2:
        return np.zeros_like(signal)
    diff = np.zeros_like(signal, dtype=float)
    diff[:-1] = (signal[1:] - signal[:-1]) / step
    return diff

def integrate_signal(deriv, first_sample, step=A):
    """Reconstruct signal from derivative using cumulative sum."""
    recon = first_sample + step * np.cumsum(deriv)
    return recon

# ---------- Möbius sieve (linear, O(K)) ----------
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

# ---------- Integral kernel (Toeplitz) ----------
def integral_kernel(K, alpha=ALPHA):
    norm = 1.0 - math.exp(-alpha * (PI + E))
    kernel = np.zeros(2*K - 1, dtype=float)
    for d in range(-(K-1), K):
        val = (1.0 - math.exp(-alpha * abs(d))) / norm
        kernel[d + (K-1)] = val
    return kernel

# ---------- FFT convolution ----------
def apply_convolution(signal, kernel):
    L = len(signal)
    N = 1 << (2*L - 1).bit_length()
    sig_pad = np.pad(signal, (0, N - L), mode='constant')
    ker_pad = np.pad(kernel, (0, N - len(kernel)), mode='constant')
    conv = ifft(fft(sig_pad) * fft(ker_pad))[:L]
    return conv

# ---------- SuperTrace ----------
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

# ---------- TSP routing (bucket sort by phase) ----------
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

# ---------- M‑matrix trace for stereo ----------
def stereo_trace(left, right, i_exp=2):
    """tr = left^(i-1) + right^(i-1)"""
    return left ** (i_exp - 1) + right ** (i_exp - 1)

# ---------- Compression with derivative ----------
def compress_stereo_derivative(left, right, i_exp=2, use_mobius=True):
    """
    Applies finite derivative to the stereo trace, then compresses.
    Returns (kept, info) where info includes the first sample of the original trace.
    """
    K = len(left)
    assert len(right) == K, "Channels must have same length"

    # 1. Compute original trace
    orig_trace = stereo_trace(left, right, i_exp)

    # 2. Derivative of the trace
    trace_deriv = finite_derivative(orig_trace, A)
    first_sample = orig_trace[0]

    # 3. Build complex pair for routing (use left+1j*right or use derivative trace itself?)
    # We'll use the derivative trace as a real signal; for TSP we need complex.
    # We'll create a complex signal from the derivative and its Hilbert transform.
    # But simpler: use the original left+1j*right for ordering (preserves stereo phase).
    z = left + 1j * right
    order = tsp_route_complex(z)
    coeffs_sorted = trace_deriv[order]

    # 4. Convolution with integral kernel
    kernel = integral_kernel(K, ALPHA)
    conv = apply_convolution(coeffs_sorted, kernel)

    # 5. Supertrace
    S = supertrace_from_coeffs(conv)
    H = entropy_from_supertrace(S, K, ALPHA)
    m = invariant_scalar(conv)
    M = max(1, int(abs(S)))
    if M > K:
        M = K

    # 6. Möbius sieve
    mu = mobius_sieve(K) if use_mobius else None

    # 7. Select top M coefficients (square‑free if requested)
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

# ---------- Reconstruction with integration ----------
def reconstruct_stereo_derivative(kept, info):
    K = info['K']
    # Reconstruct convolved derivative signal
    conv_recon = np.zeros(K, dtype=complex)
    for idx, val in kept:
        conv_recon[idx] = val

    # Invert TSP order
    inv_order = np.argsort(info['order'])
    trace_deriv_recon = np.real(conv_recon)[inv_order]

    # Integrate to recover the original trace
    first_sample = info['first_sample']
    trace_recon = integrate_signal(trace_deriv_recon, first_sample, A)

    return trace_recon

# ---------- Demonstration ----------
def main():
    # Generate synthetic stereo audio
    fs = 44100
    t = np.linspace(0, 1, fs)
    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    right = 0.5 * np.sin(2 * np.pi * 440 * t + 0.5)

    # Save original
    stereo = np.column_stack((left, right))
    sf.write('original_stereo.wav', stereo, fs)

    # Compression with derivative
    i_exp = 2
    use_mobius = True
    kept, info = compress_stereo_derivative(left, right, i_exp, use_mobius)

    print(f"Original length: {info['K']}")
    print(f"Kept coefficients: {len(kept)} (ratio {len(kept)/info['K']:.3f})")
    print(f"Supertrace S = {info['S']:.4f}, Entropy H = {info['H']:.4f}, Mass m = {info['m']:.4f}")

    # Reconstruct
    trace_recon = reconstruct_stereo_derivative(kept, info)

    # Compare with original trace
    orig_trace = stereo_trace(left, right, i_exp)
    error = orig_trace - trace_recon
    snr = 10 * np.log10(np.sum(orig_trace**2) / (np.sum(error**2) + 1e-12))
    print(f"Reconstruction SNR (trace): {snr:.2f} dB")

    # Plot a segment
    plt.figure(figsize=(12, 4))
    plt.plot(orig_trace[:1000], label='Original trace')
    plt.plot(trace_recon[:1000], label='Reconstructed trace')
    plt.legend()
    plt.title('Trace compression using derivative + integration')
    plt.show()

if __name__ == "__main__":
    main()