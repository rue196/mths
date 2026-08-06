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
    """
    left, right: 1D arrays (same length)
    Returns: array of traces where tr = left^(i-1) + right^(i-1)
    """
    return left ** (i_exp - 1) + right ** (i_exp - 1)

# ---------- Compression ----------
def compress_stereo(left, right, i_exp=2, use_mobius=True):
    """
    left, right: 1D float arrays
    Returns: compressed dict, metadata
    """
    K = len(left)
    assert len(right) == K, "Channels must have same length"

    # 1. Compute traces
    coeffs = stereo_trace(left, right, i_exp)

    # 2. Build complex pairs for routing (use left+1j*right)
    z = left + 1j * right
    order = tsp_route_complex(z)
    coeffs_sorted = coeffs[order]

    # 3. Convolution with integral kernel
    kernel = integral_kernel(K, ALPHA)
    conv = apply_convolution(coeffs_sorted, kernel)

    # 4. Supertrace
    S = supertrace_from_coeffs(conv)
    H = entropy_from_supertrace(S, K, ALPHA)
    m = invariant_scalar(conv)
    M = max(1, int(abs(S)))
    if M > K:
        M = K

    # 5. Möbius sieve
    mu = mobius_sieve(K) if use_mobius else None

    # 6. Select top M coefficients (square‑free if requested)
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

    # Store metadata
    info = {
        'S': S, 'H': H, 'm': m, 'M': M,
        'order': order, 'kernel': kernel,
        'use_mobius': use_mobius, 'i_exp': i_exp,
        'K': K
    }
    return kept, info

# ---------- Reconstruction ----------
def reconstruct_stereo(kept, info):
    K = info['K']
    # Reconstruct convolved signal (zero out non‑kept)
    conv_recon = np.zeros(K, dtype=complex)
    for idx, val in kept:
        conv_recon[idx] = val

    # We cannot recover the original coefficients without de‑convolution.
    # For a lossy representation, we take the real part as the reconstructed trace.
    trace_recon = np.real(conv_recon)
    # Invert the TSP order to get back to original time order
    inv_order = np.argsort(info['order'])
    trace_recon_original = trace_recon[inv_order]

    # To get left and right from the trace is impossible without further information.
    # We'll just return the trace as a mono signal.
    return trace_recon_original

# ---------- Demonstration ----------
def main():
    # Generate a synthetic stereo audio: left = sine, right = sine with phase shift
    fs = 44100
    t = np.linspace(0, 1, fs)
    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    right = 0.5 * np.sin(2 * np.pi * 440 * t + 0.5)

    # Save original
    stereo = np.column_stack((left, right))
    sf.write('original_stereo.wav', stereo, fs)

    # Compression
    i_exp = 2
    use_mobius = True
    kept, info = compress_stereo(left, right, i_exp, use_mobius)

    print(f"Original length: {info['K']}")
    print(f"Kept coefficients: {len(kept)} (ratio {len(kept)/info['K']:.3f})")
    print(f"Supertrace S = {info['S']:.4f}, Entropy H = {info['H']:.4f}, Mass m = {info['m']:.4f}")

    # Reconstruct (mono trace)
    trace_recon = reconstruct_stereo(kept, info)

    # Compare with original trace (ground truth)
    orig_trace = stereo_trace(left, right, i_exp)
    error = orig_trace - trace_recon
    snr = 10 * np.log10(np.sum(orig_trace**2) / np.sum(error**2))
    print(f"Reconstruction SNR (trace): {snr:.2f} dB")

    # Plot a segment
    plt.figure(figsize=(12, 4))
    plt.plot(orig_trace[:1000], label='Original trace')
    plt.plot(trace_recon[:1000], label='Reconstructed trace')
    plt.legend()
    plt.title('Trace compression using M‑matrix + supertrace')
    plt.show()

if __name__ == "__main__":
    main()