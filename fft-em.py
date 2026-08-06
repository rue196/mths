import numpy as np
import math
from scipy.signal import hilbert
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt

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

# ---------- SuperTrace and entropy ----------
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

# ---------- EM intensity from two orthogonal components ----------
def em_intensity(Ex, Ey):
    """Return total intensity I = |Ex|^2 + |Ey|^2 as a scalar sequence."""
    return np.abs(Ex)**2 + np.abs(Ey)**2

# ---------- Compression of EM signal ----------
def compress_em(Ex, Ey, use_mobius=True):
    """
    Ex, Ey: 1D complex arrays (analytic signals) or real arrays.
    Returns compressed data and metadata.
    """
    K = len(Ex)
    assert len(Ey) == K

    # 1. Compute intensity coefficients
    coeffs = em_intensity(Ex, Ey)

    # 2. Build complex pairs for routing: use Ex + i*Ey (phase information)
    z = Ex + 1j * Ey
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

    info = {
        'S': S, 'H': H, 'm': m, 'M': M,
        'order': order, 'kernel': kernel,
        'use_mobius': use_mobius, 'K': K
    }
    return kept, info

# ---------- Reconstruction (intensity) ----------
def reconstruct_em(kept, info):
    K = info['K']
    conv_recon = np.zeros(K, dtype=complex)
    for idx, val in kept:
        conv_recon[idx] = val
    # Inverse TSP order
    inv_order = np.argsort(info['order'])
    intensity_recon = np.real(conv_recon[inv_order])
    return intensity_recon

# ---------- Simulation and visualisation ----------
def main():
    # Generate synthetic EM signal: two polarisation components with a chirp
    fs = 1000
    t = np.linspace(0, 1, fs)
    Ex = np.sin(2 * np.pi * (100 + 50 * t) * t) + 0.5 * np.cos(2 * np.pi * 200 * t)
    Ey = 0.8 * np.sin(2 * np.pi * (120 + 30 * t) * t + 0.3)
    # Make analytic (complex) to get phase
    Ex = hilbert(Ex)
    Ey = hilbert(Ey)

    # Compression
    use_mobius = True
    kept, info = compress_em(Ex, Ey, use_mobius)
    print(f"Original length: {info['K']}")
    print(f"Kept coefficients: {len(kept)} (ratio {len(kept)/info['K']:.3f})")
    print(f"Supertrace S = {info['S']:.4f}, Entropy H = {info['H']:.4f}, Mass m = {info['m']:.4f}")

    # Reconstruct intensity
    intensity_orig = em_intensity(Ex, Ey)
    intensity_recon = reconstruct_em(kept, info)

    # SNR of intensity
    error = intensity_orig - intensity_recon
    snr = 10 * np.log10(np.sum(intensity_orig**2) / np.sum(error**2))
    print(f"Reconstruction SNR: {snr:.2f} dB")

    # Plot
    plt.figure(figsize=(12, 4))
    plt.plot(t, intensity_orig, label='Original intensity')
    plt.plot(t, intensity_recon, label='Reconstructed intensity')
    plt.xlabel('Time (s)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.title('EM signal compression via M‑matrix trace + supertrace + Möbius sieve')
    plt.show()

if __name__ == "__main__":
    main()