import math
import random
import numpy as np
import cmath
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)          # ≈ 2.362
NORM = 1.0 - math.exp(-ALPHA * (PI + E))

# ---------- Möbius sieve (O(K)) ----------
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

# ---------- TSP routing (bucket sort by phase, O(K)) ----------
def tsp_route_complex(z):
    """z: 1D array of complex numbers; returns ordering by phase."""
    angles = np.angle(z)
    # Bucket sort: 360 buckets
    buckets = [[] for _ in range(360)]
    for idx, a in enumerate(angles):
        a_norm = a + PI if a < 0 else a          # map to [0, 2π)
        b = int((a_norm / (2 * PI)) * 360) % 360
        buckets[b].append(idx)
    order = []
    for b in buckets:
        order.extend(b)
    return np.array(order)

# ---------- Exponential convolution (O(K)) ----------
def conv_exp_kernel(signal, alpha=ALPHA):
    K = len(signal)
    lam = math.exp(-alpha)
    # Forward pass
    f = np.zeros(K, dtype=complex if np.iscomplexobj(signal) else float)
    f[0] = signal[0]
    for i in range(1, K):
        f[i] = signal[i] + lam * f[i-1]
    # Backward pass
    b = np.zeros(K, dtype=type(signal[0]))
    b[K-1] = signal[K-1]
    for i in range(K-2, -1, -1):
        b[i] = signal[i] + lam * b[i+1]
    # Convolution with exp(-alpha|i-j|)
    conv_exp = (f + b - signal) / (1 - lam * lam)
    # Integral kernel: (1 - conv_exp) / NORM
    conv = (1.0 - conv_exp) / NORM
    return conv

# ---------- Supertrace and entropy ----------
def supertrace_and_mass(signal):
    S = 0.0
    for i, val in enumerate(signal):
        sign = 1 if (i % 2 == 0) else -1
        S += sign * abs(val)
    if S == 0:
        H = 0.0
        m = 0.0
    else:
        p = abs(S) / len(signal)
        H = -ALPHA * p * math.log(p) if p > 0 else 0.0
        m = abs(S) * math.exp(-H)
    return S, H, m

# ---------- Matrix trace (algebraic + transcendental) ----------
def matrix_trace(x, y, i_power=2):
    """
    M = [[x^(i-1), x^(-1)*y^i],
         [y^(-1)*x^i, y^(i-1)]]
    Trace = x^(i-1) + y^(i-1)
    """
    return x ** (i_power - 1) + y ** (i_power - 1)

# ---------- Generate WiFi-like addresses (subcarriers / antennas) ----------
def generate_wifi_addresses(K, seed=42, freq_range=(2.4e9, 2.5e9)):
    """
    Simulate K subcarriers with (x, y) representing in-phase/quadrature.
    """
    random.seed(seed)
    addresses = []
    for _ in range(K):
        # Randomly choose frequency offset (simulates OFDM subcarriers)
        f = random.uniform(*freq_range)
        # Phase and amplitude (x,y) as normalized I/Q components
        x = 0.5 + 0.5 * random.random()   # DC bias
        y = 0.5 + 0.5 * random.random()
        addresses.append((x, y, f))
    return addresses

# ---------- Main compression & transmission pipeline ----------
def wifi_transmission_pipeline(addresses, i_power=2, use_mobius=True):
    K = len(addresses)
    print(f"Number of subcarriers: {K}")

    # 1. Compute coefficients from M‑matrix trace
    coeffs = np.array([matrix_trace(x, y, i_power) for x, y, _ in addresses], dtype=complex)
    # Add a phase depending on frequency to simulate time evolution (transcendental part)
    # We'll multiply each coefficient by exp(i * 2π * f * t) at a reference time
    # For simplicity we take t = 1.0 and use frequency to create a complex signal.
    # But we need a time series, so we'll treat each subcarrier as a sinusoid.
    # We'll build a time-domain signal later.
    # Here we just treat coeffs as the initial complex amplitude per subcarrier.

    # 2. TSP routing by phase of the coefficient (or by address)
    # We'll use the complex coefficient's angle for ordering.
    order = tsp_route_complex(coeffs)
    coeffs_sorted = coeffs[order]

    # 3. Convolution with integral kernel (simulates propagation / filtering)
    conv = conv_exp_kernel(coeffs_sorted)

    # 4. Supertrace and entropy
    S, H, m = supertrace_and_mass(conv)
    M = max(1, int(abs(S)))
    if M > K:
        M = K
    print(f"Supertrace S = {S:.4f}, Entropy H = {H:.4f}, Mass m = {m:.4f}")
    print(f"Keeping M = {M} coefficients (based on |S|)")

    # 5. Möbius sieve
    mu = mobius_sieve(K) if use_mobius else None

    # 6. Compression: keep top M magnitudes, with μ(index) != 0 if use_mobius
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
    print(f"Compressed size: {len(kept)} (ratio {len(kept)/K:.3f})")

    # 7. Reconstruct (zero out non-kept)
    recon = np.zeros(K, dtype=complex)
    for idx, val in kept:
        recon[idx] = val
    error = np.linalg.norm(conv - recon) / (np.linalg.norm(conv) + 1e-12)
    print(f"Reconstruction relative L2 error: {error:.4e}")

    return kept, S, H, m, conv, coeffs, order, recon, error

# ---------- Convert compressed spectrum to time-domain WiFi signal ----------
def spectrum_to_wifi_signal(kept, info, fs=100e6, num_samples=1024):
    """
    Reconstruct time-domain signal from kept frequency components.
    info contains 'order' and 'K'. The kept coefficients are in the sorted order.
    We need to map back to original subcarrier order to assign correct frequencies.
    """
    K = info['K']
    # We need the original frequencies (addresses) to reconstruct.
    # For this demo we'll just treat the indices as frequency bins.
    # We'll create a complex spectrum of length K with zeros, fill kept.
    spec = np.zeros(K, dtype=complex)
    for idx, val in kept:
        spec[idx] = val
    # Inverse TSP order to restore original subcarrier order
    inv_order = info.get('inv_order', np.argsort(info['order']))
    spec_original = spec[inv_order]
    # Now we need to map these to physical frequencies. For simplicity, we'll assume
    # uniform spacing: f_i = f0 + i * df. We'll create a time vector.
    # Since we don't have actual frequency values, we use the index as frequency bin.
    # We'll zero-pad and IFFT to get time-domain signal.
    N = max(K, num_samples)
    spec_pad = np.zeros(N, dtype=complex)
    # Place the spectrum at the lower half (DC to fs/2)
    spec_pad[:K] = spec_original
    # Make Hermitian symmetric for real signal (if we want real output)
    # But here we keep complex analytic signal.
    time_signal = np.fft.ifft(spec_pad) * N
    return time_signal

# ---------- Simulation with real WiFi parameters ----------
def main():
    # Parameters
    K = 64                   # number of subcarriers (like OFDM)
    fs = 20e6                # sampling frequency
    num_samples = 512
    i_power = 2              # exponent for transcendental part

    # Generate addresses (subcarriers with random I/Q and frequencies)
    addresses = generate_wifi_addresses(K, seed=42)
    # Extract just x,y for matrix trace
    addr_xy = [(x, y) for x, y, _ in addresses]

    # Run compression pipeline on the algebraic part
    kept, S, H, m, conv, coeffs, order, recon, error = wifi_transmission_pipeline(addr_xy, i_power, use_mobius=True)

    # Store inverse order for later reconstruction
    inv_order = np.argsort(order)
    info = {'K': K, 'order': order, 'inv_order': inv_order}

    # Now simulate a time-domain transmission: we'll create a spectrum from the kept coefficients.
    time_signal = spectrum_to_wifi_signal(kept, info, fs, num_samples)

    # Also compute original intensity (sum of |coeff|^2) from the full spectrum
    # Reconstruct the full spectrum from kept + zeros
    spec_full = np.zeros(K, dtype=complex)
    for idx, val in kept:
        spec_full[idx] = val
    # Inverse TSP order
    spec_orig_order = spec_full[inv_order]
    # Create full time-domain from original spectrum (no compression)
    spec_full_pad = np.zeros(num_samples, dtype=complex)
    spec_full_pad[:K] = spec_orig_order
    time_orig = np.fft.ifft(spec_full_pad) * num_samples

    # Plot
    t = np.arange(num_samples) / fs
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, np.real(time_orig), label='Original I')
    plt.plot(t, np.imag(time_orig), label='Original Q')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Original WiFi baseband signal (I/Q)')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(t, np.real(time_signal), label='Reconstructed I')
    plt.plot(t, np.imag(time_signal), label='Reconstructed Q')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Reconstructed WiFi signal from compressed spectrum')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Compute SNR of reconstruction
    noise = time_orig - time_signal
    snr = 10 * np.log10(np.sum(np.abs(time_orig)**2) / np.sum(np.abs(noise)**2))
    print(f"Reconstruction SNR: {snr:.2f} dB")

if __name__ == "__main__":
    main()