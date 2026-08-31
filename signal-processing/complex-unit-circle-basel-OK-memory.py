#!/usr/bin/env python3
"""
mobius_complex_gate.py

Möbius complex gate with real and imaginary parts.
- Positive indices (i > 0): real part coefficients (C_i)
- Negative indices (i < 0): imaginary part coefficients (C_{-|i|})
- Zero index: trivial (0)
- Basel checksum tied to the unit circle integral: ∫ |ζ(t)|² dt = 2*(6/π²)*K
- Ternary packing for Möbius sieve (2 bits per value), with abs_sum key.
"""

import math
import struct
import numpy as np
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

# ---------- Ternary packing (2 bits per value) ----------
def pack_mobius(mu):
    K = len(mu) - 1
    if K < 1:
        return b''
    abs_sum = sum(1 for n in range(1, K+1) if mu[n] != 0)
    bit_len = 2 * K
    num_bytes = (bit_len + 7) // 8
    packed = bytearray(num_bytes)
    bit_pos = 0
    for n in range(1, K + 1):
        code = 0 if mu[n] == 0 else (1 if mu[n] == 1 else 2)
        byte_idx = bit_pos // 8
        bit_offset = 6 - (bit_pos % 8)
        packed[byte_idx] |= (code << bit_offset)
        bit_pos += 2
    header = struct.pack('>II', K, abs_sum)
    return header + bytes(packed)

def unpack_mobius(data):
    if len(data) < 8:
        raise ValueError("Data too short")
    K, abs_sum = struct.unpack('>II', data[:8])
    if K == 0:
        return [0] * (K + 1)
    packed = data[8:]
    mu = [0] * (K + 1)
    bit_pos = 0
    for n in range(1, K + 1):
        byte_idx = bit_pos // 8
        bit_offset = 6 - (bit_pos % 8)
        code = (packed[byte_idx] >> bit_offset) & 0b11
        if code == 0:
            mu[n] = 0
        elif code == 1:
            mu[n] = 1
        elif code == 2:
            mu[n] = -1
        else:
            raise ValueError(f"Invalid code {code} at position {n}")
        bit_pos += 2
    computed_sum = sum(1 for n in range(1, K+1) if mu[n] != 0)
    if computed_sum != abs_sum:
        raise ValueError(f"Integrity check failed: expected abs_sum={abs_sum}, got {computed_sum}")
    return mu

# ---------- Build square‑free coefficients (dict) ----------
def build_square_free_coeffs(K):
    mu = mobius_sieve(K)
    coeffs = {}
    for i in range(-K, K+1):
        if i == 0:
            continue
        if mu[abs(i)] != 0:
            coeffs[i] = mu[abs(i)]
    return coeffs

# ---------- Complex Gate ----------
class MobiusComplexGate:
    """
    Stores real (positive indices) and imaginary (negative indices) parts.
    Memory is O(K) for coefficients + O(K) for sieve.
    Operations: O(K) for zeta/power, O(K log K) for compression.
    Basel checksum verified via integral of |ζ(t)|² over one period.
    """

    def __init__(self, K):
        self.K = K
        self.mu = mobius_sieve(K)
        self.packed_mu = pack_mobius(self.mu)   # for storage

        # Coefficient arrays: length 2K+1, index mapping: i -> i+K
        self.coeffs = np.zeros(2*K + 1, dtype=complex)
        # We'll store real and imaginary parts separately for convenience
        self.real_coeffs = np.zeros(2*K + 1, dtype=float)   # for positive i
        self.imag_coeffs = np.zeros(2*K + 1, dtype=float)   # for negative i

        self.period = 2 * PI * ALPHA
        self.basel_ref = 2 * (6 / (PI * PI)) * K   # theoretical average

    # ---------- Accessors ----------
    def set_real(self, i, val):
        """Set coefficient for positive index i (real part)."""
        if i <= 0 or i > self.K:
            raise ValueError("Index must be positive and ≤ K")
        if self.mu[i] == 0:
            raise ValueError("Index is not square-free")
        self.real_coeffs[i + self.K] = val
        self.coeffs[i + self.K] = val + 0j

    def set_imag(self, i, val):
        """Set coefficient for negative index -i (imaginary part)."""
        if i <= 0 or i > self.K:
            raise ValueError("Index must be positive and ≤ K")
        if self.mu[i] == 0:
            raise ValueError("Index is not square-free")
        idx = -i + self.K
        self.imag_coeffs[idx] = val
        self.coeffs[idx] = 0 + 1j * val

    def get_coeff(self, i):
        """Return complex coefficient at index i (including negative)."""
        return self.coeffs[i + self.K]

    # ---------- Spectral sum ----------
    def zeta(self, t):
        """ζ(t) = Σ_{i=-K}^{K} C_i * exp(i * t * i / α)."""
        total = 0.0 + 0.0j
        # Optimised: loop over non-zero indices only (square-free)
        for i in range(-self.K, self.K+1):
            if i == 0:
                continue
            if self.mu[abs(i)] != 0:
                total += self.coeffs[i + self.K] * np.exp(1j * t * i / ALPHA)
        return total

    def power_spectrum(self, t):
        """|ζ(t)|²."""
        z = self.zeta(t)
        return np.abs(z)**2

    def integrate_power_spectrum(self, N=1000):
        """Numerical integral of |ζ(t)|² over one period."""
        t_vals = np.linspace(0, self.period, N)
        power = np.array([self.power_spectrum(t) for t in t_vals])
        # Use trapezoidal rule (compatible with old and new NumPy)
        try:
            integral = np.trapezoid(power, t_vals)
        except AttributeError:
            integral = np.trapz(power, t_vals)
        avg = integral / self.period
        return avg

    def integrate_power_spectrum_with_step(self, step=None):
        """Use a fixed step (default: ALPHA) for Riemann integration."""
        if step is None:
            step = ALPHA
        num_steps = max(1, int(self.period / step))
        t_vals = np.linspace(0, self.period, num_steps)
        power = np.array([self.power_spectrum(t) for t in t_vals])
        integral = step * np.sum(power)   # left-point Riemann sum
        avg = integral / self.period
        return avg

    def basel_error(self, N=1000):
        avg = self.integrate_power_spectrum(N)
        return abs(avg - self.basel_ref) / (self.basel_ref + 1e-12)

    def check_basel(self, tolerance=0.02, N=1000):
        err = self.basel_error(N)
        return err < tolerance, err

    # ---------- Chip compression (combines real and imaginary) ----------
    def _tsp_route(self, signal):
        K = len(signal)
        angles = np.zeros(K)
        for i in range(K):
            x = math.sin(i * 7.0) + 0.1 * math.cos(i * 13.0)
            y = math.cos(i * 11.0) + 0.1 * math.sin(i * 17.0)
            angles[i] = math.atan2(y, x) + math.pi
        buckets = [[] for _ in range(360)]
        for i, a in enumerate(angles):
            idx = int((a / (2 * PI)) * 360) % 360
            buckets[idx].append(i)
        order = []
        for b in buckets:
            order.extend(b)
        return np.array(order)

    def _conv_exp_kernel(self, signal, alpha=ALPHA):
        K = len(signal)
        lam = math.exp(-alpha)
        f = np.zeros(K, dtype=complex)
        f[0] = signal[0]
        for i in range(1, K):
            f[i] = signal[i] + lam * f[i-1]
        b = np.zeros(K, dtype=complex)
        b[K-1] = signal[K-1]
        for i in range(K-2, -1, -1):
            b[i] = signal[i] + lam * b[i+1]
        conv_exp = (f + b - signal) / (1 - lam * lam)
        conv = (1.0 - conv_exp) / NORM
        return conv

    def _supertrace_and_mass(self, signal):
        S = 0.0
        for idx, val in enumerate(signal):
            sign = 1 if (idx % 2 == 0) else -1
            S += sign * abs(val)
        if S == 0:
            H = 0.0
            m = 0.0
        else:
            p = abs(S) / len(signal)
            if p <= 0 or p >= 1:
                H = 0.0
            else:
                H = -ALPHA * p * math.log(p) if p > 0 else 0.0
            m = abs(S) * math.exp(-H)
        return S, H, m

    def compress(self):
        """
        Compress the combined complex signal (real and imaginary parts interleaved).
        Returns: (kept_indices, kept_values, S, H, m)
        where kept_indices are in the interleaved order (0..2K).
        """
        # Build a single signal of length 2K: interleave real and imaginary
        signal = np.zeros(2*self.K, dtype=complex)
        for i in range(1, self.K+1):
            idx_pos = i + self.K
            signal[2*(i-1)] = self.coeffs[idx_pos]   # real part (positive i)
            idx_neg = -i + self.K
            signal[2*(i-1)+1] = self.coeffs[idx_neg]  # imaginary part (negative i)
        K = len(signal)
        # TSP routing
        order = self._tsp_route(np.abs(signal))
        signal_sorted = signal[order]
        # Convolution
        conv = self._conv_exp_kernel(signal_sorted)
        # Supertrace
        S, H, m = self._supertrace_and_mass(conv)
        M = max(1, int(abs(S)))
        if M > K:
            M = K
        mag = np.abs(conv)
        idx_sorted = np.argsort(mag)[::-1]
        kept = []
        count = 0
        for idx in idx_sorted:
            n = idx + 1
            # we need to map back to original interleaved index
            orig_idx = order[idx]
            if self.mu[abs(orig_idx // 2) + 1] != 0:  # check if square-free
                kept.append((orig_idx, conv[idx]))
                count += 1
                if count >= M:
                    break
        return kept, S, H, m, conv

    def reconstruct_from_compressed(self, kept, K_full=None):
        if K_full is None:
            K_full = 2 * self.K
        recon = np.zeros(K_full, dtype=complex)
        for idx, val in kept:
            recon[idx] = val
        return recon

    # ---------- Storage of the sieve (for serialisation) ----------
    def get_packed_mu(self):
        return self.packed_mu

    def load_mu_from_packed(self, data):
        mu = unpack_mobius(data)
        self.mu = mu
        self.packed_mu = data

# ---------- Demonstration ----------
def demo():
    K = 20
    gate = MobiusComplexGate(K)

    # Fill with some dummy coefficients: real part = log(i), imag part = sin(i)
    for i in range(1, K+1):
        if gate.mu[i] != 0:
            gate.set_real(i, math.log(i+1))
            gate.set_imag(i, math.sin(i))

    # Basel check
    ok, err = gate.check_basel(tolerance=0.05, N=2000)
    print(f"Basel error: {err:.4f}  -> {'OK' if ok else 'FAIL'}")

    # Compute power spectrum at some t
    t = 1.5
    ps = gate.power_spectrum(t)
    print(f"|ζ({t:.2f})|² = {ps:.4f}")

    # Compress the full signal
    kept, S, H, m, conv = gate.compress()
    print(f"Compressed: kept {len(kept)} coefficients (S={S:.4f}, H={H:.4f}, m={m:.4f})")

    # Reconstruct and compare
    recon = gate.reconstruct_from_compressed(kept)
    error = np.linalg.norm(conv - recon) / (np.linalg.norm(conv) + 1e-12)
    print(f"Reconstruction relative error: {error:.4e}")

    # Show the packed sieve size
    packed = gate.get_packed_mu()
    print(f"Packed Möbius sieve size: {len(packed)} bytes")

if __name__ == "__main__":
    demo()