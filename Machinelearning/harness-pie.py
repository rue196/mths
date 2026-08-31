#!/usr/bin/env python3
"""
mobius_harness.py

Unified harness for the Möbius chip pipeline:
  1. Angle‑sort tasks (t points) using chip‑g's TSP routing.
  2. Compute elliptic Möbius Fourier spectrum |ζ(t)|².
  3. Buffer and merge results via InferenceBuffer.
  4. Compile merged data into Möbius memory.
  5. Check for coding errors using the dual‑stage checker.
"""

import math
import numpy as np
from compiler import MobiusCrossCompiler
from random_access_colla_mobius import MobiusCollatzMemory
from ML import inverse_score
from Oklogk import mu_convolution_H
import hashlib



# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)
NORM = 1.0 - math.exp(-ALPHA * (PI + E))

class ChipProcessor:
    """
    A processor that performs the chip pipeline with pre‑allocated buffers.
    This reduces garbage collection and allocation overhead.
    """

    def __init__(self, max_K=1000):
        """
        Allocate buffers for up to max_K elements.
        """
        self.max_K = max_K
        # Buffers for convolution (two passes)
        self.f = np.zeros(max_K, dtype=float)
        self.b = np.zeros(max_K, dtype=float)
        # Buffer for convolution result
        self.conv = np.zeros(max_K, dtype=float)
        # Buffer for sorted indices (TSP order)
        self.order = np.zeros(max_K, dtype=int)
        # Buffer for magnitudes (for sorting)
        self.mag = np.zeros(max_K, dtype=float)
        # Buffer for indices (for argsort)
        self.idx = np.arange(max_K, dtype=int)   # reusable index array
        # Cache for Möbius sieve (computed once)
        self.mu = None
        self._update_mu(max_K)

    def _update_mu(self, K):
        """Compute Möbius sieve up to K (if not already cached)."""
        if self.mu is None or len(self.mu) < K + 1:
            self.mu = self._mobius_sieve(K)

    @staticmethod
    def _mobius_sieve(K):
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

    def _tsp_route(self, signal, K):
        """Compute TSP routing order using bucket sort; store in self.order."""
        # Use deterministic pseudo‑angles
        angles = np.zeros(K, dtype=float)
        for i in range(K):
            x = math.sin(i * 7.0) + 0.1 * math.cos(i * 13.0)
            y = math.cos(i * 11.0) + 0.1 * math.sin(i * 17.0)
            angles[i] = math.atan2(y, x) + math.pi
        # Bucket sort: 360 buckets
        buckets = [[] for _ in range(360)]
        for i in range(K):
            a = angles[i]
            idx = int((a / (2 * math.pi)) * 360) % 360
            buckets[idx].append(i)
        order = []
        for b in buckets:
            order.extend(b)
        self.order[:K] = order

    def _conv_exp_kernel(self, signal, K, alpha=ALPHA):
        """Two‑pass exponential convolution; store result in self.conv."""
        lam = math.exp(-alpha)
        # Forward pass
        f = self.f
        f[0] = signal[0]
        for i in range(1, K):
            f[i] = signal[i] + lam * f[i-1]
        # Backward pass
        b = self.b
        b[K-1] = signal[K-1]
        for i in range(K-2, -1, -1):
            b[i] = signal[i] + lam * b[i+1]
        # Combine
        conv = self.conv
        inv_den = 1.0 / (1.0 - lam * lam)
        inv_norm = 1.0 / NORM
        for i in range(K):
            conv_exp = (f[i] + b[i] - signal[i]) * inv_den
            conv[i] = (1.0 - conv_exp) * inv_norm

    def _supertrace_and_mass(self, signal, K):
        """Compute S, H, m from the signal (assumed in self.conv)."""
        S = 0.0
        for i in range(K):
            val = signal[i]
            if i % 2 == 0:
                S += abs(val)
            else:
                S -= abs(val)
        if S == 0:
            H = 0.0
            m = 0.0
        else:
            p = abs(S) / K
            if p <= 0:
                H = 0.0
            else:
                H = -ALPHA * p * math.log(p)
            m = abs(S) * math.exp(-H)
        return S, H, m

    def process(self, signal):
        """
        Run the chip pipeline on `signal` (1D numpy array).
        Returns: (kept, S, H, m, conv)
        """
        K = len(signal)
        if K > self.max_K:
            raise ValueError(f"Signal length {K} exceeds max_K {self.max_K}; reinitialize processor with larger max_K.")

        # 1. TSP routing
        self._tsp_route(signal, K)
        order = self.order[:K]
        # Reorder signal in a temporary view? We'll create a sorted copy.
        signal_sorted = signal[order]   # This creates a new array; but we can reuse a buffer if we want.
        # Since we need the sorted signal, we'll allocate a buffer for it.
        # To avoid extra allocation, we could use a pre‑allocated buffer `self.sorted_signal`.
        if not hasattr(self, 'sorted_signal') or len(self.sorted_signal) < K:
            self.sorted_signal = np.zeros(K, dtype=signal.dtype)
        self.sorted_signal[:K] = signal[order]

        # 2. Convolution
        self._conv_exp_kernel(self.sorted_signal, K)

        # 3. Supertrace
        S, H, m = self._supertrace_and_mass(self.conv, K)
        M = max(1, int(abs(S)))
        if M > K:
            M = K

        # 4. Möbius sieve (ensure it's up to date)
        self._update_mu(K)
        mu = self.mu

        # 5. Compression: keep top M with square‑free index (μ(n) != 0)
        mag = self.mag[:K]
        for i in range(K):
            mag[i] = abs(self.conv[i])

        # Get indices sorted by magnitude descending using argsort
        # We'll use a pre‑allocated index buffer and sort.
        idx = self.idx[:K]   # already 0..K-1
        # We need to sort idx by mag descending. We'll use np.argsort on a copy of mag.
        sorted_idx = np.argsort(mag)[::-1]   # This allocates a new array; but we can reuse a buffer.
        # Since argsort always returns a new array, we can't avoid allocation easily.
        # We'll just use it; it's O(K log K) but memory allocation is small.

        kept = []
        count = 0
        for idx in sorted_idx:
            n = idx + 1
            if mu[n] != 0:
                kept.append((idx, self.conv[idx]))
                count += 1
                if count >= M:
                    break

        # Return results
        # We also return a copy of conv for external use (if needed)
        conv_copy = self.conv[:K].copy()
        return kept, S, H, m, conv_copy

# ---------- Backward‑compatible function ----------
def chip_pipeline(signal, processor=None):
    """
    Run the chip pipeline, optionally reusing a ChipProcessor.
    If processor is None, a temporary one is created.
    """
    if processor is None:
        processor = ChipProcessor(max_K=len(signal))
    return processor.process(signal)

class ChipLogicGate:
    """
    A bounded Möbius logic gate that applies a function to a signal,
    then runs the chip pipeline (convolution + supertrace + Möbius compression).
    The output is a compressed representation of the transformed signal.
    """

    def __init__(self, max_K=1000):
        self.processor = ChipProcessor(max_K=max_K)

    def apply_function(self, signal, func, *args, **kwargs):
        """
        Apply a function `func` to each element of `signal`.
        `func` can be a callable (e.g., math.log, math.exp, np.sin)
        or a string ('log', 'exp', 'sin', 'cos', 'trace').
        """
        if isinstance(func, str):
            func_name = func.lower()
            if func_name == 'log':
                # avoid log(0)
                safe_signal = np.maximum(signal, 1e-12)
                transformed = np.log(safe_signal)
            elif func_name == 'exp':
                transformed = np.exp(signal)
            elif func_name == 'sin':
                transformed = np.sin(signal)
            elif func_name == 'cos':
                transformed = np.cos(signal)
            elif func_name == 'trace':
                # Matrix trace function: assumes signal is complex and represents matrix entries
                transformed = self._matrix_trace(signal)
            else:
                raise ValueError(f"Unknown function name: {func_name}")
        else:
            # assume it's a callable
            transformed = func(signal, *args, **kwargs)

        # Run the chip pipeline on the transformed signal
        kept, S, H, m, conv = self.processor.process(transformed)

        # Return the compressed representation and invariants
        return kept, S, H, m, conv

    def _matrix_trace(self, signal):
        """
        Compute the trace of a 2x2 matrix from a signal of length 4:
        signal = [M00, M01, M10, M11]  (complex or real)
        Returns the trace (scalar) repeated to match the original length?
        For a logic gate, we treat the trace as a scalar that multiplies the signal.
        """
        if len(signal) != 4:
            raise ValueError("Signal for matrix trace must have exactly 4 elements.")
        M = np.array(signal).reshape(2, 2)
        trace = np.trace(M)
        # Return a constant signal of the same length as input (if we want element-wise)
        # But here we treat it as a scalar result; we'll expand to an array of length 1.
        return np.array([trace])

    def bounded_log(self, signal):
        """Apply log with a bound: replace log(x) with log(max(x, epsilon))."""
        return self.apply_function(signal, 'log')

    def bounded_exp(self, signal):
        """Apply exp and then clip to prevent overflow."""
        # The chip pipeline will compress, so no need to clip, but we can.
        return self.apply_function(signal, 'exp')

    # Additional functions can be added similarly
# ---------- Möbius sieve ----------
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

def build_elliptic_coeffs(K, smooth=True):
    """
    Build C_i for i = -K..K.
    If smooth=True, fill zero indices via linear interpolation.
    Returns: (coeffs, full_array)
    """
    mu = mobius_sieve(K)
    c = np.zeros(2*K + 1, dtype=float)
    # First assign the non‑zero μ values
    for i in range(-K, K+1):
        if i == 0:
            c[i + K] = 0.0
        else:
            c[i + K] = mu[abs(i)]

    if smooth:
        # For each zero index, interpolate between nearest non‑zero neighbours
        for i in range(-K, K+1):
            if i == 0:
                continue
            if c[i + K] == 0.0:
                left = i - 1
                right = i + 1
                while left >= -K and c[left + K] == 0.0:
                    left -= 1
                while right <= K and c[right + K] == 0.0:
                    right += 1
                if left < -K or right > K:
                    continue
                left_val = c[left + K]
                right_val = c[right + K]
                dist = right - left
                if dist == 0:
                    continue
                weight_left = (right - i) / dist
                weight_right = (i - left) / dist
                c[i + K] = weight_left * left_val + weight_right * right_val

    coeffs = {i: c[i + K] for i in range(-K, K+1) if abs(c[i + K]) > 1e-12}
    return coeffs, c

# ---------- Elliptic gate ----------
class EllipticMobiusGate:
    def __init__(self, K, smooth=True):
        self.K = K
        self.coeffs, self.full_array = build_elliptic_coeffs(K, smooth)
        self.indices = np.array(sorted(self.coeffs.keys()))
        self.period = 2 * PI * ALPHA
        self.smooth = smooth

    def zeta(self, t):
        """ζ(t) = Σ C_i * exp(i * t * i / α)."""
        if len(self.indices) == 0:
            return 0.0 + 0.0j
        phases = t * self.indices / ALPHA
        vals = np.array([self.coeffs[i] for i in self.indices])
        return np.sum(vals * np.exp(1j * phases))

    def power_spectrum(self, t):
        return np.abs(self.zeta(t))**2

    def power_spectrum_series(self, N_points=1000):
        """Return t_vals and power_vals for N_points equally spaced t in [0, period]."""
        t_vals = np.linspace(0, self.period, N_points)
        power_vals = np.array([self.power_spectrum(t) for t in t_vals])
        return t_vals, power_vals

    def integrate_power_spectrum(self, N=1000):
        t_vals, power = self.power_spectrum_series(N)
        try:
            integral = np.trapezoid(power, t_vals)
        except AttributeError:
            integral = np.trapz(power, t_vals)
        avg = integral / self.period
        return integral, avg

    def basel_theoretical(self):
        return 2 * (6 / (PI * PI)) * self.K

    def basel_error(self, N=1000):
        _, avg = self.integrate_power_spectrum(N)
        theo = self.basel_theoretical()
        return abs(avg - theo) / theo if theo != 0 else 0.0

    def check_basel(self, tolerance=1e-2, N=1000):
        err = self.basel_error(N)
        return err < tolerance, err

    def fourier_analysis(self, N_points=1000):
        """
        Compute the FFT of the power spectrum and return frequencies and magnitudes.
        """
        t_vals, power = self.power_spectrum_series(N_points)
        dt = t_vals[1] - t_vals[0]
        # FFT (real)
        fft_vals = np.fft.rfft(power)
        freqs = np.fft.rfftfreq(N_points, dt)
        magnitudes = np.abs(fft_vals)
        return freqs, magnitudes, t_vals, power
ALPHA_USER = 0.3628
A = ALPHA / ALPHA_USER          # ≈ 6.511 (finite difference step)

# ---------- Safe entropy ----------
def safe_entropy(S, K, alpha=ALPHA):
    """Compute entropy with bounds to avoid overflow."""
    if S == 0:
        return 0.0
    p = abs(S) / K
    # Clamp p to avoid extreme values
    if p <= 0:
        return 0.0
    if p >= 1.0:
        return 0.0
    # For very small p, p*log(p) is very small; we can set a lower bound
    if p < 1e-15:
        return 0.0
    H = -alpha * p * math.log(p)
    # H is always ≤ alpha/e ≈ 0.869, but just in case
    if H < 0:
        H = 0.0
    if H > 1.0:
        H = 1.0
    return H

# ---------- Möbius sieve ----------
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

# ---------- Chip pipeline ----------
def tsp_route(signal):
    K = len(signal)
    angles = np.zeros(K)
    for i in range(K):
        x = math.sin(i * 7.0) + 0.1 * math.cos(i * 13.0)
        y = math.cos(i * 11.0) + 0.1 * math.sin(i * 17.0)
        angles[i] = math.atan2(y, x) + math.pi
    buckets = [[] for _ in range(360)]
    for i, a in enumerate(angles):
        idx = int((a / (2 * math.pi)) * 360) % 360
        buckets[idx].append(i)
    order = []
    for b in buckets:
        order.extend(b)
    return np.array(order)

def conv_exp_kernel(signal, alpha=ALPHA):
    K = len(signal)
    lam = math.exp(-alpha)
    f = np.zeros(K)
    f[0] = signal[0]
    for i in range(1, K):
        f[i] = signal[i] + lam * f[i-1]
    b = np.zeros(K)
    b[K-1] = signal[K-1]
    for i in range(K-2, -1, -1):
        b[i] = signal[i] + lam * b[i+1]
    conv_exp = (f + b - signal) / (1 - lam * lam)
    norm = 1.0 - math.exp(-alpha * (PI + E))
    conv = (1.0 - conv_exp) / norm
    return conv

def supertrace_and_mass(conv, K):
    S = 0.0
    for i, val in enumerate(conv):
        sign = 1 if (i % 2 == 0) else -1
        S += sign * abs(val)
    H = safe_entropy(S, K)
    m = abs(S) * math.exp(-H)
    return S, H, m

def chip_compress(signal, mu):
    K = len(signal)
    order = tsp_route(signal)
    signal_sorted = signal[order]
    conv = conv_exp_kernel(signal_sorted)
    S, H, m = supertrace_and_mass(conv, K)
    M = max(1, int(abs(S)))
    if M > K:
        M = K
    mag = np.abs(conv)
    idx_sorted = np.argsort(mag)[::-1]
    kept = []
    count = 0
    for idx in idx_sorted:
        n = idx + 1
        if mu[n] != 0:
            kept.append((idx, conv[idx]))
            count += 1
            if count >= M:
                break
    return kept, S, H, m, conv

# ---------- Finite derivative ----------
def finite_derivative(signal, step=A):
    K = len(signal)
    diff = np.zeros(K)
    diff[:-1] = (signal[1:] - signal[:-1]) / step
    return diff

# ---------- Code signature ----------
def code_to_signal(code_lines):
    K = len(code_lines)
    signal = np.zeros(K, dtype=float)
    for i, line in enumerate(code_lines):
        h = int(hashlib.md5(line.encode()).hexdigest()[:8], 16) % 10000
        signal[i] = h + 0.1 * len(line)
    return signal

def signature(code_lines, mu, apply_derivative=True):
    signal = code_to_signal(code_lines)
    if apply_derivative:
        signal = finite_derivative(signal, A)
    kept, S, H, m, _ = chip_compress(signal, mu)
    return kept, S, H, m

# ---------- Code checker ----------
class CodeChecker:
    def __init__(self, K_max=1000):
        self.K_max = K_max
        self.mu = mobius_sieve(K_max)
        self.reference = None

    def load_reference(self, code_lines):
        self.reference = signature(code_lines, self.mu, apply_derivative=True)

    def check(self, code_lines, tolerance=0.1):
        if self.reference is None:
            raise ValueError("No reference loaded.")
        kept_new, S_new, H_new, m_new = signature(code_lines, self.mu, apply_derivative=True)
        kept_ref, S_ref, H_ref, m_ref = self.reference

        # Compare invariants
        score = 1.0 - (abs(S_new - S_ref) / (abs(S_ref) + 1e-12) +
                       abs(H_new - H_ref) / (abs(H_ref) + 1e-12) +
                       abs(m_new - m_ref) / (abs(m_ref) + 1e-12)) / 3.0
        # Compare kept coefficients via inverse score
        K = self.K_max
        recon_ref = np.zeros(K, dtype=complex)
        for idx, val in kept_ref:
            recon_ref[idx] = val
        recon_new = np.zeros(K, dtype=complex)
        for idx, val in kept_new:
            recon_new[idx] = val
        # Use simple correlation (cosine similarity of magnitudes)
        mag_ref = np.abs(recon_ref)
        mag_new = np.abs(recon_new)
        dot = np.dot(mag_ref, mag_new)
        norm_ref = np.linalg.norm(mag_ref)
        norm_new = np.linalg.norm(mag_new)
        cos_sim = dot / (norm_ref * norm_new + 1e-12) if norm_ref > 0 and norm_new > 0 else 0.5
        # Combine: 70% invariants, 30% structure
        similarity = 0.7 * score + 0.3 * cos_sim
        is_ok = similarity > (1.0 - tolerance)
        return is_ok, similarity, S_new, H_new, m_new
# ---------- Matrix trace (asymmetric bug gate) ----------
def matrix_trace_signal(signal, i_exp=2):
    """
    Compute the matrix trace from pairs of consecutive values:
    tr = x^(i-1) + y^(i-1), where x = signal[i], y = signal[i+1].
    Sum over all pairs.
    """
    K = len(signal)
    if K < 2:
        return 0.0
    total = 0.0
    for i in range(0, K-1, 2):
        x = abs(signal[i]) + 1e-12
        y = abs(signal[i+1]) + 1e-12
        total += x ** (i_exp - 1) + y ** (i_exp - 1)
    # If odd length, use last element paired with itself
    if K % 2 == 1:
        x = abs(signal[-1]) + 1e-12
        total += 2 * (x ** (i_exp - 1))
    return total

# ---------- Finite derivative ----------
def finite_derivative(signal, step=A):
    K = len(signal)
    diff = np.zeros(K)
    diff[:-1] = (signal[1:] - signal[:-1]) / step
    return diff

# ---------- Code signature generation ----------
def code_to_signal(code_lines):
    K = len(code_lines)
    signal = np.zeros(K, dtype=float)
    for i, line in enumerate(code_lines):
        h = int(hashlib.md5(line.encode()).hexdigest()[:8], 16) % 10000
        signal[i] = h + 0.1 * len(line)
    return signal

def signature(code_lines, mu, apply_derivative=True):
    signal = code_to_signal(code_lines)
    if apply_derivative:
        signal = finite_derivative(signal, A)
    kept, S, H, m, _ = chip_compress(signal, mu)
    return kept, S, H, m, signal

# ---------- Dual‑stage code checker ----------
class DualCodeChecker:
    def __init__(self, K_max=1000):
        self.K_max = K_max
        self.mu = mobius_sieve(K_max)
        self.good_ref = None          # (kept, S, H, m)
        self.bug_trace = None         # matrix trace of buggy code
        self.bug_tolerance = 1e-3

    def load_good_reference(self, code_lines):
        """Store the reference for good (symmetric) code."""
        kept, S, H, m, _ = signature(code_lines, self.mu, apply_derivative=True)
        self.good_ref = (kept, S, H, m)

    def load_bug_reference(self, code_lines):
        """Store the matrix trace for buggy (asymmetric) code."""
        signal = code_to_signal(code_lines)
        signal = finite_derivative(signal, A)
        self.bug_trace = matrix_trace_signal(signal, i_exp=2)

    def check(self, code_lines):
        """
        Returns: (is_ok, similarity, S, H, m, trace_diff)
        """
        # 1. Compute signal and its matrix trace
        signal = code_to_signal(code_lines)
        signal_deriv = finite_derivative(signal, A)
        trace = matrix_trace_signal(signal_deriv, i_exp=2)

        # 2. Asymmetric bug check (if bug_trace is set)
        if self.bug_trace is not None:
            trace_diff = abs(trace - self.bug_trace) / (abs(self.bug_trace) + 1e-12)
            if trace_diff > self.bug_tolerance:
                # Likely a bug: return early with low similarity
                return False, 0.0, 0.0, 0.0, 0.0, trace_diff

        # 3. Symmetric good check
        if self.good_ref is None:
            raise ValueError("No good reference loaded.")
        kept_ref, S_ref, H_ref, m_ref = self.good_ref
        kept_new, S_new, H_new, m_new, _ = signature(code_lines, self.mu, apply_derivative=True)

        # Compare invariants
        score = 1.0 - (abs(S_new - S_ref) / (abs(S_ref) + 1e-12) +
                       abs(H_new - H_ref) / (abs(H_ref) + 1e-12) +
                       abs(m_new - m_ref) / (abs(m_ref) + 1e-12)) / 3.0

        # Compare kept coefficients via cosine similarity
        K = self.K_max
        recon_ref = np.zeros(K, dtype=complex)
        for idx, val in kept_ref:
            recon_ref[idx] = val
        recon_new = np.zeros(K, dtype=complex)
        for idx, val in kept_new:
            recon_new[idx] = val
        mag_ref = np.abs(recon_ref)
        mag_new = np.abs(recon_new)
        dot = np.dot(mag_ref, mag_new)
        norm_ref = np.linalg.norm(mag_ref)
        norm_new = np.linalg.norm(mag_new)
        cos_sim = dot / (norm_ref * norm_new + 1e-12) if norm_ref > 0 and norm_new > 0 else 0.5

        similarity = 0.7 * score + 0.3 * cos_sim
        is_ok = similarity > 0.9  # threshold
        return is_ok, similarity, S_new, H_new, m_new, trace_diff
class InferenceBuffer:
    """
    An O(K log K) inference buffer that stores a signal and provides
    operations: apply gate, compress, Collatz step, and inverse score
    with a target. All operations reuse pre‑allocated buffers.
    """

    def __init__(self, max_K=1000):
        self.max_K = max_K
        # Core processors (buffers allocated once)
        self.processor = ChipProcessor(max_K=max_K)
        self.gate = ChipLogicGate(max_K=max_K)
        # Current signal storage (as numpy array)
        self.signal = np.zeros(max_K, dtype=float)
        self.length = 0
        # Output cache (last result)
        self.last_kept = []
        self.last_S = 0.0
        self.last_H = 0.0
        self.last_m = 0.0
        self.last_conv = None

    def set_signal(self, signal):
        """Set the current signal from a 1D array (copy into buffer)."""
        K = len(signal)
        if K > self.max_K:
            raise ValueError(f"Signal length {K} exceeds max_K {self.max_K}")
        self.signal[:K] = signal
        self.length = K

    def get_signal(self):
        """Return a copy of the current signal (truncated to length)."""
        return self.signal[:self.length].copy()

    def apply_gate(self, gate_name):
        """
        Apply a logic gate (log, exp, sin, cos, trace) to the current signal,
        then compress it. Updates the buffer with the compressed output.
        Returns (kept, S, H, m).
        """
        sig = self.signal[:self.length]
        kept, S, H_ent, m, conv = self.gate.apply_function(sig, gate_name)
        # Store output
        self.last_kept = kept
        self.last_S = S
        self.last_H = H_ent
        self.last_m = m
        self.last_conv = conv
        # Update buffer with reconstructed signal (zero‑padded kept coefficients)
        recon = np.zeros(self.length, dtype=complex)
        for idx, val in kept:
            if idx < self.length:
                recon[idx] = val
        self.signal[:self.length] = np.real(recon)
        return kept, S, H_ent, m

    def compress(self):
        """
        Run chip compression on the current signal (without logic gate).
        Updates buffer with the compressed signal.
        Returns (kept, S, H, m).
        """
        sig = self.signal[:self.length]
        kept, S, H_ent, m, conv = self.processor.process(sig)
        self.last_kept = kept
        self.last_S = S
        self.last_H = H_ent
        self.last_m = m
        self.last_conv = conv
        recon = np.zeros(self.length, dtype=complex)
        for idx, val in kept:
            if idx < self.length:
                recon[idx] = val
        self.signal[:self.length] = np.real(recon)
        return kept, S, H_ent, m

    def collatz(self):
        """
        Apply a Collatz step to the indices of the current signal.
        Values are carried over; indices become odd after (3n+1)/2^k.
        The memory is compressed to odd square‑free indices only.
        Updates buffer with the new signal.
        Returns list of (new_index, value) pairs.
        """
        # Pack current signal into a MobiusCollatzMemory at odd square‑free indices
        mem = MobiusCollatzMemory(max_index=self.length*3+1, use_square_free=True)
        idx = 1
        count = 0
        while count < self.length and idx <= self.length*3+1:
            if mem._valid_index(idx):
                if count < len(self.signal):
                    mem.write(idx, self.signal[count])
                    count += 1
            idx += 2
        # Apply Collatz step (value_transform = None keeps values)
        mem.collatz_step()
        # Extract new signal sorted by index
        items = sorted(mem.data.items())
        new_signal = np.array([val for _, val in items], dtype=float)
        K_new = len(new_signal)
        if K_new > self.max_K:
            raise ValueError(f"Collatz expanded to {K_new} > max_K")
        self.signal[:K_new] = new_signal
        self.length = K_new
        return items

    def inverse_score_with(self, target_signal):
        """
        Compute the inverse score (O(K log K)) between the current signal
        and a target signal (both arrays of same length).
        Returns float in [0,1].
        """
        # Align lengths: take the shorter length
        K = min(self.length, len(target_signal))
        if K < 2:
            return 0.0
        curr = self.signal[:K]
        targ = target_signal[:K]
        return inverse_score(curr.tolist(), targ.tolist())

    def print_summary(self):
        """Print current state and last output."""
        print(f"Signal length: {self.length}")
        if hasattr(self, 'last_kept'):
            print(f"Compressed coefficients: {len(self.last_kept)}")
            print(f"Supertrace S = {self.last_S:.6f}")
            print(f"Entropy H = {self.last_H:.6f}")
            print(f"Mass m = {self.last_m:.6f}")
            if self.last_kept:
                print("First 5 kept (index, value):")
                for idx, val in self.last_kept[:5]:
                    print(f"  {idx}: {val:.6f}")

class MobiusHarness:
    """
    Full harness for the Möbius pipeline.
    """

    def __init__(self, K=40, M=64, smooth=True, max_buffer=1000):
        self.K = K
        self.M = M
        self.smooth = smooth
        self.max_buffer = max_buffer

        # Core components
        self.gate = EllipticMobiusGate(K, smooth=smooth)
        self.buffer = InferenceBuffer(max_K=max_buffer)
        self.compiler = MobiusCrossCompiler(max_K=max_buffer)
        self.checker = DualCodeChecker(K_max=max_buffer)

        # Task storage
        self.tasks = []          # list of t values
        self.task_results = []   # list of (t, power_spectrum)
        self.merged_signal = None

    def add_task(self, t):
        """Add a task (a t point) to the harness."""
        self.tasks.append(t)

    def add_tasks(self, t_list):
        """Add multiple tasks."""
        self.tasks.extend(t_list)

    def process_tasks(self):
        """
        Full pipeline:
          1. Sort tasks by angle (using tsp_route from chip-g).
          2. For each t, compute power spectrum |ζ(t)|².
          3. Merge all power spectra into a single signal using the buffer.
          4. Compile the merged signal.
          5. Check for coding errors.
        """
        if not self.tasks:
            print("No tasks to process.")
            return

        # 1. Angle sorting (TSP routing) – treat the index as a point.
        # We'll use the tsp_route function on the list of t values.
        # For simplicity, we use the list of t values directly.
        # In practice, you might map each t to a 2D point (cos(t), sin(t)).
        K = len(self.tasks)
        # Convert tasks to 2D points for angle sorting
        points = np.array([[math.cos(t), math.sin(t)] for t in self.tasks])
        angles = np.arctan2(points[:,1], points[:,0])
        order = np.argsort(angles)
        sorted_tasks = [self.tasks[i] for i in order]

        # 2. Compute power spectra for each sorted task
        power_spectra = []
        for t in sorted_tasks:
            power = self.gate.power_spectrum(t)
            power_spectra.append(power)

        # 3. Merge via buffer: we can either average or take the sum.
        # Here we sum them to get a combined signal.
        combined_signal = np.sum(power_spectra, axis=0)
        # Trim to max_buffer length if needed
        if len(combined_signal) > self.max_buffer:
            combined_signal = combined_signal[:self.max_buffer]

        # 4. Set the buffer signal and compress
        self.buffer.set_signal(combined_signal)
        kept, S, H, m = self.buffer.compress()
        self.merged_signal = (kept, S, H, m)

        # 5. Compile into Möbius memory (using the compiler)
        # Convert the compressed signal to a list of lines (as code)
        # For demonstration, we treat the signal as code lines.
        code_lines = [str(val) for val in combined_signal[:100]]  # just a sample
        compiled, _, _, _ = self.compiler.compile_code(code_lines, gate_name='log', compress=True)
        print(f"Compiled {len(compiled)} coefficients.")

        # 6. Check for coding errors using the dual checker.
        # We'll use the combined signal as a "code" input.
        # Store a good reference (first run) and then check subsequent runs.
        if not hasattr(self, 'good_ref_loaded'):
            self.checker.load_good_reference(code_lines)
            self.checker.load_bug_reference(code_lines)  # placeholder
            self.good_ref_loaded = True
            print("Reference loaded.")
        else:
            ok, sim, S_new, H_new, m_new, diff = self.checker.check(code_lines)
            print(f"Check result: OK={ok}, similarity={sim:.4f}, S={S_new:.4f}, H={H_new:.4f}, m={m_new:.4f}, diff={diff:.4f}")

        return kept, S, H, m

    def run_pipeline(self, t_list):
        """Convenience method: add tasks, process, return results."""
        self.add_tasks(t_list)
        return self.process_tasks()

# ---------- Example usage ----------
def demo():
    harness = MobiusHarness(K=40, M=64, smooth=True, max_buffer=1000)
    # Generate tasks: t values from 0 to 2*period
    period = harness.gate.period
    tasks = np.linspace(0, period, 50)
    harness.run_pipeline(tasks)

if __name__ == "__main__":
    demo()