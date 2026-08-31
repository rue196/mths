#!/usr/bin/env python3
"""
color_mobius_gate.py

RGB color processing via Möbius harmonic transform.
Takes an RGB image or color values, applies the chip pipeline,
and returns a fingerprint or a transformed color.
"""

import math
import numpy as np
import hashlib
import struct
import matplotlib.pyplot as plt

PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)          # ≈ 2.362
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

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)
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

class ColorMobiusGate:
    """
    Möbius gate for RGB color processing.
    """
    def __init__(self, K=256, logic_gate='log', use_elliptic=True):
        self.K = K
        self.logic_gate = logic_gate
        self.use_elliptic = use_elliptic
        self.mu = mobius_sieve(K)
        self.processor = ChipProcessor(max_K=K)
        self.gate = ChipLogicGate(max_K=K)

    def _rgb_to_signal(self, rgb_data, flatten=True):
        """
        Convert RGB data to a 1D signal.
        rgb_data: either:
          - a list of (R,G,B) triples
          - a numpy array of shape (H, W, 3)
          - a single triple (R,G,B)
        If flatten=True, the signal is flattened to a 1D array.
        Returns: 1D numpy array of length K (padded/truncated).
        """
        if isinstance(rgb_data, tuple) and len(rgb_data) == 3:
            # single pixel
            signal = np.array(rgb_data, dtype=float)
        elif isinstance(rgb_data, list) and all(isinstance(x, tuple) for x in rgb_data):
            # list of triples
            signal = np.array([c for triple in rgb_data for c in triple], dtype=float)
        elif isinstance(rgb_data, np.ndarray):
            if rgb_data.ndim == 3 and rgb_data.shape[2] == 3:
                # image: flatten
                signal = rgb_data.flatten().astype(float)
            elif rgb_data.ndim == 1:
                signal = rgb_data.astype(float)
            else:
                raise ValueError("Unsupported numpy array shape")
        else:
            raise ValueError("Unsupported RGB data format")

        # Pad or truncate to self.K
        if len(signal) > self.K:
            signal = signal[:self.K]
        elif len(signal) < self.K:
            signal = np.pad(signal, (0, self.K - len(signal)), constant_values=0)
        return signal

    def _apply_elliptic_permutation(self, signal):
        """Reorder signal by elliptic angle (TSP routing)."""
        # Use the deterministic pseudo-angle from tsp_route
        # For a 1D signal, we treat index as the phase
        K = len(signal)
        angles = np.zeros(K)
        for i in range(K):
            x = math.sin(i * 7.0) + 0.1 * math.cos(i * 13.0)
            y = math.cos(i * 11.0) + 0.1 * math.sin(i * 17.0)
            angles[i] = math.atan2(y, x) + math.pi
        order = np.argsort(angles)
        return signal[order]

    def _apply_logic_gate(self, signal):
        """Apply the configured logic gate."""
        if self.logic_gate == 'log':
            return np.log(np.maximum(np.abs(signal), 1e-12))
        elif self.logic_gate == 'exp':
            return np.exp(signal)
        elif self.logic_gate == 'sin':
            return np.sin(signal)
        elif self.logic_gate == 'cos':
            return np.cos(signal)
        else:
            return signal

    def _chip_compress(self, signal):
        """Run the chip pipeline (convolution + supertrace + Möbius compression)."""
        # We can use the existing chip_pipeline function
        kept, S, H, m, conv = chip_pipeline(signal, processor=self.processor)
        return kept, S, H, m, conv

    def process(self, rgb_data):
        """
        Full pipeline: RGB -> signal -> elliptic permutation -> logic gate -> chip compression.
        Returns: (fingerprint, metadata)
        """
        # 1. Convert to signal
        signal = self._rgb_to_signal(rgb_data)
        # 2. Elliptic permutation (if enabled)
        if self.use_elliptic:
            signal = self._apply_elliptic_permutation(signal)
        # 3. Logic gate
        signal = self._apply_logic_gate(signal)
        # 4. Chip compression
        kept, S, H, m, conv = self._chip_compress(signal)
        # 5. Pack and hash to fingerprint
        data = bytearray()
        for idx, val in kept:
            data.extend(struct.pack('>Hd', idx, val))
        data.extend(struct.pack('>ddd', S, H, m))
        fingerprint = hashlib.sha256(data).hexdigest()
        metadata = {
            'S': S,
            'H': H,
            'm': m,
            'num_kept': len(kept),
            'conv': conv,
            'kept': kept
        }
        return fingerprint, metadata

    def transform_color(self, rgb):
        """
        Process a single RGB triple and return a transformed RGB triple.
        The transformation applies the chip pipeline and then maps the result back to RGB.
        """
        if not (isinstance(rgb, (tuple, list)) and len(rgb) == 3):
            raise ValueError("Input must be (R,G,B)")
        signal = self._rgb_to_signal(rgb)   # length K, padded with zeros
        if self.use_elliptic:
            signal = self._apply_elliptic_permutation(signal)
        signal = self._apply_logic_gate(signal)
        kept, S, H, m, conv = self._chip_compress(signal)
        # Reconstruct the convolved signal (zero-padded)
        recon = np.zeros(self.K, dtype=complex)
        for idx, val in kept:
            recon[idx] = val
        # Take the real part and map to RGB: we take the first 3 values, clamp and scale to [0,255]
        out = np.real(recon)[:3]
        out = np.clip(out, 0, 255).astype(np.uint8)
        return tuple(out)

    def plot_spectrum(self, rgb_data):
        """Plot the power spectrum of the transformed signal."""
        _, meta = self.process(rgb_data)
        plt.figure(figsize=(10,4))
        plt.plot(np.abs(meta['conv']), label='|conv|')
        plt.xlabel('Index')
        plt.ylabel('Magnitude')
        plt.title(f'Power spectrum (kept {meta["num_kept"]} coefficients)')
        plt.legend()
        plt.grid(True)
        plt.show()

# ---------- Demo ----------
def demo():
    gate = ColorMobiusGate(K=256, logic_gate='log', use_elliptic=True)

    # Example 1: Single pixel
    rgb = (128, 64, 200)
    fp, meta = gate.process(rgb)
    print(f"RGB: {rgb}")
    print(f"Fingerprint: {fp}")
    print(f"  S={meta['S']:.4f}, H={meta['H']:.4f}, m={meta['m']:.4f}, kept={meta['num_kept']}")

    # Example 2: Transform a color
    transformed = gate.transform_color(rgb)
    print(f"Transformed RGB: {transformed}")

    # Example 3: Generate a color image (synthetic gradient)
    H, W = 50, 50
    img = np.zeros((H, W, 3), dtype=np.uint8)
    for i in range(H):
        for j in range(W):
            img[i, j] = [int(255*i/H), int(255*j/W), int(255*(i+j)/(H+W))]
    fp_img, meta_img = gate.process(img)
    print(f"Image fingerprint: {fp_img[:16]}...")
    print(f"Image: S={meta_img['S']:.4f}, H={meta_img['H']:.4f}, m={meta_img['m']:.4f}, kept={meta_img['num_kept']}")

if __name__ == "__main__":
    demo()
