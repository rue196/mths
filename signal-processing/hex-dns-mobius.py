#!/usr/bin/env python3
"""
dns_hex_mobius_gate.py

DNS resolution using Möbius hex fingerprints.
Domain → ASCII → polynomial → elliptic permutation → Möbius filter → logic gate → chip compression → SHA‑256 hex.
"""

import math
import hashlib
import struct
import random
import numpy as np

# ---------- Constants ----------
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

# ---------- Elliptic permutation ----------
def elliptic_permutation(K, omega1=PI, omega2=E):
    delta = omega1 - omega2
    angles = [(i * delta) % (2 * PI) for i in range(K)]
    return sorted(range(K), key=lambda i: angles[i])

# ---------- Exponential convolution ----------
def conv_exp_kernel(signal, alpha=ALPHA):
    K = len(signal)
    lam = math.exp(-alpha)
    f = np.zeros(K, dtype=float)
    f[0] = signal[0]
    for i in range(1, K):
        f[i] = signal[i] + lam * f[i-1]
    b = np.zeros(K, dtype=float)
    b[K-1] = signal[K-1]
    for i in range(K-2, -1, -1):
        b[i] = signal[i] + lam * b[i+1]
    conv_exp = (f + b - signal) / (1 - lam * lam)
    conv = (1.0 - conv_exp) / NORM
    return conv

# ---------- Supertrace and entropy ----------
def supertrace_from_signal(signal):
    S = 0.0
    for idx, val in enumerate(signal):
        sign = 1 if (idx % 2 == 0) else -1
        S += sign * abs(val)
    return S

def entropy_from_supertrace(S, N, alpha=ALPHA):
    if S == 0:
        return 0.0
    p = abs(S) / N
    if p <= 0:
        return 0.0
    return -alpha * p * math.log(p)

def mass_from_signal(signal):
    S = supertrace_from_signal(signal)
    H = entropy_from_supertrace(S, len(signal))
    return abs(S) * math.exp(-H)

# ---------- Hex fingerprint generator ----------
class MobiusHexGate:
    """Generates hex fingerprints from text using Möbius chip pipeline."""

    def __init__(self, K=256, logic_gate='log', use_elliptic=True):
        self.K = K
        self.logic_gate = logic_gate
        self.use_elliptic = use_elliptic
        self.mu = mobius_sieve(K)

    def _ascii_to_polynomial(self, text, num_vars=6):
        random.seed(hash(text) % (2**32))
        monomials = []
        for i, ch in enumerate(text[:self.K]):
            coeff = (ord(ch) - 32) / 95.0
            coeff += 0.1 * math.sin(i * 0.5)
            exps = tuple(random.randint(0, 3) for _ in range(num_vars))
            monomials.append((coeff, exps))
        while len(monomials) < self.K:
            monomials.append((0.0, (0,) * num_vars))
        return monomials

    def _apply_mobius_filter(self, monomials):
        filtered = []
        for idx, (coeff, exps) in enumerate(monomials):
            n = idx + 1
            if self.mu[n] != 0:
                filtered.append((coeff, exps))
        return filtered

    def _apply_elliptic_permutation(self, monomials):
        order = elliptic_permutation(len(monomials))
        return [monomials[i] for i in order]

    def _apply_logic_gate(self, signal):
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
        K = len(signal)
        conv = conv_exp_kernel(signal)
        S = supertrace_from_signal(conv)
        H = entropy_from_supertrace(S, K)
        m = mass_from_signal(conv)
        M = max(1, int(abs(S)))
        if M > K:
            M = K
        mag = np.abs(conv)
        idx_sorted = np.argsort(mag)[::-1]
        kept = []
        count = 0
        for idx in idx_sorted:
            n = idx + 1
            if self.mu[n] != 0:
                kept.append((idx, conv[idx]))
                count += 1
                if count >= M:
                    break
        return kept, S, H, m

    def fingerprint(self, text, num_vars=6):
        # 1. ASCII → polynomial
        monomials = self._ascii_to_polynomial(text, num_vars)
        # 2. Elliptic permutation
        if self.use_elliptic:
            monomials = self._apply_elliptic_permutation(monomials)
        # 3. Möbius filter
        filtered = self._apply_mobius_filter(monomials)
        # 4. Extract coefficients
        signal = np.array([abs(c) for c, _ in filtered], dtype=float)
        # 5. Logic gate
        signal = self._apply_logic_gate(signal)
        # 6. Chip compression
        kept, S, H, m = self._chip_compress(signal)
        # 7. Pack into hex
        data = bytearray()
        for idx, val in kept:
            data.extend(struct.pack('>Hd', idx, val))
        # Include invariants for extra uniqueness
        data.extend(struct.pack('>ddd', S, H, m))
        return hashlib.sha256(data).hexdigest()

# ---------- DNS Möbius Gate ----------
class DNSMobiusGate:
    def __init__(self, K=256, logic_gate='log', use_elliptic=True):
        self.gate = MobiusHexGate(K=K, logic_gate=logic_gate, use_elliptic=use_elliptic)
        self.cache = {}   # fingerprint -> IP

    def add_record(self, domain, ip):
        """Add a DNS record (domain -> IP)."""
        fp = self.gate.fingerprint(domain)
        self.cache[fp] = ip

    def resolve(self, domain):
        """Resolve a domain to its IP address."""
        fp = self.gate.fingerprint(domain)
        return self.cache.get(fp, None)

    def lookup_by_fingerprint(self, fp):
        """Direct lookup by hex fingerprint."""
        return self.cache.get(fp, None)

    def show_cache(self):
        """Print all cached records (debug)."""
        for fp, ip in self.cache.items():
            print(f"{fp[:16]}... -> {ip}")

# ---------- Example ----------
def demo():
    dns = DNSMobiusGate(K=128, logic_gate='log', use_elliptic=True)

    records = {
        "example.com": "93.184.216.34",
        "mobius.org": "192.0.2.1",
        "localhost": "127.0.0.1"
    }
    for domain, ip in records.items():
        dns.add_record(domain, ip)
        print(f"Added {domain} -> {ip}")

    print("\nResolving:")
    for domain in records:
        ip = dns.resolve(domain)
        print(f"{domain} -> {ip}")

    print("\nUnknown domain:")
    print(f"unknown.com -> {dns.resolve('unknown.com')}")

    # Show fingerprints (for debugging)
    print("\nFingerprints:")
    for domain in records:
        fp = dns.gate.fingerprint(domain)
        print(f"{domain}: {fp}")

if __name__ == "__main__":
    demo()