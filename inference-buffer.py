import math
import numpy as np
from random_access_colla_mobius import MobiusCollatzMemory
from ML import inverse_score
from Oklogk import mu_convolution_H


# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)
NORM = 1.0 - math.exp(-ALPHA * (PI + E))

class ChipLogicGate:
    """
    A bounded Möbius logic gate that applies a function to a signal,
    then runs the chip pipeline (convolution + supertrace + Möbius compression).
    The output is a compressed representation of the transformed signal.
    """

    def __init__(self, max_K=1000):
        self.processor = ChipProcessor(max_K=max_K)
    
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

# ---------- Simple demonstration ----------
def demo():
    buf = InferenceBuffer(max_K=256)

    # Load a test signal: harmonic convolution F(n) = (μ*H)(n)
    from Oklogk import mu_convolution_H
    F, _, _ = mu_convolution_H(64)
    buf.set_signal(F[1:])   # n=1..64
    print("Initial signal (first 10):", buf.get_signal()[:10])

    # Apply a logic gate
    buf.apply_gate('log')
    print("\nAfter log gate:")
    buf.print_summary()

    # Compress
    buf.compress()
    print("\nAfter compression:")
    buf.print_summary()

    # Collatz step
    items = buf.collatz()
    print(f"\nAfter Collatz: {len(items)} coefficients")
    print("New signal (first 10):", buf.get_signal()[:10])

    # Compute inverse score with a target (e.g., the original signal)
    target = F[1:]
    score = buf.inverse_score_with(target)
    print(f"\nInverse score with original signal: {score:.4f}")

if __name__ == "__main__":
    demo()