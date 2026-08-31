import math
import numpy as np
from random_access_colla_mobius import MobiusCollatzMemory

import random

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)
NORM = 1.0 - math.exp(-ALPHA * (PI + E))

# ---------- 1. Address generation (elliptic‑curve points simulated) ----------
def generate_addresses(K, seed=42):
    random.seed(seed)
    # Random (x,y) in [0.5, 5.0] to avoid singularities
    addresses = [(random.uniform(0.5, 5.0), random.uniform(0.5, 5.0)) for _ in range(K)]
    return addresses

# ---------- 2. Matrix trace (scalar) ----------
def matrix_trace(x, y, i=2):
    """
    Trace of M = [[x^(i-1), x^(-1)*y^i],
                  [y^(-1)*x^i, y^(i-1)]]
    tr = x^(i-1) + y^(i-1)
    """
    return x ** (i-1) + y ** (i-1)

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


def boot_mobius_sequence(K_init=16, M_max=64, expansion_steps=3, seed=42):
    """
    Boot sequence:
    1. Generate initial coefficients at odd square‑free indices (O(K_init) time).
    2. For each expansion step:
       - Apply a logic gate (e.g., log or exp) to transform the signal.
       - Run the chip pipeline (convolution + supertrace + Möbius compression)
         to produce a compressed representation of size M (≤ M_max).
       - Store the compressed coefficients back into the memory.
    3. The memory size is always bounded by M_max (O(M) memory).
    Expansion cost per step is O(K log K) due to sorting in the chip pipeline.
    """
    # ---- Step 0: Initialisation ----
    # Generate a set of elliptic addresses (just for bootstrapping)
    K_initial = K_init
    addresses = generate_addresses(K_initial, seed=seed)
    # Compute matrix traces to get initial coefficients (O(K) time)
    coeffs_initial = [matrix_trace(x, y, i=2) for x, y in addresses]
    # Create a Möbius memory: only odd square‑free indices allowed.
    # We'll map the first K_initial odd square‑free indices.
    mem = MobiusCollatzMemory(max_index=1000, use_square_free=True)
    # Get the first K_initial odd square‑free indices
    idx = 1
    count = 0
    while count < K_initial:
        if mem._valid_index(idx):
            mem.write(idx, coeffs_initial[count])
            count += 1
        idx += 2  # only odd
    print(f"Boot: initialised {len(mem)} coefficients.")
    
    # ---- Expansion loop ----
    # Create a chip logic gate (uses the chip pipeline)
    gate = ChipLogicGate(max_K=1000)
    # We'll also use a chip processor for manual compression if needed
    processor = ChipProcessor(max_K=1000)
    
    for step in range(expansion_steps):
        # 1. Extract the current signal as a list of values (sorted by index)
        items = sorted(mem.data.items())
        signal = np.array([val for _, val in items], dtype=float)
        print(f"  Step {step+1}: signal length = {len(signal)}")
        
        # 2. Apply a logic gate to transform the signal (e.g., bounded log)
        # We can alternate gates for variety.
        if step % 2 == 0:
            kept, S, H, m, conv = gate.bounded_log(signal)
            print(f"    Applied log transform: S={S:.4f}, H={H:.4f}, m={m:.4f}")
        else:
            kept, S, H, m, conv = gate.bounded_exp(signal)
            print(f"    Applied exp transform: S={S:.4f}, H={H:.4f}, m={m:.4f}")
        
        # 3. The chip pipeline returns `kept` as a list of (index, value) pairs.
        # These indices are in the sorted order (after TSP routing).
        # We need to map them back to the original Möbius memory indices.
        # The `kept` indices are 0-based positions in the sorted signal.
        # We have the original indices from `items`.
        # Create a new memory with the compressed coefficients.
        new_mem = MobiusCollatzMemory(max_index=1000, use_square_free=True)
        for pos, val in kept:
            if pos < len(items):
                orig_idx = items[pos][0]  # the actual odd square‑free index
                if new_mem._valid_index(orig_idx):
                    new_mem.write(orig_idx, val)
        # Replace the old memory with the compressed one
        mem = new_mem
        print(f"    Compressed to {len(mem)} coefficients (≤ M_max={M_max})")
        # If we have reached the desired max, we can break
        if len(mem) >= M_max:
            print(f"    Reached memory limit {M_max}.")
            break
    
    return mem

# ---------- Main demonstration ----------
def main():
    print("=== Möbius Memory Boot Sequence ===\n")
    # Boot with small K_init, expand to M_max=64, in 3 steps
    mem = boot_mobius_sequence(K_init=16, M_max=64, expansion_steps=5, seed=42)
    
    print(f"\nFinal memory contains {len(mem)} coefficients.")
    # Show the first few indices and values
    print("First 10 stored entries (idx, value):")
    for i, (idx, val) in enumerate(sorted(mem.data.items())):
        if i >= 10:
            break
        print(f"  {idx}: {val:.6f}")
    
    # Also compute the supertrace of the final memory
    from random_access_colla_mobius import supertrace_from_coeffs
    S_final = supertrace_from_coeffs(mem.data)
    print(f"Final supertrace S = {S_final:.6f}")

if __name__ == "__main__":
    main()