import math
import numpy as np
from Oklogk import mu_convolution_H
from random_access_colla_mobius import MobiusCollatzMemory
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




# ---------- Test ----------
def main():
    # Create a reusable processor
    proc = ChipProcessor(max_K=500)
    K = 200
    signal = np.array([math.log(i+1) for i in range(K)])
    kept, S, H, m, conv = chip_pipeline(signal, processor=proc)
    print(f"Signal length: {K}")
    print(f"Supertrace S = {S:.4f}, Entropy H = {H:.4f}, Mass m = {m:.4f}")
    print(f"Kept {len(kept)} coefficients (ratio {len(kept)/K:.3f})")
    # Reconstruct
    recon = np.zeros(K, dtype=complex)
    for idx, val in kept:
        recon[idx] = val
    error = np.linalg.norm(conv - recon) / np.linalg.norm(conv)
    print(f"Reconstruction relative L2 error = {error:.4e}")
# ---------- Example usage ----------
def main():
    # Create a logic gate processor
    gate = ChipLogicGate(max_K=500)

    # Generate a test signal: a smooth curve (harmonic numbers)
    K = 200
    signal = np.array([math.log(i+1) for i in range(K)])

    print("Original signal (first 10):", signal[:10])

    # Apply bounded logarithm (already log, so same)
    kept_log, S_log, H_log, m_log, conv_log = gate.bounded_log(signal)
    print(f"\nLog transform: S={S_log:.4f}, H={H_log:.4f}, m={m_log:.4f}, kept {len(kept_log)} coeffs")

    # Apply exp to the signal
    kept_exp, S_exp, H_exp, m_exp, conv_exp = gate.bounded_exp(signal)
    print(f"Exp transform: S={S_exp:.4f}, H={H_exp:.4f}, m={m_exp:.4f}, kept {len(kept_exp)} coeffs")

    # Apply matrix trace on a 4-element signal
    mat_signal = np.array([1.0, 0.5, 0.3, 0.7], dtype=complex)
    kept_trace, S_trace, H_trace, m_trace, conv_trace = gate.apply_function(mat_signal, 'trace')
    print(f"\nMatrix trace: S={S_trace:.4f}, H={H_trace:.4f}, m={m_trace:.4f}, kept {len(kept_trace)} coeffs")


class UILogicGate:
    """
    A user‑friendly logic gate that applies operations and compression
    to a signal, with buffering for efficiency.
    """

    def __init__(self, max_K=1000):
        self.max_K = max_K
        self.processor = ChipProcessor(max_K=max_K)
        self.gate = ChipLogicGate(max_K=max_K)
        # buffers
        self.signal_buffer = np.zeros(max_K, dtype=float)
        self.current_len = 0
        # output storage
        self.output_kept = []
        self.output_S = 0.0
        self.output_H = 0.0
        self.output_m = 0.0
        self.output_conv = None

    def set_signal(self, signal):
        """Set the input signal (1D array)."""
        K = len(signal)
        if K > self.max_K:
            raise ValueError(f"Signal length {K} exceeds max_K {self.max_K}")
        self.signal_buffer[:K] = signal
        self.current_len = K

    def load_harmonic_convolution(self, K):
        """Load F(n) = (μ * H)(n) as the signal (n=1..K)."""
        F, mu, H = mu_convolution_H(K)
        self.set_signal(F[1:])   # F[0] is 0, take n=1..K

    def apply_gate(self, gate_name):
        """
        Apply a logic gate to the current signal.
        Available names: 'log', 'exp', 'sin', 'cos', 'trace'.
        Returns the compressed output (kept coefficients, S, H, m, conv).
        """
        signal = self.signal_buffer[:self.current_len]
        kept, S, H_ent, m, conv = self.gate.apply_function(signal, gate_name)
        self.output_kept = kept
        self.output_S = S
        self.output_H = H_ent
        self.output_m = m
        self.output_conv = conv
        # Optionally update the signal buffer with the reconstructed signal
        # (using the kept coefficients zero‑padded) for chaining.
        recon = np.zeros(self.current_len, dtype=complex)
        for idx, val in kept:
            recon[idx] = val
        self.signal_buffer[:self.current_len] = np.real(recon)
        return kept, S, H_ent, m, conv

    def apply_collatz(self):
        """
        Apply Collatz indexing to the current signal.
        Values are kept, indices are transformed n -> (3n+1) repeatedly until odd.
        Returns the list of (new_index, value) pairs.
        """
        K = self.current_len
        # We need to store the current values with their indices.
        # We'll use a temporary MobiusCollatzMemory with odd square‑free indices.
        # Since the current signal is just a list, we assign indices 1,3,5,7,...
        mem = MobiusCollatzMemory(max_index=K*3+1, use_square_free=True)
        idx = 1
        count = 0
        while count < K and idx <= K*3+1:
            if mem._valid_index(idx):
                if count < len(self.signal_buffer):
                    mem.write(idx, self.signal_buffer[count])
                    count += 1
            idx += 2   # only odd
        # Apply Collatz step to all indices
        mem.collatz_step()
        # Extract new signal (sorted by index)
        items = sorted(mem.data.items())
        new_signal = np.array([val for _, val in items], dtype=float)
        K_new = len(new_signal)
        if K_new > self.max_K:
            raise ValueError(f"Collatz expanded to {K_new} > max_K")
        self.signal_buffer[:K_new] = new_signal
        self.current_len = K_new
        return items

    def compress(self):
        """Run the chip compression on the current signal."""
        signal = self.signal_buffer[:self.current_len]
        kept, S, H_ent, m, conv = self.processor.process(signal)
        self.output_kept = kept
        self.output_S = S
        self.output_H = H_ent
        self.output_m = m
        self.output_conv = conv
        return kept, S, H_ent, m, conv

    def get_signal(self):
        """Return the current signal buffer (truncated to current length)."""
        return self.signal_buffer[:self.current_len]

    def get_output(self):
        """Return the compressed coefficients and invariants."""
        return self.output_kept, self.output_S, self.output_H, self.output_m, self.output_conv

    def print_summary(self):
        """Print a summary of the current state."""
        print(f"Signal length: {self.current_len}")
        print(f"Compressed coefficients: {len(self.output_kept)}")
        print(f"Supertrace S = {self.output_S:.6f}")
        print(f"Entropy H = {self.output_H:.6f}")
        print(f"Mass m = {self.output_m:.6f}")
        if self.output_kept:
            print("First 5 kept (index, value):")
            for idx, val in self.output_kept[:5]:
                print(f"  {idx}: {val:.6f}")


# ---------- Simple command‑line interface ----------
def interactive_demo():
    gate = UILogicGate(max_K=256)

    print("=== Möbius Logic Gate UI ===\n")
    print("Available commands:")
    print("  load <K>          – load (μ*H)(n) for n=1..K")
    print("  gate <name>       – apply logic gate (log, exp, sin, cos, trace)")
    print("  collatz           – apply Collatz indexing")
    print("  compress          – run chip compression")
    print("  signal            – show current signal (first 10 values)")
    print("  output            – show compressed output summary")
    print("  quit              – exit")

    while True:
        try:
            cmd = input("\n> ").strip().split()
            if not cmd:
                continue
            if cmd[0] == 'quit':
                break
            elif cmd[0] == 'load':
                if len(cmd) < 2:
                    print("Usage: load <K>")
                    continue
                K = int(cmd[1])
                gate.load_harmonic_convolution(K)
                print(f"Loaded (μ*H)(n) for n=1..{K}")
            elif cmd[0] == 'gate':
                if len(cmd) < 2:
                    print("Usage: gate <name>")
                    continue
                name = cmd[1]
                gate.apply_gate(name)
                print(f"Applied gate '{name}'")
                gate.print_summary()
            elif cmd[0] == 'collatz':
                items = gate.apply_collatz()
                print(f"Collatz step: {len(items)} coefficients remaining")
            elif cmd[0] == 'compress':
                gate.compress()
                print("Compression done.")
                gate.print_summary()
            elif cmd[0] == 'signal':
                sig = gate.get_signal()
                print("Signal (first 10):", sig[:10])
            elif cmd[0] == 'output':
                gate.print_summary()
            else:
                print("Unknown command.")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    interactive_demo()