import math
import numpy as np
import hashlib
from Oklogk import mu_convolution_H, mobius_sieve
from random_access_colla_mobius import MobiusCollatzMemory

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)
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

    # --- NEW: Möbius gate for letters ---
    def process_text(self, text, gate_name='log', max_len=None):
        """
        Convert text to ASCII signal, apply a logic gate, and compress.
        If max_len is given, truncate or pad the signal.
        """
        if max_len is None:
            max_len = self.max_K
        # Convert characters to ASCII codes (0-255)
        signal = np.array([ord(c) for c in text], dtype=float)
        # Truncate or pad to max_len
        if len(signal) > max_len:
            signal = signal[:max_len]
        elif len(signal) < max_len:
            signal = np.pad(signal, (0, max_len - len(signal)), constant_values=0)
        self.set_signal(signal)
        self.apply_gate(gate_name)
        return self.get_output()
    
class MobiusCrossCompiler:
    """
    Cross‑compiler that transforms source code into Möbius‑compressed form.
    O(K log K) where K = number of lines (or characters).
    """

    def __init__(self, max_K=1000):
        self.max_K = max_K
        self.memory = MobiusCollatzMemory(max_index=max_K, use_square_free=True)
        self.processor = ChipProcessor(max_K=max_K)
        self.gate = ChipLogicGate(max_K=max_K)
        self.signal_buffer = np.zeros(max_K, dtype=float)
        self.current_len = 0
        self.compiled_coeffs = []
        self.compiled_info = {}

    def compile_code(self, code_lines, gate_name='log', compress=True):
        """
        Compile a list of code lines (strings) into Möbius memory.
        Each line is hashed to a numeric value.
        The resulting signal is transformed by the gate and compressed.
        Returns compressed coefficients.
        """
        K = len(code_lines)
        if K > self.max_K:
            raise ValueError(f"Code has {K} lines > max_K {self.max_K}")

        # 1. Convert each line to a numeric value (hash + length)
        signal = np.zeros(K, dtype=float)
        for i, line in enumerate(code_lines):
            # Use a hash of the line (or ASCII sum) as a value
            h = int(hashlib.md5(line.encode()).hexdigest()[:8], 16) % 10000
            # also include length to distinguish similar lines
            signal[i] = h + 0.1 * len(line)

        # 2. Store in Möbius memory at odd square‑free indices
        self.memory = MobiusCollatzMemory(max_index=K*3+1, use_square_free=True)
        idx = 1
        count = 0
        while count < K and idx <= K*3+1:
            if self.memory._valid_index(idx):
                if count < len(signal):
                    self.memory.write(idx, signal[count])
                    count += 1
            idx += 2   # only odd

        # 3. Extract signal sorted by index
        items = sorted(self.memory.data.items())
        signal_sorted = np.array([val for _, val in items], dtype=float)
        self.signal_buffer[:len(signal_sorted)] = signal_sorted
        self.current_len = len(signal_sorted)

        # 4. Apply logic gate to transform signal
        kept, S, H_ent, m, conv = self.gate.apply_function(signal_sorted, gate_name)

        # 5. Optionally compress further
        if compress:
            # Run chip compression on the transformed signal
            kept2, S2, H2, m2, conv2 = self.processor.process(np.real(conv))
            kept = kept2
            S, H_ent, m = S2, H2, m2
            conv = conv2

        # Store compiled results
        self.compiled_coeffs = kept
        self.compiled_info = {
            'S': S, 'H': H_ent, 'm': m,
            'gate': gate_name,
            'num_lines': K,
            'original_signal': signal_sorted
        }
        # Keep also the compressed conv for reconstruction
        self.compiled_conv = conv
        return kept, S, H_ent, m

    def decompile(self, kept_coeffs=None):
        """
        Reconstruct code from compressed coefficients.
        Returns a list of approximate code lines (as strings).
        """
        if kept_coeffs is None:
            kept_coeffs = self.compiled_coeffs
        if not kept_coeffs:
            raise ValueError("No compiled data to decompile.")

        # Reconstruct the signal from kept coefficients (zero‑padded)
        K = self.compiled_info.get('num_lines', len(kept_coeffs))
        recon = np.zeros(K, dtype=complex)
        for idx, val in kept_coeffs:
            if idx < K:
                recon[idx] = val

        # Map values back to characters: use modulo + ASCII
        codes = []
        for val in recon:
            # Take absolute value, mod 256, clamp to printable ASCII
            c = int(abs(val.real) % 256)
            if c < 32:
                c = 32  # space
            elif c > 126:
                c = 126
            codes.append(chr(c))
        # Combine into lines (we'll treat each coefficient as a line's hash)
        # Since we lost the line structure, we just return a single string.
        return ''.join(codes)

    def compile_file(self, filename, gate_name='log', compress=True):
        """Read a file and compile its lines."""
        with open(filename, 'r') as f:
            lines = f.read().splitlines()
        return self.compile_code(lines, gate_name, compress)

    def compile_string(self, code_string, gate_name='log', compress=True):
        """Compile a multiline string."""
        lines = code_string.split('\n')
        return self.compile_code(lines, gate_name, compress)

    def print_summary(self):
        print(f"Compiled {self.compiled_info.get('num_lines', 0)} lines")
        print(f"Gate: {self.compiled_info.get('gate', 'none')}")
        print(f"Supertrace S = {self.compiled_info.get('S', 0):.6f}")
        print(f"Entropy H = {self.compiled_info.get('H', 0):.6f}")
        print(f"Mass m = {self.compiled_info.get('m', 0):.6f}")
        print(f"Compressed coefficients: {len(self.compiled_coeffs)}")

# ---------- Integration into UI ----------
class UILogicGateWithCompiler(UILogicGate):
    def __init__(self, max_K=1000):
        super().__init__(max_K)
        self.compiler = MobiusCrossCompiler(max_K=max_K)

    def compile_text(self, text, gate_name='log', compress=True):
        """Compile a text string (or code)."""
        return self.compiler.compile_string(text, gate_name, compress)

    def compile_file(self, filename, gate_name='log', compress=True):
        return self.compiler.compile_file(filename, gate_name, compress)

    def decompile(self):
        return self.compiler.decompile()

    def print_compiler_summary(self):
        self.compiler.print_summary()


# ---------- Interactive demo with compile commands ----------
def interactive_demo_with_compiler():
    gate = UILogicGateWithCompiler(max_K=256)

    print("=== Möbius Cross‑Compiler UI ===\n")
    print("Available commands:")
    print("  load <K>          – load (μ*H)(n) for n=1..K")
    print("  gate <name>       – apply logic gate (log, exp, sin, cos, trace)")
    print("  collatz           – apply Collatz indexing")
    print("  compress          – run chip compression")
    print("  text <string>     – process a text string (Möbius gate for letters)")
    print("  compile <text>    – compile code (text) using Möbius cross‑compiler")
    print("  compfile <file>   – compile a file")
    print("  decomp            – decompile the compiled code")
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
            elif cmd[0] == 'text':
                if len(cmd) < 2:
                    print("Usage: text <string>")
                    continue
                text = " ".join(cmd[1:])
                gate.process_text(text, gate_name='log')
                print(f"Processed text: '{text}'")
                gate.print_summary()
            elif cmd[0] == 'compile':
                if len(cmd) < 2:
                    print("Usage: compile <code>")
                    continue
                code = " ".join(cmd[1:])
                gate.compile_text(code, gate_name='log', compress=True)
                print(f"Compiled code: '{code[:30]}...'")
                gate.print_compiler_summary()
            elif cmd[0] == 'compfile':
                if len(cmd) < 2:
                    print("Usage: compfile <filename>")
                    continue
                filename = cmd[1]
                gate.compile_file(filename, gate_name='log', compress=True)
                print(f"Compiled file: {filename}")
                gate.print_compiler_summary()
            elif cmd[0] == 'decomp':
                result = gate.decompile()
                print("Decompiled code (approximate):")
                print(result)
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
    interactive_demo_with_compiler()