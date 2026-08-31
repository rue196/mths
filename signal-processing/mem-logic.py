import math
import numpy as np

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)

class FastLogicGate:
    """
    A logic gate with O(1) element access and O(K) processing time.
    It applies a function to a signal, computes invariants,
    and optionally compresses by keeping coefficients above a threshold.
    """

    def __init__(self, max_K=1000):
        self.max_K = max_K
        self.signal = np.zeros(max_K, dtype=float)
        self.length = 0

    def set_signal(self, signal):
        """Copy a 1D array into the gate's buffer (O(K))."""
        K = len(signal)
        if K > self.max_K:
            raise ValueError(f"Signal length {K} exceeds max_K {self.max_K}")
        self.signal[:K] = signal
        self.length = K

    def get(self, idx):
        """O(1) access to element at index idx."""
        if idx < 0 or idx >= self.length:
            raise IndexError("Index out of range")
        return self.signal[idx]

    def apply_gate(self, gate_name):
        """
        Apply a logic gate element‑wise in O(K) time.
        Available: 'log', 'exp', 'sin', 'cos', 'derivative', 'none'.
        """
        if gate_name == 'log':
            self.signal[:self.length] = np.log(np.maximum(self.signal[:self.length], 1e-12))
        elif gate_name == 'exp':
            self.signal[:self.length] = np.exp(self.signal[:self.length])
        elif gate_name == 'sin':
            self.signal[:self.length] = np.sin(self.signal[:self.length])
        elif gate_name == 'cos':
            self.signal[:self.length] = np.cos(self.signal[:self.length])
        elif gate_name == 'derivative':
            # Forward difference with step 1 (O(K))
            diff = np.zeros(self.length, dtype=float)
            diff[:-1] = (self.signal[1:] - self.signal[:-1]) / 1.0
            self.signal[:self.length] = diff
        # 'none' does nothing

    def compute_invariants(self):
        """
        Compute supertrace S, entropy H, and mass m in O(K) time.
        Returns (S, H, m).
        """
        S = 0.0
        for i in range(self.length):
            val = self.signal[i]
            sign = 1 if (i % 2 == 0) else -1
            S += sign * abs(val)
        if S == 0:
            H = 0.0
            m = 0.0
        else:
            p = abs(S) / self.length
            H = -ALPHA * p * math.log(p) if p > 0 else 0.0
            m = abs(S) * math.exp(-H)
        return S, H, m

    def compress_threshold(self, threshold=0.1):
        """
        Compress by keeping only coefficients with |value| > threshold.
        Runs in O(K) (no sorting). Returns list of (index, value).
        """
        kept = []
        for i in range(self.length):
            if abs(self.signal[i]) > threshold:
                kept.append((i, self.signal[i]))
        return kept

    def process(self, signal, gate_name='none', threshold=None):
        """
        Full pipeline: set signal, apply gate, compute invariants,
        optionally compress.
        """
        self.set_signal(signal)
        self.apply_gate(gate_name)
        S, H, m = self.compute_invariants()
        kept = self.compress_threshold(threshold) if threshold is not None else None
        return kept, S, H, m

# ---------- Demonstration ----------
def demo():
    gate = FastLogicGate(max_K=1000)
    # Test signal: harmonic numbers
    K = 20
    signal = np.array([math.log(i+1) for i in range(K)])
    print("Original signal:", signal)

    # Apply log gate
    kept, S, H, m = gate.process(signal, gate_name='log', threshold=0.5)
    print("\nAfter log gate (threshold=0.5):")
    print("  S =", S)
    print("  H =", H)
    print("  m =", m)
    print("  Kept coefficients:", kept)

    # Test element access
    print("\nElement at index 5:", gate.get(5))

if __name__ == "__main__":
    demo()