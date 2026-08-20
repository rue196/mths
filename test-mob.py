import math
import numpy as np

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)

def mobius_sieve(K):
    """Linear sieve for Möbius function, returns list mu[0..K]."""
    if K < 1:
        return [0] * (K + 1)
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


class MobiusMemoryGate:
    """
    Logic gate for memory using Möbius classification:
        μ = -1 : real numbers (stored as float)
        μ =  0 : complex numbers (stored as complex)
        μ =  1 : derived by interpolation from nearest non‑zero entries
    O(1) access, O(K) processing.
    """

    def __init__(self, K):
        self.K = K
        # Compute μ for indices -K..K
        self.mu = mobius_sieve(2*K + 1)          # length 2K+1, index offset K
        self.real_data = {}                       # idx -> float
        self.complex_data = {}                    # idx -> complex
        self.cache = None                         # full reconstructed array (complex)

    # ---------- Internal reconstruction ----------
    def _reconstruct(self):
        """Build the full array (2K+1) using stored real/complex and interpolation for μ=1."""
        arr = np.zeros(2*self.K + 1, dtype=complex)
        for idx in range(-self.K, self.K + 1):
            pos = idx + self.K
            mu_val = self.mu[pos]
            if mu_val == -1:
                arr[pos] = self.real_data.get(idx, 0.0)
            elif mu_val == 0:
                arr[pos] = self.complex_data.get(idx, 0.0 + 0.0j)
            elif mu_val == 1:
                # Interpolate from nearest μ=-1 or μ=0
                left = idx - 1
                right = idx + 1
                vals = []
                while left >= -self.K and self.mu[left + self.K] not in (-1, 0):
                    left -= 1
                while right <= self.K and self.mu[right + self.K] not in (-1, 0):
                    right += 1
                if left >= -self.K and self.mu[left + self.K] in (-1, 0):
                    vals.append(arr[left + self.K])
                if right <= self.K and self.mu[right + self.K] in (-1, 0):
                    vals.append(arr[right + self.K])
                if vals:
                    arr[pos] = np.mean(vals)
                else:
                    arr[pos] = 0.0 + 0.0j
        self.cache = arr
        return arr

    # ---------- O(1) access ----------
    def get(self, idx):
        """Return the value at index idx (real, complex, or derived). O(1)."""
        if idx < -self.K or idx > self.K:
            raise IndexError("Index out of range")
        if self.cache is None:
            self._reconstruct()
        return self.cache[idx + self.K]

    def get_real(self, idx):
        """O(1) access, only valid for μ=-1 indices."""
        if self.mu[idx + self.K] != -1:
            raise ValueError(f"Index {idx} has μ={self.mu[idx+self.K]}, not -1")
        return self.real_data.get(idx, 0.0)

    def get_complex(self, idx):
        """O(1) access, only valid for μ=0 indices."""
        if self.mu[idx + self.K] != 0:
            raise ValueError(f"Index {idx} has μ={self.mu[idx+self.K]}, not 0")
        return self.complex_data.get(idx, 0.0 + 0.0j)

    # ---------- O(1) write ----------
    def set_real(self, idx, value):
        """Store a real number at a μ=-1 index. O(1)."""
        if self.mu[idx + self.K] != -1:
            raise ValueError(f"Index {idx} has μ={self.mu[idx+self.K]}, not -1")
        self.real_data[idx] = float(value)
        self.cache = None

    def set_complex(self, idx, value):
        """Store a complex number at a μ=0 index. O(1)."""
        if self.mu[idx + self.K] != 0:
            raise ValueError(f"Index {idx} has μ={self.mu[idx+self.K]}, not 0")
        self.complex_data[idx] = complex(value)
        self.cache = None

    # ---------- Utility: list indices by type ----------
    def list_indices(self):
        """Return three lists: real_indices, complex_indices, derived_indices."""
        real = []
        complex_ = []
        derived = []
        for idx in range(-self.K, self.K + 1):
            pos = idx + self.K
            if self.mu[pos] == -1:
                real.append(idx)
            elif self.mu[pos] == 0:
                complex_.append(idx)
            elif self.mu[pos] == 1:
                derived.append(idx)
        return real, complex_, derived

    # ---------- O(K) operations ----------
    def apply_gate(self, gate_name):
        """
        Apply an element‑wise logic gate to all stored (and derived) values.
        Reconstructs the full array, transforms it, then stores back real/complex parts.
        Available gates: 'log', 'exp', 'sin', 'cos', 'derivative', 'none'.
        O(K) time.
        """
        arr = self._reconstruct()
        if gate_name == 'log':
            arr = np.log(np.maximum(np.abs(arr), 1e-12)) * np.exp(1j * np.angle(arr))
        elif gate_name == 'exp':
            arr = np.exp(arr)
        elif gate_name == 'sin':
            arr = np.sin(arr)
        elif gate_name == 'cos':
            arr = np.cos(arr)
        elif gate_name == 'derivative':
            diff = np.zeros_like(arr)
            diff[:-1] = (arr[1:] - arr[:-1])
            arr = diff
        # Store back
        self.real_data.clear()
        self.complex_data.clear()
        for idx in range(-self.K, self.K + 1):
            pos = idx + self.K
            if self.mu[pos] == -1:
                self.real_data[idx] = arr[pos].real
            elif self.mu[pos] == 0:
                self.complex_data[idx] = arr[pos]
        self.cache = arr

    def compute_invariants(self):
        """
        Compute supertrace S, entropy H, and mass m from the full array.
        O(K) time.
        """
        arr = self._reconstruct()
        S = 0.0
        for i, val in enumerate(arr):
            sign = 1 if (i % 2 == 0) else -1
            S += sign * abs(val)
        N = len(arr)
        if S == 0:
            H = 0.0
            m = 0.0
        else:
            p = abs(S) / N
            H = -ALPHA * p * math.log(p) if p > 0 else 0.0
            m = abs(S) * math.exp(-H)
        return S, H, m

    def compress_threshold(self, threshold=0.1):
        """
        Keep only indices where |value| > threshold.
        Returns list of (idx, value). O(K) time.
        """
        arr = self._reconstruct()
        kept = []
        for i, val in enumerate(arr):
            if abs(val) > threshold:
                idx = i - self.K
                kept.append((idx, val))
        return kept

    # ---------- Convenience methods ----------
    def process(self, gate_name='none', threshold=None):
        """
        Full pipeline: apply gate, compute invariants, optionally compress.
        """
        self.apply_gate(gate_name)
        S, H, m = self.compute_invariants()
        kept = self.compress_threshold(threshold) if threshold is not None else None
        return kept, S, H, m

# ---------- Demonstration ----------
def demo():
    # Create a memory gate with K=5 (indices -5..5)
    gate = MobiusMemoryGate(5)

    # Show valid indices
    real, comp, deriv = gate.list_indices()
    print("Valid indices:")
    print(f"  μ=-1 (real): {real}")
    print(f"  μ=0 (complex): {comp}")
    print(f"  μ=1 (derived): {deriv}")

    # Store at allowed indices
    for idx in real:
        gate.set_real(idx, float(idx) * 0.5)
    for idx in comp:
        gate.set_complex(idx, idx + 1.0j * idx)

    # Print some values
    print("\nStored values:")
    for idx in real[:3]:
        print(f"  real get({idx}) = {gate.get(idx)}")
    for idx in comp[:3]:
        print(f"  complex get({idx}) = {gate.get(idx)}")
    for idx in deriv[:3]:
        print(f"  derived get({idx}) = {gate.get(idx)}")

    # Apply log gate (to magnitudes)
    gate.apply_gate('log')
    print("\nAfter log gate:")
    for idx in real[:3]:
        print(f"  get({idx}) = {gate.get(idx)}")

    # Compute invariants
    S, H, m = gate.compute_invariants()
    print(f"\nSupertrace S = {S:.4f}")
    print(f"Entropy H = {H:.4f}")
    print(f"Mass m = {m:.4f}")

    # Compress
    kept = gate.compress_threshold(0.1)
    print("\nCompressed (threshold=0.1):")
    for idx, val in kept[:5]:
        print(f"  {idx}: {val}")

if __name__ == "__main__":
    demo()