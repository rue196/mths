import math
import numpy as np

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)          # ≈ 2.362

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

# ---------- Basel checksum ----------
def basel_checksum(mu, K):
    """Compute Σ_{n=1}^K μ(n) / n²."""
    S = 0.0
    for n in range(1, K + 1):
        if mu[n] != 0:
            S += mu[n] / (n * n)
    return S

def basel_density(mu, K):
    """Return the fraction of non‑zero μ(n) in 1..K."""
    return sum(1 for n in range(1, K+1) if mu[n] != 0) / K

# ---------- Möbius gate class ----------
class MobiusGate:
    def __init__(self, K, symmetric=True, check_basel=True, tolerance=1e-3):
        """
        K: length of the signal (for asymmetric) or half‑length (for symmetric).
        symmetric: if True, operate on array of length 2K+1 with indices -K..K.
                   if False, operate on length K with indices 1..K.
        check_basel: if True, raise an error when Basel sum deviates too much.
        tolerance: allowed deviation in Basel sum (absolute).
        """
        self.K = K
        self.symmetric = symmetric
        self.check_basel = check_basel
        self.tolerance = tolerance
        # Compute Möbius up to K (or 2K+1)
        self.mu = mobius_sieve(K if not symmetric else 2*K+1)
        # Pre‑compute allowed indices
        if symmetric:
            # indices: -K..K, excluding 0? Actually we allow all, but filter based on μ(|i|)
            self.allowed_indices = [i for i in range(-K, K+1) if self.mu[abs(i)] != 0]
            # Basel sum for symmetric? For symmetry, we sum over n=1..K and multiply by 2? 
            # We'll keep the same formula but for the positive half.
            self.basel_ref = basel_checksum(self.mu, K)
        else:
            self.allowed_indices = [i for i in range(1, K+1) if self.mu[i] != 0]
            self.basel_ref = basel_checksum(self.mu, K)
        # Store the last applied gate and result
        self.last_signal = None
        self.last_kept = None

    def _filter_signal(self, signal):
        """Return only the values at allowed indices (as a list)."""
        if self.symmetric:
            # signal must be of length 2K+1, index mapping: signal[i+K] corresponds to index i
            if len(signal) != 2*self.K + 1:
                raise ValueError(f"Signal length must be {2*self.K+1} for symmetric mode")
            return [signal[i + self.K] for i in self.allowed_indices]
        else:
            # signal is of length K, index mapping: signal[i-1] corresponds to i
            if len(signal) != self.K:
                raise ValueError(f"Signal length must be {self.K} for asymmetric mode")
            return [signal[i-1] for i in self.allowed_indices]

    def _apply_gate_to_signal(self, signal, gate_name):
        """Apply a gate to the full signal (not filtered)."""
        if gate_name == 'log':
            return np.log(np.maximum(signal, 1e-12))
        elif gate_name == 'exp':
            return np.exp(signal)
        elif gate_name == 'sin':
            return np.sin(signal)
        elif gate_name == 'cos':
            return np.cos(signal)
        elif gate_name == 'derivative':
            # forward difference with step ALPHA
            diff = np.zeros_like(signal, dtype=float)
            for i in range(len(signal)-1):
                diff[i] = (signal[i+1] - signal[i]) / ALPHA
            return diff
        else:
            raise ValueError(f"Unknown gate: {gate_name}")

    def _basel_check(self, signal):
        """Compute Basel sum from the given signal (full length) and compare to reference."""
        # Determine which indices are non‑zero in the signal (i.e., where signal[i] != 0)
        # But we want to check the density of the signal's support relative to the allowed indices.
        # We'll compute the sum of μ(n)/n² over indices where signal is non‑zero (or non‑zero after filtering).
        # For simplicity, we'll check the density of non‑zero entries.
        if self.symmetric:
            # Count non‑zero in the positive half (indices 1..K)
            count = 0
            for i in range(self.K):
                if abs(signal[i + self.K + 1]) > 1e-12:  # index i+1
                    if self.mu[i+1] != 0:
                        count += 1
            density = count / self.K
        else:
            count = sum(1 for i in range(1, self.K+1) if abs(signal[i-1]) > 1e-12 and self.mu[i] != 0)
            density = count / self.K
        # Expected density: 6/pi^2 ≈ 0.6079
        expected = 6 / (PI * PI)
        if abs(density - expected) > self.tolerance:
            if self.check_basel:
                raise RuntimeError(f"Basel density deviation: {density:.4f} vs expected {expected:.4f}")
            else:
                print(f"Warning: Basel density = {density:.4f}, expected {expected:.4f}")
        return density

    def process(self, signal, gate_name):
        """
        Apply the gate to the signal, filter via Möbius, and optionally check Basel.
        Returns the filtered signal (list of values at allowed indices) and the full transformed signal.
        """
        # 1. Apply gate to full signal
        transformed = self._apply_gate_to_signal(signal, gate_name)
        self.last_signal = transformed

        # 2. Keep only allowed indices
        kept_values = self._filter_signal(transformed)
        self.last_kept = kept_values

        # 3. Basel check (density of non‑zero allowed indices)
        if self.check_basel:
            self._basel_check(transformed)

        return kept_values, transformed

    def reconstruct(self, kept_values):
        """
        Reconstruct a full signal from the kept values (zero‑padded at disallowed indices).
        """
        if self.symmetric:
            full = np.zeros(2*self.K + 1, dtype=float)
            # Map kept_values back to their indices
            for val, idx in zip(kept_values, self.allowed_indices):
                full[idx + self.K] = val
            return full
        else:
            full = np.zeros(self.K, dtype=float)
            for val, idx in zip(kept_values, self.allowed_indices):
                full[idx-1] = val
            return full

# ---------- Demonstration ----------
def demo():
    K = 50
    # Generate a test signal: harmonic numbers
    signal = np.array([math.log(i+1) for i in range(2*K+1)])  # for symmetric mode
    gate = MobiusGate(K, symmetric=True, check_basel=True, tolerance=0.02)

    print("Original signal (first 10):", signal[:10])

    # Apply log gate
    kept, transformed = gate.process(signal, 'log')
    print(f"\nAfter log gate: kept {len(kept)} values out of {2*K+1}")

    # Reconstruct and verify
    reconstructed = gate.reconstruct(kept)
    print("Reconstructed (first 10):", reconstructed[:10])

    # Check Basel density
    gate._basel_check(reconstructed)

    # Try derivative gate
    kept2, trans2 = gate.process(signal, 'derivative')
    print(f"\nAfter derivative gate: kept {len(kept2)} values")

if __name__ == "__main__":
    demo()