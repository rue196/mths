import math
import random
import numpy as np
from collections import defaultdict

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)

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

# ---------- Helpers ----------
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

def generate_polynomial_embedding(seed, num_monomials=100, num_vars=6):
    """
    Generate a polynomial embedding (as a dict of index->coefficient)
    from a seed (e.g., token ID or hash).
    """
    random.seed(seed)
    monomials = []
    for _ in range(num_monomials):
        coeff = random.uniform(-1.0, 1.0)
        exps = tuple(random.randint(0, 3) for _ in range(num_vars))
        monomials.append((coeff, exps))
    # Store only at odd square-free indices (1,3,5,7,11,13,...)
    K = num_monomials
    mu = mobius_sieve(K)
    embedding = {}
    for idx, (coeff, exps) in enumerate(monomials):
        n = idx + 1
        if mu[n] != 0:   # square-free
            # Use the coefficient as the value; we can also incorporate exponents
            embedding[n] = coeff
    return embedding

def inverse_score_sparse(query_embedding, key_embedding):
    """
    Compute a sparse inverse score between two embeddings.
    Only consider indices that appear in both embeddings.
    """
    common = set(query_embedding.keys()) & set(key_embedding.keys())
    if len(common) < 2:
        return 0.5  # neutral
    q_vals = [query_embedding[i] for i in sorted(common)]
    k_vals = [key_embedding[i] for i in sorted(common)]
    # Use the ML.py inverse_score (merge‑sort inversion count)
    from ML import inverse_score
    return inverse_score(q_vals, k_vals)

class MobiusSparseAttention:
    def __init__(self, max_indices=1000, top_k=5):
        self.max_indices = max_indices
        self.top_k = top_k
        self.memory = {}  # token_id -> embedding (dict of index->coeff)
        self.basel_ref = basel_checksum(mobius_sieve(max_indices), max_indices)

    def add_token(self, token_id, embedding):
        """Store a token embedding (dict)."""
        self.memory[token_id] = embedding
        # Optional: check Basel density of the stored coefficients
        # We'll just keep it simple

    def query(self, query_embedding, threshold=0.3):
        """
        Return top‑k tokens with lowest inverse score (most similar ordering).
        Only tokens that share at least one index are considered.
        """
        scores = []
        for tid, emb in self.memory.items():
            # Compute sparse inverse score
            score = inverse_score_sparse(query_embedding, emb)
            if score < threshold:  # lower score = more similar (less inversions)
                scores.append((tid, score))
        # Sort by score ascending
        scores.sort(key=lambda x: x[1])
        return scores[:self.top_k]

# ---------- Demonstration ----------
def main():
    # Create some token embeddings
    print("Generating token embeddings...")
    tokens = ["hello", "world", "mobius", "attention", "sparse", "polynomial"]
    embeddings = {}
    for i, token in enumerate(tokens):
        # Use token hash as seed
        seed = hash(token) % (10**6)
        emb = generate_polynomial_embedding(seed, num_monomials=200, num_vars=4)
        embeddings[token] = emb

    # Create the attention memory
    attn = MobiusSparseAttention(max_indices=1000, top_k=3)

    for token, emb in embeddings.items():
        attn.add_token(token, emb)

    # Query with a new token
    query_token = "mobius"  # we already have it
    query_emb = embeddings[query_token]

    print(f"\nQuery: '{query_token}'")
    results = attn.query(query_emb, threshold=0.6)

    print("Top similar tokens (by inverse score, lower = more similar):")
    for token, score in results:
        print(f"  {token}: {score:.4f}")

    # Also show the Basel density of the stored coefficients
    print("\nBasel density check:")
    all_indices = set().union(*[set(e.keys()) for e in embeddings.values()])
    mu = mobius_sieve(1000)
    density = sum(1 for i in all_indices if mu[i] != 0) / len(all_indices) if all_indices else 0
    print(f"  Density of square-free indices in embeddings: {density:.4f} (expected ~0.6079)")

if __name__ == "__main__":
    main()