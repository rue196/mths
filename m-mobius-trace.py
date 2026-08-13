import math
import numpy as np
from collections import defaultdict

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)          # ≈ 2.362

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

def supertrace_and_mass(signal):
    S = 0.0
    for idx, val in enumerate(signal):
        sign = 1 if (idx % 2 == 0) else -1
        S += sign * abs(val)
    if S == 0:
        H = 0.0; m = 0.0
    else:
        p = abs(S) / len(signal)
        H = -ALPHA * p * math.log(p) if p > 0 else 0.0
        m = abs(S) * math.exp(-H)
    return S, H, m

class MobiusMemoryCache:
    def __init__(self, K, max_slots=None):
        """
        Initialize a cache with K potential slots (indices 1..K).
        Only square-free indices (mu[n] != 0) are used.
        max_slots: maximum number of stored entries. If None, use number of square-free indices.
        """
        self.K = K
        self.mu = mobius_sieve(K)
        self.slots = [n for n in range(1, K+1) if self.mu[n] != 0]
        if max_slots is None:
            max_slots = len(self.slots)
        self.max_slots = max_slots
        self.cache = {}          # slot_index -> trace_value (scalar or vector)
        self.trace_length = None # will be set on first store

    def _get_slot_index(self, index):
        """Map an arbitrary index to a slot index from the square-free list."""
        # Use a hash of the index to pick a slot deterministically
        h = hash(index) % len(self.slots)
        return self.slots[h]

    def store(self, index, trace_value):
        """
        Store a trace value associated with an index (e.g., time step or matrix element index).
        trace_value can be a scalar or a numpy array.
        """
        slot = self._get_slot_index(index)
        if self.trace_length is None:
            if isinstance(trace_value, (list, tuple, np.ndarray)):
                self.trace_length = len(trace_value)
            else:
                self.trace_length = 1
        # If cache is full, evict least recently used (LRU) or based on supertrace mass?
        # Simple: if full, drop the first entry.
        if len(self.cache) >= self.max_slots:
            # Evict the oldest entry (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[slot] = trace_value

    def retrieve(self, index):
        """Retrieve a trace value for an index. Returns None if not found."""
        slot = self._get_slot_index(index)
        return self.cache.get(slot, None)

    def reconstruct(self, indices, default_value=0.0):
        """
        Given a list of indices, return a list of reconstructed trace values.
        For indices that are not in cache, use default_value.
        """
        if self.trace_length is None:
            raise ValueError("Cache is empty, cannot reconstruct.")
        result = []
        for idx in indices:
            val = self.retrieve(idx)
            if val is None:
                if self.trace_length == 1:
                    result.append(default_value)
                else:
                    result.append(np.full(self.trace_length, default_value))
            else:
                result.append(val)
        return result

    def get_cached_indices(self):
        """Return the list of cached slot indices."""
        return list(self.cache.keys())

    def get_cache_size(self):
        return len(self.cache)

    def get_mass(self):
        """Compute the supertrace mass of the cached traces (if they are scalars)."""
        if not self.cache:
            return 0.0
        # Flatten if traces are vectors? We'll take the first element if vector.
        values = []
        for v in self.cache.values():
            if isinstance(v, (list, tuple, np.ndarray)):
                values.append(v[0])  # use first component for mass
            else:
                values.append(v)
        # Compute supertrace mass of the list
        S = 0.0
        for i, val in enumerate(values):
            sign = 1 if (i % 2 == 0) else -1
            S += sign * abs(val)
        if S == 0:
            return 0.0
        N = len(values)
        p = abs(S) / N
        H = -ALPHA * p * math.log(p) if p > 0 else 0.0
        return abs(S) * math.exp(-H)

# ---------- Demonstration ----------
def main():
    K = 20
    cache = MobiusMemoryCache(K, max_slots=10)
    print(f"Square-free slots: {cache.slots}")

    # Store some traces (simulate matrix traces from pendulum)
    for i in range(15):
        # Simulate a trace vector of length 4 (M-matrix elements)
        trace = np.array([1.0 + 0.1*i, 0.8 - 0.05*i, 0.2 + 0.02*i, 0.0 + 0.01*i])
        cache.store(i, trace)
        print(f"Stored index {i} -> slot {cache._get_slot_index(i)}")

    print(f"Cache size: {cache.get_cache_size()}")
    print(f"Cached slots: {cache.get_cached_indices()}")

    # Reconstruct for indices 0..19
    reconstructed = cache.reconstruct(list(range(20)), default_value=0.0)
    for idx, val in enumerate(reconstructed):
        if isinstance(val, np.ndarray):
            print(f"idx {idx}: {val.tolist()}")
        else:
            print(f"idx {idx}: {val}")

    # Compute mass of cache
    mass = cache.get_mass()
    print(f"Cache supertrace mass: {mass:.4f}")

if __name__ == "__main__":
    main()