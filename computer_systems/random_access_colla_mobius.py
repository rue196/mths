import math
import numpy as np

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


# ---------- Möbius function (for filtering) ----------
def mobius_sieve(K):
    mu = [0]*(K+1)
    mu[1] = 1
    primes = []
    is_comp = [False]*(K+1)
    for i in range(2, K+1):
        if not is_comp[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i*p > K:
                break
            is_comp[i*p] = True
            if i % p == 0:
                mu[i*p] = 0
                break
            else:
                mu[i*p] = -mu[i]
    return mu

def is_square_free(n, mu_cache=None):
    if mu_cache is not None and n < len(mu_cache):
        return mu_cache[n] != 0
    # otherwise compute by trial division
    if n < 2:
        return True
    for p in range(2, int(n**0.5)+1):
        if n % p == 0:
            count = 0
            while n % p == 0:
                n //= p
                count += 1
            if count > 1:
                return False
    return True

class MobiusCollatzMemory:
    """
    Memory storing values at odd (and optionally square-free) indices.
    Supports Collatz iteration and element-wise operations.
    """
    def __init__(self, max_index=None, use_square_free=True):
        self.max_index = max_index
        self.use_square_free = use_square_free
        self.mu_cache = None
        if max_index is not None:
            self.mu_cache = mobius_sieve(max_index)
        self.data = {}  # index -> value

    def _valid_index(self, idx):
        if idx % 2 == 0:
            return False
        if self.use_square_free:
            if self.mu_cache is not None and idx < len(self.mu_cache):
                return self.mu_cache[idx] != 0
            else:
                return is_square_free(idx, self.mu_cache)
        return True

    def write(self, idx, value):
        if not self._valid_index(idx):
            raise ValueError(f"Index {idx} is not allowed (even or non-square-free)")
        self.data[idx] = value

    def read(self, idx):
        return self.data.get(idx, None)

    def apply_function(self, func):
        """Apply func to all values in-place."""
        for idx in list(self.data.keys()):
            self.data[idx] = func(self.data[idx])

    def collatz_step(self, value_transform=None):
        """
        Apply a Collatz step to all indices:
        For each odd n, compute next = 3*n + 1, then divide by 2 until odd.
        The value is transformed by value_transform (if provided) or kept.
        Collisions: if two indices map to the same new index, sum the values.
        """
        new_data = {}
        for n, val in self.data.items():
            next_n = 3*n + 1
            # divide by 2 until odd
            while next_n % 2 == 0:
                next_n //= 2
            # only keep if valid (odd and square-free if required)
            if self._valid_index(next_n):
                if value_transform is not None:
                    new_val = value_transform(val)
                else:
                    new_val = val
                if next_n in new_data:
                    new_data[next_n] += new_val
                else:
                    new_data[next_n] = new_val
        self.data = new_data

    def items(self):
        return self.data.items()

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"MobiusCollatzMemory({dict(self.data)})"


# ---------- Collatz Logic Gate ----------
class CollatzLogicGate:
    """
    Applies a sequence of Collatz steps with optional monomial operations
    (multiplication, addition) on the values.
    """
    def __init__(self, memory):
        self.memory = memory

    def step(self, operation=None):
        """
        Perform one Collatz step. If operation is a string, apply it to values:
        - 'double': multiply by 2
        - 'halve': divide by 2
        - 'add1': add 1
        - 'monomial': apply a monomial function (e.g., x**k)
        """
        def val_transform(v):
            if operation == 'double':
                return v * 2
            elif operation == 'halve':
                return v / 2
            elif operation == 'add1':
                return v + 1
            elif operation == 'monomial':
                # assume we want v^2
                return v ** 2
            else:
                return v
        self.memory.collatz_step(val_transform)

    def run(self, steps, operation=None):
        for _ in range(steps):
            self.step(operation)
        return self.memory


# ---------- Example ----------
def main():
    # Initialize memory with some odd indices
    mem = MobiusCollatzMemory(max_index=100, use_square_free=True)
    # Store some initial values: e.g., index -> value (like monomial coefficients)
    init_data = {1: 1.0, 3: 2.0, 5: 3.0, 7: 4.0, 9: 5.0}  # 9 is square-free? 9 has square factor, so will be rejected.
    for idx, val in init_data.items():
        try:
            mem.write(idx, val)
        except ValueError as e:
            print(f"Skipping {idx}: {e}")
    print("Initial memory:", mem)

    # Apply function: multiply all values by 2
    mem.apply_function(lambda x: x*2)
    print("After doubling:", mem)

    # Apply Collatz step with no value transformation
    mem.collatz_step()
    print("After Collatz step (no op):", mem)

    # Apply Collatz step with monomial operation (square)
    gate = CollatzLogicGate(mem)
    gate.step(operation='monomial')
    print("After Collatz step with monomial (square):", mem)

    # Run multiple steps
    gate.run(3, operation='double')
    print("After 3 more steps with doubling:", mem)

if __name__ == "__main__":
    main()