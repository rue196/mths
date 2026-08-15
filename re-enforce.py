import math
import numpy as np
import matplotlib.pyplot as plt
from ML import inverse_score


# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)

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


def supertrace_from_coeffs(coeffs):
    """
    coeffs: dict {index: value} (indices are odd square-free).
    Compute S = Σ_i (-1)^{(i-1)//2} * |value|.
    This alternates every two odd indices.
    """
    S = 0.0
    for idx, val in coeffs.items():
        # idx is odd, (idx-1)//2 gives 0,1,2,...
        sign = 1 if ((idx - 1) // 2) % 2 == 0 else -1
        S += sign * abs(val)
    return S

# ---------- Create target coefficients (desired “dopamine” state) ----------
def create_target_coeffs(max_idx=100, seed=42):
    np.random.seed(seed)
    mem = MobiusCollatzMemory(max_index=max_idx, use_square_free=True)
    # Fill with some random values at odd square-free indices
    # We'll use indices that are odd and square-free
    indices = [i for i in range(1, max_idx, 2) if mem._valid_index(i)]
    values = np.random.randn(len(indices)) * 0.5 + 1.0
    for i, val in zip(indices, values):
        mem.write(i, val)
    return mem

# ---------- RL loop ----------
def rl_loop(target_mem, steps=100, lr=0.02, baseline=0.0):
    # Initialize current memory (state) with zeros at a few indices
    state_mem = MobiusCollatzMemory(max_index=target_mem.max_index, use_square_free=True)
    # Start with small random coefficients at odd square-free indices
    indices = [i for i in range(1, target_mem.max_index, 2) if state_mem._valid_index(i)]
    for i in indices[:10]:   # start with only a few active indices
        state_mem.write(i, np.random.randn() * 0.2 + 0.5)

    # Store history for plotting
    reward_history = []
    match_history = []
    supertrace_history = []

    # Collatz logic gate for actions
    gate = CollatzLogicGate(state_mem)

    for step in range(steps):
        # ---- Action: apply Collatz step to all indices, with value transform ----
        # We'll apply a Collatz step and also multiply values by a factor (action)
        action_factor = 0.8 + 0.4 * np.random.rand()  # random scaling
        def transform(v):
            return v * action_factor
        gate.step(operation='double')  # this uses collatz_step with value_transform = double (from gate.step)
        # Actually, we want a custom transform: we'll use the low-level method
        # We'll implement our own Collatz step with value scaling.
        # For simplicity, we'll just call collatz_step with a lambda.
        state_mem.collatz_step(value_transform=lambda v: v * action_factor)

        # ---- Compute reward ----
        # 1. Supertrace of current state
        S = supertrace_from_coeffs(state_mem.data)
        # 2. Inverse score between current coefficients and target coefficients
        # We need both as lists in the same order (sorted by index)
        current_items = sorted(state_mem.data.items())
        target_items = sorted(target_mem.data.items())
        # Align lengths: take the intersection of indices
        common_indices = set(state_mem.data.keys()) & set(target_mem.data.keys())
        if len(common_indices) > 1:
            curr_vals = [state_mem.data[i] for i in sorted(common_indices)]
            targ_vals = [target_mem.data[i] for i in sorted(common_indices)]
            inv_score = inverse_score(curr_vals, targ_vals)
        else:
            inv_score = 0.5  # neutral

        # Reward: supertrace + bonus for low inverse score (match)
        reward = S - inv_score  # high S and low inv_score -> high reward
        reward_history.append(reward)
        match_history.append(inv_score)
        supertrace_history.append(S)

        # ---- Update coefficients based on reward (REINFORCE‑like) ----
        # If reward > baseline, scale up all coefficients; else scale down
        if reward > baseline:
            scale = 1.0 + lr * (reward - baseline)
        else:
            scale = 1.0 / (1.0 + lr * (baseline - reward))
        for i in list(state_mem.data.keys()):
            state_mem.data[i] *= scale

        # Clamp to avoid explosion
        for i in list(state_mem.data.keys()):
            state_mem.data[i] = np.clip(state_mem.data[i], 0.1, 10.0)

    return state_mem, reward_history, match_history, supertrace_history

# ---------- Main ----------
def main():
    # Create target memory (the "desired" state)
    target = create_target_coeffs(max_idx=100, seed=42)
    print("Target supertrace:", supertrace_from_coeffs(target.data))

    # Run RL
    state, rewards, matches, S_hist = rl_loop(target, steps=80, lr=0.03, baseline=0.2)

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.subplot(2,1,1)
    plt.plot(rewards, label='Reward')
    plt.plot(matches, label='Inverse score (match)')
    plt.plot(S_hist, label='Supertrace')
    plt.xlabel('Step')
    plt.legend()
    plt.grid(True)
    plt.title('RL dynamics: reward = supertrace - inverse score')

    plt.subplot(2,1,2)
    # Show final coefficients vs target
    curr_vals = [state.data[i] for i in sorted(state.data.keys())]
    targ_vals = [target.data[i] for i in sorted(target.data.keys()) if i in state.data]
    plt.stem(sorted(state.data.keys()), curr_vals, label='Learned coefficients')
    plt.stem(sorted([i for i in target.data if i in state.data]), targ_vals, linefmt='r--', label='Target')
    plt.xlabel('Index (odd square-free)')
    plt.ylabel('Coefficient value')
    plt.legend()
    plt.grid(True)
    plt.title('Final state vs target')

    plt.tight_layout()
    plt.show()

    print("Final supertrace:", supertrace_from_coeffs(state.data))
    print("Target supertrace:", supertrace_from_coeffs(target.data))

if __name__ == "__main__":
    main()