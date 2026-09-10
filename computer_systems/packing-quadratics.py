"""
mobius_quadratic_translation.py

Extends MobiusMemoryGate with:
  1. Quadratic translation  q(n) = a*n^2 + b*n + c
  2. Binary packing into K^2 slots
  3. Möbius‑ternary packing of a quadratic form
All operations remain O(K) or O(K log K).
"""

import math
import numpy as np
from fractions import Fraction

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)
ALPHA_USER = 0.3628
A_NEW = ALPHA / ALPHA_USER    # ≈ 6.511

# ---------- Möbius sieve (O(K)) ----------
def mobius_sieve(K):
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


# ============================================================
#  Quadratic translation layer
# ============================================================
class QuadraticTranslator:
    """
    Provides three packing/translation schemes on top of a Möbius sieve.

    - Quadratic translation: q(n) = a*n^2 + b*n + c, filtered by μ(q) ≠ 0.
    - Binary packing: indices are placed in K^2 slots (simple 1‑to‑1 mapping).
    - Ternary packing: each index is written in base‑3, then the ternary
      digits are combined into a quadratic form that is Möbius‑filtered.
    """

    def __init__(self, K, a=1, b=1, c=0):
        self.K = K
        self.a, self.b, self.c = a, b, c
        # We need a Möbius table large enough for the ternary packing
        self.max_index = max(K * K, 3 ** int(math.log(max(K, 2), 3)) * 3)
        self.mu = mobius_sieve(self.max_index + 1)

    # ---------- Quadratic translation ----------
    def quadratic(self, n):
        """q(n) = a n^2 + b n + c."""
        return self.a * n * n + self.b * n + self.c

    def quadratic_translate(self, n):
        """Return q(n) if it is within range and μ(q) ≠ 0, else None."""
        q = self.quadratic(n)
        if q < 0 or q > self.max_index:
            return None
        if self.mu[q] == 0:
            return None
        return q

    # ---------- Binary packing (K^2 slots) ----------
    def binary_pack_K2(self, values):
        """
        Place each value into a slot i ∈ [0, K^2). Returns a sparse dict.
        O(K) time, O(K) space.
        """
        K2 = self.K * self.K
        packed = {}
        for i, v in enumerate(values[:K2]):
            packed[i] = v
        return packed

    def binary_unpack_K2(self, packed):
        """Inverse of binary_pack_K2. Returns list of length K^2 (0‑padded)."""
        K2 = self.K * self.K
        out = [0.0] * K2
        for i, v in packed.items():
            if 0 <= i < K2:
                out[i] = v
        return out

    # ---------- Möbius‑ternary quadratic packing ----------
    @staticmethod
    def _to_ternary(n, digits):
        """Return the ternary representation of n as a list of `digits` digits."""
        out = [0] * digits
        for i in range(digits):
            out[digits - 1 - i] = n % 3
            n //= 3
        return out

    def _ternary_quadratic_index(self, digits):
        """
        Combine ternary digits d_0..d_{m-1} into a quadratic form:
            Q = Σ_i d_i * (i+1)^2 + Σ_i d_i^2 * (i+1)
        Return Q (an integer).
        """
        m = len(digits)
        Q = 0
        for i, d in enumerate(digits):
            Q += d * (i + 1) ** 2
            Q += (d * d) * (i + 1)
        return Q

    def mobius_ternary_pack(self, values, digits=None):
        """
        Möbius‑ternary packing of a quadratic.

        For each (index i, value v):
          1. Convert i to ternary (with `digits` digits).
          2. Compute the quadratic index Q from those digits.
          3. Only keep the value if μ(Q) ≠ 0.
        Returns a sparse dict {Q: value}.
        """
        n = len(values)
        if digits is None:
            digits = max(1, int(math.log(max(n, 2), 3)) + 1)
        packed = {}
        for i, v in enumerate(values):
            tern = self._to_ternary(i, digits)
            Q = self._ternary_quadratic_index(tern)
            if 0 <= Q <= self.max_index and self.mu[Q] != 0:
                # Sum values that collide (rare) instead of overwriting
                packed[Q] = packed.get(Q, 0.0) + v
        return packed

    def mobius_ternary_unpack(self, packed, size):
        """
        Inverse of mobius_ternary_pack. Reconstructs an array of length `size`
        by mapping each ternary index back to its original position.
        Collisions are averaged.
        """
        digits = max(1, int(math.log(max(size, 2), 3)) + 1)
        # Build reverse map: Q -> list of original indices
        reverse = {}
        for i in range(size):
            tern = self._to_ternary(i, digits)
            Q = self._ternary_quadratic_index(tern)
            reverse.setdefault(Q, []).append(i)
        out = [0.0] * size
        counts = [0] * size
        for Q, v in packed.items():
            for i in reverse.get(Q, []):
                out[i] += v
                counts[i] += 1
        for i in range(size):
            if counts[i] > 0:
                out[i] /= counts[i]
        return out


# ============================================================
#  Combined memory gate with quadratic translation
# ============================================================
class MobiusQuadraticMemory:
    """
    Combines MobiusMemoryGate (from n‑th‑mobius.py) with the
    QuadraticTranslator. Provides:
      - quadratic translation of indices
      - binary K^2 packing
      - Möbius‑ternary quadratic packing
      - supertrace / entropy invariants on the packed form
    """

    def __init__(self, K, a=1, b=1, c=0):
        self.K = K
        self.translator = QuadraticTranslator(K, a, b, c)
        self.data = {}          # sparse index -> value (in quadratic space)
        self.mu = self.translator.mu

    # ---------- Write via quadratic translation ----------
    def write_quadratic(self, n, value):
        """Write value at quadratic index q(n) (must be μ(q) ≠ 0)."""
        q = self.translator.quadratic_translate(n)
        if q is None:
            raise ValueError(f"n={n} maps to a forbidden Möbius index")
        self.data[q] = value

    # ---------- Write via binary K^2 packing ----------
    def write_binary(self, values):
        """Store a sequence of values in K^2 slots."""
        packed = self.translator.binary_pack_K2(values)
        self.data.update(packed)

    # ---------- Write via ternary quadratic packing ----------
    def write_ternary(self, values):
        """Store values via the Möbius‑ternary quadratic map."""
        packed = self.translator.mobius_ternary_pack(values)
        for q, v in packed.items():
            self.data[q] = self.data.get(q, 0.0) + v

    # ---------- Read ----------
    def read_quadratic(self, n):
        q = self.translator.quadratic_translate(n)
        return self.data.get(q, 0.0) if q is not None else 0.0

    # ---------- Invariants (O(K)) ----------
    def supertrace(self):
        S = 0.0
        for i, (q, v) in enumerate(sorted(self.data.items())):
            sign = 1 if (i % 2 == 0) else -1
            S += sign * abs(v)
        return S

    def entropy(self):
        S = self.supertrace()
        N = max(len(self.data), 1)
        p = abs(S) / N
        if 0.0 < p < 1.0:
            return -ALPHA * p * math.log(p)
        return 0.0

    def mass(self):
        S = self.supertrace()
        H = self.entropy()
        return abs(S) * math.exp(-H)

    def summary(self):
        S = self.supertrace()
        H = self.entropy()
        m = self.mass()
        return dict(num_entries=len(self.data), S=S, H=H, m=m)


# ============================================================
#  Demo
# ============================================================
def demo():
    print("=== Möbius Quadratic Translation ===\n")
    K = 16
    mem = MobiusQuadraticMemory(K, a=1, b=1, c=0)

    # 1. Quadratic translation
    print("Quadratic translation q(n) = n^2 + n:")
    for n in range(1, 6):
        q = mem.translator.quadratic(n)
        keep = mem.translator.quadratic_translate(n)
        print(f"  n={n}: q={q}, μ(q)={mem.mu[q] if q <= mem.translator.max_index else '?'}, kept={keep}")

    # 2. Binary K^2 packing
    print(f"\nBinary K^2 packing (K={K}, slots={K*K}):")
    values = [math.sin(i) for i in range(K)]
    mem.write_binary(values)
    print(f"  Stored {len(values)} values in {K*K}-slot binary form")
    print(f"  First 5 entries: {list(mem.data.items())[:5]}")

    # 3. Möbius-ternary quadratic packing
    print("\nMöbius-ternary quadratic packing:")
    mem2 = MobiusQuadraticMemory(K, a=1, b=1, c=0)
    ternary_values = [math.cos(i) for i in range(K)]
    mem2.write_ternary(ternary_values)
    print(f"  Stored {len(ternary_values)} values, kept {len(mem2.data)} Möbius-allowed slots")
    print(f"  First 5 entries: {list(mem2.data.items())[:5]}")

    # 4. Invariants
    print("\nInvariants:")
    print("  Binary:  ", mem.summary())
    print("  Ternary: ", mem2.summary())

    # 5. Reconstruct ternary
    recon = mem2.translator.mobius_ternary_unpack(mem2.data, K)
    err = sum((a - b) ** 2 for a, b in zip(ternary_values, recon)) / K
    print(f"\n  Ternary reconstruction MSE: {err:.6e}")


if __name__ == "__main__":
    demo()