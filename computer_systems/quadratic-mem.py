#!/usr/bin/env python3
"""
quadratic_mobius_factoring.py

Quadratic factoring of a K-sum of length 2K+1 (K = 10^6),
with Möbius‑based filtering and reconstruction.

Scheme
------
    original signal:  c_n  for  n ∈ [-K, K]      (2K+1 values)
    quadratic factor: q(n) = a·n² + b·n + c
    Möbius filter:    keep only n with μ(n) ≠ 0  (square‑free)
    storage:          sparse dict {q(n): payload}
    reconstruction:   read payload back at source index n

Each factor stores a payload (value, optional extra data),
so the quadratic addresses "store more data" than the raw sum.

For K = 10^6:
    original length  : 2,000,001
    square‑free density: 6/π² ≈ 0.6079
    stored factors   : ≈ 1,216,000
    address range    : up to ~4·10^12 (sparse, hash‑indexed)

All operations are O(K) (sieve) + O(K) (store/read).
"""

import math
import numpy as np
from collections import defaultdict

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)                # ≈ 2.362
ALPHA_USER = 0.3628
A_STEP = ALPHA / ALPHA_USER           # ≈ 6.511
SQUAREFREE_DENSITY = 6.0 / (PI * PI)  # ≈ 0.6079


# ============================================================
#  1. Möbius sieve (linear, O(K))
# ============================================================
def mobius_sieve(K):
    """Return μ(0..K) with μ[0] unused."""
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
#  2. Quadratic factor map
# ============================================================
def quadratic_factor(n, a=1, b=0, c=0):
    """q(n) = a·n² + b·n + c.  Signed quadratic address."""
    return a * n * n + b * n + c


# ============================================================
#  3. Factored signal container
# ============================================================
class QuadraticFactoredSignal:
    """
    Stores a signal of length 2K+1 as quadratic factors,
    filtered by the Möbius function.

    Each factor is a dict entry:
        q(n)  ->  {'value': c_n, 'n': n, 'extra': payload}
    """

    def __init__(self, K, a=1, b=0, c=0, filter_on='none'):
        """
        K          : half‑length (signal = 2K+1)
        a,b,c      : quadratic coefficients
        filter_on  : 'index'  -> keep n with μ(n) ≠ 0
                     'factor' -> keep q(n) with μ(q(n)) ≠ 0
                     'both'   -> require both
                     'none'   -> no filter (store everything)
        """
        self.K = K
        self.a, self.b, self.c = a, b, c
        self.filter_on = filter_on
        self.size = 2 * K + 1
        self.mu = mobius_sieve(K)          # μ up to K (index filter)
        self.factors = {}                  # q -> {'value', 'n', 'extra'}

    # ---------- storage ----------
    def store(self, signal, extras=None):
        """
        Store a 1‑D array of length 2K+1.
        extras (optional): list of payloads, one per index.
        """
        assert len(signal) == self.size, f"expected length {self.size}"
        for n in range(-self.K, self.K + 1):
            idx = n + self.K
            value = float(signal[idx])

            # --- Möbius filter on index ---
            if self.filter_on in ('index', 'both'):
                if mu_val := (self.mu[abs(n)] if abs(n) <= self.K else 0):
                    pass
                else:
                    continue

            # --- quadratic address ---
            q = quadratic_factor(n, self.a, self.b, self.c)

            # --- Möbius filter on factor ---
            if self.filter_on in ('factor', 'both'):
                if 0 <= q <= self.K and self.mu[q] == 0:
                    continue

            extra = extras[idx] if extras is not None else None
            # Store (value, extra) at the quadratic address
            if q in self.factors:
                # Accumulate (folding) — preserves total sum
                self.factors[q]['value'] += value
                if extra is not None:
                    self.factors[q]['extra'] = extra
            else:
                self.factors[q] = {
                    'value': value,
                    'n': n,
                    'extra': extra,
                }

    # ---------- reconstruction ----------
    def reconstruct(self):
        """Return the original signal (zeros where not stored)."""
        out = np.zeros(self.size, dtype=float)
        for q, entry in self.factors.items():
            n = entry['n']
            out[n + self.K] = entry['value']
        return out

    def reconstruct_with_extras(self):
        """Return (signal, extras) arrays."""
        signal = np.zeros(self.size, dtype=float)
        extras = [None] * self.size
        for q, entry in self.factors.items():
            n = entry['n']
            signal[n + self.K] = entry['value']
            extras[n + self.K] = entry['extra']
        return signal, extras

    # ---------- invariants ----------
    def supertrace(self):
        S = 0.0
        for i, (q, entry) in enumerate(sorted(self.factors.items())):
            sign = 1 if (i % 2 == 0) else -1
            S += sign * abs(entry['value'])
        return S

    def entropy(self):
        S = self.supertrace()
        N = max(len(self.factors), 1)
        p = abs(S) / N
        if 0.0 < p < 1.0:
            return -ALPHA * p * math.log(p)
        return 0.0

    def mass(self):
        S = self.supertrace()
        H = self.entropy()
        # Guard against overflow (p ≥ 1 ⇒ H ≤ 0)
        if H > 700:
            return 0.0
        return abs(S) * math.exp(-H)

    def density(self):
        return len(self.factors) / self.size

    def summary(self):
        return {
            'original_length': self.size,
            'stored_factors': len(self.factors),
            'density': self.density(),
            'square_free_density': SQUAREFREE_DENSITY,
            'S': self.supertrace(),
            'H': self.entropy(),
            'm': self.mass(),
        }


# ============================================================
#  4. Demo — K = 10^6
# ============================================================
def demo(K=10**6):
    print(f"=== Quadratic Möbius Factoring (K = {K:,}) ===\n")
    print(f"Signal length        : {2*K+1:,}")
    print(f"Square‑free density  : {SQUAREFREE_DENSITY:.6f}")
    print(f"Expected factors     : ~{int((2*K+1)*SQUAREFREE_DENSITY):,}\n")

    # --- 1. Generate a sparse demo signal (sine + cosine) ---
    print("Generating demo signal ...")
    n_vals = np.arange(-K, K + 1, dtype=float)
    # Use a smooth signal that is cheap to build
    signal = (np.sin(n_vals * 0.001) * np.cos(n_vals * 0.0003)
              + 0.1 * np.sin(n_vals * 0.01))
    signal = signal.astype(float)
    print(f"  mean={signal.mean():.4f}  std={signal.std():.4f}\n")

    # --- 2. Quadratic factor ---
    print("Factoring (quadratic map q(n) = n² + n, Möbius filter on index) ...")
    qfs = QuadraticFactoredSignal(K, a=1, b=1, c=0, filter_on='index')
    qfs.store(signal)
    print(f"  stored {len(qfs.factors):,} factors")

    # --- 3. Reconstruction ---
    print("\nReconstructing ...")
    recon = qfs.reconstruct()
    # Only compare where factors were stored (the rest are 0)
    stored_mask = np.array([(abs(n) <= K and qfs.mu[abs(n)] != 0)
                            for n in range(-K, K + 1)])
    diff = signal[stored_mask] - recon[stored_mask]
    mse = float(np.mean(diff * diff))
    print(f"  MSE on stored entries: {mse:.4e}")

    # --- 4. Invariants ---
    print("\nInvariants:")
    info = qfs.summary()
    for k_, v in info.items():
        if isinstance(v, float):
            print(f"  {k_:>20s}: {v:.6e}")
        else:
            print(f"  {k_:>20s}: {v:,}")

    # --- 5. "More data" — store extras alongside each factor ---
    print("\nStoring extra payload per factor (e.g., a 3‑vector) ...")
    extras = [np.array([np.sin(n), np.cos(n), float(n % 7)])
              for n in range(-K, K + 1)]
    qfs2 = QuadraticFactoredSignal(K, a=1, b=1, c=0, filter_on='index')
    qfs2.store(signal, extras=extras)
    sig_out, ext_out = qfs2.reconstruct_with_extras()
    # verify a few entries
    ok = all(
        ext_out[i] is not None and np.allclose(ext_out[i], extras[i])
        for i in range(0, 2*K + 1, 100_000)
        if abs(i - K) <= K and qfs2.mu[abs(i - K)] != 0
    )
    print(f"  extras verified: {ok}")
    print(f"  total payload entries stored: {len(qfs2.factors):,}")
    print(f"  each entry holds {1 + 3} floats (value + 3‑vector)")

    # --- 6. Quadratic address range ---
    qs = [quadratic_factor(n, 1, 1, 0) for n in (-K, 0, K)]
    print(f"\nQuadratic address range: q(-K)={qs[0]:,}, "
          f"q(0)={qs[1]:,}, q(K)={qs[2]:,}")
    print(f"  address span ≈ {qs[2] - qs[0]:,} (sparse, hash‑indexed)")

    print("\nDone.")


if __name__ == "__main__":
    # For a quick run, use K = 10^5; uncomment K = 10**6 for the full test.
    import sys
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 10**5
    demo(K)