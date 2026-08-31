#!/usr/bin/env python3
"""
mobius_cache_checker.py

Möbius cache for code signatures:
  - Good code: stored symmetrically (full range -K..K)
  - Bugs / bad code: stored asymmetrically (only positive indices)
Uses Basel checksum and finite-step derivative.
"""

import math
import hashlib
import numpy as np

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)          # ≈ 2.362
ALPHA_USER = 0.3628
A = ALPHA / ALPHA_USER          # ≈ 6.511

# ---------- Safe entropy ----------
def safe_entropy(S, K, alpha=ALPHA):
    if S == 0:
        return 0.0
    p = abs(S) / K
    if p <= 0 or p >= 1.0:
        return 0.0
    if p < 1e-15:
        return 0.0
    H = -alpha * p * math.log(p)
    if H < 0:
        H = 0.0
    if H > 1.0:
        H = 1.0
    return H

# ---------- Möbius sieve ----------
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

# ---------- Möbius gate ----------
class MobiusGate:
    def __init__(self, K, symmetric=True, check_basel=True, tolerance=1e-3):
        self.K = K
        self.symmetric = symmetric
        self.check_basel = check_basel
        self.tolerance = tolerance
        self.mu = mobius_sieve(2*K+1 if symmetric else K)
        if symmetric:
            self.allowed_indices = [i for i in range(-K, K+1) if self.mu[abs(i)] != 0]
        else:
            self.allowed_indices = [i for i in range(1, K+1) if self.mu[i] != 0]
        self.last_signal = None
        self.last_kept = None

    def _filter_signal(self, signal):
        if self.symmetric:
            if len(signal) != 2*self.K + 1:
                raise ValueError(f"Signal length must be {2*self.K+1}")
            return [signal[i + self.K] for i in self.allowed_indices]
        else:
            if len(signal) != self.K:
                raise ValueError(f"Signal length must be {self.K}")
            return [signal[i-1] for i in self.allowed_indices]

    def _apply_gate_to_signal(self, signal, gate_name):
        if gate_name == 'log':
            return np.log(np.maximum(signal, 1e-12))
        elif gate_name == 'exp':
            return np.exp(signal)
        elif gate_name == 'sin':
            return np.sin(signal)
        elif gate_name == 'cos':
            return np.cos(signal)
        elif gate_name == 'derivative':
            diff = np.zeros_like(signal)
            diff[:-1] = (signal[1:] - signal[:-1]) / ALPHA
            return diff
        else:
            return signal

    def _basel_check(self, signal):
        if self.symmetric:
            count = 0
            for i in range(self.K):
                if abs(signal[i + self.K + 1]) > 1e-12:
                    if self.mu[i+1] != 0:
                        count += 1
            density = count / self.K
        else:
            count = sum(1 for i in range(1, self.K+1) if abs(signal[i-1]) > 1e-12 and self.mu[i] != 0)
            density = count / self.K
        expected = 6 / (PI * PI)
        if abs(density - expected) > self.tolerance:
            if self.check_basel:
                raise RuntimeError(f"Basel density deviation: {density:.4f} vs expected {expected:.4f}")
            else:
                print(f"Warning: Basel density = {density:.4f}, expected {expected:.4f}")
        return density

    def process(self, signal, gate_name='derivative'):
        transformed = self._apply_gate_to_signal(signal, gate_name)
        self.last_signal = transformed
        kept_values = self._filter_signal(transformed)
        self.last_kept = kept_values
        if self.check_basel:
            self._basel_check(transformed)
        return kept_values, transformed

    def reconstruct(self, kept_values):
        if self.symmetric:
            full = np.zeros(2*self.K + 1, dtype=float)
            for val, idx in zip(kept_values, self.allowed_indices):
                full[idx + self.K] = val
            return full
        else:
            full = np.zeros(self.K, dtype=float)
            for val, idx in zip(kept_values, self.allowed_indices):
                full[idx-1] = val
            return full

# ---------- Code signature ----------
def code_to_signal(code_lines, K):
    signal = np.zeros(K, dtype=float)
    for i, line in enumerate(code_lines):
        h = int(hashlib.md5(line.encode()).hexdigest()[:8], 16) % 10000
        signal[i] = h + 0.1 * len(line)
    return signal

def finite_derivative(signal, step=A):
    diff = np.zeros_like(signal)
    diff[:-1] = (signal[1:] - signal[:-1]) / step
    return diff

def signature(code_lines, gate):
    K = gate.K
    signal = code_to_signal(code_lines, K)
    signal = finite_derivative(signal, A)
    kept, _ = gate.process(signal, gate_name='derivative')
    # Also compute invariants from the full signal (before filtering)
    # We'll compute S, H, m from the transformed signal (full length)
    trans = gate.last_signal
    if trans is not None:
        S = 0.0
        for i, val in enumerate(trans):
            sign = 1 if (i % 2 == 0) else -1
            S += sign * abs(val)
        N = len(trans)
        H = safe_entropy(S, N)
        m = abs(S) * math.exp(-H)
    else:
        S, H, m = 0.0, 0.0, 0.0
    return kept, S, H, m

# ---------- Cache ----------
class MobiusCache:
    def __init__(self, K=256, symmetric_ref=True):
        self.K = K
        self.symmetric_ref = symmetric_ref
        self.good_gate = MobiusGate(K, symmetric=True) if symmetric_ref else MobiusGate(K, symmetric=False)
        self.bad_gate = MobiusGate(K, symmetric=False)
        self.reference_signature = None   # stored as (kept, S, H, m) from good_gate

    def store_good(self, code_lines):
        """Store good code symmetrically."""
        kept, S, H, m = signature(code_lines, self.good_gate)
        self.reference_signature = (kept, S, H, m)
        return kept, S, H, m

    def check(self, code_lines, tolerance=0.1):
        """Check code: compare asymmetric signature to reference."""
        if self.reference_signature is None:
            raise ValueError("No reference stored.")
        kept_good, S_good, H_good, m_good = self.reference_signature
        # Generate asymmetric signature of the new code
        kept_bad, S_bad, H_bad, m_bad = signature(code_lines, self.bad_gate)
        # Compare invariants
        score = 1.0 - (abs(S_bad - S_good) / (abs(S_good) + 1e-12) +
                       abs(H_bad - H_good) / (abs(H_good) + 1e-12) +
                       abs(m_bad - m_good) / (abs(m_good) + 1e-12)) / 3.0
        # Compare kept coefficients (cosine similarity of magnitude spectra)
        K = self.K
        recon_good = np.zeros(2*K+1 if self.symmetric_ref else K, dtype=complex)
        for idx, val in kept_good:
            recon_good[idx] = val
        recon_bad = np.zeros(K, dtype=complex)
        for idx, val in kept_bad:
            recon_bad[idx] = val
        # Align lengths: take absolute magnitudes and pad if needed
        mag_good = np.abs(recon_good)
        mag_bad = np.abs(recon_bad)
        # If good is symmetric, we take only positive half for comparison
        if self.symmetric_ref:
            mag_good = mag_good[self.K+1:]  # indices 1..K (positive)
        # Pad bad to same length if shorter
        if len(mag_bad) < len(mag_good):
            mag_bad = np.pad(mag_bad, (0, len(mag_good)-len(mag_bad)))
        else:
            mag_good = np.pad(mag_good, (0, len(mag_bad)-len(mag_good)))
        dot = np.dot(mag_good, mag_bad)
        norm_good = np.linalg.norm(mag_good)
        norm_bad = np.linalg.norm(mag_bad)
        cos_sim = dot / (norm_good * norm_bad + 1e-12)
        similarity = 0.7 * score + 0.3 * cos_sim
        is_ok = similarity > (1.0 - tolerance)
        return is_ok, similarity, S_bad, H_bad, m_bad

# ---------- Demo ----------
def demo():
    good_code = [
        "def add(a, b):",
        "    return a + b",
        "def main():",
        "    print(add(2, 3))",
        "if __name__ == '__main__':",
        "    main()"
    ]
    bad_code = [
        "def add(a, b):",
        "    return a - b",      # bug
        "def main():",
        "    print(add(2, 3))",
        "if __name__ == '__main__':",
        "    main()"
    ]

    cache = MobiusCache(K=100, symmetric_ref=True)
    cache.store_good(good_code)
    print("Reference stored (symmetric).")

    ok, sim, S, H, m = cache.check(good_code)
    print(f"Good code: OK={ok}, similarity={sim:.4f}, S={S:.4f}, H={H:.4f}, m={m:.4f}")

    ok, sim, S, H, m = cache.check(bad_code)
    print(f"Bad code:  OK={ok}, similarity={sim:.4f}, S={S:.4f}, H={H:.4f}, m={m:.4f}")

if __name__ == "__main__":
    demo()