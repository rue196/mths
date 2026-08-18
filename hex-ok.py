#!/usr/bin/env python3
"""
mobius_ascii_monomial_gate.py

Möbius ASCII Monomial Elliptical Hexadecimal Gate.
Processes ASCII text into a polynomial, applies Möbius filter,
elliptic permutation, logic gate, and returns a hex fingerprint.
"""

import math
import random
import numpy as np
import hashlib
import struct

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)          # ≈ 2.362
NORM = 1.0 - math.exp(-ALPHA * (PI + E))

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

# ---------- Elliptic permutation (TSP routing) ----------
def elliptic_permutation(K, omega1=PI, omega2=E):
    """
    Generate a permutation of indices 0..K-1 based on pseudo‑angle.
    """
    delta = omega1 - omega2   # π - e
    angles = [(i * delta) % (2 * PI) for i in range(K)]
    order = sorted(range(K), key=lambda i: angles[i])
    return order

# ---------- Exponential convolution (two‑pass, O(K)) ----------
def conv_exp_kernel(signal, alpha=ALPHA):
    K = len(signal)
    lam = math.exp(-alpha)
    f = np.zeros(K, dtype=float)
    f[0] = signal[0]
    for i in range(1, K):
        f[i] = signal[i] + lam * f[i-1]
    b = np.zeros(K, dtype=float)
    b[K-1] = signal[K-1]
    for i in range(K-2, -1, -1):
        b[i] = signal[i] + lam * b[i+1]
    conv_exp = (f + b - signal) / (1 - lam * lam)
    conv = (1.0 - conv_exp) / NORM
    return conv

# ---------- Supertrace and entropy ----------
def supertrace_from_signal(signal):
    S = 0.0
    for idx, val in enumerate(signal):
        sign = 1 if (idx % 2 == 0) else -1
        S += sign * abs(val)
    return S

def entropy_from_supertrace(S, N, alpha=ALPHA):
    if S == 0:
        return 0.0
    p = abs(S) / N
    if p <= 0:
        return 0.0
    return -alpha * p * math.log(p)

def mass_from_signal(signal):
    S = supertrace_from_signal(signal)
    H = entropy_from_supertrace(S, len(signal))
    return abs(S) * math.exp(-H)

# ---------- Basel checksum ----------
def basel_checksum(mu, K):
    S = 0.0
    for n in range(1, K + 1):
        if mu[n] != 0:
            S += mu[n] / (n * n)
    return S

# ---------- Main Gate ----------
class MobiusAsciiGate:
    """
    Processes ASCII text into a compact hexadecimal fingerprint using:
      - ASCII → polynomial coefficients (monomials)
      - Elliptic permutation (TSP routing)
      - Möbius filtering (keep square‑free indices)
      - Optional logic gate (log, exp, sin, cos)
      - Supertrace compression
    Output: hex string of the compressed coefficients (magnitudes).
    """

    def __init__(self, K=256, logic_gate='log', use_elliptic=True):
        self.K = K
        self.logic_gate = logic_gate
        self.use_elliptic = use_elliptic
        self.mu = mobius_sieve(K)
        self.basel_ref = basel_checksum(self.mu, K)

    def _ascii_to_polynomial(self, text, num_vars=6):
        """
        Convert ASCII text into a polynomial with K monomials.
        Each character's ASCII code is used to generate a coefficient.
        """
        random.seed(hash(text) % (2**32))
        monomials = []
        for i, ch in enumerate(text[:self.K]):
            coeff = (ord(ch) - 32) / 95.0   # normalize to [0,1]
            # Add some randomness based on index
            coeff += 0.1 * math.sin(i * 0.5)
            exps = tuple(random.randint(0, 3) for _ in range(num_vars))
            monomials.append((coeff, exps))
        # Pad if shorter than K
        while len(monomials) < self.K:
            monomials.append((0.0, (0,)*num_vars))
        return monomials

    def _apply_mobius_filter(self, monomials):
        """Keep only monomials with square‑free index (μ(n) != 0)."""
        filtered = []
        for idx, (coeff, exps) in enumerate(monomials):
            n = idx + 1
            if self.mu[n] != 0:
                filtered.append((coeff, exps))
        return filtered

    def _apply_elliptic_permutation(self, monomials):
        """Reorder monomials by elliptic angle."""
        K = len(monomials)
        order = elliptic_permutation(K)
        return [monomials[i] for i in order]

    def _apply_logic_gate(self, signal):
        """Apply logic gate to the signal (array of coefficients)."""
        if self.logic_gate == 'log':
            return np.log(np.maximum(np.abs(signal), 1e-12))
        elif self.logic_gate == 'exp':
            return np.exp(signal)
        elif self.logic_gate == 'sin':
            return np.sin(signal)
        elif self.logic_gate == 'cos':
            return np.cos(signal)
        else:
            return signal

    def _chip_compress(self, signal):
        """Run chip pipeline: convolution + supertrace + Möbius compression."""
        K = len(signal)
        # Convolution
        conv = conv_exp_kernel(signal)
        # Supertrace and mass
        S = supertrace_from_signal(conv)
        H = entropy_from_supertrace(S, K)
        m = mass_from_signal(conv)
        M = max(1, int(abs(S)))
        if M > K:
            M = K
        # Keep top M magnitudes with square‑free index
        mag = np.abs(conv)
        idx_sorted = np.argsort(mag)[::-1]
        kept = []
        count = 0
        for idx in idx_sorted:
            n = idx + 1
            if self.mu[n] != 0:
                kept.append((idx, conv[idx]))
                count += 1
                if count >= M:
                    break
        # Reconstruct for error check (optional)
        recon = np.zeros(K, dtype=complex)
        for idx, val in kept:
            recon[idx] = val
        error = np.linalg.norm(conv - recon) / (np.linalg.norm(conv) + 1e-12)
        return kept, S, H, m, error

    def process_text(self, text, num_vars=6):
        """
        Process text and return:
          - hex fingerprint (string)
          - metadata (S, H, m, error, basel_density)
        """
        # 1. ASCII → polynomial
        monomials = self._ascii_to_polynomial(text, num_vars)

        # 2. Elliptic permutation (optional)
        if self.use_elliptic:
            monomials = self._apply_elliptic_permutation(monomials)

        # 3. Möbius filter (keep square‑free)
        filtered = self._apply_mobius_filter(monomials)

        # 4. Extract coefficients as signal (absolute values |C_i|)
        signal = np.array([abs(c) for c, _ in filtered], dtype=float)

        # 5. Apply logic gate
        signal = self._apply_logic_gate(signal)

        # 6. Chip compression
        kept, S, H, m, error = self._chip_compress(signal)

        # 7. Basel density check
        density = basel_checksum(self.mu, self.K) / self.K if self.K > 0 else 0

        # 8. Hexadecimal fingerprint: pack kept coefficients into hex
        # Flatten (index, value) into a bytearray
        data = bytearray()
        for idx, val in kept:
            # pack index (uint16) and value as float (8 bytes)
            data.extend(struct.pack('>Hd', idx, val))
        # Hash to fixed length
        fingerprint = hashlib.sha256(data).hexdigest()

        metadata = {
            'S': S,
            'H': H,
            'm': m,
            'error': error,
            'basel_density': density,
            'num_kept': len(kept)
        }
        return fingerprint, metadata

# ---------- Example ----------
def main():
    import struct  # for packing
    gate = MobiusAsciiGate(K=128, logic_gate='log', use_elliptic=True)

    texts = [
        "Hello, world!",
        "Möbius ASCII gate",
        "Elliptic monomial hex",
        "The quick brown fox jumps over the lazy dog"
    ]

    for text in texts:
        fp, meta = gate.process_text(text, num_vars=4)
        print(f"Text: {text[:30]}...")
        print(f"  Fingerprint: {fp}")
        print(f"  S={meta['S']:.4f}, H={meta['H']:.4f}, m={meta['m']:.4f}")
        print(f"  Error={meta['error']:.2e}, Basel density={meta['basel_density']:.4f}")
        print(f"  Kept coefficients: {meta['num_kept']}\n")

if __name__ == "__main__":
    main()