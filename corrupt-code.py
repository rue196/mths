#!/usr/bin/env python3
"""
corrupt-code.py - Safe Möbius code checker with bounded entropy.
"""

import math
import hashlib
import numpy as np

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)          # ≈ 2.362
ALPHA_USER = 0.3628
A = ALPHA / ALPHA_USER          # ≈ 6.511 (finite difference step)

# ---------- Safe entropy ----------
def safe_entropy(S, K, alpha=ALPHA):
    """Compute entropy with bounds to avoid overflow."""
    if S == 0:
        return 0.0
    p = abs(S) / K
    # Clamp p to avoid extreme values
    if p <= 0:
        return 0.0
    if p >= 1.0:
        return 0.0
    # For very small p, p*log(p) is very small; we can set a lower bound
    if p < 1e-15:
        return 0.0
    H = -alpha * p * math.log(p)
    # H is always ≤ alpha/e ≈ 0.869, but just in case
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

# ---------- Chip pipeline ----------
def tsp_route(signal):
    K = len(signal)
    angles = np.zeros(K)
    for i in range(K):
        x = math.sin(i * 7.0) + 0.1 * math.cos(i * 13.0)
        y = math.cos(i * 11.0) + 0.1 * math.sin(i * 17.0)
        angles[i] = math.atan2(y, x) + math.pi
    buckets = [[] for _ in range(360)]
    for i, a in enumerate(angles):
        idx = int((a / (2 * math.pi)) * 360) % 360
        buckets[idx].append(i)
    order = []
    for b in buckets:
        order.extend(b)
    return np.array(order)

def conv_exp_kernel(signal, alpha=ALPHA):
    K = len(signal)
    lam = math.exp(-alpha)
    f = np.zeros(K)
    f[0] = signal[0]
    for i in range(1, K):
        f[i] = signal[i] + lam * f[i-1]
    b = np.zeros(K)
    b[K-1] = signal[K-1]
    for i in range(K-2, -1, -1):
        b[i] = signal[i] + lam * b[i+1]
    conv_exp = (f + b - signal) / (1 - lam * lam)
    norm = 1.0 - math.exp(-alpha * (PI + E))
    conv = (1.0 - conv_exp) / norm
    return conv

def supertrace_and_mass(conv, K):
    S = 0.0
    for i, val in enumerate(conv):
        sign = 1 if (i % 2 == 0) else -1
        S += sign * abs(val)
    H = safe_entropy(S, K)
    m = abs(S) * math.exp(-H)
    return S, H, m

def chip_compress(signal, mu):
    K = len(signal)
    order = tsp_route(signal)
    signal_sorted = signal[order]
    conv = conv_exp_kernel(signal_sorted)
    S, H, m = supertrace_and_mass(conv, K)
    M = max(1, int(abs(S)))
    if M > K:
        M = K
    mag = np.abs(conv)
    idx_sorted = np.argsort(mag)[::-1]
    kept = []
    count = 0
    for idx in idx_sorted:
        n = idx + 1
        if mu[n] != 0:
            kept.append((idx, conv[idx]))
            count += 1
            if count >= M:
                break
    return kept, S, H, m, conv

# ---------- Finite derivative ----------
def finite_derivative(signal, step=A):
    K = len(signal)
    diff = np.zeros(K)
    diff[:-1] = (signal[1:] - signal[:-1]) / step
    return diff

# ---------- Code signature ----------
def code_to_signal(code_lines):
    K = len(code_lines)
    signal = np.zeros(K, dtype=float)
    for i, line in enumerate(code_lines):
        h = int(hashlib.md5(line.encode()).hexdigest()[:8], 16) % 10000
        signal[i] = h + 0.1 * len(line)
    return signal

def signature(code_lines, mu, apply_derivative=True):
    signal = code_to_signal(code_lines)
    if apply_derivative:
        signal = finite_derivative(signal, A)
    kept, S, H, m, _ = chip_compress(signal, mu)
    return kept, S, H, m

# ---------- Code checker ----------
class CodeChecker:
    def __init__(self, K_max=1000):
        self.K_max = K_max
        self.mu = mobius_sieve(K_max)
        self.reference = None

    def load_reference(self, code_lines):
        self.reference = signature(code_lines, self.mu, apply_derivative=True)

    def check(self, code_lines, tolerance=0.1):
        if self.reference is None:
            raise ValueError("No reference loaded.")
        kept_new, S_new, H_new, m_new = signature(code_lines, self.mu, apply_derivative=True)
        kept_ref, S_ref, H_ref, m_ref = self.reference

        # Compare invariants
        score = 1.0 - (abs(S_new - S_ref) / (abs(S_ref) + 1e-12) +
                       abs(H_new - H_ref) / (abs(H_ref) + 1e-12) +
                       abs(m_new - m_ref) / (abs(m_ref) + 1e-12)) / 3.0
        # Compare kept coefficients via inverse score
        K = self.K_max
        recon_ref = np.zeros(K, dtype=complex)
        for idx, val in kept_ref:
            recon_ref[idx] = val
        recon_new = np.zeros(K, dtype=complex)
        for idx, val in kept_new:
            recon_new[idx] = val
        # Use simple correlation (cosine similarity of magnitudes)
        mag_ref = np.abs(recon_ref)
        mag_new = np.abs(recon_new)
        dot = np.dot(mag_ref, mag_new)
        norm_ref = np.linalg.norm(mag_ref)
        norm_new = np.linalg.norm(mag_new)
        cos_sim = dot / (norm_ref * norm_new + 1e-12) if norm_ref > 0 and norm_new > 0 else 0.5
        # Combine: 70% invariants, 30% structure
        similarity = 0.7 * score + 0.3 * cos_sim
        is_ok = similarity > (1.0 - tolerance)
        return is_ok, similarity, S_new, H_new, m_new

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

    checker = CodeChecker(K_max=1000)
    checker.load_reference(good_code)

    ok, sim, S, H, m = checker.check(good_code)
    print(f"Good code: OK={ok}, similarity={sim:.4f}, S={S:.4f}, H={H:.4f}, m={m:.4f}")

    ok, sim, S, H, m = checker.check(bad_code)
    print(f"Bad code:  OK={ok}, similarity={sim:.4f}, S={S:.4f}, H={H:.4f}, m={m:.4f}")

if __name__ == "__main__":
    demo()