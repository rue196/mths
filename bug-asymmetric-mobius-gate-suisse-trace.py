#!/usr/bin/env python3
"""
code_checker_dual.py

Dual‑stage code checker:
- Symmetric gate: full compressed signature (good code)
- Asymmetric bug gate: matrix trace (fast O(K) check)
Uses finite‑step derivative constant a = 1/(π−e)/0.3628.
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
    if S == 0:
        return 0.0
    p = abs(S) / K
    if p <= 0 or p >= 1.0:
        return 0.0
    if p < 1e-15:
        return 0.0
    H = -alpha * p * math.log(p)
    return min(max(H, 0.0), 1.0)

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

# ---------- Matrix trace (asymmetric bug gate) ----------
def matrix_trace_signal(signal, i_exp=2):
    """
    Compute the matrix trace from pairs of consecutive values:
    tr = x^(i-1) + y^(i-1), where x = signal[i], y = signal[i+1].
    Sum over all pairs.
    """
    K = len(signal)
    if K < 2:
        return 0.0
    total = 0.0
    for i in range(0, K-1, 2):
        x = abs(signal[i]) + 1e-12
        y = abs(signal[i+1]) + 1e-12
        total += x ** (i_exp - 1) + y ** (i_exp - 1)
    # If odd length, use last element paired with itself
    if K % 2 == 1:
        x = abs(signal[-1]) + 1e-12
        total += 2 * (x ** (i_exp - 1))
    return total

# ---------- Finite derivative ----------
def finite_derivative(signal, step=A):
    K = len(signal)
    diff = np.zeros(K)
    diff[:-1] = (signal[1:] - signal[:-1]) / step
    return diff

# ---------- Code signature generation ----------
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
    return kept, S, H, m, signal

# ---------- Dual‑stage code checker ----------
class DualCodeChecker:
    def __init__(self, K_max=1000):
        self.K_max = K_max
        self.mu = mobius_sieve(K_max)
        self.good_ref = None          # (kept, S, H, m)
        self.bug_trace = None         # matrix trace of buggy code
        self.bug_tolerance = 1e-3

    def load_good_reference(self, code_lines):
        """Store the reference for good (symmetric) code."""
        kept, S, H, m, _ = signature(code_lines, self.mu, apply_derivative=True)
        self.good_ref = (kept, S, H, m)

    def load_bug_reference(self, code_lines):
        """Store the matrix trace for buggy (asymmetric) code."""
        signal = code_to_signal(code_lines)
        signal = finite_derivative(signal, A)
        self.bug_trace = matrix_trace_signal(signal, i_exp=2)

    def check(self, code_lines):
        """
        Returns: (is_ok, similarity, S, H, m, trace_diff)
        """
        # 1. Compute signal and its matrix trace
        signal = code_to_signal(code_lines)
        signal_deriv = finite_derivative(signal, A)
        trace = matrix_trace_signal(signal_deriv, i_exp=2)

        # 2. Asymmetric bug check (if bug_trace is set)
        if self.bug_trace is not None:
            trace_diff = abs(trace - self.bug_trace) / (abs(self.bug_trace) + 1e-12)
            if trace_diff > self.bug_tolerance:
                # Likely a bug: return early with low similarity
                return False, 0.0, 0.0, 0.0, 0.0, trace_diff

        # 3. Symmetric good check
        if self.good_ref is None:
            raise ValueError("No good reference loaded.")
        kept_ref, S_ref, H_ref, m_ref = self.good_ref
        kept_new, S_new, H_new, m_new, _ = signature(code_lines, self.mu, apply_derivative=True)

        # Compare invariants
        score = 1.0 - (abs(S_new - S_ref) / (abs(S_ref) + 1e-12) +
                       abs(H_new - H_ref) / (abs(H_ref) + 1e-12) +
                       abs(m_new - m_ref) / (abs(m_ref) + 1e-12)) / 3.0

        # Compare kept coefficients via cosine similarity
        K = self.K_max
        recon_ref = np.zeros(K, dtype=complex)
        for idx, val in kept_ref:
            recon_ref[idx] = val
        recon_new = np.zeros(K, dtype=complex)
        for idx, val in kept_new:
            recon_new[idx] = val
        mag_ref = np.abs(recon_ref)
        mag_new = np.abs(recon_new)
        dot = np.dot(mag_ref, mag_new)
        norm_ref = np.linalg.norm(mag_ref)
        norm_new = np.linalg.norm(mag_new)
        cos_sim = dot / (norm_ref * norm_new + 1e-12) if norm_ref > 0 and norm_new > 0 else 0.5

        similarity = 0.7 * score + 0.3 * cos_sim
        is_ok = similarity > 0.9  # threshold
        return is_ok, similarity, S_new, H_new, m_new, trace_diff

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
    buggy_code = [
        "def add(a, b):",
        "    return a - b",      # bug
        "def main():",
        "    print(add(2, 3))",
        "if __name__ == '__main__':",
        "    main()"
    ]
    another_good = [
        "def add(a, b):",
        "    return a + b",
        "def main():",
        "    print(add(5, 7))",
        "if __name__ == '__main__':",
        "    main()"
    ]

    checker = DualCodeChecker(K_max=1000)
    checker.load_good_reference(good_code)
    checker.load_bug_reference(buggy_code)  # store buggy trace for fast detection

    # Test good code (should pass)
    ok, sim, S, H, m, diff = checker.check(good_code)
    print(f"Good code: OK={ok}, sim={sim:.4f}, S={S:.4f}, H={H:.4f}, m={m:.4f}, trace_diff={diff:.4f}")

    # Test buggy code (should fail fast)
    ok, sim, S, H, m, diff = checker.check(buggy_code)
    print(f"Buggy code: OK={ok}, sim={sim:.4f}, S={S:.4f}, H={H:.4f}, m={m:.4f}, trace_diff={diff:.4f}")

    # Test another good (should pass, trace diff small)
    ok, sim, S, H, m, diff = checker.check(another_good)
    print(f"Another good: OK={ok}, sim={sim:.4f}, S={S:.4f}, H={H:.4f}, m={m:.4f}, trace_diff={diff:.4f}")

if __name__ == "__main__":
    demo()