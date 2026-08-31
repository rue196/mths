#!/usr/bin/env python3
"""
trading_harness.py

Trading harness integrating:
- Supply-chain harmonic model (O(K) sums, supertrace, entropy, elliptic projection)
- Finite-step derivative with a = 1/(π−e)/0.3628
- Inverse score (merge-sort inversion count) for feature analysis
- Trading signal generation based on supertrace mass and entropy
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from ML import inverse_score   # only import the function, not the class

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)          # ≈ 2.362
ALPHA_USER = 0.3628
A = ALPHA / ALPHA_USER          # ≈ 6.511 (finite derivative step)

# ---------- Helper: compute inverse scores for all features ----------
def compute_inverse_scores(features, target):
    """
    features: (N_samples, D_features)
    target: (N_samples,)
    Returns: (D_features,) array of inverse scores between each feature and target.
    """
    D = features.shape[1]
    scores = np.zeros(D)
    for j in range(D):
        scores[j] = inverse_score(features[:, j], target)
    return scores

# ---------- Möbius and harmonic functions ----------
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

def harmonic_numbers(N):
    H = np.zeros(N + 1, dtype=float)
    H[1] = 1.0
    for n in range(2, N + 1):
        H[n] = H[n-1] + 1.0 / n
    return H

def build_coeffs(K):
    mu = mobius_sieve(K)
    c = np.zeros(2*K + 1, dtype=complex)
    for n in range(1, K + 1):
        c[K + n] = mu[n]          # i = n
        c[K - n] = mu[n]          # i = -n
    c[K] = 0.0                    # i = 0
    return c

def zeta_at(t, c, alpha):
    K = (len(c) - 1) // 2
    total = 0.0 + 0.0j
    for i in range(-K, K + 1):
        total += c[i + K] * np.exp(1j * t * i / alpha)
    return total

# ---------- Supertrace and entropy ----------
def supertrace_from_signal(signal):
    S = 0.0
    for idx, val in enumerate(signal):
        sign = 1 if (idx % 2 == 0) else -1
        S += sign * abs(val)
    return S

def entropy_from_supertrace(S, K, alpha=ALPHA):
    if S == 0:
        return 0.0
    p = abs(S) / K
    if p <= 0 or p >= 1:
        return 0.0
    return -alpha * p * math.log(p)

def mass_from_supertrace(S, K):
    H = entropy_from_supertrace(S, K)
    return abs(S) * math.exp(-H)

# ---------- Finite-step derivative ----------
def finite_derivative(signal, step=A):
    if len(signal) < 2:
        return np.array([0.0])
    diff = np.zeros_like(signal)
    diff[:-1] = (signal[1:] - signal[:-1]) / step
    return diff

# ---------- Supply-chain stage model ----------
def supply_chain_stages(prices, K=None):
    if K is None:
        K = len(prices) - 1
    diff = np.diff(prices)
    if len(diff) < K:
        diff = np.pad(diff, (0, K - len(diff)), constant_values=0)
    else:
        diff = diff[:K]
    return diff

# ---------- Elliptic projection (bounded scarcity) ----------
def elliptic_projection(z, alpha=ALPHA):
    x = np.real(z)
    y = np.imag(z)
    x_norm = x / PI
    y_norm = y / E
    weight = 0.5 * (1 + np.cos(2 * np.pi * x_norm) * np.cos(2 * np.pi * y_norm))
    return np.clip(weight, 0.0, 1.0)

# ---------- Trading harness ----------
class TradingHarness:
    def __init__(self, K=50, alpha=ALPHA):
        self.K = K
        self.alpha = alpha
        self.coeffs = build_coeffs(K)
        self.H = harmonic_numbers(K)
        self.zeta_vals = np.array([zeta_at(self.H[n], self.coeffs, alpha).real for n in range(1, K+1)])
        self.step = A
        self.history = {
            'prices': [],
            'stages': [],
            'S': [],
            'H_ent': [],
            'm': [],
            'derivative': [],
            'signal': []
        }

    def feed_prices(self, prices):
        """Feed a price series and update internal state."""
        stages = supply_chain_stages(prices, K=self.K)
        S = supertrace_from_signal(stages)
        H_ent = entropy_from_supertrace(S, self.K)
        m = mass_from_supertrace(S, self.K)
        deriv = finite_derivative(stages, self.step)

        self.history['prices'].append(prices[-1] if len(prices)>0 else 0.0)
        self.history['stages'].append(stages)
        self.history['S'].append(S)
        self.history['H_ent'].append(H_ent)
        self.history['m'].append(m)
        self.history['derivative'].append(deriv)

        # Trading signal: combine mass and derivative
        if m > 1.0 and np.mean(deriv) > 0:
            signal = 1   # buy
        elif m < 0.5 and np.mean(deriv) < 0:
            signal = -1  # sell
        else:
            signal = 0   # neutral
        self.history['signal'].append(signal)
        return signal

    def analyze_features(self, features, target):
        """Compute inverse scores between features and target."""
        return compute_inverse_scores(features, target)

    def plot_history(self):
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        axes[0].plot(self.history['S'], label='Supertrace S')
        axes[0].set_ylabel('S')
        axes[0].legend()
        axes[1].plot(self.history['H_ent'], label='Entropy H')
        axes[1].set_ylabel('H')
        axes[1].legend()
        axes[2].plot(self.history['m'], label='Mass m')
        axes[2].axhline(y=1.0, color='r', linestyle='--', label='buy threshold')
        axes[2].axhline(y=0.5, color='g', linestyle='--', label='sell threshold')
        axes[2].set_ylabel('m')
        axes[2].legend()
        axes[3].plot(self.history['signal'], label='Trading signal', drawstyle='steps-post')
        axes[3].set_ylabel('Signal')
        axes[3].set_xlabel('Time step')
        axes[3].legend()
        plt.tight_layout()
        plt.show()

# ---------- Demo ----------
def demo():
    np.random.seed(42)
    # Simulate price series
    T = 100
    prices = [100.0]
    for _ in range(T):
        ret = 0.001 * np.random.randn() + 0.0005 * np.sin(2*math.pi * _ / 20)
        prices.append(prices[-1] * (1 + ret))
    prices = np.array(prices)

    harness = TradingHarness(K=30)
    signals = []
    for i in range(1, len(prices)):
        window = prices[max(0, i-30):i+1]
        sig = harness.feed_prices(window)
        signals.append(sig)

    harness.plot_history()
    print("Trading signals (last 10):", signals[-10:])

if __name__ == "__main__":
    demo()
