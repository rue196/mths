import numpy as np
import math

# ============================================================
# TRADING HARNESS — Ford Motor (F), Sept 15 2026
# ============================================================

PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)          # ≈ 2.362
ALPHA_USER = 0.3628
A_STEP = ALPHA / ALPHA_USER      # ≈ 6.511

def mobius_sieve(K):
    mu = [0]*(K+1); mu[1] = 1
    primes = []; is_comp = [False]*(K+1)
    for i in range(2, K+1):
        if not is_comp[i]:
            primes.append(i); mu[i] = -1
        for p in primes:
            if i*p > K: break
            is_comp[i*p] = True
            if i % p == 0: mu[i*p] = 0; break
            else: mu[i*p] = -mu[i]
    return mu

def build_coeffs(K):
    mu = mobius_sieve(K)
    c = np.zeros(2*K+1, dtype=complex)
    for n in range(1, K+1):
        c[K+n] = mu[n]; c[K-n] = mu[n]
    c[K] = 0.0
    return c

def zeta_at(t, c, alpha):
    K = (len(c)-1)//2
    total = 0.0+0.0j
    for i in range(-K, K+1):
        total += c[i+K]*np.exp(1j*t*i/alpha)
    return total

def supertrace_from_signal(signal):
    S = 0.0
    for idx, val in enumerate(signal):
        sign = 1 if (idx % 2 == 0) else -1
        S += sign * abs(val)
    return S

def entropy_from_supertrace(S, K, alpha=ALPHA):
    if S == 0: return 0.0
    p = abs(S)/K
    if p <= 0 or p >= 1: return 0.0
    return -alpha * p * math.log(p)

def mass_from_supertrace(S, K):
    H = entropy_from_supertrace(S, K)
    return abs(S) * math.exp(-H)

def finite_derivative(signal, step=A_STEP):
    if len(signal) < 2: return np.array([0.0])
    diff = np.zeros_like(signal)
    diff[:-1] = (signal[1:] - signal[:-1]) / step
    return diff

# --- Ford price series: synthetic reconstruction from Q2/Q3 2026 data ---
# Actual anchors: $13.88 (Sep 10), $13.42 (Aug 1 approx), $14.20 (Jul 15)
# Q2 EPS $0.42 beat, Q3 EPS est $0.41
np.random.seed(15)
T = 60
base_prices = np.linspace(13.00, 13.88, T)
noise = np.random.randn(T) * 0.08
# Add tariff shock dip around t=40
base_prices[40:45] -= 0.35
ford_prices = base_prices + noise

K = 30
coeffs = build_coeffs(K)
stages_signal = np.diff(ford_prices)
if len(stages_signal) < K:
    stages_signal = np.pad(stages_signal, (0, K-len(stages_signal)))

# --- Compute harness metrics ---
S = supertrace_from_signal(stages_signal)
H_ent = entropy_from_supertrace(S, K)
m = mass_from_supertrace(S, K)
deriv = finite_derivative(stages_signal)

# --- Signal generation ---
print(f"=== TRADING HARNESS — FORD (F) — 2026-09-15 ===")
print(f"Supertrace S:        {S:.4f}")
print(f"Entropy H:           {H_ent:.4f}")
print(f"Mass m:              {m:.4f}")
print(f"Mean derivative:     {np.mean(deriv):.6f}")
print(f"Current price:       $13.88")

# Signal logic from harness
if m > 1.0 and np.mean(deriv) > 0:
    signal = "BUY"
elif m < 0.5 and np.mean(deriv) < 0:
    signal = "SELL"
else:
    signal = "NEUTRAL"

print(f"\nTrading signal:      {signal}")