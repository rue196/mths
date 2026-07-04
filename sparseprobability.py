import math
import cmath

# ------------------------------------------------------------
# 1. Letter probabilities from a string
# ------------------------------------------------------------
def letter_probabilities(text):
    freq = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    total = len(text)
    return {ch: cnt / total for ch, cnt in freq.items()}

# ------------------------------------------------------------
# 2. Build spectral coefficients (complex) with Collatz mask (even indices only)
#    C[i] = p_i * exp(i * phi_i) where phi_i = -p_i * log(p_i)
#    Odd indices are set to zero (Collatz odd → 0).
# ------------------------------------------------------------
def build_spectral_coefficients(probs, collatz_mask=True):
    letters = list(probs.keys())
    k = len(letters)
    c = [0j] * (2 * k + 1)
    for idx, letter in enumerate(letters):
        i = idx + 1                     # index 1..K
        p = probs[letter]
        entropy_contrib = -p * math.log(p)   # natural log
        # Phase = entropy contribution
        c[k + i] = cmath.exp(1j * entropy_contrib) * p
        # Hermitian symmetry (optional)
        c[k - i] = c[k + i].conjugate()
    # Apply Collatz mask: zero out odd indices (except i=0 which is even)
    if collatz_mask:
        for i in range(1, k + 1):
            if i % 2 == 1:
                c[k + i] = 0j
                c[k - i] = 0j
    return c, k

# ------------------------------------------------------------
# 3. Spectral sum ζ(t) = Σ_{i=-K}^{K} C_i * exp(i t i/α)
# ------------------------------------------------------------
def zeta_complex(t, c, alpha):
    k = (len(c) - 1) // 2
    total = c[k]   # i=0 term
    for i in range(1, k + 1):
        theta = t * i / alpha
        total += c[k + i] * cmath.exp(1j * theta)
        total += c[k - i] * cmath.exp(-1j * theta)
    return total

# ------------------------------------------------------------
# 4. Exact Shannon entropy
# ------------------------------------------------------------
def shannon_entropy(probs):
    return -sum(p * math.log(p) for p in probs.values())

# ------------------------------------------------------------
# 5. Dummy compression demo
# ------------------------------------------------------------
text = "hello spectral world"
probs = letter_probabilities(text)
c_array, K = build_spectral_coefficients(probs, collatz_mask=True)

# Original data length (characters)
original_len = len(text)
# Number of non‑zero spectral coefficients (including i=0)
non_zero_count = 1  # i=0 is always present
for i in range(1, K + 1):
    if c_array[K + i] != 0j:
        non_zero_count += 2  # both +i and -i

# Spectral array total length (including zeros)
spectral_len = len(c_array)

print("Original text length (characters):", original_len)
print("Number of distinct letters (K):", K)
print("Spectral array total length (2K+1):", spectral_len)
print("Non‑zero spectral coefficients (Collatz even only):", non_zero_count)
print("Compression ratio (original / spectral_len):", original_len / spectral_len)
print("Compression ratio (original / non_zero):", original_len / non_zero_count)
print("\nSpectral coefficients (first 10):")
for i in range(min(10, spectral_len)):
    print(f"  C[{i-K}] = {c_array[i]}")

# Example evaluation
alpha = 0.3628
t_test = 1.0
zeta_val = zeta_complex(t_test, c_array, alpha)
exact_entropy = shannon_entropy(probs)
print(f"\nAt t={t_test}: ζ(t) = {zeta_val}")
print(f"Imaginary part of ζ(t): {zeta_val.imag}")
print(f"Exact entropy: {exact_entropy}")
print(f"Difference: {zeta_val.imag - exact_entropy}")

import math, cmath

def build_spectral_coefficients(probs, collatz_mask=True, hermitian=False):
    letters = list(probs.keys())
    k = len(letters)
    c = [0j] * (2*k + 1)
    for idx, letter in enumerate(letters):
        i = idx + 1
        p = probs[letter]
        phi = -p * math.log(p)          # entropy phase
        c[k + i] = p * cmath.exp(1j * phi)   # no conjugate set
        if hermitian:
            c[k - i] = c[k + i].conjugate()
    if collatz_mask:
        for i in range(1, k + 1):
            if i % 2 == 1:
                c[k + i] = 0j
                if hermitian:
                    c[k - i] = 0j
    return c, k

# Now the imaginary part will be non‑zero if hermitian=False
text = "hello spectral world"
probs = letter_probabilities(text)   # (defined earlier)
c_arr, K = build_spectral_coefficients(probs, collatz_mask=True, hermitian=False)

alpha = 0.3628
t_test = 1.0
zeta = zeta_complex(t_test, c_arr, alpha)   # (defined earlier)
exact_entropy = shannon_entropy(probs)

print(f"ζ({t_test}) = {zeta}")
print(f"Imag part = {zeta.imag}")
print(f"Exact entropy = {exact_entropy}")
print(f"Difference = {zeta.imag - exact_entropy}")

input('Press ENTER to exit')
   