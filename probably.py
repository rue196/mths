import math
import cmath

# 1. Letter probabilities
def letter_probabilities(text):
    freq = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    total = len(text)
    return {ch: cnt / total for ch, cnt in freq.items()}

# 2. Build spectral coefficients with entropy derivative property
def build_entropy_spectral_coeffs(probs, alpha, collatz_even=True):
    letters = list(probs.keys())
    K = len(letters)
    C = [0j] * (2 * K + 1)   # indices -K .. K, C[K] = C_0
    
    for idx, letter in enumerate(letters):
        i = idx + 1          # index from 1 to K
        p = probs[letter]
        # Coefficient for positive index i
        if collatz_even and i % 2 != 0:
            C[K + i] = 0j           # odd index → zero (Collatz mask)
        else:
            # C_i = -α * p * log(p) / i   (real)
            C[K + i] = -alpha * p * math.log(p) / i
        # Negative index: we do NOT set symmetry; we keep zero
        # C[K - i] remains 0j
    
    # C_0 remains zero (no term)
    return C, K

# 3. Spectral sum ζ(t)
def zeta(t, C, alpha):
    K = (len(C) - 1) // 2
    total = C[K]   # i=0 term (zero)
    for i in range(1, K + 1):
        theta = t * i / alpha
        total += C[K + i] * cmath.exp(1j * theta)
        # Negative indices have zero coefficients
    return total

# 4. Exact Shannon entropy
def entropy(probs):
    return -sum(p * math.log(p) for p in probs.values())

# 5. Test
text = "hello spectral world"
probs = letter_probabilities(text)
alpha = 0.3628   # any positive constant
C, K = build_entropy_spectral_coeffs(probs, alpha, collatz_even=True)

# Analytical derivative at 0: ζ'(0) = (i/α) * Σ i * C_i
sum_iCi = 0.0
for i in range(1, K + 1):
    sum_iCi += i * C[K + i].real   # C are real
zeta_prime_0 = (1j / alpha) * sum_iCi
imag_part = zeta_prime_0.imag

H = entropy(probs)

print("Entropy (natural units):", H)
print("Imaginary part of ζ'(0) :", imag_part)
print("Difference:", imag_part - H)

# Optionally compute numerical derivative for verification
eps = 1e-8
zeta_eps = zeta(eps, C, alpha)
zeta_0 = zeta(0.0, C, alpha)
num_deriv = (zeta_eps - zeta_0) / eps
print("Numerical ζ'(0):", num_deriv)
print("Imag part of numerical derivative:", num_deriv.imag)

input('Press ENTER to exit')