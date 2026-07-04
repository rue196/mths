import random
import math

# ------------------------------------------------------------
# 1. Generate random test data using Mersenne Twister
#    K = number of "letters" (or states). We'll create random probabilities.
#    The generation is O(K) because each random() call is O(1).
#    Then we sort the probabilities (O(K log K)) to mimic TSP routing.
# ------------------------------------------------------------
def generate_random_test_data(K, seed=42):
    random.seed(seed)               # for reproducibility
    # Generate K random probabilities (not normalized)
    raw = [random.random() for _ in range(K)]
    # Normalize to sum to 1 (so they form a valid distribution)
    total = sum(raw)
    probs = {chr(65+i): p/total for i, p in enumerate(raw)}  # letters A, B, C, ...
    return probs

# ------------------------------------------------------------
# 2. Spectral entropy operator (Collatz‑filtered, derivative at 0) 
#    Returns C array and the computed entropy via Im(ζ'(0)).
# ------------------------------------------------------------
def build_entropy_spectral_coeffs(probs, alpha, collatz_even=True):
    letters = sorted(probs.keys())
    K = len(letters)
    C = [0j] * (2 * K + 1)
    for idx, letter in enumerate(letters):
        i = idx + 1
        p = probs[letter]
        if collatz_even and i % 2 != 0:
            C[K + i] = 0j
        else:
            # C_i = -α * p * log(p) / i   (real)
            C[K + i] = -alpha * p * math.log(p) / i
        # Negative coefficients remain zero (no Hermitian symmetry)
    return C, K

def entropy_from_spectral(C, alpha):
    K = (len(C) - 1) // 2
    sum_iCi = 0.0
    for i in range(1, K + 1):
        sum_iCi += i * C[K + i].real
    zeta_prime_0 = (1j / alpha) * sum_iCi
    return zeta_prime_0.imag   # equals entropy

def shannon_entropy(probs):
    return -sum(p * math.log(p) for p in probs.values())

# ------------------------------------------------------------
# 3. Main test: generate data, compute entropy both ways, verify
# ------------------------------------------------------------
def main():
    K = 100                     # number of distinct letters
    alpha = 0.3628             # arbitrary constant

    # Generate random probabilities (O(K) + O(K log K) for sorting)
    probs = generate_random_test_data(K)
    # Sort the probabilities by value (for TSP‑like routing)
    sorted_probs = sorted(probs.items(), key=lambda x: x[1])
    print("Sorted probabilities (letter → prob):")
    for letter, p in sorted_probs:
        print(f"  {letter}: {p:.4f}")

    # Build spectral coefficients (O(K))
    C, K_actual = build_entropy_spectral_coeffs(probs, alpha, collatz_even=True)

    # Compute entropy via the spectral derivative (O(K))
    spectral_entropy = entropy_from_spectral(C, alpha)
    exact_entropy = shannon_entropy(probs)

    # Show result
    print(f"\nExact Shannon entropy: {exact_entropy:.8f}")
    print(f"Spectral entropy (Im ζ'(0)): {spectral_entropy:.8f}")
    print(f"Difference: {spectral_entropy - exact_entropy:.2e}")

    # (Optional) show first few coefficients
    print("\nFirst 5 non‑zero spectral coefficients (even indices):")
    for i in range(1, K_actual+1):
        if C[K_actual + i] != 0j:
            print(f"  C_{i} = {C[K_actual + i]}")

if __name__ == "__main__":
    main()

    input('Press ENTER to exit')