import math

# Constants
PI = math.pi
E  = math.e
A  = 1.0 / (PI + E)          # a = 1/(π+e)

# Small epsilon to prevent overflow/underflow (optional)
EPS = 1e-12

# Exact Gaussian function (for comparison)
def exact_gaussian(x):
    u = x - PI
    # Guard against exponent overflow
    exponent = -u * u / (2.0 * E * E)
    if exponent < -700:   # beyond double precision
        return 0.0
    return A * math.exp(exponent)

# O(K) series approximation:
# g(π+u) ≈ a * Σ_{k=0}^{K-1} (-1)^k * u^{2k} / (2^k * k! * e^{2k})
# Internally uses x = -u^2 / (2*e^2) and computes Σ x^k/k! in O(K).
def series_approx(u, k_terms):
    x = - (u * u) / (2.0 * E * E)   # x = -u^2 / (2e^2)
    total = 1.0
    term = 1.0
    for k in range(1, k_terms):
        term *= x / k
        total += term
    return A * total

# Variant with modulo check (only include even k terms, i.e., multiples of 2)
def series_with_modulo(u, k_terms):
    u2 = u * u
    total = 0.0
    pow_u = 1.0          # u^(2k)
    coeff = 1.0          # (-1)^k / (2^k * k! * e^{2k})
    for k in range(k_terms):
        # Include only terms where k is even (i.e., exponent multiple of 4?)
        # The original Ruby code used: if k % 2 == 0 then sum += coeff * pow_u
        # That means include k=0,2,4,... (even k)
        if k % 2 == 0:
            total += coeff * pow_u
        # Update for next term
        pow_u *= u2
        coeff *= -1.0 / (2.0 * E * E * (k + 1))
    return total * (1.0 / (PI + E))

# Alternative: use modulo with division (like mod_using_div from earlier)
# This is just a demonstration; Python's % is fine.
def mod_using_div(a, b):
    if b == 0:
        return 0
    return a - (a // b) * b

# Test run
if __name__ == "__main__":
    print("K terms: 10")
    print("x\tapprox\t\texact\t\terror")
    # Generate x values from 0 to 20 step 0.5
    x_vals = [i * 0.5 for i in range(41)]  # 0, 0.5, 1.0, ..., 20.0
    for x in x_vals:
        u = x - PI
        approx = series_approx(u, 10)
        exact = exact_gaussian(x)
        error = approx - exact
        print(f"{x:.2f}\t{approx:.6f}\t{exact:.6f}\t{error:+.2e}")

    # Example of series_with_modulo
    print("\nTesting series_with_modulo for u=1.0, K=10:")
    val = series_with_modulo(1.0, 10)
    print(f"Result: {val:.8f}")

    # Show modulo using division function
    print("\nmod_using_div(7,3) =", mod_using_div(7,3))

    
input('Press ENTER to exit')
   