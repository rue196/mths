import math
import cmath
import numpy as np

# ---------- 1. Spectral sum ζ(t) = Σ C_i * exp(i * t * i / α) ----------
def zeta2(t, c, alpha):
    """
    t: float
    c: list of length 2K+1, c[K] corresponds to i=0
    alpha: float
    Returns real part of the sum (assuming symmetric coefficients).
    """
    K = (len(c) - 1) // 2
    total = 0.0 + 0.0j
    for i in range(-K, K + 1):
        # coefficient at index i is c[i + K]
        total += c[i + K] * cmath.exp(1j * t * i / alpha)
    return total.real

# ---------- 2. Taylor series for exp(x) (O(K)) ----------
def exp_series(x, k):
    """Return (approximation, error) using first k+1 terms."""
    sum_ = 1.0
    term = 1.0
    for n in range(1, k + 1):
        term *= x / n
        sum_ += term
    return sum_, sum_ - math.e

# ---------- 3. Leibniz series for π (O(K)) ----------
def pi_series(k):
    """Return (approximation, error) using k+1 terms."""
    sum_ = 0.0
    sign = 1.0
    for n in range(k + 1):
        sum_ += sign / (2*n + 1)
        sign = -sign
    pi_approx = 4.0 * sum_
    return pi_approx, pi_approx - math.pi

# ---------- 4. True constants ----------
TRUE_E = math.e
TRUE_PI = math.pi
TRUE_A = 1.0 / (TRUE_PI - TRUE_E)

# ---------- 5. Demonstration: spectral sum (same as Ruby) ----------
if __name__ == "__main__":
    # Spectral sum example
    K = 10
    C = [1.0] * (2*K + 1)   # all ones, symmetric
    alpha = 0.362737364

    t_values = np.linspace(0, 20, 1000)   # 500 intervals = 501 points
    results = [zeta2(t, C, alpha) for t in t_values]
    print("First 5 results of ζ(t):", results[:5])

    # Taylor and Leibniz series comparison
    print("\nTrue constants:")
    print(f"  e = {TRUE_E}")
    print(f"  π = {TRUE_PI}")
    print(f"  a = 1/(π - e) = {TRUE_A}")
    print()

    print("K | e_approx error       | π_approx error      | a_approx            | 1/a_approx + e_approx")
    print("-" * 80)

    for k in range(1, 21):
        e_ap, e_err = exp_series(1.0, k)
        pi_ap, pi_err = pi_series(k)
        a_ap = 1.0 / (pi_ap - e_ap)
        lhs = 1.0 / a_ap + e_ap   # should equal pi_ap (by construction)
        print(f"{k:2d} | {e_err:+12.4e}     | {pi_err:+12.4e}     | {a_ap:15.10f} | {lhs:15.10f}")

    print("\nNote: 1/a_ap + e_ap exactly equals π_ap (by algebra).")
    print(f"      a_ap converges to true a = {TRUE_A} as K increases.")

    # Invariance demonstration (different K for e and π)
    print("\n=== Invariance demonstration ===")
    k_e = 5
    k_pi = 10
    e_ap, _ = exp_series(1.0, k_e)
    pi_ap, _ = pi_series(k_pi)
    a_ap = 1.0 / (pi_ap - e_ap)
    print(f"e approximated with {k_e} terms  = {e_ap}")
    print(f"π approximated with {k_pi} terms = {pi_ap}")
    print(f"a = 1/(π - e) = {a_ap}")
    print(f"1/a + e = {1.0/a_ap + e_ap}")
    print(f"which exactly equals π_ap (by algebra) -> {math.isclose(1.0/a_ap + e_ap, pi_ap)}")

    input("\nPress Enter to exit")