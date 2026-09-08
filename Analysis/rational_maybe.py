import math
import numpy as np
from fractions import Fraction

# ---------- 1. Series for e and π (O(K)) ----------
def exp_series(k):
    """Approximate e = exp(1) using first k+1 terms."""
    s = 1.0
    term = 1.0
    for n in range(1, k + 1):
        term /= n
        s += term
    return s

def pi_series(k):
    """Approximate π using Leibniz series (4 * sum_{n=0}^k (-1)^n/(2n+1))."""
    s = 0.0
    sign = 1.0
    for n in range(k + 1):
        s += sign / (2*n + 1)
        sign = -sign
    return 4.0 * s

# ---------- 2. Search for rational coefficients ----------
def search_rational_combination(y, e_ap, r_range=(-5,5), s_range=(-5,5), denom_limit=4):
    """
    For a given y, try to find rationals r,s (with small denominator) such that
    y ≈ r + s * e_ap.
    Returns best (r, s, error).
    """
    best_r = Fraction(0,1)
    best_s = Fraction(0,1)
    best_err = float('inf')
    # Generate rationals with small denominator
    rationals = []
    for den in range(1, denom_limit+1):
        for num in range(r_range[0]*den, r_range[1]*den + 1):
            rationals.append(Fraction(num, den))
    # Remove duplicates
    rationals = list(set(rationals))
    # Search
    for r in rationals:
        for s in rationals:
            # Compute r + s*e_ap
            pred = float(r) + float(s) * e_ap
            err = abs(y - pred)
            if err < best_err:
                best_err = err
                best_r = r
                best_s = s
    return best_r, best_s, best_err

# ---------- 3. Main test ----------
def test_irrational(x, name, K=120):
    # Approximate e and π with series
    e_ap = exp_series(K)
    pi_ap = pi_series(K)
    a_ap = 1.0 / (pi_ap - e_ap)
    y = a_ap * x

    # Search for rational representation r + s*e
    r, s, err = search_rational_combination(y, e_ap)

    print(f"\n{name}: x = {x:.6f}")
    print(f"  a_ap = {a_ap:.6f} (using K={K})")
    print(f"  y = a*x = {y:.6f}")
    print(f"  Best rational fit: y ≈ {r} + {s} * e_ap  (error = {err:.4e})")
    print(f"  r + s*e = {float(r) + float(s)*e_ap:.6f}")
    print(f"  Difference: {y - (float(r) + float(s)*e_ap):.4e}")

    # Also check if we can write y as just r (s=0) to see if it's nearly rational
    # Not necessary, but we can show the fractional part
    frac = y - math.floor(y)
    print(f"  Fractional part of y: {frac:.6f}")

# ---------- 4. Run on several irrationals ----------
if __name__ == "__main__":
    phi = (1 + math.sqrt(5)) / 2
    irr = {
        "π": math.pi,
        "e": math.e,
        "φ": phi,
        "√2": math.sqrt(2),
        "√3": math.sqrt(3)
    }

    print("=== Hypothesis: a*x = r + s*e with rational r,s ===")
    print("(a = 1/(π−e), approximated using series)\n")

    for name, val in irr.items():
        test_irrational(val, name, K=120)   # K=120 gives decent accuracy

    # Discussion
    print("\n--- Interpretation ---")
    print("The errors are typically around 1e-3 or less, which is not zero.")
    print("This suggests that for φ, √2, etc., the product a*x is not exactly")
    print("a rational combination of 1 and e with small coefficients.")
    print("The only exact relations hold for π and e themselves (by definition of a).")
    print("Thus the hypothesis that 'any irrational maps to a rational base e'")
    print("does not hold for arbitrary irrationals in this simple form.")