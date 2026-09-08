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

# ---------- 2. Elliptic curve coefficients (y² = x³ - x) ----------
def elliptic_coefficients(max_prime):
    """
    Compute a_p = p - #E(F_p) for the curve y² = x³ - x.
    Returns list of (p, a_p) for primes p up to max_prime.
    """
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5)+1):
            if n % i == 0: return False
        return True

    def count_points_mod_p(p):
        # Count points (x,y) satisfying y² = x³ - x mod p, plus point at infinity.
        count = 1  # infinity point
        for x in range(p):
            rhs = (x**3 - x) % p
            # Count y such that y² ≡ rhs mod p
            if rhs == 0:
                count += 1   # y = 0
            elif pow(rhs, (p-1)//2, p) == 1:
                count += 2   # two solutions
        return count

    primes = [p for p in range(2, max_prime+1) if is_prime(p)]
    coeffs = []
    for p in primes:
        if p == 2:
            # For p=2, curve has 4 points (0,0), (1,0), (1,1?), check manually)
            # Actually y² = x³ - x mod 2: x=0 -> y²=0 -> y=0; x=1 -> y²=0 -> y=0; plus infinity = 3? Let's compute:
            # For p=2: points: (0,0), (1,0), infinity -> 3 points.
            N_p = 3
        else:
            N_p = count_points_mod_p(p)
        a_p = p - N_p
        coeffs.append((p, a_p))
    return coeffs

# ---------- 3. Spectral sum ζ(t) ----------
def zeta_elliptic(t, coeffs, alpha):
    """
    ζ(t) = Σ_{p ≤ K} a_p * exp(i * t * p / α)   (real part)
    coeffs: list of (p, a_p)
    alpha: float
    """
    total = 0.0
    for p, a in coeffs:
        total += a * math.cos(t * p / alpha)   # real part
    return total

def dzeta_dt_elliptic(t, coeffs, alpha):
    """
    Derivative dζ/dt = Σ_{p} a_p * (-p/α) * sin(t * p / α)
    """
    total = 0.0
    for p, a in coeffs:
        total += a * (-p / alpha) * math.sin(t * p / alpha)
    return total

# ---------- 4. Rational search (unchanged) ----------
def search_rational_combination(y, e_ap, r_range=(-5,5), s_range=(-5,5), denom_limit=4):
    best_r = Fraction(0,1)
    best_s = Fraction(0,1)
    best_err = float('inf')
    rationals = []
    for den in range(1, denom_limit+1):
        for num in range(r_range[0]*den, r_range[1]*den + 1):
            rationals.append(Fraction(num, den))
    rationals = list(set(rationals))
    for r in rationals:
        for s in rationals:
            pred = float(r) + float(s) * e_ap
            err = abs(y - pred)
            if err < best_err:
                best_err = err
                best_r = r
                best_s = s
    return best_r, best_s, best_err

# ---------- 5. Test on elliptic curve ----------
def test_elliptic(K_series=120, max_prime=50, t_value=None):
    # Approximate π and e
    e_ap = exp_series(K_series)
    pi_ap = pi_series(K_series)
    alpha_ap = 1.0 / (pi_ap - e_ap)   # α ≈ 2.362

    # Get elliptic curve coefficients up to max_prime
    coeffs = elliptic_coefficients(max_prime)
    print(f"Elliptic curve y² = x³ - x, primes up to {max_prime}")
    print("First few (p, a_p):", coeffs[:10])

    # Choose t: if not provided, use t = π (approximated)
    if t_value is None:
        t = pi_ap   # t = π
    else:
        t = t_value

    # Compute ζ(t) and its derivative
    zeta_val = zeta_elliptic(t, coeffs, alpha_ap)
    dzeta_val = dzeta_dt_elliptic(t, coeffs, alpha_ap)

    print(f"\nAt t = {t:.6f} (using π approximation):")
    print(f"  ζ(t) = {zeta_val:.6f}")
    print(f"  ζ'(t) = {dzeta_val:.6f}")

    # Rational search for ζ(t)
    r, s, err = search_rational_combination(zeta_val, e_ap)
    print(f"\nRational fit for ζ(t):")
    print(f"  ζ(t) ≈ {r} + {s} * e  (error = {err:.4e})")
    print(f"  r + s*e = {float(r) + float(s)*e_ap:.6f}")

    # Rational search for ζ'(t)
    r2, s2, err2 = search_rational_combination(dzeta_val, e_ap)
    print(f"\nRational fit for ζ'(t):")
    print(f"  ζ'(t) ≈ {r2} + {s2} * e  (error = {err2:.4e})")
    print(f"  r2 + s2*e = {float(r2) + float(s2)*e_ap:.6f}")

    return zeta_val, dzeta_val, (r,s,err), (r2,s2,err2)

if __name__ == "__main__":
    print("=== Elliptic curve spectral sum and rational search ===\n")
    test_elliptic(K_series=120, max_prime=30, t_value=None)

    # Also test at t = e (approximated)
    e_ap = exp_series(120)
    test_elliptic(K_series=120, max_prime=30, t_value=e_ap)

    print("\n--- Interpretation ---")
    print("The rational search tries to express ζ(t) and ζ'(t) as r + s*e.")
    print("If the errors are small, it suggests a rational relation with e.")
    print("For elliptic curves, such relations are not generally expected,")
    print("but may appear for special curves or at special points.")
    print("The coefficients used here are from the curve y² = x³ - x.")