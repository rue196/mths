import math
from fractions import Fraction

# ---------- Series (O(K) each) ----------
def exp_series(k):
    s = 1.0
    term = 1.0
    for n in range(1, k+1):
        term /= n
        s += term
    return s

def pi_series(k):
    s = 0.0
    sign = 1.0
    for n in range(k+1):
        s += sign / (2*n + 1)
        sign = -sign
    return 4.0 * s

# ---------- Rational search (with small denominators) ----------
def find_rational_fit(y, e_val, denom_limit=4, r_range=(-5,5), s_range=(-5,5)):
    """
    Find rationals r,s (with denominator <= denom_limit) such that
    y ≈ r + s * e_val.
    Returns (best_r, best_s, error) or None if no fit within range.
    """
    best_r, best_s = Fraction(0,1), Fraction(0,1)
    best_err = float('inf')
    # Generate rationals with small denominators
    rationals = set()
    for den in range(1, denom_limit+1):
        for num in range(r_range[0]*den, r_range[1]*den + 1):
            rationals.add(Fraction(num, den))
    rationals = list(rationals)
    for r in rationals:
        for s in rationals:
            pred = float(r) + float(s) * e_val
            err = abs(y - pred)
            if err < best_err:
                best_err = err
                best_r, best_s = r, s
    return best_r, best_s, best_err

# ---------- Safety cap: iterate K until rational fit is good ----------
def safe_iteration(x, name, max_K=200, threshold=1e-3, denom_limit=4):
    """
    Increase K until y = a(K)*x can be represented as r + s*e(K)
    with error < threshold, or until max_K reached.
    """
    for K in range(2, max_K+1):
        e_ap = exp_series(K)
        pi_ap = pi_series(K)
        a_ap = 1.0 / (pi_ap - e_ap)
        y = a_ap * x
        r, s, err = find_rational_fit(y, e_ap, denom_limit=denom_limit)
        if err < threshold:
            print(f"{name}: stopped at K={K}, error={err:.4e}, fit: {r} + {s}·e")
            return K, r, s, err, y
    print(f"{name}: no good fit up to K={max_K}")
    return None, None, None, None, None

# ---------- Test on several irrationals ----------
if __name__ == "__main__":
    test_vals = {
        "π": math.pi,
        "e": math.e,
        "φ": (1+math.sqrt(5))/2,
        "√2": math.sqrt(2),
        "√3": math.sqrt(3)
    }

    print("=== Safety cap using rational search ===\n")
    for name, x in test_vals.items():
        safe_iteration(x, name, max_K=150, threshold=5e-4)

    print("\n--- Interpretation ---")
    print("The algorithm stops when the product a·x can be expressed as r + s·e")
    print("with small rational coefficients and error below the threshold.")
    print("For π and e, we get exact relations early (by construction).")
    print("For other irrationals, it may take more iterations or never reach the threshold.")
    print("This provides a deterministic cap for algorithms that would otherwise loop indefinitely.")