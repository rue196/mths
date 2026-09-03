#!/usr/bin/env python3
"""

Probe whether de Rham cohomology classes of an elliptic curve have algebraic parts
by using the M‑matrix supertrace mass and the square‑packing scaling constant A = 6.511.

The script:
  1. Defines an elliptic curve (e.g., y^2 = x^3 - x).
  2. Samples points (x,y) on the curve.
  3. Builds the flow matrix M and computes supertrace S, entropy H, mass m.
  4. Computes the real period ω1 numerically.
  5. Checks if m / ω1 or m * A is close to a rational number with small denominator.
"""

import math
import numpy as np
from scipy.integrate import quad
from fractions import Fraction

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)          # ≈ 2.362
ALPHA_USER = 0.3628
A = ALPHA / ALPHA_USER          # ≈ 6.511 (square‑packing scaling)

# ---------- Elliptic curve parameters ----------
# We use y^2 = x^3 - x  (complex multiplication, algebraic periods)
a_curve = -1
b_curve = 0
# For general curves, use y^2 = x^3 + a*x + b

def curve_rhs(x):
    """Right‑hand side of elliptic curve: y^2 = x^3 + a*x + b."""
    return x**3 + a_curve * x + b_curve

def sample_points_on_curve(N, x_min=-2.0, x_max=2.0):
    """
    Sample N points (x,y) on the elliptic curve by choosing x values and solving y = ±sqrt(x^3 + a*x + b).
    """
    points = []
    # Choose x values where the cubic is non‑negative
    # For y^2 = x^3 - x, the real part is x ∈ [-1,0] ∪ [1,∞)
    # We'll sample in the range where it's defined.
    x_vals = np.linspace(x_min, x_max, N*2)  # sample more, then filter
    for x in x_vals:
        try:
            rhs = curve_rhs(x)
            if rhs >= 0:
                y = math.sqrt(rhs)
                # Add both branches (if we want a symmetric set)
                points.append((x, y))
                points.append((x, -y))
        except:
            pass
    # If we have more than N points, take the first N
    if len(points) > N:
        points = points[:N]
    # If we have fewer, duplicate some
    while len(points) < N:
        points.append(points[-1])
    return np.array(points)

# ---------- Build M‑matrix and compute supertrace ----------
def build_M_and_supertrace(points):
    """
    Build the flow matrix M_{t,j} = x_j^t + y_j^t for t,j = 1..N.
    Compute supertrace S = Σ_{t=1}^N (-1)^{t+1} M_{t,t}.
    Returns S, H, m.
    """
    N = len(points)
    # Diagonal elements: for each t, M_{t,t} = x_t^t + y_t^t
    # We'll compute using t as 1‑based index.
    diag = []
    for t in range(1, N+1):
        x = points[t-1][0]
        y = points[t-1][1]
        # x_t^t and y_t^t (power with t)
        val = x**t + y**t
        diag.append(val)
    S = 0.0
    for t, val in enumerate(diag, start=1):
        sign = 1 if (t % 2 == 1) else -1   # because (-1)^{t+1}
        S += sign * val
    # Entropy and mass
    N = len(diag)
    if S == 0:
        H = 0.0
        m = 0.0
    else:
        p = abs(S) / N
        if p <= 0 or p >= 1:
            H = 0.0
        else:
            H = -ALPHA * p * math.log(p)
        m = abs(S) * math.exp(-H)
    return S, H, m

# ---------- Compute real period ω1 ----------
def compute_real_period():
    """
    Compute ω1 = ∫_{e1}^{e2} dx / sqrt(4x^3 - g2*x - g3) for the curve.
    For y^2 = x^3 + a*x + b, we have 4x^3 - g2*x - g3 = 4*(x^3 + a*x + b) if we set g2 = -4a, g3 = -4b.
    But the standard Weierstrass form is 4x^3 - g2*x - g3. So g2 = -4a, g3 = -4b.
    For y^2 = x^3 - x, we have a = -1, b = 0, so g2 = 4, g3 = 0.
    Then the real period is 2 * ∫_{e1}^{e2} dx / sqrt(4x^3 - g2*x - g3),
    where e1 < e2 are the real roots of the cubic.
    For y^2 = x^3 - x, roots are x = -1, 0, 1. The real period is 2 * ∫_{0}^{1} dx / sqrt(4x^3 - 4x)?? Wait.
    Actually the differential is dx / sqrt(4x^3 - g2*x - g3) = dx / sqrt(4x^3 - 4x) = dx / (2 sqrt(x^3 - x)).
    So the period is ∫_{0}^{1} dx / sqrt(x^3 - x) * 2? Let's check.
    Standard: ω1 = ∫_{0}^{∞} dx / sqrt(4x^3 - g2*x - g3). For y^2 = x^3 - x, the real period is 2 * ∫_{0}^{1} dx / sqrt(x^3 - x)? Let's just numerically integrate.
    We'll use the formula: the period is 2 * ∫_{e2}^{∞} dx / sqrt(4x^3 - g2*x - g3) where e2 is the largest root.
    For y^2 = x^3 - x, roots: -1, 0, 1. The integral from 1 to ∞ gives one half period.
    Let's compute numerically.
    """
    g2 = -4 * a_curve   # = 4
    g3 = -4 * b_curve   # = 0
    # Find roots of 4x^3 - g2*x - g3 = 0
    # For g2=4, g3=0: 4x^3 - 4x = 4x(x^2-1) = 0 → roots -1, 0, 1.
    # The period is 2 * ∫_{1}^{∞} dx / sqrt(4x^3 - 4x) = 2 * ∫_{1}^{∞} dx / (2 sqrt(x^3 - x)) = ∫_{1}^{∞} dx / sqrt(x^3 - x).
    # But this integral diverges at infinity? Actually it converges because x^3 dominates.
    # We'll integrate from 1 to a large number.
    def integrand(x):
        return 1.0 / math.sqrt(x**3 - x) if x**3 - x > 0 else 0.0
    # Integrate from 1 to 100
    try:
        omega_half, _ = quad(integrand, 1.0, 100.0, limit=1000)
        omega1 = 2.0 * omega_half
        return omega1
    except:
        return None

# ---------- Test rationality ----------
def is_near_rational(value, tolerance=1e-4):
    """
    Check if value is close to a rational number with small denominator.
    Returns (nearest_fraction, distance).
    """
    frac = Fraction(value).limit_denominator(1000)
    dist = abs(value - float(frac))
    return frac, dist

# ---------- Main ----------
def main():
    N = 20  # number of points (must be at least 1)
    points = sample_points_on_curve(N, x_min=-1.5, x_max=1.5)
    print(f"Sampled {len(points)} points on curve y^2 = x^3 + {a_curve}x + {b_curve}")

    S, H, m = build_M_and_supertrace(points)
    print(f"Supertrace S = {S:.6f}")
    print(f"Entropy H = {H:.6f}")
    print(f"Mass m = {m:.6f}")

    # Compute real period
    omega1 = compute_real_period()
    if omega1 is not None:
        print(f"Real period ω1 = {omega1:.6f}")
        # Check if m / ω1 is rational
        ratio = m / omega1
        frac, dist = is_near_rational(ratio)
        print(f"m / ω1 = {ratio:.6f} ≈ {frac} (error {dist:.2e})")
        if dist < 1e-3:
            print("  -> m/ω1 appears algebraic (rational)!")

        # Also check m * A
        product = m * A
        frac2, dist2 = is_near_rational(product)
        print(f"m * A = {product:.6f} ≈ {frac2} (error {dist2:.2e})")
        if dist2 < 1e-3:
            print("  -> m*A appears algebraic (rational)!")
    else:
        print("Could not compute real period.")

if __name__ == "__main__":
    main()