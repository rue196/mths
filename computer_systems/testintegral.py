import math
import numpy as np

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)          # ≈ 2.362

# ---------- Supertrace entropy from sampled values ----------
def supertrace_entropy(values):
    """
    Compute an entropy-like scalar from a list of values.
    S = Σ_even v_i - Σ_odd v_i, then H = -α*(|S|/N)*log(|S|/N)
    """
    N = len(values)
    if N == 0:
        return 0.0
    S = 0.0
    for i, v in enumerate(values):
        if (i + 1) % 2 == 0:   # even 1-based -> positive
            S += v
        else:
            S -= v
    ratio = abs(S) / N
    if 0.0 < ratio < 1.0:
        return -ALPHA * ratio * math.log(ratio)
    return 0.0

# ---------- Adaptive Simpson integrator with alpha scaling ----------
def adaptive_simpson(f, a, b, tol=1e-6, max_depth=20, alpha_scale=1.0):
    """
    Recursive adaptive Simpson integration.
    The step size and tolerance are scaled by alpha_scale.
    """
    def simpson(f, a, b):
        return (b - a) / 6.0 * (f(a) + 4.0 * f((a + b) / 2.0) + f(b))

    def recursive(a, b, fa, fm, fb, S, depth):
        # Midpoint
        m = (a + b) / 2.0
        lm = (a + m) / 2.0
        rm = (m + b) / 2.0
        flm = f(lm)
        frm = f(rm)

        S_left = simpson(f, a, m)
        S_right = simpson(f, m, b)
        S_total = S_left + S_right

        # Use alpha to scale the tolerance (larger alpha -> more strict)
        scaled_tol = tol * (1.0 / (1.0 + alpha_scale * 0.1))  # example scaling
        error = abs(S_total - S)

        if depth <= 0 or error < 15.0 * scaled_tol:
            return S_total + (S_total - S) / 15.0
        return (recursive(a, m, fa, flm, fm, S_left, depth-1) +
                recursive(m, b, fm, frm, fb, S_right, depth-1))

    fa = f(a)
    fb = f(b)
    fm = f((a + b) / 2.0)
    S = simpson(f, a, b)
    return recursive(a, b, fa, fm, fb, S, max_depth)

# ---------- Wrapper that uses alpha scaling and entropy ----------
def integrate_with_alpha(f, a, b, tol=1e-6, use_entropy=True):
    """
    Integrate f from a to b, using alpha scaling.
    If use_entropy is True, sample the function at 100 points to compute
    entropy and adjust the tolerance accordingly.
    """
    # Coarse sampling to compute entropy
    if use_entropy:
        x_samples = np.linspace(a, b, 100)
        y_samples = [f(x) for x in x_samples]
        H = supertrace_entropy(y_samples)
        # Use entropy to scale tolerance: higher entropy -> lower tolerance (more accurate)
        # Clamp entropy to reasonable range.
        H = max(0.0, min(H, 1.0))
        # Adjust alpha_scale: larger H -> smaller tolerance (more strict)
        alpha_scale = 1.0 + 2.0 * H   # range 1..3
        print(f"Entropy = {H:.4f}, using alpha_scale = {alpha_scale:.2f}")
    else:
        alpha_scale = 1.0

    result = adaptive_simpson(f, a, b, tol=tol, alpha_scale=alpha_scale)
    return result

# ---------- Example usage ----------
if __name__ == "__main__":
    # Define a random integrand (you can change this to any function)
    def f(x):
        return math.sin(x) * math.exp(-0.1 * x)

    a = 0.0
    b = 10.0

    # Integrate with alpha scaling and entropy
    result = integrate_with_alpha(f, a, b, tol=1e-6, use_entropy=True)
    print(f"Integral of f from {a} to {b} = {result:.8f}")

    # Compare with scipy for accuracy (if available)
    try:
        from scipy.integrate import quad
        scipy_result, _ = quad(f, a, b)
        print(f"Scipy reference: {scipy_result:.8f}")
        print(f"Absolute error: {abs(result - scipy_result):.2e}")
    except ImportError:
        print("Scipy not installed, cannot compare.")

def g(x):
    return 1.0 / (1.0 + x**2)
print(integrate_with_alpha(g, 0, 1))
