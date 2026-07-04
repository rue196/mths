import math
import random
import cmath

# ------------------------------------------------------------
# 1. Generate K addresses as (x, y) real numbers, then treat as complex z = x + iy
# ------------------------------------------------------------
def generate_addresses(K, seed=42):
    random.seed(seed)
    addresses = [(random.uniform(0.5, 5), random.uniform(0.5, 5)) for _ in range(K)]  # avoid zero
    return addresses

# ------------------------------------------------------------
# 2. Build the 2x2 matrix M(x,y) for a given integer exponent i
#    Returns a tuple (trace, determinant) for convenience.
# ------------------------------------------------------------
def build_matrix(x, y, i):
    # M = [[x^(i-1), x^(-1)*y^i],
    #      [y^(-1)*x^i, y^(i-1)]]
    m11 = x ** (i-1)
    m12 = (x ** (-1)) * (y ** i)
    m21 = (y ** (-1)) * (x ** i)
    m22 = y ** (i-1)
    trace = m11 + m22
    det = m11 * m22 - m12 * m21
    return trace, det

# ------------------------------------------------------------
# 3. TSP routing: sort addresses by the argument (angle) of z = x+iy
#    This gives a cyclic order optimal for points on a circle.
# ------------------------------------------------------------
def tsp_angle_route(addresses):
    # Convert to complex, compute angle, and sort
    return sorted(addresses, key=lambda p: (cmath.phase(complex(p[0], p[1])), p[0], p[1]))

# ------------------------------------------------------------
# 4. Entropy diagnostic using the trace of M as a probability weight
#    We use the absolute value of the trace (or a positive shift) to form probabilities.
# ------------------------------------------------------------
def entropy_diagnostic(route, i_exp=2, alpha=0.3628, collatz_even=True):
    traces = []
    for (x, y) in route:
        tr, _ = build_matrix(x, y, i_exp)
        traces.append(abs(tr) + 1e-8)   # ensure positivity
    # Normalize to get probabilities
    sum_tr = sum(traces)
    probs = [t / sum_tr for t in traces]
    K = len(probs)
    # Build C_i = -α * p_i * log(p_i) / i  with Collatz mask (even indices only)
    C = [0.0] * K
    for idx, p in enumerate(probs, start=1):
        if collatz_even and idx % 2 != 0:
            C[idx-1] = 0.0
        else:
            C[idx-1] = -alpha * p * math.log(p + 1e-12) / idx
    # Spectral entropy = (1/α) * Σ i * C_i
    sum_iCi = sum((idx+1) * C[idx] for idx in range(K))
    spectral_entropy = sum_iCi / alpha
    shannon = -sum(p * math.log(p + 1e-12) for p in probs)
    return spectral_entropy, shannon, traces

# ------------------------------------------------------------
# 5. Main demonstration
# ------------------------------------------------------------
def main():
    K = 20
    addresses = generate_addresses(K)
    print("Generated (x,y) addresses (first 5):", addresses[:5])
    
    # TSP route by angle
    route = tsp_angle_route(addresses)
    print("\nTSP route (angle‑sorted, first 5):", route[:5])
    
    # Choose exponent i (e.g., 2)
    i_exp = 2
    spec_ent, shan_ent, traces = entropy_diagnostic(route, i_exp)
    
    print(f"\nUsing exponent i = {i_exp}")
    print(f"Traces of M (first 5): {[round(t, 3) for t in traces[:5]]}")
    print(f"Spectral entropy (Im ζ'(0)): {spec_ent:.6f}")
    print(f"Shannon entropy: {shan_ent:.6f}")
    print(f"Difference: {spec_ent - shan_ent:.6f}")

if __name__ == "__main__":
    main()

input('Press ENTER to exit')