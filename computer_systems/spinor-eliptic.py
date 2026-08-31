import math
import random
import numpy as np

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)          # ≈ 2.362

# ---------- Linear Möbius sieve (O(K)) ----------
def mobius_sieve(K):
    mu = [0] * (K + 1)
    mu[1] = 1
    primes = []
    is_comp = [False] * (K + 1)
    for i in range(2, K + 1):
        if not is_comp[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K:
                break
            is_comp[i * p] = True
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    return mu

# ---------- Elliptic‑curve‑based permutation (TSP routing) ----------
def elliptic_permutation(K, omega1=PI, omega2=E):
    """
    Generate a permutation of indices 0..K-1 based on the real part of
    the Weierstrass zeta function? For simplicity, we use a deterministic
    pseudo‑angle: θ_i = i * (omega1 - omega2) mod 2π, then sort.
    This mimics a cyclic order on the elliptic curve's real period.
    """
    delta = omega1 - omega2   # π - e ≈ 0.4233
    angles = [(i * delta) % (2 * PI) for i in range(K)]
    order = sorted(range(K), key=lambda i: angles[i])
    return order

# ---------- Generate a test polynomial ----------
def generate_polynomial(K, num_vars=5, seed=42):
    """
    Generate a polynomial with K monomials.
    Each monomial: coeff * ∏_{v=1}^{num_vars} x_v^{exp[v]}
    Exponents are random small integers.
    Returns: list of (coeff, exponents_tuple)
    """
    random.seed(seed)
    monomials = []
    for _ in range(K):
        coeff = random.uniform(-1.0, 1.0)
        exps = tuple(random.randint(0, 3) for _ in range(num_vars))
        monomials.append((coeff, exps))
    return monomials

# ---------- Compression ----------
def compress_polynomial(monomials, mu):
    """
    Keep only monomials whose index (0‑based) has μ(index+1) != 0.
    Returns: compressed list of (original_index, coeff, exps)
    """
    compressed = []
    for idx, (coeff, exps) in enumerate(monomials):
        n = idx + 1   # 1‑based index
        if mu[n] != 0:
            compressed.append((idx, coeff, exps))
    return compressed

# ---------- Reconstruct ----------
def reconstruct_polynomial(monomials, compressed, K):
    """
    Reconstruct the full coefficient array from compressed data.
    Returns: list of (coeff, exps) for all monomials (zero for non‑kept).
    """
    full = [(0.0, exps) for _, exps in monomials]  # keep exponents, set coeff=0
    for idx, coeff, exps in compressed:
        full[idx] = (coeff, exps)
    return full

# ---------- Evaluation ----------
def eval_polynomial(monomials, x_vals):
    """
    Evaluate polynomial at given x_vals (list of values for each variable).
    """
    total = 0.0
    for coeff, exps in monomials:
        term = coeff
        for v, exp in enumerate(exps):
            term *= x_vals[v] ** exp
        total += term
    return total

# ---------- Main ----------
def main():
    K = 1000   # number of monomials (choose a prime if desired)
    num_vars = 6
    print(f"Generating polynomial with {K} monomials in {num_vars} variables...")
    monomials = generate_polynomial(K, num_vars, seed=42)

    # 1. Möbius sieve
    mu = mobius_sieve(K)
    square_free_count = sum(1 for n in range(1, K+1) if mu[n] != 0)
    print(f"Square‑free indices (μ≠0): {square_free_count} out of {K} (ratio {square_free_count/K:.3f})")

    # 2. Optional elliptic permutation (TSP routing)
    order = elliptic_permutation(K, omega1=PI, omega2=E)
    # Reorder monomials according to the elliptic curve order
    monomials_ordered = [monomials[i] for i in order]
    # The compression will now use the new order; we need to keep track of original indices.
    # We'll store the original index (before permutation) in the compressed data.
    # For simplicity, we'll just compress the ordered list, and note that the indices are permuted.
    # We'll store the order as part of the key.

    # 3. Compress
    compressed = compress_polynomial(monomials_ordered, mu)
    print(f"Compressed size: {len(compressed)} monomials (ratio {len(compressed)/K:.3f})")

    # 4. Reconstruct (using the same order to map back)
    # We need to reconstruct the full array in the original order.
    # We'll reconstruct in the ordered space, then invert the permutation.
    full_ordered = [(0.0, exps) for _, exps in monomials_ordered]
    for idx, coeff, exps in compressed:
        full_ordered[idx] = (coeff, exps)
    # Invert the permutation to get back to original order
    inv_order = [0] * K
    for new_idx, orig_idx in enumerate(order):
        inv_order[new_idx] = orig_idx
    full_original = [None] * K
    for new_idx, (coeff, exps) in enumerate(full_ordered):
        orig_idx = inv_order[new_idx]
        full_original[orig_idx] = (coeff, exps)

    # 5. Test evaluation on a random point
    x_vals = [random.uniform(-1, 1) for _ in range(num_vars)]
    val_orig = eval_polynomial(monomials, x_vals)
    val_recon = eval_polynomial(full_original, x_vals)
    error = abs(val_orig - val_recon) / (abs(val_orig) + 1e-12)
    print(f"Relative evaluation error: {error:.4e}")

    # 6. Show sample compressed entries
    print("\nFirst 5 compressed entries (original index, coeff, exponents):")
    for idx, coeff, exps in compressed[:5]:
        print(f"  idx={idx}, coeff={coeff:.4f}, exps={exps}")

if __name__ == "__main__":
    main()