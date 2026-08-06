import numpy as np
import matplotlib.pyplot as plt

def neutrino_oscillation_primes(K=3, distance_max=1e6, num_points=500, seed=42):
    """
    Simulate neutrino oscillation with K flavours.
    Each flavour i (i=0..K-1) has a mass squared difference proportional to a prime.
    The travel distance is from 0 to distance_max.
    The mixing matrix is a random orthogonal matrix (SO(K)).
    The invariant total probability is computed to show it remains 1.
    """
    np.random.seed(seed)
    
    # ---- 1. Choose primes for mass squared differences ----
    # First K primes: 2,3,5,7,11,...
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29][:K]
    # Optionally use 1 as the first to avoid zero frequency
    # For more realistic, we set the smallest to 1.
    # We'll use primes but scale them.
    mass_diff = np.array(primes)  # Δm^2 in arbitrary units
    # Normalize so that the first is 1 (optional)
    # mass_diff = mass_diff / mass_diff[0]  # now 1, 1.5, 2.5, ...
    
    # ---- 2. Define mixing matrix (real orthogonal) ----
    # Generate a random orthogonal matrix via QR decomposition
    A = np.random.randn(K, K)
    Q, R = np.linalg.qr(A)
    # Ensure det(Q)=1 for SO(K)
    if np.linalg.det(Q) < 0:
        Q[:,0] *= -1
    U = Q  # mixing matrix (flavour to mass basis)
    
    # ---- 3. Initial state: electron neutrino (flavour 0) ----
    flavour_init = np.zeros(K)
    flavour_init[0] = 1.0
    # Convert to mass basis
    mass_init = U.T @ flavour_init   # U is real, so U.T is inverse
    
    # ---- 4. Propagation distances ----
    L = np.linspace(0, distance_max, num_points)
    
    # Energy (arbitrary)
    E = 1.0
    
    # Precompute phases: φ_i = - (Δm_i^2 * L) / (2E)
    # We'll include a common phase factor, but ignore it.
    # For each distance, compute the evolved flavour state.
    prob = np.zeros((num_points, K))
    
    for idx, l in enumerate(L):
        # Mass eigenstate evolution
        phases = np.exp(-1j * mass_diff * l / (2 * E))  # i = sqrt(-1)
        mass_evolved = mass_init * phases
        # Convert back to flavour basis
        flavour_evolved = U @ mass_evolved
        # Probabilities
        prob[idx, :] = np.abs(flavour_evolved)**2
    
    # ---- 5. Compute invariant projection (total probability) ----
    total_prob = np.sum(prob, axis=1)  # should be 1
    
    # ---- 6. Plot results ----
    plt.figure(figsize=(12, 6))
    for i in range(K):
        plt.plot(L, prob[:, i], label=f'Flavour {i+1}')
    plt.plot(L, total_prob, 'k--', linewidth=2, label='Total probability (invariant)')
    plt.xlabel('Distance (arbitrary units)')
    plt.ylabel('Probability')
    plt.title(f'Neutrino oscillation with {K} flavours, mass differences = {primes[:K]}')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # ---- 7. Print invariant check ----
    print(f"Maximum deviation from total probability = 1: {np.max(np.abs(total_prob - 1)):.2e}")
    print("The total probability (spinor projection) is invariant under SO(K) rotations.")

# ---- Run for 3 flavours ----
if __name__ == "__main__":
    neutrino_oscillation_primes(K=3, distance_max=1e6, num_points=1000)