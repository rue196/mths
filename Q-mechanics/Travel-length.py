import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
import math

# ---------- Spinor projection functions (unchanged) ----------
def epsilon_tensor():
    eps = {}
    for perm in permutations(range(6)):
        if len(set(perm)) != 6:
            eps[perm] = 0
            continue
        inv = sum(1 for i in range(6) for j in range(i+1, 6) if perm[i] > perm[j])
        eps[perm] = (-1)**inv
    return eps

EPS = epsilon_tensor()

def spinor_projection(vertices):
    """Compute Π = ε_{i1...i6} ∏_{k=1}^6 v_{k, i_k} using first 6 vertices."""
    v = vertices[:6]  # take first 6 vertices (12 total, but we only need 6 for contraction)
    scalar = 0.0
    for perm, sign in EPS.items():
        if sign == 0:
            continue
        prod = 1.0
        for idx, coord_idx in enumerate(perm):
            prod *= v[idx][coord_idx]
        scalar += sign * prod
    return scalar

def rotate_vertices(vertices, theta):
    """Apply a global SO(6) rotation (block-diagonal SO(2) rotations)."""
    rotated = vertices.copy()
    for v in rotated:
        for pair in [(0,1), (2,3), (4,5)]:
            x, y = v[pair[0]], v[pair[1]]
            v[pair[0]] = x * math.cos(theta) - y * math.sin(theta)
            v[pair[1]] = x * math.sin(theta) + y * math.cos(theta)
    return rotated

# ---------- Simulation ----------
def run_simulation(n_samples=500, scale_range=(0.5, 5.0), seed=42):
    np.random.seed(seed)
    
    travel_lengths = []
    contraction_lengths = []
    energies = []
    # We'll also keep track of rotated versions to show invariance
    rot_contractions = []
    
    for _ in range(n_samples):
        # Generate 12 random vertices in R^6
        vertices = np.random.randn(12, 6)
        
        # Scale them to achieve a desired travel length L
        # We'll choose L uniformly in scale_range
        L = np.random.uniform(*scale_range)
        vertices *= L / np.linalg.norm(vertices)   # normalize to norm = L
        
        # Compute the contraction scalar
        Pi = spinor_projection(vertices)
        # Contraction length = |Pi|^(1/6)
        ell_c = abs(Pi) ** (1/6)
        # Energy = 1 / ell_c
        E = 1.0 / ell_c if ell_c > 1e-12 else np.inf
        
        travel_lengths.append(L)
        contraction_lengths.append(ell_c)
        energies.append(E)
        
        # Test rotation invariance: rotate by a random angle and recompute
        theta_rand = np.random.uniform(0, 2*np.pi)
        rot_verts = rotate_vertices(vertices, theta_rand)
        Pi_rot = spinor_projection(rot_verts)
        ell_rot = abs(Pi_rot) ** (1/6)
        rot_contractions.append(ell_rot)
    
    # ---- Plotting ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Travel length vs contraction length
    axes[0,0].scatter(travel_lengths, contraction_lengths, s=10, alpha=0.6, color='blue')
    axes[0,0].set_xlabel('Travel length L (norm of vertices)')
    axes[0,0].set_ylabel('Contraction length ℓ_c = |Π|^(1/6)')
    axes[0,0].set_title('Travel length vs. SU(3) contraction length')
    axes[0,0].grid(True)
    
    # 2. Energy vs travel length
    axes[0,1].scatter(travel_lengths, energies, s=10, alpha=0.6, color='red')
    axes[0,1].set_xlabel('Travel length L')
    axes[0,1].set_ylabel('Energy E = 1/ℓ_c')
    axes[0,1].set_title('Energy required to mediate into field')
    axes[0,1].grid(True)
    
    # 3. Distribution of contraction lengths (showing typical scale)
    axes[1,0].hist(contraction_lengths, bins=30, density=True, alpha=0.7, color='green')
    axes[1,0].set_xlabel('Contraction length ℓ_c')
    axes[1,0].set_ylabel('Density')
    axes[1,0].set_title('Distribution of contraction lengths')
    
    # 4. Rotation invariance: compare original vs rotated contraction
    axes[1,1].scatter(contraction_lengths, rot_contractions, s=10, alpha=0.6, color='purple')
    axes[1,1].plot([0, max(contraction_lengths)], [0, max(contraction_lengths)], 'k--', label='y = x')
    axes[1,1].set_xlabel('ℓ_c (original)')
    axes[1,1].set_ylabel('ℓ_c (rotated)')
    axes[1,1].set_title('Invariance under SO(6) rotation')
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print some stats
    print(f"Average contraction length: {np.mean(contraction_lengths):.3f}")
    print(f"Average energy: {np.mean(energies):.3f}")
    print("Rotation invariance: mean relative difference =",
          np.mean(np.abs(np.array(contraction_lengths) - np.array(rot_contractions)) / np.array(contraction_lengths)))

if __name__ == "__main__":
    run_simulation(n_samples=800, scale_range=(0.5, 5.0))
