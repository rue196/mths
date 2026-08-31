import numpy as np
import matplotlib.pyplot as plt
import math

# ---------- SuperTrace and Entropy functions (from earlier) ----------
ALPHA = 1.0 / (math.pi - math.e)

def supertrace_from_coeffs(C):
    """Compute alternating sum of |C_i| (i even -> boson +, i odd -> fermion -)."""
    S = 0.0
    for idx, coeff in enumerate(C):
        sign = 1 if (idx % 2 == 0) else -1
        S += sign * abs(coeff)
    return S

def entropy_from_supertrace(S, N, alpha=ALPHA):
    """H = -alpha * (|S|/N) * log(|S|/N)"""
    if S == 0:
        return 0.0
    p = abs(S) / N
    if p <= 0:
        return 0.0
    return -alpha * p * math.log(p)

def invariant_mass(S, H):
    return abs(S) * math.exp(-H)

# ---------- Neutrino oscillation model ----------
def neutrino_oscillation_with_entropy(
        distance_max=1000, num_points=500,
        initial_H=0.5, K=3, seed=42):
    """
    Simulate neutrino oscillation where the mixing angles and mass splittings
    depend on the local entropy H of the surrounding composite particle system.
    The entropy evolves with distance due to interaction.
    """
    np.random.seed(seed)
    
    # ---- 1. Define mass eigenstates and mixing matrix ----
    # We use 3 flavours (electron, muon, tau)
    # The mixing matrix U will be parametrized by three angles,
    # which we make functions of H.
    def mixing_matrix(H):
        # For simplicity, we set angles as linear functions of H.
        # This mimics the effect of the medium on neutrino mixing.
        theta12 = np.radians(30 + 10 * (H - 0.5))
        theta13 = np.radians(5  + 15 * (H - 0.5))
        theta23 = np.radians(45 - 20 * (H - 0.5))
        # Build PMNS-like matrix (real for simplicity)
        c12, s12 = np.cos(theta12), np.sin(theta12)
        c13, s13 = np.cos(theta13), np.sin(theta13)
        c23, s23 = np.cos(theta23), np.sin(theta23)
        U = np.array([
            [c12*c13, s12*c13, s13],
            [-s12*c23 - c12*s13*s23, c12*c23 - s12*s13*s23, c13*s23],
            [s12*s23 - c12*s13*c23, -c12*s23 - s12*s13*c23, c13*c23]
        ])
        return U
    
    # ---- 2. Mass squared differences as functions of H ----
    # We assume they scale with entropy: larger H => larger mass differences
    def mass_splittings(H):
        # Base values in arbitrary units
        dm21_base = 0.5
        dm31_base = 2.0
        # Scale with H (entropy)
        return dm21_base * (1 + 0.5*H), dm31_base * (1 + 0.5*H)
    
    # ---- 3. Entropy evolution with distance ----
    # We model the composite particle's entropy as slowly increasing
    # due to energy exchange with the neutrino (like the second law).
    # We'll use a logistic-like increase from initial_H to a max.
    H_max = 1.2
    H0 = initial_H
    L = np.linspace(0, distance_max, num_points)
    # Entropy grows with distance and saturates
    H = H0 + (H_max - H0) * (1 - np.exp(-0.003 * L))
    
    # ---- 4. Initial neutrino state (from beta decay) ----
    # We assume a neutron decay produces an electron antineutrino
    # which is a superposition of mass eigenstates.
    # At L=0, we set the flavour state as pure electron neutrino.
    flavour_init = np.array([1.0, 0.0, 0.0])  # electron neutrino
    
    # ---- 5. Propagation ----
    # For each distance, compute the evolved flavour probabilities.
    prob = np.zeros((num_points, 3))
    total_prob = np.zeros(num_points)
    
    for i, l in enumerate(L):
        U = mixing_matrix(H[i])
        dm21, dm31 = mass_splittings(H[i])
        # Convert initial flavour to mass basis
        mass_init = U.T @ flavour_init  # real orthogonal -> inverse is transpose
        # Phases: phi1=0 (common), phi2 = -dm21 * l / (2E), phi3 = -dm31 * l / (2E)
        E = 1.0  # arbitrary energy
        phases = np.array([0.0, -dm21 * l / (2*E), -dm31 * l / (2*E)])
        mass_evolved = mass_init * np.exp(1j * phases)
        # Back to flavour basis
        flavour_evolved = U @ mass_evolved
        prob[i, :] = np.abs(flavour_evolved)**2
        total_prob[i] = np.sum(prob[i, :])
    
    # ---- 6. Plot results ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Entropy vs distance
    axes[0,0].plot(L, H, 'g-', linewidth=2)
    axes[0,0].set_xlabel('Distance')
    axes[0,0].set_ylabel('Entropy H')
    axes[0,0].set_title('Entropy of composite system (second law)')
    axes[0,0].grid(True)
    
    # Flavour probabilities
    axes[0,1].plot(L, prob[:,0], 'b-', label='Electron (ν_e)')
    axes[0,1].plot(L, prob[:,1], 'r-', label='Muon (ν_μ)')
    axes[0,1].plot(L, prob[:,2], 'orange', label='Tau (ν_τ)')
    axes[0,1].set_xlabel('Distance')
    axes[0,1].set_ylabel('Probability')
    axes[0,1].set_title('Neutrino flavour oscillation')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # Total probability (should be 1)
    axes[1,0].plot(L, total_prob, 'k--', linewidth=2)
    axes[1,0].axhline(1, color='gray', linestyle=':')
    axes[1,0].set_xlabel('Distance')
    axes[1,0].set_ylabel('Total probability')
    axes[1,0].set_title('Invariant spinor projection (|ν|² = 1)')
    axes[1,0].grid(True)
    
    # Phase space: probability of electron vs entropy
    axes[1,1].scatter(H, prob[:,0], c=L, cmap='viridis', s=10)
    axes[1,1].set_xlabel('Entropy H')
    axes[1,1].set_ylabel('P(ν_e)')
    axes[1,1].set_title('Electron probability vs entropy')
    axes[1,1].grid(True)
    plt.colorbar(axes[1,1].collections[0], ax=axes[1,1], label='Distance')
    
    plt.tight_layout()
    plt.show()
    
    # ---- 7. Print some values ----
    print(f"Initial H = {H[0]:.3f}, final H = {H[-1]:.3f}")
    print(f"Final probabilities: ν_e={prob[-1,0]:.3f}, ν_μ={prob[-1,1]:.3f}, ν_τ={prob[-1,2]:.3f}")
    print(f"Max deviation from total probability = 1: {np.max(np.abs(total_prob - 1)):.2e}")

# ---------- Run the simulation ----------
if __name__ == "__main__":
    neutrino_oscillation_with_entropy(distance_max=800, num_points=800, initial_H=0.4)