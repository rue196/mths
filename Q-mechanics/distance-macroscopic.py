import numpy as np
import matplotlib.pyplot as plt
import math

# ---------- SuperTrace and Entropy functions ----------
ALPHA = 1.0 / (math.pi - math.e)

def supertrace_from_coeffs(C):
    S = 0.0
    for idx, coeff in enumerate(C):
        sign = 1 if (idx % 2 == 0) else -1
        S += sign * abs(coeff)
    return S

def entropy_from_supertrace(S, N, alpha=ALPHA):
    if S == 0:
        return 0.0
    p = abs(S) / N
    if p <= 0:
        return 0.0
    return -alpha * p * math.log(p)

def contraction_length(S):
    """ℓ_c = |S|^(1/6) (spinor projection contraction)."""
    return abs(S) ** (1/6)

# ---------- Cosmological extension ----------
def cosmology_from_neutrinos(distance_max=10000, num_points=1000,
                             initial_H=0.4, seed=42):
    """
    Simulate neutrino oscillations and compute dark energy/dark matter densities
    based on entropy H and supertrace S.
    """
    np.random.seed(seed)
    
    # ---- 1. Neutrino oscillation model (similar to previous) ----
    # We'll reuse the same entropy evolution and mixing matrix.
    def mixing_matrix(H):
        theta12 = np.radians(30 + 10 * (H - 0.5))
        theta13 = np.radians(5  + 15 * (H - 0.5))
        theta23 = np.radians(45 - 20 * (H - 0.5))
        c12, s12 = np.cos(theta12), np.sin(theta12)
        c13, s13 = np.cos(theta13), np.sin(theta13)
        c23, s23 = np.cos(theta23), np.sin(theta23)
        U = np.array([
            [c12*c13, s12*c13, s13],
            [-s12*c23 - c12*s13*s23, c12*c23 - s12*s13*s23, c13*s23],
            [s12*s23 - c12*s13*c23, -c12*s23 - s12*s13*c23, c13*c23]
        ])
        return U
    
    def mass_splittings(H):
        dm21_base = 0.5
        dm31_base = 2.0
        return dm21_base * (1 + 0.5*H), dm31_base * (1 + 0.5*H)
    
    # Entropy evolution
    H_max = 1.2
    H0 = initial_H
    L = np.linspace(0, distance_max, num_points)
    H = H0 + (H_max - H0) * (1 - np.exp(-0.0005 * L))  # slower growth for cosmology
    
    # Initial neutrino state (electron neutrino)
    flavour_init = np.array([1.0, 0.0, 0.0])
    prob = np.zeros((num_points, 3))
    total_prob = np.zeros(num_points)
    
    # We also need the supertrace S. We'll simulate a composite particle's
    # coefficients evolving with distance. For simplicity, we set S as a
    # decreasing function of H: as entropy increases, the supertrace magnitude
    # decreases (since fermions and bosons balance more).
    # We'll generate a synthetic S that decreases with H.
    S = np.zeros(num_points)
    # We'll also compute contraction length ℓ_c.
    ell_c = np.zeros(num_points)
    
    for i, l in enumerate(L):
        # Oscillation part
        U = mixing_matrix(H[i])
        dm21, dm31 = mass_splittings(H[i])
        mass_init = U.T @ flavour_init
        E = 1.0
        phases = np.array([0.0, -dm21 * l / (2*E), -dm31 * l / (2*E)])
        mass_evolved = mass_init * np.exp(1j * phases)
        flavour_evolved = U @ mass_evolved
        prob[i, :] = np.abs(flavour_evolved)**2
        total_prob[i] = np.sum(prob[i, :])
        
        # Compute supertrace S from the current neutrino state?
        # We treat the neutrino itself as the composite system:
        # its flavour state vector (complex amplitudes) can be used to compute S.
        # For three flavours, we can take the amplitudes as the coefficients C_i.
        # We'll use the flavour amplitudes (complex) as C_i.
        C = flavour_evolved
        S[i] = supertrace_from_coeffs(C)
        ell_c[i] = contraction_length(S[i])
    
    # ---- 2. Cosmological densities ----
    # Dark energy: inversely proportional to entropy H
    rho_Lambda = 1.0 / (H + 1e-12)  # avoid division by zero
    # Normalize to make the total density constant
    rho_DM = 1.0 / (ell_c + 1e-12)  # inverse contraction length -> dark matter
    # Total density: rho_total = rho_Lambda + rho_DM
    # We'll rescale so that rho_total = constant (critical density)
    rho_total = rho_Lambda + rho_DM
    # Normalize such that at early times (small L) the total is 1
    scale = rho_total[0]
    rho_Lambda /= scale
    rho_DM /= scale
    rho_total /= scale
    
    # ---- 3. Plotting ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Entropy and SuperTrace
    axes[0,0].plot(L, H, 'g-', label='Entropy H')
    axes[0,0].set_xlabel('Cosmic distance')
    axes[0,0].set_ylabel('H')
    axes[0,0].set_title('Entropy evolution')
    axes[0,0].grid(True)
    
    axes[0,1].plot(L, S, 'purple', label='Supertrace S')
    axes[0,1].set_xlabel('Cosmic distance')
    axes[0,1].set_ylabel('S')
    axes[0,1].set_title('Supertrace (from neutrino amplitudes)')
    axes[0,1].grid(True)
    
    axes[0,2].plot(L, ell_c, 'brown', label='Contraction length ℓ_c')
    axes[0,2].set_xlabel('Cosmic distance')
    axes[0,2].set_ylabel('ℓ_c')
    axes[0,2].set_title('Contraction length (|S|^{1/6})')
    axes[0,2].grid(True)
    
    # Neutrino probabilities
    axes[1,0].plot(L, prob[:,0], 'b-', label='ν_e')
    axes[1,0].plot(L, prob[:,1], 'r-', label='ν_μ')
    axes[1,0].plot(L, prob[:,2], 'orange', label='ν_τ')
    axes[1,0].set_xlabel('Cosmic distance')
    axes[1,0].set_ylabel('Probability')
    axes[1,0].set_title('Neutrino flavours')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # Cosmological densities
    axes[1,1].plot(L, rho_Lambda, '--', color='blue', label='Dark energy (1/H)')
    axes[1,1].plot(L, rho_DM, '--', color='red', label='Dark matter (1/ℓ_c)')
    axes[1,1].plot(L, rho_total, 'k-', linewidth=2, label='Total (critical)')
    axes[1,1].set_xlabel('Cosmic distance')
    axes[1,1].set_ylabel('Energy density (normalised)')
    axes[1,1].set_title('ΛCDM densities')
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    # Ratio dark energy / dark matter
    ratio = rho_Lambda / (rho_DM + 1e-12)
    axes[1,2].plot(L, ratio, 'm-', label='ρ_Λ / ρ_DM')
    axes[1,2].set_xlabel('Cosmic distance')
    axes[1,2].set_ylabel('Ratio')
    axes[1,2].set_title('Dark energy vs dark matter')
    axes[1,2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # ---- 4. Print summary ----
    print(f"Initial H = {H[0]:.3f}, final H = {H[-1]:.3f}")
    print(f"Initial S = {S[0]:.3f}, final S = {S[-1]:.3f}")
    print(f"Initial ℓ_c = {ell_c[0]:.3f}, final ℓ_c = {ell_c[-1]:.3f}")
    print(f"Final densities: ρ_Λ={rho_Lambda[-1]:.3f}, ρ_DM={rho_DM[-1]:.3f}")
    print(f"Ratio ρ_Λ/ρ_DM at final: {ratio[-1]:.3f}")
    print(f"Max deviation total probability = {np.max(np.abs(total_prob - 1)):.2e}")

if __name__ == "__main__":
    cosmology_from_neutrinos(distance_max=20000, num_points=1000, initial_H=0.4)