import math
import cmath
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import List, Tuple, Optional
from itertools import permutations

# ============================================================
# 1. Spinor Projection Operator (from previous code)
# ============================================================
def generate_vertices(seed=42):
    np.random.seed(seed)
    return np.random.randn(12, 6)

def epsilon_tensor():
    eps = {}
    for perm in permutations(range(6)):
        if len(set(perm)) != 6:
            eps[perm] = 0
            continue
        inv = sum(1 for i in range(6) for j in range(i+1,6) if perm[i] > perm[j])
        eps[perm] = (-1)**inv
    return eps

def spinor_projection(vertices, eps):
    v = vertices[:6]  # use first 6 vertices for contraction
    scalar = 0.0
    for perm, sign in eps.items():
        if sign == 0:
            continue
        prod = 1.0
        for idx, coord_idx in enumerate(perm):
            prod *= v[idx][coord_idx]
        scalar += sign * prod
    return scalar

def rotate_vertices(vertices, theta):
    rotated = vertices.copy()
    for v in rotated:
        for pair in [(0,1), (2,3), (4,5)]:
            x, y = v[pair[0]], v[pair[1]]
            v[pair[0]] = x * math.cos(theta) - y * math.sin(theta)
            v[pair[1]] = x * math.sin(theta) + y * math.cos(theta)
    return rotated

def projection_features(vertices, num_theta=256):
    eps = epsilon_tensor()
    theta_vals = np.linspace(0, 2*np.pi, num_theta)
    proj = []
    for theta in theta_vals:
        rot = rotate_vertices(vertices, theta)
        proj.append(spinor_projection(rot, eps))
    proj = np.array(proj)
    var = np.var(proj)
    mean = np.mean(proj)
    # spectral compression (as before)
    K = len(proj)
    sigma = 2.0
    kernel = np.array([math.exp(-(d*d)/(2*sigma*sigma)) for d in range(-(K-1), K)])
    N = 1 << (2*K - 1).bit_length()
    sig_pad = np.pad(proj, (0, N - K))
    ker_pad = np.pad(kernel, (0, N - (2*K - 1)))
    conv = np.fft.ifft(np.fft.fft(sig_pad) * np.fft.fft(ker_pad))[:K]
    M = int(0.3 * K)
    idx = np.argsort(np.abs(conv))[::-1][:M]
    recon = np.zeros(K, dtype=complex)
    for i in idx:
        recon[i] = conv[i]
    error = np.linalg.norm(recon - conv)
    return var, mean, error

# ============================================================
# 2. Zeta function and supertrace (adapted from sim.py)
# ============================================================
ALPHA = 1.0 / (math.pi - math.e)   # ≈ 2.362

def generate_coefficients(K: int, seed: bytes) -> List[complex]:
    """
    Generate 2K+1 complex coefficients C_i (i=-K..K).
    Seed is derived from projection features to link geometry to state.
    """
    random.seed(seed)
    C = [complex(random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(2*K+1)]
    # Enforce symmetry for a real spectrum: C_{-i} = C_i^*
    for i in range(1, K+1):
        C[K - i] = C[K + i].conjugate()
    C[K] = C[K].real   # i=0 must be real
    return C

def zeta(t: float, C: List[complex], alpha: float = ALPHA) -> complex:
    """Evaluate ζ(t) = Σ_i C_i * exp(i * t * i / α)"""
    K = (len(C) - 1) // 2
    total = 0.0 + 0.0j
    for idx, coeff in enumerate(C):
        i = idx - K
        phase = t * i / alpha
        total += coeff * complex(math.cos(phase), math.sin(phase))
    return total

def supertrace_from_coeffs(C: List[complex]) -> float:
    """
    Compute the supertrace as the alternating sum of the magnitudes of the coefficients.
    Fermions (odd i) enter with negative sign, bosons (even i) with positive sign.
    """
    K = (len(C) - 1) // 2
    S = 0.0
    for idx, coeff in enumerate(C):
        i = idx - K
        sign = 1 if (i % 2 == 0) else -1   # even → boson (+), odd → fermion (-)
        S += sign * abs(coeff)
    return S

def entropy_from_supertrace(S: float, N: int, alpha: float = ALPHA) -> float:
    """Entropy H = -α * (|S|/N) * log(|S|/N)"""
    if S == 0:
        return 0.0
    p = abs(S) / N
    if p <= 0:
        return 0.0
    return -alpha * p * math.log(p)

def invariant_mass(S: float, H: float) -> float:
    """m = |S| * exp(-H)"""
    return abs(S) * math.exp(-H)

# ============================================================
# 3. Main simulation: combine projection and zeta dynamics
# ============================================================
def run_simulation(K: int = 5, steps: int = 150, decay_rate: float = 0.02,
                   seed_vertices: int = 42, seed_coeff: int = 123):
    """
    Simulate the evolution of the composite system under weak decay.
    The initial coefficients are seeded using the spinor projection features
    to link the geometry to the state.
    """
    # ---- 1. Generate public vertices and extract projection invariants ----
    vertices = generate_vertices(seed_vertices)
    var, mean, err = projection_features(vertices)
    print(f"Projection features: var={var:.4f}, mean={mean:.4f}, err={err:.4f}")
    
    # ---- 2. Seed coefficients using these features ----
    # Combine features into a seed for reproducibility
    seed_bytes = f"{var}_{mean}_{err}_{seed_coeff}".encode()
    C = generate_coefficients(K, seed_bytes)
    N = 2*K + 1
    
    # Store history
    hist_S = []
    hist_H = []
    hist_m = []
    hist_zeta_real = []
    hist_zeta_abs = []
    
    # Time points for evaluating zeta (we'll take the last time point as the current state)
    t_values = np.linspace(0, 10, 50)
    
    for step in range(steps):
        # Compute current supertrace, entropy, mass
        S = supertrace_from_coeffs(C)
        H = entropy_from_supertrace(S, N)
        m = invariant_mass(S, H)
        hist_S.append(S)
        hist_H.append(H)
        hist_m.append(m)
        
        # Evaluate zeta at a fixed t (e.g., t=step/10) to see evolution
        t_now = step * 0.1
        z = zeta(t_now, C)
        hist_zeta_real.append(z.real)
        hist_zeta_abs.append(abs(z))
        
        # ---- 3. Apply weak decay: reduce fermionic coefficients (odd i) ----
        for idx in range(len(C)):
            i = idx - K
            if i % 2 != 0:   # odd → fermion
                C[idx] *= (1 - decay_rate)
        # Optional: renormalize to keep overall scale? Not needed for decay.
    
    # ---- 4. Plot results ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Supertrace
    axes[0,0].plot(range(steps), hist_S, 'purple', label='Supertrace S')
    axes[0,0].axhline(0, color='black', linestyle='--')
    axes[0,0].set_ylabel('S')
    axes[0,0].set_title('Supertrace (alternating sum)')
    axes[0,0].grid(True)
    
    # Entropy
    axes[0,1].plot(range(steps), hist_H, 'orange', label='Entropy H')
    axes[0,1].set_ylabel('H')
    axes[0,1].set_title('Entropy (second law)')
    axes[0,1].grid(True)
    
    # Mass
    axes[0,2].plot(range(steps), hist_m, 'red', label='Mass m = |S| exp(-H)')
    # Fit exponential decay
    def exp_decay(t, A, lam):
        return A * np.exp(-lam * t)
    try:
        popt, _ = curve_fit(exp_decay, np.arange(steps), hist_m, p0=(hist_m[0], 0.01))
        A_fit, lam_fit = popt
        half_life = np.log(2) / lam_fit
        axes[0,2].plot(range(steps), exp_decay(np.arange(steps), A_fit, lam_fit),
                       'b--', label=f'Fit: λ={lam_fit:.3f}, T½={half_life:.1f}')
    except:
        pass
    axes[0,2].set_ylabel('Mass')
    axes[0,2].set_title('Mass decay (weak interaction)')
    axes[0,2].legend()
    axes[0,2].grid(True)
    
    # Zeta real part
    axes[1,0].plot(range(steps), hist_zeta_real, 'green', label='Re ζ(t)')
    axes[1,0].set_xlabel('Time step')
    axes[1,0].set_ylabel('Re ζ')
    axes[1,0].set_title('Zeta function (real part)')
    axes[1,0].grid(True)
    
    # Zeta magnitude
    axes[1,1].plot(range(steps), hist_zeta_abs, 'blue', label='|ζ(t)|')
    axes[1,1].set_xlabel('Time step')
    axes[1,1].set_ylabel('|ζ|')
    axes[1,1].set_title('Zeta magnitude')
    axes[1,1].grid(True)
    
    # Phase space: mass vs entropy
    axes[1,2].scatter(hist_H, hist_m, c=range(steps), cmap='viridis', s=10)
    axes[1,2].set_xlabel('Entropy H')
    axes[1,2].set_ylabel('Mass')
    axes[1,2].set_title('Mass vs Entropy (decay trajectory)')
    axes[1,2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print final values
    print(f"\nFinal: S={hist_S[-1]:.4f}, H={hist_H[-1]:.4f}, m={hist_m[-1]:.4f}")
    if 'half_life' in locals():
        print(f"Decay constant λ={lam_fit:.4f}, Half-life T½={half_life:.1f} steps")

if __name__ == "__main__":
    run_simulation(K=5, steps=150, decay_rate=0.02)
