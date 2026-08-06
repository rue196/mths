import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math
import hashlib
import struct
from typing import List, Tuple

# ---------- core functions (as before) ----------
def prng_bytes(seed: bytes, length: int) -> bytes:
    return hashlib.shake_256(seed).digest(length)

def generate_addresses(K: int, seed: bytes) -> List[Tuple[float, float]]:
    raw = prng_bytes(seed, K * 16)
    points = []
    for i in range(K):
        x = struct.unpack('>d', raw[16*i:16*i+8])[0]
        y = struct.unpack('>d', raw[16*i+8:16*i+16])[0]
        x = 0.5 + (x - math.floor(x)) * 4.5
        y = 0.5 + (y - math.floor(y)) * 4.5
        points.append((x, y))
    return points

def build_supermatrix(points: List[Tuple[float, float]], N: int) -> np.ndarray:
    M = np.zeros((N, N))
    for t in range(1, N+1):
        for j, (x, y) in enumerate(points[:N]):
            M[t-1, j] = (x ** t) + (y ** t)
    return M

def supertrace(M: np.ndarray) -> float:
    N = M.shape[0]
    trace = 0.0
    for t in range(N):
        val = M[t, t]
        if (t+1) % 2 == 0:   # bosonic (even)
            trace += val
        else:                # fermionic (odd)
            trace -= val
    return trace

def matrix_entropy(M: np.ndarray, alpha: float = 1.0/(math.pi - math.e)) -> float:
    S = abs(supertrace(M))
    if S == 0:
        return 0.0
    p = S / M.shape[0]
    if p <= 0:
        return 0.0
    return -alpha * p * math.log(p)

def strong_mass(M: np.ndarray) -> float:
    S = abs(supertrace(M))
    H = matrix_entropy(M)
    return S * math.exp(-H)

# ---------- decay simulation ----------
def run_decay_simulation(K=1, decay_rate=0.02, steps=10, seed=13):
    """
    Simulate weak decay: at each step, multiply fermion diagonal elements
    by (1 - decay_rate), mimicking conversion to bosons or loss.
    This reduces the supertrace magnitude over time, causing mass decay.
    """
    # Initialise matrix
    addr_seed = b'weak_decay_seed'
    points = generate_addresses(K, addr_seed)
    M = build_supermatrix(points, K)
    
    # Store history
    masses = []
    entropies = []
    straces = []
    fermion_contrib = []   # sum of odd diagonal (absolute)
    boson_contrib = []     # sum of even diagonal (absolute)
    
    for step in range(steps):
        # Compute current quantities
        st = supertrace(M)
        H = matrix_entropy(M)
        m = strong_mass(M)
        masses.append(m)
        entropies.append(H)
        straces.append(st)
        
        # Fermion and boson contributions (diagonal)
        diag = np.diag(M)
        fermion_sum = sum(abs(diag[t]) for t in range(K) if (t+1) % 2 == 1)
        boson_sum = sum(abs(diag[t]) for t in range(K) if (t+1) % 2 == 0)
        fermion_contrib.append(fermion_sum)
        boson_contrib.append(boson_sum)
        
        # Apply decay: reduce fermion diagonal elements by factor (1 - decay_rate)
        # This simulates weak interaction turning fermions into bosons or dissipating them.
        for t in range(K):
            if (t+1) % 2 == 1:  # odd -> fermion
                M[t, t] *= (1 - decay_rate)
        # Optionally, we could also increase boson elements to conserve total?
        # For simplicity, we only decrease fermions; this mimics decay.
    
    # ---- Plotting ----
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    # Mass vs time
    axes[0,0].plot(range(steps), masses, 'r-', label='Mass')
    axes[0,0].set_ylabel('Mass scale')
    axes[0,0].set_title('Mass decay (weak interaction)')
    axes[0,0].grid(True)
    
    # Fit exponential decay to mass (for half-life)
    def exp_decay(t, A, lam):
        return A * np.exp(-lam * t)
    try:
        popt, _ = curve_fit(exp_decay, np.arange(steps), masses, p0=(masses[0], 0.01))
        A_fit, lam_fit = popt
        half_life = np.log(2) / lam_fit
        axes[0,0].plot(range(steps), exp_decay(np.arange(steps), A_fit, lam_fit), 'b--', 
                       label=f'Fit: λ={lam_fit:.3f}, T½={half_life:.1f} steps')
    except:
        pass
    axes[0,0].legend()
    
    # Entropy vs time (should increase to equilibrium)
    axes[0,1].plot(range(steps), entropies, 'g-', label='Entropy H')
    axes[0,1].axhline(y=max(entropies)*0.9, color='k', linestyle=':', label='Equilibrium approx')
    axes[0,1].set_ylabel('Entropy')
    axes[0,1].set_title('Entropy increase (second law)')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # Supertrace vs time
    axes[1,0].plot(range(steps), straces, 'purple', label='Supertrace STr')
    axes[1,0].axhline(0, color='black', linestyle='--')
    axes[1,0].set_ylabel('Supertrace')
    axes[1,0].set_title('Supertrace evolution')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # Fermion vs Boson contributions
    axes[1,1].plot(range(steps), fermion_contrib, 'orange', label='Fermion sum (odd)')
    axes[1,1].plot(range(steps), boson_contrib, 'blue', label='Boson sum (even)')
    axes[1,1].set_ylabel('Absolute diagonal sum')
    axes[1,1].set_title('Fermion / Boson balance')
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    # Mass vs entropy (phase space)
    axes[2,0].scatter(entropies, masses, c=range(steps), cmap='viridis', s=10)
    axes[2,0].set_xlabel('Entropy H')
    axes[2,0].set_ylabel('Mass')
    axes[2,0].set_title('Mass vs Entropy (decay trajectory)')
    axes[2,0].grid(True)
    
    # Half-life annotation
    axes[2,1].axis('off')
    if 'half_life' in locals():
        axes[2,1].text(0.1, 0.7, f'Decay constant λ = {lam_fit:.4f} per step', fontsize=14)
        axes[2,1].text(0.1, 0.5, f'Half-life T½ = {half_life:.1f} steps', fontsize=14)
    axes[2,1].text(0.1, 0.3, 'Weak interaction:\nFermions decay,\nentropy rises,\nmass decreases.', fontsize=12)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_decay_simulation(K=1, decay_rate=0.03, steps=10)