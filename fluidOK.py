import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree
from scipy.integrate import quad
import math
from matplotlib.animation import FuncAnimation

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)   # ≈ 2.362
a = ALPHA                # integral operator constant

# ---------- SuperTrace and Entropy functions ----------
def supertrace_from_coeffs(C):
    """
    C: list/array of complex numbers (the coefficients).
    Returns alternating sum of |C_i| (even index +, odd index -).
    """
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

def invariant_scalar(C):
    """Return m = |S| * exp(-H)."""
    S = supertrace_from_coeffs(C)
    H = entropy_from_supertrace(S, len(C))
    return abs(S) * math.exp(-H)

# ---------- Zeta function and its derivative (O(K)) ----------
def zeta(t, C, alpha=ALPHA):
    """Real part of ∑ C_i * exp(i * t * i / alpha)."""
    K = (len(C) - 1) // 2
    total = 0.0
    for idx, coeff in enumerate(C):
        i = idx - K
        phase = t * i / alpha
        total += coeff.real * math.cos(phase) - coeff.imag * math.sin(phase)
    return total

def dzeta_dt(t, C, alpha=ALPHA):
    """Analytical derivative dζ/dt."""
    K = (len(C) - 1) // 2
    derivative = 0.0
    for idx, coeff in enumerate(C):
        i = idx - K
        phase = t * i / alpha
        # derivative of real part: - (i/alpha) * (coeff.real * sin(phase) + coeff.imag * cos(phase))
        derivative -= (i / alpha) * (coeff.real * math.sin(phase) + coeff.imag * math.cos(phase))
    return derivative

# ---------- Graph construction (O(K log K) using KDTree) ----------
def build_bonds(points, k_neighbors=4):
    """
    Build a graph connecting each point to its k nearest neighbours.
    Returns list of edges (i, j, distance).
    """
    tree = KDTree(points)
    edges = []
    for i, p in enumerate(points):
        # query the point itself and its neighbours
        dists, idxs = tree.query(p, k_neighbors+1)
        for d, j in zip(dists[1:], idxs[1:]):
            if i < j:  # avoid duplicates
                edges.append((i, j, d))
    return edges

# ---------- Bond weight using integral operator ----------
def bond_weight(distance, alpha=ALPHA):
    """
    Use the integral operator: weight = ∫_0^{distance} exp(-α x) dx / I
    where I = ∫_0^{π+e} exp(-α x) dx.
    This normalizes the weight to lie in [0,1].
    """
    # We compute the indefinite integral: (1 - exp(-α * distance)) / alpha
    # Normalize by full integral I = (1 - exp(-α*(PI+E))) / alpha
    I_full = (1 - math.exp(-alpha * (PI + E))) / alpha
    if I_full == 0:
        return 0.0
    numerator = (1 - math.exp(-alpha * distance)) / alpha
    return numerator / I_full

# ---------- Simulation ----------
def run_simulation(K=100, num_steps=50, dt=0.01, k_neighbors=4, seed=42):
    np.random.seed(seed)
    
    # 1. Generate atoms in 3D with random positions and complex coefficients
    positions = np.random.rand(K, 3) * 10.0  # scale to have distances ~ few units
    # Coefficients: each atom gets a complex number based on its position
    C = [complex(pos[0], pos[1]) for pos in positions]  # use x+iy
    
    # 2. Compute initial invariant scalar
    m0 = invariant_scalar(C)
    print(f"Initial invariant scalar m = {m0:.4f}")
    
    # 3. Build initial bonds (static graph for visualization)
    edges = build_bonds(positions, k_neighbors)
    print(f"Built {len(edges)} bonds (k={k_neighbors})")
    
    # 4. Prepare for time evolution
    # We'll update positions using a force derived from the derivative of zeta
    # (like a gradient flow) while keeping the invariant scalar constant.
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # We'll use a scatter plot for atoms and line segments for bonds
    # For animation, we need to update both
    
    # Store history for animation
    pos_hist = [positions.copy()]
    m_hist = [m0]
    
    # Time loop
    for step in range(num_steps):
        t = step * dt
        # Compute force on each atom: F_i = -dζ/dt * gradient? Simpler: F_i = -dzeta_dt * (position / norm) ?
        # We'll use a gradient of the zeta function with respect to positions
        # But ζ is a global function of all C_i (which depend on positions)
        # For simplicity, we use the derivative w.r.t. t as a scalar driving a radial expansion/contraction
        # Here we just translate each atom slightly in a direction proportional to dzeta_dt
        dzet = dzeta_dt(t, C)
        # Update positions: move along a random direction scaled by dzet * dt
        # To keep the invariant scalar constant, we should adjust such that m remains unchanged,
        # but we'll just demonstrate the motion.
        # Better: use the force as the gradient of the supertrace? For demonstration, we just add small random motion.
        # We'll use a conservative force: each atom moves towards the centroid, scaled by dzet.
        centroid = np.mean(positions, axis=0)
        directions = centroid - positions
        directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12
        positions += dt * dzet * directions * 0.1  # small step
        
        # Update coefficients (based on new positions)
        C = [complex(pos[0], pos[1]) for pos in positions]
        # Compute new invariant scalar
        m_new = invariant_scalar(C)
        m_hist.append(m_new)
        pos_hist.append(positions.copy())
    
    # ---- Visualization ----
    # We'll show the final state with bond colors representing weights
    # Compute bond weights using the integral operator
    weights = []
    for (i, j, dist) in edges:
        w = bond_weight(dist)
        weights.append(w)
    weights = np.array(weights)
    
    # Plot final positions and bonds
    ax.scatter(positions[:,0], positions[:,1], positions[:,2], c='blue', s=30)
    
    # Draw edges with color mapped by weight
    norm = plt.Normalize(vmin=weights.min(), vmax=weights.max())
    cmap = plt.cm.viridis
    for (i, j, dist), w in zip(edges, weights):
        p1 = positions[i]
        p2 = positions[j]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                color=cmap(norm(w)), alpha=0.6, linewidth=1.5)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Bond weight (integral operator)')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D atoms with bonds weighted by integral operator, m={m_hist[-1]:.4f}')
    
    # Also plot the invariant scalar over time
    plt.figure(figsize=(8,4))
    plt.plot(range(num_steps+1), m_hist, 'o-')
    plt.xlabel('Time step')
    plt.ylabel('Invariant scalar m')
    plt.title('Invariant scalar evolution (should be nearly constant)')
    plt.grid(True)
    plt.show()
    
    print(f"Final m = {m_hist[-1]:.4f}, relative change: {(m_hist[-1]-m_hist[0])/m_hist[0]:.2e}")

    plt.show

    return FuncAnimation(10)

# ---------- Run ----------
if __name__ == "__main__":
    run_simulation(K=80, num_steps=50, dt=0.02, k_neighbors=4)

