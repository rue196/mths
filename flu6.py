import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree
import math
import os

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)
a = ALPHA

# ---------- SuperTrace and Entropy (as before) ----------
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

def invariant_scalar(C):
    S = supertrace_from_coeffs(C)
    H = entropy_from_supertrace(S, len(C))
    return abs(S) * math.exp(-H)

# ---------- Zeta and derivative ----------
def zeta(t, C, alpha=ALPHA):
    K = (len(C) - 1) // 2
    total = 0.0
    for idx, coeff in enumerate(C):
        i = idx - K
        phase = t * i / alpha
        total += coeff.real * math.cos(phase) - coeff.imag * math.sin(phase)
    return total

def dzeta_dt(t, C, alpha=ALPHA):
    K = (len(C) - 1) // 2
    derivative = 0.0
    for idx, coeff in enumerate(C):
        i = idx - K
        phase = t * i / alpha
        derivative -= (i / alpha) * (coeff.real * math.sin(phase) + coeff.imag * math.cos(phase))
    return derivative

# ---------- Bond construction (static) ----------
def build_bonds(points, k_neighbors=4):
    tree = KDTree(points)
    edges = []
    for i, p in enumerate(points):
        dists, idxs = tree.query(p, k_neighbors+1)
        for d, j in zip(dists[1:], idxs[1:]):
            if i < j:
                edges.append((i, j, d))
    return edges

# ---------- Bond weight using integral ----------
def bond_weight(distance, alpha=ALPHA):
    I_full = (1 - math.exp(-alpha * (PI + E))) / alpha
    if I_full == 0:
        return 0.0
    numerator = (1 - math.exp(-alpha * distance)) / alpha
    return numerator / I_full

# ---------- Main animation with save to JPEG ----------
def animate_and_save(K=80, num_steps=50, dt=0.02, k_neighbors=4, 
                     out_dir='frames', seed=42):
    """
    Run simulation, animate, and save each frame as JPEG.
    """
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Initialise
    positions = np.random.rand(K, 3) * 10.0
    C = [complex(pos[0], pos[1]) for pos in positions]
    edges = build_bonds(positions, k_neighbors)
    
    # Precompute bond weights for static edges (they don't change)
    bond_weights = [bond_weight(d) for (_, _, d) in edges]
    weights_norm = np.array(bond_weights)
    
    # 2. Prepare history for animation
    pos_hist = [positions.copy()]
    m_hist = [invariant_scalar(C)]
    
    # Time evolution
    for step in range(num_steps):
        t = step * dt
        dzet = dzeta_dt(t, C)
        centroid = np.mean(positions, axis=0)
        directions = centroid - positions
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / (norms + 1e-12)
        positions += dt * dzet * directions * 0.1
        C = [complex(pos[0], pos[1]) for pos in positions]
        m_new = invariant_scalar(C)
        pos_hist.append(positions.copy())
        m_hist.append(m_new)
    
    # 3. Set up the 3D plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # We'll update scatter and line segments
    # For lines, we'll store all segments as a list of (x, y, z) triples
    # We'll use a Line3DCollection for efficiency, but for simplicity we'll replot each time.
    # Since we save each frame, performance is less critical.
    
    def update(frame):
        ax.clear()
        pos = pos_hist[frame]
        # Scatter atoms
        ax.scatter(pos[:,0], pos[:,1], pos[:,2], c='blue', s=30)
        # Draw bonds with weights
        for (i, j, _), w in zip(edges, weights_norm):
            p1 = pos[i]
            p2 = pos[j]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    color=plt.cm.viridis(w), alpha=0.6, linewidth=1.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Frame {frame}, m = {m_hist[frame]:.4f}')
        # Set limits to keep view fixed
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_zlim(0, 10)
    
    # 4. Save each frame as JPEG
    for frame in range(len(pos_hist)):
        update(frame)
        # Save with zero-padded index
        fname = os.path.join(out_dir, f'frame_{frame:04d}.jpg')
        plt.savefig(fname, dpi=100, bbox_inches='tight')
        print(f'Saved {fname}')
    
    plt.close()
    print(f"All {len(pos_hist)} frames saved to {out_dir}")

# Run the snippet
if __name__ == "__main__":
    animate_and_save(K=80, num_steps=30, dt=0.02, k_neighbors=4, out_dir='bond_frames')