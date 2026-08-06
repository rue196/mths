import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree
from matplotlib.animation import FuncAnimation
import math

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)

# ---------- SuperTrace and Entropy ----------
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

# ---------- Build bonds (KDTree, O(K log K)) ----------
def build_bonds(points, k_neighbors=4):
    tree = KDTree(points)
    edges = []
    for i, p in enumerate(points):
        dists, idxs = tree.query(p, k_neighbors+1)
        for d, j in zip(dists[1:], idxs[1:]):
            if i < j:
                edges.append((i, j, d))
    return edges

# ---------- Bond weight using integral operator ----------
def bond_weight(distance, alpha=ALPHA):
    I_full = (1 - math.exp(-alpha * (PI + E))) / alpha
    if I_full == 0:
        return 0.0
    numerator = (1 - math.exp(-alpha * distance)) / alpha
    return numerator / I_full

# ---------- Main simulation with animation ----------
def animate_bonds(K=80, num_steps=50, dt=0.02, k_neighbors=4, seed=42):
    np.random.seed(seed)
    
    # 1. Initialise positions and coefficients
    positions = np.random.rand(K, 3) * 10.0
    C = [complex(pos[0], pos[1]) for pos in positions]
    edges = build_bonds(positions, k_neighbors)
    bond_weights = [bond_weight(d) for (_, _, d) in edges]
    
    # 2. Run simulation and store history
    pos_hist = [positions.copy()]
    m_hist = [invariant_scalar(C)]
    
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
    
    # 3. Prepare figure with two subplots: 3D and scalar evolution
    fig = plt.figure(figsize=(12, 6))
    ax3d = fig.add_subplot(121, projection='3d')
    ax_scalar = fig.add_subplot(122)
    ax_scalar.set_xlabel('Frame')
    ax_scalar.set_ylabel('Invariant scalar m')
    ax_scalar.grid(True)
    ax_scalar.set_title('m = |S| exp(-H)')
    line_scalar, = ax_scalar.plot([], [], 'b-', lw=2)
    ax_scalar.set_xlim(0, num_steps)
    ax_scalar.set_ylim(min(m_hist)*0.95, max(m_hist)*1.05)
    
    # 4. Update function for animation
    def update(frame):
        ax3d.clear()
        pos = pos_hist[frame]
        # Scatter atoms
        ax3d.scatter(pos[:,0], pos[:,1], pos[:,2], c='blue', s=30)
        # Draw bonds with weights
        for (i, j, _), w in zip(edges, bond_weights):
            p1 = pos[i]
            p2 = pos[j]
            ax3d.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                      color=plt.cm.viridis(w), alpha=0.6, linewidth=1.5)
        ax3d.set_xlabel('X')
        ax3d.set_ylabel('Y')
        ax3d.set_zlabel('Z')
        ax3d.set_title(f'Frame {frame}, m = {m_hist[frame]:.4f}')
        ax3d.set_xlim(0, 10)
        ax3d.set_ylim(0, 10)
        ax3d.set_zlim(0, 10)
        # Update scalar plot
        line_scalar.set_data(range(frame+1), m_hist[:frame+1])
        ax_scalar.set_title(f'Invariant scalar (m = {m_hist[frame]:.4f})')
        return ax3d, line_scalar
    
    # 5. Create animation
    ani = FuncAnimation(fig, update, frames=len(pos_hist), interval=100, blit=False)
    plt.tight_layout()
    return ani  # Return the animation object

# ---------- Run and show animation ----------
if __name__ == "__main__":
    ani = animate_bonds(K=80, num_steps=50, dt=0.02, k_neighbors=4, seed=42)
    # To display in Jupyter, use: from IPython.display import HTML; HTML(ani.to_html5_video())
    plt.show()   # For interactive window