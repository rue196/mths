import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree
from matplotlib.animation import FuncAnimation
import math
from matplotlib.path import Path

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)          # ≈ 2.362
A = ALPHA                       # same as 1/(π-e)

# ---------- Prime generator (sieve) O(N log log N) ----------
def sieve_primes(limit):
    is_prime = np.ones(limit+1, dtype=bool)
    is_prime[0:2] = False
    for i in range(2, int(limit**0.5)+1):
        if is_prime[i]:
            is_prime[i*i:limit+1:i] = False
    return np.nonzero(is_prime)[0]

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

# ---------- Build bonds (KDTree, O(N log N)) ----------
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

# ---------- Define airfoil (NACA0012 symmetric) ----------
def get_airfoil_points():
    # Upper surface: quadratic approximation
    x = np.linspace(0, 1, 20)
    y = 0.06 * (2.969 * np.sqrt(x) - 1.260 * x - 3.516 * x**2 + 2.843 * x**3 - 1.036 * x**4)
    upper = np.vstack([x, y]).T
    lower = np.vstack([x, -y]).T[1:-1][::-1]
    vertices = np.vstack([upper, lower])
    return vertices

# ---------- Potential flow around airfoil (coarse grid) ----------
def compute_potential_flow(airfoil_vertices, x_grid, y_grid, Uinf=1.0, alpha=0.0):
    # Simplified: uniform flow + vortex panel method not implemented; just uniform flow for demo.
    # We'll just return uniform flow plus a small perturbation.
    X, Y = np.meshgrid(x_grid, y_grid)
    u = Uinf * np.cos(alpha) * np.ones_like(X)
    v = Uinf * np.sin(alpha) * np.ones_like(Y)
    # Add a simple vortex at the quarter chord
    xc, yc = 0.25, 0.0
    r2 = (X - xc)**2 + (Y - yc)**2
    gamma = 0.5  # circulation strength
    u += -gamma * (Y - yc) / (r2 + 1e-6)
    v += gamma * (X - xc) / (r2 + 1e-6)
    return u, v

# ---------- Main simulation ----------
def simulate_wind_3d(K=80, num_steps=100, dt=0.02, k_neighbors=4, seed=42):
    # 1. Generate N = K-th prime
    primes = sieve_primes(1000)
    N = primes[min(K-1, len(primes)-1)]  # use K-th prime as particle count
    print(f"Using N={N} particles (K={K}-th prime).")

    np.random.seed(seed)
    # 2. Domain and airfoil
    airfoil = get_airfoil_points()
    # Extrude in z: create a 3D airfoil surface (just for visualisation, not used for flow)
    z_extrude = np.array([-0.5, 0.5])
    airfoil_3d = []
    for z in z_extrude:
        for p in airfoil:
            airfoil_3d.append([p[0], p[1], z])
    airfoil_3d = np.array(airfoil_3d)

    # 3. Background flow (3D uniform flow plus vortex)
    x_min, x_max = -0.5, 1.8
    y_min, y_max = -0.5, 0.5
    z_min, z_max = -0.5, 0.5
    nx, ny, nz = 30, 20, 10
    x_grid = np.linspace(x_min, x_max, nx)
    y_grid = np.linspace(y_min, y_max, ny)
    z_grid = np.linspace(z_min, z_max, nz)
    u_flow, v_flow = compute_potential_flow(airfoil, x_grid, y_grid, Uinf=1.0, alpha=np.deg2rad(5))
    w_flow = np.zeros_like(u_flow)   # no z-component in background

    # Interpolate flow at particle positions
    def get_flow(xp, yp, zp):
        # Nearest grid interpolation
        ix = np.clip(np.searchsorted(x_grid, xp), 0, nx-2)
        iy = np.clip(np.searchsorted(y_grid, yp), 0, ny-2)
        iz = np.clip(np.searchsorted(z_grid, zp), 0, nz-2)
        u = u_flow[iy, ix]  # indices: (y, x)
        v = v_flow[iy, ix]
        w = w_flow[iy, ix]
        return u, v, w

    # 4. Initialise particles
    positions = np.random.rand(N, 3)
    positions[:,0] = x_min + (x_max - x_min) * positions[:,0]
    positions[:,1] = y_min + (y_max - y_min) * positions[:,1]
    positions[:,2] = z_min + (z_max - z_min) * positions[:,2]
    # Remove particles inside the airfoil (2D projection)
    airfoil_path = Path(airfoil)
    inside = np.zeros(N, dtype=bool)
    for i, p in enumerate(positions):
        inside[i] = airfoil_path.contains_point((p[0], p[1]))
    positions = positions[~inside]
    N = len(positions)
    print(f"After removing inside-airfoil particles: N={N}")

    # Velocities: initialise from background flow
    velocities = np.zeros((N, 3))
    for i, p in enumerate(positions):
        u, v, w = get_flow(p[0], p[1], p[2])
        velocities[i] = [u, v, w]

    # Coefficients C_i = u + i v (using horizontal components)
    C = [complex(velocities[i,0], velocities[i,1]) for i in range(N)]

    # 5. Precompute bonds (static graph for visualisation)
    edges = build_bonds(positions, k_neighbors)
    bond_weights = [bond_weight(d) for (_, _, d) in edges]

    # 6. Storage for animation
    pos_hist = [positions.copy()]
    m_hist = [invariant_scalar(C)]
    m0 = m_hist[0]

    # 7. Time evolution
    for step in range(num_steps):
        t = step * dt
        # Compute current invariant scalar
        m = invariant_scalar(C)
        # Force: adjust velocities to keep m constant
        # We add a small random perturbation to velocities (turbulence) and then
        # project back to keep m = m0.
        # Simple feedback: scale velocities by factor (m0 / m) if m deviates.
        factor = m0 / m if m != 0 else 1.0
        velocities *= factor   # this changes the magnitude of velocities, preserving direction

        # Advect particles with background flow + a small random walk (turbulence)
        for i in range(N):
            # Background flow
            u_b, v_b, w_b = get_flow(positions[i,0], positions[i,1], positions[i,2])
            # Add some stochastic term (Brownian)
            noise = 0.02 * np.random.randn(3)
            velocities[i] = [u_b, v_b, w_b] + noise
        # Recompute C from new velocities
        C = [complex(velocities[i,0], velocities[i,1]) for i in range(N)]

        # Update positions
        positions += velocities * dt
        # Keep within domain (soft boundaries)
        positions[:,0] = np.clip(positions[:,0], x_min, x_max)
        positions[:,1] = np.clip(positions[:,1], y_min, y_max)
        positions[:,2] = np.clip(positions[:,2], z_min, z_max)

        # Store history
        pos_hist.append(positions.copy())
        m_hist.append(invariant_scalar(C))

    # 8. Animation
    fig = plt.figure(figsize=(12, 6))
    ax3d = fig.add_subplot(121, projection='3d')
    ax_scalar = fig.add_subplot(122)
    ax_scalar.set_xlabel('Time step')
    ax_scalar.set_ylabel('Invariant scalar m')
    ax_scalar.grid(True)
    ax_scalar.set_title('m = |S| exp(-H)')
    line_scalar, = ax_scalar.plot([], [], 'b-', lw=2)
    ax_scalar.set_xlim(0, num_steps)
    ax_scalar.set_ylim(min(m_hist)*0.95, max(m_hist)*1.05)

    def update(frame):
        ax3d.clear()
        pos = pos_hist[frame]
        ax3d.scatter(pos[:,0], pos[:,1], pos[:,2], c='blue', s=10, alpha=0.6)
        # Draw bonds
        for (i, j, _), w in zip(edges, bond_weights):
            p1 = pos[i]
            p2 = pos[j]
            ax3d.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                      color=plt.cm.viridis(w), alpha=0.3, linewidth=0.8)
        # Draw airfoil surface (extruded)
        ax3d.plot(airfoil[:,0], airfoil[:,1], -0.5*np.ones_like(airfoil[:,0]), 'k-', lw=1)
        ax3d.plot(airfoil[:,0], airfoil[:,1],  0.5*np.ones_like(airfoil[:,0]), 'k-', lw=1)
        ax3d.set_xlim(x_min, x_max)
        ax3d.set_ylim(y_min, y_max)
        ax3d.set_zlim(z_min, z_max)
        ax3d.set_xlabel('X')
        ax3d.set_ylabel('Y')
        ax3d.set_zlabel('Z')
        ax3d.set_title(f'Frame {frame}, m={m_hist[frame]:.4f}')
        # Update scalar plot
        line_scalar.set_data(range(frame+1), m_hist[:frame+1])
        ax_scalar.set_title(f'Invariant scalar (m = {m_hist[frame]:.4f})')
        return ax3d, line_scalar

    ani = FuncAnimation(fig, update, frames=len(pos_hist), interval=50, blit=False)
    plt.tight_layout()
    return ani

# ---------- Run ----------
if __name__ == "__main__":
    ani = simulate_wind_3d(K=60, num_steps=80, dt=0.02, k_neighbors=3, seed=42)
    # To display in Jupyter: from IPython.display import HTML; HTML(ani.to_html5_video())
    plt.show()