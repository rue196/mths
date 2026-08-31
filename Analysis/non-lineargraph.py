import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import quad
import math

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)   # ≈ 2.362

# ------------------------------------------------------------
# 1. Generate points in 3D
# ------------------------------------------------------------
def generate_points(N, seed=42):
    np.random.seed(seed)
    points = np.random.rand(N, 3)   # N points in [0,1]^3
    return points

# ------------------------------------------------------------
# 2. Build an Eulerian graph (cycle) – all degrees 2 (even)
# ------------------------------------------------------------
def build_even_graph(points):
    N = points.shape[0]
    # Create a cycle: edges (i, i+1) and (N-1, 0)
    edges = [(i, (i+1)%N) for i in range(N)]
    return edges

# ------------------------------------------------------------
# 3. Compute Euclidean distances for edges
# ------------------------------------------------------------
def compute_distances(points, edges):
    dists = []
    for i, j in edges:
        dist = np.linalg.norm(points[i] - points[j])
        dists.append(dist)
    return np.array(dists)

# ------------------------------------------------------------
# 4. Define g(x) and compute integral with α
# ------------------------------------------------------------
def g(x, alpha=ALPHA):
    return np.exp(-alpha * x)

def integral_g(alpha=ALPHA):
    # integrate from 0 to pi+e
    I, err = quad(g, 0, PI + E, args=(alpha,))
    return I

# ------------------------------------------------------------
# 5. Main: construct graph, scale distances by integral
# ------------------------------------------------------------
def main():
    N = 50   # number of vertices
    points = generate_points(N)
    edges = build_even_graph(points)
    dists = compute_distances(points, edges)

    # Compute integral
    I = integral_g()
    print(f"Integral I = ∫_0^{PI+E:.4f} exp(-{ALPHA:.4f} x) dx = {I:.6f}")

    # Scale distances by 1/I (optional, to normalize)
    scaled_dists = dists / I

    # Print statistics
    print(f"Original distances: mean = {dists.mean():.4f}, std = {dists.std():.4f}")
    print(f"Scaled distances:   mean = {scaled_dists.mean():.4f}, std = {scaled_dists.std():.4f}")

    # ------------------------------------------------------------
    # Visualize the graph in 3D
    # ------------------------------------------------------------
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2], c='blue', s=50)

    # Draw edges with color mapped by scaled distance
    for idx, (i, j) in enumerate(edges):
        p1 = points[i]
        p2 = points[j]
        w = scaled_dists[idx]
        # Normalize weight for color
        norm_w = (w - scaled_dists.min()) / (scaled_dists.max() - scaled_dists.min() + 1e-12)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                color=plt.cm.viridis(norm_w), alpha=0.8, linewidth=2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Non‑linear graph (cycle), edge color = scaled distance by 1/I')
    plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax, label='Scaled distance')
    plt.show()

    # ------------------------------------------------------------
    # Also show the integral function
    # ------------------------------------------------------------
    x_vals = np.linspace(0, PI+E, 100)
    g_vals = [g(x) for x in x_vals]
    plt.figure()
    plt.plot(x_vals, g_vals)
    plt.fill_between(x_vals, 0, g_vals, alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('g(x) = exp(-αx)')
    plt.title(f'Integral from 0 to {PI+E:.4f} = {I:.6f}')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()