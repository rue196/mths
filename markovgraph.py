import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
import math

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)   # ≈ 2.362

# ------------------------------------------------------------
# 1. Generate 3D points
# ------------------------------------------------------------
def generate_points(N, seed=42):
    np.random.seed(seed)
    return np.random.rand(N, 3)   # N points in [0,1]^3

# ------------------------------------------------------------
# 2. Build a sparse Eulerian graph (cycle + extra edges)
# ------------------------------------------------------------
def build_sparse_graph(points, extra_edges_per_vertex=0):
    """
    Build a graph with even degrees.
    Start with a cycle (degree 2). Then add extra edges to increase connectivity
    while keeping total edges O(N).
    For simplicity, we add edges to the next-nearest neighbors along the cycle.
    """
    N = points.shape[0]
    edges = set()
    # Cycle edges
    for i in range(N):
        edges.add((i, (i+1)%N))
    # Extra edges: connect to vertices at distance 2 along the cycle
    for k in range(1, extra_edges_per_vertex+1):
        for i in range(N):
            j = (i + k + 1) % N   # skip immediate neighbor
            if i != j:
                edges.add((min(i,j), max(i,j)))
    # Ensure all degrees are even – if not, we can add more edges.
    # The cycle already gives degree 2 (even); extra edges add 2 each (if added symmetrically)
    # So degrees remain even.
    return list(edges)

# ------------------------------------------------------------
# 3. Compute distances and spectral coefficients
# ------------------------------------------------------------
def compute_distances(points, edges):
    dists = {}
    for i, j in edges:
        d = np.linalg.norm(points[i] - points[j])
        dists[(i,j)] = d
    return dists

def compute_spectral_coefficients(points, alpha=ALPHA):
    """
    Compute |C_i| for each vertex i.
    We define C_i as the trace of a 3x3 matrix built from the coordinates,
    e.g., M = [[x, y, z], [y, z, x], [z, x, y]] (circulant-like).
    We use the absolute value of the trace, scaled by alpha.
    This gives a positive scalar for each vertex.
    """
    N = points.shape[0]
    C_abs = np.zeros(N)
    for i in range(N):
        x, y, z = points[i]
        # Build a simple 3x3 matrix
        M = np.array([[x, y, z],
                      [y, z, x],
                      [z, x, y]])
        trace = np.trace(M)   # = x + z + y = x+y+z (since symmetric)
        # Use absolute, multiply by alpha for scaling
        C_abs[i] = alpha * abs(trace) + 1e-8   # avoid zero
    return C_abs

# ------------------------------------------------------------
# 4. Build transition matrix (sparse, O(K))
# ------------------------------------------------------------
def build_transition_matrix(points, edges, dists, C_abs, beta=1.0, collatz_mask=False):
    """
    Build a sparse row-stochastic transition matrix P (K x K).
    For each vertex i, the probability to go to neighbor j is proportional to
        exp(-beta * dist(i,j)) * C_abs[j]
    (the C_abs of the target vertex).
    If collatz_mask=True, we zero out transitions from odd-indexed vertices
    (index i) – this is a Collatz-like filter.
    """
    N = points.shape[0]
    # We'll store row indices, col indices, and values for CSR format.
    row = []
    col = []
    data = []
    for i in range(N):
        # Gather neighbors (both directions)
        neighbors = []
        for (a,b) in edges:
            if a == i:
                neighbors.append(b)
            elif b == i:
                neighbors.append(a)
        # If Collatz mask: for odd i, skip all transitions (i.e., row becomes zero)
        if collatz_mask and (i % 2 != 0):
            # We can set a self-loop to 1 to keep stochastic, or set all to 0 (but then row sum zero).
            # We'll set self-loop with probability 1 to avoid absorption.
            row.append(i)
            col.append(i)
            data.append(1.0)
            continue
        # Compute unnormalized weights
        weights = []
        for j in neighbors:
            if i == j:
                continue
            d = dists.get((min(i,j), max(i,j)), 1e-8)
            w = math.exp(-beta * d) * C_abs[j]
            weights.append((j, w))
        # Add self-loop with small probability to ensure irreducibility
        self_weight = 1e-6
        weights.append((i, self_weight))
        total = sum(w for _, w in weights)
        if total == 0:
            total = 1.0
            weights = [(i, 1.0)]
        # Normalize and store
        for j, w in weights:
            row.append(i)
            col.append(j)
            data.append(w / total)
    # Build sparse matrix
    P = csr_matrix((data, (row, col)), shape=(N, N))
    # Ensure row sums are 1 (numerical)
    return P

# ------------------------------------------------------------
# 5. Compute stationary distribution via power iteration
# ------------------------------------------------------------
def stationary_distribution(P, max_iter=1000, tol=1e-8):
    """
    Compute the stationary distribution π such that π = π P.
    Using power iteration on the transpose (since we want left eigenvector).
    """
    N = P.shape[0]
    pi = np.ones(N) / N
    P_T = P.T
    for _ in range(max_iter):
        pi_new = pi @ P_T
        diff = np.linalg.norm(pi_new - pi, ord=1)
        pi = pi_new
        if diff < tol:
            break
    return pi

# ------------------------------------------------------------
# 6. Simulate Markov chain (random walk)
# ------------------------------------------------------------
def simulate_random_walk(P, start=0, steps=100):
    states = [start]
    current = start
    for _ in range(steps):
        # Get row of P for current state
        row = P[current].toarray().flatten()
        # Sample next state from the distribution
        next_state = np.random.choice(len(row), p=row)
        states.append(next_state)
        current = next_state
    return states

# ------------------------------------------------------------
# 7. Main demonstration
# ------------------------------------------------------------
def main():
    N = 50
    points = generate_points(N)
    extra_edges = 1   # add edges to next-nearest neighbors
    edges = build_sparse_graph(points, extra_edges_per_vertex=extra_edges)

    dists = compute_distances(points, edges)
    C_abs = compute_spectral_coefficients(points, alpha=ALPHA)

    # Build transition matrix with Collatz mask (odd indices get self-loop)
    beta = 2.0   # inverse temperature
    P = build_transition_matrix(points, edges, dists, C_abs, beta=beta, collatz_mask=True)

    # Stationary distribution
    pi = stationary_distribution(P)
    print("Stationary distribution (first 5):", pi[:5])

    # Simulate a random walk
    start = 0
    steps = 50
    states = simulate_random_walk(P, start, steps)
    print("Random walk states (first 10):", states[:10])

    # Visualize the graph with transition probabilities (optional)
    # We can color edges by probability from a given vertex.
    # For brevity, we just show the 3D graph.
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2], c='blue', s=50)
    # Draw edges
    for (i,j) in edges:
        p1 = points[i]
        p2 = points[j]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='gray', alpha=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Graph (cycle + extra edges) for Markov chain')
    plt.show()

if __name__ == "__main__":
    main()