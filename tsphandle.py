import math
import random

# ------------------------------------------------------------
# 1. Generate tasks (K electrons/operations) with positions
# ------------------------------------------------------------
def generate_tasks(K, seed=43):
    random.seed(seed)
    # Each task has a position (x, y) on the chip
    tasks = [(random.random(), random.random()) for _ in range(K)]
    return tasks

# ------------------------------------------------------------
# 2. 1D TSP solution (optimal for points on a line)
#    We sort by the x-coordinate (or any scalar projection)
# ------------------------------------------------------------
def tsp_1d_route(tasks):
    # Sort tasks by x-coordinate (if ties, use y)
    sorted_tasks = sorted(tasks, key=lambda p: (p[0], p[1]))
    return sorted_tasks   # the optimal order is sorted order

# ------------------------------------------------------------
# 3. CPU Handler: simulates electron flow direction changes
#    Returns total path length and direction changes.
# ------------------------------------------------------------
def cpu_handler(tasks, route):
    # Route is the order of visiting tasks
    # Compute total path length (Manhattan or Euclidean)
    total_distance = 0.0
    direction_changes = 0
    prev = route[0]
    for curr in route[1:]:
        # Euclidean distance
        dist = math.hypot(curr[0]-prev[0], curr[1]-prev[1])
        total_distance += dist
        # Detect direction change (sign of dx, dy)
        if prev[0] != curr[0] and prev[1] != curr[1]:
            # simple: if both coordinates change, direction changed
            direction_changes += 1
        prev = curr
    # Return to starting point (cycle) – optional
    # Uncomment to include return path
    # dist = math.hypot(route[-1][0]-route[0][0], route[-1][1]-route[0][1])
    # total_distance += dist
    return total_distance, direction_changes

# ------------------------------------------------------------
# 4. Optional: entropy / spectral diagnostic (from earlier)
#    Uses the coefficients C_i = -α * p_i * log(p_i) / i
# ------------------------------------------------------------
def entropy_spectral_diagnostic(route, alpha=0.3628, collatz_even=True):
    K = len(route)
    # Use the positions to define probabilities (e.g., normalized x coordinates)
    xs = [p[0] for p in route]
    # Compute probabilities (e.g., softmax or normalized by sum)
    sum_x = sum(xs)
    probs = [x / sum_x for x in xs]
    # Build C_i
    C = []
    for i, p in enumerate(probs, start=1):
        if collatz_even and i % 2 != 0:
            C.append(0.0)
        else:
            C.append(-alpha * p * math.log(p + 1e-12) / i)
    # Compute ζ'(0) imaginary part = entropy
    sum_iCi = sum((i+1) * C[i] for i in range(len(C)))
    spectral_entropy = sum_iCi / alpha
    # Shannon entropy
    shannon = -sum(p * math.log(p + 1e-12) for p in probs)
    return spectral_entropy, shannon

# ------------------------------------------------------------
# 5. Main simulation
# ------------------------------------------------------------
def main():
    K = 20
    tasks = generate_tasks(K)
    route = tsp_1d_route(tasks)
    total_dist, dir_changes = cpu_handler(tasks, route)
    spec_ent, shan_ent = entropy_spectral_diagnostic(route)

    print("=== CPU Handler: Electron Flow Scheduling ===")
    print(f"Number of tasks (K): {K}")
    print(f"Optimal route (sorted by x): {route[:5]} ...")
    print(f"Total path length: {total_dist:.4f}")
    print(f"Number of direction changes: {dir_changes}")
    print(f"Spectral entropy (Im ζ'(0)): {spec_ent:.6f}")
    print(f"Shannon entropy: {shan_ent:.6f}")
    print("Difference: {:.2e}".format(spec_ent - shan_ent))

if __name__ == "__main__":
    main()

    
input('Press ENTER to exit')