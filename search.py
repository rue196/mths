import random
import math

# ------------------------------------------------------------
# 1. Generate K memory addresses (as integers)
# ------------------------------------------------------------
def generate_addresses(K, seed=42):
    random.seed(seed)
    # Use distinct addresses to avoid duplicates
    addresses = random.sample(range(1, K*10), K)
    return sorted(addresses)   # keep sorted for 1D TSP (optimal route)

# ------------------------------------------------------------
# 2. TSP route in 1D: the optimal tour is simply sorted order
#    (going from min to max, then optionally back to min for a cycle)
# ------------------------------------------------------------
def tsp_1d_route(addresses):
    # Sorted addresses already give the optimal path (no crossing)
    return addresses

# ------------------------------------------------------------
# 3. Compute total distance of a given access order (sum of absolute diffs)
#    This simulates the cost of moving the memory head.
# ------------------------------------------------------------
def route_distance(order):
    if len(order) < 2:
        return 0.0
    total = 0.0
    for i in range(len(order)-1):
        total += abs(order[i+1] - order[i])
    return total

# ------------------------------------------------------------
# 4. Generate a binary‑search access order from a sorted list
#    We traverse the BST in preorder: root, left subtree, right subtree.
#    This mimics the sequence of indices probed in a binary search.
# ------------------------------------------------------------
def binary_search_order(sorted_arr):
    if not sorted_arr:
        return []
    mid = len(sorted_arr) // 2
    root = sorted_arr[mid]
    left = binary_search_order(sorted_arr[:mid])
    right = binary_search_order(sorted_arr[mid+1:])
    return [root] + left + right

# ------------------------------------------------------------
# 5. Compare linear (TSP) and binary search patterns
# ------------------------------------------------------------
def compare_search_patterns(K):
    addresses = generate_addresses(K)
    # TSP (linear) order: sorted
    tsp_order = tsp_1d_route(addresses)
    # Binary search order
    bst_order = binary_search_order(addresses)

    dist_tsp = route_distance(tsp_order)
    dist_bst = route_distance(bst_order)

    print(f"K = {K}")
    print(f"Memory addresses: {addresses[:10]} ...")
    print(f"TSP (linear) order (first 10): {tsp_order[:10]} ...")
    print(f"BST (binary) order (first 10): {bst_order[:10]} ...")
    print(f"TSP travel distance: {dist_tsp:.2f}")
    print(f"BST travel distance: {dist_bst:.2f}")
    print(f"Improvement (TSP vs BST): {dist_bst - dist_tsp:.2f}")
    print("-" * 40)

# ------------------------------------------------------------
# 6. Test with different K
# ------------------------------------------------------------
if __name__ == "__main__":
    for K in [10, 50, 100, 500]:
        compare_search_patterns(K)

        
        
input('Press ENTER to exit')