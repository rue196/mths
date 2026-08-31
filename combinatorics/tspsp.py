import random
import time

def tsp_1d(points):
    """
    points: list of numbers (1D coordinates)
    Returns: (tour_order, total_distance)
    """
    if len(points) <= 1:
        return points, 0.0
    
    # Sort the points O(K log K)
    sorted_pts = sorted(points)
    
    # Optimal path: leftmost to rightmost
    tour = sorted_pts  # direct order
    
    # Total distance for cycle (return to start)
    total_distance = 2.0 * (sorted_pts[-1] - sorted_pts[0])
    
    return tour, total_distance

# Example
K = 10**5  # 100,000 cities
random.seed(44)
cities = [random.uniform(0, 1000) for _ in range(K)]

start = time.time()
tour, dist = tsp_1d(cities)
elapsed = time.time() - start

print(f"Number of cities: {K}")
print(f"Optimal tour length (cycle): {dist:.2f}")
print(f"Time taken: {elapsed:.3f} seconds (O(K log K) sorting)")
print(f"First 10 cities in order: {tour[:10000]}")

input('Press ENTER to exit')
   