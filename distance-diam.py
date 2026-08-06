import math
import sys
from collections import deque

# ---------- Prime utilities ----------
def is_prime(n):
    if n < 2: return False
    if n % 2 == 0: return n == 2
    i = 3
    while i*i <= n:
        if n % i == 0: return False
        i += 2
    return True

def first_n_primes(n):
    primes = []
    x = 2
    while len(primes) < n:
        if is_prime(x):
            primes.append(x)
        x += 1
    return primes

# ---------- Tree node ----------
class TreeNode:
    __slots__ = ('val', 'left', 'right', 'parent', 'weight')
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.parent = None
        self.weight = 0  # edge weight to parent

def build_balanced(arr, start, end, parent=None):
    if start > end:
        return None
    mid = (start + end) // 2
    node = TreeNode(arr[mid])
    node.parent = parent
    if parent:
        node.weight = abs(node.val - parent.val)
    node.left = build_balanced(arr, start, mid-1, node)
    node.right = build_balanced(arr, mid+1, end, node)
    return node

# ---------- Tree traversal (undirected) ----------
def find_farthest(start):
    """Return (farthest_node, max_distance) using iterative DFS."""
    if start is None:
        return None, 0
    stack = [(start, None, 0)]   # (node, parent, distance_from_start)
    farthest = start
    max_dist = 0
    while stack:
        node, parent, dist = stack.pop()
        if dist > max_dist:
            max_dist = dist
            farthest = node
        # Explore all neighbours: left child, right child, and parent
        for nxt in (node.left, node.right, node.parent):
            if nxt is not None and nxt is not parent:
                # Determine the edge weight to traverse
                if nxt is node.left or nxt is node.right:
                    w = nxt.weight          # child's weight to this node
                elif nxt is node.parent:
                    w = node.weight         # this node's weight to its parent
                else:
                    w = 0
                stack.append((nxt, node, dist + w))
    return farthest, max_dist

def tree_diameter(root):
    if root is None:
        return 0
    u, _ = find_farthest(root)
    v, diam = find_farthest(u)
    return diam

def total_weight(root):
    if root is None:
        return 0
    total = 0
    stack = [root]
    while stack:
        node = stack.pop()
        if node.parent:
            total += node.weight
        if node.left:
            stack.append(node.left)
        if node.right:
            stack.append(node.right)
    return total

# ---------- Main ----------
def main(K):
    primes = first_n_primes(K)
    root = build_balanced(primes, 0, K-1)
    total_w = total_weight(root)
    diam = tree_diameter(root)
    tsp_cycle = 2 * total_w
    tsp_path = 2 * total_w - diam

    print(f"K = {K}")
    print(f"Total edge weight = {total_w}")
    print(f"Diameter = {diam}")
    print(f"TSP cycle (return to start) = {tsp_cycle}")
    print(f"TSP path (open) = {tsp_path}")

if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 101
    main(K)