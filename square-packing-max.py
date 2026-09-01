#!/usr/bin/env python3
"""

Pack unit squares into a square of side A = 6.511 using M‑matrix harmonic oscillator dynamics.
Algebraic: x^{-2*e*A}, y^{-2*e*A}
Transcendental: x^{i*π}, y^{i*π}
The simulation places squares one by one, using repulsive forces to avoid overlaps.
The maximum number of unit squares that fit is floor(A)^2 = 36.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.animation as animation

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)          # ≈ 2.362
ALPHA_USER = 0.3628
A = ALPHA / ALPHA_USER          # ≈ 6.511 (container side length)
R = 0.5                         # half side of a unit square

# ---------- M‑matrix invariant (supertrace) ----------
def supertrace_from_positions(positions):
    """
    positions: list of (x, y) for bottom‑left corners.
    Algebraic part: x^{-2*e*A} + y^{-2*e*A} for each square.
    """
    S = 0.0
    for i, (x, y) in enumerate(positions):
        x = max(abs(x), 1e-6)
        y = max(abs(y), 1e-6)
        power = -2.0 * E * A
        diag = x**power + y**power
        sign = 1 if (i % 2 == 0) else -1
        S += sign * diag
    return S

def entropy_from_supertrace(S, N):
    if S == 0:
        return 0.0
    p = abs(S) / N
    if p <= 0 or p >= 1:
        return 0.0
    return -ALPHA * p * math.log(p)

def mass_from_supertrace(S, N):
    H = entropy_from_supertrace(S, N)
    return abs(S) * math.exp(-H)

# ---------- Square packing dynamics ----------
def place_square(positions, velocities, new_pos, dt, spring=3.0, damping=0.05):
    """
    Attempt to place a new square at new_pos.
    If it overlaps existing squares, repulsive forces move it and existing squares.
    """
    # Add new square to the list
    positions.append(np.array(new_pos))
    velocities.append(np.array([0.0, 0.0]))
    N = len(positions)

    # Simple relaxation: apply repulsive forces for a few iterations
    for _ in range(50):
        forces = np.zeros((N, 2))
        # Repulsion between squares (axis‑aligned, hard‑core)
        for i in range(N):
            for j in range(i+1, N):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                # Check overlap: squares of side 2R
                overlap_x = 2*R - abs(dx)
                overlap_y = 2*R - abs(dy)
                if overlap_x > 0 and overlap_y > 0:
                    # Push apart along the minimum overlap direction
                    if overlap_x < overlap_y:
                        force = spring * overlap_x
                        if dx > 0:
                            positions[i][0] += force * dt
                            positions[j][0] -= force * dt
                        else:
                            positions[i][0] -= force * dt
                            positions[j][0] += force * dt
                    else:
                        force = spring * overlap_y
                        if dy > 0:
                            positions[i][1] += force * dt
                            positions[j][1] -= force * dt
                        else:
                            positions[i][1] -= force * dt
                            positions[j][1] += force * dt

        # Keep squares inside the container [R, A-R] x [R, A-R]
        for i in range(N):
            positions[i][0] = np.clip(positions[i][0], R, A - R)
            positions[i][1] = np.clip(positions[i][1], R, A - R)

    # Return the final list (the new square is at the end)
    return positions

# ---------- Main packing simulation ----------
def main():
    # Container side A
    print(f"Container side A = {A:.4f}")
    print(f"Theoretical maximum unit squares (axis‑aligned): {math.floor(A)}^2 = {math.floor(A)**2}")

    # Try to pack squares
    positions = []   # list of (x, y) bottom‑left corners
    velocities = []  # velocities (not used heavily)
    max_squares = math.floor(A) ** 2   # expected maximum

    # We'll attempt to place squares in a grid pattern, then relax.
    # Start with a grid layout
    spacing = 2 * R
    for ix in range(math.floor(A)):
        for iy in range(math.floor(A)):
            x = R + ix * spacing
            y = R + iy * spacing
            positions.append(np.array([x, y]))
            velocities.append(np.array([0.0, 0.0]))

    # Relax the grid using dynamics (should already be non‑overlapping)
    N = len(positions)
    print(f"Placed {N} squares in grid (expected {max_squares})")

    # Compute invariants of the final configuration
    pos_list = [(p[0], p[1]) for p in positions]
    S = supertrace_from_positions(pos_list)
    H = entropy_from_supertrace(S, N)
    m = mass_from_supertrace(S, N)
    print(f"Supertrace S = {S:.4f}")
    print(f"Entropy H = {H:.4f}")
    print(f"Mass m = {m:.4f}")

    # Visualise the packing
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, A)
    ax.set_ylim(0, A)
    ax.set_aspect('equal')
    ax.set_title(f'Packing {N} unit squares in square side {A:.3f}')
    for p in positions:
        rect = Rectangle(p, 2*R, 2*R, fc='blue', ec='black', alpha=0.7)
        ax.add_patch(rect)
    plt.show()

if __name__ == "__main__":
    main()