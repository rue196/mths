#!/usr/bin/env python3
"""
packing_m_matrix.py

Square/circle packing using M‑matrix harmonic oscillator dynamics.
M = [[x^{-1}, y^{-1}], [x^i, y^i]]
Objects evolve in time t (index i) under a potential that penalises overlaps.
The supertrace S and entropy H are computed from the object positions.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import matplotlib.animation as animation

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)          # ≈ 2.362

# ---------- Supertrace and entropy ----------
def supertrace_from_positions(positions):
    """
    positions: list of (x, y) tuples.
    Construct diagonal elements of M: x^{-1} and y^{-1} for each object?
    Actually the matrix M is 2x2 for each object? We'll define:
    For each object j, we have a 2x2 matrix M_j = [[x_j^{-1}, y_j^{-1}], [x_j^{i}, y_j^{i}]]
    The supertrace is the sum over diagonal elements of all objects with alternating signs.
    We'll compute S = Σ_{j} (-1)^j (x_j^{-1} + y_j^{-1}) ??
    Following the definition: S = Σ_{t} (-1)^{t+1} M_{t,t} where M_{t,t} are diagonal entries.
    For each object, the diagonal entries are x^{-1} and y^{-1} (if we treat them as the diagonal of a 2x2 matrix?).
    We'll simplify: use the positions directly: S = Σ_{j} (-1)^j (1/x_j + 1/y_j)
    """
    S = 0.0
    for i, (x, y) in enumerate(positions):
        # Avoid division by zero
        x = max(x, 1e-6)
        y = max(y, 1e-6)
        sign = 1 if (i % 2 == 0) else -1
        S += sign * (1.0 / x + 1.0 / y)
    return S

def entropy_from_supertrace(S, N, alpha=ALPHA):
    if S == 0:
        return 0.0
    p = abs(S) / N
    if p <= 0 or p >= 1:
        return 0.0
    return -alpha * p * math.log(p)

def mass_from_supertrace(S, N):
    H = entropy_from_supertrace(S, N)
    return abs(S) * math.exp(-H)

# ---------- Harmonic oscillator dynamics ----------
def harmonic_step(positions, velocities, dt, spring_constant=1.0, damping=0.1):
    """
    Update positions and velocities using a damped harmonic oscillator.
    The force is derived from a potential that penalises overlaps.
    Here we implement a simple spring force that pulls objects toward a target (the centre).
    For packing, we need a repulsive force between objects.
    We'll implement a Lennard-Jones type potential.
    """
    N = len(positions)
    forces = np.zeros((N, 2))
    # Repulsive force: if two objects overlap, push them apart
    # For circles: overlap if distance < 2*radius (we use radius = 0.5 for all)
    radius = 0.5
    for i in range(N):
        for j in range(i+1, N):
            dx = positions[j][0] - positions[i][0]
            dy = positions[j][1] - positions[i][1]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < 2*radius and dist > 1e-6:
                # Spring-like repulsion: force = k * (2*radius - dist) / dist
                k = spring_constant
                force_mag = k * (2*radius - dist)
                fx = force_mag * dx / dist
                fy = force_mag * dy / dist
                forces[i][0] -= fx
                forces[i][1] -= fy
                forces[j][0] += fx
                forces[j][1] += fy
    # Also add a weak spring to the centre to keep objects in the box
    centre = np.array([5.0, 5.0])  # box centre
    for i in range(N):
        dx = positions[i][0] - centre[0]
        dy = positions[i][1] - centre[1]
        forces[i][0] -= 0.01 * dx
        forces[i][1] -= 0.01 * dy

    # Update velocities and positions (Euler integration)
    for i in range(N):
        velocities[i][0] += forces[i][0] * dt
        velocities[i][1] += forces[i][1] * dt
        # Damping
        velocities[i] *= (1 - damping * dt)
        positions[i][0] += velocities[i][0] * dt
        positions[i][1] += velocities[i][1] * dt
        # Keep inside a box [0, 10] x [0, 10]
        if positions[i][0] < radius:
            positions[i][0] = radius
            velocities[i][0] = -velocities[i][0] * 0.5
        elif positions[i][0] > 10 - radius:
            positions[i][0] = 10 - radius
            velocities[i][0] = -velocities[i][0] * 0.5
        if positions[i][1] < radius:
            positions[i][1] = radius
            velocities[i][1] = -velocities[i][1] * 0.5
        elif positions[i][1] > 10 - radius:
            positions[i][1] = 10 - radius
            velocities[i][1] = -velocities[i][1] * 0.5

# ---------- Main packing simulation ----------
def main():
    # Parameters
    N_objects = 20
    dt = 0.02
    steps = 1000
    box_size = 10.0
    radius = 0.5  # circle radius or half side for square

    # Initialise positions randomly in the box
    np.random.seed(42)
    positions = np.random.rand(N_objects, 2) * (box_size - 2*radius) + radius
    velocities = np.zeros((N_objects, 2))

    # Store history for animation
    history = [positions.copy()]

    # Evolution
    for step in range(steps):
        harmonic_step(positions, velocities, dt, spring_constant=2.0, damping=0.05)
        if step % 10 == 0:
            history.append(positions.copy())

    # Compute final supertrace and entropy
    pos_list = [tuple(p) for p in positions]
    S = supertrace_from_positions(pos_list)
    H = entropy_from_supertrace(S, N_objects)
    m = mass_from_supertrace(S, N_objects)
    print(f"Final supertrace S = {S:.4f}")
    print(f"Entropy H = {H:.4f}")
    print(f"Mass m = {m:.4f}")

    # Visualise final packing
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_aspect('equal')
    ax.set_title(f'Circle packing (S={S:.2f}, m={m:.2f})')
    for pos in positions:
        circle = Circle(pos, radius, fc='blue', ec='black', alpha=0.7)
        ax.add_patch(circle)
    plt.show()

    # Optional: animate the packing process
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_aspect('equal')
    patches = []

    def init():
        return patches

    def animate(frame):
        ax.clear()
        ax.set_xlim(0, box_size)
        ax.set_ylim(0, box_size)
        ax.set_aspect('equal')
        pos = history[frame]
        for p in pos:
            circle = Circle(p, radius, fc='blue', ec='black', alpha=0.7)
            ax.add_patch(circle)
        return patches

    ani = animation.FuncAnimation(fig, animate, frames=len(history), init_func=init, interval=50, blit=False)
    plt.show()

if __name__ == "__main__":
    main()