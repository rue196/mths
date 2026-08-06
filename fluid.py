import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math

# ---------- SuperTrace and Entropy functions (from earlier) ----------
ALPHA = 1.0 / (math.pi - math.e)

def supertrace_from_coeffs(C):
    """C is a list of complex numbers (the z_j). Compute alternating sum of |C_i|."""
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
    """Compute m = |S| * exp(-H) from the complex coefficients."""
    S = supertrace_from_coeffs(C)
    H = entropy_from_supertrace(S, len(C))
    return abs(S) * math.exp(-H)

# ---------- Fluid simulation ----------
def fluid_simulation(grid_size=10, num_steps=200, dt=0.05, diffusion=0.1, 
                     force_strength=0.5, seed=42):
    """
    Simulate a 2D fluid with complex state z = real (density) + i*v (velocity).
    The invariant scalar m drives the flow via a force proportional to its gradient.
    """
    np.random.seed(seed)
    
    # Grid of particles
    N = grid_size * grid_size
    # Initial state: random small perturbations around a constant
    z = np.zeros(N, dtype=complex)
    # Real part: density-like field (0..1)
    z.real = 0.5 + 0.1 * np.random.randn(N)
    # Imag part: velocity (small random)
    z.imag = 0.1 * np.random.randn(N)
    
    # 2D positions for plotting (grid)
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    pos = np.column_stack((X.ravel(), Y.ravel()))
    
    # Store history for animation
    real_hist = [z.real.copy().reshape(grid_size, grid_size)]
    imag_hist = [z.imag.copy().reshape(grid_size, grid_size)]
    
    # Precompute neighbor indices for diffusion (4-neighbor)
    idx_2d = np.arange(N).reshape(grid_size, grid_size)
    neighbors = []
    for i in range(grid_size):
        for j in range(grid_size):
            neigh = []
            if i > 0: neigh.append(idx_2d[i-1, j])
            if i < grid_size-1: neigh.append(idx_2d[i+1, j])
            if j > 0: neigh.append(idx_2d[i, j-1])
            if j < grid_size-1: neigh.append(idx_2d[i, j+1])
            neighbors.append(neigh)
    
    # Time evolution
    for step in range(num_steps):
        # 1. Compute invariant scalar from current state
        m = invariant_scalar(z)
        
        # 2. Compute force: gradient of m? But m is global, so we need a local field.
        # Instead, we treat m as a global scaling factor for a random force field.
        # To mimic pressure, we compute local gradients of the real part (density).
        # Use the real part as density, compute its gradient, and apply force proportional to m.
        rho = z.real.reshape(grid_size, grid_size)
        # Compute gradient using finite differences
        grad_x = np.zeros_like(rho)
        grad_y = np.zeros_like(rho)
        grad_x[1:-1, :] = (rho[2:, :] - rho[:-2, :]) / 2
        grad_y[:, 1:-1] = (rho[:, 2:] - rho[:, :-2]) / 2
        # Flatten
        grad_x = grad_x.ravel()
        grad_y = grad_y.ravel()
        
        # 3. Update velocities (imag part) with force proportional to m * gradient
        # Also add diffusion (Laplacian) to smooth
        force = force_strength * m * (grad_x + 1j * grad_y)
        # Diffusion: average of neighbors
        diff = np.zeros(N, dtype=complex)
        for i, neigh in enumerate(neighbors):
            if neigh:
                diff[i] = np.mean(z[neigh]) - z[i]
        z.imag += dt * (force.imag + diffusion * diff.imag)
        # Also evolve real part with advection (using velocity)
        # Simple advection: move density along velocity
        # We do a simple Euler step: real += dt * velocity (but velocity is imag part)
        # However, we need to handle boundary conditions; we'll just update with small dt.
        z.real += dt * 0.1 * z.imag  # advection coefficient
        
        # 4. Apply damping to prevent instability
        z.imag *= 0.98
        z.real = np.clip(z.real, 0, 1)
        
        # Store for animation
        real_hist.append(z.real.copy().reshape(grid_size, grid_size))
        imag_hist.append(z.imag.copy().reshape(grid_size, grid_size))
    
    # ---- Animation ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.set_title('Real part (density)')
    ax2.set_title('Imag part (velocity)')
    
    im1 = ax1.imshow(real_hist[0], cmap='viridis', vmin=0, vmax=1, animated=True)
    im2 = ax2.imshow(imag_hist[0], cmap='RdBu', vmin=-0.5, vmax=0.5, animated=True)
    plt.colorbar(im1, ax=ax1)
    plt.colorbar(im2, ax=ax2)
    
    def update(frame):
        im1.set_array(real_hist[frame])
        im2.set_array(imag_hist[frame])
        return im1, im2
    
    ani = FuncAnimation(fig, update, frames=len(real_hist), interval=50, blit=True)
    plt.tight_layout()
    plt.show()
    
    return z, real_hist, imag_hist

if __name__ == "__main__":
    fluid_simulation(grid_size=10, num_steps=150, dt=0.05, force_strength=0.5)