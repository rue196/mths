import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math

# Constants
ALPHA = 1.0 / (math.pi - math.e)

def local_supertrace(positions, window=5):
    """Sliding window supertrace (alternating sum of distances from mean)."""
    n = len(positions)
    S = np.zeros(n)
    for i in range(n):
        start = max(0, i - window//2)
        end = min(n, i + window//2 + 1)
        vals = positions[start:end]
        # Use distances from local mean as coefficients
        mean = np.mean(vals)
        coeffs = vals - mean
        # SuperTrace: alternating sum of absolute values
        s = 0.0
        for j, c in enumerate(coeffs):
            sign = 1 if j % 2 == 0 else -1
            s += sign * abs(c)
        S[i] = s
    return S

def entropy_from_supertrace(S, N, alpha=ALPHA):
    if S == 0: return 0.0
    p = abs(S) / N
    if p <= 0: return 0.0
    return -alpha * p * math.log(p)

def contraction_length(S):
    return abs(S) ** (1/6)

# Simulation parameters
N = 100           # number of particles
t_steps = 200
dt = 0.1

# Initialise positions (slightly perturbed from equilibrium)
x = np.linspace(0, 10, N) + 0.1 * np.random.randn(N)
v = np.zeros(N)

# Time evolution: wave equation with supertrace feedback
# We'll store the local supertrace and contraction length for each time.
S_hist = []
ell_hist = []

for t in range(t_steps):
    # Compute local supertrace (window of 5)
    S = local_supertrace(x, window=7)
    ell = contraction_length(S)
    S_hist.append(S.copy())
    ell_hist.append(ell.copy())
    
    # Force: each particle feels a restoring force towards neighbours,
    # modulated by the local contraction length (larger ell -> stronger spring?)
    # We'll simply update positions with a wave-like equation using the gradient of ell.
    # For simplicity, use a standard wave equation with local speed proportional to ell.
    # Actually, we can let the acceleration be proportional to the second derivative of x,
    # scaled by ell.
    dx = np.gradient(x)
    d2x = np.gradient(dx)
    # Wave speed c = 1 + 0.5 * ell (so that larger ell -> faster propagation)
    c = 1.0 + 0.3 * ell
    a = c * d2x
    v += a * dt
    x += v * dt
    # Soft boundaries
    x = np.clip(x, 0, 10)

# ---- Animation ----
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

# Pre-plot lines
line1, = ax1.plot([], [], 'b-', lw=2, label='Positions')
line2, = ax2.plot([], [], 'r-', lw=2, label='Supertrace S')
line3, = ax3.plot([], [], 'g-', lw=2, label='Contraction length ℓ')

ax1.set_ylim(-0.5, 10.5)
ax1.set_ylabel('Position')
ax2.set_ylim(-0.5, 0.5)
ax2.set_ylabel('S')
ax3.set_ylim(0, 0.5)
ax3.set_ylabel('ℓ')
ax3.set_xlabel('Particle index')

def update(frame):
    line1.set_data(range(N), x)
    line2.set_data(range(N), S_hist[frame])
    line3.set_data(range(N), ell_hist[frame])
    ax1.set_title(f'Frame {frame}')
    return line1, line2, line3

ani = FuncAnimation(fig, update, frames=len(S_hist), interval=50)
plt.tight_layout()
plt.show()