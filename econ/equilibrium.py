import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

# ------------------------------------------------------------
# 1. Spectral sum and derivative (mapping from Ruby code)
# ------------------------------------------------------------
def zeta(t, c, alpha):
    """
    Compute spectral sum ζ(t) = Σ C_i * exp(i * t * i / alpha)
    with symmetric real coefficients C_i = C_{-i}.
    """
    K = (len(c) - 1) // 2
    sum_val = c[K]  # i=0 term
    for i in range(1, K+1):
        theta = t * i / alpha
        sum_val += c[K + i] * np.cos(theta) * 2  # because C_i = C_{-i}
    return sum_val

def dzeta_dt(t, c, alpha):
    """
    Analytical derivative dζ/dt = -2 Σ i/α * C_i * sin(i*t/α)
    """
    K = (len(c) - 1) // 2
    deriv = 0.0
    for i in range(1, K+1):
        theta = t * i / alpha
        deriv += -2.0 * c[K + i] * (i / alpha) * np.sin(theta)
    return deriv

# ------------------------------------------------------------
# 2. Binomial scarcity model (from supply-chain depth paper)
# ------------------------------------------------------------
def scarcity_multiplier(p, n):
    """(1 - p)^(-n)   binomial amplifier"""
    return (1 - p) ** (-n)

# ------------------------------------------------------------
# 3. Simulation parameters
# ------------------------------------------------------------
T = 125.0
dt = 2.5
time = np.arange(0, T, dt)

# Spectral coefficients (symmetric, all ones as in Ruby)
K = 25
alpha = 0.3628
c = np.ones(2 * K + 1)   # C_i = 1 for all i

# Initial conditions for matrix M = [[x^-1, y^-1], [x^i, y^i]]
x_inv0 = 1.0   # scarcity of x
y_inv0 = 0.8   # scarcity of y
x_i0 = 1.0     # demand factor for x
y_i0 = 1.2     # demand factor for y

# Binomial parameters for x and y (scarcity waves)
p0_x, p_amp_x, freq_x, n_x = 0.2, 0.15, 0.5, 2
p0_y, p_amp_y, freq_y, n_y = 0.3, 0.10, 0.3, 3

# Storage
M = np.zeros((len(time), 2, 2))
M[0, :, :] = [[x_inv0, y_inv0], [x_i0, y_i0]]

# ------------------------------------------------------------
# 4. Simulation loop
# ------------------------------------------------------------
for idx in range(1, len(time)):
    t = time[idx]
    t_prev = time[idx-1]
    
    # --- Compute derivative of spectral sum at time t ---
    d_zeta = dzeta_dt(t, c, alpha)   # scalar
    
    # --- Scarcity factors evolve driven by d_zeta and binomial waves ---
    # Binomial failure probability oscillates (trials waves)
    p_x = p0_x + p_amp_x * np.sin(2 * np.pi * freq_x * t)
    p_y = p0_y + p_amp_y * np.sin(2 * np.pi * freq_y * t)
    # Ensure p stays in [0,1)
    p_x = np.clip(p_x, 0.0, 0.99)
    p_y = np.clip(p_y, 0.0, 0.99)
    
    # Scarcity multipliers (algebraic part)
    Sx = scarcity_multiplier(p_x, n_x)
    Sy = scarcity_multiplier(p_y, n_y)
    
    # Update scarcity row (x^-1, y^-1) using d_zeta as a growth rate
    # We use d_zeta to drive changes in scarcity (like a gradient)
    # New scarcity = old * (1 + d_zeta * dt) but also attracted to binomial baseline
    # This couples the spectral derivative to the binomial scarcity.
    x_inv = M[idx-1, 0, 0] * (1 + d_zeta * dt) + (Sx - M[idx-1, 0, 0]) * 0.01
    y_inv = M[idx-1, 0, 1] * (1 + d_zeta * dt) + (Sy - M[idx-1, 0, 1]) * 0.01
    # Ensure positive
    x_inv = max(x_inv, 0.1)
    y_inv = max(y_inv, 0.1)
    
    # --- Demand/time factors (transcendental) also driven by d_zeta ---
    # They evolve with a combination of time-decay and spectral forcing
    x_i = M[idx-1, 1, 0] * (1 + 0.02 * d_zeta * dt) + 0.01 * np.sin(2 * np.pi * 0.1 * t)
    y_i = M[idx-1, 1, 1] * (1 + 0.02 * d_zeta * dt) + 0.01 * np.cos(2 * np.pi * 0.08 * t)
    # Keep within reasonable bounds
    x_i = np.clip(x_i, 0.1, 5.0)
    y_i = np.clip(y_i, 0.1, 5.0)
    
    # Store matrix
    M[idx, :, :] = [[x_inv, y_inv], [x_i, y_i]]

# ------------------------------------------------------------
# 5. Compute derived quantities
# ------------------------------------------------------------
# Equilibrium scalar: determinant of M (or trace)
det = M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]
trace = M[:, 0, 0] + M[:, 1, 1]   # sum of diagonals

# A "consumer price" analog: product of scarcity and demand factors
price_x = M[:, 0, 0] * M[:, 1, 0]   # x^-1 * x^i = x^(i-1)  (i variable)
price_y = M[:, 0, 1] * M[:, 1, 1]

# ------------------------------------------------------------
# 6. Plotting
# ------------------------------------------------------------
fig, axes = plt.subplots(3, 2, figsize=(15, 12))

# Spectral derivative
ax = axes[0, 0]
ax.plot(time, [dzeta_dt(t, c, alpha) for t in time], color='purple')
ax.set_title('Spectral Derivative $d\zeta/dt$')
ax.set_xlabel('Time')
ax.grid(True, alpha=0.3)

# Scarcity row (x^-1, y^-1)
ax = axes[0, 1]
ax.plot(time, M[:, 0, 0], label='$x^{-1}$ (scarcity x)', color='red')
ax.plot(time, M[:, 0, 1], label='$y^{-1}$ (scarcity y)', color='blue')
ax.set_title('Scarcity Factors (binomial + spectral forcing)')
ax.legend()
ax.grid(True, alpha=0.3)

# Demand row (x^i, y^i)
ax = axes[1, 0]
ax.plot(time, M[:, 1, 0], label='$x^{i}$ (demand x)', color='orange')
ax.plot(time, M[:, 1, 1], label='$y^{i}$ (demand y)', color='green')
ax.set_title('Transcendental Demand Factors')
ax.legend()
ax.grid(True, alpha=0.3)

# Price-like derived quantities
ax = axes[1, 1]
ax.plot(time, price_x, label='Price x = $x^{-1} x^{i}$', color='crimson')
ax.plot(time, price_y, label='Price y = $y^{-1} y^{i}$', color='navy')
ax.set_title('Composite Price (scarcity × demand)')
ax.legend()
ax.grid(True, alpha=0.3)

# Determinant of M
ax = axes[2, 0]
ax.plot(time, det, color='black')
ax.axhline(y=0, linestyle='--', color='gray')
ax.set_title('Determinant of M (equilibrium indicator)')
ax.grid(True, alpha=0.3)

# Trace of M
ax = axes[2, 1]
ax.plot(time, trace, color='darkgreen')
ax.set_title('Trace of M')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 7. Print summary
# ------------------------------------------------------------
print("=== Simulation Summary ===")
print(f"Final scarcity: x^-1 = {M[-1,0,0]:.3f}, y^-1 = {M[-1,0,1]:.3f}")
print(f"Final demand:   x^i  = {M[-1,1,0]:.3f}, y^i  = {M[-1,1,1]:.3f}")
print(f"Final price x:  {price_x[-1]:.3f}, final price y: {price_y[-1]:.3f}")