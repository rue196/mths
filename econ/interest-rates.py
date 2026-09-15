import numpy as np
import matplotlib.pyplot as plt
import math

# ---------- Constants ----------
ALPHA = 1.0 / (math.pi - math.e)   # ≈ 2.362

# ---------- Spectral derivative (from equilibrium.py) ----------
def dzeta_dt(t, c, alpha=ALPHA):
    K = (len(c) - 1) // 2
    deriv = 0.0
    for i in range(1, K + 1):
        theta = t * i / alpha
        deriv += -2.0 * c[K + i] * (i / alpha) * np.sin(theta)
    return deriv

# ---------- Binomial scarcity multiplier ----------
def scarcity_multiplier(p, n):
    return (1.0 - p) ** (-n)

# ---------- Single stage cost (with interest) ----------
def stage_cost(C_prev, m, p, n, ell, w, eta, e, tau, d, mu, rho):
    """
    Compute cost at a stage given:
      C_prev  : input cost from previous stage
      m       : material conversion ratio
      p, n    : binomial failure probability and trials
      ell, w  : labour hours and wage
      eta, e  : energy intensity and price
      tau, d  : transport cost per km and distance
      mu      : stage markup
      rho     : interest rate (effective at this stage)
    Returns:
      C_k     : output cost of this stage
    """
    # Scarcity amplifier
    amp = scarcity_multiplier(p, n)
    # Material cost (includes waste)
    mat_cost = m * C_prev * amp
    # Labour, energy, logistics
    labor_cost = ell * w
    energy_cost = eta * e
    logist_cost = tau * d
    # Markup on input cost
    markup = mu * C_prev
    # Sum before interest
    stage_sum = mat_cost + labor_cost + energy_cost + logist_cost + markup
    # Apply interest (multiply entire stage cost)
    C_k = (1.0 + rho) * stage_sum
    return C_k

# ---------- Full supply chain simulation ----------
def simulate_chain(C0, stage_params, rho, retail_mu=0.35):
    """
    stage_params: list of tuples (m, p, n, ell, w, eta, e, tau, d, mu)
    rho: interest rate (applied uniformly to all stages)
    Returns: final price, labour share, raw share, cost_stack
    """
    C = C0
    cost_stack = [C0]
    total_labor_value = 0.0
    for params in stage_params:
        m, p, n, ell, w, eta, e, tau, d, mu = params
        # Recompute with given rho
        C = stage_cost(C, m, p, n, ell, w, eta, e, tau, d, mu, rho)
        cost_stack.append(C)
        # Accumulate labour value (plus-sum) amplified by downstream interest? 
        # Labour value at this stage is ell*w, but it gets amplified by downstream multipliers.
        # For labour share we'll later use formula (6) from the paper, but here we store raw cost components.
        # We'll store each stage's labour contribution (not amplified) for later share calculation.
        total_labor_value += ell * w * (1.0 + rho)  # simple approximation
    # Final consumer price with retail margin
    P_fin = C * (1.0 + retail_mu)
    # Raw material share: C0 * product of (m * amp + mu) factors? For simplicity, we'll compute as C0 / (sum of all costs)
    # Better: raw share = (C0 * product of all material conversion factors including interest) / P_fin
    # We'll approximate as C0 / P_fin (simplified)
    raw_share = C0 / P_fin
    labor_share = total_labor_value / P_fin
    return P_fin, raw_share, labor_share, cost_stack

# ---------- Parameter setup ----------
d = 8   # depth
# Define stage parameters (some realistic values)
# Stage 0 is raw material extraction (not a stage in the chain, just C0)
C0 = 1.0  # base raw cost

# For each stage: (m, p, n, ell, w, eta, e, tau, d, mu)
stage_params = [
    # Stage 1: Primary processing
    (1.2, 0.08, 1, 0.5, 20.0, 0.5, 0.10, 0.002, 100, 0.05),
    # Stage 2: Secondary processing
    (1.1, 0.05, 1, 0.3, 25.0, 1.0, 0.12, 0.0015, 200, 0.08),
    # Stage 3: Tertiary
    (1.0, 0.02, 1, 0.4, 30.0, 0.8, 0.11, 0.001, 300, 0.10),
    # Stage 4: Component manufacture
    (1.3, 0.10, 2, 0.6, 35.0, 2.0, 0.15, 0.002, 500, 0.12),
    # Stage 5: Assembly
    (1.1, 0.03, 1, 1.0, 40.0, 1.5, 0.13, 0.001, 200, 0.15),
    # Stage 6: Testing & packaging
    (1.0, 0.01, 1, 0.8, 45.0, 0.3, 0.10, 0.001, 50, 0.05),
    # Stage 7: Logistics & distribution
    (1.0, 0.02, 1, 0.2, 25.0, 0.1, 0.12, 0.005, 1000, 0.03),
    # Stage 8: Retail prep (final stage before consumer)
    (1.0, 0.01, 1, 0.1, 30.0, 0.0, 0.0, 0.0, 0, 0.20),
]

# ---------- Sweep interest rates ----------
rho_range = np.linspace(0.0, 0.20, 50)   # 0 to 20% interest
final_prices = []
raw_shares = []
labor_shares = []
cost_stacks = []

for rho in rho_range:
    P_fin, raw_share, labor_share, stack = simulate_chain(C0, stage_params, rho, retail_mu=0.35)
    final_prices.append(P_fin)
    raw_shares.append(raw_share)
    labor_shares.append(labor_share)
    cost_stacks.append(stack)

# ---------- Plot results ----------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Final price vs interest rate
axes[0,0].plot(rho_range, final_prices, 'b-', label='Final consumer price')
axes[0,0].set_xlabel('Interest rate (rho)')
axes[0,0].set_ylabel('Price (USD)')
axes[0,0].set_title('Effect of interest rates on final price')
axes[0,0].grid(True)

# Labour and raw material shares
axes[0,1].plot(rho_range, labor_shares, 'r-', label='Labour share')
axes[0,1].plot(rho_range, raw_shares, 'g--', label='Raw material share')
axes[0,1].set_xlabel('Interest rate (rho)')
axes[0,1].set_ylabel('Share of final price')
axes[0,1].set_title('Cost structure vs interest rate')
axes[0,1].legend()
axes[0,1].grid(True)

# Inflation: derivative of price wrt rho
inflation = np.gradient(final_prices, rho_range) / final_prices
axes[1,0].plot(rho_range[1:], inflation[1:], 'm-')
axes[1,0].set_xlabel('Interest rate (rho)')
axes[1,0].set_ylabel('Inflation (dP/P / drho)')
axes[1,0].set_title('Price sensitivity to interest rates')
axes[1,0].grid(True)

# Determinant and trace of the cost stack (as equilibrium indicators)
# We'll compute the determinant and trace of the matrix M = [[C0, C1], [C2, C3]] as an example.
# We'll take the first four costs from the stack (C0, C1, C2, C3).
# For each rho, we have a stack; we compute det = C0*C3 - C1*C2, trace = C0+C3.
dets = []
traces = []
for stack in cost_stacks:
    # take first four elements as a 2x2 matrix
    if len(stack) >= 4:
        M = np.array([[stack[0], stack[1]], [stack[2], stack[3]]])
        dets.append(np.linalg.det(M))
        traces.append(np.trace(M))
    else:
        dets.append(np.nan)
        traces.append(np.nan)

axes[1,1].plot(rho_range, dets, 'k-', label='Determinant of [C0 C1; C2 C3]')
axes[1,1].plot(rho_range, traces, 'c--', label='Trace')
axes[1,1].set_xlabel('Interest rate (rho)')
axes[1,1].set_ylabel('Det / Trace')
axes[1,1].legend()
axes[1,1].grid(True)
axes[1,1].set_title('Equilibrium indicators (cost matrix)')

plt.tight_layout()
plt.show()

# ---------- Also simulate time evolution with varying interest rate ----------
# Let's make interest rate follow the spectral derivative
time = np.linspace(0, 50, 100)
c = np.ones(51)  # spectral coefficients
C0 = 1.0
# We'll compute prices over time using a time-varying rho(t) = base + 0.1 * dzeta_dt
base_rho = 0.05
prices_time = []
rhos_time = []
for t in time:
    d_zeta = dzeta_dt(t, c, ALPHA)
    rho_t = base_rho + 0.1 * d_zeta   # scale to keep positive
    rho_t = max(0.0, rho_t)
    rhos_time.append(rho_t)
    P_fin, _, _, _ = simulate_chain(C0, stage_params, rho_t, retail_mu=0.35)
    prices_time.append(P_fin)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(time, rhos_time, 'r-', label='Interest rate rho(t)')
plt.xlabel('Time')
plt.ylabel('rho')
plt.grid(True)
plt.legend()
plt.subplot(1,2,2)
plt.plot(time, prices_time, 'b-', label='Final price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.grid(True)
plt.legend()
plt.suptitle('Time evolution driven by spectral derivative')
plt.tight_layout()
plt.show()

print("Simulation complete.")