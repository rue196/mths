import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit  # logistic function

# ===========================================
# 1. SUPPLY CHAIN PARAMETERS (Depth d = 6)
# ===========================================
d = 6
C0 = 1.0  # Baseline raw material cost

# Base Stage Parameters (from Eq. 3)
m_base = np.array([1.2, 1.5, 1.3, 1.1, 1.0, 1.0])  # Material conversion ratios
p_base = np.array([0.10, 0.05, 0.04, 0.03, 0.02, 0.01])  # Failure probability
n_base = np.array([1, 1, 2, 1, 1, 1])  # Trials per stage
labour = np.array([2.0, 4.0, 8.0, 12.0, 5.0, 3.0])  # Labour plus-sum (ℓ_k w_k)
energy = np.array([1.0, 3.0, 6.0, 2.0, 1.0, 0.5])  # Energy cost (η_k e_k)
logistics = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 1.0])  # Logistics cost (τ_k d_k)
mu_base = np.array([0.05, 0.08, 0.12, 0.15, 0.10, 0.05])  # Stage markup
mu_retail_base = 0.25  # Retail margin

# Speculation sensitivity (how much m and mu inflate during a bubble)
beta_m = 1.2  # Max 120% increase in material orders
beta_mu = 0.8  # Max 80% increase in markups

# Consumer demand parameters
Q_max = 1000  # Maximum market size
demand_elasticity = 0.015  # Price sensitivity

# Inventory / "Spoilage" parameters
inv_capacity = 200  # Free storage limit
obsolescence_rate = 0.08  # 8% of inventory value decays per year (economic spoilage)
fire_sale_threshold = 300  # If inventory > this, panic selling begins

# ===========================================
# 2. SIMULATION LOOP
# ===========================================
T = 100  # 100 years
spec = np.zeros(T)
P_fin = np.zeros(T)
Q_demand = np.zeros(T)
Q_produced = np.zeros(T)
inventory = np.zeros(T)
realized_price = np.zeros(T)
destroyed_capital = np.zeros(T)
cost_stack = np.zeros(T)

# Initial demand
Q_demand[0] = 500

for t in range(T):
    # --- 2a. Speculation Factor (Growth Trap wave) ---
    # Rises smoothly, peaks at t=50, then crashes.
    # Uses a logistic rise + exponential decay to mimic a "bubble".
    rise = expit((t - 25) / 8)  # S-curve from 0 to 1
    decay = np.exp(-0.06 * max(0, t - 50))  # Crash after peak
    spec[t] = 1.0 + 3.0 * rise * decay  # Spec multiplier ranges from 1.0 to ~4.0

    # --- 2b. Inflate Stage Parameters based on Speculation ---
    # Producers over-order materials (m) and raise margins (mu) expecting future scarcity
    m = m_base * (1 + beta_m * (spec[t] - 1) / 3.0)
    mu = mu_base * (1 + beta_mu * (spec[t] - 1) / 3.0)

    # --- 2c. Stage-by-Stage Cost Propagation (Equation 3) ---
    C_prev = C0
    for k in range(d):
        binomial_scarcity = (1 - p_base[k]) ** (-n_base[k])
        # The stage cost = (input * conversion * scarcity) + labour + energy + logistics + markup*input
        C_k = (m[k] * C_prev * binomial_scarcity) + labour[k] + energy[k] + logistics[k] + (mu[k] * C_prev)
        C_prev = C_k
    cost_stack[t] = C_prev  # This is C_d

    # Final consumer price before retail margin
    P_fin[t] = C_prev * (1 + mu_retail_base)
    # During a bubble, retailers also add a speculative premium on top
    if spec[t] > 1.5:
        P_fin[t] *= (1 + 0.15 * (spec[t] - 1.5))  # Extra retail gouging

    # --- 2d. Consumer Demand (real, not speculative) ---
    # Demand grows with time (population/income) but collapses if price is too high
    base_demand_trend = Q_max * (1 - 0.5 * np.exp(-0.03 * t))  # Saturation curve
    price_penalty = 1 / (1 + demand_elasticity * P_fin[t])
    Q_demand[t] = base_demand_trend * price_penalty
    # Add a late-stage demand shock (consumer fatigue / recession)
    if t > 60:
        Q_demand[t] *= (1 - 0.02 * (t - 60))

    # --- 2e. Production (driven by speculation, not current demand) ---
    # Producers extrapolate past demand + aggressively stockpile for the "boom"
    if t == 0:
        Q_produced[t] = Q_demand[0] * 1.1
    else:
        speculative_boost = 1 + 0.6 * (spec[t] - 1)  # Build inventory for future sales
        Q_produced[t] = Q_demand[t-1] * speculative_boost + 10 * spec[t]

    # --- 2f. Inventory Dynamics ("Spoilage" of capital) ---
    if t > 0:
        inventory[t] = inventory[t-1] + Q_produced[t] - Q_demand[t]
    else:
        inventory[t] = Q_produced[t] - Q_demand[t]
    inventory[t] = max(0, inventory[t])  # Can't be negative

    # --- 2g. Economic Spoilage / Obsolescence ---
    # If inventory exceeds capacity, it starts to decay (like spoilage, but for capital)
    excess_inv = max(0, inventory[t] - inv_capacity)
    obsolescence_cost = obsolescence_rate * excess_inv * P_fin[t]
    destroyed_capital[t] = obsolescence_cost

    # --- 2h. Realized Price (When bubble pops, retail margin collapses) ---
    # If inventory piles up, retailers slash prices to clear stock (fire sale)
    if inventory[t] > fire_sale_threshold:
        # Dump inventory: realized price falls below cost stack
        clearance_factor = max(0.2, 1 - (inventory[t] - fire_sale_threshold) / (fire_sale_threshold * 2))
        realized_price[t] = cost_stack[t] * clearance_factor
        # During fire sales, the extra inventory "spoils" further as unsold obsolete goods
        destroyed_capital[t] += (inventory[t] - fire_sale_threshold) * cost_stack[t] * 0.1
    else:
        realized_price[t] = P_fin[t]  # Normal retail price

    # If realized price < cost stack, that's negative capital formation
    if realized_price[t] < cost_stack[t]:
        destroyed_capital[t] += (cost_stack[t] - realized_price[t]) * Q_demand[t] * 0.5

# ===========================================
# 3. PLOTTING
# ===========================================
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
years = np.arange(T)

# Plot 1: Speculation Factor
ax = axes[0, 0]
ax.plot(years, spec, color='purple', linewidth=2)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
ax.set_title('Speculation Factor (Bubble Growth & Crash)')
ax.set_ylabel('Multiplier')
ax.grid(True, alpha=0.3)

# Plot 2: Price Dynamics (Cost Stack vs Final Price vs Realized Price)
ax = axes[0, 1]
ax.plot(years, cost_stack, label='Cost Stack ($C_d$)', color='blue', linewidth=2)
ax.plot(years, P_fin, label='Asking Price ($P_{fin}$)', color='green', linestyle='--', linewidth=2)
ax.plot(years, realized_price, label='Realized Price (Fire Sale)', color='red', linewidth=2)
ax.set_title('Price Bubble and Crash (Growth Trap)')
ax.set_ylabel('Price (units)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Production vs Real Demand
ax = axes[1, 0]
ax.fill_between(years, 0, Q_produced, color='orange', alpha=0.4, label='Production (Speculative)')
ax.fill_between(years, 0, Q_demand, color='blue', alpha=0.4, label='Consumer Demand')
ax.set_title('Supply-Demand Mismatch')
ax.set_ylabel('Quantity')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Inventory (The "Spoilage" of capital)
ax = axes[1, 1]
ax.fill_between(years, 0, inventory, color='darkred', alpha=0.5)
ax.axhline(y=inv_capacity, color='green', linestyle=':', label='Free Storage Limit')
ax.axhline(y=fire_sale_threshold, color='purple', linestyle='--', label='Fire-sale Threshold')
ax.set_title('Unsold Inventory (Capital Spoilage)')
ax.set_ylabel('Units in Storage')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Destroyed Capital (Obsolescence + Fire-sale losses)
ax = axes[2, 0]
ax.fill_between(years, 0, destroyed_capital, color='black', alpha=0.3)
ax.set_title('Destroyed Economic Value (Spoilage & Write-downs)')
ax.set_ylabel('Value Lost (units)')
ax.grid(True, alpha=0.3)

# Plot 6: Total Sector Profit (Revenue - Production Cost)
revenue = realized_price * Q_demand
prod_cost = cost_stack * Q_produced
# Account for holding costs (storage)
holding_cost = 0.05 * inventory * cost_stack
profit = revenue - prod_cost - holding_cost
ax = axes[2, 1]
ax.plot(years, profit, color='blue', linewidth=2)
ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
ax.fill_between(years, 0, profit, where=(profit < 0), color='red', alpha=0.3, label='Loss Zone')
ax.set_title('Industry Profit (Growth Trap Bites)')
ax.set_ylabel('Net Profit')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Final diagnostics
print("="*60)
print("GROWTH TRAP DIAGNOSTICS")
print("="*60)
print(f"Peak Speculation at t={np.argmax(spec)}: {np.max(spec):.2f}x")
print(f"Peak Asking Price: {np.max(P_fin):.2f}")
print(f"Peak Inventory: {np.max(inventory):.0f} units")
print(f"Total Destroyed Capital (over 100 yrs): {np.sum(destroyed_capital):.0f} units")
print(f"Final Profit (Year 99): {profit[-1]:.2f}")
