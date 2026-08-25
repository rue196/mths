import numpy as np

import matplotlib.pyplot as plt

# ==========================================
# PARAMETERS
# ==========================================
YEARS = 80
DT = 1

# Initial population: [Young (0-20), Middle (20-60), Old (60+)]
pop_base = np.zeros((YEARS, 3))
pop_shock = np.zeros((YEARS, 3))
pop_base[0] = [1200, 5000, 1800]
pop_shock[0] = [1200, 5000, 1800]

# Demographic rates (annual)
aging_ym = 1 / 20.0   # Young -> Middle (after 20 years)
aging_mo = 1 / 40.0   # Middle -> Old (after 40 working years)
death_m = 0.005       # Middle mortality (small)
death_o_base = 1 / 20.0  # Old mortality (5% per year, ~20 yrs post-retirement)

# Economic parameters
birth_base = 0.08     # Baseline births per middle-aged adult per year
productivity_base = 1.0
productivity_growth = 0.015  # 1.5% annual TFP growth

# Supply-chain mapping parameters
young_dependency_cost = 0.6   # Cost to support young per capita
retiree_dependency_cost = 1.2 # Cost to support elderly (healthcare/pensions)
initial_ym_ratio = pop_base[0, 0] / pop_base[0, 1]

# Storage for metrics
metrics = {
    'base': {'surplus': np.zeros(YEARS), 'scarcity': np.zeros(YEARS), 
             'dep_ratio': np.zeros(YEARS), 'pop_mid': np.zeros(YEARS)},
    'shock': {'surplus': np.zeros(YEARS), 'scarcity': np.zeros(YEARS),
              'dep_ratio': np.zeros(YEARS), 'pop_mid': np.zeros(YEARS)}
}

# ==========================================
# SIMULATION FUNCTION
# ==========================================
def run_simulation(pop, is_shock=False):
    surplus = np.zeros(YEARS)
    scarcity = np.zeros(YEARS)
    dep_ratio = np.zeros(YEARS)
    
    for t in range(YEARS - 1):
        Y, M, O = pop[t, 0], pop[t, 1], pop[t, 2]
        
        # ----- Demographic Transitions -----
        # Apply shocks only to the 'shock' scenario
        if is_shock:
            # BOTTLENECK 1: Low birth rate after year 10
            birth = 0.04 if t > 10 else birth_base
            # BOTTLENECK 2: Retirement longevity shock after year 30
            death_o = 1 / 40.0 if t > 30 else death_o_base
        else:
            birth = birth_base
            death_o = death_o_base
        
        ym_flow = Y * aging_ym
        mo_flow = M * aging_mo
        births = M * birth
        deaths_o = O * death_o
        deaths_m = M * death_m
        
        # Update cohorts
        pop[t+1, 0] = Y - ym_flow + births
        pop[t+1, 1] = M - mo_flow - deaths_m + ym_flow
        pop[t+1, 2] = O - deaths_o + mo_flow
        
        # Guard against negatives
        pop[t+1] = np.maximum(pop[t+1], 0)
        
        # ----- Supply-Chain Depth Mapping -----
        # 1. Dependency Ratio (total drag on middle)
        dep_ratio[t] = (Y + O) / max(M, 1)
        
        # 2. Front-end scarcity (low young/middle ratio -> high binomial p)
        current_ym = Y / max(M, 1)
        # If current Y/M falls below initial, failure probability p1 rises.
        # p1 scales from 0.05 (abundant) to 0.7 (critical shortage)
        p1 = 0.05 + 0.65 * max(0, min(1, (1 - current_ym / initial_ym_ratio) * 1.8))
        # Scarcity multiplier (n=1 from Eq. 1: (1-p)^-n)
        scarcity[t] = 1 / (1 - p1)  # This compounds downstream
        
        # 3. Plus-sum labour value (middle generation productivity)
        productivity = productivity_base * (1 + productivity_growth) ** t
        V_L = M * productivity
        
        # 4. Raw / front-end cost (supporting the young)
        C0_raw = Y * young_dependency_cost
        
        # 5. Logistic drag / back-end cost (supporting retirees)
        L_fin = O * retiree_dependency_cost
        
        # 6. Total burden on the middle generation
        #    The scarcity amplifier multiplies the front-end cost (just like Eq. 5)
        total_burden = scarcity[t] * C0_raw + L_fin
        
        # 7. Surplus (capital left for investment / final consumption)
        surplus[t] = V_L - total_burden
        
    return surplus, scarcity, dep_ratio, pop

# ==========================================
# RUN BOTH SCENARIOS
# ==========================================
surplus_base, scarcity_base, dep_base, pop_base = run_simulation(pop_base, is_shock=False)
surplus_shock, scarcity_shock, dep_shock, pop_shock = run_simulation(pop_shock, is_shock=True)

# Store middle population for reference
metrics['base']['pop_mid'] = pop_base[:, 1]
metrics['shock']['pop_mid'] = pop_shock[:, 1]
metrics['base']['surplus'] = surplus_base
metrics['shock']['surplus'] = surplus_shock
metrics['base']['scarcity'] = scarcity_base
metrics['shock']['scarcity'] = scarcity_shock
metrics['base']['dep_ratio'] = dep_base
metrics['shock']['dep_ratio'] = dep_shock

# ==========================================
# PLOTTING
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
years = np.arange(YEARS)

# 1. Population Cohorts (Shock Scenario)
ax = axes[0, 0]
ax.stackplot(years, pop_shock[:, 0], pop_shock[:, 1], pop_shock[:, 2],
             labels=['Young', 'Middle (Plus-Sum)', 'Retired'],
             colors=['#1f77b4', '#ff7f0e', '#d62728'], alpha=0.8)
ax.set_title('Population Cohorts (Double-Bottleneck Shock)')
ax.set_ylabel('Population')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# 2. Dependency Ratio (Young+Old / Middle)
ax = axes[0, 1]
ax.plot(years, dep_base, label='Baseline', color='green', linewidth=2)
ax.plot(years, dep_shock, label='Double Bottleneck', color='red', linewidth=2)
ax.axvline(x=10, color='gray', linestyle='--', alpha=0.5, label='Birth Shock (t=10)')
ax.axvline(x=30, color='purple', linestyle='--', alpha=0.5, label='Retirement Shock (t=30)')
ax.set_title('Dependency Ratio: (Young + Old) / Middle')
ax.set_ylabel('Ratio')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Front-End Scarcity Multiplier (Phi_raw)
ax = axes[1, 0]
ax.plot(years, scarcity_base, label='Baseline', color='green', linewidth=2)
ax.plot(years, scarcity_shock, label='Double Bottleneck', color='red', linewidth=2)
ax.axhline(y=1.0, color='black', linestyle=':', alpha=0.7, label='Neutral (no scarcity)')
ax.set_title('Scarcity Amplifier ($\Phi_{raw}$) from Low Birth Rate')
ax.set_ylabel('Multiplier (1 / (1-p))')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Economic Surplus (Capital left after supporting both bottlenecks)
ax = axes[1, 1]
ax.plot(years, surplus_base, label='Baseline (Stable)', color='green', linewidth=2)
ax.plot(years, surplus_shock, label='Double Bottleneck', color='red', linewidth=2)
ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
ax.fill_between(years, 0, surplus_shock, where=(surplus_shock < 0), 
                color='red', alpha=0.3, label='Capital Erosion Zone')
ax.set_title('Economic Surplus / Investment Capital')
ax.set_ylabel('Surplus (units of value)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print final diagnostics
print("="*60)
print("FINAL DIAGNOSTICS (Year 79)")
print("="*60)
print(f"Baseline - Middle Pop: {pop_base[-1,1]:.0f}, Surplus: {surplus_base[-1]:.2f}, Scarcity: {scarcity_base[-1]:.2f}")
print(f"Shock   - Middle Pop: {pop_shock[-1,1]:.0f}, Surplus: {surplus_shock[-1]:.2f}, Scarcity: {scarcity_shock[-1]:.2f}")
print(f"Δ Middle Pop: {(pop_shock[-1,1]/pop_base[-1,1]-1)*100:.1f}%")
print(f"Δ Surplus: {(surplus_shock[-1]/max(surplus_base[-1],0.01)-1)*100:.1f}%")