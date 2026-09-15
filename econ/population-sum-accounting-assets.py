"""
Intergenerational Asset Allocation Under Double-Bottleneck Demographics
======================================================================
Models Japan and Italy using real data:
- Fertility rates, aging, and mortality (2024)
- Real estate vacancy rates and price dynamics
- Inflation-adjusted asset values
- Supply-demand equilibrium for housing
- Generational wealth concentration and the "€1 house" / "akiya" phenomenon

Key insight: Asset values (especially real estate) are implicitly priced on
future population replacement. When that replacement fails, the accounting
value collapses toward a non-speculative equilibrium (land value only).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# ---- Add these two imports at the top of your file ----
import os
from datetime import datetime


# ============================================================
# REAL-WORLD PARAMETERS (2024 data)
# ============================================================

# --- Japan (source: Ministry of Health, Labour and Welfare, 2024) ---
JP = {
    'name': 'Japan',
    'tfr': 1.15,                        # Total fertility rate 2024
    'births': 686061,                    # Live births 2024
    'deaths': 1620000,                   # Deaths 2024
    'population': 124000000,             # Approximate total population
    'pop_65_plus': 0.295,                # 29.5% aged 65+
    'vacancy_rate': 0.138,               # 13.8% vacant homes (2023)
    'vacant_homes': 9000000,             # 9 million vacant homes
    'housing_depreciation_years': 22,    # Wooden structures depreciate to zero in 22 years
    'homeownership_65plus': 0.845,       # 84.5% of 65+ own homes
    'immigration_share': 0.0321,         # 3.21% foreign population
    'house_price_index_2024': 132.148,   # 2015 = 100
    'inflation_2024': 0.036,             # 3.6% CPI Dec 2024
    'housing_cpi_2024': 0.008,           # 0.8% housing CPI
    'nominal_price_change': 0.045,       # Nominal house price change
}

# --- Italy (source: ISTAT, 2024) ---
IT = {
    'name': 'Italy',
    'tfr': 1.18,                         # Total fertility rate 2024
    'births': 370000,                    # Live births 2024
    'deaths': 651000,                    # Deaths 2024
    'population': 58900000,              # Approximate total population
    'pop_65_plus': 0.24,                 # ~24% aged 65+
    'vacancy_rate': 0.272,               # 27.2% non-permanently occupied
    'vacant_homes': 9600000,             # 9.6 million vacant homes
    'homeownership': 0.78,               # 78% homeownership rate
    'wealth_70plus_realestate': 2.72e12, # €2.72 trillion in real estate held by 70+
    'immigration_2024': 166000,          # Net immigration 2024
    'foreign_population': 5310000,       # 5.31 million foreign residents
    'house_price_index_2024': 113.50,    # 2015 = 100
    'inflation_2024': 0.013,             # 1.3% CPI Dec 2024
    'housing_cpi_2024': -0.022,          # -2.2% housing CPI
    'nominal_price_change': 0.045,       # 4.5% y-o-y Q4 2024
    'real_price_change': 0.032,          # 3.2% real house price change 2024
}

# --- Simulation parameters ---
YEARS = 80
DT = 1

# ============================================================
# GENERATIONAL COHORT MODEL
# ============================================================

def initialize_population(total_pop, pop_65_plus, tfr):
    """
    Initialize three cohorts: Young (0-20), Middle (20-60), Old (60+).
    Uses realistic age distribution based on aging share.
    """
    # Approximate cohort shares based on aging structure
    # For high-aging countries (Japan): more old, fewer young
    old_share = pop_65_plus
    # Middle: working age 20-60 (~55-60% in aging societies)
    # Young: 0-20 (~12-16% in aging societies)
    if pop_65_plus > 0.28:  # Japan-like
        middle_share = 0.545
        young_share = 1 - old_share - middle_share
    else:  # Italy-like
        middle_share = 0.58
        young_share = 1 - old_share - middle_share
    
    return np.array([
        total_pop * young_share,
        total_pop * middle_share,
        total_pop * old_share
    ])

def run_asset_simulation(pop_init, params, is_shock=False):
    """
    Run the full simulation with asset allocation.
    
    Returns a dictionary of time series for:
    - Population by cohort
    - Housing supply (occupied + vacant)
    - Housing demand (based on household formation)
    - Asset values (nominal and real)
    - Generational asset ownership
    - Scarcity multiplier and dependency ratios
    """
    Y, M, O = pop_init
    pop = np.zeros((YEARS, 3))
    pop[0] = [Y, M, O]
    
    # Extract demographic rates
    tfr = params['tfr']
    immigration_rate = params.get('immigration_share', 0.0)
    
    # Annual rates (approximated from TFR and life expectancy)
    # Birth rate: TFR * (women in childbearing age / total pop) / 35 years
    birth_rate = tfr * 0.015  # ~1.15 * 0.015 = 0.01725 for Japan
    death_rate_o = 1 / 20.0   # ~5% annual old-age mortality
    death_rate_m = 0.005      # Small middle-age mortality
    
    # Aging flows
    aging_ym = 1 / 20.0       # Young -> Middle over 20 years
    aging_mo = 1 / 40.0       # Middle -> Old over 40 years
    
    # Asset / housing parameters
    vacancy_rate_init = params['vacancy_rate']
    homeownership_old = params.get('homeownership_65plus', 
                                    params.get('homeownership', 0.8))
    housing_depr_years = params.get('housing_depreciation_years', 50)
    
    # Initial housing stock (occupied + vacant)
    initial_households = M * 0.6 + O * 0.7  # Household formation rate
    total_housing = initial_households / (1 - vacancy_rate_init)
    
    # Storage
    results = {
        'pop': pop,
        'housing_supply': np.zeros(YEARS),
        'housing_demand': np.zeros(YEARS),
        'vacancy_rate': np.zeros(YEARS),
        'nominal_price': np.zeros(YEARS),
        'real_price': np.zeros(YEARS),
        'asset_value_total': np.zeros(YEARS),
        'asset_value_old': np.zeros(YEARS),
        'asset_value_middle': np.zeros(YEARS),
        'asset_value_young': np.zeros(YEARS),
        'scarcity': np.zeros(YEARS),
        'dep_ratio': np.zeros(YEARS),
        'speculative_premium': np.zeros(YEARS),
        'replacement_dependency': np.zeros(YEARS),
        'immigration_effect': np.zeros(YEARS),
    }
    
    # Initial values
    results['housing_supply'][0] = total_housing
    results['housing_demand'][0] = initial_households
    results['vacancy_rate'][0] = vacancy_rate_init
    results['nominal_price'][0] = 1.0  # Normalized
    results['real_price'][0] = 1.0
    
    # Initial asset allocation: old hold most real estate
    total_asset_value = total_housing * 1.0  # Normalized
    results['asset_value_total'][0] = total_asset_value
    results['asset_value_old'][0] = total_asset_value * 0.6
    results['asset_value_middle'][0] = total_asset_value * 0.35
    results['asset_value_young'][0] = total_asset_value * 0.05
    
    for t in range(YEARS - 1):
        Y_t, M_t, O_t = pop[t]
        
        # --- Demographic transitions ---
        if is_shock:
            birth_rate_eff = birth_rate * 0.5 if t > 10 else birth_rate
            death_rate_o_eff = death_rate_o * 0.5 if t > 30 else death_rate_o
        else:
            birth_rate_eff = birth_rate
            death_rate_o_eff = death_rate_o
        
        # Immigration (population replacement)
        # Only partially offsets natural decline
        if is_shock:
            imm_effect = 0.0  # No immigration in shock scenario
        else:
            imm_effect = immigration_rate * params['population'] * 0.01
        results['immigration_effect'][t] = imm_effect
        
        births = M_t * birth_rate_eff
        deaths_o = O_t * death_rate_o_eff
        deaths_m = M_t * death_rate_m
        ym_flow = Y_t * aging_ym
        mo_flow = M_t * aging_mo
        
        pop[t+1, 0] = Y_t - ym_flow + births + imm_effect * 0.3
        pop[t+1, 1] = M_t - mo_flow - deaths_m + ym_flow + imm_effect * 0.5
        pop[t+1, 2] = O_t - deaths_o + mo_flow + imm_effect * 0.2
        pop[t+1] = np.maximum(pop[t+1], 0)
        
        # --- Housing supply and demand ---
        # Demand: based on household formation (middle + old, scaled by propensity)
        # Young don't form independent households at high rates in aging societies
        new_demand = M_t * 0.55 + O_t * 0.65
        results['housing_demand'][t+1] = new_demand
        
        # Supply: existing stock minus depreciation + new construction
        # Construction responds to price signals and demand
        if t > 0:
            price_signal = results['nominal_price'][t] / max(results['nominal_price'][t-1], 0.01)
        else:
            price_signal = 1.0
        
        # New construction is slow and responds to demand gap
        demand_gap = new_demand - results['housing_supply'][t]
        construction = max(0, demand_gap * 0.05) + max(0, price_signal - 1) * total_housing * 0.01
        
        # Depreciation: housing becomes worthless over time if not maintained
        # In Japan, wooden structures depreciate to zero in 22 years
        depreciation_rate = 1.0 / housing_depr_years
        depreciation = results['housing_supply'][t] * depreciation_rate * 0.5  # Partial
        
        results['housing_supply'][t+1] = max(
            results['housing_supply'][t] + construction - depreciation,
            new_demand * 0.95  # Floor: can't have less than 95% of demand met
        )
        
        # Vacancy rate
        vacant = results['housing_supply'][t+1] - new_demand
        results['vacancy_rate'][t+1] = max(0, vacant / max(results['housing_supply'][t+1], 1))
        
        # --- Price determination ---
        # Fundamental value: based on supply-demand balance
        # Speculative premium: based on expected future demand (population growth)
        
        # Fundamental component (land + replacement cost)
        supply_demand_ratio = results['housing_supply'][t+1] / max(new_demand, 1)
        fundamental_price = 1.0 / max(supply_demand_ratio, 0.5)  # Scarcity premium
        
        # Speculative component: depends on population momentum
        # When population grows, speculation drives prices above fundamentals
        # When population declines, speculative premium collapses
        pop_momentum = (pop[t+1, 1] + pop[t+1, 0]) / max(pop[t, 1] + pop[t, 0], 1) - 1
        
        # Replacement dependency: how much does price depend on future population?
        # If births are far below replacement, speculative premium is fragile
        replacement_ratio = births / max(deaths_o + deaths_m, 1)
        results['replacement_dependency'][t] = replacement_ratio
        
        # Speculative premium: high when replacement is strong, collapses when weak
        if replacement_ratio > 1.0:
            speculative_premium = 0.3 * min(replacement_ratio, 2.0)
        else:
            speculative_premium = 0.3 * replacement_ratio * (replacement_ratio ** 2)
        
        results['speculative_premium'][t] = speculative_premium
        
        # Nominal price = fundamental + speculative
        nominal_price = fundamental_price + speculative_premium
        
        # Inflation adjustment
        inflation = params['inflation_2024']
        if is_shock:
            inflation_eff = inflation * 0.5  # Deflationary pressure
        else:
            inflation_eff = inflation
        
        cumulative_inflation = (1 + inflation_eff) ** t
        real_price = nominal_price / max(cumulative_inflation, 1)
        
        # Housing CPI adjustment (housing-specific inflation)
        housing_cpi = params['housing_cpi_2024']
        housing_cpi_cumulative = (1 + housing_cpi) ** t
        real_housing_price = nominal_price / max(housing_cpi_cumulative, 0.1)
        
        results['nominal_price'][t+1] = nominal_price
        results['real_price'][t+1] = real_price
        
        # --- Asset values by generation ---
        # Total housing asset value
        total_value = nominal_price * results['housing_supply'][t+1]
        results['asset_value_total'][t+1] = total_value
        
        # Asset allocation shifts with demographics
        # Old generation holds most real estate (accumulated over lifetime)
        # As they die, assets transfer to middle generation (inheritance)
        # But if middle generation is small, assets become stranded
        
        old_share_of_pop = O_t / max(Y_t + M_t + O_t, 1)
        middle_share_of_pop = M_t / max(Y_t + M_t + O_t, 1)
        
        # Old generation owns 60-70% of housing assets
        # This share declines as they die, but only if heirs exist
        inheritance_rate = min(1.0, M_t / max(O_t * 0.5, 1))  # Heirs per elderly
        
        old_ownership = min(0.7, 0.5 + 0.2 * (O_t / max(M_t, 1)) ** 0.5)
        # Adjust for inheritance failure
        old_ownership *= (1 - 0.3 * (1 - inheritance_rate))
        
        results['asset_value_old'][t+1] = total_value * old_ownership
        results['asset_value_middle'][t+1] = total_value * (1 - old_ownership) * 0.7
        results['asset_value_young'][t+1] = total_value * (1 - old_ownership) * 0.3
        
        # --- Scarcity multiplier (from demographic bottleneck) ---
        initial_ym_ratio = pop_init[0] / pop_init[1]
        current_ym = pop[t+1, 0] / max(pop[t+1, 1], 1)
        p1 = 0.05 + 0.65 * max(0, min(1, (1 - current_ym / initial_ym_ratio) * 1.8))
        results['scarcity'][t+1] = 1 / (1 - p1)
        
        # Dependency ratio
        results['dep_ratio'][t+1] = (pop[t+1, 0] + pop[t+1, 2]) / max(pop[t+1, 1], 1)
    
    return results

# ============================================================
# RUN SIMULATIONS
# ============================================================

# Initialize populations
pop_jp = initialize_population(JP['population'], JP['pop_65_plus'], JP['tfr'])
pop_it = initialize_population(IT['population'], IT['pop_65_plus'], IT['tfr'])

# Run baseline and shock scenarios for both countries
results_jp_base = run_asset_simulation(pop_jp.copy(), JP, is_shock=False)
results_jp_shock = run_asset_simulation(pop_jp.copy(), JP, is_shock=True)
results_it_base = run_asset_simulation(pop_it.copy(), IT, is_shock=False)
results_it_shock = run_asset_simulation(pop_it.copy(), IT, is_shock=True)

# ============================================================
# VISUALIZATION
# ============================================================
# ============================================================
# VISUALIZATION (fixed)
# ============================================================

fig = plt.figure(figsize=(18, 20))
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.30)

years = np.arange(YEARS)

# --- Row 1: Population Dynamics ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.stackplot(years, results_jp_shock['pop'][:, 0], results_jp_shock['pop'][:, 1],
              results_jp_shock['pop'][:, 2],
              labels=['Young', 'Middle', 'Old'],
              colors=['#4A90D9', '#F5A623', '#D0021B'], alpha=0.8)
ax1.set_title('Japan — Population Cohorts (Shock)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Population')
ax1.legend(loc='upper right', fontsize=8)
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
ax2.stackplot(years, results_it_shock['pop'][:, 0], results_it_shock['pop'][:, 1],
              results_it_shock['pop'][:, 2],
              labels=['Young', 'Middle', 'Old'],
              colors=['#4A90D9', '#F5A623', '#D0021B'], alpha=0.8)
ax2.set_title('Italy — Population Cohorts (Shock)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Population')
ax2.legend(loc='upper right', fontsize=8)
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(years, results_jp_base['pop'][:, 0], label='JP Young', color='#4A90D9', ls='--')
ax3.plot(years, results_jp_base['pop'][:, 1], label='JP Middle', color='#F5A623', ls='--')
ax3.plot(years, results_jp_base['pop'][:, 2], label='JP Old', color='#D0021B', ls='--')
ax3.plot(years, results_it_base['pop'][:, 0], label='IT Young', color='#4A90D9')
ax3.plot(years, results_it_base['pop'][:, 1], label='IT Middle', color='#F5A623')
ax3.plot(years, results_it_base['pop'][:, 2], label='IT Old', color='#D0021B')
ax3.set_title('Baseline Population Comparison', fontsize=11, fontweight='bold')
ax3.set_ylabel('Population')
ax3.legend(loc='upper right', fontsize=7, ncol=2)
ax3.grid(True, alpha=0.3)

# --- Row 2: Housing Supply, Demand, Vacancy ---
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(years, results_jp_base['housing_demand'], label='Demand', color='blue', lw=2)
ax4.plot(years, results_jp_base['housing_supply'], label='Supply', color='green', lw=2)
ax4.plot(years, results_jp_shock['housing_demand'], label='Demand (Shock)', color='blue', ls='--', lw=1.5)
ax4.plot(years, results_jp_shock['housing_supply'], label='Supply (Shock)', color='green', ls='--', lw=1.5)
ax4.set_title('Japan — Housing Supply & Demand', fontsize=11, fontweight='bold')
ax4.set_ylabel('Units')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(years, results_it_base['housing_demand'], label='Demand', color='blue', lw=2)
ax5.plot(years, results_it_base['housing_supply'], label='Supply', color='green', lw=2)
ax5.plot(years, results_it_shock['housing_demand'], label='Demand (Shock)', color='blue', ls='--', lw=1.5)
ax5.plot(years, results_it_shock['housing_supply'], label='Supply (Shock)', color='green', ls='--', lw=1.5)
ax5.set_title('Italy — Housing Supply & Demand', fontsize=11, fontweight='bold')
ax5.set_ylabel('Units')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

ax6 = fig.add_subplot(gs[1, 2])
ax6.plot(years, results_jp_base['vacancy_rate'] * 100, label='JP Baseline', color='green', lw=2)
ax6.plot(years, results_jp_shock['vacancy_rate'] * 100, label='JP Shock', color='red', lw=2, ls='--')
ax6.plot(years, results_it_base['vacancy_rate'] * 100, label='IT Baseline', color='darkblue', lw=2)
ax6.plot(years, results_it_shock['vacancy_rate'] * 100, label='IT Shock', color='purple', lw=2, ls='--')
ax6.axhline(y=13.8, color='gray', ls=':', alpha=0.5, label='JP 2023 actual (13.8%)')
ax6.axhline(y=27.2, color='gray', ls='-.', alpha=0.5, label='IT actual (27.2%)')
ax6.set_title('Vacancy Rate (%)', fontsize=11, fontweight='bold')
ax6.set_ylabel('Vacancy Rate (%)')
ax6.legend(fontsize=7)
ax6.grid(True, alpha=0.3)

# --- Row 3: Asset Prices & Generational Allocation ---
ax7 = fig.add_subplot(gs[2, 0])
ax7.plot(years, results_jp_base['nominal_price'], label='JP Nominal', color='blue', lw=2)
ax7.plot(years, results_jp_base['real_price'], label='JP Real (CPI-adj)', color='blue', ls='--', lw=2)
ax7.plot(years, results_jp_shock['nominal_price'], label='JP Nominal (Shock)', color='red', lw=1.5)
ax7.plot(years, results_jp_shock['real_price'], label='JP Real (Shock)', color='red', ls='--', lw=1.5)
ax7.axhline(y=1.0, color='black', ls=':', alpha=0.5)
ax7.set_title('Japan — Housing Price Index (Normalized)', fontsize=11, fontweight='bold')
ax7.set_ylabel('Price Index')
ax7.legend(fontsize=7)
ax7.grid(True, alpha=0.3)

ax8 = fig.add_subplot(gs[2, 1])
ax8.plot(years, results_it_base['nominal_price'], label='IT Nominal', color='darkgreen', lw=2)
ax8.plot(years, results_it_base['real_price'], label='IT Real (CPI-adj)', color='darkgreen', ls='--', lw=2)
ax8.plot(years, results_it_shock['nominal_price'], label='IT Nominal (Shock)', color='red', lw=1.5)
ax8.plot(years, results_it_shock['real_price'], label='IT Real (Shock)', color='red', ls='--', lw=1.5)
ax8.axhline(y=1.0, color='black', ls=':', alpha=0.5)
ax8.set_title('Italy — Housing Price Index (Normalized)', fontsize=11, fontweight='bold')
ax8.set_ylabel('Price Index')
ax8.legend(fontsize=7)
ax8.grid(True, alpha=0.3)

ax9 = fig.add_subplot(gs[2, 2])
ax9.plot(years, results_jp_shock['speculative_premium'] * 100, label='JP Speculative Premium', color='red', lw=2)
ax9.plot(years, results_it_shock['speculative_premium'] * 100, label='IT Speculative Premium', color='purple', lw=2)
ax9.plot(years, results_jp_base['speculative_premium'] * 100, label='JP Baseline', color='red', ls='--', lw=1.5)
ax9.plot(years, results_it_base['speculative_premium'] * 100, label='IT Baseline', color='purple', ls='--', lw=1.5)
ax9.set_title('Speculative Premium (% of Price)', fontsize=11, fontweight='bold')
ax9.set_ylabel('Premium (%)')
ax9.legend(fontsize=7)
ax9.grid(True, alpha=0.3)

# --- Row 4: Generational Asset Ownership & Replacement Dependency ---
ax10 = fig.add_subplot(gs[3, 0])
ax10.stackplot(years,
               results_jp_shock['asset_value_old'] / 1e6,
               results_jp_shock['asset_value_middle'] / 1e6,
               results_jp_shock['asset_value_young'] / 1e6,
               labels=['Old (60+)', 'Middle (20-60)', 'Young (0-20)'],
               colors=['#D0021B', '#F5A623', '#4A90D9'], alpha=0.8)
ax10.set_title('Japan — Generational Asset Ownership (Shock)', fontsize=11, fontweight='bold')
ax10.set_ylabel('Asset Value (millions)')
ax10.legend(fontsize=8)
ax10.grid(True, alpha=0.3)

ax11 = fig.add_subplot(gs[3, 1])
ax11.stackplot(years,
               results_it_shock['asset_value_old'] / 1e6,
               results_it_shock['asset_value_middle'] / 1e6,
               results_it_shock['asset_value_young'] / 1e6,
               labels=['Old (60+)', 'Middle (20-60)', 'Young (0-20)'],
               colors=['#D0021B', '#F5A623', '#4A90D9'], alpha=0.8)
ax11.set_title('Italy — Generational Asset Ownership (Shock)', fontsize=11, fontweight='bold')
ax11.set_ylabel('Asset Value (millions)')
ax11.legend(fontsize=8)
ax11.grid(True, alpha=0.3)

ax12 = fig.add_subplot(gs[3, 2])
ax12.plot(years, results_jp_shock['replacement_dependency'], label='JP Replacement Ratio', color='red', lw=2)
ax12.plot(years, results_it_shock['replacement_dependency'], label='IT Replacement Ratio', color='purple', lw=2)
ax12.plot(years, results_jp_base['replacement_dependency'], label='JP Baseline', color='red', ls='--', lw=1.5)
ax12.plot(years, results_it_base['replacement_dependency'], label='IT Baseline', color='purple', ls='--', lw=1.5)
ax12.axhline(y=1.0, color='black', ls=':', label='Replacement threshold')
ax12.axhline(y=2.1, color='green', ls=':', alpha=0.5, label='TFR replacement (2.1)')
ax12.set_title('Births / Deaths (Replacement Ratio)', fontsize=11, fontweight='bold')
ax12.set_ylabel('Ratio')
ax12.legend(fontsize=7)
ax12.grid(True, alpha=0.3)

fig.suptitle('Intergenerational Asset Allocation Under Double-Bottleneck Demographics\n'
             'Japan & Italy — Real Data (2024) with Inflation-Adjusted Housing Prices',
             fontsize=14, fontweight='bold', y=0.995)

# ---- Use subplots_adjust instead of tight_layout (avoids the gridspec warning) ----
fig.subplots_adjust(top=0.96, bottom=0.04, left=0.06, right=0.98,
                    hspace=0.45, wspace=0.30)

# ---- Portable save path (works on Windows, macOS, Linux) ----
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "asset_allocation_simulation.png")
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved figure to: {out_path}")

plt.show()
