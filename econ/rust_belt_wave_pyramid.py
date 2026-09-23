"""
Wave-Pyramid Model of Rust Belt Deindustrialization and
Superstar-City Population Replacement (1970-2024)
========================================================
Applies the "Unified Wave-Pyramid" framework (binomial access decay +
logistic ceiling + scarcity-compounding supply chains) to a concrete,
data-anchored case: the post-1978 offshoring of US manufacturing to
China and Mexico, the resulting outmigration of working-age people from
the Rust Belt to a handful of "superstar" service-economy metros, and
the resulting divergence in housing / cost-of-living between the two.

REAL DATA ANCHORS (not invented):
- US manufacturing employment 1970-2024, millions (BLS / FRED series
  LFEAMNTTUSA647N; Hightower Lowdown 2016 citing Working America; St.
  Louis Fed "Sluggish Renaissance of US Manufacturing", Aug 2025).
- Historical wave markers: Shenzhen Special Economic Zone (1980, first
  of Deng Xiaoping's SEZs), NAFTA (Jan 1994), China's WTO accession
  (Dec 2001), the Global Financial Crisis (2008-09), and the 2018+
  tariff/reshoring era.

WHAT IS ILLUSTRATIVE, NOT CENSUS DATA:
Regional population splits ("Rust Belt" vs "Superstar Cities"), housing
elasticities, wage/pull sensitivities, and the CPI shelter pass-through
are calibrated aggregates chosen to reproduce the *known qualitative and
rough quantitative* pattern (Rust Belt population stagnation/decline,
big metro housing premiums, youth outmigration) -- they are not drawn
from a specific Census/BEA table for those two aggregates, since no
single official series defines "the Rust Belt" or "superstar cities" as
a region. Treat the shapes of the curves as the model's claim, and the
manufacturing-employment curve underneath them as the real anchor.

This is a simplified teaching/illustration model, not a calibrated
econometric forecast.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# ============================================================
# 1. REAL DATA: US MANUFACTURING EMPLOYMENT, 1970-2024
# ============================================================
MFG_EMPLOYMENT_ANCHORS = {   # millions of jobs
    1970: 17.8,
    1979: 19.5,   # all-time peak, June 1979
    1994: 17.0,   # NAFTA takes effect (Jan 1994)
    2000: 17.3,   # pre-China-shock local peak
    2001: 17.1,   # China joins WTO (Dec 2001)
    2010: 11.5,   # post-Financial-Crisis trough
    2018: 12.8,
    2024: 12.6,   # BLS / St. Louis Fed, Q4 2024
}

WAVES = [
    {"label": "Pre-SEZ",          "start": 1970, "end": 1980,
     "event": "1978-80: Deng Xiaoping reforms; Shenzhen SEZ opens (1980)"},
    {"label": "Early Offshoring", "start": 1980, "end": 1994,
     "event": "1994: NAFTA enters into force"},
    {"label": "NAFTA Era",        "start": 1994, "end": 2001,
     "event": "2001: China joins WTO ('China Shock' begins)"},
    {"label": "China Shock",      "start": 2001, "end": 2010,
     "event": "2008-09: Global Financial Crisis"},
    {"label": "Post-Crisis",      "start": 2010, "end": 2018,
     "event": "2018: US-China tariff war begins"},
    {"label": "Tariff/Reshoring", "start": 2018, "end": 2024,
     "event": "2020-24: COVID shock + reshoring push"},
]

YEARS = np.arange(1970, 2025)
N = len(YEARS)

def interp_anchors(anchors):
    xs = sorted(anchors)
    ys = [anchors[x] for x in xs]
    return np.interp(YEARS, xs, ys)

MFG = interp_anchors(MFG_EMPLOYMENT_ANCHORS)          # millions, real series
mfg_growth = np.gradient(MFG) / MFG                    # annual growth rate
p_fail = np.clip(-mfg_growth, 0, None)                 # binomial "job-fails-to-survive" rate

def wave_of(year):
    for w in WAVES:
        if w["start"] <= year < w["end"]:
            return w
    return WAVES[-1]

# ============================================================
# 2. REGIONAL COHORT MODEL: RUST BELT vs SUPERSTAR CITIES
# ============================================================
# Illustrative starting stocks (millions of people), calibrated so the
# *relative* dynamics -- not the absolute headcounts -- match the known
# direction: Rust Belt working-age population stagnates/declines while
# a small set of superstar metros (Moretti 2012's "great divergence")
# absorb disproportionate high-skill in-migration.
RB0 = np.array([6.0, 16.0, 5.0])   # Rust Belt:      [young, working, old]
SC0 = np.array([3.0, 9.0, 2.0])    # Superstar cities:[young, working, old]
# Note on magnitude: the model's ~70% working-age decline is far steeper
# than "the Rust Belt" taken as whole states (Ohio/Pennsylvania grew
# slightly overall, just slower than the national/Sun Belt average).
# It is, however, close to the real experience of the manufacturing-
# dependent legacy cities that anchor the Rust Belt narrative: Detroit
# fell from ~1.85M (1950) to ~630K (2020s), Youngstown and Flint each
# lost 55-65% of their populations over the same window. Read the RB
# region here as "legacy manufacturing communities," not entire states.

BASE_BIRTH        = 0.013
DEATH_OLD         = 1/22
DEATH_WORKING     = 0.003
AGING_Y2W         = 1/20
AGING_W2O         = 1/40
BASE_MOBILITY     = 0.003      # background inter-regional mobility, any era
PUSH_SENSITIVITY  = 2.2        # how hard local manufacturing job-loss pushes workers out
PULL_SENSITIVITY  = 0.12       # how hard big-metro services growth pulls them in
MAX_MIGRATION_RATE = 0.032     # cap: real Rust Belt states lost population share, not
                                # their whole population -- annual net outmigration of a
                                # few percent compounds to a large *relative* divergence
                                # without implausibly emptying the region out entirely
SERVICES_BASE     = 0.016      # baseline superstar-city services-job growth/yr
SERVICES_BOOST = {             # extra post-shock agglomeration pull (finance/tech/services)
    2001: 0.012, 2010: 0.015, 2018: 0.008,
}

def services_growth(year):
    g = SERVICES_BASE
    for boost_year, boost in SERVICES_BOOST.items():
        if year >= boost_year:
            g += boost * np.exp(-0.05 * (year - boost_year))  # boost decays slowly
    return g

# ============================================================
# 3. HOUSING & NECESSITIES PRICE MODEL
# ============================================================
# Superstar cities: near-zero supply elasticity (zoning-constrained), so
# price obeys the wave-pyramid Model-I *scarcity regime*
# C(n) = C0 (1-p)^-n -- here p_zoning plays the binomial access-failure
# role for a new housing unit ever getting permitted and built.
P_ZONING_FAIL_SC          = 0.80
CONSTRUCTION_ELASTICITY_SC = 0.12

# Rust Belt: supply becomes *abundant* as people leave (vacant housing,
# closed factories) -- mirrors the akiya/vacant-home dynamic: price
# floors toward Pfloor = alpha0 / (I * rho).
ALPHA0             = 0.35
I_QUALITY_RB       = 0.55   # aging infrastructure quality, held roughly fixed
RHO_DENSITY_RB     = 0.60   # remaining industrial/material density
VACANCY_DECAY_RB   = 0.35   # how fast oversupply drags price toward the floor
PFLOOR_RB          = ALPHA0 / (I_QUALITY_RB * RHO_DENSITY_RB)

SHELTER_CPI_SHARE  = 0.33   # BLS CPI shelter weight - necessities pass-through

# ============================================================
# 4. SIMULATION
# ============================================================
rb = np.zeros((N, 3)); rb[0] = RB0
sc = np.zeros((N, 3)); sc[0] = SC0

price_rb = np.ones(N)
price_sc = np.ones(N)
supply_sc = np.ones(N) * (SC0[0] + SC0[1])
necessities_rb = np.ones(N)
necessities_sc = np.ones(N)
net_youth_outflow_rb = np.zeros(N)        # per-year young+working leaving Rust Belt
cum_youth_outflow_rb = np.zeros(N)
mfg_job_fail_p = p_fail

# "Myth gap" (PDF Sec 5.5): naive extrapolation of Rust Belt housing
# value assuming the pre-1979 manufacturing growth trend had continued,
# versus what the model actually produces once replacement fails.
pre1979_years = [y for y in MFG_EMPLOYMENT_ANCHORS if y <= 1979]
pre1979_cagr = (MFG_EMPLOYMENT_ANCHORS[1979] / MFG_EMPLOYMENT_ANCHORS[1970]) ** (1/9) - 1
naive_price_rb = np.ones(N)

for t in range(N - 1):
    year = YEARS[t]
    Y_rb, W_rb, O_rb = rb[t]
    Y_sc, W_sc, O_sc = sc[t]

    # --- binomial-uplift-driven push/pull migration ---
    push = PUSH_SENSITIVITY * mfg_job_fail_p[t]                 # job losses push workers out
    pull = PULL_SENSITIVITY * services_growth(year)             # services growth pulls workers in
    migration_rate = np.clip(BASE_MOBILITY + push + pull, 0, MAX_MIGRATION_RATE)

    movers_young   = Y_rb * migration_rate
    movers_working = W_rb * migration_rate * 1.2  # working-age/high-skill move disproportionately
    total_movers = movers_young + movers_working
    net_youth_outflow_rb[t] = total_movers
    cum_youth_outflow_rb[t+1] = cum_youth_outflow_rb[t] + total_movers

    # --- demographic transitions (both regions) ---
    births_rb = W_rb * BASE_BIRTH
    births_sc = W_sc * BASE_BIRTH
    deaths_o_rb, deaths_o_sc = O_rb * DEATH_OLD, O_sc * DEATH_OLD
    deaths_w_rb, deaths_w_sc = W_rb * DEATH_WORKING, W_sc * DEATH_WORKING

    rb[t+1, 0] = Y_rb - Y_rb*AGING_Y2W + births_rb - movers_young
    rb[t+1, 1] = W_rb - W_rb*AGING_W2O - deaths_w_rb + Y_rb*AGING_Y2W - movers_working
    rb[t+1, 2] = O_rb - deaths_o_rb + W_rb*AGING_W2O

    sc[t+1, 0] = Y_sc - Y_sc*AGING_Y2W + births_sc + movers_young
    sc[t+1, 1] = W_sc - W_sc*AGING_W2O - deaths_w_sc + Y_sc*AGING_Y2W + movers_working
    sc[t+1, 2] = O_sc - deaths_o_sc + W_sc*AGING_W2O

    rb[t+1] = np.maximum(rb[t+1], 0)
    sc[t+1] = np.maximum(sc[t+1], 0)

    # --- Superstar city housing: scarcity-compounding (Model I regime) ---
    demand_sc = sc[t+1, 0] + sc[t+1, 1]
    supply_sc[t+1] = supply_sc[t] + CONSTRUCTION_ELASTICITY_SC * max(demand_sc - supply_sc[t], 0)
    excess = max((demand_sc - supply_sc[t+1]) / max(supply_sc[t+1], 1e-6), 0)
    # Model-I scarcity-compounding form (cost ~ (1-p)^-n), calibrated to a
    # linearized growth rate so the compounding stays plausible over a
    # 55-year horizon: each year's excess demand raises price by a fraction
    # governed by the zoning failure probability, and (being sticky) it
    # never falls back just because inflow slows.
    price_sc[t+1] = price_sc[t] * (1 + P_ZONING_FAIL_SC * excess)

    # --- Rust Belt housing: oversupply drags price toward the floor ---
    demand_rb = rb[t+1, 0] + rb[t+1, 1]
    supply_rb_ref = RB0[0] + RB0[1]                      # legacy housing stock, roughly fixed
    vacancy = max(0.0, (supply_rb_ref - demand_rb) / supply_rb_ref)
    price_rb[t+1] = max(PFLOOR_RB, price_rb[t] * (1 - VACANCY_DECAY_RB * vacancy * 0.05))

    # --- myth gap: naive pre-1979-trend extrapolation for the Rust Belt ---
    naive_price_rb[t+1] = naive_price_rb[t] * (1 + pre1979_cagr * 0.6)

    # --- necessities / cost-of-living pass-through (CPI shelter share) ---
    necessities_rb[t+1] = 1 + SHELTER_CPI_SHARE * (price_rb[t+1] - 1)
    necessities_sc[t+1] = 1 + SHELTER_CPI_SHARE * (price_sc[t+1] - 1)

myth_gap_rb = naive_price_rb - price_rb

# ============================================================
# 5. GENERALIZATION CHECK: rural -> manufacturing -> services tipping
# ============================================================
# The same mechanism that hollows out the Rust Belt is what filled China's
# coastal cities: real, widely-reported UN/World Bank urbanization shares
# put China's urban population at roughly 19% (1980) rising to ~66%
# (2024) as the SEZ-driven manufacturing boom pulled rural labor into
# coastal manufacturing zones, which are now themselves feeding a rising
# services/city tier -- the same country-village -> manufacturing ->
# services-city cascade the wave-pyramid paper describes, just running in
# the opposite geographic direction from the US case above.
CHINA_URBAN_SHARE_ANCHORS = {1980: 0.19, 1994: 0.30, 2001: 0.37, 2010: 0.50, 2018: 0.60, 2024: 0.66}
china_urban_share = interp_anchors(CHINA_URBAN_SHARE_ANCHORS)

# ============================================================
# 6. VISUALIZATION
# ============================================================
fig = plt.figure(figsize=(17, 14))
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.28)

def shade_waves(ax):
    for i, w in enumerate(WAVES):
        ax.axvspan(w["start"], w["end"], color="gray", alpha=0.05 if i % 2 == 0 else 0.0)
    for w in WAVES[1:]:
        ax.axvline(w["start"], color="gray", ls=":", lw=0.8, alpha=0.6)

# --- Panel 1: real manufacturing employment with wave markers ---
ax1 = fig.add_subplot(gs[0, 0])
shade_waves(ax1)
ax1.plot(YEARS, MFG, color="#8B0000", lw=2.2)
for x in [1980, 1994, 2001, 2010, 2018]:
    ax1.axvline(x, color="gray", ls=":", lw=0.8)
ax1.annotate("Shenzhen SEZ\n(1980)", (1980, 19.0), fontsize=7, ha="center")
ax1.annotate("NAFTA\n(1994)", (1994, 18.3), fontsize=7, ha="center")
ax1.annotate("China WTO\n(2001)", (2001, 17.6), fontsize=7, ha="center")
ax1.annotate("Financial\nCrisis", (2010, 11.0), fontsize=7, ha="center")
ax1.set_title("US Manufacturing Employment (real, BLS/FRED)", fontsize=11, fontweight="bold")
ax1.set_ylabel("Millions of jobs")
ax1.grid(alpha=0.3)

# --- Panel 2: Rust Belt vs Superstar city working-age population ---
ax2 = fig.add_subplot(gs[0, 1])
shade_waves(ax2)
ax2.plot(YEARS, rb[:, 0] + rb[:, 1], color="#4A90D9", lw=2, label="Rust Belt (young+working)")
ax2.plot(YEARS, sc[:, 0] + sc[:, 1], color="#D0021B", lw=2, label="Superstar cities (young+working)")
ax2.set_title("Working-Age Population by Region", fontsize=11, fontweight="bold")
ax2.set_ylabel("Millions")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# --- Panel 3: cumulative youth/working-age outflow from Rust Belt ---
ax3 = fig.add_subplot(gs[1, 0])
shade_waves(ax3)
ax3.plot(YEARS, cum_youth_outflow_rb, color="#F5A623", lw=2.2)
ax3.fill_between(YEARS, 0, cum_youth_outflow_rb, color="#F5A623", alpha=0.25)
ax3.set_title("Cumulative Young + Working-Age Outmigration\nfrom Rust Belt to Superstar Cities", fontsize=11, fontweight="bold")
ax3.set_ylabel("Millions (cumulative)")
ax3.grid(alpha=0.3)

# --- Panel 4: housing price divergence ---
ax4 = fig.add_subplot(gs[1, 1])
shade_waves(ax4)
ax4.plot(YEARS, price_sc, color="#D0021B", lw=2.2, label="Superstar cities")
ax4.plot(YEARS, price_rb, color="#4A90D9", lw=2.2, label="Rust Belt")
ax4.axhline(PFLOOR_RB, color="#4A90D9", ls=":", lw=1, label="Rust Belt price floor")
ax4.set_title("Housing Price Index (1970 = 1.0)", fontsize=11, fontweight="bold")
ax4.set_ylabel("Index")
ax4.set_yscale("log")
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3)

# --- Panel 5: necessities / cost-of-living + myth gap ---
ax5 = fig.add_subplot(gs[2, 0])
shade_waves(ax5)
ax5.plot(YEARS, necessities_sc, color="#D0021B", lw=2, label="Superstar-city necessities index")
ax5.plot(YEARS, necessities_rb, color="#4A90D9", lw=2, label="Rust Belt necessities index")
ax5.set_title("Cost-of-Necessities Index (shelter pass-through)", fontsize=11, fontweight="bold")
ax5.set_ylabel("Index")
ax5.legend(fontsize=8)
ax5.grid(alpha=0.3)

ax5b = ax5.twinx()
ax5b.plot(YEARS, myth_gap_rb, color="gray", lw=1.3, ls="--", label="Myth gap (naive - actual, Rust Belt)")
ax5b.set_ylabel("Myth gap", color="gray", fontsize=8)
ax5b.tick_params(axis="y", labelcolor="gray")

# --- Panel 6: generalization -- China's mirrored rural->city cascade ---
ax6 = fig.add_subplot(gs[2, 1])
ax6.plot(YEARS[YEARS >= 1980], china_urban_share[YEARS >= 1980] * 100, color="#2E7D32", lw=2.2)
ax6.set_title("Generalization: China Urban Population Share\n(real, UN/World Bank; opposite-direction mirror)", fontsize=11, fontweight="bold")
ax6.set_ylabel("% urban")
ax6.grid(alpha=0.3)

fig.suptitle(
    "Wave-Pyramid Model: Manufacturing Offshoring, Rust Belt Outmigration,\n"
    "and Superstar-City Housing Scarcity (1970-2024)",
    fontsize=14, fontweight="bold", y=0.995,
)
fig.subplots_adjust(top=0.92, bottom=0.05, left=0.06, right=0.96, hspace=0.5, wspace=0.28)

out_dir = "/mnt/user-data/outputs"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "rust_belt_wave_pyramid_simulation.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved figure to: {out_path}")

# ============================================================
# 7. SUMMARY PRINTOUT
# ============================================================
print("\n--- Wave summary ---")
for w in WAVES:
    idx = (YEARS >= w["start"]) & (YEARS < w["end"])
    print(f"{w['label']:<18} {w['start']}-{w['end']:<6} mfg jobs: "
          f"{MFG[idx][0]:.1f}M -> {MFG[idx][-1]:.1f}M | {w['event']}")

print(f"\nTotal modeled young+working-age outmigration from Rust Belt, "
      f"1970-2024: {cum_youth_outflow_rb[-1]:.1f} million")
print(f"Superstar-city housing index in 2024: {price_sc[-1]:.1f}x (1970=1.0)")
print(f"Rust Belt housing index in 2024: {price_rb[-1]:.2f}x (1970=1.0), "
      f"floor = {PFLOOR_RB:.2f}x")
