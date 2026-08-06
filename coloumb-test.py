import numpy as np
import matplotlib.pyplot as plt
import math

# ---------- SuperTrace / Entropy functions (vectorized) ----------
ALPHA = 1.0 / (math.pi - math.e)

def entropy_from_supertrace(S, N, alpha=ALPHA):
    """
    Compute entropy H = -alpha * (|S|/N) * log(|S|/N).
    Handles both scalar and array inputs.
    """
    # Convert to array if not already
    S = np.asarray(S)
    p = np.abs(S) / N
    # Avoid log(0) by setting p=0 -> 0
    with np.errstate(divide='ignore', invalid='ignore'):
        H = -alpha * p * np.log(p)
        H = np.nan_to_num(H, nan=0.0, posinf=0.0)
    return H

def contraction_length(S):
    """ℓ_c = |S|^(1/6), works with arrays."""
    return np.abs(S) ** (1/6)

def strong_mass(S, N=6):
    """m = |S| * exp(-H)."""
    H = entropy_from_supertrace(S, N)
    return np.abs(S) * np.exp(-H)

# ---------- Potential model (fully vectorized) ----------
def nuclear_potential(r, S0=1.0, r0=1.0, alpha_s=10.0, k_e=1.44, N=6):
    """
    Compute strong and Coulomb potentials for nucleon separation r (array).
    S(r) = S0 * exp(-r/r0).
    Returns: V_strong, V_Coulomb, V_total, m_strong, ell_c, H, S.
    """
    r = np.asarray(r)
    S = S0 * np.exp(-r / r0)
    H = entropy_from_supertrace(S, N)
    m = strong_mass(S, N)
    ell_c = contraction_length(S)
    # Avoid division by zero at r=0
    with np.errstate(divide='ignore', invalid='ignore'):
        V_Coulomb = np.divide(k_e, r, out=np.full_like(r, np.inf), where=r>0)
    V_strong = -alpha_s * m
    V_total = V_strong + V_Coulomb
    return V_strong, V_Coulomb, V_total, m, ell_c, H, S

# ---------- Simulation parameters ----------
r = np.linspace(0.2, 10, 500)          # distance in fm
S0 = 1.0                               # supertrace scale
r0 = 1.0                               # force range (fm)
alpha_s = 12.0                         # strong coupling (adjust to see barrier)
k_e = 1.44                             # Coulomb constant (MeV·fm)

# Compute potentials
V_strong, V_Coulomb, V_total, m_strong, ell_c, H, S = nuclear_potential(
    r, S0, r0, alpha_s, k_e, N=6)

# ---------- Plotting (same as before) ----------
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Supertrace and entropy
axes[0,0].plot(r, S, 'b-', label='S(r)')
axes[0,0].set_xlabel('r (fm)')
axes[0,0].set_ylabel('S')
axes[0,0].set_title('Supertrace magnitude')
axes[0,0].grid(True)

axes[0,1].plot(r, H, 'r-', label='H(r)')
axes[0,1].set_xlabel('r (fm)')
axes[0,1].set_ylabel('H')
axes[0,1].set_title('Entropy')
axes[0,1].grid(True)

# 2. Contraction length and its inverse (energy scale)
axes[0,2].plot(r, ell_c, 'g-', label='ℓ_c = |S|^{1/6}')
axes[0,2].set_xlabel('r (fm)')
axes[0,2].set_ylabel('ℓ_c')
axes[0,2].set_title('Contraction length')
axes[0,2].grid(True)

# 3. Strong mass (invariant scalar)
axes[1,0].plot(r, m_strong, 'm-', label='m_strong = |S| exp(-H)')
axes[1,0].axhline(y=m_strong[0]*0.9, color='gray', linestyle='--', label='~constant for r<r0')
axes[1,0].set_xlabel('r (fm)')
axes[1,0].set_ylabel('m_strong')
axes[1,0].set_title('Strong mass (invariant)')
axes[1,0].legend()
axes[1,0].grid(True)

# 4. Potentials
axes[1,1].plot(r, V_strong, 'b--', label='Strong (attractive)')
axes[1,1].plot(r, V_Coulomb, 'r--', label='Coulomb (repulsive)')
axes[1,1].plot(r, V_total, 'k-', linewidth=2, label='Total')
axes[1,1].axhline(0, color='gray', linestyle=':')
axes[1,1].set_xlabel('r (fm)')
axes[1,1].set_ylabel('Potential (MeV)')
axes[1,1].set_title('Nuclear potential')
axes[1,1].legend()
axes[1,1].grid(True)

# 5. Energy required to mediate (1/ℓ_c)
axes[1,2].plot(r, 1/(ell_c + 1e-12), 'c-', label='1/ℓ_c')
axes[1,2].set_xlabel('r (fm)')
axes[1,2].set_ylabel('Energy (arb.)')
axes[1,2].set_title('Energy cost to mediate')
axes[1,2].grid(True)

plt.tight_layout()
plt.show()

# ---------- Additional analysis ----------
V_min = np.min(V_total)
r_min = r[np.argmin(V_total)]
# Find barrier: maximum after the minimum (if any)
mask = r > r_min
if np.any(mask):
    V_barrier = np.max(V_total[mask]) if len(V_total[mask]) > 0 else V_total[-1]
else:
    V_barrier = V_total[-1]

print(f"Minimum of total potential: {V_min:.2f} MeV at r = {r_min:.2f} fm")
print(f"Barrier height (if any): {V_barrier:.2f} MeV")

if V_min < 0:
    print("Total potential is attractive -> bound state (nuclear force overcomes Coulomb).")
else:
    print("Total potential is repulsive -> no bound state (Coulomb barrier dominates).")

# Check if strong mass is roughly constant for r < r0
r_in = r[r < r0]
m_in = m_strong[r < r0]
if len(m_in) > 1:
    rel_var = np.std(m_in) / np.mean(m_in)
    print(f"Relative variation of m_strong for r<r0: {rel_var:.3f} (small means constant)")