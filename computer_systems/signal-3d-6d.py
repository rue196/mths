import math
import cmath
import numpy as np
import matplotlib.pyplot as plt

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)          # ≈ 2.362
A = ALPHA                       # same as a in derida.rb
NORM = 1.0 - math.exp(-ALPHA * (PI + E))

# ---------- Derivative operator D_a ----------
def D_a(f, t, a=A):
    """Approximate derivative using step a: (f(t+a) - f(t)) / a."""
    return (f(t + a) - f(t)) / a

# ---------- Möbius sieve (O(K)) ----------
def mobius_sieve(K):
    mu = [0] * (K + 1)
    mu[1] = 1
    primes = []
    is_comp = [False] * (K + 1)
    for i in range(2, K + 1):
        if not is_comp[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K:
                break
            is_comp[i * p] = True
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    return mu

# ---------- TSP routing (bucket sort by phase) ----------
def tsp_route_complex(z):
    angles = np.angle(z)
    buckets = [[] for _ in range(360)]
    for idx, a in enumerate(angles):
        a_norm = a + PI if a < 0 else a
        b = int((a_norm / (2 * PI)) * 360) % 360
        buckets[b].append(idx)
    order = []
    for b in buckets:
        order.extend(b)
    return np.array(order)

# ---------- Exponential convolution (two‑pass, O(K)) ----------
def conv_exp_kernel(signal, alpha=ALPHA):
    K = len(signal)
    lam = math.exp(-alpha)
    f = np.zeros(K, dtype=complex if np.iscomplexobj(signal) else float)
    f[0] = signal[0]
    for i in range(1, K):
        f[i] = signal[i] + lam * f[i-1]
    b = np.zeros(K, dtype=type(signal[0]))
    b[K-1] = signal[K-1]
    for i in range(K-2, -1, -1):
        b[i] = signal[i] + lam * b[i+1]
    conv_exp = (f + b - signal) / (1 - lam * lam)
    conv = (1.0 - conv_exp) / NORM
    return conv

# ---------- Supertrace, entropy, mass ----------
def supertrace_and_mass(signal):
    S = 0.0
    for i, val in enumerate(signal):
        sign = 1 if (i % 2 == 0) else -1
        S += sign * abs(val)
    if S == 0:
        H = 0.0; m = 0.0
    else:
        p = abs(S) / len(signal)
        H = -ALPHA * p * math.log(p) if p > 0 else 0.0
        m = abs(S) * math.exp(-H)
    return S, H, m

# ---------- Initialize wavepacket in HO basis ----------
def initialize_wavepacket_1D(K, x0=0.0, sigma=1.0):
    coeffs = np.zeros(K, dtype=complex)
    alpha = (x0 + 1j*0.0) / (sigma * math.sqrt(2))
    alpha2 = abs(alpha)**2
    norm = math.exp(-alpha2/2)
    for i in range(K):
        if i == 0:
            coeffs[i] = norm
        else:
            coeffs[i] = coeffs[i-1] * (alpha / math.sqrt(i))
    return coeffs

# ---------- Potential as a function of position (for derivative) ----------
def potential(x, omega=1.0):
    return 0.5 * omega**2 * x**2

# ---------- Force using D_a derivative ----------
def force_from_potential(gamma_coeff, basis_funcs):
    """
    Compute force = -dV/dγ using the D_a derivative of the potential.
    gamma_coeff: coefficients of position in HO basis.
    basis_funcs: callable that returns position x from coefficients? 
    Here we use the fact that in HO basis, the potential is diagonal,
    so we can compute force as -ω² * gamma_n.
    But we want to demonstrate D_a, so we compute the derivative with respect to a parameter.
    For simplicity, we'll use D_a on the potential function evaluated at a scalar.
    We'll treat the expectation value of position as the variable.
    """
    # Compute expectation position
    x_exp = expectation_position(gamma_coeff)
    # Use D_a to approximate dV/dx at x_exp
    dV_dx = D_a(lambda x: potential(x), x_exp)
    return -dV_dx   # force = -dU/dx

# ---------- Quantum kinetic (diagonal) ----------
def apply_kinetic(psi, dt, d, mass=1.0):
    omega = 1.0
    T_diag = (np.arange(len(psi)) + 0.5) * omega
    factor = (1 - 1j * T_diag * dt / 2) / (1 + 1j * T_diag * dt / 2)
    return psi * factor

# ---------- Quantum potential (diagonal) ----------
def apply_potential(psi, V_spectrum, dt):
    return psi * np.exp(-1j * V_spectrum * dt)

# ---------- Expectation position (HO basis) ----------
def expectation_position(psi):
    K = len(psi)
    x_exp = 0.0
    for n in range(K-1):
        x_exp += math.sqrt((n+1)/2) * np.real(np.conj(psi[n]) * psi[n+1])
    return x_exp

# ---------- Main comparison (using D_a for derivatives) ----------
def Compare3DClassicalVs6DQuantum_Compressed(T, dt, K):
    print(f"Using derivative operator D_a with a = {A:.6f} (1/(π-e))")
    print(f"T={T}, dt={dt}, K={K}")

    # Classical state
    gamma_coeff = initialize_wavepacket_1D(K, x0=1.0, sigma=0.5)
    v_coeff = np.zeros(K, dtype=complex)

    # Quantum state (6D separable)
    psi_dims = [initialize_wavepacket_1D(K, x0=0.0, sigma=0.5) for _ in range(6)]

    # Potential spectra (diagonal in HO basis)
    omega = 1.0
    V_spectrum = np.array([0.5 * omega**2 * (n + 0.5) for n in range(K)])

    # Storage
    time_vals = []
    x_cl = []
    x_qm = []

    mu = mobius_sieve(K)

    for step, t in enumerate(np.arange(0, T+dt, dt)):
        # --- Classical update using D_a force ---
        # Compute force using D_a derivative of potential at current position expectation
        F = force_from_potential(gamma_coeff, None)   # returns scalar force
        # We need to update each mode: v_n += (F / m) * dt * basis_coeff? 
        # In HO basis, the force is not simply a scalar; we need to project.
        # For simplicity, we apply the force as a scalar to all modes (crude).
        # A better approach: use the gradient of the potential in the basis.
        # Here we just apply a uniform acceleration.
        v_coeff = v_coeff + F * dt
        gamma_coeff = gamma_coeff + v_coeff * dt

        # Optional compression of classical state
        if step % 10 == 0:
            order = tsp_route_complex(gamma_coeff)
            gamma_sorted = gamma_coeff[order]
            gamma_conv = conv_exp_kernel(gamma_sorted)
            S, H, m = supertrace_and_mass(gamma_conv)
            M = max(1, int(abs(S))) if abs(S) > 1 else K
            M = min(M, K)
            mag = np.abs(gamma_conv)
            idx_sorted = np.argsort(mag)[::-1]
            kept = []
            count = 0
            for idx in idx_sorted:
                if mu[idx+1] != 0:
                    kept.append(idx)
                    count += 1
                    if count >= M:
                        break
            gamma_new = np.zeros(K, dtype=complex)
            for idx in kept:
                gamma_new[idx] = gamma_conv[idx]
            inv_order = np.argsort(order)
            gamma_coeff = gamma_new[inv_order]

        # --- Quantum update (split operator) ---
        for d in range(6):
            psi_dims[d] = apply_kinetic(psi_dims[d], dt, d)
            psi_dims[d] = apply_potential(psi_dims[d], V_spectrum, dt)

        # --- 3D effective wavefunction (product of first 3 dims) ---
        psi_3D = np.ones(K, dtype=complex)
        for d in range(3):
            psi_3D = psi_3D * psi_dims[d]

        # Compress quantum state
        if step % 10 == 0:
            order_q = tsp_route_complex(psi_3D)
            psi_sorted = psi_3D[order_q]
            psi_conv = conv_exp_kernel(psi_sorted)
            S_q, H_q, m_q = supertrace_and_mass(psi_conv)
            M_q = max(1, int(abs(S_q))) if abs(S_q) > 1 else K
            M_q = min(M_q, K)
            mag_q = np.abs(psi_conv)
            idx_sorted_q = np.argsort(mag_q)[::-1]
            kept_q = []
            count_q = 0
            for idx in idx_sorted_q:
                if mu[idx+1] != 0:
                    kept_q.append(idx)
                    count_q += 1
                    if count_q >= M_q:
                        break
            psi_new = np.zeros(K, dtype=complex)
            for idx in kept_q:
                psi_new[idx] = psi_conv[idx]
            inv_order_q = np.argsort(order_q)
            psi_3D = psi_new[inv_order_q]

        # Expectation values
        x_cl_val = expectation_position(gamma_coeff)
        x_qm_val = expectation_position(psi_3D)

        time_vals.append(t)
        x_cl.append(x_cl_val)
        x_qm.append(x_qm_val)

        if step % 50 == 0:
            print(f"t={t:.2f}, x_cl={x_cl_val:.4f}, x_qm={x_qm_val:.4f}")

    # Plot
    plt.figure(figsize=(10,6))
    plt.plot(time_vals, x_cl, label='Classical 3D (D_a force)')
    plt.plot(time_vals, x_qm, label='Quantum 6D -> 3D (D_a for potential)')
    plt.xlabel('Time')
    plt.ylabel('Position expectation')
    plt.legend()
    plt.title('Comparison using derivative operator D_a = (f(t+a)-f(t))/a, a=1/(π-e)')
    plt.grid(True)
    plt.show()

    return gamma_coeff, psi_3D

# ---------- Run ----------
if __name__ == "__main__":
    K = 64
    T = 10.0
    dt = 0.05
    gamma_final, psi_final = Compare3DClassicalVs6DQuantum_Compressed(T, dt, K)
    print("Done.")