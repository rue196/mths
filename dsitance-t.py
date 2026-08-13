import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)          # ≈ 2.362
A = ALPHA
NORM = 1.0 - math.exp(-ALPHA * (PI + E))
alpha = 1.0 / math.pi - math.e

# ---------- (Reuse functions from signal-3d-6d.py) ----------
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

def conv_exp_kernel(signal, alpha=(1/PI-E)):
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

def expectation_position(psi):
    K = len(psi)
    x_exp = 0.0
    for n in range(K-1):
        x_exp += math.sqrt((n+1)/2) * np.real(np.conj(psi[n]) * psi[n+1])
    return x_exp

# ---------- Time‑of‑flight functions ----------
def distance_from_mass_ratio(m0, mr, beta=0.1):
    """
    Compute distance (in arbitrary units) from original and received masses.
    m0: original supertrace mass.
    mr: received supertrace mass.
    beta: attenuation coefficient (1/distance).
    """
    if m0 <= 0 or mr <= 0:
        return 0.0
    ratio = mr / m0
    if ratio <= 0:
        return 0.0
    return -math.log(ratio) / beta

def time_from_distance(d, v=1.0):
    """Time = distance / speed."""
    return d / v

def phase_delay(original_signal, received_signal):
    """
    Estimate time delay from the phase difference of the spectral sums.
    We compute ζ(t) for the two signals and find the phase shift.
    Returns: time delay (delta_t) in units of t (harmonic time).
    """
    # We need to evaluate zeta at some reference harmonic point.
    # For simplicity, we take the first harmonic index n=1 (t = H_1 = 1.0).
    # In practice, one would use the full spectral sum.
    # Here we use the average phase of the coefficients.
    orig_phase = np.angle(original_signal)
    recv_phase = np.angle(received_signal)
    # Average phase difference (mod 2pi)
    diff = recv_phase - orig_phase
    # Unwrap (mod 2pi) - use numpy's unwrap for 1D
    diff_unwrapped = np.unwrap(diff)
    # Average phase shift
    mean_diff = np.mean(diff_unwrapped)
    # The phase shift relates to time delay: φ = ω * t, but here the spectral sum
    # has frequencies i/alpha. We can approximate using the dominant mode.
    # For simplicity, we take the time shift = -mean_diff * alpha (since phase = t/alpha)
    # Actually, ζ(t) = Σ C_i exp(i * t * i / alpha). A time delay δt changes the phase by δt * i / alpha.
    # So δt = mean_diff * alpha / i, but we don't know i. We'll use the average i.
    K = len(original_signal) // 2
    avg_i = K / 2.0
    delta_t = mean_diff * alpha / avg_i
    return delta_t

# ---------- Simulation of transmission ----------
def simulate_transmission(K=64, distance=5.0, beta=0.1, v=1.0, seed=42):
    """
    Simulate a signal being sent over a distance.
    Returns: (original_signal, received_signal, true_time, estimated_time_mass, estimated_time_phase)
    """
    np.random.seed(seed)
    # 1. Create an original signal (a random wavepacket in HO basis)
    orig_signal = initialize_wavepacket_1D(K, x0=0.5, sigma=0.8)
    # Add some randomness to make it interesting
    orig_signal += 0.1 * (np.random.randn(K) + 1j*np.random.randn(K))

    # 2. Compute original mass
    S0, H0, m0 = supertrace_and_mass(orig_signal)
    print(f"Original mass m0 = {m0:.4f}")

    # 3. Simulate attenuation: m_received = m0 * exp(-beta * distance)
    mr = m0 * math.exp(-beta * distance)
    # Scale the signal magnitude to achieve the desired received mass
    # We'll scale the entire signal by a factor sqrt(mr/m0) because mass scales with |signal|^2?
    # Actually mass is computed from |signal| (absolute values). So scaling by factor c gives mass c * m0.
    # So we need c = mr / m0.
    scale = mr / m0
    received_signal = orig_signal * scale

    # 4. Compute received mass (should match mr)
    S_r, H_r, m_r = supertrace_and_mass(received_signal)
    print(f"Received mass m_r = {m_r:.4f} (target {mr:.4f})")

    # 5. Estimate distance and time from mass
    est_dist = distance_from_mass_ratio(m0, m_r, beta)
    est_time_mass = time_from_distance(est_dist, v)

    # 6. Estimate time from phase delay
    delta_t_phase = phase_delay(orig_signal, received_signal)
    # The phase delay gives a time shift. The true time delay is distance / v.
    true_time = distance / v
    print(f"True distance = {distance}, true time = {true_time:.4f}")
    print(f"Estimated distance (mass) = {est_dist:.4f}, time = {est_time_mass:.4f}")
    print(f"Estimated time (phase) = {delta_t_phase:.4f}")

    return orig_signal, received_signal, true_time, est_time_mass, delta_t_phase

# ---------- Main demonstration ----------
def main():
    # Parameters
    K = 64
    distances = np.linspace(0.5, 10.0, 20)
    beta = 0.2
    v = 1.0

    true_times = []
    est_times_mass = []
    est_times_phase = []

    for d in distances:
        orig, recv, true_t, est_mass_t, est_phase_t = simulate_transmission(K, distance=d, beta=beta, v=v, seed=42)
        true_times.append(true_t)
        est_times_mass.append(est_mass_t)
        est_times_phase.append(est_phase_t)

    # Plot results
    plt.figure(figsize=(10,6))
    plt.plot(distances, true_times, 'k-', label='True time (distance/v)')
    plt.plot(distances, est_times_mass, 'ro', label='Estimated time from mass')
    plt.plot(distances, est_times_phase, 'bs', label='Estimated time from phase')
    plt.xlabel('Distance')
    plt.ylabel('Time')
    plt.title('Time estimation from supertrace mass and phase delay')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Also show the relationship between mass and distance
    # Compute mass as a function of distance
    m0 = None
    masses = []
    for d in distances:
        orig, recv, _, _, _ = simulate_transmission(K, distance=d, beta=beta, v=v, seed=42)
        if m0 is None:
            S0, H0, m0 = supertrace_and_mass(orig)
        S_r, H_r, m_r = supertrace_and_mass(recv)
        masses.append(m_r)
    plt.figure()
    plt.plot(distances, masses, 'g-', label='Received mass')
    plt.xlabel('Distance')
    plt.ylabel('Mass')
    plt.title('Attenuation of supertrace mass with distance')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()