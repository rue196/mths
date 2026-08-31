import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)          # ≈ 2.362

# ---------- Möbius and harmonic functions ----------
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

def harmonic_numbers(N):
    H = np.zeros(N + 1, dtype=float)
    H[1] = 1.0
    for n in range(2, N + 1):
        H[n] = H[n-1] + 1.0 / n
    return H

def build_coeffs(K):
    mu = mobius_sieve(K)
    c = np.zeros(2*K + 1, dtype=complex)
    # C_i = μ(|i|) for i != 0, C_0 = 0
    for n in range(1, K + 1):
        c[K + n] = mu[n]          # i = n
        c[K - n] = mu[n]          # i = -n
    c[K] = 0.0
    return c

def zeta_at(t, c, alpha):
    K = (len(c) - 1) // 2
    total = 0.0 + 0.0j
    for i in range(-K, K + 1):
        total += c[i + K] * np.exp(1j * t * i / alpha)
    return total

# ---------- SuperTrace and entropy ----------
def supertrace_from_state(state):
    S = 0.0
    for idx, val in enumerate(state):
        sign = 1 if (idx % 2 == 0) else -1
        S += sign * abs(val)
    return S

def entropy_from_supertrace(S, N, alpha=ALPHA):
    if S == 0:
        return 0.0
    p = abs(S) / N
    if p <= 0:
        return 0.0
    return -alpha * p * math.log(p)

def mass_from_state(state):
    S = supertrace_from_state(state)
    H = entropy_from_supertrace(S, len(state))
    return abs(S) * math.exp(-H)

# ---------- Buffer allocation helper ----------
def create_buffer(shape, dtype=np.float32):
    return np.zeros(shape, dtype=dtype)

# ---------- Pendulum data generator with buffers ----------
def generate_pendulum_data(N_harmonics=100, K_max=60, seed=42):
    np.random.seed(seed)
    c = build_coeffs(K_max)
    H = harmonic_numbers(N_harmonics)          # length N_harmonics+1

    # Pre‑allocate buffers
    time_vals = np.zeros(N_harmonics, dtype=float)   # H[1]..H[N]
    zeta_vals = np.zeros(N_harmonics, dtype=float)
    state_history = np.zeros((N_harmonics, 4), dtype=float)

    # Fill zeta buffer (real part)
    for n in range(1, N_harmonics + 1):
        z = zeta_at(H[n], c, ALPHA)
        zeta_vals[n-1] = z.real

    # time_vals = H[1:] (length N_harmonics)
    time_vals[:] = H[1:]

    state0 = np.array([1.0, 0.8, 0.2, 0.0])
    state_history[0] = state0

    dt = 0.01
    for n in range(1, N_harmonics):
        t_prev = time_vals[n-1]
        t_cur = time_vals[n]
        n_steps = max(1, int((t_cur - t_prev) / dt))
        dt_sub = (t_cur - t_prev) / n_steps
        state = state_history[n-1]
        zeta_avg = (zeta_vals[n-1] + zeta_vals[n]) / 2.0

        def deriv(state, zeta):
            x_inv, y_inv, x_i, y_i = state
            m = (x_inv + y_inv) / 2.0
            theta = (x_i - y_i) / 2.0
            omega = (x_i - y_i) / 2.0
            torque = zeta * 0.1
            g, L, gamma = 9.81, 1.0, 0.1
            alpha_acc = torque / (m * L**2) - (g / L) * math.sin(theta) - gamma * omega
            tau_mass = 10.0
            dx_inv_dt = (1.0 - x_inv) / tau_mass
            dy_inv_dt = (1.0 - y_inv) / tau_mass
            dx_i_dt = omega + alpha_acc
            dy_i_dt = omega - alpha_acc
            return np.array([dx_inv_dt, dy_inv_dt, dx_i_dt, dy_i_dt])

        for _ in range(n_steps):
            state = state + deriv(state, zeta_avg) * dt_sub
            state = np.clip(state, 0.1, 5.0)
        state_history[n] = state

    return time_vals, zeta_vals, state_history

# ---------- LearnedPendulumCell ----------
class LearnedPendulumCell(nn.Module):
    def __init__(self, hidden_dim=4):
        super().__init__()
        self.g = nn.Parameter(torch.tensor(9.81, dtype=torch.float32))
        self.L = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.tau_mass = nn.Parameter(torch.tensor(10.0, dtype=torch.float32))
        self.coupling = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, state, zeta, dt):
        x_inv = state[:, 0:1]
        y_inv = state[:, 1:2]
        x_i   = state[:, 2:3]
        y_i   = state[:, 3:4]

        m = (x_inv + y_inv) / 2.0
        theta = (x_i - y_i) / 2.0
        omega = (x_i - y_i) / 2.0

        torque = zeta * self.coupling
        m = torch.clamp(m, min=0.1)

        alpha_acc = torque / (m * self.L**2) - (self.g / self.L) * torch.sin(theta) - self.gamma * omega

        dx_inv_dt = (1.0 - x_inv) / self.tau_mass
        dy_inv_dt = (1.0 - y_inv) / self.tau_mass
        dx_i_dt = omega + alpha_acc
        dy_i_dt = omega - alpha_acc

        dstate = torch.cat([dx_inv_dt, dy_inv_dt, dx_i_dt, dy_i_dt], dim=1)
        next_state = state + dstate * dt
        next_state = torch.clamp(next_state, 0.1, 5.0)
        return next_state

# ---------- Training with pre‑allocated buffers ----------
def train_pendulum_cell(num_epochs=200, batch_size=64, seq_len=20):
    time_vals, zeta_vals, state_hist = generate_pendulum_data(N_harmonics=150, K_max=40)
    N = len(time_vals)
    num_samples = N - seq_len

    X_zeta = create_buffer((num_samples, seq_len), dtype=np.float32)
    X_state = create_buffer((num_samples, seq_len, 4), dtype=np.float32)
    y_state = create_buffer((num_samples, seq_len, 4), dtype=np.float32)

    for i in range(num_samples):
        X_zeta[i] = zeta_vals[i:i+seq_len]
        X_state[i] = state_hist[i:i+seq_len]
        y_state[i] = state_hist[i+1:i+seq_len+1]

    dataset = TensorDataset(torch.tensor(X_zeta), torch.tensor(X_state), torch.tensor(y_state))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    cell = LearnedPendulumCell()
    optimizer = optim.Adam(cell.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    loss_history = create_buffer((num_epochs,), dtype=float)
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_zeta, batch_state, batch_y in loader:
            batch_size_cur = batch_zeta.size(0)
            state = batch_state[:, 0, :]
            pred_states = []
            for t in range(seq_len):
                zeta = batch_zeta[:, t].unsqueeze(1)
                dt = 0.05
                state = cell(state, zeta, dt)
                pred_states.append(state)
            pred_states = torch.stack(pred_states, dim=1)
            loss = criterion(pred_states, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_size_cur
        avg_loss = total_loss / len(dataset)
        loss_history[epoch] = avg_loss
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")

    plt.figure()
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training the Pendulum Cell')
    plt.grid()
    plt.show()
    return cell, loss_history

# ---------- Long thought simulation with buffer ----------
def long_thought_simulation(cell, zeta_input, initial_state, dt=0.05, steps=500):
    states = create_buffer((steps, 4), dtype=np.float32)
    state = torch.tensor(initial_state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        for t in range(steps):
            zeta = torch.tensor(zeta_input[t], dtype=torch.float32).view(1, 1)
            state = cell(state, zeta, dt)
            states[t] = state.squeeze(0).numpy()
    return states

# ---------- Box‑counting fractal dimension ----------
def box_counting_dimension(points, grid_sizes=None):
    if grid_sizes is None:
        grid_sizes = np.logspace(np.log10(1e-2), np.log10(1.0), num=20)
    counts = []
    for s in grid_sizes:
        scaled = points / s
        cells = np.floor(scaled).astype(int)
        unique_cells = np.unique(cells, axis=0)
        counts.append(len(unique_cells))
    log_inv_s = np.log(1.0 / grid_sizes)
    log_counts = np.log(counts)
    coeffs = np.polyfit(log_inv_s, log_counts, 1)
    return coeffs[0]

# ---------- Approximate Hilbert curve ----------
def hilbert_curve(order=4, scale=1.0):
    t = np.linspace(0, 2*np.pi, 1000)
    x = np.sin(t) * np.sin(0.5*t)
    y = np.cos(t) * np.cos(0.7*t)
    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())
    return np.column_stack([x, y])

# ---------- Main ----------
def main():
    print("Training the pendulum neural simulator with buffers...")
    cell, loss_hist = train_pendulum_cell(num_epochs=150, batch_size=32, seq_len=20)

    # ---- Buffer demo (like buffy.c) ----
    k = 10
    buf = create_buffer(2*k + 1, dtype=np.float64)
    buf[:] = 1.0   # fill with ones (|C_i| array)
    total = np.sum(buf)
    print(f"Buffer sum (|C_i| array length {2*k+1}) = {total}")

    # ---- Long thought simulation ----
    steps = 500
    t = np.linspace(0, 20, steps)
    zeta_input = 0.5 * np.sin(0.5 * t) + 0.3 * np.cos(1.2 * t) + 0.1 * np.sin(2.3 * t + 1.0)
    initial_state = np.array([0.9, 0.7, 0.3, 0.1])
    state_hist = long_thought_simulation(cell, zeta_input, initial_state, dt=0.04, steps=steps)

    theta = (state_hist[:, 2] - state_hist[:, 3]) / 2.0
    mass = (state_hist[:, 0] + state_hist[:, 1]) / 2.0
    S_hist = np.array([supertrace_from_state(s) for s in state_hist])
    mass_from_S = np.array([mass_from_state(s) for s in state_hist])

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    axes[0].plot(zeta_input, label='Input zeta (thought drive)')
    axes[0].set_ylabel('Zeta')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(theta, label='Angle (thought state)')
    axes[1].set_ylabel('Angle')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(mass, label='Mass intensity')
    axes[2].plot(mass_from_S, '--', label='Mass from supertrace')
    axes[2].set_xlabel('Time step')
    axes[2].set_ylabel('Mass')
    axes[2].legend()
    axes[2].grid(True)
    plt.suptitle('Long Thought Simulation – Neural Pendulum Dynamics (with buffers)')
    plt.tight_layout()
    plt.show()

    # ---- Memory test: step input ----
    print("\n=== Memory Test: Step Input Integration ===")
    zeta_step = np.zeros(300)
    zeta_step[50:150] = 1.0
    zeta_step[200:250] = -1.0
    state_mem = long_thought_simulation(cell, zeta_step, initial_state, dt=0.05, steps=300)

    theta_mem = (state_mem[:, 2] - state_mem[:, 3]) / 2.0
    omega_mem = (state_mem[:, 2] + state_mem[:, 3]) / 2.0
    mass_mem = (state_mem[:, 0] + state_mem[:, 1]) / 2.0

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    axes[0].plot(zeta_step, label='Input zeta (step)')
    axes[0].set_ylabel('Zeta')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_title('Step input')

    axes[1].plot(theta_mem, label='Angle', color='red')
    axes[1].plot(mass_mem, label='Mass', color='blue')
    axes[1].set_xlabel('Time step')
    axes[1].set_ylabel('State variables')
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_title('Response: angle and mass')
    plt.tight_layout()
    plt.show()

    # ---- Fractal analysis ----
    pts = np.column_stack([state_hist[:, 2], state_hist[:, 0]])
    pts_norm = (pts - pts.min(axis=0)) / (pts.max(axis=0) - pts.min(axis=0) + 1e-12)
    dim = box_counting_dimension(pts_norm)
    print(f"Estimated fractal dimension (theta vs mass): {dim:.3f}")

    hilbert_pts = hilbert_curve()
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes2[0, 0]
    ax.plot(theta_mem, mass_mem, color='purple', alpha=0.7, label='Trajectory')
    hx = hilbert_pts[:, 0] * (theta_mem.max() - theta_mem.min()) + theta_mem.min()
    hy = hilbert_pts[:, 1] * (mass_mem.max() - mass_mem.min()) + mass_mem.min()
    ax.plot(hx, hy, 'k--', alpha=0.3, linewidth=0.5, label='Hilbert ref')
    ax.set_xlabel('Angle')
    ax.set_ylabel('Mass')
    ax.set_title(f'θ vs mass (dim ≈ {dim:.2f})')
    ax.legend()
    ax.grid(True)

    ax = axes2[0, 1]
    ax.plot(theta_mem, omega_mem, color='green', alpha=0.7)
    ax.set_xlabel('Angle')
    ax.set_ylabel('Angular velocity')
    ax.set_title('θ vs ω')
    ax.grid(True)

    ax = axes2[1, 0]
    ax.plot(mass_mem, omega_mem, color='orange', alpha=0.7)
    ax.set_xlabel('Mass')
    ax.set_ylabel('Angular velocity')
    ax.set_title('Mass vs ω')
    ax.grid(True)

    ax = axes2[1, 1]
    ax = fig2.add_subplot(2, 2, 4, projection='3d')
    sub = slice(0, len(theta_mem), 10)
    ax.scatter(theta_mem[sub], omega_mem[sub], mass_mem[sub], c=zeta_step[sub], cmap='viridis', s=5)
    ax.set_xlabel('θ')
    ax.set_ylabel('ω')
    ax.set_zlabel('Mass')
    ax.set_title('3D state space (color = input zeta)')
    plt.tight_layout()
    plt.show()

    print("Long thought simulation complete with buffering and |C_i| array.")

if __name__ == "__main__":
    main()