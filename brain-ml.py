#!/usr/bin/env python3
"""
creative_memory_system.py

Full pipeline for one‑shot training, exploration, and creative output.

Stages:
  1. Observe: Convert input to a Möbius signature (finite‑step derivative first).
  2. Recall: Retrieve similar stored memories (inverse score via merge sort).
  3. Explore: Perturb recalled memory using the pendulum long‑chain dynamics.
  4. Create: Blend explored states, weighted by supertrace entropy and deviation.
  5. Filter: Reject tail‑end predictions (bad) using inverse score thresholds.
  6. Output: Produce final creative result.

All operations run in O(K log K) due to merge sort and sorting in chip pipeline.
"""

import math
import numpy as np
import random
from collections import defaultdict
import hashlib
import struct
import cmath
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)
A = ALPHA / 0.3628          # finite step derivative
NORM = 1.0 - math.exp(-ALPHA * (PI + E))

# ---------- Finite derivative function ----------
def finite_derivative(signal, step=A):
    """Compute forward finite difference with step size `step`."""
    if len(signal) < 2:
        return np.zeros_like(signal, dtype=float)
    diff = np.zeros_like(signal, dtype=float)
    diff[:-1] = (signal[1:] - signal[:-1]) / step
    return diff
# ---------- Supertrace entropy ----------
def supertrace_entropy(features, harmonic_indices):
    S = 0.0
    for idx, z in enumerate(features):
        val = z.real
        n = idx + 1
        if n % 2 == 0:
            S += val
        else:
            S -= val
    N = len(features)
    ratio = abs(S) / N if N > 0 else 0.0
    if 0.0 < ratio < 1.0:
        return -ALPHA * ratio * math.log(ratio)
    return 0.0

# ---------- Merge-sort inversion count ----------
def merge_and_count(arr, temp, left, mid, right):
    i, j, k = left, mid+1, left
    inv = 0
    while i <= mid and j <= right:
        if arr[i] <= arr[j]:
            temp[k] = arr[i]; i += 1
        else:
            temp[k] = arr[j]
            inv += (mid - i + 1)
            j += 1
        k += 1
    while i <= mid:
        temp[k] = arr[i]; i += 1; k += 1
    while j <= right:
        temp[k] = arr[j]; j += 1; k += 1
    for i in range(left, right+1):
        arr[i] = temp[i]
    return inv

def _merge_sort(arr, temp, left, right):
    inv = 0
    if left < right:
        mid = (left + right) // 2
        inv += _merge_sort(arr, temp, left, mid)
        inv += _merge_sort(arr, temp, mid+1, right)
        inv += merge_and_count(arr, temp, left, mid, right)
    return inv

def inversion_count(arr):
    n = len(arr)
    temp = [0]*n
    return _merge_sort(arr, temp, 0, n-1)

def inverse_score(a, b):
    pairs = sorted(zip(a, b), key=lambda x: x[0])
    b_sorted = [p[1] for p in pairs]
    inv = inversion_count(b_sorted)
    K = len(a)
    max_inv = K*(K-1)//2
    return inv / max_inv if max_inv > 0 else 0.0

# ---------- Neural network (FIXED forward) ----------
class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x, return_hidden=False):
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        out = self.fc3(h2)
        if return_hidden:
            return out, h2   # return tensor, no detach
        return out

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

class ChipProcessor:
    """
    A processor that performs the chip pipeline with pre‑allocated buffers.
    This reduces garbage collection and allocation overhead.
    """

    def __init__(self, max_K=1000):
        """
        Allocate buffers for up to max_K elements.
        """
        self.max_K = max_K
        # Buffers for convolution (two passes)
        self.f = np.zeros(max_K, dtype=float)
        self.b = np.zeros(max_K, dtype=float)
        # Buffer for convolution result
        self.conv = np.zeros(max_K, dtype=float)
        # Buffer for sorted indices (TSP order)
        self.order = np.zeros(max_K, dtype=int)
        # Buffer for magnitudes (for sorting)
        self.mag = np.zeros(max_K, dtype=float)
        # Buffer for indices (for argsort)
        self.idx = np.arange(max_K, dtype=int)   # reusable index array
        # Cache for Möbius sieve (computed once)
        self.mu = None
        self._update_mu(max_K)

    def _update_mu(self, K):
        """Compute Möbius sieve up to K (if not already cached)."""
        if self.mu is None or len(self.mu) < K + 1:
            self.mu = self._mobius_sieve(K)

    @staticmethod
    def _mobius_sieve(K):
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

    def _tsp_route(self, signal, K):
        """Compute TSP routing order using bucket sort; store in self.order."""
        # Use deterministic pseudo‑angles
        angles = np.zeros(K, dtype=float)
        for i in range(K):
            x = math.sin(i * 7.0) + 0.1 * math.cos(i * 13.0)
            y = math.cos(i * 11.0) + 0.1 * math.sin(i * 17.0)
            angles[i] = math.atan2(y, x) + math.pi
        # Bucket sort: 360 buckets
        buckets = [[] for _ in range(360)]
        for i in range(K):
            a = angles[i]
            idx = int((a / (2 * math.pi)) * 360) % 360
            buckets[idx].append(i)
        order = []
        for b in buckets:
            order.extend(b)
        self.order[:K] = order

    def _conv_exp_kernel(self, signal, K, alpha=ALPHA):
        """Two‑pass exponential convolution; store result in self.conv."""
        lam = math.exp(-alpha)
        # Forward pass
        f = self.f
        f[0] = signal[0]
        for i in range(1, K):
            f[i] = signal[i] + lam * f[i-1]
        # Backward pass
        b = self.b
        b[K-1] = signal[K-1]
        for i in range(K-2, -1, -1):
            b[i] = signal[i] + lam * b[i+1]
        # Combine
        conv = self.conv
        inv_den = 1.0 / (1.0 - lam * lam)
        inv_norm = 1.0 / NORM
        for i in range(K):
            conv_exp = (f[i] + b[i] - signal[i]) * inv_den
            conv[i] = (1.0 - conv_exp) * inv_norm

    def _supertrace_and_mass(self, signal, K):
        """Compute S, H, m from the signal (assumed in self.conv)."""
        S = 0.0
        for i in range(K):
            val = signal[i]
            if i % 2 == 0:
                S += abs(val)
            else:
                S -= abs(val)
        if S == 0:
            H = 0.0
            m = 0.0
        else:
            p = abs(S) / K
            if p <= 0:
                H = 0.0
            else:
                H = -ALPHA * p * math.log(p)
            m = abs(S) * math.exp(-H)
        return S, H, m

    def process(self, signal):
        """
        Run the chip pipeline on `signal` (1D numpy array).
        Returns: (kept, S, H, m, conv)
        """
        K = len(signal)
        if K > self.max_K:
            raise ValueError(f"Signal length {K} exceeds max_K {self.max_K}; reinitialize processor with larger max_K.")

        # 1. TSP routing
        self._tsp_route(signal, K)
        order = self.order[:K]
        # Reorder signal in a temporary view? We'll create a sorted copy.
        signal_sorted = signal[order]   # This creates a new array; but we can reuse a buffer if we want.
        # Since we need the sorted signal, we'll allocate a buffer for it.
        # To avoid extra allocation, we could use a pre‑allocated buffer `self.sorted_signal`.
        if not hasattr(self, 'sorted_signal') or len(self.sorted_signal) < K:
            self.sorted_signal = np.zeros(K, dtype=signal.dtype)
        self.sorted_signal[:K] = signal[order]

        # 2. Convolution
        self._conv_exp_kernel(self.sorted_signal, K)

        # 3. Supertrace
        S, H, m = self._supertrace_and_mass(self.conv, K)
        M = max(1, int(abs(S)))
        if M > K:
            M = K

        # 4. Möbius sieve (ensure it's up to date)
        self._update_mu(K)
        mu = self.mu

        # 5. Compression: keep top M with square‑free index (μ(n) != 0)
        mag = self.mag[:K]
        for i in range(K):
            mag[i] = abs(self.conv[i])

        # Get indices sorted by magnitude descending using argsort
        # We'll use a pre‑allocated index buffer and sort.
        idx = self.idx[:K]   # already 0..K-1
        # We need to sort idx by mag descending. We'll use np.argsort on a copy of mag.
        sorted_idx = np.argsort(mag)[::-1]   # This allocates a new array; but we can reuse a buffer.
        # Since argsort always returns a new array, we can't avoid allocation easily.
        # We'll just use it; it's O(K log K) but memory allocation is small.

        kept = []
        count = 0
        for idx in sorted_idx:
            n = idx + 1
            if mu[n] != 0:
                kept.append((idx, self.conv[idx]))
                count += 1
                if count >= M:
                    break

        # Return results
        # We also return a copy of conv for external use (if needed)
        conv_copy = self.conv[:K].copy()
        return kept, S, H, m, conv_copy

# ---------- Backward‑compatible function ----------
def chip_pipeline(signal, processor=None):
    """
    Run the chip pipeline, optionally reusing a ChipProcessor.
    If processor is None, a temporary one is created.
    """
    if processor is None:
        processor = ChipProcessor(max_K=len(signal))
    return processor.process(signal)

class ChipLogicGate:
    """
    A bounded Möbius logic gate that applies a function to a signal,
    then runs the chip pipeline (convolution + supertrace + Möbius compression).
    The output is a compressed representation of the transformed signal.
    """

    def __init__(self, max_K=1000):
        self.processor = ChipProcessor(max_K=max_K)

    def apply_function(self, signal, func, *args, **kwargs):
        """
        Apply a function `func` to each element of `signal`.
        `func` can be a callable (e.g., math.log, math.exp, np.sin)
        or a string ('log', 'exp', 'sin', 'cos', 'trace').
        """
        if isinstance(func, str):
            func_name = func.lower()
            if func_name == 'log':
                # avoid log(0)
                safe_signal = np.maximum(signal, 1e-12)
                transformed = np.log(safe_signal)
            elif func_name == 'exp':
                transformed = np.exp(signal)
            elif func_name == 'sin':
                transformed = np.sin(signal)
            elif func_name == 'cos':
                transformed = np.cos(signal)
            elif func_name == 'trace':
                # Matrix trace function: assumes signal is complex and represents matrix entries
                transformed = self._matrix_trace(signal)
            else:
                raise ValueError(f"Unknown function name: {func_name}")
        else:
            # assume it's a callable
            transformed = func(signal, *args, **kwargs)

        # Run the chip pipeline on the transformed signal
        kept, S, H, m, conv = self.processor.process(transformed)

        # Return the compressed representation and invariants
        return kept, S, H, m, conv
# ---------- 1. Address generation (elliptic‑curve points simulated) ----------
def generate_addresses(K, seed=42):
    random.seed(seed)
    # Random (x,y) in [0.5, 5.0] to avoid singularities
    addresses = [(random.uniform(0.5, 5.0), random.uniform(0.5, 5.0)) for _ in range(K)]
    return addresses

# ---------- 2. Matrix trace (scalar) ----------
def matrix_trace(x, y, i=2):
    """
    Trace of M = [[x^(i-1), x^(-1)*y^i],
                  [y^(-1)*x^i, y^(i-1)]]
    tr = x^(i-1) + y^(i-1)
    """
    return x ** (i-1) + y ** (i-1)
# ---------- 4. TSP routing (bucket sort by angle) ----------
def tsp_route(addresses):
    # Convert to complex, get phase, bucket sort
    angles = [cmath.phase(complex(x, y)) for x, y in addresses]
    # Bucket sort: 360 buckets
    buckets = [[] for _ in range(360)]
    for idx, a in enumerate(angles):
        # map a from [-π, π] to [0, 2π)
        a_norm = a + PI if a < 0 else a
        b = int((a_norm / (2 * PI)) * 360) % 360
        buckets[b].append(idx)
    order = []
    for b in buckets:
        order.extend(b)
    return order
# ---------- 6. Supertrace and mass ----------
def supertrace_and_mass(signal):
    S = 0.0
    for i, val in enumerate(signal):
        sign = 1 if (i % 2 == 0) else -1
        S += sign * abs(val)
    if S == 0:
        H = 0.0
        m = 0.0
    else:
        p = abs(S) / len(signal)
        H = -ALPHA * p * math.log(p) if p > 0 else 0.0
        m = abs(S) * math.exp(-H)
    return S, H, m

# ---------- 7. Main compression pipeline ----------
def compress_scalar_polynomial(addresses, i_exp=2):
    K = len(addresses)
    print(f"Number of addresses: {K}")

    # 1. Compute coefficients (traces)
    coeffs = np.array([matrix_trace(x, y, i_exp) for x, y in addresses], dtype=float)
    print(f"Coefficients (first 5): {coeffs[:5]}")

    # 2. TSP routing (reorder addresses and coefficients)
    order = tsp_route(addresses)
    coeffs_sorted = coeffs[order]

    # 3. Convolution with integral kernel
    conv = conv_exp_kernel(coeffs_sorted)

    # 4. Supertrace
    S, H, m = supertrace_and_mass(conv)
    M = max(1, int(abs(S)))
    if M > K:
        M = K
    print(f"Supertrace S = {S:.4f}, Entropy H = {H:.4f}, Mass m = {m:.4f}")
    print(f"Keeping M = {M} coefficients (based on |S|)")

    # 5. Möbius sieve (square‑free indices)
    mu = mobius_sieve(K)   # mu[0] unused

    # 6. Compression: keep top M magnitudes with μ(index) != 0
    mag = np.abs(conv)
    idx_sorted = np.argsort(mag)[::-1]
    kept = []
    count = 0
    for idx in idx_sorted:
        n = idx + 1   # 1‑based for μ
        if mu[n] != 0:
            kept.append((idx, conv[idx]))
            count += 1
            if count >= M:
                break
    print(f"Compressed size: {len(kept)} (ratio {len(kept)/K:.3f})")

    # 7. Reconstruct (zero out non‑kept)
    recon = np.zeros(K, dtype=complex)
    for idx, val in kept:
        recon[idx] = val
    error = np.linalg.norm(conv - recon) / np.linalg.norm(conv)
    print(f"Reconstruction relative L2 error: {error:.4e}")

    return kept, S, H, m, conv, coeffs, order

# ---------- 5. Integral kernel convolution (exponential filter, O(K)) ----------
def conv_exp_kernel(signal, alpha=ALPHA):
    K = len(signal)
    lam = math.exp(-alpha)
    # Forward pass
    f = np.zeros(K)
    f[0] = signal[0]
    for i in range(1, K):
        f[i] = signal[i] + lam * f[i-1]
    # Backward pass
    b = np.zeros(K)
    b[K-1] = signal[K-1]
    for i in range(K-2, -1, -1):
        b[i] = signal[i] + lam * b[i+1]
    # Convolution with exp(-alpha|i-j|)
    conv_exp = (f + b - signal) / (1 - lam * lam)
    # Integral kernel: (1 - conv_exp) / NORM
    conv = (1.0 - conv_exp) / NORM
    return conv

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

# ---------- Elliptic permutation (TSP routing) ----------
def elliptic_permutation(K, omega1=PI, omega2=E):
    """
    Generate a permutation of indices 0..K-1 based on pseudo‑angle.
    """
    delta = omega1 - omega2   # π - e
    angles = [(i * delta) % (2 * PI) for i in range(K)]
    order = sorted(range(K), key=lambda i: angles[i])
    return order

# ---------- Exponential convolution (two‑pass, O(K)) ----------
def conv_exp_kernel(signal, alpha=ALPHA):
    K = len(signal)
    lam = math.exp(-alpha)
    f = np.zeros(K, dtype=float)
    f[0] = signal[0]
    for i in range(1, K):
        f[i] = signal[i] + lam * f[i-1]
    b = np.zeros(K, dtype=float)
    b[K-1] = signal[K-1]
    for i in range(K-2, -1, -1):
        b[i] = signal[i] + lam * b[i+1]
    conv_exp = (f + b - signal) / (1 - lam * lam)
    conv = (1.0 - conv_exp) / NORM
    return conv

# ---------- Supertrace and entropy ----------
def supertrace_from_signal(signal):
    S = 0.0
    for idx, val in enumerate(signal):
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

def mass_from_signal(signal):
    S = supertrace_from_signal(signal)
    H = entropy_from_supertrace(S, len(signal))
    return abs(S) * math.exp(-H)

# ---------- Basel checksum ----------
def basel_checksum(mu, K):
    S = 0.0
    for n in range(1, K + 1):
        if mu[n] != 0:
            S += mu[n] / (n * n)
    return S

# ---------- Main Gate ----------
class MobiusAsciiGate:
    """
    Processes ASCII text into a compact hexadecimal fingerprint using:
      - ASCII → polynomial coefficients (monomials)
      - Elliptic permutation (TSP routing)
      - Möbius filtering (keep square‑free indices)
      - Optional logic gate (log, exp, sin, cos)
      - Supertrace compression
    Output: hex string of the compressed coefficients (magnitudes).
    """

    def __init__(self, K=256, logic_gate='log', use_elliptic=True):
        self.K = K
        self.logic_gate = logic_gate
        self.use_elliptic = use_elliptic
        self.mu = mobius_sieve(K)
        self.basel_ref = basel_checksum(self.mu, K)

    def _ascii_to_polynomial(self, text, num_vars=6):
        """
        Convert ASCII text into a polynomial with K monomials.
        Each character's ASCII code is used to generate a coefficient.
        """
        random.seed(hash(text) % (2**32))
        monomials = []
        for i, ch in enumerate(text[:self.K]):
            coeff = (ord(ch) - 32) / 95.0   # normalize to [0,1]
            # Add some randomness based on index
            coeff += 0.1 * math.sin(i * 0.5)
            exps = tuple(random.randint(0, 3) for _ in range(num_vars))
            monomials.append((coeff, exps))
        # Pad if shorter than K
        while len(monomials) < self.K:
            monomials.append((0.0, (0,)*num_vars))
        return monomials

    def _apply_mobius_filter(self, monomials):
        """Keep only monomials with square‑free index (μ(n) != 0)."""
        filtered = []
        for idx, (coeff, exps) in enumerate(monomials):
            n = idx + 1
            if self.mu[n] != 0:
                filtered.append((coeff, exps))
        return filtered

    def _apply_elliptic_permutation(self, monomials):
        """Reorder monomials by elliptic angle."""
        K = len(monomials)
        order = elliptic_permutation(K)
        return [monomials[i] for i in order]

    def _apply_logic_gate(self, signal):
        """Apply logic gate to the signal (array of coefficients)."""
        if self.logic_gate == 'log':
            return np.log(np.maximum(np.abs(signal), 1e-12))
        elif self.logic_gate == 'exp':
            return np.exp(signal)
        elif self.logic_gate == 'sin':
            return np.sin(signal)
        elif self.logic_gate == 'cos':
            return np.cos(signal)
        else:
            return signal

    def _chip_compress(self, signal):
        """Run chip pipeline: convolution + supertrace + Möbius compression."""
        K = len(signal)
        # Convolution
        conv = conv_exp_kernel(signal)
        # Supertrace and mass
        S = supertrace_from_signal(conv)
        H = entropy_from_supertrace(S, K)
        m = mass_from_signal(conv)
        M = max(1, int(abs(S)))
        if M > K:
            M = K
        # Keep top M magnitudes with square‑free index
        mag = np.abs(conv)
        idx_sorted = np.argsort(mag)[::-1]
        kept = []
        count = 0
        for idx in idx_sorted:
            n = idx + 1
            if self.mu[n] != 0:
                kept.append((idx, conv[idx]))
                count += 1
                if count >= M:
                    break
        # Reconstruct for error check (optional)
        recon = np.zeros(K, dtype=complex)
        for idx, val in kept:
            recon[idx] = val
        error = np.linalg.norm(conv - recon) / (np.linalg.norm(conv) + 1e-12)
        return kept, S, H, m, error

    def process_text(self, text, num_vars=6):
        """
        Process text and return:
          - hex fingerprint (string)
          - metadata (S, H, m, error, basel_density)
        """
        # 1. ASCII → polynomial
        monomials = self._ascii_to_polynomial(text, num_vars)

        # 2. Elliptic permutation (optional)
        if self.use_elliptic:
            monomials = self._apply_elliptic_permutation(monomials)

        # 3. Möbius filter (keep square‑free)
        filtered = self._apply_mobius_filter(monomials)

        # 4. Extract coefficients as signal (absolute values |C_i|)
        signal = np.array([abs(c) for c, _ in filtered], dtype=float)

        # 5. Apply logic gate
        signal = self._apply_logic_gate(signal)

        # 6. Chip compression
        kept, S, H, m, error = self._chip_compress(signal)

        # 7. Basel density check
        density = basel_checksum(self.mu, self.K) / self.K if self.K > 0 else 0

        # 8. Hexadecimal fingerprint: pack kept coefficients into hex
        # Flatten (index, value) into a bytearray
        data = bytearray()
        for idx, val in kept:
            # pack index (uint16) and value as float (8 bytes)
            data.extend(struct.pack('>Hd', idx, val))
        # Hash to fixed length
        fingerprint = hashlib.sha256(data).hexdigest()

        metadata = {
            'S': S,
            'H': H,
            'm': m,
            'error': error,
            'basel_density': density,
            'num_kept': len(kept)
        }
        return fingerprint, metadata

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

# ---------- CreativeMemorySystem class (unchanged except the observe method uses the function) ----------
class CreativeMemorySystem:
    def __init__(self, K=128, num_memories=100):
        self.K = K
        self.num_memories = num_memories
        self.gate = MobiusAsciiGate(K=K, logic_gate='log', use_elliptic=True)
        self.memory = {}          # signature -> (trace, metadata)
        self.exploration_cell = LearnedPendulumCell(hidden_dim=4)
        self.processor = ChipProcessor(max_K=K)
        self.history = []

    def observe(self, input_data):
        """Convert input to a Möbius signature with finite‑step derivative."""
        if isinstance(input_data, str):
            fp, meta = self.gate.process_text(input_data)
            signal = np.array([ord(c) for c in input_data[:self.K]], dtype=float)
        else:
            signal = np.array(input_data[:self.K], dtype=float)
            fp = self.gate.fingerprint(str(signal.tobytes()))  # fallback
        # Apply finite‑step derivative at the start (like in atrophy.py)
        deriv = finite_derivative(signal, step=A)   # now a function
        trace = deriv[:self.K]
        return fp, trace

        # ---------- 1. Observe ----------
    def observe(self, input_data):
        """Convert input to a Möbius signature with finite‑step derivative."""
        if isinstance(input_data, str):
            fp, meta = self.gate.process_text(input_data)
            signal = np.array([ord(c) for c in input_data[:self.K]], dtype=float)
        else:
            signal = np.array(input_data[:self.K], dtype=float)
            fp = self.gate.fingerprint(str(signal.tobytes()))  # fallback
        # Apply finite‑step derivative at the start (like in atrophy.py)
        deriv = finite_derivative(signal, step=A)
        # Store the full derivative signal as the trace (length K)
        trace = deriv[:self.K]
        # Also extract initial state for the pendulum (first 4 elements)
        initial_state = trace[:4] if len(trace) >= 4 else np.pad(trace, (0, 4 - len(trace)), constant_values=0)
        return fp, trace, initial_state

    # ---------- 2. Recall ----------
    def recall(self, query_trace, top_k=5):
        """Retrieve top‑k similar memories using inverse score (merge sort)."""
        scores = []
        for sig, (stored_trace, meta) in self.memory.items():
            # Compute inverse score only if lengths match
            if len(stored_trace) == len(query_trace):
                score = inverse_score(stored_trace.tolist(), query_trace.tolist())
                scores.append((score, sig, stored_trace, meta))
        scores.sort(key=lambda x: x[0])   # lower score = more similar
        return scores[:top_k]

    # ---------- 3. Explore ----------
    def explore(self, base_trace, initial_state=None, steps=50, dt=0.04):
        """
        Perturb the base trace using the pendulum dynamics.
        If initial_state is None, use the first 4 elements of base_trace.
        Returns a list of explored states (each a 4‑element array).
        """
        if initial_state is None:
            # Use the base_trace's first 4 elements as initial state
            init = base_trace[:4] if len(base_trace) >= 4 else np.pad(base_trace, (0, 4 - len(base_trace)), constant_values=0)
        else:
            init = initial_state
        # Ensure it's a numpy array of length 4
        init = np.array(init, dtype=float).flatten()[:4]
        if len(init) < 4:
            init = np.pad(init, (0, 4 - len(init)), constant_values=0)
        # Use the base_trace as the driving input (zeta) for the pendulum
        zeta_input = base_trace[:steps] if len(base_trace) >= steps else np.pad(base_trace, (0, steps-len(base_trace)))
        # Run long‑thought simulation
        states = long_thought_simulation(self.exploration_cell, zeta_input, init, dt=dt, steps=steps)
        # Return the full state vectors (each is a 4‑element array)
        explored_states = [s for s in states]   # already numpy arrays of shape (4,)
        return explored_states

    # ---------- 4. Create (blend) ----------
    def create(self, explored_states, original_trace, entropy_weight=0.5):
        """
        Blend explored states weighted by entropy and deviation from original.
        Returns the creative output state (4‑element array).
        """
        # original_trace is the derivative signal; we need a reference state from it
        ref_state = original_trace[:4] if len(original_trace) >= 4 else np.pad(original_trace, (0, 4 - len(original_trace)), constant_values=0)
        weights = []
        for state in explored_states:
            # state is a numpy array of length 4
            # Compute supertrace of this state
            S = supertrace_and_mass(state)[0]   # returns (S, H, m)
            # Entropy: use absolute supertrace as a measure
            ent = abs(S) if S != 0 else 0.1
            # Deviation from original reference state: inverse score between state and ref_state
            dev = inverse_score(state.tolist(), ref_state.tolist())
            # Weight: high entropy and low deviation (similar but creative)
            w = ent * (1.0 - dev)
            weights.append(w)
        # Normalise weights
        weights = np.array(weights)
        if weights.sum() == 0:
            # fallback: equal weights
            weights = np.ones_like(weights) / len(weights)
        else:
            weights = weights / weights.sum()
        # Blend states
        blended = np.zeros_like(explored_states[0])
        for state, w in zip(explored_states, weights):
            blended += w * state
        return blended

    # ---------- 5. Filter ----------
    def filter_predictions(self, predictions, threshold=0.3):
        """
        Remove predictions that have high inverse score (bad) at tail ends.
        predictions: list of states (each a 4‑element array)
        """
        good = []
        if self.memory:
            # Use the most recent memory's trace as reference (we need a state reference)
            ref_trace = list(self.memory.values())[0][0]  # first stored trace (full derivative)
            ref_state = ref_trace[:4] if len(ref_trace) >= 4 else np.pad(ref_trace, (0, 4 - len(ref_trace)), constant_values=0)
            for pred in predictions:
                score = inverse_score(pred.tolist(), ref_state.tolist())
                if score < threshold:
                    good.append(pred)
        else:
            good = predictions
        return good

    # ---------- 6. Full pipeline ----------
    def process(self, input_data, store=True):
        """
        Run the full creative pipeline on an input.
        If store=True, the input is also stored as a memory.
        Returns the creative output state and metadata.
        """
        # 1. Observe
        fp, trace, init_state = self.observe(input_data)

        # 2. Recall (if there are memories)
        if self.memory:
            similar = self.recall(trace, top_k=3)
        else:
            similar = []

        # 3. Explore: use the best matched memory's trace as base, or the input trace
        if similar:
            base_trace = similar[0][2]   # best match trace (derivative signal)
        else:
            base_trace = trace

        explored = self.explore(base_trace, initial_state=init_state, steps=50)

        # 4. Create (blend)
        creative_output = self.create(explored, trace)

        # 5. Filter bad predictions (tail ends)
        good_explored = self.filter_predictions(explored, threshold=0.4)

        # Final output: use the creative blend (we can optionally refine with good_explored)
        final_output = creative_output

        # 6. Store the input as a memory if requested
        if store:
            # Store the trace (derivative signal) and also the initial state?
            self.memory[fp] = (trace, {'input': input_data, 'timestamp': len(self.memory)})

        # Compute final entropy of the output
        final_S = supertrace_and_mass(final_output)[0]
        final_entropy = abs(final_S)

        # Use finite‑step derivative at the end (like a Gaussian smoothing) – not needed for state vector
        # We can just return the blended state.
        return final_output, {'entropy': final_entropy, 'num_explored': len(explored)}
# ---------- Demo ----------
def demo():
    system = CreativeMemorySystem(K=128)

    # Seed some memories (example: short texts)
    seed_texts = [
        "Hello world",
        "Creative AI",
        "Memory and exploration",
        "Möbius transform"
    ]
    for text in seed_texts:
        system.process(text, store=True)

    # New input
    query = "AI creativity"
    output, meta = system.process(query, store=False)

    print("Creative output (first 10 values):", output[:10])
    print("Output entropy:", meta['entropy'])
    print("Number of explored states:", meta['num_explored'])

if __name__ == "__main__":
    demo()

input('Press ENTER to exit')
   