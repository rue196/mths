import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)          # ≈ 2.362

# ---------- Mobius and harmonic functions (from harmonic-alpha-Ok.py) ----------
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

def harmonic_numbers(K):
    H = np.zeros(K + 1, dtype=float)
    if K >= 1:
        H[1] = 1.0
    for n in range(2, K + 1):
        H[n] = H[n-1] + 1.0 / n
    return H

def build_coeffs(K):
    mu = mobius_sieve(K)
    c = np.zeros(2*K + 1, dtype=complex)
    for n in range(1, K + 1):
        c[K + n] = mu[n]
        c[K - n] = mu[n]
    c[K] = 0.0
    return c

def zeta_at(t, c, alpha):
    K = (len(c) - 1) // 2
    total = 0.0 + 0.0j
    for i in range(-K, K + 1):
        total += c[i + K] * np.exp(1j * t * i / alpha)
    return total

# ---------- Supertrace and entropy ----------
def compute_supertrace_and_entropy(K_max, N_harmonics, alpha=ALPHA):
    """
    Returns:
      - S: supertrace (scalar)
      - H_ent: entropy (scalar)
      - zeta_list: list of complex zeta values (for feature construction)
    """
    c = build_coeffs(K_max)
    H = harmonic_numbers(N_harmonics)
    S = 0.0
    zeta_vals = []
    for n in range(1, N_harmonics + 1):
        t = H[n]
        z = zeta_at(t, c, alpha)
        zeta_vals.append(z)
        val = z.real
        if n % 2 == 0:      # even → boson → positive
            S += val
        else:               # odd  → fermion → negative
            S -= val
    # Entropy
    ratio = abs(S) / N_harmonics
    if ratio <= 0 or ratio >= 1:
        H_ent = 0.0
    else:
        H_ent = -alpha * ratio * math.log(ratio)
    return S, H_ent, zeta_vals

# ---------- Generate dataset with entropy as additional feature ----------
def generate_spectral_dataset(num_samples=500, K_min=10, K_max=60, N_harmonics=30):
    X = []       # features: real and imag of zeta at each harmonic
    y = []       # target: sum of |zeta|
    entropy_vals = []   # extra target: entropy of supertrace

    for _ in range(num_samples):
        K = np.random.randint(K_min, K_max+1)
        alpha = ALPHA * (0.8 + 0.4 * np.random.rand())
        S, H_ent, zeta_list = compute_supertrace_and_entropy(K, N_harmonics, alpha)
        features = []
        target_sum = 0.0
        for z in zeta_list:
            features.append(z.real)
            features.append(z.imag)
            target_sum += abs(z)
        X.append(features)
        y.append(target_sum)
        entropy_vals.append(H_ent)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    entropy_vals = np.array(entropy_vals, dtype=np.float32).reshape(-1, 1)
    return X, y, entropy_vals

# ---------- Neural network: multi‑output (target + entropy) ----------
class SpectralNNWithEntropy(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=2):  # output: (sum_z, entropy)
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# ---------- Training function ----------
def train_model(X, y, entropy, hidden_dim=128, epochs=200, batch_size=32, lr=0.001):
    # Combine targets
    Y = np.hstack([y, entropy])   # shape (N, 2)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(Y_train, dtype=torch.float32))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                            torch.tensor(Y_test, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = SpectralNNWithEntropy(X.shape[1], hidden_dim=hidden_dim, output_dim=2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_X)          # (batch, 2)
            loss = criterion(pred, batch_Y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
        train_losses.append(epoch_loss / len(train_loader.dataset))

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_Y in test_loader:
                pred = model(batch_X)
                loss = criterion(pred, batch_Y)
                test_loss += loss.item() * batch_X.size(0)
        test_losses.append(test_loss / len(test_loader.dataset))

        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}  Train Loss: {train_losses[-1]:.4f}  Test Loss: {test_losses[-1]:.4f}")

    return model, train_losses, test_losses

# ---------- Main ----------
def main():
    print("Generating dataset with supertrace entropy...")
    X, y, entropy = generate_spectral_dataset(num_samples=500, K_min=10, K_max=60, N_harmonics=30)
    print(f"Features shape: {X.shape}, Target shape: {y.shape}, Entropy shape: {entropy.shape}")

    # Train network
    model, train_loss, test_loss = train_model(X, y, entropy, hidden_dim=128, epochs=200)

    # Plot losses
    plt.figure(figsize=(8,5))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(test_loss, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('Multi‑output training (sum_z + entropy)')
    plt.grid(True)
    plt.show()

    # Example prediction
    model.eval()
    with torch.no_grad():
        sample_idx = 0
        sample_input = torch.tensor(X[sample_idx:sample_idx+1], dtype=torch.float32)
        true_sum = float(y[sample_idx, 0])
        true_ent = float(entropy[sample_idx, 0])
        pred = model(sample_input).numpy().flatten()
        pred_sum, pred_ent = pred[0], pred[1]
        print(f"\nExample sample:")
        print(f"  True sum_z = {true_sum:.4f}, Predicted sum_z = {pred_sum:.4f}")
        print(f"  True entropy = {true_ent:.4f}, Predicted entropy = {pred_ent:.4f}")

    # Show entropy bounds
    print("\nTheoretical entropy bounds:")
    x_vals = np.linspace(0.001, 0.999, 100)
    h_vals = -ALPHA * x_vals * np.log(x_vals)
    max_ent = np.max(h_vals)
    print(f"  Maximum entropy = {max_ent:.4f} (at x = 1/e ≈ 0.3679)")

if __name__ == "__main__":
    main()