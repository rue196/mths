import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ---------- Reuse functions from harmonic-alpha-Ok.py ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)          # ≈ 2.362

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

# ---------- Generate dataset (O(K * N_harmonics) per sample) ----------
def generate_spectral_dataset(num_samples=500, K_min=10, K_max=100, N_harmonics=70):
    H = harmonic_numbers(N_harmonics)
    X = []
    y = []

    for _ in range(num_samples):
        K = np.random.randint(K_min, K_max+1)
        alpha = ALPHA * (0.8 + 0.4 * np.random.rand())   # vary alpha
        c = build_coeffs(K)
        features = []
        target_sum = 0.0
        for n in range(1, N_harmonics+1):
            t = H[n]
            z = zeta_at(t, c, alpha)
            features.append(z.real)
            features.append(z.imag)
            target_sum += abs(z)
        X.append(features)
        y.append(target_sum)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)

# ---------- Build the neural network ----------
class SpectralNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
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

# ---------- Main: train and evaluate ----------
def main():
    print("Generating spectral dataset...")
    X, y = generate_spectral_dataset(num_samples=500000, K_min=10, K_max=100000, N_harmonics=30)
    print(f"Feature shape: {X.shape}, Target shape: {y.shape}")

    # Train/test split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # DataLoaders
    batch_size = 32
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model, loss, optimizer
    input_dim = X.shape[1]
    model = SpectralNN(input_dim, hidden_dim=128)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 100
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
        train_losses.append(epoch_loss / len(train_loader.dataset))

        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                pred = model(batch_X)
                loss = criterion(pred, batch_y)
                test_loss += loss.item() * batch_X.size(0)
        test_losses.append(test_loss / len(test_loader.dataset))

        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}  Train Loss: {train_losses[-1]:.4f}  Test Loss: {test_losses[-1]:.4f}")

    # Plot loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('Training Progress')
    plt.grid(True)
    plt.show()

    # Example prediction – FIX: extract scalar from array
    # Example prediction – corrected extraction
    model.eval()
    with torch.no_grad():
     sample_idx = 0
    sample_input = torch.tensor(X_test[sample_idx:sample_idx+1])
    true_val = float(y_test[sample_idx, 0])   # <--- index the column
    # or: true_val = y_test[sample_idx].item()
    pred_val = model(sample_input).item()
    print(f"\nExample prediction: True = {true_val:.4f}, Predicted = {pred_val:.4f}")
if __name__ == "__main__":
    main()