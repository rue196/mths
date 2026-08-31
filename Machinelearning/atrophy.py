import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)          # ≈ 2.362

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

# ---------- Synthetic dataset ----------
def generate_spectral_data(num_samples=500, K_max=60, N_harmonics=30):
    X, y = [], []
    for _ in range(num_samples):
        K = np.random.randint(10, K_max+1)
        zetas = [complex(np.random.uniform(-1,1), np.random.uniform(-1,1)) for _ in range(N_harmonics)]
        features = []
        for z in zetas:
            features.append(z.real); features.append(z.imag)
        target = sum(abs(z) for z in zetas)
        X.append(features); y.append(target)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1,1)

# ---------- Main ----------
def main():
    # Generate data
    print("Generating spectral dataset...")
    X, y = generate_spectral_data(num_samples=500, N_harmonics=30)
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                           torch.tensor(y_val, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    # Train base model and get reference activations
    print("Training base model...")
    model = SimpleNet(input_dim=X.shape[1], hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    for epoch in range(50):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"Base epoch {epoch+1}")

    # Reference activations (averaged over training set)
    model.eval()
    all_hidden = []
    with torch.no_grad():
        for bx, _ in train_loader:
            _, hidden = model(bx, return_hidden=True)
            all_hidden.append(hidden.cpu().numpy())
    ref_activations = np.concatenate(all_hidden, axis=0)
    mean_ref = np.mean(ref_activations, axis=0)   # (hidden_dim,)

    # Compute entropy from a sample
    sample_features = X_train[0]
    zetas = [complex(sample_features[i], sample_features[i+1]) for i in range(0, len(sample_features), 2)]
    entropy = supertrace_entropy(zetas, list(range(1, len(zetas)+1)))
    print(f"Computed entropy: {entropy:.4f}")

    # ---- Atrophy simulation with prevention ----
    print("\nSimulating atrophy with prevention...")
    sim_model = SimpleNet(input_dim=X.shape[1], hidden_dim=64)
    sim_model.load_state_dict(model.state_dict())   # start from base weights

    optimizer = optim.Adam(sim_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    loss_hist, inv_hist, drift_hist = [], [], []
    atrophy_threshold = 0.3
    noise_std = 0.01
    correction_strength = 0.5

    for epoch in range(100):
        # ---- Add atrophy noise ----
        with torch.no_grad():
            for p in sim_model.parameters():
                p.add_(torch.randn_like(p) * noise_std)

        # ---- Normal training step ----
        sim_model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            pred = sim_model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * bx.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        loss_hist.append(avg_loss)

        # ---- Current activations and inverse score ----
        sim_model.eval()
        all_cur_hidden = []
        with torch.no_grad():
            for bx, _ in val_loader:
                _, hidden = sim_model(bx, return_hidden=True)
                all_cur_hidden.append(hidden.cpu().numpy())
        cur_activations = np.concatenate(all_cur_hidden, axis=0)
        mean_cur = np.mean(cur_activations, axis=0)
        inv_score = inverse_score(mean_cur.tolist(), mean_ref.tolist())
        inv_hist.append(inv_score)

        # ---- Weight drift ----
        total_drift = 0.0
        for p, init_p in zip(sim_model.parameters(), model.parameters()):
            total_drift += torch.norm(p - init_p).item()
        drift_hist.append(total_drift)

        # ---- Prevention correction ----
        if inv_score > atrophy_threshold:
            print(f"Epoch {epoch+1}: InvScore {inv_score:.3f} > threshold -> correcting")
            # Use a batch from validation
            val_batch_X, _ = next(iter(val_loader))
            sim_model.train()
            optimizer.zero_grad()   # important: clear previous gradients
            _, hidden_tensor = sim_model(val_batch_X, return_hidden=True)
            ref_tensor = torch.tensor(mean_ref, dtype=torch.float32).unsqueeze(0).repeat(val_batch_X.size(0), 1)
            corr_loss = torch.nn.functional.mse_loss(hidden_tensor, ref_tensor)
            scale = min(1.0, entropy / 0.5) * correction_strength
            corr_loss = corr_loss * scale
            corr_loss.backward()
            optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Inv={inv_score:.3f}, Drift={total_drift:.4f}")

    # ---- Plot ----
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(loss_hist); plt.title('Loss'); plt.xlabel('Epoch'); plt.grid(True)
    plt.subplot(1, 3, 2)
    plt.plot(inv_hist); plt.axhline(y=atrophy_threshold, color='r', linestyle='--', label='threshold')
    plt.title('Inverse Score'); plt.xlabel('Epoch'); plt.legend(); plt.grid(True)
    plt.subplot(1, 3, 3)
    plt.plot(drift_hist); plt.title('Weight Drift'); plt.xlabel('Epoch'); plt.grid(True)
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()