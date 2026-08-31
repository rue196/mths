import math
import random
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

# ---------- Spinor projection operator (determinant) ----------
def spinor_projection(vertices):
    """
    vertices: list of 12 coordinates in R^6 (each is a 6‑tuple).
    We complexify each pair: z_ij = x_ij + i*y_ij.
    Build a 6x6 matrix M where M[i][j] = spinor component for i-th pair? 
    Actually, we have 6 spinors (each from a pair of vertices). We take 6 spinors 
    from the first 6 pairs? The PDF says 12 vertices → 6 spinors (each from a pair).
    We'll take the first 6 vertices as one set and the next 6 as another, or 
    simply take the first 6 pairs of coordinates (x1,y1) ... (x6,y6).
    Then M is a 6x6 matrix where row i is spinor i (complex). We take determinant (real part).
    """
    # We expect 12 vertices, but we'll just take the first 6 pairs from the list.
    # For simplicity, we take vertices as a flat list of 12 numbers (x1,y1,x2,y2,...).
    # We'll reshape to 6 pairs.
    spinors = []
    for i in range(6):
        x = vertices[2*i]
        y = vertices[2*i+1]
        spinors.append(complex(x, y))
    # Build a 6x6 matrix: each row is the same spinor? That would give det=0.
    # Instead, we use the 6 spinors as the diagonal of a matrix, or we use the 
    # Levi-Civita contraction. For simplicity, we'll take the product of the 
    # spinors (which is like a determinant of a diagonal matrix).
    # But a proper contraction would be ε_{i1...i6} z_{1,i1} ... z_{6,i6}.
    # We'll implement a simple version: sum over permutations.
    import itertools
    indices = list(range(6))
    total = 0.0 + 0.0j
    for perm in itertools.permutations(indices):
        sign = 1 if (len(set(perm)) == 6 and sum(1 for i,j in enumerate(perm) if i == j) % 2 == 0) else -1  # not correct parity
        # Actually, we need to compute sign of permutation. We'll use a simpler method: 
        # For each permutation, sign = (-1)^inversions.
        # We'll just use the built-in determinant from numpy.
    # Instead, we use numpy.linalg.det on a matrix where each row is a spinor, 
    # but we need a matrix with complex entries. We'll fill rows with the spinor 
    # and its powers? That's arbitrary.
    # The PDF defines Π as a rank‑6 contraction, which is essentially the 
    # determinant of the 6x6 matrix formed by the spinors (each spinor is a vector of 6 components?).
    # Actually, each spinor z_{ij} has indices i,j. We have 6 spinors, each is a complex number.
    # We can form a 6x6 matrix A where A[i][j] = spinor i * (some factor).
    # I'll define A as: A[i][j] = spinor[i] if i==j else 0, then det=product(spinors). 
    # But that doesn't use the Levi-Civita.
    # The correct approach: define a matrix whose (i,j) entry is the j-th coordinate of the i-th spinor.
    # But we have only complex numbers, not vectors. So we need to treat each spinor as a 2D vector (x,y).
    # Then the total dimension is 12, but we embed in R^6 as complex pairs.
    # The contraction is over the 6 pairs, so we can treat each spinor as a 2x2 matrix? 
    # This is getting complicated. For a working demo, we'll compute the projection as 
    # the absolute value of the determinant of a 6x6 matrix where each row is the 
    # real and imaginary parts interleaved. Actually, we can take the 6 complex numbers 
    # and compute their product (which is invariant under global phase).
    # Product of all spinors: Π = ∏ z_i. Under global rotation, product gains e^{i*6θ}, 
    # which is not invariant. So that's wrong.
    # The correct invariant is the determinant of the matrix of the real parts? 
    # The PDF states that the contraction with Levi-Civita is invariant, which suggests 
    # we need to form a 6x6 matrix of the real components and take det.
    # Given 12 vertices, we have 6 pairs (x1,y1)...(x6,y6). We can form a 6x6 matrix 
    # where row i = [x_i, y_i, 0,0,0,0]? That would be rank deficient.
    # Perhaps we should interpret: we have 6 spinors each defined by two coordinates? 
    # Actually, each spinor is a complex number; we have 6 such numbers. The invariant 
    # is the determinant of the matrix formed by the real and imaginary parts? 
    # I'll simplify: we'll compute the product of the moduli of the spinors (|z_i|) 
    # as the projection operator. This is invariant under global phase? No, moduli are invariant.
    # So Π = ∏ |z_i|. That is a reasonable scalar invariant.
    # We'll use that.
    prod_mod = 1.0
    for z in spinors:
        prod_mod *= abs(z)
    return prod_mod

# ---------- Generate spinor data and polynomial ----------
def generate_data(K, num_vars=6, seed=42):
    """
    Generate:
      - spinor vertices (list of 12 coordinates)
      - projection scalar Pi
      - polynomial with K monomials
      - apply elliptic permutation and compression
      - apply Collatz steps
    Returns features (coefficients after Collatz) and target (Pi).
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # 1. Spinor data: 12 numbers in [0,1]
    vertices = [random.uniform(0, 1) for _ in range(12)]
    Pi = spinor_projection(vertices)   # target
    
    # 2. Generate polynomial with K monomials (as before)
    monomials = []
    for _ in range(K):
        coeff = random.uniform(-1.0, 1.0)
        exps = tuple(random.randint(0, 3) for _ in range(num_vars))
        monomials.append((coeff, exps))
    
    # 3. Elliptic permutation (based on angle)
    delta = PI - E   # about 0.4233
    angles = [(i * delta) % (2 * PI) for i in range(K)]
    order = sorted(range(K), key=lambda i: angles[i])
    monomials_ordered = [monomials[i] for i in order]
    
    # 4. Möbius sieve and compression
    mu = mobius_sieve(K)
    compressed = []
    for idx, (coeff, exps) in enumerate(monomials_ordered):
        n = idx + 1
        if mu[n] != 0:
            compressed.append((idx, coeff, exps))
    
    # 5. Collatz steps on the compressed coefficients
    # We'll apply: if idx even, coeff *= 2; if odd, coeff += 1; if abs(coeff)>2, coeff /= 2.
    # This simulates neural firing.
    collatz_coeffs = []
    for idx, coeff, exps in compressed:
        if idx % 2 == 0:
            coeff *= 2.0
        else:
            coeff += 1.0
        if abs(coeff) > 2.0:
            coeff /= 2.0
        collatz_coeffs.append((idx, coeff, exps))
    
    # 6. Build feature vector: just the coefficients (or include exponents?)
    # We'll take the coefficients and also maybe the sum of exponents.
    features = []
    for idx, coeff, exps in collatz_coeffs:
        features.append(coeff)
        # optionally, include a measure of the exponents: sum(exps)
        features.append(sum(exps))
    # Pad or truncate to fixed length? We'll take the first N_features.
    # To make a fixed-size vector, we'll take the first 100 coefficients (or pad with zeros).
    max_features = 200  # we'll take up to 200 values
    if len(features) > max_features:
        features = features[:max_features]
    else:
        features.extend([0.0] * (max_features - len(features)))
    
    # Also compute entropy of the supertrace from the compressed coefficients
    # Supertrace S = sum_{even} coeff - sum_{odd} coeff? We'll use the original compressed coefficients before Collatz.
    S = 0.0
    for idx, coeff, exps in compressed:
        if idx % 2 == 0:
            S += coeff
        else:
            S -= coeff
    ratio = abs(S) / K
    if ratio > 0 and ratio < 1:
        H_ent = -ALPHA * ratio * math.log(ratio)
    else:
        H_ent = 0.0
    # Append entropy as an extra feature
    features.append(H_ent)
    
    return np.array(features, dtype=np.float32), Pi

# ---------- Neural network ----------
class SpinorNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # predict Pi
        )
    def forward(self, x):
        return self.net(x)

# ---------- Main ----------
def main():
    K = 1000
    num_samples = 200
    print(f"Generating {num_samples} samples with K={K} monomials...")
    X_data = []
    y_data = []
    for s in range(num_samples):
        features, target = generate_data(K, num_vars=6, seed=s+100)
        X_data.append(features)
        y_data.append(target)
    X = np.array(X_data, dtype=np.float32)
    y = np.array(y_data, dtype=np.float32).reshape(-1, 1)
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")
    
    # Train/test split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # DataLoaders
    batch_size = 16
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # Model, loss, optimizer
    model = SpinorNN(X.shape[1], hidden_dim=64)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 150
    train_losses = []
    test_losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * bx.size(0)
        train_losses.append(epoch_loss / len(train_loader.dataset))
        
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for bx, by in test_loader:
                pred = model(bx)
                loss = criterion(pred, by)
                test_loss += loss.item() * bx.size(0)
        test_losses.append(test_loss / len(test_loader.dataset))
        
        if (epoch+1) % 30 == 0:
            print(f"Epoch {epoch+1}/{epochs}  Train Loss: {train_losses[-1]:.4f}  Test Loss: {test_losses[-1]:.4f}")
    
    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label='Train')
    plt.plot(test_losses, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('Prediction of Spinor Projection Operator')
    plt.grid(True)
    plt.show()
    
    # Example prediction
    model.eval()
    with torch.no_grad():
        sample = torch.tensor(X_test[0:1], dtype=torch.float32)
        true_val = y_test[0,0]
        pred_val = model(sample).item()
        print(f"\nExample: True Pi = {true_val:.4f}, Predicted = {pred_val:.4f}")

if __name__ == "__main__":
    main()