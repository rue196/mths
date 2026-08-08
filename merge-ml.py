import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import networkx as nx
import random
import time

# ---------- Word2Vec & PyTorch ----------
try:
    import gensim
    from gensim.models import Word2Vec
    HAS_GENSIM = True
except ImportError:
    HAS_GENSIM = False
    print("Install gensim for Word2Vec training.")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Install PyTorch for neural network (optional).")

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)   # ≈ 2.362

# ===================================================================
# Part 1: Functions from 3dgraphdeep.py (graph embedding pipeline)
# ===================================================================
def load_word_vectors(num_words=50, dim=50):
    np.random.seed(42)
    words = [f"word_{i}" for i in range(num_words)]
    emb = np.random.randn(num_words, dim)
    for i in range(10):
        emb[i] = emb[0] + 0.1 * np.random.randn(dim)
    return words, emb

def project_to_3d(emb, method='pca'):
    if method == 'pca':
        pca = PCA(n_components=3)
        coords = pca.fit_transform(emb)
    else:
        tsne = TSNE(n_components=3, perplexity=min(30, len(emb)-1))
        coords = tsne.fit_transform(emb)
    coords = (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0) + 1e-12)
    return coords

def compute_spectral_coeffs(coords, alpha=ALPHA):
    return alpha * np.abs(coords.sum(axis=1)) + 1e-8

def tsp_route_words(coords):
    angles = np.arctan2(coords[:,1], coords[:,0])
    return np.argsort(angles)

def build_even_graph(N, extra_edges_per_vertex=1):
    edges = set()
    for i in range(N):
        edges.add((i, (i+1) % N))
    for k in range(2, extra_edges_per_vertex+2):
        for i in range(N):
            j = (i + k) % N
            if i != j:
                edges.add((min(i,j), max(i,j)))
    return list(edges)

def build_transition_matrix(coords, edges, C_abs, beta=1.0, collatz_mask=True):
    N = coords.shape[0]
    row, col, data = [], [], []
    for i in range(N):
        if collatz_mask and (i % 2 != 0):
            row.append(i); col.append(i); data.append(1.0)
            continue
        neighbors = []
        for (a,b) in edges:
            if a == i:
                neighbors.append(b)
            elif b == i:
                neighbors.append(a)
        neighbors = list(set(neighbors))
        weights = []
        for j in neighbors:
            if i == j: continue
            d = np.linalg.norm(coords[i] - coords[j])
            w = math.exp(-beta * d) * C_abs[j]
            weights.append((j, w))
        weights.append((i, 1e-6))
        total = sum(w for _, w in weights)
        if total == 0:
            total = 1.0
            weights = [(i, 1.0)]
        for j, w in weights:
            row.append(i); col.append(j); data.append(w / total)
    return csr_matrix((data, (row, col)), shape=(N, N))

def generate_random_walks(P, start_distribution=None, num_walks=50, walk_length=10):
    N = P.shape[0]
    if start_distribution is None:
        start_distribution = np.ones(N) / N
    walks = []
    for _ in range(num_walks):
        start = np.random.choice(N, p=start_distribution)
        walk = [start]
        current = start
        for _ in range(walk_length-1):
            row = P[current].toarray().flatten()
            next_state = np.random.choice(N, p=row)
            walk.append(next_state)
            current = next_state
        walks.append(walk)
    return walks

def train_word2vec(walks, words, vector_size=50, window=5, min_count=1, sg=1):
    if not HAS_GENSIM:
        return None
    sentences = [[words[idx] for idx in walk] for walk in walks]
    model = Word2Vec(sentences, vector_size=vector_size, window=window,
                     min_count=min_count, sg=sg, epochs=10)
    return model

# ===================================================================
# Part 2: Merge-sort inversion counting (from ML.py)
# ===================================================================
def merge_and_count(arr, temp_arr, left, mid, right):
    i, j, k = left, mid+1, left
    inv = 0
    while i <= mid and j <= right:
        if arr[i] <= arr[j]:
            temp_arr[k] = arr[i]
            i += 1
        else:
            temp_arr[k] = arr[j]
            inv += (mid - i + 1)
            j += 1
        k += 1
    while i <= mid:
        temp_arr[k] = arr[i]; i += 1; k += 1
    while j <= right:
        temp_arr[k] = arr[j]; j += 1; k += 1
    for i in range(left, right+1):
        arr[i] = temp_arr[i]
    return inv

def _merge_sort(arr, temp_arr, left, right):
    inv = 0
    if left < right:
        mid = (left + right) // 2
        inv += _merge_sort(arr, temp_arr, left, mid)
        inv += _merge_sort(arr, temp_arr, mid+1, right)
        inv += merge_and_count(arr, temp_arr, left, mid, right)
    return inv

def inversion_count(arr):
    n = len(arr)
    temp = [0]*n
    return _merge_sort(arr, temp, 0, n-1)

def inverse_score(a, b):
    """Return normalized inversion count (0..1) between two sequences."""
    pairs = sorted(zip(a, b), key=lambda x: x[0])
    b_sorted = [p[1] for p in pairs]
    inv = inversion_count(b_sorted)
    K = len(a)
    max_inv = K*(K-1)//2
    if max_inv == 0:
        return 0.0
    return inv / max_inv

def compute_inverse_scores(features, target):
    """
    features: (N, D) array
    target: (N,) array
    Returns: (D,) array of inverse scores between each feature and target.
    """
    D = features.shape[1]
    scores = np.zeros(D)
    for j in range(D):
        scores[j] = inverse_score(features[:, j], target)
    return scores

# ===================================================================
# Part 3: Neural network (optional, from neural.py)
# ===================================================================
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

def train_nn(X, y, hidden_dim=64, epochs=200, batch_size=8, lr=0.001):
    if not HAS_TORCH:
        return None, [], []
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.float32))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                            torch.tensor(y_test, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = SpectralNN(X.shape[1], hidden_dim=hidden_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, test_losses = [], []
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

        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}  Train Loss: {train_losses[-1]:.4f}  Test Loss: {test_losses[-1]:.4f}")
    return model, train_losses, test_losses

# ===================================================================
# MAIN: Combine everything
# ===================================================================
def main():
    print("=== Combined Pipeline: Graph Embedding + Merge-Sort Inverse Scoring + Optional NN ===\n")

    # 1. Generate graph data and embeddings
    num_words = 50
    words, emb = load_word_vectors(num_words, dim=50)
    coords = project_to_3d(emb, method='pca')
    C_abs = compute_spectral_coeffs(coords, alpha=ALPHA)

    tsp_order = tsp_route_words(coords)
    coords_ordered = coords[tsp_order]
    C_abs_ordered = C_abs[tsp_order]
    words_ordered = [words[i] for i in tsp_order]

    edges = build_even_graph(num_words, extra_edges_per_vertex=1)
    P = build_transition_matrix(coords_ordered, edges, C_abs_ordered, beta=2.0, collatz_mask=True)
    walks = generate_random_walks(P, num_walks=100, walk_length=10)

    if HAS_GENSIM:
        w2v = train_word2vec(walks, words_ordered, vector_size=50)
        if w2v:
            embeddings = np.array([w2v.wv[word] for word in words_ordered])
        else:
            embeddings = np.hstack([coords_ordered, C_abs_ordered.reshape(-1,1)])
    else:
        embeddings = np.hstack([coords_ordered, C_abs_ordered.reshape(-1,1)])

    print(f"Embedding shape: {embeddings.shape}")

    # 2. Merge-sort inverse scoring: compute inverse scores between each embedding component and the spectral coefficient (target)
    target = C_abs_ordered
    inv_scores = compute_inverse_scores(embeddings, target)
    print("\nInverse scores between each embedding feature and target (spectral coeff):")
    print(inv_scores)

    # Plot scores
    plt.figure(figsize=(10,4))
    plt.bar(range(len(inv_scores)), inv_scores, color='skyblue')
    plt.xlabel('Embedding feature index')
    plt.ylabel('Inverse score')
    plt.title('Merge-sort based inverse correlation with spectral coefficient')
    plt.grid(alpha=0.3)
    plt.show()

    # 3. (Optional) Train neural network on embeddings -> target for comparison
    if HAS_TORCH:
        print("\nTraining neural network (for comparison)...")
        model, train_loss, test_loss = train_nn(embeddings, target.reshape(-1,1),
                                                hidden_dim=16, epochs=100, batch_size=8)
        if model:
            plt.figure(figsize=(8,5))
            plt.plot(train_loss, label='Train Loss')
            plt.plot(test_loss, label='Test Loss')
            plt.xlabel('Epoch')
            plt.ylabel('MSE')
            plt.legend()
            plt.title('Neural Network Training (optional)')
            plt.grid(True)
            plt.show()

            # Example prediction
            model.eval()
            with torch.no_grad():
                sample = torch.tensor(embeddings[0:1], dtype=torch.float32)
                true_val = target[0]
                pred_val = model(sample).item()
                print(f"Example: True spectral coeff = {true_val:.4f}, Predicted = {pred_val:.4f}")
    else:
        print("\nPyTorch not installed – skipping neural network.")

    print("\nDone. Merge-sort inversion counting is the core analysis.")

if __name__ == "__main__":
    main()