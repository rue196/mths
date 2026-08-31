import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import math
import random
import networkx as nx

# For word embeddings and deep learning
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
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Install PyTorch for deep learning examples.")

# ------------------------------------------------------------
# 0. Constants
# ------------------------------------------------------------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)   # ≈ 2.362

# ------------------------------------------------------------
# 1. Generate synthetic word vectors (or load from file)
# ------------------------------------------------------------
def load_word_vectors(num_words=50, dim=50):
    """
    Simulate word embeddings. In practice, load from GloVe/Word2Vec.
    """
    np.random.seed(42)
    # Random vectors, but we'll make some correlations
    words = [f"word_{i}" for i in range(num_words)]
    # Create clusters to simulate semantic relations
    emb = np.random.randn(num_words, dim)
    # Add structure: first 10 words similar to each other
    for i in range(10):
        emb[i] = emb[0] + 0.1 * np.random.randn(dim)
    return words, emb

# ------------------------------------------------------------
# 2. Project to 3D (using PCA or t-SNE)
# ------------------------------------------------------------
def project_to_3d(emb, method='pca'):
    if method == 'pca':
        pca = PCA(n_components=3)
        coords = pca.fit_transform(emb)
    else:  # t-SNE (slower)
        tsne = TSNE(n_components=3, perplexity=min(30, len(emb)-1))
        coords = tsne.fit_transform(emb)
    # Normalize to [0,1] for convenience
    coords = (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0) + 1e-12)
    return coords

# ------------------------------------------------------------
# 3. Compute spectral coefficient |C_i| for each word
# ------------------------------------------------------------
def compute_spectral_coeffs(coords, alpha=ALPHA):
    """
    C_i = trace([[x,y,z],[y,z,x],[z,x,y]]) = x+y+z
    Returns |C_i| scaled by alpha.
    """
    C_abs = alpha * np.abs(coords.sum(axis=1)) + 1e-8
    return C_abs

# ------------------------------------------------------------
# 4. TSP routing: sort by angle of (x+iy)
# ------------------------------------------------------------
def tsp_route_words(coords):
    """
    Sort words by the phase of z = x + i*y (projection to XY plane).
    Returns a permutation (list of indices) in cyclic order.
    """
    angles = np.arctan2(coords[:,1], coords[:,0])   # y, x
    idx = np.argsort(angles)
    return idx

# ------------------------------------------------------------
# 5. Build sparse even graph (cycle + extra edges)
# ------------------------------------------------------------
def build_even_graph(N, extra_edges_per_vertex=1):
    """
    Returns a list of undirected edges (i, j) with i < j.
    All degrees are even.
    """
    edges = set()
    # Cycle
    for i in range(N):
        edges.add((i, (i+1) % N))
    # Extra edges: connect to vertices at distance 2,3,... up to extra_edges_per_vertex
    for k in range(2, extra_edges_per_vertex+2):
        for i in range(N):
            j = (i + k) % N
            if i != j:
                edges.add((min(i,j), max(i,j)))
    return list(edges)

# ------------------------------------------------------------
# 6. Build transition matrix (sparse, O(K))
# ------------------------------------------------------------
def build_transition_matrix(coords, edges, C_abs, beta=1.0, collatz_mask=True):
    """
    P_{i->j} ∝ exp(-beta * dist(i,j)) * C_abs[j]
    If collatz_mask=True, odd-indexed vertices (in the TSP order) get self-loop.
    """
    N = coords.shape[0]
    row, col, data = [], [], []
    for i in range(N):
        if collatz_mask and (i % 2 != 0):
            # Odd index: self-loop only
            row.append(i); col.append(i); data.append(1.0)
            continue
        # Neighbors
        neighbors = []
        for (a,b) in edges:
            if a == i:
                neighbors.append(b)
            elif b == i:
                neighbors.append(a)
        # Unique
        neighbors = list(set(neighbors))
        weights = []
        for j in neighbors:
            if i == j:
                continue
            d = np.linalg.norm(coords[i] - coords[j])
            w = math.exp(-beta * d) * C_abs[j]
            weights.append((j, w))
        # Add self-loop with small probability for irreducibility
        weights.append((i, 1e-6))
        total = sum(w for _, w in weights)
        if total == 0:
            total = 1.0
            weights = [(i, 1.0)]
        for j, w in weights:
            row.append(i); col.append(j); data.append(w / total)
    P = csr_matrix((data, (row, col)), shape=(N, N))
    return P

# ------------------------------------------------------------
# 7. Generate random walks (sequences of word indices)
# ------------------------------------------------------------
def generate_random_walks(P, start_distribution=None, num_walks=50, walk_length=10):
    """
    Generate random walks from the Markov chain.
    Returns list of lists of word indices.
    """
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

# ------------------------------------------------------------
# 8. Train Word2Vec on random walks (or use as features)
# ------------------------------------------------------------
def train_word2vec(walks, words, vector_size=50, window=5, min_count=1, sg=1):
    """
    Train a Word2Vec model using the random walks as sentences.
    Returns a gensim Word2Vec model.
    """
    if not HAS_GENSIM:
        print("Gensim not installed, skipping Word2Vec training.")
        return None
    # Convert indices to words
    sentences = [[words[idx] for idx in walk] for walk in walks]
    model = Word2Vec(sentences, vector_size=vector_size, window=window,
                     min_count=min_count, sg=sg, epochs=10)
    return model

# ------------------------------------------------------------
# 9. Deep learning: a simple classifier using the embeddings
# ------------------------------------------------------------
def build_classifier(embedding_dim, num_classes, hidden_dim=64):
    """
    Example: a simple MLP classifier on top of word embeddings.
    Assumes we have labels for each word.
    """
    if not HAS_TORCH:
        print("PyTorch not installed, skipping classifier.")
        return None
    class Net(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_classes):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, num_classes)
            self.relu = nn.ReLU()
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    return Net(embedding_dim, hidden_dim, num_classes)

# ------------------------------------------------------------
# 10. Main demonstration
# ------------------------------------------------------------
def main():
    # 1. Load/Generate words
    num_words = 50
    words, emb = load_word_vectors(num_words)
    print(f"Loaded {num_words} words with {emb.shape[1]} dim embeddings.")

    # 2. Project to 3D
    coords = project_to_3d(emb, method='pca')
    print("3D coordinates shape:", coords.shape)

    # 3. Spectral coefficients
    C_abs = compute_spectral_coeffs(coords, alpha=ALPHA)
    print("Spectral coefficients (first 5):", C_abs[:5])

    # 4. TSP routing
    tsp_order = tsp_route_words(coords)
    print("TSP order (first 5 indices):", tsp_order[:5])
    # Reorder coordinates and coefficients according to TSP
    coords_ordered = coords[tsp_order]
    C_abs_ordered = C_abs[tsp_order]
    words_ordered = [words[i] for i in tsp_order]

    # 5. Build graph
    N = num_words
    extra_edges = 1   # add edges to next-nearest
    edges = build_even_graph(N, extra_edges_per_vertex=extra_edges)
    print(f"Number of edges: {len(edges)}")

    # 6. Build transition matrix (with Collatz mask)
    beta = 2.0
    P = build_transition_matrix(coords_ordered, edges, C_abs_ordered,
                                beta=beta, collatz_mask=True)
    print("Transition matrix shape:", P.shape)

    # 7. Generate random walks
    walks = generate_random_walks(P, num_walks=100, walk_length=10)
    print("First walk (indices):", walks[0])
    print("First walk (words):", [words_ordered[idx] for idx in walks[0]])

    # 8. Train Word2Vec
    if HAS_GENSIM:
        w2v_model = train_word2vec(walks, words_ordered, vector_size=50)
        # Print a similarity example
        if w2v_model:
            print("Word2Vec model trained.")
            # Example: find similar to first word
            first_word = words_ordered[0]
            if first_word in w2v_model.wv:
                similar = w2v_model.wv.most_similar(first_word, topn=3)
                print(f"Similar words to '{first_word}': {similar}")

    # 9. Deep learning classifier (dummy example)
    if HAS_TORCH:
        # Assume we have labels (e.g., categories) for each word
        num_classes = 3
        labels = np.random.randint(0, num_classes, size=N)  # random labels for demo
        # Use the learned Word2Vec embeddings as input features
        if HAS_GENSIM and w2v_model:
            embeddings = np.array([w2v_model.wv[word] for word in words_ordered])
        else:
            # Fallback: use the spectral coefficients as features
            embeddings = C_abs_ordered.reshape(-1, 1)  # only one feature, not ideal
        # Build and train a simple classifier (just a skeleton)
        net = build_classifier(embeddings.shape[1], num_classes, hidden_dim=32)
        print("Classifier model created (training not executed in demo).")

    # 10. Visualize the 3D graph
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coords_ordered[:,0], coords_ordered[:,1], coords_ordered[:,2],
               c='blue', s=50)
    # Draw edges
    for (i,j) in edges:
        p1 = coords_ordered[i]
        p2 = coords_ordered[j]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color='gray', alpha=0.5)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('3D Graph (cycle + extra edges) for word embedding')
    plt.show()

if __name__ == "__main__":
    main()