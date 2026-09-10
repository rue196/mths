# app.py
import math
import random
import numpy as np
from flask import Flask, request, jsonify, render_template

# ---------- PDE constants (from semiotic-differential.c) ----------
K = 128               # hidden state size
VOCAB = 1000          # vocabulary size (will be adapted)
DT = 0.1
ALPHA = 0.1
EPS = 0.01

# Global parameters (randomly initialised, kept fixed)
W_in = None
W_out = None
anchors = None

def init_weights(vocab_size):
    global W_in, W_out, anchors, VOCAB
    VOCAB = vocab_size
    W_in = np.random.randn(K, VOCAB) * 0.01
    W_out = np.random.randn(VOCAB, K) * 0.01
    anchors = np.arange(0, K, K//20)  # 20 evenly spaced anchors

def laplacian(h):
    diff = np.zeros_like(h)
    diff[0] = 0.0
    diff[-1] = 0.0
    for i in range(1, K-1):
        diff[i] = h[i-1] + h[i+1] - 2.0 * h[i]
    return diff

def pde_step(h, token):
    # Input injection
    h += W_in[:, token]
    # Diffusion, nonlocal, decay
    diff = laplacian(h)
    an_sum = np.sum(h[anchors])
    nonlocal_term = ALPHA * an_sum
    h += DT * (diff + nonlocal_term - EPS * h)
    return h

def document_vector(tokens):
    h = np.zeros(K)
    for token in tokens:
        h = pde_step(h, token)
    return h

# ---------- Corpus ----------
documents = [
    {"id": 0, "title": "Cat", "content": "Cats are small carnivorous mammals. They are popular pets."},
    {"id": 1, "title": "Dog", "content": "Dogs are domesticated canines, often kept as pets."},
    {"id": 2, "title": "Car", "content": "Cars are vehicles with wheels, powered by engines."},
    {"id": 3, "title": "Python", "content": "Python is a programming language used for web development and data science."},
    {"id": 4, "title": "Chess", "content": "Chess is a two-player board game with strategy and tactics."}
]

# Build vocabulary
def tokenize(text):
    return text.lower().split()

vocab = {}
corpus_tokens = []
for doc in documents:
    tokens = tokenize(doc["content"])
    doc["tokens"] = tokens
    corpus_tokens.extend(tokens)

# Map tokens to indices (simple frequency-based)
from collections import Counter
freq = Counter(corpus_tokens)
# Keep top VOCAB-1 frequent, reserve 0 for unknown
vocab_list = [word for word, _ in freq.most_common(VOCAB-1)]
vocab = {word: i+1 for i, word in enumerate(vocab_list)}  # 1..V-1
vocab["<UNK>"] = 0
VOCAB = len(vocab)

# Initialise weights with actual vocab size
init_weights(VOCAB)

# Precompute document vectors
doc_vectors = []
for doc in documents:
    tokens = [vocab.get(w, 0) for w in doc["tokens"]]
    vec = document_vector(tokens)
    doc_vectors.append(vec)

# ---------- Merge Sort (from sorting-a.py) ----------
def merge_sort_with_a(arr, a=0.5866):
    """Sort list of (doc_id, similarity) by similarity descending using merge sort."""
    if len(arr) <= 1:
        return arr

    def less_equal_than_a(x, y):
        # x and y are tuples (doc_id, sim)
        diff = abs(x[1] - y[1])
        avg = (abs(x[1]) + abs(y[1])) / 2.0
        return x[1] >= y[1] - a * avg   # we want descending

    def merge(left, right):
        result = []
        i = j = 0
        while i < len(left) and j < len(right):
            if less_equal_than_a(left[i], right[j]):
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        result.extend(left[i:])
        result.extend(right[j:])
        return result

    mid = len(arr) // 2
    left = merge_sort_with_a(arr[:mid], a)
    right = merge_sort_with_a(arr[mid:], a)
    return merge(left, right)

# ---------- Search ----------
def search(query):
    query_tokens = tokenize(query)
    q_tokens = [vocab.get(w, 0) for w in query_tokens]
    q_vec = document_vector(q_tokens)
    # Compute cosine similarity
    similarities = []
    for i, vec in enumerate(doc_vectors):
        norm_q = np.linalg.norm(q_vec)
        norm_d = np.linalg.norm(vec)
        if norm_q == 0 or norm_d == 0:
            sim = 0.0
        else:
            sim = np.dot(q_vec, vec) / (norm_q * norm_d)
        similarities.append((i, sim))
    # Sort descending using merge sort
    sorted_sims = merge_sort_with_a(similarities)
    # Return top results
    results = []
    for doc_id, sim in sorted_sims:
        if sim > 0.01:  # threshold
            doc = documents[doc_id]
            results.append({
                "id": doc["id"],
                "title": doc["title"],
                "content": doc["content"],
                "similarity": float(sim)
            })
    return results

# ---------- Flask App ----------
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('test/search.html')

@app.route('/search')
def search_endpoint():
    query = request.args.get('q', '')
    if not query:
        return jsonify([])
    results = search(query)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)