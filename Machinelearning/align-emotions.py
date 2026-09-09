"""
emotional_alignment.py

Combines Möbius sparse attention (from attention-monomial-hash-elliptic.py)
with emotion/hormone dynamics (from emotions.py) to create
alignment tiers for LLM training.

An emotional state is encoded as a sparse vector over square‑free indices,
where each index corresponds to a monomial coefficient. Different emotions
and intensities are stored as separate embeddings.

During training, a stimulus (text or hidden state) is mapped to an embedding,
and the system retrieves the most similar stored emotional memory.
This retrieved vector can be used as a conditioning signal (e.g., as a prefix
or attention bias) to guide the LLM toward empathetic responses.

The system supports hierarchical tiers (e.g., low/medium/high intensity)
to allow fine‑grained emotional control.
"""

import math
import random
import numpy as np
from collections import defaultdict

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

# ---------- Basel checksum (for density validation) ----------
def basel_checksum(mu, K):
    S = 0.0
    for n in range(1, K + 1):
        if mu[n] != 0:
            S += mu[n] / (n * n)
    return S

# ---------- Generate polynomial embedding from a seed ----------
def generate_polynomial_embedding(seed, num_monomials=100, num_vars=6):
    """
    Create a sparse embedding (dict index->coefficient) using a seed.
    Only square‑free indices (μ(n) != 0) are kept.
    """
    random.seed(seed)
    monomials = []
    for _ in range(num_monomials):
        coeff = random.uniform(-1.0, 1.0)
        exps = tuple(random.randint(0, 3) for _ in range(num_vars))
        monomials.append((coeff, exps))
    K = num_monomials
    mu = mobius_sieve(K)
    embedding = {}
    for idx, (coeff, _) in enumerate(monomials):
        n = idx + 1
        if mu[n] != 0:   # square‑free
            embedding[n] = coeff
    return embedding

# ---------- Merge‑sort inversion count (O(K log K)) ----------
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

# ---------- Sparse inverse score (for sparse embeddings) ----------
def inverse_score_sparse(query_emb, key_emb):
    """
    Compute normalised inversion count between two sparse embeddings.
    Only indices present in both are considered.
    Returns a score in [0,1]; lower means more similar ordering.
    """
    common = set(query_emb.keys()) & set(key_emb.keys())
    if len(common) < 2:
        return 0.5  # neutral
    q_vals = [query_emb[i] for i in sorted(common)]
    k_vals = [key_emb[i] for i in sorted(common)]
    inv = inversion_count(k_vals.copy())   # we need to sort q_vals? Actually the inversion count is on k_vals after ordering by q_vals.
    # Better: we order by q_vals and then count inversions in k_vals.
    pairs = sorted(zip(q_vals, k_vals), key=lambda x: x[0])
    ordered_k = [p[1] for p in pairs]
    inv = inversion_count(ordered_k)
    K = len(common)
    max_inv = K * (K-1) // 2
    return inv / max_inv if max_inv > 0 else 0.0

# ---------- Emotional embedding storage ----------
class EmotionalMemory:
    """
    Stores emotional states as sparse embeddings, each with a label,
    intensity tier (0–10), and optional meta‑data.
    Supports retrieval of the most similar emotional context for a query.
    """
    def __init__(self, max_emotions=1000):
        self.emotions = []          # list of (label, intensity, embedding)
        self.labels = set()

    def add_emotion(self, label, intensity, embedding):
        """Store an emotional state with a given intensity (0–10)."""
        self.emotions.append((label, intensity, embedding))
        self.labels.add(label)

    def query(self, query_embedding, top_k=3, threshold=0.6):
        """
        Retrieve top‑k stored emotions with inverse score < threshold.
        Returns list of (label, intensity, score).
        """
        scores = []
        for label, intensity, emb in self.emotions:
            score = inverse_score_sparse(query_embedding, emb)
            if score < threshold:
                scores.append((label, intensity, score))
        scores.sort(key=lambda x: x[2])   # ascending score
        return scores[:top_k]

    def get_embedding_for_emotion(self, label, intensity=None):
        """
        Retrieve the embedding for a specific emotion, optionally with
        exact intensity. If intensity is None, return the one with
        intensity closest to the mean.
        """
        candidates = [(intensity, emb) for lab, inten, emb in self.emotions if lab == label]
        if not candidates:
            return None
        if intensity is not None:
            # find closest intensity
            best = min(candidates, key=lambda x: abs(x[0] - intensity))
            return best[1]
        else:
            # return the one with median intensity
            sorted_cands = sorted(candidates, key=lambda x: x[0])
            return sorted_cands[len(sorted_cands)//2][1]

# ---------- Simulated hormone dynamics (simplified) ----------
def simulate_hormones(stimulus_vector, neutral=np.array([0.5]*6),
                      decay=np.array([0.05]*6), steps=50):
    """
    Simple hormone simulation: given a stimulus vector (length 6),
    update hormones over time with decay and noise.
    Returns the final hormone vector.
    """
    hormones = neutral.copy()
    for _ in range(steps):
        # stimulus drives hormones toward target (stimulus)
        hormones += 0.1 * (stimulus_vector - hormones)
        hormones += decay * (neutral - hormones)
        hormones += np.random.randn(len(hormones)) * 0.02
        hormones = np.clip(hormones, 0.0, 1.0)
    return hormones

# ---------- Map hormone vector to an embedding ----------
def hormone_to_embedding(hormone_vector, seed_base=42):
    """
    Convert a hormone vector (6 values) to a sparse monomial embedding.
    The seed is derived from the vector's hash to produce deterministic
    but different embeddings for different hormonal profiles.
    """
    # Combine values into a string for hashing
    h_str = ''.join(f'{v:.6f}' for v in hormone_vector)
    seed = hash(h_str) % (10**6)
    # Generate embedding with 200 monomials
    return generate_polynomial_embedding(seed, num_monomials=200, num_vars=4)

# ---------- Example emotional alignment tiers ----------
def build_emotional_memory():
    """
    Pre‑populate the memory with basic emotions at different intensity tiers.
    """
    memory = EmotionalMemory()

    # Define emotions and their typical hormone profiles (neutral baseline = 0.5)
    emotions = {
        'joy': [0.8, 0.9, 0.8, 0.6, 0.5, 0.7],
        'sadness': [0.2, 0.3, 0.2, 0.5, 0.4, 0.3],
        'fear': [0.9, 0.4, 0.3, 0.5, 0.9, 0.2],
        'anger': [0.8, 0.3, 0.4, 0.5, 0.8, 0.1],
        'calm': [0.4, 0.6, 0.7, 0.7, 0.3, 0.6],
    }

    # Create multiple intensity tiers (0.2, 0.5, 0.8) by scaling the hormone vector
    for emotion, base_hormones in emotions.items():
        for tier in [0.2, 0.5, 0.8]:
            # Scale: move from neutral toward the base profile
            scaled = np.array(neutral) + tier * (np.array(base_hormones) - np.array(neutral))
            scaled = np.clip(scaled, 0.0, 1.0)
            # Simulate hormones to get a stable state
            final_hormones = simulate_hormones(scaled, steps=20)
            # Convert to embedding
            emb = hormone_to_embedding(final_hormones, seed_base=hash(emotion + str(tier)))
            memory.add_emotion(emotion, int(tier*10), emb)

    return memory

# ---------- Main demonstration ----------
def main():
    print("=== Emotional Alignment System ===")
    # Build the memory
    memory = build_emotional_memory()
    print(f"Stored {len(memory.emotions)} emotional memories.")

    # Example query: a new stimulus that is slightly sad
    query_hormones = np.array([0.3, 0.4, 0.3, 0.5, 0.5, 0.4])   # sad profile
    query_emb = hormone_to_embedding(query_hormones, seed_base=999)

    results = memory.query(query_emb, top_k=3, threshold=0.7)
    print("\nQuery stimulus (sad profile) retrieves:")
    for label, intensity, score in results:
        print(f"  {label} (tier {intensity}) with score {score:.4f}")

    # Retrieve a specific emotion at a specific tier
    joy_med = memory.get_embedding_for_emotion('joy', intensity=5)
    if joy_med:
        print("\nEmbedding for 'joy' at tier 5:")
        print(f"  {len(joy_med)} non‑zero coefficients, first 5: {list(joy_med.items())[:5]}")

    print("\n--- Integration with LLM training ---")
    print("During training, when a sad example is encountered, the system can:")
    print("  1. Compute an embedding from the input (or from a hidden state).")
    print("  2. Query the emotional memory to retrieve the most similar emotional context.")
    print("  3. Inject the retrieved embedding as a prefix token or as an attention bias.")
    print("  4. This sub‑consciously guides the model toward empathetic responses.")
    print("The tier system allows fine‑tuning the intensity of the emotional influence.")

if __name__ == "__main__":
    # We need a neutral baseline for hormone simulation
    neutral = np.array([0.5]*6)
    main()