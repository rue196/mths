import numpy as np
import matplotlib.pyplot as plt
import time
from itertools import combinations

# -------------------------------------------------------------------
# 1. Inversion count (O(K log K)) via merge sort
# -------------------------------------------------------------------
def merge_and_count(arr, temp_arr, left, mid, right):
    """
    Merge two sorted halves and count inversions.
    """
    i = left    # Starting index of left subarray
    j = mid + 1 # Starting index of right subarray
    k = left    # Starting index to be sorted
    inv_count = 0

    while i <= mid and j <= right:
        if arr[i] <= arr[j]:
            temp_arr[k] = arr[i]
            i += 1
        else:
            # There are mid - i + 1 inversions because all elements
            # from i to mid are greater than arr[j]
            temp_arr[k] = arr[j]
            inv_count += (mid - i + 1)
            j += 1
        k += 1

    # Copy remaining elements of left subarray
    while i <= mid:
        temp_arr[k] = arr[i]
        i += 1
        k += 1

    # Copy remaining elements of right subarray
    while j <= right:
        temp_arr[k] = arr[j]
        j += 1
        k += 1

    # Copy sorted subarray into original array
    for i in range(left, right + 1):
        arr[i] = temp_arr[i]

    return inv_count

def inversion_count(arr):
    """
    Returns the number of inversions in array 'arr' (list or numpy array).
    """
    n = len(arr)
    temp_arr = [0] * n
    return _merge_sort(arr, temp_arr, 0, n - 1)

def _merge_sort(arr, temp_arr, left, right):
    inv_count = 0
    if left < right:
        mid = (left + right) // 2
        inv_count += _merge_sort(arr, temp_arr, left, mid)
        inv_count += _merge_sort(arr, temp_arr, mid + 1, right)
        inv_count += merge_and_count(arr, temp_arr, left, mid, right)
    return inv_count

# -------------------------------------------------------------------
# 2. Compute inverse relationship between two sequences
# -------------------------------------------------------------------
def inverse_score(a, b):
    """
    Given two sequences a and b (same length), compute a scalar
    that measures the inverse relationship.
    We compute the number of inversions between the ordering of a and b.
    High inversion count indicates strong negative association.
    """
    # Pair up (a_i, b_i) and sort by a
    pairs = sorted(zip(a, b), key=lambda x: x[0])
    # Extract b in the order of sorted a
    b_sorted = [p[1] for p in pairs]
    # Count inversions in b_sorted
    inv = inversion_count(b_sorted)
    # Normalize: max inversions = K*(K-1)/2, we return ratio in [0,1]
    K = len(a)
    max_inv = K * (K - 1) // 2
    if max_inv == 0:
        return 0.0
    return inv / max_inv   # 1 = perfect inverse, 0 = perfect forward

# -------------------------------------------------------------------
# 3. Data analysis class: compute inverse pattern matrix
# -------------------------------------------------------------------
class InversePatternAnalyzer:
    """
    Analyzes inverse relationships between features (columns) of a data matrix.
    The analysis runs in O(D * K log K) for each feature pair if done pairwise,
    but we only compute against a target by default (O(D * K log K)).
    """
    def __init__(self, data, target=None):
        """
        data: 2D numpy array (K samples, D features)
        target: 1D array (K samples) or None. If None, the last column is used.
        """
        self.data = np.asarray(data)
        self.K, self.D = self.data.shape
        if target is None:
            self.target = self.data[:, -1]   # use last column as target
            self.feature_cols = list(range(self.D - 1))
        else:
            self.target = np.asarray(target)
            self.feature_cols = list(range(self.D))
        assert len(self.target) == self.K, "Target must have same length as samples"

    def compute_inverse_scores(self, threshold=None):
        """
        For each feature, compute inverse score with the target.
        Returns a dict {feature_index: score}.
        If threshold is given, only features with score >= threshold are returned.
        """
        scores = {}
        for j in self.feature_cols:
            score = inverse_score(self.data[:, j], self.target)
            scores[j] = score
        if threshold is not None:
            scores = {j: s for j, s in scores.items() if s >= threshold}
        return scores

    def compute_pairwise_matrix(self, features=None):
        """
        Compute inverse scores for all pairs of features (or a subset).
        Returns a D' x D' matrix (symmetric) where entry (i,j) is the score.
        WARNING: O(m^2 * K log K) where m = number of features.
        """
        if features is None:
            features = list(range(self.D))
        m = len(features)
        score_mat = np.zeros((m, m))
        for idx1, i in enumerate(features):
            for idx2, j in enumerate(features):
                if idx1 >= idx2:
                    continue
                s = inverse_score(self.data[:, i], self.data[:, j])
                score_mat[idx1, idx2] = s
                score_mat[idx2, idx1] = s
        return score_mat

# -------------------------------------------------------------------
# 4. Example: synthetic data and analysis
# -------------------------------------------------------------------
def main():
    # Generate synthetic dataset: 100 samples, 10 features
    np.random.seed(44)
    K = 30
    D = 8
    # Create some features with inverse relationships
    X = np.random.randn(K, D)
    # Feature 0 is positively correlated with feature 1
    X[:, 1] = 0.8 * X[:, 0] + 0.2 * np.random.randn(K)
    # Feature 2 is negatively correlated with feature 3
    X[:, 3] = -0.7 * X[:, 2] + 0.3 * np.random.randn(K)
    # Feature 4 is unrelated
    X[:, 4] = np.random.randn(K)
    # Target is a linear combination with inverse pattern to feature 5
    X[:, 5] = 0.5 * np.random.randn(K)  # noise
    target = -0.9 * X[:, 5] + 0.1 * np.random.randn(K)  # inverse with feature 5

    # Analyze
    analyzer = InversePatternAnalyzer(X, target)

    # Compute scores for each feature (except target if it was part of X)
    scores = analyzer.compute_inverse_scores()
    print("Inverse scores with target (higher = more inverse):")
    for j, s in scores.items():
        print(f"Feature {j}: {s:.3f}")

    # Apply a threshold (tunable scalar) to find strong inverse patterns
    threshold = 0.3   # tune this scalar for your data
    strong = analyzer.compute_inverse_scores(threshold=threshold)
    print(f"\nFeatures with inverse score >= {threshold}: {list(strong.keys())}")

    # Visualize the scores
    plt.figure(figsize=(8, 4))
    plt.bar(list(scores.keys()), list(scores.values()), color='skyblue')
    plt.axhline(y=threshold, color='red', linestyle='--', label=f'threshold={threshold}')
    plt.xlabel('Feature index')
    plt.ylabel('Inverse score')
    plt.title('Inverse relationship between each feature and target')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # Optional: compute pairwise inverse matrix (demonstration)
    print("\nComputing pairwise inverse scores (may take a few seconds)...")
    start = time.time()
    pairwise = analyzer.compute_pairwise_matrix(features=[0,1,2,3,4,5])
    elapsed = time.time() - start
    print(f"Pairwise computation took {elapsed:.2f} seconds (O(m^2 K log K)).")

    # Heatmap
    plt.figure(figsize=(6, 5))
    plt.imshow(pairwise, cmap='coolwarm', interpolation='nearest', vmin=0, vmax=1)
    plt.colorbar(label='Inverse score')
    plt.title('Pairwise inverse relationship matrix')
    plt.xlabel('Feature index')
    plt.ylabel('Feature index')
    plt.xticks(range(len([0,1,2,3,4,5])), [0,1,2,3,4,5])
    plt.yticks(range(len([0,1,2,3,4,5])), [0,1,2,3,4,5])
    plt.show()

if __name__ == "__main__":
    main()