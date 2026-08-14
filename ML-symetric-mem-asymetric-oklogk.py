import numpy as np
import matplotlib.pyplot as plt
from ML import inverse_score  # import the O(K log K) inversion count scorer

# ---------- Simulation parameters ----------
K = 50                 # length of weight and inquiry vectors
steps = 200
lr = 0.01              # learning rate for prediction error
eta = 0.05             # blending factor for alignment
threshold = 0.4        # inverse score above which we align

# ---------- Generate data ----------
np.random.seed(42)
# Symmetric weights: we enforce symmetry by taking a random vector and making it symmetric
# but for simplicity we just use a random vector, and we'll treat it as "symmetric" by design
weights = np.random.randn(K) * 0.5
# Asymmetric inquiry: random with some structure
inquiry = np.random.randn(K) + 0.5 * np.sin(np.arange(K))
# Target outcome: dot product plus noise
target = np.dot(weights, inquiry) + 0.1 * np.random.randn()

# ---------- Storage ----------
pred_errors = []
inv_scores = []
weight_drift = []

# ---------- Iterative adjustment ----------
for step in range(steps):
    # 1. Predict
    pred = np.dot(weights, inquiry)
    error = pred - target
    pred_errors.append(error**2)

    # 2. Compute inverse score between weights and inquiry
    # (we treat both as sequences of equal length)
    inv_score = inverse_score(weights.tolist(), inquiry.tolist())
    inv_scores.append(inv_score)

    # 3. Gradient update for prediction error: w -= lr * 2 * error * inquiry
    grad = 2 * error * inquiry
    weights = weights - lr * grad

    # 4. Heuristic alignment: if inverse score is high, blend weights toward inquiry
    if inv_score > threshold:
        weights = (1 - eta) * weights + eta * inquiry

    # Optional: keep weights bounded
    weights = np.clip(weights, -2, 2)

    # Store weight drift (norm of change)
    if step > 0:
        weight_drift.append(np.linalg.norm(weights - prev_weights))
    prev_weights = weights.copy()

# ---------- Plot results ----------
fig, axes = plt.subplots(3, 1, figsize=(10, 10))

axes[0].plot(pred_errors)
axes[0].set_title('Prediction error (MSE)')
axes[0].set_xlabel('Step')
axes[0].set_ylabel('Error²')
axes[0].grid(True)

axes[1].plot(inv_scores)
axes[1].axhline(y=threshold, color='r', linestyle='--', label='threshold')
axes[1].set_title('Inverse score (weights vs inquiry)')
axes[1].set_xlabel('Step')
axes[1].set_ylabel('Inverse score')
axes[1].legend()
axes[1].grid(True)

axes[2].plot(weight_drift)
axes[2].set_title('Weight drift (||Δw||)')
axes[2].set_xlabel('Step')
axes[2].set_ylabel('Drift')
axes[2].grid(True)

plt.tight_layout()
plt.show()

# ---------- Final summary ----------
print(f"Final prediction error: {pred_errors[-1]:.6f}")
print(f"Final inverse score: {inv_scores[-1]:.4f}")
print(f"Final weights (first 5): {weights[:5]}")
print(f"Final inquiry (first 5): {inquiry[:5]}")