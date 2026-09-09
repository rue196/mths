import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)          # ≈ 2.362

# ---------- Merge-sort inversion counting (O(K log K)) ----------
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
    """Normalised inversion count (0..1). Higher means more inverse relationship."""
    pairs = sorted(zip(a, b), key=lambda x: x[0])
    b_sorted = [p[1] for p in pairs]
    inv = inversion_count(b_sorted)
    K = len(a)
    max_inv = K*(K-1)//2
    return inv / max_inv if max_inv > 0 else 0.0

# ---------- Supertrace entropy (abstract) ----------
def supertrace_entropy(hormone_vector):
    """
    We treat the hormone levels as "spectral coefficients" and compute
    an entropy-like scalar: S = Σ_even h_i - Σ_odd h_i, then H = -α*(|S|/N)*log(|S|/N)
    This gives a measure of disorder in the hormonal profile.
    """
    N = len(hormone_vector)
    S = 0.0
    for i, val in enumerate(hormone_vector):
        if (i+1) % 2 == 0:   # even index (1-based) -> boson-like (positive)
            S += val
        else:
            S -= val
    ratio = abs(S) / N if N > 0 else 0.0
    if 0.0 < ratio < 1.0:
        return -ALPHA * ratio * math.log(ratio)
    else:
        return 0.0

# ---------- Stimulus-to-hormone mapper (simplified neural net) ----------
class HormoneNet(nn.Module):
    def __init__(self, stimulus_dim=10, hormone_dim=6):
        super().__init__()
        self.fc = nn.Linear(stimulus_dim, hormone_dim)

    def forward(self, x):
        # x: (batch, stimulus_dim) -> hormone levels (batch, hormone_dim)
        return torch.sigmoid(self.fc(x))  # squashes to (0,1)

# ---------- Simulation parameters ----------
HORMONE_NAMES = ['Cortisol', 'Dopamine', 'Serotonin', 'Oxytocin', 'Adrenaline', 'Endorphins']
NEUTRAL = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])   # baseline
DECAY_RATES = np.array([0.05, 0.03, 0.04, 0.02, 0.06, 0.01])  # per time step
NOISE_STD = 0.02

# ---------- Stimulus generation (storyline) ----------
def generate_stimulus_sequence(T=200):
    """
    Create a sequence of stimulus vectors that evoke different emotions.
    Returns: list of (stimulus_vector, label)
    """
    stimuli = []
    labels = []
    for t in range(T):
        # Simulate events: joy, fear, calm, etc. by changing the stimulus distribution
        if t < 30:
            # Neutral
            s = np.random.randn(10) * 0.1
            label = 'neutral'
        elif 30 <= t < 60:
            # Positive event (dopamine/serotonin boost)
            s = np.random.randn(10) * 0.2
            s[0] += 1.0   # some feature for positivity
            label = 'joy'
        elif 60 <= t < 90:
            # Stressful event (cortisol/adrenaline spike)
            s = np.random.randn(10) * 0.2
            s[1] += 1.0
            label = 'stress'
        elif 90 <= t < 120:
            # Social bonding (oxytocin)
            s = np.random.randn(10) * 0.2
            s[2] += 1.0
            label = 'bonding'
        elif 120 <= t < 150:
            # Mixed / recovery
            s = np.random.randn(10) * 0.1
            label = 'recovery'
        else:
            # Neutral again
            s = np.random.randn(10) * 0.1
            label = 'neutral'
        stimuli.append(s)
        labels.append(label)
    return stimuli, labels

# ---------- Simulation loop ----------
def run_simulation(model, stimuli, neutral=NEUTRAL, decay=DECAY_RATES,
                   homeostatic_threshold=0.4, correction_strength=0.1):
    T = len(stimuli)
    hormone_history = []
    drift_history = []
    entropy_history = []

    # Initial hormone levels (neutral)
    hormones = neutral.copy()

    for t, s in enumerate(stimuli):
        # 1. Stimulus effect
        s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0)  # (1,10)
        with torch.no_grad():
            output = model(s_tensor).squeeze().numpy()   # (hormone_dim,)
        # 2. Add stimulus-driven change (the model outputs a delta? Actually we want new levels)
        # We'll interpret the model output as the target level, and we move toward it with some inertia.
        # Alternatively, we can treat the model output as the new level directly, but we add dynamics.
        # Here, we update hormones toward the model output with a learning rate.
        target = output
        # 3. Natural decay toward neutral (homeostasis without correction)
        hormones = hormones + decay * (neutral - hormones)   # decay toward neutral
        # 4. Move toward target (stimulus effect)
        hormones = hormones + 0.1 * (target - hormones)      # gradual adoption
        # 5. Add small noise
        hormones += np.random.randn(len(hormones)) * NOISE_STD
        # Clamp to [0,1]
        hormones = np.clip(hormones, 0.0, 1.0)

        # 6. Compute drift (inverse score between hormones and neutral)
        drift = inverse_score(hormones.tolist(), neutral.tolist())
        # 7. Compute entropy of current hormone vector
        ent = supertrace_entropy(hormones)

        # 8. Homeostatic correction: if drift > threshold, pull back toward neutral
        # The correction strength is modulated by entropy (to mimic plasticity)
        if drift > homeostatic_threshold:
            # correction = (drift - threshold) * correction_strength * (1 + entropy)  # entropy scales up correction when disorder is high
            scale = 1.0 + ent  # more entropy allows more correction
            correction = (drift - homeostatic_threshold) * correction_strength * scale
            # Apply correction: move hormones slightly toward neutral
            hormones = hormones + correction * (neutral - hormones)
            hormones = np.clip(hormones, 0.0, 1.0)

        # Store
        hormone_history.append(hormones.copy())
        drift_history.append(drift)
        entropy_history.append(ent)

    return np.array(hormone_history), np.array(drift_history), np.array(entropy_history)

# ---------- Main ----------
def main():
    # 1. Instantiate the hormone net (untrained, but we use its weights as fixed random mapping)
    model = HormoneNet(stimulus_dim=10, hormone_dim=6)
    # For demonstration, we fix the weights so that different stimuli produce consistent hormone patterns.
    # We'll set random weights but keep them fixed; the simulation will still produce meaningful dynamics.

    # 2. Generate stimulus sequence
    stimuli, labels = generate_stimulus_sequence(T=300)

    # 3. Run simulation
    hormone_history, drift_history, entropy_history = run_simulation(
        model, stimuli,
        neutral=NEUTRAL,
        decay=DECAY_RATES,
        homeostatic_threshold=0.45,
        correction_strength=0.15
    )

    # 4. Plot results
    time = np.arange(len(drift_history))
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # (a) Drift over time
    axes[0].plot(time, drift_history, color='purple', label='Emotional Drift (inverse score)')
    axes[0].axhline(y=0.45, color='red', linestyle='--', label='Homeostatic threshold')
    axes[0].set_ylabel('Drift')
    axes[0].set_title('Emotional arousal (deviation from neutral)')
    axes[0].legend()
    axes[0].grid(True)

    # (b) Entropy over time
    axes[1].plot(time, entropy_history, color='green', label='Entropy (disorder)')
    axes[1].set_ylabel('Entropy')
    axes[1].set_title('Hormonal entropy')
    axes[1].legend()
    axes[1].grid(True)

    # (c) Hormone levels (stacked)
    axes[2].stackplot(time, hormone_history[:, 0], hormone_history[:, 1],
                      hormone_history[:, 2], hormone_history[:, 3],
                      hormone_history[:, 4], hormone_history[:, 5],
                      labels=HORMONE_NAMES, alpha=0.7)
    axes[2].set_xlabel('Time step')
    axes[2].set_ylabel('Hormone level')
    axes[2].set_title('Hormone dynamics')
    axes[2].legend(loc='upper right', fontsize=8)
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

    # Print some statistics
    print(f"Average drift: {np.mean(drift_history):.3f}")
    print(f"Max drift: {np.max(drift_history):.3f}")
    print(f"Average entropy: {np.mean(entropy_history):.3f}")

if __name__ == "__main__":
    main()