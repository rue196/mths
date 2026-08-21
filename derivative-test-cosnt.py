import math
import numpy as np
import matplotlib.pyplot as plt

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)          # ≈ 2.362
ALPHA_USER = 0.3628
A = ALPHA / ALPHA_USER          # ≈ 6.511

# ---------- Random smooth function ----------
# We'll create a random combination of sin and cos with random frequencies and phases
np.random.seed(42)
freq1 = np.random.uniform(0.5, 3.0)
freq2 = np.random.uniform(1.0, 4.0)
phase1 = np.random.uniform(0, 2*math.pi)
phase2 = np.random.uniform(0, 2*math.pi)
amp1 = np.random.uniform(0.5, 2.0)
amp2 = np.random.uniform(0.3, 1.5)

def f(x):
    return amp1 * np.sin(freq1 * x + phase1) + amp2 * np.cos(freq2 * x + phase2)

# ---------- True derivative (symbolic) ----------
def df_true(x):
    return amp1 * freq1 * np.cos(freq1 * x + phase1) - amp2 * freq2 * np.sin(freq2 * x + phase2)

# ---------- Finite difference using step a ----------
def df_finite(x, step=A):
    return (f(x + step) - f(x)) / step

# ---------- Test at a random point ----------
x0 = np.random.uniform(0, 5)
deriv_true = df_true(x0)
deriv_approx = df_finite(x0)

print(f"x0 = {x0:.4f}")
print(f"a = {A:.6f}")
print(f"True derivative:  {deriv_true:.6f}")
print(f"Finite diff (step a): {deriv_approx:.6f}")
print(f"Error: {deriv_approx - deriv_true:.6e}")

# ---------- Plot the function and its derivative ----------
x_vals = np.linspace(-2, 8, 200)
y_vals = f(x_vals)
y_deriv_true = df_true(x_vals)
y_deriv_finite = [(f(x + A) - f(x)) / A for x in x_vals]

plt.figure(figsize=(10, 5))
plt.plot(x_vals, y_vals, label='f(x)', linewidth=2)
plt.plot(x_vals, y_deriv_true, '--', label='True derivative', linewidth=2)
plt.plot(x_vals, y_deriv_finite, ':', label=f'Finite diff (a={A:.3f})', linewidth=2)
plt.axvline(x0, color='k', linestyle=':', alpha=0.5, label=f'x0 = {x0:.2f}')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Random function and its derivative using step a = 1/(π−e)/0.3628')
plt.grid(True)
plt.show()