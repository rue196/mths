import math
import numpy as np
import matplotlib.pyplot as plt

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)          # ≈ 2.362
ALPHA_USER = 0.3628
A = ALPHA / ALPHA_USER          # ≈ 6.511 (finite derivative step)

def harmonic_oscillator_pi_e(K, compute_derivative=True):
    """
    O(K) computation of e and π approximations using Taylor and Leibniz series.
    
    Returns:
        e_approx   : final e approximation
        pi_approx  : final π approximation
        e_series   : list of partial sums for e (length K+1)
        pi_series  : list of partial sums for π (length K+1)
        error_pi   : list of π errors
        deriv_pi   : finite‑step derivative of π series (if compute_derivative)
    """
    # Initialise series
    e_partial = 1.0          # n=0 term: 1
    pi_partial = 0.0         # sum of (-1)^n / (2n+1)
    
    e_series = [e_partial]
    pi_series = [pi_partial]
    
    term_e = 1.0             # current term for e: x^n / n! with x=1
    sign = 1.0               # sign for pi series
    
    for n in range(1, K+1):
        # Update e: term_e = 1/n!  (since x=1)
        term_e /= n
        e_partial += term_e
        e_series.append(e_partial)
        
        # Update pi: add (-1)^(n-1) / (2n-1)? Actually for n starting at 1,
        # the term is (-1)^(n) / (2n+1) if we start at n=0.
        # We'll use a separate counter for the denominator.
        denom = 2 * n - 1
        pi_partial += sign / denom
        sign = -sign
        pi_series.append(pi_partial)
    
    # Convert pi series to actual π (multiply by 4)
    pi_series = [4 * p for p in pi_series]
    pi_approx = 4 * pi_partial
    e_approx = e_partial
    
    # Compute errors
    error_pi = [p - PI for p in pi_series]
    error_e = [e - E for e in e_series]
    
    # Compute finite‑step derivative of π series using step A
    if compute_derivative:
        deriv_pi = []
        for i in range(len(pi_series) - 1):
            # derivative = (π[i+1] - π[i]) / A
            deriv_pi.append((pi_series[i+1] - pi_series[i]) / A)
        # pad last with 0
        deriv_pi.append(0.0)
    else:
        deriv_pi = None
    
    return e_approx, pi_approx, e_series, pi_series, error_pi, deriv_pi

# ---------- Demonstration ----------
def demo():
    K = 120
    e_final, pi_final, e_series, pi_series, error_pi, deriv_pi = harmonic_oscillator_pi_e(K)
    
    print(f"Final e approximation (K={K}): {e_final:.15f}  (true: {E})")
    print(f"Error in e: {e_final - E:.2e}")
    print(f"Final π approximation (K={K}): {pi_final:.15f}  (true: {PI})")
    print(f"Error in π: {pi_final - PI:.2e}")
    
    # Plot the errors
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # e error (should be essentially zero after ~60)
    axes[0,0].plot(np.arange(len(e_series)), [e - E for e in e_series], 'b-')
    axes[0,0].set_title('Error in e approximation (Taylor)')
    axes[0,0].set_xlabel('n')
    axes[0,0].set_ylabel('error')
    axes[0,0].grid(True)
    
    # π error (oscillatory)
    axes[0,1].plot(np.arange(len(error_pi)), error_pi, 'r-')
    axes[0,1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[0,1].set_title('Error in π approximation (Leibniz)')
    axes[0,1].set_xlabel('n')
    axes[0,1].set_ylabel('error')
    axes[0,1].grid(True)
    
    # Envelope of π error (decaying 1/n)
    envelope = [1.0 / (2*n+1) for n in range(len(error_pi))]
    axes[0,1].plot(np.arange(len(error_pi)), envelope, 'g--', label='1/(2n+1)')
    axes[0,1].plot(np.arange(len(error_pi)), [-e for e in envelope], 'g--')
    axes[0,1].legend()
    
    # Finite‑step derivative of π series
    if deriv_pi:
        axes[1,0].plot(np.arange(len(deriv_pi)-1), deriv_pi[:-1], 'm-')
        axes[1,0].set_title('Finite‑step derivative of π series')
        axes[1,0].set_xlabel('n')
        axes[1,0].set_ylabel('dπ/dn (step = A)')
        axes[1,0].grid(True)
    
    # Ratio of π error to envelope (should oscillate)
    ratio = [error / (1.0/(2*n+1)) if 1.0/(2*n+1) != 0 else 0 for n, error in enumerate(error_pi)]
    axes[1,1].plot(np.arange(len(ratio)), ratio, 'c-')
    axes[1,1].axhline(y=0, color='k', linestyle='--')
    axes[1,1].axhline(y=1, color='g', linestyle='--', alpha=0.5)
    axes[1,1].axhline(y=-1, color='g', linestyle='--', alpha=0.5)
    axes[1,1].set_title('π error / (1/(2n+1))')
    axes[1,1].set_xlabel('n')
    axes[1,1].set_ylabel('ratio')
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    demo()