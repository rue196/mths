#!/usr/bin/env python3
"""
screen_derivative_update.py

Simulates a screen (1D signal) evolving under a finite‑difference derivative
with step a = 1/(π−e)/0.3628. The screen is compressed using the spectral
compression method (single‑sum, FFT‑based) after each update.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.fft import fft, ifft

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)          # ≈ 2.362
ALPHA_USER = 0.3628
A = ALPHA / ALPHA_USER          # ≈ 6.511 (finite difference step)

# ---------- Spectral compression functions ----------
def sort_by_angle(signal):
    """Treat signal indices as points in 2D with angle = index (simulate)."""
    # For simplicity, we use the index as a phase: angle = 2π * i / K
    K = len(signal)
    angles = 2 * np.pi * np.arange(K) / K
    idx = np.argsort(angles)
    return signal[idx], idx

def toeplitz_kernel(K, sigma=1.0):
    kernel = np.zeros(2*K-1, dtype=float)
    for d in range(-(K-1), K):
        kernel[d + (K-1)] = math.exp(-(d*d) / (2*sigma*sigma))
    return kernel

def apply_convolution(signal, kernel):
    K = len(signal)
    N = 1 << (2*K - 1).bit_length()
    sig_pad = np.pad(signal, (0, N - K), mode='constant')
    ker_pad = np.pad(kernel, (0, N - (2*K - 1)), mode='constant')
    return ifft(fft(sig_pad) * fft(ker_pad))[:K]

def compress_spectral(signal, M, sigma=1.0):
    """Compress signal by keeping M largest coefficients in convolved spectrum."""
    signal_sorted, sort_idx = sort_by_angle(signal)
    K = len(signal_sorted)
    kernel = toeplitz_kernel(K, sigma)
    conv = apply_convolution(signal_sorted, kernel)
    idx = np.argsort(np.abs(conv))[::-1][:M]
    kept = {int(i): conv[i] for i in idx}
    # Reconstruct for error (optional)
    recon = np.zeros(K, dtype=complex)
    for i, v in kept.items():
        recon[i] = v
    # Un‑sort to return to original order
    recon_unsorted = np.zeros(K, dtype=complex)
    recon_unsorted[sort_idx] = recon
    return kept, recon_unsorted

# ---------- Screen class ----------
class MobiusScreen:
    def __init__(self, K=256, M=64, sigma=2.0, dt=A):
        self.K = K                          # number of pixels (1D)
        self.M = M                          # number of kept coefficients
        self.sigma = sigma                  # kernel width
        self.dt = dt                        # time step (a)
        # Initialise signal: random smooth pattern
        np.random.seed(42)
        x = np.linspace(0, 4*np.pi, K)
        self.signal = 0.5 * np.sin(x) + 0.3 * np.cos(2*x) + 0.1 * np.random.randn(K)
        # Keep a copy for reference
        self.original = self.signal.copy()
        # Compression state
        self.compressed = {}
        self.recon = None

    def step(self):
        """Update signal using finite‑difference derivative with step dt."""
        # Compute derivative: dS[i] = (S[i+1] - S[i]) / dt
        diff = np.zeros(self.K)
        diff[:-1] = (self.signal[1:] - self.signal[:-1]) / self.dt
        # Apply diffusion: S = S + dt * derivative (Euler)
        self.signal += self.dt * diff
        # Optionally, apply a simple smoothing / boundary condition
        # For stability, we can clamp or renormalize
        self.signal = np.clip(self.signal, -2, 2)

    def compress(self):
        """Compress the current signal and store the result."""
        self.compressed, self.recon = compress_spectral(self.signal, self.M, self.sigma)

    def get_reconstructed(self):
        """Return the reconstructed signal (real part)."""
        if self.recon is None:
            self.compress()
        return np.real(self.recon)

    def get_signal(self):
        return self.signal

# ---------- Animation ----------
def animate_screen():
    K = 256
    M = 64
    screen = MobiusScreen(K=K, M=M, sigma=2.0, dt=A)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    ax1.set_title('Original signal (top) and reconstructed (bottom)')
    ax2.set_title('Reconstructed (compressed)')
    line1, = ax1.plot(screen.get_signal(), 'b-', label='Current')
    ax1.legend()
    line2, = ax2.plot(screen.get_reconstructed(), 'r-', label='Reconstructed')
    ax2.legend()

    # Display compression ratio
    ratio_text = ax1.text(0.02, 0.9, '', transform=ax1.transAxes)

    def update(frame):
        # Step and compress every 5 frames
        screen.step()
        if frame % 5 == 0:
            screen.compress()
            recon = screen.get_reconstructed()
            line2.set_ydata(recon)
            ratio = len(screen.compressed) / K
            ratio_text.set_text(f'Compression ratio: {ratio:.2f}')
        line1.set_ydata(screen.get_signal())
        return line1, line2, ratio_text

    ani = FuncAnimation(fig, update, frames=200, interval=50, blit=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    animate_screen()