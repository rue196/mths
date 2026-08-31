import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
import math
import hashlib
import struct
from typing import List, Tuple

# ---------- helper functions from the updated scheme ----------
def prng_bytes(seed: bytes, length: int) -> bytes:
    return hashlib.shake_256(seed).digest(length)

def generate_addresses(K: int, seed: bytes) -> List[Tuple[float, float]]:
    raw = prng_bytes(seed, K * 16)
    points = []
    for i in range(K):
        x = struct.unpack('>d', raw[16*i:16*i+8])[0]
        y = struct.unpack('>d', raw[16*i+8:16*i+16])[0]
        x = 0.5 + (x - math.floor(x)) * 4.5
        y = 0.5 + (y - math.floor(y)) * 4.5
        points.append((x, y))
    return points

def build_supermatrix(points: List[Tuple[float, float]], N: int) -> np.ndarray:
    M = np.zeros((N, N))
    for t in range(1, N+1):
        for j, (x, y) in enumerate(points[:N]):
            M[t-1, j] = (x ** t) + (y ** t)
    return M

def supertrace(M: np.ndarray) -> float:
    N = M.shape[0]
    trace = 0.0
    for t in range(N):
        val = M[t, t]
        if (t+1) % 2 == 0:   # even time (bosonic)
            trace += val
        else:                # odd time (fermionic)
            trace -= val
    return trace

def matrix_entropy(M: np.ndarray, alpha: float = 1.0/(math.pi - math.e)) -> float:
    S = abs(supertrace(M))
    if S == 0:
        return 0.0
    return -alpha * S * math.log(S) / M.shape[0]

# ---------- simulation ----------
def run_simulation(seed: int = 42, K: int = 8, animate: bool = False):
    """
    Visualise the supermatrix M and its supertrace.
    If animate=True, we rotate the spatial points over time to show
    the evolution of the matrix and the supertrace.
    """
    # Generate fixed points (or base points for animation)
    base_points = generate_addresses(K, seed.to_bytes(4, 'big'))
    
    if not animate:
        # Static plot
        M = build_supermatrix(base_points, K)
        st = supertrace(M)
        H = matrix_entropy(M)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Heatmap of full matrix
        im = axes[0].imshow(M, cmap='viridis', norm=Normalize(vmin=-np.abs(M).max(), vmax=np.abs(M).max()))
        axes[0].set_title('Matrix M (t vs j)')
        axes[0].set_xlabel('Particle index j')
        axes[0].set_ylabel('Time step t (1-based)')
        plt.colorbar(im, ax=axes[0])
        
        # Diagonal values with parity coloring
        diag = np.diag(M)
        t_vals = np.arange(1, K+1)
        colors = ['red' if t%2==1 else 'blue' for t in t_vals]  # odd=red (fermion), even=blue (boson)
        axes[1].bar(t_vals, diag, color=colors)
        axes[1].axhline(0, color='black', linestyle='--', linewidth=0.5)
        axes[1].set_title('Diagonal M_{t,t}')
        axes[1].set_xlabel('Time t')
        axes[1].set_ylabel('Value')
        axes[1].legend(['Fermion (odd t)', 'Boson (even t)'], 
                       handles=[plt.Rectangle((0,0),1,1, color='red'), 
                                plt.Rectangle((0,0),1,1, color='blue')])
        
        # SuperTrace and Entropy
        axes[2].text(0.1, 0.8, f'Supertrace STr(M) = {st:.4f}', fontsize=14)
        axes[2].text(0.1, 0.6, f'Entropy H = {H:.4f}', fontsize=14)
        axes[2].set_xlim(0,1)
        axes[2].set_ylim(0,1)
        axes[2].axis('off')
        axes[2].set_title('Supersymmetry summary')
        
        plt.tight_layout()
        plt.show()
    else:
        # Animation: rotate the points slightly each frame and recompute M
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        im = axes[0].imshow(np.zeros((K,K)), cmap='viridis', vmin=-10, vmax=10)
        axes[0].set_title('Matrix M (t vs j)')
        axes[0].set_xlabel('Particle index j')
        axes[0].set_ylabel('Time step t')
        plt.colorbar(im, ax=axes[0])
        
        bar_container = axes[1].bar(np.arange(1,K+1), np.zeros(K), color=['red' if t%2==1 else 'blue' for t in range(1,K+1)])
        axes[1].axhline(0, color='black', linestyle='--', linewidth=0.5)
        axes[1].set_title('Diagonal M_{t,t}')
        axes[1].set_xlabel('Time t')
        axes[1].set_ylabel('Value')
        
        text_box = axes[2].text(0.1, 0.5, '', fontsize=14, transform=axes[2].transAxes)
        axes[2].set_xlim(0,1)
        axes[2].set_ylim(0,1)
        axes[2].axis('off')
        axes[2].set_title('Supersymmetry summary')
        
        def update(frame):
            # rotate points by a small angle
            theta = frame * 0.1  # 0.1 rad per frame
            points = []
            for (x,y) in base_points:
                # rotate in 2D
                xr = x * math.cos(theta) - y * math.sin(theta)
                yr = x * math.sin(theta) + y * math.cos(theta)
                points.append((xr, yr))
            M = build_supermatrix(points, K)
            im.set_array(M)
            im.set_clim(vmin=-np.abs(M).max(), vmax=np.abs(M).max())
            
            diag = np.diag(M)
            for i, rect in enumerate(bar_container):
                rect.set_height(diag[i])
            # update y limits
            max_diag = max(abs(diag.max()), abs(diag.min()))
            axes[1].set_ylim(-max_diag*1.1, max_diag*1.1)
            
            st = supertrace(M)
            H = matrix_entropy(M)
            text_box.set_text(f'Supertrace STr = {st:.4f}\nEntropy H = {H:.4f}')
            return [im, *bar_container, text_box]
        
        ani = animation.FuncAnimation(fig, update, frames=60, interval=200, blit=True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Run static simulation
    run_simulation(seed=13, K=20, animate=True)
    # Uncomment the line below for animation (may take a moment)
    # run_simulation(seed=42, K=8, animate=True)