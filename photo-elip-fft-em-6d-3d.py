import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft
import cmath
import math
from chip_g import ChipProcessor  # assumes chip-g.py is in the same directory

# ---------- Constants ----------
PI = math.pi
E = math.e
ALPHA = 1.0 / (PI - E)          # ≈ 2.362

# ---------- Elliptic projection weight (bounded [0,1]) ----------
def elliptic_projection_weight(z, alpha=ALPHA):
    """
    Compute a bounded weight from the elliptic projection operator.
    Here we use a simple periodic function on the torus: w = 0.5*(1 + cos(2*pi*real(z)/pi) * cos(2*pi*imag(z)/e)).
    This mimics the spinor projection on the elliptic curve with periods pi and e.
    """
    x = np.real(z)
    y = np.imag(z)
    # Normalize by periods pi and e
    x_norm = x / PI
    y_norm = y / E
    weight = 0.5 * (1 + np.cos(2 * np.pi * x_norm) * np.cos(2 * np.pi * y_norm))
    # Ensure in [0,1]
    return np.clip(weight, 0.0, 1.0)

# ---------- Image generation (unchanged) ----------
def generate_matrix_image(size=128, n=1, scale=2.0):
    x = np.linspace(-scale, scale, size)
    y = np.linspace(-scale, scale, size)
    X, Y = np.meshgrid(x, y)
    if n == 1:
        img = np.ones((size, size)) * 2.0
    else:
        img = np.abs(X)**(n-1) + np.abs(Y)**(n-1)
    img = img - img.min()
    img = img / img.max()
    return img

def load_image(n=1, size=128):
    return generate_matrix_image(size, n)

# ---------- Flatten and sort by angle ----------
def flatten_image(img):
    H, W = img.shape
    K = H * W
    coords = np.array([(i, j) for i in range(H) for j in range(W)])
    z = coords[:,0] + 1j * coords[:,1]
    intensities = img.flatten()
    return z, intensities, (H, W)

def sort_by_angle(z, intensities):
    angles = np.angle(z)
    idx = np.argsort(angles)
    z_sorted = z[idx]
    intensities_sorted = intensities[idx]
    return z_sorted, intensities_sorted, idx

# ---------- Updated compression using chip pipeline ----------
def compress_image(img, sigma=2.0, use_elliptic=False, processor=None):
    """
    Compress the image using the chip pipeline.
    If processor is None, a new ChipProcessor is created.
    The elliptic projection scales the intensities before compression.
    """
    H, W = img.shape
    z, intensities, shape = flatten_image(img)
    K = len(intensities)

    # Optional elliptic projection: scale intensities by weight
    if use_elliptic:
        weights = elliptic_projection_weight(z)
        intensities = intensities * weights

    # Sort by angle (TSP routing)
    z_sorted, intensities_sorted, sort_idx = sort_by_angle(z, intensities)

    # Create or reuse processor
    if processor is None:
        processor = ChipProcessor(max_K=K)
    else:
        if processor.max_K < K:
            raise ValueError(f"Processor max_K ({processor.max_K}) < signal length {K}")

    # Run the chip pipeline on the sorted intensities
    kept, S, H_ent, m, conv = processor.process(intensities_sorted)

    # Metadata for decompression
    meta = {
        'shape': shape,
        'sort_idx': sort_idx,
        'K': K,
        'S': S,
        'H_ent': H_ent,
        'm': m,
        'use_elliptic': use_elliptic,
        'processor_max_K': processor.max_K
    }
    return kept, meta

# ---------- Decompress: reconstruct from kept coefficients ----------
def decompress_image(kept, meta):
    K = meta['K']
    shape = meta['shape']
    sort_idx = meta['sort_idx']

    # Reconstruct the sorted signal (length K) from kept (list of (idx, value))
    recon_sorted = np.zeros(K, dtype=complex)
    for idx, val in kept:
        recon_sorted[idx] = val

    # Unsorted to original order
    recon_flat = np.zeros(K, dtype=complex)
    recon_flat[sort_idx] = recon_sorted
    recon_img = recon_flat.reshape(shape).real

    # Clamp to [0,1] (since the chip pipeline may produce values outside this range)
    recon_img = np.clip(recon_img, 0, 1)
    return recon_img

# ---------- Demonstration ----------
if __name__ == "__main__":
    # Create a reusable ChipProcessor (buffers allocated once)
    proc = ChipProcessor(max_K=200000)  # enough for 128x128 = 16384 pixels

    for n in [1, 5, 15]:
        print(f"\n--- Exponent n = {n} ---")
        img = load_image(n, size=128)
        plt.imshow(img, cmap='gray')
        plt.title(f"Original (n={n})")
        plt.show()

        # Compress with elliptic projection enabled
        kept, meta = compress_image(img, use_elliptic=True, processor=proc)

        print(f"Original pixels: {meta['K']}")
        print(f"Compressed coefficients: {len(kept)} (ratio: {len(kept)/meta['K']:.2%})")
        print(f"Supertrace S = {meta['S']:.4f}, Entropy H = {meta['H_ent']:.4f}, Mass m = {meta['m']:.4f}")

        recon = decompress_image(kept, meta)

        plt.imshow(recon, cmap='gray')
        plt.title(f"Reconstructed (n={n}, kept={len(kept)})")
        plt.show()

        mse = np.mean((img - recon)**2)
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        print(f"PSNR: {psnr:.2f} dB")