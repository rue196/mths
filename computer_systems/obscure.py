import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import permutations
import math
import hashlib

# ---------- Spinor projection functions ----------
def epsilon_tensor():
    eps = {}
    for perm in permutations(range(6)):
        if len(set(perm)) != 6:
            eps[perm] = 0
            continue
        inv = sum(1 for i in range(6) for j in range(i+1,6) if perm[i] > perm[j])
        eps[perm] = (-1)**inv
    return eps

EPS = epsilon_tensor()

def spinor_projection(vertices):
    """Compute Π = ε_{i1...i6} ∏_{k=1}^6 v_{k, i_k} using first 6 vertices."""
    v = vertices[:6]  # take first 6 vertices (12 total, but we only need 6 for contraction)
    scalar = 0.0
    for perm, sign in EPS.items():
        if sign == 0:
            continue
        prod = 1.0
        for idx, coord_idx in enumerate(perm):
            prod *= v[idx][coord_idx]
        scalar += sign * prod
    return scalar

def rotate_vertices(vertices, theta):
    rotated = vertices.copy()
    for v in rotated:
        for pair in [(0,1), (2,3), (4,5)]:
            x, y = v[pair[0]], v[pair[1]]
            v[pair[0]] = x * math.cos(theta) - y * math.sin(theta)
            v[pair[1]] = x * math.sin(theta) + y * math.cos(theta)
    return rotated

def projection_features(vertices, num_theta=256):
    eps = EPS
    theta_vals = np.linspace(0, 2*np.pi, num_theta)
    proj = []
    for theta in theta_vals:
        rot = rotate_vertices(vertices, theta)
        proj.append(spinor_projection(rot))
    proj = np.array(proj)
    var = np.var(proj)
    mean = np.mean(proj)
    return var, mean

# ---------- Encryption / Decryption ----------
def embed_image_rgb6(image, seed=42):
    """
    Convert image to 6D points: for each pixel, use RGB as first three coordinates,
    and add three random coordinates (obscuring).
    Returns: 6D points, and the random seed.
    """
    np.random.seed(seed)
    h, w, _ = image.shape
    # Normalize RGB to [0,1]
    rgb = image.astype(np.float32) / 255.0
    # Create 6D points: [R, G, B, noise1, noise2, noise3]
    points = np.zeros((h, w, 6))
    points[:,:,:3] = rgb
    # Add random noise (obscuring) scaled by 0.5 to keep within reasonable range
    points[:,:,3:] = np.random.randn(h, w, 3) * 0.5
    return points, seed

def encrypt_image(points, key_seed=42):
    """
    Encrypt the 6D points by rotating each vertex? Actually we need to obscure.
    We can compute a global rotation (spinor rotation) and apply to each point.
    The key is the seed for the rotation angles.
    """
    np.random.seed(key_seed)
    # Generate a set of 12 vertices for the spinor projection? Not needed.
    # We'll just apply a random orthogonal transformation to the 6D space.
    # Generate a random 6x6 orthogonal matrix.
    Q, _ = np.linalg.qr(np.random.randn(6, 6))
    # Apply to each point
    h, w, _ = points.shape
    points_flat = points.reshape(-1, 6)
    points_transformed = points_flat @ Q.T
    points_encrypted = points_transformed.reshape(h, w, 6)
    return points_encrypted, Q

def decrypt_image(points_encrypted, Q):
    """Apply inverse rotation."""
    h, w, _ = points_encrypted.shape
    points_flat = points_encrypted.reshape(-1, 6)
    points_dec = points_flat @ Q  # Q is orthogonal, inverse is transpose
    return points_dec.reshape(h, w, 6)

def compute_variance_projection(image_points):
    """
    For each pixel, compute the spinor projection variance (or just projection)
    to generate a scalar that can be used as brightness.
    We'll compute the projection for each pixel (using its 6 coordinates as a vertex? 
    But we need 6 vertices to compute the contraction. So we need to group pixels.
    Instead, we can treat each pixel as a vertex and compute the contraction over a sliding window of 6 pixels.
    That's too slow. We'll use a simpler approach: compute the variance of the coordinates across the image.
    Or we can compute the projection of the entire set of points (treat all pixels as 12 vertices? No.)
    Let's use a simpler method: compute the principal components of the 6D points, and use the variance of the projection onto the first PC.
    """
    # Flatten points
    h, w, d = image_points.shape
    points_flat = image_points.reshape(-1, d)
    # Compute covariance
    cov = np.cov(points_flat.T)
    # Compute variance of projection onto first principal component
    eigvals, eigvecs = np.linalg.eig(cov)
    # Projection variance = largest eigenvalue
    var = np.max(eigvals.real)
    return var

def display_3d_compressed(image_points, original_rgb):
    """
    Display a 3D scatter plot where each pixel is a point in 3D (first 3 components)
    and color is the original RGB (or transformed).
    """
    h, w, _ = image_points.shape
    # Take first 3 coordinates as 3D position
    points_3d = image_points[:,:,:3].reshape(-1, 3)
    # Original RGB (flatten)
    colors = original_rgb.reshape(-1, 3) / 255.0
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d[:,0], points_3d[:,1], points_3d[:,2], c=colors, s=1, alpha=0.6)
    ax.set_title("3D Compressed (Symmetric) - Human view")
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    plt.show()

def display_obscured_image(points_encrypted):
    """
    Display the encrypted 6D data as a 2D image using the first three coordinates as RGB.
    This shows the obscured image that a computer would read.
    """
    h, w, _ = points_encrypted.shape
    # Take first 3 coords as RGB, clip to [0,1]
    rgb_enc = points_encrypted[:,:,:3]
    # Normalize to [0,1] globally
    rgb_enc = (rgb_enc - rgb_enc.min()) / (rgb_enc.max() - rgb_enc.min() + 1e-12)
    plt.figure(figsize=(6,6))
    plt.imshow(rgb_enc)
    plt.title("Obscured Image (Anti-symmetric) - Computer view")
    plt.axis('off')
    plt.show()

# ---------- Main ----------
def main():
    # Load an image (example: create a simple shape)
    # We'll create a synthetic image with a circle for demonstration.
    size = 64
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    mask = R < 0.5
    # Create RGB image: red circle on black background
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[mask, 0] = 255  # red channel
    # Also add some gradient for better visualization
    image[mask, 1] = (R[mask] * 255).astype(np.uint8)  # green gradient
    image[mask, 2] = (0.5 - R[mask]) * 255 * 2   # blue gradient
    image = image.astype(np.uint8)

    plt.imshow(image)
    plt.title("Original Image")
    plt.show()

    # 1. Embed into 6D
    points, seed = embed_image_rgb6(image, seed=42)
    # 2. Encrypt (random orthogonal rotation)
    points_encrypted, Q = encrypt_image(points, key_seed=123)

    # 3. Display compressed 3D (human view) - using original points (before encryption) for clarity
    # Actually we want to show the encrypted points in 3D? But the human sees the original?
    # We'll show both: original points (symmetric) and encrypted points (anti-symmetric).
    print("Displaying 3D compressed view of original (symmetric):")
    display_3d_compressed(points, image)

    print("Displaying 3D compressed view of encrypted (anti-symmetric):")
    display_3d_compressed(points_encrypted, image)

    # 4. Display obscured image (computer view)
    display_obscured_image(points_encrypted)

    # 5. Decrypt and recover
    points_dec = decrypt_image(points_encrypted, Q)
    # Recover RGB from first 3 coords
    rgb_dec = points_dec[:,:,:3]
    rgb_dec = np.clip(rgb_dec, 0, 1)
    rgb_dec = (rgb_dec * 255).astype(np.uint8)
    plt.imshow(rgb_dec)
    plt.title("Decrypted Image")
    plt.show()

if __name__ == "__main__":
    main()