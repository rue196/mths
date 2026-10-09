import numpy as np
import math
from itertools import permutations

PI  = math.pi
E   = math.e
ALPHA = 1.0 / (PI - E)          # ≈ 2.362
PI0   = 1.0                     # calibration constant for pigment/scattering scale

# ---------- Precompute (12,6) permutation table with signs ----------
def _build_perm_table(n=12, k=6):
    perms = np.array(list(permutations(range(n), k)), dtype=np.int16)
    signs = np.zeros(len(perms), dtype=np.float32)
    for i, p in enumerate(perms):
        inv = 0
        for a in range(k):
            for b in range(a + 1, k):
                if p[a] > p[b]:
                    inv += 1
        signs[i] = -1.0 if (inv & 1) else 1.0
    return perms, signs

PERMS, SIGNS = _build_perm_table(12, 6)   # 665,280 terms

def levi_civita_6(Z):
    """
    Rank-6 Levi-Civita contraction:
        Π₆(Z) = Σ_σ ε(σ) Π_k Z[σ(k), k]
    Z : (12, 6) complex array.
    """
    prods = np.ones(len(PERMS), dtype=np.complex128)
    for k in range(6):
        prods *= Z[PERMS[:, k], k]
    return np.sum(SIGNS * prods)

def chirality(Z):
    """Return (χ = arg Π₆, |Π₆|)."""
    Pi = levi_civita_6(Z)
    return float(np.angle(Pi)), float(abs(Pi))

def quantized_chirality(chi):
    """c(Z) ∈ {0,1} — see chiral.pdf Eq. (2)."""
    return 0 if math.cos(chi) >= 0 else 1

def chiral_bond_weight(chi_A, chi_B):
    """w(A,B) = |χ_A - χ_B|/π ∈ [0,1]  — electron-spin coupling."""
    d = abs(chi_A - chi_B)
    d = (d + PI) % (2 * PI) - PI       # wrap to (-π, π]
    return abs(d) / PI

def contraction_scalar(Z, pi0=PI0):
    """κ = log(1 + |Π₆| / Π₀)  — wavelength scattering scalar."""
    _, mag = chirality(Z)
    return math.log1p(mag / pi0)

def wavelength_from_contraction(kappa, lam_min=380.0, lam_max=750.0):
    """More contraction → higher wavelength (bounded sigmoid)."""
    return lam_min + (lam_max - lam_min) / (1.0 + math.exp(-kappa))

def supertrace(mags, signs):
    """S = Σ (-1)^{k+1} |Π₆(Z_k)|."""
    return sum(s * m for s, m in zip(signs, mags))

def entropy_S(S, N=12):
    p = abs(S) / N
    if p <= 0.0:
        return 0.0
    return -ALPHA * p * math.log(p)

def mass_gap(S, N=12):
    return abs(S) * math.exp(-entropy_S(S, N))

def build_Z_patch(img_rgb, x, y, radius=(1, 1)):
    """
    Build a (12, 6) complex matrix from a 12-pixel neighborhood.
    img_rgb : (H, W, 3) float
    Returns Z of shape (12, 6).
    """
    H, W, _ = img_rgb.shape
    ry, rx = radius
    patch = []
    for dy in range(-ry, ry + 1):
        for dx in range(-rx, rx + 2):        # 3 rows × 4 cols = 12 vertices
            yy = np.clip(y + dy, 0, H - 1)
            xx = np.clip(x + dx, 0, W - 1)
            patch.append(img_rgb[yy, xx])
    patch = np.array(patch)                  # (12, 3)

    # Complexify: each pixel → 6 complex coordinates
    Z = np.zeros((12, 6), dtype=np.complex128)
    for k in range(12):
        r, g, b = patch[k]
        # 1..3 : RGB as complex (real part)
        Z[k, 0] = r + 1j * 0.0
        Z[k, 1] = g + 1j * 0.0
        Z[k, 2] = b + 1j * 0.0
        # 4 : luminance
        Z[k, 3] = (0.2126 * r + 0.7152 * g + 0.0722 * b) + 1j * 0.0
        # 5,6 : local gradients (phase carriers)
        Z[k, 4] = complex(np.gradient(patch[:, 0])[k], 0.0)
        Z[k, 5] = complex(np.gradient(patch[:, 1])[k], 0.0)
    return Z
def ccd_correct_pixel(Z, raw_rgb):
    """
    Z       : (12, 6) complex neighborhood patch
    raw_rgb : (3,) raw channel values
    Returns corrected_rgb, metadata
    """
    # --- 1. Per-channel modulated Z ---
    Z_ch = [Z * raw_rgb[c] for c in range(3)]

    # --- 2. Chirality per channel ---
    chis, mags = [], []
    for Zc in Z_ch:
        chi, mag = chirality(Zc)
        chis.append(chi); mags.append(mag)

    # --- 3. Chiral bond weights (electron spin coupling) ---
    w_rg = chiral_bond_weight(chis[0], chis[1])
    w_gb = chiral_bond_weight(chis[1], chis[2])
    w_br = chiral_bond_weight(chis[2], chis[0])
    W = np.array([(w_rg + w_br) / 2.0,      # R coupling
                  (w_rg + w_gb) / 2.0,      # G coupling
                  (w_gb + w_br) / 2.0])     # B coupling

    # --- 4. Contraction scalar → wavelength per channel ---
    kappas   = np.array([math.log1p(m / PI0) for m in mags])
    lambdas  = np.array([wavelength_from_contraction(k) for k in kappas])

    # --- 5. Supertrace across channels ---
    signs = np.array([+1.0, -1.0, +1.0])    # R(+), G(−), B(+)
    S = supertrace(mags, signs)
    H = entropy_S(S, N=12)
    m = mass_gap(S, N=12)

    # --- 6. Coupling gain (spin-weighted supertrace correction) ---
    gains = np.exp(-W * abs(S) * ALPHA)

    corrected = np.clip(raw_rgb * gains, 0.0, 1.0)

    meta = dict(chis=chis, mags=mags, W=W, kappas=kappas,
                lambdas=lambdas, S=S, H=H, m=m, gains=gains)
    return corrected, meta
def ccd_correct_image(img_rgb):
    H, W, _ = img_rgb.shape
    out   = np.zeros_like(img_rgb)
    meta_map = {
        "S": np.zeros((H, W)),
        "H": np.zeros((H, W)),
        "m": np.zeros((H, W)),
        "W": np.zeros((H, W, 3)),
        "lam": np.zeros((H, W, 3)),
    }
    for y in range(H):
        for x in range(W):
            Z = build_Z_patch(img_rgb, x, y)
            corr, meta = ccd_correct_pixel(Z, img_rgb[y, x])
            out[y, x] = corr
            meta_map["S"][y, x]   = meta["S"]
            meta_map["H"][y, x]   = meta["H"]
            meta_map["m"][y, x]   = meta["m"]
            meta_map["W"][y, x]   = meta["W"]
            meta_map["lam"][y, x] = meta["lambdas"]
    return out, meta_map

    # img_rgb : (H, W, 3) float in [0,1]
corrected, meta = ccd_correct_image(img_rgb)

# Supertrace image — the "color bias field"
S_field = meta["S"]           # alternating-sum of channel contractions

# Wavelength field per channel (nm)
lam_R = meta["lam"][..., 0]
lam_G = meta["lam"][..., 1]
lam_B = meta["lam"][..., 2]

# Spin coupling magnitude (0..1) — how strongly chiral mismatch affects gain
W_mag = np.linalg.norm(meta["W"], axis=-1)