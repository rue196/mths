import numpy as np
from itertools import permutations
from math import exp, log, pi, e

N = 12
ALPHA = 1.0 / (pi - e)
PI0 = 1.0  # calibrate with reference pigments
ETA = 0.5
ZETA = 0.1

def supertrace(M):
    return sum(((-1) ** (i + 1)) * M[i, i] for i in range(N))

def entropy(S):
    x = abs(S) / N
    if x <= 0:
        return 0.0
    return -ALPHA * x * log(x)

def perm_sign(seq):
    # sign of ordered distinct indices relative to sorted order
    sorted_seq = sorted(seq)
    index = {v: i for i, v in enumerate(sorted_seq)}
    p = [index[v] for v in seq]
    inv = sum(1 for i in range(len(p)) for j in range(i+1, len(p)) if p[i] > p[j])
    return -1 if inv % 2 else 1

def scalar_perm_12_6(M):
    C = 0.0
    for perm in permutations(range(N), 6):
        sign = perm_sign(perm)
        prod = 1.0
        for k, row in enumerate(perm):
            prod *= M[row, k]
        C += sign * prod
    return C

def wavelength_from_kappa(kappa):
    return 380 + 370 / (1 + np.exp(-kappa))

def score(M, n_conj=6):
    S = supertrace(M)
    H = entropy(S)
    m = abs(S) * exp(-H)
    Pi = scalar_perm_12_6(M)
    kappa = log(1 + abs(Pi) / PI0) + ETA * m + ZETA * n_conj
    lam = wavelength_from_kappa(kappa)
    return {
        "S": S,
        "H": H,
        "m_strong": m,
        "Pi": Pi,
        "kappa": kappa,
        "lambda_peak": lam,
    }