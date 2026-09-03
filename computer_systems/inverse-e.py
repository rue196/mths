import math

def supertrace_from_harmonic(K, C=1.0):
    """
    Compute the supertrace S = -C * H_K in O(K) time,
    where H_K = sum_{n=1}^K 1/n is the K-th harmonic number.
    """
    H = 0.0
    for n in range(1, K + 1):
        H += 1.0 / n
    return -C * H

if __name__ == "__main__":
    # Observed data: K -> S (rounded)
    observed = {
        10: -2.561441,
        50: -6.433116,
        100: -9.659349,
        200: -13.729237,
        500: -22.165148
    }

    # Estimate C from K=500: H_500 ≈ 6.792, S ≈ -22.165 => C ≈ 3.263
    C = 3.263

    print("K\t S (harmonic model)\t S (observed)")
    for K in sorted(observed.keys()):
        S_model = supertrace_from_harmonic(K, C)
        S_obs = observed[K]
        print(f"{K}\t {S_model:.6f}\t\t {S_obs:.6f}")