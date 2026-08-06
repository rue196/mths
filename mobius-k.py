def mobius_sieve(K):
    """
    Compute Möbius function μ(n) for all n in [1, K].
    Returns a list mu of length K+1, where mu[0] is unused.
    Time: O(K), Memory: O(K).
    """
    if K < 1:
        return [0] * (K + 1)

    mu = [0] * (K + 1)
    mu[1] = 1
    primes = []
    is_comp = [False] * (K + 1)

    for i in range(2, K + 1):
        if not is_comp[i]:
            primes.append(i)
            mu[i] = -1   # prime has one distinct factor
        for p in primes:
            if i * p > K:
                break
            is_comp[i * p] = True
            if i % p == 0:
                mu[i * p] = 0   # p divides i, so i*p has p^2 factor
                break
            else:
                mu[i * p] = -mu[i]  # adds a new prime factor
    return mu

# Example usage:
K = 200
mu_values = mobius_sieve(K)
for n in range(1, K+1):
    print(f"μ({n}) = {mu_values[n]}")