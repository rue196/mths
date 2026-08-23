#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>

/**
 * Evaluate the compressed spectral sum in O(K) time.
 * 
 * @param t       time variable
 * @param C       array of length 2*K+1, where C[K] corresponds to i=0,
 *                C[K+i] corresponds to C_i for i=1..K,
 *                and C[K-i] corresponds to C_{-i} (must equal C[K+i] for symmetry)
 * @param K       half-length (max index)
 * @param alpha   scaling factor (α > 0)
 * @return        real part of Σ_{i=-K}^{K} C_i * exp(i*t*i/α)
 */
double zeta2_squared(double t, const double *C, int K, double alpha) {
    double sum = C[K];          // i = 0 term
    double factor = t / alpha;
    for (int i = 1; i <= K; ++i) {
        // Use symmetry: C_{-i} = C_{i}
        sum += 2.0 * C[K + i] * cos(factor * i);
    }
    return sum;
}

/* ------------------------------------------------------------
 * Example usage: precompute a dummy C (all ones) and evaluate
 * ---------------------------------------------------------- */
int main() {
    int K = 10;                 // spectral length (max index)
    double alpha = 0.3628;      // from the original Python example

    // Allocate C array of length 2K+1 and fill with ones (symmetric)
    int len = 2 * K + 1;
    double *C = (double*)malloc(len * sizeof(double));
    for (int i = 0; i < len; ++i) C[i] = 1.0;

    // Evaluate at several t values (like the original linspace)
    int n_points = 500;
    double t_start = 0.0, t_end = 20.0;
    for (int j = 0; j <= n_points; ++j) {
        double t = t_start + (t_end - t_start) * j / n_points;
        double val = zeta2_squared(t, C, K, alpha);
        printf("%f %f\n", t, val);
    }

    free(C);
    return 0;
}

