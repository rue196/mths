/*
 * chip_impl.c
 *
 * Möbius chip pipeline – C implementation.
 * 
 * Stages:
 *   1. Angle sort (TSP routing) via bucket sort.
 *   2. Exponential convolution (two‑pass).
 *   3. Supertrace, entropy, mass.
 *   4. Magnitude selection (top M = floor(|S|) coefficients).
 *   5. Möbius filter (keep only square‑free indices).
 *
 * Compile: gcc -std=c99 -O2 -lm -o chip_impl chip_impl.c
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ---------- Constants ---------- */
#define PI          3.14159265358979323846
#define E           2.71828182845904523536
#define ALPHA       (1.0 / (PI - E))          /* ≈ 2.362 */
#define NORM        (1.0 - exp(-ALPHA * (PI + E)))

#define MAX_K       256
#define BUCKETS     360

/* ---------- Packed Möbius sieve ---------- */
typedef struct {
    uint32_t K;           /* length-1 of mu array */
    uint32_t abs_sum;     /* Σ|μ(i)| for i=1..K (checksum) */
    uint8_t *bits;        /* packed bits: 2 bits per value */
} MobiusSieve;

/* ---------- Chip processor state ---------- */
typedef struct {
    int K;                /* current signal length */
    double *signal;       /* original signal (size K) */
    double *sorted;       /* reordered after TSP */
    double *conv;         /* convolution result */
    double *mag;          /* magnitudes for sorting */
    int *order;           /* TSP order (size K) */
    int *kept_idx;        /* compressed indices (size K) */
    double *kept_val;     /* compressed values (size K) */
    int kept_count;       /* number of kept coefficients */
    double S, H, m;       /* invariants */
    MobiusSieve mu;       /* packed Möbius sieve */
} ChipProcessor;

/* ---------- Utility functions ---------- */
static double min_d(double a, double b) { return a < b ? a : b; }
static double max_d(double a, double b) { return a > b ? a : b; }

/* ---------- Möbius sieve (linear, O(K)) ---------- */
void mobius_sieve(int K, int *mu) {
    if (K < 1) return;
    mu[0] = 0;
    mu[1] = 1;
    int *primes = (int*)malloc((K+1) * sizeof(int));
    int *is_comp = (int*)calloc(K+1, sizeof(int));
    int prime_count = 0;
    for (int i = 2; i <= K; i++) {
        if (!is_comp[i]) {
            primes[prime_count++] = i;
            mu[i] = -1;
        }
        for (int j = 0; j < prime_count; j++) {
            int p = primes[j];
            long long prod = 1LL * i * p;
            if (prod > K) break;
            is_comp[prod] = 1;
            if (i % p == 0) {
                mu[prod] = 0;
                break;
            } else {
                mu[prod] = -mu[i];
            }
        }
    }
    free(primes);
    free(is_comp);
}

/* ---------- Pack Möbius array (2 bits per value) ---------- */
MobiusSieve pack_mobius(const int *mu, int K) {
    MobiusSieve s;
    s.K = (uint32_t)K;
    if (K < 1) {
        s.abs_sum = 0;
        s.bits = NULL;
        return s;
    }
    uint32_t abs_sum = 0;
    for (int i = 1; i <= K; i++) {
        if (mu[i] != 0) abs_sum++;
    }
    s.abs_sum = abs_sum;
    int bit_len = 2 * K;
    int num_bytes = (bit_len + 7) / 8;
    s.bits = (uint8_t*)calloc(num_bytes, 1);
    int bit_pos = 0;
    for (int i = 1; i <= K; i++) {
        int code = 0;
        if (mu[i] == 1) code = 1;
        else if (mu[i] == -1) code = 2;
        int byte_idx = bit_pos / 8;
        int bit_offset = 6 - (bit_pos % 8);
        s.bits[byte_idx] |= (code << bit_offset);
        bit_pos += 2;
    }
    return s;
}

/* ---------- Free packed sieve ---------- */
void free_mobius_sieve(MobiusSieve *s) {
    if (s->bits) { free(s->bits); s->bits = NULL; }
}

/* ---------- Check square‑free via packed bits ---------- */
int is_square_free(const MobiusSieve *s, int n) {
    if (n < 1 || n > s->K) return 0;
    int bit_pos = (n - 1) * 2;
    int byte_idx = bit_pos / 8;
    int bit_offset = 6 - (bit_pos % 8);
    int code = (s->bits[byte_idx] >> bit_offset) & 0x03;
    return (code != 0);
}

/* ---------- TSP routing (bucket sort, O(K)) ---------- */
void tsp_route(const double *signal, int K, int *order) {
    double *angles = (double*)malloc(K * sizeof(double));
    for (int i = 0; i < K; i++) {
        double x = sin(i * 7.0) + 0.1 * cos(i * 13.0);
        double y = cos(i * 11.0) + 0.1 * sin(i * 17.0);
        angles[i] = atan2(y, x) + PI;
    }
    // Bucket sort
    int **buckets = (int**)malloc(BUCKETS * sizeof(int*));
    int *bucket_sizes = (int*)calloc(BUCKETS, sizeof(int));
    for (int b = 0; b < BUCKETS; b++) {
        buckets[b] = (int*)malloc(K * sizeof(int));
    }
    for (int i = 0; i < K; i++) {
        double a = angles[i];
        int idx = (int)((a / (2 * PI)) * BUCKETS) % BUCKETS;
        buckets[idx][bucket_sizes[idx]++] = i;
    }
    int pos = 0;
    for (int b = 0; b < BUCKETS; b++) {
        for (int j = 0; j < bucket_sizes[b]; j++) {
            order[pos++] = buckets[b][j];
        }
        free(buckets[b]);
    }
    free(buckets);
    free(bucket_sizes);
    free(angles);
}

/* ---------- Exponential convolution (two‑pass, O(K)) ---------- */
void conv_exp_kernel(const double *signal, int K, double alpha, double *conv) {
    double lam = exp(-alpha);
    double *f = (double*)malloc(K * sizeof(double));
    double *b = (double*)malloc(K * sizeof(double));
    f[0] = signal[0];
    for (int i = 1; i < K; i++) {
        f[i] = signal[i] + lam * f[i-1];
    }
    b[K-1] = signal[K-1];
    for (int i = K-2; i >= 0; i--) {
        b[i] = signal[i] + lam * b[i+1];
    }
    double inv_den = 1.0 / (1.0 - lam * lam);
    double inv_norm = 1.0 / NORM;
    for (int i = 0; i < K; i++) {
        double conv_exp = (f[i] + b[i] - signal[i]) * inv_den;
        conv[i] = (1.0 - conv_exp) * inv_norm;
    }
    free(f);
    free(b);
}

/* ---------- Supertrace and mass (safe) ---------- */
void supertrace_and_mass(const double *conv, int K, double *S, double *H, double *m) {
    *S = 0.0;
    for (int i = 0; i < K; i++) {
        double val = conv[i];
        if (i % 2 == 0) *S += fabs(val);
        else *S -= fabs(val);
    }
    if (*S == 0.0) {
        *H = 0.0;
        *m = 0.0;
        return;
    }
    double p = fabs(*S) / K;
    if (p <= 0.0 || p >= 1.0) {
        *H = 0.0;
        *m = 0.0;
        return;
    }
    if (p < 1e-15) {
        *H = 0.0;
        *m = 0.0;
        return;
    }
    *H = -ALPHA * p * log(p);
    if (*H < 0) *H = 0.0;
    if (*H > 1.0) *H = 1.0;
    *m = fabs(*S) * exp(-(*H));
}

/* ---------- Chip compression pipeline ---------- */
void chip_compress(const double *signal, int K, ChipProcessor *proc) {
    // 1. TSP routing
    int *order = proc->order;
    tsp_route(signal, K, order);
    // 2. Reorder
    double *sorted = proc->sorted;
    for (int i = 0; i < K; i++) {
        sorted[i] = signal[order[i]];
    }
    // 3. Convolution
    double *conv = proc->conv;
    conv_exp_kernel(sorted, K, ALPHA, conv);
    // 4. Supertrace
    supertrace_and_mass(conv, K, &proc->S, &proc->H, &proc->m);
    int M = (int)floor(fabs(proc->S));
    if (M < 1) M = 1;
    if (M > K) M = K;
    // 5. Magnitude selection (top M) – simple selection for small K
    double *mag = proc->mag;
    int *idx = (int*)malloc(K * sizeof(int));
    for (int i = 0; i < K; i++) {
        mag[i] = fabs(conv[i]);
        idx[i] = i;
    }
    // partial selection: bubble the largest M to the front
    for (int i = 0; i < M && i < K; i++) {
        int max_pos = i;
        double max_val = mag[i];
        for (int j = i+1; j < K; j++) {
            if (mag[j] > max_val) {
                max_val = mag[j];
                max_pos = j;
            }
        }
        if (max_pos != i) {
            // swap mag
            double tmp_m = mag[i];
            mag[i] = mag[max_pos];
            mag[max_pos] = tmp_m;
            // swap idx
            int tmp_i = idx[i];
            idx[i] = idx[max_pos];
            idx[max_pos] = tmp_i;
        }
    }
    // 6. Möbius filter and store kept
    int kept_count = 0;
    for (int i = 0; i < M; i++) {
        int pos = idx[i];
        int n = pos + 1;   // 1‑based index
        if (is_square_free(&proc->mu, n)) {
            proc->kept_idx[kept_count] = pos;
            proc->kept_val[kept_count] = conv[pos];
            kept_count++;
        }
    }
    proc->kept_count = kept_count;
    free(idx);
}

/* ---------- Initialize processor ---------- */
void chip_init(ChipProcessor *proc, int K) {
    proc->K = K;
    proc->signal = (double*)malloc(K * sizeof(double));
    proc->sorted  = (double*)malloc(K * sizeof(double));
    proc->conv    = (double*)malloc(K * sizeof(double));
    proc->mag     = (double*)malloc(K * sizeof(double));
    proc->order   = (int*)malloc(K * sizeof(int));
    proc->kept_idx = (int*)malloc(K * sizeof(int));
    proc->kept_val = (double*)malloc(K * sizeof(double));
    proc->kept_count = 0;
    proc->S = proc->H = proc->m = 0.0;
    // Pre‑compute Möbius sieve and pack it
    int *mu = (int*)malloc((K+1) * sizeof(int));
    mobius_sieve(K, mu);
    proc->mu = pack_mobius(mu, K);
    free(mu);
}

/* ---------- Free processor ---------- */
void chip_free(ChipProcessor *proc) {
    free(proc->signal);
    free(proc->sorted);
    free(proc->conv);
    free(proc->mag);
    free(proc->order);
    free(proc->kept_idx);
    free(proc->kept_val);
    free_mobius_sieve(&proc->mu);
}

/* ---------- Set signal and run ---------- */
void chip_run(ChipProcessor *proc, const double *signal, int K) {
    if (K > proc->K) {
        fprintf(stderr, "Error: signal length %d exceeds max_K %d\n", K, proc->K);
        return;
    }
    memcpy(proc->signal, signal, K * sizeof(double));
    chip_compress(proc->signal, K, proc);
}

/* ---------- Print summary ---------- */
void chip_print_summary(const ChipProcessor *proc) {
    printf("Chip summary:\n");
    printf("  K = %d\n", proc->K);
    printf("  Supertrace S = %f\n", proc->S);
    printf("  Entropy H    = %f\n", proc->H);
    printf("  Mass m       = %f\n", proc->m);
    printf("  Kept coefficients: %d / %d (ratio %.3f)\n",
           proc->kept_count, proc->K,
           (double)proc->kept_count / proc->K);
    printf("  First 5 kept (index, value):\n");
    for (int i = 0; i < proc->kept_count && i < 5; i++) {
        printf("    %d: %f\n", proc->kept_idx[i], proc->kept_val[i]);
    }
}

/* ---------- Main demonstration ---------- */
int main() {
    const int K = 200;
    ChipProcessor proc;
    chip_init(&proc, K);

    // Generate a test signal: harmonic numbers
    double *signal = (double*)malloc(K * sizeof(double));
    for (int i = 0; i < K; i++) {
        signal[i] = log(i+1.0);
    }

    chip_run(&proc, signal, K);
    chip_print_summary(&proc);

    free(signal);
    chip_free(&proc);
    return 0;
}