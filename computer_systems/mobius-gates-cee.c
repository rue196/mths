/*
 * mobius_harness.c
 *
 * Möbius chip pipeline in C with minimal garbage collection.
 * Compile with: gcc -std=c99 -O2 -lm -o mobius_harness mobius_harness.c
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <complex.h>

/* ---------- Constants ---------- */
#define PI         3.14159265358979323846
#define E          2.71828182845904523536
#define ALPHA       (1.0 / (PI - E))          /* ≈ 2.362 */
#define ALPHA_USER  0.3628
#define A_NEW       (ALPHA / ALPHA_USER)      /* ≈ 6.511 */
#define NORM        (1.0 - exp(-ALPHA * (PI + E)))

#define MAX_K       256
#define MAX_BUFFER  4096
#define MAX_TASKS   100

/* ---------- Packed Möbius sieve ---------- */
typedef struct {
    uint32_t K;
    uint32_t abs_sum;
    uint8_t *bits;               /* length = (2*K + 7)/8 */
} MobiusSieve;

/* ---------- Chip processor buffers (pre‑allocated) ---------- */
typedef struct {
    int max_K;
    double *f;          /* forward pass */
    double *b;          /* backward pass */
    double *conv;       /* convolution result */
    double *mag;        /* magnitudes for sorting */
    int *order;         /* TSP order */
    int *idx;           /* index array for argsort */
    MobiusSieve mu;     /* cached sieve */
} ChipProcessor;

/* ---------- Elliptic gate ---------- */
typedef struct {
    int K;
    double *coeffs;     /* length 2*K+1, C_i = μ(|i|) */
    int *indices;       /* list of non‑zero indices (square‑free) */
    int num_indices;
    double period;
} EllipticGate;

/* ---------- Inference buffer ---------- */
typedef struct {
    int max_len;
    double *signal;     /* current signal */
    int length;
    double last_S, last_H, last_m;
    int *last_kept_idx;
    double *last_kept_val;
    int last_kept_count;
} InferenceBuffer;

/* ---------- Harness ---------- */
typedef struct {
    int K;
    int M;                 /* number of coefficients to keep */
    EllipticGate gate;
    ChipProcessor processor;
    InferenceBuffer buffer;
    double tasks[MAX_TASKS];
    int num_tasks;
} MobiusHarness;

/* ---------- Utility functions ---------- */
static double min_d(double a, double b) { return a < b ? a : b; }
static double max_d(double a, double b) { return a > b ? a : b; }
static double clamp_d(double x, double lo, double hi) {
    return max_d(lo, min_d(x, hi));
}

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

/* ---------- Chip processor initialisation ---------- */
void chip_processor_init(ChipProcessor *p, int max_K) {
    p->max_K = max_K;
    p->f = (double*)calloc(max_K, sizeof(double));
    p->b = (double*)calloc(max_K, sizeof(double));
    p->conv = (double*)calloc(max_K, sizeof(double));
    p->mag = (double*)calloc(max_K, sizeof(double));
    p->order = (int*)malloc(max_K * sizeof(int));
    p->idx = (int*)malloc(max_K * sizeof(int));
    for (int i = 0; i < max_K; i++) p->idx[i] = i;
    // Compute sieve and pack it
    int *mu = (int*)malloc((max_K+1) * sizeof(int));
    mobius_sieve(max_K, mu);
    p->mu = pack_mobius(mu, max_K);
    free(mu);
}

void chip_processor_free(ChipProcessor *p) {
    free(p->f);
    free(p->b);
    free(p->conv);
    free(p->mag);
    free(p->order);
    free(p->idx);
    free_mobius_sieve(&p->mu);
}

/* ---------- TSP routing (bucket sort by phase, O(K)) ---------- */
void tsp_route(const double *signal, int K, int *order) {
    double *angles = (double*)malloc(K * sizeof(double));
    for (int i = 0; i < K; i++) {
        double x = sin(i * 7.0) + 0.1 * cos(i * 13.0);
        double y = cos(i * 11.0) + 0.1 * sin(i * 17.0);
        angles[i] = atan2(y, x) + PI;
    }
    // Bucket sort: 360 buckets
    int **buckets = (int**)malloc(360 * sizeof(int*));
    int *bucket_sizes = (int*)calloc(360, sizeof(int));
    for (int i = 0; i < 360; i++) {
        buckets[i] = (int*)malloc(K * sizeof(int));
    }
    for (int i = 0; i < K; i++) {
        double a = angles[i];
        int idx = (int)((a / (2 * PI)) * 360) % 360;
        buckets[idx][bucket_sizes[idx]++] = i;
    }
    int pos = 0;
    for (int b = 0; b < 360; b++) {
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
    // Forward pass
    double *f = (double*)malloc(K * sizeof(double));
    f[0] = signal[0];
    for (int i = 1; i < K; i++) {
        f[i] = signal[i] + lam * f[i-1];
    }
    // Backward pass
    double *b = (double*)malloc(K * sizeof(double));
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

/* ---------- Supertrace and mass (O(K)) ---------- */
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

/* ---------- Chip compression (keeps top M with μ(n)!=0) ---------- */
void chip_compress(const double *signal, int K, const ChipProcessor *proc,
                   int *kept_idx, double *kept_val, int *kept_count,
                   double *S, double *H, double *m) {
    // 1. TSP routing
    int *order = (int*)malloc(K * sizeof(int));
    tsp_route(signal, K, order);
    // 2. Reorder signal
    double *signal_sorted = (double*)malloc(K * sizeof(double));
    for (int i = 0; i < K; i++) {
        signal_sorted[i] = signal[order[i]];
    }
    // 3. Convolution
    double *conv = (double*)malloc(K * sizeof(double));
    conv_exp_kernel(signal_sorted, K, ALPHA, conv);
    // 4. Supertrace
    supertrace_and_mass(conv, K, S, H, m);
    int M = (int)floor(fabs(*S));
    if (M < 1) M = 1;
    if (M > K) M = K;
    // 5. Compute magnitudes and select top M (simple selection)
    int *idx = (int*)malloc(K * sizeof(int));
    double *mag = (double*)malloc(K * sizeof(double));
    for (int i = 0; i < K; i++) {
        idx[i] = i;
        mag[i] = fabs(conv[i]);
    }
    // Selection sort on magnitude (descending)
    int selected = 0;
    for (int i = 0; i < K && selected < M; i++) {
        int max_pos = -1;
        double max_val = -1.0;
        for (int j = i; j < K; j++) {
            if (mag[j] > max_val) {
                max_val = mag[j];
                max_pos = j;
            }
        }
        if (max_pos < 0) break;
        // swap to front
        int tmp_idx = idx[i];
        idx[i] = idx[max_pos];
        idx[max_pos] = tmp_idx;
        double tmp_mag = mag[i];
        mag[i] = mag[max_pos];
        mag[max_pos] = tmp_mag;
        selected++;
    }
    // Filter by square‑free index (μ(n) != 0)
    int count = 0;
    for (int i = 0; i < selected; i++) {
        int pos = idx[i];
        int n = pos + 1;   // 1‑based for μ
        if (n <= proc->mu.K && proc->mu.bits) {
            int bit_pos = (n - 1) * 2;
            int byte_idx = bit_pos / 8;
            int bit_offset = 6 - (bit_pos % 8);
            int code = (proc->mu.bits[byte_idx] >> bit_offset) & 0x03;
            if (code != 0) {
                kept_idx[count] = pos;
                kept_val[count] = conv[pos];
                count++;
            }
        }
    }
    *kept_count = count;
    free(order);
    free(signal_sorted);
    free(conv);
    free(idx);
    free(mag);
}

/* ---------- Elliptic gate: ζ(t) = Σ C_i * exp(i * t * i / α) ---------- */
double complex zeta_elliptic(double t, const EllipticGate *gate) {
    double complex sum = 0.0 + 0.0*I;
    for (int j = 0; j < gate->num_indices; j++) {
        int i = gate->indices[j];
        double coeff = gate->coeffs[i + gate->K]; // offset
        double phase = t * i / ALPHA;
        sum += coeff * cexp(I * phase);
    }
    return sum;
}

double power_spectrum_elliptic(double t, const EllipticGate *gate) {
    double complex z = zeta_elliptic(t, gate);
    return creal(z)*creal(z) + cimag(z)*cimag(z);
}

/* ---------- Build elliptic gate coefficients (square‑free only) ---------- */
void elliptic_gate_init(EllipticGate *gate, int K, int smooth) {
    gate->K = K;
    gate->coeffs = (double*)calloc(2*K+1, sizeof(double));
    int *mu = (int*)malloc((K+1) * sizeof(int));
    mobius_sieve(K, mu);
    // Count square‑free indices
    int count = 0;
    for (int i = -K; i <= K; i++) {
        if (i == 0) continue;
        if (mu[abs(i)] != 0) count++;
    }
    gate->indices = (int*)malloc(count * sizeof(int));
    gate->num_indices = 0;
    for (int i = -K; i <= K; i++) {
        if (i == 0) continue;
        if (mu[abs(i)] != 0) {
            gate->indices[gate->num_indices++] = i;
            gate->coeffs[i + K] = mu[abs(i)];
        }
    }
    // Optional smoothing (if smooth) not implemented here for brevity.
    gate->period = 2 * PI * ALPHA;
    free(mu);
}

void elliptic_gate_free(EllipticGate *gate) {
    free(gate->coeffs);
    free(gate->indices);
}

/* ---------- Inference buffer ---------- */
void inference_buffer_init(InferenceBuffer *buf, int max_len) {
    buf->max_len = max_len;
    buf->signal = (double*)calloc(max_len, sizeof(double));
    buf->length = 0;
    buf->last_S = buf->last_H = buf->last_m = 0.0;
    buf->last_kept_idx = (int*)malloc(max_len * sizeof(int));
    buf->last_kept_val = (double*)malloc(max_len * sizeof(double));
    buf->last_kept_count = 0;
}

void inference_buffer_free(InferenceBuffer *buf) {
    free(buf->signal);
    free(buf->last_kept_idx);
    free(buf->last_kept_val);
}

void inference_buffer_set_signal(InferenceBuffer *buf, const double *sig, int len) {
    if (len > buf->max_len) len = buf->max_len;
    memcpy(buf->signal, sig, len * sizeof(double));
    buf->length = len;
}

void inference_buffer_compress(InferenceBuffer *buf, const ChipProcessor *proc) {
    chip_compress(buf->signal, buf->length, proc,
                  buf->last_kept_idx, buf->last_kept_val, &buf->last_kept_count,
                  &buf->last_S, &buf->last_H, &buf->last_m);
}

/* ---------- Harness ---------- */
void harness_init(MobiusHarness *h, int K, int M) {
    h->K = K;
    h->M = M;
    elliptic_gate_init(&h->gate, K, 1);
    chip_processor_init(&h->processor, K);
    inference_buffer_init(&h->buffer, K);
    h->num_tasks = 0;
}

void harness_free(MobiusHarness *h) {
    elliptic_gate_free(&h->gate);
    chip_processor_free(&h->processor);
    inference_buffer_free(&h->buffer);
}

void harness_add_task(MobiusHarness *h, double t) {
    if (h->num_tasks < MAX_TASKS) {
        h->tasks[h->num_tasks++] = t;
    }
}

void harness_run(MobiusHarness *h) {
    if (h->num_tasks == 0) {
        printf("No tasks to process.\n");
        return;
    }
    // 1. Compute power spectra for each task
    int K_tasks = h->num_tasks;
    double *spectra = (double*)malloc(K_tasks * sizeof(double));
    for (int i = 0; i < K_tasks; i++) {
        spectra[i] = power_spectrum_elliptic(h->tasks[i], &h->gate);
    }
    // 2. Angle sort by the task itself (using sin/cos)
    double *angles = (double*)malloc(K_tasks * sizeof(double));
    int *order = (int*)malloc(K_tasks * sizeof(int));
    for (int i = 0; i < K_tasks; i++) {
        angles[i] = atan2(sin(h->tasks[i]), cos(h->tasks[i])) + PI;
    }
    // Simple insertion sort of indices by angle
    for (int i = 0; i < K_tasks; i++) order[i] = i;
    for (int i = 1; i < K_tasks; i++) {
        int j = i;
        while (j > 0 && angles[order[j]] < angles[order[j-1]]) {
            int tmp = order[j];
            order[j] = order[j-1];
            order[j-1] = tmp;
            j--;
        }
    }
    // 3. Merge (sum) spectra in sorted order
    double *merged = (double*)calloc(h->K, sizeof(double));
    // We'll place the spectra into the merged signal at positions determined by order
    for (int i = 0; i < K_tasks && i < h->K; i++) {
        int idx = order[i];
        merged[i] = spectra[idx];
    }
    // 4. Set the buffer
    inference_buffer_set_signal(&h->buffer, merged, h->K);
    // 5. Compress
    inference_buffer_compress(&h->buffer, &h->processor);
    // 6. Print summary
    printf("Supertrace S = %f\n", h->buffer.last_S);
    printf("Entropy H    = %f\n", h->buffer.last_H);
    printf("Mass m       = %f\n", h->buffer.last_m);
    printf("Compressed %d coefficients (out of %d)\n",
           h->buffer.last_kept_count, h->K);

    free(spectra);
    free(angles);
    free(order);
    free(merged);
}

/* ---------- Example main ---------- */
int main() {
    MobiusHarness harness;
    harness_init(&harness, 64, 16);

    // Generate tasks: t values from 0 to 2*period
    double period = harness.gate.period;
    for (int i = 0; i < 20; i++) {
        double t = period * i / 20.0;
        harness_add_task(&harness, t);
    }

    harness_run(&harness);
    harness_free(&harness);
    return 0;
}