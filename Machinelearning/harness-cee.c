/* mobius_harness.h */

#include <stdint.h>
#include <stddef.h>
#include <complex.h>

#define MAX_K 1024
#define MAX_BUFFER 4096

/* Packed Möbius sieve (2 bits per value) */
typedef struct {
    uint32_t K;
    uint32_t abs_sum;
    uint8_t *bits;               /* length = (2*K + 7)/8 */
} MobiusSieve;

/* Pre‑allocated buffers for chip pipeline */
typedef struct {
    double *f;                  /* forward pass */
    double *b;                  /* backward pass */
    double *conv;               /* convolution result */
    double *mag;                /* magnitudes */
    int *order;                 /* TSP order */
    int *idx;                   /* index array for sorting */
    MobiusSieve mu;             /* cached sieve */
    size_t max_K;
} ChipProcessor;

/* Elliptic gate state */
typedef struct {
    int K;
    double *coeffs;             /* C_i for i = -K..K, length 2K+1 */
    int *indices;               /* non‑zero indices (square‑free) */
    size_t num_indices;
    double period;
} EllipticMobiusGate;

/* Inference buffer */
typedef struct {
    double *signal;             /* current signal, length max_buffer */
    size_t length;
    double last_S, last_H, last_m;
    int *last_kept_idx;         /* compressed indices */
    double *last_kept_val;      /* compressed values */
    size_t last_kept_count;
} InferenceBuffer;

/* Harness */
typedef struct {
    EllipticMobiusGate gate;
    ChipProcessor processor;
    InferenceBuffer buffer;
    int K, M;
    double *task_points;        /* list of t values */
    size_t num_tasks;
} MobiusHarness;

/* Sieve and pack μ */
MobiusSieve mobius_sieve(uint32_t K);

/* TSP routing (bucket sort by phase) */
void tsp_route(const double *signal, int K, int *order);

/* Exponential convolution (two‑pass) */
void conv_exp_kernel(const double *signal, int K, double alpha, double *conv);

/* Supertrace and mass */
void supertrace_and_mass(const double *conv, int K, double *S, double *H, double *m);

/* Chip compression */
void chip_compress(const double *signal, int K, const MobiusSieve *mu,
                   int *kept_idx, double *kept_val, size_t *kept_count,
                   double *S, double *H, double *m);

/* Elliptic gate: ζ(t) = Σ C_i * exp(i * t * i / α) */
double complex_zeta(double t, const EllipticMobiusGate *gate);

/* Power spectrum |ζ(t)|² */
double power_spectrum(double t, const EllipticMobiusGate *gate);

/* Full harness: sort tasks, compute spectra, merge, compile, check */
void mobius_harness_run(MobiusHarness *h, const double *tasks, size_t n);