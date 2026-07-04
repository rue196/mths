#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

// Circular buffer storing C_0 ... C_K (K+1 values)
typedef struct {
    double *coeff;   // underlying array (size K+1)
    int K;           // max index
    int head;        // index of C_0 in the circular array (always 0 for fixed storage, but can rotate)
    // For simplicity, we keep the buffer linear – a true circular buffer would need head for rotation.
    // Here we implement a classic circular buffer with head/tail for dynamic updates.
    int capacity;    // should be K+1
    int size;        // number of valid entries (normally K+1)
    int start;       // logical index of first element (C_0) in the circular array
} SpectralBuffer;

// Create a buffer for given K (O(K) memory allocation)
SpectralBuffer* sb_create(int K) {
    SpectralBuffer *sb = (SpectralBuffer*)malloc(sizeof(SpectralBuffer));
    sb->K = K;
    sb->capacity = K + 1;
    sb->coeff = (double*)malloc(sb->capacity * sizeof(double));
    sb->size = 0;
    sb->start = 0;   // logical C_0 at physical index 0 initially
    return sb;
}

// Add a new coefficient for index i (must be called in order i=0..K)
// This rotates the buffer if needed (circular behaviour)
bool sb_set(SpectralBuffer *sb, int idx, double value) {
    if (idx < 0 || idx > sb->K) return false;
    // Compute physical position: start + idx (mod capacity)
    int pos = (sb->start + idx) % sb->capacity;
    sb->coeff[pos] = value;
    if (idx + 1 > sb->size) sb->size = idx + 1;
    return true;
}

// Read coefficient for index i (0..K) with O(1) access
double sb_get(SpectralBuffer *sb, int idx) {
    if (idx < 0 || idx >= sb->size) return 0.0;
    int pos = (sb->start + idx) % sb->capacity;
    return sb->coeff[pos];
}

// Rotate the buffer: drop the oldest coefficient and insert a new one at the end (shifts logical start)
// This is O(1) – just increments start pointer.
void sb_rotate(SpectralBuffer *sb, double new_coeff_last) {
    // Overwrite the oldest element (logical index 0) with the new last coefficient?
    // Actually rotate: new C_0 becomes old C_1, ... , new C_K becomes old C_{K-1}, and we set new C_K = new_coeff_last.
    // Simpler: increment start; then set the new position for C_K.
    // But we must also adjust the size.
    if (sb->size < sb->capacity) {
        // Not full yet, just add at end
        int pos = (sb->start + sb->size) % sb->capacity;
        sb->coeff[pos] = new_coeff_last;
        sb->size++;
    } else {
        // Full: increment start (drop logical C_0)
        sb->start = (sb->start + 1) % sb->capacity;
        // The new logical C_{K-1} is at old start+K-1, now we set new C_K
        int pos = (sb->start + sb->K) % sb->capacity;
        sb->coeff[pos] = new_coeff_last;
    }
}

// Free buffer
void sb_destroy(SpectralBuffer *sb) {
    free(sb->coeff);
    free(sb);
}

// ------------------------------------------------------------
// O(K) evaluation of |ζ(t)|² using the circular buffer
// ------------------------------------------------------------
double zeta2_squared(SpectralBuffer *sb, double t, double alpha) {
    double sum = sb_get(sb, 0);          // C_0
    double factor = t / alpha;
    for (int i = 1; i <= sb->K; i++) {
        double C_i = sb_get(sb, i);
        sum += 2.0 * C_i * cos(factor * i);
    }
    return sum;
}

// ------------------------------------------------------------
// Example usage
// ------------------------------------------------------------
int main() {
    int K = 10;
    double alpha = 0.3628;

    // Create buffer and fill with dummy coefficients (all ones)
    SpectralBuffer *sb = sb_create(K);
    for (int i = 0; i <= K; i++) {
        sb_set(sb, i, 1.0);
    }

    // Evaluate at several t values (linspace 0..20, 500 points)
    int n_points = 500;
    double t_start = 0.0, t_end = 20.0;
    for (int j = 0; j <= n_points; j++) {
        double t = t_start + (t_end - t_start) * j / n_points;
        double val = zeta2_squared(sb, t, alpha);
        printf("%f %f\n", t, val);
    }

    // Example rotation: shift coefficients and add new C_K = 2.0
    sb_rotate(sb, 2.0);
    printf("\nAfter rotation, new C_0 = %f, C_K = %f\n", sb_get(sb, 0), sb_get(sb, K));

    sb_destroy(sb);
    return 0;
}