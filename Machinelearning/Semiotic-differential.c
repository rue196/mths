#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define K 128          // hidden state size
#define VOCAB 1000     // vocabulary size
#define DT 0.1f        // time step
#define ALPHA 0.1f     // anchor coupling
#define EPS 0.01f      // exponential decay

// Global parameters (learned, here fixed for demo)
float W_in[K][VOCAB];     // input projection (K x vocab)
float W_out[VOCAB][K];    // output projection (vocab x K)
int anchors[20];          // fixed anchor indices (e.g., primes)

// PDE state
float h[K];

// Helper: softmax in place (logits -> probabilities)
void softmax(float *logits, int n) {
    float max = logits[0];
    for (int i = 1; i < n; ++i) if (logits[i] > max) max = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        logits[i] = expf(logits[i] - max);
        sum += logits[i];
    }
    for (int i = 0; i < n; ++i) logits[i] /= sum;
}

// 1D discrete Laplacian (zero flux boundaries)
void laplacian(const float *h, float *diff) {
    diff[0] = 0.0f;
    diff[K-1] = 0.0f;
    for (int i = 1; i < K-1; ++i)
        diff[i] = h[i-1] + h[i+1] - 2.0f * h[i];
}

// Non‑local sum over anchors
float anchor_sum(const float *h, const int *anchors, int num_anchors) {
    float s = 0.0f;
    for (int i = 0; i < num_anchors; ++i)
        s += h[anchors[i]];
    return s;
}

// One PDE step: update h given input token (one‑hot index)
void pde_step(int token) {
    // 1. Input injection: h += W_in[:, token]
    for (int i = 0; i < K; ++i)
        h[i] += W_in[i][token];

    // 2. Compute diffusion, nonlocal, decay
    float diff[K];
    laplacian(h, diff);
    float an_sum = anchor_sum(h, anchors, sizeof(anchors)/sizeof(int));
    float nonlocal = ALPHA * an_sum;
    for (int i = 0; i < K; ++i) {
        float decay = -EPS * h[i];
        h[i] += DT * (diff[i] + nonlocal + decay);
    }
}

// Predict next token probabilities (logits -> softmax)
void predict(float *probs) {
    // Compute logits = W_out * h
    for (int j = 0; j < VOCAB; ++j) {
        float logit = 0.0f;
        for (int i = 0; i < K; ++i)
            logit += W_out[j][i] * h[i];
        probs[j] = logit;
    }
    softmax(probs, VOCAB);
}

// Example usage: process a sequence of tokens
int main() {
    // Initialize parameters (dummy values)
    for (int i = 0; i < K; ++i)
        for (int j = 0; j < VOCAB; ++j) {
            W_in[i][j] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
            W_out[j][i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
        }
    for (int i = 0; i < 20; ++i) anchors[i] = i * (K/20); // evenly spaced
    memset(h, 0, sizeof(h));

    // Input token sequence (example)
    int tokens[] = {42, 137, 5, 999, 0};
    int num_tokens = sizeof(tokens)/sizeof(int);

    for (int t = 0; t < num_tokens; ++t) {
        pde_step(tokens[t]);
        float probs[VOCAB];
        predict(probs);
        printf("After token %d, top-3 next token probabilities:\n", tokens[t]);
        for (int p = 0; p < 3; ++p) {
            int argmax = 0;
            for (int i = 1; i < VOCAB; ++i)
                if (probs[i] > probs[argmax]) argmax = i;
            printf("  token %d: %.4f\n", argmax, probs[argmax]);
            probs[argmax] = 0; // mask to get next best
        }
    }
    return 0;
}