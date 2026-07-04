#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Circular buffer for probability distribution
typedef struct {
    double *p;      // array of size K
    int K;
} Dist;

Dist* create_dist(int K) {
    Dist *d = malloc(sizeof(Dist));
    d->K = K;
    d->p = calloc(K, sizeof(double));
    return d;
}

void free_dist(Dist *d) {
    free(d->p);
    free(d);
}

// One step of the Markov chain: p_next = p * P
// For a ring: from i to i-1 (left), i+1 (right), stay
void step_ring(Dist *curr, Dist *next, double left_prob, double right_prob, double stay_prob) {
    int K = curr->K;
    for (int i = 0; i < K; i++) {
        // probability to move to i from neighbours
        double incoming = 0.0;
        // from left neighbour (i-1) moving right
        incoming += curr->p[(i-1+K)%K] * right_prob;
        // from right neighbour (i+1) moving left
        incoming += curr->p[(i+1)%K] * left_prob;
        // from itself staying
        incoming += curr->p[i] * stay_prob;
        next->p[i] = incoming;
    }
}

int main() {
    int K = 1000;
    Dist *dist = create_dist(K);
    Dist *tmp = create_dist(K);
    
    // start with all mass at state 0
    dist->p[0] = 1.0;
    
    double left = 0.3, right = 0.3, stay = 0.4;
    
    // run 100 steps (each step O(K))
    for (int step = 0; step < 100; step++) {
        step_ring(dist, tmp, left, right, stay);
        // swap pointers
        Dist *swap = dist;
        dist = tmp;
        tmp = swap;
    }
    
    // print stationary distribution (should be uniform)
    printf("Stationary approx after 100 steps:\n");
    for (int i = 0; i < 10; i++)
        printf("p[%d] = %f\n", i, dist->p[i]);
    
    free_dist(dist);
    free_dist(tmp);
    return 0;
}