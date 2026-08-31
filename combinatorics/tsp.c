#include <stdio.h>
#include <stdlib.h>

int cmp(const void *a, const void *b) {
    double da = *(double*)a, db = *(double*)b;
    return (da > db) - (da < db);
}

int main() {
    int K = 100000;
    double *cities = malloc(K * sizeof(double));
    for (int i = 0; i < K; i++)
        cities[i] = (double)rand() / RAND_MAX * 1000.0;
    
    qsort(cities, K, sizeof(double), cmp);  // O(K log K)
    double dist = 2.0 * (cities[K-1] - cities[0]);
    printf("Total cycle length: %.2f\n", dist);
    
    free(cities);
    return 0;
}