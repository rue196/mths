#include <stdlib.h>

int cmp(const void *a, const void *b) {
    return *(int*)a - *(int*)b;
}

int main() {
    int K = 1 << 20;  // 1,048,576 elements
    int *arr = malloc(K * sizeof(int));
    // ... fill array
    qsort(arr, K, sizeof(int), cmp);   // O(K log K) comparisons
    // Each comparison does O(1) memory access (via caches)
    free(arr);
}