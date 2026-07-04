#include <stdio.h>
#include <stdlib.h>

int main() {
    int k = 10;
    int len = 2 * k + 1;
    double *buffer = (double*)malloc(len * sizeof(double));   // O(K) allocation

    // Fill with ones
    for (int i = 0; i < len; i++) {
        buffer[i] = 1.0;
    }

    // O(K) sum
    double sum = 0.0;
    for (int i = 0; i < len; i++) {
        sum += buffer[i];
    }
    printf("Sum of %d elements = %f\n", len, sum);

    free(buffer);   // O(1) deallocation
    return 0;
}