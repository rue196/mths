#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

typedef struct {
    double *buffer;   // underlying array
    int capacity;     // maximum number of elements (K)
    int head;         // index of oldest element (read position)
    int tail;         // index of next write position
    int size;         // current number of elements
} CircularBuffer;

// O(K) allocation and O(1) initialisation
CircularBuffer* cb_create(int capacity) {
    CircularBuffer *cb = (CircularBuffer*)malloc(sizeof(CircularBuffer));
    if (!cb) return NULL;
    cb->buffer = (double*)malloc(capacity * sizeof(double));
    if (!cb->buffer) {
        free(cb);
        return NULL;
    }
    cb->capacity = capacity;
    cb->head = 0;
    cb->tail = 0;
    cb->size = 0;
    return cb;
}

// O(1) enqueue (returns true if successful, false if full)
bool cb_enqueue(CircularBuffer *cb, double value) {
    if (cb->size == cb->capacity) return false;
    cb->buffer[cb->tail] = value;
    cb->tail = (cb->tail + 1) % cb->capacity;
    cb->size++;
    return true;
}

// O(1) dequeue (returns true if successful, false if empty)
bool cb_dequeue(CircularBuffer *cb, double *out) {
    if (cb->size == 0) return false;
    *out = cb->buffer[cb->head];
    cb->head = (cb->head + 1) % cb->capacity;
    cb->size--;
    return true;
}

// O(1) checks
bool cb_is_empty(CircularBuffer *cb) { return cb->size == 0; }
bool cb_is_full(CircularBuffer *cb)  { return cb->size == cb->capacity; }
int cb_size(CircularBuffer *cb)      { return cb->size; }

// O(1) free
void cb_destroy(CircularBuffer *cb) {
    free(cb->buffer);
    free(cb);
}

// Example: use circular buffer to store last K spectral coefficients
int main() {
    int K = 10;   // buffer capacity (O(K) memory)
    CircularBuffer *cb = cb_create(K);
    
    // Simulate adding 20 values (older ones get overwritten)
    for (int i = 0; i < 20; i++) {
        if (!cb_enqueue(cb, (double)i)) {
            // Buffer full – dequeue oldest and retry (overwrite)
            double discarded;
            cb_dequeue(cb, &discarded);
            cb_enqueue(cb, (double)i);
        }
        printf("After adding %2d, buffer size = %d\n", i, cb_size(cb));
    }
    
    // Print remaining elements (the most recent K values)
    printf("\nRemaining elements (oldest to newest):\n");
    int idx = cb->head;
    for (int i = 0; i < cb->size; i++) {
        printf("%.0f ", cb->buffer[idx]);
        idx = (idx + 1) % cb->capacity;
    }
    printf("\n");
    
    cb_destroy(cb);
    return 0;
}