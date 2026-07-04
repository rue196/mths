#include <stdint.h>
#include <stdbool.h>

#define K 1024  // buffer size (power of two for fast modulo)

// Event descriptor
typedef struct {
    uint32_t type;
    uint32_t data;
} event_t;

// Circular buffer (shared between hardware and handler)
volatile event_t ring[K];
volatile uint32_t head = 0;   // write index (hardware fills)
volatile uint32_t tail = 0;   // read index (handler consumes)

// Helper: check if buffer is empty
static inline bool is_empty(void) {
    return head == tail;
}

// Helper: advance index with wrap‑around (power of two)
static inline uint32_t advance(uint32_t idx) {
    return (idx + 1) & (K - 1);   // assumes K is power of two
}

// Interrupt Service Routine (ISR)
// Called by CPU when hardware raises an interrupt.
// Runs with interrupts disabled (or at high priority).
void interrupt_handler(void) {
    // Process **all** pending events in the circular buffer
    // This loop runs in O(K) time in the worst case.
    while (!is_empty()) {
        event_t ev = ring[tail];   // O(1) read
        tail = advance(tail);      // O(1) update

        // Dispatch based on event type (O(1) per event)
        switch (ev.type) {
            case 1:  // handle network packet
                // ... process ev.data
                break;
            case 2:  // handle timer
                // ...
                break;
            default:
                // unknown event
                break;
        }
    }

    // Acknowledge interrupt to hardware (platform‑specific)
    // ...
}