layout(std430, binding = 0) readonly buffer CiBlock {
    double c[2 * K + 1];
};
layout(std430, binding = 1) writeonly buffer OutBlock {
    double inv[5 * K];
};

const int K = 256;

void main() {
    uint k = gl_GlobalInvocationID.x;
    if (k >= uint(K)) return;

    int slot_pos = K + int(k + 1);   // positive half  → symmetric
    int slot_neg = K - int(k + 1);   // negative half  → asymmetric

    double sym  = c[slot_pos];       // load 1
    double asym = c[slot_neg];       // load 2

    inv[5u * k + 0u] = sym;
    inv[5u * k + 1u] = asym;
    inv[5u * k + 2u] = sym * sym + asym * asym;
    inv[5u * k + 3u] = sym * sym - asym * asym;
    inv[5u * k + 4u] = c[K];         // DC term
}