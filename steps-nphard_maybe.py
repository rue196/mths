#!/usr/bin/env python3
"""
complexity_probe.py

Empirical scaling of:
  (a) rational fit per value  → expect flat / O(log K)
  (b) Collatz step count      → expect super-logarithmic, record-holders grow
"""

import math, time, struct
from fractions import Fraction

# ---------- rational search (fixed candidate set) ----------
def rational_fit(y, e_ap, R=5, D=4):
    best_err = float('inf')
    best = (Fraction(0,1), Fraction(0,1))
    for den in range(1, D+1):
        for num in range(-R*den, R*den+1):
            r = Fraction(num, den)
            for den2 in range(1, D+1):
                for num2 in range(-R*den2, R*den2+1):
                    s = Fraction(num2, den2)
                    err = abs(y - (float(r) + float(s)*e_ap))
                    if err < best_err:
                        best_err = err
                        best = (r, s)
    return best, best_err

# ---------- collatz step count ----------
def collatz_steps(n, cap=10_000_000):
    steps = 0
    while n != 1 and steps < cap:
        n = n//2 if n % 2 == 0 else 3*n + 1
        steps += 1
    return steps

# ---------- measure ----------
def main():
    E_AP = sum(1.0/math.factorial(k) for k in range(60))  # ≈ e

    print("=== (a) rational fit time vs K ===")
    print(" K      total_time_s   time_per_value_us")
    for K in [10**3, 10**4, 10**5, 10**6]:
        ys = [math.sin(i)*2.5 + 1.0 for i in range(K)]
        t0 = time.perf_counter()
        for y in ys:
            rational_fit(y, E_AP)
        dt = time.perf_counter() - t0
        print(f"{K:>7d}   {dt:>10.4f}   {1e6*dt/K:>14.3f}")

    print("\n=== (b) collatz step count for record-holders ===")
    print("   n          steps   n/steps")
    records = []
    best = 0
    for n in range(1, 200_000):
        s = collatz_steps(n)
        if s > best:
            best = s
            records.append((n, s))
    for n, s in records[-10:]:
        print(f"{n:>10d}   {s:>8d}   {n/s:>10.2f}")

    print("\n=== (c) max steps vs search range N ===")
    print("   N          max_steps   max_steps/log2(N)")
    for N in [10**2, 10**3, 10**4, 10**5, 2*10**5]:
        m = max(collatz_steps(n) for n in range(1, N+1))
        print(f"{N:>8d}   {m:>10d}   {m/math.log2(N):>14.3f}")

if __name__ == "__main__":
    main()