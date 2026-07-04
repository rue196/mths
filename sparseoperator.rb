PI = Math::PI
E  = Math::E
A  = 1.0 / (PI - E)          # a = 1/(π−e)

# Spectral sum with Collatz condition: only even i contribute.
# C is an array of length 2K+1, C[K] = C_0, C[K+i] = C_i (symmetric).
def zeta_collatz(t, c, alpha)
  k = (c.length - 1) / 2
  sum = c[k]                # i = 0 (even)
  factor = t / alpha
  i = 2
  while i <= k
    # i is even → include term
    sum += 2.0 * c[k + i] * Math.cos(factor * i)
    i += 2                  # jump by 2 to stay even
  end
  sum
end

# Example: dummy coefficients
K = 20
c = [1.0] * (2*K + 1)
alpha = 0.3628
t = 1.5

puts "Collatz-filtered ζ(t) = #{zeta_collatz(t, c, alpha)}"

puts "Press Enter to exit"
gets
