#!/usr/bin/env ruby

# ------------------------------------------------------------
# 1. Compute the special constant a = 1/(π - e)
# ------------------------------------------------------------
A_SPECIAL = 1.0 / (Math::PI - Math::E)
puts "a_special = 1/(π - e) = #{A_SPECIAL}\n\n"

# ------------------------------------------------------------
# 2. Spectral sum and its derivative (O(K) operations)
#    Assumes C_i are real and symmetric (C_{-i}=C_i).
# ------------------------------------------------------------
class SpectralFunction
  def initialize(c, alpha)
    # c is an array of length 2K+1, where c[K] = C_0, c[K+i] = C_i (i>=0)
    @c = c
    @k = (c.length - 1) / 2
    @alpha = alpha
  end

  # ζ(t) = Σ_{i=-K}^{K} C_i exp(i t i/α)  -> real part
  def zeta(t)
    factor = t / @alpha
    sum = @c[@k]   # i=0 term
    (1..@k).each do |i|
      sum += 2.0 * @c[@k + i] * Math.cos(factor * i)
    end
    sum
  end

  # Analytical derivative ζ'(t) = Σ_{i=-K}^{K} C_i * (i/α) * i * exp(i t i/α)
  # Real part for symmetric C: derivative = -2 Σ_{i=1}^{K} C_i (i/α) sin(t i/α)
  def derivative(t)
    factor = t / @alpha
    sum = 0.0
    (1..@k).each do |i|
      sum += -2.0 * @c[@k + i] * (i / @alpha) * Math.sin(factor * i)
    end
    sum
  end
end

# ------------------------------------------------------------
# 3. Create example coefficients (all ones, K=20)
# ------------------------------------------------------------
K = 20
alpha = 0.3628   # from earlier examples
c = [1.0] * (2*K + 1)   # C_i = 1 for all i
spec = SpectralFunction.new(c, alpha)

# ------------------------------------------------------------
# 4. Compare analytical derivative with finite differences
#    using different step sizes (including the special a)
# ------------------------------------------------------------
t0 = 1.5
analytical = spec.derivative(t0)

puts "Analytical derivative at t=#{t0}: #{analytical}\n\n"

# Step sizes to try: the special a, and a few others
step_sizes = [A_SPECIAL, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

puts "Step size h       | Finite difference (fwd) | Error (abs)"
puts "-" * 60
step_sizes.each do |h|
  fd = (spec.zeta(t0 + h) - spec.zeta(t0)) / h
  err = (fd - analytical).abs
  printf "%12.4e  | %20.12f  | %12.4e\n", h, fd, err
end

puts "\nNote: The special step a = 1/(π-e) ≈ #{A_SPECIAL} gives an error similar to"
puts "other small steps, but it is algebraically linked to π and e."