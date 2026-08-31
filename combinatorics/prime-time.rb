require 'complex'

# Evaluates ζ(t) = Σ_{i=-K}^{K} C[i] * exp(i * t * i / α)
# C is an array of length 2K+1, where C[K] corresponds to i=0.
def zeta2(t, c, alpha)
  k = (c.length - 1) / 2          # K
  total = 0.0
  (-k..k).each do |i|
    # c[i + k] → coefficient C_i
    total += c[i + k] * Complex.polar(1, t * i / alpha)
  end
  total.real                      # return real part (C symmetric → real result)
end

# Example usage (same as Python code)
K = 10
C = [1.0] * (2 * K + 1)          # all ones, symmetric
alpha = 0.362737364

# Generate t values from 0 to 20, 500 points
t_values = (0..500).map { |j| j * 20.0 / 1000 }
results = t_values.map { |t| zeta2(t, C, alpha) }
