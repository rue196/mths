require 'complex'

# ------------------------------------------------------------
# Spectral sum evaluation (O(K) operations)
# C is an array of length 2K+1, C[K] = C_0, C[K+i] = C_i
# Returns real part (assuming symmetric real C)
# ------------------------------------------------------------
def zeta(t, c, alpha)
  k = (c.length - 1) / 2
  sum = c[k].to_c                     # i = 0 term
  (1..k).each do |i|
    theta = t * i / alpha
    sum += c[k + i] * Complex.polar(1,  theta)   # C_i * e^{iθ}
    sum += c[k - i] * Complex.polar(1, -theta)   # C_{-i} * e^{-iθ}
  end
  sum.real
end

PI = Math::PI
E  = Math::E
A  = 1.0 / (PI + E)

# O(K) series approximation using separate recurrence for power and coefficient
def series_approx(u, k_terms)
  u2 = u * u                     # base for even powers
  # coefficient update constant: (-1) / (2 * e^2)
  coeff_factor = -1.0 / (2.0 * E * E)

  sum = 0.0
  pow_u = 1.0    # u^(2k), starts at k=0 → u^0 = 1
  coeff = 1.0    # (-1)^k / (2^k * k! * e^(2k)), starts at k=0 → 1

  k_terms.times do |k|
    sum += coeff * pow_u          # add current term

    # Update for next term (k → k+1):
    pow_u *= u2                   # exponent update: u^(2k) → u^(2k+2)
    coeff *= coeff_factor / (k + 1)  # coefficient update: divide by (k+1)
  end

  A * sum   # multiply by a = 1/(π+e) outside
end

# Exact function for comparison
def exact_gaussian(x)
  u = x - PI
  A * Math.exp(-u * u / (2.0 * E * E))
end

# Example: test at x = π (so u = 0)
puts "At x = π (u=0):"
k_vals = [1, 2, 5, 10, 20]
k_vals.each do |k|
  approx = series_approx(0.0, k)
  exact = exact_gaussian(PI)
  puts "  K=#{k}: approx=#{approx}, exact=#{exact}, error=#{approx - exact}"
end

# Example: test at x = 3.0 (u = 3 - π)
puts "\nAt x = 3.0 (u = -0.14159):"
approx = series_approx(3.0 - PI, 10)
exact = exact_gaussian(3.0)
puts "  K=10: approx=#{approx}, exact=#{exact}, error=#{approx - exact}"


puts "Press RETURN when you're done."
gets
   