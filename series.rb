# Constants
PI = Math::PI
E  = Math::E
A  = 1.0 / (PI + E)

# Exact Gaussian function (for comparison)
def exact_gaussian(x)
  u = x - PI
  A * Math.exp(-u * u / (2.0 * E * E))
end

# O(K) evaluation of the series approximation:
#   g(π+u) ≈ a * Σ_{k=0}^{K-1} (-1)^k * u^{2k} / (2^k * k! * e^{2k})
# Internally uses x = -u^2 / (2*e^2) and computes Σ x^k/k! in O(K).
def series_approx(u, k_terms)
  x = - (u * u) / (2.0 * E * E)   # x = -u^2 / (2e^2)
  sum = 1.0
  term = 1.0
  (1...k_terms).each do |k|
    term *= x / k                  # next term = previous * x/k
    sum += term
  end
  A * sum                          # multiply by a outside
end

# Example: evaluate at several points and compare with exact
puts "K terms: 10"
puts "x\tapprox\texact\terror"
(0..20).step(0.5) do |x|
  u = x - PI
  approx = series_approx(u, 10)
  exact = exact_gaussian(x)
  printf "%.2f\t%.6f\t%.6f\t%+.2e\n", x, approx, exact, approx - exact
end

def series_with_modulo(u, k_terms)
  u2 = u * u
  sum = 0.0
  pow_u = 1.0
  coeff = 1.0
  k_terms.times do |k|
    # Only include even terms (n = 2k) – we already have that by construction.
    # But we can "modulo" the exponent to restrict to, say, multiples of 4:
    if k % 2 == 0
      sum += coeff * pow_u
    end
    # Still update for next term
    pow_u *= u2
    coeff *= -1.0 / (2.0 * E * E * (k + 1))
  end
  sum * (1.0 / (PI + E))
end

if mod_using_div(k, 2) == 0
  sum += coeff * pow_u
end

puts "Press RETURN when you're done."
gets