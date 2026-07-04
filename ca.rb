PI = Math::PI
E  = Math::E
A  = 1.0 / (PI + E)          # a = 1/(π+e)   (you can change to π-e if needed)

# Exact Gaussian
def exact_gaussian(x)
  u = x - PI
  A * Math.exp(-u * u / (2.0 * E * E))
end

# O(K) Taylor series: only even powers (Collatz even condition)
def series_approx(u, k_terms)
  x = - (u * u) / (2.0 * E * E)   # x = -u^2/(2e^2)
  sum = 1.0
  term = 1.0
  (1...k_terms).each do |k|
    term *= x / k                  # recurrence: term_k = term_{k-1} * x/k
    sum += term
  end
  A * sum   # multiply by a outside
end

# Test convergence (Collatz condition: only even powers, odd terms are ignored)
puts "K\tapprox at x=π+0.5\texact\terror"
[1, 2, 5, 10, 20, 30].each do |k|
  u = 0.5
  approx = series_approx(u, k)
  exact = exact_gaussian(PI + u)
  printf "%2d\t%.8f\t%.8f\t%+.2e\n", k, approx, exact, approx - exact
end

puts "Press RETURN when you're done."
gets