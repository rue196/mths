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
alpha = 0.3628

# Generate t values from 0 to 20, 500 points
t_values = (0..500).map { |j| j * 20.0 / 500 }
results = t_values.map { |t| zeta2(t, C, alpha) }

# First few results as a quick check
puts results[0..4]

# Approximate exp(x) using Taylor series:
#   e^x = Σ_{n=0}^{K} x^n / n!   (error decreases as K increases)
# Complexity: O(K) time, O(1) extra space.
def exp_series(x, k)
  sum = 1.0          # n = 0 term
  term = 1.0         # current term = x^n / n!
  (1..k).each do |n|
    term *= x / n    # update term: x^n/n! = (x^(n-1)/(n-1)!) * (x/n)
    sum += term
  end
  sum
end

# Approximate π using Leibniz series:
#   π/4 = Σ_{n=0}^{K} (-1)^n / (2n+1)
# So π ≈ 4 * Σ_{n=0}^{K} (-1)^n/(2n+1)
# Complexity: O(K) time.
def pi_series(k)
  sum = 0.0
  sign = 1.0
  (0..k).each do |n|
    denominator = 2 * n + 1
    sum += sign / denominator
    sign = -sign        # alternate sign
  end
  4.0 * sum
end

# Example usage:
K = 1000   # number of terms – algorithm runs in O(K)
x = 1.0

exp_approx = exp_series(x, K)
pi_approx  = pi_series(K)

puts "exp(1) ≈ #{exp_approx}  (error: #{exp_approx - Math::E})"
puts "π      ≈ #{pi_approx}  (error: #{pi_approx - Math::PI})"

def exp_series(x, k)
  sum = 1.0
  term = 1.0
  (1..k).each do |n|
    term *= x / n       # key: build term from previous term, not from scratch
    sum += term
  end
  sum
end

e_approx = exp_series(1.0, 10)   # => 2.7182818011463845
a = 1.0 / (Math::PI - e_approx)  # => 2.362322215598202
1.0/a + e_approx                  # => 3.141592653589793  (still matches π?)

# Incremental O(K) Taylor series for e^x
def exp_series_incremental(x, max_k)
  return enum_for(:exp_series_incremental, x, max_k) unless block_given?
  sum = 1.0
  term = 1.0
  yield 0, sum                     # K=0 term
  (1..max_k).each do |n|
    term *= x / n
    sum += term
    yield n, sum
  end
end

# True constant a = 1/(π - e)
a_true = 1.0 / (Math::PI - Math::E)

max_k = 30                         # far more than needed for double precision
puts "#{'K'.rjust(3)} | #{'e_approx'.rjust(18)} | #{'π_approx'.rjust(18)} | #{'error'.rjust(12)}"
puts "-" * 68

exp_series_incremental(1.0, max_k) do |k, e_approx|
  pi_approx = 1.0 / a_true + e_approx
  error = pi_approx - Math::PI
  printf "%3d | %18.15f | %18.15f | %+12.2e\n", k, e_approx, pi_approx, error
end

puts "Press Enter to exit"
gets