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
t_values = (0..500).map { |j| j * 20.0 / 2000 }
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



# ------------------------------------------------------------
# O(K) series approximations (incremental, no overflow)
# ------------------------------------------------------------

# exp(x) Taylor series: returns [e_approx, error] for K terms
def exp_series(x, k)
  sum = 1.0
  term = 1.0
  (1..k).each do |n|
    term *= x / n
    sum += term
  end
  [sum, sum - Math::E]
end

# π Leibniz series: returns [π_approx, error] for K terms
def pi_series(k)
  sum = 0.0
  sign = 1.0
  (0..k).each do |n|
    sum += sign / (2*n + 1)
    sign = -sign
  end
  pi_approx = 4.0 * sum
  [pi_approx, pi_approx - Math::PI]
end

# True constants
TRUE_E = Math::E
TRUE_PI = Math::PI
TRUE_A = 1.0 / (TRUE_PI - TRUE_E)

puts "True constants:"
puts "  e = #{TRUE_E}"
puts "  π = #{TRUE_PI}"
puts "  a = 1/(π - e) = #{TRUE_A}"
puts

# ------------------------------------------------------------
# Explore how a_n behaves as K increases (same K for both series)
# ------------------------------------------------------------
puts "K | e_approx error       | π_approx error      | a_approx            | 1/a_approx + e_approx (should equal π_approx)"
puts "-" * 100

(1..20).each do |k|
  e_ap, e_err = exp_series(1.0, k)
  pi_ap, pi_err = pi_series(k)
  a_ap = 1.0 / (pi_ap - e_ap)
  lhs = 1.0 / a_ap + e_ap   # by construction = pi_ap (except rounding)
  printf "%2d | %+12.4e     | %+12.4e     | %15.10f | %15.10f\n",
         k, e_err, pi_err, a_ap, lhs
end

puts "\nNote: 1/a_ap + e_ap exactly equals π_ap (invariant)."
puts "      a_ap converges to true a = #{TRUE_A} as K increases."

# ------------------------------------------------------------
# Demonstration of the invariant for arbitrary approximations
# ------------------------------------------------------------
puts "\n=== Invariance demonstration ==="
# Choose any two approximations (different K for e and π)
k_e = 5
k_pi = 10
e_ap, _ = exp_series(1.0, k_e)
pi_ap, _ = pi_series(k_pi)
a_ap = 1.0 / (pi_ap - e_ap)

puts "e approximated with #{k_e} terms  = #{e_ap}"
puts "π approximated with #{k_pi} terms = #{pi_ap}"
puts "a = 1/(π - e) = #{a_ap}"
puts "1/a + e = #{1.0/a_ap + e_ap}"
puts "which exactly equals π_ap (by algebra) -> #{(1.0/a_ap + e_ap) == pi_ap}"
puts

puts "Press Enter to exit"
gets