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
    sum += sign / (2*n + 1).to_f
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

# exp_series.rb
def exp_series(x, k)
  sum = 1.0
  term = 1.0
  (1..k).each do |n|
    term *= x / n
    sum += term
  end
  sum
end

def pi_series(k)
  sum = 0.0
  sign = 1.0
  (0..k).each do |n|
    sum += sign / (2*n + 1)
    sign = -sign
  end
  4.0 * sum
end

K = 1000
puts "exp(1) = #{exp_series(1.0, K)}"
puts "π      = #{pi_series(K)}"
puts "\nPress Enter to exit."
gets

# O(K) Taylor series for exp(x)
def exp_series(x, k)
  sum = 1.0
  term = 1.0
  (1..k).each do |n|
    term *= x / n
    sum += term
  end
  sum
end

# Approximate pi using |2.31975^-1 + e|
k = 1000                     # number of terms (O(K) time)
e_approx = exp_series(1.0, k)
pi_approx = (2.31975 ** -1 + e_approx).abs   # absolute value (already positive)

puts "Approximated π = #{pi_approx}"
puts "Actual π        = #{Math::PI}"
puts "Error           = #{pi_approx - Math::PI}"


# O(K) Taylor series for exp(x)
def exp_series(x, k)
  sum = 1.0
  term = 1.0
  (1..k).each do |n|
    term *= x / n
    sum += term
  end
  sum
end

# Parameters
K = 1000               # number of terms → O(K) operations
e_approx = exp_series(1.0, K)

# Solve for a:  1/a + e = π  →  a = 1/(π - e)
pi_exact = Math::PI
a = 1.0 / (pi_exact - e_approx)

# Verify the equation
left_side = 1.0 / a + e_approx

puts "Computed e (K=#{K}) : #{e_approx}"
puts "a                     = #{a}"
puts "Check: 1/a + e        = #{left_side}"
puts "Actual π              = #{pi_exact}"
puts "Error                 = #{left_side - pi_exact}"

# Compute e^x using incremental multiplication – O(K), no overflow
def exp_series(x, k)
  sum = 1.0      # term for n=0
  term = 1.0     # current term = x^n / n!
  (1..k).each do |n|
    term *= x / n   # update from previous term
    sum += term
  end
  sum
end

def solve_a(k)
  e_approx = exp_series(1.0, k)   # O(K)
  a = 1.0 / (Math::PI - e_approx) # O(1)
  a
end

# Verify the equation
k = 1000
a = solve_a(k)
e_val = exp_series(1.0, k)
lhs = 1.0 / a + e_val

puts "a = #{a}"
puts "1/a + e = #{lhs}"
puts "π = #{Math::PI}"
puts "Error = #{lhs - Math::PI}"

def pi_leibniz(k)
  sum = 0.0
  sign = 1.0
  (0..k).each do |n|
    sum += sign / (2*n + 1)
    sign = -sign
  end
  4.0 * sum
end

k = 10000                # Leibniz converges slowly
e_approx = exp_series(1.0, k)
pi_approx = pi_leibniz(k)
a = 1.0 / (pi_approx - e_approx)

lhs = 1.0 / a + e_approx
puts "Using computed π: #{pi_approx}"
puts "a = #{a}, 1/a+e = #{lhs}, error = #{lhs - pi_approx}

"puts "Done. Press Enter to close."
STDIN.gets