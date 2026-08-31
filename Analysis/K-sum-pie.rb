#!/usr/bin/env ruby
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
# ------------------------------------------------------------
# Helper: factorial (for coefficients)
# ------------------------------------------------------------
def factorial(n)
  (1..n).reduce(1, :*)
end

# ------------------------------------------------------------
# Integral of an even‑power series:
#   f(x) = Σ_{k=0}^{K-1} c(k) * x^(2k)
#   ∫_0^L f(x) dx = Σ c(k) * L^(2k+1) / (2k+1)
# mask_even_k: if true, only include k where k is even (Collatz mask)
# ------------------------------------------------------------
def integral_even_series(c_func, l, k, mask_even_k: false)
  total = 0.0
  pow_l = l.to_f          # L^(2k+1) for k=0 => L^1
  (0...k).each do |k_idx|
    unless mask_even_k && k_idx.odd?
      ck = c_func.call(k_idx)
      total += ck * pow_l / (2 * k_idx + 1)
    end
    pow_l *= l * l        # L^(2(k+1)+1) = L^(2k+3)
  end
  total
end

# ------------------------------------------------------------
# Integral of a polynomial: P(x) = Σ a_n * x^n
#   ∫_0^L P(x) dx = Σ a_n * L^(n+1) / (n+1)
# mask_even_only: if true, skip odd n
# ------------------------------------------------------------
def polynomial_integral(coeffs, l, mask_even_only: false)
  total = 0.0
  pow_l = l.to_f          # L^(1) for n=0
  coeffs.each_with_index do |a, n|
    unless mask_even_only && n.odd?
      total += a * pow_l / (n + 1)
    end
    pow_l *= l            # next power: L^(n+2)
  end
  total
end

# ------------------------------------------------------------
# Example 1: Gaussian e^{-x^2}, coefficients c(k) = (-1)^k / k!
# ------------------------------------------------------------
gaussian_coeff = ->(k) { (-1)**k / factorial(k).to_f }

l = 2.0
k_terms = 20
ig_full = integral_even_series(gaussian_coeff, l, k_terms, mask_even_k: false)
ig_collatz = integral_even_series(gaussian_coeff, l, k_terms, mask_even_k: true)

# Reference: use a very high‑order series (K=50) as "exact"
exact_gaussian = integral_even_series(gaussian_coeff, l, 50, mask_even_k: false)

puts "Gaussian integral ∫₀^2 e^{-x²} dx"
puts "  Reference (K=50): #{exact_gaussian}"
puts "  Full series (K=20): #{ig_full}  error: #{ig_full - exact_gaussian}"
puts "  Collatz mask (K=20): #{ig_collatz}  error: #{ig_collatz - exact_gaussian}"
puts

# ------------------------------------------------------------
# Example 2: Sinc sin(x)/x, coefficients c(k) = (-1)^k / (2k+1)!
# ------------------------------------------------------------
sinc_coeff = ->(k) { (-1)**k / factorial(2*k + 1).to_f }

l = 3.0
k_terms = 15
is_full = integral_even_series(sinc_coeff, l, k_terms, mask_even_k: false)
is_collatz = integral_even_series(sinc_coeff, l, k_terms, mask_even_k: true)
exact_sinc = integral_even_series(sinc_coeff, l, 50, mask_even_k: false)

puts "Sinc integral ∫₀³ sin(x)/x dx"
puts "  Reference (K=50): #{exact_sinc}"
puts "  Full series (K=15): #{is_full}  error: #{is_full - exact_sinc}"
puts "  Collatz mask (K=15): #{is_collatz}  error: #{is_collatz - exact_sinc}"
puts

# ------------------------------------------------------------
# Example 3: Polynomial with random coefficients (degree 20)
# ------------------------------------------------------------
# Generate random coefficients
srand(42)
n_deg = 20
coeffs = Array.new(n_deg + 1) { rand(-2.0..2.0) }

l = 1.5
ip_full = polynomial_integral(coeffs, l, mask_even_only: false)
ip_even = polynomial_integral(coeffs, l, mask_even_only: true)

# The exact integral is just the full series (since polynomial is finite)
ip_exact = ip_full

puts "Polynomial ∫₀^1.5 P(x) dx, degree 20"
puts "  Exact (full): #{ip_exact}"
puts "  Full series:  #{ip_full}  error: #{ip_full - ip_exact}"
puts "  Even‑only:    #{ip_even}  error: #{ip_even - ip_exact}"
puts

# ------------------------------------------------------------
# Additional: series_with_modulo from earlier (even‑only k)
# ------------------------------------------------------------
def series_with_modulo(u, k_terms)
  u2 = u * u
  total = 0.0
  pow_u = 1.0
  coeff = 1.0
  e = Math::E
  (0...k_terms).each do |k|
    if k.even?
      total += coeff * pow_u
    end
    pow_u *= u2
    coeff *= -1.0 / (2.0 * e * e * (k + 1))
  end
  total * (1.0 / (Math::PI + e))
end

puts "series_with_modulo(1.0, 10) = #{series_with_modulo(1.0, 10)}"

# ------------------------------------------------------------
# Demonstration of mod_using_div (division‑based modulo)
# ------------------------------------------------------------
def mod_using_div(a, b)
  return 0 if b == 0
  a - (a / b) * b
end

puts "mod_using_div(7,3) = #{mod_using_div(7, 3)}"

# ============================================================
# ADDITION: Definite and indefinite integration using the series
# ============================================================

# Indefinite integral (antiderivative) of the series:
#   ∫ g(π+u) du ≈ a * Σ_{k=0}^{K-1} c_k * u^(2k+1) / (2k+1)
# where c_k = (-1)^k / (2^k * k! * e^(2k))
# Returns the value of the antiderivative at u.
def indefinite_series(u, k_terms)
  x = - (u * u) / (2.0 * E * E)   # same x as in series_approx
  # We need Σ (x^k / k!) * u / (2k+1)? Actually easier to compute directly:
  # term_k = c_k * u^(2k+1) / (2k+1)
  # We can update term iteratively:
  # term_{k+1} = term_k * ( -u^2 / (2*e^2*(k+1)) ) * (2k+1)/(2k+3) ?
  # Or we can compute using the power recurrence we already have.
  # We'll just build the sum with powers.
  sum = 0.0
  pow_u = u          # u^(2k+1) for k=0 => u^1
  coeff = 1.0        # c_0 = 1
  (0...k_terms).each do |k|
    sum += coeff * pow_u / (2*k + 1)
    # update for next k:
    pow_u *= u * u           # u^(2k+3)
    coeff *= -1.0 / (2.0 * E * E * (k + 1))
  end
  A * sum   # multiply by a = 1/(π+e)
end

# Definite integral: ∫_a^b g(x) dx  (with x in original coordinates)
# We use u = x - π, so the antiderivative is indefinite_series(u, K)
def definite_integral(a, b, k_terms)
  u_a = a - PI
  u_b = b - PI
  indefinite_series(u_b, k_terms) - indefinite_series(u_a, k_terms)
end

# Demonstration
puts "\n--- Integration examples ---"

# Test indefinite integral at u = 0: should be 0 (since the series has only odd powers)
puts "Indefinite at u=0: #{indefinite_series(0.0, 10)} (should be ~0)"

# Definite integral from 0 to π+e (i.e., x=0 to x=π+e)
a = 0.0
b = PI + E
K_int = 20
I_approx = definite_integral(a, b, K_int)
# Reference: we can compute using the exact Gaussian integral (with erf)
def exact_integral_gaussian(a, b)
  # ∫ a * exp(-(x-π)^2/(2e^2)) dx = a * e * sqrt(pi/2) * ( erf((b-π)/(e*sqrt(2))) - erf((a-π)/(e*sqrt(2))) )
  factor = A * E * Math.sqrt(Math::PI / 2.0)
  u_a = (a - PI) / (E * Math.sqrt(2.0))
  u_b = (b - PI) / (E * Math.sqrt(2.0))
  factor * (Math.erf(u_b) - Math.erf(u_a))
end
I_exact = exact_integral_gaussian(a, b)

puts "Definite integral from 0 to π+e:"
puts "  Approx (K=#{K_int}): #{I_approx}"
puts "  Exact (erf):        #{I_exact}"
puts "  Error:              #{I_approx - I_exact}"

# Also test with modulo version (even k only)
def indefinite_series_modulo(u, k_terms)
  u2 = u * u
  sum = 0.0
  pow_u = u          # u^(2k+1)
  coeff = 1.0
  (0...k_terms).each do |k|
    if k.even?   # only include even k (Collatz mask)
      sum += coeff * pow_u / (2*k + 1)
    end
    pow_u *= u2
    coeff *= -1.0 / (2.0 * E * E * (k + 1))
  end
  A * sum
end

def definite_integral_modulo(a, b, k_terms)
  u_a = a - PI
  u_b = b - PI
  indefinite_series_modulo(u_b, k_terms) - indefinite_series_modulo(u_a, k_terms)
end

I_mod = definite_integral_modulo(a, b, 20)
puts "\nWith Collatz mask (even k only):"
puts "  Approx: #{I_mod}"
puts "  Error:  #{I_mod - I_exact}"

def exp_series_incremental(x, k)
  sums = [1.0]          # S_0 = 1
  term = 1.0
  (1..k).each do |n|
    term *= x / n       # term = x^n / n!
    sums << sums.last + term
  end
  sums
end

partials = exp_series_incremental(1.0, 5)
partials.each_with_index { |s, n| puts "S_#{n} = #{s}" }

def exp_series_each(x, k)
  sum = 1.0
  term = 1.0
  yield sum            # n = 0
  (1..k).each do |n|
    term *= x / n
    sum += term
    yield sum
  end
end

exp_series_each(1.0, 5) do |s|
  puts s
end

def exp_series(x, k)
  sum = 1.0
  term = 1.0
  if block_given?
    yield sum, 0
  end
  (1..k).each do |n|
    term *= x / n
    sum += term
    yield sum, n if block_given?
  end
  sum
end
