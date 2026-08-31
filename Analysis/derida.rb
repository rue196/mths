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

# Analytical derivative dζ/dt (O(K))
def dzeta_dt(t, c, alpha)
  k = (c.length - 1) / 2
  sum = 0.0
  (1..k).each do |i|
    theta = t * i / alpha
    # derivative of C_i e^{iθ} + C_{-i} e^{-iθ}
    # = C_i * i*(i/α) e^{iθ} + C_{-i} * (-i)*(i/α) e^{-iθ}
    # Real part after simplifying: - (i/α) * (C_i sinθ + C_{-i} sinθ?) Actually let's compute correctly:
    # For real C_i = C_{-i}, the sum is 2 C_i cosθ. Derivative = -2 C_i (i/α) sinθ.
    # So:
    sum += -2.0 * c[k + i] * (i / alpha) * Math.sin(theta)
  end
  sum
end

# ------------------------------------------------------------
# Precompute example coefficients (symmetric, all ones)
# ------------------------------------------------------------
K = 20
alpha = 0.3628   # from earlier examples
c = [1.0] * (2*K + 1)   # C_i = 1 for all i

# True constant a = 1/(π - e)
a_true = 1.0 / (Math::PI - Math::E)
puts "a = 1/(π - e) = #{a_true}\n\n"

# ------------------------------------------------------------
# Compare analytical derivative vs finite difference using step a
# ------------------------------------------------------------
t0 = 1.5
analytical = dzeta_dt(t0, c, alpha)
fd_forward = (zeta(t0 + a_true, c, alpha) - zeta(t0, c, alpha)) / a_true
fd_central = (zeta(t0 + a_true, c, alpha) - zeta(t0 - a_true, c, alpha)) / (2 * a_true)

puts "At t = #{t0}"
puts "Analytical derivative:        #{analytical}"
puts "Forward difference (step = a): #{fd_forward}"
puts "Central difference (step = a): #{fd_central}"
puts "Forward error:                #{fd_forward - analytical}"
puts "Central error:                #{fd_central - analytical}"

# ------------------------------------------------------------
# Show that a acts as a "natural" step size derived from π and e
# ------------------------------------------------------------
puts "\n--- Interpretation ---"
puts "The step size a = 1/(π - e) ≈ #{a_true} comes from the identity a⁻¹ + e = π."
puts "It can be used as a derivative operator:"
puts "  D_a f(t) = (f(t+a) - f(t))/a  approximates f'(t)."
puts "The error depends on the second derivative and a."

# ------------------------------------------------------------
# Constant a = 1/(π - e) using high‑precision Ruby constants
# ------------------------------------------------------------
A = 1.0 / (Math::PI - Math::E)
puts "a = 1/(π - e) = #{A}"
puts

# Analytical value of the integral I = a * ln(1 + 1/a)
I_exact = A * Math.log(1 + 1.0/A)
puts "Analytical I = a * ln(1 + 1/a) = #{I_exact}"
puts

# ------------------------------------------------------------
# Numerical integration of ∫₀¹ a/(x+a) dx using Riemann sum
# (O(K) operations, K = number of subintervals)
# ------------------------------------------------------------
def integrate_a_over_x_plus_a(a, k)
  # Integrate f(x) = a / (x + a) from 0 to 1
  dx = 1.0 / k
  sum = 0.0
  (0...k).each do |i|
    x = i * dx
    sum += a / (x + a) * dx
  end
  sum
end

# Try increasing K (number of intervals) to see convergence
puts "K    | Numerical I       | Error (numerical - analytical)"
puts "-" * 50
[10, 100, 1000, 10000, 100000].each do |k|
  num = integrate_a_over_x_plus_a(A, k)
  err = num - I_exact
  printf "%5d | %16.12f | %+12.4e\n", k, num, err
end

# ------------------------------------------------------------
# Verify the identity that the integral equals a * ln(1+1/a)
# by symbolic reasoning (already done).
puts
puts "The integral equals a * ln(1+1/a) analytically."
puts "Numerical check above confirms it (error → 0 as K increases)."

puts "Press RETURN when you're done."
gets
   