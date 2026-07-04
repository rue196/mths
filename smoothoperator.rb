PI = Math::PI
E  = Math::E
A  = 1.0 / (PI + E)          # a = 1/(π-e) ?

# Wait: earlier we had a = 1/(π - e). But here the boundary condition gives a = 1/(π - e). Let's check.
# The user wrote a=1/(π−e). So we use that.
A = 1.0 / (PI - E)   # This is the constant from the identity a^{-1}+e=π.

# Actually we need to reconcile: In the earlier problem we had a = 1/(π - e). So we'll keep that.

# The integral approximation in O(K)
def integral_approx(k_terms)
  # We need powers: e^(2k+1) and pi^(2k+1)
  # We'll maintain pow_pi = pi^(2k+1), pow_e = e^(2k+1) starting at k=0: pi^1, e^1.
  pow_pi = PI
  pow_e  = E

  # Coefficient c_k = (-1)^k / (2^k * k! * e^{2k})
  # We'll maintain coeff = c_k, starting at k=0: 1
  coeff = 1.0
  # We'll also maintain denom = 2k+1
  denom = 1

  sum = 0.0
  k_terms.times do |k|
    # Add term: coeff / (2k+1) * (pow_e + pow_pi)
    sum += coeff / denom * (pow_e + pow_pi)

    # Update for next k:
    # coeff_{k+1} = coeff_k * (-1) / (2 * e^2 * (k+1))
    coeff *= -1.0 / (2.0 * E * E * (k + 1))
    # pow_e = e^(2k+3) = pow_e * e^2
    pow_e *= E * E
    # pow_pi = pi^(2k+3) = pow_pi * pi^2
    pow_pi *= PI * PI
    denom += 2
  end

  A * sum
end

# Exact integral of the full Gaussian (known)
def exact_integral
  # ∫_0^(π+e) a * exp(-(x-π)^2/(2e^2)) dx
  # Substitute u = (x-π)/(e*sqrt(2))
  # This is a * e * sqrt(2) * integral from u1 to u2 of exp(-u^2) du
  # where u1 = -π/(e*sqrt(2)), u2 = e/(e*sqrt(2)) = 1/sqrt(2)
  # erf function: integral of exp(-u^2) du = sqrt(pi)/2 * erf(u)
  # So result = a * e * sqrt(2) * (sqrt(pi)/2) * (erf(u2) - erf(u1))
  # = a * e * sqrt(pi/2) * (erf(1/sqrt(2)) + erf(π/(e*sqrt(2))))
  require 'erb'  # not needed, we can use Math.erf
  u1 = -PI / (E * Math.sqrt(2))
  u2 = 1.0 / Math.sqrt(2)
  factor = A * E * Math.sqrt(Math::PI / 2.0)
  factor * (Math.erf(u2) - Math.erf(u1))
end

# Test convergence
puts "K\tintegral_approx\terror"
[1, 2, 3, 5, 10, 20, 30].each do |k|
  approx = integral_approx(k)
  exact = exact_integral
  printf "%2d\t%.8f\t%+.4e\n", k, approx, approx - exact
end

puts "Press RETURN when you're done."
gets