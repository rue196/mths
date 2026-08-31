require 'complex'

# ------------------------------------------------------------
# 1. Compute letter probabilities from a given string
# ------------------------------------------------------------
def letter_probabilities(str)
  freq = Hash.new(0)
  str.each_char { |c| freq[c] += 1 }
  total = str.length.to_f
  probabilities = {}
  freq.each { |c, count| probabilities[c] = count / total }
  probabilities
end

# ------------------------------------------------------------
# 2. Build spectral coefficients C (complex) for indices -K..K
#    We map each letter to a unique index i (1..K)
#    C[i] = p_i * exp(i * phi_i) where phi_i = -p_i * log(p_i)
#    C[-i] = conj(C[i]) to keep the total sum real if desired,
#    but we'll skip symmetry to keep imaginary part non‑zero.
# ------------------------------------------------------------
def build_spectral_coefficients(probabilities)
  letters = probabilities.keys
  k = letters.size
  c = Array.new(2*k + 1, Complex(0,0))
  
  letters.each_with_index do |letter, idx|
    i = idx + 1                # indices 1..K
    p = probabilities[letter]
    entropy_contrib = -p * Math.log(p)   # Shannon term (natural log)
    # Phase chosen to encode the entropy contribution
    # We set C_i = p * exp(i * entropy_contrib)   (pure imaginary exponent)
    # The imaginary part of the total sum will then relate to total entropy.
    c[k + i] = Complex(p * Math.cos(entropy_contrib), p * Math.sin(entropy_contrib))
    c[k - i] = c[k + i].conj   # make it Hermitian if you want real ζ(t)
  end
  c
end

# ------------------------------------------------------------
# 3. Complex spectral sum ζ(t) = Σ_{i=-K}^{K} C_i * exp(i t i/α)
#    O(K) evaluation, returns a Complex number.
# ------------------------------------------------------------
def zeta_complex(t, c, alpha)
  k = (c.length - 1) / 2
  sum = c[k]   # i = 0 term
  (1..k).each do |i|
    theta = t * i / alpha
    exp_pos = Complex.polar(1,  theta)
    exp_neg = Complex.polar(1, -theta)
    sum += c[k + i] * exp_pos + c[k - i] * exp_neg
  end
  sum
end

# ------------------------------------------------------------
# 4. Exact Shannon entropy of the letter distribution
# ------------------------------------------------------------
def shannon_entropy(probabilities)
  probabilities.values.reduce(0) do |h, p|
    h - p * Math.log(p)   # natural log
  end
end

# ------------------------------------------------------------
# Example: fit letters of a sentence into the spectral sum
# ------------------------------------------------------------
text = "hello spectral world"
probs = letter_probabilities(text)
puts "Letter probabilities:"
probs.each { |c, p| puts "  '#{c}' => %.4f" % p }
puts

c_array = build_spectral_coefficients(probs)
k = (c_array.length - 1) / 2
puts "Spectral array length: #{c_array.length} (2K+1, K=#{k})"
puts

alpha = 0.3628   # from original example
t_test = 1.0
zeta_val = zeta_complex(t_test, c_array, alpha)

exact_entropy = shannon_entropy(probs)
imag_part = zeta_val.imag

puts "At t = #{t_test}:"
puts "  ζ(t) = #{zeta_val}"
puts "  Imaginary part of ζ(t) = #{imag_part}"
puts "  Exact Shannon entropy (natural) = #{exact_entropy}"
puts "  Difference: #{imag_part - exact_entropy}"

require 'complex'

# ------------------------------------------------------------
# Helper: get the n-th bit (0-indexed) of the fractional part of a float
# ------------------------------------------------------------
def fractional_bit(x, n)
  # Extract fractional part, multiply by 2^(n+1), then check the n-th fractional bit
  frac = x - x.floor
  (frac * (2**(n+1))).to_i & 1
end

# ------------------------------------------------------------
# Build spectral array C from bits of two irrationals (pi and e)
# K = number of bits used (spectral length = 2K+1)
# C is made symmetric: C[i] = (bit_pi[i] + bit_e[i]) / 2.0  for i>0
# ------------------------------------------------------------
def build_spectral_from_bits(irr1, irr2, k)
  c = Array.new(2*k + 1, 0.0)
  (1..k).each do |i|
    bit1 = fractional_bit(irr1, i-1)   # 0 or 1
    bit2 = fractional_bit(irr2, i-1)
    value = (bit1 + bit2) / 2.0        # can be 0, 0.5, or 1.0
    c[k + i] = value
    c[k - i] = value                   # symmetric
  end
  c
end

# ------------------------------------------------------------
# Spectral sum (real symmetric version, returns real)
# ------------------------------------------------------------
def zeta(t, c, alpha)
  k = (c.length - 1) / 2
  sum = c[k].to_c                     # i=0 term (always zero in our build)
  (1..k).each do |i|
    theta = t * i / alpha
    sum += c[k + i] * Complex.polar(1,  theta)
    sum += c[k - i] * Complex.polar(1, -theta)
  end
  sum.real
end

# ------------------------------------------------------------
# Analytical derivative (real symmetric version)
# ------------------------------------------------------------
def dzeta_dt(t, c, alpha)
  k = (c.length - 1) / 2
  sum = 0.0
  (1..k).each do |i|
    theta = t * i / alpha
    # derivative of 2 * C_i * cos(theta) = -2 C_i (i/α) sin(theta)
    sum += -2.0 * c[k + i] * (i / alpha) * Math.sin(theta)
  end
  sum
end

# ------------------------------------------------------------
# Main: use bits of π and e up to K bits, compute derivative and FD
# ------------------------------------------------------------
K = 20                     # bit length (spectral size 2K+1)
alpha = 0.3628             # from original problem
a_true = 1.0 / (Math::PI - Math::E)

# Build C array from binary expansions of π and e
c = build_spectral_from_bits(Math::PI, Math::E, K)

puts "Spectral array built from first #{K} bits of π and e (fractional parts)."
puts "a = 1/(π - e) = #{a_true}\n\n"

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

# Optional: print first few C_i
puts "\nFirst 5 C_i (i=1..5):"
(1..5).each do |i|
  puts "C_#{i} = #{c[K + i]}"
end

require 'complex'

# ------------------------------------------------------------
# 1. Letter probabilities from a string
# ------------------------------------------------------------
def letter_probabilities(str)
  freq = Hash.new(0)
  str.each_char { |c| freq[c] += 1 }
  total = str.length.to_f
  probs = {}
  freq.each { |c, cnt| probs[c] = cnt / total }
  probs
end

# ------------------------------------------------------------
# 2. Build complex spectral array C (length 2K+1, K = number of distinct letters)
#    |C_i| = probability p_i
#    arg(C_i) = -p_i * log(p_i)   (entropy contribution)
#    We place positive indices for letters, negative indices get conjugate.
# ------------------------------------------------------------
def build_spectral_from_probs(probs)
  letters = probs.keys
  k = letters.size
  c = Array.new(2*k + 1, Complex(0,0))
  letters.each_with_index do |letter, idx|
    i = idx + 1                     # index from 1 to K
    p = probs[letter]
    phase = -p * Math.log(p)        # entropy term (natural log)
    c[k + i] = Complex(p * Math.cos(phase), p * Math.sin(phase))
    c[k - i] = c[k + i].conj        # Hermitian symmetry (optional)
  end
  c
end

# ------------------------------------------------------------
# 3. Spectral sum ζ(t) = Σ_{i=-K}^{K} C_i e^{i t i/α}  (complex)
# ------------------------------------------------------------
def zeta_complex(t, c, alpha)
  k = (c.length - 1) / 2
  sum = c[k]   # i = 0 term (usually 0)
  (1..k).each do |i|
    theta = t * i / alpha
    sum += c[k + i] * Complex.polar(1,  theta)
    sum += c[k - i] * Complex.polar(1, -theta)
  end
  sum
end

# ------------------------------------------------------------
# 4. Analytical derivative dζ/dt (complex)
# ------------------------------------------------------------
def dzeta_dt_complex(t, c, alpha)
  k = (c.length - 1) / 2
  sum = Complex(0,0)
  (1..k).each do |i|
    theta = t * i / alpha
    # derivative of C_i e^{iθ} is C_i * i*(i/α) e^{iθ}
    # derivative of C_{-i} e^{-iθ} is C_{-i} * (-i)*(i/α) e^{-iθ}
    term = c[k + i] * Complex(0, i/alpha) * Complex.polar(1,  theta)
    term += c[k - i] * Complex(0, -i/alpha) * Complex.polar(1, -theta)
    sum += term
  end
  sum
end

# ------------------------------------------------------------
# 5. Finite difference using step a = 1/(π - e)
# ------------------------------------------------------------
text = "hello spectral world"
probs = letter_probabilities(text)
puts "Letter probabilities:"
probs.each { |c, p| puts "  '#{c}' => %.4f" % p }
puts

c_array = build_spectral_from_probs(probs)
k = (c_array.length - 1) / 2
puts "Spectral array length: #{c_array.length} (2K+1, K=#{k})"
puts "First few C_i (i=1..5):"
(1..[5,k].min).each do |i|
  puts "  C_#{i} = #{c_array[k + i]}"
end

alpha = 0.3628
a_true = 1.0 / (Math::PI - Math::E)
t0 = 1.5

zeta_t0 = zeta_complex(t0, c_array, alpha)
deriv_analytical = dzeta_dt_complex(t0, c_array, alpha)

fd_forward = (zeta_complex(t0 + a_true, c_array, alpha) - zeta_t0) / a_true
fd_central = (zeta_complex(t0 + a_true, c_array, alpha) - zeta_complex(t0 - a_true, c_array, alpha)) / (2 * a_true)

puts "\nAt t = #{t0}"
puts "ζ(t) = #{zeta_t0}"
puts "Analytical derivative:        #{deriv_analytical}"
puts "Forward difference (step = a): #{fd_forward}"
puts "Central difference (step = a): #{fd_central}"
puts "Forward error (abs):           #{(fd_forward - deriv_analytical).abs}"
puts "Central error (abs):           #{(fd_central - deriv_analytical).abs}"



puts "Press RETURN when you're done."
gets

# Modulo using division and subtraction (no built‑in %)
def mod_using_div(a, b)
  return 0 if b == 0
  # For positive integers (Ruby's integer division truncates toward −∞,
  # but we use it for non‑negative a,b so it works like floor).
  a - (a / b) * b
end

# Test: circular buffer indexing with K slots, running K steps
K = 10
head = 0
K.times do |step|
  idx = mod_using_div(head, K)   # equivalent to head % K
  puts "Step #{step}: head = #{head}, idx = #{idx}"
  head += 1   # simulate advancing
end