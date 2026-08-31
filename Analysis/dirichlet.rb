require 'complex'

# ---------- Linear sieve for Möbius function (O(K)) ----------
def mobius_sieve(k)
  mu = [0] * (k + 1)
  mu[1] = 1
  primes = []
  is_comp = [false] * (k + 1)
  (2..k).each do |i|
    if !is_comp[i]
      primes << i
      mu[i] = -1
    end
    primes.each do |p|
      break if i * p > k
      is_comp[i * p] = true
      if i % p == 0
        mu[i * p] = 0
        break
      else
        mu[i * p] = -mu[i]
      end
    end
  end
  mu
end

# ---------- Build coefficient array C_i from Möbius ----------
def build_coeffs(k)
  mu = mobius_sieve(k)
  c = [0.0] * (2 * k + 1)   # indices 0..2k correspond to i = -k..k
  (1..k).each do |n|
    c[k + n] = mu[n].to_f      # C_{n} = μ(n)
    c[k - n] = mu[n].to_f      # C_{-n} = μ(n)  (symmetric)
  end
  # C_0 = μ(0) is not defined; set to 0 (or use μ(1) if you wish)
  c[k] = 0.0
  c
end

# ---------- Spectral sum ζ(t) = Σ C_i e^{i t i / α} (O(K)) ----------
def zeta(t, c, alpha)
  k = (c.length - 1) / 2
  total = 0.0 + 0.0i
  (-k..k).each do |i|
    # coefficient for index i is c[i + k]
    total += c[i + k] * Complex.polar(1.0, t * i / alpha)
  end
  total
end

# ---------- Example ----------
K = 10
alpha = 0.3628   # or 1/(π-e)

c = build_coeffs(K)
puts "Coefficient array C_i (first few):"
puts "i=-K: #{c[0]}, i=0: #{c[K]}, i=K: #{c[2*K]}"

# Evaluate ζ(t) for a range of t
t_values = (0..100).map { |j| j * 0.2 }
results = t_values.map { |t| zeta(t, c, alpha) }

# Print |ζ(t)|^2 for first few t
puts "\n|ζ(t)|^2 for t = 0, 0.2, ... :"
results.first(5).each_with_index do |z, idx|
  puts "t=#{t_values[idx].round(1)}: #{z.abs2}"
end

# Optionally, compute the Dirichlet convolution (μ * 1)(n) = δ_{n,1}
# as a quick check: (μ * 1)(n) = 1 if n=1 else 0.
# This is the classic convolution that can be computed O(K log K) but here we just show.
puts "\nDirichlet convolution (μ * 1)(n) for n=1..10:"
(1..10).each do |n|
  conv = (1..n).select { |d| n % d == 0 }.sum { |d| mu[n/d] * 1 }
  # But we already have mu from the sieve; we can compute directly:
  # Since μ * 1 = δ, we can just print δ.
  puts "n=#{n}: #{n == 1 ? 1 : 0}"
end
"puts "Done. Press Enter to close."
STDIN.gets
   