require 'complex'

class SpectralComponent
  # `f` is an array of length (k_max - k_min + 1) representing f[k]
  # or you can directly pass the convolved C array.
  def initialize(c_array)
    # c_array should be of length 2K+1, where c_array[K] = C_0
    @c = c_array
    @k = (c_array.length - 1) / 2
  end

  attr_reader :k, :c

  # O(1) element access (Ruby array indexing)
  def [](i)
    @c[i]
  end

  # Evaluate |ζ(t)|² in O(K) time
  # Assumes real symmetric C (C_{-i} = C_i)
  def zeta2_squared(t, alpha)
    factor = t / alpha
    sum = @c[@k]  # i = 0 term
    (1..@k).each do |i|
      sum += 2.0 * @c[@k + i] * Math.cos(factor * i)
    end
    sum
  end

  # Alternative complex version (preserves imaginary part)
  def zeta2(t, alpha)
    total = Complex(0,0)
    (-@k..@k).each do |i|
      total += @c[@k + i] * Complex.polar(1, t * i / alpha)
    end
    total.real
  end

  # Build C from f array via convolution (O(K²) one‑time setup)
  def self.from_f(f)
    k = f.length - 1   # f index from 0..K, where f[k] corresponds to integer k
    c_len = 2 * k + 1
    c = Array.new(c_len, 0.0)
    (0..k).each do |j|
      (0..k).each do |l|
        i = j - l
        c[k + i] += f[j] * f[l]
      end
    end
    new(c)
  end
end

# Example usage
K = 10
f = Array.new(K+1) { |i| 1.0 / Math.sqrt(i+1) }  # dummy f
spec = SpectralComponent.from_f(f)

alpha = 0.3628
t = 1.5
puts "ζ²(t) ≈ #{spec.zeta2_squared(t, alpha)}"

# O(1) access example
puts "C_0 = #{spec[spec.k]}"
puts "C_3 = #{spec[spec.k + 3]}"

require 'fiddle'
require 'fiddle/import'

class SpectralComponentC
  include Fiddle::Imports

  def initialize(c_ptr, k)
    # c_ptr is a Fiddle::Pointer to a double array of length 2*k+1
    @ptr = c_ptr
    @k = k
    @len = 2*k + 1
  end

  attr_reader :k

  # O(1) element access (index from 0 to 2K)
  def [](i)
    @ptr[i].to_f
  end

  def []=(i, value)
    @ptr[i] = value
  end

  # Evaluate in O(K) time
  def zeta2_squared(t, alpha)
    factor = t / alpha
    sum = self[@k]  # i=0 term
    (1..@k).each do |i|
      sum += 2.0 * self[@k + i] * Math.cos(factor * i)
    end
    sum
  end

  # Factory: allocate C memory and fill from Ruby array
  def self.from_ruby_array(arr)
    k = (arr.length - 1) / 2
    size = arr.length * Fiddle::SIZEOF_DOUBLE
    ptr = Fiddle::Pointer.malloc(size, Fiddle::RUBY_FREE)
    ptr.write_array_of_double(arr)
    new(ptr, k)
  end
end

# Example
arr = [1.0] * (2*10 + 1)   # all ones
spec_c = SpectralComponentC.from_ruby_array(arr)
puts spec_c.zeta2_squared(1.5, 0.3628)

def exp_series(x, k)
  sum = 1.0; term = 1.0
  (1..k).each { |n| term *= x / n; sum += term }
  sum
end

A = 1.0 / (Math::PI - exp_series(1.0, 20))   # a ≈ 2.36233

puts "Press RETURN when you're done."
gets
   