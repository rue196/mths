require 'complex'

# ------------------------------------------------------------
# 1. Letter probabilities and spectral coefficients
# ------------------------------------------------------------
def letter_probabilities(str)
  freq = Hash.new(0)
  str.each_char { |c| freq[c] += 1 }
  total = str.length.to_f
  probs = {}
  freq.each { |c, cnt| probs[c] = cnt / total }
  probs
end

def build_spectral_coeffs(probs)
  letters = probs.keys
  k = letters.size
  c = Array.new(2*k + 1, Complex(0,0))
  letters.each_with_index do |letter, idx|
    i = idx + 1
    p = probs[letter]
    phase = -p * Math.log(p)   # entropy contribution
    c[k + i] = Complex(p * Math.cos(phase), p * Math.sin(phase))
    c[k - i] = c[k + i].conj
  end
  [c, letters]
end

# ------------------------------------------------------------
# 2. Group letters by probability (or by full complex coefficient)
# ------------------------------------------------------------
def group_by_probability(probs)
  groups = Hash.new { |h, k| h[k] = [] }
  probs.each { |letter, p| groups[p] << letter }
  groups
end

# ------------------------------------------------------------
# 3. Linear-time counting sort on probability values
#    Assumes probabilities are multiples of 1/L where L = total letters.
# ------------------------------------------------------------
def counting_sort_by_probability(groups, total_letters)
  # probability -> integer numerator = p * total_letters
  max_num = total_letters
  buckets = Array.new(max_num + 1) { [] }
  groups.each do |p, letters|
    numerator = (p * total_letters).round   # exact integer
    buckets[numerator] = letters
  end
  # Flatten in increasing numerator order (i.e., increasing probability)
  buckets.flat_map { |letters| letters }
end

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
text = "hello spectral world"
probs = letter_probabilities(text)
total_len = text.length
groups = group_by_probability(probs)

puts "Groups (probability -> letters):"
groups.each { |p, letters| puts "  p=#{p.round(4)} : #{letters}" }

sorted_letters = counting_sort_by_probability(groups, total_len)
puts "\nLetters sorted by probability (ascending):"
puts sorted_letters.inspect

# Build spectral array (as before)
c_array, letters_order = build_spectral_coeffs(probs)
puts "\nFirst few spectral coefficients (C_i for i=1..5):"
k = (c_array.length - 1) / 2
(1..[5,k].min).each do |i|
  puts "  C_#{i} = #{c_array[k + i]}"
end

def linear_sort_by_probability(groups, total_letters)
  # Use a hash: numerator -> list of letters
  buckets = {}
  groups.each do |p, letters|
    numerator = (p * total_letters).round
    buckets[numerator] = letters
  end
  # Sort only the keys that exist (at most K keys)
  buckets.keys.sort.map { |num| buckets[num] }.flatten
end

puts "Press RETURN when you're done."
gets
   