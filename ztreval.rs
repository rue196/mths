/// Evaluate the compressed spectral sum in O(K) time.
///
/// # Arguments
/// * `t` - time variable
/// * `c` - slice of length `2*K+1`, where `c[K]` corresponds to i=0,
///        `c[K+i]` corresponds to C_i for i=1..K,
///        and `c[K-i]` corresponds to C_{-i} (must equal `c[K+i]` for symmetry)
/// * `k` - half-length (max index)
/// * `alpha` - scaling factor (α > 0)
///
/// # Returns
/// Real part of Σ_{i=-K}^{K} C_i * exp(i*t*i/α)
fn zeta2_squared(t: f64, c: &[f64], k: usize, alpha: f64) -> f64 {
    // i = 0 term
    let mut sum = c[k];
    let factor = t / alpha;
    for i in 1..=k {
        // Use symmetry: C_{-i} = C_i
        sum += 2.0 * c[k + i] * (factor * i as f64).cos();
    }
    sum
}

fn main() {
    let k = 10;                 // spectral length (max index)
    let alpha = 0.3628;         // from the original Python example

    // Allocate C vector of length 2*K+1 and fill with ones (symmetric)
    let len = 2 * k + 1;
    let c = vec![1.0; len];

    // Evaluate at several t values (linspace 0..20, 500 points)
    let n_points = 500;
    let t_start = 0.0;
    let t_end = 20.0;
    for j in 0..=n_points {
        let t = t_start + (t_end - t_start) * (j as f64) / (n_points as f64);
        let val = zeta2_squared(t, &c, k, alpha);
        println!("{} {}", t, val);
    }
}