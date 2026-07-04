/// Evaluates |ζ(t)|² ≈ Σ_{i=-K}^{K} C_i * exp(i*t*i/α)
/// where the coefficients C_i are stored in `coeff` of length 2K+1,
/// with `coeff[K]` = C_0, `coeff[K+i]` = C_i, `coeff[K-i]` = C_{-i}.
/// The coefficients are assumed real and symmetric (C_{-i} = C_i).
fn zeta_squared(t: f64, coeff: &[f64], alpha: f64) -> f64 {
    let k = (coeff.len() - 1) / 2;
    let factor = t / alpha;
    let mut sum = coeff[k]; // i = 0 term
    for i in 1..=k {
        sum += 2.0 * coeff[k + i] * (factor * i as f64).cos();
    }
    sum
}
/// Stores only the non‑negative coefficients: `c_nonneg[i] = C_i` for i = 0..K.
/// Assumes real and symmetric.
fn zeta_squared_sym(t: f64, c_nonneg: &[f64], alpha: f64) -> f64 {
    let k = c_nonneg.len() - 1;
    let factor = t / alpha;
    let mut sum = c_nonneg[0]; // i = 0
    for i in 1..=k {
        sum += 2.0 * c_nonneg[i] * (factor * i as f64).cos();
    }
    sum
}
fn main() {
    // Parameters
    let k = 10;
    let alpha = 0.3628;

    // Pre‑compute dummy |C_i| (all ones for illustration)
    let c_full: Vec<f64> = vec![1.0; 2 * k + 1];
    // Alternative: store only symmetric part
    let c_sym: Vec<f64> = vec![1.0; k + 1];

    // Evaluate at 500 points from t=0 to t=20
    let n_points = 500;
    let t_start = 0.0;
    let t_end = 20.0;

    println!("# Using full array storage (2K+1 entries)");
    for j in 0..=n_points {
        let t = t_start + (t_end - t_start) * (j as f64) / (n_points as f64);
        let val = zeta_squared(t, &c_full, alpha);
        println!("{} {}", t, val);
    }

    // Similarly using symmetric storage
    println!("\n# Using symmetric storage (K+1 entries)");
    for j in 0..=n_points {
        let t = t_start + (t_end - t_start) * (j as f64) / (n_points as f64);
        let val = zeta_squared_sym(t, &c_sym, alpha);
        println!("{} {}", t, val);
    }
}