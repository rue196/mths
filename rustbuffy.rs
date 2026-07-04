fn main() {
    let k: usize = 10;                     // K
    let len = 2 * k + 1;                   // buffer length
    let mut buffer = vec![1.0_f64; len];   // allocates O(K) memory

    // O(K) iteration: sum all elements
    let sum: f64 = buffer.iter().sum();
    println!("Sum of {} elements = {}", len, sum);
}