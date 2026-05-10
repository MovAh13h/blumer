use std::f64::consts::LN_2;

/// Optimal number of bits for a given capacity and target false positive rate.
/// Formula: m = -(n * ln(p)) / ln(2)²
pub(crate) fn optimal_bit_count(capacity: usize, fpr: f64) -> usize {
    let m = -(capacity as f64 * fpr.ln()) / (LN_2 * LN_2);
    (m.ceil() as usize).max(64)
}

/// Optimal number of hash functions given bit count and capacity.
/// Formula: k = (m / n) * ln(2)
pub(crate) fn optimal_hash_count(bits: usize, capacity: usize) -> usize {
    let k = (bits as f64 / capacity as f64) * LN_2;
    (k.round() as usize).max(1)
}

/// Current false positive rate given filter parameters and insertion count.
/// Formula: (1 - e^(-k * count / m))^k
pub(crate) fn estimated_fpr(k: usize, m: usize, count: usize) -> f64 {
    if count == 0 {
        return 0.0;
    }
    let k_f = k as f64;
    (1.0 - (-(k_f * count as f64) / m as f64).exp()).powi(k as i32)
}
