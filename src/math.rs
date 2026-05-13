use std::f64::consts::LN_2;

use crate::error::BloomError;

/// Optimal number of bits for a given capacity and target false positive rate.
/// Formula: m = -(n * ln(p)) / ln(2)²
///
/// The result is floored at 64 bits — filters smaller than one `u64` word
/// are impractical and would produce degenerate k values.
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

/// Validates `new(capacity, fpr)` arguments and returns the optimal `(m, k)`.
///
/// Centralises the validation + parameter computation shared by every filter's
/// `new` constructor, eliminating repeated boilerplate.
pub(crate) fn validated_params(
    capacity: usize,
    fpr: f64,
) -> Result<(usize, usize), BloomError> {
    if capacity == 0 {
        return Err(BloomError::InvalidCapacity(capacity));
    }
    if !fpr.is_finite() || fpr <= 0.0 || fpr >= 1.0 {
        return Err(BloomError::InvalidFpr(fpr));
    }
    let m = optimal_bit_count(capacity, fpr);
    let k = optimal_hash_count(m, capacity);
    Ok((m, k))
}

/// Validates `with_params(slots, hash_fns)` arguments and back-derives the
/// design capacity using `n = (m · ln2) / k`.
///
/// Centralises the validation + capacity derivation shared by every filter's
/// `with_params` constructor.
pub(crate) fn validated_with_params(
    slots: usize,
    hash_fns: usize,
) -> Result<usize, BloomError> {
    if slots == 0 {
        return Err(BloomError::InvalidBitCount(slots));
    }
    if hash_fns == 0 {
        return Err(BloomError::InvalidHashCount(hash_fns));
    }
    let n = ((slots as f64 * LN_2) / hash_fns as f64).round() as usize;
    Ok(n.max(1))
}
