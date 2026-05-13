//! Error types for bloom filter operations.

use std::fmt;

/// Errors that can occur during bloom filter operations.
///
/// Variants are returned as the `Err` side of a [`Result`] from fallible
/// methods on [`BloomFilter`]. Currently all errors arise at construction time,
/// since insertion and membership testing are infallible. New variants will be
/// added here as additional fallible operations (such as merging filters) are
/// introduced.
///
/// [`BloomFilter`]: crate::BloomFilter
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum BloomError {
    /// The `capacity` argument was `0`.
    ///
    /// A bloom filter must be designed for at least one item. The value carried
    /// is the capacity that was provided.
    InvalidCapacity(usize),

    /// The `fpr` (false positive rate) argument was outside the range `(0, 1)`,
    /// or was `NaN` or infinite.
    ///
    /// Valid values are strictly between `0.0` and `1.0` exclusive. The value
    /// carried is the FPR that was provided.
    InvalidFpr(f64),

    /// The `bits` (or `counters`) argument to a `with_params` constructor was `0`.
    ///
    /// A filter must have at least one bit. The value carried is the count
    /// that was provided.
    InvalidBitCount(usize),

    /// The `hash_fns` argument to a `with_params` constructor was `0`.
    ///
    /// A filter must use at least one hash function. The value carried is the
    /// count that was provided.
    InvalidHashCount(usize),
}

impl PartialEq for BloomError {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::InvalidCapacity(a), Self::InvalidCapacity(b)) => a == b,
            (Self::InvalidBitCount(a), Self::InvalidBitCount(b)) => a == b,
            (Self::InvalidHashCount(a), Self::InvalidHashCount(b)) => a == b,
            // Treat NaN == NaN so that error values round-trip through equality
            // checks. This diverges from IEEE 754 float equality intentionally.
            (Self::InvalidFpr(a), Self::InvalidFpr(b)) => a == b || (a.is_nan() && b.is_nan()),
            _ => false,
        }
    }
}

impl fmt::Display for BloomError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BloomError::InvalidCapacity(n) => {
                write!(f, "capacity must be > 0, got {n}")
            }
            BloomError::InvalidFpr(p) => {
                write!(f, "false positive rate must be in (0, 1), got {p}")
            }
            BloomError::InvalidBitCount(b) => {
                write!(f, "bit count must be > 0, got {b}")
            }
            BloomError::InvalidHashCount(k) => {
                write!(f, "hash function count must be > 0, got {k}")
            }
        }
    }
}

impl std::error::Error for BloomError {}
