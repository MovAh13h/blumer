//! Counting bloom filter — a standard bloom filter extended with deletion.

use crate::error::BloomError;
use crate::hash::{bit_positions, hash_pair};
use crate::math::{estimated_fpr, optimal_bit_count, optimal_hash_count};
use crate::traits::{Filter, MutableFilter, RemovableFilter};
use crate::Bloomable;

/// A counting bloom filter that supports deletion.
///
/// Extends a standard bloom filter by replacing each 1-bit slot with a `u8`
/// counter. Insertion increments counters; deletion decrements them. An item
/// is considered present when all of its `k` counters are non-zero.
///
/// # Counter overflow
///
/// Each counter saturates at 255. Overflow requires 255 distinct items to hash
/// to the exact same `k` positions simultaneously — astronomically unlikely at
/// any sane load. If it does occur, the saturated counter is never decremented,
/// so deletion silently stops working for that slot. Resize the filter before
/// reaching this state.
///
/// # Deletion correctness
///
/// Only remove items you have previously inserted. Removing an item that was
/// never inserted decrements counters shared with other items, causing
/// **false negatives**. The filter has no way to detect this; the contract is
/// enforced by the caller.
///
/// # Memory usage
///
/// Each counter slot occupies one byte, so a counting filter uses 8× the
/// memory of an equivalent standard filter.
///
/// | Items (`n`) | FPR | RAM       |
/// |-------------|-----|-----------|
/// | 1 000       | 1%  | ~9.4 KB   |
/// | 100 000     | 1%  | ~938 KB   |
/// | 1 000 000   | 1%  | ~9.4 MB   |
///
/// # Examples
///
/// ```rust
/// use blume::prelude::*;
///
/// let mut filter = CountingBloomFilter::new(1_000, 0.01).unwrap();
///
/// filter.insert("alice");
/// filter.insert("bob");
///
/// assert!(filter.contains("alice"));
/// assert!(filter.remove("alice"));    // returns true — was present
/// assert!(!filter.contains("alice")); // now absent
/// assert!(filter.contains("bob"));    // unaffected
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CountingBloomFilter {
    counters: Vec<u8>,
    k: usize,
    m: usize,
    n: usize,
    count: usize,
}

impl CountingBloomFilter {
    /// Creates a counting bloom filter optimised for `capacity` items at the
    /// given false positive rate.
    ///
    /// `m` (counter count) and `k` (hash function count) are derived using
    /// the same formulas as [`BloomFilter::new`]:
    ///
    /// ```text
    /// m = -(n · ln(p)) / ln(2)²
    /// k =  (m / n)    · ln(2)
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`BloomError::InvalidCapacity`] if `capacity` is `0`.
    /// Returns [`BloomError::InvalidFpr`] if `fpr` is not in `(0, 1)` or is
    /// `NaN` or infinite.
    ///
    /// [`BloomFilter::new`]: crate::BloomFilter::new
    ///
    /// # Examples
    ///
    /// ```rust
    /// use blume::prelude::*;
    ///
    /// let filter = CountingBloomFilter::new(10_000, 0.01).unwrap();
    ///
    /// assert!(CountingBloomFilter::new(0, 0.01).is_err());
    /// assert!(CountingBloomFilter::new(100, 0.0).is_err());
    /// ```
    pub fn new(capacity: usize, fpr: f64) -> Result<Self, BloomError> {
        if capacity == 0 {
            return Err(BloomError::InvalidCapacity(capacity));
        }
        if !fpr.is_finite() || fpr <= 0.0 || fpr >= 1.0 {
            return Err(BloomError::InvalidFpr(fpr));
        }
        let m = optimal_bit_count(capacity, fpr);
        let k = optimal_hash_count(m, capacity);
        Ok(Self::raw(m, k, capacity))
    }

    /// Creates a counting bloom filter with explicit counter count and hash
    /// function count.
    ///
    /// Prefer [`CountingBloomFilter::new`] for normal use. This constructor
    /// is intended for cases where you need to match the exact geometry of a
    /// previously constructed filter — for example, when rehydrating from
    /// storage.
    ///
    /// # Errors
    ///
    /// Returns [`BloomError::InvalidBitCount`] if `counters` is `0`.
    /// Returns [`BloomError::InvalidHashCount`] if `hash_fns` is `0`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use blume::prelude::*;
    ///
    /// let mut filter = CountingBloomFilter::with_params(9_585, 7).unwrap();
    /// filter.insert("hello");
    /// assert!(filter.contains("hello"));
    /// ```
    pub fn with_params(counters: usize, hash_fns: usize) -> Result<Self, BloomError> {
        if counters == 0 {
            return Err(BloomError::InvalidBitCount(counters));
        }
        if hash_fns == 0 {
            return Err(BloomError::InvalidHashCount(hash_fns));
        }
        let n = ((counters as f64 * std::f64::consts::LN_2) / hash_fns as f64).round() as usize;
        Ok(Self::raw(counters, hash_fns, n.max(1)))
    }

    fn raw(m: usize, k: usize, n: usize) -> Self {
        Self {
            counters: vec![0u8; m],
            k,
            m,
            n,
            count: 0,
        }
    }
}

impl Filter for CountingBloomFilter {
    /// Returns `true` if all `k` counters for `item` are non-zero.
    #[inline]
    fn contains<T: Bloomable + ?Sized>(&self, item: &T) -> bool {
        let (h1, h2) = hash_pair(item);
        bit_positions(self.k, self.m, h1, h2).all(|pos| self.counters[pos] > 0)
    }

    fn item_count(&self) -> usize {
        self.count
    }

    fn bit_size(&self) -> usize {
        self.m
    }

    fn capacity(&self) -> usize {
        self.n
    }

    fn estimated_fpr(&self) -> f64 {
        estimated_fpr(self.k, self.m, self.count)
    }
}

impl MutableFilter for CountingBloomFilter {
    /// Increments the `k` counters for `item`, saturating at 255.
    #[inline]
    fn insert<T: Bloomable + ?Sized>(&mut self, item: &T) {
        let (h1, h2) = hash_pair(item);
        for pos in bit_positions(self.k, self.m, h1, h2) {
            self.counters[pos] = self.counters[pos].saturating_add(1);
        }
        self.count += 1;
    }

    fn clear(&mut self) {
        self.counters.fill(0);
        self.count = 0;
    }
}

impl RemovableFilter for CountingBloomFilter {
    /// Removes `item` by decrementing its `k` counters.
    ///
    /// Two passes over the `k` bit positions — both derived from a single
    /// hash computation — with no heap allocation:
    ///
    /// 1. **Check pass:** verify all counters are non-zero. Returns `false`
    ///    immediately if any counter is zero (item definitely absent).
    /// 2. **Decrement pass:** decrement each counter, saturating at zero.
    ///
    /// Returns `true` if the item was probably present and counters were
    /// decremented, `false` if it was definitely absent.
    #[inline]
    fn remove<T: Bloomable + ?Sized>(&mut self, item: &T) -> bool {
        let (h1, h2) = hash_pair(item);
        if !bit_positions(self.k, self.m, h1, h2).all(|pos| self.counters[pos] > 0) {
            return false;
        }
        for pos in bit_positions(self.k, self.m, h1, h2) {
            self.counters[pos] = self.counters[pos].saturating_sub(1);
        }
        self.count = self.count.saturating_sub(1);
        true
    }
}
