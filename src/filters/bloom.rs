//! Standard fixed-size bloom filter.

use crate::error::BloomError;
use crate::hash::{bit_positions, hash_pair};
use crate::math::{estimated_fpr, validated_params, validated_with_params};
use crate::traits::{Filter, MutableFilter};
use crate::Bloomable;

/// A standard fixed-size bloom filter.
///
/// A bloom filter is a space-efficient probabilistic data structure for
/// membership testing. It can report false positives (claiming an item is
/// present when it is not) but never false negatives (it will always correctly
/// identify items that have been inserted).
///
/// # Bit packing
///
/// Bits are packed into an array of `u64` words. Each `u64` holds 64 bits, so
/// the array has `ceil(m / 64)` words for a filter with `m` total bits.
///
/// To set or test bit `i`:
///
/// ```text
/// word  = bits[i / 64]   — which u64 holds this bit
/// shift = i % 64         — which bit within that word
///
/// set:  bits[i / 64] |=  (1u64 << shift)
/// test: bits[i / 64] &   (1u64 << shift) != 0
/// ```
///
/// Packing into 64-bit words keeps memory aligned, lets the CPU load and test
/// an entire word in a single instruction, and plays well with the cache line
/// size (typically 64 bytes = 8 × `u64`).
///
/// # Hashing
///
/// Each item is hashed twice with [AHash] using independent seeds, producing
/// two 64-bit values `h1` and `h2`. From those, `k` bit positions are derived
/// without any further hash calls using the double hashing formula:
///
/// ```text
/// pos_i = (h1 + i · h2) mod m,  for i in 0..k
/// ```
///
/// This is the Kirsch–Mitzenmacher optimization (2006), which proves that two
/// hash functions are sufficient to achieve the same asymptotic false positive
/// rate as `k` independent ones. `h2` is forced odd to prevent cycles: if
/// `h2` were even and `m` were also even, some bit positions would never be
/// reached.
///
/// [AHash]: https://github.com/tkaitchuck/aHash
///
/// # Choosing parameters
///
/// Use [`BloomFilter::new`] and provide the number of items you expect to insert
/// and your acceptable false positive rate. The filter will automatically compute
/// the optimal bit count `m` and hash function count `k`:
///
/// | FPR target | Bits per item (`m/n`) | Hash functions (`k`) |
/// |------------|-----------------------|----------------------|
/// | 1%         | ~9.6                  | 7                    |
/// | 0.1%       | ~14.4                 | 10                   |
/// | 0.01%      | ~19.2                 | 14                   |
///
/// # Memory usage
///
/// The internal bit array occupies exactly `ceil(m / 8)` bytes, where
/// `m = -(n · ln(p)) / ln(2)²`. Because bits are stored in `u64` words,
/// the actual allocation is rounded up to the nearest 8 bytes.
///
/// Quick reference for the `new(n, p)` constructor:
///
/// | Items (`n`) | FPR (`p`) | Bit array  | RAM       |
/// |-------------|-----------|------------|-----------|
/// | 1 000       | 1%        | ~9 600 b   | ~1.2 KB   |
/// | 100 000     | 1%        | ~960 000 b | ~117 KB   |
/// | 1 000 000   | 1%        | ~9.6 Mb    | ~1.2 MB   |
/// | 1 000 000   | 0.1%      | ~14.4 Mb   | ~1.7 MB   |
///
/// For comparison, storing the same 1 000 000 items in a `HashSet<u64>` would
/// typically use ~40 MB — roughly 30× more.
///
/// # Examples
///
/// ```rust
/// use blume::prelude::*;
///
/// let mut filter = BloomFilter::new(1_000, 0.01).unwrap();
///
/// filter.insert("alice");
/// filter.insert("bob");
/// filter.insert(&42u64);
///
/// assert!(filter.contains("alice"));
/// assert!(filter.contains("bob"));
/// assert!(filter.contains(&42u64));
/// ```
///
/// # Feature flags
///
/// With the `serde` feature enabled, `BloomFilter` implements
/// `serde::Serialize` and `serde::Deserialize`, allowing filters to be
/// saved to disk or sent over the network.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BloomFilter {
    bits: Vec<u64>,
    k: usize,
    m: usize,
    n: usize,
    count: usize,
}

impl BloomFilter {
    /// Creates a bloom filter optimised for `capacity` items at the given false
    /// positive rate. See the [crate-level FPR documentation](crate#false-positive-rate-fpr)
    /// for a full explanation of what the `fpr` value means.
    ///
    /// ## How `m` (bit count) and `k` (hash functions) are derived
    ///
    /// Given `n = capacity` and `p = fpr`, the two parameters are computed as:
    ///
    /// ```text
    /// m = -(n · ln(p)) / ln(2)²     — total bits
    /// k =  (m / n)    · ln(2)       — hash functions
    /// ```
    ///
    /// **Intuition for `m`:** a lower FPR requires more bits per item to keep
    /// the bit array sparse enough that collisions (false positives) are rare.
    /// The `ln(p)` term is always negative (since `p < 1`), so the leading `-`
    /// makes `m` positive. `ln(2)²` is the normalisation constant that falls
    /// out of minimising the FPR with respect to the number of bits.
    ///
    /// **Intuition for `k`:** each hash function sets one bit per insertion.
    /// Too few functions and each item claims too few bits, making collisions
    /// likely. Too many and the array fills up quickly, also raising the FPR.
    /// The optimal `k = (m/n) · ln(2)` balances these two effects.
    ///
    /// Concretely, for `capacity = 1_000` and `fpr = 0.01`:
    ///
    /// ```text
    /// m = -(1000 · ln(0.01)) / ln(2)²  ≈  9 586 bits  (≈ 1.2 KB)
    /// k =  (9586 / 1000)    · ln(2)    ≈  7 hash functions
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`BloomError::InvalidCapacity`] if `capacity` is `0`.
    /// Returns [`BloomError::InvalidFpr`] if `fpr` is not in the range `(0, 1)`
    /// or is `NaN` or infinite.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use blume::prelude::*;
    ///
    /// let filter = BloomFilter::new(10_000, 0.01).unwrap();
    ///
    /// assert!(BloomFilter::new(0, 0.01).is_err());
    /// assert!(BloomFilter::new(100, 0.0).is_err());
    /// assert!(BloomFilter::new(100, 1.0).is_err());
    /// ```
    pub fn new(capacity: usize, fpr: f64) -> Result<Self, BloomError> {
        let (m, k) = validated_params(capacity, fpr)?;
        Ok(Self::raw(m, k, capacity))
    }

    /// Creates a bloom filter with explicit bit count and hash function count.
    ///
    /// This is an expert constructor for cases where you want precise control
    /// over the filter's internal parameters — for example, when you are
    /// rehydrating a filter from storage and must match the exact geometry of
    /// the original.
    ///
    /// The designed capacity `n` is back-derived from `bits` and `hash_fns`
    /// by inverting the optimal `k` formula:
    ///
    /// ```text
    /// k = (m / n) · ln(2)   →   n = (m · ln(2)) / k
    /// ```
    ///
    /// This gives the item count at which `hash_fns` would have been the
    /// optimal choice for `bits`. It is an approximation — if you constructed
    /// the original filter with [`BloomFilter::new`], the reported capacity
    /// will be close but may differ by a small rounding amount.
    ///
    /// Prefer [`BloomFilter::new`] for all normal use.
    ///
    /// # Errors
    ///
    /// Returns [`BloomError::InvalidBitCount`] if `bits` is `0`.
    /// Returns [`BloomError::InvalidHashCount`] if `hash_fns` is `0`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use blume::prelude::*;
    ///
    /// // Manually sized for ~1 000 items at ~1% FPR.
    /// let mut filter = BloomFilter::with_params(9_585, 7).unwrap();
    /// filter.insert("hello");
    /// assert!(filter.contains("hello"));
    /// ```
    pub fn with_params(bits: usize, hash_fns: usize) -> Result<Self, BloomError> {
        let n = validated_with_params(bits, hash_fns)?;
        Ok(Self::raw(bits, hash_fns, n))
    }

    fn raw(m: usize, k: usize, n: usize) -> Self {
        Self {
            bits: vec![0u64; m.div_ceil(64)],
            k,
            m,
            n,
            count: 0,
        }
    }

    /// Creates a new filter that is the union of `self` and `other`.
    ///
    /// A bit in the result is set if it is set in either filter. After
    /// merging, `contains` returns `true` for any item inserted into
    /// either source filter.
    ///
    /// # Errors
    ///
    /// Returns [`BloomError::IncompatibleGeometry`] if the two filters have
    /// different `m` (bit count) or `k` (hash function count). Both filters
    /// must be constructed with identical parameters to be mergeable — use
    /// the same `capacity` and `fpr`, or the same `with_params` arguments.
    ///
    /// # Item count
    ///
    /// The merged filter's `item_count` is the sum of both filters' counts,
    /// which is an upper bound — items present in both filters are counted
    /// twice.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use blume::prelude::*;
    ///
    /// let mut a = BloomFilter::new(1_000, 0.01).unwrap();
    /// let mut b = BloomFilter::new(1_000, 0.01).unwrap();
    ///
    /// a.insert("alice");
    /// b.insert("bob");
    ///
    /// let merged = a.merge(&b).unwrap();
    /// assert!(merged.contains("alice"));
    /// assert!(merged.contains("bob"));
    ///
    /// // Filters with different parameters cannot be merged.
    /// let other = BloomFilter::new(500, 0.01).unwrap();
    /// assert!(a.merge(&other).is_err());
    /// ```
    pub fn merge(&self, other: &Self) -> Result<Self, BloomError> {
        if self.m != other.m || self.k != other.k {
            return Err(BloomError::IncompatibleGeometry {
                m: (self.m, other.m),
                k: (self.k, other.k),
            });
        }
        Ok(Self {
            bits: self.bits.iter().zip(&other.bits).map(|(a, b)| a | b).collect(),
            k: self.k,
            m: self.m,
            n: self.n,
            count: self.count + other.count,
        })
    }
}

impl Filter for BloomFilter {
    /// Returns `true` if `item` is probably in the filter, `false` if it is
    /// definitely not.
    ///
    /// This operation is `O(k)` where `k` is the number of hash functions.
    #[inline]
    fn contains<T: Bloomable + ?Sized>(&self, item: &T) -> bool {
        let (h1, h2) = hash_pair(item);
        bit_positions(self.k, self.m, h1, h2)
            .all(|pos| self.bits[pos / 64] & (1u64 << (pos % 64)) != 0)
    }

    /// Returns the total number of [`MutableFilter::insert`] calls, including
    /// duplicates.
    fn item_count(&self) -> usize {
        self.count
    }

    /// Returns the total number of bits in the internal bit array (`m`).
    fn bit_size(&self) -> usize {
        self.m
    }

    /// Returns the number of items the filter was sized for (`n`).
    fn capacity(&self) -> usize {
        self.n
    }

    /// Returns the estimated false positive rate given the current number of
    /// insertions.
    fn estimated_fpr(&self) -> f64 {
        estimated_fpr(self.k, self.m, self.count)
    }
}

impl MutableFilter for BloomFilter {
    /// Inserts `item` into the filter.
    ///
    /// This operation is `O(k)` where `k` is the number of hash functions.
    #[inline]
    fn insert<T: Bloomable + ?Sized>(&mut self, item: &T) {
        let (h1, h2) = hash_pair(item);
        for pos in bit_positions(self.k, self.m, h1, h2) {
            self.bits[pos / 64] |= 1u64 << (pos % 64);
        }
        self.count += 1;
    }

    /// Resets all bits to zero and sets the item count to zero.
    ///
    /// The filter's capacity, bit size, and hash function count are unchanged.
    fn clear(&mut self) {
        self.bits.fill(0);
        self.count = 0;
    }
}
