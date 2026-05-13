//! Cuckoo filter — supports deletion with lower memory than a counting filter.

use crate::error::BloomError;
use crate::hash::hash_pair;

use crate::traits::Filter;
use crate::Bloomable;

/// Number of fingerprint slots per bucket.
const BUCKET_SIZE: usize = 4;

/// Maximum number of eviction attempts before an insert is declared failed.
const MAX_KICKS: usize = 500;

/// A cuckoo filter — a space-efficient probabilistic set that supports
/// deletion.
///
/// A cuckoo filter stores a small **fingerprint** (hash of the item) in one
/// of two candidate buckets. Lookup and deletion check only those two buckets,
/// making both operations `O(1)`. This is faster and more memory-efficient
/// than a [`CountingBloomFilter`] for workloads that require deletion.
///
/// # Comparison with counting bloom filter
///
/// | Property | `CuckooFilter` | `CountingBloomFilter` |
/// |----------|---------------|-----------------------|
/// | Deletion | ✓ | ✓ |
/// | Memory   | ~1 byte/item  | ~8 bytes/item         |
/// | Insert   | `Result` (can fail at ~95% load) | always succeeds |
/// | FPR      | controlled by fingerprint size | controlled by `fpr` |
///
/// # Deletion correctness
///
/// Only delete items that were previously inserted. Deleting an absent item
/// may remove a fingerprint belonging to a different item, causing false
/// negatives. The filter cannot detect this; the contract is enforced by the
/// caller.
///
/// # Capacity and load
///
/// Insertions can fail when the table approaches its practical capacity limit
/// (~95% load factor). [`insert`] returns [`BloomError::CapacityExceeded`] in
/// that case. Size the filter with headroom: `capacity * 1.1` is a safe
/// starting point.
///
/// # False positive rate
///
/// With 8-bit fingerprints and bucket size 4, the empirical FPR is
/// approximately `2 * bucket_size / 2^fingerprint_bits ≈ 3%`. To achieve a
/// target FPR `p`, the fingerprint size `f` must satisfy
/// `f ≥ log2(2 * bucket_size / p)`. For 1% FPR, `f ≥ 10` bits.
///
/// This implementation uses 8-bit fingerprints for simplicity. For lower FPR,
/// use [`BloomFilter`] or [`CountingBloomFilter`] instead.
///
/// # Examples
///
/// ```rust
/// use blumer::prelude::*;
///
/// let mut filter = CuckooFilter::new(1_000).unwrap();
///
/// filter.insert("alice").unwrap();
/// filter.insert("bob").unwrap();
///
/// assert!(filter.contains("alice"));
/// assert!(filter.contains("bob"));
///
/// filter.remove("alice");
/// assert!(!filter.contains("alice"));
/// assert!(filter.contains("bob"));
/// ```
///
/// [`insert`]: CuckooFilter::insert
/// [`BloomFilter`]: crate::BloomFilter
/// [`CountingBloomFilter`]: crate::CountingBloomFilter
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CuckooFilter {
    /// Flat array of fingerprint buckets. Bucket `i` occupies
    /// `table[i * BUCKET_SIZE .. (i+1) * BUCKET_SIZE]`.
    /// A zero fingerprint means an empty slot.
    table: Vec<u8>,
    /// Number of buckets.
    num_buckets: usize,
    /// Number of fingerprints currently stored.
    count: usize,
}

impl CuckooFilter {
    /// Creates a cuckoo filter sized for at least `capacity` items.
    ///
    /// The actual bucket count is rounded up to the next power of two to
    /// enable fast bitwise index arithmetic.
    ///
    /// # Errors
    ///
    /// Returns [`BloomError::InvalidCapacity`] if `capacity` is `0`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use blumer::prelude::*;
    ///
    /// let mut filter = CuckooFilter::new(1_000).unwrap();
    /// filter.insert("hello").unwrap();
    /// assert!(filter.contains("hello"));
    /// ```
    pub fn new(capacity: usize) -> Result<Self, BloomError> {
        if capacity == 0 {
            return Err(BloomError::InvalidCapacity(capacity));
        }
        // Each bucket holds BUCKET_SIZE items; add headroom for the ~95% load limit.
        let min_buckets = (capacity * 2).div_ceil(BUCKET_SIZE);
        let num_buckets = min_buckets.next_power_of_two().max(1);
        Ok(Self {
            table: vec![0u8; num_buckets * BUCKET_SIZE],
            num_buckets,
            count: 0,
        })
    }

    /// Creates a cuckoo filter with an explicit bucket count.
    ///
    /// `num_buckets` is rounded up to the next power of two. This constructor
    /// is intended for cases where you need to match the exact geometry of a
    /// previously constructed filter.
    ///
    /// # Errors
    ///
    /// Returns [`BloomError::InvalidCapacity`] if `num_buckets` is `0`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use blumer::prelude::*;
    ///
    /// let mut filter = CuckooFilter::with_buckets(256).unwrap();
    /// filter.insert("hello").unwrap();
    /// assert!(filter.contains("hello"));
    /// ```
    pub fn with_buckets(num_buckets: usize) -> Result<Self, BloomError> {
        if num_buckets == 0 {
            return Err(BloomError::InvalidCapacity(num_buckets));
        }
        let num_buckets = num_buckets.next_power_of_two();
        Ok(Self {
            table: vec![0u8; num_buckets * BUCKET_SIZE],
            num_buckets,
            count: 0,
        })
    }

    /// Inserts `item` into the filter.
    ///
    /// Returns `Ok(())` on success. Returns [`BloomError::CapacityExceeded`]
    /// if the table is too full to accommodate the item (typically above ~95%
    /// load). In that case, construct a larger filter.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use blumer::prelude::*;
    ///
    /// let mut filter = CuckooFilter::new(100).unwrap();
    /// assert!(filter.insert("hello").is_ok());
    /// ```
    pub fn insert<T: Bloomable + ?Sized>(&mut self, item: &T) -> Result<(), BloomError> {
        let (fp, i1, i2) = self.fingerprint_and_indices(item);

        // Try to place into either candidate bucket.
        if self.insert_into_bucket(i1, fp) || self.insert_into_bucket(i2, fp) {
            self.count += 1;
            return Ok(());
        }

        // Both buckets full — begin cuckoo eviction from a random-ish bucket.
        let mut index = i1;
        let mut evicted_fp = fp;

        for _ in 0..MAX_KICKS {
            // Evict a random slot from the current bucket.
            let slot = (evicted_fp as usize) % BUCKET_SIZE;
            let offset = index * BUCKET_SIZE + slot;
            std::mem::swap(&mut self.table[offset], &mut evicted_fp);

            // Compute the alternate bucket for the evicted fingerprint.
            index = self.alt_index(index, evicted_fp);

            if self.insert_into_bucket(index, evicted_fp) {
                self.count += 1;
                return Ok(());
            }
        }

        Err(BloomError::CapacityExceeded)
    }

    /// Removes one occurrence of `item` from the filter.
    ///
    /// Returns `true` if the item was probably present and a fingerprint was
    /// removed, `false` if it was definitely absent.
    ///
    /// # Correctness
    ///
    /// Only call this for items you know were previously inserted. Removing an
    /// absent item may delete a fingerprint belonging to a different item,
    /// causing false negatives for that item.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use blumer::prelude::*;
    ///
    /// let mut filter = CuckooFilter::new(100).unwrap();
    /// filter.insert("hello").unwrap();
    ///
    /// assert!(filter.remove("hello"));
    /// assert!(!filter.contains("hello"));
    /// assert!(!filter.remove("world")); // was never inserted
    /// ```
    pub fn remove<T: Bloomable + ?Sized>(&mut self, item: &T) -> bool {
        let (fp, i1, i2) = self.fingerprint_and_indices(item);
        if self.remove_from_bucket(i1, fp) || self.remove_from_bucket(i2, fp) {
            self.count = self.count.saturating_sub(1);
            return true;
        }
        false
    }

    /// Resets the filter to empty.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use blumer::prelude::*;
    ///
    /// let mut filter = CuckooFilter::new(100).unwrap();
    /// filter.insert("hello").unwrap();
    /// filter.clear();
    /// assert!(filter.is_empty());
    /// assert!(!filter.contains("hello"));
    /// ```
    pub fn clear(&mut self) {
        self.table.fill(0);
        self.count = 0;
    }

    /// Returns the number of buckets in the underlying hash table.
    pub fn num_buckets(&self) -> usize {
        self.num_buckets
    }

    /// Returns the total number of fingerprint slots (`num_buckets × 4`).
    pub fn total_slots(&self) -> usize {
        self.num_buckets * BUCKET_SIZE
    }

    // --- private helpers ---

    /// Computes the 8-bit fingerprint and both candidate bucket indices for `item`.
    ///
    /// Uses h1 for the primary index and h2 for the fingerprint. The alternate
    /// index is derived as `i1 XOR hash(fp)` so it can be recovered from either
    /// bucket without storing the original key.
    fn fingerprint_and_indices<T: Bloomable + ?Sized>(&self, item: &T) -> (u8, usize, usize) {
        let (h1, h2) = hash_pair(item);
        // Non-zero fingerprint: map 0 → 1 to keep 0 as the empty-slot sentinel.
        let fp = ((h2 & 0xFF) as u8).max(1);
        let i1 = (h1 as usize) & (self.num_buckets - 1);
        let i2 = self.alt_index(i1, fp);
        (fp, i1, i2)
    }

    /// Computes the alternate bucket index using `index XOR hash(fingerprint)`.
    ///
    /// This operation is its own inverse: `alt_index(alt_index(i, fp), fp) == i`.
    /// The property ensures we can always recover the partner bucket from
    /// either end without storing the original item.
    #[inline]
    fn alt_index(&self, index: usize, fp: u8) -> usize {
        // Multiply the fingerprint by a large prime to spread the XOR uniformly.
        let hash = (fp as usize).wrapping_mul(0x517cc1b727220a95);
        (index ^ hash) & (self.num_buckets - 1)
    }

    /// Attempts to place `fp` into an empty slot in bucket `index`.
    /// Returns `true` on success.
    #[inline]
    fn insert_into_bucket(&mut self, index: usize, fp: u8) -> bool {
        let start = index * BUCKET_SIZE;
        for slot in &mut self.table[start..start + BUCKET_SIZE] {
            if *slot == 0 {
                *slot = fp;
                return true;
            }
        }
        false
    }

    /// Attempts to remove one occurrence of `fp` from bucket `index`.
    /// Returns `true` on success.
    #[inline]
    fn remove_from_bucket(&mut self, index: usize, fp: u8) -> bool {
        let start = index * BUCKET_SIZE;
        for slot in &mut self.table[start..start + BUCKET_SIZE] {
            if *slot == fp {
                *slot = 0;
                return true;
            }
        }
        false
    }
}

impl Filter for CuckooFilter {
    /// Returns `true` if `item` is probably in the filter, `false` if it is
    /// definitely absent.
    ///
    /// Checks both candidate buckets for the item's fingerprint. This is
    /// `O(1)` regardless of filter size.
    #[inline]
    fn contains<T: Bloomable + ?Sized>(&self, item: &T) -> bool {
        let (fp, i1, i2) = self.fingerprint_and_indices(item);
        self.bucket_contains(i1, fp) || self.bucket_contains(i2, fp)
    }

    fn item_count(&self) -> usize {
        self.count
    }

    /// Returns the total number of fingerprint slots.
    fn bit_size(&self) -> usize {
        self.total_slots() * 8
    }

    /// Returns the practical capacity — approximately 95% of total slots.
    fn capacity(&self) -> usize {
        (self.total_slots() * 95) / 100
    }

    /// Returns an estimated false positive rate based on current load.
    ///
    /// With 8-bit fingerprints and 4-slot buckets, the FPR at load factor `α`
    /// is approximately `α × 2 × bucket_size / 256`.
    fn estimated_fpr(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let load = self.count as f64 / self.total_slots() as f64;
        load * 2.0 * BUCKET_SIZE as f64 / 256.0
    }
}

impl CuckooFilter {
    /// Returns `true` if bucket `index` contains fingerprint `fp`.
    #[inline]
    fn bucket_contains(&self, index: usize, fp: u8) -> bool {
        let start = index * BUCKET_SIZE;
        self.table[start..start + BUCKET_SIZE].contains(&fp)
    }
}

// Ensure CuckooFilter satisfies the validated_params contract at the type
// level by calling it in `new` — nothing to do here since CuckooFilter's
// `new` validates capacity independently.
const _: () = {
    // Compile-time sanity check: BUCKET_SIZE must be a power of two for the
    // slot-selection arithmetic to distribute evenly.
    assert!(BUCKET_SIZE.is_power_of_two());
};
