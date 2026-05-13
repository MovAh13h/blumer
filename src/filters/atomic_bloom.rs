//! Lock-free bloom filter backed by atomic bit operations.

use std::fmt;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use crate::error::BloomError;
use crate::hash::{bit_positions, hash_pair};
use crate::math::{estimated_fpr, validated_params, validated_with_params};
use crate::traits::{ConcurrentFilter, Filter};
use crate::Bloomable;

/// A lock-free bloom filter for concurrent use.
///
/// Equivalent to [`BloomFilter`] in behaviour and API, but backed by
/// [`AtomicU64`] words instead of plain `u64`. Any number of threads can call
/// [`insert`] and [`contains`] simultaneously without external locking.
///
/// # Memory ordering
///
/// - **`insert`** sets bits with `fetch_or(Release)`. After an insert
///   completes, any thread that subsequently loads the same bit with `Acquire`
///   ordering is guaranteed to see it set.
/// - **`contains`** loads each word with `Acquire`, pairing with the `Release`
///   writes in `insert`. This guarantees the no-false-negatives property across
///   threads: if thread A inserts an item and thread B calls `contains` after
///   A's insert returns, B will find the item.
/// - **`item_count`** is tracked with `Relaxed` atomics and is approximate
///   under concurrent use — see [`Filter::item_count`].
///
/// # Approximate item count
///
/// [`Filter::item_count`] and [`Filter::estimated_fpr`] are eventually
/// consistent. Under high concurrency, the reported count may lag behind the
/// true number of insertions. Do not rely on exact values while other threads
/// are actively inserting.
///
/// # Resetting the filter
///
/// There is no concurrent `clear`. [`AtomicBloomFilter::clear`] is available
/// but requires `&mut self`, enforcing exclusive access at the type level.
/// If you need to reset a shared filter, synchronise externally (e.g. replace
/// the `Arc` with a fresh filter).
///
/// # Serde
///
/// With the `serde` feature, `AtomicBloomFilter` serializes to and deserializes
/// from the same JSON/binary format as [`BloomFilter`], so the two types can
/// round-trip into each other:
///
/// ```toml
/// blumer = { version = "0.2", features = ["serde"] }
/// ```
///
/// # Examples
///
/// ```rust
/// use blumer::prelude::*;
/// use std::sync::Arc;
///
/// let filter = Arc::new(AtomicBloomFilter::new(1_000, 0.01).unwrap());
///
/// let f1 = Arc::clone(&filter);
/// let f2 = Arc::clone(&filter);
///
/// let t1 = std::thread::spawn(move || f1.insert("alice"));
/// let t2 = std::thread::spawn(move || f2.insert("bob"));
///
/// t1.join().unwrap();
/// t2.join().unwrap();
///
/// assert!(filter.contains("alice"));
/// assert!(filter.contains("bob"));
/// ```
///
/// [`BloomFilter`]: crate::BloomFilter
/// [`insert`]: ConcurrentFilter::insert
/// [`contains`]: Filter::contains
pub struct AtomicBloomFilter {
    bits: Vec<AtomicU64>,
    k: usize,
    m: usize,
    n: usize,
    count: AtomicUsize,
}

impl AtomicBloomFilter {
    /// Creates an atomic bloom filter optimised for `capacity` items at the
    /// given false positive rate.
    ///
    /// `m` and `k` are computed using the same formulas as [`BloomFilter::new`]:
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
    /// use blumer::prelude::*;
    ///
    /// let filter = AtomicBloomFilter::new(10_000, 0.01).unwrap();
    ///
    /// assert!(AtomicBloomFilter::new(0, 0.01).is_err());
    /// assert!(AtomicBloomFilter::new(100, 0.0).is_err());
    /// ```
    pub fn new(capacity: usize, fpr: f64) -> Result<Self, BloomError> {
        let (m, k) = validated_params(capacity, fpr)?;
        Ok(Self::raw(m, k, capacity))
    }

    /// Creates an atomic bloom filter with explicit bit count and hash
    /// function count.
    ///
    /// Prefer [`AtomicBloomFilter::new`] for normal use. This constructor
    /// is intended for cases where you need to match the exact geometry of a
    /// previously constructed filter — for example, when rehydrating from
    /// storage.
    ///
    /// # Errors
    ///
    /// Returns [`BloomError::InvalidBitCount`] if `bits` is `0`.
    /// Returns [`BloomError::InvalidHashCount`] if `hash_fns` is `0`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use blumer::prelude::*;
    ///
    /// let filter = AtomicBloomFilter::with_params(9_585, 7).unwrap();
    /// filter.insert("hello");
    /// assert!(filter.contains("hello"));
    /// ```
    pub fn with_params(bits: usize, hash_fns: usize) -> Result<Self, BloomError> {
        let n = validated_with_params(bits, hash_fns)?;
        Ok(Self::raw(bits, hash_fns, n))
    }

    /// Resets all bits to zero and the item count to zero.
    ///
    /// Requires exclusive access (`&mut self`) to ensure no other thread is
    /// reading or writing concurrently. If the filter is shared behind an
    /// `Arc`, synchronise externally or replace the `Arc` with a fresh filter
    /// instead.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use blumer::prelude::*;
    ///
    /// let mut filter = AtomicBloomFilter::new(100, 0.01).unwrap();
    /// filter.insert("hello");
    /// filter.clear();
    /// assert!(filter.is_empty());
    /// assert!(!filter.contains("hello"));
    /// ```
    pub fn clear(&mut self) {
        for word in &self.bits {
            word.store(0, Ordering::Relaxed);
        }
        self.count.store(0, Ordering::Relaxed);
    }

    fn raw(m: usize, k: usize, n: usize) -> Self {
        let bits = (0..m.div_ceil(64)).map(|_| AtomicU64::new(0)).collect();
        Self {
            bits,
            k,
            m,
            n,
            count: AtomicUsize::new(0),
        }
    }

    /// Creates a new filter that is the union of `self` and `other`.
    ///
    /// A bit in the result is set if it is set in either filter. The result
    /// is a snapshot — concurrent inserts to `self` or `other` during the
    /// merge may or may not be reflected.
    ///
    /// # Errors
    ///
    /// Returns [`BloomError::IncompatibleGeometry`] if the two filters have
    /// different `m` or `k`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use blumer::prelude::*;
    ///
    /// let a = AtomicBloomFilter::new(1_000, 0.01).unwrap();
    /// let b = AtomicBloomFilter::new(1_000, 0.01).unwrap();
    ///
    /// a.insert("alice");
    /// b.insert("bob");
    ///
    /// let merged = a.merge(&b).unwrap();
    /// assert!(merged.contains("alice"));
    /// assert!(merged.contains("bob"));
    /// ```
    pub fn merge(&self, other: &Self) -> Result<Self, BloomError> {
        if self.m != other.m || self.k != other.k {
            return Err(BloomError::IncompatibleGeometry {
                m: (self.m, other.m),
                k: (self.k, other.k),
            });
        }
        let bits = self.bits.iter()
            .zip(&other.bits)
            .map(|(a, b)| AtomicU64::new(a.load(Ordering::Acquire) | b.load(Ordering::Acquire)))
            .collect();
        Ok(Self {
            bits,
            k: self.k,
            m: self.m,
            n: self.n,
            count: AtomicUsize::new(
                self.count.load(Ordering::Relaxed) + other.count.load(Ordering::Relaxed),
            ),
        })
    }

    /// Atomically merges all bits from `other` into `self` in place.
    ///
    /// Each word is updated with `fetch_or(Release)`, making all of `other`'s
    /// items visible to subsequent `contains` calls. Safe to call concurrently
    /// with `insert` and `contains` — no bits are ever cleared.
    ///
    /// # Errors
    ///
    /// Returns [`BloomError::IncompatibleGeometry`] if the two filters have
    /// different `m` or `k`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use blumer::prelude::*;
    /// use std::sync::Arc;
    ///
    /// let dst = Arc::new(AtomicBloomFilter::new(1_000, 0.01).unwrap());
    /// let src = AtomicBloomFilter::new(1_000, 0.01).unwrap();
    ///
    /// dst.insert("alice");
    /// src.insert("bob");
    ///
    /// dst.merge_from(&src).unwrap();
    /// assert!(dst.contains("alice"));
    /// assert!(dst.contains("bob"));
    /// ```
    pub fn merge_from(&self, other: &Self) -> Result<(), BloomError> {
        if self.m != other.m || self.k != other.k {
            return Err(BloomError::IncompatibleGeometry {
                m: (self.m, other.m),
                k: (self.k, other.k),
            });
        }
        for (dst, src) in self.bits.iter().zip(&other.bits) {
            dst.fetch_or(src.load(Ordering::Acquire), Ordering::Release);
        }
        self.count.fetch_add(other.count.load(Ordering::Relaxed), Ordering::Relaxed);
        Ok(())
    }
}

impl Filter for AtomicBloomFilter {
    /// Returns `true` if all `k` bit positions for `item` are set.
    ///
    /// Each word is loaded with `Acquire` ordering, pairing with the `Release`
    /// writes in [`ConcurrentFilter::insert`]. This guarantees that any insert
    /// that completed before this call is visible.
    #[inline]
    fn contains<T: Bloomable + ?Sized>(&self, item: &T) -> bool {
        let (h1, h2) = hash_pair(item);
        bit_positions(self.k, self.m, h1, h2)
            .all(|pos| self.bits[pos / 64].load(Ordering::Acquire) & (1u64 << (pos % 64)) != 0)
    }

    /// Returns the number of `insert` calls so far.
    ///
    /// Under concurrent use this value is approximate — see the
    /// [type-level docs](AtomicBloomFilter#approximate-item-count).
    fn item_count(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }

    fn bit_size(&self) -> usize {
        self.m
    }

    fn capacity(&self) -> usize {
        self.n
    }

    /// Returns the estimated false positive rate based on the current item
    /// count.
    ///
    /// Because [`item_count`] is approximate under concurrency, this value is
    /// also approximate. It is safe to call from any thread.
    ///
    /// [`item_count`]: Filter::item_count
    fn estimated_fpr(&self) -> f64 {
        estimated_fpr(self.k, self.m, self.count.load(Ordering::Relaxed))
    }
}

impl ConcurrentFilter for AtomicBloomFilter {
    /// Sets the `k` bit positions for `item` using `fetch_or(Release)`.
    ///
    /// Multiple threads may call this simultaneously. Each bit position is
    /// updated atomically — no bit set by one thread will be lost due to a
    /// concurrent write from another.
    #[inline]
    fn insert<T: Bloomable + ?Sized>(&self, item: &T) {
        let (h1, h2) = hash_pair(item);
        for pos in bit_positions(self.k, self.m, h1, h2) {
            self.bits[pos / 64].fetch_or(1u64 << (pos % 64), Ordering::Release);
        }
        self.count.fetch_add(1, Ordering::Relaxed);
    }
}

impl fmt::Debug for AtomicBloomFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AtomicBloomFilter")
            .field("m", &self.m)
            .field("k", &self.k)
            .field("n", &self.n)
            .field("count", &self.count.load(Ordering::Relaxed))
            .finish()
    }
}

impl Clone for AtomicBloomFilter {
    /// Creates a snapshot of the filter at this point in time.
    ///
    /// Each atomic word is loaded with `Acquire` ordering — sufficient to
    /// observe all inserts that completed before the clone call. This is not
    /// an instantaneous snapshot: concurrent inserts may interleave with the
    /// word-by-word copy, but the result is always a valid filter state.
    fn clone(&self) -> Self {
        Self {
            bits: self
                .bits
                .iter()
                .map(|w| AtomicU64::new(w.load(Ordering::Acquire)))
                .collect(),
            k: self.k,
            m: self.m,
            n: self.n,
            count: AtomicUsize::new(self.count.load(Ordering::Acquire)),
        }
    }
}

// --- serde ---

#[cfg(feature = "serde")]
mod serde_impl {
    use super::*;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    /// Shadow struct matching [`BloomFilter`]'s serialized layout so that the
    /// two types can round-trip into each other.
    ///
    /// [`BloomFilter`]: crate::BloomFilter
    #[derive(Serialize, Deserialize)]
    struct Data {
        bits: Vec<u64>,
        k: usize,
        m: usize,
        n: usize,
        count: usize,
    }

    impl Serialize for AtomicBloomFilter {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            // Relaxed is sufficient here — serialization is inherently
            // single-threaded and we only need a consistent snapshot of the
            // current values, not synchronisation with other threads.
            Data {
                bits: self.bits.iter().map(|w| w.load(Ordering::Relaxed)).collect(),
                k: self.k,
                m: self.m,
                n: self.n,
                count: self.count.load(Ordering::Relaxed),
            }
            .serialize(serializer)
        }
    }

    impl<'de> Deserialize<'de> for AtomicBloomFilter {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            let data = Data::deserialize(deserializer)?;
            Ok(AtomicBloomFilter {
                bits: data.bits.into_iter().map(AtomicU64::new).collect(),
                k: data.k,
                m: data.m,
                n: data.n,
                count: AtomicUsize::new(data.count),
            })
        }
    }
}
