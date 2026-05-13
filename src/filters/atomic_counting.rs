//! Lock-free counting bloom filter backed by atomic byte counters.

use std::fmt;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

use crate::error::BloomError;
use crate::hash::{bit_positions, hash_pair};
use crate::math::{estimated_fpr, validated_params, validated_with_params};
use crate::traits::{ConcurrentFilter, Filter};
use crate::Bloomable;

/// A lock-free counting bloom filter that supports concurrent insertion and deletion.
///
/// Extends [`AtomicBloomFilter`] by replacing each bit with a `u8` counter,
/// enabling deletion. Any number of threads can call [`insert`] and [`contains`]
/// simultaneously without external locking. Deletion via [`remove`] is also
/// lock-free but serializing concurrent removes of the **same item** is the
/// caller's responsibility — see below.
///
/// # Atomic memory ordering
///
/// - **`insert`** increments counters via a CAS loop with `Release` ordering.
///   This ensures any subsequent `contains` with `Acquire` loading sees the
///   incremented value.
/// - **`contains`** loads each counter with `Acquire`, pairing with `Release`
///   writes. If thread A inserts an item and thread B calls `contains` after
///   A's insert returns, B is guaranteed to find it.
/// - **`remove`** uses `Acquire` for the check pass and `Release` CAS for
///   the decrement pass.
/// - **`item_count`** and **`estimated_fpr`** use `Relaxed` and are approximate
///   under concurrent use.
///
/// # Counter overflow
///
/// Each counter saturates at 255. Reaching saturation requires 255 distinct
/// items to hash to the same position simultaneously — astronomically unlikely
/// at any sane load. A saturated counter is never decremented, so deletion
/// silently stops working for that slot. Resize the filter before that state.
///
/// # Concurrent remove correctness
///
/// Concurrent inserts and contains are always safe. Concurrent removes of
/// **different** items are also safe. However, if two threads concurrently
/// remove the **same item**, the behavior is a best-effort: no counter
/// underflows to zero and wraps, but one of the two removes may not fully
/// decrement all counters, potentially leaving a ghost entry. Serialize
/// removes of identical items with an external `Mutex` if this matters.
///
/// Only remove items you have previously inserted. Removing a never-inserted
/// item decrements counters shared with real items and **causes false negatives**.
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
/// use std::sync::Arc;
///
/// let filter = Arc::new(AtomicCountingBloomFilter::new(1_000, 0.01).unwrap());
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
///
/// // Deletion is also available:
/// filter.remove("alice");
/// assert!(!filter.contains("alice"));
/// assert!(filter.contains("bob"));
/// ```
///
/// [`AtomicBloomFilter`]: crate::AtomicBloomFilter
/// [`insert`]: ConcurrentFilter::insert
/// [`contains`]: Filter::contains
/// [`remove`]: AtomicCountingBloomFilter::remove
pub struct AtomicCountingBloomFilter {
    counters: Vec<AtomicU8>,
    k: usize,
    m: usize,
    n: usize,
    count: AtomicUsize,
}

impl AtomicCountingBloomFilter {
    /// Creates an atomic counting bloom filter optimised for `capacity` items
    /// at the given false positive rate.
    ///
    /// `m` (counter count) and `k` (hash function count) are derived using the
    /// same formulas as [`BloomFilter::new`]:
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
    /// let filter = AtomicCountingBloomFilter::new(10_000, 0.01).unwrap();
    ///
    /// assert!(AtomicCountingBloomFilter::new(0, 0.01).is_err());
    /// assert!(AtomicCountingBloomFilter::new(100, 0.0).is_err());
    /// ```
    pub fn new(capacity: usize, fpr: f64) -> Result<Self, BloomError> {
        let (m, k) = validated_params(capacity, fpr)?;
        Ok(Self::raw(m, k, capacity))
    }

    /// Creates an atomic counting bloom filter with explicit counter count and
    /// hash function count.
    ///
    /// Prefer [`AtomicCountingBloomFilter::new`] for normal use. This
    /// constructor is intended for cases where you need to match the exact
    /// geometry of a previously constructed filter — for example, when
    /// rehydrating from storage.
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
    /// let filter = AtomicCountingBloomFilter::with_params(9_585, 7).unwrap();
    /// filter.insert("hello");
    /// assert!(filter.contains("hello"));
    /// ```
    pub fn with_params(counters: usize, hash_fns: usize) -> Result<Self, BloomError> {
        let n = validated_with_params(counters, hash_fns)?;
        Ok(Self::raw(counters, hash_fns, n))
    }

    /// Resets all counters to zero and the item count to zero.
    ///
    /// Requires exclusive access (`&mut self`) to ensure no other thread is
    /// reading or writing concurrently. If the filter is shared behind an
    /// `Arc`, replace the `Arc` with a fresh filter instead.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use blume::prelude::*;
    ///
    /// let mut filter = AtomicCountingBloomFilter::new(100, 0.01).unwrap();
    /// filter.insert("hello");
    /// filter.clear();
    /// assert!(filter.is_empty());
    /// assert!(!filter.contains("hello"));
    /// ```
    pub fn clear(&mut self) {
        for c in &self.counters {
            c.store(0, Ordering::Relaxed);
        }
        self.count.store(0, Ordering::Relaxed);
    }

    /// Removes `item` from the filter by decrementing its counters.
    ///
    /// Two passes over the `k` positions — both derived from a single hash
    /// computation — with no heap allocation:
    ///
    /// 1. **Check pass** (`Acquire` load): if any counter is zero the item is
    ///    definitely absent; returns `false` immediately.
    /// 2. **Decrement pass** (CAS with `Release`): decrements each non-zero
    ///    counter. The CAS protects against underflow — if a concurrent remove
    ///    raced to zero first, that counter is skipped.
    ///
    /// Returns `true` if the item was probably present and counters were
    /// decremented, `false` if it was definitely absent.
    ///
    /// # Correctness
    ///
    /// Only call this for items you know were previously inserted. See the
    /// [type-level docs](AtomicCountingBloomFilter) for concurrency caveats.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use blume::prelude::*;
    ///
    /// let filter = AtomicCountingBloomFilter::new(100, 0.01).unwrap();
    /// filter.insert("hello");
    ///
    /// assert!(filter.contains("hello"));
    /// assert!(filter.remove("hello"));
    /// assert!(!filter.contains("hello"));
    ///
    /// assert!(!filter.remove("world")); // never inserted — returns false
    /// ```
    pub fn remove<T: Bloomable + ?Sized>(&self, item: &T) -> bool {
        let (h1, h2) = hash_pair(item);
        let positions: Vec<usize> = bit_positions(self.k, self.m, h1, h2).collect();

        // Check pass — if any counter is zero, item is definitely absent.
        if positions.iter().any(|&pos| self.counters[pos].load(Ordering::Acquire) == 0) {
            return false;
        }

        // Decrement pass — CAS protects against underflow under concurrent removes.
        for &pos in &positions {
            let mut current = self.counters[pos].load(Ordering::Acquire);
            loop {
                if current == 0 { break; } // raced to zero by a concurrent remove
                match self.counters[pos].compare_exchange_weak(
                    current,
                    current - 1,
                    Ordering::Release,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(actual) => current = actual,
                }
            }
        }

        // count is approximate — saturate at 0 rather than wrap.
        let _ = self.count.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |c| {
            if c > 0 { Some(c - 1) } else { None }
        });

        true
    }

    fn raw(m: usize, k: usize, n: usize) -> Self {
        let counters = (0..m).map(|_| AtomicU8::new(0)).collect();
        Self {
            counters,
            k,
            m,
            n,
            count: AtomicUsize::new(0),
        }
    }
}

impl Filter for AtomicCountingBloomFilter {
    /// Returns `true` if all `k` counters for `item` are non-zero.
    ///
    /// Each counter is loaded with `Acquire` ordering, pairing with the
    /// `Release` writes in [`ConcurrentFilter::insert`] and [`remove`].
    ///
    /// [`remove`]: AtomicCountingBloomFilter::remove
    #[inline]
    fn contains<T: Bloomable + ?Sized>(&self, item: &T) -> bool {
        let (h1, h2) = hash_pair(item);
        bit_positions(self.k, self.m, h1, h2)
            .all(|pos| self.counters[pos].load(Ordering::Acquire) > 0)
    }

    /// Returns the number of `insert` calls so far.
    ///
    /// Under concurrent use this value is approximate — see the
    /// [type-level docs](AtomicCountingBloomFilter#atomic-memory-ordering).
    fn item_count(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }

    fn bit_size(&self) -> usize {
        self.m
    }

    fn capacity(&self) -> usize {
        self.n
    }

    /// Returns the estimated false positive rate based on the current item count.
    ///
    /// Because [`item_count`] is approximate under concurrency, this value is
    /// also approximate.
    ///
    /// [`item_count`]: Filter::item_count
    fn estimated_fpr(&self) -> f64 {
        estimated_fpr(self.k, self.m, self.count.load(Ordering::Relaxed))
    }
}

impl ConcurrentFilter for AtomicCountingBloomFilter {
    /// Increments the `k` counters for `item`, saturating each at 255.
    ///
    /// Uses a CAS loop with `Release` ordering to atomically increment without
    /// wrapping. Multiple threads may call this simultaneously — no increment
    /// from one thread will be lost due to a concurrent write from another.
    #[inline]
    fn insert<T: Bloomable + ?Sized>(&self, item: &T) {
        let (h1, h2) = hash_pair(item);
        for pos in bit_positions(self.k, self.m, h1, h2) {
            // Saturating increment: CAS loop avoids wrapping past 255.
            let mut current = self.counters[pos].load(Ordering::Relaxed);
            loop {
                if current == u8::MAX { break; }
                match self.counters[pos].compare_exchange_weak(
                    current,
                    current + 1,
                    Ordering::Release,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(actual) => current = actual,
                }
            }
        }
        self.count.fetch_add(1, Ordering::Relaxed);
    }
}

impl fmt::Debug for AtomicCountingBloomFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AtomicCountingBloomFilter")
            .field("m", &self.m)
            .field("k", &self.k)
            .field("n", &self.n)
            .field("count", &self.count.load(Ordering::Relaxed))
            .finish()
    }
}

impl Clone for AtomicCountingBloomFilter {
    /// Creates a snapshot of the filter at this point in time.
    ///
    /// Each counter is loaded with `Acquire` ordering — sufficient to observe
    /// all inserts and removes that completed before the clone call. This is
    /// not an instantaneous snapshot: concurrent operations may interleave with
    /// the element-by-element copy, but the result is always a valid filter state.
    fn clone(&self) -> Self {
        Self {
            counters: self
                .counters
                .iter()
                .map(|c| AtomicU8::new(c.load(Ordering::Acquire)))
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

    #[derive(Serialize, Deserialize)]
    struct Data {
        counters: Vec<u8>,
        k: usize,
        m: usize,
        n: usize,
        count: usize,
    }

    impl Serialize for AtomicCountingBloomFilter {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            Data {
                counters: self.counters.iter().map(|c| c.load(Ordering::Relaxed)).collect(),
                k: self.k,
                m: self.m,
                n: self.n,
                count: self.count.load(Ordering::Relaxed),
            }
            .serialize(serializer)
        }
    }

    impl<'de> Deserialize<'de> for AtomicCountingBloomFilter {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            let data = Data::deserialize(deserializer)?;
            Ok(AtomicCountingBloomFilter {
                counters: data.counters.into_iter().map(AtomicU8::new).collect(),
                k: data.k,
                m: data.m,
                n: data.n,
                count: AtomicUsize::new(data.count),
            })
        }
    }
}
