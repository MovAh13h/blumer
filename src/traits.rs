//! Core traits that define the bloom filter interface.

use crate::Bloomable;

/// Read-only operations shared by all bloom filter types.
///
/// This trait provides membership testing and inspection methods. It does not
/// include insertion — see [`MutableFilter`] for that.
///
/// # False positives
///
/// [`Filter::contains`] may return `true` for items that were never inserted.
/// The probability of this is bounded by the filter's false positive rate (FPR),
/// which you can inspect at any time with [`Filter::estimated_fpr`]. It will
/// never return `false` for an item that was inserted.
pub trait Filter {
    /// Returns `true` if `item` is **probably** in the filter, `false` if it is
    /// **definitely not**.
    ///
    /// A return value of `true` is probabilistic: there is a small chance of a
    /// false positive. A return value of `false` is guaranteed to be correct.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use blumer::prelude::*;
    ///
    /// let mut filter = BloomFilter::new(100, 0.01).unwrap();
    /// filter.insert("hello");
    ///
    /// assert!(filter.contains("hello"));  // definitely present
    /// assert!(!filter.contains("world")); // definitely absent (barring false positive)
    /// ```
    #[must_use]
    fn contains<T: Bloomable + ?Sized>(&self, item: &T) -> bool;

    /// Returns the total number of [`MutableFilter::insert`] calls, including
    /// duplicate insertions. It is not deduplicated — bloom filters do not track
    /// whether an item was already present.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use blumer::prelude::*;
    ///
    /// let mut filter = BloomFilter::new(100, 0.01).unwrap();
    /// assert_eq!(filter.item_count(), 0);
    /// filter.insert("a");
    /// filter.insert("a"); // duplicate — still counted
    /// assert_eq!(filter.item_count(), 2);
    /// ```
    #[must_use]
    fn item_count(&self) -> usize;

    /// Returns the total number of bits in the filter's internal bit array.
    ///
    /// This is the `m` parameter in bloom filter literature. A larger bit array
    /// reduces the false positive rate for a given number of insertions.
    #[must_use]
    fn bit_size(&self) -> usize;

    /// Returns the number of items the filter was designed to hold while
    /// maintaining its target false positive rate.
    ///
    /// This is the `n` parameter passed to [`crate::BloomFilter::new`]. Inserting
    /// significantly more items than this will cause the actual FPR to exceed
    /// the target.
    #[must_use]
    fn capacity(&self) -> usize;

    /// Returns the current estimated false positive rate based on the number of
    /// items inserted so far.
    ///
    /// Returns `0.0` if no items have been inserted. As more items are inserted,
    /// this value rises toward and eventually beyond the target FPR.
    ///
    /// The estimate is computed from the formula `(1 - e^(-k·n/m))^k`, where
    /// `k` is the number of hash functions, `n` is [`item_count`], and `m` is
    /// [`bit_size`].
    ///
    /// [`item_count`]: Filter::item_count
    /// [`bit_size`]: Filter::bit_size
    ///
    /// # Examples
    ///
    /// ```rust
    /// use blumer::prelude::*;
    ///
    /// let mut filter = BloomFilter::new(100, 0.01).unwrap();
    /// assert_eq!(filter.estimated_fpr(), 0.0);
    ///
    /// for i in 0..100u64 {
    ///     filter.insert(&i);
    /// }
    /// assert!(filter.estimated_fpr() > 0.0);
    /// ```
    #[must_use]
    fn estimated_fpr(&self) -> f64;

    /// Returns `true` if no items have been inserted into the filter.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use blumer::prelude::*;
    ///
    /// let mut filter = BloomFilter::new(100, 0.01).unwrap();
    /// assert!(filter.is_empty());
    /// filter.insert("hello");
    /// assert!(!filter.is_empty());
    /// ```
    #[must_use]
    fn is_empty(&self) -> bool {
        self.item_count() == 0
    }
}

/// A bloom filter that supports insertion and clearing.
///
/// Extends [`Filter`] with write operations. Implementors use `&mut self` for
/// mutations, making this suitable for single-threaded use.
///
/// For concurrent access, wrap the filter in a `Mutex` or `RwLock`.
pub trait MutableFilter: Filter {
    /// Inserts `item` into the filter.
    ///
    /// After this call, [`Filter::contains`] is guaranteed to return `true` for
    /// the same item. Inserting the same item multiple times is safe but
    /// increments [`Filter::item_count`] each time.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use blumer::prelude::*;
    ///
    /// let mut filter = BloomFilter::new(100, 0.01).unwrap();
    /// filter.insert("hello");
    /// assert!(filter.contains("hello"));
    /// ```
    fn insert<T: Bloomable + ?Sized>(&mut self, item: &T);

    /// Resets the filter to its initial empty state.
    ///
    /// All bits are cleared and the item count is set to zero. The filter's
    /// capacity, bit size, and hash function count are unchanged.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use blumer::prelude::*;
    ///
    /// let mut filter = BloomFilter::new(100, 0.01).unwrap();
    /// filter.insert("hello");
    /// filter.clear();
    /// assert!(filter.is_empty());
    /// assert!(!filter.contains("hello"));
    /// ```
    fn clear(&mut self);
}

/// A bloom filter that supports deletion.
///
/// Extends [`MutableFilter`] with a `remove` operation. Not all filter types
/// support deletion — standard bit-array filters cannot remove items because
/// a single bit may be shared by multiple items. Use [`CountingBloomFilter`]
/// or another counting-based implementation when you need this capability.
///
/// # Correctness contract
///
/// `remove` is only safe to call for items that were previously inserted.
/// Removing an item that was never inserted decrements counters that other
/// items depend on, which **will cause false negatives** — a serious
/// correctness violation. The caller is responsible for tracking what has
/// been inserted.
///
/// [`CountingBloomFilter`]: crate::CountingBloomFilter
pub trait RemovableFilter: MutableFilter {
    /// Removes `item` from the filter by decrementing its counters.
    ///
    /// Returns `true` if the item was probably present and was removed,
    /// `false` if it was definitely absent (no counters were modified).
    ///
    /// # Correctness
    ///
    /// Only call this for items you know were previously inserted. Removing
    /// an item that was never inserted will corrupt the filter and cause
    /// future [`Filter::contains`] calls to return incorrect results.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use blumer::prelude::*;
    ///
    /// let mut filter = CountingBloomFilter::new(100, 0.01).unwrap();
    /// filter.insert("hello");
    ///
    /// assert!(filter.contains("hello"));
    /// assert!(filter.remove("hello"));
    /// assert!(!filter.contains("hello"));
    ///
    /// // Removing an absent item returns false and leaves the filter unchanged.
    /// assert!(!filter.remove("world"));
    /// ```
    fn remove<T: Bloomable + ?Sized>(&mut self, item: &T) -> bool;
}

/// A bloom filter that supports concurrent insertion and membership testing
/// from multiple threads without external locking.
///
/// Unlike [`MutableFilter`], which requires exclusive `&mut self` access,
/// `ConcurrentFilter` uses atomic operations internally, allowing any number
/// of threads to call [`insert`] and [`Filter::contains`] simultaneously.
///
/// # No false negatives under concurrency
///
/// The guarantee holds across threads: if thread A inserts an item and thread
/// B subsequently calls `contains`, it will return `true`. This relies on
/// `Release` ordering on writes and `Acquire` ordering on reads, which form
/// a happens-before relationship across threads.
///
/// # Approximate item count
///
/// [`Filter::item_count`] is eventually consistent under concurrent use. Two
/// threads inserting simultaneously may observe stale counts. Do not treat
/// the value as exact when other threads are actively inserting.
///
/// # No concurrent `clear`
///
/// `clear` is intentionally absent from this trait. Resetting a filter while
/// other threads are reading or writing produces undefined logical state.
/// Implementations may provide `clear(&mut self)` directly — the `&mut self`
/// requirement enforces exclusive access at the type level.
///
/// [`insert`]: ConcurrentFilter::insert
pub trait ConcurrentFilter: Filter + Send + Sync {
    /// Inserts `item` into the filter.
    ///
    /// Takes `&self` rather than `&mut self`, allowing concurrent calls from
    /// multiple threads. Each bit position is set atomically using
    /// `fetch_or` with `Release` ordering.
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
    fn insert<T: Bloomable + ?Sized>(&self, item: &T);
}
