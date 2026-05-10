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
    /// use blume::{BloomFilter, Filter, MutableFilter};
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
    /// use blume::{BloomFilter, Filter, MutableFilter};
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
    /// use blume::{BloomFilter, Filter, MutableFilter};
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
    /// use blume::{BloomFilter, Filter, MutableFilter};
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
    /// use blume::{BloomFilter, Filter, MutableFilter};
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
    /// use blume::{BloomFilter, Filter, MutableFilter};
    ///
    /// let mut filter = BloomFilter::new(100, 0.01).unwrap();
    /// filter.insert("hello");
    /// filter.clear();
    /// assert!(filter.is_empty());
    /// assert!(!filter.contains("hello"));
    /// ```
    fn clear(&mut self);
}
