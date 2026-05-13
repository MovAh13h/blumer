//! Scalable bloom filter — grows automatically as items are inserted.

use crate::error::BloomError;
use crate::filters::BloomFilter;
use crate::traits::{Filter, MutableFilter};
use crate::Bloomable;

/// Default capacity growth factor between slices.
const DEFAULT_GROWTH: u32 = 2;

/// Default FPR tightening ratio between slices.
const DEFAULT_TIGHTENING: f64 = 0.5;

/// A bloom filter that grows automatically as items are inserted.
///
/// A standard bloom filter has a fixed capacity — inserting beyond it causes
/// the false positive rate to rise above the target. A `ScalableBloomFilter`
/// avoids this by maintaining a list of **slices** (plain [`BloomFilter`]s),
/// adding a new, larger slice whenever the current one fills up.
///
/// # How it works
///
/// The filter starts with one slice of `initial_capacity` items. When that
/// slice is full (its `item_count` reaches its capacity), a new slice is
/// added. Each successive slice is `growth` times larger in capacity.
///
/// To keep the overall FPR bounded by the target, each new slice's FPR is
/// tightened by multiplying by `tightening`. Slice `i` has:
///
/// ```text
/// capacity_i = initial_capacity × growth^i
/// fpr_i      = target_fpr × (1 − tightening) × tightening^i
/// ```
///
/// The geometric series sums to exactly `target_fpr`:
///
/// ```text
/// Σ fpr_i = target_fpr × (1 − tightening) × Σ tightening^i
///         = target_fpr × (1 − tightening) / (1 − tightening)
///         = target_fpr
/// ```
///
/// # Defaults
///
/// | Parameter    | Default | Notes |
/// |--------------|---------|-------|
/// | `growth`     | `2`     | Each slice is 2× the previous capacity |
/// | `tightening` | `0.5`   | Each slice targets half the previous FPR |
///
/// # Contains
///
/// An item may be in any slice, so `contains` checks all slices. This is
/// `O(slices × k)` — typically `O(log n)` slices for a filter holding `n`
/// items.
///
/// # Memory
///
/// Total memory grows with the number of slices but is bounded by roughly
/// `2 × memory(last_slice)` because the geometric sum of slice sizes is
/// dominated by the last term.
///
/// # Examples
///
/// ```rust
/// use blumer::prelude::*;
///
/// let mut filter = ScalableBloomFilter::new(100, 0.01).unwrap();
///
/// // Insert well beyond the initial capacity — the filter grows automatically.
/// for i in 0u64..1_000 {
///     filter.insert(&i);
/// }
///
/// for i in 0u64..1_000 {
///     assert!(filter.contains(&i));
/// }
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ScalableBloomFilter {
    slices: Vec<BloomFilter>,
    initial_capacity: usize,
    target_fpr: f64,
    growth: u32,
    tightening: f64,
}

impl ScalableBloomFilter {
    /// Creates a scalable bloom filter with default growth (`2`) and
    /// tightening (`0.5`) parameters.
    ///
    /// # Errors
    ///
    /// Returns [`BloomError::InvalidCapacity`] if `initial_capacity` is `0`.
    /// Returns [`BloomError::InvalidFpr`] if `target_fpr` is not in `(0, 1)`
    /// or is `NaN` or infinite.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use blumer::prelude::*;
    ///
    /// let mut filter = ScalableBloomFilter::new(1_000, 0.01).unwrap();
    /// filter.insert("hello");
    /// assert!(filter.contains("hello"));
    /// ```
    pub fn new(initial_capacity: usize, target_fpr: f64) -> Result<Self, BloomError> {
        Self::with_options(initial_capacity, target_fpr, DEFAULT_GROWTH, DEFAULT_TIGHTENING)
    }

    /// Creates a scalable bloom filter with explicit growth and tightening
    /// parameters.
    ///
    /// - `growth` — capacity multiplier per slice; must be `>= 2`.
    /// - `tightening` — FPR multiplier per slice; must be in `(0, 1)`.
    ///   Smaller values reduce FPR faster but use more memory per slice.
    ///
    /// # Errors
    ///
    /// Returns [`BloomError::InvalidCapacity`] if `initial_capacity` is `0`.
    /// Returns [`BloomError::InvalidFpr`] if `target_fpr` is not in `(0, 1)`.
    /// Returns [`BloomError::InvalidGrowthFactor`] if `growth < 2`.
    /// Returns [`BloomError::InvalidTighteningRatio`] if `tightening` is not
    /// in `(0, 1)`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use blumer::prelude::*;
    ///
    /// // 4× growth, aggressive tightening — fewer slices, each using more memory.
    /// let mut filter = ScalableBloomFilter::with_options(1_000, 0.01, 4, 0.9).unwrap();
    /// filter.insert("hello");
    /// assert!(filter.contains("hello"));
    /// ```
    pub fn with_options(
        initial_capacity: usize,
        target_fpr: f64,
        growth: u32,
        tightening: f64,
    ) -> Result<Self, BloomError> {
        if initial_capacity == 0 {
            return Err(BloomError::InvalidCapacity(initial_capacity));
        }
        if !target_fpr.is_finite() || target_fpr <= 0.0 || target_fpr >= 1.0 {
            return Err(BloomError::InvalidFpr(target_fpr));
        }
        if growth < 2 {
            return Err(BloomError::InvalidGrowthFactor(growth));
        }
        if !tightening.is_finite() || tightening <= 0.0 || tightening >= 1.0 {
            return Err(BloomError::InvalidTighteningRatio(tightening));
        }

        let mut filter = Self {
            slices: Vec::new(),
            initial_capacity,
            target_fpr,
            growth,
            tightening,
        };
        filter.add_slice();
        Ok(filter)
    }

    /// Returns the number of slices currently allocated.
    pub fn slice_count(&self) -> usize {
        self.slices.len()
    }

    /// Adds a new slice sized for the next tier of capacity and FPR.
    fn add_slice(&mut self) {
        let i = self.slices.len();
        let capacity = self.initial_capacity.saturating_mul((self.growth as usize).pow(i as u32));
        let fpr = self.target_fpr * (1.0 - self.tightening) * self.tightening.powi(i as i32);
        // fpr_i converges toward 0 as i grows; clamp to the smallest positive
        // f64 to keep it within BloomFilter::new's valid range.
        let fpr = fpr.max(f64::MIN_POSITIVE);
        // capacity uses saturating_mul so overflow produces usize::MAX rather
        // than wrapping. BloomFilter::new will then attempt to allocate an
        // enormous bit array and the allocator will abort — acceptable since
        // reaching this point requires an astronomically large number of slices.
        self.slices.push(BloomFilter::new(capacity, fpr).unwrap());
    }
}

impl Filter for ScalableBloomFilter {
    /// Returns `true` if `item` is probably in any slice, `false` if it is
    /// definitely absent from all slices.
    ///
    /// Checks slices in insertion order, short-circuiting on the first hit.
    #[inline]
    fn contains<T: Bloomable + ?Sized>(&self, item: &T) -> bool {
        self.slices.iter().any(|s| s.contains(item))
    }

    /// Returns the total number of insertions across all slices.
    fn item_count(&self) -> usize {
        self.slices.iter().map(|s| s.item_count()).sum()
    }

    /// Returns the total number of bits across all slices.
    fn bit_size(&self) -> usize {
        self.slices.iter().map(|s| s.bit_size()).sum()
    }

    /// Returns the total capacity across all current slices.
    fn capacity(&self) -> usize {
        self.slices.iter().map(|s| s.capacity()).sum()
    }

    /// Returns the estimated overall false positive rate.
    ///
    /// Computed as `1 − ∏(1 − fpr_i)` over all slices, which is the
    /// probability that at least one slice produces a false positive for an
    /// absent item.
    fn estimated_fpr(&self) -> f64 {
        if self.item_count() == 0 {
            return 0.0;
        }
        1.0 - self.slices.iter().map(|s| 1.0 - s.estimated_fpr()).product::<f64>()
    }
}

impl MutableFilter for ScalableBloomFilter {
    /// Inserts `item` into the current slice.
    ///
    /// If the current slice has reached its designed capacity, a new slice is
    /// added automatically before insertion.
    #[inline]
    fn insert<T: Bloomable + ?Sized>(&mut self, item: &T) {
        if self.slices.last().is_none_or(|s| s.item_count() >= s.capacity()) {
            self.add_slice();
        }
        self.slices.last_mut().unwrap().insert(item);
    }

    /// Resets the filter to its initial state — one empty slice.
    fn clear(&mut self) {
        self.slices.clear();
        self.add_slice();
    }
}
