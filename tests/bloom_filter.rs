//! Integration tests for [`BloomFilter`].
//!
//! Tests are grouped into four categories:
//!
//! - **Construction** — valid and invalid parameter combinations, correct error
//!   variants.
//! - **No false negatives** — property-based: every inserted item is found.
//!   Covers `u64`, `&str`, and a custom [`Bloomable`] type.
//! - **Behavioural** — property-based: `clear`, `item_count`, and `estimated_fpr`
//!   behave according to their contracts.
//! - **Statistical** — one large fixed-dataset test that measures the actual
//!   false positive rate and asserts it stays within 3× the target. Kept as a
//!   plain `#[test]` (not a proptest) because statistical accuracy requires a
//!   large, stable dataset; small random inputs produce too much noise.

mod common;

use blume::{BloomError, BloomFilter, Filter, MutableFilter};
use common::data::arb_user_id;
use proptest::prelude::*;
use rstest::rstest;

// --- construction: valid params ---

/// `BloomFilter::new` accepts any positive capacity with an FPR strictly
/// between 0 and 1.
#[rstest]
#[case(1, 0.5)]
#[case(1_000, 0.01)]
#[case(1_000_000, 0.001)]
fn new_valid_params(#[case] capacity: usize, #[case] fpr: f64) {
    assert!(BloomFilter::new(capacity, fpr).is_ok());
}

/// `BloomFilter::with_params` accepts any positive bit count and hash function
/// count.
#[rstest]
#[case(64, 1)]
#[case(9_585, 7)]
#[case(100_000, 10)]
fn with_params_valid(#[case] bits: usize, #[case] hash_fns: usize) {
    assert!(BloomFilter::with_params(bits, hash_fns).is_ok());
}

// --- construction: invalid params ---

/// `BloomFilter::new` rejects zero capacity and FPR values outside `(0, 1)`.
#[rstest]
#[case(0, 0.01)]   // zero capacity
#[case(100, 0.0)]  // fpr boundary: zero
#[case(100, 1.0)]  // fpr boundary: one
#[case(100, -0.1)] // fpr negative
#[case(100, 1.1)]  // fpr above one
fn new_invalid_params(#[case] capacity: usize, #[case] fpr: f64) {
    assert!(BloomFilter::new(capacity, fpr).is_err());
}

/// `BloomFilter::with_params` rejects zero bits or zero hash functions.
#[rstest]
#[case(0, 7)]      // zero bits
#[case(9_585, 0)]  // zero hash functions
fn with_params_invalid(#[case] bits: usize, #[case] hash_fns: usize) {
    assert!(BloomFilter::with_params(bits, hash_fns).is_err());
}

/// Verifies that each invalid input maps to the correct [`BloomError`] variant
/// and, where applicable, the correct payload value.
///
/// Kept as a single test rather than an rstest parameterised case because the
/// cases produce different enum variants, which rstest cannot unify in a single
/// pattern parameter.
#[test]
fn construction_error_variants() {
    assert!(matches!(BloomFilter::new(0, 0.01),        Err(BloomError::InvalidCapacity(0))));
    assert!(matches!(BloomFilter::new(100, 0.0),       Err(BloomError::InvalidFpr(_))));
    assert!(matches!(BloomFilter::new(100, 1.0),       Err(BloomError::InvalidFpr(_))));
    assert!(matches!(BloomFilter::new(100, f64::NAN),  Err(BloomError::InvalidFpr(_))));
    assert!(matches!(BloomFilter::with_params(0, 7),   Err(BloomError::InvalidBitCount(0))));
    assert!(matches!(BloomFilter::with_params(100, 0), Err(BloomError::InvalidHashCount(0))));
}

// --- proptest: no false negatives ---
//
// A bloom filter must never report that an inserted item is absent. These
// three property tests cover the three major Bloomable impls: a primitive
// integer, a string slice, and a custom struct.

proptest! {
    /// Every `u64` inserted into the filter is subsequently found by `contains`.
    #[test]
    fn no_false_negatives_u64(items in prop::collection::vec(any::<u64>(), 1..1_000)) {
        let mut f = BloomFilter::new(items.len(), 0.01).unwrap();
        for item in &items { f.insert(item); }
        for item in &items { prop_assert!(f.contains(item)); }
    }

    /// Every `&str` inserted into the filter is subsequently found by `contains`.
    ///
    /// Uses a `hash_set` strategy so all strings are unique, keeping
    /// `item_count` equal to the number of distinct items and making the
    /// proptest assertions clean.
    #[test]
    fn no_false_negatives_string(
        items in prop::collection::hash_set("[a-z]{1,20}", 1..200usize)
    ) {
        let items: Vec<String> = items.into_iter().collect();
        let mut f = BloomFilter::new(items.len(), 0.01).unwrap();
        for item in &items { f.insert(item.as_str()); }
        for item in &items { prop_assert!(f.contains(item.as_str())); }
    }

    /// Every custom [`UserId`] inserted into the filter is subsequently found
    /// by `contains`. Exercises the user-defined [`Bloomable`] path end-to-end.
    #[test]
    fn no_false_negatives_user_id(
        items in prop::collection::vec(arb_user_id(), 1..500)
    ) {
        let mut f = BloomFilter::new(items.len(), 0.01).unwrap();
        for item in &items { f.insert(item); }
        for item in &items { prop_assert!(f.contains(item)); }
    }
}

// --- proptest: behavioural ---

proptest! {
    /// After `clear`, the filter reports empty and no previously inserted item
    /// is found. Verifies that `clear` zeroes all bits and resets the count.
    #[test]
    fn clear_resets_filter(items in prop::collection::vec(any::<u64>(), 1..500)) {
        let mut f = BloomFilter::new(items.len(), 0.01).unwrap();
        for item in &items { f.insert(item); }
        f.clear();
        prop_assert!(f.is_empty());
        for item in &items { prop_assert!(!f.contains(item)); }
    }

    /// `item_count` equals the number of `insert` calls, including duplicates.
    /// Bloom filters do not deduplicate; every insertion increments the counter.
    #[test]
    fn item_count_tracks_insertions(items in prop::collection::vec(any::<u64>(), 1..500)) {
        let mut f = BloomFilter::new(items.len(), 0.01).unwrap();
        for item in &items { f.insert(item); }
        prop_assert_eq!(f.item_count(), items.len());
    }

    /// `estimated_fpr` is exactly `0.0` on an empty filter and strictly
    /// positive after insertions, confirming that the FPR estimate responds
    /// correctly to the item count.
    #[test]
    fn estimated_fpr_starts_at_zero_and_rises(n in 10usize..200) {
        let mut f = BloomFilter::new(n, 0.01).unwrap();
        prop_assert_eq!(f.estimated_fpr(), 0.0);
        for i in 0..n as u64 { f.insert(&i); }
        prop_assert!(f.estimated_fpr() > 0.0);
    }
}

// --- statistical: fpr within bounds ---
//
// Kept as a plain test rather than a proptest because statistical validity
// requires a large fixed dataset — proptest's small random inputs would make
// this measurement too noisy to be meaningful.

/// Measures the empirical false positive rate against a disjoint probe set and
/// asserts it stays below 3× the configured target.
///
/// **Setup:** insert 10 000 contiguous `u64` values starting at `0`, then
/// probe 100 000 values starting at `1_000_000_000` (guaranteed not inserted).
/// Each probe that returns `true` is a false positive.
///
/// **Tolerance:** the 3× multiplier accounts for natural statistical variance
/// while still catching a badly broken filter. The expected measured rate for a
/// 1% target filter is well under 1.5%, so 3% provides a comfortable margin
/// without making the test meaningless.
#[test]
fn fpr_within_bounds() {
    let n = 10_000;
    let target = 0.01;
    let mut f = BloomFilter::new(n, target).unwrap();
    for i in 0..n as u64 { f.insert(&i); }

    let offset = 1_000_000_000u64;
    let trials = 100_000u64;
    let false_positives = (offset..offset + trials).filter(|i| f.contains(i)).count();
    let measured = false_positives as f64 / trials as f64;
    assert!(
        measured < target * 3.0,
        "measured FPR {measured:.4} exceeded 3× target {target}"
    );
}
