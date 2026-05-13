//! Integration tests for [`ScalableBloomFilter`].
//!
//! - **Construction** — valid and invalid parameters, error variants.
//! - **No false negatives** — every inserted item is always found.
//! - **Auto-growth** — new slices are added automatically when capacity fills.
//! - **Behavioural** — `clear`, `item_count`, `slice_count`, `estimated_fpr`.
//! - **Statistical** — empirical FPR within 3× target after heavy load.

mod common;

use blumer::prelude::*;
use common::data::arb_user_id;
use proptest::prelude::*;
use rstest::rstest;

// --- construction: valid params ---

#[rstest]
#[case(1, 0.5)]
#[case(1_000, 0.01)]
#[case(1_000_000, 0.001)]
fn new_valid_params(#[case] capacity: usize, #[case] fpr: f64) {
    assert!(ScalableBloomFilter::new(capacity, fpr).is_ok());
}

#[rstest]
#[case(100, 0.01, 2, 0.5)]
#[case(100, 0.01, 4, 0.9)]
#[case(100, 0.01, 2, 0.1)]
fn with_options_valid(
    #[case] capacity: usize,
    #[case] fpr: f64,
    #[case] growth: u32,
    #[case] tightening: f64,
) {
    assert!(ScalableBloomFilter::with_options(capacity, fpr, growth, tightening).is_ok());
}

// --- construction: invalid params ---

#[rstest]
#[case(0, 0.01)]
#[case(100, 0.0)]
#[case(100, 1.0)]
#[case(100, -0.1)]
#[case(100, 1.1)]
fn new_invalid_params(#[case] capacity: usize, #[case] fpr: f64) {
    assert!(ScalableBloomFilter::new(capacity, fpr).is_err());
}

#[rstest]
#[case(0, 0.01, 2, 0.5)]   // zero capacity
#[case(100, 0.0, 2, 0.5)]  // invalid fpr
#[case(100, 0.01, 1, 0.5)] // growth < 2
#[case(100, 0.01, 0, 0.5)] // growth = 0
#[case(100, 0.01, 2, 0.0)] // tightening boundary
#[case(100, 0.01, 2, 1.0)] // tightening boundary
fn with_options_invalid(
    #[case] capacity: usize,
    #[case] fpr: f64,
    #[case] growth: u32,
    #[case] tightening: f64,
) {
    assert!(ScalableBloomFilter::with_options(capacity, fpr, growth, tightening).is_err());
}

#[test]
fn construction_error_variants() {
    assert!(matches!(
        ScalableBloomFilter::new(0, 0.01),
        Err(BloomError::InvalidCapacity(0))
    ));
    assert!(matches!(
        ScalableBloomFilter::new(100, 0.0),
        Err(BloomError::InvalidFpr(_))
    ));
    assert!(matches!(
        ScalableBloomFilter::with_options(100, 0.01, 1, 0.5),
        Err(BloomError::InvalidGrowthFactor(1))
    ));
    assert!(matches!(
        ScalableBloomFilter::with_options(100, 0.01, 2, 0.0),
        Err(BloomError::InvalidTighteningRatio(_))
    ));
}

// --- proptest: no false negatives ---

proptest! {
    /// Every inserted u64 is found, even after the filter grows across slices.
    #[test]
    fn no_false_negatives_u64(items in prop::collection::vec(any::<u64>(), 1..2_000)) {
        let mut f = ScalableBloomFilter::new(100, 0.01).unwrap();
        for item in &items { f.insert(item); }
        for item in &items { prop_assert!(f.contains(item)); }
    }

    #[test]
    fn no_false_negatives_string(
        items in prop::collection::hash_set("[a-z]{1,20}", 1..500usize)
    ) {
        let items: Vec<String> = items.into_iter().collect();
        let mut f = ScalableBloomFilter::new(50, 0.01).unwrap();
        for item in &items { f.insert(item.as_str()); }
        for item in &items { prop_assert!(f.contains(item.as_str())); }
    }

    #[test]
    fn no_false_negatives_user_id(
        items in prop::collection::vec(arb_user_id(), 1..500)
    ) {
        let mut f = ScalableBloomFilter::new(50, 0.01).unwrap();
        for item in &items { f.insert(item); }
        for item in &items { prop_assert!(f.contains(item)); }
    }
}

// --- auto-growth ---

/// The filter adds a second slice once the first is full.
#[test]
fn grows_beyond_initial_capacity() {
    let initial = 100usize;
    let mut f = ScalableBloomFilter::new(initial, 0.01).unwrap();
    assert_eq!(f.slice_count(), 1);

    for i in 0..initial as u64 * 2 {
        f.insert(&i);
    }

    assert!(f.slice_count() > 1, "expected more than one slice after overflow");
}

/// Items inserted before and after a slice boundary are both found.
#[test]
fn items_across_slice_boundary_are_found() {
    let initial = 50usize;
    let mut f = ScalableBloomFilter::new(initial, 0.01).unwrap();

    for i in 0..initial as u64 * 4 {
        f.insert(&i);
    }

    for i in 0..initial as u64 * 4 {
        assert!(f.contains(&i), "item {i} missing after crossing slice boundary");
    }
}

// --- is_empty ---

#[test]
fn is_empty_reflects_insertions() {
    let mut f = ScalableBloomFilter::new(100, 0.01).unwrap();
    assert!(f.is_empty());
    f.insert(&1u64);
    assert!(!f.is_empty());
}

// --- proptest: behavioural ---

proptest! {
    #[test]
    fn clear_resets_to_one_slice(items in prop::collection::vec(any::<u64>(), 1..500)) {
        let mut f = ScalableBloomFilter::new(10, 0.01).unwrap();
        for item in &items { f.insert(item); }
        f.clear();
        prop_assert!(f.is_empty());
        prop_assert_eq!(f.slice_count(), 1);
        for item in &items { prop_assert!(!f.contains(item)); }
    }

    #[test]
    fn item_count_tracks_insertions(items in prop::collection::vec(any::<u64>(), 1..500)) {
        let mut f = ScalableBloomFilter::new(10, 0.01).unwrap();
        for item in &items { f.insert(item); }
        prop_assert_eq!(f.item_count(), items.len());
    }

    #[test]
    fn estimated_fpr_starts_at_zero_and_rises(n in 10usize..200) {
        let mut f = ScalableBloomFilter::new(n, 0.01).unwrap();
        prop_assert_eq!(f.estimated_fpr(), 0.0);
        for i in 0..n as u64 { f.insert(&i); }
        prop_assert!(f.estimated_fpr() > 0.0);
    }
}

// --- statistical: fpr within bounds ---

/// Inserts 10× the initial capacity to force multiple slices, then measures
/// the empirical FPR against a disjoint probe set.
#[test]
fn fpr_within_bounds_across_slices() {
    let initial = 1_000usize;
    let target = 0.01;
    let mut f = ScalableBloomFilter::new(initial, target).unwrap();

    // Insert 5× initial capacity to force growth.
    for i in 0..initial as u64 * 5 {
        f.insert(&i);
    }

    let offset = 1_000_000_000u64;
    let trials = 100_000u64;
    let false_positives = (offset..offset + trials).filter(|i| f.contains(i)).count();
    let measured = false_positives as f64 / trials as f64;

    assert!(
        measured < target * 3.0,
        "measured FPR {measured:.4} exceeded 3× target {target} with {} slices",
        f.slice_count()
    );
}
