//! Integration tests for [`CountingBloomFilter`].
//!
//! Mirrors the structure of `bloom_filter.rs` and adds a deletion-specific
//! section that exercises the contracts unique to [`RemovableFilter`]:
//!
//! - **Construction** — valid and invalid parameters, correct error variants.
//! - **No false negatives** — every inserted item is found.
//! - **Deletion** — removed items become absent; non-removed items are
//!   unaffected; removing absent items is a no-op.
//! - **Behavioural** — `clear`, `item_count`, `estimated_fpr` contracts.
//! - **Statistical** — empirical FPR within 3× target after full load.

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
    assert!(CountingBloomFilter::new(capacity, fpr).is_ok());
}

#[rstest]
#[case(64, 1)]
#[case(9_585, 7)]
#[case(100_000, 10)]
fn with_params_valid(#[case] counters: usize, #[case] hash_fns: usize) {
    assert!(CountingBloomFilter::with_params(counters, hash_fns).is_ok());
}

// --- construction: invalid params ---

#[rstest]
#[case(0, 0.01)]
#[case(100, 0.0)]
#[case(100, 1.0)]
#[case(100, -0.1)]
#[case(100, 1.1)]
fn new_invalid_params(#[case] capacity: usize, #[case] fpr: f64) {
    assert!(CountingBloomFilter::new(capacity, fpr).is_err());
}

#[rstest]
#[case(0, 7)]
#[case(9_585, 0)]
fn with_params_invalid(#[case] counters: usize, #[case] hash_fns: usize) {
    assert!(CountingBloomFilter::with_params(counters, hash_fns).is_err());
}

#[test]
fn construction_error_variants() {
    assert!(matches!(CountingBloomFilter::new(0, 0.01),        Err(BloomError::InvalidCapacity(0))));
    assert!(matches!(CountingBloomFilter::new(100, 0.0),       Err(BloomError::InvalidFpr(_))));
    assert!(matches!(CountingBloomFilter::new(100, 1.0),       Err(BloomError::InvalidFpr(_))));
    assert!(matches!(CountingBloomFilter::new(100, f64::NAN),  Err(BloomError::InvalidFpr(_))));
    assert!(matches!(CountingBloomFilter::with_params(0, 7),   Err(BloomError::InvalidBitCount(0))));
    assert!(matches!(CountingBloomFilter::with_params(100, 0), Err(BloomError::InvalidHashCount(0))));
}

// --- proptest: no false negatives ---

proptest! {
    #[test]
    fn no_false_negatives_u64(items in prop::collection::vec(any::<u64>(), 1..1_000)) {
        let mut f = CountingBloomFilter::new(items.len(), 0.01).unwrap();
        for item in &items { f.insert(item); }
        for item in &items { prop_assert!(f.contains(item)); }
    }

    #[test]
    fn no_false_negatives_string(
        items in prop::collection::hash_set("[a-z]{1,20}", 1..200usize)
    ) {
        let items: Vec<String> = items.into_iter().collect();
        let mut f = CountingBloomFilter::new(items.len(), 0.01).unwrap();
        for item in &items { f.insert(item.as_str()); }
        for item in &items { prop_assert!(f.contains(item.as_str())); }
    }

    #[test]
    fn no_false_negatives_user_id(
        items in prop::collection::vec(arb_user_id(), 1..500)
    ) {
        let mut f = CountingBloomFilter::new(items.len(), 0.01).unwrap();
        for item in &items { f.insert(item); }
        for item in &items { prop_assert!(f.contains(item)); }
    }
}

// --- deletion ---
//
// Absence after removal is only guaranteed in a single-item filter: with
// multiple items, shared counter positions can keep an item's counters above
// zero even after decrementing, producing a post-removal false positive.
// The multi-item tests therefore only assert what is unconditionally true:
// `remove` returns the right bool and non-removed items are unaffected.

/// The only item in the filter is absent after removal — no shared counters,
/// no false positives possible.
#[test]
fn remove_only_item_makes_absent() {
    let mut f = CountingBloomFilter::new(1_000, 0.01).unwrap();
    f.insert(&42u64);
    assert!(f.remove(&42u64));
    assert!(!f.contains(&42u64));
}

proptest! {
    /// Removing an inserted item returns `true`; all other items remain present.
    ///
    /// Does not assert the removed item becomes absent — shared counters can
    /// keep all k positions non-zero even after decrement.
    #[test]
    fn remove_inserted_item_returns_true(
        items in prop::collection::hash_set(any::<u64>(), 2..500usize)
    ) {
        let items: Vec<u64> = items.into_iter().collect();
        let mut f = CountingBloomFilter::new(items.len(), 0.01).unwrap();
        for item in &items { f.insert(item); }

        prop_assert!(f.remove(&items[0]));

        for item in &items[1..] {
            prop_assert!(f.contains(item));
        }
    }

    /// Removing from an empty filter always returns `false`.
    ///
    /// Empty filter is used to guarantee absence — no false positives possible.
    #[test]
    fn remove_absent_item_returns_false(item in any::<u64>()) {
        let mut f = CountingBloomFilter::new(1_000, 0.01).unwrap();
        prop_assert!(!f.remove(&item));
    }

    /// Inserting an item twice and removing it once keeps it present; removing
    /// it a second time makes it absent. Only one item is in the filter so
    /// there are no shared counters and absence is guaranteed.
    #[test]
    fn double_insert_requires_double_remove(item in any::<u64>()) {
        let mut f = CountingBloomFilter::new(100, 0.01).unwrap();
        f.insert(&item);
        f.insert(&item);

        f.remove(&item);
        prop_assert!(f.contains(&item), "still present after one remove of two inserts");

        f.remove(&item);
        prop_assert!(!f.contains(&item), "absent after two removes matching two inserts");
    }
}

// --- is_empty ---

#[test]
fn is_empty_reflects_insertions() {
    let mut f = CountingBloomFilter::new(100, 0.01).unwrap();
    assert!(f.is_empty());
    f.insert(&1u64);
    assert!(!f.is_empty());
    f.clear();
    assert!(f.is_empty());
}

// --- proptest: behavioural ---

proptest! {
    #[test]
    fn clear_resets_filter(items in prop::collection::vec(any::<u64>(), 1..500)) {
        let mut f = CountingBloomFilter::new(items.len(), 0.01).unwrap();
        for item in &items { f.insert(item); }
        f.clear();
        prop_assert!(f.is_empty());
        for item in &items { prop_assert!(!f.contains(item)); }
    }

    #[test]
    fn item_count_tracks_insertions(items in prop::collection::vec(any::<u64>(), 1..500)) {
        let mut f = CountingBloomFilter::new(items.len(), 0.01).unwrap();
        for item in &items { f.insert(item); }
        prop_assert_eq!(f.item_count(), items.len());
    }

    /// `item_count` decrements by one after a successful remove.
    #[test]
    fn item_count_decrements_on_remove(
        items in prop::collection::hash_set(any::<u64>(), 1..500usize)
    ) {
        let items: Vec<u64> = items.into_iter().collect();
        let mut f = CountingBloomFilter::new(items.len(), 0.01).unwrap();
        for item in &items { f.insert(item); }

        let before = f.item_count();
        f.remove(&items[0]);
        prop_assert_eq!(f.item_count(), before - 1);
    }

    /// `item_count` is unchanged when removing from an empty filter.
    ///
    /// Empty filter is used to guarantee the item is absent — a populated
    /// filter can produce false positives that would trigger an actual remove.
    #[test]
    fn item_count_unchanged_on_remove_absent(item in any::<u64>()) {
        let mut f = CountingBloomFilter::new(1_000, 0.01).unwrap();
        f.remove(&item);
        prop_assert_eq!(f.item_count(), 0);
    }

    #[test]
    fn estimated_fpr_starts_at_zero_and_rises(n in 10usize..200) {
        let mut f = CountingBloomFilter::new(n, 0.01).unwrap();
        prop_assert_eq!(f.estimated_fpr(), 0.0);
        for i in 0..n as u64 { f.insert(&i); }
        prop_assert!(f.estimated_fpr() > 0.0);
    }
}

// --- statistical: fpr within bounds ---

#[test]
fn fpr_within_bounds() {
    let n = 10_000;
    let target = 0.01;
    let mut f = CountingBloomFilter::new(n, target).unwrap();
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
