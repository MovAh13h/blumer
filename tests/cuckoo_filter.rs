//! Integration tests for [`CuckooFilter`].
//!
//! - **Construction** — valid and invalid parameters, error variants.
//! - **No false negatives** — every inserted item is found.
//! - **Deletion** — removed items become absent; others are unaffected.
//! - **Capacity** — insert fails gracefully when the filter is full.
//! - **Behavioural** — `clear`, `item_count`, `estimated_fpr`.
//! - **Statistical** — empirical FPR within expected bounds.

mod common;

use blumer::prelude::*;
use proptest::prelude::*;
use rstest::rstest;

// --- construction ---

#[rstest]
#[case(1)]
#[case(100)]
#[case(10_000)]
fn new_valid(#[case] capacity: usize) {
    assert!(CuckooFilter::new(capacity).is_ok());
}

#[test]
fn new_zero_capacity_errors() {
    assert!(matches!(
        CuckooFilter::new(0),
        Err(BloomError::InvalidCapacity(0))
    ));
}

#[test]
fn with_buckets_zero_errors() {
    assert!(matches!(
        CuckooFilter::with_buckets(0),
        Err(BloomError::InvalidCapacity(0))
    ));
}

#[test]
fn with_buckets_rounds_to_power_of_two() {
    let f = CuckooFilter::with_buckets(100).unwrap();
    assert!(f.num_buckets().is_power_of_two());
    assert!(f.num_buckets() >= 100);
}

// --- proptest: no false negatives ---

proptest! {
    /// Every inserted item is found immediately after insertion.
    #[test]
    fn no_false_negatives_u64(items in prop::collection::hash_set(any::<u64>(), 1..500usize)) {
        let items: Vec<u64> = items.into_iter().collect();
        let mut f = CuckooFilter::new(items.len() * 2).unwrap();
        for item in &items {
            f.insert(item).expect("insert failed");
        }
        for item in &items {
            prop_assert!(f.contains(item));
        }
    }

    #[test]
    fn no_false_negatives_string(
        items in prop::collection::hash_set("[a-z]{1,20}", 1..200usize)
    ) {
        let items: Vec<String> = items.into_iter().collect();
        let mut f = CuckooFilter::new(items.len() * 2).unwrap();
        for item in &items {
            f.insert(item.as_str()).expect("insert failed");
        }
        for item in &items {
            prop_assert!(f.contains(item.as_str()));
        }
    }
}

// --- deletion ---

/// Removing an inserted item makes it absent; other items are unaffected.
#[test]
fn remove_makes_item_absent() {
    let mut f = CuckooFilter::new(100).unwrap();
    f.insert("alice").unwrap();
    f.insert("bob").unwrap();

    assert!(f.remove("alice"));
    assert!(!f.contains("alice"));
    assert!(f.contains("bob"));
}

/// Removing from an empty filter returns false.
#[test]
fn remove_absent_returns_false() {
    let mut f = CuckooFilter::new(100).unwrap();
    assert!(!f.remove("never_inserted"));
}

/// Insert twice, remove once — item is still present (first fingerprint gone,
/// second remains).
#[test]
fn double_insert_requires_double_remove() {
    let mut f = CuckooFilter::new(100).unwrap();
    f.insert(&42u64).unwrap();
    f.insert(&42u64).unwrap();

    f.remove(&42u64);
    assert!(f.contains(&42u64));

    f.remove(&42u64);
    assert!(!f.contains(&42u64));
}

proptest! {
    /// Removing an item in a single-item filter always produces absence.
    #[test]
    fn remove_only_item_makes_absent(item in any::<u64>()) {
        let mut f = CuckooFilter::new(100).unwrap();
        f.insert(&item).unwrap();
        f.remove(&item);
        prop_assert!(!f.contains(&item));
    }
}

// --- capacity ---

/// Inserting at very high load eventually returns `CapacityExceeded`.
#[test]
fn insert_fails_when_full() {
    // Very small filter — fill it completely.
    let mut f = CuckooFilter::with_buckets(4).unwrap();
    let mut succeeded = 0usize;
    let mut failed = false;

    for i in 0u64..1_000 {
        match f.insert(&i) {
            Ok(()) => succeeded += 1,
            Err(BloomError::CapacityExceeded) => {
                failed = true;
                break;
            }
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    assert!(failed, "expected CapacityExceeded but all {succeeded} inserts succeeded");
}

// --- is_empty ---

#[test]
fn is_empty_reflects_insertions() {
    let mut f = CuckooFilter::new(100).unwrap();
    assert!(f.is_empty());
    f.insert(&1u64).unwrap();
    assert!(!f.is_empty());
}

// --- clear ---

#[test]
fn clear_resets_filter() {
    let mut f = CuckooFilter::new(100).unwrap();
    f.insert(&1u64).unwrap();
    f.insert(&2u64).unwrap();
    f.clear();
    assert!(f.is_empty());
    assert!(!f.contains(&1u64));
    assert!(!f.contains(&2u64));
}

// --- proptest: behavioural ---

proptest! {
    #[test]
    fn item_count_tracks_insertions(
        items in prop::collection::hash_set(any::<u64>(), 1..200usize)
    ) {
        let items: Vec<u64> = items.into_iter().collect();
        let mut f = CuckooFilter::new(items.len() * 2).unwrap();
        for item in &items { f.insert(item).unwrap(); }
        prop_assert_eq!(f.item_count(), items.len());
    }

    #[test]
    fn item_count_decrements_on_remove(
        items in prop::collection::hash_set(any::<u64>(), 2..200usize)
    ) {
        let items: Vec<u64> = items.into_iter().collect();
        let mut f = CuckooFilter::new(items.len() * 2).unwrap();
        for item in &items { f.insert(item).unwrap(); }

        let before = f.item_count();
        assert!(f.remove(&items[0]));
        assert_eq!(f.item_count(), before - 1);
    }

    #[test]
    fn estimated_fpr_starts_at_zero_and_rises(n in 10usize..100) {
        let mut f = CuckooFilter::new(n * 4).unwrap();
        prop_assert_eq!(f.estimated_fpr(), 0.0);
        for i in 0..n as u64 { f.insert(&i).unwrap(); }
        prop_assert!(f.estimated_fpr() > 0.0);
    }
}

// --- statistical: fpr within bounds ---

/// Inserts 1 000 distinct items, then probes 100 000 disjoint values and
/// asserts the measured FPR is below 5% (theoretical ~3% with 8-bit fingerprints).
#[test]
fn fpr_within_expected_bounds() {
    let n = 1_000usize;
    let mut f = CuckooFilter::new(n * 2).unwrap();
    for i in 0..n as u64 { f.insert(&i).unwrap(); }

    let offset = 1_000_000_000u64;
    let trials = 100_000u64;
    let false_positives = (offset..offset + trials).filter(|i| f.contains(i)).count();
    let measured = false_positives as f64 / trials as f64;

    // With 8-bit fingerprints the expected FPR is ~3%. Allow up to 5% for
    // statistical variance.
    assert!(
        measured < 0.05,
        "measured FPR {measured:.4} exceeded 5% bound for 8-bit cuckoo filter"
    );
}
