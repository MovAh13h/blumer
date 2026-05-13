//! Integration tests for [`AtomicCountingBloomFilter`].
//!
//! - **Construction** — valid and invalid parameters, error variants.
//! - **No false negatives** — every inserted item is found.
//! - **Deletion** — removed items become absent; others are unaffected.
//! - **Concurrency** — concurrent inserts, contains, and removes are correct.
//! - **Behavioural** — `clear`, `item_count`, `is_empty`, `estimated_fpr`.
//! - **Statistical** — empirical FPR within expected bounds.

use blume::prelude::*;
use proptest::prelude::*;
use rstest::rstest;
use std::sync::Arc;

// --- construction ---

#[rstest]
#[case(1, 0.5)]
#[case(100, 0.01)]
#[case(10_000, 0.001)]
fn new_valid(#[case] capacity: usize, #[case] fpr: f64) {
    assert!(AtomicCountingBloomFilter::new(capacity, fpr).is_ok());
}

#[test]
fn new_zero_capacity_errors() {
    assert!(matches!(
        AtomicCountingBloomFilter::new(0, 0.01),
        Err(BloomError::InvalidCapacity(0))
    ));
}

#[rstest]
#[case(0.0)]
#[case(-0.5)]
#[case(1.0)]
#[case(1.5)]
#[case(f64::NAN)]
#[case(f64::INFINITY)]
fn new_invalid_fpr_errors(#[case] fpr: f64) {
    assert!(matches!(
        AtomicCountingBloomFilter::new(100, fpr),
        Err(BloomError::InvalidFpr(_))
    ));
}

#[test]
fn with_params_zero_counters_errors() {
    assert!(matches!(
        AtomicCountingBloomFilter::with_params(0, 7),
        Err(BloomError::InvalidBitCount(0))
    ));
}

#[test]
fn with_params_zero_hash_fns_errors() {
    assert!(matches!(
        AtomicCountingBloomFilter::with_params(1_000, 0),
        Err(BloomError::InvalidHashCount(0))
    ));
}

// --- no false negatives ---

proptest! {
    #[test]
    fn no_false_negatives_u64(
        items in prop::collection::hash_set(any::<u64>(), 1..500usize)
    ) {
        let items: Vec<u64> = items.into_iter().collect();
        let f = AtomicCountingBloomFilter::new(items.len() * 2, 0.01).unwrap();
        for item in &items { f.insert(item); }
        for item in &items { prop_assert!(f.contains(item)); }
    }

    #[test]
    fn no_false_negatives_string(
        items in prop::collection::hash_set("[a-z]{1,20}", 1..200usize)
    ) {
        let items: Vec<String> = items.into_iter().collect();
        let f = AtomicCountingBloomFilter::new(items.len() * 2, 0.01).unwrap();
        for item in &items { f.insert(item.as_str()); }
        for item in &items { prop_assert!(f.contains(item.as_str())); }
    }
}

// --- deletion ---

#[test]
fn remove_makes_item_absent() {
    let f = AtomicCountingBloomFilter::new(100, 0.01).unwrap();
    f.insert("alice");
    f.insert("bob");

    assert!(f.remove("alice"));
    assert!(!f.contains("alice"));
    assert!(f.contains("bob"));
}

#[test]
fn remove_absent_returns_false() {
    let f = AtomicCountingBloomFilter::new(100, 0.01).unwrap();
    assert!(!f.remove("never_inserted"));
}

#[test]
fn double_insert_requires_double_remove() {
    let f = AtomicCountingBloomFilter::new(100, 0.01).unwrap();
    f.insert(&42u64);
    f.insert(&42u64);

    f.remove(&42u64);
    assert!(f.contains(&42u64));

    f.remove(&42u64);
    assert!(!f.contains(&42u64));
}

proptest! {
    #[test]
    fn remove_only_item_makes_absent(item in any::<u64>()) {
        let f = AtomicCountingBloomFilter::new(100, 0.01).unwrap();
        f.insert(&item);
        f.remove(&item);
        prop_assert!(!f.contains(&item));
    }
}

// --- is_empty ---

#[test]
fn is_empty_reflects_insertions() {
    let f = AtomicCountingBloomFilter::new(100, 0.01).unwrap();
    assert!(f.is_empty());
    f.insert(&1u64);
    assert!(!f.is_empty());
}

// --- clear ---

#[test]
fn clear_resets_filter() {
    let mut f = AtomicCountingBloomFilter::new(100, 0.01).unwrap();
    f.insert(&1u64);
    f.insert(&2u64);
    f.clear();
    assert!(f.is_empty());
    assert!(!f.contains(&1u64));
    assert!(!f.contains(&2u64));
}

// --- item_count ---

proptest! {
    #[test]
    fn item_count_tracks_insertions(
        items in prop::collection::hash_set(any::<u64>(), 1..200usize)
    ) {
        let items: Vec<u64> = items.into_iter().collect();
        let f = AtomicCountingBloomFilter::new(items.len() * 2, 0.01).unwrap();
        for item in &items { f.insert(item); }
        prop_assert_eq!(f.item_count(), items.len());
    }

    #[test]
    fn item_count_decrements_on_remove(
        items in prop::collection::hash_set(any::<u64>(), 2..200usize)
    ) {
        let items: Vec<u64> = items.into_iter().collect();
        let f = AtomicCountingBloomFilter::new(items.len() * 2, 0.01).unwrap();
        for item in &items { f.insert(item); }

        let before = f.item_count();
        assert!(f.remove(&items[0]));
        assert_eq!(f.item_count(), before - 1);
    }

    #[test]
    fn estimated_fpr_starts_at_zero_and_rises(n in 10usize..100) {
        let f = AtomicCountingBloomFilter::new(n * 4, 0.01).unwrap();
        prop_assert_eq!(f.estimated_fpr(), 0.0);
        for i in 0..n as u64 { f.insert(&i); }
        prop_assert!(f.estimated_fpr() > 0.0);
    }
}

// --- concurrency ---

/// 8 threads each insert 1 000 distinct items; all items must be found after
/// all threads complete.
#[test]
fn concurrent_inserts_no_false_negatives() {
    let f = Arc::new(AtomicCountingBloomFilter::new(100_000, 0.01).unwrap());
    let threads: Vec<_> = (0..8u64)
        .map(|t| {
            let f = Arc::clone(&f);
            std::thread::spawn(move || {
                for i in 0..1_000u64 {
                    f.insert(&(t * 1_000 + i));
                }
            })
        })
        .collect();
    for t in threads { t.join().unwrap(); }

    for t in 0..8u64 {
        for i in 0..1_000u64 {
            assert!(f.contains(&(t * 1_000 + i)));
        }
    }
}

/// Concurrent inserts and contains: items inserted by one thread are visible
/// to another after the inserting thread joins.
#[test]
fn concurrent_insert_and_contains() {
    let f = Arc::new(AtomicCountingBloomFilter::new(10_000, 0.01).unwrap());
    let f2 = Arc::clone(&f);
    let inserter = std::thread::spawn(move || {
        for i in 0..1_000u64 { f2.insert(&i); }
    });
    inserter.join().unwrap();
    for i in 0..1_000u64 {
        assert!(f.contains(&i));
    }
}

/// Items inserted then removed by one thread are absent after the thread joins.
#[test]
fn concurrent_insert_then_remove() {
    let f = Arc::new(AtomicCountingBloomFilter::new(10_000, 0.01).unwrap());
    let f2 = Arc::clone(&f);
    let worker = std::thread::spawn(move || {
        for i in 0..500u64 { f2.insert(&i); }
        for i in 0..500u64 { f2.remove(&i); }
    });
    worker.join().unwrap();
    // Every item was inserted and removed — none should remain.
    let still_present = (0..500u64).filter(|i| f.contains(i)).count();
    // Allow a small number of false positives from hash collisions.
    assert!(
        still_present <= 10,
        "{still_present} items still present after remove (expected ~0)"
    );
}

// --- statistical: fpr within bounds ---

#[test]
fn fpr_within_expected_bounds() {
    let n = 1_000usize;
    let f = AtomicCountingBloomFilter::new(n, 0.01).unwrap();
    for i in 0..n as u64 { f.insert(&i); }

    let offset = 1_000_000_000u64;
    let trials = 100_000u64;
    let false_positives = (offset..offset + trials).filter(|i| f.contains(i)).count();
    let measured = false_positives as f64 / trials as f64;

    assert!(
        measured < 0.05,
        "measured FPR {measured:.4} exceeded 5% bound"
    );
}
