//! Integration tests for [`AtomicBloomFilter`].
//!
//! Covers the same ground as `bloom_filter.rs` plus concurrent correctness:
//!
//! - **Construction** — valid and invalid parameters, correct error variants.
//! - **No false negatives** — every inserted item is found, single-threaded.
//! - **Concurrent correctness** — inserts from multiple threads are all visible
//!   after all threads complete.
//! - **Behavioural** — `clear`, `item_count`, `estimated_fpr`.
//! - **Statistical** — empirical FPR within 3× target after full load.

mod common;

use std::sync::Arc;

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
    assert!(AtomicBloomFilter::new(capacity, fpr).is_ok());
}

#[rstest]
#[case(64, 1)]
#[case(9_585, 7)]
#[case(100_000, 10)]
fn with_params_valid(#[case] bits: usize, #[case] hash_fns: usize) {
    assert!(AtomicBloomFilter::with_params(bits, hash_fns).is_ok());
}

// --- construction: invalid params ---

#[rstest]
#[case(0, 0.01)]
#[case(100, 0.0)]
#[case(100, 1.0)]
#[case(100, -0.1)]
#[case(100, 1.1)]
fn new_invalid_params(#[case] capacity: usize, #[case] fpr: f64) {
    assert!(AtomicBloomFilter::new(capacity, fpr).is_err());
}

#[rstest]
#[case(0, 7)]
#[case(9_585, 0)]
fn with_params_invalid(#[case] bits: usize, #[case] hash_fns: usize) {
    assert!(AtomicBloomFilter::with_params(bits, hash_fns).is_err());
}

#[test]
fn construction_error_variants() {
    assert!(matches!(AtomicBloomFilter::new(0, 0.01),        Err(BloomError::InvalidCapacity(0))));
    assert!(matches!(AtomicBloomFilter::new(100, 0.0),       Err(BloomError::InvalidFpr(_))));
    assert!(matches!(AtomicBloomFilter::new(100, 1.0),       Err(BloomError::InvalidFpr(_))));
    assert!(matches!(AtomicBloomFilter::new(100, f64::NAN),  Err(BloomError::InvalidFpr(_))));
    assert!(matches!(AtomicBloomFilter::with_params(0, 7),   Err(BloomError::InvalidBitCount(0))));
    assert!(matches!(AtomicBloomFilter::with_params(100, 0), Err(BloomError::InvalidHashCount(0))));
}

// --- proptest: no false negatives (single-threaded) ---

proptest! {
    #[test]
    fn no_false_negatives_u64(items in prop::collection::vec(any::<u64>(), 1..1_000)) {
        let f = AtomicBloomFilter::new(items.len(), 0.01).unwrap();
        for item in &items { f.insert(item); }
        for item in &items { prop_assert!(f.contains(item)); }
    }

    #[test]
    fn no_false_negatives_string(
        items in prop::collection::hash_set("[a-z]{1,20}", 1..200usize)
    ) {
        let items: Vec<String> = items.into_iter().collect();
        let f = AtomicBloomFilter::new(items.len(), 0.01).unwrap();
        for item in &items { f.insert(item.as_str()); }
        for item in &items { prop_assert!(f.contains(item.as_str())); }
    }

    #[test]
    fn no_false_negatives_user_id(
        items in prop::collection::vec(arb_user_id(), 1..500)
    ) {
        let f = AtomicBloomFilter::new(items.len(), 0.01).unwrap();
        for item in &items { f.insert(item); }
        for item in &items { prop_assert!(f.contains(item)); }
    }
}

// --- concurrent correctness ---

/// Items inserted by separate threads are all visible after all threads join.
///
/// Uses 8 threads each inserting `n / 8` non-overlapping items. After joining,
/// every item must be found — verifying the Release/Acquire ordering guarantee.
#[test]
fn concurrent_inserts_no_false_negatives() {
    let n = 10_000usize;
    let thread_count = 8;
    let per_thread = n / thread_count;
    let filter = Arc::new(AtomicBloomFilter::new(n, 0.01).unwrap());

    let handles: Vec<_> = (0..thread_count)
        .map(|t| {
            let f = Arc::clone(&filter);
            std::thread::spawn(move || {
                let start = (t * per_thread) as u64;
                for i in start..start + per_thread as u64 {
                    f.insert(&i);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    for i in 0..n as u64 {
        assert!(filter.contains(&i), "item {i} missing after concurrent inserts");
    }
}

/// Concurrent inserts and contains calls on overlapping items do not produce
/// false negatives for items that were inserted before the contains call.
#[test]
fn concurrent_insert_and_contains() {
    let n = 5_000usize;
    let filter = Arc::new(AtomicBloomFilter::new(n * 2, 0.01).unwrap());

    // Pre-insert a known set before spawning threads.
    let pre_items: Vec<u64> = (0..n as u64).collect();
    for item in &pre_items {
        filter.insert(item);
    }

    // Concurrently insert a second set while checking the pre-inserted items.
    let f_insert = Arc::clone(&filter);
    let inserter = std::thread::spawn(move || {
        for i in n as u64..(n * 2) as u64 {
            f_insert.insert(&i);
        }
    });

    // Pre-inserted items must always be found, regardless of concurrent writes.
    for item in &pre_items {
        assert!(filter.contains(item), "pre-inserted item {item} missing");
    }

    inserter.join().unwrap();
}

// --- clone ---

/// A cloned filter contains all items from the original and is independent —
/// inserts into the clone do not affect the original.
#[test]
fn clone_preserves_state() {
    let f = AtomicBloomFilter::new(100, 0.01).unwrap();
    f.insert(&1u64);
    f.insert(&2u64);

    let clone = f.clone();

    assert!(clone.contains(&1u64));
    assert!(clone.contains(&2u64));
    assert_eq!(f.item_count(), clone.item_count());
    assert_eq!(f.bit_size(), clone.bit_size());
    assert_eq!(f.capacity(), clone.capacity());
}

/// Inserting into a clone does not affect the original.
#[test]
fn clone_is_independent() {
    let f = AtomicBloomFilter::new(100, 0.01).unwrap();
    let clone = f.clone();
    clone.insert(&42u64);
    assert!(!f.contains(&42u64));
}

// --- debug ---

#[test]
fn debug_format_contains_type_name() {
    let f = AtomicBloomFilter::new(100, 0.01).unwrap();
    let s = format!("{f:?}");
    assert!(s.contains("AtomicBloomFilter"));
}

// --- is_empty ---

#[test]
fn is_empty_reflects_insertions() {
    let f = AtomicBloomFilter::new(100, 0.01).unwrap();
    assert!(f.is_empty());
    f.insert(&1u64);
    assert!(!f.is_empty());
}

// --- clear ---

#[test]
fn clear_resets_filter() {
    let mut f = AtomicBloomFilter::new(100, 0.01).unwrap();
    f.insert(&1u64);
    f.insert(&2u64);
    f.clear();
    assert!(f.is_empty());
    assert!(!f.contains(&1u64));
    assert!(!f.contains(&2u64));
}

// --- merge ---

/// Merged filter contains items from both source filters.
#[test]
fn merge_contains_all_items() {
    let a = AtomicBloomFilter::new(1_000, 0.01).unwrap();
    let b = AtomicBloomFilter::new(1_000, 0.01).unwrap();
    a.insert("alice");
    b.insert("bob");

    let merged = a.merge(&b).unwrap();
    assert!(merged.contains("alice"));
    assert!(merged.contains("bob"));
}

/// Merging with an empty filter contains all items from the original.
#[test]
fn merge_with_empty_is_identity() {
    let a = AtomicBloomFilter::new(1_000, 0.01).unwrap();
    let b = AtomicBloomFilter::new(1_000, 0.01).unwrap();
    for i in 0..100u64 { a.insert(&i); }

    let merged = a.merge(&b).unwrap();
    for i in 0..100u64 { assert!(merged.contains(&i)); }
}

/// `item_count` on the merged filter is the sum of both source counts.
#[test]
fn merge_item_count_is_sum() {
    let a = AtomicBloomFilter::new(1_000, 0.01).unwrap();
    let b = AtomicBloomFilter::new(1_000, 0.01).unwrap();
    for i in 0..50u64 { a.insert(&i); }
    for i in 50..100u64 { b.insert(&i); }

    let merged = a.merge(&b).unwrap();
    assert_eq!(merged.item_count(), 100);
}

/// Merging filters with different geometries returns an error.
#[test]
fn merge_incompatible_returns_error() {
    let a = AtomicBloomFilter::new(1_000, 0.01).unwrap();
    let b = AtomicBloomFilter::new(500, 0.01).unwrap();
    assert!(matches!(a.merge(&b), Err(BloomError::IncompatibleGeometry { .. })));
}

/// `merge_from` atomically ORs all bits from `other` into `self`.
#[test]
fn merge_from_contains_all_items() {
    let dst = Arc::new(AtomicBloomFilter::new(1_000, 0.01).unwrap());
    let src = AtomicBloomFilter::new(1_000, 0.01).unwrap();
    dst.insert("alice");
    src.insert("bob");

    dst.merge_from(&src).unwrap();
    assert!(dst.contains("alice"));
    assert!(dst.contains("bob"));
}

/// `merge_from` is safe to call concurrently with `insert` and `contains`.
#[test]
fn merge_from_concurrent() {
    let dst = Arc::new(AtomicBloomFilter::new(10_000, 0.01).unwrap());
    let src = Arc::new(AtomicBloomFilter::new(10_000, 0.01).unwrap());

    for i in 0..1_000u64 { src.insert(&i); }

    let dst2 = Arc::clone(&dst);
    let src2 = Arc::clone(&src);
    let inserter = std::thread::spawn(move || {
        for i in 1_000..2_000u64 { dst2.insert(&i); }
    });

    dst.merge_from(&src).unwrap();
    inserter.join().unwrap();

    for i in 0..1_000u64 {
        assert!(dst.contains(&i), "item from src missing after merge_from: {i}");
    }
}

// --- proptest: behavioural ---

proptest! {
    #[test]
    fn item_count_tracks_insertions(items in prop::collection::vec(any::<u64>(), 1..500)) {
        let f = AtomicBloomFilter::new(items.len(), 0.01).unwrap();
        for item in &items { f.insert(item); }
        prop_assert_eq!(f.item_count(), items.len());
    }

    #[test]
    fn estimated_fpr_starts_at_zero_and_rises(n in 10usize..200) {
        let f = AtomicBloomFilter::new(n, 0.01).unwrap();
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
    let f = AtomicBloomFilter::new(n, target).unwrap();
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
