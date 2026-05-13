//! Round-trip serialization tests for all filter types.
//!
//! This file requires the `serde` feature:
//!
//! ```sh
//! cargo test --features serde
//! ```
//!
//! # Adding tests for a new filter type
//!
//! Call the macro with the snake_case module name and the filter type:
//!
//! ```rust,ignore
//! serde_tests!(scalable_filter, ScalableBloomFilter);
//! ```
//!
//! This generates a full test module covering empty, half-full, and at-capacity
//! round-trips, plus a `with_params` constructor round-trip.

use blume::prelude::*;
use serde::{Serialize, de::DeserializeOwned};

// --- shared helper ---

/// Serializes `original` to JSON, deserializes it back, and asserts that the
/// restored filter:
/// - has identical metadata (`item_count`, `bit_size`, `capacity`, `is_empty`, `estimated_fpr`)
/// - returns `true` for every item in `inserted`
/// - returns the same `contains` result as `original` for every item in both
///   `inserted` and `absent` (guarantees bit-for-bit state equality)
fn assert_round_trip<F>(original: &F, inserted: &[u64], absent: &[u64])
where
    F: Serialize + DeserializeOwned + Filter,
{
    let json = serde_json::to_string(original).expect("serialization failed");
    let restored: F = serde_json::from_str(&json).expect("deserialization failed");

    assert_eq!(original.item_count(),    restored.item_count(),    "item_count mismatch");
    assert_eq!(original.bit_size(),      restored.bit_size(),      "bit_size mismatch");
    assert_eq!(original.capacity(),      restored.capacity(),      "capacity mismatch");
    assert_eq!(original.is_empty(),      restored.is_empty(),      "is_empty mismatch");
    assert_eq!(original.estimated_fpr(), restored.estimated_fpr(), "estimated_fpr mismatch");

    for item in inserted {
        assert!(
            restored.contains(item),
            "inserted item {item} not found after round-trip"
        );
        assert_eq!(
            original.contains(item),
            restored.contains(item),
            "contains diverged for inserted item {item}"
        );
    }

    for item in absent {
        assert_eq!(
            original.contains(item),
            restored.contains(item),
            "contains diverged for absent item {item}"
        );
    }
}

// --- per-filter test generation ---

/// Generates serde round-trip tests for a filter type.
///
/// Two forms:
/// - `serde_tests!(name, Type)` — for filters with a `with_params(bits,
///   hash_fns)` constructor. Generates 4 tests including `with_params_round_trip`.
/// - `serde_tests!(name, Type, no_with_params)` — for filters whose expert
///   constructor has a different signature (e.g. `ScalableBloomFilter`).
///   Generates the 3 core round-trip tests only.
macro_rules! serde_tests {
    ($mod_name:ident, $filter:ty) => {
        mod $mod_name {
            use blume::prelude::*;
            use super::assert_round_trip;

            fn make(n: usize, p: f64) -> $filter {
                <$filter>::new(n, p).unwrap()
            }

            #[test]
            fn empty_round_trip() {
                let f = make(1_000, 0.01);
                assert_round_trip(&f, &[], &[]);
            }

            #[test]
            fn half_capacity_round_trip() {
                let mut f = make(1_000, 0.01);
                let items: Vec<u64> = (0..500).collect();
                for item in &items { f.insert(item); }
                let absent: Vec<u64> = (1_000_000u64..1_000_500).collect();
                assert_round_trip(&f, &items, &absent);
            }

            #[test]
            fn full_capacity_round_trip() {
                let n = 1_000usize;
                let mut f = make(n, 0.01);
                let items: Vec<u64> = (0..n as u64).collect();
                for item in &items { f.insert(item); }
                let absent: Vec<u64> = (1_000_000u64..1_000_000 + n as u64).collect();
                assert_round_trip(&f, &items, &absent);
            }

            /// Verifies that explicit geometry (`with_params`) survives serialization.
            #[test]
            fn with_params_round_trip() {
                let mut f = <$filter>::with_params(9_585, 7).unwrap();
                let items: Vec<u64> = (0..500).collect();
                for item in &items { f.insert(item); }
                let absent: Vec<u64> = (1_000_000u64..1_000_500).collect();
                assert_round_trip(&f, &items, &absent);
            }
        }
    };
    ($mod_name:ident, $filter:ty, no_with_params) => {
        mod $mod_name {
            use blume::prelude::*;
            use super::assert_round_trip;

            fn make(n: usize, p: f64) -> $filter {
                <$filter>::new(n, p).unwrap()
            }

            #[test]
            fn empty_round_trip() {
                let f = make(1_000, 0.01);
                assert_round_trip(&f, &[], &[]);
            }

            #[test]
            fn half_capacity_round_trip() {
                let mut f = make(1_000, 0.01);
                let items: Vec<u64> = (0..500).collect();
                for item in &items { f.insert(item); }
                let absent: Vec<u64> = (1_000_000u64..1_000_500).collect();
                assert_round_trip(&f, &items, &absent);
            }

            #[test]
            fn full_capacity_round_trip() {
                let n = 1_000usize;
                let mut f = make(n, 0.01);
                let items: Vec<u64> = (0..n as u64).collect();
                for item in &items { f.insert(item); }
                let absent: Vec<u64> = (1_000_000u64..1_000_000 + n as u64).collect();
                assert_round_trip(&f, &items, &absent);
            }
        }
    };
}

// --- filter registrations ---
// To add a new filter: serde_tests!(module_name, FilterType);

serde_tests!(bloom_filter, BloomFilter);
serde_tests!(counting_filter, CountingBloomFilter);
serde_tests!(atomic_bloom_filter, AtomicBloomFilter);
serde_tests!(atomic_counting_bloom_filter, AtomicCountingBloomFilter);
serde_tests!(scalable_bloom_filter, ScalableBloomFilter, no_with_params);

// CuckooFilter::new takes only capacity (no FPR), so it can't use the macro.
mod cuckoo_filter {
    use blume::prelude::*;
    use super::assert_round_trip;

    #[test]
    fn empty_round_trip() {
        let f = CuckooFilter::new(1_000).unwrap();
        assert_round_trip(&f, &[], &[]);
    }

    #[test]
    fn populated_round_trip() {
        let mut f = CuckooFilter::new(1_000).unwrap();
        let items: Vec<u64> = (0..200).collect();
        for item in &items { f.insert(item).unwrap(); }
        let absent: Vec<u64> = (1_000_000u64..1_000_200).collect();
        assert_round_trip(&f, &items, &absent);
    }

    #[test]
    fn with_buckets_round_trip() {
        let mut f = CuckooFilter::with_buckets(256).unwrap();
        let items: Vec<u64> = (0..100).collect();
        for item in &items { f.insert(item).unwrap(); }
        let absent: Vec<u64> = (1_000_000u64..1_000_100).collect();
        assert_round_trip(&f, &items, &absent);
    }
}
