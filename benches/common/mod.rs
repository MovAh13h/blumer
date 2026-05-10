//! Shared benchmark suite for all bloom filter types.
//!
//! Each function takes a `make` closure that constructs a filter sized for `n`
//! items. To add benchmarks for a new filter type, create a new bench file and
//! call these functions with the appropriate constructor — no benchmark logic
//! needs to be duplicated.

use blume::MutableFilter;
use criterion::{BenchmarkGroup, BenchmarkId, Throughput, black_box, measurement::WallTime};

/// Capacities at which every benchmark is run.
///
/// Chosen to span three orders of magnitude so that cache effects
/// (L1 → L2 → L3/RAM) are visible in the results.
pub const SIZES: &[usize] = &[1_000, 100_000, 1_000_000];

/// False positive rate used when constructing filters.
pub const FPR: f64 = 0.01;

/// Benchmarks a single `insert` call at each capacity in [`SIZES`].
///
/// The filter is constructed once per size outside the hot loop. Items are
/// inserted with a wrapping counter so the compiler cannot constant-fold
/// the inputs away, and the filter never needs to be rebuilt between iterations.
pub fn bench_insert<F, MakeF>(group: &mut BenchmarkGroup<WallTime>, make: MakeF)
where
    F: MutableFilter,
    MakeF: Fn(usize) -> F,
{
    for &n in SIZES {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let mut f = make(n);
            let mut i = 0u64;
            b.iter(|| {
                f.insert(black_box(&i));
                i = i.wrapping_add(1);
            });
        });
    }
}

/// Benchmarks a single `contains` call for an item that **is** present.
///
/// All `n` items are inserted before timing begins. Lookups cycle through
/// the inserted set to prevent the CPU from predicting a constant address.
pub fn bench_contains_hit<F, MakeF>(group: &mut BenchmarkGroup<WallTime>, make: MakeF)
where
    F: MutableFilter,
    MakeF: Fn(usize) -> F,
{
    for &n in SIZES {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let items: Vec<u64> = (0..n as u64).collect();
            let mut f = make(n);
            for item in &items {
                f.insert(item);
            }
            let mut i = 0usize;
            b.iter(|| {
                let r = black_box(f.contains(&items[i % n]));
                i += 1;
                r
            });
        });
    }
}

/// Benchmarks a single `contains` call for an item that is **absent**.
///
/// Probes are drawn from a range far outside the inserted set, guaranteeing
/// they were never inserted. Any `true` result here is a genuine false positive.
pub fn bench_contains_miss<F, MakeF>(group: &mut BenchmarkGroup<WallTime>, make: MakeF)
where
    F: MutableFilter,
    MakeF: Fn(usize) -> F,
{
    for &n in SIZES {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let items: Vec<u64> = (0..n as u64).collect();
            let probes: Vec<u64> = (1_000_000_000u64..1_000_000_000 + n as u64).collect();
            let mut f = make(n);
            for item in &items {
                f.insert(item);
            }
            let mut i = 0usize;
            b.iter(|| {
                let r = black_box(f.contains(&probes[i % n]));
                i += 1;
                r
            });
        });
    }
}
