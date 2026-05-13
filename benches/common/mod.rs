//! Shared benchmark suite for all bloom filter types.
//!
//! Each function accepts a `make` closure that constructs a filter sized for
//! `n` items. To benchmark a new filter type, create a new bench file and call
//! these functions with the appropriate constructor.

use blume::{MutableFilter, RemovableFilter};
use criterion::{
    BatchSize, BenchmarkGroup, BenchmarkId, Throughput, black_box, measurement::WallTime,
};

/// Capacities at which every benchmark is run.
///
/// Chosen to span three orders of magnitude so that cache effects
/// (L1 → L2 → L3/RAM) are visible in the results.
pub const SIZES: &[usize] = &[1_000, 100_000, 1_000_000];

/// False positive rate used when constructing filters.
pub const FPR: f64 = 0.01;

/// Benchmarks a single `insert` call at each capacity in [`SIZES`].
///
/// Items are inserted with a wrapping counter so the compiler cannot
/// constant-fold the inputs away. The filter is constructed once per size
/// and is never rebuilt between iterations.
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
/// the full inserted set to prevent the CPU from predicting a constant address.
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
/// Probes are drawn from a range far outside the inserted set (`0..n`),
/// guaranteeing they were never inserted. Any `true` result is a genuine
/// false positive.
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

/// Benchmarks a single `remove` call for an item that **is** present.
///
/// A fresh, fully-populated filter is set up for each measurement via
/// `iter_batched` so the item is always present when timing begins. This
/// avoids the contamination that occurs when a shared filter empties and
/// subsequent calls hit the fast-exit absent path instead.
#[allow(dead_code)]
pub fn bench_remove_hit<F, MakeF>(group: &mut BenchmarkGroup<WallTime>, make: MakeF)
where
    F: RemovableFilter,
    MakeF: Fn(usize) -> F,
{
    for &n in SIZES {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let items: Vec<u64> = (0..n as u64).collect();
            b.iter_batched(
                || {
                    let mut f = make(n);
                    for item in &items {
                        f.insert(item);
                    }
                    f
                },
                |mut f| black_box(f.remove(&items[0])),
                BatchSize::LargeInput,
            );
        });
    }
}

/// Benchmarks a single `remove` call for an item that is **absent**.
///
/// Probes are drawn from a range far outside the inserted set, guaranteeing
/// they were never inserted. The filter is fully populated and never mutated
/// during timing (absent removes return `false` without modifying state).
#[allow(dead_code)]
pub fn bench_remove_miss<F, MakeF>(group: &mut BenchmarkGroup<WallTime>, make: MakeF)
where
    F: RemovableFilter,
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
                let r = black_box(f.remove(&probes[i % n]));
                i += 1;
                r
            });
        });
    }
}
