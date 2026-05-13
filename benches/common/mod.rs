//! Shared benchmark suite for all bloom filter types.
//!
//! Each function accepts a `make` closure that constructs a filter sized for
//! `n` items. To benchmark a new filter type, create a new bench file and call
//! these functions with the appropriate constructor.

// Each bench binary is a separate compilation unit and only calls a subset of
// these helpers. The unused-code lint fires per-binary, not across all binaries,
// so we suppress it at the module level rather than on each function.
#![allow(dead_code)]

use blumer::{ConcurrentFilter, MutableFilter, RemovableFilter};
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
/// The filter is pre-loaded with `n` items once. Each iteration times only the
/// `remove` call, then immediately re-inserts the item (outside the timing
/// window) to restore state for the next iteration. This keeps all `n` items
/// in the filter throughout, maintaining realistic memory pressure at every
/// size without the per-iteration setup overhead that breaks `iter_batched` at
/// large `n`.
pub fn bench_remove_hit<F, MakeF>(group: &mut BenchmarkGroup<WallTime>, make: MakeF)
where
    F: RemovableFilter,
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
            let mut idx = 0usize;
            b.iter_custom(|iters| {
                let mut total = std::time::Duration::ZERO;
                for _ in 0..iters {
                    let target = &items[idx % n];
                    idx += 1;
                    let start = std::time::Instant::now();
                    black_box(f.remove(target));
                    total += start.elapsed();
                    f.insert(target); // restore — not timed
                }
                total
            });
        });
    }
}

/// Benchmarks a single `remove` call for an item that is **absent**.
///
/// Probes are drawn from a range far outside the inserted set, guaranteeing
/// they were never inserted. The filter is fully populated and never mutated
/// during timing (absent removes return `false` without modifying state).
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

/// Benchmarks a single `insert` call on a [`ConcurrentFilter`] at each
/// capacity in [`SIZES`].
///
/// Uses `&self` insertion, measuring the overhead of atomic `fetch_or` vs the
/// plain bit-set in `bench_insert`. Single-threaded — the goal is to isolate
/// the per-operation atomic cost, not multi-threaded throughput.
pub fn bench_concurrent_insert<F, MakeF>(group: &mut BenchmarkGroup<WallTime>, make: MakeF)
where
    F: ConcurrentFilter,
    MakeF: Fn(usize) -> F,
{
    for &n in SIZES {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let f = make(n);
            let mut i = 0u64;
            b.iter(|| {
                f.insert(black_box(&i));
                i = i.wrapping_add(1);
            });
        });
    }
}

/// Benchmarks a single `contains` call on a [`ConcurrentFilter`] for an item
/// that **is** present.
///
/// Measures the overhead of `Acquire` loads vs plain loads in `bench_contains_hit`.
pub fn bench_concurrent_contains_hit<F, MakeF>(group: &mut BenchmarkGroup<WallTime>, make: MakeF)
where
    F: ConcurrentFilter,
    MakeF: Fn(usize) -> F,
{
    for &n in SIZES {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let items: Vec<u64> = (0..n as u64).collect();
            let f = make(n);
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

/// Benchmarks a single `contains` call on a [`ConcurrentFilter`] for an item
/// that is **absent**.
pub fn bench_concurrent_contains_miss<F, MakeF>(group: &mut BenchmarkGroup<WallTime>, make: MakeF)
where
    F: ConcurrentFilter,
    MakeF: Fn(usize) -> F,
{
    for &n in SIZES {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let items: Vec<u64> = (0..n as u64).collect();
            let probes: Vec<u64> = (1_000_000_000u64..1_000_000_000 + n as u64).collect();
            let f = make(n);
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
