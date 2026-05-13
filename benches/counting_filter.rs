mod common;

use blume::CountingBloomFilter;
use criterion::{Criterion, criterion_group, criterion_main};

fn make(n: usize) -> CountingBloomFilter {
    CountingBloomFilter::new(n, common::FPR).unwrap()
}

fn insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("counting_filter/insert");
    common::bench_insert(&mut group, make);
    group.finish();
}

fn contains_hit(c: &mut Criterion) {
    let mut group = c.benchmark_group("counting_filter/contains_hit");
    common::bench_contains_hit(&mut group, make);
    group.finish();
}

fn contains_miss(c: &mut Criterion) {
    let mut group = c.benchmark_group("counting_filter/contains_miss");
    common::bench_contains_miss(&mut group, make);
    group.finish();
}

fn remove_hit(c: &mut Criterion) {
    let mut group = c.benchmark_group("counting_filter/remove_hit");
    common::bench_remove_hit(&mut group, make);
    group.finish();
}

fn remove_miss(c: &mut Criterion) {
    let mut group = c.benchmark_group("counting_filter/remove_miss");
    common::bench_remove_miss(&mut group, make);
    group.finish();
}

criterion_group!(benches, insert, contains_hit, contains_miss, remove_hit, remove_miss);
criterion_main!(benches);
