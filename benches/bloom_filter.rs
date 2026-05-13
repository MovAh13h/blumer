mod common;

use blumer::BloomFilter;
use criterion::{Criterion, criterion_group, criterion_main};

fn make(n: usize) -> BloomFilter {
    BloomFilter::new(n, common::FPR).unwrap()
}

fn insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("bloom_filter/insert");
    common::bench_insert(&mut group, make);
    group.finish();
}

fn contains_hit(c: &mut Criterion) {
    let mut group = c.benchmark_group("bloom_filter/contains_hit");
    common::bench_contains_hit(&mut group, make);
    group.finish();
}

fn contains_miss(c: &mut Criterion) {
    let mut group = c.benchmark_group("bloom_filter/contains_miss");
    common::bench_contains_miss(&mut group, make);
    group.finish();
}

criterion_group!(benches, insert, contains_hit, contains_miss);
criterion_main!(benches);
