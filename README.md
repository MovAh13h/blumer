<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/blume-lockup-dark.svg">
    <img src="assets/blume-lockup.svg" alt="blume" width="340">
  </picture>

  <p>A high-performance, bit-optimized bloom filter library for Rust.</p>
</div>

A bloom filter is a space-efficient probabilistic data structure for membership
testing. It can return **false positives** (reporting an item as present when it
was never inserted), but never **false negatives** (an inserted item is always
found). This makes bloom filters ideal when a small probability of false
positives is acceptable in exchange for significant memory savings.

## Installation

```toml
[dependencies]
blume = "0.1"
```

## Quick start

```rust
use blume::{BloomFilter, Filter, MutableFilter};

// Create a filter sized for 1 000 items at a 1% false positive rate.
let mut filter = BloomFilter::new(1_000, 0.01).unwrap();

filter.insert("alice");
filter.insert("bob");

assert!(filter.contains("alice")); // definitely present
assert!(filter.contains("bob"));   // definitely present
assert!(!filter.contains("eve"));  // definitely absent (barring false positive)
```

## Supported types

Any type that implements [`Bloomable`] can be inserted. The following types are
supported out of the box:

```rust
use blume::{BloomFilter, Filter, MutableFilter};

let mut filter = BloomFilter::new(1_000, 0.01).unwrap();

filter.insert("a string slice");
filter.insert(&String::from("an owned string"));
filter.insert(&42u64);
filter.insert(&[1u8, 2, 3][..]);
```

Full list of built-in implementations:

| Type | Notes |
|------|-------|
| `u8`, `u16`, `u32`, `u64`, `u128`, `usize` | Little-endian byte representation |
| `i8`, `i16`, `i32`, `i64`, `i128`, `isize` | Little-endian byte representation |
| `f32`, `f64` | Little-endian IEEE 754 bits |
| `bool` | Single byte (`0` or `1`) |
| `&str`, `String` | UTF-8 bytes |
| `&[u8]`, `Vec<u8>` | Raw bytes |

## Custom types

Implement `Bloomable` to use your own types. The callback pattern passes a
`&[u8]` reference to the hashing logic with zero allocation — no intermediate
`Vec<u8>` is created.

```rust
use blume::{Bloomable, BloomFilter, Filter, MutableFilter};

struct UserId(u64);

impl Bloomable for UserId {
    fn with_bloom_bytes<R>(&self, f: impl FnOnce(&[u8]) -> R) -> R {
        f(&self.0.to_le_bytes())
    }
}

let mut filter = BloomFilter::new(1_000, 0.01).unwrap();
filter.insert(&UserId(42));
assert!(filter.contains(&UserId(42)));
```

For composite types, pack all fields into a fixed-size stack buffer:

```rust
use blume::Bloomable;

struct UserId { namespace: u32, id: u64 }

impl Bloomable for UserId {
    fn with_bloom_bytes<R>(&self, f: impl FnOnce(&[u8]) -> R) -> R {
        let mut buf = [0u8; 12];
        buf[..4].copy_from_slice(&self.namespace.to_le_bytes());
        buf[4..].copy_from_slice(&self.id.to_le_bytes());
        f(&buf)
    }
}
```

## False positive rate (FPR)

The FPR is the probability that `contains` returns `true` for an item that was
**never inserted**. Pass it as a `f64` in the open interval `(0, 1)`:

| `fpr` value | Meaning |
|-------------|---------|
| `0.01`      | 1 in 100 non-inserted items falsely reported as present |
| `0.001`     | 1 in 1 000 |
| `0.0001`    | 1 in 10 000 |

The FPR you pass to `BloomFilter::new` is a **target at the stated capacity**.
It holds precisely when the number of insertions equals `capacity`. Inserting
more items causes the actual FPR to rise; inserting fewer keeps it below the
target.

Use `Filter::estimated_fpr` to observe the actual rate at runtime as the filter
fills.

## Memory usage

The filter stores `m` bits in a packed `u64` array, where:

```
m = -(n · ln(p)) / ln(2)²
```

Quick reference for common configurations:

| Items (`n`) | FPR (`p`) | Bit array   | RAM      |
|-------------|-----------|-------------|----------|
| 1 000       | 1%        | ~9 600 b    | ~1.2 KB  |
| 100 000     | 1%        | ~960 000 b  | ~117 KB  |
| 1 000 000   | 1%        | ~9.6 Mb     | ~1.2 MB  |
| 1 000 000   | 0.1%      | ~14.4 Mb    | ~1.7 MB  |

For comparison, storing the same 1 000 000 items in a `HashSet<u64>` typically
uses ~40 MB — roughly 30× more.

## Constructors

| Constructor | Use case |
|-------------|----------|
| `BloomFilter::new(capacity, fpr)` | Normal use — computes optimal `m` and `k` automatically |
| `BloomFilter::with_params(bits, hash_fns)` | Expert use — explicit parameters, e.g. when rehydrating from storage |

## Feature flags

| Flag | Description |
|------|-------------|
| `serde` | Enables `serde::Serialize` / `serde::Deserialize` on `BloomFilter` |

```toml
[dependencies]
blume = { version = "0.1", features = ["serde"] }
```

## Design notes

- **Hashing:** each item is hashed twice with [AHash] using fixed high-entropy
  seeds. From those two values, `k` bit positions are derived using the
  [Kirsch–Mitzenmacher] double-hashing optimization — no further hash calls
  needed.
- **Range reduction:** bit positions use the multiplication trick
  `(h × m) >> 64` instead of `h % m`, replacing a ~30-cycle integer division
  with a ~3-cycle multiply-and-shift.
- **Bit packing:** bits are stored in `u64` words (64 bits per word), keeping
  memory aligned and allowing single-instruction word loads on every lookup.

[AHash]: https://github.com/tkaitchuck/aHash
[Kirsch–Mitzenmacher]: https://www.eecs.harvard.edu/~michaelm/postscripts/tr-02-05.pdf
