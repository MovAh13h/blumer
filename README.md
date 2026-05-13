<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/MovAh13h/blumer/master/assets/blumer-lockup-dark.svg">
    <img src="https://raw.githubusercontent.com/MovAh13h/blumer/master/assets/blumer-lockup.svg" alt="blumer" width="340">
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
blumer = "0.4"
```

## Quick start

```rust
use blumer::prelude::*;

let mut filter = BloomFilter::new(1_000, 0.01).unwrap();

filter.insert("alice");
filter.insert("bob");

assert!(filter.contains("alice")); // definitely present
assert!(filter.contains("bob"));   // definitely present
assert!(!filter.contains("eve"));  // definitely absent (barring false positive)
```

## Supported types

Any type that implements `Bloomable` can be inserted. The following types are
supported out of the box:

| Type | Byte representation |
|------|---------------------|
| `u8`–`u128`, `usize` | Little-endian |
| `i8`–`i128`, `isize` | Little-endian |
| `f32`, `f64` | IEEE 754 bit pattern, little-endian |
| `bool` | Single byte: `0` or `1` |
| `&str`, `String` | UTF-8 bytes |
| `&[u8]`, `Vec<u8>` | Raw bytes |

## Custom types

Implement `Bloomable` to use your own types. The callback pattern passes
`&[u8]` to the hashing logic with zero allocation:

```rust
use blumer::Bloomable;

struct UserId(u64);

impl Bloomable for UserId {
    fn with_bloom_bytes<R>(&self, f: impl FnOnce(&[u8]) -> R) -> R {
        f(&self.0.to_le_bytes())
    }
}
```

For composite types, pack all fields into a fixed-size stack buffer:

```rust
use blumer::Bloomable;

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

## How it works — the math

A bloom filter is defined by three parameters:

| Symbol | Name | Role |
|--------|------|------|
| `n` | capacity | number of items the filter is designed to hold |
| `p` | false positive rate | target probability of a false positive at capacity |
| `m` | bit count | total bits in the internal array |
| `k` | hash count | number of hash positions set per item |

**Computing m and k from n and p**

```
m = -(n · ln p) / ln(2)²     ← total bits needed
k =  (m / n)   · ln(2)       ← optimal number of hash functions
```

**Why these formulas?**

After inserting `n` items with `k` hash functions into an `m`-bit array, any
given bit is set with probability:

```
P(bit is set) = 1 - (1 - 1/m)^(k·n)  ≈  1 - e^(-k·n/m)
```

A false positive occurs when all `k` positions of an uninserted item happen to
be set by other items. Assuming independence:

```
p = (1 - e^(-k·n/m))^k
```

Minimising this expression over `m` (for fixed `n` and `p`) and then solving
for the optimal `k` yields the formulas above. The `ln(2)²` in `m` and `ln(2)`
in `k` are the normalisation constants that fall out of that minimisation. At
the optimum, exactly half the bits in the array are set.

**Practical values**

| FPR target | Bits per item (`m/n`) | Hash functions (`k`) |
|------------|-----------------------|----------------------|
| 1%         | ~9.6                  | 7                    |
| 0.1%       | ~14.4                 | 10                   |
| 0.01%      | ~19.2                 | 14                   |

Each halving of the FPR adds ~1.4 bits per item (`ln 2` extra bits).

## False positive rate

The FPR you pass to `new` is a **target at the stated capacity** — it holds
when insertions equal `capacity`. Inserting more items causes the actual rate
to rise above the target; inserting fewer keeps it below.

| `fpr` | Meaning |
|-------|---------|
| `0.01` | 1 in 100 non-inserted items falsely reported present |
| `0.001` | 1 in 1 000 |
| `0.0001` | 1 in 10 000 |

Use `Filter::estimated_fpr` at runtime to observe the current rate as the
filter fills.

## Memory usage

`BloomFilter` stores bits packed into a `u64` array (`ceil(m / 64)` words).
`CountingBloomFilter` stores one `u8` counter per slot (`m` bytes),
costing 8× the memory of a standard filter.

| Items (`n`) | FPR | Standard | Counting |
|-------------|-----|----------|----------|
| 1 000       | 1%  | ~1.2 KB  | ~9.4 KB  |
| 100 000     | 1%  | ~117 KB  | ~938 KB  |
| 1 000 000   | 1%  | ~1.2 MB  | ~9.4 MB  |

For comparison, 1 000 000 items in a `HashSet<u64>` uses ~40 MB — ~30× more
than the standard filter.

## Filters

| Type | Thread-safe | Deletion | Growable | Memory | Use when |
|------|-------------|----------|----------|--------|----------|
| `BloomFilter` | No | No | No | 1× | fixed capacity, single-threaded |
| `CountingBloomFilter` | No | Yes | No | 8× | deletion, high accuracy |
| `CuckooFilter` | No | Yes | No | ~1× | deletion, memory-efficient |
| `AtomicBloomFilter` | Yes | No | No | 1× | concurrent insert + lookup |
| `AtomicCountingBloomFilter` | Yes | Yes | No | 8× | concurrent insert + deletion |
| `ScalableBloomFilter` | No | No | Yes | 1×+ | unknown or unbounded item count |

## Constructors

| Constructor | Use case |
|-------------|----------|
| `BloomFilter::new(n, p)` | Normal use — optimal `m` and `k` computed automatically |
| `BloomFilter::with_params(bits, hash_fns)` | Expert use — match the exact geometry of an existing filter (e.g. deserialising from storage) |

Same constructors exist on `CountingBloomFilter`, `AtomicBloomFilter`, and `AtomicCountingBloomFilter`.

## Concurrent use

`AtomicBloomFilter` allows any number of threads to insert and query
simultaneously without external locking:

```rust
use blumer::prelude::*;
use std::sync::Arc;

let filter = Arc::new(AtomicBloomFilter::new(1_000, 0.01).unwrap());

let f1 = Arc::clone(&filter);
let f2 = Arc::clone(&filter);

let t1 = std::thread::spawn(move || f1.insert("alice"));
let t2 = std::thread::spawn(move || f2.insert("bob"));

t1.join().unwrap();
t2.join().unwrap();

assert!(filter.contains("alice"));
assert!(filter.contains("bob"));
```

Inserts use `fetch_or(Release)` and lookups use `load(Acquire)`, ensuring
every insert that completes before a `contains` call is visible to it.

## Prelude

Import everything at once:

```rust
use blumer::prelude::*;
// BloomFilter, CountingBloomFilter, AtomicBloomFilter, AtomicCountingBloomFilter,
// CuckooFilter, ScalableBloomFilter, Filter, MutableFilter, RemovableFilter,
// ConcurrentFilter, Bloomable, and BloomError are all in scope.
```

## Feature flags

| Flag | Default | Description |
|------|---------|-------------|
| `serde` | off | `Serialize` / `Deserialize` on all filter types |

```toml
blumer = { version = "0.4", features = ["serde"] }
```

## Design notes

- **Hashing:** each item is hashed twice with [AHash] using fixed high-entropy
  seeds. `k` bit positions are derived from those two values using the
  [Kirsch–Mitzenmacher] double-hashing formula — no further hash calls needed.
- **Range reduction:** `(h × m) >> 64` instead of `h % m` — replaces a
  ~30-cycle integer division with a ~3-cycle multiply-and-shift.
- **Bit packing:** `BloomFilter` packs bits into `u64` words (64 bits/word);
  `CountingBloomFilter` stores one `u8` counter per slot; `AtomicBloomFilter`
  packs bits into `AtomicU64` words with the same layout as `BloomFilter`;
  `AtomicCountingBloomFilter` stores one `AtomicU8` counter per slot.

[AHash]: https://github.com/tkaitchuck/aHash
[Kirsch–Mitzenmacher]: https://www.eecs.harvard.edu/~michaelm/postscripts/tr-02-05.pdf
