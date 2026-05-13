//! # blume
//!
//! A high-performance, bit-optimized bloom filter library for Rust.
//!
//! A bloom filter is a space-efficient probabilistic data structure that tests
//! whether an element is a member of a set. It can return false positives (saying
//! an item is present when it is not), but never false negatives (it will never
//! say an item is absent when it has been inserted). This makes bloom filters
//! ideal for use cases where a small probability of false positives is acceptable
//! in exchange for significant memory savings.
//!
//! ## Quick start
//!
//! ```rust
//! use blume::prelude::*;
//!
//! // Create a filter for 1 000 items with a 1% false positive rate.
//! let mut filter = BloomFilter::new(1_000, 0.01).unwrap();
//!
//! filter.insert("alice");
//! filter.insert("bob");
//!
//! assert!(filter.contains("alice"));
//! assert!(filter.contains("bob"));
//! assert!(!filter.contains("eve")); // very likely false
//! ```
//!
//! ## Inserting different types
//!
//! Any type that implements [`Bloomable`] can be inserted. All common types are
//! supported out of the box:
//!
//! ```rust
//! use blume::prelude::*;
//!
//! let mut filter = BloomFilter::new(1_000, 0.01).unwrap();
//!
//! filter.insert("a string slice");
//! filter.insert(&String::from("an owned string"));
//! filter.insert(&42u64);
//! filter.insert(&[1u8, 2, 3][..]);
//! ```
//!
//! ## Custom types
//!
//! Implement [`Bloomable`] to use your own types:
//!
//! ```rust
//! use blume::prelude::*;
//!
//! struct UserId(u64);
//!
//! impl Bloomable for UserId {
//!     fn with_bloom_bytes<R>(&self, f: impl FnOnce(&[u8]) -> R) -> R {
//!         f(&self.0.to_le_bytes())
//!     }
//! }
//!
//! let mut filter = BloomFilter::new(1_000, 0.01).unwrap();
//! filter.insert(&UserId(42));
//! assert!(filter.contains(&UserId(42)));
//! ```
//!
//! ## False positive rate (FPR)
//!
//! The false positive rate is the probability that [`Filter::contains`] returns
//! `true` for an item that was **never inserted**. It is expressed as a `f64`
//! in the range `(0, 1)`:
//!
//! | `fpr` value | Meaning |
//! |-------------|---------|
//! | `0.01`      | 1 in 100 non-inserted items will be falsely reported as present |
//! | `0.001`     | 1 in 1 000 |
//! | `0.0001`    | 1 in 10 000 |
//!
//! The FPR you pass to [`BloomFilter::new`] is a **target** at the stated
//! capacity — it holds precisely when the number of insertions equals
//! `capacity`. Inserting more items than `capacity` causes the actual FPR to
//! rise above the target. Inserting fewer keeps it below.
//!
//! Lower FPR targets require more memory: halving the FPR adds roughly 1.4
//! bits per item. Use [`Filter::estimated_fpr`] at runtime to observe the
//! actual rate as the filter fills.
//!
//! ## Feature flags
//!
//! | Flag | Description |
//! |------|-------------|
//! | `serde` | Enables `serde::Serialize` and `serde::Deserialize` on [`BloomFilter`] |
//!
//! Enable serde support in `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! blume = { version = "*", features = ["serde"] }
//! ```

pub mod bloomable;
pub mod error;
pub mod filters;
pub mod prelude;
pub mod traits;
pub(crate) mod hash;
pub(crate) mod math;

pub use bloomable::Bloomable;
pub use error::BloomError;
pub use filters::{AtomicBloomFilter, BloomFilter, CountingBloomFilter};
pub use traits::{ConcurrentFilter, Filter, MutableFilter, RemovableFilter};
