//! Convenience re-exports for the most common items.
//!
//! Import everything at once with:
//!
//! ```rust
//! use blume::prelude::*;
//!
//! let mut filter = BloomFilter::new(1_000, 0.01).unwrap();
//! filter.insert("hello");
//! assert!(filter.contains("hello"));
//! ```

pub use crate::bloomable::Bloomable;
pub use crate::error::BloomError;
pub use crate::filters::{
    AtomicBloomFilter, AtomicCountingBloomFilter, BloomFilter, CountingBloomFilter, CuckooFilter,
    ScalableBloomFilter,
};
pub use crate::traits::{ConcurrentFilter, Filter, MutableFilter, RemovableFilter};
