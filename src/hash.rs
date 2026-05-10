use std::hash::{BuildHasher, Hasher};
use std::sync::LazyLock;

use ahash::RandomState;

use crate::Bloomable;

// Seeds derived from the SHA-256 initial hash values (fractional parts of the
// square roots of the first 8 primes). These are well-known high-entropy
// constants with good distribution properties.
static H1: LazyLock<RandomState> = LazyLock::new(|| {
    RandomState::with_seeds(
        0x6a09e667f3bcc908,
        0xbb67ae8584caa73b,
        0x3c6ef372fe94f82b,
        0xa54ff53a5f1d36f1,
    )
});

static H2: LazyLock<RandomState> = LazyLock::new(|| {
    RandomState::with_seeds(
        0x510e527fade682d1,
        0x9b05688c2b3e6c1f,
        0x1f83d9abfb41bd6b,
        0x5be0cd19137e2179,
    )
});

/// Returns two independent 64-bit hashes for use with double hashing.
/// `h2` is forced odd to prevent hash cycles regardless of filter size.
pub(crate) fn hash_pair<T: Bloomable + ?Sized>(item: &T) -> (u64, u64) {
    item.with_bloom_bytes(|bytes| {
        let mut h1 = H1.build_hasher();
        let mut h2 = H2.build_hasher();
        h1.write(bytes);
        h2.write(bytes);
        (h1.finish(), h2.finish() | 1)
    })
}

/// Derives `k` bit positions from two hashes using double hashing.
///
/// Uses the multiplication-based range reduction `(h * m) >> 64` instead of
/// `h % m` to avoid integer division (~20–40 cycles) at the cost of a single
/// 64×64→128 multiply and shift (~3 cycles).
#[inline]
pub(crate) fn bit_positions(k: usize, m: usize, h1: u64, h2: u64) -> impl Iterator<Item = usize> {
    (0..k).map(move |i| {
        let h = h1.wrapping_add((i as u64).wrapping_mul(h2));
        ((h as u128 * m as u128) >> 64) as usize
    })
}
