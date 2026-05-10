//! The [`Bloomable`] trait for types that can be inserted into a bloom filter.

/// A type that can be represented as a byte slice for hashing in a bloom filter.
///
/// Rather than returning a `Vec<u8>` (which allocates) or `&[u8]` (which cannot
/// represent stack-allocated data like integer byte arrays), `Bloomable` uses a
/// callback pattern: the implementor temporarily exposes its bytes to a closure,
/// keeping everything on the stack with zero allocation.
///
/// # Provided implementations
///
/// The following types implement `Bloomable` out of the box:
///
/// | Type | Byte representation |
/// |------|---------------------|
/// | `str`, `String` | UTF-8 bytes |
/// | `[u8]`, `Vec<u8>` | raw bytes as-is |
/// | `u8`–`u128`, `usize` | little-endian bytes |
/// | `i8`–`i128`, `isize` | little-endian bytes |
///
/// # Implementing for custom types
///
/// Call `f` with a byte slice that uniquely and stably identifies your value.
/// Two values that should be treated as equal by the filter **must** produce
/// identical byte slices.
///
/// ```rust
/// use blume::Bloomable;
///
/// struct UserId(u64);
///
/// impl Bloomable for UserId {
///     fn with_bloom_bytes<R>(&self, f: impl FnOnce(&[u8]) -> R) -> R {
///         f(&self.0.to_le_bytes())
///     }
/// }
/// ```
///
/// For types that already hold their data as a contiguous byte slice, simply
/// forward the reference:
///
/// ```rust
/// use blume::Bloomable;
///
/// struct IpAddr([u8; 4]);
///
/// impl Bloomable for IpAddr {
///     fn with_bloom_bytes<R>(&self, f: impl FnOnce(&[u8]) -> R) -> R {
///         f(&self.0)
///     }
/// }
/// ```
///
/// # Warning: stability
///
/// The byte representation must be **stable across time and processes**. Avoid
/// deriving it from `std::hash::Hash`, which is not guaranteed to be consistent
/// across Rust versions or process restarts.
pub trait Bloomable {
    /// Passes the byte representation of `self` to `f` and returns the result.
    ///
    /// The bytes passed to `f` must uniquely and stably identify the value.
    /// This method must not allocate for types whose data is already contiguous
    /// in memory (e.g. `str`, `[u8]`).
    fn with_bloom_bytes<R>(&self, f: impl FnOnce(&[u8]) -> R) -> R;
}

impl Bloomable for str {
    fn with_bloom_bytes<R>(&self, f: impl FnOnce(&[u8]) -> R) -> R {
        f(self.as_bytes())
    }
}

impl Bloomable for String {
    fn with_bloom_bytes<R>(&self, f: impl FnOnce(&[u8]) -> R) -> R {
        f(self.as_bytes())
    }
}

impl Bloomable for [u8] {
    fn with_bloom_bytes<R>(&self, f: impl FnOnce(&[u8]) -> R) -> R {
        f(self)
    }
}

impl Bloomable for Vec<u8> {
    fn with_bloom_bytes<R>(&self, f: impl FnOnce(&[u8]) -> R) -> R {
        f(self)
    }
}

macro_rules! impl_bloomable_int {
    ($($t:ty),*) => {
        $(
            impl Bloomable for $t {
                fn with_bloom_bytes<R>(&self, f: impl FnOnce(&[u8]) -> R) -> R {
                    f(&self.to_le_bytes())
                }
            }
        )*
    };
}

impl_bloomable_int!(u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize);
