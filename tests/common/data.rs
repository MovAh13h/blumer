use blume::Bloomable;
use proptest::prelude::*;

/// A composite user identifier used to verify that user-defined [`Bloomable`]
/// implementations work correctly end-to-end with bloom filters.
///
/// `UserId` combines a 32-bit namespace (e.g. tenant or shard ID) with a
/// 64-bit item ID. It exercises the fixed-width stack-allocated buffer pattern:
/// serialize both fields into a `[u8; 12]` array and pass a reference to the
/// callback — zero heap allocation, no intermediate `Vec<u8>`.
#[derive(Debug)]
pub struct UserId {
    pub namespace: u32,
    pub id: u64,
}

impl Bloomable for UserId {
    fn with_bloom_bytes<R>(&self, f: impl FnOnce(&[u8]) -> R) -> R {
        // Fixed-width layout: [namespace: 4 bytes LE][id: 8 bytes LE]
        // Little-endian keeps the layout consistent across platforms.
        let mut buf = [0u8; 12];
        buf[..4].copy_from_slice(&self.namespace.to_le_bytes());
        buf[4..].copy_from_slice(&self.id.to_le_bytes());
        f(&buf)
    }
}

/// Proptest strategy that generates arbitrary [`UserId`] values by combining
/// independent uniform distributions over `u32` and `u64`.
prop_compose! {
    pub fn arb_user_id()(namespace in any::<u32>(), id in any::<u64>()) -> UserId {
        UserId { namespace, id }
    }
}
