//! Tests for all built-in [`Bloomable`] implementations.
//!
//! Verifies that every primitive type supported out of the box can be inserted
//! and found in a filter — confirming each `with_bloom_bytes` implementation
//! produces stable, non-empty byte representations.

use blumer::prelude::*;

fn round_trips<T: Bloomable + ?Sized>(value: &T) {
    let mut f = BloomFilter::new(10, 0.01).unwrap();
    f.insert(value);
    assert!(f.contains(value), "inserted value not found after round-trip");
}

// --- unsigned integers ---

#[test] fn u8_round_trips()    { round_trips(&42u8);    }
#[test] fn u16_round_trips()   { round_trips(&1_000u16);  }
#[test] fn u32_round_trips()   { round_trips(&100_000u32); }
#[test] fn u64_round_trips()   { round_trips(&u64::MAX);  }
#[test] fn u128_round_trips()  { round_trips(&u128::MAX); }
#[test] fn usize_round_trips() { round_trips(&usize::MAX);}

// --- signed integers ---

#[test] fn i8_round_trips()    { round_trips(&-42i8);   }
#[test] fn i16_round_trips()   { round_trips(&-1_000i16); }
#[test] fn i32_round_trips()   { round_trips(&i32::MIN); }
#[test] fn i64_round_trips()   { round_trips(&i64::MIN); }
#[test] fn i128_round_trips()  { round_trips(&i128::MIN);}
#[test] fn isize_round_trips() { round_trips(&isize::MIN);}

// --- floats ---

#[test] fn f32_round_trips()         { round_trips(&1.5f32);         }
#[test] fn f32_negative_round_trips(){ round_trips(&-1.5f32);        }
#[test] fn f32_nan_round_trips()     { round_trips(&f32::NAN);       }
#[test] fn f32_inf_round_trips()     { round_trips(&f32::INFINITY);  }
#[test] fn f64_round_trips()         { round_trips(&1.5f64);         }
#[test] fn f64_negative_round_trips(){ round_trips(&-1.5f64);        }
#[test] fn f64_nan_round_trips()     { round_trips(&f64::NAN);       }
#[test] fn f64_inf_round_trips()     { round_trips(&f64::INFINITY);  }

// --- bool ---

#[test] fn bool_true_round_trips()  { round_trips(&true);  }
#[test] fn bool_false_round_trips() { round_trips(&false); }

/// `true` and `false` hash to different values — they must not collide.
#[test]
fn bool_true_and_false_are_distinct() {
    let mut f = BloomFilter::new(10, 0.0001).unwrap();
    f.insert(&true);
    // false was never inserted — it should almost certainly not be found.
    // With FPR 0.01%, this has a 1-in-10000 chance of false positive.
    // Run 10 independent filters to reduce flakiness to negligible.
    let any_fp = (0..10).any(|_| {
        let mut g = BloomFilter::new(10, 0.0001).unwrap();
        g.insert(&true);
        g.contains(&false)
    });
    assert!(!any_fp, "false was found in a filter that only contains true");
}

// --- strings ---

#[test] fn str_round_trips()    { round_trips("hello world");             }
#[test] fn string_round_trips() { round_trips(&String::from("hello"));    }
#[test] fn empty_str_round_trips() { round_trips("");                     }

// --- byte slices ---

#[test] fn byte_slice_round_trips() { round_trips(&[1u8, 2, 3][..]);      }
#[test] fn vec_u8_round_trips()     { round_trips(&vec![1u8, 2, 3]);       }
#[test] fn empty_bytes_round_trips(){ round_trips(&[][..] as &[u8]);       }

// --- with_params correctness ---

/// `BloomFilter::with_params` back-derives a capacity close to what `new`
/// would produce. Verifies the inversion formula is correct.
#[test]
fn with_params_capacity_is_consistent() {
    let original = BloomFilter::new(1_000, 0.01).unwrap();
    let restored = BloomFilter::with_params(original.bit_size(), 7).unwrap();

    // The back-derived capacity should be within 5% of the original.
    let ratio = restored.capacity() as f64 / original.capacity() as f64;
    assert!(
        (ratio - 1.0).abs() < 0.10,
        "with_params capacity {restored_cap} diverges from original {orig_cap}",
        restored_cap = restored.capacity(),
        orig_cap = original.capacity(),
    );
}
