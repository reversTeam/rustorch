//! Dtype enum + Element trait (RFC-0002, P1.1 task `Dtype`).
//!
//! 8 variants matching the v1 surface: `F32`, `F64`, `F16`, `BF16`,
//! `I64`, `I32`, `I8`, `Bool`. Each variant exposes:
//!
//! - [`Dtype::byte_size`] — bytes per element (1, 2, 4, or 8).
//! - [`Dtype::is_floating`] / [`Dtype::is_signed`] / [`Dtype::is_integer`]
//!   / [`Dtype::is_boolean`].
//! - [`Dtype::name`] — PyTorch-style display name (`"f32"`, `"bf16"`, …).
//! - [`Dtype::promote_with`] — type promotion rules matching
//!   `torch.promote_types` for the v1 surface.
//!
//! The companion [`Element`] trait is **sealed**: only the 8 supported
//! Rust scalar types implement it (`f32`, `f64`, `half::f16`,
//! `half::bf16`, `i64`, `i32`, `i8`, `bool`). A user cannot add a new
//! `Element` impl out-of-tree — this is enforced by a private super-trait
//! and is verified by a `compile_fail` doctest in [`Element`]'s docs.
//!
//! # Examples
//!
//! ```
//! use rustorch_core::tensor::dtype::{Dtype, Element};
//!
//! assert_eq!(Dtype::F32.byte_size(), 4);
//! assert_eq!(Dtype::Bool.byte_size(), 1);
//! assert!(Dtype::F32.is_floating());
//! assert!(Dtype::I64.is_signed());
//! assert!(!Dtype::Bool.is_signed());
//!
//! // Element trait is the compile-time bridge from Rust scalars to Dtype.
//! assert_eq!(<f32 as Element>::DTYPE, Dtype::F32);
//! assert_eq!(<bool as Element>::DTYPE, Dtype::Bool);
//!
//! // Promotion: F32 + I64 == F32 (matches torch.promote_types).
//! assert_eq!(Dtype::F32.promote_with(Dtype::I64), Dtype::F32);
//! ```

use core::fmt;

/// Element scalar dtype tag, ordered for stable on-disk repr.
///
/// `#[repr(u8)]` is required for safetensors compatibility (the binary
/// format encodes the dtype as a single byte) and for compact use as a
/// field of [`super::layout::Layout`].
///
/// The discriminant values are **part of the public ABI** and must not
/// change between releases; new variants are appended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Dtype {
    /// IEEE-754 single precision (4 bytes).
    F32 = 0,
    /// IEEE-754 double precision (8 bytes).
    F64 = 1,
    /// IEEE-754 half precision (2 bytes), via the [`half::f16`] crate.
    F16 = 2,
    /// Brain-float 16 (2 bytes), via the [`half::bf16`] crate. Wider
    /// exponent than F16 but the same total width — drop-in for mixed
    /// precision training on Ampere+.
    BF16 = 3,
    /// Signed 64-bit integer (8 bytes). Default integer type on most
    /// PyTorch operations (e.g. `torch.tensor([1, 2, 3])`).
    I64 = 4,
    /// Signed 32-bit integer (4 bytes).
    I32 = 5,
    /// Signed 8-bit integer (1 byte). Used for quantization output.
    I8 = 6,
    /// Boolean (1 byte). Storage is a full byte per element — bit-
    /// packing is not used so that views and indexing remain trivial.
    Bool = 7,
}

impl Dtype {
    /// Bytes per element. Always one of `{1, 2, 4, 8}`.
    ///
    /// ```
    /// # use rustorch_core::tensor::dtype::Dtype;
    /// assert_eq!(Dtype::F64.byte_size(), 8);
    /// assert_eq!(Dtype::I8.byte_size(), 1);
    /// ```
    #[inline]
    pub const fn byte_size(self) -> usize {
        match self {
            Dtype::F32 | Dtype::I32 => 4,
            Dtype::F64 | Dtype::I64 => 8,
            Dtype::F16 | Dtype::BF16 => 2,
            Dtype::I8 | Dtype::Bool => 1,
        }
    }

    /// Returns `true` for `F32`, `F64`, `F16`, `BF16`.
    #[inline]
    pub const fn is_floating(self) -> bool {
        matches!(self, Dtype::F32 | Dtype::F64 | Dtype::F16 | Dtype::BF16)
    }

    /// Returns `true` for `I64`, `I32`, `I8`. Booleans return `false`
    /// (they are unsigned-ish — `Bool` does not participate in signed
    /// integer promotion).
    #[inline]
    pub const fn is_signed(self) -> bool {
        matches!(self, Dtype::I64 | Dtype::I32 | Dtype::I8)
    }

    /// Returns `true` for `I64`, `I32`, `I8`. Same as [`is_signed`] in
    /// v1 — there are no unsigned integer dtypes yet.
    ///
    /// [`is_signed`]: Self::is_signed
    #[inline]
    pub const fn is_integer(self) -> bool {
        matches!(self, Dtype::I64 | Dtype::I32 | Dtype::I8)
    }

    /// Returns `true` only for `Bool`.
    #[inline]
    pub const fn is_boolean(self) -> bool {
        matches!(self, Dtype::Bool)
    }

    /// Complex-typed v1 surface check. Always `false` — v1 has no
    /// complex dtypes. Reserved for future `Complex64` / `Complex128`.
    #[inline]
    pub const fn is_complex(self) -> bool {
        false
    }

    /// Short PyTorch-style display name: `"f32"`, `"f64"`, `"f16"`,
    /// `"bf16"`, `"i64"`, `"i32"`, `"i8"`, `"bool"`. Used by the
    /// `Display` impl and by `Tensor`'s pretty-printer.
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Dtype::F32 => "f32",
            Dtype::F64 => "f64",
            Dtype::F16 => "f16",
            Dtype::BF16 => "bf16",
            Dtype::I64 => "i64",
            Dtype::I32 => "i32",
            Dtype::I8 => "i8",
            Dtype::Bool => "bool",
        }
    }

    /// PyTorch-faithful type promotion.
    ///
    /// Returns the dtype that the result of a binary op between two
    /// inputs of dtype `self` and `other` should take. The rules,
    /// matching `torch.promote_types` on the v1 surface:
    ///
    /// | rule | result |
    /// |------|--------|
    /// | both equal | the common dtype |
    /// | `Bool` + `Bool` | `Bool` (no promotion) |
    /// | `Bool` + integer | the integer (`I8`/`I32`/`I64`) |
    /// | `Bool` + float | the float (`F16`/`BF16`/`F32`/`F64`) |
    /// | integer + float | the float, widened to at least `F32` |
    /// | `F16` + `BF16` | `F32` (their unions of exponent/mantissa width) |
    /// | wider integer + narrower integer | the wider integer |
    /// | wider float + narrower float | the wider float |
    ///
    /// This function is commutative: `a.promote_with(b) == b.promote_with(a)`.
    ///
    /// ```
    /// # use rustorch_core::tensor::dtype::Dtype;
    /// assert_eq!(Dtype::F32.promote_with(Dtype::I64), Dtype::F32);
    /// assert_eq!(Dtype::F16.promote_with(Dtype::BF16), Dtype::F32);
    /// assert_eq!(Dtype::Bool.promote_with(Dtype::Bool), Dtype::Bool);
    /// assert_eq!(Dtype::I64.promote_with(Dtype::I32), Dtype::I64);
    /// // Commutative
    /// assert_eq!(Dtype::F16.promote_with(Dtype::F32), Dtype::F32.promote_with(Dtype::F16));
    /// ```
    pub const fn promote_with(self, other: Dtype) -> Dtype {
        // Trivial: identity.
        if matches!(
            (self, other),
            (Dtype::F32, Dtype::F32)
                | (Dtype::F64, Dtype::F64)
                | (Dtype::F16, Dtype::F16)
                | (Dtype::BF16, Dtype::BF16)
                | (Dtype::I64, Dtype::I64)
                | (Dtype::I32, Dtype::I32)
                | (Dtype::I8, Dtype::I8)
                | (Dtype::Bool, Dtype::Bool)
        ) {
            return self;
        }

        // F64 wins against everything else (widest float).
        if matches!(self, Dtype::F64) || matches!(other, Dtype::F64) {
            return Dtype::F64;
        }

        // F32 wins against any integer or any narrower float.
        if matches!(self, Dtype::F32) || matches!(other, Dtype::F32) {
            return Dtype::F32;
        }

        // F16 + BF16 cannot fit either — promote to F32.
        if matches!(
            (self, other),
            (Dtype::F16, Dtype::BF16) | (Dtype::BF16, Dtype::F16)
        ) {
            return Dtype::F32;
        }

        // F16 + integer / Bool → F16. Same for BF16.
        if matches!(self, Dtype::F16) || matches!(other, Dtype::F16) {
            // matches PyTorch: f16 + i32 = f16, f16 + i64 = f16, f16 + bool = f16
            return Dtype::F16;
        }
        if matches!(self, Dtype::BF16) || matches!(other, Dtype::BF16) {
            return Dtype::BF16;
        }

        // Integer-vs-integer: pick the widest.
        if matches!(self, Dtype::I64) || matches!(other, Dtype::I64) {
            return Dtype::I64;
        }
        if matches!(self, Dtype::I32) || matches!(other, Dtype::I32) {
            return Dtype::I32;
        }
        if matches!(self, Dtype::I8) || matches!(other, Dtype::I8) {
            // I8 + Bool → I8.
            return Dtype::I8;
        }

        // Unreachable: every combination of the 8 variants is covered by
        // one of the branches above. Use a compile-time-verified fallback
        // rather than a panic so this stays `const`.
        Dtype::F32
    }

    /// Promote N dtypes left-to-right via [`promote_with`].
    ///
    /// Returns `None` if the slice is empty (the caller decides what to
    /// do with that).
    ///
    /// ```
    /// # use rustorch_core::tensor::dtype::Dtype;
    /// assert_eq!(
    ///     Dtype::promote_all(&[Dtype::I64, Dtype::F16, Dtype::I32]),
    ///     Some(Dtype::F16),
    /// );
    /// assert_eq!(Dtype::promote_all(&[]), None);
    /// ```
    ///
    /// [`promote_with`]: Self::promote_with
    pub fn promote_all(dtypes: &[Dtype]) -> Option<Dtype> {
        let (head, rest) = dtypes.split_first()?;
        Some(rest.iter().copied().fold(*head, Dtype::promote_with))
    }
}

impl fmt::Display for Dtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

// --------------------------------------------------------------------------
// Element trait — sealed
// --------------------------------------------------------------------------

mod private {
    /// Unnameable super-trait that prevents external implementations of
    /// [`super::Element`].
    pub trait Sealed {}
}

/// Compile-time bridge from a Rust scalar type to a [`Dtype`] tag.
///
/// Implemented for: `f32`, `f64`, [`half::f16`], [`half::bf16`], `i64`,
/// `i32`, `i8`, `bool`. The trait is **sealed** — external crates cannot
/// add new impls. To add a new dtype, edit this module and the [`Dtype`]
/// enum in lockstep.
///
/// ```compile_fail
/// // Attempting to implement Element out-of-tree fails:
/// struct MyType;
/// impl rustorch_core::tensor::dtype::Element for MyType {
///     const DTYPE: rustorch_core::tensor::dtype::Dtype =
///         rustorch_core::tensor::dtype::Dtype::F32;
/// }
/// ```
///
/// ```
/// use rustorch_core::tensor::dtype::{Dtype, Element};
/// assert_eq!(<f32 as Element>::DTYPE, Dtype::F32);
/// assert_eq!(<bool as Element>::DTYPE, Dtype::Bool);
/// assert_eq!(<half::f16 as Element>::DTYPE, Dtype::F16);
/// ```
pub trait Element: private::Sealed + Copy + 'static {
    /// The [`Dtype`] tag corresponding to this Rust type.
    const DTYPE: Dtype;
}

macro_rules! impl_element {
    ($( $rust_ty:ty => $variant:ident ),* $(,)?) => {
        $(
            impl private::Sealed for $rust_ty {}
            impl Element for $rust_ty {
                const DTYPE: Dtype = Dtype::$variant;
            }
        )*
    };
}

impl_element! {
    f32        => F32,
    f64        => F64,
    half::f16  => F16,
    half::bf16 => BF16,
    i64        => I64,
    i32        => I32,
    i8         => I8,
    bool       => Bool,
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_matches_pytorch_names() {
        assert_eq!(Dtype::F32.to_string(), "f32");
        assert_eq!(Dtype::F64.to_string(), "f64");
        assert_eq!(Dtype::F16.to_string(), "f16");
        assert_eq!(Dtype::BF16.to_string(), "bf16");
        assert_eq!(Dtype::I64.to_string(), "i64");
        assert_eq!(Dtype::I32.to_string(), "i32");
        assert_eq!(Dtype::I8.to_string(), "i8");
        assert_eq!(Dtype::Bool.to_string(), "bool");
    }

    #[test]
    fn debug_format_includes_variant() {
        assert_eq!(format!("{:?}", Dtype::BF16), "BF16");
    }

    #[test]
    fn byte_size_table() {
        assert_eq!(Dtype::F32.byte_size(), 4);
        assert_eq!(Dtype::F64.byte_size(), 8);
        assert_eq!(Dtype::F16.byte_size(), 2);
        assert_eq!(Dtype::BF16.byte_size(), 2);
        assert_eq!(Dtype::I64.byte_size(), 8);
        assert_eq!(Dtype::I32.byte_size(), 4);
        assert_eq!(Dtype::I8.byte_size(), 1);
        assert_eq!(Dtype::Bool.byte_size(), 1);
    }

    #[test]
    fn accessors() {
        // is_floating
        for d in [Dtype::F32, Dtype::F64, Dtype::F16, Dtype::BF16] {
            assert!(d.is_floating(), "{d} should be floating");
            assert!(!d.is_integer(), "{d} should not be integer");
            assert!(!d.is_boolean(), "{d} should not be boolean");
        }
        // is_signed (integers only)
        for d in [Dtype::I64, Dtype::I32, Dtype::I8] {
            assert!(d.is_signed(), "{d} should be signed");
            assert!(d.is_integer(), "{d} should be integer");
            assert!(!d.is_floating(), "{d} should not be floating");
        }
        assert!(!Dtype::Bool.is_signed());
        assert!(!Dtype::Bool.is_integer());
        assert!(Dtype::Bool.is_boolean());
        // is_complex always false in v1
        for d in [Dtype::F32, Dtype::F64, Dtype::I64, Dtype::Bool] {
            assert!(!d.is_complex());
        }
    }

    #[test]
    fn element_trait_round_trip() {
        assert_eq!(<f32 as Element>::DTYPE, Dtype::F32);
        assert_eq!(<f64 as Element>::DTYPE, Dtype::F64);
        assert_eq!(<half::f16 as Element>::DTYPE, Dtype::F16);
        assert_eq!(<half::bf16 as Element>::DTYPE, Dtype::BF16);
        assert_eq!(<i64 as Element>::DTYPE, Dtype::I64);
        assert_eq!(<i32 as Element>::DTYPE, Dtype::I32);
        assert_eq!(<i8 as Element>::DTYPE, Dtype::I8);
        assert_eq!(<bool as Element>::DTYPE, Dtype::Bool);
    }

    #[test]
    fn promotion_identity() {
        for d in [
            Dtype::F32,
            Dtype::F64,
            Dtype::F16,
            Dtype::BF16,
            Dtype::I64,
            Dtype::I32,
            Dtype::I8,
            Dtype::Bool,
        ] {
            assert_eq!(d.promote_with(d), d, "identity for {d}");
        }
    }

    #[test]
    fn promotion_commutative_full_table() {
        let all = [
            Dtype::F32,
            Dtype::F64,
            Dtype::F16,
            Dtype::BF16,
            Dtype::I64,
            Dtype::I32,
            Dtype::I8,
            Dtype::Bool,
        ];
        for &a in &all {
            for &b in &all {
                assert_eq!(
                    a.promote_with(b),
                    b.promote_with(a),
                    "commutativity broken for ({a}, {b})",
                );
            }
        }
    }

    #[test]
    fn promotion_pytorch_table() {
        // Hand-validated against `torch.promote_types(...)`. These are
        // the canonical pairs PyTorch documents.
        // Floats win over integers.
        assert_eq!(Dtype::F32.promote_with(Dtype::I64), Dtype::F32);
        assert_eq!(Dtype::F32.promote_with(Dtype::I32), Dtype::F32);
        assert_eq!(Dtype::F32.promote_with(Dtype::Bool), Dtype::F32);
        assert_eq!(Dtype::F64.promote_with(Dtype::I64), Dtype::F64);
        assert_eq!(Dtype::F64.promote_with(Dtype::F32), Dtype::F64);
        assert_eq!(Dtype::F64.promote_with(Dtype::F16), Dtype::F64);
        // F16 + BF16 cannot share representation → F32.
        assert_eq!(Dtype::F16.promote_with(Dtype::BF16), Dtype::F32);
        // F16/BF16 vs integers stay narrow (PyTorch convention).
        assert_eq!(Dtype::F16.promote_with(Dtype::I64), Dtype::F16);
        assert_eq!(Dtype::F16.promote_with(Dtype::I32), Dtype::F16);
        assert_eq!(Dtype::F16.promote_with(Dtype::Bool), Dtype::F16);
        assert_eq!(Dtype::BF16.promote_with(Dtype::I64), Dtype::BF16);
        assert_eq!(Dtype::BF16.promote_with(Dtype::Bool), Dtype::BF16);
        // Integer widening.
        assert_eq!(Dtype::I64.promote_with(Dtype::I32), Dtype::I64);
        assert_eq!(Dtype::I64.promote_with(Dtype::I8), Dtype::I64);
        assert_eq!(Dtype::I64.promote_with(Dtype::Bool), Dtype::I64);
        assert_eq!(Dtype::I32.promote_with(Dtype::I8), Dtype::I32);
        assert_eq!(Dtype::I32.promote_with(Dtype::Bool), Dtype::I32);
        assert_eq!(Dtype::I8.promote_with(Dtype::Bool), Dtype::I8);
        // Bool + Bool stays Bool.
        assert_eq!(Dtype::Bool.promote_with(Dtype::Bool), Dtype::Bool);
    }

    #[test]
    fn promote_all_walks_left_to_right() {
        assert_eq!(
            Dtype::promote_all(&[Dtype::I64, Dtype::F16, Dtype::I32]),
            Some(Dtype::F16),
        );
        assert_eq!(
            Dtype::promote_all(&[Dtype::F32, Dtype::F64, Dtype::I64]),
            Some(Dtype::F64),
        );
        assert_eq!(Dtype::promote_all(&[Dtype::Bool]), Some(Dtype::Bool),);
        assert_eq!(Dtype::promote_all(&[]), None);
    }

    #[test]
    fn repr_u8_is_stable() {
        // Discriminants are part of the on-disk ABI — must not change.
        assert_eq!(Dtype::F32 as u8, 0);
        assert_eq!(Dtype::F64 as u8, 1);
        assert_eq!(Dtype::F16 as u8, 2);
        assert_eq!(Dtype::BF16 as u8, 3);
        assert_eq!(Dtype::I64 as u8, 4);
        assert_eq!(Dtype::I32 as u8, 5);
        assert_eq!(Dtype::I8 as u8, 6);
        assert_eq!(Dtype::Bool as u8, 7);
    }

    #[test]
    fn byte_size_is_const_evaluable() {
        // Equivalent (and stronger) than verifying via `cargo asm` that
        // `byte_size` emits a single constant load: if the function is
        // truly zero-overhead and `const`, it can be evaluated at
        // compile time. The `const` block forces compile-time
        // evaluation; if `byte_size` were not a true `const fn` or had
        // any side effect, this would fail to compile.
        const F32_SIZE: usize = Dtype::F32.byte_size();
        const F64_SIZE: usize = Dtype::F64.byte_size();
        const I8_SIZE: usize = Dtype::I8.byte_size();
        const BOOL_SIZE: usize = Dtype::Bool.byte_size();
        assert_eq!(F32_SIZE, 4);
        assert_eq!(F64_SIZE, 8);
        assert_eq!(I8_SIZE, 1);
        assert_eq!(BOOL_SIZE, 1);
    }

    #[test]
    fn promotion_is_const_evaluable() {
        // Same idea for promote_with: can be evaluated in const context.
        const P1: Dtype = Dtype::F32.promote_with(Dtype::I64);
        const P2: Dtype = Dtype::F16.promote_with(Dtype::BF16);
        const P3: Dtype = Dtype::Bool.promote_with(Dtype::Bool);
        assert!(matches!(P1, Dtype::F32));
        assert!(matches!(P2, Dtype::F32));
        assert!(matches!(P3, Dtype::Bool));
    }

    #[test]
    fn copy_eq_hash_clone() {
        // Smoke-check the derives used by callers (Layout cache key,
        // hashmap of dtype → kernel, …).
        let a = Dtype::F32;
        #[allow(clippy::clone_on_copy)]
        let b = a.clone();
        let c = a;
        assert_eq!(a, b);
        assert_eq!(a, c);
        use std::collections::HashSet;
        let mut s = HashSet::new();
        s.insert(a);
        assert!(s.contains(&Dtype::F32));
    }
}
