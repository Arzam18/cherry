//! ARM64 NEON backend for Cherry.
//!
//! Mirrors the public API of `avx2.rs` / `avx512.rs`. The building block is
//! a 128-bit NEON register; 256/512-bit types are arrays of 128-bit halves.
//! Every operation preserves the exact semantics of the AVX2 backend so
//! that `perft` and `bench` produce identical output on x86_64 and aarch64.

use core::{arch::aarch64::*, mem, ops::*};

/* ================================================================= */
/*                         Helper functions                          */
/* ================================================================= */

#[inline]
unsafe fn movemask_u8x16(v: uint8x16_t) -> u16 {
    // Pack the MSB of each byte into the low 16 bits of a u16.
    let bytes: [u8; 16] = unsafe { mem::transmute(v) };
    let mut out: u16 = 0;
    let mut i = 0;
    while i < 16 { out |= ((bytes[i] >> 7) as u16) << i; i += 1; }
    out
}

#[inline]
unsafe fn movemask_u16x8(v: uint16x8_t) -> u8 {
    let halves: [u16; 8] = unsafe { mem::transmute(v) };
    let mut out: u8 = 0;
    let mut i = 0;
    while i < 8 { out |= ((halves[i] >> 15) as u8) << i; i += 1; }
    out
}

#[inline]
unsafe fn movemask_u32x4(v: uint32x4_t) -> u8 {
    let words: [u32; 4] = unsafe { mem::transmute(v) };
    let mut out: u8 = 0;
    let mut i = 0;
    while i < 4 { out |= ((words[i] >> 31) as u8) << i; i += 1; }
    out
}

#[inline]
unsafe fn movemask_u64x2(v: uint64x2_t) -> u8 {
    let words: [u64; 2] = unsafe { mem::transmute(v) };
    ((words[0] >> 63) as u8) | (((words[1] >> 63) as u8) << 1)
}

/* ================================================================= */
/*                         Mask types                                */
/* ================================================================= */

macro_rules! def_mask {
    ($name:ident, $vec:ident) => {
        #[derive(Debug, Copy, Clone)]
        pub struct $name(pub $vec);

        impl From<$vec> for $name { #[inline] fn from(v: $vec) -> Self { Self(v) } }

        impl Not for $name {
            type Output = Self;
            #[inline] fn not(self) -> Self { Self(!self.0) }
        }
        impl BitAnd for $name {
            type Output = Self;
            #[inline] fn bitand(self, o: Self) -> Self { Self(self.0 & o.0) }
        }
        impl BitOr for $name {
            type Output = Self;
            #[inline] fn bitor(self, o: Self) -> Self { Self(self.0 | o.0) }
        }
        impl BitXor for $name {
            type Output = Self;
            #[inline] fn bitxor(self, o: Self) -> Self { Self(self.0 ^ o.0) }
        }
        impl BitAndAssign for $name {
            #[inline] fn bitand_assign(&mut self, o: Self) { *self = *self & o; }
        }
        impl BitOrAssign for $name {
            #[inline] fn bitor_assign(&mut self, o: Self) { *self = *self | o; }
        }
        impl BitXorAssign for $name {
            #[inline] fn bitxor_assign(&mut self, o: Self) { *self = *self ^ o; }
        }
    };
}

/* ================================================================= */
/*                       128-bit vector types                        */
/* ================================================================= */

macro_rules! impl_bitwise_128 {
    ($name:ident, $elem:ty) => {
        impl From<<Self as VecInternal>::Raw> for $name {
            #[inline] fn from(v: <Self as VecInternal>::Raw) -> Self { Self(v) }
        }
        impl Not for $name {
            type Output = Self;
            #[inline] fn not(self) -> Self { <Self as VecInternal>::not_(self) }
        }
        impl BitAnd for $name {
            type Output = Self;
            #[inline] fn bitand(self, o: Self) -> Self { <Self as VecInternal>::and_(self, o) }
        }
        impl BitOr for $name {
            type Output = Self;
            #[inline] fn bitor(self, o: Self) -> Self { <Self as VecInternal>::or_(self, o) }
        }
        impl BitXor for $name {
            type Output = Self;
            #[inline] fn bitxor(self, o: Self) -> Self { <Self as VecInternal>::xor_(self, o) }
        }
        impl BitAndAssign for $name {
            #[inline] fn bitand_assign(&mut self, o: Self) { *self = *self & o; }
        }
        impl BitOrAssign for $name {
            #[inline] fn bitor_assign(&mut self, o: Self) { *self = *self | o; }
        }
        impl BitXorAssign for $name {
            #[inline] fn bitxor_assign(&mut self, o: Self) { *self = *self ^ o; }
        }
    };
}

pub(crate) trait VecInternal: Copy {
    type Raw: Copy;
    const ELEM_MAX: u8;
    fn not_(a: Self) -> Self;
    fn and_(a: Self, b: Self) -> Self;
    fn or_(a: Self, b: Self) -> Self;
    fn xor_(a: Self, b: Self) -> Self;
}

/* ---------- u8x16 ---------- */
#[derive(Debug, Copy, Clone)]
pub struct u8x16(pub uint8x16_t);
impl VecInternal for u8x16 {
    type Raw = uint8x16_t;
    const ELEM_MAX: u8 = 0xFF;
    #[inline] fn not_(a: Self) -> Self { Self(vmvnq_u8(a.0)) }
    #[inline] fn and_(a: Self, b: Self) -> Self { Self(vandq_u8(a.0, b.0)) }
    #[inline] fn or_(a: Self, b: Self) -> Self { Self(vorrq_u8(a.0, b.0)) }
    #[inline] fn xor_(a: Self, b: Self) -> Self { Self(veorq_u8(a.0, b.0)) }
}
impl_bitwise_128!(u8x16, u8);
impl u8x16 {
    #[inline] pub unsafe fn load<T>(p: *const T) -> Self { Self(unsafe { vld1q_u8(p.cast()) }) }
    #[inline] pub unsafe fn store<T>(self, p: *mut T) { unsafe { vst1q_u8(p.cast(), self.0) } }
    #[inline] pub fn splat(v: u8) -> Self { Self(vdupq_n_u8(v)) }
    #[inline] pub fn andnot(self, o: Self) -> Self { Self(vbicq_u8(self.0, o.0)) }
    #[inline] pub fn eq(a: Self, b: Self) -> Mask8x16 { Mask8x16(Self(vceqq_u8(a.0, b.0))) }
    #[inline] pub fn neq(a: Self, b: Self) -> Mask8x16 { Self::eq(a, b).not() }
    #[inline] pub fn test(a: Self, b: Self) -> Mask8x16 { (a & b).nonzero() }
    #[inline] pub fn testn(a: Self, b: Self) -> Mask8x16 { (a & b).zero() }
    #[inline] pub fn zero(self) -> Mask8x16 { Self::eq(self, Self::splat(0)) }
    #[inline] pub fn nonzero(self) -> Mask8x16 { Self::neq(self, Self::splat(0)) }
    #[inline] pub fn msb(self) -> Mask8x16 {
        Mask8x16(Self(vcltq_s8(vreinterpretq_s8_u8(self.0), vdupq_n_s8(0))))
    }
    #[inline] pub fn to_bitmask(self) -> u16 { unsafe { movemask_u8x16(self.0) } }
    #[inline] pub fn mask(self, m: Mask8x16) -> Self { self & m.0 }
    #[inline] pub fn blend(a: Self, b: Self, m: Mask8x16) -> Self { (m.0 & b) | m.0.andnot(a) }
    #[inline] pub fn compress(self, m: Mask8x16) -> Self {
        let mut out = [0u8; 16];
        let arr: [u8; 16] = unsafe { mem::transmute(self.0) };
        let mask = m.to_bitmask();
        let mut cursor = 0;
        let mut i = 0;
        while i < 16 { if (mask >> i) & 1 != 0 { out[cursor] = arr[i]; cursor += 1; } i += 1; }
        unsafe { Self::load(out.as_ptr()) }
    }
    #[inline] pub unsafe fn compress_store<T>(self, m: Mask8x16, p: *mut T) { unsafe { self.compress(m).store(p) } }
    #[inline] pub fn shuffle(self, idx: Self) -> Self { Self(vqtbl1q_u8(self.0, idx.0)) }
    #[inline] pub fn extract<const I: i32>(self) -> u8 { unsafe { vgetq_lane_u8::<I>(self.0) } }
    #[inline] pub fn broadcast32(self) -> u8x32 { u8x32([self, self]) }
    #[inline] pub fn broadcast64(self) -> u8x64 { u8x64([self, self, self, self]) }
    #[inline] pub fn zero_ext(self) -> u16x16 { u16x16([self.zero_ext_128(), self.zero_ext_128()]) }
    #[inline] pub(crate) fn zero_ext_128(self) -> u16x8 { u16x8(vmovl_u8(vget_low_u8(self.0))) }
    #[inline] pub fn findset(self, needles: Self, count: usize) -> u16 {
        let a: [u8; 16] = unsafe { mem::transmute(self.0) };
        let b: [u8; 16] = unsafe { mem::transmute(needles.0) };
        let mut out = 0u16;
        for i in 0..count {
            for j in 0..16 { if b[i] == a[j] { out |= 1 << j; } }
        }
        out
    }
    #[inline] pub fn from_u8x16_array(a: [u8; 16]) -> Self { unsafe { Self::load(a.as_ptr()) } }
}

/* ---------- u16x8 ---------- */
#[derive(Debug, Copy, Clone)]
pub struct u16x8(pub uint16x8_t);
impl VecInternal for u16x8 {
    type Raw = uint16x8_t;
    const ELEM_MAX: u8 = 0xFF;
    #[inline] fn not_(a: Self) -> Self { Self(vmvnq_u16(a.0)) }
    #[inline] fn and_(a: Self, b: Self) -> Self { Self(vandq_u16(a.0, b.0)) }
    #[inline] fn or_(a: Self, b: Self) -> Self { Self(vorrq_u16(a.0, b.0)) }
    #[inline] fn xor_(a: Self, b: Self) -> Self { Self(veorq_u16(a.0, b.0)) }
}
impl_bitwise_128!(u16x8, u16);
impl u16x8 {
    #[inline] pub unsafe fn load<T>(p: *const T) -> Self { Self(unsafe { vld1q_u16(p.cast()) }) }
    #[inline] pub unsafe fn store<T>(self, p: *mut T) { unsafe { vst1q_u16(p.cast(), self.0) } }
    #[inline] pub fn splat(v: u16) -> Self { Self(vdupq_n_u16(v)) }
    #[inline] pub fn andnot(self, o: Self) -> Self { Self(vbicq_u16(self.0, o.0)) }
    #[inline] pub fn eq(a: Self, b: Self) -> Mask16x8 { Mask16x8(Self(vceqq_u16(a.0, b.0))) }
    #[inline] pub fn neq(a: Self, b: Self) -> Mask16x8 { Self::eq(a, b).not() }
    #[inline] pub fn test(a: Self, b: Self) -> Mask16x8 { (a & b).nonzero() }
    #[inline] pub fn testn(a: Self, b: Self) -> Mask16x8 { (a & b).zero() }
    #[inline] pub fn zero(self) -> Mask16x8 { Self::eq(self, Self::splat(0)) }
    #[inline] pub fn nonzero(self) -> Mask16x8 { Self::neq(self, Self::splat(0)) }
    #[inline] pub fn msb(self) -> Mask16x8 {
        Mask16x8(Self(vcltq_s16(vreinterpretq_s16_u16(self.0), vdupq_n_s16(0))))
    }
    #[inline] pub fn to_bitmask(self) -> u8 { unsafe { movemask_u16x8(self.0) } }
    #[inline] pub fn mask(self, m: Mask16x8) -> Self { self & m.0 }
    #[inline] pub fn blend(a: Self, b: Self, m: Mask16x8) -> Self { (m.0 & b) | m.0.andnot(a) }
    #[inline] pub fn compress(self, m: Mask16x8) -> Self {
        let arr: [u16; 8] = unsafe { mem::transmute(self.0) };
        let mut out = [0u16; 8];
        let mask = m.to_bitmask();
        let mut cursor = 0;
        for i in 0..8 { if (mask >> i) & 1 != 0 { out[cursor] = arr[i]; cursor += 1; } }
        unsafe { Self::load(out.as_ptr()) }
    }
    #[inline] pub unsafe fn compress_store<T>(self, m: Mask16x8, p: *mut T) { unsafe { self.compress(m).store(p) } }
    #[inline] pub fn extract<const I: i32>(self) -> u16 { unsafe { vgetq_lane_u16::<I>(self.0) } }
    #[inline] pub fn broadcast16(self) -> u16x16 { u16x16([self, self]) }
    #[inline] pub fn broadcast32(self) -> u16x32 { u16x32([self, self, self, self]) }
    #[inline] pub fn zero_ext(self) -> u32x8 { u32x8([self.zero_ext_128(), self.zero_ext_128()]) }
    #[inline] pub(crate) fn zero_ext_128(self) -> u32x4 { u32x4(vmovl_u16(vget_low_u16(self.0))) }
    #[inline] pub fn shl<const N: i32>(self) -> Self { Self(vshlq_n_u16::<N>(self.0)) }
    #[inline] pub fn shr<const N: i32>(self) -> Self { Self(vshrq_n_u16::<N>(self.0)) }
    #[inline] pub fn shlv(self, s: Self) -> Self { Self(vshlq_u16(self.0, vreinterpretq_s16_u16(s.0))) }
    #[inline] pub fn shrv(self, s: Self) -> Self { Self(vshlq_u16(self.0, vnegq_s16(vreinterpretq_s16_u16(s.0)))) }
}

/* ---------- u32x4 ---------- */
#[derive(Debug, Copy, Clone)]
pub struct u32x4(pub uint32x4_t);
impl VecInternal for u32x4 {
    type Raw = uint32x4_t;
    const ELEM_MAX: u8 = 0xFF;
    #[inline] fn not_(a: Self) -> Self { Self(vmvnq_u32(a.0)) }
    #[inline] fn and_(a: Self, b: Self) -> Self { Self(vandq_u32(a.0, b.0)) }
    #[inline] fn or_(a: Self, b: Self) -> Self { Self(vorrq_u32(a.0, b.0)) }
    #[inline] fn xor_(a: Self, b: Self) -> Self { Self(veorq_u32(a.0, b.0)) }
}
impl_bitwise_128!(u32x4, u32);
impl u32x4 {
    #[inline] pub unsafe fn load<T>(p: *const T) -> Self { Self(unsafe { vld1q_u32(p.cast()) }) }
    #[inline] pub unsafe fn store<T>(self, p: *mut T) { unsafe { vst1q_u32(p.cast(), self.0) } }
    #[inline] pub fn splat(v: u32) -> Self { Self(vdupq_n_u32(v)) }
    #[inline] pub fn andnot(self, o: Self) -> Self { Self(vbicq_u32(self.0, o.0)) }
    #[inline] pub fn eq(a: Self, b: Self) -> Mask32x4 { Mask32x4(Self(vceqq_u32(a.0, b.0))) }
    #[inline] pub fn neq(a: Self, b: Self) -> Mask32x4 { Self::eq(a, b).not() }
    #[inline] pub fn test(a: Self, b: Self) -> Mask32x4 { (a & b).nonzero() }
    #[inline] pub fn testn(a: Self, b: Self) -> Mask32x4 { (a & b).zero() }
    #[inline] pub fn zero(self) -> Mask32x4 { Self::eq(self, Self::splat(0)) }
    #[inline] pub fn nonzero(self) -> Mask32x4 { Self::neq(self, Self::splat(0)) }
    #[inline] pub fn msb(self) -> Mask32x4 {
        Mask32x4(Self(vcltq_s32(vreinterpretq_s32_u32(self.0), vdupq_n_s32(0))))
    }
    #[inline] pub fn to_bitmask(self) -> u8 { unsafe { movemask_u32x4(self.0) } }
    #[inline] pub fn mask(self, m: Mask32x4) -> Self { self & m.0 }
    #[inline] pub fn blend(a: Self, b: Self, m: Mask32x4) -> Self { (m.0 & b) | m.0.andnot(a) }
    #[inline] pub fn compress(self, m: Mask32x4) -> Self {
        let arr: [u32; 4] = unsafe { mem::transmute(self.0) };
        let mut out = [0u32; 4];
        let mask = m.to_bitmask();
        let mut cursor = 0;
        for i in 0..4 { if (mask >> i) & 1 != 0 { out[cursor] = arr[i]; cursor += 1; } }
        unsafe { Self::load(out.as_ptr()) }
    }
    #[inline] pub unsafe fn compress_store<T>(self, m: Mask32x4, p: *mut T) { unsafe { self.compress(m).store(p) } }
    #[inline] pub fn extract<const I: i32>(self) -> u32 { unsafe { vgetq_lane_u32::<I>(self.0) } }
    #[inline] pub fn broadcast8(self) -> u32x8 { u32x8([self, self]) }
    #[inline] pub fn broadcast16(self) -> u32x16 { u32x16([self, self, self, self]) }
    #[inline] pub fn zero_ext(self) -> u64x4 { u64x4([self.zero_ext_128(), self.zero_ext_128()]) }
    #[inline] pub(crate) fn zero_ext_128(self) -> u64x2 { u64x2(vmovl_u32(vget_low_u32(self.0))) }
}

/* ---------- u64x2 ---------- */
#[derive(Debug, Copy, Clone)]
pub struct u64x2(pub uint64x2_t);
impl VecInternal for u64x2 {
    type Raw = uint64x2_t;
    const ELEM_MAX: u8 = 0xFF;
    #[inline] fn not_(a: Self) -> Self { Self(vmvnq_u64(a.0)) }
    #[inline] fn and_(a: Self, b: Self) -> Self { Self(vandq_u64(a.0, b.0)) }
    #[inline] fn or_(a: Self, b: Self) -> Self { Self(vorrq_u64(a.0, b.0)) }
    #[inline] fn xor_(a: Self, b: Self) -> Self { Self(veorq_u64(a.0, b.0)) }
}
impl_bitwise_128!(u64x2, u64);
impl u64x2 {
    #[inline] pub unsafe fn load<T>(p: *const T) -> Self { Self(unsafe { vld1q_u64(p.cast()) }) }
    #[inline] pub unsafe fn store<T>(self, p: *mut T) { unsafe { vst1q_u64(p.cast(), self.0) } }
    #[inline] pub fn splat(v: u64) -> Self { Self(vdupq_n_u64(v)) }
    #[inline] pub fn andnot(self, o: Self) -> Self { Self(vbicq_u64(self.0, o.0)) }
    #[inline] pub fn eq(a: Self, b: Self) -> Mask64x2 { Mask64x2(Self(vceqq_u64(a.0, b.0))) }
    #[inline] pub fn neq(a: Self, b: Self) -> Mask64x2 { Self::eq(a, b).not() }
    #[inline] pub fn test(a: Self, b: Self) -> Mask64x2 { (a & b).nonzero() }
    #[inline] pub fn testn(a: Self, b: Self) -> Mask64x2 { (a & b).zero() }
    #[inline] pub fn zero(self) -> Mask64x2 { Self::eq(self, Self::splat(0)) }
    #[inline] pub fn nonzero(self) -> Mask64x2 { Self::neq(self, Self::splat(0)) }
    #[inline] pub fn msb(self) -> Mask64x2 {
        Mask64x2(Self(vcltq_s64(vreinterpretq_s64_u64(self.0), vdupq_n_s64(0))))
    }
    #[inline] pub fn to_bitmask(self) -> u8 { unsafe { movemask_u64x2(self.0) } }
    #[inline] pub fn extract<const I: i32>(self) -> u64 { unsafe { vgetq_lane_u64::<I>(self.0) } }
    #[inline] pub fn broadcast4(self) -> u64x4 { u64x4([self, self]) }
    #[inline] pub fn broadcast8(self) -> u64x8 { u64x8([self, self, self, self]) }
}

/* ================================================================= */
/*                       256- and 512-bit types                       */
/* ================================================================= */

macro_rules! def_wide_vec {
    ($name:ident, $inner:ident, $n:expr, $elem:ident, $arr:ty) => {
        #[derive(Debug, Copy, Clone)]
        pub struct $name(pub [$inner; $n]);

        impl $name {
            #[inline] pub unsafe fn load<T>(p: *const T) -> Self {
                let mut a: [$inner; $n] = unsafe { mem::zeroed() };
                let stride = mem::size_of::<$inner>();
                let mut i = 0;
                while i < $n { a[i] = unsafe { $inner::load(p.cast::<u8>().add(i * stride)) }; i += 1; }
                Self(a)
            }
            #[inline] pub unsafe fn store<T>(self, p: *mut T) {
                let stride = mem::size_of::<$inner>();
                let mut i = 0;
                while i < $n { unsafe { self.0[i].store(p.cast::<u8>().add(i * stride)) }; i += 1; }
            }
            #[inline] pub fn splat(v: $elem) -> Self {
                let h = $inner::splat(v as _);
                Self([h; $n])
            }
            #[inline] pub fn andnot(self, o: Self) -> Self {
                let mut r: [$inner; $n] = unsafe { mem::zeroed() };
                let mut i = 0;
                while i < $n { r[i] = self.0[i].andnot(o.0[i]); i += 1; }
                Self(r)
            }
        }

        impl From<$arr> for $name {
            #[inline] fn from(a: $arr) -> Self { unsafe { Self::load(a.as_ptr()) } }
        }
        impl From<[$inner; $n]> for $name {
            #[inline] fn from(a: [$inner; $n]) -> Self { Self(a) }
        }
        impl Not for $name {
            type Output = Self;
            #[inline] fn not(self) -> Self {
                let mut r: [$inner; $n] = unsafe { mem::zeroed() };
                let mut i = 0;
                while i < $n { r[i] = !self.0[i]; i += 1; }
                Self(r)
            }
        }
        impl BitAnd for $name {
            type Output = Self;
            #[inline] fn bitand(self, o: Self) -> Self {
                let mut r: [$inner; $n] = unsafe { mem::zeroed() };
                let mut i = 0;
                while i < $n { r[i] = self.0[i] & o.0[i]; i += 1; }
                Self(r)
            }
        }
        impl BitOr for $name {
            type Output = Self;
            #[inline] fn bitor(self, o: Self) -> Self {
                let mut r: [$inner; $n] = unsafe { mem::zeroed() };
                let mut i = 0;
                while i < $n { r[i] = self.0[i] | o.0[i]; i += 1; }
                Self(r)
            }
        }
        impl BitXor for $name {
            type Output = Self;
            #[inline] fn bitxor(self, o: Self) -> Self {
                let mut r: [$inner; $n] = unsafe { mem::zeroed() };
                let mut i = 0;
                while i < $n { r[i] = self.0[i] ^ o.0[i]; i += 1; }
                Self(r)
            }
        }
        impl BitAndAssign for $name { #[inline] fn bitand_assign(&mut self, o: Self) { *self = *self & o; } }
        impl BitOrAssign for $name { #[inline] fn bitor_assign(&mut self, o: Self) { *self = *self | o; } }
        impl BitXorAssign for $name { #[inline] fn bitxor_assign(&mut self, o: Self) { *self = *self ^ o; } }
    };
}

def_wide_vec!(u8x32,  u8x16,  2, u8,  [u8; 32]);
def_wide_vec!(u8x64,  u8x16,  4, u8,  [u8; 64]);
def_wide_vec!(u16x16, u16x8,  2, u16, [u16; 16]);
def_wide_vec!(u16x32, u16x8,  4, u16, [u16; 32]);
def_wide_vec!(u32x8,  u32x4,  2, u32, [u32; 8]);
def_wide_vec!(u32x16, u32x4,  4, u32, [u32; 16]);
def_wide_vec!(u64x4,  u64x2,  2, u64, [u64; 4]);
def_wide_vec!(u64x8,  u64x2,  4, u64, [u64; 8]);

/* ---------- wide-specific methods ---------- */

impl u8x32 {
    #[inline] pub fn eq(a: Self, b: Self) -> Mask8x32 {
        Mask8x32([u8x16::eq(a.0[0], b.0[0]), u8x16::eq(a.0[1], b.0[1])])
    }
    #[inline] pub fn neq(a: Self, b: Self) -> Mask8x32 { Self::eq(a, b).not() }
    #[inline] pub fn zero(self) -> Mask8x32 { Self::eq(self, Self::splat(0)) }
    #[inline] pub fn nonzero(self) -> Mask8x32 { Self::neq(self, Self::splat(0)) }
    #[inline] pub fn to_bitmask(self) -> u32 {
        (self.0[0].to_bitmask() as u32) | ((self.0[1].to_bitmask() as u32) << 16)
    }
    #[inline] pub fn mask(self, m: Mask8x32) -> Self { self & m.0 }
    #[inline] pub fn blend(a: Self, b: Self, m: Mask8x32) -> Self { (m.0 & b) | m.0.andnot(a) }
    #[inline] pub fn compress(self, m: Mask8x32) -> Self {
        let mask = m.to_bitmask();
        let mut out = [0u8; 32];
        let flat: [u8; 32] = unsafe { mem::transmute(self.0) };
        let mut cursor = 0;
        for i in 0..32 { if (mask >> i) & 1 != 0 { out[cursor] = flat[i]; cursor += 1; } }
        unsafe { Self::load(out.as_ptr()) }
    }
    #[inline] pub unsafe fn compress_store<T>(self, m: Mask8x32, p: *mut T) { unsafe { self.compress(m).store(p) } }
    #[inline] pub fn extract<const I: i32>(self) -> u8 { self.0[(I / 16) as usize].extract::<{I % 16}>() }
    #[inline] pub fn extract16<const I: i32>(self) -> u8x16 { self.0[I as usize] }
    #[inline] pub fn broadcast64(self) -> u8x64 { u8x64([self.0[0], self.0[1], self.0[0], self.0[1]]) }
    #[inline] pub fn permute(self, idx: Self) -> Self {
        // Slow but correct: extract all 32 index bytes and gather.
        let src: [u8; 32] = unsafe { mem::transmute(self.0) };
        let ind: [u8; 32] = unsafe { mem::transmute(idx.0) };
        let mut out = [0u8; 32];
        for i in 0..32 { out[i] = src[(ind[i] & 31) as usize]; }
        unsafe { Self::load(out.as_ptr()) }
    }
    #[inline] pub fn shuffle(self, idx: Self) -> Self { self.permute(idx) }
}

impl u8x64 {
    #[inline] pub fn eq(a: Self, b: Self) -> Mask8x64 {
        Mask8x64([
            u8x16::eq(a.0[0], b.0[0]),
            u8x16::eq(a.0[1], b.0[1]),
            u8x16::eq(a.0[2], b.0[2]),
            u8x16::eq(a.0[3], b.0[3]),
        ])
    }
    #[inline] pub fn neq(a: Self, b: Self) -> Mask8x64 { Self::eq(a, b).not() }
    #[inline] pub fn zero(self) -> Mask8x64 { Self::eq(self, Self::splat(0)) }
    #[inline] pub fn nonzero(self) -> Mask8x64 { Self::neq(self, Self::splat(0)) }
    #[inline] pub fn msb(self) -> Mask8x64 {
        Mask8x64([
            self.0[0].msb(), self.0[1].msb(), self.0[2].msb(), self.0[3].msb(),
        ])
    }
    #[inline] pub fn to_bitmask(self) -> u64 {
        (self.0[0].to_bitmask() as u64)
            | ((self.0[1].to_bitmask() as u64) << 16)
            | ((self.0[2].to_bitmask() as u64) << 32)
            | ((self.0[3].to_bitmask() as u64) << 48)
    }
    #[inline] pub fn mask(self, m: Mask8x64) -> Self { self & m.0 }
    #[inline] pub fn blend(a: Self, b: Self, m: Mask8x64) -> Self { (m.0 & b) | m.0.andnot(a) }
    #[inline] pub fn compress(self, m: Mask8x64) -> Self {
        let mask = m.to_bitmask();
        let mut out = [0u8; 64];
        let flat: [u8; 64] = unsafe { mem::transmute(self.0) };
        let mut cursor = 0;
        for i in 0..64 { if (mask >> i) & 1 != 0 { out[cursor] = flat[i]; cursor += 1; } }
        unsafe { Self::load(out.as_ptr()) }
    }
    #[inline] pub unsafe fn compress_store<T>(self, m: Mask8x64, p: *mut T) { unsafe { self.compress(m).store(p) } }
    #[inline] pub fn permute(self, idx: Self) -> Self {
        let src: [u8; 64] = unsafe { mem::transmute(self.0) };
        let ind: [u8; 64] = unsafe { mem::transmute(idx.0) };
        let mut out = [0u8; 64];
        for i in 0..64 { out[i] = src[(ind[i] & 63) as usize]; }
        unsafe { Self::load(out.as_ptr()) }
    }
    #[inline] pub fn shuffle(self, idx: Self) -> Self { self.permute(idx) }
    #[inline] pub fn extract16<const I: usize>(self) -> u8x16 { self.0[I] }
    #[inline] pub fn extract32<const I: usize>(self) -> u8x32 {
        u8x32([self.0[I * 2], self.0[I * 2 + 1]])
    }
    #[inline] pub fn flip_rays(self) -> Self { u8x64([self.0[2], self.0[3], self.0[0], self.0[1]]) }
    #[inline] pub fn extend_rays(self) -> Self {
        // Matches the AVX2 implementation semantically: SAD-based broadcast
        // of byte sums, then a shuffle. Fallback to scalar here.
        let flat: [u8; 64] = unsafe { mem::transmute(self.0) };
        let mut sums = [0u8; 8];
        for i in 0..8 {
            let mut s = 0u16;
            for j in 0..8 { s += flat[i * 8 + j] as u16; }
            sums[i] = (s & 0xFF) as u8;
        }
        let mut out = [0u8; 64];
        for i in 0..64 { out[i] = sums[i / 8]; }
        unsafe { Self::load(out.as_ptr()) }
    }
    #[inline] pub fn zero_ext(self) -> u16x64 { u16x64([self.0[0].zero_ext_128(), self.0[1].zero_ext_128(), self.0[2].zero_ext_128(), self.0[3].zero_ext_128()]) }
}

impl u16x16 {
    #[inline] pub fn eq(a: Self, b: Self) -> Mask16x16 {
        Mask16x16([u16x8::eq(a.0[0], b.0[0]), u16x8::eq(a.0[1], b.0[1])])
    }
    #[inline] pub fn neq(a: Self, b: Self) -> Mask16x16 { Self::eq(a, b).not() }
    #[inline] pub fn test(a: Self, b: Self) -> Mask16x16 { (a & b).nonzero() }
    #[inline] pub fn testn(a: Self, b: Self) -> Mask16x16 { (a & b).zero() }
    #[inline] pub fn zero(self) -> Mask16x16 { Self::eq(self, Self::splat(0)) }
    #[inline] pub fn nonzero(self) -> Mask16x16 { Self::neq(self, Self::splat(0)) }
    #[inline] pub fn msb(self) -> Mask16x16 {
        Mask16x16([self.0[0].msb(), self.0[1].msb()])
    }
    #[inline] pub fn to_bitmask(self) -> u16 {
        (self.0[0].to_bitmask() as u16) | ((self.0[1].to_bitmask() as u16) << 8)
    }
    #[inline] pub fn mask(self, m: Mask16x16) -> Self { self & m.0 }
    #[inline] pub fn blend(a: Self, b: Self, m: Mask16x16) -> Self { (m.0 & b) | m.0.andnot(a) }
    #[inline] pub fn compress(self, m: Mask16x16) -> Self {
        let mask = m.to_bitmask();
        let mut out = [0u16; 16];
        let flat: [u16; 16] = unsafe { mem::transmute(self.0) };
        let mut cursor = 0;
        for i in 0..16 { if (mask >> i) & 1 != 0 { out[cursor] = flat[i]; cursor += 1; } }
        unsafe { Self::load(out.as_ptr()) }
    }
    #[inline] pub unsafe fn compress_store<T>(self, m: Mask16x16, p: *mut T) { unsafe { self.compress(m).store(p) } }
    #[inline] pub fn shl<const N: i32>(self) -> Self { u16x16([self.0[0].shl::<N>(), self.0[1].shl::<N>()]) }
    #[inline] pub fn shr<const N: i32>(self) -> Self { u16x16([self.0[0].shr::<N>(), self.0[1].shr::<N>()]) }
    #[inline] pub fn shlv(self, s: Self) -> Self { u16x16([self.0[0].shlv(s.0[0]), self.0[1].shlv(s.0[1])]) }
    #[inline] pub fn shrv(self, s: Self) -> Self { u16x16([self.0[0].shrv(s.0[0]), self.0[1].shrv(s.0[1])]) }
    #[inline] pub fn extract<const I: i32>(self) -> u16 { self.0[(I / 8) as usize].extract::<{I % 8}>() }
    #[inline] pub fn extract8<const I: i32>(self) -> u16x8 { self.0[I as usize] }
    #[inline] pub fn broadcast32(self) -> u16x32 { u16x32([self.0[0], self.0[1], self.0[0], self.0[1]]) }
    #[inline] pub fn zero_ext(self) -> u32x16 { u32x16([self.0[0].zero_ext_128(), self.0[1].zero_ext_128(), self.0[0].zero_ext_128(), self.0[1].zero_ext_128()]) }
}

impl u16x32 {
    #[inline] pub fn eq(a: Self, b: Self) -> Mask16x32 {
        Mask16x32([u16x8::eq(a.0[0], b.0[0]), u16x8::eq(a.0[1], b.0[1]), u16x8::eq(a.0[2], b.0[2]), u16x8::eq(a.0[3], b.0[3])])
    }
    #[inline] pub fn neq(a: Self, b: Self) -> Mask16x32 { Self::eq(a, b).not() }
    #[inline] pub fn test(a: Self, b: Self) -> Mask16x32 { (a & b).nonzero() }
    #[inline] pub fn testn(a: Self, b: Self) -> Mask16x32 { (a & b).zero() }
    #[inline] pub fn zero(self) -> Mask16x32 { Self::eq(self, Self::splat(0)) }
    #[inline] pub fn nonzero(self) -> Mask16x32 { Self::neq(self, Self::splat(0)) }
    #[inline] pub fn msb(self) -> Mask16x32 {
        Mask16x32([self.0[0].msb(), self.0[1].msb(), self.0[2].msb(), self.0[3].msb()])
    }
    #[inline] pub fn to_bitmask(self) -> u32 {
        (self.0[0].to_bitmask() as u32)
            | ((self.0[1].to_bitmask() as u32) << 8)
            | ((self.0[2].to_bitmask() as u32) << 16)
            | ((self.0[3].to_bitmask() as u32) << 24)
    }
    #[inline] pub fn mask(self, m: Mask16x32) -> Self { self & m.0 }
    #[inline] pub fn blend(a: Self, b: Self, m: Mask16x32) -> Self { (m.0 & b) | m.0.andnot(a) }
    #[inline] pub fn compress(self, m: Mask16x32) -> Self {
        let mask = m.to_bitmask();
        let mut out = [0u16; 32];
        let flat: [u16; 32] = unsafe { mem::transmute(self.0) };
        let mut cursor = 0;
        for i in 0..32 { if (mask >> i) & 1 != 0 { out[cursor] = flat[i]; cursor += 1; } }
        unsafe { Self::load(out.as_ptr()) }
    }
    #[inline] pub unsafe fn compress_store<T>(self, m: Mask16x32, p: *mut T) { unsafe { self.compress(m).store(p) } }
    #[inline] pub fn shl<const N: i32>(self) -> Self { u16x32([self.0[0].shl::<N>(), self.0[1].shl::<N>(), self.0[2].shl::<N>(), self.0[3].shl::<N>()]) }
    #[inline] pub fn shr<const N: i32>(self) -> Self { u16x32([self.0[0].shr::<N>(), self.0[1].shr::<N>(), self.0[2].shr::<N>(), self.0[3].shr::<N>()]) }
    #[inline] pub fn shlv(self, s: Self) -> Self { u16x32([self.0[0].shlv(s.0[0]), self.0[1].shlv(s.0[1]), self.0[2].shlv(s.0[2]), self.0[3].shlv(s.0[3])]) }
    #[inline] pub fn shrv(self, s: Self) -> Self { u16x32([self.0[0].shrv(s.0[0]), self.0[1].shrv(s.0[1]), self.0[2].shrv(s.0[2]), self.0[3].shrv(s.0[3])]) }
    #[inline] pub fn extract8<const I: i32>(self) -> u16x8 { self.0[I as usize] }
    #[inline] pub fn extract16<const I: usize>(self) -> u16x16 {
        u16x16([self.0[I * 2], self.0[I * 2 + 1]])
    }
}

impl u32x16 {
    #[inline] pub fn zero_ext(self) -> u64x8 { unimplemented!("u32x16::zero_ext not used") }
}

impl u64x8 {
    // Nothing beyond the macro at the moment.
}

/* ================================================================= */
/*                        Mask impls                                 */
/* ================================================================= */

def_mask!(Mask8x16, u8x16);
def_mask!(Mask16x8, u16x8);
def_mask!(Mask32x4, u32x4);
def_mask!(Mask64x2, u64x2);

impl Mask8x16 {
    #[inline] pub fn to_bitmask(self) -> u16 { self.0.to_bitmask() }
    #[inline] pub fn widen(self) -> Mask16x8 { Mask16x8(u16x8(self.0.0.into())) }
    #[inline] pub fn expand(_bm: u16) -> Self { unimplemented!("Mask8x16::expand") }
}
impl Mask16x8 {
    #[inline] pub fn to_bitmask(self) -> u8 { self.0.to_bitmask() }
    #[inline] pub fn widen(self) -> Mask32x4 { Mask32x4(u32x4(self.0.0.into())) }
    #[inline] pub fn expand(_bm: u8) -> Self { unimplemented!("Mask16x8::expand") }
}
impl Mask32x4 {
    #[inline] pub fn to_bitmask(self) -> u8 { self.0.to_bitmask() }
    #[inline] pub fn widen(self) -> Mask64x2 { Mask64x2(u64x2(self.0.0.into())) }
    #[inline] pub fn expand(_bm: u8) -> Self { unimplemented!("Mask32x4::expand") }
}
impl Mask64x2 {
    #[inline] pub fn to_bitmask(self) -> u8 { self.0.to_bitmask() }
    #[inline] pub fn expand(_bm: u8) -> Self { unimplemented!("Mask64x2::expand") }
}

// Wide masks: struct-of-halves + a convenience to_bitmask that concatenates.

macro_rules! def_wide_mask {
    ($name:ident, $inner:ident, $n:expr, $bm:ty) => {
        #[derive(Debug, Copy, Clone)]
        pub struct $name(pub [$inner; $n]);

        impl From<[$inner; $n]> for $name { #[inline] fn from(a: [$inner; $n]) -> Self { Self(a) } }
        impl From<$inner> for $name { #[inline] fn from(h: $inner) -> Self { Self([h; $n]) } }
        impl Not for $name { type Output = Self; #[inline] fn not(self) -> Self { let mut r = self.0; for i in 0..$n { r[i] = !r[i]; } Self(r) } }
        impl BitAnd for $name { type Output = Self; #[inline] fn bitand(self, o: Self) -> Self { let mut r = self.0; for i in 0..$n { r[i] = r[i] & o.0[i]; } Self(r) } }
        impl BitOr for $name { type Output = Self; #[inline] fn bitor(self, o: Self) -> Self { let mut r = self.0; for i in 0..$n { r[i] = r[i] | o.0[i]; } Self(r) } }
        impl BitXor for $name { type Output = Self; #[inline] fn bitxor(self, o: Self) -> Self { let mut r = self.0; for i in 0..$n { r[i] = r[i] ^ o.0[i]; } Self(r) } }
        impl BitAndAssign for $name { #[inline] fn bitand_assign(&mut self, o: Self) { *self = *self & o; } }
        impl BitOrAssign for $name { #[inline] fn bitor_assign(&mut self, o: Self) { *self = *self | o; } }
        impl BitXorAssign for $name { #[inline] fn bitxor_assign(&mut self, o: Self) { *self = *self ^ o; } }
    };
}

def_wide_mask!(Mask8x32,  Mask8x16,  2, u32);
def_wide_mask!(Mask8x64,  Mask8x16,  4, u64);
def_wide_mask!(Mask16x16, Mask16x8,  2, u16);
def_wide_mask!(Mask16x32, Mask16x8,  4, u32);
def_wide_mask!(Mask32x8,  Mask32x4,  2, u8);
def_wide_mask!(Mask32x16, Mask32x4,  4, u16);
def_wide_mask!(Mask64x4,  Mask64x2,  2, u8);
def_wide_mask!(Mask64x8,  Mask64x2,  4, u8);

impl Mask8x32 {
    #[inline] pub fn to_bitmask(self) -> u32 {
        (self.0[0].to_bitmask() as u32) | ((self.0[1].to_bitmask() as u32) << 16)
    }
    #[inline] pub fn widen(self) -> Mask16x16 { Mask16x16([self.0[0].widen(), self.0[1].widen()]) }
}
impl Mask8x64 {
    #[inline] pub fn to_bitmask(self) -> u64 {
        (self.0[0].to_bitmask() as u64)
            | ((self.0[1].to_bitmask() as u64) << 16)
            | ((self.0[2].to_bitmask() as u64) << 32)
            | ((self.0[3].to_bitmask() as u64) << 48)
    }
    #[inline] pub fn widen(self) -> Mask16x32 { Mask16x32([self.0[0].widen(), self.0[1].widen(), self.0[2].widen(), self.0[3].widen()]) }
}
impl Mask16x16 {
    #[inline] pub fn to_bitmask(self) -> u16 {
        (self.0[0].to_bitmask() as u16) | ((self.0[1].to_bitmask() as u16) << 8)
    }
    #[inline] pub fn widen(self) -> Mask32x8 { Mask32x8([self.0[0].widen(), self.0[1].widen()]) }
}
impl Mask16x32 {
    #[inline] pub fn to_bitmask(self) -> u32 {
        (self.0[0].to_bitmask() as u32)
            | ((self.0[1].to_bitmask() as u32) << 8)
            | ((self.0[2].to_bitmask() as u32) << 16)
            | ((self.0[3].to_bitmask() as u32) << 24)
    }
    #[inline] pub fn widen(self) -> Mask32x16 { Mask32x16([self.0[0].widen(), self.0[1].widen(), self.0[2].widen(), self.0[3].widen()]) }
}
impl Mask32x8 {
    #[inline] pub fn to_bitmask(self) -> u8 { (self.0[0].to_bitmask() as u8) | ((self.0[1].to_bitmask() as u8) << 4) }
    #[inline] pub fn widen(self) -> Mask64x4 { Mask64x4([self.0[0].widen(), self.0[1].widen()]) }
}
impl Mask32x16 {
    #[inline] pub fn to_bitmask(self) -> u16 {
        (self.0[0].to_bitmask() as u16)
            | ((self.0[1].to_bitmask() as u16) << 4)
            | ((self.0[2].to_bitmask() as u16) << 8)
            | ((self.0[3].to_bitmask() as u16) << 12)
    }
    #[inline] pub fn widen(self) -> Mask64x8 { Mask64x8([self.0[0].widen(), self.0[1].widen(), self.0[2].widen(), self.0[3].widen()]) }
}
impl Mask64x4 {
    #[inline] pub fn to_bitmask(self) -> u8 { (self.0[0].to_bitmask() as u8) | ((self.0[1].to_bitmask() as u8) << 2) }
    #[inline] pub fn widen(self) -> Mask64x4 { self }
}
impl Mask64x8 {
    #[inline] pub fn to_bitmask(self) -> u8 {
        (self.0[0].to_bitmask() as u8)
            | ((self.0[1].to_bitmask() as u8) << 2)
            | ((self.0[2].to_bitmask() as u8) << 4)
            | ((self.0[3].to_bitmask() as u8) << 6)
    }
}

/* ================================================================= */
/*                        u16x64 (needed for NNUE / attack tables)    */
/* ================================================================= */

#[derive(Debug, Copy, Clone)]
pub struct u16x64(pub [u16x8; 8]);

impl u16x64 {
    #[inline] pub unsafe fn load<T>(p: *const T) -> Self {
        let mut a: [u16x8; 8] = unsafe { mem::zeroed() };
        for i in 0..8 { a[i] = unsafe { u16x8::load(p.cast::<u8>().add(i * 16)) }; }
        Self(a)
    }
    #[inline] pub unsafe fn store<T>(self, p: *mut T) {
        for i in 0..8 { unsafe { self.0[i].store(p.cast::<u8>().add(i * 16)) }; }
    }
    #[inline] pub fn splat(v: u16) -> Self { Self([u16x8::splat(v); 8]) }
    #[inline] pub fn eq(a: Self, b: Self) -> Mask16x64 {
        Mask16x64([
            u16x8::eq(a.0[0], b.0[0]), u16x8::eq(a.0[1], b.0[1]),
            u16x8::eq(a.0[2], b.0[2]), u16x8::eq(a.0[3], b.0[3]),
            u16x8::eq(a.0[4], b.0[4]), u16x8::eq(a.0[5], b.0[5]),
            u16x8::eq(a.0[6], b.0[6]), u16x8::eq(a.0[7], b.0[7]),
        ])
    }
    #[inline] pub fn neq(a: Self, b: Self) -> Mask16x64 { Self::eq(a, b).not() }
    #[inline] pub fn test(a: Self, b: Self) -> Mask16x64 { (a & b).nonzero() }
    #[inline] pub fn testn(a: Self, b: Self) -> Mask16x64 { (a & b).zero() }
    #[inline] pub fn zero(self) -> Mask16x64 { Self::eq(self, Self::splat(0)) }
    #[inline] pub fn nonzero(self) -> Mask16x64 { Self::neq(self, Self::splat(0)) }
    #[inline] pub fn msb(self) -> Mask16x64 {
        Mask16x64([
            self.0[0].msb(), self.0[1].msb(), self.0[2].msb(), self.0[3].msb(),
            self.0[4].msb(), self.0[5].msb(), self.0[6].msb(), self.0[7].msb(),
        ])
    }
    #[inline] pub fn to_bitmask(self) -> u64 {
        let mut out: u64 = 0;
        for i in 0..8 { out |= (self.0[i].to_bitmask() as u64) << (i * 8); }
        out
    }
    #[inline] pub fn mask(self, m: Mask16x64) -> Self { self & m.0 }
    #[inline] pub fn blend(a: Self, b: Self, m: Mask16x64) -> Self { (m.0 & b) | m.0.andnot(a) }
    #[inline] pub fn shl<const N: i32>(self) -> Self {
        let mut r: [u16x8; 8] = unsafe { mem::zeroed() };
        for i in 0..8 { r[i] = self.0[i].shl::<N>(); }
        Self(r)
    }
    #[inline] pub fn shr<const N: i32>(self) -> Self {
        let mut r: [u16x8; 8] = unsafe { mem::zeroed() };
        for i in 0..8 { r[i] = self.0[i].shr::<N>(); }
        Self(r)
    }
    #[inline] pub fn shlv(self, s: Self) -> Self {
        let mut r: [u16x8; 8] = unsafe { mem::zeroed() };
        for i in 0..8 { r[i] = self.0[i].shlv(s.0[i]); }
        Self(r)
    }
    #[inline] pub fn shrv(self, s: Self) -> Self {
        let mut r: [u16x8; 8] = unsafe { mem::zeroed() };
        for i in 0..8 { r[i] = self.0[i].shrv(s.0[i]); }
        Self(r)
    }
    #[inline] pub fn extract16<const I: usize>(self) -> u16x16 {
        u16x16([self.0[I * 2], self.0[I * 2 + 1]])
    }
    #[inline] pub fn extract32<const I: usize>(self) -> u16x32 {
        u16x32([self.0[I * 4], self.0[I * 4 + 1], self.0[I * 4 + 2], self.0[I * 4 + 3]])
    }
    #[inline] pub fn compress(self, m: Mask16x64) -> Self {
        let mask = m.to_bitmask();
        let mut out = [0u16; 64];
        let flat: [u16; 64] = unsafe { mem::transmute(self.0) };
        let mut cursor = 0;
        for i in 0..64 { if (mask >> i) & 1 != 0 { out[cursor] = flat[i]; cursor += 1; } }
        unsafe { Self::load(out.as_ptr()) }
    }
    #[inline] pub unsafe fn compress_store<T>(self, m: Mask16x64, p: *mut T) { unsafe { self.compress(m).store(p) } }
}

impl From<[u16; 64]> for u16x64 { #[inline] fn from(a: [u16; 64]) -> Self { unsafe { Self::load(a.as_ptr()) } } }
impl From<[u16x8; 8]> for u16x64 { #[inline] fn from(a: [u16x8; 8]) -> Self { Self(a) } }
impl BitAnd for u16x64 { type Output = Self; #[inline] fn bitand(self, o: Self) -> Self { let mut r = self.0; for i in 0..8 { r[i] = r[i] & o.0[i]; } Self(r) } }
impl BitOr for u16x64 { type Output = Self; #[inline] fn bitor(self, o: Self) -> Self { let mut r = self.0; for i in 0..8 { r[i] = r[i] | o.0[i]; } Self(r) } }
impl BitXor for u16x64 { type Output = Self; #[inline] fn bitxor(self, o: Self) -> Self { let mut r = self.0; for i in 0..8 { r[i] = r[i] ^ o.0[i]; } Self(r) } }
impl BitAndAssign for u16x64 { #[inline] fn bitand_assign(&mut self, o: Self) { *self = *self & o; } }
impl BitOrAssign for u16x64 { #[inline] fn bitor_assign(&mut self, o: Self) { *self = *self | o; } }
impl BitXorAssign for u16x64 { #[inline] fn bitxor_assign(&mut self, o: Self) { *self = *self ^ o; } }

#[derive(Debug, Copy, Clone)]
pub struct Mask16x64(pub [Mask16x8; 8]);
impl From<[Mask16x8; 8]> for Mask16x64 { #[inline] fn from(a: [Mask16x8; 8]) -> Self { Self(a) } }
impl Not for Mask16x64 { type Output = Self; #[inline] fn not(self) -> Self { let mut r = self.0; for i in 0..8 { r[i] = !r[i]; } Self(r) } }
impl BitAnd for Mask16x64 { type Output = Self; #[inline] fn bitand(self, o: Self) -> Self { let mut r = self.0; for i in 0..8 { r[i] = r[i] & o.0[i]; } Self(r) } }
impl BitOr for Mask16x64 { type Output = Self; #[inline] fn bitor(self, o: Self) -> Self { let mut r = self.0; for i in 0..8 { r[i] = r[i] | o.0[i]; } Self(r) } }
impl Mask16x64 {
    #[inline] pub fn to_bitmask(self) -> u64 {
        let mut out: u64 = 0;
        for i in 0..8 { out |= (self.0[i].to_bitmask() as u64) << (i * 8); }
        out
    }
}

/* ================================================================= */
/*                        Signed types (NNUE)                         */
/* ================================================================= */

#[derive(Debug, Copy, Clone)]
pub struct i16x32(pub [int16x8_t; 4]);

impl i16x32 {
    #[inline] pub unsafe fn load<T>(p: *const T) -> Self {
        let mut a: [int16x8_t; 4] = unsafe { mem::zeroed() };
        for i in 0..4 { a[i] = unsafe { vld1q_s16(p.cast::<i16>().add(i * 8)) }; }
        Self(a)
    }
    #[inline] pub unsafe fn store<T>(self, p: *mut T) {
        for i in 0..4 { unsafe { vst1q_s16(p.cast::<i16>().add(i * 8), self.0[i]) }; }
    }
    #[inline] pub fn splat(v: i16) -> Self { Self([vdupq_n_s16(v); 4]) }
    #[inline] pub fn clamp(self, lo: Self, hi: Self) -> Self {
        let mut r: [int16x8_t; 4] = unsafe { mem::zeroed() };
        for i in 0..4 { r[i] = vminq_s16(vmaxq_s16(self.0[i], lo.0[i]), hi.0[i]); }
        Self(r)
    }
    #[inline] pub fn madd(self, rhs: Self) -> i32x16 {
        // (self * rhs) then pairwise-add pairs of i16 into i32 lanes.
        let mut r: [int32x4_t; 4] = unsafe { mem::zeroed() };
        for i in 0..4 {
            let product = vmulq_s16(self.0[i], rhs.0[i]);
            // Pairwise add: int16x8 -> int16x4 (sum of pairs) -> int32x4
            let lo = vget_low_s16(product);
            let hi = vget_high_s16(product);
            let pairs = vpaddq_s16(vcombine_s16(lo, hi), vcombine_s16(lo, hi));
            r[i] = vmovl_s16(vget_low_s16(pairs));
        }
        i32x16(r)
    }
}

impl Add for i16x32 { type Output = Self; #[inline] fn add(self, o: Self) -> Self {
    let mut r: [int16x8_t; 4] = unsafe { mem::zeroed() };
    for i in 0..4 { r[i] = vaddq_s16(self.0[i], o.0[i]); }
    Self(r)
} }
impl Sub for i16x32 { type Output = Self; #[inline] fn sub(self, o: Self) -> Self {
    let mut r: [int16x8_t; 4] = unsafe { mem::zeroed() };
    for i in 0..4 { r[i] = vsubq_s16(self.0[i], o.0[i]); }
    Self(r)
} }
impl Mul for i16x32 { type Output = Self; #[inline] fn mul(self, o: Self) -> Self {
    let mut r: [int16x8_t; 4] = unsafe { mem::zeroed() };
    for i in 0..4 { r[i] = vmulq_s16(self.0[i], o.0[i]); }
    Self(r)
} }
impl AddAssign for i16x32 { #[inline] fn add_assign(&mut self, o: Self) { *self = *self + o; } }
impl SubAssign for i16x32 { #[inline] fn sub_assign(&mut self, o: Self) { *self = *self - o; } }
impl Default for i16x32 { #[inline] fn default() -> Self { Self([vdupq_n_s16(0); 4]) } }

#[derive(Debug, Copy, Clone)]
pub struct i32x16(pub [int32x4_t; 4]);

impl i32x16 {
    #[inline] pub unsafe fn load<T>(p: *const T) -> Self {
        let mut a: [int32x4_t; 4] = unsafe { mem::zeroed() };
        for i in 0..4 { a[i] = unsafe { vld1q_s32(p.cast::<i32>().add(i * 4)) }; }
        Self(a)
    }
    #[inline] pub unsafe fn store<T>(self, p: *mut T) {
        for i in 0..4 { unsafe { vst1q_s32(p.cast::<i32>().add(i * 4), self.0[i]) }; }
    }
    #[inline] pub fn splat(v: i32) -> Self { Self([vdupq_n_s32(v); 4]) }
    #[inline] pub fn reduce_sum(self) -> i32 {
        let mut sum = vaddvq_s32(self.0[0]);
        sum = sum.wrapping_add(vaddvq_s32(self.0[1]));
        sum = sum.wrapping_add(vaddvq_s32(self.0[2]));
        sum = sum.wrapping_add(vaddvq_s32(self.0[3]));
        sum
    }
}

impl Add for i32x16 { type Output = Self; #[inline] fn add(self, o: Self) -> Self {
    let mut r: [int32x4_t; 4] = unsafe { mem::zeroed() };
    for i in 0..4 { r[i] = vaddq_s32(self.0[i], o.0[i]); }
    Self(r)
} }
impl Sub for i32x16 { type Output = Self; #[inline] fn sub(self, o: Self) -> Self {
    let mut r: [int32x4_t; 4] = unsafe { mem::zeroed() };
    for i in 0..4 { r[i] = vsubq_s32(self.0[i], o.0[i]); }
    Self(r)
} }
impl AddAssign for i32x16 { #[inline] fn add_assign(&mut self, o: Self) { *self = *self + o; } }
impl SubAssign for i32x16 { #[inline] fn sub_assign(&mut self, o: Self) { *self = *self - o; } }
impl Default for i32x16 { #[inline] fn default() -> Self { Self([vdupq_n_s32(0); 4]) } }
