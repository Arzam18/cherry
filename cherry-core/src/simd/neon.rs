//! ARM64 NEON backend for Cherry.
//!
//! Mirrors the public API of `avx2.rs`. NEON is natively 128-bit, so
//! 256/512-bit types are composed as arrays of 128-bit halves. Every
//! operation matches the AVX2 semantics so `perft` and `bench` produce
//! identical output across x86_64 and aarch64.
//!
//! On aarch64 every NEON intrinsic is gated behind
//! `#[target_feature(enable = "neon")]` and is unsafe to call, so method
//! bodies wrap the intrinsic calls in `unsafe { }`.

#![allow(unused_unsafe)]
#![allow(dead_code)]

use core::{arch::aarch64::*, mem, ops::*};

/* ==================== helper functions ==================== */

#[inline] fn movemask_u8x16(v: uint8x16_t) -> u16 {
    let b: [u8; 16] = unsafe { mem::transmute(v) };
    let mut o: u16 = 0; let mut i = 0;
    while i < 16 { o |= ((b[i] >> 7) as u16) << i; i += 1; }
    o
}
#[inline] fn movemask_u16x8(v: uint16x8_t) -> u8 {
    let h: [u16; 8] = unsafe { mem::transmute(v) };
    let mut o: u8 = 0; let mut i = 0;
    while i < 8 { o |= ((h[i] >> 15) as u8) << i; i += 1; }
    o
}
#[inline] fn movemask_u32x4(v: uint32x4_t) -> u8 {
    let w: [u32; 4] = unsafe { mem::transmute(v) };
    let mut o: u8 = 0; let mut i = 0;
    while i < 4 { o |= ((w[i] >> 31) as u8) << i; i += 1; }
    o
}
#[inline] fn movemask_u64x2(v: uint64x2_t) -> u8 {
    let w: [u64; 2] = unsafe { mem::transmute(v) };
    ((w[0] >> 63) as u8) | (((w[1] >> 63) as u8) << 1)
}

/* ==================== u8x16 ==================== */

#[derive(Debug, Copy, Clone)]
pub struct u8x16(pub uint8x16_t);
impl u8x16 {
    #[inline] pub unsafe fn load<T>(p: *const T) -> Self { unsafe { Self(vld1q_u8(p.cast())) } }
    #[inline] pub unsafe fn store<T>(self, p: *mut T) { unsafe { vst1q_u8(p.cast(), self.0) } }
    #[inline] pub fn splat(v: u8) -> Self { unsafe { Self(vdupq_n_u8(v)) } }
    #[inline] pub fn andnot(self, o: Self) -> Self { unsafe { Self(vbicq_u8(self.0, o.0)) } }
    #[inline] pub fn eq(a: Self, b: Self) -> Mask8x16 { unsafe { Mask8x16(Self(vceqq_u8(a.0, b.0))) } }
    #[inline] pub fn neq(a: Self, b: Self) -> Mask8x16 { Self::eq(a, b).not() }
    #[inline] pub fn test(a: Self, b: Self) -> Mask8x16 { (a & b).nonzero() }
    #[inline] pub fn testn(a: Self, b: Self) -> Mask8x16 { (a & b).zero() }
    #[inline] pub fn zero(self) -> Mask8x16 { Self::eq(self, Self::splat(0)) }
    #[inline] pub fn nonzero(self) -> Mask8x16 { Self::neq(self, Self::splat(0)) }
    #[inline] pub fn msb(self) -> Mask8x16 { unsafe { Mask8x16(Self(vcltq_s8(vreinterpretq_s8_u8(self.0), vdupq_n_s8(0)))) } }
    #[inline] pub fn to_bitmask(self) -> u16 { movemask_u8x16(self.0) }
    #[inline] pub fn mask(self, m: Mask8x16) -> Self { self & m.0 }
    #[inline] pub fn blend(a: Self, b: Self, m: Mask8x16) -> Self { (m.0 & b) | m.0.andnot(a) }
    #[inline] pub fn compress(self, m: Mask8x16) -> Self { unsafe {
        let arr: [u8; 16] = mem::transmute(self.0);
        let mut out = [0u8; 16]; let mask = m.to_bitmask(); let mut c = 0; let mut i = 0;
        while i < 16 { if (mask >> i) & 1 != 0 { out[c] = arr[i]; c += 1; } i += 1; }
        Self::load(out.as_ptr())
    } }
    #[inline] pub unsafe fn compress_store<T>(self, m: Mask8x16, p: *mut T) { unsafe { self.compress(m).store(p) } }
    #[inline] pub fn shuffle(self, idx: Self) -> Self { unsafe { Self(vqtbl1q_u8(self.0, idx.0)) } }
    #[inline] pub fn extract<const I: i32>(self) -> u8 { unsafe { vgetq_lane_u8::<I>(self.0) } }
    #[inline] pub fn broadcast64(self) -> u8x64 { u8x64([self, self, self, self]) }
    #[inline] pub fn to_u16x8(self) -> u16x8 { unsafe { u16x8(vreinterpretq_u16_u8(self.0)) } }
    #[inline] pub fn to_u32x4(self) -> u32x4 { unsafe { u32x4(vreinterpretq_u32_u8(self.0)) } }
    #[inline] pub fn to_u64x2(self) -> u64x2 { unsafe { u64x2(vreinterpretq_u64_u8(self.0)) } }
    #[inline] pub fn zero_ext(self) -> u16x16 { unsafe {
        u16x16([u16x8(vmovl_u8(vget_low_u8(self.0))), u16x8(vmovl_u8(vget_high_u8(self.0)))])
    } }
    #[inline] pub fn findset(self, needles: Self, count: usize) -> u16 {
        let a: [u8; 16] = unsafe { mem::transmute(self.0) };
        let b: [u8; 16] = unsafe { mem::transmute(needles.0) };
        let mut out: u16 = 0;
        let mut i = 0;
        while i < count { let mut j = 0;
            while j < 16 { if b[i] == a[j] { out |= 1u16 << j; } j += 1; }
            i += 1;
        }
        out
    }
}
impl From<uint8x16_t> for u8x16 { #[inline] fn from(v: uint8x16_t) -> Self { Self(v) } }
impl From<[u8; 16]> for u8x16 { #[inline] fn from(a: [u8; 16]) -> Self { unsafe { Self::load(a.as_ptr()) } } }
impl Not for u8x16 { type Output = Self; #[inline] fn not(self) -> Self { unsafe { Self(vmvnq_u8(self.0)) } } }
impl BitAnd for u8x16 { type Output = Self; #[inline] fn bitand(self, o: Self) -> Self { unsafe { Self(vandq_u8(self.0, o.0)) } } }
impl BitOr  for u8x16 { type Output = Self; #[inline] fn bitor (self, o: Self) -> Self { unsafe { Self(vorrq_u8(self.0, o.0)) } } }
impl BitXor for u8x16 { type Output = Self; #[inline] fn bitxor(self, o: Self) -> Self { unsafe { Self(veorq_u8(self.0, o.0)) } } }
impl BitAndAssign for u8x16 { #[inline] fn bitand_assign(&mut self, o: Self) { *self = *self & o; } }
impl BitOrAssign  for u8x16 { #[inline] fn bitor_assign (&mut self, o: Self) { *self = *self | o; } }
impl BitXorAssign for u8x16 { #[inline] fn bitxor_assign(&mut self, o: Self) { *self = *self ^ o; } }
impl Default for u8x16 { #[inline] fn default() -> Self { Self::splat(0) } }

/* ==================== u16x8 ==================== */

#[derive(Debug, Copy, Clone)]
pub struct u16x8(pub uint16x8_t);
impl u16x8 {
    #[inline] pub unsafe fn load<T>(p: *const T) -> Self { unsafe { Self(vld1q_u16(p.cast())) } }
    #[inline] pub unsafe fn store<T>(self, p: *mut T) { unsafe { vst1q_u16(p.cast(), self.0) } }
    #[inline] pub fn splat(v: u16) -> Self { unsafe { Self(vdupq_n_u16(v)) } }
    #[inline] pub fn andnot(self, o: Self) -> Self { unsafe { Self(vbicq_u16(self.0, o.0)) } }
    #[inline] pub fn eq(a: Self, b: Self) -> Mask16x8 { unsafe { Mask16x8(Self(vceqq_u16(a.0, b.0))) } }
    #[inline] pub fn neq(a: Self, b: Self) -> Mask16x8 { Self::eq(a, b).not() }
    #[inline] pub fn test(a: Self, b: Self) -> Mask16x8 { (a & b).nonzero() }
    #[inline] pub fn testn(a: Self, b: Self) -> Mask16x8 { (a & b).zero() }
    #[inline] pub fn zero(self) -> Mask16x8 { Self::eq(self, Self::splat(0)) }
    #[inline] pub fn nonzero(self) -> Mask16x8 { Self::neq(self, Self::splat(0)) }
    #[inline] pub fn msb(self) -> Mask16x8 { unsafe { Mask16x8(Self(vcltq_s16(vreinterpretq_s16_u16(self.0), vdupq_n_s16(0)))) } }
    #[inline] pub fn to_bitmask(self) -> u8 { movemask_u16x8(self.0) }
    #[inline] pub fn mask(self, m: Mask16x8) -> Self { self & m.0 }
    #[inline] pub fn blend(a: Self, b: Self, m: Mask16x8) -> Self { (m.0 & b) | m.0.andnot(a) }
    #[inline] pub fn compress(self, m: Mask16x8) -> Self { unsafe {
        let arr: [u16; 8] = mem::transmute(self.0);
        let mut out = [0u16; 8]; let mask = m.to_bitmask(); let mut c = 0;
        for i in 0..8 { if (mask >> i) & 1 != 0 { out[c] = arr[i]; c += 1; } }
        Self::load(out.as_ptr())
    } }
    #[inline] pub unsafe fn compress_store<T>(self, m: Mask16x8, p: *mut T) { unsafe { self.compress(m).store(p) } }
    #[inline] pub fn shl<const N: i32>(self) -> Self { unsafe { Self(vshlq_n_u16::<N>(self.0)) } }
    #[inline] pub fn shr<const N: i32>(self) -> Self { unsafe { Self(vshrq_n_u16::<N>(self.0)) } }
    #[inline] pub fn shlv(self, s: Self) -> Self { unsafe { Self(vshlq_u16(self.0, vreinterpretq_s16_u16(s.0))) } }
    #[inline] pub fn shrv(self, s: Self) -> Self { unsafe { Self(vshlq_u16(self.0, vnegq_s16(vreinterpretq_s16_u16(s.0)))) } }
    #[inline] pub fn extract<const I: i32>(self) -> u16 { unsafe { vgetq_lane_u16::<I>(self.0) } }
    #[inline] pub fn broadcast32(self) -> u16x32 { u16x32([self, self, self, self]) }
    #[inline] pub fn to_u8x16(self) -> u8x16 { unsafe { u8x16(vreinterpretq_u8_u16(self.0)) } }
    #[inline] pub fn to_u32x4(self) -> u32x4 { unsafe { u32x4(vreinterpretq_u32_u16(self.0)) } }
    #[inline] pub fn to_u64x2(self) -> u64x2 { unsafe { u64x2(vreinterpretq_u64_u16(self.0)) } }
    #[inline] pub fn zero_ext(self) -> u32x8 { unsafe {
        u32x8([u32x4(vmovl_u16(vget_low_u16(self.0))), u32x4(vmovl_u16(vget_high_u16(self.0)))])
    } }
}
impl From<uint16x8_t> for u16x8 { #[inline] fn from(v: uint16x8_t) -> Self { Self(v) } }
impl From<[u16; 8]> for u16x8 { #[inline] fn from(a: [u16; 8]) -> Self { unsafe { Self::load(a.as_ptr()) } } }
impl Not for u16x8 { type Output = Self; #[inline] fn not(self) -> Self { unsafe { Self(vmvnq_u16(self.0)) } } }
impl BitAnd for u16x8 { type Output = Self; #[inline] fn bitand(self, o: Self) -> Self { unsafe { Self(vandq_u16(self.0, o.0)) } } }
impl BitOr  for u16x8 { type Output = Self; #[inline] fn bitor (self, o: Self) -> Self { unsafe { Self(vorrq_u16(self.0, o.0)) } } }
impl BitXor for u16x8 { type Output = Self; #[inline] fn bitxor(self, o: Self) -> Self { unsafe { Self(veorq_u16(self.0, o.0)) } } }
impl BitAndAssign for u16x8 { #[inline] fn bitand_assign(&mut self, o: Self) { *self = *self & o; } }
impl BitOrAssign  for u16x8 { #[inline] fn bitor_assign (&mut self, o: Self) { *self = *self | o; } }
impl BitXorAssign for u16x8 { #[inline] fn bitxor_assign(&mut self, o: Self) { *self = *self ^ o; } }
impl Default for u16x8 { #[inline] fn default() -> Self { Self::splat(0) } }

/* ==================== u32x4 ==================== */

#[derive(Debug, Copy, Clone)]
pub struct u32x4(pub uint32x4_t);
impl u32x4 {
    #[inline] pub unsafe fn load<T>(p: *const T) -> Self { unsafe { Self(vld1q_u32(p.cast())) } }
    #[inline] pub unsafe fn store<T>(self, p: *mut T) { unsafe { vst1q_u32(p.cast(), self.0) } }
    #[inline] pub fn splat(v: u32) -> Self { unsafe { Self(vdupq_n_u32(v)) } }
    #[inline] pub fn andnot(self, o: Self) -> Self { unsafe { Self(vbicq_u32(self.0, o.0)) } }
    #[inline] pub fn eq(a: Self, b: Self) -> Mask32x4 { unsafe { Mask32x4(Self(vceqq_u32(a.0, b.0))) } }
    #[inline] pub fn neq(a: Self, b: Self) -> Mask32x4 { Self::eq(a, b).not() }
    #[inline] pub fn test(a: Self, b: Self) -> Mask32x4 { (a & b).nonzero() }
    #[inline] pub fn testn(a: Self, b: Self) -> Mask32x4 { (a & b).zero() }
    #[inline] pub fn zero(self) -> Mask32x4 { Self::eq(self, Self::splat(0)) }
    #[inline] pub fn nonzero(self) -> Mask32x4 { Self::neq(self, Self::splat(0)) }
    #[inline] pub fn msb(self) -> Mask32x4 { unsafe { Mask32x4(Self(vcltq_s32(vreinterpretq_s32_u32(self.0), vdupq_n_s32(0)))) } }
    #[inline] pub fn to_bitmask(self) -> u8 { movemask_u32x4(self.0) }
    #[inline] pub fn mask(self, m: Mask32x4) -> Self { self & m.0 }
    #[inline] pub fn blend(a: Self, b: Self, m: Mask32x4) -> Self { (m.0 & b) | m.0.andnot(a) }
    #[inline] pub fn compress(self, m: Mask32x4) -> Self { unsafe {
        let arr: [u32; 4] = mem::transmute(self.0);
        let mut out = [0u32; 4]; let mask = m.to_bitmask(); let mut c = 0;
        for i in 0..4 { if (mask >> i) & 1 != 0 { out[c] = arr[i]; c += 1; } }
        Self::load(out.as_ptr())
    } }
    #[inline] pub unsafe fn compress_store<T>(self, m: Mask32x4, p: *mut T) { unsafe { self.compress(m).store(p) } }
    #[inline] pub fn shl<const N: i32>(self) -> Self { unsafe { Self(vshlq_n_u32::<N>(self.0)) } }
    #[inline] pub fn shr<const N: i32>(self) -> Self { unsafe { Self(vshrq_n_u32::<N>(self.0)) } }
    #[inline] pub fn extract<const I: i32>(self) -> u32 { unsafe { vgetq_lane_u32::<I>(self.0) } }
    #[inline] pub fn broadcast16(self) -> u32x16 { u32x16([self, self, self, self]) }
    #[inline] pub fn to_u8x16(self) -> u8x16 { unsafe { u8x16(vreinterpretq_u8_u32(self.0)) } }
    #[inline] pub fn to_u16x8(self) -> u16x8 { unsafe { u16x8(vreinterpretq_u16_u32(self.0)) } }
    #[inline] pub fn to_u64x2(self) -> u64x2 { unsafe { u64x2(vreinterpretq_u64_u32(self.0)) } }
    #[inline] pub fn zero_ext(self) -> u64x4 { unsafe {
        u64x4([u64x2(vmovl_u32(vget_low_u32(self.0))), u64x2(vmovl_u32(vget_high_u32(self.0)))])
    } }
}
impl From<uint32x4_t> for u32x4 { #[inline] fn from(v: uint32x4_t) -> Self { Self(v) } }
impl From<[u32; 4]> for u32x4 { #[inline] fn from(a: [u32; 4]) -> Self { unsafe { Self::load(a.as_ptr()) } } }
impl Not for u32x4 { type Output = Self; #[inline] fn not(self) -> Self { unsafe { Self(vmvnq_u32(self.0)) } } }
impl BitAnd for u32x4 { type Output = Self; #[inline] fn bitand(self, o: Self) -> Self { unsafe { Self(vandq_u32(self.0, o.0)) } } }
impl BitOr  for u32x4 { type Output = Self; #[inline] fn bitor (self, o: Self) -> Self { unsafe { Self(vorrq_u32(self.0, o.0)) } } }
impl BitXor for u32x4 { type Output = Self; #[inline] fn bitxor(self, o: Self) -> Self { unsafe { Self(veorq_u32(self.0, o.0)) } } }
impl BitAndAssign for u32x4 { #[inline] fn bitand_assign(&mut self, o: Self) { *self = *self & o; } }
impl BitOrAssign  for u32x4 { #[inline] fn bitor_assign (&mut self, o: Self) { *self = *self | o; } }
impl BitXorAssign for u32x4 { #[inline] fn bitxor_assign(&mut self, o: Self) { *self = *self ^ o; } }
impl Default for u32x4 { #[inline] fn default() -> Self { Self::splat(0) } }

/* ==================== u64x2 ==================== */

#[derive(Debug, Copy, Clone)]
pub struct u64x2(pub uint64x2_t);
impl u64x2 {
    #[inline] pub unsafe fn load<T>(p: *const T) -> Self { unsafe { Self(vld1q_u64(p.cast())) } }
    #[inline] pub unsafe fn store<T>(self, p: *mut T) { unsafe { vst1q_u64(p.cast(), self.0) } }
    #[inline] pub fn splat(v: u64) -> Self { unsafe { Self(vdupq_n_u64(v)) } }
    #[inline] pub fn andnot(self, o: Self) -> Self { unsafe { Self(vbicq_u64(self.0, o.0)) } }
    #[inline] pub fn eq(a: Self, b: Self) -> Mask64x2 { unsafe { Mask64x2(Self(vceqq_u64(a.0, b.0))) } }
    #[inline] pub fn neq(a: Self, b: Self) -> Mask64x2 { Self::eq(a, b).not() }
    #[inline] pub fn test(a: Self, b: Self) -> Mask64x2 { (a & b).nonzero() }
    #[inline] pub fn testn(a: Self, b: Self) -> Mask64x2 { (a & b).zero() }
    #[inline] pub fn zero(self) -> Mask64x2 { Self::eq(self, Self::splat(0)) }
    #[inline] pub fn nonzero(self) -> Mask64x2 { Self::neq(self, Self::splat(0)) }
    #[inline] pub fn msb(self) -> Mask64x2 { unsafe { Mask64x2(Self(vcltq_s64(vreinterpretq_s64_u64(self.0), vdupq_n_s64(0)))) } }
    #[inline] pub fn to_bitmask(self) -> u8 { movemask_u64x2(self.0) }
    #[inline] pub fn shl<const N: i32>(self) -> Self { unsafe { Self(vshlq_n_u64::<N>(self.0)) } }
    #[inline] pub fn shr<const N: i32>(self) -> Self { unsafe { Self(vshrq_n_u64::<N>(self.0)) } }
    #[inline] pub fn extract<const I: i32>(self) -> u64 { unsafe { vgetq_lane_u64::<I>(self.0) } }
    #[inline] pub fn broadcast8(self) -> u64x8 { u64x8([self, self, self, self]) }
}
impl From<uint64x2_t> for u64x2 { #[inline] fn from(v: uint64x2_t) -> Self { Self(v) } }
impl From<[u64; 2]> for u64x2 { #[inline] fn from(a: [u64; 2]) -> Self { unsafe { Self::load(a.as_ptr()) } } }
// NOTE: aarch64 NEON has no `vmvnq_u64` (the MVN instruction is only defined for
// 8/16/32-bit elements). Emulate NOT by XORing with all-ones.
impl Not for u64x2 { type Output = Self; #[inline] fn not(self) -> Self { unsafe { Self(veorq_u64(self.0, vdupq_n_u64(u64::MAX))) } } }
impl BitAnd for u64x2 { type Output = Self; #[inline] fn bitand(self, o: Self) -> Self { unsafe { Self(vandq_u64(self.0, o.0)) } } }
impl BitOr  for u64x2 { type Output = Self; #[inline] fn bitor (self, o: Self) -> Self { unsafe { Self(vorrq_u64(self.0, o.0)) } } }
impl BitXor for u64x2 { type Output = Self; #[inline] fn bitxor(self, o: Self) -> Self { unsafe { Self(veorq_u64(self.0, o.0)) } } }
impl BitAndAssign for u64x2 { #[inline] fn bitand_assign(&mut self, o: Self) { *self = *self & o; } }
impl BitOrAssign  for u64x2 { #[inline] fn bitor_assign (&mut self, o: Self) { *self = *self | o; } }
impl BitXorAssign for u64x2 { #[inline] fn bitxor_assign(&mut self, o: Self) { *self = *self ^ o; } }
impl Default for u64x2 { #[inline] fn default() -> Self { Self::splat(0) } }

/* ==================== wide vector macro ==================== */

macro_rules! def_wide_vec {
    ($name:ident, $inner:ident, $n:expr, $elem:ty, $arr:ty, $zero:expr) => {
        #[derive(Debug, Copy, Clone)]
        pub struct $name(pub [$inner; $n]);
        impl $name {
            #[inline] pub unsafe fn load<T>(p: *const T) -> Self { unsafe {
                let mut a: [$inner; $n] = [$zero; $n];
                let stride = mem::size_of::<$inner>();
                let mut i = 0;
                while i < $n { a[i] = $inner::load(p.cast::<u8>().add(i * stride)); i += 1; }
                Self(a)
            } }
            #[inline] pub unsafe fn store<T>(self, p: *mut T) { unsafe {
                let stride = mem::size_of::<$inner>();
                let mut i = 0;
                while i < $n { self.0[i].store(p.cast::<u8>().add(i * stride)); i += 1; }
            } }
            #[inline] pub fn splat(v: $elem) -> Self { Self([$inner::splat(v); $n]) }
            #[inline] pub fn andnot(self, o: Self) -> Self {
                let mut r: [$inner; $n] = [$zero; $n];
                let mut i = 0;
                while i < $n { r[i] = self.0[i].andnot(o.0[i]); i += 1; }
                Self(r)
            }
        }
        impl From<$arr> for $name { #[inline] fn from(a: $arr) -> Self { unsafe { Self::load(a.as_ptr()) } } }
        impl Not for $name { type Output = Self; #[inline] fn not(self) -> Self {
            let mut r = self.0; let mut i = 0; while i < $n { r[i] = !self.0[i]; i += 1; } Self(r)
        } }
        impl BitAnd for $name { type Output = Self; #[inline] fn bitand(self, o: Self) -> Self {
            let mut r = self.0; let mut i = 0; while i < $n { r[i] = self.0[i] & o.0[i]; i += 1; } Self(r)
        } }
        impl BitOr for $name { type Output = Self; #[inline] fn bitor(self, o: Self) -> Self {
            let mut r = self.0; let mut i = 0; while i < $n { r[i] = self.0[i] | o.0[i]; i += 1; } Self(r)
        } }
        impl BitXor for $name { type Output = Self; #[inline] fn bitxor(self, o: Self) -> Self {
            let mut r = self.0; let mut i = 0; while i < $n { r[i] = self.0[i] ^ o.0[i]; i += 1; } Self(r)
        } }
        impl BitAndAssign for $name { #[inline] fn bitand_assign(&mut self, o: Self) { *self = *self & o; } }
        impl BitOrAssign  for $name { #[inline] fn bitor_assign (&mut self, o: Self) { *self = *self | o; } }
        impl BitXorAssign for $name { #[inline] fn bitxor_assign(&mut self, o: Self) { *self = *self ^ o; } }
    };
}

def_wide_vec!(u8x32,  u8x16,  2, u8,  [u8; 32],  u8x16::splat(0));
def_wide_vec!(u8x64,  u8x16,  4, u8,  [u8; 64],  u8x16::splat(0));
def_wide_vec!(u16x16, u16x8,  2, u16, [u16; 16], u16x8::splat(0));
def_wide_vec!(u16x32, u16x8,  4, u16, [u16; 32], u16x8::splat(0));
def_wide_vec!(u32x8,  u32x4,  2, u32, [u32; 8],  u32x4::splat(0));
def_wide_vec!(u32x16, u32x4,  4, u32, [u32; 16], u32x4::splat(0));
def_wide_vec!(u64x4,  u64x2,  2, u64, [u64; 4],  u64x2::splat(0));
def_wide_vec!(u64x8,  u64x2,  4, u64, [u64; 8],  u64x2::splat(0));

/* ==================== wide u8x32 ==================== */

impl u8x32 {
    #[inline] pub fn eq(a: Self, b: Self) -> Mask8x32 { Mask8x32([u8x16::eq(a.0[0], b.0[0]), u8x16::eq(a.0[1], b.0[1])]) }
    #[inline] pub fn neq(a: Self, b: Self) -> Mask8x32 { Self::eq(a, b).not() }
    #[inline] pub fn zero(self) -> Mask8x32 { Self::eq(self, Self::splat(0)) }
    #[inline] pub fn nonzero(self) -> Mask8x32 { Self::neq(self, Self::splat(0)) }
    #[inline] pub fn msb(self) -> Mask8x32 { Mask8x32([self.0[0].msb(), self.0[1].msb()]) }
    #[inline] pub fn to_bitmask(self) -> u32 {
        (self.0[0].to_bitmask() as u32) | ((self.0[1].to_bitmask() as u32) << 16)
    }
    #[inline] pub fn mask(self, m: Mask8x32) -> Self {
        u8x32([self.0[0].mask(m.0[0]), self.0[1].mask(m.0[1])])
    }
    #[inline] pub fn blend(a: Self, b: Self, m: Mask8x32) -> Self {
        u8x32([u8x16::blend(a.0[0], b.0[0], m.0[0]),
               u8x16::blend(a.0[1], b.0[1], m.0[1])])
    }
    #[inline] pub fn compress(self, m: Mask8x32) -> Self { unsafe {
        let mask = m.to_bitmask();
        let flat: [u8; 32] = mem::transmute(self.0);
        let mut out = [0u8; 32]; let mut c = 0;
        for i in 0..32 { if (mask >> i) & 1 != 0 { out[c] = flat[i]; c += 1; } }
        Self::load(out.as_ptr())
    } }
    #[inline] pub unsafe fn compress_store<T>(self, m: Mask8x32, p: *mut T) { unsafe { self.compress(m).store(p) } }
    #[inline] pub fn extract16<const I: i32>(self) -> u8x16 { self.0[I as usize] }
    #[inline] pub fn zero_ext(self) -> u16x32 {
        let a = self.0[0].zero_ext();
        let b = self.0[1].zero_ext();
        u16x32([a.0[0], a.0[1], b.0[0], b.0[1]])
    }
}

/* ==================== wide u8x64 ==================== */

impl u8x64 {
    #[inline] pub fn eq(a: Self, b: Self) -> Mask8x64 {
        Mask8x64([u8x16::eq(a.0[0], b.0[0]), u8x16::eq(a.0[1], b.0[1]),
                   u8x16::eq(a.0[2], b.0[2]), u8x16::eq(a.0[3], b.0[3])])
    }
    #[inline] pub fn neq(a: Self, b: Self) -> Mask8x64 { Self::eq(a, b).not() }
    #[inline] pub fn zero(self) -> Mask8x64 { Self::eq(self, Self::splat(0)) }
    #[inline] pub fn nonzero(self) -> Mask8x64 { Self::neq(self, Self::splat(0)) }
    #[inline] pub fn msb(self) -> Mask8x64 {
        Mask8x64([self.0[0].msb(), self.0[1].msb(), self.0[2].msb(), self.0[3].msb()])
    }
    #[inline] pub fn to_bitmask(self) -> u64 {
        (self.0[0].to_bitmask() as u64) | ((self.0[1].to_bitmask() as u64) << 16)
            | ((self.0[2].to_bitmask() as u64) << 32) | ((self.0[3].to_bitmask() as u64) << 48)
    }
    #[inline] pub fn mask(self, m: Mask8x64) -> Self {
        u8x64([self.0[0].mask(m.0[0]), self.0[1].mask(m.0[1]),
               self.0[2].mask(m.0[2]), self.0[3].mask(m.0[3])])
    }
    #[inline] pub fn blend(a: Self, b: Self, m: Mask8x64) -> Self {
        u8x64([u8x16::blend(a.0[0], b.0[0], m.0[0]),
               u8x16::blend(a.0[1], b.0[1], m.0[1]),
               u8x16::blend(a.0[2], b.0[2], m.0[2]),
               u8x16::blend(a.0[3], b.0[3], m.0[3])])
    }
    #[inline] pub fn compress(self, m: Mask8x64) -> Self { unsafe {
        let mask = m.to_bitmask();
        let flat: [u8; 64] = mem::transmute(self.0);
        let mut out = [0u8; 64]; let mut c = 0;
        for i in 0..64 { if (mask >> i) & 1 != 0 { out[c] = flat[i]; c += 1; } }
        Self::load(out.as_ptr())
    } }
    #[inline] pub unsafe fn compress_store<T>(self, m: Mask8x64, p: *mut T) { unsafe { self.compress(m).store(p) } }
    #[inline] pub fn permute(self, idx: Self) -> Self { unsafe {
        let src: [u8; 64] = mem::transmute(self.0);
        let ind: [u8; 64] = mem::transmute(idx.0);
        let mut out = [0u8; 64];
        for i in 0..64 { out[i] = src[(ind[i] & 63) as usize]; }
        Self::load(out.as_ptr())
    } }
    #[inline] pub fn shuffle(self, idx: Self) -> Self { self.permute(idx) }
    #[inline] pub fn extract16<const I: usize>(self) -> u8x16 { self.0[I] }
    #[inline] pub fn extract32<const I: usize>(self) -> u8x32 { u8x32([self.0[I*2], self.0[I*2+1]]) }
    #[inline] pub fn flip_rays(self) -> Self { u8x64([self.0[2], self.0[3], self.0[0], self.0[1]]) }
    #[inline] pub fn extend_rays(self) -> Self { unsafe {
        let flat: [u8; 64] = mem::transmute(self.0);
        let mut sums = [0u8; 8];
        for i in 0..8 {
            let mut s = 0u16;
            for j in 0..8 { s += flat[i * 8 + j] as u16; }
            sums[i] = (s & 0xFF) as u8;
        }
        let mut out = [0u8; 64];
        for i in 0..64 { out[i] = sums[i / 8]; }
        Self::load(out.as_ptr())
    } }
    #[inline] pub fn zero_ext(self) -> u16x64 {
        let a = self.0[0].zero_ext();
        let b = self.0[1].zero_ext();
        let c = self.0[2].zero_ext();
        let d = self.0[3].zero_ext();
        u16x64([a.0[0], a.0[1], b.0[0], b.0[1], c.0[0], c.0[1], d.0[0], d.0[1]])
    }
    #[inline] pub fn to_u16x32(self) -> u16x32 { unsafe {
        u16x32([
            u16x8(vreinterpretq_u16_u8(self.0[0].0)), u16x8(vreinterpretq_u16_u8(self.0[1].0)),
            u16x8(vreinterpretq_u16_u8(self.0[2].0)), u16x8(vreinterpretq_u16_u8(self.0[3].0)),
        ])
    } }
}

/* ==================== wide u16x16 ==================== */

impl u16x16 {
    #[inline] pub fn eq(a: Self, b: Self) -> Mask16x16 { Mask16x16([u16x8::eq(a.0[0], b.0[0]), u16x8::eq(a.0[1], b.0[1])]) }
    #[inline] pub fn neq(a: Self, b: Self) -> Mask16x16 { Self::eq(a, b).not() }
    #[inline] pub fn test(a: Self, b: Self) -> Mask16x16 { (a & b).nonzero() }
    #[inline] pub fn testn(a: Self, b: Self) -> Mask16x16 { (a & b).zero() }
    #[inline] pub fn zero(self) -> Mask16x16 { Self::eq(self, Self::splat(0)) }
    #[inline] pub fn nonzero(self) -> Mask16x16 { Self::neq(self, Self::splat(0)) }
    #[inline] pub fn msb(self) -> Mask16x16 { Mask16x16([self.0[0].msb(), self.0[1].msb()]) }
    #[inline] pub fn to_bitmask(self) -> u16 {
        (self.0[0].to_bitmask() as u16) | ((self.0[1].to_bitmask() as u16) << 8)
    }
    #[inline] pub fn mask(self, m: Mask16x16) -> Self {
        u16x16([self.0[0].mask(m.0[0]), self.0[1].mask(m.0[1])])
    }
    #[inline] pub fn blend(a: Self, b: Self, m: Mask16x16) -> Self {
        u16x16([u16x8::blend(a.0[0], b.0[0], m.0[0]),
                u16x8::blend(a.0[1], b.0[1], m.0[1])])
    }
    #[inline] pub fn compress(self, m: Mask16x16) -> Self { unsafe {
        let mask = m.to_bitmask();
        let flat: [u16; 16] = mem::transmute(self.0);
        let mut out = [0u16; 16]; let mut c = 0;
        for i in 0..16 { if (mask >> i) & 1 != 0 { out[c] = flat[i]; c += 1; } }
        Self::load(out.as_ptr())
    } }
    #[inline] pub unsafe fn compress_store<T>(self, m: Mask16x16, p: *mut T) { unsafe { self.compress(m).store(p) } }
    #[inline] pub fn shl<const N: i32>(self) -> Self { u16x16([self.0[0].shl::<N>(), self.0[1].shl::<N>()]) }
    #[inline] pub fn shr<const N: i32>(self) -> Self { u16x16([self.0[0].shr::<N>(), self.0[1].shr::<N>()]) }
    #[inline] pub fn shlv(self, s: Self) -> Self { u16x16([self.0[0].shlv(s.0[0]), self.0[1].shlv(s.0[1])]) }
    #[inline] pub fn shrv(self, s: Self) -> Self { u16x16([self.0[0].shrv(s.0[0]), self.0[1].shrv(s.0[1])]) }
    #[inline] pub fn extract16<const I: usize>(self) -> u16x8 { self.0[I] }
    #[inline] pub fn zero_ext(self) -> u32x16 {
        let a = self.0[0].zero_ext();
        let b = self.0[1].zero_ext();
        u32x16([a.0[0], a.0[1], b.0[0], b.0[1]])
    }
}

/* ==================== wide u16x32 ==================== */

impl u16x32 {
    #[inline] pub fn eq(a: Self, b: Self) -> Mask16x32 {
        Mask16x32([u16x8::eq(a.0[0], b.0[0]), u16x8::eq(a.0[1], b.0[1]),
                    u16x8::eq(a.0[2], b.0[2]), u16x8::eq(a.0[3], b.0[3])])
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
        (self.0[0].to_bitmask() as u32) | ((self.0[1].to_bitmask() as u32) << 8)
            | ((self.0[2].to_bitmask() as u32) << 16) | ((self.0[3].to_bitmask() as u32) << 24)
    }
    #[inline] pub fn mask(self, m: Mask16x32) -> Self {
        u16x32([self.0[0].mask(m.0[0]), self.0[1].mask(m.0[1]),
                self.0[2].mask(m.0[2]), self.0[3].mask(m.0[3])])
    }
    #[inline] pub fn blend(a: Self, b: Self, m: Mask16x32) -> Self {
        u16x32([u16x8::blend(a.0[0], b.0[0], m.0[0]),
                u16x8::blend(a.0[1], b.0[1], m.0[1]),
                u16x8::blend(a.0[2], b.0[2], m.0[2]),
                u16x8::blend(a.0[3], b.0[3], m.0[3])])
    }
    #[inline] pub fn compress(self, m: Mask16x32) -> Self { unsafe {
        let mask = m.to_bitmask();
        let flat: [u16; 32] = mem::transmute(self.0);
        let mut out = [0u16; 32]; let mut c = 0;
        for i in 0..32 { if (mask >> i) & 1 != 0 { out[c] = flat[i]; c += 1; } }
        Self::load(out.as_ptr())
    } }
    #[inline] pub unsafe fn compress_store<T>(self, m: Mask16x32, p: *mut T) { unsafe { self.compress(m).store(p) } }
    #[inline] pub fn shl<const N: i32>(self) -> Self {
        u16x32([self.0[0].shl::<N>(), self.0[1].shl::<N>(), self.0[2].shl::<N>(), self.0[3].shl::<N>()])
    }
    #[inline] pub fn shr<const N: i32>(self) -> Self {
        u16x32([self.0[0].shr::<N>(), self.0[1].shr::<N>(), self.0[2].shr::<N>(), self.0[3].shr::<N>()])
    }
    #[inline] pub fn shlv(self, s: Self) -> Self {
        u16x32([self.0[0].shlv(s.0[0]), self.0[1].shlv(s.0[1]), self.0[2].shlv(s.0[2]), self.0[3].shlv(s.0[3])])
    }
    #[inline] pub fn shrv(self, s: Self) -> Self {
        u16x32([self.0[0].shrv(s.0[0]), self.0[1].shrv(s.0[1]), self.0[2].shrv(s.0[2]), self.0[3].shrv(s.0[3])])
    }
    #[inline] pub fn to_u8x64(self) -> u8x64 { unsafe {
        u8x64([
            u8x16(vreinterpretq_u8_u16(self.0[0].0)), u8x16(vreinterpretq_u8_u16(self.0[1].0)),
            u8x16(vreinterpretq_u8_u16(self.0[2].0)), u8x16(vreinterpretq_u8_u16(self.0[3].0)),
        ])
    } }
}

/* ==================== wide u32x8 / u32x16 ==================== */

impl u32x8 { #[inline] pub fn zero_ext(self) -> u64x8 {
    let a = self.0[0].zero_ext(); let b = self.0[1].zero_ext();
    u64x8([a.0[0], a.0[1], b.0[0], b.0[1]])
} }

// u32x16 has no zero_ext: zero-extending 16×u32 → 16×u64 needs a 1024-bit
// output type that does not exist and is not needed by Cherry.

/* ==================== 128-bit masks ==================== */

macro_rules! def_mask {
    ($name:ident, $vec:ident, $bitmask:ty) => {
        #[derive(Debug, Copy, Clone)]
        pub struct $name(pub $vec);
        impl From<$vec> for $name { #[inline] fn from(v: $vec) -> Self { Self(v) } }
        impl From<$bitmask> for $name { #[inline] fn from(v: $bitmask) -> Self { Self::expand(v) } }
        impl Not for $name { type Output = Self; #[inline] fn not(self) -> Self { Self(!self.0) } }
        impl BitAnd for $name { type Output = Self; #[inline] fn bitand(self, o: Self) -> Self { Self(self.0 & o.0) } }
        impl BitOr  for $name { type Output = Self; #[inline] fn bitor (self, o: Self) -> Self { Self(self.0 | o.0) } }
        impl BitXor for $name { type Output = Self; #[inline] fn bitxor(self, o: Self) -> Self { Self(self.0 ^ o.0) } }
        impl BitAndAssign for $name { #[inline] fn bitand_assign(&mut self, o: Self) { *self = *self & o; } }
        impl BitOrAssign  for $name { #[inline] fn bitor_assign (&mut self, o: Self) { *self = *self | o; } }
        impl BitXorAssign for $name { #[inline] fn bitxor_assign(&mut self, o: Self) { *self = *self ^ o; } }
        impl BitAnd<$bitmask> for $name { type Output = Self; #[inline] fn bitand(self, rhs: $bitmask) -> Self { self & Self::expand(rhs) } }
        impl BitOr <$bitmask> for $name { type Output = Self; #[inline] fn bitor (self, rhs: $bitmask) -> Self { self | Self::expand(rhs) } }
        impl BitXor<$bitmask> for $name { type Output = Self; #[inline] fn bitxor(self, rhs: $bitmask) -> Self { self ^ Self::expand(rhs) } }
    };
}
def_mask!(Mask8x16, u8x16, u16);
def_mask!(Mask16x8, u16x8, u8);
def_mask!(Mask32x4, u32x4, u8);
def_mask!(Mask64x2, u64x2, u8);

impl Mask8x16 {
    #[inline] pub fn to_bitmask(self) -> u16 { self.0.to_bitmask() }
    #[inline] pub fn widen(self) -> Mask16x16 {
        let v = self.0.zero_ext();
        let w = v | v.shl::<8>();
        Mask16x16([Mask16x8(w.0[0]), Mask16x8(w.0[1])])
    }
    #[inline] pub fn expand(bm: u16) -> Self {
        let mut b = [0u8; 16]; let mut i = 0;
        while i < 16 { if (bm >> i) & 1 != 0 { b[i] = 0xFF; } i += 1; }
        Mask8x16(u8x16::from(b))
    }
}
impl Mask16x8 {
    #[inline] pub fn to_bitmask(self) -> u8 { self.0.to_bitmask() }
    #[inline] pub fn widen(self) -> Mask32x8 {
        let v = self.0.zero_ext();
        let a = v.0[0] | v.0[0].shl::<16>();
        let b = v.0[1] | v.0[1].shl::<16>();
        Mask32x8([Mask32x4(a), Mask32x4(b)])
    }
    #[inline] pub fn expand(bm: u8) -> Self {
        let mut h = [0u16; 8]; let mut i = 0;
        while i < 8 { if (bm >> i) & 1 != 0 { h[i] = 0xFFFF; } i += 1; }
        Mask16x8(u16x8::from(h))
    }
}
impl Mask32x4 {
    #[inline] pub fn to_bitmask(self) -> u8 { self.0.to_bitmask() }
    #[inline] pub fn widen(self) -> Mask64x4 {
        let v = self.0.zero_ext();
        let a = v.0[0] | v.0[0].shl::<32>();
        let b = v.0[1] | v.0[1].shl::<32>();
        Mask64x4([Mask64x2(a), Mask64x2(b)])
    }
    #[inline] pub fn expand(bm: u8) -> Self {
        let mut w = [0u32; 4]; let mut i = 0;
        while i < 4 { if (bm >> i) & 1 != 0 { w[i] = 0xFFFF_FFFF; } i += 1; }
        Mask32x4(u32x4::from(w))
    }
}
impl Mask64x2 {
    #[inline] pub fn to_bitmask(self) -> u8 { self.0.to_bitmask() }
    #[inline] pub fn expand(bm: u8) -> Self {
        let mut w = [0u64; 2]; let mut i = 0;
        while i < 2 { if (bm >> i) & 1 != 0 { w[i] = 0xFFFF_FFFF_FFFF_FFFF; } i += 1; }
        Mask64x2(u64x2::from(w))
    }
}

/* ==================== wide masks ==================== */

macro_rules! def_wide_mask {
    ($name:ident, $inner:ident, $n:expr) => {
        #[derive(Debug, Copy, Clone)]
        pub struct $name(pub [$inner; $n]);
        impl From<[$inner; $n]> for $name { #[inline] fn from(a: [$inner; $n]) -> Self { Self(a) } }
        impl Not for $name { type Output = Self; #[inline] fn not(self) -> Self {
            let mut r = self.0; let mut i = 0; while i < $n { r[i] = !r[i]; i += 1; } Self(r)
        } }
        impl BitAnd for $name { type Output = Self; #[inline] fn bitand(self, o: Self) -> Self {
            let mut r = self.0; let mut i = 0; while i < $n { r[i] = r[i] & o.0[i]; i += 1; } Self(r)
        } }
        impl BitOr for $name { type Output = Self; #[inline] fn bitor(self, o: Self) -> Self {
            let mut r = self.0; let mut i = 0; while i < $n { r[i] = r[i] | o.0[i]; i += 1; } Self(r)
        } }
        impl BitXor for $name { type Output = Self; #[inline] fn bitxor(self, o: Self) -> Self {
            let mut r = self.0; let mut i = 0; while i < $n { r[i] = r[i] ^ o.0[i]; i += 1; } Self(r)
        } }
        impl BitAndAssign for $name { #[inline] fn bitand_assign(&mut self, o: Self) { *self = *self & o; } }
        impl BitOrAssign  for $name { #[inline] fn bitor_assign (&mut self, o: Self) { *self = *self | o; } }
        impl BitXorAssign for $name { #[inline] fn bitxor_assign(&mut self, o: Self) { *self = *self ^ o; } }
    };
}
def_wide_mask!(Mask8x32,  Mask8x16,  2);
def_wide_mask!(Mask8x64,  Mask8x16,  4);
def_wide_mask!(Mask16x16, Mask16x8,  2);
def_wide_mask!(Mask16x32, Mask16x8,  4);
def_wide_mask!(Mask32x8,  Mask32x4,  2);
def_wide_mask!(Mask32x16, Mask32x4,  4);
def_wide_mask!(Mask64x4,  Mask64x2,  2);
def_wide_mask!(Mask64x8,  Mask64x2,  4);

impl Mask8x32 {
    #[inline] pub fn to_bitmask(self) -> u32 { (self.0[0].to_bitmask() as u32) | ((self.0[1].to_bitmask() as u32) << 16) }
    #[inline] pub fn widen(self) -> Mask16x32 {
        let a = self.0[0].widen(); let b = self.0[1].widen();
        Mask16x32([a.0[0], a.0[1], b.0[0], b.0[1]])
    }
    #[inline] pub fn expand(bm: u32) -> Self { Mask8x32([Mask8x16::expand(bm as u16), Mask8x16::expand((bm >> 16) as u16)]) }
}
impl From<u32> for Mask8x32 { #[inline] fn from(v: u32) -> Self { Self::expand(v) } }
impl BitAnd<u32> for Mask8x32 { type Output = Self; #[inline] fn bitand(self, rhs: u32) -> Self { self & Self::expand(rhs) } }
impl BitOr <u32> for Mask8x32 { type Output = Self; #[inline] fn bitor (self, rhs: u32) -> Self { self | Self::expand(rhs) } }

impl Mask8x64 {
    #[inline] pub fn to_bitmask(self) -> u64 {
        (self.0[0].to_bitmask() as u64) | ((self.0[1].to_bitmask() as u64) << 16)
            | ((self.0[2].to_bitmask() as u64) << 32) | ((self.0[3].to_bitmask() as u64) << 48)
    }
    #[inline] pub fn widen(self) -> Mask16x64 {
        let a = self.0[0].widen(); let b = self.0[1].widen();
        let c = self.0[2].widen(); let d = self.0[3].widen();
        Mask16x64([a.0[0], a.0[1], b.0[0], b.0[1], c.0[0], c.0[1], d.0[0], d.0[1]])
    }
    #[inline] pub fn expand(bm: u64) -> Self {
        Mask8x64([Mask8x16::expand(bm as u16), Mask8x16::expand((bm >> 16) as u16),
                  Mask8x16::expand((bm >> 32) as u16), Mask8x16::expand((bm >> 48) as u16)])
    }
}
impl From<u64> for Mask8x64 { #[inline] fn from(v: u64) -> Self { Self::expand(v) } }
impl BitAnd<u64> for Mask8x64 { type Output = Self; #[inline] fn bitand(self, rhs: u64) -> Self { self & Self::expand(rhs) } }
impl BitOr <u64> for Mask8x64 { type Output = Self; #[inline] fn bitor (self, rhs: u64) -> Self { self | Self::expand(rhs) } }
impl BitXor<u64> for Mask8x64 { type Output = Self; #[inline] fn bitxor(self, rhs: u64) -> Self { self ^ Self::expand(rhs) } }

impl Mask16x16 {
    #[inline] pub fn to_bitmask(self) -> u16 { (self.0[0].to_bitmask() as u16) | ((self.0[1].to_bitmask() as u16) << 8) }
    #[inline] pub fn widen(self) -> Mask32x16 {
        let a = self.0[0].widen(); let b = self.0[1].widen();
        Mask32x16([a.0[0], a.0[1], b.0[0], b.0[1]])
    }
    #[inline] pub fn expand(bm: u16) -> Self { Mask16x16([Mask16x8::expand(bm as u8), Mask16x8::expand((bm >> 8) as u8)]) }
}
impl From<u16> for Mask16x16 { #[inline] fn from(v: u16) -> Self { Self::expand(v) } }
impl BitAnd<u16> for Mask16x16 { type Output = Self; #[inline] fn bitand(self, rhs: u16) -> Self { self & Self::expand(rhs) } }
impl BitOr <u16> for Mask16x16 { type Output = Self; #[inline] fn bitor (self, rhs: u16) -> Self { self | Self::expand(rhs) } }

impl Mask16x32 {
    #[inline] pub fn to_bitmask(self) -> u32 {
        (self.0[0].to_bitmask() as u32) | ((self.0[1].to_bitmask() as u32) << 8)
            | ((self.0[2].to_bitmask() as u32) << 16) | ((self.0[3].to_bitmask() as u32) << 24)
    }
    #[inline] pub fn expand(bm: u32) -> Self {
        Mask16x32([Mask16x8::expand(bm as u8), Mask16x8::expand((bm >> 8) as u8),
                   Mask16x8::expand((bm >> 16) as u8), Mask16x8::expand((bm >> 24) as u8)])
    }
}
impl From<u32> for Mask16x32 { #[inline] fn from(v: u32) -> Self { Self::expand(v) } }
impl BitAnd<u32> for Mask16x32 { type Output = Self; #[inline] fn bitand(self, rhs: u32) -> Self { self & Self::expand(rhs) } }
impl BitOr <u32> for Mask16x32 { type Output = Self; #[inline] fn bitor (self, rhs: u32) -> Self { self | Self::expand(rhs) } }

impl Mask32x8 {
    #[inline] pub fn to_bitmask(self) -> u8 { (self.0[0].to_bitmask() as u8) | ((self.0[1].to_bitmask() as u8) << 4) }
    #[inline] pub fn widen(self) -> Mask64x8 {
        let a = self.0[0].widen(); let b = self.0[1].widen();
        Mask64x8([a.0[0], a.0[1], b.0[0], b.0[1]])
    }
    #[inline] pub fn expand(bm: u8) -> Self { Mask32x8([Mask32x4::expand(bm & 0xF), Mask32x4::expand((bm >> 4) & 0xF)]) }
}
impl From<u8> for Mask32x8 { #[inline] fn from(v: u8) -> Self { Self::expand(v) } }
impl BitAnd<u8> for Mask32x8 { type Output = Self; #[inline] fn bitand(self, rhs: u8) -> Self { self & Self::expand(rhs) } }
impl BitOr <u8> for Mask32x8 { type Output = Self; #[inline] fn bitor (self, rhs: u8) -> Self { self | Self::expand(rhs) } }

impl Mask32x16 {
    #[inline] pub fn to_bitmask(self) -> u16 {
        (self.0[0].to_bitmask() as u16) | ((self.0[1].to_bitmask() as u16) << 4)
            | ((self.0[2].to_bitmask() as u16) << 8) | ((self.0[3].to_bitmask() as u16) << 12)
    }
    #[inline] pub fn expand(bm: u16) -> Self {
        Mask32x16([Mask32x4::expand((bm & 0xF) as u8), Mask32x4::expand(((bm >> 4) & 0xF) as u8),
                   Mask32x4::expand(((bm >> 8) & 0xF) as u8), Mask32x4::expand(((bm >> 12) & 0xF) as u8)])
    }
}
impl From<u16> for Mask32x16 { #[inline] fn from(v: u16) -> Self { Self::expand(v) } }
impl BitAnd<u16> for Mask32x16 { type Output = Self; #[inline] fn bitand(self, rhs: u16) -> Self { self & Self::expand(rhs) } }
impl BitOr <u16> for Mask32x16 { type Output = Self; #[inline] fn bitor (self, rhs: u16) -> Self { self | Self::expand(rhs) } }

impl Mask64x4 { #[inline] pub fn to_bitmask(self) -> u8 {
    (self.0[0].to_bitmask() as u8) | ((self.0[1].to_bitmask() as u8) << 2)
} }
impl Mask64x8 { #[inline] pub fn to_bitmask(self) -> u8 {
    (self.0[0].to_bitmask() as u8) | ((self.0[1].to_bitmask() as u8) << 2)
        | ((self.0[2].to_bitmask() as u8) << 4) | ((self.0[3].to_bitmask() as u8) << 6)
} }

/* ==================== Mask16x64 ==================== */

#[derive(Debug, Copy, Clone)]
pub struct Mask16x64(pub [Mask16x8; 8]);
impl From<[Mask16x8; 8]> for Mask16x64 { #[inline] fn from(a: [Mask16x8; 8]) -> Self { Self(a) } }
impl Not for Mask16x64 { type Output = Self; #[inline] fn not(self) -> Self {
    let mut r = self.0; let mut i = 0; while i < 8 { r[i] = !r[i]; i += 1; } Self(r)
} }
impl BitAnd for Mask16x64 { type Output = Self; #[inline] fn bitand(self, o: Self) -> Self {
    let mut r = self.0; let mut i = 0; while i < 8 { r[i] = r[i] & o.0[i]; i += 1; } Self(r)
} }
impl BitOr for Mask16x64 { type Output = Self; #[inline] fn bitor(self, o: Self) -> Self {
    let mut r = self.0; let mut i = 0; while i < 8 { r[i] = r[i] | o.0[i]; i += 1; } Self(r)
} }
impl BitXor for Mask16x64 { type Output = Self; #[inline] fn bitxor(self, o: Self) -> Self {
    let mut r = self.0; let mut i = 0; while i < 8 { r[i] = r[i] ^ o.0[i]; i += 1; } Self(r)
} }
impl BitAndAssign for Mask16x64 { #[inline] fn bitand_assign(&mut self, o: Self) { *self = *self & o; } }
impl BitOrAssign  for Mask16x64 { #[inline] fn bitor_assign (&mut self, o: Self) { *self = *self | o; } }
impl BitXorAssign for Mask16x64 { #[inline] fn bitxor_assign(&mut self, o: Self) { *self = *self ^ o; } }
impl Mask16x64 {
    #[inline] pub fn to_bitmask(self) -> u64 {
        let mut o: u64 = 0; let mut i = 0;
        while i < 8 { o |= (self.0[i].to_bitmask() as u64) << (i * 8); i += 1; }
        o
    }
    #[inline] pub fn expand(bm: u64) -> Self {
        let mut m = [Mask16x8::expand(0); 8]; let mut i = 0;
        while i < 8 { m[i] = Mask16x8::expand(((bm >> (i * 8)) & 0xFF) as u8); i += 1; }
        Self(m)
    }
}
impl From<u64> for Mask16x64 { #[inline] fn from(bm: u64) -> Self { Self::expand(bm) } }
impl BitAnd<u64> for Mask16x64 { type Output = Self; #[inline] fn bitand(self, rhs: u64) -> Self { self & Self::expand(rhs) } }
impl BitOr <u64> for Mask16x64 { type Output = Self; #[inline] fn bitor (self, rhs: u64) -> Self { self | Self::expand(rhs) } }

/* ==================== u16x64 ==================== */

#[derive(Debug, Copy, Clone)]
pub struct u16x64(pub [u16x8; 8]);
impl u16x64 {
    #[inline] pub unsafe fn load<T>(p: *const T) -> Self { unsafe {
        let mut a: [u16x8; 8] = [u16x8::splat(0); 8];
        let mut i = 0;
        while i < 8 { a[i] = u16x8::load(p.cast::<u8>().add(i * 16)); i += 1; }
        Self(a)
    } }
    #[inline] pub unsafe fn store<T>(self, p: *mut T) { unsafe {
        let mut i = 0;
        while i < 8 { self.0[i].store(p.cast::<u8>().add(i * 16)); i += 1; }
    } }
    #[inline] pub fn splat(v: u16) -> Self { Self([u16x8::splat(v); 8]) }
    #[inline] pub fn eq(a: Self, b: Self) -> Mask16x64 { Mask16x64([
        u16x8::eq(a.0[0], b.0[0]), u16x8::eq(a.0[1], b.0[1]),
        u16x8::eq(a.0[2], b.0[2]), u16x8::eq(a.0[3], b.0[3]),
        u16x8::eq(a.0[4], b.0[4]), u16x8::eq(a.0[5], b.0[5]),
        u16x8::eq(a.0[6], b.0[6]), u16x8::eq(a.0[7], b.0[7]),
    ]) }
    #[inline] pub fn neq(a: Self, b: Self) -> Mask16x64 { Self::eq(a, b).not() }
    #[inline] pub fn test(a: Self, b: Self) -> Mask16x64 { (a & b).nonzero() }
    #[inline] pub fn testn(a: Self, b: Self) -> Mask16x64 { (a & b).zero() }
    #[inline] pub fn zero(self) -> Mask16x64 { Self::eq(self, Self::splat(0)) }
    #[inline] pub fn nonzero(self) -> Mask16x64 { Self::neq(self, Self::splat(0)) }
    #[inline] pub fn msb(self) -> Mask16x64 { Mask16x64([
        self.0[0].msb(), self.0[1].msb(), self.0[2].msb(), self.0[3].msb(),
        self.0[4].msb(), self.0[5].msb(), self.0[6].msb(), self.0[7].msb(),
    ]) }
    #[inline] pub fn to_bitmask(self) -> u64 {
        let mut o: u64 = 0; let mut i = 0;
        while i < 8 { o |= (self.0[i].to_bitmask() as u64) << (i * 8); i += 1; }
        o
    }
    #[inline] pub fn mask(self, m: Mask16x64) -> Self {
        let mut r = self.0; let mut i = 0;
        while i < 8 { r[i] = r[i] & m.0[i].0; i += 1; }
        u16x64(r)
    }
    #[inline] pub fn blend(a: Self, b: Self, m: Mask16x64) -> Self {
        let mut r = a.0; let mut i = 0;
        while i < 8 { r[i] = (m.0[i].0 & b.0[i]) | m.0[i].0.andnot(a.0[i]); i += 1; }
        u16x64(r)
    }
    #[inline] pub fn shl<const N: i32>(self) -> Self {
        let mut r = self.0; let mut i = 0; while i < 8 { r[i] = self.0[i].shl::<N>(); i += 1; } Self(r)
    }
    #[inline] pub fn shr<const N: i32>(self) -> Self {
        let mut r = self.0; let mut i = 0; while i < 8 { r[i] = self.0[i].shr::<N>(); i += 1; } Self(r)
    }
    #[inline] pub fn shlv(self, s: Self) -> Self {
        let mut r = self.0; let mut i = 0; while i < 8 { r[i] = self.0[i].shlv(s.0[i]); i += 1; } Self(r)
    }
    #[inline] pub fn shrv(self, s: Self) -> Self {
        let mut r = self.0; let mut i = 0; while i < 8 { r[i] = self.0[i].shrv(s.0[i]); i += 1; } Self(r)
    }
    #[inline] pub fn extract16<const I: usize>(self) -> u16x16 { u16x16([self.0[I*2], self.0[I*2+1]]) }
    #[inline] pub fn extract32<const I: usize>(self) -> u16x32 {
        u16x32([self.0[I*4], self.0[I*4+1], self.0[I*4+2], self.0[I*4+3]])
    }
    #[inline] pub fn compress(self, m: Mask16x64) -> Self { unsafe {
        let mask = m.to_bitmask();
        let flat: [u16; 64] = mem::transmute(self.0);
        let mut out = [0u16; 64]; let mut c = 0;
        for i in 0..64 { if (mask >> i) & 1 != 0 { out[c] = flat[i]; c += 1; } }
        Self::load(out.as_ptr())
    } }
    #[inline] pub unsafe fn compress_store<T>(self, m: Mask16x64, p: *mut T) { unsafe { self.compress(m).store(p) } }
}
impl From<[u16; 64]> for u16x64 { #[inline] fn from(a: [u16; 64]) -> Self { unsafe { Self::load(a.as_ptr()) } } }
impl From<[u16x8; 8]> for u16x64 { #[inline] fn from(a: [u16x8; 8]) -> Self { Self(a) } }
impl Not for u16x64 { type Output = Self; #[inline] fn not(self) -> Self {
    let mut r = self.0; let mut i = 0; while i < 8 { r[i] = !self.0[i]; i += 1; } Self(r)
} }
impl BitAnd for u16x64 { type Output = Self; #[inline] fn bitand(self, o: Self) -> Self {
    let mut r = self.0; let mut i = 0; while i < 8 { r[i] = self.0[i] & o.0[i]; i += 1; } Self(r)
} }
impl BitOr for u16x64 { type Output = Self; #[inline] fn bitor(self, o: Self) -> Self {
    let mut r = self.0; let mut i = 0; while i < 8 { r[i] = self.0[i] | o.0[i]; i += 1; } Self(r)
} }
impl BitXor for u16x64 { type Output = Self; #[inline] fn bitxor(self, o: Self) -> Self {
    let mut r = self.0; let mut i = 0; while i < 8 { r[i] = self.0[i] ^ o.0[i]; i += 1; } Self(r)
} }
impl BitAndAssign for u16x64 { #[inline] fn bitand_assign(&mut self, o: Self) { *self = *self & o; } }
impl BitOrAssign  for u16x64 { #[inline] fn bitor_assign (&mut self, o: Self) { *self = *self | o; } }
impl BitXorAssign for u16x64 { #[inline] fn bitxor_assign(&mut self, o: Self) { *self = *self ^ o; } }

/* ==================== signed ==================== */

#[derive(Debug, Copy, Clone)]
pub struct i16x32(pub [int16x8_t; 4]);
impl i16x32 {
    #[inline] pub unsafe fn load<T>(p: *const T) -> Self { unsafe {
        let mut a: [int16x8_t; 4] = [vdupq_n_s16(0); 4];
        let mut i = 0;
        while i < 4 { a[i] = vld1q_s16(p.cast::<i16>().add(i * 8)); i += 1; }
        Self(a)
    } }
    #[inline] pub unsafe fn store<T>(self, p: *mut T) { unsafe {
        let mut i = 0;
        while i < 4 { vst1q_s16(p.cast::<i16>().add(i * 8), self.0[i]); i += 1; }
    } }
    #[inline] pub fn splat(v: i16) -> Self { unsafe { Self([vdupq_n_s16(v); 4]) } }
    #[inline] pub fn clamp(self, lo: Self, hi: Self) -> Self { unsafe {
        let mut r: [int16x8_t; 4] = [vdupq_n_s16(0); 4];
        let mut i = 0;
        while i < 4 { r[i] = vminq_s16(vmaxq_s16(self.0[i], lo.0[i]), hi.0[i]); i += 1; }
        Self(r)
    } }
    /// Matches AVX2's `_mm256_madd_epi16`: for each 128-bit half, computes
    /// `r[k] = self[2k]*rhs[2k] + self[2k+1]*rhs[2k+1]` as int32.
    #[inline] pub fn madd(self, rhs: Self) -> i32x16 { unsafe {
        let mut r: [int32x4_t; 4] = [vdupq_n_s32(0); 4];
        let mut i = 0;
        while i < 4 {
            let lo_a = vget_low_s16(self.0[i]);
            let lo_b = vget_low_s16(rhs.0[i]);
            let hi_a = vget_high_s16(self.0[i]);
            let hi_b = vget_high_s16(rhs.0[i]);
            let pl = vmull_s16(lo_a, lo_b);
            let ph = vmull_s16(hi_a, hi_b);
            r[i] = vpaddq_s32(pl, ph);
            i += 1;
        }
        i32x16(r)
    } }
}
impl Add for i16x32 { type Output = Self; #[inline] fn add(self, o: Self) -> Self { unsafe {
    let mut r: [int16x8_t; 4] = [vdupq_n_s16(0); 4];
    let mut i = 0; while i < 4 { r[i] = vaddq_s16(self.0[i], o.0[i]); i += 1; } Self(r)
} } }
impl Sub for i16x32 { type Output = Self; #[inline] fn sub(self, o: Self) -> Self { unsafe {
    let mut r: [int16x8_t; 4] = [vdupq_n_s16(0); 4];
    let mut i = 0; while i < 4 { r[i] = vsubq_s16(self.0[i], o.0[i]); i += 1; } Self(r)
} } }
impl Mul for i16x32 { type Output = Self; #[inline] fn mul(self, o: Self) -> Self { unsafe {
    let mut r: [int16x8_t; 4] = [vdupq_n_s16(0); 4];
    let mut i = 0; while i < 4 { r[i] = vmulq_s16(self.0[i], o.0[i]); i += 1; } Self(r)
} } }
impl AddAssign for i16x32 { #[inline] fn add_assign(&mut self, o: Self) { *self = *self + o; } }
impl SubAssign for i16x32 { #[inline] fn sub_assign(&mut self, o: Self) { *self = *self - o; } }
impl Default for i16x32 { #[inline] fn default() -> Self { unsafe { Self([vdupq_n_s16(0); 4]) } } }

#[derive(Debug, Copy, Clone)]
pub struct i32x16(pub [int32x4_t; 4]);
impl i32x16 {
    #[inline] pub unsafe fn load<T>(p: *const T) -> Self { unsafe {
        let mut a: [int32x4_t; 4] = [vdupq_n_s32(0); 4];
        let mut i = 0;
        while i < 4 { a[i] = vld1q_s32(p.cast::<i32>().add(i * 4)); i += 1; }
        Self(a)
    } }
    #[inline] pub unsafe fn store<T>(self, p: *mut T) { unsafe {
        let mut i = 0;
        while i < 4 { vst1q_s32(p.cast::<i32>().add(i * 4), self.0[i]); i += 1; }
    } }
    #[inline] pub fn splat(v: i32) -> Self { unsafe { Self([vdupq_n_s32(v); 4]) } }
    #[inline] pub fn reduce_sum(self) -> i32 { unsafe {
        let mut s = vaddvq_s32(self.0[0]);
        s = s.wrapping_add(vaddvq_s32(self.0[1]));
        s = s.wrapping_add(vaddvq_s32(self.0[2]));
        s = s.wrapping_add(vaddvq_s32(self.0[3]));
        s
    } }
}
impl Add for i32x16 { type Output = Self; #[inline] fn add(self, o: Self) -> Self { unsafe {
    let mut r: [int32x4_t; 4] = [vdupq_n_s32(0); 4];
    let mut i = 0; while i < 4 { r[i] = vaddq_s32(self.0[i], o.0[i]); i += 1; } Self(r)
} } }
impl Sub for i32x16 { type Output = Self; #[inline] fn sub(self, o: Self) -> Self { unsafe {
    let mut r: [int32x4_t; 4] = [vdupq_n_s32(0); 4];
    let mut i = 0; while i < 4 { r[i] = vsubq_s32(self.0[i], o.0[i]); i += 1; } Self(r)
} } }
impl AddAssign for i32x16 { #[inline] fn add_assign(&mut self, o: Self) { *self = *self + o; } }
impl SubAssign for i32x16 { #[inline] fn sub_assign(&mut self, o: Self) { *self = *self - o; } }
impl Default for i32x16 { #[inline] fn default() -> Self { unsafe { Self([vdupq_n_s32(0); 4]) } } }
