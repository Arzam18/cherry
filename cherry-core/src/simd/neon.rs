use core::{arch::aarch64::*, ops::*};

/*----------------------------------------------------------------*/

macro_rules! impl_mask {
    ($name:ident, $bits:ty) => {
        #[derive(Debug, Copy, Clone, PartialEq, Eq)]
        pub struct $name($bits);

        impl From<$bits> for $name {
            #[inline]
            fn from(bits: $bits) -> Self {
                Self(bits)
            }
        }

        impl $name {
            #[inline]
            pub fn to_bitmask(self) -> $bits {
                self.0
            }
        }

        impl Not for $name {
            type Output = Self;

            #[inline]
            fn not(self) -> Self::Output {
                Self(!self.0)
            }
        }

        impl BitAnd for $name {
            type Output = Self;

            #[inline]
            fn bitand(self, rhs: Self) -> Self::Output {
                Self(self.0 & rhs.0)
            }
        }

        impl BitOr for $name {
            type Output = Self;

            #[inline]
            fn bitor(self, rhs: Self) -> Self::Output {
                Self(self.0 | rhs.0)
            }
        }

        impl BitXor for $name {
            type Output = Self;

            #[inline]
            fn bitxor(self, rhs: Self) -> Self::Output {
                Self(self.0 ^ rhs.0)
            }
        }

        impl BitAndAssign for $name {
            #[inline]
            fn bitand_assign(&mut self, rhs: Self) {
                self.0 &= rhs.0;
            }
        }

        impl BitOrAssign for $name {
            #[inline]
            fn bitor_assign(&mut self, rhs: Self) {
                self.0 |= rhs.0;
            }
        }

        impl BitXorAssign for $name {
            #[inline]
            fn bitxor_assign(&mut self, rhs: Self) {
                self.0 ^= rhs.0;
            }
        }
    };
}

impl_mask!(Mask8x16, u16);
impl_mask!(Mask8x64, u64);
impl_mask!(Mask16x64, u64);

impl BitAnd<u64> for Mask8x64 {
    type Output = Self;

    #[inline]
    fn bitand(self, rhs: u64) -> Self::Output {
        Self(self.0 & rhs)
    }
}

impl BitOr<u64> for Mask8x64 {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: u64) -> Self::Output {
        Self(self.0 | rhs)
    }
}

impl BitXor<u64> for Mask8x64 {
    type Output = Self;

    #[inline]
    fn bitxor(self, rhs: u64) -> Self::Output {
        Self(self.0 ^ rhs)
    }
}

impl BitAndAssign<u64> for Mask8x64 {
    #[inline]
    fn bitand_assign(&mut self, rhs: u64) {
        self.0 &= rhs;
    }
}

impl BitOrAssign<u64> for Mask8x64 {
    #[inline]
    fn bitor_assign(&mut self, rhs: u64) {
        self.0 |= rhs;
    }
}

impl BitXorAssign<u64> for Mask8x64 {
    #[inline]
    fn bitxor_assign(&mut self, rhs: u64) {
        self.0 ^= rhs;
    }
}

impl Mask8x64 {
    #[inline]
    pub fn widen(self) -> Mask16x64 {
        Mask16x64(self.0)
    }
}

/*----------------------------------------------------------------*/

#[repr(transparent)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct u8x16([u8; 16]);

impl From<[u8; 16]> for u8x16 {
    #[inline]
    fn from(value: [u8; 16]) -> Self {
        Self(value)
    }
}

impl u8x16 {
    #[inline]
    pub unsafe fn load<T>(src: *const T) -> Self {
        let mut value = [0u8; 16];
        unsafe {
            let loaded = vld1q_u8(src.cast::<u8>());
            vst1q_u8(value.as_mut_ptr(), loaded);
        }
        Self(value)
    }

    #[inline]
    pub unsafe fn store<T>(self, dest: *mut T) {
        unsafe { vst1q_u8(dest.cast::<u8>(), vld1q_u8(self.0.as_ptr())) }
    }

    #[inline]
    pub fn splat(value: u8) -> Self {
        let mut out = [0u8; 16];
        unsafe { vst1q_u8(out.as_mut_ptr(), vdupq_n_u8(value)) };
        Self(out)
    }

    #[inline]
    pub fn extract<const INDEX: i32>(self) -> u8 {
        self.0[INDEX as usize]
    }

    #[inline]
    pub fn broadcast64(self) -> u8x64 {
        let mut out = [0u8; 64];
        for chunk in out.chunks_exact_mut(16) {
            chunk.copy_from_slice(&self.0);
        }
        u8x64(out)
    }

    #[inline]
    pub fn findset(self, needles: u8x16, needle_count: usize) -> u16 {
        let count = needle_count.min(16);
        let mut result = 0u16;
        for (index, value) in self.0.iter().enumerate() {
            if needles.0[..count].contains(value) {
                result |= 1u16 << index;
            }
        }
        result
    }

    #[inline]
    pub fn mask(self, mask: Mask8x16) -> Self {
        let mut out = [0u8; 16];
        for (index, value) in out.iter_mut().enumerate() {
            if (mask.0 >> index) & 1 != 0 {
                *value = self.0[index];
            }
        }
        Self(out)
    }

    #[inline]
    pub fn eq(a: Self, b: Self) -> Mask8x16 {
        let mut lanes = [0u8; 16];
        unsafe {
            let av = vld1q_u8(a.0.as_ptr());
            let bv = vld1q_u8(b.0.as_ptr());
            vst1q_u8(lanes.as_mut_ptr(), vceqq_u8(av, bv));
        }
        let mut bits = 0u16;
        for (index, lane) in lanes.iter().enumerate() {
            bits |= u16::from(*lane != 0) << index;
        }
        Mask8x16(bits)
    }

    #[inline]
    pub fn neq(a: Self, b: Self) -> Mask8x16 {
        !Self::eq(a, b)
    }

    #[inline]
    pub fn nonzero(self) -> Mask8x16 {
        Self::neq(self, Self::splat(0))
    }

    #[inline]
    pub fn zero(self) -> Mask8x16 {
        Self::eq(self, Self::splat(0))
    }

    #[inline]
    pub fn to_u64x2(self) -> u64x2 {
        u64x2(unsafe { core::mem::transmute(self.0) })
    }
}

macro_rules! impl_u8_ops {
    ($name:ident, $width:expr) => {
        impl BitAnd for $name {
            type Output = Self;
            #[inline]
            fn bitand(self, rhs: Self) -> Self::Output {
                let mut out = [0u8; $width];
                let mut offset = 0;
                unsafe {
                    while offset < $width {
                        let a = vld1q_u8(self.0.as_ptr().add(offset));
                        let b = vld1q_u8(rhs.0.as_ptr().add(offset));
                        vst1q_u8(out.as_mut_ptr().add(offset), vandq_u8(a, b));
                        offset += 16;
                    }
                }
                Self(out)
            }
        }

        impl BitOr for $name {
            type Output = Self;
            #[inline]
            fn bitor(self, rhs: Self) -> Self::Output {
                let mut out = [0u8; $width];
                let mut offset = 0;
                unsafe {
                    while offset < $width {
                        let a = vld1q_u8(self.0.as_ptr().add(offset));
                        let b = vld1q_u8(rhs.0.as_ptr().add(offset));
                        vst1q_u8(out.as_mut_ptr().add(offset), vorrq_u8(a, b));
                        offset += 16;
                    }
                }
                Self(out)
            }
        }

        impl BitXor for $name {
            type Output = Self;
            #[inline]
            fn bitxor(self, rhs: Self) -> Self::Output {
                let mut out = [0u8; $width];
                let mut offset = 0;
                unsafe {
                    while offset < $width {
                        let a = vld1q_u8(self.0.as_ptr().add(offset));
                        let b = vld1q_u8(rhs.0.as_ptr().add(offset));
                        vst1q_u8(out.as_mut_ptr().add(offset), veorq_u8(a, b));
                        offset += 16;
                    }
                }
                Self(out)
            }
        }

        impl Not for $name {
            type Output = Self;
            #[inline]
            fn not(self) -> Self::Output {
                let mut out = [0u8; $width];
                let mut offset = 0;
                unsafe {
                    while offset < $width {
                        let value = vld1q_u8(self.0.as_ptr().add(offset));
                        vst1q_u8(out.as_mut_ptr().add(offset), vmvnq_u8(value));
                        offset += 16;
                    }
                }
                Self(out)
            }
        }

        impl BitAndAssign for $name {
            #[inline]
            fn bitand_assign(&mut self, rhs: Self) { *self = *self & rhs; }
        }
        impl BitOrAssign for $name {
            #[inline]
            fn bitor_assign(&mut self, rhs: Self) { *self = *self | rhs; }
        }
        impl BitXorAssign for $name {
            #[inline]
            fn bitxor_assign(&mut self, rhs: Self) { *self = *self ^ rhs; }
        }
    };
}

impl_u8_ops!(u8x16, 16);

/*----------------------------------------------------------------*/

#[repr(transparent)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct u64x2([u64; 2]);

impl u64x2 {
    #[inline]
    pub fn splat(value: u64) -> Self {
        Self([value; 2])
    }

    #[inline]
    pub fn to_u8x16(self) -> u8x16 {
        u8x16(unsafe { core::mem::transmute(self.0) })
    }

    #[inline]
    pub fn extract<const INDEX: i32>(self) -> u64 {
        self.0[INDEX as usize]
    }

    #[inline]
    pub fn mask(self, mask: Mask8x16) -> Self {
        self.to_u8x16().mask(mask).to_u64x2()
    }
}

/*----------------------------------------------------------------*/

/// Eight 64-bit lanes, represented as four 128-bit NEON vectors.
///
/// Cherry uses this vector in static-exchange evaluation to select the first
/// piece class whose ray bitboard intersects the current attackers.
#[repr(transparent)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct u64x8([u64; 8]);

/// Eight-lane mask returned by `u64x8::nonzero`. Bit `n` corresponds to lane `n`.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Mask64x8(u8);

impl Mask64x8 {
    #[inline]
    pub fn to_bitmask(self) -> u8 {
        self.0
    }
}

impl From<[u64; 8]> for u64x8 {
    #[inline]
    fn from(value: [u64; 8]) -> Self {
        Self(value)
    }
}

impl u64x8 {
    #[inline]
    pub fn splat(value: u64) -> Self {
        let mut out = [0u64; 8];
        unsafe {
            let lanes = vdupq_n_u64(value);
            for offset in (0..8).step_by(2) {
                vst1q_u64(out.as_mut_ptr().add(offset), lanes);
            }
        }
        Self(out)
    }

    #[inline]
    pub fn nonzero(self) -> Mask64x8 {
        let mut bits = 0u8;
        unsafe {
            let zero = vdupq_n_u64(0);
            for offset in (0..8).step_by(2) {
                let lanes = vld1q_u64(self.0.as_ptr().add(offset));
                let equal = vceqq_u64(lanes, zero);
                let mut compared = [0u64; 2];
                vst1q_u64(compared.as_mut_ptr(), equal);

                if compared[0] == 0 {
                    bits |= 1u8 << offset;
                }
                if compared[1] == 0 {
                    bits |= 1u8 << (offset + 1);
                }
            }
        }
        Mask64x8(bits)
    }
}

impl BitAnd for u64x8 {
    type Output = Self;

    #[inline]
    fn bitand(self, rhs: Self) -> Self::Output {
        let mut out = [0u64; 8];
        unsafe {
            for offset in (0..8).step_by(2) {
                let lhs = vld1q_u64(self.0.as_ptr().add(offset));
                let rhs = vld1q_u64(rhs.0.as_ptr().add(offset));
                vst1q_u64(out.as_mut_ptr().add(offset), vandq_u64(lhs, rhs));
            }
        }
        Self(out)
    }
}

impl BitAndAssign for u64x8 {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}

/*----------------------------------------------------------------*/

#[repr(transparent)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct u8x64([u8; 64]);

impl From<[u8; 64]> for u8x64 {
    #[inline]
    fn from(value: [u8; 64]) -> Self { Self(value) }
}

impl_u8_ops!(u8x64, 64);

fn bits_from_u8_lanes(lanes: &[u8; 64]) -> u64 {
    let mut bits = 0u64;
    for (index, lane) in lanes.iter().enumerate() {
        if *lane != 0 { bits |= 1u64 << index; }
    }
    bits
}

impl u8x64 {
    #[inline]
    pub fn splat(value: u8) -> Self {
        let mut out = [0u8; 64];
        for offset in (0..64).step_by(16) {
            unsafe { vst1q_u8(out.as_mut_ptr().add(offset), vdupq_n_u8(value)); }
        }
        Self(out)
    }

    #[inline]
    pub unsafe fn load<T>(src: *const T) -> Self {
        let mut out = [0u8; 64];
        for offset in (0..64).step_by(16) {
            unsafe {
                let value = vld1q_u8(src.cast::<u8>().add(offset));
                vst1q_u8(out.as_mut_ptr().add(offset), value);
            }
        }
        Self(out)
    }

    #[inline]
    pub unsafe fn store<T>(self, dest: *mut T) {
        for offset in (0..64).step_by(16) {
            unsafe {
                let value = vld1q_u8(self.0.as_ptr().add(offset));
                vst1q_u8(dest.cast::<u8>().add(offset), value);
            }
        }
    }

    #[inline]
    pub fn eq(a: Self, b: Self) -> Mask8x64 {
        let mut lanes = [0u8; 64];
        for offset in (0..64).step_by(16) {
            unsafe {
                let av = vld1q_u8(a.0.as_ptr().add(offset));
                let bv = vld1q_u8(b.0.as_ptr().add(offset));
                vst1q_u8(lanes.as_mut_ptr().add(offset), vceqq_u8(av, bv));
            }
        }
        Mask8x64(bits_from_u8_lanes(&lanes))
    }

    #[inline]
    pub fn neq(a: Self, b: Self) -> Mask8x64 { !Self::eq(a, b) }

    #[inline]
    pub fn zero(self) -> Mask8x64 { Self::eq(self, Self::splat(0)) }

    #[inline]
    pub fn nonzero(self) -> Mask8x64 { Self::neq(self, Self::splat(0)) }

    #[inline]
    pub fn msb(self) -> Mask8x64 {
        let mut lanes = [0u8; 64];
        for offset in (0..64).step_by(16) {
            unsafe {
                let value = vld1q_u8(self.0.as_ptr().add(offset));
                vst1q_u8(lanes.as_mut_ptr().add(offset), vshrq_n_u8::<7>(value));
            }
        }
        Mask8x64(bits_from_u8_lanes(&lanes))
    }

    #[inline]
    pub fn mask(self, mask: Mask8x64) -> Self {
        let mut mask_lanes = [0u8; 64];
        for (index, lane) in mask_lanes.iter_mut().enumerate() {
            *lane = if (mask.0 >> index) & 1 != 0 { u8::MAX } else { 0 };
        }
        let mut out = [0u8; 64];
        for offset in (0..64).step_by(16) {
            unsafe {
                let value = vld1q_u8(self.0.as_ptr().add(offset));
                let bits = vld1q_u8(mask_lanes.as_ptr().add(offset));
                vst1q_u8(out.as_mut_ptr().add(offset), vandq_u8(value, bits));
            }
        }
        Self(out)
    }

    #[inline]
    pub fn blend(a: Self, b: Self, mask: Mask8x64) -> Self {
        let mut mask_lanes = [0u8; 64];
        for (index, lane) in mask_lanes.iter_mut().enumerate() {
            *lane = if (mask.0 >> index) & 1 != 0 { u8::MAX } else { 0 };
        }
        let mut out = [0u8; 64];
        for offset in (0..64).step_by(16) {
            unsafe {
                let av = vld1q_u8(a.0.as_ptr().add(offset));
                let bv = vld1q_u8(b.0.as_ptr().add(offset));
                let mv = vld1q_u8(mask_lanes.as_ptr().add(offset));
                vst1q_u8(out.as_mut_ptr().add(offset), vbslq_u8(mv, bv, av));
            }
        }
        Self(out)
    }

    #[inline]
    pub fn compress(self, mask: Mask8x64) -> Self {
        let mut out = [0u8; 64];
        let mut count = 0usize;
        let mut bits = mask.0;
        while bits != 0 {
            let index = bits.trailing_zeros() as usize;
            out[count] = self.0[index];
            count += 1;
            bits &= bits - 1;
        }
        Self(out)
    }

    #[inline]
    pub unsafe fn compress_store<T>(self, mask: Mask8x64, dest: *mut T) {
        unsafe { self.compress(mask).store(dest); }
    }

    #[inline]
    pub fn extract16<const INDEX: usize>(self) -> u8x16 {
        let start = INDEX * 16;
        assert!(start + 16 <= 64);
        let mut out = [0u8; 16];
        out.copy_from_slice(&self.0[start..start + 16]);
        u8x16(out)
    }

    #[inline]
    pub fn zero_ext(self) -> u16x64 {
        let mut out = [0u16; 64];
        for offset in (0..64).step_by(16) {
            unsafe {
                let bytes = vld1q_u8(self.0.as_ptr().add(offset));
                vst1q_u16(out.as_mut_ptr().add(offset), vmovl_u8(vget_low_u8(bytes)));
                vst1q_u16(out.as_mut_ptr().add(offset + 8), vmovl_u8(vget_high_u8(bytes)));
            }
        }
        u16x64(out)
    }

    #[inline]
    pub fn to_u16x32(self) -> u16x32 {
        u16x32(unsafe { core::mem::transmute(self.0) })
    }

    #[inline]
    pub fn flip_rays(self) -> Self {
        let mut out = [0u8; 64];
        out[..32].copy_from_slice(&self.0[32..]);
        out[32..].copy_from_slice(&self.0[..32]);
        Self(out)
    }

    #[inline]
    pub fn extend_rays(self) -> Self {
        let mut out = [0u8; 64];
        for offset in (0..64).step_by(8) {
            let sum = unsafe { vaddv_u8(vld1_u8(self.0.as_ptr().add(offset))) };
            unsafe { vst1_u8(out.as_mut_ptr().add(offset), vdup_n_u8(sum)); }
        }
        Self(out)
    }

    #[inline]
    pub fn permute(self, index: Self) -> Self {
        let mut out = [0u8; 64];
        unsafe {
            let table0 = vld1q_u8(self.0.as_ptr());
            let table1 = vld1q_u8(self.0.as_ptr().add(16));
            let table2 = vld1q_u8(self.0.as_ptr().add(32));
            let table3 = vld1q_u8(self.0.as_ptr().add(48));
            let base16 = vdupq_n_u8(16);
            let base32 = vdupq_n_u8(32);
            let base48 = vdupq_n_u8(48);

            for offset in (0..64).step_by(16) {
                let idx = vld1q_u8(index.0.as_ptr().add(offset));
                let a = vqtbl1q_u8(table0, idx);
                let b = vqtbl1q_u8(table1, vsubq_u8(idx, base16));
                let c = vqtbl1q_u8(table2, vsubq_u8(idx, base32));
                let d = vqtbl1q_u8(table3, vsubq_u8(idx, base48));
                let value = vorrq_u8(vorrq_u8(a, b), vorrq_u8(c, d));
                vst1q_u8(out.as_mut_ptr().add(offset), value);
            }
        }
        Self(out)
    }

    #[inline]
    pub fn shuffle(self, index: Self) -> Self {
        let mut out = [0u8; 64];
        unsafe {
            let low_nibble = vdupq_n_u8(0x0f);
            let high_bit = vdupq_n_u8(0x80);
            let zero = vdupq_n_u8(0);
            for offset in (0..64).step_by(16) {
                let values = vld1q_u8(self.0.as_ptr().add(offset));
                let indices = vld1q_u8(index.0.as_ptr().add(offset));
                let lookup = vandq_u8(indices, low_nibble);
                let selected = vqtbl1q_u8(values, lookup);
                let keep = vceqq_u8(vandq_u8(indices, high_bit), zero);
                vst1q_u8(out.as_mut_ptr().add(offset), vandq_u8(selected, keep));
            }
        }
        Self(out)
    }
}

/*----------------------------------------------------------------*/

#[repr(transparent)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct u16x32([u16; 32]);

impl u16x32 {
    #[inline]
    pub fn shr<const SHIFT: i32>(self) -> Self {
        let mut out = [0u16; 32];
        for offset in (0..32).step_by(8) {
            unsafe {
                let value = vld1q_u16(self.0.as_ptr().add(offset));
                vst1q_u16(out.as_mut_ptr().add(offset), vshrq_n_u16::<SHIFT>(value));
            }
        }
        Self(out)
    }

    #[inline]
    pub fn to_u8x64(self) -> u8x64 {
        u8x64(unsafe { core::mem::transmute(self.0) })
    }
}

/*----------------------------------------------------------------*/

#[repr(transparent)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct u16x64([u16; 64]);

impl From<[u16; 64]> for u16x64 {
    #[inline]
    fn from(value: [u16; 64]) -> Self { Self(value) }
}

macro_rules! impl_u16_ops {
    ($name:ident, $width:expr) => {
        impl BitAnd for $name {
            type Output = Self;
            #[inline]
            fn bitand(self, rhs: Self) -> Self::Output {
                let mut out = [0u16; $width];
                let mut offset = 0;
                unsafe {
                    while offset < $width {
                        let a = vld1q_u16(self.0.as_ptr().add(offset));
                        let b = vld1q_u16(rhs.0.as_ptr().add(offset));
                        vst1q_u16(out.as_mut_ptr().add(offset), vandq_u16(a, b));
                        offset += 8;
                    }
                }
                Self(out)
            }
        }
        impl BitOr for $name {
            type Output = Self;
            #[inline]
            fn bitor(self, rhs: Self) -> Self::Output {
                let mut out = [0u16; $width];
                let mut offset = 0;
                unsafe {
                    while offset < $width {
                        let a = vld1q_u16(self.0.as_ptr().add(offset));
                        let b = vld1q_u16(rhs.0.as_ptr().add(offset));
                        vst1q_u16(out.as_mut_ptr().add(offset), vorrq_u16(a, b));
                        offset += 8;
                    }
                }
                Self(out)
            }
        }
        impl BitXor for $name {
            type Output = Self;
            #[inline]
            fn bitxor(self, rhs: Self) -> Self::Output {
                let mut out = [0u16; $width];
                let mut offset = 0;
                unsafe {
                    while offset < $width {
                        let a = vld1q_u16(self.0.as_ptr().add(offset));
                        let b = vld1q_u16(rhs.0.as_ptr().add(offset));
                        vst1q_u16(out.as_mut_ptr().add(offset), veorq_u16(a, b));
                        offset += 8;
                    }
                }
                Self(out)
            }
        }
        impl Not for $name {
            type Output = Self;
            #[inline]
            fn not(self) -> Self::Output {
                let mut out = [0u16; $width];
                let mut offset = 0;
                unsafe {
                    while offset < $width {
                        let value = vld1q_u16(self.0.as_ptr().add(offset));
                        vst1q_u16(out.as_mut_ptr().add(offset), vmvnq_u16(value));
                        offset += 8;
                    }
                }
                Self(out)
            }
        }
        impl BitAndAssign for $name {
            #[inline]
            fn bitand_assign(&mut self, rhs: Self) { *self = *self & rhs; }
        }
        impl BitOrAssign for $name {
            #[inline]
            fn bitor_assign(&mut self, rhs: Self) { *self = *self | rhs; }
        }
        impl BitXorAssign for $name {
            #[inline]
            fn bitxor_assign(&mut self, rhs: Self) { *self = *self ^ rhs; }
        }
    };
}

impl_u16_ops!(u16x64, 64);

impl u16x64 {
    #[inline]
    pub fn splat(value: u16) -> Self {
        let mut out = [0u16; 64];
        for offset in (0..64).step_by(8) {
            unsafe { vst1q_u16(out.as_mut_ptr().add(offset), vdupq_n_u16(value)); }
        }
        Self(out)
    }

    #[inline]
    pub fn test(a: Self, b: Self) -> Mask16x64 {
        (a & b).nonzero()
    }

    #[inline]
    pub fn eq(a: Self, b: Self) -> Mask16x64 {
        let mut lanes = [0u16; 64];
        for offset in (0..64).step_by(8) {
            unsafe {
                let av = vld1q_u16(a.0.as_ptr().add(offset));
                let bv = vld1q_u16(b.0.as_ptr().add(offset));
                vst1q_u16(lanes.as_mut_ptr().add(offset), vceqq_u16(av, bv));
            }
        }
        let mut bits = 0u64;
        for (index, lane) in lanes.iter().enumerate() {
            if *lane != 0 { bits |= 1u64 << index; }
        }
        Mask16x64(bits)
    }

    #[inline]
    pub fn neq(a: Self, b: Self) -> Mask16x64 { !Self::eq(a, b) }

    #[inline]
    pub fn zero(self) -> Mask16x64 { Self::eq(self, Self::splat(0)) }

    #[inline]
    pub fn nonzero(self) -> Mask16x64 { Self::neq(self, Self::splat(0)) }

    #[inline]
    pub fn mask(self, mask: Mask16x64) -> Self {
        let mut mask_lanes = [0u16; 64];
        for (index, lane) in mask_lanes.iter_mut().enumerate() {
            *lane = if (mask.0 >> index) & 1 != 0 { u16::MAX } else { 0 };
        }
        let mask_vec = Self(mask_lanes);
        self & mask_vec
    }

    #[inline]
    pub fn shlv(self, shift: Self) -> Self {
        let mut out = [0u16; 64];
        for offset in (0..64).step_by(8) {
            unsafe {
                let value = vld1q_u16(self.0.as_ptr().add(offset));
                let count = vreinterpretq_s16_u16(vld1q_u16(shift.0.as_ptr().add(offset)));
                vst1q_u16(out.as_mut_ptr().add(offset), vshlq_u16(value, count));
            }
        }
        Self(out)
    }
}

/*----------------------------------------------------------------*/

#[repr(transparent)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct i16x32([i16; 32]);

impl i16x32 {
    #[inline]
    pub unsafe fn load(src: *const i16) -> Self {
        let mut out = [0i16; 32];
        for offset in (0..32).step_by(8) {
            unsafe {
                let value = vld1q_s16(src.add(offset));
                vst1q_s16(out.as_mut_ptr().add(offset), value);
            }
        }
        Self(out)
    }

    #[inline]
    pub unsafe fn store(self, dest: *mut i16) {
        for offset in (0..32).step_by(8) {
            unsafe {
                let value = vld1q_s16(self.0.as_ptr().add(offset));
                vst1q_s16(dest.add(offset), value);
            }
        }
    }

    #[inline]
    pub fn splat(value: i16) -> Self {
        let mut out = [0i16; 32];
        for offset in (0..32).step_by(8) {
            unsafe { vst1q_s16(out.as_mut_ptr().add(offset), vdupq_n_s16(value)); }
        }
        Self(out)
    }

    #[inline]
    pub fn clamp(self, min: Self, max: Self) -> Self {
        let mut out = [0i16; 32];
        for offset in (0..32).step_by(8) {
            unsafe {
                let value = vld1q_s16(self.0.as_ptr().add(offset));
                let lo = vld1q_s16(min.0.as_ptr().add(offset));
                let hi = vld1q_s16(max.0.as_ptr().add(offset));
                let clamped = vminq_s16(vmaxq_s16(value, lo), hi);
                vst1q_s16(out.as_mut_ptr().add(offset), clamped);
            }
        }
        Self(out)
    }

    #[inline]
    pub fn madd(self, rhs: Self) -> i32x16 {
        let mut out = [0i32; 16];
        for offset in (0..32).step_by(8) {
            unsafe {
                let a = vld1q_s16(self.0.as_ptr().add(offset));
                let b = vld1q_s16(rhs.0.as_ptr().add(offset));
                let low = vmull_s16(vget_low_s16(a), vget_low_s16(b));
                let high = vmull_s16(vget_high_s16(a), vget_high_s16(b));
                let pairs = vpaddq_s32(low, high);
                vst1q_s32(out.as_mut_ptr().add(offset / 2), pairs);
            }
        }
        i32x16(out)
    }
}

impl Mul for i16x32 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        let mut out = [0i16; 32];
        for offset in (0..32).step_by(8) {
            unsafe {
                let a = vld1q_s16(self.0.as_ptr().add(offset));
                let b = vld1q_s16(rhs.0.as_ptr().add(offset));
                vst1q_s16(out.as_mut_ptr().add(offset), vmulq_s16(a, b));
            }
        }
        Self(out)
    }
}

impl Add for i16x32 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        let mut out = [0i16; 32];
        for offset in (0..32).step_by(8) {
            unsafe {
                let a = vld1q_s16(self.0.as_ptr().add(offset));
                let b = vld1q_s16(rhs.0.as_ptr().add(offset));
                vst1q_s16(out.as_mut_ptr().add(offset), vaddq_s16(a, b));
            }
        }
        Self(out)
    }
}

impl Sub for i16x32 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        let mut out = [0i16; 32];
        for offset in (0..32).step_by(8) {
            unsafe {
                let a = vld1q_s16(self.0.as_ptr().add(offset));
                let b = vld1q_s16(rhs.0.as_ptr().add(offset));
                vst1q_s16(out.as_mut_ptr().add(offset), vsubq_s16(a, b));
            }
        }
        Self(out)
    }
}

impl AddAssign for i16x32 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) { *self = *self + rhs; }
}

impl SubAssign for i16x32 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) { *self = *self - rhs; }
}

/*----------------------------------------------------------------*/

#[repr(transparent)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct i32x16([i32; 16]);

impl i32x16 {
    #[inline]
    pub fn splat(value: i32) -> Self {
        let mut out = [0i32; 16];
        for offset in (0..16).step_by(4) {
            unsafe { vst1q_s32(out.as_mut_ptr().add(offset), vdupq_n_s32(value)); }
        }
        Self(out)
    }

    #[inline]
    pub fn reduce_sum(self) -> i32 {
        let mut sum = 0i32;
        for offset in (0..16).step_by(4) {
            unsafe {
                sum += vaddvq_s32(vld1q_s32(self.0.as_ptr().add(offset)));
            }
        }
        sum
    }
}

impl Add for i32x16 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        let mut out = [0i32; 16];
        for offset in (0..16).step_by(4) {
            unsafe {
                let a = vld1q_s32(self.0.as_ptr().add(offset));
                let b = vld1q_s32(rhs.0.as_ptr().add(offset));
                vst1q_s32(out.as_mut_ptr().add(offset), vaddq_s32(a, b));
            }
        }
        Self(out)
    }
}

impl Sub for i32x16 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        let mut out = [0i32; 16];
        for offset in (0..16).step_by(4) {
            unsafe {
                let a = vld1q_s32(self.0.as_ptr().add(offset));
                let b = vld1q_s32(rhs.0.as_ptr().add(offset));
                vst1q_s32(out.as_mut_ptr().add(offset), vsubq_s32(a, b));
            }
        }
        Self(out)
    }
}

impl AddAssign for i32x16 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) { *self = *self + rhs; }
}

impl SubAssign for i32x16 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) { *self = *self - rhs; }
}

/*----------------------------------------------------------------*/

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_vector_permutation_and_shuffle() {
        let input = u8x64::from(core::array::from_fn(|i| i as u8));
        let indices = u8x64::from(core::array::from_fn(|i| (63 - i) as u8));
        let reversed = input.permute(indices);
        assert_eq!(reversed.0[0], 63);
        assert_eq!(reversed.0[63], 0);

        let lane_indices = u8x64::from(core::array::from_fn(|i| (15 - (i % 16)) as u8));
        let shuffled = input.shuffle(lane_indices);
        assert_eq!(shuffled.0[0], 15);
        assert_eq!(shuffled.0[16], 31);
        assert_eq!(shuffled.0[63], 48);
    }

    #[test]
    fn byte_masks_and_zero_extension() {
        let values = u8x64::from(core::array::from_fn(|i| if i % 3 == 0 { 0x80 } else { i as u8 }));
        assert_eq!(values.msb().to_bitmask().count_ones(), 22);
        let widened = values.zero_ext();
        assert_eq!(widened.0[63], values.0[63] as u16);
    }

    #[test]
    fn signed_pairwise_multiply_add() {
        let a = i16x32::splat(3);
        let b = i16x32::splat(5);
        assert_eq!(a.madd(b).reduce_sum(), 32 * 15);
    }
}
