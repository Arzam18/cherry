#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
mod avx2;
#[cfg(target_feature = "avx512f")]
mod avx512;
#[cfg(target_arch = "aarch64")]
mod neon;

#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
pub use avx2::*;
#[cfg(target_feature = "avx512f")]
pub use avx512::*;
#[cfg(target_arch = "aarch64")]
pub use neon::*;
