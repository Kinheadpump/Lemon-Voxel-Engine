mod block;
mod coords;
mod direction;

pub use block::BlockId;
pub use coords::{ChunkPos, LocalPos};
pub use direction::{Axis, FaceDir};

pub const CHUNK_SIZE: usize = 32;
pub const CHUNK_SIZE_I32: i32 = CHUNK_SIZE as i32;
pub const CHUNK_AREA: usize = CHUNK_SIZE * CHUNK_SIZE;
pub const CHUNK_VOLUME: usize = CHUNK_AREA * CHUNK_SIZE;
pub const PADDED_SIZE: usize = CHUNK_SIZE + 2;
pub const PADDED_VOLUME: usize = PADDED_SIZE * PADDED_SIZE * PADDED_SIZE;

const _: () = assert!(CHUNK_SIZE.is_power_of_two());
const _: () = assert!(
    PADDED_SIZE <= 64,
    "binary meshing stores a padded column in one u64"
);

#[inline(always)]
pub const fn padded_index(x: usize, y: usize, z: usize) -> usize {
    (y * PADDED_SIZE + z) * PADDED_SIZE + x
}
