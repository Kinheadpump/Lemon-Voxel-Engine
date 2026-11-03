use crate::{CHUNK_SIZE, CHUNK_SIZE_I32};
use glam::IVec3;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ChunkPos(pub IVec3);

impl ChunkPos {
    const KEY_BITS: u32 = 21;
    const KEY_MASK: u64 = (1 << Self::KEY_BITS) - 1;
    const KEY_BIAS: i32 = 1 << (Self::KEY_BITS - 1);

    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self(IVec3::new(x, y, z))
    }

    #[inline]
    pub fn containing_block(world_block: IVec3) -> Self {
        Self(world_block >> CHUNK_SIZE.trailing_zeros() as i32)
    }

    #[inline]
    pub fn min_block(self) -> IVec3 {
        self.0 * CHUNK_SIZE_I32
    }

    #[inline]
    pub fn offset(self, delta: IVec3) -> Self {
        Self(self.0 + delta)
    }

    #[inline]
    pub fn packed_key(self) -> u64 {
        let pack = |v: i32| u64::from((v + Self::KEY_BIAS) as u32) & Self::KEY_MASK;
        pack(self.0.x)
            | (pack(self.0.y) << Self::KEY_BITS)
            | (pack(self.0.z) << (2 * Self::KEY_BITS))
    }

    #[inline]
    pub fn from_packed_key(key: u64) -> Self {
        let unpack = |shift: u32| ((key >> shift) & Self::KEY_MASK) as i32 - Self::KEY_BIAS;
        Self::new(
            unpack(0),
            unpack(Self::KEY_BITS),
            unpack(2 * Self::KEY_BITS),
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LocalPos {
    pub x: u8,
    pub y: u8,
    pub z: u8,
}

impl LocalPos {
    #[inline(always)]
    pub const fn new(x: u8, y: u8, z: u8) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub fn of_world_block(world_block: IVec3) -> Self {
        let local = world_block & (CHUNK_SIZE_I32 - 1);
        Self::new(local.x as u8, local.y as u8, local.z as u8)
    }

    #[inline(always)]
    pub const fn index(self) -> usize {
        (self.y as usize * CHUNK_SIZE + self.z as usize) * CHUNK_SIZE + self.x as usize
    }

    #[inline(always)]
    pub const fn from_index(index: usize) -> Self {
        Self::new(
            (index % CHUNK_SIZE) as u8,
            (index / (CHUNK_SIZE * CHUNK_SIZE)) as u8,
            ((index / CHUNK_SIZE) % CHUNK_SIZE) as u8,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CHUNK_VOLUME;

    #[test]
    fn packed_key_round_trips_negative_and_positive() {
        for pos in [
            ChunkPos::new(0, 0, 0),
            ChunkPos::new(-1, 5, -700_000),
            ChunkPos::new(1_000_000, -1_000_000, 3),
        ] {
            assert_eq!(ChunkPos::from_packed_key(pos.packed_key()), pos);
        }
    }

    #[test]
    fn containing_block_floors_negative_coordinates() {
        assert_eq!(
            ChunkPos::containing_block(IVec3::new(-1, 0, 31)),
            ChunkPos::new(-1, 0, 0)
        );
        assert_eq!(
            ChunkPos::containing_block(IVec3::new(32, -33, -32)),
            ChunkPos::new(1, -2, -1)
        );
        assert_eq!(
            LocalPos::of_world_block(IVec3::new(-1, 32, -33)),
            LocalPos::new(31, 0, 31)
        );
    }

    #[test]
    fn local_index_round_trips() {
        for index in (0..CHUNK_VOLUME).step_by(97) {
            assert_eq!(LocalPos::from_index(index).index(), index);
        }
    }
}
