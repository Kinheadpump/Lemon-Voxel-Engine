use bytemuck::{Pod, Zeroable};
use vx_core::{BlockId, FaceDir};

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Pod, Zeroable)]
pub struct PackedQuad(pub u64);

impl PackedQuad {
    const COORD_BITS: u32 = 6;
    const COORD_MASK: u64 = (1 << Self::COORD_BITS) - 1;
    const Y_SHIFT: u32 = 6;
    const Z_SHIFT: u32 = 12;
    const WIDTH_SHIFT: u32 = 18;
    const HEIGHT_SHIFT: u32 = 24;
    const DIR_SHIFT: u32 = 30;
    const BLOCK_SHIFT: u32 = 33;

    #[inline(always)]
    pub fn new(origin: [usize; 3], width: u32, height: u32, dir: FaceDir, block: BlockId) -> Self {
        debug_assert!((1..=64).contains(&width) && (1..=64).contains(&height));
        Self(
            origin[0] as u64
                | (origin[1] as u64) << Self::Y_SHIFT
                | (origin[2] as u64) << Self::Z_SHIFT
                | u64::from(width - 1) << Self::WIDTH_SHIFT
                | u64::from(height - 1) << Self::HEIGHT_SHIFT
                | (dir as u64) << Self::DIR_SHIFT
                | u64::from(block.0) << Self::BLOCK_SHIFT,
        )
    }

    #[inline(always)]
    fn field(self, shift: u32) -> u32 {
        ((self.0 >> shift) & Self::COORD_MASK) as u32
    }

    pub fn origin(self) -> [u32; 3] {
        [
            self.field(0),
            self.field(Self::Y_SHIFT),
            self.field(Self::Z_SHIFT),
        ]
    }

    pub fn width(self) -> u32 {
        self.field(Self::WIDTH_SHIFT) + 1
    }

    pub fn height(self) -> u32 {
        self.field(Self::HEIGHT_SHIFT) + 1
    }

    pub fn dir(self) -> FaceDir {
        FaceDir::ALL[((self.0 >> Self::DIR_SHIFT) & 0b111) as usize]
    }

    pub fn block(self) -> BlockId {
        BlockId((self.0 >> Self::BLOCK_SHIFT) as u16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fields_round_trip_at_extremes() {
        let quad = PackedQuad::new([63, 0, 31], 64, 1, FaceDir::NegZ, BlockId(u16::MAX));
        assert_eq!(quad.origin(), [63, 0, 31]);
        assert_eq!((quad.width(), quad.height()), (64, 1));
        assert_eq!(quad.dir(), FaceDir::NegZ);
        assert_eq!(quad.block(), BlockId(u16::MAX));
    }
}
