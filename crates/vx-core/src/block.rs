use bytemuck::{Pod, Zeroable};

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Pod, Zeroable)]
pub struct BlockId(pub u16);

impl BlockId {
    pub const AIR: Self = Self(0);
    pub const STONE: Self = Self(1);
    pub const DIRT: Self = Self(2);
    pub const GRASS: Self = Self(3);
    pub const SAND: Self = Self(4);
    pub const SNOW: Self = Self(5);

    #[inline(always)]
    pub const fn is_air(self) -> bool {
        self.0 == 0
    }

    #[inline(always)]
    pub const fn is_opaque(self) -> bool {
        self.0 != 0
    }
}
