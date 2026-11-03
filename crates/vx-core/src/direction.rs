#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Axis {
    X = 0,
    Y = 1,
    Z = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum FaceDir {
    PosX = 0,
    NegX = 1,
    PosY = 2,
    NegY = 3,
    PosZ = 4,
    NegZ = 5,
}

impl Axis {
    pub const ALL: [Self; 3] = [Self::X, Self::Y, Self::Z];

    #[inline(always)]
    pub const fn index(self) -> usize {
        self as usize
    }

    #[inline(always)]
    pub const fn depth_u_v_to_xyz(self, depth: usize, u: usize, v: usize) -> [usize; 3] {
        match self {
            Self::X => [depth, u, v],
            Self::Y => [u, depth, v],
            Self::Z => [u, v, depth],
        }
    }
}

impl FaceDir {
    pub const ALL: [Self; 6] = [
        Self::PosX,
        Self::NegX,
        Self::PosY,
        Self::NegY,
        Self::PosZ,
        Self::NegZ,
    ];

    #[inline(always)]
    pub const fn index(self) -> usize {
        self as usize
    }

    #[inline(always)]
    pub const fn axis(self) -> Axis {
        match self {
            Self::PosX | Self::NegX => Axis::X,
            Self::PosY | Self::NegY => Axis::Y,
            Self::PosZ | Self::NegZ => Axis::Z,
        }
    }

    #[inline(always)]
    pub const fn is_positive(self) -> bool {
        matches!(self, Self::PosX | Self::PosY | Self::PosZ)
    }

    #[inline(always)]
    pub const fn normal(self) -> [i32; 3] {
        match self {
            Self::PosX => [1, 0, 0],
            Self::NegX => [-1, 0, 0],
            Self::PosY => [0, 1, 0],
            Self::NegY => [0, -1, 0],
            Self::PosZ => [0, 0, 1],
            Self::NegZ => [0, 0, -1],
        }
    }
}
