pub const CHUNK_SIZE: i32 = 32;
pub const CHUNK_VOLUME: usize = 32 * 32 * 32;

pub struct Chunk {
    pub blocks: [u16; CHUNK_VOLUME],
}

impl Chunk {
    pub fn empty() -> Self {
        Self { blocks: [0u16; CHUNK_VOLUME] }
    }

    pub fn clear(&mut self) {
        self.blocks.fill(0);
    }

    #[inline(always)]
    pub const fn index_from_pos(x: u32, y: u32, z: u32) -> usize {
        (x + y * 32 + z * 1024) as usize
    }

    #[inline(always)]
    pub const fn pos_from_index(index: usize) -> (u32, u32, u32) {
        let index = index as u32;
        let x = index % 32;
        let y = (index / 32) % 32;
        let z = index / 1024;
        (x, y, z)
    }

    #[inline(always)]
    pub fn set_block(&mut self, x: i32, y: i32, z: i32, id: u16) {
        if x < 0 || y < 0 || z < 0 || x >= CHUNK_SIZE || y >= CHUNK_SIZE || z >= CHUNK_SIZE {
            return;
        }
        self.blocks[Self::index_from_pos(x as u32, y as u32, z as u32)] = id;
    }

    #[inline(always)]
    pub fn get_block(&self, x: i32, y: i32, z: i32) -> u16 {
        if x < 0 || y < 0 || z < 0 || x >= CHUNK_SIZE || y >= CHUNK_SIZE || z >= CHUNK_SIZE {
            return 0;
        }
        self.blocks[Self::index_from_pos(x as u32, y as u32, z as u32)]
    }
}
