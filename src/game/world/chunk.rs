pub const CHUNK_SIZE: i32 = 32;
pub const CHUNK_VOLUME: usize = 32 * 32 * 32;
/// log2(CHUNK_SIZE) - CHUNK_SIZE ist eine Zweierpotenz, die Flat-Array-Indizierung nutzt deshalb
/// zwingend Bitshifts statt Multiplikation/Division/Modulo.
const CHUNK_SHIFT: u32 = 5;

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

    /// Guenstige Vorabpruefung, um das teure Greedy-Meshing (6 Richtungen * 32 Ebenen) fuer
    /// Chunks zu ueberspringen, die ohnehin keine Faces erzeugen wuerden - bei vertikal gestapelten
    /// Chunks ist der weit ueberwiegende Teil (alles oberhalb der Terrainhoehe) reine Luft.
    pub fn is_empty(&self) -> bool {
        self.blocks.iter().all(|&block| block == 0)
    }

    #[inline(always)]
    pub const fn index_from_pos(x: u32, y: u32, z: u32) -> usize {
        (x + (y << CHUNK_SHIFT) + (z << (CHUNK_SHIFT * 2))) as usize
    }

    #[inline(always)]
    pub const fn pos_from_index(index: usize) -> (u32, u32, u32) {
        let index = index as u32;
        let mask = (1u32 << CHUNK_SHIFT) - 1;
        let x = index & mask;
        let y = (index >> CHUNK_SHIFT) & mask;
        let z = index >> (CHUNK_SHIFT * 2);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_chunk_is_empty() {
        assert!(Chunk::empty().is_empty());
    }

    #[test]
    fn placing_a_block_makes_it_non_empty() {
        let mut chunk = Chunk::empty();
        chunk.set_block(1, 2, 3, 7);
        assert!(!chunk.is_empty());
        assert_eq!(chunk.get_block(1, 2, 3), 7);
    }

    #[test]
    fn out_of_bounds_access_is_a_noop() {
        let mut chunk = Chunk::empty();
        chunk.set_block(-1, 0, 0, 9);
        chunk.set_block(CHUNK_SIZE, 0, 0, 9);
        assert!(chunk.is_empty());
        assert_eq!(chunk.get_block(-1, 0, 0), 0);
    }
}
