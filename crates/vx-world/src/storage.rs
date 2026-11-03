use vx_core::{BlockId, CHUNK_VOLUME, LocalPos};

pub type DenseChunk = [BlockId; CHUNK_VOLUME];

pub fn new_dense_chunk() -> Box<DenseChunk> {
    vec![BlockId::AIR; CHUNK_VOLUME]
        .into_boxed_slice()
        .try_into()
        .expect("exact chunk volume")
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChunkVoxels {
    Uniform(BlockId),
    Packed(PackedVoxels),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedVoxels {
    palette: Vec<BlockId>,
    bits_per_voxel: u32,
    words: Box<[u64]>,
}

const MAX_BITS_PER_VOXEL: u32 = 16;

impl ChunkVoxels {
    pub fn from_dense(dense: &DenseChunk) -> Self {
        let first = dense[0];
        if dense.iter().all(|&block| block == first) {
            return Self::Uniform(first);
        }
        Self::Packed(PackedVoxels::from_dense(dense))
    }

    #[inline]
    pub fn get(&self, pos: LocalPos) -> BlockId {
        match self {
            Self::Uniform(block) => *block,
            Self::Packed(packed) => packed.get(pos.index()),
        }
    }

    pub fn set(&mut self, pos: LocalPos, block: BlockId) {
        match self {
            Self::Uniform(current) if *current == block => {}
            Self::Uniform(current) => {
                let mut packed = PackedVoxels::filled(*current, block);
                packed.set(pos.index(), block);
                *self = Self::Packed(packed);
            }
            Self::Packed(packed) => packed.set(pos.index(), block),
        }
    }

    pub fn decode_into(&self, out: &mut DenseChunk) {
        match self {
            Self::Uniform(block) => out.fill(*block),
            Self::Packed(packed) => packed.decode_into(out),
        }
    }

    pub fn uniform_block(&self) -> Option<BlockId> {
        match self {
            Self::Uniform(block) => Some(*block),
            Self::Packed(_) => None,
        }
    }

    pub fn heap_bytes(&self) -> usize {
        match self {
            Self::Uniform(_) => 0,
            Self::Packed(packed) => {
                packed.words.len() * size_of::<u64>()
                    + packed.palette.capacity() * size_of::<BlockId>()
            }
        }
    }
}

impl PackedVoxels {
    fn word_count(bits_per_voxel: u32) -> usize {
        CHUNK_VOLUME * bits_per_voxel as usize / 64
    }

    fn bits_for_palette_len(len: usize) -> u32 {
        let needed = usize::BITS - (len.max(2) - 1).leading_zeros();
        needed.next_power_of_two().min(MAX_BITS_PER_VOXEL)
    }

    fn filled(existing: BlockId, incoming: BlockId) -> Self {
        Self {
            palette: vec![existing, incoming],
            bits_per_voxel: 1,
            words: vec![0; Self::word_count(1)].into_boxed_slice(),
        }
    }

    fn from_dense(dense: &DenseChunk) -> Self {
        let mut palette = vec![dense[0]];
        let mut previous = dense[0];
        for &block in dense {
            if block != previous {
                palette_index_or_insert(&mut palette, block);
                previous = block;
            }
        }
        let bits_per_voxel = Self::bits_for_palette_len(palette.len());
        let mut packed = Self {
            palette,
            bits_per_voxel,
            words: vec![0; Self::word_count(bits_per_voxel)].into_boxed_slice(),
        };
        let mut cached = (dense[0], 0u16);
        for (voxel, &block) in dense.iter().enumerate() {
            if block != cached.0 {
                cached = (
                    block,
                    palette_index_or_insert(&mut packed.palette, block) as u16,
                );
            }
            packed.write_index(voxel, cached.1);
        }
        packed
    }

    #[inline(always)]
    fn read_index(&self, voxel: usize) -> u16 {
        let bit = voxel * self.bits_per_voxel as usize;
        let mask = (1u64 << self.bits_per_voxel) - 1;
        ((self.words[bit / 64] >> (bit % 64)) & mask) as u16
    }

    #[inline(always)]
    fn write_index(&mut self, voxel: usize, palette_index: u16) {
        let bit = voxel * self.bits_per_voxel as usize;
        let mask = (1u64 << self.bits_per_voxel) - 1;
        let word = &mut self.words[bit / 64];
        *word = (*word & !(mask << (bit % 64))) | (u64::from(palette_index) << (bit % 64));
    }

    #[inline]
    fn get(&self, voxel: usize) -> BlockId {
        self.palette[self.read_index(voxel) as usize]
    }

    fn set(&mut self, voxel: usize, block: BlockId) {
        let palette_index = palette_index_or_insert(&mut self.palette, block);
        if self.palette.len() > 1 << self.bits_per_voxel {
            self.widen();
        }
        self.write_index(voxel, palette_index as u16);
    }

    fn widen(&mut self) {
        let mut widened = Self {
            palette: Vec::new(),
            bits_per_voxel: (self.bits_per_voxel * 2).min(MAX_BITS_PER_VOXEL),
            words: Box::new([]),
        };
        widened.words = vec![0; Self::word_count(widened.bits_per_voxel)].into_boxed_slice();
        for voxel in 0..CHUNK_VOLUME {
            widened.write_index(voxel, self.read_index(voxel));
        }
        widened.palette = std::mem::take(&mut self.palette);
        *self = widened;
    }

    fn decode_into(&self, out: &mut DenseChunk) {
        let bits = self.bits_per_voxel as usize;
        let per_word = 64 / bits;
        let mask = (1u64 << bits) - 1;
        for (word, chunk) in self.words.iter().zip(out.chunks_exact_mut(per_word)) {
            let mut word = *word;
            for slot in chunk {
                *slot = self.palette[(word & mask) as usize];
                word >>= bits;
            }
        }
    }
}

#[inline]
fn palette_index_or_insert(palette: &mut Vec<BlockId>, block: BlockId) -> usize {
    if let Some(found) = palette.iter().position(|&existing| existing == block) {
        found
    } else {
        palette.push(block);
        palette.len() - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    #[test]
    fn uniform_chunk_has_no_heap() {
        let mut dense = new_dense_chunk();
        dense.fill(BlockId::STONE);
        let voxels = ChunkVoxels::from_dense(&dense);
        assert_eq!(voxels.uniform_block(), Some(BlockId::STONE));
        assert_eq!(voxels.heap_bytes(), 0);
    }

    #[test]
    fn dense_round_trip_and_widening_preserve_every_voxel() {
        let mut rng = 0x9E37_79B9_7F4A_7C15;
        let mut dense = new_dense_chunk();
        for block in dense.iter_mut() {
            *block = BlockId((xorshift(&mut rng) % 3) as u16);
        }
        let mut voxels = ChunkVoxels::from_dense(&dense);
        let mut decoded = new_dense_chunk();
        voxels.decode_into(&mut decoded);
        assert_eq!(decoded, dense);

        for step in 0..600u16 {
            let voxel = (xorshift(&mut rng) as usize) % CHUNK_VOLUME;
            let block = BlockId(100 + step % 300);
            dense[voxel] = block;
            voxels.set(LocalPos::from_index(voxel), block);
        }
        voxels.decode_into(&mut decoded);
        assert_eq!(decoded, dense);
        for (voxel, &block) in dense.iter().enumerate() {
            assert_eq!(voxels.get(LocalPos::from_index(voxel)), block);
        }
    }

    #[test]
    fn set_on_uniform_promotes_to_one_bit_packing() {
        let mut voxels = ChunkVoxels::Uniform(BlockId::AIR);
        voxels.set(LocalPos::new(3, 4, 5), BlockId::DIRT);
        assert_eq!(voxels.get(LocalPos::new(3, 4, 5)), BlockId::DIRT);
        assert_eq!(voxels.get(LocalPos::new(0, 0, 0)), BlockId::AIR);
        assert_eq!(
            voxels.heap_bytes(),
            CHUNK_VOLUME / 8 + 2 * size_of::<BlockId>()
        );
    }
}
