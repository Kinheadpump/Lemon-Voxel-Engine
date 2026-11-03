use fastnoise_lite::{FastNoiseLite, FractalType, NoiseType};
use vx_core::{BlockId, CHUNK_AREA, CHUNK_SIZE, CHUNK_SIZE_I32, ChunkPos};

use crate::{ChunkVoxels, DenseChunk};

#[derive(Clone, Copy, Debug)]
pub struct TerrainSettings {
    pub seed: i32,
    pub sea_level: i32,
    pub amplitude: f32,
    pub frequency: f32,
    pub octaves: i32,
}

impl Default for TerrainSettings {
    fn default() -> Self {
        Self {
            seed: 1337,
            sea_level: 64,
            amplitude: 48.0,
            frequency: 0.004,
            octaves: 5,
        }
    }
}

pub struct TerrainGenerator {
    settings: TerrainSettings,
    height_noise: FastNoiseLite,
}

impl TerrainGenerator {
    pub fn new(settings: TerrainSettings) -> Self {
        let mut height_noise = FastNoiseLite::with_seed(settings.seed);
        height_noise.set_noise_type(Some(NoiseType::OpenSimplex2));
        height_noise.set_fractal_type(Some(FractalType::FBm));
        height_noise.set_fractal_octaves(Some(settings.octaves));
        height_noise.set_frequency(Some(settings.frequency));
        Self {
            settings,
            height_noise,
        }
    }

    pub fn surface_heights(&self, pos: ChunkPos, heights: &mut [i32; CHUNK_AREA]) {
        let origin = pos.min_block();
        for (column, height) in heights.iter_mut().enumerate() {
            let x = (origin.x + (column % CHUNK_SIZE) as i32) as f32;
            let z = (origin.z + (column / CHUNK_SIZE) as i32) as f32;
            let noise = self.height_noise.get_noise_2d(x, z);
            *height = self.settings.sea_level + (noise * self.settings.amplitude) as i32;
        }
    }

    pub fn generate(&self, pos: ChunkPos, dense_scratch: &mut DenseChunk) -> ChunkVoxels {
        let mut heights = [0i32; CHUNK_AREA];
        self.surface_heights(pos, &mut heights);
        let chunk_bottom = pos.min_block().y;
        let chunk_top = chunk_bottom + CHUNK_SIZE_I32;
        let (lowest, highest) = heights
            .iter()
            .fold((i32::MAX, i32::MIN), |(lo, hi), &h| (lo.min(h), hi.max(h)));
        if chunk_bottom > highest {
            return ChunkVoxels::Uniform(BlockId::AIR);
        }
        if chunk_top + 4 <= lowest {
            return ChunkVoxels::Uniform(BlockId::STONE);
        }
        for y in 0..CHUNK_SIZE {
            let world_y = chunk_bottom + y as i32;
            for (column, &surface) in heights.iter().enumerate() {
                dense_scratch[y * CHUNK_AREA + column] = Self::block_at_depth(surface - world_y);
            }
        }
        ChunkVoxels::from_dense(dense_scratch)
    }

    #[inline(always)]
    fn block_at_depth(depth_below_surface: i32) -> BlockId {
        match depth_below_surface {
            i32::MIN..0 => BlockId::AIR,
            0 => BlockId::GRASS,
            1..4 => BlockId::DIRT,
            _ => BlockId::STONE,
        }
    }
}
