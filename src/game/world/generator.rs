use bevy_math::Vec2;
use noiz::Noise;
use noiz::prelude::*;

use super::chunk::{CHUNK_SIZE, Chunk};

const WORLD_SEED: u32 = 1337;
const NOISE_FREQUENCY: f32 = 0.02;
const BASE_HEIGHT: f32 = 12.0;
const HEIGHT_AMPLITUDE: f32 = 10.0;

pub struct TerrainGenerator {
    noise: Noise<common_noise::Perlin>,
}

impl TerrainGenerator {
    pub fn new() -> Self {
        let mut noise = Noise::<common_noise::Perlin>::default();
        noise.set_seed(WORLD_SEED);
        Self { noise }
    }

    pub fn height_at(&self, world_x: i32, world_z: i32) -> i32 {
        let sample_point =
            Vec2::new(world_x as f32 * NOISE_FREQUENCY, world_z as f32 * NOISE_FREQUENCY);
        let raw: f32 = self.noise.sample(sample_point);
        (BASE_HEIGHT + raw * HEIGHT_AMPLITUDE) as i32
    }

    pub fn generate_chunk(&self, chunk_x: i32, chunk_z: i32, chunk: &mut Chunk) {
        chunk.clear();

        for local_z in 0..CHUNK_SIZE {
            for local_x in 0..CHUNK_SIZE {
                let world_x = chunk_x * CHUNK_SIZE + local_x;
                let world_z = chunk_z * CHUNK_SIZE + local_z;
                let height = self.height_at(world_x, world_z).clamp(0, CHUNK_SIZE - 1);

                for local_y in 0..=height {
                    chunk.set_block(local_x, local_y, local_z, 1);
                }
            }
        }
    }
}

impl Default for TerrainGenerator {
    fn default() -> Self {
        Self::new()
    }
}
