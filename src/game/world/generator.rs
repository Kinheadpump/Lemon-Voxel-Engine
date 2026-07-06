use bevy_math::Vec2;
use noiz::Noise;
use noiz::prelude::*;

use crate::engine::render::textures::{TEXTURE_LAYER_DIRT, TEXTURE_LAYER_GRASS, TEXTURE_LAYER_STONE};

use super::chunk::{CHUNK_SIZE, Chunk};

const WORLD_SEED: u32 = 1337;
const NOISE_FREQUENCY: f32 = 0.02;
const BASE_HEIGHT: f32 = 12.0;
const HEIGHT_AMPLITUDE: f32 = 10.0;
const DIRT_LAYER_DEPTH: i32 = 3;

/// Die Ursprungszelle des `noiz`-Gradientenrauschens (Welt-Koordinaten nahe (0,0)) ist
/// degeneriert und liefert dort konstant 0.0 unabhaengig von der Position. Ein fixer Offset
/// verschiebt jede Sample-Koordinate weit weg vom Ursprung und umgeht das vollstaendig.
const NOISE_ORIGIN_OFFSET: f32 = 10_000.0;

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
        let sample_point = Vec2::new(
            world_x as f32 * NOISE_FREQUENCY + NOISE_ORIGIN_OFFSET,
            world_z as f32 * NOISE_FREQUENCY + NOISE_ORIGIN_OFFSET,
        );
        let raw: f32 = self.noise.sample(sample_point);
        (BASE_HEIGHT + raw * HEIGHT_AMPLITUDE) as i32
    }

    /// Einzige Quelle der Wahrheit fuer Voxel-Festigkeit ausserhalb geladener Chunk-Daten -
    /// genutzt vom Mesher (Nachbar-Check ueber Chunk-Grenzen) UND von der Physik (Kollision).
    /// Da das Terrain rein prozedural ist (noch keine Block-Edits), ist eine Hoehenabfrage
    /// ausreichend und immer verfuegbar, auch fuer noch nicht gemeshte Chunks.
    pub fn is_solid(&self, world_x: i32, world_y: i32, world_z: i32) -> bool {
        world_y >= 0 && world_y <= self.height_at(world_x, world_z).clamp(0, CHUNK_SIZE - 1)
    }

    pub fn generate_chunk(&self, chunk_x: i32, chunk_z: i32, chunk: &mut Chunk) {
        chunk.clear();

        for local_z in 0..CHUNK_SIZE {
            for local_x in 0..CHUNK_SIZE {
                let world_x = chunk_x * CHUNK_SIZE + local_x;
                let world_z = chunk_z * CHUNK_SIZE + local_z;
                let height = self.height_at(world_x, world_z).clamp(0, CHUNK_SIZE - 1);

                for local_y in 0..=height {
                    let depth_from_surface = height - local_y;
                    let block_id = if depth_from_surface == 0 {
                        TEXTURE_LAYER_GRASS
                    } else if depth_from_surface <= DIRT_LAYER_DEPTH {
                        TEXTURE_LAYER_DIRT
                    } else {
                        TEXTURE_LAYER_STONE
                    } as u16;

                    chunk.set_block(local_x, local_y, local_z, block_id);
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
