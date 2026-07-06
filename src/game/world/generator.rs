use bevy_math::Vec2;
use noiz::Noise;
use noiz::prelude::*;

use crate::engine::config::EngineConfig;
use crate::engine::render::textures::{TEXTURE_LAYER_DIRT, TEXTURE_LAYER_GRASS, TEXTURE_LAYER_STONE};

use super::chunk::{CHUNK_SIZE, Chunk};

pub struct TerrainGenerator {
    noise: Noise<common_noise::Perlin>,
    noise_frequency: f32,
    base_height: f32,
    height_amplitude: f32,
    dirt_layer_depth: i32,
    /// Die Ursprungszelle des `noiz`-Gradientenrauschens (Welt-Koordinaten nahe (0,0)) ist
    /// degeneriert und liefert dort konstant 0.0 unabhaengig von der Position. Ein fixer Offset
    /// verschiebt jede Sample-Koordinate weit weg vom Ursprung und umgeht das vollstaendig.
    noise_origin_offset: f32,
}

impl TerrainGenerator {
    pub fn new(config: &EngineConfig) -> Self {
        let mut noise = Noise::<common_noise::Perlin>::default();
        noise.set_seed(config.terrain_seed);
        Self {
            noise,
            noise_frequency: config.terrain_noise_frequency,
            base_height: config.terrain_base_height,
            height_amplitude: config.terrain_height_amplitude,
            dirt_layer_depth: config.terrain_dirt_layer_depth,
            noise_origin_offset: config.terrain_noise_origin_offset,
        }
    }

    pub fn height_at(&self, world_x: i32, world_z: i32) -> i32 {
        let sample_point = Vec2::new(
            world_x as f32 * self.noise_frequency + self.noise_origin_offset,
            world_z as f32 * self.noise_frequency + self.noise_origin_offset,
        );
        let raw: f32 = self.noise.sample(sample_point);
        (self.base_height + raw * self.height_amplitude) as i32
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
                    } else if depth_from_surface <= self.dirt_layer_depth {
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
