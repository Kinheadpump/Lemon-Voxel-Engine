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

    /// Fallback-Quelle der Wahrheit fuer Voxel-Festigkeit ausserhalb geladener/editierter
    /// Chunk-Daten - genutzt vom Mesher (Nachbar-Check ueber Chunk-Grenzen an noch nicht gemeshten
    /// Chunks) UND von `ChunkManager::is_solid_at` fuer Regionen, die (noch) nicht geladen sind.
    /// Kein Limit mehr nach oben ODER unten: alles oberhalb der Terrainoberflaeche ist Luft, alles
    /// darunter ist (unendlich tief) massiver Fels - die Chunk-Aufteilung in Y ist reine
    /// Streaming-Granularitaet, kein Weltrand.
    pub fn is_solid(&self, world_x: i32, world_y: i32, world_z: i32) -> bool {
        world_y <= self.height_at(world_x, world_z)
    }

    pub fn generate_chunk(&self, chunk_x: i32, chunk_y: i32, chunk_z: i32, chunk: &mut Chunk) {
        chunk.clear();

        for local_z in 0..CHUNK_SIZE {
            for local_x in 0..CHUNK_SIZE {
                let world_x = chunk_x * CHUNK_SIZE + local_x;
                let world_z = chunk_z * CHUNK_SIZE + local_z;
                let height = self.height_at(world_x, world_z);

                for local_y in 0..CHUNK_SIZE {
                    let world_y = chunk_y * CHUNK_SIZE + local_y;
                    if world_y > height {
                        continue;
                    }

                    let depth_from_surface = height - world_y;
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
