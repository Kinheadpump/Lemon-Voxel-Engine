use std::sync::Arc;

use crate::engine::config::EngineConfig;
use crate::engine::render::renderer::ChunkRenderer;
use crate::game::math::cascades::{Cascade, MAX_SHADOW_CASCADES};

use super::generator::TerrainGenerator;
use super::manager::ChunkManager;
use super::raycast::RaycastHit;

/// Fassade zwischen App und Chunk-Streaming - haelt die Welt-API stabil, waehrend die dahinter
/// liegende Streaming-Struktur (aktuell ein einzelner Voll-Detail-`ChunkManager`) austauschbar
/// bleibt.
pub struct WorldStreamer {
    chunks: ChunkManager,
}

impl WorldStreamer {
    pub fn new(config: &EngineConfig) -> Self {
        Self { chunks: ChunkManager::new(config) }
    }

    pub fn generator(&self) -> &Arc<TerrainGenerator> {
        self.chunks.generator()
    }

    pub fn is_solid_at(&self, world_x: i32, world_y: i32, world_z: i32) -> bool {
        self.chunks.is_solid_at(world_x, world_y, world_z)
    }

    pub fn raycast(&self, origin: glam::Vec3, direction: glam::Vec3, max_distance: f32) -> Option<RaycastHit> {
        self.chunks.raycast(origin, direction, max_distance)
    }

    pub fn set_block(
        &mut self,
        world_x: i32,
        world_y: i32,
        world_z: i32,
        block_id: u16,
        queue: &wgpu::Queue,
        renderer: &mut ChunkRenderer,
    ) -> bool {
        self.chunks.set_block(world_x, world_y, world_z, block_id, queue, renderer)
    }

    pub fn loaded_chunk_count(&self) -> usize {
        self.chunks.loaded_chunk_count()
    }

    pub fn update(
        &mut self,
        camera_position: glam::Vec3,
        cascades: &[Cascade; MAX_SHADOW_CASCADES],
        cascade_count: u32,
        queue: &wgpu::Queue,
        renderer: &mut ChunkRenderer,
    ) {
        self.chunks.update(camera_position, queue, renderer);
        self.chunks.update_shadow_visibility(cascades, cascade_count, queue, renderer);
    }
}
