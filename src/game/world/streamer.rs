use std::sync::Arc;

use crate::engine::config::EngineConfig;
use crate::engine::render::renderer::ChunkRenderer;
use crate::game::math::cascades::{Cascade, MAX_SHADOW_CASCADES};

use super::generator::TerrainGenerator;
use super::manager::ChunkManager;
use super::raycast::RaycastHit;

/// Koordiniert alle LOD-Ringe (s. `EngineConfig::lod_ring_runtimes`) - Ring 0 ist immer LOD0
/// (voller Detailgrad, editierbar, wirft Schatten). Editier-/Physik-/Schatten-Operationen sind
/// bewusst NUR gegen Ring 0 exponiert: LOD-Ringe sind rein prozedural und nie Ziel von
/// Bearbeitung, Kollision oder Schattenwurf (s. `generator/lod.rs`-Kommentar) - eine generische
/// "irgendein Ring"-API wuerde diese Einschraenkung nur verschleiern.
pub struct WorldStreamer {
    rings: Vec<ChunkManager>,
}

impl WorldStreamer {
    pub fn new(config: &EngineConfig) -> Self {
        let rings = config.lod_ring_runtimes().iter().map(|ring| ChunkManager::new(config, ring)).collect();
        Self { rings }
    }

    fn lod0(&self) -> &ChunkManager {
        &self.rings[0]
    }

    pub fn generator(&self) -> &Arc<TerrainGenerator> {
        self.lod0().generator()
    }

    pub fn is_solid_at(&self, world_x: i32, world_y: i32, world_z: i32) -> bool {
        self.lod0().is_solid_at(world_x, world_y, world_z)
    }

    pub fn raycast(&self, origin: glam::Vec3, direction: glam::Vec3, max_distance: f32) -> Option<RaycastHit> {
        self.lod0().raycast(origin, direction, max_distance)
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
        self.rings[0].set_block(world_x, world_y, world_z, block_id, queue, renderer)
    }

    pub fn loaded_chunk_count(&self) -> usize {
        self.rings.iter().map(ChunkManager::loaded_chunk_count).sum()
    }

    /// Treibt das Streaming ALLER Ringe voran, dann Schatten-Sichtbarkeit NUR fuer Ring 0 (LOD-
    /// Ringe werfen keine Schatten, s. Typ-Kommentar).
    pub fn update(
        &mut self,
        camera_position: glam::Vec3,
        cascades: &[Cascade; MAX_SHADOW_CASCADES],
        cascade_count: u32,
        queue: &wgpu::Queue,
        renderer: &mut ChunkRenderer,
    ) {
        for ring in &mut self.rings {
            ring.update(camera_position, queue, renderer);
        }
        self.rings[0].update_shadow_visibility(cascades, cascade_count, queue, renderer);
    }
}
