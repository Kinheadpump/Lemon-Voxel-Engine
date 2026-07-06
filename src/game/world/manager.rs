use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};

use rayon::prelude::*;

use crate::engine::core::mesher::{DirectionalMesh, mesh_chunk};
use crate::engine::render::renderer::{ChunkGpuHandle, ChunkRenderer};
use crate::game::math::frustum::Frustum;

use super::chunk::{CHUNK_SIZE, Chunk};
use super::generator::TerrainGenerator;

/// Vorallozierter Chunk-Pool. Deckt Render-Distanz bis 32 ab ((2*32+1)^2 = 4225 Chunks) mit
/// Reserve. Jeder Chunk belegt 64 KiB RAM.
pub const POOL_SIZE: usize = 4300;

type ChunkCoord = (i32, i32);

struct LoadedChunk {
    pool_slot: usize,
    gpu_handle: ChunkGpuHandle,
}

struct GenerationResult {
    coord: ChunkCoord,
    pool_slot: usize,
    chunk: Chunk,
    mesh: DirectionalMesh,
}

pub struct ChunkManager {
    pool: Vec<Option<Chunk>>,
    pool_free_list: Vec<usize>,
    loaded: HashMap<ChunkCoord, LoadedChunk>,
    in_flight: HashSet<ChunkCoord>,
    generator: Arc<TerrainGenerator>,
    result_tx: Sender<GenerationResult>,
    result_rx: Receiver<GenerationResult>,
    render_distance_chunks: i32,
    visible_handles: Vec<ChunkGpuHandle>,
    visible_count: usize,
}

impl ChunkManager {
    pub fn new(render_distance_chunks: i32) -> Self {
        let pool = (0..POOL_SIZE).map(|_| Some(Chunk::empty())).collect();
        let pool_free_list = (0..POOL_SIZE).collect();
        let (result_tx, result_rx) = channel();

        Self {
            pool,
            pool_free_list,
            loaded: HashMap::new(),
            in_flight: HashSet::new(),
            generator: Arc::new(TerrainGenerator::new()),
            result_tx,
            result_rx,
            render_distance_chunks,
            visible_handles: Vec::new(),
            visible_count: 0,
        }
    }

    pub fn loaded_chunk_count(&self) -> usize {
        self.loaded.len()
    }

    pub fn visible_chunk_count(&self) -> usize {
        self.visible_count
    }

    pub fn update(
        &mut self,
        camera_position: glam::Vec3,
        frustum: &Frustum,
        queue: &wgpu::Queue,
        renderer: &mut ChunkRenderer,
    ) {
        self.apply_completed_generations(queue, renderer);

        let center_x = (camera_position.x / CHUNK_SIZE as f32).floor() as i32;
        let center_z = (camera_position.z / CHUNK_SIZE as f32).floor() as i32;

        let mut desired = HashSet::new();
        for dz in -self.render_distance_chunks..=self.render_distance_chunks {
            for dx in -self.render_distance_chunks..=self.render_distance_chunks {
                desired.insert((center_x + dx, center_z + dz));
            }
        }

        let to_unload: Vec<ChunkCoord> =
            self.loaded.keys().copied().filter(|coord| !desired.contains(coord)).collect();
        for coord in to_unload {
            self.unload_chunk(coord, renderer);
        }

        for coord in desired {
            if self.loaded.contains_key(&coord) || self.in_flight.contains(&coord) {
                continue;
            }

            let Some(pool_slot) = self.pool_free_list.pop() else {
                continue;
            };

            let mut chunk = self.pool[pool_slot].take().expect("Pool-Slot bereits leer");
            self.in_flight.insert(coord);

            let generator = Arc::clone(&self.generator);
            let tx = self.result_tx.clone();

            rayon::spawn(move || {
                generator.generate_chunk(coord.0, coord.1, &mut chunk);

                let mesh = mesh_chunk(&chunk, coord.0, coord.1, |world_x, world_y, world_z| {
                    world_y >= 0
                        && world_y <= generator.height_at(world_x, world_z).clamp(0, CHUNK_SIZE - 1)
                });

                let _ = tx.send(GenerationResult { coord, pool_slot, chunk, mesh });
            });
        }

        self.update_visibility(frustum, queue, renderer);
    }

    fn update_visibility(&mut self, frustum: &Frustum, queue: &wgpu::Queue, renderer: &mut ChunkRenderer) {
        let visible_coords: Vec<ChunkCoord> = self
            .loaded
            .par_iter()
            .filter_map(|(coord, _)| {
                let min =
                    glam::Vec3::new((coord.0 * CHUNK_SIZE) as f32, 0.0, (coord.1 * CHUNK_SIZE) as f32);
                let max = min + glam::Vec3::splat(CHUNK_SIZE as f32);
                frustum.intersects_aabb(min, max).then_some(*coord)
            })
            .collect();

        self.visible_handles.clear();
        for coord in &visible_coords {
            if let Some(loaded) = self.loaded.get(coord) {
                self.visible_handles.push(loaded.gpu_handle);
            }
        }
        self.visible_count = self.visible_handles.len();

        renderer.set_visible(queue, &self.visible_handles);
    }

    fn apply_completed_generations(&mut self, queue: &wgpu::Queue, renderer: &mut ChunkRenderer) {
        while let Ok(result) = self.result_rx.try_recv() {
            self.in_flight.remove(&result.coord);

            let origin = glam::Vec3::new(
                (result.coord.0 * CHUNK_SIZE) as f32,
                0.0,
                (result.coord.1 * CHUNK_SIZE) as f32,
            );

            let gpu_handle = renderer.alloc_chunk(queue, &result.mesh, origin);

            self.pool[result.pool_slot] = Some(result.chunk);
            self.loaded.insert(
                result.coord,
                LoadedChunk { pool_slot: result.pool_slot, gpu_handle },
            );
        }
    }

    fn unload_chunk(&mut self, coord: ChunkCoord, renderer: &mut ChunkRenderer) {
        let Some(loaded) = self.loaded.remove(&coord) else {
            return;
        };

        renderer.free_chunk(&loaded.gpu_handle);

        if let Some(mut chunk) = self.pool[loaded.pool_slot].take() {
            chunk.clear();
            self.pool[loaded.pool_slot] = Some(chunk);
        }
        self.pool_free_list.push(loaded.pool_slot);
    }
}
