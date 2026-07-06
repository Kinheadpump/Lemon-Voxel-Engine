use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};

use rayon::prelude::*;

use crate::engine::config::EngineConfig;
use crate::engine::core::mesher::{DirectionalMesh, mesh_chunk};
use crate::engine::render::renderer::{ChunkGpuHandle, ChunkRenderer};
use crate::game::math::frustum::Frustum;

use super::chunk::{CHUNK_SIZE, Chunk};
use super::generator::TerrainGenerator;

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
    desired_scratch: HashSet<ChunkCoord>,
    unload_scratch: Vec<ChunkCoord>,
    visible_coords_scratch: Vec<ChunkCoord>,
}

impl ChunkManager {
    /// Deckt bei `chunk_pool_size = 4300` eine Render-Distanz bis 32 ab ((2*32+1)^2 = 4225 Chunks)
    /// mit Reserve. Jeder Chunk belegt 64 KiB RAM.
    pub fn new(config: &EngineConfig) -> Self {
        let pool = (0..config.chunk_pool_size).map(|_| Some(Chunk::empty())).collect();
        let pool_free_list = (0..config.chunk_pool_size).collect();
        let (result_tx, result_rx) = channel();

        Self {
            pool,
            pool_free_list,
            loaded: HashMap::new(),
            in_flight: HashSet::new(),
            generator: Arc::new(TerrainGenerator::new(config)),
            result_tx,
            result_rx,
            render_distance_chunks: config.render_distance_chunks,
            visible_handles: Vec::new(),
            visible_count: 0,
            desired_scratch: HashSet::new(),
            unload_scratch: Vec::new(),
            visible_coords_scratch: Vec::new(),
        }
    }

    pub fn loaded_chunk_count(&self) -> usize {
        self.loaded.len()
    }

    pub fn visible_chunk_count(&self) -> usize {
        self.visible_count
    }

    pub fn generator(&self) -> &Arc<TerrainGenerator> {
        &self.generator
    }

    pub fn update(
        &mut self,
        camera_position: glam::Vec3,
        frustum: &Frustum,
        queue: &wgpu::Queue,
        renderer: &mut ChunkRenderer,
    ) {
        let center_x = (camera_position.x / CHUNK_SIZE as f32).floor() as i32;
        let center_z = (camera_position.z / CHUNK_SIZE as f32).floor() as i32;

        self.apply_completed_generations(center_x, center_z, queue, renderer);

        self.desired_scratch.clear();
        for dz in -self.render_distance_chunks..=self.render_distance_chunks {
            for dx in -self.render_distance_chunks..=self.render_distance_chunks {
                self.desired_scratch.insert((center_x + dx, center_z + dz));
            }
        }

        self.unload_scratch.clear();
        self.unload_scratch.extend(
            self.loaded.keys().copied().filter(|coord| !self.desired_scratch.contains(coord)),
        );
        while let Some(coord) = self.unload_scratch.pop() {
            self.unload_chunk(coord, renderer);
        }

        for coord in &self.desired_scratch {
            let coord = *coord;
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
                    generator.is_solid(world_x, world_y, world_z)
                });

                let _ = tx.send(GenerationResult { coord, pool_slot, chunk, mesh });
            });
        }

        self.update_visibility(frustum, queue, renderer);
    }

    fn update_visibility(&mut self, frustum: &Frustum, queue: &wgpu::Queue, renderer: &mut ChunkRenderer) {
        self.visible_coords_scratch.clear();
        self.visible_coords_scratch.par_extend(self.loaded.par_iter().filter_map(|(coord, _)| {
            let min = glam::Vec3::new((coord.0 * CHUNK_SIZE) as f32, 0.0, (coord.1 * CHUNK_SIZE) as f32);
            let max = min + glam::Vec3::splat(CHUNK_SIZE as f32);
            frustum.intersects_aabb(min, max).then_some(*coord)
        }));

        self.visible_handles.clear();
        for coord in &self.visible_coords_scratch {
            if let Some(loaded) = self.loaded.get(coord) {
                self.visible_handles.push(loaded.gpu_handle);
            }
        }
        self.visible_count = self.visible_handles.len();

        renderer.set_visible(queue, &self.visible_handles);
    }

    /// Soft-Cancellation: Ein auf dem Rayon-Thread fertiggestellter Chunk wird verworfen, wenn die
    /// Kamera sich waehrend der Generierungszeit bereits so weit wegbewegt hat, dass der Chunk
    /// nicht mehr innerhalb der Render-Distanz liegt - so entsteht kein unnoetiger GPU-Upload fuer
    /// Chunks, die im selben Frame wieder entladen wuerden.
    fn apply_completed_generations(
        &mut self,
        center_x: i32,
        center_z: i32,
        queue: &wgpu::Queue,
        renderer: &mut ChunkRenderer,
    ) {
        while let Ok(result) = self.result_rx.try_recv() {
            self.in_flight.remove(&result.coord);

            let still_in_range = (result.coord.0 - center_x).abs() <= self.render_distance_chunks
                && (result.coord.1 - center_z).abs() <= self.render_distance_chunks;

            if !still_in_range {
                self.pool[result.pool_slot] = Some(result.chunk);
                self.pool_free_list.push(result.pool_slot);
                continue;
            }

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
