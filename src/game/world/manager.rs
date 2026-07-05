use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};

use crate::engine::core::mesher::{DirectionalMesh, mesh_chunk};
use crate::engine::render::renderer::{ChunkRenderer, GPU_RENDER_SLOTS};

use super::chunk::{CHUNK_SIZE, Chunk};
use super::generator::TerrainGenerator;

pub const POOL_SIZE: usize = 1000;
pub const RENDER_DISTANCE_CHUNKS: i32 = 4;

type ChunkCoord = (i32, i32);

struct LoadedChunk {
    pool_slot: usize,
    gpu_slot: usize,
}

struct GenerationResult {
    coord: ChunkCoord,
    pool_slot: usize,
    gpu_slot: usize,
    chunk: Chunk,
    mesh: DirectionalMesh,
}

pub struct ChunkManager {
    pool: Vec<Option<Chunk>>,
    pool_free_list: Vec<usize>,
    gpu_free_list: Vec<usize>,
    loaded: HashMap<ChunkCoord, LoadedChunk>,
    in_flight: HashSet<ChunkCoord>,
    generator: Arc<TerrainGenerator>,
    result_tx: Sender<GenerationResult>,
    result_rx: Receiver<GenerationResult>,
}

impl ChunkManager {
    pub fn new() -> Self {
        let pool = (0..POOL_SIZE).map(|_| Some(Chunk::empty())).collect();
        let pool_free_list = (0..POOL_SIZE).collect();
        let gpu_free_list = (0..GPU_RENDER_SLOTS).collect();
        let (result_tx, result_rx) = channel();

        Self {
            pool,
            pool_free_list,
            gpu_free_list,
            loaded: HashMap::new(),
            in_flight: HashSet::new(),
            generator: Arc::new(TerrainGenerator::new()),
            result_tx,
            result_rx,
        }
    }

    pub fn loaded_chunk_count(&self) -> usize {
        self.loaded.len()
    }

    pub fn update(
        &mut self,
        camera_position: glam::Vec3,
        queue: &wgpu::Queue,
        renderer: &ChunkRenderer,
    ) {
        self.apply_completed_generations(queue, renderer);

        let center_x = (camera_position.x / CHUNK_SIZE as f32).floor() as i32;
        let center_z = (camera_position.z / CHUNK_SIZE as f32).floor() as i32;

        let mut desired = HashSet::new();
        for dz in -RENDER_DISTANCE_CHUNKS..=RENDER_DISTANCE_CHUNKS {
            for dx in -RENDER_DISTANCE_CHUNKS..=RENDER_DISTANCE_CHUNKS {
                desired.insert((center_x + dx, center_z + dz));
            }
        }

        let to_unload: Vec<ChunkCoord> =
            self.loaded.keys().copied().filter(|coord| !desired.contains(coord)).collect();
        for coord in to_unload {
            self.unload_chunk(coord, queue, renderer);
        }

        for coord in desired {
            if self.loaded.contains_key(&coord) || self.in_flight.contains(&coord) {
                continue;
            }

            let Some(pool_slot) = self.pool_free_list.pop() else {
                continue;
            };
            let Some(gpu_slot) = self.gpu_free_list.pop() else {
                self.pool_free_list.push(pool_slot);
                continue;
            };

            let mut chunk = self.pool[pool_slot].take().expect("Pool-Slot bereits leer");
            self.in_flight.insert(coord);

            let generator = Arc::clone(&self.generator);
            let tx = self.result_tx.clone();

            rayon::spawn(move || {
                generator.generate_chunk(coord.0, coord.1, &mut chunk);
                let mesh = mesh_chunk(&chunk);
                let _ = tx.send(GenerationResult { coord, pool_slot, gpu_slot, chunk, mesh });
            });
        }
    }

    fn apply_completed_generations(&mut self, queue: &wgpu::Queue, renderer: &ChunkRenderer) {
        while let Ok(result) = self.result_rx.try_recv() {
            self.in_flight.remove(&result.coord);

            let origin = glam::Vec3::new(
                (result.coord.0 * CHUNK_SIZE) as f32,
                0.0,
                (result.coord.1 * CHUNK_SIZE) as f32,
            );
            renderer.upload_chunk(queue, result.gpu_slot, &result.mesh, origin);

            let total_faces: usize = result.mesh.faces.iter().map(|f| f.len()).sum();
            log::debug!(
                "Chunk {:?} hochgeladen: slot={} origin={:?} faces={:?} total={}",
                result.coord,
                result.gpu_slot,
                origin,
                result.mesh.faces.iter().map(|f| f.len()).collect::<Vec<_>>(),
                total_faces
            );

            self.pool[result.pool_slot] = Some(result.chunk);
            self.loaded.insert(
                result.coord,
                LoadedChunk { pool_slot: result.pool_slot, gpu_slot: result.gpu_slot },
            );
        }
    }

    fn unload_chunk(&mut self, coord: ChunkCoord, queue: &wgpu::Queue, renderer: &ChunkRenderer) {
        let Some(loaded) = self.loaded.remove(&coord) else {
            return;
        };

        if let Some(mut chunk) = self.pool[loaded.pool_slot].take() {
            chunk.clear();
            self.pool[loaded.pool_slot] = Some(chunk);
        }
        self.pool_free_list.push(loaded.pool_slot);

        renderer.clear_slot(queue, loaded.gpu_slot);
        self.gpu_free_list.push(loaded.gpu_slot);
    }
}

impl Default for ChunkManager {
    fn default() -> Self {
        Self::new()
    }
}
