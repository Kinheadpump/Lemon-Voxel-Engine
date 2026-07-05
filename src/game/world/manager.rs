use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};

use rayon::prelude::*;

use crate::engine::core::mesher::{DirectionalMesh, mesh_chunk};
use crate::game::math::frustum::Frustum;

use super::chunk::{CHUNK_SIZE, Chunk};
use super::generator::TerrainGenerator;

pub const POOL_SIZE: usize = 1000;

type ChunkCoord = (i32, i32);

struct LoadedChunk {
    pool_slot: usize,
    mesh: DirectionalMesh,
    origin: glam::Vec3,
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
    visible_coords: Vec<ChunkCoord>,
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
            visible_coords: Vec::new(),
        }
    }

    pub fn loaded_chunk_count(&self) -> usize {
        self.loaded.len()
    }

    pub fn visible_chunk_count(&self) -> usize {
        self.visible_coords.len()
    }

    /// Meshes aller aktuell sichtbaren (frustum-getesteten) Chunks, fuer die Kompaktierung
    /// in den Renderer-Frame-Buffer.
    pub fn visible_chunks(&self) -> impl Iterator<Item = (&DirectionalMesh, glam::Vec3)> {
        self.visible_coords
            .iter()
            .filter_map(move |coord| self.loaded.get(coord).map(|loaded| (&loaded.mesh, loaded.origin)))
    }

    pub fn update(&mut self, camera_position: glam::Vec3, frustum: &Frustum) {
        self.apply_completed_generations();

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
            self.unload_chunk(coord);
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

        self.update_visibility(frustum);
    }

    fn update_visibility(&mut self, frustum: &Frustum) {
        self.visible_coords = self
            .loaded
            .par_iter()
            .filter_map(|(coord, _)| {
                let min =
                    glam::Vec3::new((coord.0 * CHUNK_SIZE) as f32, 0.0, (coord.1 * CHUNK_SIZE) as f32);
                let max = min + glam::Vec3::splat(CHUNK_SIZE as f32);
                frustum.intersects_aabb(min, max).then_some(*coord)
            })
            .collect();
    }

    fn apply_completed_generations(&mut self) {
        while let Ok(result) = self.result_rx.try_recv() {
            self.in_flight.remove(&result.coord);

            let origin = glam::Vec3::new(
                (result.coord.0 * CHUNK_SIZE) as f32,
                0.0,
                (result.coord.1 * CHUNK_SIZE) as f32,
            );

            let total_faces: usize = result.mesh.faces.iter().map(|f| f.len()).sum();
            log::debug!(
                "Chunk {:?} gemesht: faces={:?} total={}",
                result.coord,
                result.mesh.faces.iter().map(|f| f.len()).collect::<Vec<_>>(),
                total_faces
            );

            self.pool[result.pool_slot] = Some(result.chunk);
            self.loaded.insert(
                result.coord,
                LoadedChunk { pool_slot: result.pool_slot, mesh: result.mesh, origin },
            );
        }
    }

    fn unload_chunk(&mut self, coord: ChunkCoord) {
        let Some(loaded) = self.loaded.remove(&coord) else {
            return;
        };

        if let Some(mut chunk) = self.pool[loaded.pool_slot].take() {
            chunk.clear();
            self.pool[loaded.pool_slot] = Some(chunk);
        }
        self.pool_free_list.push(loaded.pool_slot);
    }
}
