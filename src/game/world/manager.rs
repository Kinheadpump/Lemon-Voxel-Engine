use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};

use glam::IVec3;
use rayon::prelude::*;

use crate::engine::config::EngineConfig;
use crate::engine::core::mesher::{DirectionalMesh, mesh_chunk};
use crate::engine::render::renderer::{ChunkGpuHandle, ChunkRenderer};
use crate::game::math::frustum::Frustum;

use super::chunk::{CHUNK_SIZE, Chunk};
use super::generator::TerrainGenerator;
use super::raycast::{RaycastHit, raycast};

/// Maximale Raycast-Reichweite fuer Abbauen/Platzieren.
pub const INTERACTION_REACH: f32 = 6.0;

/// (chunk_x, chunk_y, chunk_z) - Y ist Teil der Chunk-Koordinate, damit Terrain vertikal ueber
/// beliebig viele Chunks gestapelt werden kann statt in eine einzelne Schicht oder einen festen
/// Hoehenbereich gezwungen zu sein.
type ChunkCoord = (i32, i32, i32);

struct LoadedChunk {
    pool_slot: usize,
    gpu_handle: ChunkGpuHandle,
    /// Reiner Luft-Chunk (z.B. weit oberhalb des Terrains) - wird aus der Sichtbarkeitspruefung
    /// ausgeklammert, damit die pro-Frame-Kosten mit der tatsaechlich sichtbaren Geometrie skalieren
    /// statt mit der (bei vertikal unbegrenzter Welt potenziell riesigen) Gesamtzahl geladener Chunks.
    is_empty: bool,
}

struct GenerationResult {
    coord: ChunkCoord,
    pool_slot: usize,
    chunk: Chunk,
    mesh: DirectionalMesh,
    is_empty: bool,
}

fn chunk_origin(coord: ChunkCoord) -> glam::Vec3 {
    glam::Vec3::new((coord.0 * CHUNK_SIZE) as f32, (coord.1 * CHUNK_SIZE) as f32, (coord.2 * CHUNK_SIZE) as f32)
}

/// Zerlegt eine Weltkoordinate in Chunk-Koordinate + lokale Blockkoordinate. `div_euclid`/
/// `rem_euclid` runden zum negativen Unendlichen statt zur Null, das ist bei negativen
/// Weltkoordinaten (jenseits des Ursprungs) der einzig korrekte Umrechnungsweg.
fn chunk_and_local(world_x: i32, world_y: i32, world_z: i32) -> (ChunkCoord, IVec3) {
    let coord = (world_x.div_euclid(CHUNK_SIZE), world_y.div_euclid(CHUNK_SIZE), world_z.div_euclid(CHUNK_SIZE));
    let local = IVec3::new(
        world_x.rem_euclid(CHUNK_SIZE),
        world_y.rem_euclid(CHUNK_SIZE),
        world_z.rem_euclid(CHUNK_SIZE),
    );
    (coord, local)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_world_coords_convert_correctly() {
        let (coord, local) = chunk_and_local(33, 5, 65);
        assert_eq!(coord, (1, 0, 2));
        assert_eq!(local, IVec3::new(1, 5, 1));
    }

    #[test]
    fn negative_world_coords_floor_towards_negative_infinity() {
        let (coord, local) = chunk_and_local(-1, -1, -33);
        assert_eq!(coord, (-1, -1, -2));
        assert_eq!(local, IVec3::new(31, 31, 31));
    }

    #[test]
    fn chunk_origin_matches_chunk_coord_times_size() {
        assert_eq!(chunk_origin((1, -1, 2)), glam::Vec3::new(32.0, -32.0, 64.0));
    }
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
    vertical_render_distance_chunks: i32,
    visible_handles: Vec<ChunkGpuHandle>,
    visible_count: usize,
    /// Chunk-Koordinate der Kamera im letzten Frame, in dem das Ladefenster neu aufgebaut wurde.
    /// Der komplette Soll/Entlade-Scan (O(Ladevolumen)) muss nur laufen, wenn sich diese
    /// Mittelpunkt-Koordinate tatsaechlich aendert - im Stillstand (oder bei reiner Kamerarotation)
    /// entstehen sonst pro Frame tausende ueberfluessige HashSet-Operationen.
    last_center: Option<ChunkCoord>,
    desired_scratch: HashSet<ChunkCoord>,
    unload_scratch: Vec<ChunkCoord>,
    /// Noch nicht dispatchte, aber gewuenschte Chunks (neu ins Ladefenster gerutscht oder wegen
    /// Pool-Erschoepfung zurueckgestellt). Wird pro Frame nur um tatsaechlich neue Arbeit erweitert
    /// bzw. abgearbeitet - kein Full-Rescan des gesamten Ladevolumens mehr pro Frame.
    pending_scratch: Vec<ChunkCoord>,
    pending_set: HashSet<ChunkCoord>,
    visible_coords_scratch: Vec<ChunkCoord>,
}

impl ChunkManager {
    /// Der Pool muss `(2*render_distance_chunks+1)^2 * (2*vertical_render_distance_chunks+1)`
    /// Chunks abdecken, sonst werden Chunks am Rand der Render-Distanz stillschweigend nicht
    /// geladen (siehe `chunk_pool_size`-Kommentar in `EngineConfig`). Jeder Chunk belegt 64 KiB RAM.
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
            vertical_render_distance_chunks: config.vertical_render_distance_chunks,
            visible_handles: Vec::new(),
            visible_count: 0,
            last_center: None,
            desired_scratch: HashSet::new(),
            unload_scratch: Vec::new(),
            pending_scratch: Vec::new(),
            pending_set: HashSet::new(),
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

    /// Voxel-Festigkeit unter Beruecksichtigung geladener/editierter Chunk-Daten. Ist der Chunk an
    /// dieser Position nicht geladen, wird auf die rein prozedurale Vorhersage zurueckgefallen -
    /// das reicht fuer Physik/Raycast, da beide ohnehin nur innerhalb der Render-Distanz abfragen.
    pub fn is_solid_at(&self, world_x: i32, world_y: i32, world_z: i32) -> bool {
        let (coord, local) = chunk_and_local(world_x, world_y, world_z);
        if let Some(loaded) = self.loaded.get(&coord) {
            if let Some(chunk) = &self.pool[loaded.pool_slot] {
                return chunk.get_block(local.x, local.y, local.z) != 0;
            }
        }
        self.generator.is_solid(world_x, world_y, world_z)
    }

    pub fn raycast(&self, origin: glam::Vec3, direction: glam::Vec3, max_distance: f32) -> Option<RaycastHit> {
        raycast(origin, direction, max_distance, |x, y, z| self.is_solid_at(x, y, z))
    }

    /// Setzt einen Block in Weltkoordinaten und meshed den betroffenen Chunk (und alle Nachbarn,
    /// deren Randflaechen von diesem Block abhaengen) synchron neu. Liegt der Zielchunk nicht
    /// geladen vor, ist die Position ausserhalb der Reichweite des Spielers und wird ignoriert.
    pub fn set_block(
        &mut self,
        world_x: i32,
        world_y: i32,
        world_z: i32,
        block_id: u16,
        queue: &wgpu::Queue,
        renderer: &mut ChunkRenderer,
    ) -> bool {
        let (coord, local) = chunk_and_local(world_x, world_y, world_z);
        let Some(pool_slot) = self.loaded.get(&coord).map(|loaded| loaded.pool_slot) else {
            return false;
        };
        let Some(chunk) = self.pool[pool_slot].as_mut() else {
            return false;
        };
        chunk.set_block(local.x, local.y, local.z, block_id);

        self.remesh_chunk(coord, queue, renderer);
        if local.x == 0 {
            self.remesh_chunk((coord.0 - 1, coord.1, coord.2), queue, renderer);
        }
        if local.x == CHUNK_SIZE - 1 {
            self.remesh_chunk((coord.0 + 1, coord.1, coord.2), queue, renderer);
        }
        if local.y == 0 {
            self.remesh_chunk((coord.0, coord.1 - 1, coord.2), queue, renderer);
        }
        if local.y == CHUNK_SIZE - 1 {
            self.remesh_chunk((coord.0, coord.1 + 1, coord.2), queue, renderer);
        }
        if local.z == 0 {
            self.remesh_chunk((coord.0, coord.1, coord.2 - 1), queue, renderer);
        }
        if local.z == CHUNK_SIZE - 1 {
            self.remesh_chunk((coord.0, coord.1, coord.2 + 1), queue, renderer);
        }
        true
    }

    fn remesh_chunk(&mut self, coord: ChunkCoord, queue: &wgpu::Queue, renderer: &mut ChunkRenderer) {
        let Some(loaded) = self.loaded.get(&coord) else {
            return;
        };
        let pool_slot = loaded.pool_slot;
        let old_handle = loaded.gpu_handle;

        let Some(chunk) = self.pool[pool_slot].as_ref() else {
            return;
        };
        let is_empty = chunk.is_empty();
        let mesh = if is_empty {
            DirectionalMesh::default()
        } else {
            mesh_chunk(chunk, coord.0, coord.1, coord.2, |world_x, world_y, world_z| {
                self.is_solid_at(world_x, world_y, world_z)
            })
        };

        renderer.free_chunk(&old_handle);
        let new_handle = renderer.alloc_chunk(queue, &mesh, chunk_origin(coord));

        if let Some(loaded) = self.loaded.get_mut(&coord) {
            loaded.gpu_handle = new_handle;
            loaded.is_empty = is_empty;
        }
    }

    pub fn update(
        &mut self,
        camera_position: glam::Vec3,
        frustum: &Frustum,
        queue: &wgpu::Queue,
        renderer: &mut ChunkRenderer,
    ) {
        let center = (
            (camera_position.x / CHUNK_SIZE as f32).floor() as i32,
            (camera_position.y / CHUNK_SIZE as f32).floor() as i32,
            (camera_position.z / CHUNK_SIZE as f32).floor() as i32,
        );

        self.apply_completed_generations(center, queue, renderer);

        if self.last_center != Some(center) {
            self.rebuild_load_window(center, renderer);
            self.last_center = Some(center);
        }

        self.dispatch_pending();
        self.update_visibility(frustum, queue, renderer);
    }

    /// Baut Soll-Menge, Entlade-Liste und die Liste neu zu ladender Chunks komplett neu auf - nur
    /// noetig, wenn der Kamera-Mittelpunkt tatsaechlich einen neuen Chunk betreten hat.
    fn rebuild_load_window(&mut self, center: ChunkCoord, renderer: &mut ChunkRenderer) {
        self.desired_scratch.clear();
        for dz in -self.render_distance_chunks..=self.render_distance_chunks {
            for dx in -self.render_distance_chunks..=self.render_distance_chunks {
                for dy in -self.vertical_render_distance_chunks..=self.vertical_render_distance_chunks {
                    self.desired_scratch.insert((center.0 + dx, center.1 + dy, center.2 + dz));
                }
            }
        }

        self.unload_scratch.clear();
        self.unload_scratch.extend(
            self.loaded.keys().copied().filter(|coord| !self.desired_scratch.contains(coord)),
        );
        while let Some(coord) = self.unload_scratch.pop() {
            self.unload_chunk(coord, renderer);
        }

        // Chunks, die aus dem Fenster gewandert sind, bevor sie je dispatcht wurden, muessen aus
        // der Pending-Liste verschwinden - sonst wuerde spaeter fuer eine laengst irrelevante
        // Position noch ein Generierungs-Job gestartet.
        let desired = &self.desired_scratch;
        self.pending_set.retain(|coord| desired.contains(coord));
        self.pending_scratch.retain(|coord| self.pending_set.contains(coord));

        for &coord in &self.desired_scratch {
            if self.loaded.contains_key(&coord) || self.in_flight.contains(&coord) {
                continue;
            }
            if self.pending_set.insert(coord) {
                self.pending_scratch.push(coord);
            }
        }
    }

    /// Dispatcht so viele ausstehende Chunks wie der Pool hergibt. Laeuft jeden Frame, kostet im
    /// eingeschwungenen Zustand (nichts mehr ausstehend) aber O(1) statt O(Ladevolumen).
    fn dispatch_pending(&mut self) {
        while let Some(coord) = self.pending_scratch.pop() {
            self.pending_set.remove(&coord);

            if self.loaded.contains_key(&coord) || self.in_flight.contains(&coord) {
                continue;
            }

            let Some(pool_slot) = self.pool_free_list.pop() else {
                self.pending_set.insert(coord);
                self.pending_scratch.push(coord);
                break;
            };

            let mut chunk = self.pool[pool_slot].take().expect("Pool-Slot bereits leer");
            self.in_flight.insert(coord);

            let generator = Arc::clone(&self.generator);
            let tx = self.result_tx.clone();

            rayon::spawn(move || {
                generator.generate_chunk(coord.0, coord.1, coord.2, &mut chunk);

                // Reine Luft-Chunks (z.B. weit oberhalb des Terrains) erzeugen ohnehin keine
                // Faces - das teure Greedy-Meshing (6 Richtungen * 32 Ebenen) lohnt sich dafuer
                // nicht.
                let is_empty = chunk.is_empty();
                let mesh = if is_empty {
                    DirectionalMesh::default()
                } else {
                    mesh_chunk(&chunk, coord.0, coord.1, coord.2, |world_x, world_y, world_z| {
                        generator.is_solid(world_x, world_y, world_z)
                    })
                };

                let _ = tx.send(GenerationResult { coord, pool_slot, chunk, mesh, is_empty });
            });
        }
    }

    fn update_visibility(&mut self, frustum: &Frustum, queue: &wgpu::Queue, renderer: &mut ChunkRenderer) {
        self.visible_coords_scratch.clear();
        self.visible_coords_scratch.par_extend(self.loaded.par_iter().filter_map(|(coord, loaded)| {
            if loaded.is_empty {
                return None;
            }
            let min = chunk_origin(*coord);
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
    fn apply_completed_generations(&mut self, center: ChunkCoord, queue: &wgpu::Queue, renderer: &mut ChunkRenderer) {
        while let Ok(result) = self.result_rx.try_recv() {
            self.in_flight.remove(&result.coord);

            let still_in_range = (result.coord.0 - center.0).abs() <= self.render_distance_chunks
                && (result.coord.1 - center.1).abs() <= self.vertical_render_distance_chunks
                && (result.coord.2 - center.2).abs() <= self.render_distance_chunks;

            if !still_in_range {
                self.pool[result.pool_slot] = Some(result.chunk);
                self.pool_free_list.push(result.pool_slot);
                continue;
            }

            let gpu_handle = renderer.alloc_chunk(queue, &result.mesh, chunk_origin(result.coord));

            self.pool[result.pool_slot] = Some(result.chunk);
            self.loaded.insert(
                result.coord,
                LoadedChunk { pool_slot: result.pool_slot, gpu_handle, is_empty: result.is_empty },
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
