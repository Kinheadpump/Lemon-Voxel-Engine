use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};

use glam::IVec3;

use crate::engine::config::{EngineConfig, LodRingRuntime};
use crate::engine::core::mesher::{DirectionalMesh, NEIGHBOR_OFFSETS, mesh_chunk};
use crate::engine::render::renderer::{ChunkGpuHandle, ChunkRenderer};
use crate::game::math::cascades::{Cascade, MAX_SHADOW_CASCADES};

use super::blocks;
use super::chunk::{CHUNK_SIZE, Chunk};
use super::generator::TerrainGenerator;
use super::raycast::{RaycastHit, raycast};

/// Maximale Raycast-Reichweite fuer Abbauen/Platzieren.
pub const INTERACTION_REACH: f32 = 6.0;

/// (chunk_x, chunk_y, chunk_z) - Y ist Teil der Chunk-Koordinate, damit Terrain vertikal ueber
/// beliebig viele Chunks gestapelt werden kann statt in eine einzelne Schicht oder einen festen
/// Hoehenbereich gezwungen zu sein.
type ChunkCoord = (i32, i32, i32);

/// FxHash-artiger multiplikativer Hasher fuer ChunkCoord-Keys. Der SipHash-Default von
/// `std::collections::HashMap` ist DoS-resistent, aber fuer interne (nicht angreifbare)
/// Koordinaten-Keys unnoetig teuer - `loaded`/`pending_set`/`in_flight` werden im Streaming-Pfad
/// zehntausendfach pro Ladefenster-Rebuild und pro `is_solid_at`-Voxelabfrage (Physik/Raycast)
/// getroffen.
#[derive(Default)]
struct CoordHasher(u64);

impl Hasher for CoordHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 = (self.0 ^ u64::from(byte)).wrapping_mul(0x0100_0000_01B3);
        }
    }

    #[inline]
    fn write_i32(&mut self, value: i32) {
        self.0 = (self.0.rotate_left(5) ^ u64::from(value as u32)).wrapping_mul(0x517C_C1B7_2722_0A95);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

type CoordMap<V> = HashMap<ChunkCoord, V, BuildHasherDefault<CoordHasher>>;
type CoordSet = HashSet<ChunkCoord, BuildHasherDefault<CoordHasher>>;

struct LoadedChunk {
    pool_slot: usize,
    gpu_handle: ChunkGpuHandle,
    /// Reiner Luft-Chunk (z.B. weit oberhalb des Terrains) - wird aus der Sichtbarkeitspruefung
    /// ausgeklammert, damit die pro-Frame-Kosten mit der tatsaechlich sichtbaren Geometrie skalieren
    /// statt mit der (bei vertikal unbegrenzter Welt potenziell riesigen) Gesamtzahl geladener Chunks.
    is_empty: bool,
    /// Bereits in `unload_scratch` eingereiht (aber noch nicht abgearbeitet). Verhindert, dass
    /// derselbe Chunk bei mehreren Fenster-Rebuilds vor dem gedeckelten Entladen mehrfach in die
    /// Warteschlange gelegt wird.
    queued_for_unload: bool,
}

/// `chunk` ist geboxt, damit ueber den gesamten asynchronen Pfad (Pool-Take -> rayon-Closure ->
/// mpsc-Send -> Pool-Rueckgabe) nur ein 8-Byte-Pointer wandert. Mit Inline-`Chunk` wurde der volle
/// 64-KiB-Block VIERMAL pro Chunk kopiert und die geboxte rayon-Closure sowie der mpsc-Node wurden
/// zu 64-KiB-Heap-Allokationen auf dem Main-Thread - bei 128 Dispatches+Uploads/Frame ~24 MB
/// Alloc-/Memcpy-Traffic pro Frame waehrend des Streamens (Frame-Stutter).
struct GenerationResult {
    coord: ChunkCoord,
    pool_slot: usize,
    chunk: Box<Chunk>,
    mesh: DirectionalMesh,
    is_empty: bool,
}

/// Welt-Ursprung eines Chunks IN DIESEM Ring - `voxel_scale=1` fuer LOD0 (identisch zum bisherigen
/// Verhalten), bei LOD-Ringen deckt sowohl die Chunk-Koordinate als auch jeder lokale Schritt
/// `voxel_scale` Weltbloecke ab (s. `generator/lod.rs`).
fn chunk_origin_scaled(coord: ChunkCoord, voxel_scale: i32) -> glam::Vec3 {
    let step = (CHUNK_SIZE * voxel_scale) as f32;
    glam::Vec3::new(coord.0 as f32 * step, coord.1 as f32 * step, coord.2 as f32 * step)
}

fn sphere_intersects_aabb(center: glam::Vec3, radius: f32, min: glam::Vec3, max: glam::Vec3) -> bool {
    let closest = center.clamp(min, max);
    center.distance_squared(closest) <= radius * radius
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

/// Wie viele Generierungs-Jobs gleichzeitig auf dem Rayon-Pool sitzen duerfen, bevor
/// `dispatch_pending` weitere zurueckhaelt - s. Kommentar dort. Ein kleines Vielfaches der
/// tatsaechlichen Worker-Anzahl gibt jedem Kern genug Nachschub, um nie leerzulaufen (Work-Stealing-
/// Overhead), ohne einen unbegrenzten Rueckstau zuzulassen.
fn max_in_flight_generations() -> usize {
    rayon::current_num_threads().max(1) * 2
}

/// Skaliert den per-Kaskade tolerierten Kugel-Versatz mit dem Kaskadenradius: die ferne Kaskade
/// wandert bei reiner Kamerarotation deutlich weiter als die nahe (ihr Zentrum liegt weit entlang
/// der Blickrichtung) - ein fixer kleiner Slack wuerde dort trotzdem jeden Frame einen Rebuild
/// ausloesen. Die Kugeln werden beim Sichtbarkeitstest um diesen Slack AUFGEBLAEHT, wodurch die
/// gecachte Menge eine korrekte OBERMENGE der exakten bleibt, solange sich kein Zentrum weiter als
/// den Slack bewegt hat (Dreiecksungleichung) - erst dann wird neu aufgebaut.
fn shadow_cascade_slack(radius: f32) -> f32 {
    (radius * 0.2).clamp(8.0, 64.0)
}

pub struct ChunkManager {
    /// Geboxte Chunks (s. `GenerationResult`) - `take()`/Zurueckstecken bewegt nur den Pointer.
    pool: Vec<Option<Box<Chunk>>>,
    pool_free_list: Vec<usize>,
    loaded: CoordMap<LoadedChunk>,
    in_flight: CoordSet,
    generator: Arc<TerrainGenerator>,
    result_tx: Sender<GenerationResult>,
    result_rx: Receiver<GenerationResult>,
    render_distance_chunks: i32,
    vertical_render_distance_chunks: i32,
    /// 1 fuer LOD0, sonst der Weltblock-Faktor dieses Rings (s. `generator/lod.rs`/
    /// `LodRingRuntime`) - bestimmt Generierungspfad (`generate_chunk` vs. `generate_lod_chunk`)
    /// UND die Chunk-Weltgroesse (`CHUNK_SIZE * voxel_scale`).
    voxel_scale: i32,
    /// Offset dieses Rings im GEMEINSAMEN `chunk_meta_buffer`-Adressraum des `ChunkRenderer` - lokale
    /// `pool`-Indices (0..pool_size) werden nur bei GPU-Aufrufen (`update_chunk_meta`/
    /// `clear_chunk_meta`) um diesen Wert verschoben, sonst bleibt der Ring komplett unabhaengig.
    pool_slot_base: usize,
    /// Fenster-Zentrum (Kamera-Chunk-Koordinate) des letzten Rebuilds. Ob eine Koordinate zum
    /// Ladefenster gehoert, ist ein O(1)-Arithmetik-Praedikat gegen dieses Zentrum
    /// (`Self::in_window`) - es existiert bewusst KEINE materialisierte "Soll-Menge" mehr: das
    /// fruehere `HashSet` mit O(Ladevolumen) Inserts pro Fenster-Rebuild (38k Hashes bei
    /// render_distance=32, jedes Mal beim Ueberqueren einer Chunk-Grenze) war die Haupt-Stutter-
    /// Quelle beim Bewegen.
    last_center: Option<ChunkCoord>,
    unload_scratch: Vec<ChunkCoord>,
    /// Noch nicht dispatchte, aber gewuenschte Chunks - absteigend nach Distanz zum Fenster-Zentrum
    /// sortiert, sodass `pop()` (Dispatch-Reihenfolge) immer den NAECHSTGELEGENEN Chunk zuerst
    /// liefert. Vorher war die Reihenfolge HashSet-Iterationszufall - sichtbar als "Chunks laden an
    /// zufaelligen Stellen zuerst".
    pending_scratch: Vec<ChunkCoord>,
    pending_set: CoordSet,
    shadow_visible_handles: Vec<(ChunkGpuHandle, glam::Vec3)>,
    /// Kaskaden-Kugeln (Zentrum, Radius) zum Zeitpunkt des letzten Schatten-Sichtbarkeits-Rebuilds
    /// plus Dirty-Flag: der fruehere Voll-Scan ueber ALLE geladenen Chunks lief jeden Frame (O(N)
    /// par_iter + kompletter Indirect-Buffer-Reupload), obwohl sich die Menge zwischen zwei Frames
    /// fast nie aendert. Jetzt laeuft er nur noch, wenn Chunks geladen/entladen/editiert wurden oder
    /// sich eine Kaskaden-Kugel weiter als ihren Slack bewegt hat (s. `shadow_cascade_slack`).
    shadow_last_cascades: [(glam::Vec3, f32); MAX_SHADOW_CASCADES],
    shadow_last_cascade_count: u32,
    shadow_set_dirty: bool,
    /// Frame-Budgets fuer `dispatch_pending`/`apply_completed_generations`/`drain_unloads` - siehe
    /// Kommentar an `EngineConfig::max_chunk_dispatches_per_frame`. Ohne diese Grenzen dispatcht/
    /// uploaded/entlaedt ein grosser Backlog (Welt-Start, schnelles Fliegen, vertikales Fallen)
    /// tausende Chunks in einem einzigen Frame.
    max_chunk_dispatches_per_frame: usize,
    max_chunk_uploads_per_frame: usize,
    max_chunk_unloads_per_frame: usize,
}

impl ChunkManager {
    /// `ring.pool_size` ist bereits in `EngineConfig::lod_ring_runtimes` auf das Ladevolumen DIESES
    /// Rings normalisiert, `ring.pool_slot_base` auf seinen Anteil am gemeinsamen
    /// `ChunkRenderer`-Adressraum - beide garantiert konsistent mit den Renderer-Buffern
    /// (`chunk_meta_buffer` etc.), die exakt `EngineConfig::total_chunk_pool_size` gross sind. Jeder
    /// Chunk belegt 64 KiB RAM, unabhaengig von `voxel_scale` (identische `[u16; 32768]`-Groesse).
    pub fn new(config: &EngineConfig, ring: &LodRingRuntime) -> Self {
        let pool_size = ring.pool_size;
        let pool = (0..pool_size).map(|_| Some(Box::new(Chunk::empty()))).collect();
        let pool_free_list = (0..pool_size).collect();
        let (result_tx, result_rx) = channel();

        Self {
            pool,
            pool_free_list,
            loaded: CoordMap::default(),
            in_flight: CoordSet::default(),
            generator: Arc::new(TerrainGenerator::new(config)),
            result_tx,
            result_rx,
            render_distance_chunks: ring.render_distance_chunks,
            vertical_render_distance_chunks: ring.vertical_render_distance_chunks,
            voxel_scale: ring.voxel_scale,
            pool_slot_base: ring.pool_slot_base,
            last_center: None,
            unload_scratch: Vec::new(),
            pending_scratch: Vec::new(),
            pending_set: CoordSet::default(),
            shadow_visible_handles: Vec::new(),
            shadow_last_cascades: [(glam::Vec3::ZERO, 0.0); MAX_SHADOW_CASCADES],
            shadow_last_cascade_count: 0,
            shadow_set_dirty: true,
            max_chunk_dispatches_per_frame: config.dev.max_chunk_dispatches_per_frame,
            max_chunk_uploads_per_frame: config.dev.max_chunk_uploads_per_frame,
            max_chunk_unloads_per_frame: config.dev.max_chunk_unloads_per_frame,
        }
    }

    pub fn loaded_chunk_count(&self) -> usize {
        self.loaded.len()
    }

    pub fn generator(&self) -> &Arc<TerrainGenerator> {
        &self.generator
    }

    #[inline(always)]
    fn chunk_origin(&self, coord: ChunkCoord) -> glam::Vec3 {
        chunk_origin_scaled(coord, self.voxel_scale)
    }

    #[inline(always)]
    fn in_window(&self, coord: ChunkCoord, center: ChunkCoord) -> bool {
        (coord.0 - center.0).abs() <= self.render_distance_chunks
            && (coord.1 - center.1).abs() <= self.vertical_render_distance_chunks
            && (coord.2 - center.2).abs() <= self.render_distance_chunks
    }

    /// PHYSIKALISCHE Voxel-Festigkeit (Kollision/Raycast) unter Beruecksichtigung geladener/
    /// editierter Chunk-Daten - Wasser ist begehbar/durchschwimmbar und zaehlt NICHT als solide
    /// (der Mesher nutzt fuer Okklusion stattdessen `TerrainGenerator::is_solid`, das Wasser als
    /// sichtbaren Block einschliesst). Ist der Chunk nicht geladen, faellt es auf die prozedurale
    /// Vorhersage zurueck - das reicht fuer Physik/Raycast, da beide ohnehin nur innerhalb der
    /// Render-Distanz abfragen.
    pub fn is_solid_at(&self, world_x: i32, world_y: i32, world_z: i32) -> bool {
        let (coord, local) = chunk_and_local(world_x, world_y, world_z);
        if let Some(loaded) = self.loaded.get(&coord)
            && let Some(chunk) = self.pool[loaded.pool_slot].as_deref()
        {
            let block = chunk.get_block(local.x, local.y, local.z);
            return block != 0 && block != blocks::WATER;
        }
        self.generator.is_physically_solid(world_x, world_y, world_z)
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
        let Some(chunk) = self.pool[pool_slot].as_deref_mut() else {
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

    /// Loest fuer alle 6 Richtungen die tatsaechlich geladenen Nachbar-Chunk-Referenzen auf - EINMAL
    /// vor dem Meshing statt einer HashMap-Lookup pro Rand-Voxel (bis zu 6144 pro Chunk). Nur im
    /// synchronen Remesh-Pfad (Block-Editierung) sinnvoll/moeglich: hier lebt `&self` lange genug,
    /// dass die Referenzen den kompletten `mesh_chunk`-Aufruf ueberleben. Der asynchrone
    /// Rayon-Dispatch-Pfad (`dispatch_pending`) kann das NICHT nutzen - die Referenzen wuerden die
    /// Thread-Grenze nicht ueberleben.
    fn neighbor_chunk_refs(&self, coord: ChunkCoord) -> [Option<&Chunk>; 6] {
        std::array::from_fn(|dir| {
            let (ox, oy, oz) = NEIGHBOR_OFFSETS[dir];
            let neighbor_coord = (coord.0 + ox, coord.1 + oy, coord.2 + oz);
            self.loaded.get(&neighbor_coord).and_then(|loaded| self.pool[loaded.pool_slot].as_deref())
        })
    }

    fn remesh_chunk(&mut self, coord: ChunkCoord, queue: &wgpu::Queue, renderer: &mut ChunkRenderer) {
        let Some(loaded) = self.loaded.get(&coord) else {
            return;
        };
        let pool_slot = loaded.pool_slot;
        let old_handle = loaded.gpu_handle;

        let Some(chunk) = self.pool[pool_slot].as_deref() else {
            return;
        };
        let is_empty = chunk.is_empty();
        let mesh = if is_empty {
            DirectionalMesh::default()
        } else {
            let neighbors = self.neighbor_chunk_refs(coord);
            mesh_chunk(chunk, coord.0, coord.1, coord.2, neighbors, |world_x, world_y, world_z| {
                self.is_solid_at(world_x, world_y, world_z)
            })
        };

        renderer.free_chunk(&old_handle);
        let new_handle = renderer.alloc_chunk(queue, &mesh);

        let gpu_pool_slot = self.pool_slot_base + pool_slot;
        if is_empty {
            renderer.clear_chunk_meta(queue, gpu_pool_slot);
        } else {
            let min = self.chunk_origin(coord);
            let max = min + glam::Vec3::splat((CHUNK_SIZE * self.voxel_scale) as f32);
            renderer.update_chunk_meta(queue, gpu_pool_slot, min, max, self.voxel_scale as f32, &new_handle);
        }

        if let Some(loaded) = self.loaded.get_mut(&coord) {
            loaded.gpu_handle = new_handle;
            loaded.is_empty = is_empty;
        }
        self.shadow_set_dirty = true;
    }

    /// Streaming-Update fuer EINEN Ring: Pool/Ladefenster/Dispatch/Entladen. Schatten-Sichtbarkeit
    /// ist bewusst NICHT Teil dieser Methode (s. `update_shadow_visibility`) - nur LOD0 wirft
    /// Schatten, der Aufrufer (`WorldStreamer`) ruft sie deshalb gezielt nur fuer Ring 0 auf.
    pub fn update(&mut self, camera_position: glam::Vec3, queue: &wgpu::Queue, renderer: &mut ChunkRenderer) {
        let step = CHUNK_SIZE * self.voxel_scale;
        let center = (
            (camera_position.x / step as f32).floor() as i32,
            (camera_position.y / step as f32).floor() as i32,
            (camera_position.z / step as f32).floor() as i32,
        );

        self.apply_completed_generations(center, queue, renderer);

        if self.last_center != Some(center) {
            self.rebuild_load_window(center);
            self.last_center = Some(center);
        }

        self.drain_unloads(center, queue, renderer);
        self.dispatch_pending();
    }

    /// Fenster-Rebuild beim Betreten eines neuen Kamera-Chunks. Neu zu ladende Chunks werden
    /// INKREMENTELL bestimmt: nur Koordinaten, die im neuen, aber nicht im alten Fenster liegen
    /// (beim typischen Ein-Chunk-Schritt eine einzelne Randebene statt des vollen Ladevolumens).
    /// Der Skip-Test gegen das alte Fenster ist reine Arithmetik - im Gegensatz zum frueheren
    /// HashSet-Aufbau fallen fuer die (bei render_distance=32 rund 38k) unveraenderten Koordinaten
    /// weder Hashes noch Inserts an. Der Entlade-Scan bleibt O(geladene Chunks), aber ebenfalls mit
    /// reinem Arithmetik-Praedikat.
    fn rebuild_load_window(&mut self, center: ChunkCoord) {
        let old_center = self.last_center;

        for (coord, loaded) in self.loaded.iter_mut() {
            let outside = (coord.0 - center.0).abs() > self.render_distance_chunks
                || (coord.1 - center.1).abs() > self.vertical_render_distance_chunks
                || (coord.2 - center.2).abs() > self.render_distance_chunks;
            if outside && !loaded.queued_for_unload {
                loaded.queued_for_unload = true;
                self.unload_scratch.push(*coord);
            }
        }

        // Chunks, die aus dem Fenster gewandert sind, bevor sie je dispatcht wurden, muessen aus
        // der Pending-Liste verschwinden - sonst wuerde spaeter fuer eine laengst irrelevante
        // Position noch ein Generierungs-Job gestartet.
        let r = self.render_distance_chunks;
        let rv = self.vertical_render_distance_chunks;
        self.pending_set.retain(|&(x, y, z)| {
            (x - center.0).abs() <= r && (y - center.1).abs() <= rv && (z - center.2).abs() <= r
        });
        let pending_set = &self.pending_set;
        self.pending_scratch.retain(|coord| pending_set.contains(coord));

        for dz in -r..=r {
            for dx in -r..=r {
                for dy in -rv..=rv {
                    let coord = (center.0 + dx, center.1 + dy, center.2 + dz);
                    if let Some(old) = old_center
                        && self.in_window(coord, old)
                    {
                        continue;
                    }
                    if self.loaded.contains_key(&coord) || self.in_flight.contains(&coord) {
                        continue;
                    }
                    if self.pending_set.insert(coord) {
                        self.pending_scratch.push(coord);
                    }
                }
            }
        }

        // Absteigend nach Distanz sortieren, damit `pop()` in `dispatch_pending` nearest-first
        // arbeitet. Nach dem initialen Voll-Scan ist das einmalig O(Volumen log Volumen), bei
        // inkrementellen Rebuilds nur noch ueber die kleine Rest+Randebenen-Menge.
        self.pending_scratch.sort_unstable_by_key(|&(x, y, z)| {
            let dx = (x - center.0) as i64;
            let dy = (y - center.1) as i64;
            let dz = (z - center.2) as i64;
            -(dx * dx + dy * dy + dz * dz)
        });
    }

    /// Arbeitet bis zu `max_chunk_unloads_per_frame` Eintraege der Entlade-Warteschlange ab. Ein
    /// zwischenzeitlich wieder ins Fenster gewanderter Chunk wird nicht entladen, sondern nur wieder
    /// freigegeben.
    fn drain_unloads(&mut self, center: ChunkCoord, queue: &wgpu::Queue, renderer: &mut ChunkRenderer) {
        for _ in 0..self.max_chunk_unloads_per_frame {
            let Some(coord) = self.unload_scratch.pop() else { break };

            if self.in_window(coord, center) {
                if let Some(loaded) = self.loaded.get_mut(&coord) {
                    loaded.queued_for_unload = false;
                }
                continue;
            }

            self.unload_chunk(coord, queue, renderer);
        }
    }

    /// Dispatcht bis zu `max_chunk_dispatches_per_frame` ausstehende Chunks (nearest-first, s.
    /// `pending_scratch`-Sortierung) - ABER NICHT, wenn bereits `max_in_flight_generations()` Jobs
    /// unbeantwortet auf dem Rayon-Pool sitzen.
    ///
    /// Der reine Pro-Frame-Deckel (`max_chunk_dispatches_per_frame`) reicht allein NICHT: er
    /// begrenzt, wie viele NEUE Tasks pro Frame gespawnt werden, aber nicht, wie viele INSGESAMT
    /// gleichzeitig laufen. Solange Generierung schneller war als der Dispatch-Takt, war das
    /// irrelevant - seit Hoehlen/Tunnel/fBm aber mehrere ms/Chunk kosten koennen (v.a. tief unter der
    /// Oberflaeche in Tunnel-Regionen), spawnt dieser Deckel bei 60 FPS potenziell 128 NEUE Tasks
    /// alle ~16ms, waehrend die vorherigen 128 noch gar nicht fertig sind - die Rayon-Warteschlange
    /// waechst dadurch UNBEGRENZT. Ergebnis: alle CPU-Kerne dauerhaft mit Chunk-Generierung gesaettigt,
    /// der Main-Thread bekommt vom OS-Scheduler kein Zeitfenster mehr fuer sein eigenes (triviales)
    /// Pro-Frame-Setup - Frame-Zeiten von >200ms bei praktisch leerlaufender GPU (0.2ms), obwohl die
    /// GPU-Kullung selbst blitzschnell ist. Die In-Flight-Grenze bremst den Dispatch-Takt auf das,
    /// was der Pool tatsaechlich verarbeiten kann, statt blind auf Vorrat zu spawnen.
    fn dispatch_pending(&mut self) {
        let max_in_flight = max_in_flight_generations();

        for _ in 0..self.max_chunk_dispatches_per_frame {
            if self.in_flight.len() >= max_in_flight {
                break;
            }

            let Some(coord) = self.pending_scratch.pop() else { break };
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
            let voxel_scale = self.voxel_scale;

            rayon::spawn(move || {
                // LOD-Ringe (voxel_scale > 1) nutzen den schlanken `generate_lod_chunk`-Pfad (keine
                // Hoehlen/Baeume, s. `generator/lod.rs`) - der teure `boundary_planes`-Batch lohnt
                // sich dort nicht, `is_solid_lod` ist bereits guenstig genug fuer den direkten
                // Mesher-Fallback-Aufruf.
                if voxel_scale == 1 {
                    generator.generate_chunk(coord.0, coord.1, coord.2, &mut chunk);
                } else {
                    generator.generate_lod_chunk(coord.0, coord.1, coord.2, voxel_scale, &mut chunk);
                }

                // Reine Luft-Chunks (z.B. weit oberhalb des Terrains) erzeugen ohnehin keine
                // Faces - das teure Greedy-Meshing (6 Richtungen * 32 Ebenen) lohnt sich dafuer
                // nicht.
                let is_empty = chunk.is_empty();
                let mesh = if is_empty {
                    DirectionalMesh::default()
                } else if voxel_scale == 1 {
                    // Keine Nachbar-Referenzen moeglich (anderer Thread, s. Kommentar an
                    // `ChunkManager::neighbor_chunk_refs`) - `compute_exposure` faellt fuer ALLE 6
                    // Seiten auf die prozedurale Welt-Vorhersage zurueck. Statt bis zu 6144
                    // Einzelaufrufen von `generator.is_solid` (je bis zu 16 gehashte Gitter-Eckwert-
                    // Lookups) werden die 6 Rand-Ebenen EINMAL gebatcht vorberechnet
                    // (`TerrainGenerator::boundary_planes`) und die Closure macht nur noch simple
                    // Array-Lookups - der dominante Meshing-Kostenblock (s. Profiling: 81.5% von
                    // `mesh_chunk`), der JEDEN frisch geladenen Chunk trifft.
                    let ox = coord.0 * CHUNK_SIZE;
                    let oy = coord.1 * CHUNK_SIZE;
                    let oz = coord.2 * CHUNK_SIZE;
                    let planes = generator.boundary_planes(coord.0, coord.1, coord.2);
                    mesh_chunk(&chunk, coord.0, coord.1, coord.2, [None; 6], move |world_x, world_y, world_z| {
                        if world_x < ox {
                            planes.neg_x[(world_y - oy) as usize][(world_z - oz) as usize]
                        } else if world_x >= ox + CHUNK_SIZE {
                            planes.pos_x[(world_y - oy) as usize][(world_z - oz) as usize]
                        } else if world_y < oy {
                            planes.neg_y[(world_x - ox) as usize][(world_z - oz) as usize]
                        } else if world_y >= oy + CHUNK_SIZE {
                            planes.pos_y[(world_x - ox) as usize][(world_z - oz) as usize]
                        } else if world_z < oz {
                            planes.neg_z[(world_x - ox) as usize][(world_y - oy) as usize]
                        } else {
                            planes.pos_z[(world_x - ox) as usize][(world_y - oy) as usize]
                        }
                    })
                } else {
                    // LOD-Randabfrage: mesher-lokale Koordinaten sind hier "Chunk-Koordinate *
                    // CHUNK_SIZE + Offset" in NATIVEN (voxel_scale-freien) Einheiten - mit
                    // `voxel_scale` multipliziert ergeben sie exakt die echte Weltkoordinate (die
                    // Chunk-Ursprungs-Skalierung UND der lokale Schritt sind beide affin in
                    // `voxel_scale`, die Multiplikation verteilt sich also korrekt).
                    mesh_chunk(&chunk, coord.0, coord.1, coord.2, [None; 6], move |wx, wy, wz| {
                        generator.is_solid_lod(wx * voxel_scale, wy * voxel_scale, wz * voxel_scale)
                    })
                };

                let _ = tx.send(GenerationResult { coord, pool_slot, chunk, mesh, is_empty });
            });
        }
    }

    /// Schatten-Sichtbarkeit ueber Licht-Kugel- statt Kamera-Frustum-Kullung: ein Chunk gilt als
    /// schatten-relevant, wenn seine AABB IRGENDEINE aktive (um ihren Slack aufgeblaehte)
    /// Kaskaden-Kugel schneidet. Der Voll-Scan + Indirect-Buffer-Reupload laeuft NUR, wenn sich die
    /// Chunk-Menge geaendert hat oder eine Kaskaden-Kugel weiter als ihren Slack gewandert ist -
    /// dazwischen bleibt die zuletzt hochgeladene (dank Aufblaehung garantiert vollstaendige)
    /// Obermenge einfach stehen, der Main-Thread fasst pro Frame keinen einzigen Chunk an.
    pub fn update_shadow_visibility(
        &mut self,
        cascades: &[Cascade; MAX_SHADOW_CASCADES],
        cascade_count: u32,
        queue: &wgpu::Queue,
        renderer: &mut ChunkRenderer,
    ) {
        let active = &cascades[..cascade_count as usize];

        let cascades_stable = cascade_count == self.shadow_last_cascade_count
            && active.iter().enumerate().all(|(i, c)| {
                let (last_center, last_radius) = self.shadow_last_cascades[i];
                let slack = shadow_cascade_slack(last_radius);
                (c.radius - last_radius).abs() < 1.0 && c.center.distance_squared(last_center) < slack * slack
            });
        if cascades_stable && !self.shadow_set_dirty {
            return;
        }

        self.shadow_visible_handles.clear();
        for (coord, loaded) in self.loaded.iter() {
            if loaded.is_empty {
                continue;
            }
            let min = self.chunk_origin(*coord);
            let max = min + glam::Vec3::splat((CHUNK_SIZE * self.voxel_scale) as f32);
            let relevant = active
                .iter()
                .any(|c| sphere_intersects_aabb(c.center, c.radius + shadow_cascade_slack(c.radius), min, max));
            if relevant {
                self.shadow_visible_handles.push((loaded.gpu_handle, min));
            }
        }

        renderer.set_shadow_visible(queue, &self.shadow_visible_handles);

        for (i, c) in active.iter().enumerate() {
            self.shadow_last_cascades[i] = (c.center, c.radius);
        }
        self.shadow_last_cascade_count = cascade_count;
        self.shadow_set_dirty = false;
    }

    /// Soft-Cancellation: Ein auf dem Rayon-Thread fertiggestellter Chunk wird verworfen, wenn die
    /// Kamera sich waehrend der Generierungszeit bereits so weit wegbewegt hat, dass der Chunk
    /// nicht mehr innerhalb der Render-Distanz liegt - so entsteht kein unnoetiger GPU-Upload fuer
    /// Chunks, die im selben Frame wieder entladen wuerden.
    ///
    /// Gedeckelt auf `max_chunk_uploads_per_frame` - nicht abgeholte Ergebnisse bleiben im Channel
    /// und werden im naechsten Frame weiterverarbeitet.
    fn apply_completed_generations(&mut self, center: ChunkCoord, queue: &wgpu::Queue, renderer: &mut ChunkRenderer) {
        for _ in 0..self.max_chunk_uploads_per_frame {
            let Ok(result) = self.result_rx.try_recv() else { break };
            self.in_flight.remove(&result.coord);

            if !self.in_window(result.coord, center) {
                self.pool[result.pool_slot] = Some(result.chunk);
                self.pool_free_list.push(result.pool_slot);
                continue;
            }

            let gpu_handle = renderer.alloc_chunk(queue, &result.mesh);
            let gpu_pool_slot = self.pool_slot_base + result.pool_slot;

            if result.is_empty {
                renderer.clear_chunk_meta(queue, gpu_pool_slot);
            } else {
                let min = self.chunk_origin(result.coord);
                let max = min + glam::Vec3::splat((CHUNK_SIZE * self.voxel_scale) as f32);
                renderer.update_chunk_meta(queue, gpu_pool_slot, min, max, self.voxel_scale as f32, &gpu_handle);
            }

            self.pool[result.pool_slot] = Some(result.chunk);
            self.loaded.insert(
                result.coord,
                LoadedChunk {
                    pool_slot: result.pool_slot,
                    gpu_handle,
                    is_empty: result.is_empty,
                    queued_for_unload: false,
                },
            );
            if !result.is_empty {
                self.shadow_set_dirty = true;
            }
        }
    }

    fn unload_chunk(&mut self, coord: ChunkCoord, queue: &wgpu::Queue, renderer: &mut ChunkRenderer) {
        let Some(loaded) = self.loaded.remove(&coord) else {
            return;
        };

        renderer.free_chunk(&loaded.gpu_handle);
        renderer.clear_chunk_meta(queue, self.pool_slot_base + loaded.pool_slot);

        // Kein `chunk.clear()` hier: `generate_chunk` beginnt selbst mit `clear()`, und zwischen
        // Freigabe und Neuvergabe liest niemand den Pool-Slot (`is_solid_at` prueft `loaded`
        // zuerst) - das memset (64 KiB * bis zu 192 Unloads/Frame = 12 MB) war reine Verschwendung.
        self.pool_free_list.push(loaded.pool_slot);
        if !loaded.is_empty {
            self.shadow_set_dirty = true;
        }
    }
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
        assert_eq!(chunk_origin_scaled((1, -1, 2), 1), glam::Vec3::new(32.0, -32.0, 64.0));
    }

    #[test]
    fn chunk_origin_scales_with_voxel_scale() {
        assert_eq!(chunk_origin_scaled((1, -1, 2), 4), glam::Vec3::new(128.0, -128.0, 256.0));
    }

    #[test]
    fn coord_hasher_distributes_neighboring_coords() {
        let mut seen = std::collections::HashSet::new();
        for x in -8..8 {
            for y in -8..8 {
                for z in -8..8 {
                    let mut hasher = CoordHasher::default();
                    hasher.write_i32(x);
                    hasher.write_i32(y);
                    hasher.write_i32(z);
                    seen.insert(hasher.finish());
                }
            }
        }
        assert_eq!(seen.len(), 16 * 16 * 16, "CoordHasher kollidiert auf benachbarten Koordinaten");
    }
}
