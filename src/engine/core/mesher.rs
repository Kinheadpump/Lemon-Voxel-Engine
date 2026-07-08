use std::cell::RefCell;

use crate::game::world::chunk::{Chunk, CHUNK_SIZE};

pub const DIR_NEG_X: usize = 0;
pub const DIR_POS_X: usize = 1;
pub const DIR_NEG_Y: usize = 2;
pub const DIR_POS_Y: usize = 3;
pub const DIR_NEG_Z: usize = 4;
pub const DIR_POS_Z: usize = 5;

pub const NEIGHBOR_OFFSETS: [(i32, i32, i32); 6] = [
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
];

const CHUNK_SIZE_USIZE: usize = CHUNK_SIZE as usize;

type FaceMask = [[u16; CHUNK_SIZE_USIZE]; CHUNK_SIZE_USIZE];
/// Eine Achsen-Spalten-Tabelle: 32x32 Positionen in der Ebene senkrecht zur Achse, je ein u32-
/// Bitfeld ueber die 32 Voxel ENTLANG der Achse (Bit i = Voxel an Achsenposition i ist massiv).
/// CHUNK_SIZE=32 passt exakt in ein u32 - das ist die Grundlage von "Binary Greedy Meshing".
type AxisColumns = [[u32; CHUNK_SIZE_USIZE]; CHUNK_SIZE_USIZE];

struct MeshingScratch {
    mask: FaceMask,
    solid_x: AxisColumns,
    solid_y: AxisColumns,
    solid_z: AxisColumns,
    /// Pro Richtung (Index = DIR_*) das Ergebnis von `compute_exposure`: Bit `layer` gesetzt heisst
    /// "Voxel bei dieser Achsenposition hat auf der `dir`-Seite kein massives Nachbarvoxel" - fertig
    /// vorberechnet fuer den kompletten Chunk, bevor irgendein Face gebaut wird.
    exposure: [AxisColumns; 6],
    /// Pro Richtung die OR-Reduktion aller 1024 `exposure`-Spalten - Bit `layer` gesetzt heisst "in
    /// dieser Ebene liegt IRGENDWO mindestens ein Face". Ebenen mit Bit 0 werden in `mesh_chunk`
    /// komplett uebersprungen (kein Populate, kein Merge-Aufruf).
    any_exposed: [u32; 6],
}

thread_local! {
    /// Wiederverwendeter Scratch-Speicher (~39 KiB: 2 KiB Merge-Maske + 3x4 KiB Achsen-Spalten +
    /// 6x4 KiB Exposure-Tabellen), EINMAL pro Rayon-Worker-Thread alloziert statt pro Chunk. Passt
    /// bequem in L2-Cache.
    static SCRATCH: RefCell<MeshingScratch> = const {
        RefCell::new(MeshingScratch {
            mask: [[0u16; CHUNK_SIZE_USIZE]; CHUNK_SIZE_USIZE],
            solid_x: [[0u32; CHUNK_SIZE_USIZE]; CHUNK_SIZE_USIZE],
            solid_y: [[0u32; CHUNK_SIZE_USIZE]; CHUNK_SIZE_USIZE],
            solid_z: [[0u32; CHUNK_SIZE_USIZE]; CHUNK_SIZE_USIZE],
            exposure: [[[0u32; CHUNK_SIZE_USIZE]; CHUNK_SIZE_USIZE]; 6],
            any_exposed: [0u32; 6],
        })
    };
}

#[derive(Default)]
pub struct DirectionalMesh {
    pub faces: [Vec<u32>; 6],
}

#[inline(always)]
fn encode_face(x: u32, y: u32, z: u32, texture_id: u32, width_m1: u32, height_m1: u32) -> u32 {
    (x & 0x1F)
        | ((y & 0x1F) << 5)
        | ((z & 0x1F) << 10)
        | ((texture_id & 0x7F) << 15)
        | ((width_m1 & 0x1F) << 22)
        | ((height_m1 & 0x1F) << 27)
}

#[inline(always)]
fn pos_from_layer_uv(dir: usize, layer: i32, u: i32, v: i32) -> (i32, i32, i32) {
    match dir {
        DIR_NEG_X => (layer, v, u),
        DIR_POS_X => (layer, u, v),
        DIR_NEG_Y => (u, layer, v),
        DIR_POS_Y => (v, layer, u),
        DIR_NEG_Z => (v, u, layer),
        DIR_POS_Z => (u, v, layer),
        _ => unreachable!(),
    }
}

/// Fuer welche Richtungen `pos_from_layer_uv` `v` (statt `u`) auf die X-Achse des flachen
/// `x + y*32 + z*1024`-Arrays abbildet - X ist die einzige Achse mit Stride 1 (echt
/// zusammenhaengender Speicher). Nur noch fuer `merge_mask_into_faces`/das finale Mask-Array
/// relevant (die teure Rand-Pruefung selbst laeuft jetzt ueber `compute_exposure`, s.u.), bleibt
/// aber aus dem urspruenglichen Profiling erhalten, weil Zugriffsreihenfolge weiterhin zaehlt.
const SWAP_UV_LOOP_ORDER: [bool; 6] = [
    false, // DIR_NEG_X: v->Y (Stride 32) ist bereits die beste erreichbare Achse (X ist hier "layer", fix)
    true,  // DIR_POS_X: u->Y (Stride 32) statt v->Z (Stride 1024)
    true,  // DIR_NEG_Y: u->X (Stride 1) statt v->Z (Stride 1024)
    false, // DIR_POS_Y: v->X (Stride 1) ist bereits optimal
    false, // DIR_NEG_Z: v->X (Stride 1) ist bereits optimal
    true,  // DIR_POS_Z: u->X (Stride 1) statt v->Y (Stride 32)
];

/// Phase 1 (Binary Greedy Meshing): EIN sequenzieller Durchlauf ueber alle 32768 Voxel des Chunks
/// (exakt die Flat-Array-Reihenfolge: x schnellst-, z langsamst-laufend) fuellt gleichzeitig alle
/// drei Achsen-Bitfelder. Ersetzt die vorherigen bis zu 196.608 Einzelpruefungen (6 Richtungen * 32
/// Ebenen * 1024 Zellen) durch genau 32768 sequenzielle, cache-optimale Lesevorgaenge - jedes Voxel
/// wird nur noch EIN einziges Mal angefasst statt bis zu 6-mal ueber alle Richtungen hinweg.
fn build_solidity_columns(chunk: &Chunk, solid_x: &mut AxisColumns, solid_y: &mut AxisColumns, solid_z: &mut AxisColumns) {
    for z in 0..CHUNK_SIZE_USIZE {
        for y in 0..CHUNK_SIZE_USIZE {
            for x in 0..CHUNK_SIZE_USIZE {
                if chunk.get_block(x as i32, y as i32, z as i32) == 0 {
                    continue;
                }
                solid_x[y][z] |= 1 << x;
                solid_y[x][z] |= 1 << y;
                solid_z[x][y] |= 1 << z;
            }
        }
    }
}

/// Liefert, ob die Position exakt einen Schritt ausserhalb des Chunks entlang einer Achse massiv
/// ist - genau EIN Aufruf pro Rand-Spalte statt vorher potenziell einer pro Rand-VOXEL. `local` ist
/// die Position bereits in den Nachbar-Chunk gewrappt (0 oder 31), `world` die absolute
/// Weltkoordinate fuer den prozeduralen Fallback.
#[inline(always)]
fn boundary_solid<F: Fn(i32, i32, i32) -> bool>(
    neighbor: Option<&Chunk>,
    local: (i32, i32, i32),
    world: (i32, i32, i32),
    neighbor_solid_at_world: &F,
) -> bool {
    if let Some(chunk) = neighbor {
        chunk.get_block(local.0, local.1, local.2) != 0
    } else {
        neighbor_solid_at_world(world.0, world.1, world.2)
    }
}

/// Phase 2: aus den Achsen-Bitfeldern werden pro Spalte BEIDE Face-Richtungen gleichzeitig per
/// Shift+AND-NOT bestimmt - `col & !(col >> 1 mit eingeschobenem Rand-Bit)` markiert alle 32
/// moeglichen Positionen einer Spalte auf einmal als "Richtung +Achse exponiert", `col << 1`
/// analog fuer "-Achse exponiert". Das ersetzt 32 skalare Nachbar-Vergleiche pro Spalte durch 2
/// Bit-Operationen. Das Rand-Bit (Nachbar-Chunk oder prozeduraler Fallback) wird genau EINMAL pro
/// Spalte eingespeist statt einmal pro Voxel. `any_exposed[dir]` sammelt zusaetzlich per OR ueber
/// alle 1024 Spalten, in welchen Ebenen ueberhaupt IRGENDEIN Face liegt - `mesh_chunk` ueberspringt
/// damit komplett leere Ebenen (haeufig bei durchgehend massiven oder komplett umschlossenen
/// Chunks) statt sie trotzdem leer zu durchlaufen.
#[allow(clippy::too_many_arguments)]
fn compute_exposure<F: Fn(i32, i32, i32) -> bool>(
    chunk_x: i32,
    chunk_y: i32,
    chunk_z: i32,
    neighbors: [Option<&Chunk>; 6],
    solid_x: &AxisColumns,
    solid_y: &AxisColumns,
    solid_z: &AxisColumns,
    exposure: &mut [AxisColumns; 6],
    any_exposed: &mut [u32; 6],
    neighbor_solid_at_world: &F,
) {
    for z in 0..CHUNK_SIZE_USIZE {
        for y in 0..CHUNK_SIZE_USIZE {
            let col = solid_x[y][z];
            let below = boundary_solid(
                neighbors[DIR_NEG_X],
                (CHUNK_SIZE - 1, y as i32, z as i32),
                (chunk_x * CHUNK_SIZE - 1, chunk_y * CHUNK_SIZE + y as i32, chunk_z * CHUNK_SIZE + z as i32),
                neighbor_solid_at_world,
            );
            let above = boundary_solid(
                neighbors[DIR_POS_X],
                (0, y as i32, z as i32),
                (chunk_x * CHUNK_SIZE + CHUNK_SIZE, chunk_y * CHUNK_SIZE + y as i32, chunk_z * CHUNK_SIZE + z as i32),
                neighbor_solid_at_world,
            );
            let extended_above = (col >> 1) | ((above as u32) << 31);
            let extended_below = (col << 1) | (below as u32);
            let pos = col & !extended_above;
            let neg = col & !extended_below;
            exposure[DIR_POS_X][y][z] = pos;
            exposure[DIR_NEG_X][y][z] = neg;
            any_exposed[DIR_POS_X] |= pos;
            any_exposed[DIR_NEG_X] |= neg;
        }
    }

    for z in 0..CHUNK_SIZE_USIZE {
        for x in 0..CHUNK_SIZE_USIZE {
            let col = solid_y[x][z];
            let below = boundary_solid(
                neighbors[DIR_NEG_Y],
                (x as i32, CHUNK_SIZE - 1, z as i32),
                (chunk_x * CHUNK_SIZE + x as i32, chunk_y * CHUNK_SIZE - 1, chunk_z * CHUNK_SIZE + z as i32),
                neighbor_solid_at_world,
            );
            let above = boundary_solid(
                neighbors[DIR_POS_Y],
                (x as i32, 0, z as i32),
                (chunk_x * CHUNK_SIZE + x as i32, chunk_y * CHUNK_SIZE + CHUNK_SIZE, chunk_z * CHUNK_SIZE + z as i32),
                neighbor_solid_at_world,
            );
            let extended_above = (col >> 1) | ((above as u32) << 31);
            let extended_below = (col << 1) | (below as u32);
            let pos = col & !extended_above;
            let neg = col & !extended_below;
            exposure[DIR_POS_Y][x][z] = pos;
            exposure[DIR_NEG_Y][x][z] = neg;
            any_exposed[DIR_POS_Y] |= pos;
            any_exposed[DIR_NEG_Y] |= neg;
        }
    }

    for y in 0..CHUNK_SIZE_USIZE {
        for x in 0..CHUNK_SIZE_USIZE {
            let col = solid_z[x][y];
            let below = boundary_solid(
                neighbors[DIR_NEG_Z],
                (x as i32, y as i32, CHUNK_SIZE - 1),
                (chunk_x * CHUNK_SIZE + x as i32, chunk_y * CHUNK_SIZE + y as i32, chunk_z * CHUNK_SIZE - 1),
                neighbor_solid_at_world,
            );
            let above = boundary_solid(
                neighbors[DIR_POS_Z],
                (x as i32, y as i32, 0),
                (chunk_x * CHUNK_SIZE + x as i32, chunk_y * CHUNK_SIZE + y as i32, chunk_z * CHUNK_SIZE + CHUNK_SIZE),
                neighbor_solid_at_world,
            );
            let extended_above = (col >> 1) | ((above as u32) << 31);
            let extended_below = (col << 1) | (below as u32);
            let pos = col & !extended_above;
            let neg = col & !extended_below;
            exposure[DIR_POS_Z][x][y] = pos;
            exposure[DIR_NEG_Z][x][y] = neg;
            any_exposed[DIR_POS_Z] |= pos;
            any_exposed[DIR_NEG_Z] |= neg;
        }
    }
}

/// Phase 3: fuellt die 2D-Merge-Maske fuer eine (Richtung, Ebene) NUR noch per Bit-Test aus der
/// vorberechneten Exposure-Tabelle - keine Nachbar-Pruefung, kein zweiter Speicherzugriff mehr pro
/// Zelle. `chunk.get_block` wird nur noch fuer tatsaechlich exponierte (also garantiert massive)
/// Zellen aufgerufen, um die Textur-ID zu holen.
fn populate_mask_from_exposure(
    chunk: &Chunk,
    exposure_for_dir: &AxisColumns,
    dir: usize,
    layer: i32,
    mask: &mut FaceMask,
) {
    debug_assert!(
        mask.iter().flatten().all(|&cell| cell == 0),
        "Mask-Scratch war beim Eintritt nicht vollstaendig 0 - Selbstreinigungs-Invariante von \
         `merge_mask_into_faces` verletzt"
    );

    let swap = SWAP_UV_LOOP_ORDER[dir];

    for a in 0..CHUNK_SIZE {
        for b in 0..CHUNK_SIZE {
            let (u, v) = if swap { (b, a) } else { (a, b) };
            let (x, y, z) = pos_from_layer_uv(dir, layer, u, v);

            // Exposure-Tabellen sind wie ihre zugehoerige `solid_*`-Spaltentabelle indiziert:
            // X-Richtungen ueber (y,z), Y-Richtungen ueber (x,z), Z-Richtungen ueber (x,y).
            let column = match dir {
                DIR_NEG_X | DIR_POS_X => exposure_for_dir[y as usize][z as usize],
                DIR_NEG_Y | DIR_POS_Y => exposure_for_dir[x as usize][z as usize],
                _ => exposure_for_dir[x as usize][y as usize],
            };

            if (column >> layer) & 1 != 0 {
                mask[u as usize][v as usize] = chunk.get_block(x, y, z);
            }
        }
    }
}

/// Greedy-Merge UND Face-Encoding in einem Durchgang: verbrauchte Zellen werden auf 0
/// zurueckgesetzt (dadurch ist die Maske nach dieser Funktion garantiert wieder komplett 0 -
/// `populate_mask_from_exposure` muss sie folglich nie explizit leeren) und jedes fertige Rechteck
/// wird SOFORT als komprimiertes u32 in `output` gepusht. Es existiert bewusst keine Zwischen-Liste
/// aus Rechtecken mehr, die spaeter erst in den Output kopiert wuerde.
fn merge_mask_into_faces(mask: &mut FaceMask, dir: usize, layer: i32, output: &mut Vec<u32>) {
    for v in 0..CHUNK_SIZE_USIZE {
        for u in 0..CHUNK_SIZE_USIZE {
            let texture_id = mask[u][v];
            if texture_id == 0 {
                continue;
            }

            let mut width = 1;
            while u + width < CHUNK_SIZE_USIZE && mask[u + width][v] == texture_id {
                width += 1;
            }

            let mut height = 1;
            'grow_height: while v + height < CHUNK_SIZE_USIZE {
                for k in 0..width {
                    if mask[u + k][v + height] != texture_id {
                        break 'grow_height;
                    }
                }
                height += 1;
            }

            for du in 0..width {
                for dv in 0..height {
                    mask[u + du][v + dv] = 0;
                }
            }

            let (x, y, z) = pos_from_layer_uv(dir, layer, u as i32, v as i32);
            output.push(encode_face(
                x as u32,
                y as u32,
                z as u32,
                texture_id as u32,
                (width - 1) as u32,
                (height - 1) as u32,
            ));
        }
    }
}

/// `neighbors[dir]` ist die bereits geladene Nachbar-Chunk-Referenz in Richtung `dir` (Reihenfolge
/// wie `NEIGHBOR_OFFSETS`/`DIR_*`), sofern vorhanden - vom Aufrufer EINMALIG vor dem Meshing
/// aufgeloest (siehe `ChunkManager::neighbor_chunk_refs`), statt pro Rand-Voxel neu nachzuschlagen.
pub fn mesh_chunk<F: Fn(i32, i32, i32) -> bool>(
    chunk: &Chunk,
    chunk_x: i32,
    chunk_y: i32,
    chunk_z: i32,
    neighbors: [Option<&Chunk>; 6],
    neighbor_solid_at_world: F,
) -> DirectionalMesh {
    let mut mesh = DirectionalMesh::default();

    SCRATCH.with_borrow_mut(|scratch| {
        let MeshingScratch { mask, solid_x, solid_y, solid_z, exposure, any_exposed } = &mut *scratch;

        build_solidity_columns(chunk, solid_x, solid_y, solid_z);
        compute_exposure(
            chunk_x,
            chunk_y,
            chunk_z,
            neighbors,
            solid_x,
            solid_y,
            solid_z,
            exposure,
            any_exposed,
            &neighbor_solid_at_world,
        );

        for dir in 0..6 {
            if any_exposed[dir] == 0 {
                continue;
            }

            // Popcount ueber die Exposure-Spalten = exakte Face-Anzahl VOR dem Greedy-Merge, also
            // eine garantierte Obergrenze des Outputs - EINE Allokation exakt passender Groesse
            // statt wiederholtem Verdoppeln (und Kopieren) durch `Vec::push`-Wachstum.
            let exposed_face_upper_bound: u32 = exposure[dir].iter().flatten().map(|col| col.count_ones()).sum();
            mesh.faces[dir].reserve(exposed_face_upper_bound as usize);

            for layer in 0..CHUNK_SIZE {
                // Ganze Ebenen ohne ein einziges exponiertes Face ueberspringen (haeufig bei
                // durchgehend massiven oder komplett umschlossenen Chunks) - spart den vollen
                // 1024-Zellen-Populate- UND den Merge-Durchlauf fuer diese Ebene komplett.
                if (any_exposed[dir] >> layer) & 1 == 0 {
                    continue;
                }
                populate_mask_from_exposure(chunk, &exposure[dir], dir, layer, mask);
                merge_mask_into_faces(mask, dir, layer, &mut mesh.faces[dir]);
            }
        }

        // Die Achsen-/Exposure-/Aggregat-Tabellen sind NICHT selbstreinigend (anders als `mask`) -
        // sie werden pro Chunk komplett ueberschrieben, aber `build_solidity_columns`/
        // `compute_exposure` nutzen `|=` (additiv), muessen also von 0 starten. Ein `memset` auf
        // drei 4-KiB-, sechs 4-KiB- und ein 24-Byte-Array ist immer noch um Groessenordnungen
        // billiger als der eingesparte Scan.
        for column in solid_x.iter_mut().chain(solid_y.iter_mut()).chain(solid_z.iter_mut()) {
            column.fill(0);
        }
        for table in exposure.iter_mut() {
            for column in table.iter_mut() {
                column.fill(0);
            }
        }
        any_exposed.fill(0);
    });

    mesh
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_world_neighbors(_x: i32, _y: i32, _z: i32) -> bool {
        false
    }

    #[test]
    fn single_isolated_voxel_produces_exactly_one_face_per_direction() {
        let mut chunk = Chunk::empty();
        chunk.set_block(5, 5, 5, 1);

        let mesh = mesh_chunk(&chunk, 0, 0, 0, [None; 6], no_world_neighbors);

        for faces in &mesh.faces {
            assert_eq!(faces.len(), 1);
        }
    }

    #[test]
    fn adjacent_same_texture_voxels_merge_into_one_larger_face() {
        let mut chunk = Chunk::empty();
        chunk.set_block(5, 5, 5, 1);
        chunk.set_block(6, 5, 5, 1);

        let mesh = mesh_chunk(&chunk, 0, 0, 0, [None; 6], no_world_neighbors);

        // +Y/-Y/+Z/-Z sehen beide Bloecke nebeneinander - greedy merge zu EINEM 2x1-Face.
        assert_eq!(mesh.faces[DIR_POS_Y].len(), 1);
        assert_eq!(mesh.faces[DIR_NEG_Y].len(), 1);
        // +X/-X sind die Stirnseiten - je ein 1x1-Face, unveraendert getrennt.
        assert_eq!(mesh.faces[DIR_POS_X].len(), 1);
        assert_eq!(mesh.faces[DIR_NEG_X].len(), 1);
    }

    #[test]
    fn cached_neighbor_reference_occludes_boundary_face() {
        let mut chunk = Chunk::empty();
        chunk.set_block(31, 5, 5, 1);
        let mut neighbor = Chunk::empty();
        neighbor.set_block(0, 5, 5, 1);

        let mut neighbors: [Option<&Chunk>; 6] = [None; 6];
        neighbors[DIR_POS_X] = Some(&neighbor);

        let mesh = mesh_chunk(&chunk, 0, 0, 0, neighbors, no_world_neighbors);

        // Die +X-Seite ist durch den gecachten Nachbarn verdeckt - kein Face in dieser Richtung.
        assert_eq!(mesh.faces[DIR_POS_X].len(), 0);
        // Alle anderen Seiten sind weiterhin frei.
        assert_eq!(mesh.faces[DIR_NEG_X].len(), 1);
    }

    #[test]
    fn boundary_face_exposed_when_neighbor_not_loaded_and_world_reports_air() {
        let mut chunk = Chunk::empty();
        chunk.set_block(31, 5, 5, 1);

        // Kein gecachter Nachbar UND die prozedurale Welt-Vorhersage sagt "Luft" -> Face muss da sein.
        let mesh = mesh_chunk(&chunk, 0, 0, 0, [None; 6], no_world_neighbors);
        assert_eq!(mesh.faces[DIR_POS_X].len(), 1);
    }

    #[test]
    fn boundary_face_hidden_when_world_fallback_reports_solid() {
        let mut chunk = Chunk::empty();
        chunk.set_block(31, 5, 5, 1);

        // Nur die +X-Seite (x=32) ist ein echter Chunk-Rand und faellt auf den Welt-Fallback
        // zurueck. Alle anderen 5 Nachbarpositionen (x=30, y=4/6, z=4/6) liegen im lokalen Chunk-
        // Inneren und sind dort Luft - unabhaengig vom Fallback exponiert.
        let mesh = mesh_chunk(&chunk, 0, 0, 0, [None; 6], |_, _, _| true);
        assert_eq!(mesh.faces[DIR_POS_X].len(), 0);
        assert_eq!(mesh.faces[DIR_NEG_X].len(), 1);
        assert_eq!(mesh.faces[DIR_NEG_Y].len(), 1);
        assert_eq!(mesh.faces[DIR_POS_Y].len(), 1);
        assert_eq!(mesh.faces[DIR_NEG_Z].len(), 1);
        assert_eq!(mesh.faces[DIR_POS_Z].len(), 1);
    }

    #[test]
    fn fully_solid_chunk_produces_no_interior_faces() {
        let mut chunk = Chunk::empty();
        for i in 0..CHUNK_SIZE {
            for j in 0..CHUNK_SIZE {
                for k in 0..CHUNK_SIZE {
                    chunk.set_block(i, j, k, 1);
                }
            }
        }

        // Komplett von massiven Nachbarn umgeben -> ueberhaupt keine Faces.
        let mesh = mesh_chunk(&chunk, 0, 0, 0, [None; 6], |_, _, _| true);
        for faces in &mesh.faces {
            assert_eq!(faces.len(), 0);
        }
    }

    #[test]
    fn mask_scratch_is_clean_after_repeated_calls() {
        // Regressionstest fuer die Selbstreinigungs-Invariante: mehrere aufeinanderfolgende
        // `mesh_chunk`-Aufrufe im selben Thread duerfen den `debug_assert` in
        // `populate_mask_from_exposure` nicht verletzen (liefe in Debug-Builds sofort in einen
        // Panic, falls die Maske nicht vollstaendig auf 0 zurueckgesetzt wird).
        for i in 0..3 {
            let mut chunk = Chunk::empty();
            chunk.set_block(i, i, i, 1);
            let _ = mesh_chunk(&chunk, 0, 0, 0, [None; 6], no_world_neighbors);
        }
    }

    /// Diagnose-Benchmark, KEIN Korrektheitstest - misst die reine `mesh_chunk`-Zeit direkt (die
    /// Phasenaufteilung aus der Vorgaenger-Version ist mit dem Binary-Greedy-Umbau ueberholt, die
    /// Phasen laufen jetzt nicht mehr pro (Richtung,Ebene) einzeln). Manuell ausfuehren mit:
    /// `cargo test --release --lib -- --ignored --nocapture profile_mesh_chunk_total`
    #[test]
    #[ignore = "Diagnose-Tool, kein automatisierter Test - siehe Doc-Kommentar"]
    fn profile_mesh_chunk_total() {
        use std::time::Instant;

        use crate::engine::config::EngineConfig;
        use crate::game::world::generator::TerrainGenerator;

        let config = EngineConfig::default();
        let generator = TerrainGenerator::new(&config);
        let mut chunk = Chunk::empty();
        generator.generate_chunk(4, 0, 4, &mut chunk);
        assert!(!chunk.is_empty());

        let neighbor_solid = |x: i32, y: i32, z: i32| generator.is_solid(x, y, z);
        const ITERATIONS: usize = 2000;

        for _ in 0..50 {
            std::hint::black_box(mesh_chunk(&chunk, 4, 0, 4, [None; 6], neighbor_solid));
        }

        let start = Instant::now();
        for _ in 0..ITERATIONS {
            std::hint::black_box(mesh_chunk(&chunk, 4, 0, 4, [None; 6], neighbor_solid));
        }
        let elapsed = start.elapsed();
        let per_chunk_us = elapsed.as_secs_f64() * 1_000_000.0 / ITERATIONS as f64;
        println!("Warm-Meshing (Binary Greedy): {per_chunk_us:.2} us/Chunk");
    }

    /// Diagnose-Benchmark, KEIN Korrektheitstest - zerlegt `mesh_chunk` in seine 4 Phasen
    /// (Spalten-Bau, Exposure-Berechnung, Populate+Merge, Scratch-Reinigung). Manuell ausfuehren:
    /// `cargo test --release --lib -- --ignored --nocapture profile_mesh_chunk_binary_phases`
    #[test]
    #[ignore = "Diagnose-Tool, kein automatisierter Test - siehe Doc-Kommentar"]
    fn profile_mesh_chunk_binary_phases() {
        use std::time::{Duration, Instant};

        use crate::engine::config::EngineConfig;
        use crate::game::world::generator::TerrainGenerator;

        let config = EngineConfig::default();
        let generator = TerrainGenerator::new(&config);
        let mut chunk = Chunk::empty();
        generator.generate_chunk(4, 0, 4, &mut chunk);
        assert!(!chunk.is_empty());

        let neighbor_solid = |x: i32, y: i32, z: i32| generator.is_solid(x, y, z);
        const ITERATIONS: usize = 2000;

        let run = || {
            let mut columns_time = Duration::ZERO;
            let mut exposure_time = Duration::ZERO;
            let mut populate_merge_time = Duration::ZERO;
            let mut cleanup_time = Duration::ZERO;
            let mut mesh = DirectionalMesh::default();

            SCRATCH.with_borrow_mut(|scratch| {
                let MeshingScratch { mask, solid_x, solid_y, solid_z, exposure, any_exposed } = &mut *scratch;

                let t0 = Instant::now();
                build_solidity_columns(&chunk, solid_x, solid_y, solid_z);
                columns_time += t0.elapsed();

                let t1 = Instant::now();
                compute_exposure(4, 0, 4, [None; 6], solid_x, solid_y, solid_z, exposure, any_exposed, &neighbor_solid);
                exposure_time += t1.elapsed();

                let t2 = Instant::now();
                for dir in 0..6 {
                    for layer in 0..CHUNK_SIZE {
                        if (any_exposed[dir] >> layer) & 1 == 0 {
                            continue;
                        }
                        populate_mask_from_exposure(&chunk, &exposure[dir], dir, layer, mask);
                        merge_mask_into_faces(mask, dir, layer, &mut mesh.faces[dir]);
                    }
                }
                populate_merge_time += t2.elapsed();

                let t3 = Instant::now();
                for column in solid_x.iter_mut().chain(solid_y.iter_mut()).chain(solid_z.iter_mut()) {
                    column.fill(0);
                }
                for table in exposure.iter_mut() {
                    for column in table.iter_mut() {
                        column.fill(0);
                    }
                }
                any_exposed.fill(0);
                cleanup_time += t3.elapsed();
            });

            for faces in &mut mesh.faces {
                faces.clear();
            }

            (columns_time, exposure_time, populate_merge_time, cleanup_time)
        };

        for _ in 0..50 {
            std::hint::black_box(run());
        }

        let mut totals = (Duration::ZERO, Duration::ZERO, Duration::ZERO, Duration::ZERO);
        for _ in 0..ITERATIONS {
            let (a, b, c, d) = run();
            totals.0 += a;
            totals.1 += b;
            totals.2 += c;
            totals.3 += d;
        }

        let us = |d: Duration| d.as_secs_f64() * 1_000_000.0 / ITERATIONS as f64;
        let (c_us, e_us, p_us, cl_us) = (us(totals.0), us(totals.1), us(totals.2), us(totals.3));
        let total = c_us + e_us + p_us + cl_us;
        println!("Phase 1 build_solidity_columns: {c_us:8.2} us/Chunk ({:5.1}%)", c_us / total * 100.0);
        println!("Phase 2 compute_exposure:        {e_us:8.2} us/Chunk ({:5.1}%)", e_us / total * 100.0);
        println!("Phase 3 populate+merge (6*32):    {p_us:8.2} us/Chunk ({:5.1}%)", p_us / total * 100.0);
        println!("Phase 4 scratch-cleanup:         {cl_us:8.2} us/Chunk ({:5.1}%)", cl_us / total * 100.0);
        println!("Summe: {total:8.2} us/Chunk");
    }
}
