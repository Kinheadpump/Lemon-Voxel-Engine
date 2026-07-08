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

thread_local! {
    /// Wiederverwendeter Scratch-Puffer fuer die 2D-Face-Maske (32x32 = 2 KiB), EINMAL pro
    /// Rayon-Worker-Thread alloziert statt 192-mal pro Chunk (6 Richtungen * 32 Ebenen). Bleibt
    /// zwischen Aufrufen garantiert komplett auf 0 - siehe Invariante an `build_face_mask`.
    static MASK_SCRATCH: RefCell<FaceMask> = const { RefCell::new([[0u16; CHUNK_SIZE_USIZE]; CHUNK_SIZE_USIZE]) };
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

/// Liest einen Nachbarblock. Liegt die Position innerhalb des lokalen Chunks, wird das lokale
/// Array genutzt. Liegt sie ausserhalb (immer exakt 1 Schritt in GENAU der `dir`-Achse, da `x,y,z`
/// hier stets aus `pos_from_layer_uv` + einem Einheitsversatz stammen), wird zuerst die lokal
/// gecachte `neighbor`-Referenz probiert (billiger Array-Zugriff statt HashMap-Lookup) und nur wenn
/// dieser Nachbar gar nicht geladen ist (z.B. Rand der Render-Distanz, oder der asynchrone
/// Rayon-Meshing-Pfad, der keine Referenzen ueber die Thread-Grenze halten kann) auf die
/// prozedurale Welt-Vorhersage zurueckgefallen.
#[inline(always)]
fn is_solid<F: Fn(i32, i32, i32) -> bool>(
    chunk: &Chunk,
    neighbor: Option<&Chunk>,
    chunk_x: i32,
    chunk_y: i32,
    chunk_z: i32,
    x: i32,
    y: i32,
    z: i32,
    neighbor_solid_at_world: &F,
) -> bool {
    if (0..CHUNK_SIZE).contains(&x) && (0..CHUNK_SIZE).contains(&y) && (0..CHUNK_SIZE).contains(&z) {
        return chunk.get_block(x, y, z) != 0;
    }

    if let Some(neighbor_chunk) = neighbor {
        return neighbor_chunk.get_block(x.rem_euclid(CHUNK_SIZE), y.rem_euclid(CHUNK_SIZE), z.rem_euclid(CHUNK_SIZE))
            != 0;
    }

    neighbor_solid_at_world(chunk_x * CHUNK_SIZE + x, chunk_y * CHUNK_SIZE + y, chunk_z * CHUNK_SIZE + z)
}

fn build_face_mask<F: Fn(i32, i32, i32) -> bool>(
    chunk: &Chunk,
    neighbor: Option<&Chunk>,
    chunk_x: i32,
    chunk_y: i32,
    chunk_z: i32,
    dir: usize,
    layer: i32,
    neighbor_solid_at_world: &F,
    mask: &mut FaceMask,
) {
    debug_assert!(
        mask.iter().flatten().all(|&cell| cell == 0),
        "Mask-Scratch war beim Eintritt nicht vollstaendig 0 - Selbstreinigungs-Invariante von \
         `merge_mask_into_faces` verletzt"
    );

    let (ox, oy, oz) = NEIGHBOR_OFFSETS[dir];

    for u in 0..CHUNK_SIZE {
        for v in 0..CHUNK_SIZE {
            let (x, y, z) = pos_from_layer_uv(dir, layer, u, v);
            let block_id = chunk.get_block(x, y, z);
            if block_id == 0 {
                continue;
            }

            let neighbor_solid = is_solid(
                chunk,
                neighbor,
                chunk_x,
                chunk_y,
                chunk_z,
                x + ox,
                y + oy,
                z + oz,
                neighbor_solid_at_world,
            );
            if !neighbor_solid {
                mask[u as usize][v as usize] = block_id;
            }
        }
    }
}

/// Greedy-Merge UND Face-Encoding in einem Durchgang: verbrauchte Zellen werden auf 0
/// zurueckgesetzt (dadurch ist die Maske nach dieser Funktion garantiert wieder komplett 0 -
/// `build_face_mask` muss sie folglich nie explizit leeren) und jedes fertige Rechteck wird SOFORT
/// als komprimiertes u32 in `output` gepusht. Es existiert bewusst keine Zwischen-Liste aus
/// Rechtecken mehr, die spaeter erst in den Output kopiert wuerde.
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

    MASK_SCRATCH.with_borrow_mut(|mask| {
        for dir in 0..6 {
            let neighbor = neighbors[dir];
            for layer in 0..CHUNK_SIZE {
                build_face_mask(
                    chunk,
                    neighbor,
                    chunk_x,
                    chunk_y,
                    chunk_z,
                    dir,
                    layer,
                    &neighbor_solid_at_world,
                    mask,
                );
                merge_mask_into_faces(mask, dir, layer, &mut mesh.faces[dir]);
            }
        }
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
    fn mask_scratch_is_clean_after_repeated_calls() {
        // Regressionstest fuer die Selbstreinigungs-Invariante: mehrere aufeinanderfolgende
        // `mesh_chunk`-Aufrufe im selben Thread duerfen den `debug_assert` in `build_face_mask`
        // nicht verletzen (liefe in Debug-Builds sofort in einen Panic, falls die Maske nicht
        // vollstaendig auf 0 zurueckgesetzt wird).
        for i in 0..3 {
            let mut chunk = Chunk::empty();
            chunk.set_block(i, i, i, 1);
            let _ = mesh_chunk(&chunk, 0, 0, 0, [None; 6], no_world_neighbors);
        }
    }
}
