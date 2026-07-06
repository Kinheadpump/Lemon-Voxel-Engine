use crate::game::world::chunk::{Chunk, CHUNK_SIZE};

pub const DIR_NEG_X: usize = 0;
pub const DIR_POS_X: usize = 1;
pub const DIR_NEG_Y: usize = 2;
pub const DIR_POS_Y: usize = 3;
pub const DIR_NEG_Z: usize = 4;
pub const DIR_POS_Z: usize = 5;

const NEIGHBOR_OFFSETS: [(i32, i32, i32); 6] = [
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
];

const CHUNK_SIZE_USIZE: usize = CHUNK_SIZE as usize;

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
/// Array genutzt. Liegt sie ausserhalb (X/Y/Z-Chunkgrenze), wird `neighbor_solid_at_world`
/// befragt, damit an Chunk-Nahtstellen (auch vertikal, zwischen gestapelten Chunks) keine
/// Phantom-Waende entstehen, obwohl der Nachbar-Chunk dort tatsaechlich Terrain hat.
#[inline(always)]
fn is_solid<F: Fn(i32, i32, i32) -> bool>(
    chunk: &Chunk,
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

    neighbor_solid_at_world(chunk_x * CHUNK_SIZE + x, chunk_y * CHUNK_SIZE + y, chunk_z * CHUNK_SIZE + z)
}

fn build_face_mask<F: Fn(i32, i32, i32) -> bool>(
    chunk: &Chunk,
    chunk_x: i32,
    chunk_y: i32,
    chunk_z: i32,
    dir: usize,
    layer: i32,
    neighbor_solid_at_world: &F,
) -> [[u16; CHUNK_SIZE_USIZE]; CHUNK_SIZE_USIZE] {
    let mut mask = [[0u16; CHUNK_SIZE_USIZE]; CHUNK_SIZE_USIZE];
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

    mask
}

fn greedy_merge_mask(
    mask: &mut [[u16; CHUNK_SIZE_USIZE]; CHUNK_SIZE_USIZE],
) -> Vec<(usize, usize, usize, usize, u16)> {
    let mut rects = Vec::new();

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

            rects.push((u, v, width, height, texture_id));
        }
    }

    rects
}

pub fn mesh_chunk<F: Fn(i32, i32, i32) -> bool>(
    chunk: &Chunk,
    chunk_x: i32,
    chunk_y: i32,
    chunk_z: i32,
    neighbor_solid_at_world: F,
) -> DirectionalMesh {
    let mut mesh = DirectionalMesh::default();

    for dir in 0..6 {
        for layer in 0..CHUNK_SIZE {
            let mut mask =
                build_face_mask(chunk, chunk_x, chunk_y, chunk_z, dir, layer, &neighbor_solid_at_world);

            for (u, v, width, height, texture_id) in greedy_merge_mask(&mut mask) {
                let (x, y, z) = pos_from_layer_uv(dir, layer, u as i32, v as i32);
                let face = encode_face(
                    x as u32,
                    y as u32,
                    z as u32,
                    texture_id as u32,
                    (width - 1) as u32,
                    (height - 1) as u32,
                );
                mesh.faces[dir].push(face);
            }
        }
    }

    mesh
}
