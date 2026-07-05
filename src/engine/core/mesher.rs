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

#[derive(Default)]
pub struct DirectionalMesh {
    pub faces: [Vec<u32>; 6],
}

#[inline(always)]
fn encode_face(x: u32, y: u32, z: u32, texture_id: u32) -> u32 {
    (x & 0x1F)
        | ((y & 0x1F) << 5)
        | ((z & 0x1F) << 10)
        | ((texture_id & 0x7F) << 15)
        | (0u32 << 22)
        | (0u32 << 27)
}

pub fn mesh_chunk(chunk: &Chunk) -> DirectionalMesh {
    let mut mesh = DirectionalMesh::default();

    for z in 0..CHUNK_SIZE {
        for y in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                let block_id = chunk.get_block(x, y, z);
                if block_id == 0 {
                    continue;
                }

                let texture_id = block_id as u32;

                for (dir, (ox, oy, oz)) in NEIGHBOR_OFFSETS.iter().enumerate() {
                    let neighbor = chunk.get_block(x + ox, y + oy, z + oz);
                    if neighbor == 0 {
                        let face = encode_face(x as u32, y as u32, z as u32, texture_id);
                        mesh.faces[dir].push(face);
                    }
                }
            }
        }
    }

    mesh
}
