use vx_core::{CHUNK_SIZE, FaceDir, padded_index};

use crate::{PackedQuad, PaddedVoxels};

pub fn mesh_culled_unit_faces(voxels: &PaddedVoxels, out: &mut Vec<PackedQuad>) {
    for dir in FaceDir::ALL {
        let [nx, ny, nz] = dir.normal();
        for y in 1..=CHUNK_SIZE {
            for z in 1..=CHUNK_SIZE {
                for x in 1..=CHUNK_SIZE {
                    let block = voxels[padded_index(x, y, z)];
                    let neighbour = voxels[padded_index(
                        x.wrapping_add_signed(nx as isize),
                        y.wrapping_add_signed(ny as isize),
                        z.wrapping_add_signed(nz as isize),
                    )];
                    if block.is_opaque() && !neighbour.is_opaque() {
                        out.push(PackedQuad::new([x - 1, y - 1, z - 1], 1, 1, dir, block));
                    }
                }
            }
        }
    }
}
