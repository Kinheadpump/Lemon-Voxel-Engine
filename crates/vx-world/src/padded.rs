use glam::IVec3;
use vx_core::{BlockId, CHUNK_SIZE, ChunkPos, FaceDir, LocalPos, PADDED_VOLUME, padded_index};

use crate::{ChunkVoxels, ChunkWindow, DenseChunk};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MissingNeighbour(pub ChunkPos);

pub fn gather_padded(
    window: &ChunkWindow<ChunkVoxels>,
    center: ChunkPos,
    dense_scratch: &mut DenseChunk,
    out: &mut [BlockId; PADDED_VOLUME],
) -> Result<(), MissingNeighbour> {
    let center_voxels = window.get(center).ok_or(MissingNeighbour(center))?;
    let mut neighbours = [center_voxels; 6];
    for (neighbour, dir) in neighbours.iter_mut().zip(FaceDir::ALL) {
        let pos = center.offset(IVec3::from_array(dir.normal()));
        *neighbour = window.get(pos).ok_or(MissingNeighbour(pos))?;
    }

    center_voxels.decode_into(dense_scratch);
    for y in 0..CHUNK_SIZE {
        for z in 0..CHUNK_SIZE {
            let source = (y * CHUNK_SIZE + z) * CHUNK_SIZE;
            let target = padded_index(1, y + 1, z + 1);
            out[target..target + CHUNK_SIZE]
                .copy_from_slice(&dense_scratch[source..source + CHUNK_SIZE]);
        }
    }

    for (dir, neighbour) in FaceDir::ALL.into_iter().zip(neighbours) {
        copy_border_slab(dir, neighbour, out);
    }
    Ok(())
}

fn copy_border_slab(dir: FaceDir, neighbour: &ChunkVoxels, out: &mut [BlockId; PADDED_VOLUME]) {
    let axis = dir.axis();
    let (source_depth, padded_depth) = if dir.is_positive() {
        (0, CHUNK_SIZE + 1)
    } else {
        (CHUNK_SIZE - 1, 0)
    };
    for v in 0..CHUNK_SIZE {
        for u in 0..CHUNK_SIZE {
            let [sx, sy, sz] = axis.depth_u_v_to_xyz(source_depth, u, v);
            let [px, py, pz] = axis.depth_u_v_to_xyz(padded_depth, u + 1, v + 1);
            out[padded_index(px, py, pz)] =
                neighbour.get(LocalPos::new(sx as u8, sy as u8, sz as u8));
        }
    }
}
