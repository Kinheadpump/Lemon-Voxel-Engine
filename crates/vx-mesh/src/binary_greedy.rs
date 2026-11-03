use std::ops::Range;

use vx_core::{BlockId, CHUNK_SIZE, FaceDir, PADDED_SIZE, padded_index};

use crate::{PackedQuad, PaddedVoxels};

const COLUMNS_PER_AXIS: usize = PADDED_SIZE * PADDED_SIZE;
const INTERIOR_MASK: u64 = u64::MAX >> (64 - CHUNK_SIZE);

type PlaneRows = [u64; CHUNK_SIZE];

#[derive(Default)]
pub struct ChunkMesh {
    pub quads: Vec<PackedQuad>,
    pub direction_ranges: [Range<u32>; 6],
}

impl ChunkMesh {
    pub fn clear(&mut self) {
        self.quads.clear();
        self.direction_ranges = Default::default();
    }

    pub fn direction(&self, dir: FaceDir) -> &[PackedQuad] {
        let range = &self.direction_ranges[dir.index()];
        &self.quads[range.start as usize..range.end as usize]
    }
}

pub struct BinaryGreedyMesher {
    opaque_columns: Box<[[u64; COLUMNS_PER_AXIS]; 3]>,
    layer_planes: Vec<Vec<(BlockId, PlaneRows)>>,
}

impl Default for BinaryGreedyMesher {
    fn default() -> Self {
        Self {
            opaque_columns: vec![[0; COLUMNS_PER_AXIS]; 3]
                .into_boxed_slice()
                .try_into()
                .expect("three axes"),
            layer_planes: (0..CHUNK_SIZE).map(|_| Vec::with_capacity(4)).collect(),
        }
    }
}

impl BinaryGreedyMesher {
    pub fn mesh(&mut self, voxels: &PaddedVoxels, mesh: &mut ChunkMesh) {
        mesh.clear();
        self.build_opaque_columns(voxels);
        for dir in FaceDir::ALL {
            let start = mesh.quads.len() as u32;
            self.collect_face_planes(voxels, dir);
            self.merge_planes(dir, &mut mesh.quads);
            mesh.direction_ranges[dir.index()] = start..mesh.quads.len() as u32;
        }
    }

    fn build_opaque_columns(&mut self, voxels: &PaddedVoxels) {
        let [x_columns, y_columns, z_columns] = &mut *self.opaque_columns;
        x_columns.fill(0);
        y_columns.fill(0);
        z_columns.fill(0);
        for y in 0..PADDED_SIZE {
            for z in 0..PADDED_SIZE {
                let row = padded_index(0, y, z);
                for x in 0..PADDED_SIZE {
                    if voxels[row + x].is_opaque() {
                        x_columns[z * PADDED_SIZE + y] |= 1 << x;
                        y_columns[z * PADDED_SIZE + x] |= 1 << y;
                        z_columns[y * PADDED_SIZE + x] |= 1 << z;
                    }
                }
            }
        }
    }

    fn collect_face_planes(&mut self, voxels: &PaddedVoxels, dir: FaceDir) {
        let axis = dir.axis();
        let columns = &self.opaque_columns[axis.index()];
        for planes in &mut self.layer_planes {
            planes.clear();
        }
        for v in 0..CHUNK_SIZE {
            for u in 0..CHUNK_SIZE {
                let column = columns[(v + 1) * PADDED_SIZE + (u + 1)];
                let exposed = if dir.is_positive() {
                    column & !(column >> 1)
                } else {
                    column & !(column << 1)
                };
                let mut faces = (exposed >> 1) & INTERIOR_MASK;
                while faces != 0 {
                    let depth = faces.trailing_zeros() as usize;
                    faces &= faces - 1;
                    let [x, y, z] = axis.depth_u_v_to_xyz(depth + 1, u + 1, v + 1);
                    let block = voxels[padded_index(x, y, z)];
                    plane_rows_for(&mut self.layer_planes[depth], block)[v] |= 1 << u;
                }
            }
        }
    }

    fn merge_planes(&mut self, dir: FaceDir, quads: &mut Vec<PackedQuad>) {
        let axis = dir.axis();
        for (depth, planes) in self.layer_planes.iter_mut().enumerate() {
            for (block, rows) in planes.iter_mut() {
                merge_plane(rows, |u, v, width, height| {
                    let origin = axis.depth_u_v_to_xyz(depth, u, v);
                    quads.push(PackedQuad::new(origin, width, height, dir, *block));
                });
            }
        }
    }
}

#[inline]
fn plane_rows_for(planes: &mut Vec<(BlockId, PlaneRows)>, block: BlockId) -> &mut PlaneRows {
    let index = if let Some(found) = planes.iter().position(|(existing, _)| *existing == block) {
        found
    } else {
        planes.push((block, [0; CHUNK_SIZE]));
        planes.len() - 1
    };
    &mut planes[index].1
}

#[inline]
fn merge_plane(rows: &mut PlaneRows, mut emit: impl FnMut(usize, usize, u32, u32)) {
    for v in 0..CHUNK_SIZE {
        while rows[v] != 0 {
            let u = rows[v].trailing_zeros();
            let width = (rows[v] >> u).trailing_ones();
            let run = (u64::MAX >> (64 - width)) << u;
            rows[v] &= !run;
            let mut height = 1;
            while v + height < CHUNK_SIZE && rows[v + height] & run == run {
                rows[v + height] &= !run;
                height += 1;
            }
            emit(u as usize, v, width, height as u32);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use vx_core::{Axis, BlockId};

    use super::*;
    use crate::mesh_culled_unit_faces;

    fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    fn expand_to_unit_faces(quads: &[PackedQuad]) -> HashMap<([u32; 3], FaceDir), BlockId> {
        let mut faces = HashMap::new();
        for quad in quads {
            let axis = quad.dir().axis();
            let origin = quad.origin();
            let (depth, u0, v0) = match axis {
                Axis::X => (origin[0], origin[1], origin[2]),
                Axis::Y => (origin[1], origin[0], origin[2]),
                Axis::Z => (origin[2], origin[0], origin[1]),
            };
            for dv in 0..quad.height() {
                for du in 0..quad.width() {
                    let [x, y, z] = axis.depth_u_v_to_xyz(
                        depth as usize,
                        (u0 + du) as usize,
                        (v0 + dv) as usize,
                    );
                    let key = ([x as u32, y as u32, z as u32], quad.dir());
                    assert!(
                        faces.insert(key, quad.block()).is_none(),
                        "overlapping quads at {key:?}"
                    );
                }
            }
        }
        faces
    }

    fn assert_matches_reference(voxels: &PaddedVoxels) -> usize {
        let mut mesher = BinaryGreedyMesher::default();
        let mut mesh = ChunkMesh::default();
        mesher.mesh(voxels, &mut mesh);
        let mut reference = Vec::new();
        mesh_culled_unit_faces(voxels, &mut reference);
        assert_eq!(
            expand_to_unit_faces(&mesh.quads),
            expand_to_unit_faces(&reference)
        );
        for dir in FaceDir::ALL {
            assert!(mesh.direction(dir).iter().all(|quad| quad.dir() == dir));
        }
        mesh.quads.len()
    }

    #[test]
    fn random_noise_covers_exactly_the_exposed_faces() {
        let mut rng = 0xDEAD_BEEF_CAFE_F00D;
        for density in [2, 3, 7] {
            let mut voxels = crate::new_padded_voxels();
            for block in voxels.iter_mut() {
                let roll = xorshift(&mut rng);
                if roll.is_multiple_of(density) {
                    *block = BlockId(1 + (roll >> 32) as u16 % 3);
                }
            }
            assert_matches_reference(&voxels);
        }
    }

    #[test]
    fn solid_chunk_with_air_border_is_six_full_quads() {
        let mut voxels = crate::new_padded_voxels();
        for y in 1..=CHUNK_SIZE {
            for z in 1..=CHUNK_SIZE {
                for x in 1..=CHUNK_SIZE {
                    voxels[padded_index(x, y, z)] = BlockId::STONE;
                }
            }
        }
        assert_eq!(assert_matches_reference(&voxels), 6);
    }

    #[test]
    fn solid_neighbours_cull_every_border_face() {
        let mut voxels = crate::new_padded_voxels();
        voxels.fill(BlockId::STONE);
        assert_eq!(assert_matches_reference(&voxels), 0);
    }
}
