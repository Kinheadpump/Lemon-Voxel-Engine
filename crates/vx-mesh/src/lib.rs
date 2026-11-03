mod binary_greedy;
mod culled_reference;
mod quad;

pub use binary_greedy::{BinaryGreedyMesher, ChunkMesh};
pub use culled_reference::mesh_culled_unit_faces;
pub use quad::PackedQuad;

pub type PaddedVoxels = [vx_core::BlockId; vx_core::PADDED_VOLUME];

pub fn new_padded_voxels() -> Box<PaddedVoxels> {
    vec![vx_core::BlockId::AIR; vx_core::PADDED_VOLUME]
        .into_boxed_slice()
        .try_into()
        .expect("exact padded volume")
}
