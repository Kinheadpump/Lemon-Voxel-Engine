use std::time::{Duration, Instant};

use rayon::prelude::*;
use vx_core::{BlockId, ChunkPos};
use vx_mesh::{BinaryGreedyMesher, ChunkMesh, new_padded_voxels};
use vx_world::{
    ChunkVoxels, ChunkWindow, TerrainGenerator, TerrainSettings, gather_padded, new_dense_chunk,
};

pub const VERTICAL_CHUNKS: std::ops::Range<i32> = 0..5;

pub struct MeshedRegion {
    pub meshes: Vec<(ChunkPos, ChunkMesh)>,
    pub generated_chunks: usize,
    pub voxel_heap_bytes: usize,
    pub generate_time: Duration,
    pub mesh_time: Duration,
}

fn columns_within(radius: i32) -> impl Iterator<Item = ChunkPos> {
    (-radius..=radius).flat_map(move |x| {
        (-radius..=radius).flat_map(move |z| VERTICAL_CHUNKS.map(move |y| ChunkPos::new(x, y, z)))
    })
}

pub fn build_region(view_radius: u32) -> MeshedRegion {
    let radius = view_radius as i32;
    let generator = TerrainGenerator::new(TerrainSettings::default());
    let generate_started = Instant::now();
    let generated: Vec<(ChunkPos, ChunkVoxels)> = columns_within(radius + 1)
        .collect::<Vec<_>>()
        .into_par_iter()
        .map_init(new_dense_chunk, |scratch, pos| {
            (pos, generator.generate(pos, scratch))
        })
        .collect();
    let generate_time = generate_started.elapsed();

    let mut window = ChunkWindow::for_view_radius(view_radius + 1, VERTICAL_CHUNKS.len() as u32);
    let generated_chunks = generated.len();
    let voxel_heap_bytes = generated
        .iter()
        .map(|(_, voxels)| voxels.heap_bytes())
        .sum();
    for (pos, voxels) in generated {
        window.insert(pos, voxels);
    }
    let border = ChunkVoxels::Uniform(BlockId::AIR);
    for x in -radius - 1..=radius + 1 {
        for z in -radius - 1..=radius + 1 {
            window.insert(
                ChunkPos::new(x, VERTICAL_CHUNKS.start - 1, z),
                border.clone(),
            );
            window.insert(ChunkPos::new(x, VERTICAL_CHUNKS.end, z), border.clone());
        }
    }

    let mesh_started = Instant::now();
    let meshes = columns_within(radius)
        .filter(|pos| window.get(*pos).and_then(ChunkVoxels::uniform_block) != Some(BlockId::AIR))
        .collect::<Vec<_>>()
        .into_par_iter()
        .map_init(
            || {
                (
                    BinaryGreedyMesher::default(),
                    new_dense_chunk(),
                    new_padded_voxels(),
                )
            },
            |(mesher, dense, padded), pos| {
                let mut mesh = ChunkMesh::default();
                if gather_padded(&window, pos, dense, padded).is_ok() {
                    mesher.mesh(padded, &mut mesh);
                }
                (pos, mesh)
            },
        )
        .filter(|(_, mesh)| !mesh.quads.is_empty())
        .collect();
    MeshedRegion {
        meshes,
        generated_chunks,
        voxel_heap_bytes,
        generate_time,
        mesh_time: mesh_started.elapsed(),
    }
}

impl MeshedRegion {
    pub fn log_summary(&self) {
        let quads: usize = self.meshes.iter().map(|(_, mesh)| mesh.quads.len()).sum();
        log::info!(
            "region: {} chunks generated in {:.1} ms ({:.1} MiB voxel heap), {} non-empty meshes / {} quads ({:.1} MiB) in {:.1} ms on {} threads",
            self.generated_chunks,
            self.generate_time.as_secs_f64() * 1e3,
            self.voxel_heap_bytes as f64 / f64::from(1u32 << 20),
            self.meshes.len(),
            quads,
            (quads * 8) as f64 / f64::from(1u32 << 20),
            self.mesh_time.as_secs_f64() * 1e3,
            rayon::current_num_threads(),
        );
    }
}
