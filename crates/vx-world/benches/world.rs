use divan::black_box;
use vx_core::ChunkPos;
use vx_world::{ChunkVoxels, TerrainGenerator, TerrainSettings, new_dense_chunk};

fn main() {
    divan::main();
}

fn surface_chunk() -> ChunkPos {
    ChunkPos::new(3, 2, -5)
}

#[divan::bench]
fn generate_surface_chunk(bencher: divan::Bencher) {
    let generator = TerrainGenerator::new(TerrainSettings::default());
    let mut scratch = new_dense_chunk();
    bencher.bench_local(|| generator.generate(black_box(surface_chunk()), &mut scratch));
}

#[divan::bench]
fn palette_decode_surface_chunk(bencher: divan::Bencher) {
    let generator = TerrainGenerator::new(TerrainSettings::default());
    let mut scratch = new_dense_chunk();
    let voxels = generator.generate(surface_chunk(), &mut scratch);
    assert!(matches!(voxels, ChunkVoxels::Packed(_)));
    bencher.bench_local(|| black_box(&voxels).decode_into(&mut scratch));
}
