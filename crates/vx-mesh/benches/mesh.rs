use divan::black_box;
use vx_core::{BlockId, ChunkPos, PADDED_SIZE, padded_index};
use vx_mesh::{
    BinaryGreedyMesher, ChunkMesh, PaddedVoxels, mesh_culled_unit_faces, new_padded_voxels,
};
use vx_world::{
    ChunkVoxels, ChunkWindow, TerrainGenerator, TerrainSettings, gather_padded, new_dense_chunk,
};

fn main() {
    divan::main();
}

fn padded_surface_chunk() -> Box<PaddedVoxels> {
    let generator = TerrainGenerator::new(TerrainSettings::default());
    let center = ChunkPos::new(3, 2, -5);
    let mut window: ChunkWindow<ChunkVoxels> = ChunkWindow::for_view_radius(2, 2);
    let mut scratch = new_dense_chunk();
    for dx in -1..=1 {
        for dy in -1..=1 {
            for dz in -1..=1 {
                let pos = ChunkPos::new(center.0.x + dx, center.0.y + dy, center.0.z + dz);
                window.insert(pos, generator.generate(pos, &mut scratch));
            }
        }
    }
    let mut padded = new_padded_voxels();
    gather_padded(&window, center, &mut scratch, &mut padded).expect("neighbours present");
    padded
}

fn checkerboard_chunk() -> Box<PaddedVoxels> {
    let mut padded = new_padded_voxels();
    for y in 0..PADDED_SIZE {
        for z in 0..PADDED_SIZE {
            for x in 0..PADDED_SIZE {
                if (x + y + z) % 2 == 0 {
                    padded[padded_index(x, y, z)] = BlockId::STONE;
                }
            }
        }
    }
    padded
}

#[divan::bench(args = ["surface", "checkerboard"])]
fn binary_greedy(bencher: divan::Bencher, scene: &str) {
    let voxels = if scene == "surface" {
        padded_surface_chunk()
    } else {
        checkerboard_chunk()
    };
    let mut mesher = BinaryGreedyMesher::default();
    let mut mesh = ChunkMesh::default();
    mesher.mesh(&voxels, &mut mesh);
    println!("{scene}: {} quads", mesh.quads.len());
    bencher.bench_local(|| mesher.mesh(black_box(&voxels), &mut mesh));
}

#[divan::bench]
fn culled_reference_surface(bencher: divan::Bencher) {
    let voxels = padded_surface_chunk();
    let mut quads = Vec::with_capacity(16 * 1024);
    bencher.bench_local(|| {
        quads.clear();
        mesh_culled_unit_faces(black_box(&voxels), &mut quads);
    });
}

#[divan::bench]
fn gather_padded_surface(bencher: divan::Bencher) {
    let generator = TerrainGenerator::new(TerrainSettings::default());
    let center = ChunkPos::new(3, 2, -5);
    let mut window: ChunkWindow<ChunkVoxels> = ChunkWindow::for_view_radius(2, 2);
    let mut scratch = new_dense_chunk();
    for dx in -1..=1 {
        for dy in -1..=1 {
            for dz in -1..=1 {
                let pos = ChunkPos::new(center.0.x + dx, center.0.y + dy, center.0.z + dz);
                window.insert(pos, generator.generate(pos, &mut scratch));
            }
        }
    }
    let mut padded = new_padded_voxels();
    bencher.bench_local(|| gather_padded(&window, black_box(center), &mut scratch, &mut padded));
}
