// Isoliert die Kosten von `TerrainGenerator::is_solid` (Fallback-Hotpath des Meshers, bis zu ~6144
// Aufrufe/Chunk) von der reinen Chunk-Generierung, um herauszufinden, wo die Zeit tatsaechlich hingeht.

use std::time::Instant;

use voxel_engine::engine::config::EngineConfig;
use voxel_engine::game::world::generator::TerrainGenerator;

const ITERATIONS: usize = 200_000;

fn main() {
    let config = EngineConfig::default();
    let generator = TerrainGenerator::new(&config);

    for _ in 0..1000 {
        std::hint::black_box(generator.is_solid(4, 4, 4));
    }

    let start = Instant::now();
    let mut acc = 0u32;
    for i in 0..ITERATIONS {
        let x = (i % 4096) as i32;
        let z = (i / 4096) as i32;
        acc += generator.is_solid(x, 4, z) as u32;
    }
    std::hint::black_box(acc);
    let elapsed = start.elapsed();
    println!("is_solid: {:.1} ns/Aufruf ueber {ITERATIONS} Aufrufe", elapsed.as_secs_f64() * 1e9 / ITERATIONS as f64);

    let start = Instant::now();
    let mut acc2 = 0i32;
    for i in 0..ITERATIONS {
        let x = (i % 4096) as i32;
        let z = (i / 4096) as i32;
        acc2 += generator.height_at(x, z);
    }
    std::hint::black_box(acc2);
    let elapsed = start.elapsed();
    println!("height_at: {:.1} ns/Aufruf ueber {ITERATIONS} Aufrufe", elapsed.as_secs_f64() * 1e9 / ITERATIONS as f64);
}
