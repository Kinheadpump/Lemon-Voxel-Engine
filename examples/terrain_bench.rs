// Mikrobenchmark fuer `TerrainGenerator::generate_chunk`: misst die reine Generierungszeit (Hoehen-
// raster, Hangneigung, Straende, Hoehlenraster, Block-Klassifikation) ohne Meshing/GPU-Uploads.
// Warm-Lauf ist die relevante Zahl - im echten Betrieb generiert ein Rayon-Worker viele Chunks
// nacheinander.

use std::time::Instant;

use voxel_engine::engine::config::EngineConfig;
use voxel_engine::game::world::chunk::Chunk;
use voxel_engine::game::world::generator::TerrainGenerator;

const WARMUP_ITERATIONS: usize = 50;
const MEASURED_ITERATIONS: usize = 2000;

fn bench(label: &str, generator: &TerrainGenerator, chunk_x: i32, chunk_y: i32, chunk_z: i32) -> f64 {
    let mut chunk = Chunk::empty();

    for _ in 0..WARMUP_ITERATIONS {
        generator.generate_chunk(chunk_x, chunk_y, chunk_z, &mut chunk);
        std::hint::black_box(&chunk);
    }

    let start = Instant::now();
    for _ in 0..MEASURED_ITERATIONS {
        generator.generate_chunk(chunk_x, chunk_y, chunk_z, &mut chunk);
        std::hint::black_box(&chunk);
    }
    let elapsed = start.elapsed();

    let per_chunk_us = elapsed.as_secs_f64() * 1_000_000.0 / MEASURED_ITERATIONS as f64;
    println!("{label}: {per_chunk_us:.2} us/Chunk ueber {MEASURED_ITERATIONS} Iterationen");
    per_chunk_us
}

fn main() {
    let config = EngineConfig::default();
    let generator = TerrainGenerator::new(&config);

    // Oberflaechen-Chunk: Hoehen-, Hangneigungs- UND Hoehlenraster aktiv - der teuerste Fall.
    let surface_us = bench("Oberflaeche (chunk_y=0)", &generator, 4, 0, 4);
    // Tief unter der Oberflaeche: kein fruehzeitiger Sky-Abbruch, aber trivialer Hoehen-Test pro
    // Saeule - Hoehlenraster bleibt aktiv.
    let underground_us = bench("Tiefe (chunk_y=-5)", &generator, 4, -5, 4);
    // Weit ueber der Oberflaeche: fruehzeitiger Abbruch nach dem Hoehenraster, kein Hoehlenraster.
    let sky_us = bench("Himmel (chunk_y=20)", &generator, 4, 20, 4);

    let target_met = surface_us < 2000.0 && underground_us < 2000.0 && sky_us < 2000.0;
    println!("Ziel <2ms/Chunk: {}", if target_met { "ERREICHT" } else { "VERFEHLT" });
}
