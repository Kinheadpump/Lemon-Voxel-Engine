// Mikrobenchmark fuer `TerrainGenerator::generate_chunk` UND den realistischen End-zu-End-Pfad des
// asynchronen Chunk-Ladens (`ChunkManager::dispatch_pending`): Generierung + Meshing, wobei das
// Meshing IMMER auf den prozeduralen `is_solid`-Fallback zurueckfaellt (nie echte Nachbar-Chunk-
// Referenzen - das ist auf dem Rayon-Worker-Thread unmoeglich, s. Kommentar an
// `ChunkManager::dispatch_pending`). Das ist der mit Abstand haeufigste Aufrufpfad in der Praxis
// (jeder frisch generierte Chunk beim initialen Laden/schnellen Fliegen).

use std::time::Instant;

use voxel_engine::engine::config::EngineConfig;
use voxel_engine::engine::core::mesher::mesh_chunk;
use voxel_engine::game::world::chunk::Chunk;
use voxel_engine::game::world::generator::TerrainGenerator;

const WARMUP_ITERATIONS: usize = 50;
const MEASURED_ITERATIONS: usize = 2000;

fn bench_generate(label: &str, generator: &TerrainGenerator, chunk_x: i32, chunk_y: i32, chunk_z: i32) -> f64 {
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

fn bench_generate_and_mesh(
    label: &str,
    generator: &TerrainGenerator,
    chunk_x: i32,
    chunk_y: i32,
    chunk_z: i32,
) -> f64 {
    let neighbor_solid = |x: i32, y: i32, z: i32| generator.is_solid(x, y, z);
    let mut chunk = Chunk::empty();

    for _ in 0..WARMUP_ITERATIONS {
        generator.generate_chunk(chunk_x, chunk_y, chunk_z, &mut chunk);
        std::hint::black_box(mesh_chunk(&chunk, chunk_x, chunk_y, chunk_z, [None; 6], neighbor_solid));
    }

    let start = Instant::now();
    for _ in 0..MEASURED_ITERATIONS {
        generator.generate_chunk(chunk_x, chunk_y, chunk_z, &mut chunk);
        std::hint::black_box(mesh_chunk(&chunk, chunk_x, chunk_y, chunk_z, [None; 6], neighbor_solid));
    }
    let elapsed = start.elapsed();

    let per_chunk_us = elapsed.as_secs_f64() * 1_000_000.0 / MEASURED_ITERATIONS as f64;
    println!("{label}: {per_chunk_us:.2} us/Chunk ueber {MEASURED_ITERATIONS} Iterationen");
    per_chunk_us
}

fn main() {
    let config = EngineConfig::default();
    let generator = TerrainGenerator::new(&config);

    // (4, -10, 8) per `examples/tunnel_diagnostic.rs` als dichtestes bekanntes Tunnelgebiet
    // identifiziert (29.5% Luft/Hoehlen) - realistischer Tunnel-Worst-Case statt reinem Zufallstreffer.
    println!("-- reine Generierung --");
    let surface_us = bench_generate("Oberflaeche (chunk_y=0)", &generator, 4, 0, 4);
    let underground_us = bench_generate("Tiefe (chunk_y=-5)", &generator, 4, -5, 4);
    let sky_us = bench_generate("Himmel (chunk_y=20)", &generator, 4, 20, 4);
    let tunnel_us = bench_generate("Tunnelgebiet (chunk_y=-10)", &generator, 4, -10, 8);

    println!("-- Generierung + Meshing, IMMER mit is_solid-Fallback (realistischer Lade-Pfad) --");
    let surface_mesh_us = bench_generate_and_mesh("Oberflaeche+Mesh (chunk_y=0)", &generator, 4, 0, 4);
    let underground_mesh_us = bench_generate_and_mesh("Tiefe+Mesh (chunk_y=-5)", &generator, 4, -5, 4);
    let tunnel_mesh_us = bench_generate_and_mesh("Tunnelgebiet+Mesh (chunk_y=-10)", &generator, 4, -10, 8);

    let target_met = surface_us < 2000.0
        && underground_us < 2000.0
        && sky_us < 2000.0
        && tunnel_us < 2000.0
        && surface_mesh_us < 2000.0
        && underground_mesh_us < 2000.0
        && tunnel_mesh_us < 2000.0;
    println!("Ziel <2ms/Chunk: {}", if target_met { "ERREICHT" } else { "VERFEHLT" });
}
