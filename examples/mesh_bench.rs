// Mikrobenchmark fuer die Greedy-Meshing-Pipeline: misst die reine `mesh_chunk`-Zeit (kein I/O,
// keine GPU-Uploads) fuer einen realistischen, an der Terrainoberflaeche liegenden Chunk. Warm-Lauf
// (nach Aufwaermphase) ist die relevante Zahl - im echten Betrieb meshed ein Rayon-Worker-Thread
// ueber seine gesamte Lebenszeit viele Chunks nacheinander, der Thread-lokale Mask-Puffer ist also
// so gut wie nie "kalt".

use std::time::Instant;

use voxel_engine::engine::config::EngineConfig;
use voxel_engine::engine::core::mesher::mesh_chunk;
use voxel_engine::game::world::chunk::Chunk;
use voxel_engine::game::world::generator::TerrainGenerator;

const WARMUP_ITERATIONS: usize = 50;
const MEASURED_ITERATIONS: usize = 2000;

fn main() {
    let config = EngineConfig::default();
    let generator = TerrainGenerator::new(&config);

    // chunk_y=0 deckt world_y 0..32 ab - bei terrain_base_height=12 liegt die Oberflaeche mitten im
    // Chunk, das erzeugt den ungünstigsten Fall (viele Luft/Fels-Uebergaenge, keine trivial leere
    // oder trivial volle Ebene wie bei Chunks weit ueber/unter der Oberflaeche).
    let mut chunk = Chunk::empty();
    generator.generate_chunk(4, 0, 4, &mut chunk);
    assert!(!chunk.is_empty(), "Benchmark-Chunk ist leer - taugt nicht als realistischer Testfall");

    let neighbor_solid = |x: i32, y: i32, z: i32| generator.is_solid(x, y, z);

    for _ in 0..WARMUP_ITERATIONS {
        std::hint::black_box(mesh_chunk(&chunk, 4, 0, 4, [None; 6], neighbor_solid));
    }

    let start = Instant::now();
    for _ in 0..MEASURED_ITERATIONS {
        std::hint::black_box(mesh_chunk(&chunk, 4, 0, 4, [None; 6], neighbor_solid));
    }
    let elapsed = start.elapsed();

    let per_chunk_us = elapsed.as_secs_f64() * 1_000_000.0 / MEASURED_ITERATIONS as f64;
    println!("Warm-Meshing: {per_chunk_us:.2} us/Chunk ueber {MEASURED_ITERATIONS} Iterationen (Oberflaechen-Chunk)");

    // Zweiter Testfall: durchgehend massiver Chunk (tief unter der Oberflaeche) - Face-Anzahl nahe 0
    // (fast alles intern verdeckt), zeigt die Kosten des reinen Masken-Scans ohne nennenswerten
    // Merge-/Encode-Anteil.
    let mut solid_chunk = Chunk::empty();
    generator.generate_chunk(4, -5, 4, &mut solid_chunk);
    assert!(!solid_chunk.is_empty());

    for _ in 0..WARMUP_ITERATIONS {
        std::hint::black_box(mesh_chunk(&solid_chunk, 4, -5, 4, [None; 6], neighbor_solid));
    }
    let start = Instant::now();
    for _ in 0..MEASURED_ITERATIONS {
        std::hint::black_box(mesh_chunk(&solid_chunk, 4, -5, 4, [None; 6], neighbor_solid));
    }
    let elapsed_solid = start.elapsed();
    let per_chunk_us_solid = elapsed_solid.as_secs_f64() * 1_000_000.0 / MEASURED_ITERATIONS as f64;
    println!("Warm-Meshing: {per_chunk_us_solid:.2} us/Chunk ueber {MEASURED_ITERATIONS} Iterationen (durchgehend massiver Chunk)");

    let target_met = per_chunk_us < 1000.0 && per_chunk_us_solid < 1000.0;
    println!("Ziel <1ms/Chunk: {}", if target_met { "ERREICHT" } else { "VERFEHLT" });
}
