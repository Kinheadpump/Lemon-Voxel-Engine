// Einmal-Diagnose fuer das neue Tunnelsystem: findet einen garantiert "Hoehlen-aktiven" Chunk fuer
// realistische Worst-Case-Benchmarks und druckt eine grobe Verteilungsstatistik (wie viele Spalten
// ueberhaupt Tunnel enthalten).

use voxel_engine::engine::config::EngineConfig;
use voxel_engine::game::world::chunk::Chunk;
use voxel_engine::game::world::generator::TerrainGenerator;

fn main() {
    let config = EngineConfig::default();
    let generator = TerrainGenerator::new(&config);

    // Fuer viele Chunk-XZ-Koordinaten pruefen, ob eine tiefe Saeule (weit unter jeder Oberflaeche,
    // also garantiert massiv ausser wo Hoehlen/Tunnel carven) ueberhaupt jemals carved ist.
    let mut active_columns = 0u32;
    let mut total_columns = 0u32;
    let mut worst_case_chunk: Option<(i32, i32, i32)> = None;
    let mut worst_case_carved_samples = 0usize;

    for cz in -30..30 {
        for cx in -30..30 {
            total_columns += 1;
            let world_x = cx * 32 + 16;
            let world_z = cz * 32 + 16;
            let mut carved_here = 0usize;
            for y in (-400..-100).step_by(4) {
                if !generator.is_solid(world_x, y, world_z) {
                    carved_here += 1;
                }
            }
            if carved_here > 0 {
                active_columns += 1;
            }
            if carved_here > worst_case_carved_samples {
                worst_case_carved_samples = carved_here;
                worst_case_chunk = Some((cx, -10, cz));
            }
        }
    }

    println!(
        "Tunnel-aktive Spalten (grob, Tiefensample): {active_columns}/{total_columns} ({:.1}%)",
        active_columns as f64 / total_columns as f64 * 100.0
    );

    if let Some((cx, cy, cz)) = worst_case_chunk {
        println!(
            "Chunk mit meisten Carves im Tiefensample: ({cx}, {cy}, {cz}) - {worst_case_carved_samples} von 75 Samples"
        );

        let mut chunk = Chunk::empty();
        generator.generate_chunk(cx, cy, cz, &mut chunk);
        let solid_count = (0..32 * 32 * 32).filter(|&i| chunk.blocks[i] != 0).count();
        println!(
            "Dieser Chunk: {solid_count}/32768 solide Bloecke ({:.1}% Luft/Hoehlen)",
            (1.0 - solid_count as f64 / 32768.0) * 100.0
        );
    } else {
        println!("KEIN Chunk mit Tunneln im gescannten Bereich gefunden - Schwellwerte vermutlich zu restriktiv.");
    }
}
