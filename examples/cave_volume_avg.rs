use voxel_engine::engine::config::EngineConfig;
use voxel_engine::game::world::chunk::Chunk;
use voxel_engine::game::world::generator::TerrainGenerator;

fn main() {
    let config = EngineConfig::default();
    let generator = TerrainGenerator::new(&config);
    let mut chunk = Chunk::empty();
    let mut total_solid = 0u64;
    let mut total_blocks = 0u64;
    let mut worst = 100.0f64;
    let mut best = 0.0f64;
    let mut n = 0;
    for cz in -8..8 {
        for cx in -8..8 {
            for cy in [-2, -5, -10, -15] {
                generator.generate_chunk(cx, cy, cz, &mut chunk);
                let solid = (0..32768).filter(|&i| chunk.blocks[i] != 0).count() as u64;
                total_solid += solid;
                total_blocks += 32768;
                let air_pct = (1.0 - solid as f64 / 32768.0) * 100.0;
                worst = worst.min(100.0 - air_pct);
                best = best.max(air_pct);
                n += 1;
            }
        }
    }
    println!("Chunks: {n}, Durchschnitt solide: {:.1}%, schlechtester Chunk Luftanteil: {:.1}%",
        total_solid as f64 / total_blocks as f64 * 100.0, best);
}
