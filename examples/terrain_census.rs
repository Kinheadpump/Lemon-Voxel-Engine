// Diagnose: Hoehen-Histogramm + Oberflaechen-Block-Zaehlung ueber ein grosses Gebiet, um die
// gemeldeten Symptome (zu kleine Meere, Sand ueberall, chaotisches Relief) mit Zahlen zu belegen.

use voxel_engine::engine::config::EngineConfig;
use voxel_engine::engine::render::textures::{
    TEXTURE_LAYER_DIRT, TEXTURE_LAYER_GRASS, TEXTURE_LAYER_SAND, TEXTURE_LAYER_STONE, TEXTURE_LAYER_WATER,
};
use voxel_engine::game::world::chunk::{CHUNK_SIZE, Chunk};
use voxel_engine::game::world::generator::TerrainGenerator;

const AREA_CHUNKS: i32 = 24; // 24*32 = 768 Bloecke Kantenlaenge

fn main() {
    let config = EngineConfig::default();
    let generator = TerrainGenerator::new(&config);

    // 1) Hoehen-Histogramm ueber alle Spalten.
    let mut heights = Vec::new();
    let half = AREA_CHUNKS * CHUNK_SIZE / 2;
    for wz in (-half..half).step_by(2) {
        for wx in (-half..half).step_by(2) {
            heights.push(generator.height_at(wx, wz));
        }
    }
    heights.sort_unstable();
    let pct = |p: f64| heights[((heights.len() as f64 - 1.0) * p) as usize];
    let ocean = heights.iter().filter(|&&h| h < 0).count() as f64 / heights.len() as f64 * 100.0;
    let near_sea = heights.iter().filter(|&&h| h.abs() <= 5).count() as f64 / heights.len() as f64 * 100.0;
    println!("== Hoehen ({} Spalten) ==", heights.len());
    println!(
        "min={} p5={} p25={} p50={} p75={} p95={} max={}",
        heights[0],
        pct(0.05),
        pct(0.25),
        pct(0.50),
        pct(0.75),
        pct(0.95),
        heights[heights.len() - 1]
    );
    println!("Ozean (h<0): {ocean:.1}%   |h|<=5 (Strandband-Kandidat): {near_sea:.1}%");

    // Glaettheits-Check: Transekt entlang X und maximale Pro-Block-Hoehendifferenz (Steilheit).
    let mut max_step = 0;
    let mut prev = generator.height_at(-half, 0);
    for wx in -half + 1..half {
        let h = generator.height_at(wx, 0);
        max_step = max_step.max((h - prev).abs());
        prev = h;
    }
    println!("Max. Hoehensprung pro Block entlang X-Transekt: {max_step} (>4 = abrupt/klippig)");
    print!("Transekt-Ausschnitt (je 8 Bloecke): ");
    for wx in (0..256).step_by(8) {
        print!("{} ", generator.height_at(wx, 0));
    }
    println!();

    // 2) Oberflaechen-Block-Zaehlung: pro Spalte den obersten nicht-Luft-Block ermitteln.
    let mut counts = [0u64; 6];
    let mut total = 0u64;
    // Chunk-Y-Bereich, der die gemessene Hoehenspanne + Wasserspiegel abdeckt.
    let cy_lo = (heights[0].div_euclid(CHUNK_SIZE)) - 1;
    let cy_hi = (heights[heights.len() - 1].div_euclid(CHUNK_SIZE)) + 1;

    let mut chunk = Chunk::empty();
    for cz in -AREA_CHUNKS / 2..AREA_CHUNKS / 2 {
        for cx in -AREA_CHUNKS / 2..AREA_CHUNKS / 2 {
            // Pro Spalte den obersten sichtbaren Block ueber den vertikalen Stapel finden.
            let mut surface = [0u16; (CHUNK_SIZE * CHUNK_SIZE) as usize];
            for cy in cy_lo..=cy_hi {
                generator.generate_chunk(cx, cy, cz, &mut chunk);
                for lz in 0..CHUNK_SIZE {
                    for lx in 0..CHUNK_SIZE {
                        for ly in 0..CHUNK_SIZE {
                            let b = chunk.get_block(lx, ly, lz);
                            if b != 0 {
                                surface[(lz * CHUNK_SIZE + lx) as usize] = b;
                            }
                        }
                    }
                }
            }
            for &b in surface.iter() {
                total += 1;
                if (b as u32) < 6 {
                    counts[b as usize] += 1;
                }
            }
        }
    }

    let f = |n: u64| n as f64 / total as f64 * 100.0;
    println!("\n== Oberflaechen-Bloecke ({total} Spalten) ==");
    println!("Luft (h.<cy_lo?): {:.1}%", f(counts[0]));
    println!("Gras:   {:.1}%", f(counts[TEXTURE_LAYER_GRASS as usize]));
    println!("Erde:   {:.1}%", f(counts[TEXTURE_LAYER_DIRT as usize]));
    println!("Stein:  {:.1}%", f(counts[TEXTURE_LAYER_STONE as usize]));
    println!("Sand:   {:.1}%", f(counts[TEXTURE_LAYER_SAND as usize]));
    println!("Wasser: {:.1}%", f(counts[TEXTURE_LAYER_WATER as usize]));
}
