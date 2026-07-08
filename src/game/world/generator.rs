use bevy_math::{Vec2, Vec3};
use noiz::Noise;
use noiz::prelude::*;

use crate::engine::config::EngineConfig;

use super::blocks;
use super::chunk::{CHUNK_SIZE, Chunk};
use super::terrain_grid::{CaveGrid, HeightGrid};

/// Fixe Meereshoehe - keine Konfigurationsoption, sondern eine architektonische Festlegung: alle
/// Shaping-Funktionen (See-Kompression, Straende) sind relativ zu `y=0` formuliert.
const SEA_LEVEL: i32 = 0;

/// Formt die "cliffy" Regional-Karte: Exponent < 1 auf `|noise|` drueckt die meisten Werte schnell
/// Richtung +-1 (breite Plateaus), nur nahe der Nulldurchgaenge bleibt eine schmale, steile Rampe -
/// das ist die "Erosion Discontinuity" aus Yosemite-artigen Klippen ohne echtes 3D-Dichtefeld.
const CLIFF_CONTRAST_EXPONENT: f32 = 0.35;
/// Formt die Blend-Maske zwischen sanftem und "cliffy" Hoehenfeld: kleiner Exponent = weicherer,
/// aber dennoch kontrastreicher Uebergang zwischen den Regionen (kein hartes Ein/Aus).
const MASK_CONTRAST_EXPONENT: f32 = 0.6;
/// Der oberste Block einer Saeule wird nie von Hoehlen durchbrochen, sonst entstehen einzelne
/// Ein-Block-Loecher direkt im Gras.
const MIN_CAVE_DEPTH: i32 = 1;

#[inline(always)]
fn signed_pow(value: f32, exponent: f32) -> f32 {
    value.signum() * value.abs().powf(exponent)
}

#[inline(always)]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Multi-Stage-Terraingenerator: 2D-Rauschen fuer Hoehe/Klippen/Straende, 3D-Rauschen nur fuer
/// Hoehlen (und dort nur unterhalb der bereits bekannten Oberflaeche) - siehe Kommentare an den
/// einzelnen Feldern fuer die Rolle jeder Rauschschicht.
pub struct TerrainGenerator {
    /// Sehr niedrige Frequenz, haelt Land/Ozean auf kontinentaler Ebene auseinander.
    continental: Noise<common_noise::Perlin>,
    continental_frequency: f32,
    continental_amplitude: f32,
    /// Sanfte Huegel-Variante der Regional-Karte.
    regional_smooth: Noise<common_noise::Perlin>,
    /// Kontrastierte Variante derselben Skala - erzeugt die Klippen-Kandidaten.
    regional_cliff: Noise<common_noise::Perlin>,
    regional_frequency: f32,
    regional_amplitude: f32,
    /// Blend-Maske zwischen `regional_smooth` und `regional_cliff`.
    cliff_mask: Noise<common_noise::Perlin>,
    cliff_mask_frequency: f32,
    sea_compression_range: f32,
    sea_compression_exponent: f32,
    height_sample_stride: i32,
    /// 3D-Perlin fuer "Cheese Caves" (Cutoff-Hoehlen).
    cave: Noise<common_noise::Perlin>,
    cave_frequency: f32,
    cave_threshold: f32,
    cave_sample_stride: i32,
    dirt_layer_depth: i32,
    /// Die Ursprungszelle des `noiz`-Gradientenrauschens (Welt-Koordinaten nahe (0,0)) ist
    /// degeneriert und liefert dort konstant 0.0 unabhaengig von der Position. Ein fixer Offset
    /// verschiebt jede Sample-Koordinate weit weg vom Ursprung und umgeht das vollstaendig.
    noise_origin_offset: f32,
}

impl TerrainGenerator {
    pub fn new(config: &EngineConfig) -> Self {
        let mut continental = Noise::<common_noise::Perlin>::default();
        continental.set_seed(config.terrain_seed);

        let mut regional_smooth = Noise::<common_noise::Perlin>::default();
        regional_smooth.set_seed(config.terrain_seed.wrapping_add(0x9E37_79B9));

        let mut regional_cliff = Noise::<common_noise::Perlin>::default();
        regional_cliff.set_seed(config.terrain_seed.wrapping_add(0x85EB_CA77));

        let mut cliff_mask = Noise::<common_noise::Perlin>::default();
        cliff_mask.set_seed(config.terrain_seed.wrapping_add(0xC2B2_AE3D));

        let mut cave = Noise::<common_noise::Perlin>::default();
        cave.set_seed(config.terrain_seed.wrapping_add(0x27D4_EB2F));

        Self {
            continental,
            continental_frequency: config.terrain_continental_frequency,
            continental_amplitude: config.terrain_continental_amplitude,
            regional_smooth,
            regional_cliff,
            regional_frequency: config.terrain_regional_frequency,
            regional_amplitude: config.terrain_regional_amplitude,
            cliff_mask,
            cliff_mask_frequency: config.terrain_cliff_mask_frequency,
            sea_compression_range: config.terrain_sea_compression_range.max(1.0),
            sea_compression_exponent: config.terrain_sea_compression_exponent,
            height_sample_stride: config.terrain_height_sample_stride,
            cave,
            cave_frequency: config.terrain_cave_frequency,
            cave_threshold: config.terrain_cave_threshold,
            cave_sample_stride: config.terrain_cave_sample_stride,
            dirt_layer_depth: config.terrain_dirt_layer_depth,
            noise_origin_offset: config.terrain_noise_origin_offset,
        }
    }

    #[inline]
    fn sample2d(&self, noise: &Noise<common_noise::Perlin>, frequency: f32, world_x: i32, world_z: i32) -> f32 {
        let point = Vec2::new(
            world_x as f32 * frequency + self.noise_origin_offset,
            world_z as f32 * frequency + self.noise_origin_offset,
        );
        noise.sample(point)
    }

    /// Exakte (nicht interpolierte) Oberflaechenhoehe - einzige Quelle der Wahrheit, genutzt fuer
    /// Chunk-Rand-/Nachbar-Abfragen und als Basis fuer das sparse interpolierte Innere eines Chunks.
    pub fn height_at(&self, world_x: i32, world_z: i32) -> i32 {
        self.raw_height_at(world_x, world_z).round() as i32
    }

    fn raw_height_at(&self, world_x: i32, world_z: i32) -> f32 {
        let continental = self.sample2d(&self.continental, self.continental_frequency, world_x, world_z);
        let smooth = self.sample2d(&self.regional_smooth, self.regional_frequency, world_x, world_z);
        let cliff_raw = self.sample2d(&self.regional_cliff, self.regional_frequency, world_x, world_z);
        let cliff = signed_pow(cliff_raw, CLIFF_CONTRAST_EXPONENT);
        let mask_raw = self.sample2d(&self.cliff_mask, self.cliff_mask_frequency, world_x, world_z);
        let blend = signed_pow(mask_raw, MASK_CONTRAST_EXPONENT) * 0.5 + 0.5;

        let regional_shape = lerp(smooth, cliff, blend);
        let raw_height =
            SEA_LEVEL as f32 + continental * self.continental_amplitude + regional_shape * self.regional_amplitude;

        self.compress_toward_sea_level(raw_height)
    }

    /// Drueckt Hoehen nahe `SEA_LEVEL` mit sanft steigender Staerke Richtung Meereshoehe (flache
    /// Straende), laesst Werte jenseits von `sea_compression_range` unveraendert linear weiterlaufen
    /// (Gebirge/Tiefsee werden nicht gedeckelt).
    fn compress_toward_sea_level(&self, height: f32) -> f32 {
        let range = self.sea_compression_range;
        let delta = height - SEA_LEVEL as f32;
        let clamped = delta.clamp(-range, range);
        let excess = delta - clamped;
        let shaped = signed_pow(clamped / range, self.sea_compression_exponent) * range;
        SEA_LEVEL as f32 + shaped + excess
    }

    fn cave_density(&self, world_x: i32, world_y: i32, world_z: i32) -> f32 {
        let point = Vec3::new(
            world_x as f32 * self.cave_frequency + self.noise_origin_offset,
            world_y as f32 * self.cave_frequency + self.noise_origin_offset,
            world_z as f32 * self.cave_frequency + self.noise_origin_offset,
        );
        self.cave.sample(point)
    }

    fn is_cave(&self, world_x: i32, world_y: i32, world_z: i32) -> bool {
        self.cave_density(world_x, world_y, world_z) > self.cave_threshold
    }

    /// Fallback-Quelle der Wahrheit fuer Voxel-Festigkeit ausserhalb geladener/editierter
    /// Chunk-Daten - genutzt vom Mesher (Nachbar-Check ueber Chunk-Grenzen an noch nicht gemeshten
    /// Chunks) UND von `ChunkManager::is_solid_at` fuer Regionen, die (noch) nicht geladen sind.
    pub fn is_solid(&self, world_x: i32, world_y: i32, world_z: i32) -> bool {
        world_y <= self.height_at(world_x, world_z) && !self.is_cave(world_x, world_y, world_z)
    }

    pub fn generate_chunk(&self, chunk_x: i32, chunk_y: i32, chunk_z: i32, chunk: &mut Chunk) {
        chunk.clear();

        let chunk_origin_x = chunk_x * CHUNK_SIZE;
        let chunk_origin_y = chunk_y * CHUNK_SIZE;
        let chunk_origin_z = chunk_z * CHUNK_SIZE;

        let height_grid = HeightGrid::fill(self.height_sample_stride, |local_x, local_z| {
            self.raw_height_at(chunk_origin_x + local_x, chunk_origin_z + local_z)
        });

        let mut local_height = [0i32; (CHUNK_SIZE * CHUNK_SIZE) as usize];
        let mut chunk_max_height = i32::MIN;
        for local_z in 0..CHUNK_SIZE {
            for local_x in 0..CHUNK_SIZE {
                let h = height_grid.sample(local_x, local_z).round() as i32;
                local_height[(local_z * CHUNK_SIZE + local_x) as usize] = h;
                chunk_max_height = chunk_max_height.max(h);
            }
        }

        // Chunk liegt vollstaendig ueber der Terrainoberflaeche - reine Luft, `chunk.clear()` oben
        // reicht bereits. Spart das gesamte 3D-Hoehlenraster (teuerster Teil der Generierung).
        if chunk_origin_y > chunk_max_height {
            return;
        }

        let cave_grid = CaveGrid::fill(self.cave_sample_stride, chunk_origin_y, |local_x, world_y, local_z| {
            self.cave_density(chunk_origin_x + local_x, world_y, chunk_origin_z + local_z)
        });

        let height_lookup = |local_x: i32, local_z: i32| -> i32 {
            if (0..CHUNK_SIZE).contains(&local_x) && (0..CHUNK_SIZE).contains(&local_z) {
                local_height[(local_z * CHUNK_SIZE + local_x) as usize]
            } else {
                self.height_at(chunk_origin_x + local_x, chunk_origin_z + local_z)
            }
        };

        let beach_half_range = (self.sea_compression_range * 0.25).max(1.0) as i32;

        for local_z in 0..CHUNK_SIZE {
            for local_x in 0..CHUNK_SIZE {
                let height = local_height[(local_z * CHUNK_SIZE + local_x) as usize];
                if chunk_origin_y > height {
                    continue;
                }

                let slope = (height - height_lookup(local_x - 1, local_z))
                    .abs()
                    .max((height - height_lookup(local_x + 1, local_z)).abs())
                    .max((height - height_lookup(local_x, local_z - 1)).abs())
                    .max((height - height_lookup(local_x, local_z + 1)).abs());
                let is_beach = (height - SEA_LEVEL).abs() <= beach_half_range;

                for local_y in 0..CHUNK_SIZE {
                    let world_y = chunk_origin_y + local_y;
                    if world_y > height {
                        continue;
                    }

                    let depth_from_surface = height - world_y;
                    let is_carved =
                        depth_from_surface >= MIN_CAVE_DEPTH && cave_grid.sample(local_x, local_y, local_z) > self.cave_threshold;
                    if is_carved {
                        continue;
                    }

                    let block_id = blocks::surface_block(depth_from_surface, slope, self.dirt_layer_depth, is_beach);
                    chunk.set_block(local_x, local_y, local_z, block_id);
                }
            }
        }
    }
}
