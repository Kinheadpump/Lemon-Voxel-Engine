use std::cell::RefCell;

use bevy_math::{Vec2, Vec3};
use noiz::cell_noise::WorleyDifference;
use noiz::{Noise, NoiseFunction};
use noiz::prelude::*;

use crate::engine::config::EngineConfig;

use super::blocks::{self, ColumnSurface};
use super::chunk::{CHUNK_SIZE, Chunk};

/// Fraktales Perlin (fBm) fuer die Regional-Heightmap - mehrere Octaves mit konfigurierbarer
/// Lacunarity/Gain statt einer einzelnen, glatten Frequenz (echtes Relief statt sanftem Gewoge).
type RegionalFbm = common_noise::Fbm<common_noise::Perlin>;

/// F2-F1-Distanz (WorleyDifference): "Distanz zum 2. naechsten" minus "Distanz zum naechsten"
/// Zellpunkt, unorm - nahe 0 heisst "genau auf der Voronoi-Zellgrenze". Zellgrenzen bilden (anders
/// als einfaches F1, das nur isolierte Kugeln um Zellzentren ergibt - genau die "isolierten
/// Blasen statt Tunnel", die wir NICHT wollen) ein zusammenhaengendes Netzwerk durch den GESAMTEN
/// Raum - ein Schwellwert-Cutoff darauf ergibt lange, verbundene Tunnelroehren.
type WorleyTunnel = PerCellPointDistances<Voronoi, EuclideanLength, WorleyDifference>;

/// Fixe Meereshoehe - keine Konfigurationsoption, sondern eine architektonische Festlegung: alle
/// Shaping-Funktionen (See-Kompression, Straende) sind relativ zu `y=0` formuliert.
const SEA_LEVEL: i32 = 0;
/// Wasseroberflaeche: Spalten, deren Terrainhoehe darunter liegt, werden bis hierher mit Wasser
/// aufgefuellt (`is_water_position`) - NUR ueber der Terrainoberflaeche, nie in Hoehlen (die liegen
/// per Definition UNTER der Oberflaeche und `MIN_CAVE_DEPTH` haelt die Deckschicht des Ozeanbodens
/// geschlossen, s. `is_carved`).
const WATER_LEVEL: i32 = SEA_LEVEL;

/// Formt die "cliffy" Regional-Karte: Exponent < 1 auf `|noise|` drueckt die meisten Werte Richtung
/// +-1 (breite Plateaus), nur nahe der Nulldurchgaenge bleibt eine schmale, steile Rampe - das ist
/// die "Erosion Discontinuity" aus Yosemite-artigen Klippen ohne echtes 3D-Dichtefeld. Nicht zu
/// aggressiv (0.55 statt frueher 0.35), sonst wirken die Klippen ueberall statt nur gelegentlich.
const CLIFF_CONTRAST_EXPONENT: f32 = 0.55;
/// Formt die Blend-Maske zwischen sanftem und "cliffy" Hoehenfeld: kleiner Exponent = weicherer,
/// aber dennoch kontrastreicher Uebergang zwischen den Regionen (kein hartes Ein/Aus).
const MASK_CONTRAST_EXPONENT: f32 = 0.6;
/// Der oberste Block einer Saeule wird nie von Hoehlen durchbrochen, sonst entstehen einzelne
/// Ein-Block-Loecher direkt im Gras.
const MIN_CAVE_DEPTH: i32 = 1;
/// Halbe vertikale Breite (Bloecke) des Sand-Kuestenstreifens um den Wasserspiegel - schmal und
/// FIX (nicht mehr an die Sea-Compression-Range gekoppelt, die frueher eine halbe Weltbreite Sand
/// erzeugte). Bei sanft geneigten Kuesten entspricht das ein paar Bloecken horizontalem Strand.
const BEACH_HALF_RANGE: i32 = 2;
/// Basis-Hoehe, ab der Gipfel nackten Fels statt Gras zeigen. Wird pro Spalte um die Temperatur
/// (`ROCK_HEIGHT_TEMPERATURE_DITHER`) verschoben, damit keine flache Höhenlinie entsteht.
const ROCK_HEIGHT: f32 = 92.0;
/// Warme Spalten brauchen mehr Hoehe fuer Fels, kalte weniger - dithert die Fels-Grenze mit dem
/// (bereits gesampelten, glatten) Temperaturfeld statt einer harten Kontur.
const ROCK_HEIGHT_TEMPERATURE_DITHER: f32 = 14.0;

#[inline(always)]
fn signed_pow(value: f32, exponent: f32) -> f32 {
    value.signum() * value.abs().powf(exponent)
}

#[inline(always)]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Direkt-gemapptes Memo fuer 2D-Spalten-Werte (`height_at`, `is_tunnel_region`) - Potenz von 2,
/// damit der Slot-Index eine reine Bitmaske statt eines Modulo ist. Gross genug fuer ALLE 1024
/// Spalten eines Chunks gleichzeitig (plus Kollisions-Spielraum): das direkt auf `generate_chunk`
/// folgende `mesh_chunk` desselben Rayon-Tasks prueft seine Ober-/Unterkante gegen exakt dieselben
/// 1024 Spalten (`is_solid` bei y=origin-1/origin+32) - mit zu wenigen Slots evictete der Chunk
/// seine eigenen Spalten wieder, bevor sie ein zweites Mal gebraucht wurden.
const COLUMN_CACHE_SLOTS: usize = 4096;

#[inline(always)]
fn column_cache_slot(world_x: i32, world_z: i32) -> usize {
    let hash = (world_x as u32).wrapping_mul(0x9E37_79B1) ^ (world_z as u32).wrapping_mul(0x85EB_CA6B);
    (hash as usize) & (COLUMN_CACHE_SLOTS - 1)
}

thread_local! {
    /// Thread-lokal, weil `TerrainGenerator` per `Arc` ueber Rayon-Worker geteilt wird (kein
    /// gemeinsames `RefCell`-Feld an einem per `Sync` geteilten Typ moeglich).
    static HEIGHT_CACHE: RefCell<[(i32, i32, i32); COLUMN_CACHE_SLOTS]> =
        const { RefCell::new([(i32::MIN, i32::MIN, 0); COLUMN_CACHE_SLOTS]) };
    /// Wie `HEIGHT_CACHE`, aber fuer `is_tunnel_region` - `generate_chunk` fragt pro Spalte einmal
    /// (guenstiger Cache-Miss), der Mesher-Fallback-Hotpath dafuer bis zu 32x hintereinander pro
    /// Rand-Spalte (s. `height_at`-Kommentar) - ohne Cache waere das eine zusaetzliche Rauschprobe
    /// pro dieser Wiederholungen.
    static TUNNEL_REGION_CACHE: RefCell<[(i32, i32, bool); COLUMN_CACHE_SLOTS]> =
        const { RefCell::new([(i32::MIN, i32::MIN, false); COLUMN_CACHE_SLOTS]) };
}

/// Gitterabstand (Weltbloecke) BEIDER sparse Hoehlenraster (Cheese Caves + Tunnel) - s.
/// `cheese_grid_corner`/`tunnel_grid_corner`. Fein genug, um Tunnelwaende nicht zu verlieren; fuer
/// die (bewusst grossskaligen) Cheese Caves ist das ohnehin weit unter ihrer Featuregroesse.
const CAVE_GRID_STRIDE: i32 = 4;
/// 4096 Slots * 16 Byte = 64 KiB pro Raster, thread-lokal, trivial. Ein voller Chunk braucht bei
/// Stride 4 im schlimmsten Fall 9^3=729 DISTINKTE Gitterpunkte - bei zu wenigen Slots kollidiert
/// das Direct-Mapping (Geburtstagsparadoxon) so haeufig, dass Slots wiederholt eviktiert und
/// dieselben Gitterpunkte mehrfach neu berechnet werden (empirisch beobachtet: 1024 Slots bei 729
/// Eintraegen kosteten trotz Raster noch mehrere ms/Chunk extra). Bei 4096 Slots (Lastfaktor 0.18)
/// bleibt die Kollisionsrate niedrig.
const CAVE_GRID_CACHE_SLOTS: usize = 4096;

#[inline(always)]
fn cave_grid_slot(gx: i32, gy: i32, gz: i32) -> usize {
    let hash = (gx as u32).wrapping_mul(0x9E37_79B1)
        ^ (gy as u32).wrapping_mul(0x85EB_CA6B)
        ^ (gz as u32).wrapping_mul(0xC2B2_AE35);
    (hash as usize) & (CAVE_GRID_CACHE_SLOTS - 1)
}

/// (gx, gy, gz, Rohwert) an EINEM Gitterpunkt eines der beiden Hoehlenraster.
type CaveGridCacheSlot = (i32, i32, i32, f32);

thread_local! {
    /// Cheese-Cave-Dichte (1 Perlin-Rohprobe/Gitterpunkt). Cheese Caves sind UNGEGATET (ueberall
    /// praesent, s. Kommentar an `TerrainGenerator::cheese_region_threshold`... nein, es gibt keine
    /// Gate - absichtlich, im Gegensatz zum Tunnelsystem).
    static CHEESE_GRID_CACHE: RefCell<[CaveGridCacheSlot; CAVE_GRID_CACHE_SLOTS]> =
        const { RefCell::new([(i32::MIN, i32::MIN, i32::MIN, 0.0); CAVE_GRID_CACHE_SLOTS]) };
    /// Tunnel-Worley-F2-F1-Distanz (1 Worley-Rohprobe/Gitterpunkt) - nur befuellt fuer Spalten, die
    /// `is_tunnel_region` passieren.
    static TUNNEL_GRID_CACHE: RefCell<[CaveGridCacheSlot; CAVE_GRID_CACHE_SLOTS]> =
        const { RefCell::new([(i32::MIN, i32::MIN, i32::MIN, 0.0); CAVE_GRID_CACHE_SLOTS]) };
}

/// Anzahl Gitter-Y-Ebenen, die eine volle Chunk-Spalte (`CHUNK_SIZE` Voxel) im schlechtesten Fall
/// (ungluecklichste Ausrichtung zum Gitter) beruehren kann: `CHUNK_SIZE/CAVE_GRID_STRIDE + 2`.
const CAVE_COLUMN_MAX_LAYERS: usize = (CHUNK_SIZE / CAVE_GRID_STRIDE) as usize + 2;

/// `(Anzahl gueltiger Ebenen, c00, c10, c01, c11)` - die 4 rohen XZ-Gitter-Eckwert-Stapel EINER
/// Zelle ueber alle beruehrten Y-Ebenen, s. `TerrainGenerator::cave_grid_stack`.
type CaveGridStack = (
    usize,
    [f32; CAVE_COLUMN_MAX_LAYERS],
    [f32; CAVE_COLUMN_MAX_LAYERS],
    [f32; CAVE_COLUMN_MAX_LAYERS],
    [f32; CAVE_COLUMN_MAX_LAYERS],
);

/// Ergebnis von "Extremity Bound Checking" (s. Kommentar an `bound_check`) fuer eine ganze Spalte:
/// da trilineare Interpolation eine KONVEXE Kombination ihrer 8 Eckwerte ist, liegt jeder
/// interpolierte Wert innerhalb einer Zelle garantiert zwischen deren Minimum und Maximum. Liegen
/// Minimum/Maximum einer ganzen Spalte (ueber alle beruehrten Gitter-Ecken) bereits eindeutig auf
/// einer Seite der (tiefenabhaengigen) Schwelle, ist das Ergebnis fuer JEDEN Voxel der Spalte
/// mathematisch feststehend - die teure Pro-Voxel-Interpolation+Vergleich kann komplett entfallen.
#[derive(Clone, Copy, PartialEq)]
enum CarveBound {
    Never,
    Always,
    Maybe,
}

/// Multi-Stage-Terraingenerator: 2D-Rauschen fuer Hoehe/Klippen/Straende/Biome, 3D-Rauschen nur
/// unterhalb der bereits bekannten Oberflaeche fuer die zwei Hoehlensysteme - siehe Kommentare an
/// den einzelnen Feldern fuer die Rolle jeder Rauschschicht.
pub struct TerrainGenerator {
    /// Sehr niedrige Frequenz, haelt Land/Ozean auf kontinentaler Ebene auseinander UND treibt den
    /// exponentiellen Berg-Boost (s. `raw_height_at`).
    continental: Noise<common_noise::Perlin>,
    continental_frequency: f32,
    continental_amplitude: f32,
    mountain_amplitude: f32,
    mountain_exponent: f32,
    /// Fraktale (fBm) Huegel-Variante der Regional-Karte - 4-5 Octaves fuer echtes Relief.
    regional_smooth: Noise<RegionalFbm>,
    /// Kontrastierte Einzel-Frequenz derselben Skala - erzeugt die Klippen-Kandidaten.
    regional_cliff: Noise<common_noise::Perlin>,
    regional_frequency: f32,
    regional_amplitude: f32,
    /// Blend-Maske zwischen `regional_smooth` und `regional_cliff`.
    cliff_mask: Noise<common_noise::Perlin>,
    cliff_mask_frequency: f32,
    sea_compression_range: f32,
    sea_compression_exponent: f32,
    /// Strikte 2D-Biom-Achsen: Wueste nur bei hoher Temperatur UND geringer Feuchtigkeit.
    temperature: Noise<common_noise::Perlin>,
    temperature_frequency: f32,
    humidity: Noise<common_noise::Perlin>,
    humidity_frequency: f32,
    desert_temperature_min: f32,
    desert_humidity_max: f32,
    /// Grosse Cheese Caves: 3D-Perlin-Cutoff mit niedriger Frequenz (ausgedehnte Kavernen statt
    /// kleiner Blasen). Ungegatet - ueberall im Untergrund praesent.
    cheese: Noise<common_noise::Perlin>,
    cheese_frequency: f32,
    cheese_threshold: f32,
    /// Lange, verbundene Tunnel via einfacher Worley-Zellgrenzen (`WorleyTunnel`) statt zweier
    /// Ridged-Perlin-Karten - guenstiger (1 statt 2+ Rauschproben/Gitterpunkt) UND naeher am
    /// Wunsch nach "simplem Worley Noise fuer lange Tunnel".
    tunnel: Noise<WorleyTunnel>,
    tunnel_frequency: f32,
    tunnel_threshold: f32,
    /// Gemeinsamer Tiefenfaktor: sowohl Cheese Caves als auch Tunnel werden graduell groesser, je
    /// weiter man unter `SEA_LEVEL` kommt - erreicht sein Maximum nach dieser Blocktiefe.
    cave_widen_depth_range: f32,
    /// Um wie viel `cheese_threshold` in maximaler Tiefe SINKT (leichter zu ueberschreiten -> mehr
    /// Volumen ausgehoehlt).
    cheese_widen_amount: f32,
    /// Um welchen Faktor `tunnel_threshold` in maximaler Tiefe MULTIPLIZIERT wird (groesser ->
    /// breitere Roehren, da der Cutoff auf der F2-F1-Distanz dann grosszuegiger ist).
    tunnel_widen_multiplier: f32,
    /// Grobes 2D-Gate (1 Rauschprobe, gecached): nur "Hoehlen-aktive" Regionen zahlen ueberhaupt
    /// fuer das Tunnelsystem - s. Kommentar an `is_tunnel_region`.
    cave_region: Noise<common_noise::Perlin>,
    cave_region_frequency: f32,
    cave_region_threshold: f32,
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

        let mut regional_smooth = Noise::from(LayeredNoise::new(
            Normed::default(),
            Persistence(config.terrain_regional_gain),
            FractalLayers {
                layer: Octave(common_noise::Perlin::default()),
                lacunarity: config.terrain_regional_lacunarity,
                amount: config.terrain_regional_octaves,
            },
        ));
        regional_smooth.set_seed(config.terrain_seed.wrapping_add(0x9E37_79B9));

        let mut regional_cliff = Noise::<common_noise::Perlin>::default();
        regional_cliff.set_seed(config.terrain_seed.wrapping_add(0x85EB_CA77));

        let mut cliff_mask = Noise::<common_noise::Perlin>::default();
        cliff_mask.set_seed(config.terrain_seed.wrapping_add(0xC2B2_AE3D));

        let mut temperature = Noise::<common_noise::Perlin>::default();
        temperature.set_seed(config.terrain_seed.wrapping_add(0x68E3_1DA4));
        let mut humidity = Noise::<common_noise::Perlin>::default();
        humidity.set_seed(config.terrain_seed.wrapping_add(0xB529_7A4D));

        let mut cheese = Noise::<common_noise::Perlin>::default();
        cheese.set_seed(config.terrain_seed.wrapping_add(0x27D4_EB2F));

        let mut tunnel = Noise::<WorleyTunnel>::default();
        tunnel.set_seed(config.terrain_seed.wrapping_add(0x9E3B_2265));

        let mut cave_region = Noise::<common_noise::Perlin>::default();
        cave_region.set_seed(config.terrain_seed.wrapping_add(0x1656_67B1));

        Self {
            continental,
            continental_frequency: config.terrain_continental_frequency,
            continental_amplitude: config.terrain_continental_amplitude,
            mountain_amplitude: config.terrain_mountain_amplitude,
            mountain_exponent: config.terrain_mountain_exponent,
            regional_smooth,
            regional_cliff,
            regional_frequency: config.terrain_regional_frequency,
            regional_amplitude: config.terrain_regional_amplitude,
            cliff_mask,
            cliff_mask_frequency: config.terrain_cliff_mask_frequency,
            sea_compression_range: config.terrain_sea_compression_range.max(1.0),
            sea_compression_exponent: config.terrain_sea_compression_exponent,
            temperature,
            temperature_frequency: config.terrain_temperature_frequency,
            humidity,
            humidity_frequency: config.terrain_humidity_frequency,
            desert_temperature_min: config.terrain_desert_temperature_min,
            desert_humidity_max: config.terrain_desert_humidity_max,
            cheese,
            cheese_frequency: config.terrain_cheese_frequency,
            cheese_threshold: config.terrain_cheese_threshold,
            tunnel,
            tunnel_frequency: config.terrain_tunnel_frequency,
            tunnel_threshold: config.terrain_tunnel_threshold,
            cave_widen_depth_range: config.terrain_cave_widen_depth_range.max(1.0),
            cheese_widen_amount: config.terrain_cheese_widen_amount,
            tunnel_widen_multiplier: config.terrain_tunnel_widen_multiplier,
            cave_region,
            cave_region_frequency: config.terrain_cave_region_frequency,
            cave_region_threshold: config.terrain_cave_region_threshold,
            dirt_layer_depth: config.terrain_dirt_layer_depth,
            noise_origin_offset: config.terrain_noise_origin_offset,
        }
    }

    #[inline]
    fn sample2d<N: NoiseFunction<Vec2, Output = f32>>(
        &self,
        noise: &Noise<N>,
        frequency: f32,
        world_x: i32,
        world_z: i32,
    ) -> f32 {
        let point = Vec2::new(
            world_x as f32 * frequency + self.noise_origin_offset,
            world_z as f32 * frequency + self.noise_origin_offset,
        );
        noise.sample(point)
    }

    #[inline]
    fn sample3d<N: NoiseFunction<Vec3, Output = f32>>(
        &self,
        noise: &Noise<N>,
        frequency: f32,
        world_x: i32,
        world_y: i32,
        world_z: i32,
    ) -> f32 {
        let point = Vec3::new(
            world_x as f32 * frequency + self.noise_origin_offset,
            world_y as f32 * frequency + self.noise_origin_offset,
            world_z as f32 * frequency + self.noise_origin_offset,
        );
        noise.sample(point)
    }

    /// Exakte Oberflaechenhoehe - einzige Quelle der Wahrheit, genutzt fuer Chunk-Rand-/Nachbar-
    /// Abfragen UND als per-Spalte-Wert innerhalb von `generate_chunk`.
    ///
    /// Gecached ueber `HEIGHT_CACHE`: der Mesher fragt seinen Boundary-Fallback pro Rand-SPALTE ab
    /// (s. `compute_exposure` in mesher.rs), aber die X-/Z-Richtungs-Checks iterieren dabei ueber die
    /// jeweils andere Achse mit - fragen denselben (world_x, world_z) bis zu 32x hintereinander (mit
    /// dem jeweils anderen Rand-Nachbarn verschraenkt) ab, obwohl die Hoehe gar nicht von Y abhaengt.
    /// Ohne Cache kostet jeder dieser Aufrufe erneut 4 Rauschproben (~200ns) - bei bis zu 6144
    /// Fallback-Aufrufen/Chunk machte allein das den Mesher deutlich langsamer als das <1ms-Ziel. Der
    /// Mesher selbst bleibt dabei bewusst ahnungslos ueber diese Terrain-Interna (kein Sonderfall im
    /// generischen Meshing-Code).
    pub fn height_at(&self, world_x: i32, world_z: i32) -> i32 {
        let slot = column_cache_slot(world_x, world_z);
        HEIGHT_CACHE.with_borrow_mut(|cache| {
            let (cached_x, cached_z, cached_height) = cache[slot];
            if cached_x == world_x && cached_z == world_z {
                return cached_height;
            }
            let height = self.raw_height_at(world_x, world_z).round() as i32;
            cache[slot] = (world_x, world_z, height);
            height
        })
    }

    fn raw_height_at(&self, world_x: i32, world_z: i32) -> f32 {
        let continental = self.sample2d(&self.continental, self.continental_frequency, world_x, world_z);
        let smooth = self.sample2d(&self.regional_smooth, self.regional_frequency, world_x, world_z);
        let cliff_raw = self.sample2d(&self.regional_cliff, self.regional_frequency, world_x, world_z);
        let cliff = signed_pow(cliff_raw, CLIFF_CONTRAST_EXPONENT);
        let mask_raw = self.sample2d(&self.cliff_mask, self.cliff_mask_frequency, world_x, world_z);
        let blend = signed_pow(mask_raw, MASK_CONTRAST_EXPONENT) * 0.5 + 0.5;

        // Continentalness-Shaping: `unorm^exponent` haelt Ebenen/Meere flach (unorm um 0.5 traegt
        // kaum bei) und laesst NUR die Kontinentalmaxima exponentiell zu Bergmassiven hochschiessen -
        // reuse desselben Kontinental-Samples, keine zusaetzliche Rauschprobe.
        let mountain = ((continental + 1.0) * 0.5).powf(self.mountain_exponent) * self.mountain_amplitude;

        let regional_shape = lerp(smooth, cliff, blend);
        let raw_height = SEA_LEVEL as f32
            + continental * self.continental_amplitude
            + mountain
            + regional_shape * self.regional_amplitude;

        self.compress_toward_sea_level(raw_height)
    }

    /// Wasserfuellung: NUR ueber der Terrainoberflaeche bis zum Wasserspiegel (Ozeane/Seen in
    /// Senken) - Positionen unter der Oberflaeche (auch ausgehoehlte) bleiben unberuehrt, Hoehlen
    /// fluten also nie. Reines O(1)-Praedikat aus der Spaltenhoehe, keine zusaetzliche Rauschprobe -
    /// exakt dieselbe Formel fuer `generate_chunk` (Befuellung) und `is_solid` (Fallback).
    #[inline(always)]
    fn is_water_position(height: i32, world_y: i32) -> bool {
        world_y > height && world_y <= WATER_LEVEL
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

    /// 0 bei/ueber `SEA_LEVEL`, waechst linear bis 1 nach `cave_widen_depth_range` Bloecken Tiefe
    /// und bleibt danach dort - der EINE gemeinsame Tiefenfaktor, den sowohl Cheese Caves als auch
    /// Tunnel auf ihre jeweilige Schwelle anwenden (s. `cheese_threshold_at`/`tunnel_threshold_at`).
    #[inline(always)]
    fn depth_factor(&self, world_y: i32) -> f32 {
        ((SEA_LEVEL - world_y) as f32 / self.cave_widen_depth_range).clamp(0.0, 1.0)
    }

    /// Effektive Cheese-Cave-Schwelle bei `world_y`: SINKT mit der Tiefe (leichter zu
    /// ueberschreiten) - groessere Kavernen, je tiefer man kommt.
    #[inline(always)]
    fn cheese_threshold_at(&self, world_y: i32) -> f32 {
        self.cheese_threshold - self.cheese_widen_amount * self.depth_factor(world_y)
    }

    /// Effektive Tunnel-Schwelle bei `world_y`: WAECHST mit der Tiefe (grosszuegigerer Cutoff auf
    /// der F2-F1-Distanz) - breitere Roehren, je tiefer man kommt.
    #[inline(always)]
    fn tunnel_threshold_at(&self, world_y: i32) -> f32 {
        self.tunnel_threshold * (1.0 + self.tunnel_widen_multiplier * self.depth_factor(world_y))
    }

    /// Rohe Cheese-Cave-Dichte an EINEM Gitterpunkt (`gx`/`gy`/`gz` bereits durch
    /// `CAVE_GRID_STRIDE` geteilt), `CHEESE_GRID_CACHE`-gecached.
    fn cheese_grid_corner(&self, gx: i32, gy: i32, gz: i32) -> f32 {
        let slot = cave_grid_slot(gx, gy, gz);
        CHEESE_GRID_CACHE.with_borrow_mut(|cache| {
            let (cached_x, cached_y, cached_z, value) = cache[slot];
            if cached_x == gx && cached_y == gy && cached_z == gz {
                return value;
            }
            let (wx, wy, wz) = (gx * CAVE_GRID_STRIDE, gy * CAVE_GRID_STRIDE, gz * CAVE_GRID_STRIDE);
            let value = self.sample3d(&self.cheese, self.cheese_frequency, wx, wy, wz);
            cache[slot] = (gx, gy, gz, value);
            value
        })
    }

    /// Rohe Tunnel-F2-F1-Distanz an EINEM Gitterpunkt, `TUNNEL_GRID_CACHE`-gecached. Nur aufgerufen,
    /// wenn `is_tunnel_region` bereits zugestimmt hat.
    fn tunnel_grid_corner(&self, gx: i32, gy: i32, gz: i32) -> f32 {
        let slot = cave_grid_slot(gx, gy, gz);
        TUNNEL_GRID_CACHE.with_borrow_mut(|cache| {
            let (cached_x, cached_y, cached_z, value) = cache[slot];
            if cached_x == gx && cached_y == gy && cached_z == gz {
                return value;
            }
            let (wx, wy, wz) = (gx * CAVE_GRID_STRIDE, gy * CAVE_GRID_STRIDE, gz * CAVE_GRID_STRIDE);
            let value = self.sample3d(&self.tunnel, self.tunnel_frequency, wx, wy, wz);
            cache[slot] = (gx, gy, gz, value);
            value
        })
    }

    /// Trilineare Interpolation EINES Hoehlenrasters an einer beliebigen Weltposition, ueber die 8
    /// umschliessenden Gitterpunkte - fuer Einzelabfragen (`is_solid`-Fallback via `is_carved`).
    /// Nutzt DIESELBEN Gitter-Ecken (`corner`) wie die Zellen-Bulk-Fuellung `cave_grid_stack` in
    /// `generate_chunk` - beide Pfade duerfen fuer denselben Voxel NIE unterschiedliche Werte
    /// liefern, s. Kommentar an `is_carved`.
    fn cave_fields_at(&self, corner: impl Fn(&Self, i32, i32, i32) -> f32, world_x: i32, world_y: i32, world_z: i32) -> f32 {
        let gx0 = world_x.div_euclid(CAVE_GRID_STRIDE);
        let gy0 = world_y.div_euclid(CAVE_GRID_STRIDE);
        let gz0 = world_z.div_euclid(CAVE_GRID_STRIDE);
        let tx = world_x.rem_euclid(CAVE_GRID_STRIDE) as f32 / CAVE_GRID_STRIDE as f32;
        let ty = world_y.rem_euclid(CAVE_GRID_STRIDE) as f32 / CAVE_GRID_STRIDE as f32;
        let tz = world_z.rem_euclid(CAVE_GRID_STRIDE) as f32 / CAVE_GRID_STRIDE as f32;

        let xz_layer = |gy: i32| {
            let c00 = corner(self, gx0, gy, gz0);
            let c10 = corner(self, gx0 + 1, gy, gz0);
            let c01 = corner(self, gx0, gy, gz0 + 1);
            let c11 = corner(self, gx0 + 1, gy, gz0 + 1);
            lerp(lerp(c00, c10, tx), lerp(c01, c11, tx), tz)
        };
        lerp(xz_layer(gy0), xz_layer(gy0 + 1), ty)
    }

    /// Grobes, sehr guenstiges 2D-Gate (1 Rauschprobe): nur in "Hoehlen-aktiven" Regionen wird das
    /// Tunnelsystem ueberhaupt ausgewertet. Reale Tunnelsysteme sind regional konzentriert
    /// (Karstgebiete) statt gleichmaessig verteilt - dieses Gate bildet das nach UND spart in den
    /// meisten Bereichen des Untergrunds die komplette Worley-Auswertung, statt jeden Voxel im
    /// Spiel dafuer bezahlen zu lassen. Cheese Caves bleiben bewusst UNGEGATET (sollen ueberall
    /// vorkommen koennen).
    ///
    /// Gecached ueber `TUNNEL_REGION_CACHE` - Y-unabhaengig wie `height_at`, mit identischem
    /// Wiederholungsmuster im Mesher-Fallback-Hotpath (s. dortigen Kommentar).
    fn is_tunnel_region(&self, world_x: i32, world_z: i32) -> bool {
        let slot = column_cache_slot(world_x, world_z);
        TUNNEL_REGION_CACHE.with_borrow_mut(|cache| {
            let (cached_x, cached_z, cached_region) = cache[slot];
            if cached_x == world_x && cached_z == world_z {
                return cached_region;
            }
            let region =
                self.sample2d(&self.cave_region, self.cave_region_frequency, world_x, world_z) > self.cave_region_threshold;
            cache[slot] = (world_x, world_z, region);
            region
        })
    }

    /// Vereinigung der beiden Hoehlensysteme (Cheese Caves + regional gegatetes Tunnelnetz) UNTER
    /// dem `MIN_CAVE_DEPTH`-Mindestabstand zur Oberflaeche - einzige Stelle, die diese Regel kennt.
    /// Nutzt `cave_fields_at` (trilineare Einzelabfrage ueber DIESELBEN Gitter-Ecken wie
    /// `generate_chunk`s Zellen-Bulk-Pfad `cave_grid_stack`) - NICHT die rohe, exakte
    /// Rauschprobe. Waere dieser Fallback exakt statt interpoliert (wie er es einmal war, bevor
    /// dieser Kommentar geschrieben wurde - ein direkter `sample3d`-Aufruf sah beim Rewrite
    /// harmlos aus, brach die Konsistenz aber genauso wie beim frueheren Hoehen-Bug), koennten
    /// `is_solid` und `generate_chunk` fuer denselben Voxel unterschiedliche Antworten geben -
    /// dauerhafte Luecken an Chunk-Naehten (niemand re-mesht spaeter) bzw. ein Spieler bleibt im
    /// nachtraeglich materialisierten Fels stecken. Der Regressionstest unten deckt das ab.
    fn is_carved(&self, world_x: i32, world_y: i32, world_z: i32, depth_from_surface: i32) -> bool {
        if depth_from_surface < MIN_CAVE_DEPTH {
            return false;
        }
        let cheese_density = self.cave_fields_at(Self::cheese_grid_corner, world_x, world_y, world_z);
        if cheese_density > self.cheese_threshold_at(world_y) {
            return true;
        }
        if !self.is_tunnel_region(world_x, world_z) {
            return false;
        }
        let tunnel_diff = self.cave_fields_at(Self::tunnel_grid_corner, world_x, world_y, world_z);
        tunnel_diff < self.tunnel_threshold_at(world_y)
    }

    /// Fallback-Quelle der Wahrheit fuer OKKLUSION ("belegt diese Position einen sichtbaren Block?",
    /// inklusive Wasser - opak gerendert) ausserhalb geladener Chunk-Daten - genutzt vom Mesher
    /// (Nachbar-Check ueber Chunk-Grenzen an noch nicht gemeshten Chunks). MUSS exakt `block != 0`
    /// des spaeter generierten Chunks vorhersagen, also Wasser einschliessen - sonst entstehen an
    /// Ozean-Chunk-Naehten dieselben dauerhaften Face-Fehler wie beim fruehen Hoehen-Bug.
    pub fn is_solid(&self, world_x: i32, world_y: i32, world_z: i32) -> bool {
        let height = self.height_at(world_x, world_z);
        if Self::is_water_position(height, world_y) {
            return true;
        }
        world_y <= height && !self.is_carved(world_x, world_y, world_z, height - world_y)
    }

    /// Physik-Variante: Wasser ist begehbar/durchschwimmbar, also NICHT solide - nur echtes Terrain
    /// blockiert. Fallback fuer `ChunkManager::is_solid_at` (Kollision/Raycast), waehrend der Mesher
    /// die Okklusions-Variante `is_solid` nutzt.
    pub fn is_physically_solid(&self, world_x: i32, world_y: i32, world_z: i32) -> bool {
        let height = self.height_at(world_x, world_z);
        world_y <= height && !self.is_carved(world_x, world_y, world_z, height - world_y)
    }

    /// Praeberechnet fuer eine GANZE `CAVE_GRID_STRIDE`x`CAVE_GRID_STRIDE`-Zelle (16 Spalten) alle
    /// beruehrten Gitter-Y-Ebenen EINES Hoehlenrasters an den 4 gemeinsamen XZ-Gitter-Ecken. Diese
    /// Ecken sind fuer ALLE 16 Spalten der Zelle IDENTISCH, nur die XZ-Interpolationsgewichte
    /// (`tx`/`tz`, s. `cave_column_from_stack`) unterscheiden sich pro Spalte. Ohne dieses
    /// Zellen-Batching wuerde jede Spalte dieselben 4 Ecken erneut ueber den gehashten Cache abfragen
    /// (Hash+RefCell+Vergleich pro Treffer) - hier werden sie EINMAL pro Zelle geholt und von den 16
    /// Spalten per simplem Lerp (reine Registerarithmetik, kein weiterer Cache-Zugriff) kombiniert.
    fn cave_grid_stack(
        &self,
        corner: impl Fn(&Self, i32, i32, i32) -> f32,
        gx0: i32,
        gz0: i32,
        gy_min: i32,
        gy_max: i32,
    ) -> CaveGridStack {
        let mut c00 = [0.0f32; CAVE_COLUMN_MAX_LAYERS];
        let mut c10 = [0.0f32; CAVE_COLUMN_MAX_LAYERS];
        let mut c01 = [0.0f32; CAVE_COLUMN_MAX_LAYERS];
        let mut c11 = [0.0f32; CAVE_COLUMN_MAX_LAYERS];
        let mut count = 0usize;
        for i in 0..CAVE_COLUMN_MAX_LAYERS {
            let gy = gy_min + i as i32;
            if gy > gy_max {
                break;
            }
            c00[i] = corner(self, gx0, gy, gz0);
            c10[i] = corner(self, gx0 + 1, gy, gz0);
            c01[i] = corner(self, gx0, gy, gz0 + 1);
            c11[i] = corner(self, gx0 + 1, gy, gz0 + 1);
            count = i + 1;
        }
        (count, c00, c10, c01, c11)
    }

    /// Kombiniert eine per `cave_grid_stack` geholte Zellen-Ecken-Tabelle mit den
    /// Interpolationsgewichten EINER Spalte zu deren Y-Ebenen + Roh-Min/Max - reine Arithmetik, kein
    /// Rauschen/Cache-Zugriff.
    #[inline(always)]
    fn cave_column_from_stack(
        tx: f32,
        tz: f32,
        count: usize,
        c00: &[f32; CAVE_COLUMN_MAX_LAYERS],
        c10: &[f32; CAVE_COLUMN_MAX_LAYERS],
        c01: &[f32; CAVE_COLUMN_MAX_LAYERS],
        c11: &[f32; CAVE_COLUMN_MAX_LAYERS],
    ) -> ([f32; CAVE_COLUMN_MAX_LAYERS], [f32; CAVE_COLUMN_MAX_LAYERS], [f32; CAVE_COLUMN_MAX_LAYERS]) {
        let mut layers = [0.0f32; CAVE_COLUMN_MAX_LAYERS];
        let mut layer_min = [0.0f32; CAVE_COLUMN_MAX_LAYERS];
        let mut layer_max = [0.0f32; CAVE_COLUMN_MAX_LAYERS];
        for i in 0..count {
            let (a, b, c, d) = (c00[i], c10[i], c01[i], c11[i]);
            layer_min[i] = a.min(b).min(c).min(d);
            layer_max[i] = a.max(b).max(c).max(d);
            layers[i] = lerp(lerp(a, b, tx), lerp(c, d, tx), tz);
        }
        (layers, layer_min, layer_max)
    }

    /// Extremity Bound Checking PRO SLAB (4-Voxel-Y-Streifen zwischen zwei Gitter-Ebenen) statt fuer
    /// die gesamte Spalte: bei voll unterirdischen Chunks (Y-Spanne = 32 Voxel) ueberspannen
    /// Minimum/Maximum der GANZEN Spalte fast immer die Schwelle irgendwo - der Bound-Check
    /// degeneriert dann auf `Maybe` fuer praktisch jeden Voxel und die teure Interpolation muss
    /// trotzdem ueberall laufen. Pro Slab (nur die 8 Eckwerte der zwei angrenzenden Gitter-Ebenen)
    /// ist der Wertebereich viel enger, wodurch die meisten Slabs selbst tief unter der Oberflaeche
    /// eindeutig `Always`/`Never` werden.
    fn slab_bounds(
        gy_min: i32,
        layer_count: usize,
        layer_min: &[f32; CAVE_COLUMN_MAX_LAYERS],
        layer_max: &[f32; CAVE_COLUMN_MAX_LAYERS],
        threshold_at: impl Fn(i32) -> f32,
        is_less_than: bool,
    ) -> [CarveBound; CAVE_COLUMN_MAX_LAYERS] {
        let mut bounds = [CarveBound::Never; CAVE_COLUMN_MAX_LAYERS];
        for i in 0..layer_count.saturating_sub(1) {
            let min = layer_min[i].min(layer_min[i + 1]);
            let max = layer_max[i].max(layer_max[i + 1]);
            let y_lo = (gy_min + i as i32) * CAVE_GRID_STRIDE;
            let y_hi = y_lo + CAVE_GRID_STRIDE - 1;
            bounds[i] = Self::bound_check(min, max, y_lo, y_hi, &threshold_at, is_less_than);
        }
        bounds
    }

    /// Interpoliert `world_y` aus den per `cave_column_from_stack` vorberechneten Y-Ebenen.
    #[inline(always)]
    fn cave_from_layers(world_y: i32, gy_min: i32, layers: &[f32; CAVE_COLUMN_MAX_LAYERS]) -> f32 {
        let gy0 = world_y.div_euclid(CAVE_GRID_STRIDE);
        let ty = world_y.rem_euclid(CAVE_GRID_STRIDE) as f32 / CAVE_GRID_STRIDE as f32;
        lerp(layers[(gy0 - gy_min) as usize], layers[(gy0 - gy_min) as usize + 1], ty)
    }

    /// Extremity Bound Checking: da trilineare Interpolation eine konvexe Kombination ihrer
    /// Eckwerte ist, liegt JEDER interpolierte Wert der Spalte garantiert in `[min, max]`. Die
    /// Schwelle selbst variiert mit der Tiefe (`threshold_at`) - `is_less_than` waehlt, in welche
    /// Richtung "carved" bedeutet (Cheese: Dichte > Schwelle; Tunnel: Distanz < Schwelle). Liegt
    /// selbst die permissivste/restriktivste Schwelle im beruehrten Y-Bereich schon eindeutig
    /// jenseits von `[min, max]`, steht das Ergebnis fuer JEDEN Voxel der Spalte fest - keine
    /// Naeherung, ein mathematisch sicherer Kurzschluss.
    fn bound_check(
        min: f32,
        max: f32,
        y_min: i32,
        y_max: i32,
        threshold_at: impl Fn(i32) -> f32,
        is_less_than: bool,
    ) -> CarveBound {
        let threshold_lo = threshold_at(y_min).min(threshold_at(y_max));
        let threshold_hi = threshold_at(y_min).max(threshold_at(y_max));
        if is_less_than {
            // carved <=> wert < schwelle(y)
            if max < threshold_lo {
                CarveBound::Always
            } else if min >= threshold_hi {
                CarveBound::Never
            } else {
                CarveBound::Maybe
            }
        } else {
            // carved <=> wert > schwelle(y)
            if min > threshold_hi {
                CarveBound::Always
            } else if max <= threshold_lo {
                CarveBound::Never
            } else {
                CarveBound::Maybe
            }
        }
    }

    /// Hoehenkarte wird IMMER exakt berechnet (1 Rauschprobe, Y-unabhaengig - ein interpoliertes
    /// Innen/exaktes-Rand-Schema wuerde auf die volle 32x32-Flaeche entarten, s. Git-Historie).
    /// Cheese Caves UND Tunnel sind dagegen ueber sparse Gitter interpoliert (`cave_grid_stack`)
    /// PLUS extremity-bound-geprueft (`bound_check`) - der `is_solid`-Fallback und dieser Bulk-Pfad
    /// nutzen fuer beide Systeme dieselben Gitter-Ecken-Werte (ueber `CHEESE_GRID_CACHE`/
    /// `TUNNEL_GRID_CACHE`), nur mit unterschiedlicher Praezision/Kurzschluss-Strategie - beide
    /// koennen sich also nie widersprechen.
    pub fn generate_chunk(&self, chunk_x: i32, chunk_y: i32, chunk_z: i32, chunk: &mut Chunk) {
        chunk.clear();

        let chunk_origin_x = chunk_x * CHUNK_SIZE;
        let chunk_origin_y = chunk_y * CHUNK_SIZE;
        let chunk_origin_z = chunk_z * CHUNK_SIZE;

        let mut local_height = [0i32; (CHUNK_SIZE * CHUNK_SIZE) as usize];
        let mut chunk_max_height = i32::MIN;
        for local_z in 0..CHUNK_SIZE {
            for local_x in 0..CHUNK_SIZE {
                let h = self.height_at(chunk_origin_x + local_x, chunk_origin_z + local_z);
                local_height[(local_z * CHUNK_SIZE + local_x) as usize] = h;
                chunk_max_height = chunk_max_height.max(h);
            }
        }

        // Chunk liegt vollstaendig ueber Terrainoberflaeche UND Wasserspiegel - reine Luft,
        // `chunk.clear()` oben reicht bereits. Spart die komplette Hoehlen-/Wasser-Auswertung.
        if chunk_origin_y > chunk_max_height && chunk_origin_y > WATER_LEVEL {
            return;
        }

        let height_lookup = |local_x: i32, local_z: i32| -> i32 {
            if (0..CHUNK_SIZE).contains(&local_x) && (0..CHUNK_SIZE).contains(&local_z) {
                local_height[(local_z * CHUNK_SIZE + local_x) as usize]
            } else {
                self.height_at(chunk_origin_x + local_x, chunk_origin_z + local_z)
            }
        };

        // Hoehlenraster-Y-Spanne ist fuer den GANZEN Chunk identisch (haengt nur von
        // `chunk_origin_y` ab, nicht von der individuellen Spaltenhoehe) - einmal ausserhalb aller
        // Schleifen bestimmt.
        let cave_gy_min = chunk_origin_y.div_euclid(CAVE_GRID_STRIDE);
        let cave_gy_max = (chunk_origin_y + CHUNK_SIZE - 1).div_euclid(CAVE_GRID_STRIDE) + 1;
        let cells_per_axis = CHUNK_SIZE / CAVE_GRID_STRIDE;

        for cell_z in 0..cells_per_axis {
            for cell_x in 0..cells_per_axis {
                let cell_origin_x = chunk_origin_x + cell_x * CAVE_GRID_STRIDE;
                let cell_origin_z = chunk_origin_z + cell_z * CAVE_GRID_STRIDE;
                let gx0 = cell_origin_x.div_euclid(CAVE_GRID_STRIDE);
                let gz0 = cell_origin_z.div_euclid(CAVE_GRID_STRIDE);

                // Auf die tatsaechlich benoetigte Y-Spanne DIESER Zelle geklemmt (nicht die ganze
                // Chunk-Hoehe) - an der Oberflaeche liegt die groesste lokale Hoehe oft weit unter
                // dem Chunk-Top, und ohne dieses Clipping wuerden flache Zellen unnoetig weit ueber
                // ihre eigene Terrainoberflaeche hinaus Gitter-Ecken auswerten.
                let mut cell_max_height = i32::MIN;
                for cell_local_z in 0..CAVE_GRID_STRIDE {
                    for cell_local_x in 0..CAVE_GRID_STRIDE {
                        let local_x = cell_x * CAVE_GRID_STRIDE + cell_local_x;
                        let local_z = cell_z * CAVE_GRID_STRIDE + cell_local_z;
                        cell_max_height = cell_max_height.max(local_height[(local_z * CHUNK_SIZE + local_x) as usize]);
                    }
                }
                let cell_gy_max = cave_gy_max.min(cell_max_height.min(chunk_origin_y + CHUNK_SIZE - 1).div_euclid(CAVE_GRID_STRIDE) + 1);

                // Cheese Caves: ungegatet, EINMAL PRO ZELLE (16 Spalten) statt pro Spalte geholt -
                // s. Kommentar an `cave_grid_stack`.
                let (cheese_count, cheese_c00, cheese_c10, cheese_c01, cheese_c11) =
                    self.cave_grid_stack(Self::cheese_grid_corner, gx0, gz0, cave_gy_min, cell_gy_max);

                // Tunnel-Zellen-Stack wird NUR bei Bedarf geholt (erste Spalte der Zelle mit
                // `is_tunnel_region == true`) und dann fuer den Rest der Zelle wiederverwendet - bei
                // `cave_region_frequency` (500-Block-Wellenlaenge) ist das Ergebnis innerhalb einer
                // 4-Block-Zelle so gut wie immer fuer alle 16 Spalten identisch.
                let mut tunnel_stack: Option<CaveGridStack> = None;

                for cell_local_z in 0..CAVE_GRID_STRIDE {
                    for cell_local_x in 0..CAVE_GRID_STRIDE {
                        let local_x = cell_x * CAVE_GRID_STRIDE + cell_local_x;
                        let local_z = cell_z * CAVE_GRID_STRIDE + cell_local_z;
                        let height = local_height[(local_z * CHUNK_SIZE + local_x) as usize];
                        let column_has_terrain = chunk_origin_y <= height;
                        let column_has_water = height < WATER_LEVEL && chunk_origin_y <= WATER_LEVEL;
                        if !column_has_terrain && !column_has_water {
                            continue;
                        }

                        let world_x = chunk_origin_x + local_x;
                        let world_z = chunk_origin_z + local_z;

                        // Wasserfuellung zuerst - billig (kein Rauschen, s. `is_water_position`) und
                        // unabhaengig von Hangneigung/Biom/Hoehlen.
                        if column_has_water {
                            let water_bottom = (height + 1).max(chunk_origin_y);
                            let water_top = WATER_LEVEL.min(chunk_origin_y + CHUNK_SIZE - 1);
                            for world_y in water_bottom..=water_top {
                                chunk.set_block(local_x, world_y - chunk_origin_y, local_z, blocks::WATER);
                            }
                        }
                        if !column_has_terrain {
                            continue;
                        }

                        let slope = (height - height_lookup(local_x - 1, local_z))
                            .abs()
                            .max((height - height_lookup(local_x + 1, local_z)).abs())
                            .max((height - height_lookup(local_x, local_z - 1)).abs())
                            .max((height - height_lookup(local_x, local_z + 1)).abs());
                        let surface = self.column_surface(world_x, world_z, height);

                        let tx = world_x.rem_euclid(CAVE_GRID_STRIDE) as f32 / CAVE_GRID_STRIDE as f32;
                        let tz = world_z.rem_euclid(CAVE_GRID_STRIDE) as f32 / CAVE_GRID_STRIDE as f32;

                        // Bounds PRO SLAB (4-Voxel-Y-Streifen) statt fuer die ganze Spalte - bei voll
                        // unterirdischen 32-Voxel-Spalten wuerde ein spaltenweiter Bound fast immer
                        // auf `Maybe` degenerieren (s. Kommentar an `slab_bounds`), pro Slab loesen
                        // sich die meisten Streifen dagegen eindeutig auf.
                        let (cheese_layers, cheese_layer_min, cheese_layer_max) = Self::cave_column_from_stack(
                            tx,
                            tz,
                            cheese_count,
                            &cheese_c00,
                            &cheese_c10,
                            &cheese_c01,
                            &cheese_c11,
                        );
                        let cheese_bounds = Self::slab_bounds(
                            cave_gy_min,
                            cheese_count,
                            &cheese_layer_min,
                            &cheese_layer_max,
                            |y| self.cheese_threshold_at(y),
                            false,
                        );
                        let cheese_any_carve = cheese_bounds[..cheese_count.saturating_sub(1)]
                            .iter()
                            .any(|b| *b != CarveBound::Never);

                        // Tunnel: nur ausgewertet, wenn die Spalte ueberhaupt in einer
                        // Hoehlen-aktiven Region liegt - CarveBound::Never ohne jede
                        // Gitter-Auswertung, wenn nicht.
                        let tunnel_region = self.is_tunnel_region(world_x, world_z);
                        let (tunnel_count, tunnel_bounds, tunnel_any_carve, tunnel_layers) = if tunnel_region {
                            let (count, c00, c10, c01, c11) = tunnel_stack.get_or_insert_with(|| {
                                self.cave_grid_stack(Self::tunnel_grid_corner, gx0, gz0, cave_gy_min, cell_gy_max)
                            });
                            let (layers, layer_min, layer_max) =
                                Self::cave_column_from_stack(tx, tz, *count, c00, c10, c01, c11);
                            let bounds = Self::slab_bounds(
                                cave_gy_min,
                                *count,
                                &layer_min,
                                &layer_max,
                                |y| self.tunnel_threshold_at(y),
                                true,
                            );
                            let any_carve = bounds[..count.saturating_sub(1)].iter().any(|b| *b != CarveBound::Never);
                            (*count, bounds, any_carve, layers)
                        } else {
                            (0, [CarveBound::Never; CAVE_COLUMN_MAX_LAYERS], false, [0.0; CAVE_COLUMN_MAX_LAYERS])
                        };

                        for local_y in 0..CHUNK_SIZE {
                            let world_y = chunk_origin_y + local_y;
                            if world_y > height {
                                continue;
                            }

                            let depth_from_surface = height - world_y;
                            if depth_from_surface >= MIN_CAVE_DEPTH && (cheese_any_carve || tunnel_any_carve) {
                                let slab = (world_y.div_euclid(CAVE_GRID_STRIDE) - cave_gy_min) as usize;
                                let cheese_slab = cheese_bounds[slab];
                                let tunnel_slab = if tunnel_count > 0 { tunnel_bounds[slab] } else { CarveBound::Never };
                                let carved = cheese_slab == CarveBound::Always
                                    || tunnel_slab == CarveBound::Always
                                    || (cheese_slab == CarveBound::Maybe
                                        && Self::cave_from_layers(world_y, cave_gy_min, &cheese_layers)
                                            > self.cheese_threshold_at(world_y))
                                    || (tunnel_slab == CarveBound::Maybe
                                        && Self::cave_from_layers(world_y, cave_gy_min, &tunnel_layers)
                                            < self.tunnel_threshold_at(world_y));
                                if carved {
                                    continue;
                                }
                            }

                            let block_id = blocks::surface_block(depth_from_surface, slope, self.dirt_layer_depth, surface);
                            chunk.set_block(local_x, local_y, local_z, block_id);
                        }
                    }
                }
            }
        }
    }

    /// Oberflaechen-Kontext einer Spalte: Hoehenband-Strand, Unterwasser-Boden und das strikte
    /// 2D-Biom-Mapping (Wueste NUR bei Temperatur > min UND Feuchtigkeit < max). Temperatur/
    /// Feuchtigkeit sind sehr niedrigfrequent + hart geschwellt - grosse zusammenhaengende Biome,
    /// kein Einzelspalten-Bleeding. Bewusst NICHT Teil von `is_solid`: Biome aendern nur die
    /// Block-ID (Sand vs. Gras), nie die Festigkeit - keine Konsistenzanforderung an den Fallback,
    /// keine zusaetzlichen Rauschproben im Mesher-/Physik-Hotpath.
    fn column_surface(&self, world_x: i32, world_z: i32, height: i32) -> ColumnSurface {
        let temperature = self.sample2d(&self.temperature, self.temperature_frequency, world_x, world_z);
        let humidity = self.sample2d(&self.humidity, self.humidity_frequency, world_x, world_z);
        let rock_height = ROCK_HEIGHT + temperature * ROCK_HEIGHT_TEMPERATURE_DITHER;
        ColumnSurface {
            is_beach: (height - WATER_LEVEL).abs() <= BEACH_HALF_RANGE,
            is_underwater: height < WATER_LEVEL,
            is_desert: temperature > self.desert_temperature_min && humidity < self.desert_humidity_max,
            is_rock: height as f32 > rock_height,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::config::EngineConfig;

    /// `TerrainGenerator::is_solid` ist die Vorhersage, auf die sowohl der Mesher (Nachbar-Check an
    /// noch nicht geladenen Chunks) als auch `ChunkManager::is_solid_at` (Physik/Raycast waehrend
    /// ein Chunk noch generiert) zurueckfallen. Sie MUSS mit dem uebereinstimmen, was
    /// `generate_chunk` tatsaechlich erzeugt - sonst entstehen dauerhafte Luecken an Chunk-Naehten
    /// (niemand re-mesht spaeter) bzw. bewegt sich ein Spieler waehrend des Ladens durch eine
    /// Stelle, an der sich die Vorhersage nachtraeglich als falsch herausstellt, bleibt er im
    /// nachtraeglich materialisierten Fels stecken. Prueft deshalb ALLE 32768 Voxel (nicht nur den
    /// Rand) ueber viele diverse Chunk-Koordinaten inkl. vertikaler Stapelung - eine fruehere Version
    /// mit nur 4 Koordinaten hat einen Depth-Gate-Bug nicht gefangen, weil er nur selten am obersten
    /// Block einer Saeule ausgeloest wird.
    #[test]
    fn is_solid_prediction_matches_generated_blocks_everywhere() {
        let generator = TerrainGenerator::new(&EngineConfig::default());

        let coords: Vec<(i32, i32, i32)> = [(0, 0, 0), (3, -2, -5), (-4, 1, 2), (7, 0, -1)]
            .into_iter()
            .chain((0..12).map(|i| (i * 5 - 30, (i % 5) - 2, i * 7 - 40)))
            .collect();

        for &(chunk_x, chunk_y, chunk_z) in &coords {
            let mut chunk = Chunk::empty();
            generator.generate_chunk(chunk_x, chunk_y, chunk_z, &mut chunk);

            let origin_x = chunk_x * CHUNK_SIZE;
            let origin_y = chunk_y * CHUNK_SIZE;
            let origin_z = chunk_z * CHUNK_SIZE;

            for local_y in 0..CHUNK_SIZE {
                for local_z in 0..CHUNK_SIZE {
                    for local_x in 0..CHUNK_SIZE {
                        let world = (origin_x + local_x, origin_y + local_y, origin_z + local_z);
                        let predicted = generator.is_solid(world.0, world.1, world.2);
                        let actual = chunk.get_block(local_x, local_y, local_z) != 0;
                        assert_eq!(
                            predicted, actual,
                            "Voxel {world:?} in Chunk ({chunk_x},{chunk_y},{chunk_z}) lokal \
                             ({local_x},{local_y},{local_z}): is_solid-Vorhersage={predicted}, \
                             tatsaechlich generiert={actual}"
                        );
                    }
                }
            }
        }
    }

    /// Diagnose-Tool, KEIN Korrektheitstest - misst die tatsaechliche Verteilung von Cheese-Cave-
    /// Dichte, Tunnel-F2-F1-Distanz und Region-Gate an vielen Punkten, um die Schwellwerte empirisch
    /// zu kalibrieren statt sie zu erraten (ein naiv "klein wirkender" Schwellwert kann je nach
    /// Rauschverteilung trotzdem den Grossteil des Volumens aushoehlen). Manuell ausfuehren mit:
    /// `cargo test --release --lib -- --ignored --nocapture calibrate_cave_thresholds`
    #[test]
    #[ignore = "Diagnose-Tool, kein automatisierter Test - siehe Doc-Kommentar"]
    fn calibrate_cave_thresholds() {
        let generator = TerrainGenerator::new(&EngineConfig::default());
        const N: usize = 200_000;

        let percentile = |values: &mut [f32], p: f64| -> f32 {
            values.sort_unstable_by(|a, b| a.total_cmp(b));
            values[((values.len() as f64 - 1.0) * p) as usize]
        };

        let mut cheese: Vec<f32> = (0..N)
            .map(|i| {
                let (x, y, z) = (i as i32 * 37 - 700_000, i as i32 * 11 - 50_000, i as i32 * 53 - 900_000);
                generator.sample3d(&generator.cheese, generator.cheese_frequency, x, y, z)
            })
            .collect();
        println!(
            "cheese density: p50={:.4} p80={:.4} p90={:.4} p95={:.4} p99={:.4}",
            percentile(&mut cheese, 0.50),
            percentile(&mut cheese, 0.80),
            percentile(&mut cheese, 0.90),
            percentile(&mut cheese, 0.95),
            percentile(&mut cheese, 0.99),
        );

        let mut tunnel: Vec<f32> = (0..N)
            .map(|i| {
                let (x, y, z) = (i as i32 * 37 - 700_000, i as i32 * 11 - 50_000, i as i32 * 53 - 900_000);
                generator.sample3d(&generator.tunnel, generator.tunnel_frequency, x, y, z)
            })
            .collect();
        println!(
            "tunnel F2-F1: p1={:.4} p2={:.4} p5={:.4} p10={:.4} p50={:.4}",
            percentile(&mut tunnel, 0.01),
            percentile(&mut tunnel, 0.02),
            percentile(&mut tunnel, 0.05),
            percentile(&mut tunnel, 0.10),
            percentile(&mut tunnel, 0.50),
        );

        let mut region: Vec<f32> = (0..N)
            .map(|i| {
                let (x, z) = (i as i32 * 37 - 700_000, i as i32 * 53 - 900_000);
                generator.sample2d(&generator.cave_region, generator.cave_region_frequency, x, z)
            })
            .collect();
        println!(
            "cave_region: p50={:.4} p60={:.4} p70={:.4} p80={:.4} p90={:.4}",
            percentile(&mut region, 0.50),
            percentile(&mut region, 0.60),
            percentile(&mut region, 0.70),
            percentile(&mut region, 0.80),
            percentile(&mut region, 0.90),
        );
    }
}
