use std::cell::RefCell;

use bevy_math::{Vec2, Vec3};
use noiz::{Noise, NoiseFunction};
use noiz::prelude::*;

use crate::engine::config::EngineConfig;

use super::blocks::{self, ColumnSurface};
use super::chunk::{CHUNK_SIZE, Chunk};

/// Fraktales Perlin (fBm) fuer die Regional-Heightmap - mehrere Octaves mit konfigurierbarer
/// Lacunarity/Gain statt einer einzelnen, glatten Frequenz (echtes Relief statt sanftem Gewoge).
type RegionalFbm = common_noise::Fbm<common_noise::Perlin>;

/// Fixe Meereshoehe - keine Konfigurationsoption, sondern eine architektonische Festlegung: alle
/// Shaping-Funktionen (See-Kompression, Straende) sind relativ zu `y=0` formuliert.
const SEA_LEVEL: i32 = 0;
/// Wasseroberflaeche: Spalten, deren Terrainhoehe darunter liegt, werden bis hierher mit Wasser
/// aufgefuellt (`is_water_position`) - NUR ueber der Terrainoberflaeche, nie in Hoehlen (die liegen
/// per Definition UNTER der Oberflaeche und `MIN_CAVE_DEPTH` haelt die Deckschicht des Ozeanbodens
/// geschlossen, s. `is_carved`).
const WATER_LEVEL: i32 = SEA_LEVEL;

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

/// Direkt-gemapptes Memo fuer 2D-Spalten-Werte (`height_at`, `is_tunnel_region`) - Potenz von 2,
/// damit der Slot-Index eine reine Bitmaske statt eines Modulo ist. Gross genug fuer ALLE 1024
/// Spalten eines Chunks gleichzeitig (plus Kollisions-Spielraum): das direkt auf `generate_chunk`
/// folgende `mesh_chunk` desselben Rayon-Tasks prueft seine Ober-/Unterkante gegen exakt dieselben
/// 1024 Spalten (`is_solid` bei y=origin-1/origin+32) - mit nur 128 Slots evictete der Chunk seine
/// eigenen Spalten, und der Mesher bezahlte alle ~2000 Randabfragen erneut mit der vollen
/// Hoehenberechnung (8 Rauschproben). 4096 Slots * 12 B = 48 KiB thread-lokal, im L2 irrelevant.
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

/// Gitterabstand (Weltbloecke) des sparse ausgewerteten Tunnel-Dichterasters - s. `tunnel_grid_corner`.
const TUNNEL_GRID_STRIDE: i32 = 4;
/// 4096 Slots * 20 Byte = 80 KiB, trivial. Ein voller Chunk braucht bei Stride 4 im schlimmsten Fall
/// 9^3=729 DISTINKTE Gitterpunkte - bei nur 1024 Slots kollidierte das Direct-Mapping laut
/// Geburtstagsparadoxon so haeufig (~729 Eintraege auf 1024 Slots, Lastfaktor 0.71), dass Slots
/// wiederholt eviktiert und dieselben Gitterpunkte mehrfach neu berechnet wurden - ein "Tiefe"-
/// Testchunk kostete dadurch trotz Grid noch 8ms statt der erwarteten <1ms. Bei 4096 Slots
/// (Lastfaktor 0.18) bleibt die Kollisionsrate niedrig.
const TUNNEL_GRID_CACHE_SLOTS: usize = 4096;

#[inline(always)]
fn tunnel_grid_slot(gx: i32, gy: i32, gz: i32) -> usize {
    let hash = (gx as u32).wrapping_mul(0x9E37_79B1)
        ^ (gy as u32).wrapping_mul(0x85EB_CA6B)
        ^ (gz as u32).wrapping_mul(0xC2B2_AE35);
    (hash as usize) & (TUNNEL_GRID_CACHE_SLOTS - 1)
}

/// Die 4 Ridged-Traegerwerte an einem Gitterpunkt: `[|tunnel_a|, |tunnel_b|, |connector_a|,
/// |connector_b|]`. Die Betraege werden VOR der Interpolation genommen (an den Gitterpunkten) und
/// dann pro Kanal getrennt interpoliert - glatter als der Betrag eines interpolierten Werts.
type TunnelGridChannels = [f32; 4];

/// (gx, gy, gz, Kanalwerte) an EINEM Gitterpunkt.
type TunnelGridCacheSlot = (i32, i32, i32, TunnelGridChannels);

thread_local! {
    /// Alle `TUNNEL_GRID_STRIDE` Bloecke. 4 Perlin-Rohproben pro Gitterpunkt statt pro Voxel - ein
    /// voller Chunk (32768 Voxel) braucht bei einem direkten Pro-Voxel-Ansatz bis zu 131072
    /// Rauschproben (>10ms), mit dem Raster nur 729*4=2916 plus billige trilineare Interpolation.
    /// `generate_chunk` (Bulk-Fuellung, riesige Wiederverwendung ueber 32768 Voxel) UND `is_solid`
    /// (Einzel-Fallback-Abfrage) teilen sich DENSELBEN Cache und rufen exakt dieselbe
    /// Interpolationsfunktion auf - beide Pfade liefern also IMMER identische Ergebnisse (anders als
    /// beim fruehen Hoehen-/Hoehlendichte-Bug, wo Bulk- und Fallback-Pfad zwei VERSCHIEDENE Formeln
    /// nutzten). Reine Performance-Optimierung, keine Genauigkeits-Abweichung zwischen den Pfaden.
    static TUNNEL_GRID_CACHE: RefCell<[TunnelGridCacheSlot; TUNNEL_GRID_CACHE_SLOTS]> =
        const { RefCell::new([(i32::MIN, i32::MIN, i32::MIN, [0.0; 4]); TUNNEL_GRID_CACHE_SLOTS]) };
}

#[inline(always)]
fn lerp_channels(a: TunnelGridChannels, b: TunnelGridChannels, t: f32) -> TunnelGridChannels {
    [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t), lerp(a[3], b[3], t)]
}

/// Multi-Stage-Terraingenerator: 2D-Rauschen fuer Hoehe/Klippen/Straende, 3D-Rauschen nur fuer
/// Hoehlen (und dort nur unterhalb der bereits bekannten Oberflaeche) - siehe Kommentare an den
/// einzelnen Feldern fuer die Rolle jeder Rauschschicht.
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
    /// 3D-Perlin fuer "Cheese Caves" (Cutoff-Hoehlen).
    cave: Noise<common_noise::Perlin>,
    cave_frequency: f32,
    cave_threshold: f32,
    /// Grobes 2D-Gate (1 Rauschprobe): nur "Hoehlen-aktive" Regionen zahlen ueberhaupt fuer das
    /// teure Tunnelsystem - s. Kommentar an `is_tunnel_region`.
    cave_region: Noise<common_noise::Perlin>,
    cave_region_frequency: f32,
    cave_region_threshold: f32,
    /// Haupt-Tunnelsystem: Ridged-Schnitt zweier unabhaengiger 3D-Perlin-Karten
    /// (`|a| < t && |b| < t`) - s. Kommentar an `is_tunnel`.
    tunnel_a: Noise<common_noise::Perlin>,
    tunnel_b: Noise<common_noise::Perlin>,
    tunnel_frequency: f32,
    tunnel_threshold: f32,
    /// Wie viele Bloecke unterhalb `SEA_LEVEL` die Tunnel-Verbreiterung ihr Maximum erreicht.
    tunnel_widen_depth_range: f32,
    /// Faktor, um den `tunnel_threshold` in maximaler Tiefe multipliziert wird (breitere Hoehlen je
    /// tiefer).
    tunnel_widen_max_multiplier: f32,
    /// Feineres Verbindungs-Tunnelsystem (hoehere Frequenz als das Hauptsystem) - schafft kleine
    /// Querverbindungen zwischen Haupttunneln und nebenbei mehr Oberflaecheneingaenge.
    connector_a: Noise<common_noise::Perlin>,
    connector_b: Noise<common_noise::Perlin>,
    connector_frequency: f32,
    connector_threshold: f32,
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

        let mut cave = Noise::<common_noise::Perlin>::default();
        cave.set_seed(config.terrain_seed.wrapping_add(0x27D4_EB2F));

        let mut cave_region = Noise::<common_noise::Perlin>::default();
        cave_region.set_seed(config.terrain_seed.wrapping_add(0x1656_67B1));

        let mut tunnel_a = Noise::<common_noise::Perlin>::default();
        tunnel_a.set_seed(config.terrain_seed.wrapping_add(0x9E3B_2265));
        let mut tunnel_b = Noise::<common_noise::Perlin>::default();
        tunnel_b.set_seed(config.terrain_seed.wrapping_add(0xD35A_2D97));

        let mut connector_a = Noise::<common_noise::Perlin>::default();
        connector_a.set_seed(config.terrain_seed.wrapping_add(0x4F51_1D37));
        let mut connector_b = Noise::<common_noise::Perlin>::default();
        connector_b.set_seed(config.terrain_seed.wrapping_add(0x9749_2F09));

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
            cave,
            cave_frequency: config.terrain_cave_frequency,
            cave_threshold: config.terrain_cave_threshold,
            cave_region,
            cave_region_frequency: config.terrain_cave_region_frequency,
            cave_region_threshold: config.terrain_cave_region_threshold,
            tunnel_a,
            tunnel_b,
            tunnel_frequency: config.terrain_tunnel_frequency,
            tunnel_threshold: config.terrain_tunnel_threshold,
            tunnel_widen_depth_range: config.terrain_tunnel_widen_depth_range.max(1.0),
            tunnel_widen_max_multiplier: config.terrain_tunnel_widen_max_multiplier,
            connector_a,
            connector_b,
            connector_frequency: config.terrain_connector_frequency,
            connector_threshold: config.terrain_connector_threshold,
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

    fn cave_density(&self, world_x: i32, world_y: i32, world_z: i32) -> f32 {
        self.sample3d(&self.cave, self.cave_frequency, world_x, world_y, world_z)
    }

    fn is_cheese_cave(&self, world_x: i32, world_y: i32, world_z: i32) -> bool {
        self.cave_density(world_x, world_y, world_z) > self.cave_threshold
    }

    /// Grobes, sehr guenstiges 2D-Gate (1 Rauschprobe): nur in "Hoehlen-aktiven" Regionen wird das
    /// Tunnelsystem (bis zu 4 Worley-Proben/Voxel) ueberhaupt ausgewertet. Reale Hoehlensysteme sind
    /// regional konzentriert (Karstgebiete) statt gleichmaessig verteilt - dieses Gate bildet das
    /// nach UND spart in den meisten Bereichen des Untergrunds die komplette Tunnel-Berechnung, statt
    /// jeden Voxel im Spiel dafuer bezahlen zu lassen.
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

    /// Rohwerte EINES Tunnel-Gitterpunkts (die 4 Ridged-Betraege, s. `TunnelGridChannels`),
    /// `TUNNEL_GRID_CACHE`-gecached - s. Kommentar dort. `gx`/`gy`/`gz` sind bereits durch
    /// `TUNNEL_GRID_STRIDE` geteilte Gitterkoordinaten, keine Weltkoordinaten.
    fn tunnel_grid_corner(&self, gx: i32, gy: i32, gz: i32) -> TunnelGridChannels {
        let slot = tunnel_grid_slot(gx, gy, gz);
        TUNNEL_GRID_CACHE.with_borrow_mut(|cache| {
            let (cached_x, cached_y, cached_z, channels) = cache[slot];
            if cached_x == gx && cached_y == gy && cached_z == gz {
                return channels;
            }

            let world_x = gx * TUNNEL_GRID_STRIDE;
            let world_y = gy * TUNNEL_GRID_STRIDE;
            let world_z = gz * TUNNEL_GRID_STRIDE;
            let channels = [
                self.sample3d(&self.tunnel_a, self.tunnel_frequency, world_x, world_y, world_z).abs(),
                self.sample3d(&self.tunnel_b, self.tunnel_frequency, world_x, world_y, world_z).abs(),
                self.sample3d(&self.connector_a, self.connector_frequency, world_x, world_y, world_z).abs(),
                self.sample3d(&self.connector_b, self.connector_frequency, world_x, world_y, world_z).abs(),
            ];
            cache[slot] = (gx, gy, gz, channels);
            channels
        })
    }

    /// XZ-bilineare Interpolation der Tunnel-Traegerwerte auf EINER Gitter-Y-Ebene (`gy`, bereits
    /// durch `TUNNEL_GRID_STRIDE` geteilt) - Baustein von `tunnel_fields_at` UND
    /// `tunnel_column_layers`, damit beide Pfade garantiert dieselbe Formel nutzen.
    fn tunnel_xz_layer(&self, gx0: i32, gz0: i32, tx: f32, tz: f32, gy: i32) -> TunnelGridChannels {
        let c00 = self.tunnel_grid_corner(gx0, gy, gz0);
        let c10 = self.tunnel_grid_corner(gx0 + 1, gy, gz0);
        let c01 = self.tunnel_grid_corner(gx0, gy, gz0 + 1);
        let c11 = self.tunnel_grid_corner(gx0 + 1, gy, gz0 + 1);
        lerp_channels(lerp_channels(c00, c10, tx), lerp_channels(c01, c11, tx), tz)
    }

    /// Trilineare Interpolation der Tunnel-Traegerwerte an einer beliebigen Weltposition - fuer
    /// Einzelabfragen (`is_solid`-Fallback). Fuer die Bulk-Fuellung in `generate_chunk` s. stattdessen
    /// `tunnel_column_layers`/`tunnel_from_layers`, die dieselbe XZ-Ebene ueber 32 Y-Voxel
    /// wiederverwenden statt sie pro Voxel neu zu berechnen.
    fn tunnel_fields_at(&self, world_x: i32, world_y: i32, world_z: i32) -> TunnelGridChannels {
        let gx0 = world_x.div_euclid(TUNNEL_GRID_STRIDE);
        let gz0 = world_z.div_euclid(TUNNEL_GRID_STRIDE);
        let gy0 = world_y.div_euclid(TUNNEL_GRID_STRIDE);
        let tx = world_x.rem_euclid(TUNNEL_GRID_STRIDE) as f32 / TUNNEL_GRID_STRIDE as f32;
        let ty = world_y.rem_euclid(TUNNEL_GRID_STRIDE) as f32 / TUNNEL_GRID_STRIDE as f32;
        let tz = world_z.rem_euclid(TUNNEL_GRID_STRIDE) as f32 / TUNNEL_GRID_STRIDE as f32;

        let bottom = self.tunnel_xz_layer(gx0, gz0, tx, tz, gy0);
        let top = self.tunnel_xz_layer(gx0, gz0, tx, tz, gy0 + 1);
        lerp_channels(bottom, top, ty)
    }

    /// Ob die 4 interpolierten Ridged-Betraege bei `world_y` einen Tunnel ergeben - einzige Stelle,
    /// die die Schwellwert-/Tiefenverbreiterungs-Formel kennt, genutzt von `is_tunnel` UND
    /// `tunnel_from_layers` (garantiert identisches Ergebnis fuer beide Performance-Pfade).
    /// Ridged-Schnitt: `|a| < t` allein ergibt gekruemmte 2D-FLAECHEN (die Null-Isoflaeche eines
    /// 3D-Perlin); der Schnitt ZWEIER unabhaengiger solcher Flaechen (`max(|a|,|b|) < t`) ergibt
    /// deren 1D-Schnittkurven - wurmartige, zusammenhaengende Roehren ("Spaghetti Caves").
    fn tunnel_threshold_check(&self, world_y: i32, channels: TunnelGridChannels) -> bool {
        let depth_factor = ((SEA_LEVEL - world_y) as f32 / self.tunnel_widen_depth_range).clamp(0.0, 1.0);
        let widened_threshold = self.tunnel_threshold * (1.0 + self.tunnel_widen_max_multiplier * depth_factor);
        channels[0].max(channels[1]) < widened_threshold
            || channels[2].max(channels[3]) < self.connector_threshold
    }

    /// Haupt- + Verbindungs-Spaghetti-Tunnel via Ridged-Schnitt (s. `tunnel_threshold_check`).
    /// Haupttunnel werden mit der Tiefe breiter; die feineren, konstant duennen Verbindungstunnel
    /// schaffen Querverbindungen und Oberflaecheneingaenge. Nur aufgerufen, wenn `is_tunnel_region`
    /// bereits zugestimmt hat. Einzelabfrage-Pfad fuer `is_solid` - s. Kommentar an
    /// `tunnel_fields_at`.
    fn is_tunnel(&self, world_x: i32, world_y: i32, world_z: i32) -> bool {
        let channels = self.tunnel_fields_at(world_x, world_y, world_z);
        self.tunnel_threshold_check(world_y, channels)
    }

    /// Anzahl Tunnel-Gitter-Y-Ebenen, die eine volle Chunk-Spalte (`CHUNK_SIZE` Voxel) im
    /// schlechtesten Fall (ungluecklichste Ausrichtung zum Gitter) beruehren kann:
    /// `CHUNK_SIZE/TUNNEL_GRID_STRIDE + 2` aufgerundet.
    const TUNNEL_COLUMN_MAX_LAYERS: usize = (CHUNK_SIZE / TUNNEL_GRID_STRIDE) as usize + 2;

    /// Praeberechnet fuer eine FESTE Spalte (world_x, world_z) alle beruehrten Tunnel-Gitter-
    /// Y-Ebenen XZ-bilinear vorinterpoliert. `generate_chunk` durchlaeuft pro Spalte `CHUNK_SIZE`
    /// (32) Y-Voxel; ohne dieses Praecaching wuerde jeder einzelne Voxel `tunnel_fields_at` mit 8
    /// frischen Gitter-Ecken-Lookups (Hash+RefCell+Array-Zugriff) aufrufen, obwohl `gx0`/`gz0` fuer
    /// die GESAMTE Spalte konstant sind und sich `gy0` nur alle `TUNNEL_GRID_STRIDE` Y-Schritte
    /// aendert - das allein kostete trotz `TUNNEL_GRID_CACHE` noch mehrere ms/Chunk (256 statt
    /// hoechstens 40 Lookup-Operationen pro Spalte). Mathematisch identisch zu `tunnel_fields_at` je
    /// Voxel (s. `tunnel_from_layers`), nur mit wiederverwendeten Zwischenergebnissen - keine
    /// Genauigkeitsabweichung.
    fn tunnel_column_layers(
        &self,
        world_x: i32,
        world_z: i32,
        y_min: i32,
        y_max: i32,
    ) -> (i32, [TunnelGridChannels; Self::TUNNEL_COLUMN_MAX_LAYERS]) {
        let gx0 = world_x.div_euclid(TUNNEL_GRID_STRIDE);
        let gz0 = world_z.div_euclid(TUNNEL_GRID_STRIDE);
        let tx = world_x.rem_euclid(TUNNEL_GRID_STRIDE) as f32 / TUNNEL_GRID_STRIDE as f32;
        let tz = world_z.rem_euclid(TUNNEL_GRID_STRIDE) as f32 / TUNNEL_GRID_STRIDE as f32;

        let gy_min = y_min.div_euclid(TUNNEL_GRID_STRIDE);
        let gy_max = y_max.div_euclid(TUNNEL_GRID_STRIDE) + 1;

        let mut layers = [[0.0f32; 4]; Self::TUNNEL_COLUMN_MAX_LAYERS];
        for (i, layer) in layers.iter_mut().enumerate() {
            let gy = gy_min + i as i32;
            if gy > gy_max {
                break;
            }
            *layer = self.tunnel_xz_layer(gx0, gz0, tx, tz, gy);
        }
        (gy_min, layers)
    }

    /// Interpoliert `world_y` aus den per `tunnel_column_layers` vorberechneten Y-Ebenen und prueft
    /// die Schwelle - Bulk-Pfad-Gegenstueck zu `is_tunnel`, s. dortigen Kommentar zur garantierten
    /// Ergebnisgleichheit.
    fn tunnel_from_layers(
        &self,
        world_y: i32,
        gy_min: i32,
        layers: &[TunnelGridChannels; Self::TUNNEL_COLUMN_MAX_LAYERS],
    ) -> bool {
        let gy0 = world_y.div_euclid(TUNNEL_GRID_STRIDE);
        let ty = world_y.rem_euclid(TUNNEL_GRID_STRIDE) as f32 / TUNNEL_GRID_STRIDE as f32;
        let bottom = layers[(gy0 - gy_min) as usize];
        let top = layers[(gy0 - gy_min) as usize + 1];
        self.tunnel_threshold_check(world_y, lerp_channels(bottom, top, ty))
    }

    /// Vereinigung aller Hoehlensysteme (Cheese Caves + regional gegatetes Tunnelnetz) UNTER dem
    /// `MIN_CAVE_DEPTH`-Mindestabstand zur Oberflaeche - einzige Stelle, die diese Regel kennt.
    /// `is_solid` UND `generate_chunk` rufen ausschliesslich diese Funktion fuer die Untergrund-
    /// Aushoehlung auf, damit beide GARANTIERT uebereinstimmen (s. Kommentar an `generate_chunk`) -
    /// eine fruehere, nur in `generate_chunk` inline geprueft Depth-Gate liess `is_solid` am
    /// obersten Block einer Saeule gelegentlich abweichen (der Test unten deckt das jetzt ab).
    fn is_carved(&self, world_x: i32, world_y: i32, world_z: i32, depth_from_surface: i32) -> bool {
        depth_from_surface >= MIN_CAVE_DEPTH
            && (self.is_cheese_cave(world_x, world_y, world_z)
                || (self.is_tunnel_region(world_x, world_z) && self.is_tunnel(world_x, world_y, world_z)))
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

    /// Hoehenkarte UND Cheese-Cave-Dichte werden IMMER exakt (nicht interpoliert) berechnet - beide
    /// sind mit 1 bzw. 4 Rauschproben guenstig genug, dass Interpolation keinen messbaren Vorteil
    /// braechte. Das Tunnelsystem dagegen (`is_tunnel`) IST interpoliert (sparse Worley-Gitter, s.
    /// `TUNNEL_GRID_CACHE`) - 4 Worley-Proben/Voxel waeren bei 32768 Voxeln allein >30ms. Das ist hier
    /// SICHER, weil `generate_chunk` und `is_solid` (ueber `is_carved`) exakt dieselbe interpolierte
    /// Funktion `tunnel_fields_at` aufrufen statt zwei verschiedener Formeln - anders als beim
    /// fruehen Hoehen-/Cheese-Cave-Bug (interpolierte Bulk-Fuellung vs. exakter Fallback, siehe
    /// Kommentar an `TUNNEL_GRID_CACHE`), der dauerhafte Chunk-Naht-Luecken und "im Boden stecken
    /// bleiben" verursachte, weil beide Pfade fuer denselben Voxel verschiedene Antworten gaben.
    /// `is_tunnel_region` bleibt das zusaetzliche, sehr guenstige 2D-Gate davor.
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

        for local_z in 0..CHUNK_SIZE {
            for local_x in 0..CHUNK_SIZE {
                let height = local_height[(local_z * CHUNK_SIZE + local_x) as usize];
                let column_has_terrain = chunk_origin_y <= height;
                let column_has_water = height < WATER_LEVEL && chunk_origin_y <= WATER_LEVEL;
                if !column_has_terrain && !column_has_water {
                    continue;
                }

                let world_x = chunk_origin_x + local_x;
                let world_z = chunk_origin_z + local_z;

                // Wasserfuellung zuerst - billig (kein Rauschen, s. `is_water_position`) und
                // unabhaengig von Hangneigung/Biom/Tunneln.
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

                // Einmal pro Spalte statt pro Voxel vorberechnet (s. Kommentar an
                // `tunnel_column_layers`) - nur wenn `is_tunnel_region` ueberhaupt zustimmt. Deckt
                // exakt den tatsaechlich durchlaufenen Y-Bereich dieser Saeule ab (durch `height`
                // UND das Chunk-Ende gedeckelt, wie die innere Schleife unten).
                let column_y_max = height.min(chunk_origin_y + CHUNK_SIZE - 1);
                let tunnel_layers = self
                    .is_tunnel_region(world_x, world_z)
                    .then(|| self.tunnel_column_layers(world_x, world_z, chunk_origin_y, column_y_max));

                for local_y in 0..CHUNK_SIZE {
                    let world_y = chunk_origin_y + local_y;
                    if world_y > height {
                        continue;
                    }

                    let depth_from_surface = height - world_y;
                    if depth_from_surface >= MIN_CAVE_DEPTH {
                        let carved = self.is_cheese_cave(world_x, world_y, world_z)
                            || tunnel_layers
                                .is_some_and(|(gy_min, layers)| self.tunnel_from_layers(world_y, gy_min, &layers));
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

    /// Oberflaechen-Kontext einer Spalte: Hoehenband-Strand, Unterwasser-Boden und das strikte
    /// 2D-Biom-Mapping (Wueste NUR bei Temperatur > min UND Feuchtigkeit < max). Temperatur/
    /// Feuchtigkeit sind sehr niedrigfrequent + hart geschwellt - grosse zusammenhaengende Biome,
    /// kein Einzelspalten-Bleeding. Bewusst NICHT Teil von `is_solid`: Biome aendern nur die
    /// Block-ID (Sand vs. Gras), nie die Festigkeit - keine Konsistenzanforderung an den Fallback,
    /// keine zusaetzlichen Rauschproben im Mesher-/Physik-Hotpath.
    fn column_surface(&self, world_x: i32, world_z: i32, height: i32) -> ColumnSurface {
        let beach_half_range = (self.sea_compression_range * 0.25).max(1.0) as i32;
        let temperature = self.sample2d(&self.temperature, self.temperature_frequency, world_x, world_z);
        let humidity = self.sample2d(&self.humidity, self.humidity_frequency, world_x, world_z);
        ColumnSurface {
            is_beach: (height - SEA_LEVEL).abs() <= beach_half_range,
            is_underwater: height < WATER_LEVEL,
            is_desert: temperature > self.desert_temperature_min && humidity < self.desert_humidity_max,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::config::EngineConfig;

    /// `TerrainGenerator::is_solid` ist die Vorhersage, auf die sowohl der Mesher (Nachbar-Check an
    /// noch nicht geladenen Chunks) als auch `ChunkManager::is_solid_at` (Physik/Raycast waehrend
    /// ein Chunk noch generiert) zurueckfallen. Hoehe UND jedes Hoehlensystem werden in
    /// `generate_chunk` bewusst nirgends interpoliert, genau damit diese Vorhersage IMMER exakt mit
    /// dem uebereinstimmt, was tatsaechlich generiert wird - sonst entstehen dauerhafte Luecken an
    /// Chunk-Naehten (niemand re-mesht spaeter) bzw. bewegt sich ein Spieler waehrend des Ladens
    /// durch eine Stelle, an der sich die Vorhersage nachtraeglich als falsch herausstellt, bleibt er
    /// im nachtraeglich materialisierten Fels stecken. Prueft deshalb ALLE 32768 Voxel (nicht nur den
    /// Rand) ueber viele diverse Chunk-Koordinaten inkl. vertikaler Stapelung - eine fruehere Version
    /// mit nur 4 Koordinaten hat den `MIN_CAVE_DEPTH`-Depth-Gate-Bug (s. `is_carved`-Kommentar) nicht
    /// gefangen, weil er nur selten am obersten Block einer Saeule ausgeloest wird.
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

    /// Diagnose-Tool, KEIN Korrektheitstest - misst die tatsaechliche Verteilung der Ridged-
    /// Tunnel-Traegerwerte (`max(|a|,|b|)` beider Perlin-Karten je System) an vielen zufaelligen
    /// Punkten, um `terrain_tunnel_threshold`/`terrain_connector_threshold`/
    /// `terrain_cave_region_threshold` empirisch zu kalibrieren, statt sie zu erraten. Manuell
    /// ausfuehren mit:
    /// `cargo test --release --lib -- --ignored --nocapture calibrate_tunnel_thresholds`
    #[test]
    #[ignore = "Diagnose-Tool, kein automatisierter Test - siehe Doc-Kommentar"]
    fn calibrate_tunnel_thresholds() {
        let generator = TerrainGenerator::new(&EngineConfig::default());
        const N: usize = 200_000;

        let percentile = |values: &mut [f32], p: f64| -> f32 {
            values.sort_unstable_by(|a, b| a.total_cmp(b));
            values[((values.len() as f64 - 1.0) * p) as usize]
        };

        let mut ridged_percentiles = |label: &str,
                                      map_a: &Noise<common_noise::Perlin>,
                                      map_b: &Noise<common_noise::Perlin>,
                                      frequency: f32| {
            let mut values: Vec<f32> = (0..N)
                .map(|i| {
                    let (x, y, z) = (i as i32 * 37 - 700_000, i as i32 * 11 - 50_000, i as i32 * 53 - 900_000);
                    let a = generator.sample3d(map_a, frequency, x, y, z).abs();
                    let b = generator.sample3d(map_b, frequency, x, y, z).abs();
                    a.max(b)
                })
                .collect();
            println!(
                "{label} max(|a|,|b|): p1={:.4} p2={:.4} p5={:.4} p10={:.4} p50={:.4}",
                percentile(&mut values, 0.01),
                percentile(&mut values, 0.02),
                percentile(&mut values, 0.05),
                percentile(&mut values, 0.10),
                percentile(&mut values, 0.50),
            );
        };

        ridged_percentiles("tunnel", &generator.tunnel_a, &generator.tunnel_b, generator.tunnel_frequency);
        ridged_percentiles(
            "connector",
            &generator.connector_a,
            &generator.connector_b,
            generator.connector_frequency,
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
