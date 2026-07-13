use std::cell::RefCell;

use bevy_math::{Vec2, Vec3};
use noiz::cell_noise::WorleyDifference;
use noiz::prelude::*;
use noiz::{Noise, NoiseFunction};

use crate::engine::config::EngineConfig;

use super::blocks::{self, ColumnSurface};
use super::chunk::{CHUNK_SIZE, Chunk};

mod flora;
pub(crate) mod pyramid;
use flora::{MAX_NEARBY_TREES, TREE_HEIGHT_SAFETY_MARGIN, TreeSpawn};
use pyramid::TerrainPyramid;

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
    let hash =
        (world_x as u32).wrapping_mul(0x9E37_79B1) ^ (world_z as u32).wrapping_mul(0x85EB_CA6B);
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
    /// Wie `TUNNEL_GRID_CACHE`, fuer das zweite (Connector-)Worley-Tunnelsystem - eigener Cache,
    /// weil dieselben (gx,gy,gz)-Koordinaten bei unterschiedlicher Frequenz einen ANDEREN Rohwert
    /// ergeben (unterschiedliche Weltposition pro Gitterpunkt) - ein gemeinsamer Cache wuerde die
    /// beiden Systeme gegenseitig eviktieren.
    static CONNECTOR_GRID_CACHE: RefCell<[CaveGridCacheSlot; CAVE_GRID_CACHE_SLOTS]> =
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

/// Kantenlaenge einer Rand-Ebene - identisch `CHUNK_SIZE`, eigener `usize`-Name fuer die
/// Ebenen-Arrays der `solid_plane_*`-Funktionen.
const PLANE_SIZE: usize = CHUNK_SIZE as usize;
/// `is_solid`-Ergebnis einer GANZEN 32x32-Rand-Ebene einer Chunk-Seite. Indizierung passend zu den
/// jeweiligen `solid_*`-Achsentabellen im Mesher: X-Ebenen `[y][z]`, Y-Ebenen `[x][z]`, Z-Ebenen
/// `[x][y]` (lokale Indizes relativ zum jeweiligen Chunk-Origin).
pub type BoundaryPlane = [[bool; PLANE_SIZE]; PLANE_SIZE];
/// Roh-Eckwert-Ausschnitt EINES Hoehlenrasters ueber einen 2D-Gitterbereich - Baustein der
/// `solid_plane_*`-Funktionen, s. `TerrainGenerator::grid_slice`.
type GridSlice = [[f32; CAVE_COLUMN_MAX_LAYERS]; CAVE_COLUMN_MAX_LAYERS];

/// Alle 6 Rand-Ebenen EINES Chunks, s. `TerrainGenerator::boundary_planes`.
pub struct BoundaryPlanes {
    pub neg_x: BoundaryPlane,
    pub pos_x: BoundaryPlane,
    pub neg_y: BoundaryPlane,
    pub pos_y: BoundaryPlane,
    pub neg_z: BoundaryPlane,
    pub pos_z: BoundaryPlane,
}

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

/// Multi-Stage-Terraingenerator: Hoehe/Klima aus der hierarchischen Fenster-Pyramide
/// (`generator/pyramid.rs`, InfiniteDiffusion-Schema), 3D-Rauschen nur unterhalb der bereits
/// bekannten Oberflaeche fuer die zwei Hoehlensysteme.
pub struct TerrainGenerator {
    /// Hierarchische Hoehen-/Klima-Synthese - einzige Quelle fuer Oberflaechenhoehe, Temperatur
    /// und Feuchtigkeit (seed-konsistent, O(1)-Random-Access, s. Modul-Kommentar).
    pyramid: TerrainPyramid,
    /// Strikte 2D-Biom-Achsen: Wueste nur bei hoher Temperatur UND geringer Feuchtigkeit.
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
    /// Zweites, UNABHAENGIGES Worley-Tunnelsystem (andere Frequenz UND anderer Seed als `tunnel`) -
    /// verbindet isolierte Sackgassen des primaeren Systems: zwei unabhaengige Voronoi-Zellgrenz-
    /// Netze ueberschneiden sich an voellig anderen Stellen als ein einzelnes Netz mit sich selbst,
    /// dadurch findet praktisch jede primaere Sackgasse frueher oder spaeter eine Querverbindung.
    /// Nutzt DIESELBE `is_tunnel_region`-Gate wie `tunnel` (geografisch an dieselben "Hoehlen-
    /// aktiven" Regionen gebunden, keine zusaetzliche 2D-Rauschprobe noetig).
    connector: Noise<WorleyTunnel>,
    connector_frequency: f32,
    connector_threshold: f32,
    /// Gemeinsamer Tiefenfaktor: sowohl Cheese Caves als auch Tunnel werden graduell groesser, je
    /// weiter man unter `SEA_LEVEL` kommt - erreicht sein Maximum nach dieser Blocktiefe.
    cave_widen_depth_range: f32,
    /// Um wie viel `cheese_threshold` in maximaler Tiefe SINKT (leichter zu ueberschreiten -> mehr
    /// Volumen ausgehoehlt).
    cheese_widen_amount: f32,
    /// Um welchen Faktor `tunnel_threshold` in maximaler Tiefe MULTIPLIZIERT wird (groesser ->
    /// breitere Roehren, da der Cutoff auf der F2-F1-Distanz dann grosszuegiger ist).
    tunnel_widen_multiplier: f32,
    /// Wie `tunnel_widen_multiplier`, fuer `connector_threshold`.
    connector_widen_multiplier: f32,
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
    /// Salt fuer den Baum-Spawn-Hash - kein `Noise<>`-Feld, Baeume brauchen keine Gradienten-
    /// Kontinuitaet (White Noise reicht fuer diskrete Spawn-Entscheidungen), nur Determinismus.
    tree_seed: u32,
    tree_grid_size: i32,
    tree_spawn_chance: f32,
    tree_trunk_height_min: i32,
    tree_trunk_height_max: i32,
    tree_crown_radius_min: i32,
    tree_crown_radius_max: i32,
}

impl TerrainGenerator {
    pub fn new(config: &EngineConfig) -> Self {
        let mut cheese = Noise::<common_noise::Perlin>::default();
        cheese.set_seed(config.dev.terrain_seed.wrapping_add(0x27D4_EB2F));

        let mut tunnel = Noise::<WorleyTunnel>::default();
        tunnel.set_seed(config.dev.terrain_seed.wrapping_add(0x9E3B_2265));

        let mut connector = Noise::<WorleyTunnel>::default();
        connector.set_seed(config.dev.terrain_seed.wrapping_add(0x5BD1_E995));

        let mut cave_region = Noise::<common_noise::Perlin>::default();
        cave_region.set_seed(config.dev.terrain_seed.wrapping_add(0x1656_67B1));

        Self {
            pyramid: TerrainPyramid::new(config),
            desert_temperature_min: config.dev.terrain_desert_temperature_min,
            desert_humidity_max: config.dev.terrain_desert_humidity_max,
            cheese,
            cheese_frequency: config.dev.terrain_cheese_frequency,
            cheese_threshold: config.dev.terrain_cheese_threshold,
            tunnel,
            tunnel_frequency: config.dev.terrain_tunnel_frequency,
            tunnel_threshold: config.dev.terrain_tunnel_threshold,
            connector,
            connector_frequency: config.dev.terrain_connector_frequency,
            connector_threshold: config.dev.terrain_connector_threshold,
            cave_widen_depth_range: config.dev.terrain_cave_widen_depth_range.max(1.0),
            cheese_widen_amount: config.dev.terrain_cheese_widen_amount,
            tunnel_widen_multiplier: config.dev.terrain_tunnel_widen_multiplier,
            connector_widen_multiplier: config.dev.terrain_connector_widen_multiplier,
            cave_region,
            cave_region_frequency: config.dev.terrain_cave_region_frequency,
            cave_region_threshold: config.dev.terrain_cave_region_threshold,
            dirt_layer_depth: config.dev.terrain_dirt_layer_depth,
            noise_origin_offset: config.dev.terrain_noise_origin_offset,
            tree_seed: config.dev.terrain_seed.wrapping_add(0x4B72_E68F),
            tree_grid_size: config.dev.terrain_tree_grid_size.max(1),
            tree_spawn_chance: config.dev.terrain_tree_spawn_chance.clamp(0.0, 1.0),
            tree_trunk_height_min: config.dev.terrain_tree_trunk_height_min.max(1),
            tree_trunk_height_max: config
                .dev
                .terrain_tree_trunk_height_max
                .max(config.dev.terrain_tree_trunk_height_min.max(1)),
            tree_crown_radius_min: config.dev.terrain_tree_crown_radius_min.max(0),
            tree_crown_radius_max: config
                .dev
                .terrain_tree_crown_radius_max
                .max(config.dev.terrain_tree_crown_radius_min.max(0)),
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
            let height = self.pyramid.sample(world_x, world_z).height.round() as i32;
            cache[slot] = (world_x, world_z, height);
            height
        })
    }

    /// Wasserfuellung: NUR ueber der Terrainoberflaeche bis zum Wasserspiegel (Ozeane/Seen in
    /// Senken) - Positionen unter der Oberflaeche (auch ausgehoehlte) bleiben unberuehrt, Hoehlen
    /// fluten also nie. Reines O(1)-Praedikat aus der Spaltenhoehe, keine zusaetzliche Rauschprobe -
    /// exakt dieselbe Formel fuer `generate_chunk` (Befuellung) und `is_solid` (Fallback).
    #[inline(always)]
    fn is_water_position(height: i32, world_y: i32) -> bool {
        world_y > height && world_y <= WATER_LEVEL
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

    /// Wie `tunnel_threshold_at`, fuer das zweite (Connector-)Worley-Tunnelsystem.
    #[inline(always)]
    fn connector_threshold_at(&self, world_y: i32) -> f32 {
        self.connector_threshold
            * (1.0 + self.connector_widen_multiplier * self.depth_factor(world_y))
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
            let (wx, wy, wz) = (
                gx * CAVE_GRID_STRIDE,
                gy * CAVE_GRID_STRIDE,
                gz * CAVE_GRID_STRIDE,
            );
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
            let (wx, wy, wz) = (
                gx * CAVE_GRID_STRIDE,
                gy * CAVE_GRID_STRIDE,
                gz * CAVE_GRID_STRIDE,
            );
            let value = self.sample3d(&self.tunnel, self.tunnel_frequency, wx, wy, wz);
            cache[slot] = (gx, gy, gz, value);
            value
        })
    }

    /// Wie `tunnel_grid_corner`, fuer das zweite (Connector-)Worley-Tunnelsystem - andere Frequenz
    /// UND anderer Noise-Seed, deshalb eigener Cache (`CONNECTOR_GRID_CACHE`). Nur aufgerufen, wenn
    /// `is_tunnel_region` bereits zugestimmt hat (dieselbe Gate wie `tunnel_grid_corner`).
    fn connector_grid_corner(&self, gx: i32, gy: i32, gz: i32) -> f32 {
        let slot = cave_grid_slot(gx, gy, gz);
        CONNECTOR_GRID_CACHE.with_borrow_mut(|cache| {
            let (cached_x, cached_y, cached_z, value) = cache[slot];
            if cached_x == gx && cached_y == gy && cached_z == gz {
                return value;
            }
            let (wx, wy, wz) = (
                gx * CAVE_GRID_STRIDE,
                gy * CAVE_GRID_STRIDE,
                gz * CAVE_GRID_STRIDE,
            );
            let value = self.sample3d(&self.connector, self.connector_frequency, wx, wy, wz);
            cache[slot] = (gx, gy, gz, value);
            value
        })
    }

    /// Trilineare Interpolation EINES Hoehlenrasters an einer beliebigen Weltposition, ueber die 8
    /// umschliessenden Gitterpunkte - fuer Einzelabfragen (`is_solid`-Fallback via `is_carved`).
    /// Nutzt DIESELBEN Gitter-Ecken (`corner`) wie die Zellen-Bulk-Fuellung `cave_grid_stack` in
    /// `generate_chunk` - beide Pfade duerfen fuer denselben Voxel NIE unterschiedliche Werte
    /// liefern, s. Kommentar an `is_carved`.
    fn cave_fields_at(
        &self,
        corner: impl Fn(&Self, i32, i32, i32) -> f32,
        world_x: i32,
        world_y: i32,
        world_z: i32,
    ) -> f32 {
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
            let region = self.sample2d(
                &self.cave_region,
                self.cave_region_frequency,
                world_x,
                world_z,
            ) > self.cave_region_threshold;
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
        let cheese_density =
            self.cave_fields_at(Self::cheese_grid_corner, world_x, world_y, world_z);
        if cheese_density > self.cheese_threshold_at(world_y) {
            return true;
        }
        if !self.is_tunnel_region(world_x, world_z) {
            return false;
        }
        let tunnel_diff = self.cave_fields_at(Self::tunnel_grid_corner, world_x, world_y, world_z);
        if tunnel_diff < self.tunnel_threshold_at(world_y) {
            return true;
        }
        let connector_diff =
            self.cave_fields_at(Self::connector_grid_corner, world_x, world_y, world_z);
        connector_diff < self.connector_threshold_at(world_y)
    }

    /// Fallback-Quelle der Wahrheit fuer OKKLUSION ("belegt diese Position einen sichtbaren Block?",
    /// inklusive Wasser - opak gerendert) ausserhalb geladener Chunk-Daten - genutzt vom Mesher
    /// (Nachbar-Check ueber Chunk-Grenzen an noch nicht gemeshten Chunks). MUSS exakt `block != 0`
    /// des spaeter generierten Chunks vorhersagen, also Wasser UND Baeume einschliessen - sonst
    /// entstehen an Chunk-Naehten dieselben dauerhaften Face-Fehler wie beim fruehen Hoehen-Bug.
    pub fn is_solid(&self, world_x: i32, world_y: i32, world_z: i32) -> bool {
        let height = self.height_at(world_x, world_z);
        if Self::is_water_position(height, world_y) {
            return true;
        }
        if world_y <= height {
            return !self.is_carved(world_x, world_y, world_z, height - world_y);
        }
        self.tree_occupies(world_x, world_y, world_z)
    }

    /// Physik-Variante: Wasser ist begehbar/durchschwimmbar, also NICHT solide - nur echtes Terrain
    /// (inkl. Baeume) blockiert. Fallback fuer `ChunkManager::is_solid_at` (Kollision/Raycast),
    /// waehrend der Mesher die Okklusions-Variante `is_solid` nutzt.
    pub fn is_physically_solid(&self, world_x: i32, world_y: i32, world_z: i32) -> bool {
        let height = self.height_at(world_x, world_z);
        if world_y <= height {
            return !self.is_carved(world_x, world_y, world_z, height - world_y);
        }
        self.tree_occupies(world_x, world_y, world_z)
    }

    /// Holt `corner(to_grid(i,j))` fuer `i in i_min..=i_max`, `j in j_min..=j_max` in
    /// `out[i-i_min][j-j_min]` - gemeinsamer Baustein aller drei `solid_plane_*`-Funktionen: jede
    /// Rand-Ebene braucht genau ZWEI solcher 2D-Ausschnitte (an den beiden Gitter-Nachbarn ihrer
    /// festen Achse), aus denen anschliessend PRO PUNKT exakt dieselbe Trilinear-Formel wie
    /// `cave_fields_at` zusammengesetzt wird - bit-identisch zum `is_solid`-Einzelpunktpfad, nur
    /// ohne die pro Punkt wiederholten Gitter-Cache-Lookups (bei 1024 Punkten je Ebene sonst bis zu
    /// 8192 gehashte Eckwert-Abfragen PRO Hoehlensystem).
    #[allow(clippy::too_many_arguments)]
    fn grid_slice(
        &self,
        corner: impl Fn(&Self, i32, i32, i32) -> f32,
        to_grid: impl Fn(i32, i32) -> (i32, i32, i32),
        i_min: i32,
        i_max: i32,
        j_min: i32,
        j_max: i32,
        out: &mut GridSlice,
    ) {
        let mut i = i_min;
        while i <= i_max {
            let mut j = j_min;
            while j <= j_max {
                let (gx, gy, gz) = to_grid(i, j);
                out[(i - i_min) as usize][(j - j_min) as usize] = corner(self, gx, gy, gz);
                j += 1;
            }
            i += 1;
        }
    }

    /// Rand-Ebene senkrecht zur X-Achse bei festem `world_x` (Chunk-Rand), ueber `world_y`/`world_z`
    /// je 32 Werte ab `chunk_origin_y`/`chunk_origin_z`. Ersetzt bis zu 1024 Einzelaufrufe von
    /// `is_solid` (je bis zu 16 gecachte Gitter-Eckwert-Lookups) durch EINEN Zellen-Batch: Cheese-/
    /// Tunnel-Gitter-Ecken werden je Achse (X fest, `gx0`/`gx0+1`) EINMAL ueber die volle
    /// (Y,Z)-Flaeche geholt und pro Punkt per Lerp kombiniert - exakt dieselbe Trilinear-Formel wie
    /// `cave_fields_at` (X innen, Z mitte, Y aussen), nur mit Array- statt Cache-Zugriffen.
    pub fn solid_plane_x(
        &self,
        world_x: i32,
        chunk_origin_y: i32,
        chunk_origin_z: i32,
    ) -> BoundaryPlane {
        let mut out = [[false; PLANE_SIZE]; PLANE_SIZE];

        let mut height_by_z = [0i32; PLANE_SIZE];
        for (z, height) in height_by_z.iter_mut().enumerate() {
            *height = self.height_at(world_x, chunk_origin_z + z as i32);
        }

        // Baum-Suchradius haengt nur von (world_x, world_z) ab, nicht von Y - hier ist Y aber die
        // AEUSSERE Schleife und Z die innere: ohne dieses Vorab-Array pro Z-Wert wuerde dieselbe
        // Gitterzellen-Suche fuer denselben (world_x, world_z) bis zu 32x wiederholt (einmal pro
        // Y-Wert dieser Spalte).
        let nearby_trees_by_z: [([TreeSpawn; MAX_NEARBY_TREES], usize); PLANE_SIZE] =
            std::array::from_fn(|z| {
                self.nearby_tree_candidates(world_x, chunk_origin_z + z as i32)
            });

        let gx0 = world_x.div_euclid(CAVE_GRID_STRIDE);
        let tx = world_x.rem_euclid(CAVE_GRID_STRIDE) as f32 / CAVE_GRID_STRIDE as f32;
        let gy_min = chunk_origin_y.div_euclid(CAVE_GRID_STRIDE);
        let gy_max = (chunk_origin_y + CHUNK_SIZE - 1).div_euclid(CAVE_GRID_STRIDE) + 1;
        let gz_min = chunk_origin_z.div_euclid(CAVE_GRID_STRIDE);
        let gz_max = (chunk_origin_z + CHUNK_SIZE - 1).div_euclid(CAVE_GRID_STRIDE) + 1;

        let mut cheese_a = [[0.0f32; CAVE_COLUMN_MAX_LAYERS]; CAVE_COLUMN_MAX_LAYERS];
        let mut cheese_b = [[0.0f32; CAVE_COLUMN_MAX_LAYERS]; CAVE_COLUMN_MAX_LAYERS];
        self.grid_slice(
            Self::cheese_grid_corner,
            |gy, gz| (gx0, gy, gz),
            gy_min,
            gy_max,
            gz_min,
            gz_max,
            &mut cheese_a,
        );
        self.grid_slice(
            Self::cheese_grid_corner,
            |gy, gz| (gx0 + 1, gy, gz),
            gy_min,
            gy_max,
            gz_min,
            gz_max,
            &mut cheese_b,
        );

        let any_tunnel_region =
            (0..PLANE_SIZE).any(|z| self.is_tunnel_region(world_x, chunk_origin_z + z as i32));
        let mut tunnel_a = [[0.0f32; CAVE_COLUMN_MAX_LAYERS]; CAVE_COLUMN_MAX_LAYERS];
        let mut tunnel_b = [[0.0f32; CAVE_COLUMN_MAX_LAYERS]; CAVE_COLUMN_MAX_LAYERS];
        let mut connector_a = [[0.0f32; CAVE_COLUMN_MAX_LAYERS]; CAVE_COLUMN_MAX_LAYERS];
        let mut connector_b = [[0.0f32; CAVE_COLUMN_MAX_LAYERS]; CAVE_COLUMN_MAX_LAYERS];
        if any_tunnel_region {
            self.grid_slice(
                Self::tunnel_grid_corner,
                |gy, gz| (gx0, gy, gz),
                gy_min,
                gy_max,
                gz_min,
                gz_max,
                &mut tunnel_a,
            );
            self.grid_slice(
                Self::tunnel_grid_corner,
                |gy, gz| (gx0 + 1, gy, gz),
                gy_min,
                gy_max,
                gz_min,
                gz_max,
                &mut tunnel_b,
            );
            self.grid_slice(
                Self::connector_grid_corner,
                |gy, gz| (gx0, gy, gz),
                gy_min,
                gy_max,
                gz_min,
                gz_max,
                &mut connector_a,
            );
            self.grid_slice(
                Self::connector_grid_corner,
                |gy, gz| (gx0 + 1, gy, gz),
                gy_min,
                gy_max,
                gz_min,
                gz_max,
                &mut connector_b,
            );
        }

        for (y, row) in out.iter_mut().enumerate() {
            let world_y = chunk_origin_y + y as i32;
            let gy0 = world_y.div_euclid(CAVE_GRID_STRIDE);
            let ty = world_y.rem_euclid(CAVE_GRID_STRIDE) as f32 / CAVE_GRID_STRIDE as f32;
            let (iy0, iy1) = ((gy0 - gy_min) as usize, (gy0 - gy_min + 1) as usize);

            for (z, cell) in row.iter_mut().enumerate() {
                let world_z = chunk_origin_z + z as i32;
                let height = height_by_z[z];

                if Self::is_water_position(height, world_y) {
                    *cell = true;
                    continue;
                }
                if world_y > height {
                    let (trees, count) = &nearby_trees_by_z[z];
                    *cell = Self::tree_occupies_among(&trees[..*count], world_x, world_y, world_z);
                    continue;
                }
                if height - world_y < MIN_CAVE_DEPTH {
                    *cell = true;
                    continue;
                }

                let gz0 = world_z.div_euclid(CAVE_GRID_STRIDE);
                let tz = world_z.rem_euclid(CAVE_GRID_STRIDE) as f32 / CAVE_GRID_STRIDE as f32;
                let (jz0, jz1) = ((gz0 - gz_min) as usize, (gz0 - gz_min + 1) as usize);

                let xz_layer = |grid_a: &GridSlice, grid_b: &GridSlice, iy: usize| {
                    lerp(
                        lerp(grid_a[iy][jz0], grid_b[iy][jz0], tx),
                        lerp(grid_a[iy][jz1], grid_b[iy][jz1], tx),
                        tz,
                    )
                };
                let cheese_density = lerp(
                    xz_layer(&cheese_a, &cheese_b, iy0),
                    xz_layer(&cheese_a, &cheese_b, iy1),
                    ty,
                );
                if cheese_density > self.cheese_threshold_at(world_y) {
                    continue;
                }
                if !any_tunnel_region || !self.is_tunnel_region(world_x, world_z) {
                    *cell = true;
                    continue;
                }
                let tunnel_diff = lerp(
                    xz_layer(&tunnel_a, &tunnel_b, iy0),
                    xz_layer(&tunnel_a, &tunnel_b, iy1),
                    ty,
                );
                if tunnel_diff < self.tunnel_threshold_at(world_y) {
                    // Von Tunnel ausgehoehlt - bleibt Luft (`cell` bleibt `false`), Connector-Check
                    // ueberfluessig.
                    continue;
                }
                let connector_diff = lerp(
                    xz_layer(&connector_a, &connector_b, iy0),
                    xz_layer(&connector_a, &connector_b, iy1),
                    ty,
                );
                if connector_diff < self.connector_threshold_at(world_y) {
                    continue;
                }
                *cell = true;
            }
        }

        out
    }

    /// Rand-Ebene senkrecht zur Y-Achse bei festem `world_y` - s. `solid_plane_x` fuer das
    /// allgemeine Prinzip. Y ist in `cave_fields_at`s Formel die AEUSSERE Lerp-Achse: bei fixem Y
    /// liefert je EIN Gitter-Ausschnitt (an `gy0` bzw. `gy0+1`) direkt die komplette XZ-Ebene, ohne
    /// dass Grid A/B pro Punkt gemischt werden muessen (anders als bei X-/Z-Ebenen).
    pub fn solid_plane_y(
        &self,
        world_y: i32,
        chunk_origin_x: i32,
        chunk_origin_z: i32,
    ) -> BoundaryPlane {
        let mut out = [[false; PLANE_SIZE]; PLANE_SIZE];

        let mut height = [[0i32; PLANE_SIZE]; PLANE_SIZE];
        for (x, row) in height.iter_mut().enumerate() {
            for (z, h) in row.iter_mut().enumerate() {
                *h = self.height_at(chunk_origin_x + x as i32, chunk_origin_z + z as i32);
            }
        }

        let gy0 = world_y.div_euclid(CAVE_GRID_STRIDE);
        let ty = world_y.rem_euclid(CAVE_GRID_STRIDE) as f32 / CAVE_GRID_STRIDE as f32;
        let gx_min = chunk_origin_x.div_euclid(CAVE_GRID_STRIDE);
        let gx_max = (chunk_origin_x + CHUNK_SIZE - 1).div_euclid(CAVE_GRID_STRIDE) + 1;
        let gz_min = chunk_origin_z.div_euclid(CAVE_GRID_STRIDE);
        let gz_max = (chunk_origin_z + CHUNK_SIZE - 1).div_euclid(CAVE_GRID_STRIDE) + 1;

        let mut cheese_a = [[0.0f32; CAVE_COLUMN_MAX_LAYERS]; CAVE_COLUMN_MAX_LAYERS];
        let mut cheese_b = [[0.0f32; CAVE_COLUMN_MAX_LAYERS]; CAVE_COLUMN_MAX_LAYERS];
        self.grid_slice(
            Self::cheese_grid_corner,
            |gx, gz| (gx, gy0, gz),
            gx_min,
            gx_max,
            gz_min,
            gz_max,
            &mut cheese_a,
        );
        self.grid_slice(
            Self::cheese_grid_corner,
            |gx, gz| (gx, gy0 + 1, gz),
            gx_min,
            gx_max,
            gz_min,
            gz_max,
            &mut cheese_b,
        );

        let mut any_tunnel_region = false;
        for x in 0..PLANE_SIZE {
            for z in 0..PLANE_SIZE {
                if self.is_tunnel_region(chunk_origin_x + x as i32, chunk_origin_z + z as i32) {
                    any_tunnel_region = true;
                }
            }
        }
        let mut tunnel_a = [[0.0f32; CAVE_COLUMN_MAX_LAYERS]; CAVE_COLUMN_MAX_LAYERS];
        let mut tunnel_b = [[0.0f32; CAVE_COLUMN_MAX_LAYERS]; CAVE_COLUMN_MAX_LAYERS];
        let mut connector_a = [[0.0f32; CAVE_COLUMN_MAX_LAYERS]; CAVE_COLUMN_MAX_LAYERS];
        let mut connector_b = [[0.0f32; CAVE_COLUMN_MAX_LAYERS]; CAVE_COLUMN_MAX_LAYERS];
        if any_tunnel_region {
            self.grid_slice(
                Self::tunnel_grid_corner,
                |gx, gz| (gx, gy0, gz),
                gx_min,
                gx_max,
                gz_min,
                gz_max,
                &mut tunnel_a,
            );
            self.grid_slice(
                Self::tunnel_grid_corner,
                |gx, gz| (gx, gy0 + 1, gz),
                gx_min,
                gx_max,
                gz_min,
                gz_max,
                &mut tunnel_b,
            );
            self.grid_slice(
                Self::connector_grid_corner,
                |gx, gz| (gx, gy0, gz),
                gx_min,
                gx_max,
                gz_min,
                gz_max,
                &mut connector_a,
            );
            self.grid_slice(
                Self::connector_grid_corner,
                |gx, gz| (gx, gy0 + 1, gz),
                gx_min,
                gx_max,
                gz_min,
                gz_max,
                &mut connector_b,
            );
        }

        for (x, row) in out.iter_mut().enumerate() {
            let world_x = chunk_origin_x + x as i32;
            let gx0 = world_x.div_euclid(CAVE_GRID_STRIDE);
            let tx = world_x.rem_euclid(CAVE_GRID_STRIDE) as f32 / CAVE_GRID_STRIDE as f32;
            let (ix0, ix1) = ((gx0 - gx_min) as usize, (gx0 - gx_min + 1) as usize);

            for (z, cell) in row.iter_mut().enumerate() {
                let world_z = chunk_origin_z + z as i32;
                let h = height[x][z];

                if Self::is_water_position(h, world_y) {
                    *cell = true;
                    continue;
                }
                if world_y > h {
                    *cell = self.tree_occupies(world_x, world_y, world_z);
                    continue;
                }
                if h - world_y < MIN_CAVE_DEPTH {
                    *cell = true;
                    continue;
                }

                let gz0 = world_z.div_euclid(CAVE_GRID_STRIDE);
                let tz = world_z.rem_euclid(CAVE_GRID_STRIDE) as f32 / CAVE_GRID_STRIDE as f32;
                let (jz0, jz1) = ((gz0 - gz_min) as usize, (gz0 - gz_min + 1) as usize);

                let xz_layer = |grid: &GridSlice| {
                    lerp(
                        lerp(grid[ix0][jz0], grid[ix1][jz0], tx),
                        lerp(grid[ix0][jz1], grid[ix1][jz1], tx),
                        tz,
                    )
                };
                let cheese_density = lerp(xz_layer(&cheese_a), xz_layer(&cheese_b), ty);
                if cheese_density > self.cheese_threshold_at(world_y) {
                    continue;
                }
                if !any_tunnel_region || !self.is_tunnel_region(world_x, world_z) {
                    *cell = true;
                    continue;
                }
                let tunnel_diff = lerp(xz_layer(&tunnel_a), xz_layer(&tunnel_b), ty);
                if tunnel_diff < self.tunnel_threshold_at(world_y) {
                    continue;
                }
                let connector_diff = lerp(xz_layer(&connector_a), xz_layer(&connector_b), ty);
                if connector_diff < self.connector_threshold_at(world_y) {
                    continue;
                }
                *cell = true;
            }
        }

        out
    }

    /// Rand-Ebene senkrecht zur Z-Achse bei festem `world_z` - s. `solid_plane_x` fuer das
    /// allgemeine Prinzip. Z ist die MITTLERE Lerp-Achse in `cave_fields_at`s Formel: bei fixem Z
    /// kombinieren beide Gitter-Ausschnitte (an `gz0`/`gz0+1`) ueber `tz`, waehrend X (innen, `tx`)
    /// weiterhin INNERHALB jedes Ausschnitts kombiniert wird und Y (aussen, `ty`) wie gehabt zwei
    /// Zeilen mischt.
    pub fn solid_plane_z(
        &self,
        world_z: i32,
        chunk_origin_x: i32,
        chunk_origin_y: i32,
    ) -> BoundaryPlane {
        let mut out = [[false; PLANE_SIZE]; PLANE_SIZE];

        let mut height_by_x = [0i32; PLANE_SIZE];
        for (x, height) in height_by_x.iter_mut().enumerate() {
            *height = self.height_at(chunk_origin_x + x as i32, world_z);
        }

        let gz0 = world_z.div_euclid(CAVE_GRID_STRIDE);
        let tz = world_z.rem_euclid(CAVE_GRID_STRIDE) as f32 / CAVE_GRID_STRIDE as f32;
        let gx_min = chunk_origin_x.div_euclid(CAVE_GRID_STRIDE);
        let gx_max = (chunk_origin_x + CHUNK_SIZE - 1).div_euclid(CAVE_GRID_STRIDE) + 1;
        let gy_min = chunk_origin_y.div_euclid(CAVE_GRID_STRIDE);
        let gy_max = (chunk_origin_y + CHUNK_SIZE - 1).div_euclid(CAVE_GRID_STRIDE) + 1;

        let mut cheese_a = [[0.0f32; CAVE_COLUMN_MAX_LAYERS]; CAVE_COLUMN_MAX_LAYERS];
        let mut cheese_b = [[0.0f32; CAVE_COLUMN_MAX_LAYERS]; CAVE_COLUMN_MAX_LAYERS];
        self.grid_slice(
            Self::cheese_grid_corner,
            |gx, gy| (gx, gy, gz0),
            gx_min,
            gx_max,
            gy_min,
            gy_max,
            &mut cheese_a,
        );
        self.grid_slice(
            Self::cheese_grid_corner,
            |gx, gy| (gx, gy, gz0 + 1),
            gx_min,
            gx_max,
            gy_min,
            gy_max,
            &mut cheese_b,
        );

        let any_tunnel_region =
            (0..PLANE_SIZE).any(|x| self.is_tunnel_region(chunk_origin_x + x as i32, world_z));
        let mut tunnel_a = [[0.0f32; CAVE_COLUMN_MAX_LAYERS]; CAVE_COLUMN_MAX_LAYERS];
        let mut tunnel_b = [[0.0f32; CAVE_COLUMN_MAX_LAYERS]; CAVE_COLUMN_MAX_LAYERS];
        let mut connector_a = [[0.0f32; CAVE_COLUMN_MAX_LAYERS]; CAVE_COLUMN_MAX_LAYERS];
        let mut connector_b = [[0.0f32; CAVE_COLUMN_MAX_LAYERS]; CAVE_COLUMN_MAX_LAYERS];
        if any_tunnel_region {
            self.grid_slice(
                Self::tunnel_grid_corner,
                |gx, gy| (gx, gy, gz0),
                gx_min,
                gx_max,
                gy_min,
                gy_max,
                &mut tunnel_a,
            );
            self.grid_slice(
                Self::tunnel_grid_corner,
                |gx, gy| (gx, gy, gz0 + 1),
                gx_min,
                gx_max,
                gy_min,
                gy_max,
                &mut tunnel_b,
            );
            self.grid_slice(
                Self::connector_grid_corner,
                |gx, gy| (gx, gy, gz0),
                gx_min,
                gx_max,
                gy_min,
                gy_max,
                &mut connector_a,
            );
            self.grid_slice(
                Self::connector_grid_corner,
                |gx, gy| (gx, gy, gz0 + 1),
                gx_min,
                gx_max,
                gy_min,
                gy_max,
                &mut connector_b,
            );
        }

        for (x, row) in out.iter_mut().enumerate() {
            let world_x = chunk_origin_x + x as i32;
            let gx0 = world_x.div_euclid(CAVE_GRID_STRIDE);
            let tx = world_x.rem_euclid(CAVE_GRID_STRIDE) as f32 / CAVE_GRID_STRIDE as f32;
            let (ix0, ix1) = ((gx0 - gx_min) as usize, (gx0 - gx_min + 1) as usize);
            let height = height_by_x[x];

            // Baum-Suchradius haengt nur von (world_x, world_z) ab, nicht von Y - EINMAL pro
            // Spalte geholt statt bis zu 32x (einmal pro Y-Wert), s. Kommentar an
            // `nearby_tree_candidates`.
            let (nearby_trees, nearby_tree_count) = self.nearby_tree_candidates(world_x, world_z);

            for (y, cell) in row.iter_mut().enumerate() {
                let world_y = chunk_origin_y + y as i32;

                if Self::is_water_position(height, world_y) {
                    *cell = true;
                    continue;
                }
                if world_y > height {
                    *cell = Self::tree_occupies_among(
                        &nearby_trees[..nearby_tree_count],
                        world_x,
                        world_y,
                        world_z,
                    );
                    continue;
                }
                if height - world_y < MIN_CAVE_DEPTH {
                    *cell = true;
                    continue;
                }

                let gy0 = world_y.div_euclid(CAVE_GRID_STRIDE);
                let ty = world_y.rem_euclid(CAVE_GRID_STRIDE) as f32 / CAVE_GRID_STRIDE as f32;
                let (iy0, iy1) = ((gy0 - gy_min) as usize, (gy0 - gy_min + 1) as usize);

                let layer_at = |grid_a: &GridSlice, grid_b: &GridSlice, iy: usize| {
                    lerp(
                        lerp(grid_a[ix0][iy], grid_a[ix1][iy], tx),
                        lerp(grid_b[ix0][iy], grid_b[ix1][iy], tx),
                        tz,
                    )
                };
                let cheese_density = lerp(
                    layer_at(&cheese_a, &cheese_b, iy0),
                    layer_at(&cheese_a, &cheese_b, iy1),
                    ty,
                );
                if cheese_density > self.cheese_threshold_at(world_y) {
                    continue;
                }
                if !any_tunnel_region || !self.is_tunnel_region(world_x, world_z) {
                    *cell = true;
                    continue;
                }
                let tunnel_diff = lerp(
                    layer_at(&tunnel_a, &tunnel_b, iy0),
                    layer_at(&tunnel_a, &tunnel_b, iy1),
                    ty,
                );
                if tunnel_diff < self.tunnel_threshold_at(world_y) {
                    continue;
                }
                let connector_diff = lerp(
                    layer_at(&connector_a, &connector_b, iy0),
                    layer_at(&connector_a, &connector_b, iy1),
                    ty,
                );
                if connector_diff < self.connector_threshold_at(world_y) {
                    continue;
                }
                *cell = true;
            }
        }

        out
    }

    /// Alle 6 Rand-Ebenen EINES Chunks in einem Rutsch - genutzt vom asynchronen Rayon-Ladepfad
    /// (`ChunkManager::dispatch_pending`), der wegen der Thread-Grenze NIE echte Nachbar-Chunk-
    /// Referenzen an `mesh_chunk` uebergeben kann und dessen `compute_exposure` deshalb IMMER auf
    /// den prozeduralen Fallback zurueckfaellt - vorher 6144 (6 Seiten * 1024 Randzellen)
    /// Einzelaufrufe von `is_solid` pro Chunk, jetzt 6 gebatchte Ebenen-Berechnungen.
    pub fn boundary_planes(&self, chunk_x: i32, chunk_y: i32, chunk_z: i32) -> BoundaryPlanes {
        let ox = chunk_x * CHUNK_SIZE;
        let oy = chunk_y * CHUNK_SIZE;
        let oz = chunk_z * CHUNK_SIZE;
        BoundaryPlanes {
            neg_x: self.solid_plane_x(ox - 1, oy, oz),
            pos_x: self.solid_plane_x(ox + CHUNK_SIZE, oy, oz),
            neg_y: self.solid_plane_y(oy - 1, ox, oz),
            pos_y: self.solid_plane_y(oy + CHUNK_SIZE, ox, oz),
            neg_z: self.solid_plane_z(oz - 1, ox, oy),
            pos_z: self.solid_plane_z(oz + CHUNK_SIZE, ox, oy),
        }
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
    ) -> (
        [f32; CAVE_COLUMN_MAX_LAYERS],
        [f32; CAVE_COLUMN_MAX_LAYERS],
        [f32; CAVE_COLUMN_MAX_LAYERS],
    ) {
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
        lerp(
            layers[(gy0 - gy_min) as usize],
            layers[(gy0 - gy_min) as usize + 1],
            ty,
        )
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

        // Chunk liegt vollstaendig ueber Terrainoberflaeche UND Wasserspiegel (PLUS Sicherheitsmarge
        // fuer Baumkronen aus Nachbar-Spalten ausserhalb der eigenen 1024 Saeulen, s.
        // `TREE_HEIGHT_SAFETY_MARGIN`) - reine Luft, `chunk.clear()` oben reicht bereits.
        if chunk_origin_y > chunk_max_height + TREE_HEIGHT_SAFETY_MARGIN
            && chunk_origin_y > WATER_LEVEL
        {
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
                        cell_max_height = cell_max_height
                            .max(local_height[(local_z * CHUNK_SIZE + local_x) as usize]);
                    }
                }
                let cell_gy_max = cave_gy_max.min(
                    cell_max_height
                        .min(chunk_origin_y + CHUNK_SIZE - 1)
                        .div_euclid(CAVE_GRID_STRIDE)
                        + 1,
                );

                // Cheese Caves: ungegatet, EINMAL PRO ZELLE (16 Spalten) statt pro Spalte geholt -
                // s. Kommentar an `cave_grid_stack`.
                let (cheese_count, cheese_c00, cheese_c10, cheese_c01, cheese_c11) = self
                    .cave_grid_stack(Self::cheese_grid_corner, gx0, gz0, cave_gy_min, cell_gy_max);

                // Tunnel-/Connector-Zellen-Stacks werden NUR bei Bedarf geholt (erste Spalte der
                // Zelle mit `is_tunnel_region == true`) und dann fuer den Rest der Zelle
                // wiederverwendet - bei `cave_region_frequency` (500-Block-Wellenlaenge) ist das
                // Ergebnis innerhalb einer 4-Block-Zelle so gut wie immer fuer alle 16 Spalten
                // identisch. Beide Systeme teilen sich dieselbe Gate (`is_tunnel_region`).
                let mut tunnel_stack: Option<CaveGridStack> = None;
                let mut connector_stack: Option<CaveGridStack> = None;

                for cell_local_z in 0..CAVE_GRID_STRIDE {
                    for cell_local_x in 0..CAVE_GRID_STRIDE {
                        let local_x = cell_x * CAVE_GRID_STRIDE + cell_local_x;
                        let local_z = cell_z * CAVE_GRID_STRIDE + cell_local_z;
                        let height = local_height[(local_z * CHUNK_SIZE + local_x) as usize];
                        let column_has_terrain = chunk_origin_y <= height;
                        let column_has_water =
                            height < WATER_LEVEL && chunk_origin_y <= WATER_LEVEL;
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
                                chunk.set_block(
                                    local_x,
                                    world_y - chunk_origin_y,
                                    local_z,
                                    blocks::WATER,
                                );
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

                        let tx =
                            world_x.rem_euclid(CAVE_GRID_STRIDE) as f32 / CAVE_GRID_STRIDE as f32;
                        let tz =
                            world_z.rem_euclid(CAVE_GRID_STRIDE) as f32 / CAVE_GRID_STRIDE as f32;

                        // Bounds PRO SLAB (4-Voxel-Y-Streifen) statt fuer die ganze Spalte - bei voll
                        // unterirdischen 32-Voxel-Spalten wuerde ein spaltenweiter Bound fast immer
                        // auf `Maybe` degenerieren (s. Kommentar an `slab_bounds`), pro Slab loesen
                        // sich die meisten Streifen dagegen eindeutig auf.
                        let (cheese_layers, cheese_layer_min, cheese_layer_max) =
                            Self::cave_column_from_stack(
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

                        // Tunnel/Connector: nur ausgewertet, wenn die Spalte ueberhaupt in einer
                        // Hoehlen-aktiven Region liegt - CarveBound::Never ohne jede
                        // Gitter-Auswertung, wenn nicht.
                        let tunnel_region = self.is_tunnel_region(world_x, world_z);
                        let (tunnel_count, tunnel_bounds, tunnel_any_carve, tunnel_layers) =
                            if tunnel_region {
                                let (count, c00, c10, c01, c11) =
                                    tunnel_stack.get_or_insert_with(|| {
                                        self.cave_grid_stack(
                                            Self::tunnel_grid_corner,
                                            gx0,
                                            gz0,
                                            cave_gy_min,
                                            cell_gy_max,
                                        )
                                    });
                                let (layers, layer_min, layer_max) = Self::cave_column_from_stack(
                                    tx, tz, *count, c00, c10, c01, c11,
                                );
                                let bounds = Self::slab_bounds(
                                    cave_gy_min,
                                    *count,
                                    &layer_min,
                                    &layer_max,
                                    |y| self.tunnel_threshold_at(y),
                                    true,
                                );
                                let any_carve = bounds[..count.saturating_sub(1)]
                                    .iter()
                                    .any(|b| *b != CarveBound::Never);
                                (*count, bounds, any_carve, layers)
                            } else {
                                (
                                    0,
                                    [CarveBound::Never; CAVE_COLUMN_MAX_LAYERS],
                                    false,
                                    [0.0; CAVE_COLUMN_MAX_LAYERS],
                                )
                            };
                        // Connector-Fetch nur, wenn er ueberhaupt etwas beitragen KOENNTE: wenn
                        // Cheese oder Tunnel bereits JEDES beruehrte Slab auf `Always` aufloesen,
                        // wird die Spalte per Kurzschluss (`||`) im Voxel-Loop unten sowieso nie bis
                        // zum Connector-Check kommen - dann lohnt sich nicht mal die Gitter-Abfrage.
                        let fully_resolved_without_connector = (0..cheese_count.saturating_sub(1))
                            .all(|slab| {
                                cheese_bounds[slab] == CarveBound::Always
                                    || tunnel_bounds[slab] == CarveBound::Always
                            });
                        let (
                            connector_count,
                            connector_bounds,
                            connector_any_carve,
                            connector_layers,
                        ) = if tunnel_region && !fully_resolved_without_connector {
                            let (count, c00, c10, c01, c11) =
                                connector_stack.get_or_insert_with(|| {
                                    self.cave_grid_stack(
                                        Self::connector_grid_corner,
                                        gx0,
                                        gz0,
                                        cave_gy_min,
                                        cell_gy_max,
                                    )
                                });
                            let (layers, layer_min, layer_max) =
                                Self::cave_column_from_stack(tx, tz, *count, c00, c10, c01, c11);
                            let bounds = Self::slab_bounds(
                                cave_gy_min,
                                *count,
                                &layer_min,
                                &layer_max,
                                |y| self.connector_threshold_at(y),
                                true,
                            );
                            let any_carve = bounds[..count.saturating_sub(1)]
                                .iter()
                                .any(|b| *b != CarveBound::Never);
                            (*count, bounds, any_carve, layers)
                        } else {
                            (
                                0,
                                [CarveBound::Never; CAVE_COLUMN_MAX_LAYERS],
                                false,
                                [0.0; CAVE_COLUMN_MAX_LAYERS],
                            )
                        };

                        for local_y in 0..CHUNK_SIZE {
                            let world_y = chunk_origin_y + local_y;
                            if world_y > height {
                                continue;
                            }

                            let depth_from_surface = height - world_y;
                            if depth_from_surface >= MIN_CAVE_DEPTH
                                && (cheese_any_carve || tunnel_any_carve || connector_any_carve)
                            {
                                let slab =
                                    (world_y.div_euclid(CAVE_GRID_STRIDE) - cave_gy_min) as usize;
                                let cheese_slab = cheese_bounds[slab];
                                let tunnel_slab = if tunnel_count > 0 {
                                    tunnel_bounds[slab]
                                } else {
                                    CarveBound::Never
                                };
                                let connector_slab = if connector_count > 0 {
                                    connector_bounds[slab]
                                } else {
                                    CarveBound::Never
                                };
                                let carved = cheese_slab == CarveBound::Always
                                    || tunnel_slab == CarveBound::Always
                                    || connector_slab == CarveBound::Always
                                    || (cheese_slab == CarveBound::Maybe
                                        && Self::cave_from_layers(
                                            world_y,
                                            cave_gy_min,
                                            &cheese_layers,
                                        ) > self.cheese_threshold_at(world_y))
                                    || (tunnel_slab == CarveBound::Maybe
                                        && Self::cave_from_layers(
                                            world_y,
                                            cave_gy_min,
                                            &tunnel_layers,
                                        ) < self.tunnel_threshold_at(world_y))
                                    || (connector_slab == CarveBound::Maybe
                                        && Self::cave_from_layers(
                                            world_y,
                                            cave_gy_min,
                                            &connector_layers,
                                        ) < self.connector_threshold_at(world_y));
                                if carved {
                                    continue;
                                }
                            }

                            let block_id = blocks::surface_block(
                                depth_from_surface,
                                slope,
                                self.dirt_layer_depth,
                                surface,
                            );
                            chunk.set_block(local_x, local_y, local_z, block_id);
                        }
                    }
                }
            }
        }

        self.place_flora(chunk_x, chunk_y, chunk_z, chunk);
    }

    /// Oberflaechen-Kontext einer Spalte: Hoehenband-Strand, Unterwasser-Boden und das strikte
    /// 2D-Biom-Mapping (Wueste NUR bei Temperatur > min UND Feuchtigkeit < max). Temperatur/
    /// Feuchtigkeit sind sehr niedrigfrequent + hart geschwellt - grosse zusammenhaengende Biome,
    /// kein Einzelspalten-Bleeding. Bewusst NICHT Teil von `is_solid`: Biome aendern nur die
    /// Block-ID (Sand vs. Gras), nie die Festigkeit - keine Konsistenzanforderung an den Fallback,
    /// keine zusaetzlichen Rauschproben im Mesher-/Physik-Hotpath.
    fn column_surface(&self, world_x: i32, world_z: i32, height: i32) -> ColumnSurface {
        let sample = self.pyramid.sample(world_x, world_z);
        let rock_height = ROCK_HEIGHT + sample.temperature * ROCK_HEIGHT_TEMPERATURE_DITHER;
        ColumnSurface {
            is_beach: (height - WATER_LEVEL).abs() <= BEACH_HALF_RANGE,
            is_underwater: height < WATER_LEVEL,
            is_desert: sample.temperature > self.desert_temperature_min
                && sample.humidity < self.desert_humidity_max,
            is_rock: height as f32 > rock_height,
            temperature: sample.temperature,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::engine::config::EngineConfig;

    /// Akzeptanztest fuer das Cross-Chunk-Radius-Sampling: findet einen Baum-Kandidaten, dessen
    /// Stamm+Kronen-Footprint mindestens eine Chunk-Grenze ueberschreitet, generiert NUR die
    /// betroffenen Chunks (jeden unabhaengig, wie im echten `dispatch_pending`-Pfad - keiner liest
    /// vom anderen), und verifiziert JEDEN erwarteten Baum-Voxel gegen den tatsaechlich generierten
    /// Chunk. Erwartung wird gegen `is_solid` VOR der Platzierung abgeglichen (ein Voxel, das schon
    /// durch Terrain/Wasser belegt ist, bleibt zurecht ausgespart, s. `place_tree_voxel`) - deckt
    /// damit exakt das ab, was die Aufgabenstellung "Radius-Check (Cross-Chunk)" fordert: derselbe
    /// Baum, unabhaengig generiert ueber mehrere Chunks hinweg, ergibt keine Luecken/Duplikate.
    #[test]
    fn tree_footprint_matches_across_independently_generated_chunks() {
        let mut config = EngineConfig::default();
        config.dev.terrain_tree_spawn_chance = 1.0;
        // Gitter deutlich groesser als der Isolations-Mindestabstand (2*crown_radius+1=9) - sonst
        // ueberlappen selbst gejitterte Nachbarzellen fast immer und kein Kandidat besteht den
        // Isolations-Check unten.
        config.dev.terrain_tree_grid_size = 20;
        config.dev.terrain_tree_trunk_height_min = 5;
        config.dev.terrain_tree_trunk_height_max = 5;
        config.dev.terrain_tree_crown_radius_min = 4;
        config.dev.terrain_tree_crown_radius_max = 4;
        let generator = TerrainGenerator::new(&config);

        // Reiner Terrain/Wasser-Belegungs-Check OHNE Baum-Wissen (anders als `is_solid`, das jetzt
        // absichtlich auch `tree_occupies` einschliesst) - modelliert exakt, was `place_tree_voxel`
        // in einem frisch generierten Chunk VOR jeder Baum-Platzierung als "nicht mehr Luft" sieht.
        let terrain_occupied = |world_x: i32, world_y: i32, world_z: i32| {
            let height = generator.height_at(world_x, world_z);
            TerrainGenerator::is_water_position(height, world_y) || world_y <= height
        };

        let mut chunk_cache: HashMap<(i32, i32, i32), Chunk> = HashMap::new();
        let mut best_touched = 0usize;

        // Nicht jeder geometrisch grenzueberschreitende Kandidat bleibt das auch NACH dem
        // Terrain-Occlusion-Check (Nachbarsaeulen koennen den ganzen Teil jenseits der Grenze
        // verdecken) - iteriert deshalb ueber mehrere Kandidaten und nimmt den ersten, der
        // tatsaechlich >= 2 unabhaengig generierte Chunks mit verifizierbaren Voxeln beruehrt UND
        // isoliert genug von Nachbarbaeumen steht (sonst koennte ein frueher verarbeiteter
        // Nachbarbaum Voxel "wegschnappen", was hier eine andere, aber KORREKTE Interaktion waere,
        // die dieser Test bewusst nicht mitprueft).
        'search: for cell_z in -20..20 {
            for cell_x in -20..20 {
                let Some(tree) = generator.tree_candidate(cell_x, cell_z) else {
                    continue;
                };
                let crosses_x = (tree.world_x - tree.crown_radius).div_euclid(CHUNK_SIZE)
                    != (tree.world_x + tree.crown_radius).div_euclid(CHUNK_SIZE);
                let crosses_z = (tree.world_z - tree.crown_radius).div_euclid(CHUNK_SIZE)
                    != (tree.world_z + tree.crown_radius).div_euclid(CHUNK_SIZE);
                if !crosses_x && !crosses_z {
                    continue;
                }

                let isolated = (-2..=2).all(|nz| {
                    (-2..=2).all(|nx| {
                        if nx == 0 && nz == 0 {
                            return true;
                        }
                        match generator.tree_candidate(cell_x + nx, cell_z + nz) {
                            None => true,
                            Some(other) => {
                                let min_gap = tree.crown_radius + other.crown_radius + 1;
                                (tree.world_x - other.world_x).abs() > min_gap
                                    || (tree.world_z - other.world_z).abs() > min_gap
                            }
                        }
                    })
                });
                if !isolated {
                    continue;
                }

                // Stamm ZUERST eingetragen und Skelett (Aeste/Blaetter) nur `or_insert` - `place_flora`
                // platziert in derselben Reihenfolge und ueberschreibt nichts bereits Belegtes (s.
                // `place_tree_voxel`). Nutzt dieselben Formeln (`point_to_segment_distance`,
                // `leaf_cluster_radius`) wie `place_flora`/`tree_occupies_among` statt sie zu
                // duplizieren - dieser Test prueft die Cross-Chunk-Verteilung, nicht die Geometrie
                // selbst (die deckt `is_solid_prediction_matches_generated_blocks_everywhere` ab).
                let trunk_top = tree.ground_y + tree.trunk_height;
                let mut expected: HashMap<(i32, i32, i32), u16> = HashMap::new();
                for world_y in (tree.ground_y + 1)..=trunk_top {
                    expected.insert((tree.world_x, world_y, tree.world_z), blocks::LOG);
                }
                let root =
                    glam::Vec3::new(tree.world_x as f32, trunk_top as f32, tree.world_z as f32);
                let node_count = tree.node_count as usize;
                for i in 1..node_count {
                    let from = root + tree.nodes[tree.parents[i] as usize];
                    let to = root + tree.nodes[i];
                    let min = from.min(to) - glam::Vec3::splat(super::flora::BRANCH_RADIUS);
                    let max = from.max(to) + glam::Vec3::splat(super::flora::BRANCH_RADIUS);
                    for wy in min.y.floor() as i32..=max.y.ceil() as i32 {
                        for wz in min.z.floor() as i32..=max.z.ceil() as i32 {
                            for wx in min.x.floor() as i32..=max.x.ceil() as i32 {
                                let p = glam::Vec3::new(wx as f32, wy as f32, wz as f32);
                                if super::flora::point_to_segment_distance(p, from, to)
                                    <= super::flora::BRANCH_RADIUS
                                {
                                    expected.entry((wx, wy, wz)).or_insert(blocks::LOG);
                                }
                            }
                        }
                    }
                }
                let leaf_radius =
                    super::flora::leaf_cluster_radius(tree.species, tree.crown_radius as f32);
                for i in 0..node_count {
                    let center = root + tree.nodes[i];
                    let min = center - glam::Vec3::splat(leaf_radius);
                    let max = center + glam::Vec3::splat(leaf_radius);
                    for wy in min.y.floor() as i32..=max.y.ceil() as i32 {
                        for wz in min.z.floor() as i32..=max.z.ceil() as i32 {
                            for wx in min.x.floor() as i32..=max.x.ceil() as i32 {
                                let p = glam::Vec3::new(wx as f32, wy as f32, wz as f32);
                                if p.distance(center) <= leaf_radius {
                                    expected.entry((wx, wy, wz)).or_insert(blocks::LEAVES);
                                }
                            }
                        }
                    }
                }

                let mut touched_chunks = std::collections::HashSet::new();
                let mut verified: Vec<(i32, i32, i32, u16, (i32, i32, i32))> = Vec::new();
                for (&(world_x, world_y, world_z), &expected_block) in &expected {
                    if terrain_occupied(world_x, world_y, world_z) {
                        // Vorbelegt durch Terrain/Wasser einer Nachbarsaeule - `place_tree_voxel`
                        // uebermalt das bewusst nicht, hier also keine Erwartung.
                        continue;
                    }
                    let chunk_coord = (
                        world_x.div_euclid(CHUNK_SIZE),
                        world_y.div_euclid(CHUNK_SIZE),
                        world_z.div_euclid(CHUNK_SIZE),
                    );
                    touched_chunks.insert(chunk_coord);
                    verified.push((world_x, world_y, world_z, expected_block, chunk_coord));
                }
                best_touched = best_touched.max(touched_chunks.len());
                if touched_chunks.len() < 2 {
                    continue;
                }

                for (world_x, world_y, world_z, expected_block, chunk_coord) in verified {
                    let chunk = chunk_cache.entry(chunk_coord).or_insert_with(|| {
                        let mut c = Chunk::empty();
                        generator.generate_chunk(
                            chunk_coord.0,
                            chunk_coord.1,
                            chunk_coord.2,
                            &mut c,
                        );
                        c
                    });
                    let local_x = world_x.rem_euclid(CHUNK_SIZE);
                    let local_y = world_y.rem_euclid(CHUNK_SIZE);
                    let local_z = world_z.rem_euclid(CHUNK_SIZE);
                    assert_eq!(
                        chunk.get_block(local_x, local_y, local_z),
                        expected_block,
                        "Baum-Voxel Welt({world_x},{world_y},{world_z}) in Chunk {chunk_coord:?} lokal \
                         ({local_x},{local_y},{local_z}): erwartet {expected_block}"
                    );
                }
                break 'search;
            }
        }

        assert!(
            best_touched >= 2,
            "kein Baum-Kandidat im Suchbereich beruehrte nach Terrain-Occlusion >= 2 Chunks (bester Fund: {best_touched})"
        );
    }

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

    /// `TerrainGenerator::boundary_planes` MUSS fuer jeden Randpunkt exakt dasselbe liefern wie der
    /// per-Punkt-Fallback `is_solid` - beide muessen bit-identisch sein, sonst wiederholt sich
    /// exakt der Bulk/Fallback-Divergenz-Bug, der `is_carved` schon einmal getroffen hat (s. dortiger
    /// Kommentar). Deckt alle 6 Ebenen ueber dieselben diversen Chunk-Koordinaten wie der
    /// `is_solid`-Gesamttest ab, inklusive vertikaler Stapelung (Tunnel-/Cheese-Gitter unterscheiden
    /// sich je nach Achse strukturell, s. `solid_plane_x/y/z`-Kommentare - alle drei muessen separat
    /// geprueft werden).
    #[test]
    fn boundary_planes_match_is_solid_everywhere() {
        let generator = TerrainGenerator::new(&EngineConfig::default());

        let coords: Vec<(i32, i32, i32)> = [(0, 0, 0), (3, -2, -5), (-4, 1, 2), (7, 0, -1)]
            .into_iter()
            .chain((0..12).map(|i| (i * 5 - 30, (i % 5) - 2, i * 7 - 40)))
            .collect();

        for &(chunk_x, chunk_y, chunk_z) in &coords {
            let ox = chunk_x * CHUNK_SIZE;
            let oy = chunk_y * CHUNK_SIZE;
            let oz = chunk_z * CHUNK_SIZE;
            let planes = generator.boundary_planes(chunk_x, chunk_y, chunk_z);

            for y in 0..CHUNK_SIZE {
                for z in 0..CHUNK_SIZE {
                    let expected = generator.is_solid(ox - 1, oy + y, oz + z);
                    assert_eq!(
                        planes.neg_x[y as usize][z as usize], expected,
                        "neg_x bei Chunk ({chunk_x},{chunk_y},{chunk_z}) y={y} z={z}"
                    );
                    let expected = generator.is_solid(ox + CHUNK_SIZE, oy + y, oz + z);
                    assert_eq!(
                        planes.pos_x[y as usize][z as usize], expected,
                        "pos_x bei Chunk ({chunk_x},{chunk_y},{chunk_z}) y={y} z={z}"
                    );
                }
            }
            for x in 0..CHUNK_SIZE {
                for z in 0..CHUNK_SIZE {
                    let expected = generator.is_solid(ox + x, oy - 1, oz + z);
                    assert_eq!(
                        planes.neg_y[x as usize][z as usize], expected,
                        "neg_y bei Chunk ({chunk_x},{chunk_y},{chunk_z}) x={x} z={z}"
                    );
                    let expected = generator.is_solid(ox + x, oy + CHUNK_SIZE, oz + z);
                    assert_eq!(
                        planes.pos_y[x as usize][z as usize], expected,
                        "pos_y bei Chunk ({chunk_x},{chunk_y},{chunk_z}) x={x} z={z}"
                    );
                }
            }
            for x in 0..CHUNK_SIZE {
                for y in 0..CHUNK_SIZE {
                    let expected = generator.is_solid(ox + x, oy + y, oz - 1);
                    assert_eq!(
                        planes.neg_z[x as usize][y as usize], expected,
                        "neg_z bei Chunk ({chunk_x},{chunk_y},{chunk_z}) x={x} y={y}"
                    );
                    let expected = generator.is_solid(ox + x, oy + y, oz + CHUNK_SIZE);
                    assert_eq!(
                        planes.pos_z[x as usize][y as usize], expected,
                        "pos_z bei Chunk ({chunk_x},{chunk_y},{chunk_z}) x={x} y={y}"
                    );
                }
            }
        }
    }

    /// Diagnose-Tool, KEIN Korrektheitstest - misst die tatsaechliche Verteilung von Cheese-Cave-
    /// Dichte, Tunnel-F2-F1-Distanz und Region-Gate an vielen Punkten, um die Schwellwerte empirisch
    /// zu kalibrieren statt sie zu erraten (ein naiv "klein wirkender" Schwellwert kann je nach
    /// Durchsatz der Chunk-Generierung (Pyramide + Hoehlen + Flora) ueber ein frisches
    /// Oberflaechen-Streaming-Fenster, inklusive Kalt-Start der Fenster-Caches. Manuell:
    /// `cargo test --release --lib -- --ignored --nocapture profile_generate_chunk`
    #[test]
    #[ignore = "Diagnose-Tool, kein automatisierter Test - siehe Doc-Kommentar"]
    fn profile_generate_chunk() {
        use std::time::Instant;

        let generator = TerrainGenerator::new(&EngineConfig::default());
        let mut chunk = Chunk::empty();

        let start = Instant::now();
        let mut generated = 0u32;
        for chunk_x in -8..8 {
            for chunk_z in -8..8 {
                for chunk_y in -2..2 {
                    generator.generate_chunk(chunk_x, chunk_y, chunk_z, &mut chunk);
                    generated += 1;
                }
            }
        }
        let elapsed = start.elapsed();
        println!(
            "{generated} Chunks in {:.1} ms -> {:.3} ms/Chunk ({:.0} Chunks/s, 1 Thread)",
            elapsed.as_secs_f64() * 1000.0,
            elapsed.as_secs_f64() * 1000.0 / generated as f64,
            generated as f64 / elapsed.as_secs_f64(),
        );
    }

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
                let (x, y, z) = (
                    i as i32 * 37 - 700_000,
                    i as i32 * 11 - 50_000,
                    i as i32 * 53 - 900_000,
                );
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
                let (x, y, z) = (
                    i as i32 * 37 - 700_000,
                    i as i32 * 11 - 50_000,
                    i as i32 * 53 - 900_000,
                );
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

        let mut connector: Vec<f32> = (0..N)
            .map(|i| {
                let (x, y, z) = (
                    i as i32 * 37 - 700_000,
                    i as i32 * 11 - 50_000,
                    i as i32 * 53 - 900_000,
                );
                generator.sample3d(&generator.connector, generator.connector_frequency, x, y, z)
            })
            .collect();
        println!(
            "connector F2-F1: p1={:.4} p2={:.4} p5={:.4} p10={:.4} p50={:.4}",
            percentile(&mut connector, 0.01),
            percentile(&mut connector, 0.02),
            percentile(&mut connector, 0.05),
            percentile(&mut connector, 0.10),
            percentile(&mut connector, 0.50),
        );

        let mut region: Vec<f32> = (0..N)
            .map(|i| {
                let (x, z) = (i as i32 * 37 - 700_000, i as i32 * 53 - 900_000);
                generator.sample2d(
                    &generator.cave_region,
                    generator.cave_region_frequency,
                    x,
                    z,
                )
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
