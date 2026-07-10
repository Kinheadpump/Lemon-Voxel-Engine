use glam::Vec3;

use crate::engine::config::EngineConfig;

use super::generator::TerrainGenerator;

/// Eine Godray-Billboard-Instanz, wie sie im SSBO liegt (siehe `render/godray_compute.wgsl` und
/// `render/godray_render.wgsl`). `intensity` wird ausschliesslich vom Compute-Pass geschrieben/
/// gelesen (In-Place-Temporal-Blend) - die CPU-Seite setzt sie bei einer Neu-Platzierung nur
/// einmalig auf 0 und ruehrt sie danach nicht an.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GodrayInstanceData {
    /// xyz = Basis-Position (direkt AUF der Terrainoberflaeche, nicht frei schwebend - der
    /// Kantenerkennungs-Punkt liegt `sample_height` darueber), w = Intensity.
    pub position_intensity: [f32; 4],
    /// x = Sample-Radius/Billboard-Breite, y = sichtbare Strahllaenge entlang der Lichtrichtung,
    /// z = Hoehe des Kantenerkennungs-Punkts ueber der Basis, w = ungenutzt.
    pub size: [f32; 4],
}

/// Wie viele Bloecke unter der (glatten) 2D-Oberflaeche `find_nearby_cave_opening` nach einer
/// Hoehlendecke sucht - klein halten, jede Stufe kostet einen `is_solid`-Aufruf (in Hoehlenregionen
/// mehrere hundert ns) UND die Kandidaten-Regenerierung laeuft synchron auf dem Main-Thread.
const CAVE_PROBE_MAX_DEPTH: i32 = 6;
/// Minimaler Hoehenunterschied zu den 4 Nachbar-Spalten, ab dem eine reine Oberflaechen-Position
/// (kein Hoehleneinschlag gefunden) ueberhaupt als Godray-Kandidat akzeptiert wird - auf komplett
/// flachem Land kann die GPU-seitige Kantenerkennung (16 Samples im `sample_height`-Radius) nie
/// einen echten Licht/Schatten-Uebergang finden, der Slot waere verschwendet.
const MIN_SURFACE_RELIEF: i32 = 2;

/// Platziert Godray-Kandidaten auf einem an die Terrainoberflaeche angehefteten Gitter um die
/// Kamera. Regeneriert nur bei ausreichender Kamerabewegung (wie das Chunk-Ladefenster) statt jeden
/// Frame Rauschen abzufragen und den kompletten SSBO neu hochzuladen.
pub struct GodrayField {
    count: u32,
    grid_spacing: f32,
    sample_height: f32,
    width: f32,
    beam_length: f32,
    last_center: Option<Vec3>,
    regen_threshold: f32,
    instances: Vec<GodrayInstanceData>,
}

/// Sucht straight nach unten ab der (2D-)Oberflaechenhoehe nach einem nahen Luftpocket unter festem
/// Fels - genau das ist eine Hoehlendecke, durch die Tageslicht faellt. `generator.is_solid` sieht
/// (anders als die reine Heightmap `height_at`) die tatsaechliche 3D-Aushoehlung (Cheese Caves,
/// Tunnelnetz). Liefert die Y-Koordinate des ersten Luftblocks unter der Oberflaeche, falls
/// innerhalb `CAVE_PROBE_MAX_DEPTH` einer gefunden wird.
fn find_nearby_cave_opening(generator: &TerrainGenerator, world_x: i32, world_z: i32, surface_y: i32) -> Option<i32> {
    (1..=CAVE_PROBE_MAX_DEPTH)
        .map(|depth| surface_y - depth)
        .find(|&y| !generator.is_solid(world_x, y, world_z))
}

/// Groesster Hoehenunterschied zu den 4 Nachbar-Spalten (1 Block Abstand) - billiges Relief-Mass
/// aus bereits gecachten `height_at`-Aufrufen, um flache (fuer Kantenerkennung nutzlose) Kandidaten
/// zu verwerfen.
fn local_relief(generator: &TerrainGenerator, world_x: i32, world_z: i32, height: i32) -> i32 {
    [(1, 0), (-1, 0), (0, 1), (0, -1)]
        .into_iter()
        .map(|(dx, dz)| (height - generator.height_at(world_x + dx, world_z + dz)).abs())
        .max()
        .unwrap_or(0)
}

impl GodrayField {
    pub fn new(config: &EngineConfig) -> Self {
        let grid_dim = (config.dev.godray_count as f32).sqrt().ceil() as u32;
        Self {
            count: config.dev.godray_count,
            grid_spacing: config.dev.godray_grid_spacing,
            sample_height: config.dev.godray_sample_height,
            width: config.dev.godray_width,
            beam_length: config.dev.godray_beam_length,
            last_center: None,
            regen_threshold: config.dev.godray_grid_spacing * grid_dim as f32 * 0.5,
            instances: Vec::with_capacity(config.dev.godray_count as usize),
        }
    }

    pub fn capacity(&self) -> u32 {
        self.count
    }

    /// Regeneriert das Gitter, wenn sich die Kamera weit genug vom letzten Platzierungs-Zentrum
    /// entfernt hat. Liefert bei Regenerierung die volle Instanz-Liste fuer einen SSBO-Reupload,
    /// sonst `None` - vermeidet unnoetige Noise-Abfragen und Buffer-Schreibvorgaenge pro Frame.
    pub fn update(&mut self, camera_position: Vec3, generator: &TerrainGenerator) -> Option<&[GodrayInstanceData]> {
        if let Some(last) = self.last_center
            && last.distance(camera_position) < self.regen_threshold
        {
            return None;
        }
        self.last_center = Some(camera_position);

        let grid_dim = (self.count as f32).sqrt().ceil() as i32;
        let half = grid_dim / 2;

        // Simples xorshift-Hash statt eines perfekten Gitters - eine exakt regelmaessige
        // Godray-Verteilung wirkt sofort sichtbar kuenstlich/repetitiv.
        let mut state: u32 = 0x9E3779B9 ^ (camera_position.x as i32 as u32).wrapping_mul(0x85EBCA6B);
        let mut next_jitter = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 / u32::MAX as f32) - 0.5
        };

        // Der GPU-seitige Compute-/Render-Pass verarbeitet IMMER exakt `capacity` Slots (fixer
        // Dispatch/Draw-Count, s. `GodrayPass`) - ungeschriebene Slots wuerden also veraltete Daten
        // vom letzten Upload zeigen. Pro Gitterzelle werden deshalb bis zu `CELL_RETRIES` jitterte
        // Positionen versucht (Hoehlenoeffnung > Oberflaechen-Relief > irgendeine), aber IMMER
        // genau eine Instanz geschrieben, nie eine Zelle ausgelassen.
        const CELL_RETRIES: u32 = 3;

        self.instances.clear();
        'grid: for gz in -half..=half {
            for gx in -half..=half {
                if self.instances.len() as u32 >= self.count {
                    break 'grid;
                }

                let mut best: Option<GodrayInstanceData> = None;
                for _ in 0..CELL_RETRIES {
                    let world_x =
                        camera_position.x + gx as f32 * self.grid_spacing + next_jitter() * self.grid_spacing * 0.6;
                    let world_z =
                        camera_position.z + gz as f32 * self.grid_spacing + next_jitter() * self.grid_spacing * 0.6;
                    let column_x = world_x.floor() as i32;
                    let column_z = world_z.floor() as i32;
                    let surface_y = generator.height_at(column_x, column_z);

                    // Hoehlenoeffnung gefunden: Basis DIREKT IM Luftpocket unter der Oberflaeche
                    // platzieren, mit deutlich kleinerem Kantenerkennungs-Versatz (die Decke ist
                    // nur wenige Bloecke ueber dem Pocket, nicht `sample_height` wie im Freien) -
                    // das ist die eigentliche "Lichtschacht durch ein Loch in der Hoehlendecke".
                    if let Some(air_y) = find_nearby_cave_opening(generator, column_x, column_z, surface_y) {
                        best = Some(GodrayInstanceData {
                            position_intensity: [world_x, air_y as f32 + 0.5, world_z, 0.0],
                            size: [self.width, self.beam_length, (self.sample_height * 0.5).max(0.5), 0.0],
                        });
                        break;
                    }

                    // Keine Hoehle: nur bei echtem lokalem Relief (Klippenkante, Huegelkamm)
                    // behalten - auf komplett flachem Land kann die GPU-Kantenerkennung ohnehin nie
                    // triggern. `Some(_)` wird nur durch eine bessere/gleich gute Alternative in der
                    // naechsten Iteration ersetzt, s.u.
                    if local_relief(generator, column_x, column_z, surface_y) >= MIN_SURFACE_RELIEF {
                        // Direkt AUF der Oberflaeche (nur minimaler Epsilon-Versatz gegen
                        // Z-Fighting) statt frei schwebend - der Kantenerkennungs-Punkt
                        // (`sample_height` darueber) soll mit der tatsaechlichen Voxel-Silhouette
                        // an dieser Stelle interagieren koennen.
                        best = Some(GodrayInstanceData {
                            position_intensity: [world_x, surface_y as f32 + 1.05, world_z, 0.0],
                            size: [self.width, self.beam_length, self.sample_height, 0.0],
                        });
                        break;
                    }

                    // Weder Hoehle noch Relief - als schwaechster Fallback merken, falls auch die
                    // letzte Runde nichts Besseres findet (nie eine Zelle auslassen).
                    best.get_or_insert(GodrayInstanceData {
                        position_intensity: [world_x, surface_y as f32 + 1.05, world_z, 0.0],
                        size: [self.width, self.beam_length, self.sample_height, 0.0],
                    });
                }

                self.instances.push(best.expect("CELL_RETRIES > 0 garantiert mind. einen Fallback-Kandidaten"));
            }
        }

        Some(&self.instances)
    }
}
