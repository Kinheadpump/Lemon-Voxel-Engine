use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::game::math::cascades::MAX_SHADOW_CASCADES;

pub const CONFIG_PATH: &str = "config.toml";

#[derive(Clone, Copy, Debug)]
pub struct EngineConfig {
    pub movement_speed: f32,
    pub sprint_multiplier: f32,
    pub mouse_sensitivity: f32,
    pub fov_y_radians: f32,
    pub render_distance_chunks: i32,
    /// Vertikale Ladedistanz in Chunks, zentriert auf die AKTUELLE Chunk-Y-Position der Kamera
    /// (nicht auf einen festen Weltboden). Es gibt dadurch keine harte Bau-/Grabgrenze mehr - das
    /// Ladefenster wandert einfach mit, egal wie hoch/tief man baut oder graebt. Absichtlich von
    /// `render_distance_chunks` entkoppelt, weil man vertikal selten so weit sehen muss wie
    /// horizontal (ein grosses render_distance_chunks soll nicht automatisch den Chunk-Pool und
    /// die Pro-Frame-Sichtbarkeitspruefung in Y explodieren lassen).
    pub vertical_render_distance_chunks: i32,
    pub clear_color: wgpu::Color,
    pub hud_visible_default: bool,
    pub msaa_samples: u32,
    pub ssao_enabled: bool,
    pub ssao_radius: f32,
    pub ssao_strength: f32,
    /// Bilateral-Blur-Kantenschwelle (NDC-Tiefendifferenz) fuer den SSAO-Denoise-Pass: Nachbar-
    /// Texel mit groesserer Tiefendifferenz gelten als andere Oberflaeche und fliessen nicht in den
    /// Blur ein - verhindert Ueberblenden ueber Geometriekanten hinweg.
    pub ssao_blur_depth_threshold: f32,
    pub gravity: f32,
    pub jump_speed: f32,
    /// Maximale Fallgeschwindigkeit (Betrag, Bloecke/s). In der vertikal unbegrenzten Welt wuerde
    /// die Geschwindigkeit sonst bei tiefen Faellen unbegrenzt wachsen - der Lande-Frame muesste
    /// dann einen entsprechend riesigen Sweep-Kollisions-Scan abarbeiten (Frame-Drop bei Aufprall).
    pub terminal_velocity: f32,
    pub start_flying: bool,

    /// Reale Sekunden fuer einen vollen Tag/Nacht-Zyklus (Sonnenwinkel 0..2*PI).
    pub sun_cycle_seconds: f32,
    /// Startpunkt im Zyklus, 0.0 = Sonnenaufgang, 0.25 = Zenit, 0.5 = Sonnenuntergang.
    pub sun_initial_time_of_day: f32,
    pub ambient_light: f32,
    pub sun_intensity: f32,

    /// 3-4 Kaskaden: mehr Kaskaden = feinere Aufloesungs-Staffelung nahe der Kamera, aber ein
    /// zusaetzlicher Shadow-Pass-Durchlauf pro Kaskade.
    pub shadow_cascade_count: u32,
    pub shadow_map_resolution: u32,
    /// Distanz ab der Kamera, bis zu der ueberhaupt Schatten berechnet werden - unabhaengig von der
    /// (potenziell unendlichen) Reverse-Z-Fernsicht der Hauptkamera.
    pub shadow_max_distance: f32,
    /// Mischung zwischen logarithmischer und linearer Kaskaden-Aufteilung (0 = linear, 1 = log).
    /// Log gewichtet mehr Aufloesung nahe der Kamera, was fuer Voxel-Kanten am wichtigsten ist.
    pub shadow_split_lambda: f32,
    pub shadow_depth_bias: f32,
    pub shadow_depth_bias_slope_scale: f32,

    pub sky_zenith_day_color: [f32; 3],
    pub sky_horizon_day_color: [f32; 3],
    pub sky_night_color: [f32; 3],

    /// Maximale Anzahl gleichzeitiger Godray-Billboards (SSBO-Kapazitaet, feste Groesse).
    pub godray_count: u32,
    /// Gitterabstand der Godray-Kandidaten-Positionen in Weltblöcken.
    pub godray_grid_spacing: f32,
    /// Hoehe des Kantenerkennungs-Punkts ueber der Terrainoberflaeche - bewusst klein (nah an der
    /// Oberflaeche), damit die Sample-Kugel tatsaechlich mit benachbarten Voxel-Hoehenunterschieden
    /// (Bergkaemme, Hoehleneingaenge) interagiert statt frei in der Luft zu schweben, wo es fast nie
    /// einen Licht/Schatten-Uebergang gibt.
    pub godray_sample_height: f32,
    /// Sample-Radius der Kantenerkennung UND sichtbare Billboard-Breite, in Weltblöcken.
    pub godray_width: f32,
    /// Sichtbare Strahllaenge entlang der tatsaechlichen Lichtrichtung (nicht mehr fix vertikal) -
    /// dadurch zeigen die Strahlen wirklich zur Sonne statt immer im selben Winkel zu stehen.
    pub godray_beam_length: f32,
    /// Mischfaktor pro Frame zwischen alter und neu berechneter Intensity (0 = einfriert, 1 = kein
    /// Glaetten). Klein halten, sonst flackert es bei Kamerabewegung trotz Temporal-Blend.
    pub godray_temporal_blend: f32,

    pub terrain_seed: u32,
    pub terrain_continental_frequency: f32,
    pub terrain_continental_amplitude: f32,
    /// Amplitude des Berg-Boosts: `unorm(continental)^exponent * amplitude` - der Exponent (>1)
    /// laesst Ebenen (kleine unorm-Werte) flach und nur die Kontinentalmaxima massiv hochschiessen.
    pub terrain_mountain_amplitude: f32,
    pub terrain_mountain_exponent: f32,
    pub terrain_regional_frequency: f32,
    pub terrain_regional_amplitude: f32,
    /// fBm-Octave-Anzahl der Regional-Heightmap (4-5 fuer echtes Relief statt einer einzelnen,
    /// glatten Perlin-Frequenz).
    pub terrain_regional_octaves: u32,
    /// Frequenz-Multiplikator pro fBm-Octave.
    pub terrain_regional_lacunarity: f32,
    /// Amplituden-Abfall pro fBm-Octave (Persistence/Gain).
    pub terrain_regional_gain: f32,
    pub terrain_cliff_mask_frequency: f32,
    /// Sehr niedrigfrequente Biom-Karten (Features ueber viele hundert Bloecke) - strikte
    /// Schwellwerte darauf ergeben grosse, zusammenhaengende Biome ohne Einzelblock-Bleeding.
    pub terrain_temperature_frequency: f32,
    pub terrain_humidity_frequency: f32,
    /// Wueste NUR wenn Temperatur > min UND Feuchtigkeit < max (snorm -1..1) - striktes 2D-Mapping.
    pub terrain_desert_temperature_min: f32,
    pub terrain_desert_humidity_max: f32,
    pub terrain_sea_compression_range: f32,
    pub terrain_sea_compression_exponent: f32,
    /// Niedrige Frequenz fuer grosse, ausgedehnte Cheese Caves statt kleiner Blasen.
    pub terrain_cheese_frequency: f32,
    /// Perlin-Werte oberhalb dieser Schwelle (Bereich -1..1) werden zu Cheese-Cave-Hohlraum.
    pub terrain_cheese_threshold: f32,
    pub terrain_tunnel_frequency: f32,
    /// `WorleyDifference`-Werte (unorm, ueblicherweise klein) unterhalb dieser Schwelle werden zu
    /// Tunnel - s. `calibrate_cave_thresholds` zum empirischen Kalibrieren.
    pub terrain_tunnel_threshold: f32,
    /// Bloecke unterhalb `SEA_LEVEL`, ueber die BEIDE Hoehlensysteme ihre maximale Verbreiterung
    /// erreichen - ein gemeinsamer Tiefenfaktor fuer Cheese Caves UND Tunnel.
    pub terrain_cave_widen_depth_range: f32,
    /// Um wie viel `terrain_cheese_threshold` in maximaler Tiefe sinkt (groessere Kavernen).
    pub terrain_cheese_widen_amount: f32,
    /// Faktor, um den `terrain_tunnel_threshold` in maximaler Tiefe multipliziert wird (breitere
    /// Roehren).
    pub terrain_tunnel_widen_multiplier: f32,
    /// Frequenz des 2D-Gates, das entscheidet, ob eine Region ueberhaupt Tunnel bekommt.
    pub terrain_cave_region_frequency: f32,
    /// Perlin-Werte oberhalb dieser Schwelle (Bereich -1..1) gelten als "Hoehlen-aktive" Region.
    pub terrain_cave_region_threshold: f32,
    pub terrain_dirt_layer_depth: i32,
    pub terrain_noise_origin_offset: f32,

    pub player_half_width: f32,
    pub player_height: f32,
    pub player_eye_height: f32,
    pub ground_probe_distance: f32,
    pub fixed_timestep: f32,
    pub max_physics_steps_per_frame: u32,

    pub chunk_pool_size: usize,
    pub max_faces_per_direction: usize,
    pub max_draws_per_direction: usize,

    /// Obergrenze, wie viele Chunks pro Frame vom Rayon-Pool dispatcht bzw. wie viele fertige
    /// Generierungs-Ergebnisse pro Frame in GPU-Uploads uebersetzt werden. Ohne diese Grenze
    /// versucht der Main-Thread bei grossem `render_distance_chunks` (grosser Backlog beim
    /// Welt-Start oder schnellem Fliegen), tausende Chunks in einem einzigen Frame zu dispatchen/
    /// hochzuladen - das erzeugt Mehrsekunden-Freezes statt verteilter Frame-Zeit.
    pub max_chunk_dispatches_per_frame: usize,
    pub max_chunk_uploads_per_frame: usize,
    /// Obergrenze fuer Chunk-Entladungen pro Frame. Beim Ueberqueren einer Chunk-Grenze (v.a. der
    /// vertikalen beim Fallen/Landen) wandert eine ganze Chunk-Ebene aus dem Ladefenster - ohne
    /// Deckel wuerden alle betroffenen Chunks (Tausende) in einem einzigen Frame entladen, jede
    /// `free_chunk`-Freigabe ist dabei O(Freelist). Das war die Ursache der Ruckler beim Landen.
    pub max_chunk_unloads_per_frame: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            movement_speed: 12.0,
            sprint_multiplier: 4.0,
            mouse_sensitivity: 0.0025,
            fov_y_radians: 60f32.to_radians(),
            render_distance_chunks: 4,
            vertical_render_distance_chunks: 4,
            clear_color: wgpu::Color { r: 0.02, g: 0.02, b: 0.02, a: 1.0 },
            hud_visible_default: true,
            msaa_samples: 4,
            ssao_enabled: true,
            ssao_radius: 2.0,
            ssao_strength: 1.4,
            ssao_blur_depth_threshold: 0.0008,
            gravity: 26.0,
            jump_speed: 9.0,
            terminal_velocity: 80.0,
            start_flying: true,

            sun_cycle_seconds: 1200.0,
            sun_initial_time_of_day: 0.28,
            ambient_light: 0.2,
            sun_intensity: 1.0,

            shadow_cascade_count: 4,
            shadow_map_resolution: 2048,
            shadow_max_distance: 220.0,
            shadow_split_lambda: 0.6,
            shadow_depth_bias: 2.0,
            shadow_depth_bias_slope_scale: 2.0,

            sky_zenith_day_color: [0.21, 0.47, 0.81],
            sky_horizon_day_color: [0.64, 0.72, 0.81],
            sky_night_color: [0.01, 0.015, 0.03],

            godray_count: 512,
            godray_grid_spacing: 6.0,
            godray_sample_height: 1.5,
            godray_width: 3.5,
            godray_beam_length: 12.0,
            godray_temporal_blend: 0.12,

            terrain_seed: 1337,
            // Kontinental-Skala ~625 Bloecke: Land/Ozean-Wechsel innerhalb einer explorierbaren
            // Distanz (fruehere ~2650-Block-Periode war lokal unsichtbar -> alles wirkte flach).
            terrain_continental_frequency: 0.0016,
            terrain_continental_amplitude: 55.0,
            // unorm(cont)^5.5 * 130: hoher Exponent konzentriert die Berge auf die OBERSTEN
            // Kontinental-Werte - das meiste Land bleibt sanftes Huegelland, nur die Kontinentalkerne
            // tuermen sich zu (dann umso markanteren) Massiven auf. Kleiner Exponent liess frueher ein
            // Viertel der Welt ueber Hoehe 80 liegen ("ueberall Berge").
            terrain_mountain_amplitude: 130.0,
            terrain_mountain_exponent: 5.5,
            // Regional-Skala ~180 Bloecke, 4 Octaves: sanfte Huegel/Taeler statt hochfrequenter
            // Zacken (frueher Periode 83 + 5 Octaves -> kleinteiliges Chaos auf der Sandebene).
            terrain_regional_frequency: 0.0055,
            terrain_regional_amplitude: 22.0,
            terrain_regional_octaves: 4,
            terrain_regional_lacunarity: 2.0,
            terrain_regional_gain: 0.45,
            terrain_cliff_mask_frequency: 0.008,
            // Biom-Features ~650 Bloecke - grosse zusammenhaengende Wuesten/Graslaender.
            terrain_temperature_frequency: 0.0015,
            terrain_humidity_frequency: 0.0017,
            terrain_desert_temperature_min: 0.25,
            terrain_desert_humidity_max: -0.05,
            // SANFT und SCHMAL: glaettet nur die unmittelbare Wasserlinie (±6 Bloecke, Exp 1.3),
            // statt wie zuvor (±20, Exp 2.2) das halbe Relief auf Meereshoehe zu quetschen und damit
            // eine riesige flache Sandebene zu erzeugen.
            terrain_sea_compression_range: 6.0,
            terrain_sea_compression_exponent: 1.3,
            // Niedrige Frequenz (~50 Bloecke/Feature, gegenueber vorher 20) fuer grosse, ausgedehnte
            // Kavernen statt kleiner Blasen. Schwelle empirisch kalibriert, s.
            // `calibrate_cave_thresholds`.
            terrain_cheese_frequency: 0.02,
            terrain_cheese_threshold: 0.62,
            // Feature-Groesse ~35 Bloecke. `WorleyDifference` ist NICHT uniform verteilt (kleine
            // Werte = nahe Zellgrenze sind selten, Median liegt bei ~0.046!) - Schwelle empirisch
            // kalibriert, s. `calibrate_cave_thresholds` (p1=0.0007, p2=0.0015, p5=0.0037), NICHT
            // geraten. 0.0012 (~p1.5) haelt Tunnel duenn statt den kompletten Zellraum auszuhoehlen.
            terrain_tunnel_frequency: 0.028,
            terrain_tunnel_threshold: 0.0012,
            // Gemeinsamer Tiefenfaktor fuer beide Systeme - ab hier (Bloecke unter Meeresspiegel)
            // volle Verbreiterung.
            terrain_cave_widen_depth_range: 150.0,
            terrain_cheese_widen_amount: 0.12,
            terrain_tunnel_widen_multiplier: 1.5,
            // Grosse Regionen (~500 Bloecke/Feature). Schwelle 0.3 haelt Tunnelnetze regional
            // konzentriert (Karstgebiet-Charakter) statt ueberall gleich dicht UND spart die teure
            // Worley-Auswertung im Rest des Untergrunds komplett.
            terrain_cave_region_frequency: 0.002,
            terrain_cave_region_threshold: 0.3,
            terrain_dirt_layer_depth: 3,
            terrain_noise_origin_offset: 10_000.0,

            player_half_width: 0.3,
            player_height: 1.8,
            player_eye_height: 1.6,
            ground_probe_distance: 0.1,
            fixed_timestep: 1.0 / 60.0,
            max_physics_steps_per_frame: 8,

            // Muss mind. (2*render_distance_chunks+1)^2 * (2*vertical_render_distance_chunks+1)
            // abdecken, sonst werden Chunks am Rand der Render-Distanz stillschweigend nicht
            // geladen (Pool erschoepft).
            chunk_pool_size: 800,
            max_faces_per_direction: 3_000_000,
            max_draws_per_direction: 4300,

            // Vor dem Binary-Greedy-Meshing-Umbau war das Meshing selbst der Flaschenhals; jetzt
            // ist der Upload-/Dispatch-Takt (64/Frame) die haertere Bremse (bei ~18 FPS waehrend
            // des Ladens ergab das rechnerisch exakt die beobachtete ~1100-1300 Chunks/s
            // Laderate). Verdoppelt, weil Upload/Dispatch selbst billig sind (paar Buffer-Writes
            // bzw. ein Rayon-Spawn) - das eigentliche Meshing laeuft ohnehin asynchron auf
            // Worker-Threads und begrenzt hier nicht mehr.
            max_chunk_dispatches_per_frame: 128,
            max_chunk_uploads_per_frame: 128,
            max_chunk_unloads_per_frame: 192,
        }
    }
}

/// Sicherheitsobergrenze fuer den aus der Render-Distanz abgeleiteten Chunk-Pool - verhindert, dass
/// eine extreme Kombination aus horizontaler UND vertikaler Render-Distanz beim Start unbemerkt
/// mehrere GB RAM alloziert. 65536 Chunks * 64 KiB = 4 GiB.
pub const CHUNK_POOL_SAFETY_CAP: usize = 65_536;

/// `(2*render_distance+1)^2 * (2*vertical_render_distance+1)` - die Anzahl Chunks, die gleichzeitig
/// innerhalb des Ladefensters liegen koennen.
fn required_chunk_pool_size(render_distance_chunks: i32, vertical_render_distance_chunks: i32) -> usize {
    let horizontal_span = 2 * render_distance_chunks as i64 + 1;
    let vertical_span = 2 * vertical_render_distance_chunks as i64 + 1;
    (horizontal_span * horizontal_span * vertical_span) as usize
}

impl EngineConfig {
    /// Leitet die voneinander abhaengigen Kapazitaeten EINMAL zentral ab - ALLE Konsumenten
    /// (`ChunkManager`-Pool, `ChunkRenderer`-Buffer wie `chunk_meta_buffer`/Indirect/ChunkData,
    /// Cull-Dispatch-Grenze) lesen danach dieselben normalisierten Werte. Vorher skalierte der
    /// `ChunkManager` seinen Pool intern hoch, waehrend die Renderer-Buffer auf dem rohen
    /// Config-Wert blieben - `update_chunk_meta` schrieb dann bei hoher Render-Distanz hinter das
    /// Buffer-Ende (Pool-Slot-Index >= Buffer-Kapazitaet).
    ///
    /// - `chunk_pool_size`: mindestens das Ladevolumen der Render-Distanz (Config-Wert ist
    ///   Untergrenze, nie Obergrenze), gedeckelt auf `CHUNK_POOL_SAFETY_CAP`.
    /// - `max_draws_per_direction`: mindestens `chunk_pool_size` - der GPU-Cull-Pass kompaktiert
    ///   bis zu ALLE Pool-Slots in die Indirect-Buffer; ein kleinerer Wert liess bei hoher
    ///   Render-Distanz sichtbare Chunks kommentarlos aus dem Draw fallen ("kuenstlich limitierte
    ///   Sichtweite"). Kostet 2*16 Byte pro Slot ueber 6 Richtungen - vernachlaessigbar.
    fn normalized(mut self) -> Self {
        let required = required_chunk_pool_size(self.render_distance_chunks, self.vertical_render_distance_chunks);
        if required > CHUNK_POOL_SAFETY_CAP {
            log::warn!(
                "Render-Distanz {}x{} braeuchte {} Chunk-Pool-Slots, gedeckelt auf {} ({} GiB RAM) - \
                 Chunks am Rand der Render-Distanz werden nicht geladen",
                self.render_distance_chunks,
                self.vertical_render_distance_chunks,
                required,
                CHUNK_POOL_SAFETY_CAP,
                CHUNK_POOL_SAFETY_CAP * 64 / 1024 / 1024,
            );
        }
        self.chunk_pool_size = self.chunk_pool_size.max(required.min(CHUNK_POOL_SAFETY_CAP));
        self.max_draws_per_direction = self.max_draws_per_direction.max(self.chunk_pool_size);
        self
    }

    /// Laedt die Konfiguration aus `config.toml`. Existiert die Datei nicht, wird sie mit den
    /// Default-Werten erzeugt, damit der Nutzer eine editierbare Vorlage erhaelt. Bei Parse-Fehlern
    /// wird geloggt und auf Defaults zurueckgefallen, statt abzustuerzen.
    pub fn load_or_create(path: impl AsRef<Path>) -> Self {
        let path = path.as_ref();

        match std::fs::read_to_string(path) {
            Ok(contents) => match toml::from_str::<ConfigFile>(&contents) {
                Ok(file) => {
                    log::info!("Konfiguration aus {} geladen", path.display());
                    file.into()
                }
                Err(error) => {
                    log::error!("Konfiguration {} fehlerhaft ({error}) - nutze Defaults", path.display());
                    Self::default().normalized()
                }
            },
            Err(_) => {
                let default = Self::default();
                let file = ConfigFile::from(default);
                match toml::to_string_pretty(&file) {
                    Ok(serialized) => {
                        if let Err(error) = std::fs::write(path, serialized) {
                            log::warn!("Konnte {} nicht schreiben: {error}", path.display());
                        } else {
                            log::info!("Standard-Konfiguration nach {} geschrieben", path.display());
                        }
                    }
                    Err(error) => log::warn!("Konnte Konfiguration nicht serialisieren: {error}"),
                }
                default.normalized()
            }
        }
    }
}

/// Serde-serialisierbares Spiegelbild von [`EngineConfig`] mit editorfreundlichen Einheiten
/// (FOV in Grad, Farbe als RGB-Array). Trennt das Datei-Format von der Laufzeit-Repraesentation.
///
/// BEWUSST ohne `deny_unknown_fields`: Ein einzelnes umbenanntes/entferntes Feld (z.B. bei einem
/// Terrain-Schema-Wechsel) wuerde sonst den GESAMTEN Parse-Vorgang abbrechen und `load_or_create`
/// faellt dann auf komplette Defaults zurueck - das setzt still ALLE anderen, unveraendert
/// gebliebenen Einstellungen (Render-Distanz, Maus-Sensitivitaet, ...) zurueck, nicht nur die
/// tatsaechlich verschobenen Felder. Unbekannte Felder in einer alten `config.toml` werden jetzt
/// einfach ignoriert, alle anderen Felder bleiben erhalten.
#[derive(Serialize, Deserialize)]
#[serde(default)]
struct ConfigFile {
    movement_speed: f32,
    sprint_multiplier: f32,
    mouse_sensitivity: f32,
    fov_degrees: f32,
    render_distance_chunks: i32,
    vertical_render_distance_chunks: i32,
    clear_color_rgb: [f64; 3],
    hud_visible_default: bool,
    msaa_samples: u32,
    ssao_enabled: bool,
    ssao_radius: f32,
    ssao_strength: f32,
    ssao_blur_depth_threshold: f32,
    gravity: f32,
    jump_speed: f32,
    terminal_velocity: f32,
    start_flying: bool,

    sun_cycle_seconds: f32,
    sun_initial_time_of_day: f32,
    ambient_light: f32,
    sun_intensity: f32,

    shadow_cascade_count: u32,
    shadow_map_resolution: u32,
    shadow_max_distance: f32,
    shadow_split_lambda: f32,
    shadow_depth_bias: f32,
    shadow_depth_bias_slope_scale: f32,

    sky_zenith_day_color: [f32; 3],
    sky_horizon_day_color: [f32; 3],
    sky_night_color: [f32; 3],

    godray_count: u32,
    godray_grid_spacing: f32,
    godray_sample_height: f32,
    godray_width: f32,
    godray_beam_length: f32,
    godray_temporal_blend: f32,

    terrain_seed: u32,
    terrain_continental_frequency: f32,
    terrain_continental_amplitude: f32,
    terrain_mountain_amplitude: f32,
    terrain_mountain_exponent: f32,
    terrain_regional_frequency: f32,
    terrain_regional_amplitude: f32,
    terrain_regional_octaves: u32,
    terrain_regional_lacunarity: f32,
    terrain_regional_gain: f32,
    terrain_cliff_mask_frequency: f32,
    terrain_temperature_frequency: f32,
    terrain_humidity_frequency: f32,
    terrain_desert_temperature_min: f32,
    terrain_desert_humidity_max: f32,
    terrain_sea_compression_range: f32,
    terrain_sea_compression_exponent: f32,
    terrain_cheese_frequency: f32,
    terrain_cheese_threshold: f32,
    terrain_tunnel_frequency: f32,
    terrain_tunnel_threshold: f32,
    terrain_cave_widen_depth_range: f32,
    terrain_cheese_widen_amount: f32,
    terrain_tunnel_widen_multiplier: f32,
    terrain_cave_region_frequency: f32,
    terrain_cave_region_threshold: f32,
    terrain_dirt_layer_depth: i32,
    terrain_noise_origin_offset: f32,

    player_half_width: f32,
    player_height: f32,
    player_eye_height: f32,
    ground_probe_distance: f32,
    fixed_timestep: f32,
    max_physics_steps_per_frame: u32,

    chunk_pool_size: usize,
    max_faces_per_direction: usize,
    max_draws_per_direction: usize,

    /// Obergrenze, wie viele Chunks pro Frame vom Rayon-Pool dispatcht bzw. wie viele fertige
    /// Generierungs-Ergebnisse pro Frame in GPU-Uploads uebersetzt werden. Ohne diese Grenze
    /// versucht der Main-Thread bei grossem `render_distance_chunks` (grosser Backlog beim
    /// Welt-Start oder schnellem Fliegen), tausende Chunks in einem einzigen Frame zu dispatchen/
    /// hochzuladen - das erzeugt Mehrsekunden-Freezes statt verteilter Frame-Zeit.
    max_chunk_dispatches_per_frame: usize,
    max_chunk_uploads_per_frame: usize,
    max_chunk_unloads_per_frame: usize,
}

impl Default for ConfigFile {
    fn default() -> Self {
        Self::from(EngineConfig::default())
    }
}

impl From<EngineConfig> for ConfigFile {
    fn from(c: EngineConfig) -> Self {
        Self {
            movement_speed: c.movement_speed,
            sprint_multiplier: c.sprint_multiplier,
            mouse_sensitivity: c.mouse_sensitivity,
            fov_degrees: c.fov_y_radians.to_degrees(),
            render_distance_chunks: c.render_distance_chunks,
            vertical_render_distance_chunks: c.vertical_render_distance_chunks,
            clear_color_rgb: [c.clear_color.r, c.clear_color.g, c.clear_color.b],
            hud_visible_default: c.hud_visible_default,
            msaa_samples: c.msaa_samples,
            ssao_enabled: c.ssao_enabled,
            ssao_radius: c.ssao_radius,
            ssao_strength: c.ssao_strength,
            ssao_blur_depth_threshold: c.ssao_blur_depth_threshold,
            gravity: c.gravity,
            jump_speed: c.jump_speed,
            terminal_velocity: c.terminal_velocity,
            start_flying: c.start_flying,

            sun_cycle_seconds: c.sun_cycle_seconds,
            sun_initial_time_of_day: c.sun_initial_time_of_day,
            ambient_light: c.ambient_light,
            sun_intensity: c.sun_intensity,

            shadow_cascade_count: c.shadow_cascade_count,
            shadow_map_resolution: c.shadow_map_resolution,
            shadow_max_distance: c.shadow_max_distance,
            shadow_split_lambda: c.shadow_split_lambda,
            shadow_depth_bias: c.shadow_depth_bias,
            shadow_depth_bias_slope_scale: c.shadow_depth_bias_slope_scale,

            sky_zenith_day_color: c.sky_zenith_day_color,
            sky_horizon_day_color: c.sky_horizon_day_color,
            sky_night_color: c.sky_night_color,

            godray_count: c.godray_count,
            godray_grid_spacing: c.godray_grid_spacing,
            godray_sample_height: c.godray_sample_height,
            godray_width: c.godray_width,
            godray_beam_length: c.godray_beam_length,
            godray_temporal_blend: c.godray_temporal_blend,

            terrain_seed: c.terrain_seed,
            terrain_continental_frequency: c.terrain_continental_frequency,
            terrain_continental_amplitude: c.terrain_continental_amplitude,
            terrain_mountain_amplitude: c.terrain_mountain_amplitude,
            terrain_mountain_exponent: c.terrain_mountain_exponent,
            terrain_regional_frequency: c.terrain_regional_frequency,
            terrain_regional_amplitude: c.terrain_regional_amplitude,
            terrain_regional_octaves: c.terrain_regional_octaves,
            terrain_regional_lacunarity: c.terrain_regional_lacunarity,
            terrain_regional_gain: c.terrain_regional_gain,
            terrain_cliff_mask_frequency: c.terrain_cliff_mask_frequency,
            terrain_temperature_frequency: c.terrain_temperature_frequency,
            terrain_humidity_frequency: c.terrain_humidity_frequency,
            terrain_desert_temperature_min: c.terrain_desert_temperature_min,
            terrain_desert_humidity_max: c.terrain_desert_humidity_max,
            terrain_sea_compression_range: c.terrain_sea_compression_range,
            terrain_sea_compression_exponent: c.terrain_sea_compression_exponent,
            terrain_cheese_frequency: c.terrain_cheese_frequency,
            terrain_cheese_threshold: c.terrain_cheese_threshold,
            terrain_tunnel_frequency: c.terrain_tunnel_frequency,
            terrain_tunnel_threshold: c.terrain_tunnel_threshold,
            terrain_cave_widen_depth_range: c.terrain_cave_widen_depth_range,
            terrain_cheese_widen_amount: c.terrain_cheese_widen_amount,
            terrain_tunnel_widen_multiplier: c.terrain_tunnel_widen_multiplier,
            terrain_cave_region_frequency: c.terrain_cave_region_frequency,
            terrain_cave_region_threshold: c.terrain_cave_region_threshold,
            terrain_dirt_layer_depth: c.terrain_dirt_layer_depth,
            terrain_noise_origin_offset: c.terrain_noise_origin_offset,

            player_half_width: c.player_half_width,
            player_height: c.player_height,
            player_eye_height: c.player_eye_height,
            ground_probe_distance: c.ground_probe_distance,
            fixed_timestep: c.fixed_timestep,
            max_physics_steps_per_frame: c.max_physics_steps_per_frame,

            chunk_pool_size: c.chunk_pool_size,
            max_faces_per_direction: c.max_faces_per_direction,
            max_draws_per_direction: c.max_draws_per_direction,

            max_chunk_dispatches_per_frame: c.max_chunk_dispatches_per_frame,
            max_chunk_uploads_per_frame: c.max_chunk_uploads_per_frame,
            max_chunk_unloads_per_frame: c.max_chunk_unloads_per_frame,
        }
    }
}

impl From<ConfigFile> for EngineConfig {
    fn from(f: ConfigFile) -> Self {
        Self {
            movement_speed: f.movement_speed,
            sprint_multiplier: f.sprint_multiplier,
            mouse_sensitivity: f.mouse_sensitivity,
            fov_y_radians: f.fov_degrees.to_radians(),
            render_distance_chunks: f.render_distance_chunks.clamp(1, 32),
            vertical_render_distance_chunks: f.vertical_render_distance_chunks.clamp(1, 32),
            clear_color: wgpu::Color {
                r: f.clear_color_rgb[0],
                g: f.clear_color_rgb[1],
                b: f.clear_color_rgb[2],
                a: 1.0,
            },
            hud_visible_default: f.hud_visible_default,
            msaa_samples: f.msaa_samples.clamp(1, 8),
            ssao_enabled: f.ssao_enabled,
            ssao_radius: f.ssao_radius,
            ssao_strength: f.ssao_strength,
            ssao_blur_depth_threshold: f.ssao_blur_depth_threshold.max(0.0),
            gravity: f.gravity,
            jump_speed: f.jump_speed,
            terminal_velocity: f.terminal_velocity.max(1.0),
            start_flying: f.start_flying,

            sun_cycle_seconds: f.sun_cycle_seconds.max(1.0),
            sun_initial_time_of_day: f.sun_initial_time_of_day.rem_euclid(1.0),
            ambient_light: f.ambient_light.clamp(0.0, 1.0),
            sun_intensity: f.sun_intensity.max(0.0),

            shadow_cascade_count: f.shadow_cascade_count.clamp(3, MAX_SHADOW_CASCADES as u32),
            shadow_map_resolution: f.shadow_map_resolution.clamp(256, 8192),
            shadow_max_distance: f.shadow_max_distance.max(16.0),
            shadow_split_lambda: f.shadow_split_lambda.clamp(0.0, 1.0),
            shadow_depth_bias: f.shadow_depth_bias,
            shadow_depth_bias_slope_scale: f.shadow_depth_bias_slope_scale,

            sky_zenith_day_color: f.sky_zenith_day_color,
            sky_horizon_day_color: f.sky_horizon_day_color,
            sky_night_color: f.sky_night_color,

            godray_count: f.godray_count.clamp(1, 8192),
            godray_grid_spacing: f.godray_grid_spacing.max(0.5),
            godray_sample_height: f.godray_sample_height,
            godray_width: f.godray_width.max(0.01),
            godray_beam_length: f.godray_beam_length.max(0.1),
            godray_temporal_blend: f.godray_temporal_blend.clamp(0.001, 1.0),

            terrain_seed: f.terrain_seed,
            terrain_continental_frequency: f.terrain_continental_frequency,
            terrain_continental_amplitude: f.terrain_continental_amplitude,
            terrain_mountain_amplitude: f.terrain_mountain_amplitude.max(0.0),
            terrain_mountain_exponent: f.terrain_mountain_exponent.max(1.0),
            terrain_regional_frequency: f.terrain_regional_frequency,
            terrain_regional_amplitude: f.terrain_regional_amplitude,
            terrain_regional_octaves: f.terrain_regional_octaves.clamp(1, 8),
            terrain_regional_lacunarity: f.terrain_regional_lacunarity.max(1.0),
            terrain_regional_gain: f.terrain_regional_gain.clamp(0.0, 1.0),
            terrain_cliff_mask_frequency: f.terrain_cliff_mask_frequency,
            terrain_temperature_frequency: f.terrain_temperature_frequency,
            terrain_humidity_frequency: f.terrain_humidity_frequency,
            terrain_desert_temperature_min: f.terrain_desert_temperature_min,
            terrain_desert_humidity_max: f.terrain_desert_humidity_max,
            terrain_sea_compression_range: f.terrain_sea_compression_range.max(1.0),
            terrain_sea_compression_exponent: f.terrain_sea_compression_exponent.max(1.0),
            terrain_cheese_frequency: f.terrain_cheese_frequency,
            terrain_cheese_threshold: f.terrain_cheese_threshold,
            terrain_tunnel_frequency: f.terrain_tunnel_frequency,
            terrain_tunnel_threshold: f.terrain_tunnel_threshold,
            terrain_cave_widen_depth_range: f.terrain_cave_widen_depth_range.max(1.0),
            terrain_cheese_widen_amount: f.terrain_cheese_widen_amount.max(0.0),
            terrain_tunnel_widen_multiplier: f.terrain_tunnel_widen_multiplier.max(0.0),
            terrain_cave_region_frequency: f.terrain_cave_region_frequency,
            terrain_cave_region_threshold: f.terrain_cave_region_threshold,
            terrain_dirt_layer_depth: f.terrain_dirt_layer_depth,
            terrain_noise_origin_offset: f.terrain_noise_origin_offset,

            player_half_width: f.player_half_width,
            player_height: f.player_height,
            player_eye_height: f.player_eye_height,
            ground_probe_distance: f.ground_probe_distance,
            fixed_timestep: f.fixed_timestep.max(1.0 / 480.0),
            max_physics_steps_per_frame: f.max_physics_steps_per_frame.max(1),

            chunk_pool_size: f.chunk_pool_size.max(1),
            max_faces_per_direction: f.max_faces_per_direction.max(1),
            max_draws_per_direction: f.max_draws_per_direction.max(1),

            max_chunk_dispatches_per_frame: f.max_chunk_dispatches_per_frame.max(1),
            max_chunk_uploads_per_frame: f.max_chunk_uploads_per_frame.max(1),
            max_chunk_unloads_per_frame: f.max_chunk_unloads_per_frame.max(1),
        }
        .normalized()
    }
}
