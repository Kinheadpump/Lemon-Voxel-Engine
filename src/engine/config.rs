use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::game::math::cascades::MAX_SHADOW_CASCADES;

pub const CONFIG_PATH: &str = "config.toml";

/// Einstellungen, die ein SPIELER ueber ein Optionsmenue anpassen wuerde: Eingabe-Empfindlichkeit,
/// Sichtfeld, Sicht-/Grafikqualitaet. Strikt getrennt von `DevSettings` - ein zukuenftiges
/// In-Game-Optionsmenue muss nur DIESE Struct anfassen koennen, ohne versehentlich an Terrain-Seeds
/// oder Performance-Budgets zu ruehren. `config.toml` spiegelt die Trennung als `[player]`-Tabelle.
#[derive(Clone, Copy, Debug)]
pub struct PlayerSettings {
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
    pub hud_visible_default: bool,
    pub msaa_samples: u32,
    pub ssao_enabled: bool,
    pub ssao_radius: f32,
    pub ssao_strength: f32,
    /// Bilateral-Blur-Kantenschwelle (NDC-Tiefendifferenz) fuer den SSAO-Denoise-Pass: Nachbar-
    /// Texel mit groesserer Tiefendifferenz gelten als andere Oberflaeche und fliessen nicht in den
    /// Blur ein - verhindert Ueberblenden ueber Geometriekanten hinweg.
    pub ssao_blur_depth_threshold: f32,
    /// 3-4 Kaskaden: mehr Kaskaden = feinere Aufloesungs-Staffelung nahe der Kamera, aber ein
    /// zusaetzlicher Shadow-Pass-Durchlauf pro Kaskade.
    pub shadow_cascade_count: u32,
    pub shadow_map_resolution: u32,
    /// Distanz ab der Kamera, bis zu der ueberhaupt Schatten berechnet werden - unabhaengig von der
    /// (potenziell unendlichen) Reverse-Z-Fernsicht der Hauptkamera.
    pub shadow_max_distance: f32,
    pub start_flying: bool,
}

/// Alles andere: Welt-Generierung, Physik-/Rendering-Internals, Performance-Budgets, Art Direction -
/// Stellschrauben fuer den ENTWICKLER, nicht fuer den Spieler. `config.toml` spiegelt diese Trennung
/// als `[dev]`-Tabelle wider.
#[derive(Clone, Copy, Debug)]
pub struct DevSettings {
    pub clear_color: wgpu::Color,
    pub gravity: f32,
    pub jump_speed: f32,
    /// Maximale Fallgeschwindigkeit (Betrag, Bloecke/s). In der vertikal unbegrenzten Welt wuerde
    /// die Geschwindigkeit sonst bei tiefen Faellen unbegrenzt wachsen - der Lande-Frame muesste
    /// dann einen entsprechend riesigen Sweep-Kollisions-Scan abarbeiten (Frame-Drop bei Aufprall).
    pub terminal_velocity: f32,

    /// Reale Sekunden fuer einen vollen Tag/Nacht-Zyklus (Sonnenwinkel 0..2*PI).
    pub sun_cycle_seconds: f32,
    /// Startpunkt im Zyklus, 0.0 = Sonnenaufgang, 0.25 = Zenit, 0.5 = Sonnenuntergang.
    pub sun_initial_time_of_day: f32,
    pub ambient_light: f32,
    pub sun_intensity: f32,

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
    /// Zweites, unabhaengiges Worley-Tunnelsystem (andere Frequenz/Seed als `terrain_tunnel_*`) -
    /// verbindet isolierte Tunnelsegmente/Kavernen des primaeren Systems: zwei unabhaengige Voronoi-
    /// Zellgrenz-Netze ueberlappen sich an ganz anderen Stellen als ein einzelnes, dadurch werden
    /// Sackgassen des einen Systems oft vom anderen durchbrochen.
    pub terrain_connector_frequency: f32,
    pub terrain_connector_threshold: f32,
    pub terrain_connector_widen_multiplier: f32,
    /// Frequenz des 2D-Gates, das entscheidet, ob eine Region ueberhaupt Tunnel bekommt.
    pub terrain_cave_region_frequency: f32,
    /// Perlin-Werte oberhalb dieser Schwelle (Bereich -1..1) gelten als "Hoehlen-aktive" Region.
    pub terrain_cave_region_threshold: f32,
    pub terrain_dirt_layer_depth: i32,
    pub terrain_noise_origin_offset: f32,

    /// Zellgroesse (Weltbloecke) des Baum-Spawn-Gitters - eine Zelle traegt hoechstens einen
    /// gejitterten Kandidaten.
    pub terrain_tree_grid_size: i32,
    /// Wahrscheinlichkeit (nach Jitter, vor Biom-Check), dass eine Gitterzelle einen Baum traegt.
    pub terrain_tree_spawn_chance: f32,
    pub terrain_tree_trunk_height_min: i32,
    pub terrain_tree_trunk_height_max: i32,
    pub terrain_tree_crown_radius_min: i32,
    pub terrain_tree_crown_radius_max: i32,

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

#[derive(Clone, Copy, Debug, Default)]
pub struct EngineConfig {
    pub player: PlayerSettings,
    pub dev: DevSettings,
}

impl Default for PlayerSettings {
    fn default() -> Self {
        Self {
            movement_speed: 12.0,
            sprint_multiplier: 4.0,
            mouse_sensitivity: 0.0025,
            fov_y_radians: 60f32.to_radians(),
            render_distance_chunks: 4,
            vertical_render_distance_chunks: 4,
            hud_visible_default: true,
            msaa_samples: 4,
            ssao_enabled: true,
            ssao_radius: 2.0,
            ssao_strength: 1.4,
            ssao_blur_depth_threshold: 0.0008,
            shadow_cascade_count: 4,
            shadow_map_resolution: 2048,
            shadow_max_distance: 220.0,
            start_flying: true,
        }
    }
}

impl Default for DevSettings {
    fn default() -> Self {
        Self {
            clear_color: wgpu::Color {
                r: 0.02,
                g: 0.02,
                b: 0.02,
                a: 1.0,
            },
            gravity: 26.0,
            jump_speed: 9.0,
            terminal_velocity: 80.0,

            sun_cycle_seconds: 1200.0,
            sun_initial_time_of_day: 0.28,
            ambient_light: 0.2,
            sun_intensity: 1.0,

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
            // Hoehere Frequenz (kleinere Zellen) als das primaere Tunnelsystem - ein unabhaengiges,
            // ANDERS geformtes Zellgrenz-Netz, damit es Sackgassen aus einer voellig anderen Richtung
            // durchbricht statt einfach eine zweite Kopie desselben Musters zu sein. Schwelle
            // absichtlich etwas kleiner/spaerlicher als das primaere System (0.0012) - "kleine
            // Verbindungstunnel", kein zweites Vollnetz. Platzhalter, empirisch kalibriert via
            // `calibrate_cave_thresholds`.
            terrain_connector_frequency: 0.045,
            terrain_connector_threshold: 0.0008,
            terrain_connector_widen_multiplier: 1.0,
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

            // ~8-Block-Gitter mit 12% Spawnchance -> im Schnitt ein Baum alle ~8 Zellen, natuerlich
            // sparsam ohne Kronen-Ueberlappung (Kronenradius max. 3 << halbe Zellgroesse).
            terrain_tree_grid_size: 8,
            terrain_tree_spawn_chance: 0.12,
            terrain_tree_trunk_height_min: 4,
            terrain_tree_trunk_height_max: 7,
            terrain_tree_crown_radius_min: 2,
            terrain_tree_crown_radius_max: 3,

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
fn required_chunk_pool_size(
    render_distance_chunks: i32,
    vertical_render_distance_chunks: i32,
) -> usize {
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
        let required = required_chunk_pool_size(
            self.player.render_distance_chunks,
            self.player.vertical_render_distance_chunks,
        )
        .min(CHUNK_POOL_SAFETY_CAP);
        self.dev.chunk_pool_size = self.dev.chunk_pool_size.max(required);
        self.dev.max_draws_per_direction = self
            .dev
            .max_draws_per_direction
            .max(self.dev.chunk_pool_size);
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
                    log::error!(
                        "Konfiguration {} fehlerhaft ({error}) - nutze Defaults",
                        path.display()
                    );
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
                            log::info!(
                                "Standard-Konfiguration nach {} geschrieben",
                                path.display()
                            );
                        }
                    }
                    Err(error) => log::warn!("Konnte Konfiguration nicht serialisieren: {error}"),
                }
                default.normalized()
            }
        }
    }
}

/// Serde-serialisierbares Spiegelbild von [`PlayerSettings`] - eigene TOML-Tabelle `[player]`.
#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
struct PlayerSettingsFile {
    movement_speed: f32,
    sprint_multiplier: f32,
    mouse_sensitivity: f32,
    fov_degrees: f32,
    render_distance_chunks: i32,
    vertical_render_distance_chunks: i32,
    hud_visible_default: bool,
    msaa_samples: u32,
    ssao_enabled: bool,
    ssao_radius: f32,
    ssao_strength: f32,
    ssao_blur_depth_threshold: f32,
    shadow_cascade_count: u32,
    shadow_map_resolution: u32,
    shadow_max_distance: f32,
    start_flying: bool,
}

/// Serde-serialisierbares Spiegelbild von [`DevSettings`] - eigene TOML-Tabelle `[dev]`.
#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
struct DevSettingsFile {
    clear_color_rgb: [f64; 3],
    gravity: f32,
    jump_speed: f32,
    terminal_velocity: f32,

    sun_cycle_seconds: f32,
    sun_initial_time_of_day: f32,
    ambient_light: f32,
    sun_intensity: f32,

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
    terrain_connector_frequency: f32,
    terrain_connector_threshold: f32,
    terrain_connector_widen_multiplier: f32,
    terrain_cave_widen_depth_range: f32,
    terrain_cheese_widen_amount: f32,
    terrain_tunnel_widen_multiplier: f32,
    terrain_cave_region_frequency: f32,
    terrain_cave_region_threshold: f32,
    terrain_dirt_layer_depth: i32,
    terrain_noise_origin_offset: f32,
    terrain_tree_grid_size: i32,
    terrain_tree_spawn_chance: f32,
    terrain_tree_trunk_height_min: i32,
    terrain_tree_trunk_height_max: i32,
    terrain_tree_crown_radius_min: i32,
    terrain_tree_crown_radius_max: i32,

    player_half_width: f32,
    player_height: f32,
    player_eye_height: f32,
    ground_probe_distance: f32,
    fixed_timestep: f32,
    max_physics_steps_per_frame: u32,

    chunk_pool_size: usize,
    max_faces_per_direction: usize,
    max_draws_per_direction: usize,

    max_chunk_dispatches_per_frame: usize,
    max_chunk_uploads_per_frame: usize,
    max_chunk_unloads_per_frame: usize,
}

/// Serde-serialisierbares Spiegelbild von [`EngineConfig`], zwei TOML-Tabellen `[player]`/`[dev]`
/// mit editorfreundlichen Einheiten (FOV in Grad, Farben als RGB-Arrays). Trennt das Datei-Format
/// von der Laufzeit-Repraesentation.
///
/// BEWUSST ohne `deny_unknown_fields` auf allen drei Ebenen: Ein einzelnes umbenanntes/entferntes
/// Feld (z.B. bei einem Terrain-Schema-Wechsel) wuerde sonst den GESAMTEN Parse-Vorgang abbrechen
/// und `load_or_create` faellt dann auf komplette Defaults zurueck - das setzt still ALLE anderen,
/// unveraendert gebliebenen Einstellungen zurueck, nicht nur die tatsaechlich verschobenen Felder.
/// Unbekannte Felder (oder eine fehlende `[player]`/`[dev]`-Tabelle in einer alten, noch flachen
/// `config.toml`) werden einfach ignoriert bzw. defaulten feldweise, alle anderen bleiben erhalten.
#[derive(Serialize, Deserialize)]
#[serde(default)]
struct ConfigFile {
    player: PlayerSettingsFile,
    dev: DevSettingsFile,
}

impl Default for PlayerSettingsFile {
    fn default() -> Self {
        Self::from(PlayerSettings::default())
    }
}

impl Default for DevSettingsFile {
    fn default() -> Self {
        Self::from(DevSettings::default())
    }
}

impl Default for ConfigFile {
    fn default() -> Self {
        Self::from(EngineConfig::default())
    }
}

impl From<PlayerSettings> for PlayerSettingsFile {
    fn from(p: PlayerSettings) -> Self {
        Self {
            movement_speed: p.movement_speed,
            sprint_multiplier: p.sprint_multiplier,
            mouse_sensitivity: p.mouse_sensitivity,
            fov_degrees: p.fov_y_radians.to_degrees(),
            render_distance_chunks: p.render_distance_chunks,
            vertical_render_distance_chunks: p.vertical_render_distance_chunks,
            hud_visible_default: p.hud_visible_default,
            msaa_samples: p.msaa_samples,
            ssao_enabled: p.ssao_enabled,
            ssao_radius: p.ssao_radius,
            ssao_strength: p.ssao_strength,
            ssao_blur_depth_threshold: p.ssao_blur_depth_threshold,
            shadow_cascade_count: p.shadow_cascade_count,
            shadow_map_resolution: p.shadow_map_resolution,
            shadow_max_distance: p.shadow_max_distance,
            start_flying: p.start_flying,
        }
    }
}

impl From<PlayerSettingsFile> for PlayerSettings {
    fn from(f: PlayerSettingsFile) -> Self {
        Self {
            movement_speed: f.movement_speed,
            sprint_multiplier: f.sprint_multiplier,
            mouse_sensitivity: f.mouse_sensitivity,
            fov_y_radians: f.fov_degrees.to_radians(),
            render_distance_chunks: f.render_distance_chunks.clamp(1, 32),
            vertical_render_distance_chunks: f.vertical_render_distance_chunks.clamp(1, 32),
            hud_visible_default: f.hud_visible_default,
            msaa_samples: f.msaa_samples.clamp(1, 8),
            ssao_enabled: f.ssao_enabled,
            ssao_radius: f.ssao_radius,
            ssao_strength: f.ssao_strength,
            ssao_blur_depth_threshold: f.ssao_blur_depth_threshold.max(0.0),
            shadow_cascade_count: f.shadow_cascade_count.clamp(3, MAX_SHADOW_CASCADES as u32),
            shadow_map_resolution: f.shadow_map_resolution.clamp(256, 8192),
            shadow_max_distance: f.shadow_max_distance.max(16.0),
            start_flying: f.start_flying,
        }
    }
}

impl From<DevSettings> for DevSettingsFile {
    fn from(d: DevSettings) -> Self {
        Self {
            clear_color_rgb: [d.clear_color.r, d.clear_color.g, d.clear_color.b],
            gravity: d.gravity,
            jump_speed: d.jump_speed,
            terminal_velocity: d.terminal_velocity,

            sun_cycle_seconds: d.sun_cycle_seconds,
            sun_initial_time_of_day: d.sun_initial_time_of_day,
            ambient_light: d.ambient_light,
            sun_intensity: d.sun_intensity,

            shadow_split_lambda: d.shadow_split_lambda,
            shadow_depth_bias: d.shadow_depth_bias,
            shadow_depth_bias_slope_scale: d.shadow_depth_bias_slope_scale,

            sky_zenith_day_color: d.sky_zenith_day_color,
            sky_horizon_day_color: d.sky_horizon_day_color,
            sky_night_color: d.sky_night_color,

            godray_count: d.godray_count,
            godray_grid_spacing: d.godray_grid_spacing,
            godray_sample_height: d.godray_sample_height,
            godray_width: d.godray_width,
            godray_beam_length: d.godray_beam_length,
            godray_temporal_blend: d.godray_temporal_blend,

            terrain_seed: d.terrain_seed,
            terrain_continental_frequency: d.terrain_continental_frequency,
            terrain_continental_amplitude: d.terrain_continental_amplitude,
            terrain_mountain_amplitude: d.terrain_mountain_amplitude,
            terrain_mountain_exponent: d.terrain_mountain_exponent,
            terrain_regional_frequency: d.terrain_regional_frequency,
            terrain_regional_amplitude: d.terrain_regional_amplitude,
            terrain_regional_octaves: d.terrain_regional_octaves,
            terrain_regional_lacunarity: d.terrain_regional_lacunarity,
            terrain_regional_gain: d.terrain_regional_gain,
            terrain_cliff_mask_frequency: d.terrain_cliff_mask_frequency,
            terrain_temperature_frequency: d.terrain_temperature_frequency,
            terrain_humidity_frequency: d.terrain_humidity_frequency,
            terrain_desert_temperature_min: d.terrain_desert_temperature_min,
            terrain_desert_humidity_max: d.terrain_desert_humidity_max,
            terrain_sea_compression_range: d.terrain_sea_compression_range,
            terrain_sea_compression_exponent: d.terrain_sea_compression_exponent,
            terrain_cheese_frequency: d.terrain_cheese_frequency,
            terrain_cheese_threshold: d.terrain_cheese_threshold,
            terrain_tunnel_frequency: d.terrain_tunnel_frequency,
            terrain_tunnel_threshold: d.terrain_tunnel_threshold,
            terrain_connector_frequency: d.terrain_connector_frequency,
            terrain_connector_threshold: d.terrain_connector_threshold,
            terrain_connector_widen_multiplier: d.terrain_connector_widen_multiplier,
            terrain_cave_widen_depth_range: d.terrain_cave_widen_depth_range,
            terrain_cheese_widen_amount: d.terrain_cheese_widen_amount,
            terrain_tunnel_widen_multiplier: d.terrain_tunnel_widen_multiplier,
            terrain_cave_region_frequency: d.terrain_cave_region_frequency,
            terrain_cave_region_threshold: d.terrain_cave_region_threshold,
            terrain_dirt_layer_depth: d.terrain_dirt_layer_depth,
            terrain_noise_origin_offset: d.terrain_noise_origin_offset,
            terrain_tree_grid_size: d.terrain_tree_grid_size,
            terrain_tree_spawn_chance: d.terrain_tree_spawn_chance,
            terrain_tree_trunk_height_min: d.terrain_tree_trunk_height_min,
            terrain_tree_trunk_height_max: d.terrain_tree_trunk_height_max,
            terrain_tree_crown_radius_min: d.terrain_tree_crown_radius_min,
            terrain_tree_crown_radius_max: d.terrain_tree_crown_radius_max,

            player_half_width: d.player_half_width,
            player_height: d.player_height,
            player_eye_height: d.player_eye_height,
            ground_probe_distance: d.ground_probe_distance,
            fixed_timestep: d.fixed_timestep,
            max_physics_steps_per_frame: d.max_physics_steps_per_frame,

            chunk_pool_size: d.chunk_pool_size,
            max_faces_per_direction: d.max_faces_per_direction,
            max_draws_per_direction: d.max_draws_per_direction,

            max_chunk_dispatches_per_frame: d.max_chunk_dispatches_per_frame,
            max_chunk_uploads_per_frame: d.max_chunk_uploads_per_frame,
            max_chunk_unloads_per_frame: d.max_chunk_unloads_per_frame,
        }
    }
}

impl From<DevSettingsFile> for DevSettings {
    fn from(f: DevSettingsFile) -> Self {
        Self {
            clear_color: wgpu::Color {
                r: f.clear_color_rgb[0],
                g: f.clear_color_rgb[1],
                b: f.clear_color_rgb[2],
                a: 1.0,
            },
            gravity: f.gravity,
            jump_speed: f.jump_speed,
            terminal_velocity: f.terminal_velocity.max(1.0),

            sun_cycle_seconds: f.sun_cycle_seconds.max(1.0),
            sun_initial_time_of_day: f.sun_initial_time_of_day.rem_euclid(1.0),
            ambient_light: f.ambient_light.clamp(0.0, 1.0),
            sun_intensity: f.sun_intensity.max(0.0),

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
            terrain_connector_frequency: f.terrain_connector_frequency,
            terrain_connector_threshold: f.terrain_connector_threshold,
            terrain_connector_widen_multiplier: f.terrain_connector_widen_multiplier.max(0.0),
            terrain_cave_widen_depth_range: f.terrain_cave_widen_depth_range.max(1.0),
            terrain_cheese_widen_amount: f.terrain_cheese_widen_amount.max(0.0),
            terrain_tunnel_widen_multiplier: f.terrain_tunnel_widen_multiplier.max(0.0),
            terrain_cave_region_frequency: f.terrain_cave_region_frequency,
            terrain_cave_region_threshold: f.terrain_cave_region_threshold,
            terrain_dirt_layer_depth: f.terrain_dirt_layer_depth,
            terrain_noise_origin_offset: f.terrain_noise_origin_offset,
            terrain_tree_grid_size: f.terrain_tree_grid_size.max(1),
            terrain_tree_spawn_chance: f.terrain_tree_spawn_chance.clamp(0.0, 1.0),
            terrain_tree_trunk_height_min: f.terrain_tree_trunk_height_min.max(1),
            terrain_tree_trunk_height_max: f
                .terrain_tree_trunk_height_max
                .max(f.terrain_tree_trunk_height_min.max(1)),
            terrain_tree_crown_radius_min: f.terrain_tree_crown_radius_min.max(0),
            terrain_tree_crown_radius_max: f
                .terrain_tree_crown_radius_max
                .max(f.terrain_tree_crown_radius_min.max(0)),

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
    }
}

impl From<EngineConfig> for ConfigFile {
    fn from(c: EngineConfig) -> Self {
        Self {
            player: PlayerSettingsFile::from(c.player),
            dev: DevSettingsFile::from(c.dev),
        }
    }
}

impl From<ConfigFile> for EngineConfig {
    fn from(f: ConfigFile) -> Self {
        Self {
            player: PlayerSettings::from(f.player),
            dev: DevSettings::from(f.dev),
        }
        .normalized()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalized_pool_covers_the_full_load_window() {
        let config = EngineConfig::default().normalized();
        let required = required_chunk_pool_size(
            config.player.render_distance_chunks,
            config.player.vertical_render_distance_chunks,
        );
        assert!(config.dev.chunk_pool_size >= required);
        assert!(config.dev.max_draws_per_direction >= config.dev.chunk_pool_size);
    }
}
