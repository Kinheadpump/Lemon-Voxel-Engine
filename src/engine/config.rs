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
    pub terrain_regional_frequency: f32,
    pub terrain_regional_amplitude: f32,
    pub terrain_cliff_mask_frequency: f32,
    pub terrain_sea_compression_range: f32,
    pub terrain_sea_compression_exponent: f32,
    pub terrain_cave_frequency: f32,
    /// Perlin-Werte oberhalb dieser Schwelle (Bereich -1..1) werden zu Hoehlen.
    pub terrain_cave_threshold: f32,
    /// Rasterabstand (in Bloecken) des sparse ausgewerteten 3D-Hoehlendichterasters - dazwischen
    /// wird trilinear interpoliert. Muss `CHUNK_SIZE` (32) teilen, Mindestwert 2.
    pub terrain_cave_sample_stride: i32,
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
            // Regional-Skala ~83 Bloecke/Feature, Kontinental-Skala 32x groesser (~2650 Bloecke) -
            // haelt Land/Ozean auf kontinentaler Ebene auseinander statt regional zu flackern.
            terrain_continental_frequency: 0.012 / 32.0,
            terrain_continental_amplitude: 40.0,
            terrain_regional_frequency: 0.012,
            terrain_regional_amplitude: 18.0,
            terrain_cliff_mask_frequency: 0.008,
            terrain_sea_compression_range: 20.0,
            terrain_sea_compression_exponent: 2.2,
            terrain_cave_frequency: 0.05,
            terrain_cave_threshold: 0.6,
            terrain_cave_sample_stride: 4,
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

impl EngineConfig {
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
                    Self::default()
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
                default
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
    terrain_regional_frequency: f32,
    terrain_regional_amplitude: f32,
    terrain_cliff_mask_frequency: f32,
    terrain_sea_compression_range: f32,
    terrain_sea_compression_exponent: f32,
    terrain_cave_frequency: f32,
    terrain_cave_threshold: f32,
    terrain_cave_sample_stride: i32,
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
            terrain_regional_frequency: c.terrain_regional_frequency,
            terrain_regional_amplitude: c.terrain_regional_amplitude,
            terrain_cliff_mask_frequency: c.terrain_cliff_mask_frequency,
            terrain_sea_compression_range: c.terrain_sea_compression_range,
            terrain_sea_compression_exponent: c.terrain_sea_compression_exponent,
            terrain_cave_frequency: c.terrain_cave_frequency,
            terrain_cave_threshold: c.terrain_cave_threshold,
            terrain_cave_sample_stride: c.terrain_cave_sample_stride,
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
            terrain_regional_frequency: f.terrain_regional_frequency,
            terrain_regional_amplitude: f.terrain_regional_amplitude,
            terrain_cliff_mask_frequency: f.terrain_cliff_mask_frequency,
            terrain_sea_compression_range: f.terrain_sea_compression_range.max(1.0),
            terrain_sea_compression_exponent: f.terrain_sea_compression_exponent.max(1.0),
            terrain_cave_frequency: f.terrain_cave_frequency,
            terrain_cave_threshold: f.terrain_cave_threshold,
            terrain_cave_sample_stride: f.terrain_cave_sample_stride.clamp(2, 16),
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
    }
}
