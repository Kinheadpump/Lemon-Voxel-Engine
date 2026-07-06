use std::path::Path;

use serde::{Deserialize, Serialize};

pub const CONFIG_PATH: &str = "config.toml";

#[derive(Clone, Copy, Debug)]
pub struct EngineConfig {
    pub movement_speed: f32,
    pub sprint_multiplier: f32,
    pub mouse_sensitivity: f32,
    pub fov_y_radians: f32,
    pub render_distance_chunks: i32,
    pub clear_color: wgpu::Color,
    pub hud_visible_default: bool,
    pub msaa_samples: u32,
    pub ssao_enabled: bool,
    pub ssao_radius: f32,
    pub ssao_strength: f32,
    pub gravity: f32,
    pub jump_speed: f32,
    pub start_flying: bool,

    pub terrain_seed: u32,
    pub terrain_noise_frequency: f32,
    pub terrain_base_height: f32,
    pub terrain_height_amplitude: f32,
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
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            movement_speed: 12.0,
            sprint_multiplier: 4.0,
            mouse_sensitivity: 0.0025,
            fov_y_radians: 60f32.to_radians(),
            render_distance_chunks: 4,
            clear_color: wgpu::Color { r: 0.02, g: 0.02, b: 0.02, a: 1.0 },
            hud_visible_default: true,
            msaa_samples: 4,
            ssao_enabled: true,
            ssao_radius: 2.0,
            ssao_strength: 1.4,
            gravity: 26.0,
            jump_speed: 9.0,
            start_flying: true,

            terrain_seed: 1337,
            terrain_noise_frequency: 0.02,
            terrain_base_height: 12.0,
            terrain_height_amplitude: 10.0,
            terrain_dirt_layer_depth: 3,
            terrain_noise_origin_offset: 10_000.0,

            player_half_width: 0.3,
            player_height: 1.8,
            player_eye_height: 1.6,
            ground_probe_distance: 0.1,
            fixed_timestep: 1.0 / 60.0,
            max_physics_steps_per_frame: 8,

            chunk_pool_size: 4300,
            max_faces_per_direction: 3_000_000,
            max_draws_per_direction: 4300,
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
#[derive(Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct ConfigFile {
    movement_speed: f32,
    sprint_multiplier: f32,
    mouse_sensitivity: f32,
    fov_degrees: f32,
    render_distance_chunks: i32,
    clear_color_rgb: [f64; 3],
    hud_visible_default: bool,
    msaa_samples: u32,
    ssao_enabled: bool,
    ssao_radius: f32,
    ssao_strength: f32,
    gravity: f32,
    jump_speed: f32,
    start_flying: bool,

    terrain_seed: u32,
    terrain_noise_frequency: f32,
    terrain_base_height: f32,
    terrain_height_amplitude: f32,
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
            clear_color_rgb: [c.clear_color.r, c.clear_color.g, c.clear_color.b],
            hud_visible_default: c.hud_visible_default,
            msaa_samples: c.msaa_samples,
            ssao_enabled: c.ssao_enabled,
            ssao_radius: c.ssao_radius,
            ssao_strength: c.ssao_strength,
            gravity: c.gravity,
            jump_speed: c.jump_speed,
            start_flying: c.start_flying,

            terrain_seed: c.terrain_seed,
            terrain_noise_frequency: c.terrain_noise_frequency,
            terrain_base_height: c.terrain_base_height,
            terrain_height_amplitude: c.terrain_height_amplitude,
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
            gravity: f.gravity,
            jump_speed: f.jump_speed,
            start_flying: f.start_flying,

            terrain_seed: f.terrain_seed,
            terrain_noise_frequency: f.terrain_noise_frequency,
            terrain_base_height: f.terrain_base_height,
            terrain_height_amplitude: f.terrain_height_amplitude,
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
        }
    }
}
