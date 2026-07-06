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
            render_distance_chunks: f.render_distance_chunks.clamp(1, 15),
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
        }
    }
}
