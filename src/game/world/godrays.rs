use glam::Vec3;

use crate::engine::config::EngineConfig;

use super::generator::TerrainGenerator;

/// Eine Godray-Billboard-Instanz, wie sie im SSBO liegt (siehe `render/godray.wgsl`). `intensity`
/// wird ausschliesslich vom Compute-Pass geschrieben/gelesen (In-Place-Temporal-Blend) - die
/// CPU-Seite setzt sie bei einer Neu-Platzierung nur einmalig auf 0 und ruehrt sie danach nicht an.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GodrayInstanceData {
    /// xyz = Basis-Position (auf der Terrainoberflaeche + Hoehenversatz), w = Intensity.
    pub position_intensity: [f32; 4],
    /// x = Billboard-Breite (dient GLEICHZEITIG als Sample-Radius der Kantenerkennung), y = Hoehe.
    pub size: [f32; 4],
}

/// Platziert Godray-Kandidaten auf einem an die Terrainoberflaeche angehefteten Gitter um die
/// Kamera. Regeneriert nur bei ausreichender Kamerabewegung (wie das Chunk-Ladefenster) statt jeden
/// Frame Rauschen abzufragen und den kompletten SSBO neu hochzuladen.
pub struct GodrayField {
    count: u32,
    grid_spacing: f32,
    height_offset: f32,
    width: f32,
    beam_height: f32,
    last_center: Option<Vec3>,
    regen_threshold: f32,
    instances: Vec<GodrayInstanceData>,
}

impl GodrayField {
    pub fn new(config: &EngineConfig) -> Self {
        let grid_dim = (config.godray_count as f32).sqrt().ceil() as u32;
        Self {
            count: config.godray_count,
            grid_spacing: config.godray_grid_spacing,
            height_offset: config.godray_height_offset,
            width: config.godray_width,
            beam_height: config.godray_beam_height,
            last_center: None,
            regen_threshold: config.godray_grid_spacing * grid_dim as f32 * 0.5,
            instances: Vec::with_capacity(config.godray_count as usize),
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

        self.instances.clear();
        'grid: for gz in -half..=half {
            for gx in -half..=half {
                if self.instances.len() as u32 >= self.count {
                    break 'grid;
                }

                let world_x = camera_position.x + gx as f32 * self.grid_spacing + next_jitter() * self.grid_spacing * 0.6;
                let world_z = camera_position.z + gz as f32 * self.grid_spacing + next_jitter() * self.grid_spacing * 0.6;
                let surface_y = generator.height_at(world_x.floor() as i32, world_z.floor() as i32) as f32;
                let base_y = surface_y + 1.0 + self.height_offset;

                self.instances.push(GodrayInstanceData {
                    position_intensity: [world_x, base_y, world_z, 0.0],
                    size: [self.width, self.beam_height, 0.0, 0.0],
                });
            }
        }

        Some(&self.instances)
    }
}
