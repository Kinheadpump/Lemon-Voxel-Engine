#[derive(Clone, Copy, Debug)]
pub struct EngineConfig {
    pub movement_speed: f32,
    pub mouse_sensitivity: f32,
    pub fov_y_radians: f32,
    pub render_distance_chunks: i32,
    pub clear_color: wgpu::Color,
    pub hud_visible_default: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            movement_speed: 12.0,
            mouse_sensitivity: 0.0025,
            fov_y_radians: 60f32.to_radians(),
            render_distance_chunks: 4,
            clear_color: wgpu::Color { r: 0.02, g: 0.02, b: 0.02, a: 1.0 },
            hud_visible_default: true,
        }
    }
}
