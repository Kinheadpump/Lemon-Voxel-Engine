use glam::Vec3;

/// Direktionale Lichtquelle mit simplem Tag/Nacht-Zyklus. Reines Datenmodell (keine GPU-Typen) -
/// das Rendering liest nur `light_direction()`/`direction_to_sun()` aus.
pub struct Sun {
    /// Fortschritt im Zyklus, 0.0 = Sonnenaufgang, 0.25 = Zenit, 0.5 = Sonnenuntergang, wrapped in [0, 1).
    time_of_day: f32,
}

impl Sun {
    pub fn new(initial_time_of_day: f32) -> Self {
        Self { time_of_day: initial_time_of_day.rem_euclid(1.0) }
    }

    pub fn advance(&mut self, dt: f32, cycle_seconds: f32) {
        self.time_of_day = (self.time_of_day + dt / cycle_seconds).rem_euclid(1.0);
    }

    /// Bogen von Sonnenaufgang (+X) ueber den Zenit nach Sonnenuntergang (-X), mit leichter
    /// Neigung in +Z, damit die Bahn nicht in einer einzigen Ebene entartet.
    pub fn direction_to_sun(&self) -> Vec3 {
        let angle = self.time_of_day * std::f32::consts::TAU;
        Vec3::new(angle.cos(), angle.sin(), 0.35).normalize()
    }

    /// Richtung, in die das Sonnenlicht faellt - das Gegenteil von `direction_to_sun`.
    pub fn light_direction(&self) -> Vec3 {
        -self.direction_to_sun()
    }

    pub fn is_above_horizon(&self) -> bool {
        self.direction_to_sun().y > 0.0
    }
}
