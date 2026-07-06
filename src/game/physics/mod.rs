use glam::Vec3;

use crate::engine::config::EngineConfig;
use crate::game::world::generator::TerrainGenerator;

/// Physik-Zustand des Spielers: Fallen/Springen/Kollision (wenn nicht im Flugmodus) via fester
/// Zeitschrittweite, damit Gravitation und Sprunghoehe unabhaengig von der Framerate sind.
pub struct PlayerPhysics {
    pub velocity: Vec3,
    pub grounded: bool,
    pub flying: bool,
    accumulator: f32,
    fixed_timestep: f32,
    max_steps_per_frame: u32,
    player_half_width: f32,
    player_height: f32,
    player_eye_height: f32,
    ground_probe_distance: f32,
}

impl PlayerPhysics {
    pub fn new(start_flying: bool, config: &EngineConfig) -> Self {
        Self {
            velocity: Vec3::ZERO,
            grounded: false,
            flying: start_flying,
            accumulator: 0.0,
            fixed_timestep: config.fixed_timestep,
            max_steps_per_frame: config.max_physics_steps_per_frame,
            player_half_width: config.player_half_width,
            player_height: config.player_height,
            player_eye_height: config.player_eye_height,
            ground_probe_distance: config.ground_probe_distance,
        }
    }

    pub fn eye_height(&self) -> f32 {
        self.player_eye_height
    }

    pub fn toggle_flying(&mut self) {
        self.flying = !self.flying;
        self.velocity = Vec3::ZERO;
    }

    /// Verarbeitet die reale Frame-Zeit in festen Zeitschritten (kappt Nachhol-Schritte bei
    /// Framerate-Einbruechen, um eine "Spiral of Death" zu vermeiden).
    pub fn advance(
        &mut self,
        frame_dt: f32,
        generator: &TerrainGenerator,
        eye_position: &mut Vec3,
        horizontal_move: Vec3,
        jump_held: bool,
        gravity: f32,
        jump_speed: f32,
    ) {
        self.accumulator += frame_dt.min(0.25);

        let mut steps = 0;
        while self.accumulator >= self.fixed_timestep && steps < self.max_steps_per_frame {
            self.step(generator, eye_position, horizontal_move, jump_held, gravity, jump_speed);
            self.accumulator -= self.fixed_timestep;
            steps += 1;
        }
    }

    fn step(
        &mut self,
        generator: &TerrainGenerator,
        eye_position: &mut Vec3,
        horizontal_move: Vec3,
        jump_held: bool,
        gravity: f32,
        jump_speed: f32,
    ) {
        self.velocity.x = horizontal_move.x;
        self.velocity.z = horizontal_move.z;

        if self.grounded && jump_held {
            self.velocity.y = jump_speed;
        } else {
            self.velocity.y -= gravity * self.fixed_timestep;
        }

        let mut feet = *eye_position - Vec3::new(0.0, self.player_eye_height, 0.0);
        let delta = self.velocity * self.fixed_timestep;

        feet.x += delta.x;
        if aabb_overlaps_solid(generator, feet, self.player_half_width, self.player_height) {
            feet.x -= delta.x;
            self.velocity.x = 0.0;
        }

        feet.z += delta.z;
        if aabb_overlaps_solid(generator, feet, self.player_half_width, self.player_height) {
            feet.z -= delta.z;
            self.velocity.z = 0.0;
        }

        if resolve_vertical_collision(
            generator,
            &mut feet,
            delta.y,
            self.player_half_width,
            self.player_height,
        ) {
            self.velocity.y = 0.0;
        }

        self.grounded = aabb_overlaps_solid(
            generator,
            feet - Vec3::new(0.0, self.ground_probe_distance, 0.0),
            self.player_half_width,
            self.player_height,
        );

        *eye_position = feet + Vec3::new(0.0, self.player_eye_height, 0.0);
    }
}

/// Loest eine vertikale Bewegung um `delta_y` auf. Statt bei Kollision nur den kompletten
/// Bewegungsschritt zurueckzunehmen (verursacht Jittering, da die Fuesse dann leicht ueber/unter
/// der tatsaechlichen Voxel-Oberflaeche schweben), wird die Fussposition exakt auf die getroffene
/// Voxel-Grenze gesnappt. Liefert `true`, wenn eine Kollision aufgetreten ist.
fn resolve_vertical_collision(
    generator: &TerrainGenerator,
    feet: &mut Vec3,
    delta_y: f32,
    player_half_width: f32,
    player_height: f32,
) -> bool {
    const EPSILON: f32 = 1e-4;
    let x0 = (feet.x - player_half_width).floor() as i32;
    let x1 = (feet.x + player_half_width - EPSILON).floor() as i32;
    let z0 = (feet.z - player_half_width).floor() as i32;
    let z1 = (feet.z + player_half_width - EPSILON).floor() as i32;

    let footprint_solid_at = |y: i32| (z0..=z1).any(|z| (x0..=x1).any(|x| generator.is_solid(x, y, z)));

    if delta_y < 0.0 {
        let scan_top = feet.y.floor() as i32;
        let scan_bottom = (feet.y + delta_y).floor() as i32;
        if let Some(surface_y) = (scan_bottom..=scan_top).rev().find(|&y| footprint_solid_at(y)) {
            feet.y = (surface_y + 1) as f32;
            return true;
        }
        feet.y += delta_y;
        false
    } else {
        let scan_bottom = (feet.y + player_height).floor() as i32;
        let scan_top = (feet.y + delta_y + player_height).floor() as i32;
        if let Some(ceiling_y) = (scan_bottom..=scan_top).find(|&y| footprint_solid_at(y)) {
            feet.y = ceiling_y as f32 - player_height;
            return true;
        }
        feet.y += delta_y;
        false
    }
}

fn aabb_overlaps_solid(generator: &TerrainGenerator, feet: Vec3, player_half_width: f32, player_height: f32) -> bool {
    let min = feet - Vec3::new(player_half_width, 0.0, player_half_width);
    let max = feet + Vec3::new(player_half_width, player_height, player_half_width);
    const EPSILON: f32 = 1e-4;

    let x0 = min.x.floor() as i32;
    let x1 = (max.x - EPSILON).floor() as i32;
    let y0 = min.y.floor() as i32;
    let y1 = (max.y - EPSILON).floor() as i32;
    let z0 = min.z.floor() as i32;
    let z1 = (max.z - EPSILON).floor() as i32;

    for z in z0..=z1 {
        for y in y0..=y1 {
            for x in x0..=x1 {
                if generator.is_solid(x, y, z) {
                    return true;
                }
            }
        }
    }
    false
}
