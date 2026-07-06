use glam::Vec3;

use crate::game::world::generator::TerrainGenerator;

pub const FIXED_TIMESTEP: f32 = 1.0 / 60.0;
const MAX_STEPS_PER_FRAME: u32 = 8;

const PLAYER_HALF_WIDTH: f32 = 0.3;
const PLAYER_HEIGHT: f32 = 1.8;
pub const PLAYER_EYE_HEIGHT: f32 = 1.6;
const GROUND_PROBE_DISTANCE: f32 = 0.05;

/// Physik-Zustand des Spielers: Fallen/Springen/Kollision (wenn nicht im Flugmodus) via fester
/// Zeitschrittweite, damit Gravitation und Sprunghoehe unabhaengig von der Framerate sind.
pub struct PlayerPhysics {
    pub velocity: Vec3,
    pub grounded: bool,
    pub flying: bool,
    accumulator: f32,
}

impl PlayerPhysics {
    pub fn new(start_flying: bool) -> Self {
        Self { velocity: Vec3::ZERO, grounded: false, flying: start_flying, accumulator: 0.0 }
    }

    pub fn toggle_flying(&mut self) {
        self.flying = !self.flying;
        self.velocity = Vec3::ZERO;
    }

    /// Verarbeitet die reale Frame-Zeit in festen 1/60s-Schritten (kappt Nachhol-Schritte bei
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
        while self.accumulator >= FIXED_TIMESTEP && steps < MAX_STEPS_PER_FRAME {
            self.step(generator, eye_position, horizontal_move, jump_held, gravity, jump_speed);
            self.accumulator -= FIXED_TIMESTEP;
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
            self.velocity.y -= gravity * FIXED_TIMESTEP;
        }

        let mut feet = *eye_position - Vec3::new(0.0, PLAYER_EYE_HEIGHT, 0.0);
        let delta = self.velocity * FIXED_TIMESTEP;

        feet.x += delta.x;
        if aabb_overlaps_solid(generator, feet) {
            feet.x -= delta.x;
            self.velocity.x = 0.0;
        }

        feet.z += delta.z;
        if aabb_overlaps_solid(generator, feet) {
            feet.z -= delta.z;
            self.velocity.z = 0.0;
        }

        feet.y += delta.y;
        if aabb_overlaps_solid(generator, feet) {
            feet.y -= delta.y;
            self.velocity.y = 0.0;
        }

        self.grounded =
            aabb_overlaps_solid(generator, feet - Vec3::new(0.0, GROUND_PROBE_DISTANCE, 0.0));

        *eye_position = feet + Vec3::new(0.0, PLAYER_EYE_HEIGHT, 0.0);
    }
}

fn aabb_overlaps_solid(generator: &TerrainGenerator, feet: Vec3) -> bool {
    let min = feet - Vec3::new(PLAYER_HALF_WIDTH, 0.0, PLAYER_HALF_WIDTH);
    let max = feet + Vec3::new(PLAYER_HALF_WIDTH, PLAYER_HEIGHT, PLAYER_HALF_WIDTH);
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
