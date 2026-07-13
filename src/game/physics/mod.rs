use glam::{IVec3, Vec3};

use crate::engine::config::EngineConfig;

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
    terminal_velocity: f32,
    step_height: f32,
}

impl PlayerPhysics {
    pub fn new(start_flying: bool, config: &EngineConfig) -> Self {
        Self {
            velocity: Vec3::ZERO,
            grounded: false,
            flying: start_flying,
            accumulator: 0.0,
            fixed_timestep: config.dev.fixed_timestep,
            max_steps_per_frame: config.dev.max_physics_steps_per_frame,
            player_half_width: config.dev.player_half_width,
            player_height: config.dev.player_height,
            player_eye_height: config.dev.player_eye_height,
            ground_probe_distance: config.dev.ground_probe_distance,
            terminal_velocity: config.dev.terminal_velocity,
            step_height: config.dev.player_step_height,
        }
    }

    pub fn eye_height(&self) -> f32 {
        self.player_eye_height
    }

    pub fn toggle_flying(&mut self) {
        self.flying = !self.flying;
        self.velocity = Vec3::ZERO;
    }

    /// Prueft, ob der angegebene Block mit der aktuellen Spieler-AABB (aus der Augenposition
    /// abgeleitet) ueberlappt - genutzt, um zu verhindern, dass sich der Spieler durch Platzieren
    /// selbst einsperrt.
    pub fn occupies_block(&self, eye_position: Vec3, block: IVec3) -> bool {
        let feet = eye_position - Vec3::new(0.0, self.player_eye_height, 0.0);
        let min = feet - Vec3::new(self.player_half_width, 0.0, self.player_half_width);
        let max = feet + Vec3::new(self.player_half_width, self.player_height, self.player_half_width);

        let block_min = block.as_vec3();
        let block_max = block_min + Vec3::ONE;

        min.x < block_max.x
            && max.x > block_min.x
            && min.y < block_max.y
            && max.y > block_min.y
            && min.z < block_max.z
            && max.z > block_min.z
    }

    /// Verarbeitet die reale Frame-Zeit in festen Zeitschritten (kappt Nachhol-Schritte bei
    /// Framerate-Einbruechen, um eine "Spiral of Death" zu vermeiden).
    pub fn advance<F: Fn(i32, i32, i32) -> bool>(
        &mut self,
        frame_dt: f32,
        is_solid: &F,
        eye_position: &mut Vec3,
        horizontal_move: Vec3,
        jump_held: bool,
        gravity: f32,
        jump_speed: f32,
    ) {
        self.accumulator += frame_dt.min(0.25);

        let mut steps = 0;
        while self.accumulator >= self.fixed_timestep && steps < self.max_steps_per_frame {
            self.step(is_solid, eye_position, horizontal_move, jump_held, gravity, jump_speed);
            self.accumulator -= self.fixed_timestep;
            steps += 1;
        }
    }

    fn step<F: Fn(i32, i32, i32) -> bool>(
        &mut self,
        is_solid: &F,
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
            // Terminal-Velocity-Clamp: verhindert unbegrenzt wachsende Fallgeschwindigkeit in der
            // vertikal unbegrenzten Welt - sonst muesste der Lande-Frame einen zu `delta_y`
            // proportionalen (also riesigen) Sweep-Kollisions-Scan abarbeiten und ruckelt.
            self.velocity.y = (self.velocity.y - gravity * self.fixed_timestep).max(-self.terminal_velocity);
        }

        let mut feet = *eye_position - Vec3::new(0.0, self.player_eye_height, 0.0);
        let delta = self.velocity * self.fixed_timestep;

        if !self.try_move_axis(is_solid, &mut feet, Vec3::new(delta.x, 0.0, 0.0)) {
            self.velocity.x = 0.0;
        }
        if !self.try_move_axis(is_solid, &mut feet, Vec3::new(0.0, 0.0, delta.z)) {
            self.velocity.z = 0.0;
        }

        if resolve_vertical_collision(is_solid, &mut feet, delta.y, self.player_half_width, self.player_height) {
            self.velocity.y = 0.0;
        }

        self.grounded = aabb_overlaps_solid(
            is_solid,
            feet - Vec3::new(0.0, self.ground_probe_distance, 0.0),
            self.player_half_width,
            self.player_height,
        );

        *eye_position = feet + Vec3::new(0.0, self.player_eye_height, 0.0);
    }

    /// Bewegt `feet` entlang EINER horizontalen Achse (`axis_delta` hat nur eine Nicht-Null-
    /// Komponente). Blockiert eine normale Kollision die Bewegung, wird zusaetzlich versucht,
    /// `feet` um bis zu `step_height` anzuheben ("Auto-Step"): auf gequantisiertem Blockterrain
    /// unterscheiden sich Nachbarspalten praktisch ueberall um mindestens 1 Block (jede
    /// Hoehenkarte wird auf ganze Voxel gerundet) - ohne Auto-Step blieb der Spieler an jeder
    /// dieser Kanten haengen und kam nur per Sprung darueber hinweg ("Bewegung nur nach Sprung").
    /// Die Landung nach dem Step ist exakt flush mit der neuen Oberflaeche (kein Nachsetzen
    /// noetig): Bloecke sind volle Einheitswuerfel, `step_height` deckt genau einen Block ab, die
    /// naechste `resolve_vertical_collision` haelt die Fuesse dort einfach fest.
    ///
    /// Liefert `true`, wenn die Bewegung (ggf. mit Step-Up) erfolgreich war.
    fn try_move_axis<F: Fn(i32, i32, i32) -> bool>(
        &self,
        is_solid: &F,
        feet: &mut Vec3,
        axis_delta: Vec3,
    ) -> bool {
        let moved = *feet + axis_delta;
        if !aabb_overlaps_solid(is_solid, moved, self.player_half_width, self.player_height) {
            *feet = moved;
            return true;
        }

        if self.step_height <= 0.0 {
            return false;
        }
        let stepped = moved + Vec3::new(0.0, self.step_height, 0.0);
        if aabb_overlaps_solid(is_solid, stepped, self.player_half_width, self.player_height) {
            return false;
        }
        *feet = stepped;
        true
    }
}

/// Loest eine vertikale Bewegung um `delta_y` auf. Statt bei Kollision nur den kompletten
/// Bewegungsschritt zurueckzunehmen (verursacht Jittering, da die Fuesse dann leicht ueber/unter
/// der tatsaechlichen Voxel-Oberflaeche schweben), wird die Fussposition exakt auf die getroffene
/// Voxel-Grenze gesnappt. Liefert `true`, wenn eine Kollision aufgetreten ist.
fn resolve_vertical_collision<F: Fn(i32, i32, i32) -> bool>(
    is_solid: &F,
    feet: &mut Vec3,
    delta_y: f32,
    player_half_width: f32,
    player_height: f32,
) -> bool {
    // SKIN_WIDTH schrumpft die effektive Kollisions-AABB um einen Hauch auf allen Seiten
    // ("Skin Width", Standardtechnik in Physik-Engines): `feet.y` rundet sich rein durch das
    // Hin-und-Her von `eye_position -> feet -> eye_position` (Subtraktion/Addition von
    // `player_eye_height`, s. `step`) jeden Frame um wenige ULP - z.B. exakt 1.0 wird zu
    // 0.9999999. OHNE Toleranz auf der UNTEREN Grenze rundet `floor()` das eigene Standbein dann
    // eine ganze Blockebene zu tief, die naechste Kollisionspruefung findet den Bodenblock DIREKT
    // UNTER den Fuessen "im eigenen Koerper" und blockiert JEDE Bewegung - reproduzierbar exakt
    // als "Spieler kann sich am Boden nicht bewegen, nur nach einem Sprung" (Sprung hebt `feet.y`
    // weit genug von der Ganzzahl-Grenze weg, um dem Rundungsfehler zu entkommen). Auf der
    // OBEREN Grenze existierte die Toleranz bereits (verhindert, dass die exakt anliegende
    // Nachbarzelle mitgezaehlt wird) - hier wird sie fehlend ergaenzt, symmetrisch auf allen
    // sechs Seiten der AABB.
    const SKIN_WIDTH: f32 = 1e-4;
    let x0 = (feet.x - player_half_width + SKIN_WIDTH).floor() as i32;
    let x1 = (feet.x + player_half_width - SKIN_WIDTH).floor() as i32;
    let z0 = (feet.z - player_half_width + SKIN_WIDTH).floor() as i32;
    let z1 = (feet.z + player_half_width - SKIN_WIDTH).floor() as i32;

    let footprint_solid_at = |y: i32| (z0..=z1).any(|z| (x0..=x1).any(|x| is_solid(x, y, z)));

    if delta_y < 0.0 {
        let scan_top = (feet.y + SKIN_WIDTH).floor() as i32;
        let scan_bottom = (feet.y + delta_y).floor() as i32;
        if let Some(surface_y) = (scan_bottom..=scan_top).rev().find(|&y| footprint_solid_at(y)) {
            feet.y = (surface_y + 1) as f32;
            return true;
        }
        feet.y += delta_y;
        false
    } else {
        let scan_bottom = (feet.y + player_height + SKIN_WIDTH).floor() as i32;
        let scan_top = (feet.y + delta_y + player_height).floor() as i32;
        if let Some(ceiling_y) = (scan_bottom..=scan_top).find(|&y| footprint_solid_at(y)) {
            feet.y = ceiling_y as f32 - player_height;
            return true;
        }
        feet.y += delta_y;
        false
    }
}

fn aabb_overlaps_solid<F: Fn(i32, i32, i32) -> bool>(
    is_solid: &F,
    feet: Vec3,
    player_half_width: f32,
    player_height: f32,
) -> bool {
    let min = feet - Vec3::new(player_half_width, 0.0, player_half_width);
    let max = feet + Vec3::new(player_half_width, player_height, player_half_width);
    // Skin Width auf allen sechs Seiten - s. ausfuehrlicher Kommentar an `resolve_vertical_collision`.
    const SKIN_WIDTH: f32 = 1e-4;

    let x0 = (min.x + SKIN_WIDTH).floor() as i32;
    let x1 = (max.x - SKIN_WIDTH).floor() as i32;
    let y0 = (min.y + SKIN_WIDTH).floor() as i32;
    let y1 = (max.y - SKIN_WIDTH).floor() as i32;
    let z0 = (min.z + SKIN_WIDTH).floor() as i32;
    let z1 = (max.z - SKIN_WIDTH).floor() as i32;

    for z in z0..=z1 {
        for y in y0..=y1 {
            for x in x0..=x1 {
                if is_solid(x, y, z) {
                    return true;
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::config::EngineConfig;

    fn flat_floor(_x: i32, y: i32, _z: i32) -> bool {
        y < 0
    }

    #[test]
    fn grounded_player_can_walk_horizontally() {
        let config = EngineConfig::default();
        let mut physics = PlayerPhysics::new(false, &config);
        let mut eye = Vec3::new(0.0, config.dev.player_eye_height, 0.0);

        for _ in 0..30 {
            physics.advance(
                1.0 / 60.0,
                &flat_floor,
                &mut eye,
                Vec3::ZERO,
                false,
                config.dev.gravity,
                config.dev.jump_speed,
            );
        }
        assert!(physics.grounded, "Spieler sollte nach dem Fallen auf dem Boden stehen");

        let x_before = eye.x;
        for _ in 0..10 {
            physics.advance(
                1.0 / 60.0,
                &flat_floor,
                &mut eye,
                Vec3::new(5.0, 0.0, 0.0),
                false,
                config.dev.gravity,
                config.dev.jump_speed,
            );
        }
        assert!(
            eye.x > x_before + 0.1,
            "Spieler sollte sich am Boden horizontal bewegen koennen, x blieb bei {}",
            eye.x
        );
    }

    /// Regressionstest fuer "Bewegung nur nach Sprung": ein einzelner 1-Block-Bordstein darf
    /// horizontale Bewegung nicht mehr komplett stoppen (Auto-Step, s. `try_move_axis`-Kommentar) -
    /// der reale prozedurale Terrain-Generator quantisiert jede Spaltenhoehe auf ganze Bloecke,
    /// benachbarte Spalten unterscheiden sich dadurch praktisch ueberall um mindestens 1 Block.
    #[test]
    fn grounded_player_auto_steps_over_single_block_ledge() {
        let step_at_x = |x: i32, y: i32, _z: i32| if x < 4 { y < 0 } else { y < 1 };

        let config = EngineConfig::default();
        let mut physics = PlayerPhysics::new(false, &config);
        let mut eye = Vec3::new(0.0, config.dev.player_eye_height, 0.0);

        for _ in 0..30 {
            physics.advance(1.0 / 60.0, &step_at_x, &mut eye, Vec3::ZERO, false, config.dev.gravity, config.dev.jump_speed);
        }
        assert!(physics.grounded);

        for _ in 0..120 {
            physics.advance(
                1.0 / 60.0,
                &step_at_x,
                &mut eye,
                Vec3::new(5.0, 0.0, 0.0),
                false,
                config.dev.gravity,
                config.dev.jump_speed,
            );
        }

        assert!(eye.x > 4.5, "Spieler haette die 1-Block-Stufe ueberwinden sollen, x blieb bei {}", eye.x);
        let expected_eye_y = 1.0 + config.dev.player_eye_height;
        assert!(
            (eye.y - expected_eye_y).abs() < 0.01,
            "Spieler sollte flush auf der angehobenen Oberflaeche stehen, eye.y={} erwartet={}",
            eye.y,
            expected_eye_y
        );
    }

    /// Ein Bordstein hoeher als `step_height` bleibt eine echte Wand - Auto-Step darf keine
    /// Kollisionspruefung fuer beliebig hohe Klippen aushebeln.
    #[test]
    fn tall_ledge_still_blocks_movement() {
        let wall_at_x = |x: i32, y: i32, _z: i32| if x < 4 { y < 0 } else { y < 3 };

        let config = EngineConfig::default();
        let mut physics = PlayerPhysics::new(false, &config);
        let mut eye = Vec3::new(0.0, config.dev.player_eye_height, 0.0);

        for _ in 0..30 {
            physics.advance(1.0 / 60.0, &wall_at_x, &mut eye, Vec3::ZERO, false, config.dev.gravity, config.dev.jump_speed);
        }

        for _ in 0..120 {
            physics.advance(
                1.0 / 60.0,
                &wall_at_x,
                &mut eye,
                Vec3::new(5.0, 0.0, 0.0),
                false,
                config.dev.gravity,
                config.dev.jump_speed,
            );
        }

        assert!(eye.x < 4.0, "Eine 3-Block-Wand sollte den Spieler weiterhin blockieren, x={}", eye.x);
    }
}
