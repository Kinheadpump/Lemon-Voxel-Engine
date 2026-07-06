use glam::{IVec3, Vec3};

pub struct RaycastHit {
    pub block: IVec3,
    /// Auswaerts zeigende Normale der getroffenen Flaeche - `block + normal` ist die
    /// Nachbarposition, an der ein neuer Block platziert werden würde.
    pub normal: IVec3,
}

/// Liefert (Schrittrichtung, Zeit pro vollem Voxel, Zeit bis zur ersten Voxelgrenze) fuer eine
/// Achse. Amanatides-Woo Voxel-DDA statt Sampling in Fixed-Steps - trifft dadurch exakt den
/// ersten Voxel entlang des Strahls, ohne bei hoher `max_distance` durch duenne Waende zu tunneln.
fn axis_dda(origin: f32, voxel: i32, direction: f32) -> (i32, f32, f32) {
    if direction > 0.0 {
        (1, 1.0 / direction, ((voxel + 1) as f32 - origin) / direction)
    } else if direction < 0.0 {
        (-1, 1.0 / -direction, (voxel as f32 - origin) / direction)
    } else {
        (0, f32::INFINITY, f32::INFINITY)
    }
}

pub fn raycast<F: Fn(i32, i32, i32) -> bool>(
    origin: Vec3,
    direction: Vec3,
    max_distance: f32,
    is_solid: F,
) -> Option<RaycastHit> {
    let direction = direction.normalize_or_zero();
    if direction == Vec3::ZERO {
        return None;
    }

    let mut voxel = IVec3::new(origin.x.floor() as i32, origin.y.floor() as i32, origin.z.floor() as i32);

    let (step_x, t_delta_x, mut t_max_x) = axis_dda(origin.x, voxel.x, direction.x);
    let (step_y, t_delta_y, mut t_max_y) = axis_dda(origin.y, voxel.y, direction.y);
    let (step_z, t_delta_z, mut t_max_z) = axis_dda(origin.z, voxel.z, direction.z);

    let mut normal = IVec3::ZERO;
    let mut traveled = 0.0;

    while traveled <= max_distance {
        if is_solid(voxel.x, voxel.y, voxel.z) {
            return Some(RaycastHit { block: voxel, normal });
        }

        if t_max_x < t_max_y && t_max_x < t_max_z {
            voxel.x += step_x;
            traveled = t_max_x;
            t_max_x += t_delta_x;
            normal = IVec3::new(-step_x, 0, 0);
        } else if t_max_y < t_max_z {
            voxel.y += step_y;
            traveled = t_max_y;
            t_max_y += t_delta_y;
            normal = IVec3::new(0, -step_y, 0);
        } else {
            voxel.z += step_z;
            traveled = t_max_z;
            t_max_z += t_delta_z;
            normal = IVec3::new(0, 0, -step_z);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hits_floor_looking_straight_down() {
        let is_solid = |_x: i32, y: i32, _z: i32| y <= 0;
        let hit = raycast(Vec3::new(0.5, 5.0, 0.5), Vec3::new(0.0, -1.0, 0.0), 10.0, is_solid)
            .expect("Strahl muss den Boden treffen");

        assert_eq!(hit.block, IVec3::new(0, 0, 0));
        assert_eq!(hit.normal, IVec3::new(0, 1, 0));
    }

    #[test]
    fn hits_wall_looking_sideways() {
        let is_solid = |x: i32, _y: i32, _z: i32| x >= 3;
        let hit = raycast(Vec3::new(0.5, 0.5, 0.5), Vec3::new(1.0, 0.0, 0.0), 10.0, is_solid)
            .expect("Strahl muss die Wand treffen");

        assert_eq!(hit.block, IVec3::new(3, 0, 0));
        assert_eq!(hit.normal, IVec3::new(-1, 0, 0));
    }

    #[test]
    fn misses_when_nothing_within_range() {
        let is_solid = |_x: i32, _y: i32, _z: i32| false;
        assert!(raycast(Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0), 5.0, is_solid).is_none());
    }

    #[test]
    fn zero_direction_returns_none() {
        let is_solid = |_x: i32, _y: i32, _z: i32| true;
        assert!(raycast(Vec3::ZERO, Vec3::ZERO, 5.0, is_solid).is_none());
    }
}
