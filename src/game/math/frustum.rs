use glam::{Mat4, Vec3, Vec4};

#[derive(Clone, Copy)]
struct Plane {
    normal: Vec3,
    distance: f32,
}

impl Plane {
    fn from_vec4(v: Vec4) -> Self {
        let normal = Vec3::new(v.x, v.y, v.z);
        let length = normal.length();
        Self { normal: normal / length, distance: v.w / length }
    }

    /// Vorzeichenbehafteter Abstand des AABB-Eckpunkts, der in Richtung der Plane-Normale am
    /// weitesten aussen liegt. Negativ bedeutet: die komplette AABB liegt ausserhalb dieser Plane.
    fn signed_distance_to_positive_vertex(&self, min: Vec3, max: Vec3) -> f32 {
        let positive = Vec3::new(
            if self.normal.x >= 0.0 { max.x } else { min.x },
            if self.normal.y >= 0.0 { max.y } else { min.y },
            if self.normal.z >= 0.0 { max.z } else { min.z },
        );
        self.normal.dot(positive) + self.distance
    }
}

/// Kamera-Frustum aus View-Projection-Matrix. Die Far-Plane wird ausgelassen, da die Engine eine
/// Reverse-Z-Projektion mit unendlicher Far-Distanz nutzt (die Far-Plane liegt im Unendlichen und
/// cullt nie).
pub struct Frustum {
    planes: [Plane; 5],
}

impl Frustum {
    pub fn from_view_projection(view_proj: Mat4) -> Self {
        let row0 = view_proj.row(0);
        let row1 = view_proj.row(1);
        let row2 = view_proj.row(2);
        let row3 = view_proj.row(3);

        let planes = [
            Plane::from_vec4(row3 + row0),
            Plane::from_vec4(row3 - row0),
            Plane::from_vec4(row3 + row1),
            Plane::from_vec4(row3 - row1),
            Plane::from_vec4(row3 - row2),
        ];

        Self { planes }
    }

    pub fn intersects_aabb(&self, min: Vec3, max: Vec3) -> bool {
        self.planes.iter().all(|plane| plane.signed_distance_to_positive_vertex(min, max) >= 0.0)
    }
}
