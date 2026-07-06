use glam::{Mat4, Vec3};

use super::camera::Camera;

/// Obergrenze fuer Kaskaden - Shadow-Depth-Array und WGSL-Uniform-Arrays sind fest auf diese
/// Groesse allokiert. `EngineConfig::shadow_cascade_count` waehlt, wie viele davon tatsaechlich
/// genutzt werden (3 oder 4); ungenutzte Slots bleiben unbefuellt und werden per `split_far =
/// f32::MAX` nie als Treffer ausgewaehlt.
pub const MAX_SHADOW_CASCADES: usize = 4;

#[derive(Clone, Copy)]
pub struct Cascade {
    pub view_proj: Mat4,
    /// Entfernung von der Kamera entlang ihres Forward-Vektors, bis zu der diese Kaskade zustaendig
    /// ist - im Fragment-Shader direkt mit `dot(camera_forward, world_pos - camera_pos)`
    /// vergleichbar, ohne die Hauptkamera-Tiefe (Reverse-Z) rekonstruieren zu muessen.
    pub split_far: f32,
}

impl Default for Cascade {
    fn default() -> Self {
        Self { view_proj: Mat4::IDENTITY, split_far: f32::MAX }
    }
}

/// Practical-Split-Scheme (Zhang et al.): mischt logarithmische und lineare Aufteilung. Log
/// gewichtet mehr Aufloesung nahe der Kamera - genau dort, wo einzelne Voxel-Kanten den groessten
/// Anteil am Bildschirm einnehmen.
fn compute_split_distances(cascade_count: usize, near: f32, far: f32, lambda: f32) -> [f32; MAX_SHADOW_CASCADES] {
    let mut splits = [far; MAX_SHADOW_CASCADES];
    for (i, split) in splits.iter_mut().enumerate().take(cascade_count) {
        let p = (i + 1) as f32 / cascade_count as f32;
        let log = near * (far / near).powf(p);
        let uniform = near + (far - near) * p;
        *split = lambda * log + (1.0 - lambda) * uniform;
    }
    splits
}

/// Berechnet pro Kaskade eine texel-snapped orthografische Licht-View-Projection, passend zum
/// Kamera-Sichtkegel. Jede Kaskade deckt einen Tiefenabschnitt `[split_near, split_far]` entlang
/// der Kamera ab; die engste Kaskade bekommt die hoechste effektive Aufloesung.
pub fn compute_cascades(
    camera: &Camera,
    aspect: f32,
    light_direction: Vec3,
    cascade_count: u32,
    max_distance: f32,
    split_lambda: f32,
    shadow_map_resolution: u32,
) -> [Cascade; MAX_SHADOW_CASCADES] {
    let cascade_count = (cascade_count as usize).clamp(1, MAX_SHADOW_CASCADES);
    let splits = compute_split_distances(cascade_count, camera.near, max_distance, split_lambda);

    let forward = camera.forward();
    let right = camera.right();
    let up = right.cross(forward);

    let tan_half_v = (camera.fov_y_radians * 0.5).tan();
    let tan_half_h = tan_half_v * aspect;

    let light_up = if light_direction.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };

    let mut cascades = [Cascade::default(); MAX_SHADOW_CASCADES];
    let mut split_near = camera.near;

    for i in 0..cascade_count {
        let split_far = splits[i];

        let mut corners = [Vec3::ZERO; 8];
        for (slot, &depth) in [split_near, split_far].iter().enumerate() {
            let half_v = depth * tan_half_v;
            let half_h = depth * tan_half_h;
            let base = camera.position + forward * depth;
            corners[slot * 4] = base + right * half_h + up * half_v;
            corners[slot * 4 + 1] = base + right * half_h - up * half_v;
            corners[slot * 4 + 2] = base - right * half_h + up * half_v;
            corners[slot * 4 + 3] = base - right * half_h - up * half_v;
        }

        let center = corners.iter().fold(Vec3::ZERO, |acc, &c| acc + c) / corners.len() as f32;
        let radius = corners.iter().map(|&c| center.distance(c)).fold(0.0f32, f32::max).max(0.1);

        let eye = center - light_direction * radius * 2.0;
        let light_view = glam::camera::rh::view::look_to_mat4(eye, light_direction, light_up);

        // Texel-Snapping: der Kaskaden-Mittelpunkt wandert kontinuierlich mit der Kamera, das
        // Shadow-Map-Texelraster darf das aber nicht - sonst "schwimmt" die Sub-Texel-Position des
        // Rasters von Frame zu Frame und erzeugt das beruechtigte Shimmering auf den Kanten. Da die
        // Sphere-Radius pro Kaskade bei fixer Kamera-FOV/Split konstant bleibt, reicht es, das
        // Projektions-Fenster auf ein Vielfaches der Texelgroesse zu runden.
        let texel_size = (radius * 2.0) / shadow_map_resolution as f32;
        let center_light_space = light_view.transform_point3(center);
        let snapped_x = (center_light_space.x / texel_size).floor() * texel_size;
        let snapped_y = (center_light_space.y / texel_size).floor() * texel_size;
        let delta_x = snapped_x - center_light_space.x;
        let delta_y = snapped_y - center_light_space.y;

        let light_proj = glam::camera::rh::proj::directx::orthographic(
            -radius + delta_x,
            radius + delta_x,
            -radius + delta_y,
            radius + delta_y,
            0.0,
            radius * 4.0,
        );

        cascades[i] = Cascade { view_proj: light_proj * light_view, split_far };
        split_near = split_far;
    }

    cascades
}
