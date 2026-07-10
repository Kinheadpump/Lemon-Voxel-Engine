use glam::{Mat4, Vec3};

use super::camera::Camera;

/// Obergrenze fuer Kaskaden - Shadow-Depth-Array und WGSL-Uniform-Arrays sind fest auf diese
/// Groesse allokiert. `EngineConfig::shadow_cascade_count` waehlt, wie viele davon tatsaechlich
/// genutzt werden (3 oder 4); ungenutzte Slots bleiben unbefuellt und werden per `split_far =
/// f32::MAX` nie als Treffer ausgewaehlt.
pub const MAX_SHADOW_CASCADES: usize = 4;

/// Feste Sicherheitsmarge (Weltbloecke) auf den geometrisch berechneten Kaskaden-Radius. Der reine
/// Kamera-Frustum-Ausschnitt deckt NUR ab, was aktuell sichtbar ist - eine Hoehlendecke direkt ueber
/// dem Spieler, der seitwaerts durch einen Tunnel schaut, liegt haeufig ausserhalb dieses Kegels
/// (das Sichtfeld zeigt nach vorne, nicht nach oben) und wuerde ohne diese Marge NIE als
/// Schatten-Werfer gezeichnet - Sonnenlicht "leckt" dann durch eine Felsdecke, die nirgends in der
/// Shadow-Map existiert (s. `sample_shadow`s Fallback "ausserhalb des Kaskaden-Frustums = unbeschattet").
/// Additiv statt multiplikativ, damit sie nicht mit der (bei fernen Kaskaden potenziell riesigen)
/// Sichtweite mitwaechst - kostet dafuer bei der naechsten (kleinsten) Kaskade anteilig etwas
/// Texel-Aufloesung, das ist der Preis fuer "keine Luecken in geschlossenen Raeumen".
const CASCADE_CASTER_MARGIN: f32 = 24.0;

#[derive(Clone, Copy)]
pub struct Cascade {
    pub view_proj: Mat4,
    /// Entfernung von der Kamera entlang ihres Forward-Vektors, bis zu der diese Kaskade zustaendig
    /// ist - im Fragment-Shader direkt mit `dot(camera_forward, world_pos - camera_pos)`
    /// vergleichbar, ohne die Hauptkamera-Tiefe (Reverse-Z) rekonstruieren zu muessen.
    pub split_far: f32,
    /// Umschliessende Kugel des Kamera-Frustum-Ausschnitts dieser Kaskade, in Weltkoordinaten -
    /// fuer die CPU-seitige Schatten-Sichtbarkeitspruefung in `ChunkManager` (Sphere-vs-AABB gegen
    /// alle geladenen Chunks, NICHT gegen das Kamera-Frustum - siehe dortigen Kommentar).
    pub center: Vec3,
    pub radius: f32,
}

impl Default for Cascade {
    fn default() -> Self {
        Self {
            view_proj: Mat4::IDENTITY,
            split_far: f32::MAX,
            center: Vec3::ZERO,
            radius: 0.0,
        }
    }
}

/// Practical-Split-Scheme (Zhang et al.): mischt logarithmische und lineare Aufteilung. Log
/// gewichtet mehr Aufloesung nahe der Kamera - genau dort, wo einzelne Voxel-Kanten den groessten
/// Anteil am Bildschirm einnehmen.
fn compute_split_distances(
    cascade_count: usize,
    near: f32,
    far: f32,
    lambda: f32,
) -> [f32; MAX_SHADOW_CASCADES] {
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

    let light_up = if light_direction.y.abs() > 0.99 {
        Vec3::Z
    } else {
        Vec3::Y
    };

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
        // Umschliessende Kugel (rotationsinvariant): unter reiner Kamerarotation dreht sich der
        // Frustum-Ausschnitt starr um die Kamera, die Eckdistanzen zum Schwerpunkt bleiben also
        // gleich - der Radius (und damit die Texelgroesse) ist von der Blickrichtung unabhaengig.
        // Plus `CASCADE_CASTER_MARGIN` (s. dortiger Kommentar) gegen Schatten-Lecks in geschlossenen
        // Raeumen ausserhalb des reinen Sichtkegels.
        let radius = corners
            .iter()
            .map(|&c| center.distance(c))
            .fold(0.0f32, f32::max)
            .max(0.1)
            + CASCADE_CASTER_MARGIN;
        let texel_size = (radius * 2.0) / shadow_map_resolution as f32;

        // Texel-Snapping muss in einem BLICKRICHTUNGS-UNABHAENGIGEN Licht-Frame passieren. Zuvor
        // wurde im Frame der finalen Licht-View gesnappt, deren `eye` selbst am wandernden `center`
        // klebte - `center` lag darin quasi konstant bei (0,0,-2r), das Snapping war also praktisch
        // wirkungslos und die Schatten "schwammen" bei jeder Kopfdrehung. Jetzt: Kugelmittelpunkt in
        // eine reine Rotations-View (Augpunkt im Ursprung) projizieren, dort aufs Texelraster runden
        // und zurueck in Weltkoordinaten - so rastet das Schattenraster stabil ein.
        let light_rotation =
            glam::camera::rh::view::look_to_mat4(Vec3::ZERO, light_direction, light_up);
        let center_light_space = light_rotation.transform_point3(center);
        let snapped_light_space = Vec3::new(
            (center_light_space.x / texel_size).round() * texel_size,
            (center_light_space.y / texel_size).round() * texel_size,
            center_light_space.z,
        );
        let snapped_center = light_rotation
            .inverse()
            .transform_point3(snapped_light_space);

        let eye = snapped_center - light_direction * radius * 2.0;
        let light_view = glam::camera::rh::view::look_to_mat4(eye, light_direction, light_up);
        let light_proj = glam::camera::rh::proj::directx::orthographic(
            -radius,
            radius,
            -radius,
            radius,
            0.0,
            radius * 4.0,
        );

        cascades[i] = Cascade {
            view_proj: light_proj * light_view,
            split_far,
            center: snapped_center,
            radius,
        };
        split_near = split_far;
    }

    cascades
}
