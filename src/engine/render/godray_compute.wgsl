const MAX_SHADOW_CASCADES: u32 = 4u;
const SAMPLE_COUNT: u32 = 16u;

struct GodrayInstance {
    position_intensity: vec4<f32>,
    size: vec4<f32>,
};

struct GodrayUniform {
    view_proj: mat4x4<f32>,
    cascade_view_proj: array<mat4x4<f32>, MAX_SHADOW_CASCADES>,
    cascade_split_far: vec4<f32>,
    camera_pos: vec4<f32>,
    camera_forward: vec4<f32>,
    sun_direction_to_sun: vec4<f32>,
    sun_color_intensity: vec4<f32>,
    // x = Kaskaden-Anzahl, y = Ray-Anzahl, z = Temporal-Blend-Faktor, w = ungenutzt.
    params: vec4<f32>,
};

@group(0) @binding(0) var<storage, read_write> rays: array<GodrayInstance>;
@group(0) @binding(1) var<uniform> u: GodrayUniform;
@group(0) @binding(2) var shadow_maps: texture_depth_2d_array;

/// 4x4-Gitter in [-1, 1]^2, skaliert mit der Billboard-Breite - deckt Kanten in beliebiger
/// Orientierung ab (nicht nur entlang einer festen Achse), da Terrain-Silhouetten (Bergkaemme,
/// Hoehleneingaenge) beliebig ausgerichtet sein koennen.
const SAMPLE_OFFSETS: array<vec2<f32>, 16> = array<vec2<f32>, 16>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(-0.33, -1.0), vec2<f32>(0.33, -1.0), vec2<f32>(1.0, -1.0),
    vec2<f32>(-1.0, -0.33), vec2<f32>(-0.33, -0.33), vec2<f32>(0.33, -0.33), vec2<f32>(1.0, -0.33),
    vec2<f32>(-1.0, 0.33), vec2<f32>(-0.33, 0.33), vec2<f32>(0.33, 0.33), vec2<f32>(1.0, 0.33),
    vec2<f32>(-1.0, 1.0), vec2<f32>(-0.33, 1.0), vec2<f32>(0.33, 1.0), vec2<f32>(1.0, 1.0),
);

/// Kaskaden-Auswahl ueber Kamera-Vorwaerts-Distanz - identische Metrik zu `shader.wgsl`, damit
/// Compute- und Main-Pass-Sampling konsistent dieselbe Kaskade fuer denselben Weltpunkt waehlen.
fn select_cascade(world_pos: vec3<f32>) -> i32 {
    let view_depth = dot(u.camera_forward.xyz, world_pos - u.camera_pos.xyz);
    let cascade_count = u32(u.params.x);
    for (var i = 0u; i < cascade_count; i++) {
        if view_depth <= u.cascade_split_far[i] {
            return i32(i);
        }
    }
    return -1;
}

/// Ausserhalb der Schatten-Reichweite/des Kaskaden-Frustums gilt ein Punkt als beleuchtet - das
/// vermeidet falsch-positive Kantenerkennung am Rand der Schatten-Distanz.
fn is_lit(world_pos: vec3<f32>) -> bool {
    let cascade = select_cascade(world_pos);
    if cascade < 0 {
        return true;
    }

    let light_clip = u.cascade_view_proj[cascade] * vec4<f32>(world_pos, 1.0);
    let light_ndc = light_clip.xyz / light_clip.w;
    if light_ndc.x < -1.0 || light_ndc.x > 1.0 || light_ndc.y < -1.0 || light_ndc.y > 1.0 {
        return true;
    }

    let shadow_uv = vec2<f32>(light_ndc.x * 0.5 + 0.5, 0.5 - light_ndc.y * 0.5);
    let dims = vec2<f32>(textureDimensions(shadow_maps));
    let texel = vec2<i32>(shadow_uv * dims);
    let stored_depth = textureLoad(shadow_maps, texel, cascade, 0);
    return light_ndc.z <= stored_depth;
}

/// Nimmt 16 Samples am oberen Rand (der simulierten Spitze) des Strahls und zaehlt, wie viele
/// beleuchtet sind. Die Intensity ist maximal (1.0) bei exakt 8/16 (Kante trifft die Spitze mittig),
/// faellt zu beiden Seiten linear ab und ist bei 0/16 oder 16/16 (kein Kantenuebergang) exakt 0.0.
@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if index >= u32(u.params.y) {
        return;
    }

    let ray = rays[index];
    let tip = ray.position_intensity.xyz + vec3<f32>(0.0, ray.size.y, 0.0);
    let radius = ray.size.x * 0.5;

    var lit_count = 0u;
    for (var i = 0u; i < SAMPLE_COUNT; i++) {
        let offset = SAMPLE_OFFSETS[i] * radius;
        let sample_pos = tip + vec3<f32>(offset.x, 0.0, offset.y);
        if is_lit(sample_pos) {
            lit_count += 1u;
        }
    }

    let computed = 1.0 - abs(f32(lit_count) - 8.0) / 8.0;
    let previous = ray.position_intensity.w;
    let blend = u.params.z;

    rays[index].position_intensity.w = mix(previous, computed, blend);
}
