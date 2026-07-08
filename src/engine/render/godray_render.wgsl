const MAX_SHADOW_CASCADES: u32 = 4u;

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
    params: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> rays: array<GodrayInstance>;
@group(0) @binding(1) var<uniform> u: GodrayUniform;
@group(0) @binding(2) var depth_texture: {DEPTH_TEXTURE_TYPE};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) intensity: f32,
};

const CORNER_OFFSETS: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0),
    vec2<f32>(1.0, 0.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(0.0, 1.0),
);

/// Zylindrisches Billboard ENTLANG DER TATSAECHLICHEN LICHTRICHTUNG (nicht mehr fix vertikal mit
/// kleiner Scherung): die Strahl-Achse ist `light_dir` (die Richtung, in die das Sonnenlicht
/// faellt), nur die Breiten-Achse rotiert um diese Achse zur Kamera. Dadurch stehen die Strahlen
/// tatsaechlich im Sonnenwinkel (flach bei tiefstehender Sonne, steil bei hochstehender) statt immer
/// annaehernd senkrecht zu wirken. Der Start-/"helle" Punkt (`tip`) ist EXAKT derselbe Weltpunkt, an
/// dem der Compute-Pass die Licht/Schatten-Kante erkannt hat (siehe godray_compute.wgsl) - der
/// Strahl "haengt" also sichtbar an einer echten Kante in der Voxel-Geometrie, statt frei im Raum zu
/// stehen.
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance_index: u32) -> VertexOutput {
    var out: VertexOutput;

    let ray = rays[instance_index];
    let intensity = ray.position_intensity.w;
    if intensity <= 0.0 {
        out.clip_position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.uv = vec2<f32>(0.0);
        out.intensity = 0.0;
        return out;
    }

    let base = ray.position_intensity.xyz;
    let width = ray.size.x;
    let beam_length = ray.size.y;
    let sample_height = ray.size.z;

    // Identisch zum Kantenerkennungs-Punkt im Compute-Pass - Ursprung des sichtbaren Strahls.
    let tip = base + vec3<f32>(0.0, sample_height, 0.0);
    let light_dir = normalize(-u.sun_direction_to_sun.xyz);
    let beam_end = tip + light_dir * beam_length;

    let view_dir = normalize(u.camera_pos.xyz - tip);
    var right = cross(light_dir, view_dir);
    if dot(right, right) < 1e-6 {
        right = vec3<f32>(1.0, 0.0, 0.0);
    }
    right = normalize(right) * width * 0.5;

    let corner = CORNER_OFFSETS[vertex_index % 6u];
    let side = (corner.x * 2.0 - 1.0) * right;
    let world_pos = mix(tip, beam_end, corner.y) + side;

    out.clip_position = u.view_proj * vec4<f32>(world_pos, 1.0);
    out.uv = vec2<f32>(corner.x, corner.y);
    out.intensity = intensity;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if in.intensity <= 0.0 {
        discard;
    }

    // Manueller Reverse-Z-Depth-Test (GreaterEqual): dieser Pass laeuft spaet in der Pipeline
    // (nach dem SSAO/Blur-Composite, siehe Kommentar in context.rs::render), wo Farb- und
    // Tiefenziel unterschiedliche Sample-Counts haben (aufgeloeste Swapchain vs. multisampled
    // Depth) - eine echte Depth-Stencil-Attachment-Bindung waere dort ein Sample-Count-Mismatch.
    // Stattdessen wird die Tiefe als Textur gebunden und `GreaterEqual` von Hand nachgebildet.
    let pixel_coord = vec2<i32>(in.clip_position.xy);
    let stored_depth = textureLoad(depth_texture, pixel_coord, 0);
    if in.clip_position.z < stored_depth {
        discard;
    }

    let edge_fade = 1.0 - abs(in.uv.x * 2.0 - 1.0);
    // Hell an der Kante/Spitze (uv.y = 0, wo der Compute-Pass den Licht/Schatten-Uebergang erkannt
    // hat), verblasst zum anderen Ende des Strahls (uv.y = 1) - "faellt entlang des Strahls ab".
    let vertical_fade = 1.0 - in.uv.y;
    let alpha = in.intensity * edge_fade * vertical_fade;
    if alpha <= 0.001 {
        discard;
    }

    return vec4<f32>(u.sun_color_intensity.rgb, alpha);
}
