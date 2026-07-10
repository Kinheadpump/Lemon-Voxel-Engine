enable draw_index;

const MAX_SHADOW_CASCADES: u32 = 4u;

struct CameraUniform {
    view_proj: mat4x4<f32>,
    debug_mode: vec4<u32>,
    camera_pos: vec4<f32>,
    camera_forward: vec4<f32>,
};

struct DirectionUniform {
    normal: vec4<f32>,
    u_axis: vec4<f32>,
    v_axis: vec4<f32>,
};

/// Ein Eintrag pro Indirect-Draw-Aufruf (also ein Eintrag pro sichtbarem Chunk in dieser
/// Richtung), adressiert per `@builtin(draw_index)` - nicht pro Face wie zuvor. Reduziert die
/// Chunk-Origin-Speicherlast von O(Faces) auf O(sichtbare Chunks).
struct ChunkData {
    origin: vec4<f32>,
};

struct LightingUniform {
    cascade_view_proj: array<mat4x4<f32>, MAX_SHADOW_CASCADES>,
    cascade_split_far: vec4<f32>,
    sun_direction: vec4<f32>,
    sun_color_intensity: vec4<f32>,
    ambient_count_resolution: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<uniform> direction: DirectionUniform;
@group(0) @binding(2) var<storage, read> faces: array<u32>;
@group(0) @binding(3) var<storage, read> chunk_data: array<ChunkData>;
@group(0) @binding(4) var block_textures: texture_2d_array<f32>;
@group(0) @binding(5) var block_sampler: sampler;
@group(0) @binding(6) var<uniform> lighting: LightingUniform;
@group(0) @binding(7) var shadow_maps: texture_depth_2d_array;
@group(0) @binding(8) var shadow_sampler: sampler_comparison;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) tex_layer: u32,
    @location(2) @interpolate(flat) quad_size: vec2<f32>,
    @location(3) world_pos: vec3<f32>,
};

const CORNER_OFFSETS: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0),
    vec2<f32>(1.0, 0.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(0.0, 1.0),
);

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
    @builtin(draw_index) draw_index: u32,
) -> VertexOutput {
    let packed = faces[instance_index];

    let local_x = f32(packed & 0x1Fu);
    let local_y = f32((packed >> 5u) & 0x1Fu);
    let local_z = f32((packed >> 10u) & 0x1Fu);
    let tex_layer = (packed >> 15u) & 0x7Fu;
    let width = f32(((packed >> 22u) & 0x1Fu) + 1u);
    let height = f32(((packed >> 27u) & 0x1Fu) + 1u);

    // `origin.w` = voxel_scale (LOD-Ring-Faktor, s. `ChunkMetaGpu`-Kommentar in `cull_pipeline.rs`) -
    // 1 fuer LOD0. EIN Voxel dieses Chunks deckt `voxel_scale` Weltbloecke pro Achse ab, also
    // skaliert die GESAMTE lokale Geometrie (Position, Face-Breite/-Hoehe) damit - die Textur-UVs
    // bleiben bewusst unskaliert (grobe LOD-Chunks brauchen keine Pro-Block-Texturschaerfe).
    let voxel_scale = chunk_data[draw_index].origin.w;
    let local_pos = vec3<f32>(local_x, local_y, local_z) * voxel_scale;
    let plane_offset = max(direction.normal.xyz, vec3<f32>(0.0)) * voxel_scale;
    let chunk_origin = chunk_data[draw_index].origin.xyz;
    let origin = chunk_origin + local_pos + plane_offset;

    let corner = CORNER_OFFSETS[vertex_index % 6u];
    let world_pos = origin
        + direction.u_axis.xyz * corner.x * width * voxel_scale
        + direction.v_axis.xyz * corner.y * height * voxel_scale;

    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.uv = vec2<f32>(corner.x * width, corner.y * height);
    out.tex_layer = tex_layer;
    out.quad_size = vec2<f32>(width, height);
    out.world_pos = world_pos;
    return out;
}

/// Distanz zum naechsten Rand oder zur Dreiecks-Diagonale des Greedy-Mesh-Quads, in normierten
/// (0..1) Quad-Koordinaten. Zeigt die tatsaechlichen Mesh-Dreiecke (inkl. Diagonale), nicht ein
/// Pro-Block-Gitter - so wird sichtbar, ob Greedy Meshing grosse Rechtecke erzeugt.
fn wireframe_edge_factor(uv_normalized: vec2<f32>) -> f32 {
    let to_edge = min(uv_normalized, vec2<f32>(1.0) - uv_normalized);
    let to_diagonal = abs(uv_normalized.x - uv_normalized.y);
    let nearest = min(min(to_edge.x, to_edge.y), to_diagonal);
    let line_width = max(fwidth(nearest) * 1.5, 0.001);
    return 1.0 - smoothstep(0.0, line_width, nearest);
}

/// Waehlt die naechstgelegene Kaskade ueber die Kamera-Vorwaerts-Distanz (nicht die
/// Reverse-Z-NDC-Tiefe der Hauptkamera - die CPU-Seite splittet ebenfalls nach dieser Metrik, s.
/// `compute_cascades`), transformiert `world_pos` in deren Licht-Clip-Raum und mittelt ein 3x3
/// Percentage-Closer-Filter aus dem Hardware-Vergleichs-Sampler. Ausserhalb der maximalen
/// Schatten-Distanz oder ausserhalb des Kaskaden-Frustums gilt eine Position als unbeschattet -
/// das vermeidet Randartefakte durch die Bounding-Sphere-Naeherung der Kaskaden.
fn sample_shadow(world_pos: vec3<f32>, n_dot_l: f32) -> f32 {
    let view_depth = dot(camera.camera_forward.xyz, world_pos - camera.camera_pos.xyz);
    let cascade_count = u32(lighting.ambient_count_resolution.y);

    var cascade = 0u;
    var found = false;
    for (var i = 0u; i < cascade_count; i++) {
        if view_depth <= lighting.cascade_split_far[i] {
            cascade = i;
            found = true;
            break;
        }
    }
    if !found || n_dot_l <= 0.0 {
        return 1.0;
    }

    let light_clip = lighting.cascade_view_proj[cascade] * vec4<f32>(world_pos, 1.0);
    let light_ndc = light_clip.xyz / light_clip.w;
    let shadow_uv = vec2<f32>(light_ndc.x * 0.5 + 0.5, 0.5 - light_ndc.y * 0.5);

    if shadow_uv.x < 0.0 || shadow_uv.x > 1.0 || shadow_uv.y < 0.0 || shadow_uv.y > 1.0
        || light_ndc.z < 0.0 || light_ndc.z > 1.0 {
        return 1.0;
    }

    let texel = 1.0 / lighting.ambient_count_resolution.z;
    var sum = 0.0;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let offset = vec2<f32>(f32(dx), f32(dy)) * texel;
            sum += textureSampleCompareLevel(shadow_maps, shadow_sampler, shadow_uv + offset, cascade, light_ndc.z);
        }
    }
    return sum / 9.0;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if camera.debug_mode.x != 0u {
        let uv_normalized = in.uv / in.quad_size;
        let edge = wireframe_edge_factor(uv_normalized);
        return vec4<f32>(vec3<f32>(edge), 1.0);
    }

    let base = textureSample(block_textures, block_sampler, in.uv, i32(in.tex_layer));

    let n_dot_l = max(dot(direction.normal.xyz, lighting.sun_direction.xyz), 0.0);
    let shadow = sample_shadow(in.world_pos, n_dot_l);
    let sun_light = lighting.sun_color_intensity.rgb * lighting.sun_color_intensity.a * n_dot_l * shadow;
    let ambient = lighting.ambient_count_resolution.x;

    return vec4<f32>(base.rgb * (ambient + sun_light), base.a);
}
