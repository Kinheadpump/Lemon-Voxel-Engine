enable draw_index;

struct CameraUniform {
    view_proj: mat4x4<f32>,
    debug_mode: vec4<u32>,
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

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<uniform> direction: DirectionUniform;
@group(0) @binding(2) var<storage, read> faces: array<u32>;
@group(0) @binding(3) var<storage, read> chunk_data: array<ChunkData>;
@group(0) @binding(4) var block_textures: texture_2d_array<f32>;
@group(0) @binding(5) var block_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) tex_layer: u32,
    @location(2) @interpolate(flat) quad_size: vec2<f32>,
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

    let local_pos = vec3<f32>(local_x, local_y, local_z);
    let plane_offset = max(direction.normal.xyz, vec3<f32>(0.0));
    let chunk_origin = chunk_data[draw_index].origin.xyz;
    let origin = chunk_origin + local_pos + plane_offset;

    let corner = CORNER_OFFSETS[vertex_index % 6u];
    let world_pos = origin
        + direction.u_axis.xyz * corner.x * width
        + direction.v_axis.xyz * corner.y * height;

    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.uv = vec2<f32>(corner.x * width, corner.y * height);
    out.tex_layer = tex_layer;
    out.quad_size = vec2<f32>(width, height);
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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if camera.debug_mode.x != 0u {
        let uv_normalized = in.uv / in.quad_size;
        let edge = wireframe_edge_factor(uv_normalized);
        return vec4<f32>(vec3<f32>(edge), 1.0);
    }

    return textureSample(block_textures, block_sampler, in.uv, i32(in.tex_layer));
}
