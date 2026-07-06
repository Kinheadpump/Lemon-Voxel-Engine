enable draw_index;

struct DirectionUniform {
    normal: vec4<f32>,
    u_axis: vec4<f32>,
    v_axis: vec4<f32>,
};

struct ChunkData {
    origin: vec4<f32>,
};

struct Immediates {
    light_view_proj: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> direction: DirectionUniform;
@group(0) @binding(1) var<storage, read> faces: array<u32>;
@group(0) @binding(2) var<storage, read> chunk_data: array<ChunkData>;
var<immediate> im: Immediates;

const CORNER_OFFSETS: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0),
    vec2<f32>(1.0, 0.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(0.0, 1.0),
);

/// Geometrie-Rekonstruktion identisch zu `shader.wgsl::vs_main` - der Schatten-Pass zeichnet
/// dieselben gepackten Face-Instanzen, nur mit der Licht- statt der Kamera-Projektion.
@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
    @builtin(draw_index) draw_index: u32,
) -> @builtin(position) vec4<f32> {
    let packed = faces[instance_index];

    let local_x = f32(packed & 0x1Fu);
    let local_y = f32((packed >> 5u) & 0x1Fu);
    let local_z = f32((packed >> 10u) & 0x1Fu);
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

    return im.light_view_proj * vec4<f32>(world_pos, 1.0);
}
