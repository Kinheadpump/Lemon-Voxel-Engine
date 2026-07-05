struct CameraUniform {
    view_proj: mat4x4<f32>,
};

struct DirectionUniform {
    normal: vec4<f32>,
    u_axis: vec4<f32>,
    v_axis: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<uniform> direction: DirectionUniform;
@group(0) @binding(2) var<storage, read> faces: array<u32>;
@group(0) @binding(3) var<storage, read> chunk_origins: array<vec4<f32>>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) @interpolate(flat) tex_id: u32,
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
) -> VertexOutput {
    let packed = faces[instance_index];

    let local_x = f32(packed & 0x1Fu);
    let local_y = f32((packed >> 5u) & 0x1Fu);
    let local_z = f32((packed >> 10u) & 0x1Fu);
    let tex_id = (packed >> 15u) & 0x7Fu;
    let width = f32(((packed >> 22u) & 0x1Fu) + 1u);
    let height = f32(((packed >> 27u) & 0x1Fu) + 1u);

    let local_pos = vec3<f32>(local_x, local_y, local_z);
    let plane_offset = max(direction.normal.xyz, vec3<f32>(0.0));
    let chunk_origin = chunk_origins[instance_index].xyz;
    let origin = chunk_origin + local_pos + plane_offset;

    let corner = CORNER_OFFSETS[vertex_index % 6u];
    let world_pos = origin
        + direction.u_axis.xyz * corner.x * width
        + direction.v_axis.xyz * corner.y * height;

    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.tex_id = tex_id;
    return out;
}

fn tex_id_debug_color(id: u32) -> vec3<f32> {
    let r = f32((id * 47u) % 251u) / 251.0;
    let g = f32((id * 97u) % 251u) / 251.0;
    let b = f32((id * 173u) % 251u) / 251.0;
    return vec3<f32>(r, g, b);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(tex_id_debug_color(in.tex_id), 1.0);
}
