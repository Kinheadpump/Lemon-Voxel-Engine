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
@group(0) @binding(3) var<storage, read> chunk_origins: array<vec2<f32>>;
@group(0) @binding(4) var block_textures: texture_2d_array<f32>;
@group(0) @binding(5) var block_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) tex_layer: u32,
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
    let tex_layer = (packed >> 15u) & 0x7Fu;
    let width = f32(((packed >> 22u) & 0x1Fu) + 1u);
    let height = f32(((packed >> 27u) & 0x1Fu) + 1u);

    let local_pos = vec3<f32>(local_x, local_y, local_z);
    let plane_offset = max(direction.normal.xyz, vec3<f32>(0.0));
    let chunk_xz = chunk_origins[instance_index];
    let chunk_origin = vec3<f32>(chunk_xz.x, 0.0, chunk_xz.y);
    let origin = chunk_origin + local_pos + plane_offset;

    let corner = CORNER_OFFSETS[vertex_index % 6u];
    let world_pos = origin
        + direction.u_axis.xyz * corner.x * width
        + direction.v_axis.xyz * corner.y * height;

    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.uv = vec2<f32>(corner.x * width, corner.y * height);
    out.tex_layer = tex_layer;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(block_textures, block_sampler, in.uv, i32(in.tex_layer));
}
