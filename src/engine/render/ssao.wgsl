const KERNEL_SIZE: u32 = 16u;

struct SsaoParams {
    inverse_projection: mat4x4<f32>,
    projection: mat4x4<f32>,
    screen_size_radius_strength: vec4<f32>,
    enabled: vec4<u32>,
    kernel: array<vec4<f32>, KERNEL_SIZE>,
};

@group(0) @binding(0) var depth_texture: texture_depth_multisampled_2d;
@group(0) @binding(1) var color_texture: texture_2d<f32>;
@group(0) @binding(2) var color_sampler: sampler;
@group(0) @binding(3) var<uniform> params: SsaoParams;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let uv = vec2<f32>(f32((vertex_index << 1u) & 2u), f32(vertex_index & 2u));
    var out: VertexOutput;
    out.clip_position = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    out.uv = uv;
    return out;
}

fn view_pos_from_uv(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let view = params.inverse_projection * ndc;
    return view.xyz / view.w;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let base_color = textureSample(color_texture, color_sampler, in.uv);

    if params.enabled.x == 0u {
        return base_color;
    }

    let screen_size = params.screen_size_radius_strength.xy;
    let radius = params.screen_size_radius_strength.z;
    let strength = params.screen_size_radius_strength.w;

    let pixel_coord = vec2<i32>(in.uv * screen_size);
    let depth = textureLoad(depth_texture, pixel_coord, 0);

    if depth <= 0.0 {
        return base_color;
    }

    let view_pos = view_pos_from_uv(in.uv, depth);
    let normal_raw = normalize(cross(dpdx(view_pos), dpdy(view_pos)));
    let normal = select(-normal_raw, normal_raw, dot(normal_raw, -view_pos) > 0.0);

    let random_angle =
        fract(sin(dot(in.uv * screen_size, vec2<f32>(12.9898, 78.233))) * 43758.5453) * 6.28318530718;
    let random_vec = vec3<f32>(cos(random_angle), sin(random_angle), 0.0);

    let tangent = normalize(random_vec - normal * dot(random_vec, normal));
    let bitangent = cross(normal, tangent);
    let tbn = mat3x3<f32>(tangent, bitangent, normal);

    var occlusion = 0.0;
    for (var i = 0u; i < KERNEL_SIZE; i++) {
        let sample_view = view_pos + (tbn * params.kernel[i].xyz) * radius;

        let sample_clip = params.projection * vec4<f32>(sample_view, 1.0);
        let sample_ndc = sample_clip.xyz / sample_clip.w;
        let sample_uv = vec2<f32>(sample_ndc.x * 0.5 + 0.5, 0.5 - sample_ndc.y * 0.5);

        if sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0 {
            continue;
        }

        let sample_pixel = vec2<i32>(sample_uv * screen_size);
        let sampled_depth = textureLoad(depth_texture, sample_pixel, 0);
        let sampled_view_pos = view_pos_from_uv(sample_uv, sampled_depth);

        let range_check = smoothstep(0.0, 1.0, radius / max(abs(view_pos.z - sampled_view_pos.z), 0.0001));
        occlusion += select(0.0, 1.0, sampled_view_pos.z >= sample_view.z + 0.025) * range_check;
    }

    let ao = clamp(1.0 - (occlusion / f32(KERNEL_SIZE)) * strength, 0.0, 1.0);
    return vec4<f32>(base_color.rgb * ao, base_color.a);
}
