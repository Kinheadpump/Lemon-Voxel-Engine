struct SkyboxParams {
    inverse_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    direction_to_sun: vec4<f32>,
    zenith_day: vec4<f32>,
    horizon_day: vec4<f32>,
    night: vec4<f32>,
};

@group(0) @binding(0) var depth_texture: {DEPTH_TEXTURE_TYPE};
@group(0) @binding(1) var<uniform> params: SkyboxParams;

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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(depth_texture));
    let pixel_coord = vec2<i32>(in.uv * dims);

    // Reverse-Z: der Opaque-Pass clearet auf 0.0 ("unendlich fern"). Alles > 0.0 hat also bereits
    // Geometrie - dort darf der Himmel nichts ueberschreiben.
    if textureLoad(depth_texture, pixel_coord, 0) > 0.0 {
        discard;
    }

    let ndc_xy = vec2<f32>(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
    let world_h = params.inverse_view_proj * vec4<f32>(ndc_xy, 0.0, 1.0);
    let world_point = world_h.xyz / world_h.w;
    let ray_dir = normalize(world_point - params.camera_pos.xyz);

    let sun_dir = normalize(params.direction_to_sun.xyz);
    // Weiches Ein-/Ausblenden um die Horizontlinie statt hartem Tag/Nacht-Schnitt bei sun_dir.y = 0.
    let day_t = clamp(sun_dir.y * 2.0 + 0.3, 0.0, 1.0);

    let horizon_to_zenith = smoothstep(0.0, 0.6, clamp(ray_dir.y, 0.0, 1.0));
    let day_sky = mix(params.horizon_day.rgb, params.zenith_day.rgb, horizon_to_zenith);
    var sky_color = mix(params.night.rgb, day_sky, day_t);

    let sun_glow = pow(max(dot(ray_dir, sun_dir), 0.0), 256.0) * day_t;
    sky_color += sun_glow * vec3<f32>(1.0, 0.9, 0.7);

    return vec4<f32>(sky_color, 1.0);
}
