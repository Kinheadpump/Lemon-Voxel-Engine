/// Downsample-Schritt Mip[N-1] -> Mip[N]: MINIMUM ueber den 2x2-Block (konservativ fuer
/// Reverse-Z-Occlusion-Culling, s. `hzb_copy.wgsl`-Kommentar). Odd-groesse Quellebenen werden am
/// Rand geclampt statt out-of-bounds zu lesen.
@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let out_dims = textureDimensions(output_tex);
    if gid.x >= out_dims.x || gid.y >= out_dims.y {
        return;
    }

    let in_dims = vec2<i32>(textureDimensions(input_tex, 0));
    let max_coord = in_dims - vec2<i32>(1, 1);
    let base = vec2<i32>(gid.xy) * 2;

    let c00 = clamp(base, vec2<i32>(0, 0), max_coord);
    let c10 = clamp(base + vec2<i32>(1, 0), vec2<i32>(0, 0), max_coord);
    let c01 = clamp(base + vec2<i32>(0, 1), vec2<i32>(0, 0), max_coord);
    let c11 = clamp(base + vec2<i32>(1, 1), vec2<i32>(0, 0), max_coord);

    let d00 = textureLoad(input_tex, c00, 0).r;
    let d10 = textureLoad(input_tex, c10, 0).r;
    let d01 = textureLoad(input_tex, c01, 0).r;
    let d11 = textureLoad(input_tex, c11, 0).r;
    let result = min(min(d00, d10), min(d01, d11));

    textureStore(output_tex, vec2<i32>(gid.xy), vec4<f32>(result, 0.0, 0.0, 0.0));
}
