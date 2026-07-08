/// Fuellt Mip 0 der HZB-Pyramide aus dem Haupt-Depth-Buffer des VORHERIGEN Frames. Reverse-Z:
/// Near=1.0/Far=0.0, also ist der kleinste Wert der konservativste (am weitesten entfernte)
/// Naeherungswert - bei aktivem MSAA wird deshalb ueber alle Samples das Minimum genommen statt
/// zu mitteln (Mitteln waere fuer eine konservative Occlusion-Schranke falsch).
@group(0) @binding(0) var depth_tex: {DEPTH_TEXTURE_TYPE};
@group(0) @binding(1) var out_tex: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(out_tex);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }
    let coord = vec2<i32>(gid.xy);

    var d = 1.0;
    {SAMPLE_LOOP}

    textureStore(out_tex, coord, vec4<f32>(d, 0.0, 0.0, 0.0));
}
