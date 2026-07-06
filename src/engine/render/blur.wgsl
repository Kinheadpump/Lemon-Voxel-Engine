struct BlurParams {
    // x = Breite, y = Hoehe, z = NDC-Tiefenschwelle (Kantenerkennung), w = ungenutzt.
    screen_size_depth_threshold: vec4<f32>,
};

@group(0) @binding(0) var ao_texture: texture_2d<f32>;
@group(0) @binding(1) var depth_texture: texture_depth_multisampled_2d;
@group(0) @binding(2) var color_texture: texture_2d<f32>;
@group(0) @binding(3) var<uniform> params: BlurParams;

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

const RADIUS: i32 = 2;

/// 5x5 Bilateral-("Edge-Preserving")-Blur ueber den rohen AO-Faktor: Nachbar-Texel mit stark
/// abweichender Tiefe gelten als andere Oberflaeche und fliessen mit Gewicht 0 ein - das verhindert
/// Aufhellen/Verdunkeln ueber Geometriekanten hinweg, waehrend das Rauschen innerhalb einer
/// Oberflaeche weggemittelt wird. Bewusst NDC-Tiefe statt rekonstruierter View-Space-Tiefe als
/// Distanzmass (billiger: kein Matrix-Multiply pro Tap) - fuer einen Denoise-Pass reicht die grobe
/// Naeherung, echte Kantenerkennung braucht hier keine metrische Genauigkeit.
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let screen_size = params.screen_size_depth_threshold.xy;
    let depth_threshold = params.screen_size_depth_threshold.z;

    let center_pixel = vec2<i32>(in.uv * screen_size);
    let base_color = textureLoad(color_texture, center_pixel, 0);
    let center_depth = textureLoad(depth_texture, center_pixel, 0);

    if center_depth <= 0.0 {
        return base_color;
    }

    var sum = 0.0;
    var weight_total = 0.0;
    for (var dy = -RADIUS; dy <= RADIUS; dy++) {
        for (var dx = -RADIUS; dx <= RADIUS; dx++) {
            let sample_pixel = center_pixel + vec2<i32>(dx, dy);
            let sample_depth = textureLoad(depth_texture, sample_pixel, 0);
            if abs(sample_depth - center_depth) > depth_threshold {
                continue;
            }
            sum += textureLoad(ao_texture, sample_pixel, 0).r;
            weight_total += 1.0;
        }
    }

    let ao = select(1.0, sum / weight_total, weight_total > 0.0);
    return vec4<f32>(base_color.rgb * ao, base_color.a);
}
