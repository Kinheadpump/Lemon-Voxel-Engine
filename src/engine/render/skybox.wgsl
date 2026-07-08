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

{SKY_HELPERS}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(depth_texture));
    let pixel_coord = vec2<i32>(in.uv * dims);

    // Reverse-Z: der Opaque-Pass clearet auf 0.0 ("unendlich fern"). Alles > 0.0 hat also bereits
    // Geometrie - dort darf der Himmel nichts ueberschreiben. Bei MSAA MUSS ueber ALLE Samples
    // geprueft werden: an Silhouettenkanten ist Sample 0 oft unbedeckt (0.0), waehrend andere
    // Samples Geometrie treffen. Nur Sample 0 zu testen malte den Himmel ueber die bereits
    // aufgeloeste Kantenfarbe und riss so beim Umschauen flackernde Loecher in die Silhouetten.
    if {DEPTH_HAS_GEOMETRY} {
        discard;
    }

    let ndc_xy = vec2<f32>(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
    // NDC-Z = 1.0 (Near-Ebene in unserer Reverse-Z-Konvention), NICHT 0.0 (Far): bei einer
    // UNENDLICHEN Reverse-Z-Projektion ist Z=0.0 exakt der Punkt im Unendlichen - die inverse
    // Projektionsmatrix liefert dort einen homogenen Vektor mit w=0 (reine Richtung ohne Position),
    // und die anschliessende Division durch w erzeugt Inf/NaN. Jeder endliche Z-Wert entlang
    // desselben Sichtstrahls liefert nach der Normalisierung dieselbe Richtung - Z=1.0 ist einfach
    // der naechste sichere Punkt. Dieser Bug liess die Sonnen-/Mond-Scheibe (die exakte Naehe zu
    // dot()=1.0 braucht) nie treffen, waehrend der grobe Himmel-Farbverlauf durch GPU-eigene
    // NaN-Clamp-Heuristiken zufaellig noch halbwegs plausibel aussah.
    let world_h = params.inverse_view_proj * vec4<f32>(ndc_xy, 1.0, 1.0);
    let world_point = world_h.xyz / world_h.w;
    let ray_dir = normalize(world_point - params.camera_pos.xyz);

    let sun_dir = normalize(params.direction_to_sun.xyz);
    // Weiches Ein-/Ausblenden um die Horizontlinie statt hartem Tag/Nacht-Schnitt bei sun_dir.y = 0.
    let day_t = clamp(sun_dir.y * 2.0 + 0.3, 0.0, 1.0);

    let horizon_to_zenith = smoothstep(0.0, 0.6, clamp(ray_dir.y, 0.0, 1.0));
    let day_sky = mix(params.horizon_day.rgb, params.zenith_day.rgb, horizon_to_zenith);
    var sky_color = mix(params.night.rgb, day_sky, day_t);

    // Sonnen-Scheibe + weicher Streuungs-Halo, beide reine ALU (kein zusaetzlicher Draw-Call/
    // Textur) - bewegt sich automatisch mit `sun_dir`, das schon Tageszeit-gesteuert ist.
    let sun_angle = dot(ray_dir, sun_dir);
    let sun_halo = pow(max(sun_angle, 0.0), 256.0) * day_t;
    sky_color += sun_halo * vec3<f32>(1.0, 0.9, 0.7);
    let sun_disc = smoothstep(0.9997, 0.9999, sun_angle) * step(0.0, sun_dir.y);
    sky_color += sun_disc * vec3<f32>(1.0, 0.97, 0.9);

    // Mond exakt gegenueber der Sonne (einfaches Modell ohne eigene Umlaufbahn) - dadurch
    // erscheint er automatisch genau dann ueber dem Horizont, wenn die Sonne darunter ist.
    let moon_dir = -sun_dir;
    let moon_angle = dot(ray_dir, moon_dir);
    let moon_disc = smoothstep(0.998, 0.9995, moon_angle) * step(0.0, moon_dir.y);
    let moon_halo = pow(max(moon_angle, 0.0), 512.0) * step(0.0, moon_dir.y) * 0.3;
    sky_color += (moon_disc + moon_halo) * vec3<f32>(0.85, 0.88, 0.95);

    return vec4<f32>(sky_color, 1.0);
}
