/// GPU-Driven Culling: 1 Thread pro Chunk-Pool-Slot. Frustum-Test (5 Planes, keine Far-Plane -
/// Reverse-Z mit unendlicher Fernsicht, s. `frustum.rs`-Kommentar), dann HZB-Occlusion-Test gegen
/// den im vorherigen Frame erzeugten Hi-Z-Mip-Chain. Sichtbare Chunks schreiben ihre
/// Indirect-Draw-Argumente + Chunk-Origin per `atomicAdd`-Kompaktierung direkt in die
/// GPU-Buffer - die CPU sieht diesen Pass nie, sie ruft nur noch `multi_draw_indirect_count` auf.

struct ChunkMeta {
    aabb_min: vec4<f32>,
    aabb_max: vec4<f32>,
    slots: array<vec2<u32>, 6>,
};

struct CullUniform {
    view_proj: mat4x4<f32>,
    screen_size: vec4<f32>,
    counts: vec4<u32>,
    camera_pos: vec4<f32>,
};

/// Kamera-nahe Chunks (eigener Chunk + direkte Nachbarn) ueberspringen den HZB-Occlusion-Test
/// komplett - s. Kommentar an `CullUniformData::camera_pos`. Weltblock-Einheiten; deckt bewusst
/// mehr als nur die eigene Chunk-AABB ab, da die relevante Gefahr (Kamera dicht an/in Geometrie)
/// nicht an Chunk-Grenzen halt macht.
const OCCLUSION_TEST_SKIP_RADIUS: f32 = 48.0;

fn closest_point_on_aabb(point: vec3<f32>, box_min: vec3<f32>, box_max: vec3<f32>) -> vec3<f32> {
    return clamp(point, box_min, box_max);
}

struct DrawIndirectArgs {
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
};

struct ChunkData {
    origin: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> chunk_meta: array<ChunkMeta>;
@group(0) @binding(1) var<uniform> cull: CullUniform;
@group(0) @binding(2) var hzb: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> indirect_args: array<DrawIndirectArgs>;
@group(0) @binding(4) var<storage, read_write> chunk_data: array<ChunkData>;
@group(0) @binding(5) var<storage, read_write> counters: array<atomic<u32>, 6>;

fn mat_row(m: mat4x4<f32>, i: u32) -> vec4<f32> {
    return vec4<f32>(m[0][i], m[1][i], m[2][i], m[3][i]);
}

/// Vorzeichenbehafteter Test des AABB-Eckpunkts, der in Richtung der Plane-Normale am weitesten
/// aussen liegt - negativ heisst: die komplette AABB liegt ausserhalb dieser Plane. Unnormalisiert
/// (nur das Vorzeichen zaehlt), identisch zur CPU-Variante in `frustum.rs`.
fn outside_plane(plane: vec4<f32>, box_min: vec3<f32>, box_max: vec3<f32>) -> bool {
    let positive = vec3<f32>(
        select(box_min.x, box_max.x, plane.x >= 0.0),
        select(box_min.y, box_max.y, plane.y >= 0.0),
        select(box_min.z, box_max.z, plane.z >= 0.0),
    );
    return dot(plane.xyz, positive) + plane.w < 0.0;
}

fn frustum_culled(box_min: vec3<f32>, box_max: vec3<f32>) -> bool {
    let row0 = mat_row(cull.view_proj, 0u);
    let row1 = mat_row(cull.view_proj, 1u);
    let row2 = mat_row(cull.view_proj, 2u);
    let row3 = mat_row(cull.view_proj, 3u);

    if outside_plane(row3 + row0, box_min, box_max) { return true; }
    if outside_plane(row3 - row0, box_min, box_max) { return true; }
    if outside_plane(row3 + row1, box_min, box_max) { return true; }
    if outside_plane(row3 - row1, box_min, box_max) { return true; }
    if outside_plane(row3 - row2, box_min, box_max) { return true; }
    return false;
}

const BOX_CORNERS: array<vec3<f32>, 8> = array<vec3<f32>, 8>(
    vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 0.0),
    vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 1.0, 0.0),
    vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 1.0),
    vec3<f32>(0.0, 1.0, 1.0), vec3<f32>(1.0, 1.0, 1.0),
);

/// Projiziert die 8 AABB-Ecken in NDC, sampelt den HZB konservativ (Minimum ueber 4 Texel, s.
/// `hzb_downsample.wgsl`) auf dem passenden Mip-Level und vergleicht gegen den naechstgelegenen
/// (im Reverse-Z groessten) Eckpunkt der Box. Ist selbst der naechste Punkt noch hinter dem
/// gespeicherten (konservativ entferntesten) Tiefenwert, ist die gesamte Box verdeckt.
fn occlusion_culled(box_min: vec3<f32>, box_max: vec3<f32>) -> bool {
    var ndc_min = vec2<f32>(1.0, 1.0);
    var ndc_max = vec2<f32>(-1.0, -1.0);
    var max_z = -1.0;
    var any_valid = false;

    for (var i = 0u; i < 8u; i++) {
        let corner = mix(box_min, box_max, BOX_CORNERS[i]);
        let clip = cull.view_proj * vec4<f32>(corner, 1.0);
        if clip.w <= 1e-4 {
            continue;
        }
        any_valid = true;
        let ndc = clip.xyz / clip.w;
        ndc_min = min(ndc_min, ndc.xy);
        ndc_max = max(ndc_max, ndc.xy);
        max_z = max(max_z, ndc.z);
    }

    if !any_valid {
        return false;
    }

    let u0 = ndc_min.x * 0.5 + 0.5;
    let u1 = ndc_max.x * 0.5 + 0.5;
    let v0 = 0.5 - ndc_max.y * 0.5;
    let v1 = 0.5 - ndc_min.y * 0.5;
    let uv_min = clamp(vec2<f32>(min(u0, u1), min(v0, v1)), vec2<f32>(0.0), vec2<f32>(1.0));
    let uv_max = clamp(vec2<f32>(max(u0, u1), max(v0, v1)), vec2<f32>(0.0), vec2<f32>(1.0));

    let texel_span = max(max((uv_max.x - uv_min.x) * cull.screen_size.x, (uv_max.y - uv_min.y) * cull.screen_size.y), 1.0);
    let max_level = cull.counts.z - 1u;
    let level = u32(clamp(ceil(log2(texel_span)), 0.0, f32(max_level)));

    let dims = textureDimensions(hzb, i32(level));
    let max_coord = vec2<i32>(dims) - vec2<i32>(1, 1);
    let corner_min = clamp(vec2<i32>(uv_min * vec2<f32>(dims)), vec2<i32>(0, 0), max_coord);
    let corner_max = clamp(vec2<i32>(uv_max * vec2<f32>(dims)), vec2<i32>(0, 0), max_coord);

    let d00 = textureLoad(hzb, vec2<i32>(corner_min.x, corner_min.y), i32(level)).r;
    let d10 = textureLoad(hzb, vec2<i32>(corner_max.x, corner_min.y), i32(level)).r;
    let d01 = textureLoad(hzb, vec2<i32>(corner_min.x, corner_max.y), i32(level)).r;
    let d11 = textureLoad(hzb, vec2<i32>(corner_max.x, corner_max.y), i32(level)).r;
    let sampled_hzb_depth = min(min(d00, d10), min(d01, d11));

    return max_z < sampled_hzb_depth;
}

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= cull.counts.y {
        return;
    }

    let chunk = chunk_meta[gid.x];
    if chunk.aabb_min.w < 0.5 {
        return;
    }

    let box_min = chunk.aabb_min.xyz;
    let box_max = chunk.aabb_max.xyz;

    if frustum_culled(box_min, box_max) {
        return;
    }

    let closest = closest_point_on_aabb(cull.camera_pos.xyz, box_min, box_max);
    let near_camera = distance(closest, cull.camera_pos.xyz) < OCCLUSION_TEST_SKIP_RADIUS;
    if !near_camera && occlusion_culled(box_min, box_max) {
        return;
    }

    let max_draws = cull.counts.x;
    let chunk_data_stride = cull.counts.w;
    for (var dir = 0u; dir < 6u; dir++) {
        let slot = chunk.slots[dir];
        if slot.y == 0u {
            continue;
        }

        let local_index = atomicAdd(&counters[dir], 1u);
        if local_index >= max_draws {
            continue;
        }

        indirect_args[dir * max_draws + local_index] = DrawIndirectArgs(6u, slot.y, 0u, slot.x);
        chunk_data[dir * chunk_data_stride + local_index] = ChunkData(vec4<f32>(box_min, chunk.aabb_max.w));
    }
}
